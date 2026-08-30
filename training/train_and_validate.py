from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import joblib
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from scripts.pipeline_common import (
    SEED, TARGET, CLASSES, FEATURES, CATEGORICAL,
    load_and_merge, category_levels_from, prepare_features, make_model,
    calibrate_temperature, apply_temperature, build_cost_matrix,
    evaluate_predictions, classification_metrics, subgroup_metrics,
    expected_calibration_error_binary, make_bundle, save_feature_importance,
    save_json,
)

MODEL_VERSION = "triagegeist-lgbm-safety-v0.6-repro"


def pkg_ver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in df.columns if c not in {"fold", "n_train", "n_valid", "temperature"}]
    for c in metric_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            rows.append({"metric": c, "mean": float(df[c].mean()), "std": float(df[c].std(ddof=1)), "min": float(df[c].min()), "max": float(df[c].max())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Reproduce PatientTriage.ai LightGBM training and full validation.")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--locked-test-fraction", type=float, default=0.15)
    ap.add_argument("--undertriage-weight", type=float, default=3.0)
    ap.add_argument("--overtriage-weight", type=float, default=1.0)
    ap.add_argument("--with-site-heldout-inline", action="store_true", help="Optional: run site-held-out evaluation in this same process. Recommended workflow uses site_heldout_validate.py separately.")
    args = ap.parse_args()

    root = args.project_root.resolve()
    raw_dir = root / "data_raw"
    model_dir = root / "model"
    eval_dir = root / "final_evaluation"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_and_merge(raw_dir, labelled=True)
    y_all = df[TARGET].astype(int).to_numpy()

    # The locked holdout is created first and never participates in CV/calibration/safety tuning.
    dev_idx, locked_idx = train_test_split(
        np.arange(len(df)), test_size=args.locked_test_fraction, random_state=SEED, stratify=y_all
    )
    dev = df.iloc[dev_idx].reset_index(drop=True)
    locked = df.iloc[locked_idx].reset_index(drop=True)
    y_dev = dev[TARGET].astype(int).to_numpy()
    y_locked = locked[TARGET].astype(int).to_numpy()

    category_levels = category_levels_from(dev)
    X_dev = prepare_features(dev, category_levels)
    X_locked = prepare_features(locked, category_levels)

    # ---------------- 5-fold development-only CV ----------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_raw = np.zeros((len(dev), 5), dtype=float)
    fold_rows = []
    cost_matrix = build_cost_matrix(args.undertriage_weight, args.overtriage_weight)

    for fold, (tr, va) in enumerate(skf.split(X_dev, y_dev), start=1):
        model = make_model()
        model.fit(X_dev.iloc[tr], y_dev[tr], categorical_feature=CATEGORICAL)
        raw = model.predict_proba(X_dev.iloc[va])
        oof_raw[va] = raw
        # Fold-local temperature is only for reporting stability. Final temperature is fit to all OOF predictions below.
        fold_t = calibrate_temperature(raw, y_dev[va])
        p = apply_temperature(raw, fold_t)
        fold_metrics, _, safe_pred = evaluate_predictions(y_dev[va], p, cost_matrix)
        row = {
            "fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va)), "temperature": float(fold_t),
            **{f"argmax_{k}": v for k, v in fold_metrics["argmax"].items()},
            **{f"safety_{k}": v for k, v in fold_metrics["safety_operating_point"].items()},
            **fold_metrics["probability_quality"],
        }
        fold_rows.append(row)
        print(f"Fold {fold}/5 complete: safety undertriage={100*row['safety_undertriage_rate']:.2f}%")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(eval_dir / "five_fold_per_fold_metrics.csv", index=False)
    summarize_cv(fold_df).to_csv(eval_dir / "five_fold_cv_summary.csv", index=False)

    # One development-only calibration model: temperature from OOF predictions.
    temperature = calibrate_temperature(oof_raw, y_dev)
    oof_proba = apply_temperature(oof_raw, temperature)

    # Safety frontier on OOF only. The application-compatible selected operating point remains 3.0 by default.
    frontier_rows = []
    for uw in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        C = build_cost_matrix(uw, args.overtriage_weight)
        m, _, _ = evaluate_predictions(y_dev, oof_proba, C)
        frontier_rows.append({"undertriage_weight": uw, **m["safety_operating_point"]})
    pd.DataFrame(frontier_rows).to_csv(eval_dir / "safety_tradeoff_development_oof.csv", index=False)

    # ---------------- Site-held-out development evaluation ----------------
    site_rows = []
    if args.with_site_heldout_inline and "site_id" in dev.columns:
        for site in sorted(dev["site_id"].dropna().astype(str).unique()):
            site_test_mask = dev["site_id"].astype(str).eq(site).to_numpy()
            train_pool = dev.loc[~site_test_mask].reset_index(drop=True)
            held = dev.loc[site_test_mask].reset_index(drop=True)
            y_pool = train_pool[TARGET].astype(int).to_numpy()
            y_held = held[TARGET].astype(int).to_numpy()
            fit_idx, cal_idx = train_test_split(
                np.arange(len(train_pool)), test_size=0.15, random_state=SEED, stratify=y_pool
            )
            site_levels = category_levels_from(train_pool.iloc[fit_idx])
            X_fit = prepare_features(train_pool.iloc[fit_idx], site_levels)
            X_cal = prepare_features(train_pool.iloc[cal_idx], site_levels)
            X_held = prepare_features(held, site_levels)
            m_site = make_model()
            m_site.fit(X_fit, y_pool[fit_idx], categorical_feature=CATEGORICAL)
            site_t = calibrate_temperature(m_site.predict_proba(X_cal), y_pool[cal_idx])
            p_held = apply_temperature(m_site.predict_proba(X_held), site_t)
            sm, _, _ = evaluate_predictions(y_held, p_held, cost_matrix)
            r = {
                "held_out_site": site, "n_train_pool": int(len(train_pool)), "n_held_out": int(len(held)),
                "temperature": float(site_t),
                **{f"argmax_{k}": v for k, v in sm["argmax"].items()},
                **{f"safety_{k}": v for k, v in sm["safety_operating_point"].items()},
                **sm["probability_quality"],
            }
            site_rows.append(r)
            print(f"Held-out site {site}: safety undertriage={100*r['safety_undertriage_rate']:.2f}%", flush=True)
    pd.DataFrame(site_rows).to_csv(eval_dir / "site_held_out_evaluation_development_only.csv", index=False)

    # ---------------- Fit final application model on development only ----------------
    final_model = make_model()
    final_model.fit(X_dev, y_dev, categorical_feature=CATEGORICAL)
    raw_locked = final_model.predict_proba(X_locked)
    p_locked = apply_temperature(raw_locked, temperature)
    locked_metrics, argmax_pred, safety_pred = evaluate_predictions(y_locked, p_locked, cost_matrix)

    # Confusion matrices and per-case predictions.
    pd.DataFrame(
        np.asarray(__import__('sklearn.metrics').metrics.confusion_matrix(y_locked, argmax_pred, labels=CLASSES)),
        index=[f"true_{i}" for i in CLASSES], columns=[f"pred_{i}" for i in CLASSES],
    ).to_csv(eval_dir / "locked_test_confusion_matrix_argmax.csv")
    pd.DataFrame(
        np.asarray(__import__('sklearn.metrics').metrics.confusion_matrix(y_locked, safety_pred, labels=CLASSES)),
        index=[f"true_{i}" for i in CLASSES], columns=[f"pred_{i}" for i in CLASSES],
    ).to_csv(eval_dir / "locked_test_confusion_matrix_safety.csv")

    pred_df = locked[["patient_id", "site_id", "age", "age_group", "sex", "arrival_mode", TARGET]].copy()
    for i in range(5):
        pred_df[f"p_esi{i+1}"] = p_locked[:, i]
    pred_df["p_urgent_esi1_2"] = p_locked[:, :2].sum(axis=1)
    pred_df["argmax_priority"] = argmax_pred
    pred_df["safety_priority"] = safety_pred
    pred_df["undertriage_proxy"] = safety_pred > y_locked
    pred_df["severe_undertriage_proxy"] = (safety_pred - y_locked) >= 2
    pred_df.to_csv(eval_dir / "locked_test_predictions.csv", index=False)

    # Locked subgroup checks.
    subgroup_metrics(
        locked, y_locked, safety_pred,
        ["age_group", "sex", "arrival_mode", "site_id", "transport_origin"], min_n=50,
    ).to_csv(eval_dir / "locked_test_subgroup_metrics.csv", index=False)

    y_urgent = (y_locked <= 2).astype(int)
    p_urgent = p_locked[:, :2].sum(axis=1)
    ece, cal_bins = expected_calibration_error_binary(y_urgent, p_urgent, n_bins=10)
    cal_bins.to_csv(eval_dir / "locked_test_urgent_calibration_bins.csv", index=False)

    # Feature importance and model artifact compatible with risk_engine_ml.py/app_ml.py.
    save_feature_importance(final_model, model_dir / "feature_importance.csv")
    bundle = make_bundle(
        final_model, dev, category_levels, temperature,
        args.undertriage_weight, args.overtriage_weight, MODEL_VERSION,
        len(locked),
        "Locked 15% holdout created before model development; 5-fold stratified CV on development; OOF temperature calibration and safety trade-off on development only; site-held-out validation on development; final model fit on development only.",
    )
    joblib.dump(bundle, model_dir / "triage_model_bundle.joblib", compress=3)

    # App-compatible metrics.json plus richer validation metadata.
    metrics_json = {
        "model_version": MODEL_VERSION,
        "seed": SEED,
        "data": {"total_labelled": int(len(df)), "development": int(len(dev)), "locked_holdout": int(len(locked))},
        "operating_point": {"undertriage_weight": args.undertriage_weight, "overtriage_weight": args.overtriage_weight, "temperature": temperature},
        "locked_test": locked_metrics,
        "five_fold_cv": {
            "n_folds": 5,
            "safety_accuracy_mean": float(fold_df["safety_accuracy"].mean()),
            "safety_accuracy_std": float(fold_df["safety_accuracy"].std(ddof=1)),
            "safety_undertriage_mean": float(fold_df["safety_undertriage_rate"].mean()),
            "safety_undertriage_std": float(fold_df["safety_undertriage_rate"].std(ddof=1)),
            "safety_severe_undertriage_mean": float(fold_df["safety_severe_undertriage_rate"].mean()),
            "safety_qwk_mean": float(fold_df["safety_qwk"].mean()),
        },
        "site_held_out": site_rows,
        "urgent_calibration_ece_10bin_locked": float(ece),
    }
    save_json(model_dir / "metrics.json", metrics_json)
    save_json(eval_dir / "LOCKED_TEST_FINAL_METRICS.json", metrics_json)

    # Reproducibility manifest.
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {k: pkg_ver(k) for k in ["numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "joblib"]},
        "seed": SEED,
        "model_version": MODEL_VERSION,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    save_json(eval_dir / "run_manifest.json", manifest)

    # Human-readable report.
    a = locked_metrics["argmax"]
    s = locked_metrics["safety_operating_point"]
    report = "# PatientTriage.ai Reproducible Model Evaluation\n\n"
    report += f"- Model: LightGBM 5-class acuity classifier (`{MODEL_VERSION}`)\n"
    report += f"- Labelled synthetic Triagegeist encounters: {len(df):,}\n"
    report += f"- Development pool: {len(dev):,}\n"
    report += f"- Locked holdout: {len(locked):,}\n"
    report += f"- Safety undertriage weight: {args.undertriage_weight:g}x; overtriage weight: {args.overtriage_weight:g}x\n"
    report += f"- OOF temperature: {temperature:.4f}\n\n"
    report += "## Locked holdout\n\n"
    report += "| Metric | Plain argmax | Safety operating point |\n|---|---:|---:|\n"
    for label, key in [
        ("Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("QWK", "qwk"),
        ("Undertriage proxy", "undertriage_rate"), ("Severe undertriage proxy", "severe_undertriage_rate"),
        ("Overtriage", "overtriage_rate"), ("ESI-1 recall", "recall_esi1"), ("ESI-2 recall", "recall_esi2")]:
        report += f"| {label} | {a[key]:.4f} | {s[key]:.4f} |\n"
    report += "\n## Validation design\n\n"
    report += "1. Locked holdout created before development and excluded from all fitting/tuning.\n"
    report += "2. Five-fold stratified CV performed only on the development pool.\n"
    report += "3. Probability temperature fitted using development OOF predictions.\n"
    report += "4. Site-held-out evaluation performed only inside the development pool.\n"
    report += "5. Final deployment model trained on development pool only, then evaluated once on locked holdout.\n\n"
    report += "## Important limitation\n\nThese are synthetic Triagegeist benchmark results, not real-world clinical performance claims. `triage_acuity` is an acuity label, so the reported undertriage metric is an acuity under-classification proxy.\n"
    (eval_dir / "FINAL_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")

    print("\nTraining and validation complete.")
    print(f"Model: {model_dir / 'triage_model_bundle.joblib'}")
    print(f"Locked safety accuracy: {100*s['accuracy']:.2f}%")
    print(f"Locked safety undertriage proxy: {100*s['undertriage_rate']:.2f}%")
    print(f"Locked severe undertriage proxy: {100*s['severe_undertriage_rate']:.3f}%")
    print(f"Locked ESI-1 recall: {100*s['recall_esi1']:.2f}%")


if __name__ == "__main__":
    main()

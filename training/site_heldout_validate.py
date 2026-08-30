from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.pipeline_common import (
    SEED, TARGET, CATEGORICAL, load_and_merge, category_levels_from, prepare_features,
    make_model, calibrate_temperature, apply_temperature, build_cost_matrix, evaluate_predictions,
)


def main():
    ap = argparse.ArgumentParser(description="Development-only leave-one-site-out validation for PatientTriage.ai.")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--locked-test-fraction", type=float, default=0.15)
    ap.add_argument("--undertriage-weight", type=float, default=3.0)
    ap.add_argument("--overtriage-weight", type=float, default=1.0)
    args = ap.parse_args()

    root = args.project_root.resolve()
    eval_dir = root / "final_evaluation"; eval_dir.mkdir(parents=True, exist_ok=True)
    model_metrics_path = root / "model" / "metrics.json"

    df = load_and_merge(root / "data_raw", labelled=True)
    y = df[TARGET].astype(int).to_numpy()
    dev_idx, _ = train_test_split(np.arange(len(df)), test_size=args.locked_test_fraction, random_state=SEED, stratify=y)
    dev = df.iloc[dev_idx].reset_index(drop=True)
    C = build_cost_matrix(args.undertriage_weight, args.overtriage_weight)

    rows = []
    for site in sorted(dev["site_id"].dropna().astype(str).unique()):
        held_mask = dev["site_id"].astype(str).eq(site).to_numpy()
        train_pool = dev.loc[~held_mask].reset_index(drop=True)
        held = dev.loc[held_mask].reset_index(drop=True)
        y_pool = train_pool[TARGET].astype(int).to_numpy()
        y_held = held[TARGET].astype(int).to_numpy()
        fit_idx, cal_idx = train_test_split(np.arange(len(train_pool)), test_size=0.15, random_state=SEED, stratify=y_pool)
        levels = category_levels_from(train_pool.iloc[fit_idx])
        X_fit = prepare_features(train_pool.iloc[fit_idx], levels)
        X_cal = prepare_features(train_pool.iloc[cal_idx], levels)
        X_held = prepare_features(held, levels)
        model = make_model(n_jobs=2)
        model.fit(X_fit, y_pool[fit_idx], categorical_feature=CATEGORICAL)
        temp = calibrate_temperature(model.predict_proba(X_cal), y_pool[cal_idx])
        proba = apply_temperature(model.predict_proba(X_held), temp)
        m, _, _ = evaluate_predictions(y_held, proba, C)
        r = {
            "held_out_site": site, "n_train_pool": int(len(train_pool)), "n_held_out": int(len(held)),
            "temperature": float(temp),
            **{f"argmax_{k}": v for k, v in m["argmax"].items()},
            **{f"safety_{k}": v for k, v in m["safety_operating_point"].items()},
            **m["probability_quality"],
        }
        rows.append(r)
        print(f"{site}: safety undertriage={100*r['safety_undertriage_rate']:.2f}%", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(eval_dir / "site_held_out_evaluation_development_only.csv", index=False)

    if model_metrics_path.exists():
        metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
        metrics["site_held_out"] = rows
        model_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (eval_dir / "LOCKED_TEST_FINAL_METRICS.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved site-held-out evaluation: {eval_dir / 'site_held_out_evaluation_development_only.csv'}")

if __name__ == "__main__":
    main()

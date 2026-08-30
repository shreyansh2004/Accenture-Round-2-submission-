from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

SEED = 20260829
TARGET = "triage_acuity"
CLASSES = np.arange(1, 6, dtype=int)

EXCLUDED_FEATURES = [
    "arrival_day", "arrival_hour", "arrival_month", "arrival_season",
    "chief_complaint_raw", "disposition", "ed_los_hours", "insurance_type",
    "language", "news2_score", "patient_id", "shift", "site_id",
    "triage_acuity", "triage_nurse_id",
]

FEATURES = [
    "arrival_mode", "age", "age_group", "sex", "transport_origin",
    "pain_location", "mental_status_triage", "chief_complaint_system",
    "num_prior_ed_visits_12m", "num_prior_admissions_12m",
    "num_active_medications", "num_comorbidities", "systolic_bp",
    "diastolic_bp", "mean_arterial_pressure", "pulse_pressure", "heart_rate",
    "respiratory_rate", "temperature_c", "spo2", "gcs_total", "pain_score",
    "weight_kg", "height_cm", "bmi", "shock_index", "hx_hypertension",
    "hx_diabetes_type2", "hx_diabetes_type1", "hx_asthma", "hx_copd",
    "hx_heart_failure", "hx_atrial_fibrillation", "hx_ckd",
    "hx_liver_disease", "hx_malignancy", "hx_obesity", "hx_depression",
    "hx_anxiety", "hx_dementia", "hx_epilepsy", "hx_hypothyroidism",
    "hx_hyperthyroidism", "hx_hiv", "hx_coagulopathy",
    "hx_immunosuppressed", "hx_pregnant", "hx_substance_use_disorder",
    "hx_coronary_artery_disease", "hx_stroke_prior",
    "hx_peripheral_vascular_disease",
]

CATEGORICAL = [
    "arrival_mode", "age_group", "sex", "transport_origin", "pain_location",
    "mental_status_triage", "chief_complaint_system",
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

DEFAULT_LGBM_PARAMS = dict(
    boosting_type="gbdt",
    objective="multiclass",
    num_class=5,
    n_estimators=140,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.15,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
    deterministic=True,
    force_col_wise=True,
)


def load_and_merge(raw_dir: Path, labelled: bool = True) -> pd.DataFrame:
    base_path = raw_dir / ("train.csv" if labelled else "test.csv")
    hist_path = raw_dir / "patient_history.csv"
    complaints_path = raw_dir / "chief_complaints.csv"
    if not base_path.exists() or not hist_path.exists() or not complaints_path.exists():
        raise FileNotFoundError(
            f"Expected train/test, patient_history.csv, and chief_complaints.csv under {raw_dir}"
        )
    base = pd.read_csv(base_path)
    hist = pd.read_csv(hist_path)
    complaints = pd.read_csv(complaints_path)
    # Raw complaint is merged for audit/display but deliberately excluded from model features.
    complaint_raw = complaints[["patient_id", "chief_complaint_raw"]].drop_duplicates("patient_id")
    out = base.merge(hist, on="patient_id", how="left", validate="one_to_one")
    out = out.merge(complaint_raw, on="patient_id", how="left", validate="one_to_one")
    if labelled and TARGET not in out.columns:
        raise ValueError(f"{TARGET} missing from labelled training data")
    return out


def category_levels_from(df: pd.DataFrame) -> Dict[str, List[str]]:
    levels = {}
    for c in CATEGORICAL:
        vals = sorted(df[c].dropna().astype(str).unique().tolist())
        if "UNKNOWN" not in vals:
            vals.append("UNKNOWN")
        levels[c] = vals
    return levels


def prepare_features(df: pd.DataFrame, category_levels: Dict[str, List[str]]) -> pd.DataFrame:
    work = df.copy()
    # Recompute derived physiology so training and live re-assessment share one convention.
    if {"systolic_bp", "diastolic_bp"}.issubset(work.columns):
        work["mean_arterial_pressure"] = (work["systolic_bp"] + 2 * work["diastolic_bp"]) / 3.0
        work["pulse_pressure"] = work["systolic_bp"] - work["diastolic_bp"]
    if {"heart_rate", "systolic_bp"}.issubset(work.columns):
        work["shock_index"] = work["heart_rate"] / work["systolic_bp"]
    if {"weight_kg", "height_cm"}.issubset(work.columns):
        h = work["height_cm"] / 100.0
        work["bmi"] = work["weight_kg"] / (h * h)

    for c in FEATURES:
        if c not in work.columns:
            work[c] = np.nan
    X = work[FEATURES].copy()
    for c in CATEGORICAL:
        s = X[c].fillna("UNKNOWN").astype(str)
        levels = category_levels[c]
        s = s.where(s.isin(levels), "UNKNOWN")
        X[c] = pd.Categorical(s, categories=levels)
    return X


def make_model(n_jobs: int | None = None) -> LGBMClassifier:
    params = dict(DEFAULT_LGBM_PARAMS)
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)
    return LGBMClassifier(**params)


def calibrate_temperature(raw_proba: np.ndarray, y_true: np.ndarray) -> float:
    y_idx = np.asarray(y_true, dtype=int) - 1
    raw = np.clip(np.asarray(raw_proba, dtype=float), 1e-12, 1.0)

    def objective(log_t: float) -> float:
        t = math.exp(log_t)
        logp = np.log(raw) / t
        logp -= logp.max(axis=1, keepdims=True)
        q = np.exp(logp)
        q /= q.sum(axis=1, keepdims=True)
        return float(log_loss(y_idx, q, labels=np.arange(5)))

    res = minimize_scalar(objective, bounds=(math.log(0.25), math.log(4.0)), method="bounded")
    return float(math.exp(res.x))


def apply_temperature(raw_proba: np.ndarray, temperature: float) -> np.ndarray:
    raw = np.clip(np.asarray(raw_proba, dtype=float), 1e-12, 1.0)
    logp = np.log(raw) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(axis=1, keepdims=True)


def build_cost_matrix(undertriage_weight: float = 3.0, overtriage_weight: float = 1.0) -> np.ndarray:
    C = np.zeros((5, 5), dtype=float)
    for i, actual in enumerate(CLASSES):
        for j, pred in enumerate(CLASSES):
            if pred == actual:
                C[i, j] = 0.0
            elif pred > actual:  # under-triage: larger ESI number = less urgent
                C[i, j] = undertriage_weight * float((pred - actual) ** 2)
            else:  # over-triage
                C[i, j] = overtriage_weight * float((actual - pred) ** 2)
    return C


def safety_decision(proba: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    expected = np.asarray(proba) @ np.asarray(cost_matrix)
    return CLASSES[np.argmin(expected, axis=1)]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "undertriage_rate": float(np.mean(y_pred > y_true)),
        "severe_undertriage_rate": float(np.mean((y_pred - y_true) >= 2)),
        "overtriage_rate": float(np.mean(y_pred < y_true)),
    }
    for cls in CLASSES:
        out[f"recall_esi{cls}"] = float(recall_score(y_true == cls, y_pred == cls, zero_division=0))
    if prefix:
        return {f"{prefix}{k}": v for k, v in out.items()}
    return out


def urgent_probability_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_urgent = (np.asarray(y_true, dtype=int) <= 2).astype(int)
    p_urgent = np.asarray(proba)[:, :2].sum(axis=1)
    return {
        "urgent_roc_auc": float(roc_auc_score(y_urgent, p_urgent)),
        "urgent_pr_auc": float(average_precision_score(y_urgent, p_urgent)),
        "urgent_brier": float(brier_score_loss(y_urgent, p_urgent)),
    }


def expected_calibration_error_binary(y_true_binary: np.ndarray, p: np.ndarray, n_bins: int = 10):
    y = np.asarray(y_true_binary, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & ((p < hi) if b < n_bins - 1 else (p <= hi))
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": b + 1, "lower": lo, "upper": hi, "n": 0, "mean_prediction": np.nan, "observed_rate": np.nan})
            continue
        mean_p = float(p[mask].mean())
        obs = float(y[mask].mean())
        ece += (n / len(y)) * abs(mean_p - obs)
        rows.append({"bin": b + 1, "lower": lo, "upper": hi, "n": n, "mean_prediction": mean_p, "observed_rate": obs})
    return float(ece), pd.DataFrame(rows)


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, cost_matrix: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
    argmax_pred = CLASSES[np.argmax(proba, axis=1)]
    safe_pred = safety_decision(proba, cost_matrix)
    metrics = {
        "argmax": classification_metrics(y_true, argmax_pred),
        "safety_operating_point": classification_metrics(y_true, safe_pred),
        "probability_quality": urgent_probability_metrics(y_true, proba),
    }
    y_urgent = (np.asarray(y_true) <= 2).astype(int)
    p_urgent = proba[:, :2].sum(axis=1)
    ece, bins = expected_calibration_error_binary(y_urgent, p_urgent)
    metrics["probability_quality"]["urgent_ece_10bin"] = ece
    return metrics, argmax_pred, safe_pred


def subgroup_metrics(df_meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, group_cols: Iterable[str], min_n: int = 50) -> pd.DataFrame:
    rows = []
    temp = df_meta.reset_index(drop=True).copy()
    temp["_y_true"] = np.asarray(y_true, dtype=int)
    temp["_y_pred"] = np.asarray(y_pred, dtype=int)
    for col in group_cols:
        if col not in temp.columns:
            continue
        for val, g in temp.groupby(col, dropna=False):
            if len(g) < min_n:
                continue
            m = classification_metrics(g["_y_true"].to_numpy(), g["_y_pred"].to_numpy())
            rows.append({"group_column": col, "group_value": str(val), "n": int(len(g)), **m})
    return pd.DataFrame(rows)


def feature_quantiles(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    q05, q95 = {}, {}
    for c in NUMERIC:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            q05[c] = float(s.quantile(0.05))
            q95[c] = float(s.quantile(0.95))
    return q05, q95


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def make_bundle(model, dev_df: pd.DataFrame, category_levels: Dict[str, List[str]], temperature: float,
                undertriage_weight: float, overtriage_weight: float, model_version: str,
                locked_test_rows: int, selection_basis: str):
    q05, q95 = feature_quantiles(dev_df)
    return {
        "model": model,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "category_levels": category_levels,
        "numeric_q05": q05,
        "numeric_q95": q95,
        "classes": CLASSES,
        "temperature": float(temperature),
        "undertriage_weight": float(undertriage_weight),
        "overtriage_weight": float(overtriage_weight),
        "cost_matrix": build_cost_matrix(undertriage_weight, overtriage_weight),
        "excluded_features": EXCLUDED_FEATURES,
        "model_version": model_version,
        "training_rows": int(len(dev_df)),
        "locked_test_rows_excluded_from_training": int(locked_test_rows),
        "selection_basis": selection_basis,
    }


def save_feature_importance(model, path: Path) -> pd.DataFrame:
    split = model.booster_.feature_importance(importance_type="split")
    gain = model.booster_.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": FEATURES, "split_importance": split, "gain_importance": gain})
    total_gain = df["gain_importance"].sum()
    df["gain_fraction"] = df["gain_importance"] / total_gain if total_gain > 0 else 0.0
    df = df.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

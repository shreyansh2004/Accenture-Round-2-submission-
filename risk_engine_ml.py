"""PatientTriage.ai ML risk engine v0.4.

Competition proof-of-concept only. The model is trained on the synthetic Triagegeist
benchmark and is NOT clinically validated or intended for patient care.

Core design:
- Safety-selected LightGBM multiclass model (winner of LightGBM/CatBoost/XGBoost + ensemble benchmark) trained on triage-time structured data + patient history
- leakage-prone post-triage variables excluded
- raw chief-complaint text excluded from the primary model because the synthetic
  benchmark contains near-label-revealing severity phrases; complaint system remains
- temperature-scaled probabilities
- asymmetric post-hoc decision layer that penalizes under-triage more than over-triage
- deterministic safety guardrails remain outside the ML model
- temporal re-inference: updated vitals are re-scored, and ML risk delta is tracked
- uncertainty from probability entropy + data quality/missingness
- assessment freshness and configurable reassessment clock
- information-value probe for missing intake variables
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "model" / "triage_model_bundle.joblib"

MODEL_VERSION = "triagegeist-lgbm-safety-v0.6-repro"

# Illustrative prototype intervals only; hospitals would configure these to local protocols.
REASSESSMENT_MINUTES = {1: 0, 2: 10, 3: 30, 4: 60, 5: 120}


@dataclass
class MLEvaluation:
    patient_id: str
    recommended_priority: int
    argmax_priority: int
    probability_esi1: float
    probability_esi2: float
    probability_esi3: float
    probability_esi4: float
    probability_esi5: float
    urgent_probability: float
    current_risk: str
    confidence: float
    uncertainty: str
    trajectory: str
    ml_risk_delta: float
    physiological_delta: Dict[str, float]
    freshness: str
    reassessment_due_in_min: int
    attention_score: float
    alert: str
    manual_review: bool
    next_best_information: str
    discordance: str
    top_model_drivers: List[str]
    model_version: str = MODEL_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TriageML:
    def __init__(self, bundle_path: Path | str = BUNDLE_PATH):
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle["model"]
        self.features = self.bundle["features"]
        self.categorical = self.bundle["categorical"]
        self.category_levels = self.bundle["category_levels"]
        self.classes = np.asarray(self.bundle["classes"], dtype=int)
        self.temperature = float(self.bundle["temperature"])
        self.cost_matrix = np.asarray(self.bundle["cost_matrix"], dtype=float)
        self.numeric_q05 = self.bundle.get("numeric_q05", {})
        self.numeric_q95 = self.bundle.get("numeric_q95", {})

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        # Recompute clinically derived fields when the source vitals change.
        if {"systolic_bp", "diastolic_bp"}.issubset(work.columns):
            work["mean_arterial_pressure"] = (work["systolic_bp"] + 2 * work["diastolic_bp"]) / 3.0
            work["pulse_pressure"] = work["systolic_bp"] - work["diastolic_bp"]
        if {"heart_rate", "systolic_bp"}.issubset(work.columns):
            work["shock_index"] = work["heart_rate"] / work["systolic_bp"]
        if {"weight_kg", "height_cm"}.issubset(work.columns):
            h = work["height_cm"] / 100.0
            work["bmi"] = work["weight_kg"] / (h * h)

        for c in self.features:
            if c not in work.columns:
                work[c] = np.nan
        X = work[self.features].copy()
        for c in self.categorical:
            s = X[c].fillna("UNKNOWN").astype(str)
            levels = self.category_levels[c]
            s = s.where(s.isin(levels), "UNKNOWN")
            X[c] = pd.Categorical(s, categories=levels)
        return X

    def calibrated_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(self.prepare(df))
        logp = np.log(np.clip(raw, 1e-12, 1.0)) / self.temperature
        logp -= logp.max(axis=1, keepdims=True)
        q = np.exp(logp)
        return q / q.sum(axis=1, keepdims=True)

    def safety_decision(self, proba: np.ndarray) -> np.ndarray:
        expected = proba @ self.cost_matrix
        return np.arange(1, 6)[np.argmin(expected, axis=1)]

    def argmax_decision(self, proba: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(proba, axis=1)]

    @staticmethod
    def entropy_confidence(proba: np.ndarray) -> np.ndarray:
        p = np.clip(proba, 1e-12, 1.0)
        entropy = -(p * np.log(p)).sum(axis=1) / math.log(p.shape[1])
        return np.clip(100.0 * (1.0 - entropy), 0, 100)

    def top_contributions(self, row: pd.DataFrame, target_class: int, n: int = 5) -> List[str]:
        """LightGBM contribution values for explanation; association, not causation."""
        X = self.prepare(row)
        contrib = np.asarray(self.model.booster_.predict(X, pred_contrib=True))
        nfeat = len(self.features) + 1
        try:
            block = contrib[0].reshape(5, nfeat)[target_class - 1, :-1]
        except Exception:
            return []
        idx = np.argsort(np.abs(block))[::-1][:n]
        out = []
        for i in idx:
            val = row.iloc[0].get(self.features[i], np.nan)
            sign = "+" if block[i] >= 0 else "−"
            out.append(f"{self.features[i]}={val} ({sign} model contribution)")
        return out

    def next_best_information(self, row: pd.DataFrame) -> str:
        """Approximate information-value probe for missing core intake variables.

        For each missing variable, substitute the training 5th and 95th percentile and
        measure the spread in urgent (ESI 1/2) probability. The largest spread is surfaced.
        """
        candidates = [
            "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
            "temperature_c", "spo2", "gcs_total", "pain_score",
        ]
        missing = [c for c in candidates if c in row.columns and pd.isna(row.iloc[0][c])]
        if not missing:
            # Data-quality flags can still create a next action.
            if str(row.iloc[0].get("spo2_signal_quality", "good")).lower() in {"noisy", "poor", "unreliable"}:
                return "Repeat SpO₂ with a reliable signal; current measurement quality is poor."
            return "No critical intake field is currently missing. Continue on the reassessment clock."

        base = row.copy()
        best = None
        for c in missing:
            if c not in self.numeric_q05 or c not in self.numeric_q95:
                continue
            low = base.copy(); high = base.copy()
            low.loc[low.index[0], c] = self.numeric_q05[c]
            high.loc[high.index[0], c] = self.numeric_q95[c]
            pl = self.calibrated_proba(low)[0, :2].sum()
            ph = self.calibrated_proba(high)[0, :2].sum()
            spread = abs(float(ph - pl))
            if best is None or spread > best[0]:
                best = (spread, c)
        if best:
            pretty = {
                "systolic_bp":"systolic blood pressure", "diastolic_bp":"diastolic blood pressure",
                "heart_rate":"heart rate", "respiratory_rate":"respiratory rate",
                "temperature_c":"temperature", "spo2":"oxygen saturation",
                "gcs_total":"GCS", "pain_score":"pain score",
            }.get(best[1], best[1])
            return f"Obtain/repeat {pretty}; it has the highest estimated information value among missing intake variables."
        return "Manual review required because key intake data are incomplete."


def _risk_label(priority: int, urgent_p: float) -> str:
    if priority == 1 or urgent_p >= 0.80:
        return "Critical"
    if priority == 2 or urgent_p >= 0.45:
        return "High"
    if priority == 3:
        return "Moderate"
    if priority == 4:
        return "Low"
    return "Very Low"


def _physiological_delta(current: pd.Series, previous: Optional[pd.Series], minutes: Optional[float]) -> Tuple[Dict[str, float], str]:
    if previous is None or minutes is None or minutes <= 0:
        return {}, "No repeat vitals yet"
    scale = 10.0 / float(minutes)
    cols = ["spo2", "heart_rate", "respiratory_rate", "systolic_bp", "temperature_c"]
    out: Dict[str, float] = {}
    for c in cols:
        a = pd.to_numeric(pd.Series([current.get(c)]), errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([previous.get(c)]), errors="coerce").iloc[0]
        if pd.notna(a) and pd.notna(b):
            out[c] = float((a - b) * scale)

    # A compact directional description, not a clinical diagnosis.
    worsening = 0
    if out.get("spo2", 0) <= -2: worsening += 1
    if out.get("heart_rate", 0) >= 10: worsening += 1
    if out.get("respiratory_rate", 0) >= 4: worsening += 1
    if out.get("systolic_bp", 0) <= -10: worsening += 1
    improving = 0
    if out.get("spo2", 0) >= 2: improving += 1
    if out.get("heart_rate", 0) <= -10: improving += 1
    if out.get("respiratory_rate", 0) <= -4: improving += 1
    if out.get("systolic_bp", 0) >= 10: improving += 1
    if worsening >= 3: label = "Rapidly worsening"
    elif worsening >= 1: label = "Worsening"
    elif improving >= 2: label = "Improving"
    else: label = "Stable"
    return out, label


def _hard_safety_alert(row: pd.Series) -> Optional[str]:
    """Prototype guardrails only; not a validated clinical rule set."""
    def f(name):
        try:
            v = float(row.get(name))
            return v if np.isfinite(v) else None
        except Exception:
            return None
    spo2, sbp, gcs, rr = f("spo2"), f("systolic_bp"), f("gcs_total"), f("respiratory_rate")
    if spo2 is not None and spo2 <= 88:
        return "CRITICAL SAFETY ALERT — extreme oxygen saturation; immediate clinician review."
    if sbp is not None and sbp <= 80:
        return "CRITICAL SAFETY ALERT — extreme systolic blood pressure; immediate clinician review."
    if gcs is not None and gcs <= 8:
        return "CRITICAL SAFETY ALERT — severely reduced consciousness; immediate clinician review."
    if rr is not None and (rr <= 8 or rr >= 35):
        return "CRITICAL SAFETY ALERT — extreme respiratory rate; immediate clinician review."
    return None


def _uncertainty(confidence: float, row: pd.Series) -> Tuple[str, bool]:
    missing_core = sum(pd.isna(row.get(c)) for c in ["systolic_bp","heart_rate","respiratory_rate","spo2","temperature_c","gcs_total"])
    quality = str(row.get("spo2_signal_quality", "good")).lower()
    history_available = bool(row.get("history_available", True))
    adjusted = float(confidence)
    adjusted -= 7.5 * missing_core
    if quality in {"noisy", "poor", "unreliable"}: adjusted -= 15
    if not history_available: adjusted -= 8
    adjusted = max(0.0, adjusted)
    if adjusted < 50: return "High", True
    if adjusted < 75: return "Medium", True
    return "Low", False


def _freshness(priority: int, minutes_since_assessed: float) -> Tuple[str, int]:
    interval = REASSESSMENT_MINUTES.get(int(priority), 30)
    if interval == 0:
        return "Immediate", 0
    due = int(round(interval - float(minutes_since_assessed)))
    if due < 0:
        return f"Stale / overdue by {abs(due)} min", due
    if due <= max(5, int(interval * 0.25)):
        return f"Aging / due in {due} min", due
    return f"Fresh / due in {due} min", due


def _empirical_news2_priority(news2: Any) -> Optional[int]:
    try:
        n = float(news2)
    except Exception:
        return None
    if not np.isfinite(n):
        return None
    # This is an empirical demo comparator, not a formal NEWS2-to-ESI conversion.
    if n >= 12: return 1
    if n >= 6: return 2
    if n >= 2: return 3
    if n >= 1: return 4
    return 5


def evaluate_patient(engine: TriageML, current_row: pd.Series, previous_row: Optional[pd.Series] = None,
                     elapsed_minutes: Optional[float] = None) -> MLEvaluation:
    row_df = pd.DataFrame([current_row])
    p = engine.calibrated_proba(row_df)[0]
    argmax = int(engine.argmax_decision(p.reshape(1,-1))[0])
    priority = int(engine.safety_decision(p.reshape(1,-1))[0])
    urgent_p = float(p[:2].sum())
    base_conf = float(engine.entropy_confidence(p.reshape(1,-1))[0])
    uncertainty, manual_review = _uncertainty(base_conf, current_row)

    # Re-score the previous snapshot with the same ML model so trajectory is not hand-weighted.
    ml_delta = 0.0
    if previous_row is not None and elapsed_minutes and elapsed_minutes > 0:
        pp = engine.calibrated_proba(pd.DataFrame([previous_row]))[0]
        ml_delta = float(urgent_p - pp[:2].sum())
    phys_delta, phys_label = _physiological_delta(current_row, previous_row, elapsed_minutes)
    if ml_delta >= 0.25 or phys_label == "Rapidly worsening":
        trajectory = "Rapidly worsening"
    elif ml_delta >= 0.08 or phys_label == "Worsening":
        trajectory = "Worsening"
    elif ml_delta <= -0.15 or phys_label == "Improving":
        trajectory = "Improving"
    elif previous_row is None:
        trajectory = "No repeat vitals yet"
    else:
        trajectory = "Stable"

    hard = _hard_safety_alert(current_row)
    if hard:
        priority = 1
    elif uncertainty == "High" and priority > 2:
        # Safety-first abstention: high uncertainty cannot produce a reassuring low-priority result.
        priority = 2
        manual_review = True

    minutes_since = float(current_row.get("last_assessed_minutes", 0) or 0)
    freshness, due = _freshness(priority, minutes_since)

    protocol_priority = _empirical_news2_priority(current_row.get("news2_score"))
    discordance = "None"
    if protocol_priority is not None and abs(protocol_priority - priority) >= 2:
        discordance = f"Model/protocol disagreement ({priority} vs empirical NEWS2 signal {protocol_priority}) — second review recommended."
        manual_review = True

    alert = "No active alert"
    if hard:
        alert = hard
    elif trajectory == "Rapidly worsening":
        alert = "RAPID DETERIORATION ALERT — immediate clinician reassessment recommended."
    elif trajectory == "Worsening":
        alert = "RISING-RISK ALERT — clinician reassessment recommended."
    elif due < 0:
        alert = "REASSESSMENT OVERDUE — waiting patient requires review."
    elif manual_review:
        alert = "MANUAL REVIEW REQUIRED — uncertainty or signal discordance is material."

    # Queue attention score is intentionally separate from clinical priority.
    attention = (6 - priority) * 20.0
    attention += max(0.0, ml_delta) * 50.0
    if trajectory == "Rapidly worsening": attention += 18
    elif trajectory == "Worsening": attention += 8
    if uncertainty == "High": attention += 12
    elif uncertainty == "Medium": attention += 5
    if due < 0: attention += min(15, abs(due) * 0.5)
    attention = float(min(100, attention))

    current_risk = _risk_label(priority, urgent_p)
    top = engine.top_contributions(row_df, argmax, n=5)
    nbi = engine.next_best_information(row_df)

    return MLEvaluation(
        patient_id=str(current_row.get("demo_id", current_row.get("patient_id", "UNKNOWN"))),
        recommended_priority=priority,
        argmax_priority=argmax,
        probability_esi1=float(p[0]), probability_esi2=float(p[1]), probability_esi3=float(p[2]),
        probability_esi4=float(p[3]), probability_esi5=float(p[4]), urgent_probability=urgent_p,
        current_risk=current_risk, confidence=base_conf, uncertainty=uncertainty,
        trajectory=trajectory, ml_risk_delta=ml_delta, physiological_delta=phys_delta,
        freshness=freshness, reassessment_due_in_min=due, attention_score=attention,
        alert=alert, manual_review=manual_review, next_best_information=nbi,
        discordance=discordance, top_model_drivers=top,
    )

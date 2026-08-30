from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from scripts.pipeline_common import load_and_merge, prepare_features, apply_temperature, safety_decision


def main():
    ap = argparse.ArgumentParser(description="Score the unlabeled Triagegeist test.csv with a trained PatientTriage.ai bundle.")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    root = args.project_root.resolve()
    output = args.output or (root / "final_evaluation" / "test_predictions.csv")
    bundle = joblib.load(root / "model" / "triage_model_bundle.joblib")
    df = load_and_merge(root / "data_raw", labelled=False)
    X = prepare_features(df, bundle["category_levels"])
    raw = bundle["model"].predict_proba(X)
    p = apply_temperature(raw, bundle["temperature"])
    argmax = np.asarray(bundle["classes"])[np.argmax(p, axis=1)]
    safe = safety_decision(p, np.asarray(bundle["cost_matrix"]))
    out = df[["patient_id"]].copy()
    for i in range(5):
        out[f"p_esi{i+1}"] = p[:, i]
    out["p_urgent_esi1_2"] = p[:, :2].sum(axis=1)
    out["argmax_priority"] = argmax
    out["recommended_priority"] = safe
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved {len(out):,} predictions to {output}")

if __name__ == "__main__":
    main()

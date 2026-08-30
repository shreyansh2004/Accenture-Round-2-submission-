# Model Card — PatientTriage.ai `triagegeist-lgbm-safety-v0.6-repro`

## Intended use
Competition proof-of-concept for emergency triage decision support. The model estimates a 5-level acuity distribution from triage-time structured data and patient-history indicators. It is **not clinically validated and must not be used for patient care**.

## Training data
Synthetic Triagegeist benchmark:

- 80,000 labelled encounters
- 68,000 development rows
- 12,000 locked holdout rows
- joined with `patient_history.csv`
- raw complaint text is deliberately excluded from primary ML fitting

## Model
LightGBM multiclass classifier, 51 features, 5 classes (ESI-like levels 1–5).

Key hyperparameters:

- `n_estimators=140`
- `learning_rate=0.08`
- `num_leaves=63`
- `min_child_samples=30`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `reg_alpha=0.15`
- `reg_lambda=2.0`
- random seed `20260829`

Probabilities are temperature-scaled from development out-of-fold predictions.

## Safety decision policy
The default operating point penalizes under-triage 3× more heavily than over-triage, with squared acuity distance. The final recommendation minimizes expected error cost rather than simply taking argmax probability.

## Current locked-holdout performance

| Metric | Argmax | Safety operating point |
|---|---:|---:|
| Accuracy | 85.49% | 82.30% |
| Macro-F1 | 87.29% | 83.07% |
| QWK | 0.930 | 0.909 |
| Undertriage proxy | 7.67% | 3.04% |
| Severe undertriage proxy | 0.108% | 0.025% |
| Overtriage | 6.84% | 14.66% |
| ESI-1 recall | 94.00% | 96.07% |
| ESI-2 recall | 96.92% | 97.07% |

Urgent (ESI 1/2) probability on the locked synthetic benchmark:

- ROC-AUC ≈ 0.99964
- PR-AUC ≈ 0.99883
- Brier ≈ 0.00445
- 10-bin ECE ≈ 0.00168

## Validation

- locked 15% holdout excluded from fitting/tuning;
- 5-fold stratified development cross-validation;
- development OOF probability calibration;
- development-only safety trade-off analysis;
- development-only leave-one-site-out generalization;
- subgroup checks by age group, sex, arrival mode, site and transport origin.

## Important limitations

1. Triagegeist is synthetic.
2. The target is triage acuity, not mortality, ICU transfer or future deterioration.
3. “Undertriage” in these reports means acuity under-classification relative to synthetic `triage_acuity`.
4. Repeat-vital behavior is temporal re-estimation: new measurements are rescored by the same model; the model does not predict future vital signs.
5. Hard safety guardrails in the demo are prototype rules rather than validated clinical protocols.
6. External/prospective validation is required before real clinical use.

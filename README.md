# PatientTriage.ai - Reproducible ML Training + Validation + Prototype

**Project:** Accenture Innovation Challenge 2026 — PatientTriage.ai  
**Model:** LightGBM multiclass acuity classifier + calibrated, safety-biased decision layer  
**Data:** synthetic Triagegeist benchmark — **research/demo only, not for clinical use**

---

## 1. What this package gives you

This folder is designed so the complete ML pipeline can be rerun on another Windows/macOS/Linux device from the original CSV files. It contains:

- deterministic data loading and joins;
- leakage-aware feature selection;
- 51-feature LightGBM training;
- a permanent 15% locked holdout;
- 5-fold stratified development cross-validation;
- out-of-fold temperature calibration;
- the asymmetric under-triage safety decision layer;
- a development-only safety trade-off table;
- development-only leave-one-site-out evaluation;
- locked-test classification, under-triage and calibration metrics;
- subgroup evaluation;
- confusion matrices and per-patient locked-test predictions;
- feature importance;
- a serialized model bundle consumed directly by the prototype;
- prediction on the unlabeled `test.csv`;
- the current Streamlit ED prototype.

The training script creates the exact runtime filenames expected by `risk_engine_ml.py` and `app_ml.py`:

```text
model/triage_model_bundle.joblib
model/metrics.json
model/feature_importance.csv
final_evaluation/*.csv / *.json / *.md
```

---

# 2. Implementation approach

## 2.1 Problem framing

PatientTriage.ai predicts an **ESI-like 5-level triage acuity distribution** from information available at triage. It then applies a separate safety decision policy that deliberately penalizes **under-triage** more strongly than over-triage.

The system is intentionally not a one-shot classifier. During the prototype, new/repeat vitals are sent through the same model again. The application then computes:

- new acuity probabilities;
- change in urgent probability, `ΔP(ESI 1/2)`;
- `ΔVitals / Δt`;
- uncertainty;
- assessment freshness;
- updated queue attention rank.

This is a **temporal acuity re-estimation prototype**, not a clinically validated future-deterioration model.

## 2.2 Data used


```text
train.csv
 └─ 80,000 labelled encounters

test.csv
 └─ 20,000 unlabelled encounters

patient_history.csv
 └─ 100,000 patient-history rows

chief_complaints.csv
 └─ 100,000 complaint rows
```

The pipeline joins `patient_history.csv` by `patient_id`. `chief_complaint_raw` is merged for audit/display only and is **not used by the primary model**.

## 2.3 Leakage controls

The following variables are explicitly excluded from ML fitting:

```text
patient_id
site_id
triage_nurse_id
triage_acuity              <- prediction target
chief_complaint_raw         <- excluded because synthetic text showed label-shortcut behavior
disposition                 <- post-triage leakage
ed_los_hours                <- post-triage leakage
news2_score                 <- kept as an independent runtime comparator, not an ML feature
arrival_hour/day/month/season
shift
language
insurance_type
```

`site_id` is excluded from prediction but retained in metadata so that site-held-out generalization can be tested.

---

# 3. Model architecture

```text
 TRIAGE-TIME PATIENT DATA
         │
         ├── demographics / arrival
         ├── structured complaint
         ├── vitals
         ├── GCS / pain
         ├── prior utilization
         └── patient-history flags
         │
         ▼
 FEATURE PIPELINE (51 features)
         │
         ├── MAP
         ├── pulse pressure
         ├── shock index
         └── BMI
         │
         ▼
 LIGHTGBM MULTICLASS MODEL
         │
         ▼
 P(ESI1), P(ESI2), P(ESI3), P(ESI4), P(ESI5)
         │
         ▼
 TEMPERATURE CALIBRATION
         │
         ├─────────────► Probability entropy / uncertainty
         │
         ▼
 ASYMMETRIC EXPECTED-COST DECISION
         │
         ▼
 RECOMMENDED ACUITY LEVEL
         │
         ├── deterministic critical-value guardrails
         ├── repeat-vital ML risk delta
         ├── ΔVitals / Δt
         ├── data-quality uncertainty
         ├── reassessment freshness
         └── independent protocol discordance check
         │
         ▼
 RISK ENVELOPE + LIVE SAFETY QUEUE
         │
         ▼
 CLINICIAN ACCEPT / MODIFY / ESCALATE
```

---

# 4. Final model features

The model uses **51 triage-time features**.

### Categorical (7)

```text
arrival_mode
age_group
sex
transport_origin
pain_location
mental_status_triage
chief_complaint_system
```

### Numeric / binary (44)

These include age; prior ED visits/admissions; active medication and comorbidity counts; systolic/diastolic BP; MAP; pulse pressure; HR; RR; temperature; SpO₂; GCS; pain; weight/height/BMI; shock index; and the 25 `hx_*` history flags.

The authoritative list lives in:

```text
scripts/pipeline_common.py -> FEATURES
```

---

# 5. Safety operating point

The raw LightGBM model produces class probabilities. Ordinary multiclass classification would choose:

```text
argmax P(ESI=k)
```

PatientTriage.ai instead calculates the expected cost of every possible recommended acuity level.

For actual level `i` and prediction `j`:

- correct prediction: cost = `0`
- over-triage: `1 × distance²`
- under-triage: `3 × distance²`

With the default 3× under-triage penalty, the cost matrix is:

```text
            Predicted
Actual      1   2   3   4   5
  1         0   3  12  27  48
  2         1   0   3  12  27
  3         4   1   0   3  12
  4         9   4   1   0   3
  5        16   9   4   1   0
```

The recommended priority is:

```text
argmin_j Σ_i P(ESI=i) × Cost(i,j)
```

A development-only safety frontier for weights `1.0 ... 5.0` is automatically written to:

```text
final_evaluation/safety_tradeoff_development_oof.csv
```

The app-compatible default remains **3.0** so the operating point is fixed before the locked holdout is scored.

---

# 6. Evaluation protocol

The validation process intentionally separates model development from final evaluation.

## 6.1 Locked holdout

The 80,000 labelled records are split first:

```text
68,000 development records (85%)
12,000 locked holdout records (15%)
```

The locked 12,000 rows are excluded from:

- model fitting;
- cross-validation;
- temperature fitting;
- safety weight analysis;
- site-held-out validation.

The final application model is trained on the 68,000 development records only, then evaluated once on the locked 12,000.

## 6.2 Five-fold cross-validation

Five-fold stratified CV is performed only inside the 68,000-row development pool.

The current reproducible run produced approximately:

```text
Safety accuracy mean        82.40%
Safety undertriage mean      3.05%
Severe undertriage mean      0.010%
QWK mean                     0.910
ESI-1 recall mean            94.96%
ESI-2 recall mean            96.75%
```

Exact per-device results may differ slightly with library versions and CPU architecture even with fixed seeds.

## 6.3 Site-held-out validation

`site_heldout_validate.py` repeats development-only validation by excluding one hospital site entirely from model fitting/calibration, then evaluating on that unseen site.

Current run:

```text
SITE-HEL-01 undertriage proxy   2.69%
SITE-HEL-02                     2.94%
SITE-OUL-01                     3.41%
SITE-TMP-01                     2.95%
SITE-TUR-01                     3.17%
```

This is a generalization stress test; it does not prove real-world external clinical validity because all sites are synthetic.

## 6.4 Locked-test metrics from the included reproducible run

### Plain argmax

```text
Accuracy                    85.49%
QWK                         0.930
Undertriage proxy            7.67%
Severe undertriage proxy     0.108%
Overtriage                   6.84%
ESI-1 recall                94.00%
ESI-2 recall                96.92%
```

### Safety operating point

```text
Accuracy                    82.30%
QWK                         0.909
Undertriage proxy            3.04%
Severe undertriage proxy     0.025%
Overtriage                  14.66%
ESI-1 recall                96.07%
ESI-2 recall                97.07%
```

### Urgent-probability quality

Urgent is defined as ESI 1 or 2. The pipeline also reports:

```text
ROC-AUC
PR-AUC
Brier score
10-bin Expected Calibration Error
```

Current locked ECE is approximately `0.0017` on this synthetic benchmark.

> **Important:** these are synthetic Triagegeist benchmark metrics, not clinical performance claims. The reported under-triage metric is an **acuity under-classification proxy** relative to `triage_acuity`.

---

# 7. Dependencies

## Required for ML training/evaluation

```text
Python 3.10+ recommended
numpy
pandas
scikit-learn
lightgbm
scipy
joblib
```

## Additional packages for the prototype UI

```text
streamlit
plotly
```

Install using:

```bash
pip install -r requirements.txt
```

A virtual environment is strongly recommended.

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 8. Execution instructions

## One-command full reproduction

### macOS / Linux

```bash
./run_training.sh
```

### Windows

```bat
run_training.bat
```

This performs:

```text
1. 5-fold development CV
2. OOF calibration
3. safety trade-off calculation
4. final development-only model fit
5. locked-holdout evaluation
6. site-held-out development validation
7. prediction of the unlabeled Triagegeist test.csv
```

The full workflow can take several minutes depending on CPU.

## Run each stage manually

### A. Train + CV + locked evaluation

```bash
python train_and_validate.py
```

### B. Site-held-out generalization

```bash
python site_heldout_validate.py
```

### C. Score the unlabeled test set

```bash
python predict_test.py
```

### D. Score one JSON patient

```bash
python score_patient.py example_patient.json
```

### E. Start the ED prototype

```bash
streamlit run app_ml.py
```

or use `run_app.sh` / `run_app.bat`.

---

# 9. Output files

After `run_training`, the important generated artifacts are:

```text
model/
├── triage_model_bundle.joblib
├── metrics.json
└── feature_importance.csv

final_evaluation/
├── FINAL_EVALUATION_REPORT.md
├── LOCKED_TEST_FINAL_METRICS.json
├── five_fold_per_fold_metrics.csv
├── five_fold_cv_summary.csv
├── safety_tradeoff_development_oof.csv
├── site_held_out_evaluation_development_only.csv
├── locked_test_confusion_matrix_argmax.csv
├── locked_test_confusion_matrix_safety.csv
├── locked_test_predictions.csv
├── locked_test_subgroup_metrics.csv
├── locked_test_urgent_calibration_bins.csv
├── test_predictions.csv
└── run_manifest.json
```

`run_manifest.json` records Python/platform/package versions for reproducibility.

---

# 10. How the serialized model is stored

`model/triage_model_bundle.joblib` contains more than the LightGBM trees. It stores:

```text
model                        trained LightGBM estimator
features                     exact 51-feature order
categorical / numeric        feature types
category_levels              categorical vocabularies + UNKNOWN
numeric_q05 / numeric_q95    used by Next Best Information
classes                      [1,2,3,4,5]
temperature                  probability calibration parameter
undertriage_weight           3.0 by default
overtriage_weight            1.0 by default
cost_matrix                  safety decision matrix
excluded_features            leakage/shortcut audit list
model_version
training_rows
locked_test_rows_excluded_from_training
selection_basis
```

The prototype runtime loads this artifact through `risk_engine_ml.py`.

---

# 11. Relationship between training model and live prototype

The training model predicts the **current acuity probability distribution**. The live prototype adds workflow intelligence around it:

```text
Initial patient snapshot
    -> trained model
    -> current probabilities / priority

New repeat vitals
    -> same trained model again
    -> new probabilities
    -> ΔP(urgent)
    -> ΔVitals/Δt
    -> updated Risk Envelope
    -> updated attention queue
```

The `+10 min` demonstration does **not predict future vitals**. It applies scripted repeat observations from `data/demo_events.csv` and demonstrates how the trained model reacts. The UI also supports entering real manual repeat vitals.

---

# 12. Project structure

```text
PatientTriage_AI_Reproducible_v06/
│
├── app_ml.py                         current Streamlit UI
├── risk_engine_ml.py                 runtime risk / uncertainty / temporal logic
├── train_and_validate.py             core training + CV + locked evaluation
├── site_heldout_validate.py          development-only cross-site test
├── predict_test.py                   score unlabeled test.csv
├── score_patient.py                  CLI score for one patient JSON
├── example_patient.json
├── requirements.txt
├── run_training.sh / .bat
├── run_app.sh / .bat
├── README.md
│
├── scripts/
│   └── pipeline_common.py            shared feature/model/metric functions
│
├── data_raw/
│   ├── train.csv
│   ├── test.csv
│   ├── patient_history.csv
│   └── chief_complaints.csv
│
├── data/
│   ├── demo_patients.csv
│   ├── demo_events.csv
│   ├── surge_patients.csv
│   └── manual_patients.csv
│
├── model/                            generated model artifacts
└── final_evaluation/                 generated validation artifacts
```

---

# 13. Reproducibility notes

1. Random seed is fixed at `20260829`.
2. The 15% locked holdout is created before development.
3. Site is excluded from model features.
4. LightGBM is configured with deterministic mode enabled.
5. The exact installed versions are written to `run_manifest.json`.
6. Small numeric differences can still occur between operating systems, LightGBM versions, compilers and CPU architectures.
7. Do not repeatedly use the locked holdout to select new hyperparameters. If you materially redesign the model after examining locked-test results, create a new untouched evaluation cohort or use nested development validation.

---

# 14. Clinical / scientific limitations

- Triagegeist is synthetic.
- The target is triage acuity, not mortality, ICU transfer or prospective deterioration.
- Dynamic risk is implemented as **re-estimation when new observations arrive**, not future-vital prediction.
- Deterministic critical-value rules and reassessment intervals in the demo are prototype rules, not validated clinical protocols.
- Model associations are not causal explanations.
- Real deployment would require external clinical data, prospective silent-mode validation, calibration at deployment sites, regulatory review, privacy/security implementation and licensed clinical governance.

**Do not use this package for real patient care.**

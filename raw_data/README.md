# Raw data placement

The training/evaluation scripts expect these files in this folder:

- `train.csv` — labelled Triagegeist training encounters
- `test.csv` — unlabelled Triagegeist test encounters
- `patient_history.csv` — history flags joined by `patient_id`
- `chief_complaints.csv` — raw complaint text used only for audit/display; not used as a primary model predictor

The supplied package already contains the files used for the reproducible run. If redistributing the code without data, keep this README and place the four CSVs here before executing the pipeline.

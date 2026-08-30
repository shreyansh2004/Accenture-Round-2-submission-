# PatientTriage.ai v0.5 — Demo Scenario Guide

This guide is for the Accenture Innovation Challenge Round 2 demonstration. The 20 base records are curated from the synthetic Triagegeist benchmark and then configured to exercise the workflow requirements. They are **not clinical cases and are not intended for patient care**.

## Base population design
- 20 active demo patients.
- 10 have a prior health record available; 10 intentionally have no prior record. When history is unavailable, history fields are stored as `NaN`/unknown, not as “no disease”.
- Includes pediatric, adult, geriatric, ambiguous, missing-data, noisy-sensor, zero-history, high-risk, low-risk and deterioration scenarios.
- 3× surge mode adds 40 more arrivals (60 total), with mixed history availability.

## Recommended live demo

### Start — show the Live Safety Queue
Point out that the board prioritizes clinical attention using current acuity, trajectory, uncertainty and freshness. The model prediction and queue-attention score are separate concepts.

### +10 minutes
Click **+10 min**.
- **P014**: respiratory deterioration begins. It remains a moderate-priority case initially, but the trajectory becomes worsening.
- **P016**: geriatric respiratory case worsens to critical physiology and triggers a hard safety alert.
- **P006**: initial BP was missing and SpO₂ signal was noisy. A repeat BP and verified pulse-ox arrive; uncertainty drops.
- **P009**: suspicious low SpO₂ is re-measured using a reliable probe and normalizes, demonstrating sensor-artifact handling instead of automatic escalation.

### +20 minutes
Click **+10 min** again.
- **P014**: continued deterioration drives a large increase in ML urgent-risk probability and moves the recommendation to Level 2 with a rapid-deterioration alert.
- **P003**: pediatric repeat vitals worsen and trigger reassessment.
- **P017**: subtle geriatric presentation develops objective deterioration and is reprioritized.
- **P011**: high-risk chest-pain patient receives a stable repeat; the system does not invent deterioration simply because time passed.

### +30 minutes
Click **+10 min** again.
- **P020**: immunosuppressed infection worsens during waiting and triggers a rising-risk alert.
- **P004**: low-risk pediatric repeat remains stable, demonstrating that the system can avoid unnecessary escalation.

## Manual interaction demonstrations

### Add a completely new patient
Open **Add Patient**. Enter patient/arrival information, complaint system, vitals, signal quality and optionally the complete prior-history feature set used by the ML model. If no history exists, turn **Prior health record available** off. The row is added to the live board and scored immediately.

### Record arbitrary repeat vitals
Open **Record Repeat Vitals**, choose any active patient, enter a new observation and specify the elapsed time. The system:
1. stores the previous snapshot,
2. re-runs the same LightGBM model,
3. calculates `ΔVitals/Δt`,
4. calculates the change in `P(ESI 1) + P(ESI 2)`, and
5. updates the Risk Envelope and live queue.

### Clinician override / audit
Open **Clinician Actions**, select a patient, accept or change the AI-recommended level, enter a reason, and record the decision. The session audit trail captures the AI level, clinician level, confidence, urgent probability, rationale, model version and timestamp.

### Remove a patient
Use **Remove patient from live list** for a session-level disposition. If a demo patient row is physically deleted from `data/demo_patients.csv`, the next app rerun automatically removes that patient from the dashboard source as well.

## What the scripted +10-minute button means
The button does **not** predict future vital signs. `data/demo_events.csv` supplies controlled simulated repeat observations. PatientTriage.ai's response to those new observations — re-inference, ML Risk Delta, ΔVitals/Δt, uncertainty, freshness and queue reprioritization — is calculated at runtime.

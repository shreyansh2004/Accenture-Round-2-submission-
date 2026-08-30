# PatientTriage.ai — Final Reproducible Evaluation Report

- Model version: `triagegeist-lgbm-safety-v0.6-repro`
- Labelled encounters: 80,000
- Development pool: 68,000
- Locked holdout: 12,000
- Undertriage penalty: 3×
- Temperature: 1.0998

## Locked holdout

| Metric | Plain argmax | Safety operating point |
|---|---:|---:|
| Accuracy | 0.8549 | 0.8230 |
| Macro F1 | 0.8729 | 0.8307 |
| QWK | 0.9301 | 0.9090 |
| Undertriage proxy | 0.0767 | 0.0304 |
| Severe undertriage proxy | 0.0011 | 0.0003 |
| Overtriage | 0.0684 | 0.1466 |
| ESI-1 recall | 0.9400 | 0.9607 |
| ESI-2 recall | 0.9692 | 0.9707 |

## Urgent-probability quality (ESI 1/2)

- ROC-AUC: 0.999640
- PR-AUC: 0.998832
- Brier score: 0.004446
- 10-bin ECE: 0.001677

## Five-fold development CV

- Safety accuracy: 0.8240 ± 0.0030
- Undertriage proxy: 0.0305 ± 0.0004
- Severe undertriage proxy: 0.000103 ± 0.000099
- QWK: 0.9097 ± 0.0020
- ESI-1 recall: 0.9496
- ESI-2 recall: 0.9675

## Development-only site-held-out evaluation

| Held-out site | N | Accuracy | Undertriage | Severe undertriage | QWK | ESI-1 recall | ESI-2 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| SITE-HEL-01 | 13766 | 0.8195 | 0.0269 | 0.000000 | 0.9082 | 0.9639 | 0.9606 |
| SITE-HEL-02 | 13518 | 0.8259 | 0.0294 | 0.000148 | 0.9106 | 0.9526 | 0.9680 |
| SITE-OUL-01 | 13452 | 0.8227 | 0.0341 | 0.000297 | 0.9087 | 0.9367 | 0.9653 |
| SITE-TMP-01 | 13458 | 0.8244 | 0.0295 | 0.000149 | 0.9096 | 0.9666 | 0.9670 |
| SITE-TUR-01 | 13806 | 0.8221 | 0.0317 | 0.000072 | 0.9078 | 0.9225 | 0.9646 |

## Validation protocol

1. The locked 15% holdout is created before development.
2. Five-fold CV, probability calibration and safety analysis use development data only.
3. Site-held-out tests are performed only inside the development pool.
4. The final runtime model is fit on the development pool only.
5. The locked holdout is evaluated after the operating point is fixed.

## Interpretation limitation

Triagegeist is synthetic and the target is triage acuity. Therefore the undertriage metric is an acuity under-classification proxy, not a demonstrated real-hospital harm rate. External and prospective clinical validation would be required before any clinical deployment.

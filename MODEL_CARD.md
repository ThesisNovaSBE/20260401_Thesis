# Model Card — Readmission Prediction Pipeline

## Model Details

- **Stage 1:** Classical ML classifiers (Logistic Regression, XGBoost, HistGradientBoosting) on structured EHR features
- **Stage 2:** Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`) on discharge notes — 4096-token context, trained on MIMIC-III
- **Stage 3 (optional):** Local generative explanation via Ollama (`phi4-mini`)
- **Developed by:** Nova SBE thesis team (M.Sc. Business Analytics)
- **Model type:** Two-stage triage-and-verify classification pipeline
- **Language:** English (clinical notes)

## Intended Use

- **Primary use:** Predict unplanned 30-day hospital readmissions from MIMIC-IV data
- **Primary users:** Clinical decision support research
- **Out of scope:** Direct clinical deployment without further validation

## Training Data

- MIMIC-IV v3.1 (structured tables) — credentialed access via PhysioNet
- MIMIC-IV-Note (discharge summaries) — credentialed access via PhysioNet
- Population: Adult patients (age >= 18), excluding in-hospital deaths and elective readmissions

## Metrics

| Metric | Stage 1 (Triage) | Stage 1+2 (Verified) |
|--------|-------------------|----------------------|
| Recall | TBD (target: 0.85+) | TBD |
| Precision | TBD | TBD (improved) |
| AUROC | TBD | TBD |
| F1 | TBD | TBD |

## Limitations

- Trained and evaluated on MIMIC-IV only (single US academic medical center)
- Temporal and demographic generalization not validated
- Not intended for real-time clinical use

## Ethical Considerations

- MIMIC-IV data is de-identified but originates from real patient encounters
- All data handling follows the PhysioNet Data Use Agreement
- Readmission prediction models may encode demographic biases present in the training data — fairness analysis is planned

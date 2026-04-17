# Credit Risk PD Model (Python)

This project trains a Probability of Default (PD) model using a calibrated logistic regression pipeline.
Outputs include:
- Model artifact (`models/pd_model.joblib`)
- Metrics (`reports/metrics.json`) with AUC, KS, Brier
- Scored applications (`reports/scored_applications.csv`) with PD + risk grade
- Stress test report (`reports/stress_report.csv`)
- ROC + calibration plots (`reports/*.png`)

## Setup
```bash
pip install -r requirements.txt

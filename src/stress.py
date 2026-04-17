from __future__ import annotations

import joblib
import numpy as np
import pandas as pd


def apply_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple scenario stress. Edit column names to match your dataset.
    Stress logic is intentionally transparent (interview-friendly).
    """
    stressed = df.copy()

    # Income down 10%
    for col in ["annual_income", "income"]:
        if col in stressed.columns:
            stressed[col] = stressed[col] * 0.90

    # DTI up +3 percentage points
    for col in ["dti", "debt_to_income"]:
        if col in stressed.columns:
            stressed[col] = stressed[col] + 3.0

    # Utilisation up +10pp (cap at 100 if it's a percentage)
    for col in ["utilization", "revol_util"]:
        if col in stressed.columns:
            stressed[col] = np.minimum(stressed[col] + 10.0, 100.0)

    return stressed


def main() -> None:
    artifact = joblib.load("../models/pd_model.joblib")
    model = artifact["model"]
    target = artifact.get("target", "default")

    df = pd.read_csv("../data/raw/loans.csv")

    # If the raw file includes target, drop it for scoring
    X = df.drop(columns=[target]) if target in df.columns else df

    base_pd = model.predict_proba(X)[:, 1]
    stressed_X = apply_stress(X)
    stressed_pd = model.predict_proba(stressed_X)[:, 1]

    report = pd.DataFrame({
        "base_pd_mean": [float(base_pd.mean())],
        "stressed_pd_mean": [float(stressed_pd.mean())],
        "pd_uplift_bp": [float((stressed_pd.mean() - base_pd.mean()) * 10000.0)],  # basis points
    })

    out_path = "../reports/stress_report.csv"
    report.to_csv(out_path, index=False)

    print(f"Saved stress report to: {out_path}")
    print(report)


if __name__ == "__main__":
    main()

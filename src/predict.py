from __future__ import annotations

import joblib
import pandas as pd

from utils import pd_to_grade


def main() -> None:
    artifact = joblib.load("../models/pd_model.joblib")
    model = artifact["model"]

    # Provide applications with same feature columns as training data (minus target)
    new_apps = pd.read_csv("../data/raw/new_applications.csv")

    pd_scores = model.predict_proba(new_apps)[:, 1]

    out = new_apps.copy()
    out["pd"] = pd_scores
    out["risk_grade"] = [pd_to_grade(x) for x in pd_scores]

    out_path = "../reports/scored_applications.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved scored applications to: {out_path}")
    print(out[["pd", "risk_grade"]].head(10))


if __name__ == "__main__":
    main()

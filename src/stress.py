from __future__ import annotations

import joblib
import pandas as pd

from utils import from_root, load_config, load_dataset, pd_to_grade

# Each positive current-account balance band moves one band lower
LOWER_BALANCE = {">= 200 DM or salary account": "0-200 DM", "0-200 DM": "< 0 DM"}


def apply_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Liquidity shock, intentionally simple and transparent: every borrower with
    a positive current-account balance drops one balance band.

    Only variables whose default rate moves in a consistent direction are
    stressed. In this data the instalment rate does not: borrowers paying
    >= 35% of income default less often than those paying < 20%, so
    "heavier instalments" would lower predicted risk.
    """
    stressed = df.copy()
    stressed["checking_status"] = stressed["checking_status"].replace(LOWER_BALANCE)
    return stressed


def main() -> None:
    cfg = load_config()
    model = joblib.load(from_root(cfg["output"]["model_path"]))["model"]
    X, _ = load_dataset(cfg)

    base_pd = model.predict_proba(X)[:, 1]
    stressed_pd = model.predict_proba(apply_stress(X))[:, 1]
    worst_grade = list(cfg["risk_bands"])[-1]

    def share_worst(pds) -> float:
        return round(float(pd.Series([pd_to_grade(p, cfg["risk_bands"]) for p in pds]).eq(worst_grade).mean()), 4)

    report = pd.DataFrame({
        "base_pd_mean": [round(float(base_pd.mean()), 4)],
        "stressed_pd_mean": [round(float(stressed_pd.mean()), 4)],
        "pd_uplift_bp": [round(float((stressed_pd.mean() - base_pd.mean()) * 10000.0), 1)],  # basis points
        f"share_grade_{worst_grade}_base": [share_worst(base_pd)],
        f"share_grade_{worst_grade}_stressed": [share_worst(stressed_pd)],
    })

    out_path = from_root(cfg["output"]["stress_path"])
    report.to_csv(out_path, index=False)

    print(f"Saved stress report to: {out_path}")
    print(report.T)


if __name__ == "__main__":
    main()

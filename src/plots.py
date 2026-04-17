from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from utils import load_config, map_default_label


def main() -> None:
    cfg = load_config("config.yaml")
    target = cfg["data"]["target"]

    # Load dataset and (optional) label mapping to ensure consistent target
    df = pd.read_csv(cfg["data"]["raw_path"])
    df = map_default_label(df, cfg)
    df = df.dropna(subset=[target]).copy()

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    artifact = joblib.load(cfg["output"]["model_path"])
    model = artifact["model"]

    p_test = model.predict_proba(X_test)[:, 1]

    # ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, p_test)
    roc_path = "../reports/roc_curve.png"
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {roc_path}")

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    cal_path = "../reports/calibration_curve.png"
    plt.savefig(cal_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cal_path}")


if __name__ == "__main__":
    main()

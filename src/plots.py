from __future__ import annotations

import joblib
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from utils import from_root, load_config, load_dataset


def main() -> None:
    cfg = load_config()
    X, y = load_dataset(cfg)

    # Same split as train.py, so the plots use the held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    artifact = joblib.load(from_root(cfg["output"]["model_path"]))
    model = artifact["model"]

    p_test = model.predict_proba(X_test)[:, 1]

    # ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, p_test)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("ROC Curve (test set)")
    roc_path = from_root(cfg["output"]["roc_path"])
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
    plt.title("Calibration Curve (test set)")
    cal_path = from_root(cfg["output"]["calibration_path"])
    plt.savefig(cal_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cal_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import ks_stat, load_config, map_default_label, save_json


def main() -> None:
    cfg = load_config("config.yaml")

    raw_path = cfg["data"]["raw_path"]
    target = cfg["data"]["target"]

    df = pd.read_csv(raw_path)
    df = map_default_label(df, cfg)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"If your labels are strings, enable label_mapping in config.yaml."
        )

    df = df.dropna(subset=[target]).copy()
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Identify feature types
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipelines
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    base_clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        class_weight=cfg["model"]["class_weight"],
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", base_clf),
    ])

    # --- Make CV robust to small datasets ---
    # Minimum: 2 folds. Maximum: 5 folds, but never more than the training samples.
    cv_folds = min(5, max(2, len(X_train)))

    # If you have very few samples, ROC-AUC may fail (needs both classes in each fold).
    # Fall back to "neg_log_loss" which is more stable for tiny datasets.
    scoring = "roc_auc"
    if y_train.nunique() < 2 or len(y_train) < 10:
        scoring = "neg_log_loss"

    grid = GridSearchCV(
        pipe,
        param_grid={"clf__C": cfg["model"]["c_values"]},
        scoring=scoring,
        cv=cv_folds,
        n_jobs=-1
    )

        grid.fit(X_train, y_train)
    best_pipe: Pipeline = grid.best_estimator_

    # Calibrate probabilities (critical for PD)
    method = cfg["model"].get("calibration_method", "isotonic")
    calibrated = CalibratedClassifierCV(best_pipe, method=method, cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluate
    p_test = calibrated.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))
    ks = ks_stat(y_test.to_numpy(), p_test)

    metrics = {
        "auc": auc,
        "brier": brier,
        "ks": ks,
        "best_params": grid.best_params_,
        "calibration_method": method,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
    }

    print("Training complete. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save model artifact
    artifact = {
        "model": calibrated,
        "target": target,
        "feature_info": {"numeric": num_cols, "categorical": cat_cols},
        "config": cfg,
    }
    joblib.dump(artifact, cfg["output"]["model_path"])
    save_json(cfg["output"]["metrics_path"], metrics)

    # Optional: export top coefficients (interpretable, very “credit risk”)
    # NOTE: coefficients only exist for the underlying logistic regression inside best_pipe.
    try:
        prep = best_pipe.named_steps["prep"]
        clf = best_pipe.named_steps["clf"]
        feature_names = prep.get_feature_names_out()
        coefs = clf.coef_.ravel()

        top = pd.DataFrame({"feature": feature_names, "coef": coefs})
        top["abs_coef"] = top["coef"].abs()
        top = top.sort_values("abs_coef", ascending=False).head(25)
        top.to_csv(cfg["output"]["feature_importance_path"], index=False)
    except Exception as e:
        print(f"Warning: could not export coefficients ({e}).")

    print(f"\nSaved model to: {cfg['output']['model_path']}")
    print(f"Saved metrics to: {cfg['output']['metrics_path']}")


if __name__ == "__main__":
    main()

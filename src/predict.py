from __future__ import annotations

import joblib
import pandas as pd

from utils import features, from_root, load_config, pd_to_grade


def main() -> None:
    cfg = load_config()
    artifact = joblib.load(from_root(cfg["output"]["model_path"]))
    model = artifact["model"]

    # Applications with the same columns as the training data, minus the target
    new_apps = pd.read_csv(from_root(cfg["data"]["new_applications_path"]))

    pd_scores = model.predict_proba(features(new_apps, cfg))[:, 1]

    out = new_apps.copy()
    out["pd"] = pd_scores.round(4)
    out["risk_grade"] = [pd_to_grade(x, cfg["risk_bands"]) for x in pd_scores]

    out_path = from_root(cfg["output"]["scored_path"])
    out.to_csv(out_path, index=False)

    print(f"Saved scored applications to: {out_path}")
    print(out[["pd", "risk_grade"]].head(10))


if __name__ == "__main__":
    main()

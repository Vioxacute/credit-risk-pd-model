from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def from_root(path: str) -> Path:
    """Resolve a path from config.yaml, which is relative to the repository root."""
    return ROOT / path


def load_config(name: str = "config.yaml") -> dict:
    with open(Path(__file__).parent / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic for binary classification scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    order = np.argsort(y_score)
    y_true = y_true[order]

    pos = (y_true == 1).astype(int)
    neg = (y_true == 0).astype(int)

    cum_pos = np.cumsum(pos) / max(pos.sum(), 1)
    cum_neg = np.cumsum(neg) / max(neg.sum(), 1)

    return float(np.max(np.abs(cum_pos - cum_neg)))


def map_default_label(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Optionally map string status labels to a binary 'default' column.
    Controlled by cfg['label_mapping'].
    """
    lm = cfg.get("label_mapping", {}) or {}
    if not lm.get("enabled", False):
        return df

    source_col = lm["source_col"]
    default_values = set(lm["default_values"])
    nondefault_values = set(lm["nondefault_values"])

    if source_col not in df.columns:
        raise ValueError(f"label_mapping.source_col '{source_col}' not found in dataset")

    # Create/overwrite cfg.data.target column with mapped 0/1
    target = cfg["data"]["target"]
    status = df[source_col].astype(str)

    is_default = status.isin(default_values)
    is_nondefault = status.isin(nondefault_values)

    mapped = pd.Series(np.nan, index=df.index, dtype="float")
    mapped[is_default] = 1.0
    mapped[is_nondefault] = 0.0

    df = df.copy()
    df[target] = mapped
    return df


def features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Model inputs: everything except the target and the excluded columns."""
    drop = [cfg["data"]["target"], *cfg["data"].get("drop_columns", [])]
    return df.drop(columns=[c for c in drop if c in df.columns])


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training data as (features, target), applying the optional label mapping."""
    target = cfg["data"]["target"]
    df = map_default_label(pd.read_csv(from_root(cfg["data"]["raw_path"])), cfg)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"If your labels are strings, enable label_mapping in config.yaml."
        )

    df = df.dropna(subset=[target]).copy()
    return features(df, cfg), df[target].astype(int)


def pd_to_grade(pd_value: float, bands: dict[str, float]) -> str:
    """Risk grade for a PD, given each grade's upper PD bound in ascending order."""
    for grade, upper in bands.items():
        if pd_value < upper:
            return grade
    return list(bands)[-1]

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, payload: dict) -> None:
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


@dataclass(frozen=True)
class RiskBand:
    lo: float
    hi: float
    grade: str


DEFAULT_RISK_BANDS: List[RiskBand] = [
    RiskBand(0.00, 0.01, "A"),
    RiskBand(0.01, 0.03, "B"),
    RiskBand(0.03, 0.07, "C"),
    RiskBand(0.07, 0.15, "D"),
    RiskBand(0.15, 1.01, "E"),
]


def pd_to_grade(pd_value: float, bands: List[RiskBand] | None = None) -> str:
    bands = bands or DEFAULT_RISK_BANDS
    for b in bands:
        if b.lo <= pd_value < b.hi:
            return b.grade
    return bands[-1].grade

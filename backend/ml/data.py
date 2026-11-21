"""Loading and splitting the engine-health dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.features import LABEL_COLUMN, SENSOR_COLUMNS, normalise_columns


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_dataset(path: Path, positive_label: int = 1) -> pd.DataFrame:
    """Read the CSV, normalise headers, drop duplicates and build a 0/1 fault label."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first (see data/README.md)."
        )
    df = normalise_columns(pd.read_csv(path))
    missing = [c for c in [*SENSOR_COLUMNS, LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[[*SENSOR_COLUMNS, LABEL_COLUMN]].drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=[LABEL_COLUMN])
    df["fault"] = (df[LABEL_COLUMN].astype(int) == positive_label).astype(int)
    return df


def dataset_info(path: Path) -> dict[str, str]:
    """Provenance written by scripts/download_data.py next to the CSV."""
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        return dict(json.loads(meta.read_text(encoding="utf-8")))
    return {"source": "unknown"}


def split(df: pd.DataFrame, val_size: float, test_size: float, random_state: int) -> Splits:
    """Stratified train/validation/test split (default 70/15/15)."""
    X, y = df[SENSOR_COLUMNS], df["fault"]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=rel_val, stratify=y_rest, random_state=random_state
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)

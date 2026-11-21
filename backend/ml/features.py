"""Feature definitions and the fitted preprocessing shared by training and serving.

Training fits a :class:`Preprocessor` on the training split only and saves it to
``artifacts/scaler.joblib``. The API loads that same file, so the transformation
applied at serving time is byte-for-byte the one the model was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Raw sensor columns, in the order the API and the UI use them.
SENSOR_COLUMNS: list[str] = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]

LABEL_COLUMN = "engine_condition"

# Map from the public Kaggle dataset's headers to our snake_case names.
RAW_COLUMN_ALIASES: dict[str, str] = {
    "Engine rpm": "engine_rpm",
    "Lub oil pressure": "lub_oil_pressure",
    "Fuel pressure": "fuel_pressure",
    "Coolant pressure": "coolant_pressure",
    "lub oil temp": "lub_oil_temp",
    "Coolant temp": "coolant_temp",
    "Engine Condition": LABEL_COLUMN,
}

SENSOR_LABELS: dict[str, str] = {
    "engine_rpm": "Engine RPM",
    "lub_oil_pressure": "Lubricant oil pressure",
    "fuel_pressure": "Fuel pressure",
    "coolant_pressure": "Coolant pressure",
    "lub_oil_temp": "Lubricant oil temperature",
    "coolant_temp": "Coolant temperature",
}

RPM_BAND_EDGES: tuple[float, float] = (600.0, 1200.0)
ENGINEERED_COLUMNS: list[str] = [
    "oil_fuel_ratio",
    "coolant_oil_ratio",
    "temp_delta",
    "rpm_band_low",
    "rpm_band_mid",
    "rpm_band_high",
]
FEATURE_COLUMNS: list[str] = SENSOR_COLUMNS + ENGINEERED_COLUMNS

_EPS = 1e-3


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename dataset headers to snake_case; unknown columns pass through."""
    renamed = df.rename(columns={k: v for k, v in RAW_COLUMN_ALIASES.items() if k in df.columns})
    renamed.columns = [str(c).strip().lower().replace(" ", "_") for c in renamed.columns]
    return renamed


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pressure ratios, the oil/coolant temperature delta and one-hot RPM bands."""
    out = df.copy()
    out["oil_fuel_ratio"] = out["lub_oil_pressure"] / (out["fuel_pressure"] + _EPS)
    out["coolant_oil_ratio"] = out["coolant_pressure"] / (out["lub_oil_pressure"] + _EPS)
    out["temp_delta"] = out["lub_oil_temp"] - out["coolant_temp"]
    low, high = RPM_BAND_EDGES
    rpm = out["engine_rpm"]
    out["rpm_band_low"] = (rpm < low).astype(float)
    out["rpm_band_mid"] = ((rpm >= low) & (rpm < high)).astype(float)
    out["rpm_band_high"] = (rpm >= high).astype(float)
    return out


@dataclass
class Preprocessor:
    """Median imputation, 1st/99th percentile clipping, feature engineering, scaling."""

    clip_low: dict[str, float] = field(default_factory=dict)
    clip_high: dict[str, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    healthy_mean: dict[str, float] = field(default_factory=dict)
    healthy_std: dict[str, float] = field(default_factory=dict)
    scaler: StandardScaler | None = None
    feature_order: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame, y: pd.Series | np.ndarray) -> Preprocessor:
        raw = df[SENSOR_COLUMNS].astype(float)
        self.medians = raw.median().to_dict()
        filled = raw.fillna(self.medians)
        self.clip_low = filled.quantile(0.01).to_dict()
        self.clip_high = filled.quantile(0.99).to_dict()
        clipped = self._clip(filled)
        y_arr = np.asarray(y)
        healthy = clipped[y_arr == 0]
        self.healthy_mean = healthy.mean().to_dict()
        self.healthy_std = healthy.std(ddof=0).replace(0, 1.0).to_dict()
        features = add_engineered_features(clipped)[self.feature_order]
        self.scaler = StandardScaler().fit(features.to_numpy())
        return self

    # ---------------------------------------------------------------- transform
    def _clip(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in SENSOR_COLUMNS:
            out[col] = out[col].clip(self.clip_low[col], self.clip_high[col])
        return out

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute and clip raw sensor readings (no scaling)."""
        raw = df[SENSOR_COLUMNS].astype(float).fillna(self.medians)
        return self._clip(raw)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Preprocessor is not fitted")
        features = add_engineered_features(self.clean(df))[self.feature_order]
        return np.asarray(self.scaler.transform(features.to_numpy()), dtype=np.float32)

    def deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signed deviation of each sensor from the healthy-class mean, in std units."""
        cleaned = self.clean(df)
        mean = pd.Series(self.healthy_mean)
        std = pd.Series(self.healthy_std)
        return (cleaned[SENSOR_COLUMNS] - mean[SENSOR_COLUMNS]) / std[SENSOR_COLUMNS]

    # -------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        joblib.dump(self.to_dict(), path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "medians": self.medians,
            "healthy_mean": self.healthy_mean,
            "healthy_std": self.healthy_std,
            "scaler": self.scaler,
            "feature_order": self.feature_order,
        }

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        data = joblib.load(path)
        return cls(**data)

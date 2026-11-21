from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import (
    FEATURE_COLUMNS,
    SENSOR_COLUMNS,
    Preprocessor,
    add_engineered_features,
    normalise_columns,
)


def _frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(10, 2, n) for c in SENSOR_COLUMNS})
    df["engine_rpm"] = rng.uniform(100, 2000, n)
    y = (rng.random(n) < 0.6).astype(int)
    return df, y


def test_normalise_columns_maps_kaggle_headers() -> None:
    df = pd.DataFrame(columns=["Engine rpm", "Lub oil pressure", "Engine Condition"])
    assert list(normalise_columns(df).columns) == ["engine_rpm", "lub_oil_pressure", "engine_condition"]


def test_engineered_features() -> None:
    row = {
        **dict.fromkeys(SENSOR_COLUMNS, 1.0),
        "engine_rpm": 700.0,
        "lub_oil_temp": 80.0,
        "coolant_temp": 75.0,
    }
    out = add_engineered_features(pd.DataFrame([row])).iloc[0]
    assert out["temp_delta"] == 5.0
    assert out["rpm_band_mid"] == 1.0 and out["rpm_band_low"] == 0.0
    assert abs(out["oil_fuel_ratio"] - 1.0) < 0.01


def test_preprocessor_scales_train_to_zero_mean() -> None:
    df, y = _frame()
    pre = Preprocessor().fit(df, y)
    X = pre.transform(df)
    assert X.shape == (len(df), len(FEATURE_COLUMNS))
    assert X.dtype == np.float32
    np.testing.assert_allclose(X[:, :6].mean(axis=0), 0, atol=1e-4)


def test_preprocessor_clips_and_imputes() -> None:
    df, y = _frame()
    pre = Preprocessor().fit(df, y)
    extreme = df.head(1).copy()
    extreme["fuel_pressure"] = 1e6
    extreme["coolant_temp"] = np.nan
    cleaned = pre.clean(extreme)
    assert cleaned["fuel_pressure"].iloc[0] == pre.clip_high["fuel_pressure"]
    assert not cleaned.isna().any().any()


def test_preprocessor_roundtrip(tmp_path: Path) -> None:
    df, y = _frame()
    pre = Preprocessor().fit(df, y)
    pre.save(tmp_path / "p.joblib")
    loaded = Preprocessor.load(tmp_path / "p.joblib")
    np.testing.assert_array_equal(pre.transform(df), loaded.transform(df))

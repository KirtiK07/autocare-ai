"""FaultPredictor: loads the trained artifacts and scores sensor readings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from app.schemas import FeatureContribution, Prediction, RiskBand, SensorReading
from ml.features import SENSOR_COLUMNS, SENSOR_LABELS, Preprocessor
from ml.nets import EngineFaultNet


class ModelNotReadyError(RuntimeError):
    """Raised when artifacts are missing (run `python -m ml.train`)."""


class FaultPredictor:
    """Wraps the preprocessor + served model saved by ``ml.train``.

    The served model is whichever won on validation F1 (PyTorch or a scikit-learn
    baseline); ``model_meta.json`` records which, plus the tuned threshold.
    """

    def __init__(self, artifacts_dir: Path, top_k: int = 3):
        meta_path = artifacts_dir / "model_meta.json"
        if not meta_path.exists():
            raise ModelNotReadyError(f"{meta_path} not found; run `python -m ml.train` first")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.preprocessor = Preprocessor.load(artifacts_dir / "scaler.joblib")
        self.threshold = float(self.meta["threshold"])
        self.medium_cutoff = float(self.meta["risk_bands"]["medium"])
        self.version = str(self.meta["model_version"])
        self.served = str(self.meta["served_model"])
        self.top_k = top_k
        self._net: EngineFaultNet | None = None
        self._sk: Any = None
        if self.served == "pytorch":
            net = EngineFaultNet(
                int(self.meta["n_features"]),
                hidden=tuple(self.meta["pytorch"]["hidden"]),  # type: ignore[arg-type]
                dropout=float(self.meta["pytorch"]["dropout"]),
            )
            net.load_state_dict(
                torch.load(artifacts_dir / "fault_net.pt", map_location="cpu", weights_only=True)
            )
            net.eval()
            self._net = net
        else:
            self._sk = joblib.load(artifacts_dir / "sklearn_model.joblib")

    # ----------------------------------------------------------------- scoring
    def _proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is not None:
            with torch.no_grad():
                return torch.sigmoid(self._net(torch.as_tensor(X))).numpy()
        return np.asarray(self._sk.predict_proba(X)[:, 1])

    def risk_band(self, probability: float) -> RiskBand:
        if probability >= self.threshold:
            return "high"
        if probability >= self.medium_cutoff:
            return "medium"
        return "low"

    def predict(self, reading: SensorReading) -> Prediction:
        return self.predict_many([reading])[0]

    def predict_many(self, readings: list[SensorReading]) -> list[Prediction]:
        if not readings:
            return []
        df = pd.DataFrame([r.model_dump() for r in readings])[SENSOR_COLUMNS]
        proba = np.clip(self._proba(self.preprocessor.transform(df)), 0.0, 1.0)
        deviations = self.preprocessor.deviations(df)
        impacts = self._occlusion_impacts(df, proba)
        out: list[Prediction] = []
        for i, p in enumerate(proba):
            p = float(p)
            dev = deviations.iloc[i]
            impact = impacts.iloc[i]
            # Rank by how much each reading pushes the probability up (toward "faulty").
            top = impact.sort_values(ascending=False).index[: self.top_k]
            contributions = [
                FeatureContribution(
                    feature=f,
                    label=SENSOR_LABELS[f],
                    value=float(df.iloc[i][f]),
                    impact=round(float(impact[f]), 4),
                    deviation=round(float(dev[f]), 3),
                    direction="high" if dev[f] >= 0 else "low",
                )
                for f in top
            ]
            out.append(
                Prediction(
                    probability=round(p, 4),
                    label="faulty" if p >= self.threshold else "healthy",
                    risk_band=self.risk_band(p),
                    top_features=contributions,
                    model_version=self.version,
                )
            )
        return out

    def _occlusion_impacts(self, df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
        """Model-faithful local attribution by occlusion.

        For each sensor, replace its value with the healthy-class mean and re-score;
        impact = original probability - probability with that sensor "made healthy".
        Positive impact means the reading pushes the prediction toward "faulty".
        """
        healthy = self.preprocessor.healthy_mean
        frames = []
        for col in SENSOR_COLUMNS:
            masked = df.copy()
            masked[col] = healthy[col]
            frames.append(masked)
        stacked = pd.concat(frames, ignore_index=True)
        masked_proba = np.clip(self._proba(self.preprocessor.transform(stacked)), 0.0, 1.0)
        n = len(df)
        data = {col: proba - masked_proba[j * n : (j + 1) * n] for j, col in enumerate(SENSOR_COLUMNS)}
        return pd.DataFrame(data)

    # -------------------------------------------------------------- metadata
    def samples(self) -> dict[str, dict[str, float]]:
        return dict(self.meta.get("samples", {}))

    def sensor_ranges(self) -> dict[str, Any]:
        return dict(self.meta.get("sensor_ranges", {}))

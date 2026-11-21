import json

from app.schemas import SensorReading
from app.services.fault_predictor import FaultPredictor
from tests.conftest import ARTIFACTS, SAMPLE, needs_artifacts


@needs_artifacts
def test_predict_shape_and_range(predictor: FaultPredictor) -> None:
    p = predictor.predict(SensorReading(**SAMPLE))
    assert 0.0 <= p.probability <= 1.0
    assert p.risk_band in {"low", "medium", "high"}
    assert len(p.top_features) == 3
    assert (p.label == "faulty") == (p.probability >= predictor.threshold)


@needs_artifacts
def test_samples_classified_as_labelled(predictor: FaultPredictor) -> None:
    s = predictor.samples()
    assert predictor.predict(SensorReading(**s["healthy"])).label == "healthy"
    assert predictor.predict(SensorReading(**s["faulty"])).label == "faulty"


@needs_artifacts
def test_batch_matches_single(predictor: FaultPredictor) -> None:
    readings = [SensorReading(**SAMPLE), SensorReading(**predictor.samples()["healthy"])]
    batch = predictor.predict_many(readings)
    assert [b.probability for b in batch] == [predictor.predict(r).probability for r in readings]


@needs_artifacts
def test_model_beats_dummy_baseline() -> None:
    """Model sanity gate: the served model must clearly beat chance / the majority-class baseline.

    F1 is not compared: when faults are the majority class, "always faulty" already has a high F1.
    """
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    by_key = {m["key"]: m for m in metrics["models"]}
    served = by_key[metrics["served_model"]]["test"]
    dummy = by_key["dummy"]["test"]
    assert served["roc_auc"] > dummy["roc_auc"] + 0.1
    assert served["balanced_accuracy"] > dummy["balanced_accuracy"] + 0.05
    assert served["mcc"] > 0.1


@needs_artifacts
def test_occlusion_impacts_explain_probability(predictor: FaultPredictor) -> None:
    p = predictor.predict(SensorReading(**predictor.samples()["faulty"]))
    impacts = [f.impact for f in p.top_features]
    assert impacts == sorted(impacts, reverse=True)
    assert impacts[0] > 0  # the top driver raises the fault probability
    assert all(abs(i) <= 1 for i in impacts)

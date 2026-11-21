import numpy as np

from ml.evaluate import classification_metrics, tune_threshold


def test_threshold_respects_recall_floor() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.6).astype(int)
    proba = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    t = tune_threshold(y, proba, min_recall=0.85)
    assert classification_metrics(y, proba, t)["recall"] >= 0.85


def test_metrics_keys() -> None:
    m = classification_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.4, 0.6]), 0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert set(m) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}

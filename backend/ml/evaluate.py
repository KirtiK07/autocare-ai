"""Metrics, threshold tuning and permutation importance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ProbaFn = Callable[[np.ndarray], np.ndarray]


def tune_threshold(y_true: np.ndarray, proba: np.ndarray, min_recall: float = 0.85) -> float:
    """Highest-precision threshold on validation whose recall is still >= ``min_recall``.

    Missing a real fault costs more than a false alarm, so recall is a hard floor;
    within that floor we take the threshold with the fewest false alarms. (Maximising
    F1/F2 directly degenerates to "always faulty" when faults are the majority class.)
    """
    best_t, best_key = 0.5, (-1.0, -1.0)
    for t in np.round(np.arange(0.02, 0.981, 0.01), 2):
        pred = proba >= t
        rec = recall_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        key = (float(precision_score(y_true, pred, zero_division=0)), float(t))
        if key > best_key:
            best_t, best_key = float(t), key
    return best_t


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def roc_points(y_true: np.ndarray, proba: np.ndarray, max_points: int = 60) -> list[dict[str, float]]:
    fpr, tpr, _ = roc_curve(y_true, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


def permutation_importance(
    proba_fn: ProbaFn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> list[dict[str, float | str]]:
    """Mean drop in ROC-AUC when each feature column is shuffled (model-agnostic)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, proba_fn(X))
    out: list[dict[str, float | str]] = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, proba_fn(Xp)))
        out.append({"feature": name, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return sorted(out, key=lambda d: float(d["importance"]), reverse=True)

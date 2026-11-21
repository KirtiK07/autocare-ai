"""scikit-learn baselines tuned with GridSearchCV (ROC-AUC) on stratified 5-fold CV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass
class BaselineResult:
    name: str
    estimator: Any
    strategy: str  # "none" | "class_weight" | "smote"
    cv_auc: float
    best_params: dict[str, Any]


def fit_dummy(X: np.ndarray, y: np.ndarray) -> BaselineResult:
    est = DummyClassifier(strategy="most_frequent").fit(X, y)
    return BaselineResult("Dummy (majority)", est, "none", float("nan"), {})


def _grid(
    name: str,
    clf: Any,
    params: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    imbalanced: bool,
    seed: int,
    n_jobs: int,
) -> BaselineResult:
    """Pick the best of class_weight='balanced' vs SMOTE (or plain, if balanced)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates: list[tuple[str, Any, dict[str, list[Any]]]] = []
    if imbalanced:
        weighted = {f"clf__{k}": v for k, v in params.items()} | {"clf__class_weight": ["balanced"]}
        candidates.append(("class_weight", ImbPipeline([("clf", clf)]), weighted))
        smote = {f"clf__{k}": v for k, v in params.items()}
        candidates.append(("smote", ImbPipeline([("smote", SMOTE(random_state=seed)), ("clf", clf)]), smote))
    else:
        candidates.append(("none", ImbPipeline([("clf", clf)]), {f"clf__{k}": v for k, v in params.items()}))

    best: BaselineResult | None = None
    for strategy, pipe, grid in candidates:
        # SMOTE sits inside the pipeline, so it only ever sees each CV training fold.
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=True)
        search.fit(X, y)
        result = BaselineResult(
            name, search.best_estimator_, strategy, float(search.best_score_), search.best_params_
        )
        if best is None or result.cv_auc > best.cv_auc:
            best = result
    assert best is not None
    return best


def fit_logistic(
    X: np.ndarray, y: np.ndarray, imbalanced: bool, seed: int, n_jobs: int = -1
) -> BaselineResult:
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    return _grid("Logistic Regression", clf, {"C": [0.01, 0.1, 1.0, 10.0]}, X, y, imbalanced, seed, n_jobs)


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, imbalanced: bool, seed: int, n_jobs: int = -1
) -> BaselineResult:
    clf = RandomForestClassifier(random_state=seed, n_jobs=1)
    params: dict[str, list[Any]] = {
        "n_estimators": [300],
        "max_depth": [8, 16, None],
        "min_samples_leaf": [1, 5],
    }
    return _grid("Random Forest", clf, params, X, y, imbalanced, seed, n_jobs)


def fit_gradient_boosting(
    X: np.ndarray, y: np.ndarray, imbalanced: bool, seed: int, n_jobs: int = -1
) -> BaselineResult:
    clf = HistGradientBoostingClassifier(random_state=seed, early_stopping=True)
    params: dict[str, list[Any]] = {
        "learning_rate": [0.03, 0.1],
        "max_leaf_nodes": [15, 31],
        "l2_regularization": [0.0, 1.0],
    }
    return _grid("Gradient Boosting", clf, params, X, y, imbalanced, seed, n_jobs)

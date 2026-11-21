"""End-to-end training: preprocessing, baselines, PyTorch, evaluation, artifacts.

Usage::

    python -m ml.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from ml.baselines import fit_dummy, fit_logistic, fit_random_forest
from ml.data import dataset_info, load_dataset, split
from ml.evaluate import classification_metrics, permutation_importance, roc_points, tune_threshold
from ml.features import FEATURE_COLUMNS, SENSOR_COLUMNS, Preprocessor
from ml.nets import predict_proba, seed_everything, train_net

log = logging.getLogger("ml.train")
BACKEND_DIR = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BACKEND_DIR / p


def _pick_sample(
    X: pd.DataFrame, y: np.ndarray, proba: np.ndarray, threshold: float, label: int
) -> dict[str, float]:
    """A correctly classified, typical (median-confidence) example of the given class."""
    pred = (proba >= threshold).astype(int)
    mask = (y == label) & (pred == label)
    if not mask.any():
        mask = y == label
    idx = np.where(mask)[0]
    order = idx[np.argsort(proba[idx])]
    chosen = order[len(order) // 2]
    return {c: round(float(X.iloc[chosen][c]), 3) for c in SENSOR_COLUMNS}


def run(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    seed_everything(seed)
    data_path = _resolve(cfg["data"]["path"])
    out_dir = _resolve(cfg["artifacts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path, positive_label=int(cfg["data"]["positive_label"]))
    info = dataset_info(data_path)
    log.info("Loaded %d rows (source: %s), fault share %.3f", len(df), info.get("source"), df["fault"].mean())
    s = split(df, cfg["split"]["val_size"], cfg["split"]["test_size"], seed)

    pre = Preprocessor().fit(s.X_train, s.y_train)
    Xtr, Xva, Xte = pre.transform(s.X_train), pre.transform(s.X_val), pre.transform(s.X_test)
    ytr, yva, yte = s.y_train.to_numpy(), s.y_val.to_numpy(), s.y_test.to_numpy()

    minority = min(ytr.mean(), 1 - ytr.mean())
    imbalanced = minority < float(cfg["imbalance_threshold"])
    min_recall = float(cfg["min_recall"])
    log.info("Minority share %.3f -> imbalanced=%s", minority, imbalanced)

    results: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    dummy = fit_dummy(Xtr, ytr)
    results.append(
        {
            "key": "dummy",
            "name": dummy.name,
            "strategy": "none",
            "params": {},
            "val": classification_metrics(yva, dummy.estimator.predict_proba(Xva)[:, 1], 0.5),
            "test": classification_metrics(yte, dummy.estimator.predict_proba(Xte)[:, 1], 0.5),
        }
    )

    for key, fit in (("logistic_regression", fit_logistic), ("random_forest", fit_random_forest)):
        log.info("Fitting %s ...", key)
        res = fit(Xtr, ytr, imbalanced, seed)
        models[key] = res.estimator
        pv = res.estimator.predict_proba(Xva)[:, 1]
        t = tune_threshold(yva, pv, min_recall)
        results.append(
            {
                "key": key,
                "name": res.name,
                "strategy": res.strategy,
                "cv_f1": res.cv_f1,
                "params": {k.replace("clf__", ""): v for k, v in res.best_params.items()},
                "val": classification_metrics(yva, pv, t),
                "test": classification_metrics(yte, res.estimator.predict_proba(Xte)[:, 1], t),
            }
        )

    log.info("Training PyTorch EngineFaultNet ...")
    tcfg = cfg["pytorch"]
    tr = train_net(
        Xtr,
        ytr,
        Xva,
        yva,
        lr=float(tcfg["lr"]),
        batch_size=int(tcfg["batch_size"]),
        max_epochs=int(tcfg["max_epochs"]),
        patience=int(tcfg["patience"]),
        dropout=float(tcfg["dropout"]),
        use_pos_weight=imbalanced,
        seed=seed,
    )
    models["pytorch"] = tr.model
    pv = predict_proba(tr.model, Xva)
    t = tune_threshold(yva, pv, min_recall)
    results.append(
        {
            "key": "pytorch",
            "name": "PyTorch EngineFaultNet",
            "strategy": "pos_weight" if imbalanced else "none",
            "params": {"best_epoch": tr.best_epoch, "epochs_run": tr.epochs_run, **tcfg},
            "val": classification_metrics(yva, pv, t),
            "test": classification_metrics(yte, predict_proba(tr.model, Xte), t),
        }
    )

    # Model selection uses validation F1 only; the test set is for reporting.
    candidates = [r for r in results if r["key"] != "dummy"]
    served = max(candidates, key=lambda r: r["val"]["f1"])
    served_key = served["key"]
    threshold = float(served["val"]["threshold"])
    log.info("Served model: %s (val F1 %.4f)", served_key, served["val"]["f1"])

    def proba_fn(X: np.ndarray) -> np.ndarray:
        if served_key == "pytorch":
            return predict_proba(tr.model, X)
        return np.asarray(models[served_key].predict_proba(X)[:, 1])

    test_proba = proba_fn(Xte)
    importance = permutation_importance(proba_fn, Xte, yte, list(FEATURE_COLUMNS), seed=seed)
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    version = f"{served_key}-{trained_at[:10]}"

    pre.save(out_dir / "scaler.joblib")
    torch.save(tr.model.state_dict(), out_dir / "fault_net.pt")
    sklearn_path = out_dir / "sklearn_model.joblib"
    if served_key != "pytorch":
        # Compressed: an unpruned forest can exceed GitHub's 100 MB file limit otherwise.
        joblib.dump(models[served_key], sklearn_path, compress=3)
    elif sklearn_path.exists():
        sklearn_path.unlink()

    meta = {
        "model_version": version,
        "served_model": served_key,
        "served_model_name": served["name"],
        "feature_order": list(FEATURE_COLUMNS),
        "sensor_columns": list(SENSOR_COLUMNS),
        "threshold": threshold,
        "risk_bands": {"medium": round(threshold * 0.6, 3), "high": threshold},
        "n_features": len(FEATURE_COLUMNS),
        "pytorch": {"hidden": [64, 32], "dropout": float(tcfg["dropout"])},
        "trained_at": trained_at,
        "dataset": {**info, "rows": int(len(df)), "fault_share": round(float(df["fault"].mean()), 4)},
        "test_metrics": served["test"],
        "samples": {
            "healthy": _pick_sample(s.X_test, yte, test_proba, threshold, 0),
            "faulty": _pick_sample(s.X_test, yte, test_proba, threshold, 1),
        },
        "sensor_ranges": {
            c: {
                "min": round(float(df[c].min()), 3),
                "max": round(float(df[c].max()), 3),
                "healthy_p05": round(float(df.loc[df.fault == 0, c].quantile(0.05)), 3),
                "healthy_p95": round(float(df.loc[df.fault == 0, c].quantile(0.95)), 3),
            }
            for c in SENSOR_COLUMNS
        },
    }
    metrics = {
        "model_version": version,
        "served_model": served_key,
        "imbalanced": bool(imbalanced),
        "min_recall": min_recall,
        "split_sizes": {"train": len(ytr), "val": len(yva), "test": len(yte)},
        "models": results,
        "roc_curve": roc_points(yte, test_proba),
        "feature_importance": importance,
        "training_history": tr.history,
        "dataset": meta["dataset"],
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    table = results_table(results)
    (out_dir / "results.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nServed: {served['name']}  threshold={threshold}  version={version}")
    return metrics


def results_table(results: list[dict[str, Any]]) -> str:
    """Markdown table of test-set metrics, exactly as measured (4 d.p.)."""
    lines = [
        "| Model | Imbalance strategy | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r["test"]
        lines.append(
            f"| {r['name']} | {r['strategy']} | {m['threshold']:.2f} | {m['accuracy']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the AutoCare AI fault model")
    parser.add_argument("--config", type=Path, default=BACKEND_DIR / "configs" / "train.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

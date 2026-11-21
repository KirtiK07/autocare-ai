# Model card: AutoCare engine fault classifier

## Summary

Binary classifier: given one snapshot of six engine sensor readings, estimate the probability that the engine needs attention. The served model is the one with the best validation F1, chosen between Logistic Regression, Random Forest and a PyTorch feed-forward network. `artifacts/model_meta.json` records which one was chosen, its threshold, the dataset provenance and the training date.

> The committed artifacts were trained on the **synthetic stand-in dataset**. Retrain on the real data before using or quoting any metric. The UI shows a warning until you do.

## Data

- Source: Kaggle "Automotive Vehicles Engine Health Dataset" (see `backend/data/README.md` for link, licence and row count).
- Inputs: engine RPM, lubricant oil pressure, fuel pressure, coolant pressure, lubricant oil temperature, coolant temperature.
- Label: `Engine Condition` (1 = needs attention).
- Split: stratified 70 / 15 / 15 train / validation / test, `random_state=42`. Duplicates are dropped before splitting.

## Features and preprocessing

All of this is fitted on the training split only and saved as `artifacts/scaler.joblib`, which the API loads, so training and serving never drift.

1. Median imputation per column.
2. Clipping at the 1st / 99th percentile.
3. Engineered: oil/fuel and coolant/oil pressure ratios, oil − coolant temperature, one-hot RPM bands (< 600, 600–1200, ≥ 1200).
4. `StandardScaler`.

## Models

| Model | Setup |
| --- | --- |
| Dummy | Majority class; the floor every model must beat (enforced by a CI test) |
| Logistic Regression | `GridSearchCV` over C, 5-fold stratified CV, F1 scoring |
| Random Forest | `GridSearchCV` over depth and min-samples-leaf, 300 trees |
| PyTorch `EngineFaultNet` | 12 → 64 → 32 → 1, BatchNorm1d, ReLU, dropout 0.2, `BCEWithLogitsLoss` with `pos_weight`, Adam (lr 1e-3), batch 256, early stopping on validation loss (patience 10), seeded |

Imbalance: when the minority class is below 45% of training rows, `class_weight='balanced'` is compared against SMOTE. SMOTE sits inside an imblearn pipeline, so it only ever sees CV training folds and never leaks into validation.

## Threshold

For each model, the threshold is the one with the **highest validation precision while recall stays ≥ 0.85**. Missing a real fault costs more than a false alarm. Maximising F1 or F2 directly was tried first. With faults as the majority class, it collapsed every model to "always faulty", matching the Dummy baseline. Risk bands: High ≥ threshold, Medium ≥ 0.6 × threshold, Low below that.

## Metrics

Test-set metrics for every model are in `artifacts/results.md` and `artifacts/metrics.json`, and on the UI's Model page. Paste the real-data table here after retraining.

## Explainability

- Global: permutation importance (mean drop in test ROC-AUC over 5 shuffles per feature).
- Per prediction: each sensor's deviation from the healthy-class mean in standard units; the top 3 are shown. These are easy to read, but they aren't causal attributions.

## Limitations

- Snapshot readings only: no trends, duty cycle, mileage or maintenance history.
- Not from a specific manufacturer; units aren't documented in the source data.
- Class overlap in this kind of data is high, so expect moderate accuracy. The comparison table shows the honest numbers.
- Not a safety system. Always have a qualified mechanic inspect a vehicle.

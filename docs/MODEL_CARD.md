# Model card: AutoCare engine fault classifier

## Summary

Binary classifier: given one snapshot of six engine sensor readings, estimate the probability that the engine needs attention. The served model is the one with the best **validation ROC-AUC**, chosen between Logistic Regression, Random Forest, Gradient Boosting and a PyTorch feed-forward network; on the Kaggle data that is the PyTorch network. `artifacts/model_meta.json` records the choice, its threshold, the dataset provenance and the training date.

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
| Logistic Regression | `GridSearchCV` over C, 5-fold stratified CV, ROC-AUC scoring |
| Random Forest | `GridSearchCV` over depth and min-samples-leaf, 300 trees |
| Gradient Boosting | `HistGradientBoostingClassifier`, `GridSearchCV` over learning rate, leaves and L2, early stopping |
| PyTorch `EngineFaultNet` | 12 → 64 → 32 → 1, BatchNorm1d, ReLU, dropout 0.2, `BCEWithLogitsLoss` with `pos_weight`, Adam (lr 1e-3), batch 256, early stopping on validation loss (patience 15), seeded |

Imbalance: when the minority class is below 45% of training rows, `class_weight='balanced'` is compared against SMOTE. SMOTE sits inside an imblearn pipeline, so it only ever sees CV training folds and never leaks into validation.

## Threshold

For each model, the threshold is the one with the **highest validation precision while recall stays ≥ 0.85**. Missing a real fault costs more than a false alarm. Maximising F1 or F2 directly was tried first. With faults as the majority class, it collapsed every model to "always faulty", matching the Dummy baseline. Risk bands: High ≥ threshold, Medium ≥ 0.6 × threshold, Low below that.

## Metrics (held-out test set, 2,931 rows)

| Model | Imbalance strategy | Threshold | ROC-AUC | Balanced acc. | MCC | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy (majority) | none | 0.50 | 0.5000 | 0.5000 | 0.0000 | 0.6305 | 0.6305 | 1.0000 | 0.7734 |
| Logistic Regression | smote | 0.38 | 0.6957 | 0.5975 | 0.2222 | 0.6602 | 0.6898 | 0.8377 | 0.7566 |
| Random Forest | smote | 0.38 | 0.6968 | 0.5996 | 0.2275 | 0.6626 | 0.6910 | 0.8409 | 0.7586 |
| Gradient Boosting | smote | 0.36 | 0.6962 | 0.5950 | 0.2213 | 0.6615 | 0.6872 | 0.8501 | 0.7600 |
| PyTorch EngineFaultNet | pos_weight | 0.37 | 0.7008 | 0.5996 | 0.2253 | 0.6609 | 0.6915 | 0.8344 | 0.7563 |

How to read this:
- **The dataset is weakly separable.** Every model reaches ROC-AUC of about 0.70, and the four are within 0.005 of each other, so more model capacity doesn't help; the limit is the information in six snapshot readings.
- **F1 isn't used to choose a model.** With 63% faults, "always faulty" gets F1 0.773, higher than any real model. Balanced accuracy and MCC are 0.50 and 0.00 for that dummy; the served model reaches 0.60 and 0.23. CI enforces this gap.
- At the chosen operating point, the model catches about 83% of faults and raises a false alarm on about 64% of healthy engines. That's the cost of a high recall floor on overlapping classes.

## Explainability

- Global: permutation importance (mean drop in test ROC-AUC over 5 shuffles per feature).
- Per prediction: occlusion. Each sensor is reset to its healthy-class mean and the change in fault probability is measured; the top 3 are shown in percentage points. This is faithful to the model, but it reflects learned associations, not mechanical cause.

## Limitations

- Snapshot readings only: no trends, duty cycle, mileage or maintenance history.
- Not from a specific manufacturer; units aren't documented in the source data.
- Class overlap in this kind of data is high, so expect moderate accuracy. The comparison table shows the honest numbers.
- Not a safety system. Always have a qualified mechanic inspect a vehicle.

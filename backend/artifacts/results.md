| Model | Imbalance strategy | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy (majority) | none | 0.50 | 0.6240 | 0.6240 | 1.0000 | 0.7685 | 0.5000 |
| Logistic Regression | smote | 0.35 | 0.6991 | 0.7220 | 0.8420 | 0.7774 | 0.7612 |
| Random Forest | class_weight | 0.38 | 0.7049 | 0.7248 | 0.8496 | 0.7823 | 0.7706 |
| PyTorch EngineFaultNet | pos_weight | 0.32 | 0.7199 | 0.7400 | 0.8496 | 0.7910 | 0.7906 |

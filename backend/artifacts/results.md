| Model | Imbalance strategy | Threshold | ROC-AUC | Balanced acc. | MCC | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy (majority) | none | 0.50 | 0.5000 | 0.5000 | 0.0000 | 0.6305 | 0.6305 | 1.0000 | 0.7734 |
| Logistic Regression | smote | 0.38 | 0.6957 | 0.5975 | 0.2222 | 0.6602 | 0.6898 | 0.8377 | 0.7566 |
| Random Forest | smote | 0.38 | 0.6968 | 0.5996 | 0.2275 | 0.6626 | 0.6910 | 0.8409 | 0.7586 |
| Gradient Boosting | smote | 0.36 | 0.6962 | 0.5950 | 0.2213 | 0.6615 | 0.6872 | 0.8501 | 0.7600 |
| PyTorch EngineFaultNet | pos_weight | 0.37 | 0.7008 | 0.5996 | 0.2253 | 0.6609 | 0.6915 | 0.8344 | 0.7563 |

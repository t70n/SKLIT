# Model Evaluation

[Home](../README.md)

This section explains how to measure the performance of a model honestly: which metric to use for each type of task, how to estimate generalization with cross-validation, how to choose a splitting strategy that respects the structure of the data, and how to diagnose underfitting and overfitting.

## Pages

### Metrics

| Page | Summary |
| --- | --- |
| [Metrics and Scoring Overview](Metrics-and-Scoring.md) | `score`, metric functions, the `scoring` parameter, the `neg_` convention, custom scorers |
| [Classification Metrics](Classification-Metrics.md) | Confusion matrix, accuracy, balanced accuracy, precision, recall, F1, ROC and PR curves, thresholds, class imbalance |
| [Regression Metrics](Regression-Metrics.md) | MAE, MSE, RMSE, median absolute error, MAPE, R², deviances, residual plots, target transformation |

### Validation

| Page | Summary |
| --- | --- |
| [Cross-Validation](Cross-Validation.md) | `cross_validate`, `cross_val_score`, outputs, comparison with a baseline |
| [Cross-Validation Strategies](Cross-Validation-Strategies.md) | K-fold, stratification, grouped samples, time series |

### Diagnostics

| Page | Summary |
| --- | --- |
| [Learning Curves](Learning-Curves.md) | Performance as a function of the training-set size |
| [Validation Curves](Validation-Curves.md) | Performance as a function of one hyperparameter |

## Always start with a baseline

A score is only meaningful compared with the score of a model that ignores the features: [DummyClassifier](../07-Models/Baselines/DummyClassifier.md) for classification and [DummyRegressor](../07-Models/Baselines/DummyRegressor.md) for regression.

## Related recipes

- [Comparing a Classifier with Dummy Baselines](../10-Recipes/Dummy-Classifier-Baselines.md)
- [Blood Transfusion: Evaluating an Imbalanced Classifier](../10-Recipes/Blood-Transfusion-Classification-Metrics.md)
- [Regression Error Analysis and Target Transformation](../10-Recipes/Regression-Error-Analysis-and-Target-Transformation.md)
- [Digits: Group-Aware Cross-Validation](../10-Recipes/Digits-Group-Aware-Cross-Validation.md)

## Navigation

- Previous section: [Models](../07-Models/README.md)
- Next section: [Hyperparameter Tuning](../09-Hyperparameter-Tuning/README.md)
- [Back to the home page](../README.md)

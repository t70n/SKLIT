# Validation Curves

[Home](../README.md) / [Model Evaluation](README.md)

## Purpose

A validation curve shows how model performance changes as a hyperparameter varies.

This is useful when you want to understand the model's sensitivity to a setting like:

- `max_depth` for decision trees
- `n_neighbors` for k-NN
- `gamma` for SVMs
- `C` for regularized models

The goal is to find the parameter value that gives the best balance between underfitting and overfitting.

## Typical example

```python
import numpy as np
from sklearn.model_selection import ValidationCurveDisplay

max_depth = np.array([1, 3, 5, 10, 15, 20, 25])

disp = ValidationCurveDisplay.from_estimator(
    regressor,
    X,
    y,
    param_name="max_depth",
    param_range=max_depth,
    cv=5,
    scoring="neg_mean_absolute_error",
    negate_score=True,
    std_display_style="errorbar",
    n_jobs=2,
)
```

`negate_score=True` turns the negated MAE back into a positive error on the plot. `ValidationCurveDisplay` is available since scikit-learn 1.3.

A simpler version, without the display object, is to compute the validation scores directly:

```python
from sklearn.model_selection import validation_curve

param_range = np.array([1, 3, 5, 10, 15, 20, 25])
train_scores, valid_scores = validation_curve(
    regressor,
    X,
    y,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="neg_mean_absolute_error",
)
```

## Interpretation

A validation curve is usually read from left to right:

- low complexity: underfitting
- medium complexity: good trade-off
- high complexity: overfitting

When the training score keeps improving while the validation score starts to get worse, the model is likely becoming too complex.

The best hyperparameter is often where the validation score is highest or where the gap between train and validation performance is still reasonable.

## For SVM gamma

```python
gamma_values = np.logspace(-3, 2, num=30)

disp = ValidationCurveDisplay.from_estimator(
    pipeline,
    X,
    y,
    param_name="svc__gamma",
    param_range=gamma_values,
    cv=5,
    score_type="both",
    scoring="accuracy",
    n_jobs=2,
)
```

This is a common way to inspect how `gamma` affects an SVM. If `gamma` is too small, the model may be underfitting. If it is too large, it may overfit the training data. For a parameter of a pipeline step, use the `<step>__<parameter>` name, here `svc__gamma` for a pipeline built with `make_pipeline`. Parameters that span several orders of magnitude, such as `gamma` or `C`, are best explored on a logarithmic grid (`np.logspace`).

## Key takeaway

Validation curves help you see how a hyperparameter changes the bias-variance trade-off and identify a reasonable operating point before tuning with a more expensive search.

## Validation curve or grid search?

A validation curve varies a single hyperparameter and shows both the training and the test scores, which makes it a diagnostic tool. To select the values of several interacting hyperparameters, use [GridSearchCV](../09-Hyperparameter-Tuning/GridSearchCV.md) or [RandomizedSearchCV](../09-Hyperparameter-Tuning/RandomizedSearchCV.md), and estimate the performance of the tuned model with [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md).

## Related pages

- [Learning Curves](Learning-Curves.md)
- [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md)
- [Manual Tuning](../09-Hyperparameter-Tuning/Manual-Tuning.md)
- [scikit-learn user guide: Validation curves and learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)

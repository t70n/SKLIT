# Metrics and Scoring Overview

[Home](../README.md) / [Model Evaluation](README.md)

## Purpose

A metric quantifies how well predictions match the true targets. The appropriate metric depends first on the type of task, then on the cost of the different kinds of errors:

| Task | Target | Detailed page |
| --- | --- | --- |
| Classification | Discrete labels | [Classification Metrics](Classification-Metrics.md) |
| Regression | Continuous values | [Regression Metrics](Regression-Metrics.md) |

Unsupervised tasks such as clustering have their own metrics (for example the adjusted Rand index or the silhouette coefficient); they are described in the scikit-learn user guide and are outside the scope of this wiki.

This page explains the mechanics shared by all metrics in scikit-learn: how scores are computed, the "higher is better" convention, and how to evaluate several metrics at once.

## Three ways to evaluate a model

### 1. The `score` method

Every predictor has a `score(X, y)` method with a default metric:

- classifiers return the **accuracy**
- regressors return the **coefficient of determination** $R^2$

```python
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

Convenient, but the default metric is not always the right one, for example accuracy on imbalanced classes.

### 2. Metric functions

The `sklearn.metrics` module provides functions with the signature `metric(y_true, y_pred)`:

```python
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)      # classification
mean_absolute_error(y_test, y_pred)          # regression
```

Some metrics need continuous scores rather than labels, such as `roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])`.

### 3. The `scoring` parameter

Model-selection tools (`cross_validate`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`, `learning_curve`, `validation_curve`) accept a `scoring` argument, given as a string name or a scorer object:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy")
```

All available names are listed by:

```python
from sklearn.metrics import get_scorer_names

print(get_scorer_names())
```

## The "higher is better" convention

Model-selection utilities always **maximize** the scorer. Error metrics, for which lower is better, are therefore exposed as negated scores with a `neg_` prefix:

| Scorer name | Returned value |
| --- | --- |
| `"neg_mean_absolute_error"` | $-\text{MAE}$ |
| `"neg_mean_squared_error"` | $-\text{MSE}$ |
| `"neg_root_mean_squared_error"` | $-\text{RMSE}$ |
| `"neg_log_loss"` | $-\text{log loss}$ |

Negate the returned values to report the usual positive error:

```python
from sklearn.model_selection import cross_validate

results = cross_validate(model, X, y, cv=5, scoring="neg_mean_absolute_error")
mae = -results["test_score"]
print(f"MAE: {mae.mean():.2f} +/- {mae.std():.2f}")
```

### A common interpretation trap

If all the values returned by `cross_val_score(model_A, X, y, scoring="neg_mean_squared_error")` are strictly lower than those returned for `model_B`, then `model_B` generalizes **better** than `model_A`. A lower negated MSE means a higher MSE, that is, larger errors for `model_A`.

## Evaluating several metrics at once

`cross_validate` accepts a list of scorer names or a dictionary of scorers. Each metric produces its own `test_<name>` key (and `train_<name>` with `return_train_score=True`):

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y, cv=5,
    scoring=["accuracy", "balanced_accuracy", "roc_auc"],
)
results["test_accuracy"], results["test_balanced_accuracy"], results["test_roc_auc"]
```

With several metrics, `GridSearchCV` needs `refit="<metric name>"` to know which metric selects the best model.

## Custom scorers with `make_scorer`

`make_scorer` turns a metric function into a scorer usable by the `scoring` parameter. It is required when the metric needs extra arguments:

```python
from sklearn.metrics import fbeta_score, make_scorer, mean_absolute_error, precision_score

# Positive class given by a string label
precision_donated = make_scorer(precision_score, pos_label="donated")

# F-beta score emphasizing recall
f2 = make_scorer(fbeta_score, beta=2)

# An error metric: lower is better, so the scorer negates it
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
```

Use `response_method="predict_proba"` (or `"decision_function"`) when the metric needs continuous scores instead of labels.

The string scorers `"precision"`, `"recall"`, and `"f1"` assume that the positive label is `1`. With string labels such as `"donated"` and `"not donated"`, they raise `ValueError: pos_label=1 is not a valid label`; use `make_scorer` with an explicit `pos_label` instead.

## Good practice

- choose the metric before comparing models, from the cost of errors in the application
- compare every model with a dummy baseline evaluated with the same splits and metric ([DummyClassifier](../07-Models/Baselines/DummyClassifier.md), [DummyRegressor](../07-Models/Baselines/DummyRegressor.md))
- report the mean and the standard deviation across folds, and inspect the distribution of fold scores
- report several complementary metrics when a single number would hide a failure mode
- never select a model or a threshold using the final test set

## Related pages

- [Classification Metrics](Classification-Metrics.md)
- [Regression Metrics](Regression-Metrics.md)
- [Cross-Validation](Cross-Validation.md)
- [scikit-learn user guide: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)

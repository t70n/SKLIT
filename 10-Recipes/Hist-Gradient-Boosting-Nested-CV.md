# Histogram Gradient Boosting with Nested Cross-Validation

[Home](../README.md) / [Recipes](README.md)

## Goal

Use an inner `GridSearchCV` and an outer `cross_validate` loop to tune `HistGradientBoostingRegressor` while estimating generalization.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y = y * 100

base_model = HistGradientBoostingRegressor(
    max_iter=1000,
    early_stopping=True,
    random_state=42,
)

search = GridSearchCV(
    base_model,
    param_grid={
        "max_depth": [3, 8],
        "max_leaf_nodes": [15, 31],
        "learning_rate": [0.1, 1.0],
    },
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=2,
    return_train_score=True,
)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
result = cross_validate(
    search,
    X,
    y,
    cv=outer_cv,
    scoring="neg_mean_absolute_error",
    return_estimator=True,
    n_jobs=2,
)

scores = result["test_score"]
print(f"MAE: {-scores.mean():.3f} +/- {scores.std():.3f}")
```

## Inspect each outer fold

```python
for fold, fitted_search in enumerate(result["estimator"], start=1):
    best_model = fitted_search.best_estimator_
    print(f"Fold {fold}")
    print("best parameters:", fitted_search.best_params_)
    print("trees used:", best_model.n_iter_)
```

`n_iter_` can be smaller than `max_iter` because early stopping may terminate training.

## Aggregate inner-search results

```python
import pandas as pd

frames = []
for fold, fitted_search in enumerate(result["estimator"], start=1):
    frame = pd.DataFrame(fitted_search.cv_results_)
    frame["outer_fold"] = fold
    frame["mae"] = -frame["mean_test_score"]
    frames.append(frame)

all_results = pd.concat(frames, ignore_index=True)
summary = (
    all_results
    .groupby([
        "param_max_depth",
        "param_max_leaf_nodes",
        "param_learning_rate",
    ])["mae"]
    .agg(["mean", "std", "count"])
    .sort_values("mean")
)
print(summary)
```

Nested cross-validation estimates the full tuning procedure. The outer test folds are not used to select parameters.

## Related pages

- [HistGradientBoostingRegressor](../07-Models/Ensembles/HistGradientBoostingRegressor.md)
- [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md)
- [Boosting Hyperparameter Tuning](../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md)

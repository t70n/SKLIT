# Boosting Hyperparameter Tuning

[Home](../README.md) / [Hyperparameter Tuning](README.md)

## Main trade-off

Boosting adds weak trees sequentially:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

`learning_rate` controls correction size; `n_estimators` or `max_iter` controls the number of corrections.

- small learning rate plus many trees is slower but often smoother
- large learning rate plus few trees can overfit or underfit
- too many trees without stopping can overfit

Tune learning rate together with the iteration limit.

## Tree complexity

Start with weak learners using `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, and `min_samples_split`. Depths around 3 to 8 are common starting points for classic boosting, but validation should decide. `max_depth` and `max_leaf_nodes` both limit the size of each tree; searching both at once creates redundant candidates, so it is often clearer to tune one of them primarily.

## Classic GradientBoosting search

```python
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    GradientBoostingRegressor(
        n_estimators=1000,
        n_iter_no_change=5,
        random_state=42,
    ),
    param_distributions={
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": [2, 3, 5, 8],
        "max_leaf_nodes": [7, 15, 31, 63],
        "min_samples_leaf": [1, 5, 10, 20],
        "subsample": [0.6, 0.8, 1.0],
    },
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
)
```

## Histogram Gradient Boosting search

```python
from sklearn.ensemble import HistGradientBoostingRegressor

search = RandomizedSearchCV(
    HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    ),
    param_distributions={
        "learning_rate": loguniform(0.01, 1.0),
        "max_leaf_nodes": [7, 15, 31, 63, 127],
        "max_depth": [None, 3, 5, 8],
        "min_samples_leaf": [10, 20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    },
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
)
```

Set `max_iter` generously with early stopping and inspect `n_iter_` after fitting. For nested selection, put the search inside outer `cross_validate`; see [Histogram Gradient Boosting with Nested Cross-Validation](../10-Recipes/Hist-Gradient-Boosting-Nested-CV.md).

## Random Forest controls

Random Forest has different tuning logic: `max_features` controls tree correlation, `max_depth` and `max_leaf_nodes` control structure, `min_samples_leaf` smooths predictions, and `n_estimators` stabilizes averaging. It normally has no `learning_rate`.

## Report results

```python
import pandas as pd

results = pd.DataFrame(search.cv_results_)
results["mae"] = -results["mean_test_score"]
print(results.sort_values("mae").head())
```

Report metric, folds, search space, best parameters, fit time, and variation across folds.

## Related pages

- [Ensemble Models](../07-Models/Ensembles/Ensemble-Models.md)
- [HistGradientBoostingRegressor](../07-Models/Ensembles/HistGradientBoostingRegressor.md)
- [RandomizedSearchCV](RandomizedSearchCV.md)
- [Parallel Coordinates for Hyperparameter Search Results](Parallel-Coordinates-Visualization.md)

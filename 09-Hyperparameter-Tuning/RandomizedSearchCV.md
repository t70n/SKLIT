# RandomizedSearchCV

[Home](../README.md) / [Hyperparameter Tuning](README.md)

## Purpose

`RandomizedSearchCV` samples random hyperparameter combinations instead of evaluating a full grid.

This is useful when the search space is large, when some parameters are continuous, or when the computational budget is fixed.

## Example

```python
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV


class loguniform_int:
    """Integer valued version of the log-uniform distribution."""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample."""
        return self._distribution.rvs(*args, **kwargs).astype(int)


param_distributions = {
    "classifier__l2_regularization": loguniform(1e-6, 1e3),
    "classifier__learning_rate": loguniform(0.001, 10),
    "classifier__max_leaf_nodes": loguniform_int(2, 256),
    "classifier__min_samples_leaf": loguniform_int(1, 100),
}

model_random_search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    random_state=0,
    verbose=1,
)
model_random_search.fit(X_train, y_train)
```

## Defining the search space

Each entry of `param_distributions` can be:

- a list of values, sampled uniformly (for example `["gini", "entropy"]`)
- a distribution object with an `rvs` method, such as those of `scipy.stats`

| Distribution | Use for |
| --- | --- |
| `scipy.stats.loguniform(a, b)` | Positive parameters spanning several orders of magnitude: `learning_rate`, `C`, `alpha`, `gamma`, `l2_regularization` |
| `scipy.stats.uniform(loc, scale)` | Parameters on a linear scale, sampled in `[loc, loc + scale]` |
| `scipy.stats.randint(low, high)` | Integers on a linear scale, sampled in `[low, high)` |
| `loguniform_int(a, b)` (custom class above) | Integers spanning orders of magnitude: `max_leaf_nodes`, `min_samples_leaf` |

A log-uniform distribution gives the same probability to the intervals `[0.001, 0.01]`, `[0.01, 0.1]`, and `[0.1, 1]`, which matches how these parameters influence the model.

## Budget

`n_iter` sets the number of sampled candidates, so the cost is `n_iter * n_splits` fits (plus the final refit) whatever the size of the search space. Set `random_state` to reproduce the same candidates. The attributes (`best_params_`, `best_score_`, `best_estimator_`, `cv_results_`) are the same as for [GridSearchCV](GridSearchCV.md).

## Why it is useful

- faster than exhaustive grid search
- good when the space is large or sparse
- explores more distinct values of each parameter than a grid with the same budget
- often enough to find a strong configuration

## Important caution

It explores a limited number of configurations, so it may miss the globally best combination. As with grid search, `best_score_` is optimistic: use an untouched test set or [Nested Cross-Validation](Nested-Cross-Validation.md) to estimate the generalization performance of the tuned model.

## Related pages

- [GridSearchCV](GridSearchCV.md)
- [Parallel Coordinates for Hyperparameter Search Results](Parallel-Coordinates-Visualization.md)
- [Boosting Hyperparameter Tuning](Boosting-Hyperparameter-Tuning.md)
- [scikit-learn API reference: RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

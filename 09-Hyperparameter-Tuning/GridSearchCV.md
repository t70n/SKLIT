# GridSearchCV

[Home](../README.md) / [Hyperparameter Tuning](README.md)

## Purpose

`GridSearchCV` searches a predefined grid of hyperparameter values and selects the best combination using cross-validation.

## Example

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__learning_rate": (0.01, 0.1, 1, 10),
    "classifier__max_leaf_nodes": (3, 10, 30),
}

model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
model_grid_search.fit(X_train, y_train)

print(model_grid_search.best_params_)
```

Here `model` is a pipeline whose final step is named `classifier`; nested parameters are addressed as `<step>__<parameter>`. Use `model.get_params().keys()` to list the valid names.

## What it does

It tries every combination in the grid and evaluates each one with cross-validation. The grid above contains $4\times3=12$ candidates; with `cv=2`, the search fits $12\times2=24$ models, plus one final refit.

After the search, with the default `refit=True`, the best combination is refitted on the whole training set, and the search object can be used directly as a model:

```python
model_grid_search.predict(X_test)
model_grid_search.score(X_test, y_test)
```

## Useful attributes

| Attribute | Content |
| --- | --- |
| `best_params_` | The best combination of hyperparameters |
| `best_score_` | The mean cross-validated score of the best combination |
| `best_estimator_` | The model refitted on the whole training set with `best_params_` |
| `cv_results_` | Scores, fit times, and parameters of every candidate |

## Important parameters

- `scoring`: the metric used to rank the candidates (default: the estimator's `score` method); see [Metrics and Scoring Overview](../08-Model-Evaluation/Metrics-and-Scoring.md)
- `cv`: the cross-validation strategy; an integer or a splitter object
- `refit`: refit the best model on the whole training set; with several metrics, `refit` must name the metric used for selection
- `n_jobs`: number of parallel jobs
- `return_train_score`: also store the training scores, useful to diagnose overfitting

A grid can also be a list of dictionaries, to search different parameters for different model configurations:

```python
param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]},
]
```

## Advantages

- systematic search
- easy to understand
- good when the parameter space is small

## Limitation

It can become expensive when there are many parameters or many possible values: the number of candidates is the product of the number of values of each parameter. It also only evaluates the grid points and can miss good values between them.

## Access to results

```python
import pandas as pd

cv_results = pd.DataFrame(model_grid_search.cv_results_)
cv_results.sort_values("rank_test_score").head()
```

This contains the scores for each hyperparameter combination, with columns such as `param_<name>`, `mean_test_score`, `std_test_score`, and `rank_test_score`. A heatmap of `mean_test_score` (for two parameters) or a [parallel-coordinates plot](Parallel-Coordinates-Visualization.md) helps to see how the score depends on the parameters.

## Caution: `best_score_` is optimistic

`best_score_` was computed on the same cross-validation folds that were used to choose the best combination, so it is not a fair estimate of the generalization performance, especially for large grids. Evaluate the tuned model on an untouched test set, or wrap the search in an outer cross-validation loop: see [Nested Cross-Validation](Nested-Cross-Validation.md) and [Nested Cross-Validation with SVC](../10-Recipes/Nested-Cross-Validation-with-SVC.md).

## Related pages

- [RandomizedSearchCV](RandomizedSearchCV.md)
- [Manual Tuning](Manual-Tuning.md)
- [Nested Cross-Validation](Nested-Cross-Validation.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)
- [scikit-learn user guide: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)

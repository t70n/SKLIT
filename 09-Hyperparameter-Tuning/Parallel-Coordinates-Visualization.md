# Parallel Coordinates for Hyperparameter Search Results

[Home](../README.md) / [Hyperparameter Tuning](README.md)

## Purpose

A parallel-coordinates plot is a good way to visualize many hyperparameter configurations at once.

It is especially useful when you want to see:

- which parameter values are associated with the best scores
- trade-offs between different hyperparameters
- whether some parameters are strongly linked to good performance

This kind of plot is often used with the results stored in `cv_results_` from scikit-learn search objects. The examples below use `plotly`, an external library (`pip install plotly`), which produces interactive plots in which the range of each axis can be selected to highlight the matching configurations.

## The idea

The DataFrame contains one row per evaluated configuration, and one column per parameter plus the score columns.

A parallel-coordinates plot draws one line per configuration, with each axis corresponding to a parameter or a score.

This makes it easier to inspect patterns across many experiments.

## Preparation step: shorten parameter names

When a model is inside a pipeline, parameter names often look like:

- `classifier__learning_rate`
- `classifier__max_leaf_nodes`

This is useful for scikit-learn, but for plotting it is nicer to rename them to shorter names.

```python
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


cv_results = cv_results.rename(shorten_param, axis=1)
cv_results.head()
```

## Example: plot parallel coordinates for randomized search

```python
import numpy as np
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results.rename(shorten_param, axis=1).apply(
        {
            "learning_rate": np.log10,
            "max_leaf_nodes": np.log2,
            "max_bins": np.log2,
            "min_samples_leaf": np.log10,
            "l2_regularization": np.log10,
            "mean_test_score": lambda x: x,
        }
    ),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show(renderer="notebook")
```

## Why these transformations?

The transformed axes help readability:

- `learning_rate`: often logarithmic because values may span several orders of magnitude
- `max_leaf_nodes`: often displayed in log scale for similar reasons
- `max_bins`: similar to `max_leaf_nodes`
- `min_samples_leaf`: small integer values may be better shown on a log-like axis
- `l2_regularization`: can also vary over large ranges

This makes the plot less dominated by a few large values and easier to interpret.

## Can I do this with `RandomizedSearchCV`?

Yes, this is the most natural use case.

Why?

- `RandomizedSearchCV` explores many sampled configurations
- the plot is useful when you have a lot of rows
- you can see clusters of good-performing combinations and how parameters vary together

This works especially well when the search space is large and you want to inspect patterns across many experiments.

## Can I do this with `GridSearchCV`?

Yes, you can also do it for `GridSearchCV`.

It is valid, but there are two important differences:

- the number of configurations is usually smaller
- the plot may look sparse or less informative if the grid is small

Example:

```python
import numpy as np
import plotly.express as px

cv_results = pd.DataFrame(model_grid_search.cv_results_)
cv_results = cv_results.rename(shorten_param, axis=1)

fig = px.parallel_coordinates(
    cv_results[[
        "learning_rate",
        "max_leaf_nodes",
        "mean_test_score",
    ]],
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show(renderer="notebook")
```

## When to prefer which one?

### Prefer randomized-search results when

- you have a large search space
- you want a broad overview of many sampled configurations
- you want to see dense structure in the search results

### Prefer grid-search results when

- the grid is small and interpretable
- you want to inspect a hand-designed search space
- you want to compare exact combinations you deliberately planned

## Important practical note

The plot only works if your DataFrame contains:

- parameter columns like `learning_rate`, `max_leaf_nodes`, etc.
- a score column like `mean_test_score`

If you want to keep the plot clean, it is often best to keep only the relevant parameter columns and the score column before calling `px.parallel_coordinates`.

## Recommended workflow

1. build `cv_results` from `model_random_search.cv_results_` or `model_grid_search.cv_results_`
2. rename parameter columns to short names
3. select only the relevant parameter columns + score
4. apply log transforms for parameters with large ranges
5. view the plot and inspect the best-performing lines

## Practical interpretation

In a good plot:

- the best-performing configurations tend to cluster in a region of parameter values
- you may see that some parameters are more important than others
- you may detect that a parameter is almost irrelevant if the color does not change strongly along its axis

## Related pages

- [RandomizedSearchCV](RandomizedSearchCV.md)
- [GridSearchCV](GridSearchCV.md)
- [Boosting Hyperparameter Tuning](Boosting-Hyperparameter-Tuning.md)
- [plotly API reference: parallel_coordinates](https://plotly.com/python-api-reference/generated/plotly.express.parallel_coordinates.html)

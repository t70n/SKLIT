# DummyRegressor

[Home](../../README.md) / [Models](../README.md) / [Baselines](../README.md#baselines)

## Idea

`DummyRegressor` predicts a constant value that ignores the input features, such as the mean of the training target. It gives the reference error that any useful regressor must beat.

## Import

```python
from sklearn.dummy import DummyRegressor
```

## Minimal example

```python
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
print(f"R2 score for a regressor predicting the mean: {dummy.score(X_test, y_test):.3f}")
```

## Strategies

| `strategy` | Constant prediction | Minimizes on the training set |
| --- | --- | --- |
| `"mean"` (default) | Mean of `y_train` | Mean squared error |
| `"median"` | Median of `y_train` | Mean absolute error |
| `"quantile"` | Quantile of `y_train` given by `quantile` | Pinball (quantile) loss |
| `"constant"` | Value given by `constant` | Not applicable |

Choose the strategy that matches the evaluation metric: the median is the best constant prediction for the mean absolute error, the mean is the best one for the mean squared error.

## Relationship with R²

The coefficient of determination compares a model with the constant prediction of the mean:

$$
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

A model that always predicts the mean of the evaluated targets obtains $R^2=0$. On a test set, the dummy regressor predicts the mean of the *training* targets, so its test $R^2$ is close to 0 and usually slightly negative. Any model with a negative $R^2$ is worse than this constant baseline. See [Regression Metrics](../../08-Model-Evaluation/Regression-Metrics.md).

## Compare a model with the baseline using cross-validation

Using the same splits for both models makes the comparison fair:

```python
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeRegressor

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

tree_results = cross_validate(
    DecisionTreeRegressor(), data, target,
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=2,
)
dummy_results = cross_validate(
    DummyRegressor(strategy="mean"), data, target,
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=2,
)

all_errors = pd.concat(
    [
        pd.Series(-tree_results["test_score"], name="Decision tree regressor"),
        pd.Series(-dummy_results["test_score"], name="Dummy regressor"),
    ],
    axis=1,
)
all_errors.describe()
```

The `neg_` scorer returns negated errors because scikit-learn maximizes scores; multiplying by $-1$ recovers the usual positive MAE. Plotting both columns as overlapping histograms shows whether the two error distributions are clearly separated:

```python
import matplotlib.pyplot as plt
import numpy as np

bins = np.linspace(start=0, stop=100, num=80)
all_errors.plot.hist(bins=bins, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Cross-validation testing errors")
```

With the California housing data (target in thousands of dollars), the decision tree errors are clearly lower than those of the dummy regressor, which confirms that the tree learned a useful relationship.

## When to use it

- as the first model of every regression project
- to translate an error into context: an MAE of 45 k$ is only meaningful compared with the MAE of the constant prediction
- to detect bugs or leakage: a model that does not beat the dummy has not learned anything useful

## Related pages

- [DummyClassifier](DummyClassifier.md)
- [Regression Metrics](../../08-Model-Evaluation/Regression-Metrics.md)
- [Cross-Validation](../../08-Model-Evaluation/Cross-Validation.md)
- [scikit-learn API reference: DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)

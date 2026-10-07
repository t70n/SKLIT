# Bagging

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

Bagging, or bootstrap aggregating, trains several base estimators on different bootstrap samples and combines their predictions. Each bootstrap sample is drawn with replacement from the original training set.

For regression, the ensemble averages predictions:

$$
\hat{f}_{bag}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)
$$

Averaging models with different errors mainly reduces variance. Bagging is especially useful for unstable base learners such as deep decision trees.

## Import

```python
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
```

## Minimal regression example

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

model = BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=42),
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

## Bootstrap samples

A bootstrap sample has the same size as the original training set, but samples rows with replacement. Some rows appear multiple times and some are left out. The left-out rows are called out-of-bag (OOB) observations.

```python
import numpy as np


def bootstrap_sample(data, target, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        np.arange(len(target)),
        size=len(target),
        replace=True,
    )
    return data.iloc[indices], target.iloc[indices]
```

The manual version is useful for intuition; use `BaggingRegressor` or `BaggingClassifier` for production workflows.

## Important parameters

- `estimator`: base model; trees are common but other estimators are possible
- `n_estimators`: number of models to average
- `max_samples`: rows used per base estimator, as a fraction or count
- `max_features`: features used per base estimator
- `bootstrap`: sample rows with replacement
- `bootstrap_features`: sample features with replacement
- `oob_score`: estimate performance using left-out rows
- `n_jobs`: parallel fitting and prediction

Bagging samples at the model level. Random Forest also randomizes candidate features at every tree split.

## Bagging versus boosting

| Method | Base models | Training data | Main effect |
| --- | --- | --- | --- |
| Bagging | Fit independently | Bootstrap samples | Reduces variance by averaging |
| Boosting | Fit sequentially | Reweighted observations | Reduces bias through corrections |

Bagging is easier to parallelize because its base models are independent. Boosting is sequential because each learner depends on previous errors.

## Out-of-bag evaluation

```python
model = BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=42),
    n_estimators=200,
    oob_score=True,
    random_state=42,
)
model.fit(X_train, y_train)
print(model.oob_score_)
```

OOB performance is useful as an internal diagnostic, but use cross-validation or a held-out test set for final model comparison.

## When to use it

Use bagging when:

- the base estimator has high variance
- independent parallel models are desirable
- a single tree is unstable
- nonlinear tabular relationships matter

Random Forest is usually a stronger default for tree bagging because it adds split-level feature randomness.

## Related pages

- [Random Forest](Random-Forest.md)
- [Ensemble Models](Ensemble-Models.md)
- [Synthetic Regression and Bootstrap Trees](../../10-Recipes/Synthetic-Regression-and-Bootstrap.md)
- [scikit-learn API reference: BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)

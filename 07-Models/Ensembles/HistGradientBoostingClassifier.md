# HistGradientBoostingClassifier

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

This is a gradient boosting classifier designed for tabular data and often performs very well on structured data.

It builds many shallow trees sequentially, each one correcting the error of the previous ones.

## Import

```python
from sklearn.ensemble import HistGradientBoostingClassifier
```

## Minimal example

```python
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)
```

## How it works mathematically

Gradient boosting builds an additive model one tree at a time:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

The next tree $h_m$ approximates the negative gradient of the loss at the current predictions. For classification, the model commonly minimizes log loss; final scores are converted into probabilities with a sigmoid for binary classification or a softmax for multiclass classification. This is sequential error correction, unlike a random forest's independent trees and averaging.

## Why histograms help

Instead of considering every distinct numeric threshold, the implementation bins feature values into a limited number of histogram bins. It searches splits over these bins, reducing computation and memory use on large tabular datasets. Binning is an approximation, so `max_bins` and the data distribution can affect the result.

## Why it is often useful

- strong performance on tabular data
- robust for many structured datasets
- often better than simpler baselines in real-world tasks

## Important parameters

- `learning_rate`: contribution of each tree; smaller values usually need more iterations
- `max_iter`: maximum number of boosting iterations
- `max_leaf_nodes` and `max_depth`: control each tree's complexity
- `min_samples_leaf`: keeps leaves from becoming too small
- `l2_regularization`: shrinks leaf values
- `early_stopping`: stops when a validation score stops improving

The learning-rate/iteration trade-off is central: many small corrections can generalize better than a few large ones, but cost more computation.

## Preprocessing remarks

For this model:

- numerical features usually do not need scaling
- categorical features can often be encoded with `OrdinalEncoder`
- one-hot encoding is often unnecessary and may be slower

Numerical missing values are handled natively. Categorical features are also supported natively through `categorical_features`: a list of column names or indices, a boolean mask, or `"from_dtype"`, which treats DataFrame columns of `category` dtype as categorical (the default since scikit-learn 1.6). Each categorical feature can have at most `max_bins` categories (255 by default).

```python
X = X.astype({"workclass": "category", "occupation": "category"})
model = HistGradientBoostingClassifier(categorical_features="from_dtype")
```

Ordinal encoding is still a valid option when configured consistently. One-hot encoding is not universally wrong, but it can create many sparse columns and increase split-search cost. For imbalanced data, `class_weight="balanced"` is available.

`predict_proba` returns class probabilities and `decision_function` returns raw class scores. Keep validation data separate when using early stopping, and use cross-validation when comparing hyperparameters.

## Important caution

Scaling has no effect on the tree splits of this model, but the representation of categorical features still matters.

## Related pages

- [HistGradientBoostingRegressor](HistGradientBoostingRegressor.md)
- [GradientBoostingClassifier](GradientBoostingClassifier.md)
- [Boosting Hyperparameter Tuning](../../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md)
- [scikit-learn API reference: HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

# HistGradientBoostingRegressor

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

`HistGradientBoostingRegressor` is a histogram-based gradient boosting regressor for structured numerical or mixed tabular data. It bins feature values before finding splits, making training faster and more memory-efficient than checking every distinct threshold.

It builds an additive predictor:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

Each new tree is a small correction in the direction that reduces the selected regression loss.

## Import and example

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    early_stopping=True,
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
```

## Important parameters

- `max_iter`: maximum boosting iterations; early stopping may use fewer
- `learning_rate`: size of each correction
- `max_leaf_nodes` or `max_depth`: tree complexity
- `min_samples_leaf`: minimum samples per leaf
- `l2_regularization`: shrinkage of leaf values
- `loss`: regression objective such as squared error, absolute error, or quantile loss
- `early_stopping`: whether to use validation-based stopping
- `validation_fraction`, `n_iter_no_change`, and `tol`: stopping controls
- `max_bins`: number of histogram bins for numeric features

Use a large enough `max_iter` when early stopping is enabled. Tuning a small fixed `max_iter` can prevent the model from reaching a useful stopping point.

## Histograms and missing values

Numeric values are assigned to bins, and candidate splits are searched over those bins. This reduces split-search cost at the price of an approximation controlled partly by `max_bins`.

The estimator handles missing numerical values natively. Categorical features are supported natively through `categorical_features` (column names, indices, a boolean mask, or `"from_dtype"` for DataFrame columns of `category` dtype, the default since scikit-learn 1.6); alternatively, encode them with an `OrdinalEncoder` inside a pipeline.

## Tuning strategy

A practical search varies:

- `learning_rate` on a logarithmic scale
- `max_leaf_nodes` or `max_depth`
- `l2_regularization`
- `min_samples_leaf`

Use early stopping with a sufficiently large `max_iter`, and evaluate the complete search inside an outer cross-validation loop when estimating generalization.

## When to use it

Use it for medium or large tabular regression problems with nonlinearities, interactions, and missing numerical values. It is often a strong built-in scikit-learn alternative to external gradient-boosting libraries.

## Related pages

- [HistGradientBoostingClassifier](HistGradientBoostingClassifier.md)
- [GradientBoostingRegressor](GradientBoostingRegressor.md)
- [Histogram Gradient Boosting with Nested Cross-Validation](../../10-Recipes/Hist-Gradient-Boosting-Nested-CV.md)
- [scikit-learn API reference: HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

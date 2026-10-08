# GradientBoostingRegressor

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

`GradientBoostingRegressor` predicts a continuous target by adding shallow regression trees sequentially. Each tree approximates a loss gradient or corrects the current residual structure.

For squared-error loss, the first correction is closely related to residual fitting:

$$
r_i^{(m)}=y_i-F_{m-1}(x_i)
$$

and the additive predictor is:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

## Import and example

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
```

## Important parameters

- `n_estimators`: number of boosting stages
- `learning_rate`: shrinkage of each correction
- `max_depth` or `max_leaf_nodes`: weak-tree complexity
- `min_samples_leaf`: controls leaf smoothness
- `loss`: `squared_error`, `absolute_error`, `huber`, or `quantile`
- `subsample`: stochastic gradient boosting when below 1.0
- `n_iter_no_change`, `validation_fraction`, and `tol`: early stopping

A smaller learning rate often needs more trees. Shallow trees are deliberate: each weak learner should underfit so the ensemble can build the function gradually.

## Early stopping

```python
model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_iter_no_change=5,
    validation_fraction=0.1,
    random_state=42,
)
```

The estimator monitors an internal validation fraction and stops adding trees when improvement stalls. When using nested cross-validation or an external validation protocol, be clear about which data controls stopping.

## Compare with Random Forest

- Random Forest fits deep trees independently and averages them.
- Gradient boosting fits shallow trees sequentially.
- Random Forest usually has no `learning_rate`.
- Gradient boosting can overfit when too many trees are added, but early stopping can help.
- Gradient boosting often achieves strong accuracy with lower prediction variance, but requires more careful tuning.

## Preprocessing and limitations

Scaling is not required for tree splits. Encode categories, and impute missing values because classic gradient boosting does not accept them. Classic gradient boosting can be slower than histogram gradient boosting on large tabular datasets and does not extrapolate smoothly beyond the target patterns learned by its trees.

Use MAE, RMSE, or a domain-specific metric explicitly; `.score()` returns $R^2$ for regression.

## Related pages

- [GradientBoostingClassifier](GradientBoostingClassifier.md)
- [HistGradientBoostingRegressor](HistGradientBoostingRegressor.md)
- [Gradient Boosting Regression Comparison](../../10-Recipes/Gradient-Boosting-Regression-Comparison.md)
- [Boosting Hyperparameter Tuning](../../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md)
- [scikit-learn API reference: GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

# DecisionTreeRegressor

[Home](../../README.md) / [Models](../README.md) / [Decision Trees](../README.md#decision-trees)

## Idea

A decision tree regressor predicts a continuous numerical target by splitting the data according to feature thresholds.

## Import

```python
from sklearn.tree import DecisionTreeRegressor
```

## Minimal example

```python
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

pred = model.predict(X_test)
```

## How it works mathematically

At each node, the tree greedily searches feature thresholds and chooses the split that most reduces within-node target variation. A leaf makes a constant prediction, usually the mean target:

$$
\hat{y}_L=\frac{1}{|L|}\sum_{i\in L}y_i
$$

For squared-error regression, the leaf objective is the sum of squared residuals:

$$
\operatorname{SSE}(L)=\sum_{i\in L}(y_i-\bar{y}_L)^2
$$

A split is useful when parent SSE is larger than the weighted SSE of its children:

$$
\Delta=\operatorname{SSE}(P)-\operatorname{SSE}(L)-\operatorname{SSE}(R)
$$

The resulting function is piecewise constant. It can model nonlinear interactions, but it does not extrapolate smoothly beyond the target values seen in its leaves.

## Typical metric

```python
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, pred)
print(error)
```

## Important parameters

### `max_depth`

Controls model complexity.

- shallow tree: underfitting
- deep tree: overfitting

## Why this model learns memorization if too deep

A tree grown very deep can store almost every training sample in a leaf node, which gives near-perfect training performance but poor test performance.

Because each leaf predicts a constant, a regression tree produces a piecewise-constant function and does not extrapolate a smooth trend beyond the observed feature range. Compare its prediction geometry with a linear model when extrapolation is part of the task.

## More parameters and criteria

- `min_samples_leaf` and `min_samples_split`: prevent tiny, unstable leaves
- `max_features`: limits features considered at each split
- `criterion`: `squared_error`, `friedman_mse`, `absolute_error`, or `poisson`
- `ccp_alpha`: cost-complexity pruning strength
- `random_state`: makes randomized tie-breaking reproducible

Squared error is sensitive to outliers because it squares residuals. `absolute_error` is more robust and produces a median-like leaf prediction, but can be slower. `score` returns $R^2$ for this regressor; use explicit MAE or RMSE when that matches the application's cost.

See [Decision Tree Interpretation and Tuning](../../10-Recipes/Decision-Tree-Interpretation-and-Tuning.md) for a complete regression comparison and cross-validated complexity search.

## Related pages

- [Tree-Based Models](Tree-Based-Models.md)
- [DecisionTreeClassifier](DecisionTreeClassifier.md)
- [Ames Housing: Linear Model vs Decision Tree](../../10-Recipes/Ames-Housing-Linear-vs-Tree.md)
- [Synthetic Regression and Bootstrap Trees](../../10-Recipes/Synthetic-Regression-and-Bootstrap.md)
- [scikit-learn API reference: DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

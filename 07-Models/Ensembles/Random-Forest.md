# Random Forest

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

A Random Forest is an ensemble of decision trees. Each tree sees a bootstrap sample of rows and considers a random subset of features at each split. The forest averages regression predictions or votes across classification trees.

The two sources of randomness make trees less correlated. Averaging less-correlated high-variance trees reduces variance while preserving nonlinear interactions.

## Mathematics

For regression:

$$
\hat{f}(x)=\frac{1}{B}\sum_{b=1}^{B}f_b(x)
$$

For classification, the predicted class is commonly the majority vote:

$$
\hat{y}(x)=\operatorname{mode}\{f_b(x):b=1,\ldots,B\}
$$

Averaging helps when individual tree errors are not perfectly correlated. If all trees make the same error, adding trees cannot remove that error.

## Minimal regression example

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
forest.fit(X_train, y_train)
pred = forest.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
```

For classification, use `RandomForestClassifier` and inspect `predict_proba` when probabilities are needed.

## Important parameters

- `n_estimators`: number of trees; more trees usually stabilize results but increase cost
- `max_features`: features considered at each split; lower values decorrelate trees. The default is `"sqrt"` for `RandomForestClassifier` and `1.0` (all features) for `RandomForestRegressor`
- `max_depth`: maximum tree depth
- `min_samples_leaf`: minimum samples per leaf; larger values smooth predictions
- `min_samples_split`: minimum samples required for a split
- `max_samples`: bootstrap sample size
- `bootstrap`: whether to sample rows with replacement
- `class_weight`: adjusts class influence in classification
- `oob_score`: use out-of-bag rows for an internal estimate
- `n_jobs`: parallelism

## Random Forest versus Bagging

| Ensemble | Base estimator | Feature randomness |
| --- | --- | --- |
| `BaggingRegressor`/`Classifier` | User-selected model | Usually model-level feature sampling |
| `RandomForestRegressor`/`Classifier` | Decision tree | Random feature subset at every split |

A Random Forest is specialized tree bagging. Use generic Bagging when you want a different base estimator or a custom sampling strategy.

## Inspect individual trees

```python
import numpy as np

forest_predictions = forest.predict(X_grid)
individual_predictions = np.array([
    tree.predict(X_grid.to_numpy())
    for tree in forest.estimators_
])

print(individual_predictions.shape)
print(np.allclose(individual_predictions.mean(axis=0), forest_predictions))
```

Plotting individual predictions shows the variability reduced by the ensemble average. When the forest is fitted on a DataFrame, its internal trees are fitted on arrays without feature names, so pass them `X_grid.to_numpy()` to avoid a feature-name warning; the forest itself accepts the DataFrame.

## Feature importance

Impurity-based importance is available through `feature_importances_`:

```python
import pandas as pd

importance = pd.Series(
    forest.feature_importances_,
    index=X_train.columns,
).sort_values(ascending=False)
```

It can favor continuous or high-cardinality features and is not causal. Compare it with permutation importance on held-out data.

## Categorical features and preprocessing

Trees do not require scaling. Categorical values still need appropriate handling unless the estimator/version supports them natively. Ordinal encoding can create artificial order, while one-hot encoding can increase dimensionality. Validate the encoding choice.

Random forests accept missing numerical values natively since scikit-learn 1.4; with earlier versions, impute them inside the pipeline.

## Strengths and limitations

Strengths:

- strong tabular baseline
- captures nonlinearities and interactions
- parallelizable
- less unstable than one deep tree
- often robust to scaling and moderate outliers

Limitations:

- less interpretable than a single tree or linear model
- can use substantial memory
- piecewise-constant regression does not extrapolate smoothly
- impurity importance can be biased
- many trees do not fix target leakage or poor validation design

## Related pages

- [Bagging](Bagging.md)
- [Ensemble Models](Ensemble-Models.md)
- [DecisionTreeRegressor](../Decision-Trees/DecisionTreeRegressor.md)
- [Penguins Random Forest Regression](../../10-Recipes/Penguins-Random-Forest-Regression.md)
- [Gradient Boosting Regression Comparison](../../10-Recipes/Gradient-Boosting-Regression-Comparison.md)
- [scikit-learn API reference: RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn API reference: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

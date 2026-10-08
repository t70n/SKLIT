# GradientBoostingClassifier

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

`GradientBoostingClassifier` builds a classifier sequentially from shallow decision trees. Each new tree is fitted to the current model's loss gradient, so it corrects errors made by the existing ensemble.

The additive model is:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

where $\eta$ is `learning_rate` and $h_m$ is the next regression tree. For binary classification, the scores are converted to probabilities with a logistic link; multiclass models use a multiclass loss.

## Import and example

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## Important parameters

- `n_estimators`: number of sequential trees
- `learning_rate`: contribution of each tree
- `max_depth` or `max_leaf_nodes`: complexity of each tree
- `min_samples_leaf`: minimum observations in a leaf
- `subsample`: fraction of rows used for each boosting stage
- `max_features`: feature subsampling at split search
- `n_iter_no_change`: early stopping after validation improvement stalls
- `validation_fraction`: fraction reserved for early stopping
- `tol`: minimum improvement required for early stopping

Small learning rates often need more trees. Deep trees and high learning rates can overfit quickly.

## Preprocessing

Numerical scaling is usually unnecessary because trees split by thresholds. Encode categorical data appropriately. Classic `GradientBoostingClassifier` does not accept missing values, so put an imputer in the pipeline. It is not the same as histogram gradient boosting and has different performance characteristics.

## Outputs

- `predict`: class labels
- `predict_proba`: class probabilities
- `decision_function`: raw class scores
- `estimators_`: fitted weak learners arranged by boosting stage
- `feature_importances_`: impurity-based importance, which is not causal

## Gradient boosting versus Random Forest

Random Forest trees are fitted independently on bootstrap samples and averaged. Gradient boosting fits trees sequentially, with each tree correcting the previous ensemble. Random Forest mainly reduces variance; boosting mainly reduces bias but is more sensitive to learning rate, tree complexity, and noisy labels.

## When to use it

Use it for small-to-medium tabular classification problems when nonlinear interactions matter and you want a strong classical boosting baseline. For larger datasets, `HistGradientBoostingClassifier` is often faster because it uses histogram-based split search.

## Related pages

- [GradientBoostingRegressor](GradientBoostingRegressor.md)
- [HistGradientBoostingClassifier](HistGradientBoostingClassifier.md)
- [Boosting Hyperparameter Tuning](../../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md)
- [scikit-learn API reference: GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

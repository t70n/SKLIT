# Ensemble Models

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Choose a family

- [Bagging](Bagging.md): independent bootstrap models averaged to reduce variance
- [Random Forest](Random-Forest.md): bagged decision trees with random features at each split
- [AdaBoost](AdaBoost.md): sequential learners that emphasize previous errors
- [GradientBoostingClassifier](GradientBoostingClassifier.md): classic sequential classification trees
- [GradientBoostingRegressor](GradientBoostingRegressor.md): classic sequential regression trees
- [HistGradientBoostingClassifier](HistGradientBoostingClassifier.md): histogram-based classification boosting
- [HistGradientBoostingRegressor](HistGradientBoostingRegressor.md): histogram-based regression boosting
- [XGBoost](XGBoost.md): optimized regularized gradient boosting
- [CatBoost](CatBoost.md): boosting with native categorical handling

## Bagging versus boosting

| Bagging and Random Forests | Boosting |
| --- | --- |
| Fit trees independently | Fit trees sequentially |
| Individual trees may overfit | Individual trees are usually shallow and weak |
| Averaging reduces variance | Adding corrections reduces bias |
| More trees usually stabilize performance | Too many trees can overfit |
| No `learning_rate` in ordinary bagging | `learning_rate` controls correction size |
| Easy to parallelize | Sequential stages limit parallelism |

## Practical tuning rules

For Random Forest, tune `max_features`, `max_depth` or `max_leaf_nodes`, `min_samples_leaf`, and `n_estimators`. Smaller `max_features` decorrelates trees, while larger leaves smooth noisy predictions.

For boosting, tune `learning_rate` with `n_estimators` or `max_iter`, keep trees shallow initially, tune `max_leaf_nodes` and `min_samples_leaf`, and use early stopping with a large iteration limit.

See [Boosting Hyperparameter Tuning](../../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md).

## Voting and stacking

Bagging and boosting combine many models of the same family. Two other ensembles combine different model families:

- `VotingClassifier` and `VotingRegressor` average the predictions (or class probabilities with `voting="soft"`) of several fitted models.
- `StackingClassifier` and `StackingRegressor` train a final estimator on the cross-validated predictions of several base models.

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

stack = StackingClassifier(
    estimators=[
        ("linear", make_pipeline(StandardScaler(), LogisticRegression())),
        ("forest", RandomForestClassifier(random_state=0)),
    ],
    final_estimator=LogisticRegression(),
)
```

These ensembles help most when the base models make different kinds of errors. Evaluate them with the same cross-validation as their components.

## Related pages

- [Tree-Based Models](../Decision-Trees/Tree-Based-Models.md)
- [Model Overview](../README.md)
- [scikit-learn user guide: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html)

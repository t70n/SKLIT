# Feature Selection

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

Feature selection keeps a subset of columns and removes others. It differs from feature extraction: PCA creates new components, while feature selection preserves original columns.

Selection can improve speed, reduce noise, simplify interpretation, and limit overfitting. It can also remove useful variables if performed only from a single correlation or importance ranking.

## Three categories

### Filter methods

Use statistics before fitting the final model:

- variance thresholds
- missingness and data-quality rules
- univariate tests
- correlation or redundancy screening
- mutual information

They are fast but may miss interactions.

### Wrapper methods

Repeatedly fit models to evaluate feature subsets:

- recursive feature elimination
- sequential selection
- permutation-based selection

They can model interactions but are more expensive.

### Embedded methods

The estimator performs selection while fitting:

- L1 regularization
- tree-based importance
- boosting-based importance
- sparse linear models

Importance can be unstable with correlated features, so inspect stability across folds.

## Remove constants

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_reduced = selector.fit_transform(X_train)
```

Fit the selector inside a pipeline during validation.

## Select with a model

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    SelectFromModel(
        LogisticRegression(l1_ratio=1.0, solver="liblinear", C=0.1),
    ),
    LogisticRegression(max_iter=1000),
)
```

L1 can set some coefficients to zero, and `SelectFromModel` keeps the features with non-zero (or large enough) coefficients. Scaling matters because regularization depends on coefficient size; a smaller `C` removes more features.

In scikit-learn 1.8 and later, the penalty of `LogisticRegression` is selected with `l1_ratio` (`1.0` for L1, `0.0` for L2); the former `penalty="l1"` argument is deprecated. With earlier versions, write `LogisticRegression(penalty="l1", solver="liblinear", C=0.1)`.

## Recursive feature elimination

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

selector = RFE(
    LogisticRegression(max_iter=1000),
    n_features_to_select=10,
)
selector.fit(X_train, y_train)
print(selector.support_)
print(selector.ranking_)
```

For cross-validated selection, use `RFECV` and keep it inside the modeling workflow.

## Permutation importance

Permutation importance evaluates the performance drop after shuffling one feature:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    fitted_model,
    X_valid,
    y_valid,
    n_repeats=10,
    random_state=42,
    scoring="accuracy",
)
```

Use validation data, not training data, for a realistic importance estimate. Correlated variables can hide one another's importance.

## Selection after feature generation

Feature generation can produce many sums, differences, products, ratios, or group statistics. A practical pattern is:

1. create a deliberately chosen feature family
2. fit a simple model or forest
3. inspect importance across folds
4. retain a stable subset
5. refit and validate the final model

Do not generate features using the full test set or select them using test performance.

## Leakage rule

Any selection using $y$ or learned statistics must be fitted only on training data. A safe structure is:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    SelectKBest(f_classif, k=20),
    LogisticRegression(max_iter=1000),
)
```

When this pipeline is passed to cross-validation, feature selection is refitted inside every training fold.

## What to report

- selection method and threshold
- number of original and retained features
- whether selection used the target
- validation protocol
- stability across folds
- performance before and after selection
- whether removed fields are needed for future inference or auditing

## Related pages

- [Feature Engineering Overview](Feature-Engineering-Overview.md)
- [Constant and Redundant Features](../03-EDA/Constant-and-Redundant-Features.md)
- [Principal Component Analysis](Principal-Component-Analysis.md)
- [scikit-learn user guide: Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)

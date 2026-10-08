# XGBoost

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

XGBoost (eXtreme Gradient Boosting) is a highly optimized gradient-boosted tree implementation. It builds trees sequentially, where each new tree improves the current predictions by following the gradient of the loss. It adds explicit regularization, efficient split finding, missing-value handling, and strong engineering defaults.

It is especially popular for competitive and production tabular-data problems.

## Import

```python
from xgboost import XGBClassifier, XGBRegressor
```

XGBoost is an external package, so install it separately when it is not already available:

```bash
pip install xgboost
```

## Minimal classification example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
```

For regression, use `XGBRegressor` and choose a regression metric such as RMSE or MAE.

## How boosting works

Let $F_{t-1}(x)$ be the current prediction after $t-1$ trees. XGBoost adds a new tree $f_t$:

$$
F_t(x) = F_{t-1}(x) + \eta f_t(x)
$$

where $\eta$ is the learning rate. Instead of fitting the target directly, the next tree fits the direction that reduces the chosen loss.

For training examples $i$, calculate the first and second derivatives of the loss:

$$
 g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i},
\qquad
 h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}
$$

Using a second-order Taylor approximation, the objective for a new tree is approximately:

$$
\sum_i \left[g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_i)^2\right] + \Omega(f_t)
$$

A typical tree regularizer is:

$$
\Omega(f_t) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2
$$

where $T$ is the number of leaves, $w_j$ is a leaf score, $\gamma$ penalizes extra leaves, and $\lambda$ shrinks leaf scores. The second-order information helps XGBoost choose both leaf values and useful splits efficiently.

## Split gain intuition

For a candidate split, XGBoost compares the regularized objective before and after dividing the examples between two child nodes. A simplified gain is:

$$
\frac{1}{2}\left[
\frac{G_L^2}{H_L+\lambda} +
\frac{G_R^2}{H_R+\lambda} -
\frac{G^2}{H+\lambda}
\right] - \gamma
$$

where $G$ and $H$ are sums of gradients and Hessians in a node. A split is useful only when its improvement exceeds the regularization cost. This explains why XGBoost can create nonlinear interactions while still resisting unnecessary tree complexity.

## Key parameters

### `n_estimators` and `learning_rate`

The number of trees and each tree's contribution trade off against each other. Smaller learning rates often require more trees and can generalize better.

### `max_depth` or `max_leaves`

Controls tree complexity. Deeper trees capture more interactions but increase overfitting risk. `max_leaves` gives a direct limit on leaf count.

### `min_child_weight`

Prevents a split from creating a child with too little total Hessian weight. Increasing it makes the model more conservative.

### `subsample` and `colsample_bytree`

Randomly use a fraction of rows or features for each tree. Values below 1.0 can reduce variance and improve generalization.

### `reg_alpha` and `reg_lambda`

L1 and L2 regularization on leaf weights. Increase them when the model is too flexible or features are noisy.

### `gamma`

Minimum loss reduction needed to make a split. Larger values produce more conservative trees.

## Early stopping

Reserve a validation set, carved out of the training data, and stop when the validation metric stops improving. This can save work and choose a useful number of trees, but the validation set must not be reused as an unbiased final test set.

```python
from sklearn.model_selection import train_test_split

X_fit, X_valid, y_fit, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

model = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    early_stopping_rounds=50,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)], verbose=False)
print("best iteration:", model.best_iteration)
```

In XGBoost 1.6 and later, `early_stopping_rounds` is a constructor parameter (it was previously a `fit` argument). Keep the final test set untouched: it must not be used in `eval_set`.

## Preprocessing

- numerical features usually do not need scaling
- missing values can be handled natively for numerical features; the learned default direction is used at a split
- categorical data needs explicit support and careful version-specific configuration, or encoding such as one-hot encoding
- avoid target encoding outside a cross-validated pipeline because it can leak target information
- sparse matrices and large feature sets are supported, but memory and representation still matter

## When to use it

Use XGBoost when:

- the data is mostly structured/tabular
- nonlinearities and feature interactions matter
- you need a high-performing, configurable baseline
- you can afford tuning and want strong predictive performance

A linear model is preferable when a transparent additive relationship is central. HistGradientBoosting may be simpler when you want a built-in scikit-learn implementation. Neural networks may be more appropriate for raw images, audio, or very large unstructured datasets.

## Strengths and limitations

Strengths:

- excellent performance on many tabular problems
- explicit regularization and early stopping
- handles nonlinear interactions and missing numerical values
- supports classification, regression, ranking, and custom objectives

Limitations:

- many interacting parameters make tuning easy to get wrong
- trees are less transparent than a linear model
- sequential boosting can overfit noisy labels
- feature importance is not automatically causal; use permutation importance or SHAP carefully
- external dependency and version differences matter

## Related pages

- [Ensemble Models](Ensemble-Models.md)
- [CatBoost](CatBoost.md)
- [HistGradientBoostingClassifier](HistGradientBoostingClassifier.md)
- [Boosting Hyperparameter Tuning](../../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md)
- [XGBoost documentation](https://xgboost.readthedocs.io/)

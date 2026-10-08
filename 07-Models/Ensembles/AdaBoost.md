# AdaBoost

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

AdaBoost, short for Adaptive Boosting, combines many weak learners into a strong predictor. It trains the learners sequentially. After each learner, examples that were predicted poorly receive more weight, so the next learner focuses on the difficult cases.

The standard practical choice is a collection of shallow decision trees called decision stumps or weak trees.

## Import

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
```

## Minimal classification example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

For older scikit-learn versions, the base-estimator argument was named `base_estimator` instead of `estimator`.

## How it learns

For binary classification, encode labels as $y_i \in \{-1, +1\}$. A weak learner $h_t(x)$ predicts $-1$ or $+1$. AdaBoost builds an additive model:

$$
F(x) = \sum_{t=1}^{T}\alpha_t h_t(x)
$$

and predicts $\operatorname{sign}(F(x))$.

At round $t$, the training examples have weights $D_t(i)$. The learner's weighted error is:

$$
\epsilon_t = \sum_{i=1}^{n}D_t(i)\mathbf{1}[h_t(x_i) \ne y_i]
$$

A learner that performs better than random receives a positive weight:

$$
\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)
$$

The sample weights are then increased for mistakes and decreased for correct predictions:

$$
D_{t+1}(i) \propto D_t(i)\exp(-\alpha_t y_i h_t(x_i))
$$

This procedure greedily minimizes the exponential loss:

$$
\sum_i \exp(-y_iF(x_i))
$$

The model therefore concentrates effort on observations that the current ensemble finds difficult.

## Important parameters

### `n_estimators`

Number of sequential weak learners. More estimators can improve performance, but increase training and prediction time and may overfit noisy data.

### `learning_rate`

Shrinks each learner's contribution. A smaller value usually needs more estimators. The product of model capacity and learning rate controls the strength of the ensemble.

### `estimator`

The weak learner. Shallow trees are common because they capture simple rules and keep the ensemble from becoming too complex at each step.

### `algorithm` (older releases only)

Older scikit-learn releases exposed an `algorithm` parameter with the values `"SAMME"` and `"SAMME.R"`. `"SAMME.R"` was deprecated in 1.4 and removed, and recent releases (including 1.9) no longer have the parameter: `AdaBoostClassifier` always uses the SAMME algorithm, which also handles multiclass problems.

## Preprocessing

- decision trees do not require feature scaling
- numeric missing values must be handled according to the base estimator's capabilities
- encode categorical variables before fitting unless the chosen estimator natively supports them
- keep preprocessing inside a pipeline

Because AdaBoost changes weights based on errors, mislabeled examples and extreme outliers can receive disproportionate attention. Inspect data quality when performance is unstable.

## AdaBoost regression

`AdaBoostRegressor` uses a weak regressor and adapts the weight of observations with large errors. A common loss is linear, square, or exponential:

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
    n_estimators=100,
    learning_rate=0.05,
    loss="linear",
    random_state=42,
)
```

## When to use it

Use AdaBoost when:

- you want a strong ensemble from simple base learners
- the data is clean enough that hard examples are meaningful
- a compact, often accurate classifier is useful
- you want to control complexity with shallow trees and a learning rate

Gradient boosting, Random Forest, or HistGradientBoosting are often preferable for noisy tabular data. Compare them with cross-validation rather than assuming one ensemble wins.

## Strengths and limitations

Strengths:

- simple boosting idea with a clear interpretation
- shallow trees can model nonlinear boundaries and interactions
- often good accuracy without very deep trees

Limitations:

- sensitive to mislabeled observations and outliers
- sequential fitting is less parallel than bagging methods
- probability estimates may need calibration for decision-making
- weak learners that are too strong can overfit early

## Related pages

- [Ensemble Models](Ensemble-Models.md)
- [Boosting with Sample Weights](../../10-Recipes/Boosting-with-Sample-Weights.md)
- [GradientBoostingClassifier](GradientBoostingClassifier.md)
- [scikit-learn API reference: AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

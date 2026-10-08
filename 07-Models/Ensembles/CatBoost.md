# CatBoost

[Home](../../README.md) / [Models](../README.md) / [Ensembles](../README.md#ensembles)

## Idea

CatBoost is a gradient boosting library designed to work especially well with categorical features. It combines boosted decision trees with ordered statistics and ordered boosting to reduce target leakage and prediction shift caused by naive target encoding.

Its trees are usually symmetric (also called oblivious): each depth level uses the same split rule for every node at that level. This makes the model efficient and gives a regular, compact structure.

## Import

```python
from catboost import CatBoostClassifier, CatBoostRegressor
```

CatBoost is an external package:

```bash
pip install catboost
```

## Minimal classification example with categories

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

X = pd.DataFrame({
    "age": [22, 35, 41, 28, 52, 46, 31, 63],
    "plan": ["basic", "pro", "pro", "basic", "enterprise", "pro", "basic", "enterprise"],
    "region": ["north", "south", "north", "west", "south", "west", "north", "south"],
})
y = [0, 1, 1, 0, 1, 1, 0, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

categorical_columns = ["plan", "region"]
model = CatBoostClassifier(
    iterations=300,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=False,
    random_seed=42,
)
model.fit(X_train, y_train, cat_features=categorical_columns)
print(accuracy_score(y_test, model.predict(X_test)))
```

For a real dataset, use a larger validation set. This tiny example illustrates the API, not reliable statistical performance.

## How gradient boosting works

As with other gradient boosting methods, CatBoost builds an additive predictor:

$$
F_t(x) = F_{t-1}(x) + \eta f_t(x)
$$

Each new tree reduces the loss of the current ensemble. For binary classification, the model commonly optimizes log loss:

$$
L = -\sum_i \left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

where $p_i$ is the predicted probability of the positive class. The tree structure and leaf values are chosen to improve the loss, with regularization controlling complexity.

## Ordered target statistics

A categorical value such as `"pro"` has no natural numeric order. A common encoding is a target statistic, such as the average target for that category. Computing that average from all training rows can leak information: a row's own target influences the feature used to predict that same row.

CatBoost uses a random permutation of the training data. For position $i$, it computes the category statistic using only earlier rows in the permutation:

$$
\operatorname{Enc}(x_i) =
\frac{\sum_{j<i}\mathbf{1}[x_j=x_i]y_j + aP}
{\sum_{j<i}\mathbf{1}[x_j=x_i] + a}
$$

Here $P$ is a prior and $a$ controls smoothing. Early rows rely more on the prior because little history is available. At prediction time, statistics are computed from the training data, never from the target of the new row.

This ordered construction reduces leakage while retaining useful information from high-cardinality categories.

## Ordered boosting

A related problem is prediction shift: if a model is trained using statistics influenced by an example's own target, training-time representations differ from representations at prediction time. Ordered boosting uses permutations and prefix-based information so each training example is updated using a model that does not train on its own target in the same way. The goal is a closer match between training and inference conditions.

## Key parameters

### `iterations` and `learning_rate`

The number of boosting rounds and the contribution of each tree. Lower learning rates often need more iterations.

### `depth`

Depth of the symmetric trees. Larger values model more interactions but increase overfitting and runtime.

### `l2_leaf_reg`

L2 regularization applied to leaf values. Increase it when the model is too flexible.

### `loss_function`

Selects the training objective, such as `Logloss`, `MultiClass`, `RMSE`, or `MAE`. Match it to the task and desired error behavior.

### `cat_features`

Identifies categorical columns by names or indices. Do not silently convert categories to arbitrary numbers and then treat those numbers as continuous values.

### `eval_set` and `early_stopping_rounds`

Use validation data to stop when additional iterations no longer improve the selected metric.

```python
model.fit(
    X_train,
    y_train,
    cat_features=categorical_columns,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=50,
    verbose=False,
)
```

## Preprocessing

- pass categorical columns through `cat_features` so CatBoost can process them natively
- do not one-hot encode every categorical column by default; this removes much of CatBoost's main advantage
- numerical scaling is usually unnecessary for tree-based models
- represent missing categorical values consistently and do not mix incompatible types within one categorical column
- keep train, validation, and test preprocessing rules separate from target information

## Regression and ranking

For continuous targets, use `CatBoostRegressor`:

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=False,
    random_seed=42,
)
```

CatBoost also supports ranking objectives, which are useful when the goal is to order items for each query rather than predict an independent label.

## When to use it

Use CatBoost when:

- categorical features are important, numerous, or high-cardinality
- you want a strong tabular baseline with limited manual encoding
- categories arrive at prediction time and unknown values must be handled robustly
- interactions between categorical and numerical features matter

XGBoost or HistGradientBoosting can be a better fit when the data is almost entirely numerical, tight scikit-learn integration is the priority, or you need a specific ecosystem feature. Benchmark on the same splits.

## Strengths and limitations

Strengths:

- native categorical-feature support
- ordered statistics reduce target-encoding leakage
- strong performance with relatively little feature engineering
- supports classification, regression, ranking, GPU training, and text features

Limitations:

- external dependency and version-specific APIs
- training can be slower or heavier than a simple linear model
- symmetric trees may be less flexible for some datasets
- categorical column types and indices must be specified correctly
- model explanations describe associations, not causation

## Related pages

- [Ensemble Models](Ensemble-Models.md)
- [XGBoost](XGBoost.md)
- [Handling Unknown Categories](../../05-Preprocessing/Handling-Unknown-Categories.md)
- [CatBoost documentation](https://catboost.ai/docs/)

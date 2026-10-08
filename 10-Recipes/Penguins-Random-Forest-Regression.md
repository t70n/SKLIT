# Penguins Random Forest Regression

[Home](../README.md) / [Recipes](README.md)

## Goal

Fit a three-tree Random Forest to predict penguin body mass from flipper length, evaluate it with MAE, and plot individual tree predictions against the forest prediction.

## Load and split the data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("../datasets/penguins_regression.csv")
feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"

data = penguins[[feature_name]]
target = penguins[target_name]

data_train, data_test, target_train, target_test = train_test_split(
    data,
    target,
    random_state=0,
)
```

## Fit and evaluate the forest

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

random_forest = RandomForestRegressor(
    n_estimators=3,
    n_jobs=2,
    random_state=0,
)
random_forest.fit(data_train, target_train)

predictions = random_forest.predict(data_test)
mae = mean_absolute_error(target_test, predictions)
print(f"MAE: {mae:.1f}")
```

`RandomForestRegressor.score()` returns $R^2$, not MAE. Use `mean_absolute_error` explicitly when MAE is the requested metric.

## Create a prediction grid

```python
import numpy as np

flipper_lengths = pd.DataFrame(
    np.linspace(170, 230, 100), columns=[feature_name]
)
```

Using a DataFrame with the training column name avoids a feature-name warning when predicting with the forest.

## Inspect individual trees and the forest

The fitted trees are available in `random_forest.estimators_`:

```python
individual_predictions = np.array([
    tree.predict(flipper_lengths.to_numpy())
    for tree in random_forest.estimators_
])
forest_predictions = random_forest.predict(flipper_lengths)
```

The internal trees are fitted on arrays without feature names, so they receive `flipper_lengths.to_numpy()`.

Plot the results:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(
    data[feature_name],
    target,
    color="black",
    alpha=0.3,
    label="observations",
)

for index, prediction in enumerate(individual_predictions):
    plt.plot(
        flipper_lengths[feature_name],
        prediction,
        linestyle="--",
        alpha=0.7,
        label=f"tree {index + 1}",
    )

plt.plot(
    flipper_lengths[feature_name],
    forest_predictions,
    linewidth=3,
    color="tab:orange",
    label="random forest",
)
plt.xlabel(feature_name)
plt.ylabel(target_name)
plt.legend()
plt.grid(True)
plt.show()
```

The forest prediction is the average of the individual tree predictions. With only three trees, visible variability remains; increasing `n_estimators` usually stabilizes the curve.

## Inspect parameters

```python
print(random_forest.get_params())
print("number of fitted trees:", len(random_forest.estimators_))
```

Use cross-validation when comparing forests with different numbers of trees, depths, leaf sizes, or feature-sampling settings.

## Related pages

- [Random Forest](../07-Models/Ensembles/Random-Forest.md)
- [Synthetic Regression and Bootstrap Trees](Synthetic-Regression-and-Bootstrap.md)
- [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md)

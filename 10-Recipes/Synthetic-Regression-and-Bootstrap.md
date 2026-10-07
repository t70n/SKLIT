# Synthetic Regression and Bootstrap Trees

[Home](../README.md) / [Recipes](README.md)

## Goal

Create a noisy nonlinear regression dataset, visualize bootstrap samples, fit trees on different resamples, and average their predictions to understand bagging.

## Generate reproducible data

```python
import numpy as np
import pandas as pd


def generate_data(n_samples=30, seed=1):
    x_min, x_max = -3, 3
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n_samples)
    noise = 4.0 * rng.normal(size=n_samples)
    y = x**3 - 0.5 * (x + 1) ** 2 + noise
    y /= y.std()

    data_train = pd.DataFrame(x, columns=["Feature"])
    data_test = pd.DataFrame(
        np.linspace(x_min, x_max, num=300),
        columns=["Feature"],
    )
    target_train = pd.Series(y, name="Target")
    return data_train, data_test, target_train
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

data_train, data_test, target_train = generate_data()
sns.scatterplot(
    x=data_train["Feature"],
    y=target_train,
    color="black",
    alpha=0.5,
)
plt.title("Synthetic regression data")
plt.show()
```

## Draw bootstrap samples

```python

def bootstrap_sample(data, target, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        np.arange(len(target)),
        size=len(target),
        replace=True,
    )
    return data.iloc[indices], target.iloc[indices]
```

```python
for bootstrap_idx in range(3):
    data_bootstrap, target_bootstrap = bootstrap_sample(
        data_train,
        target_train,
        seed=bootstrap_idx,
    )
    plt.scatter(
        data_bootstrap["Feature"],
        target_bootstrap,
        facecolors="none",
        edgecolors="tab:blue",
        s=100,
        label="bootstrap sample",
    )
    plt.scatter(
        data_train["Feature"],
        target_train,
        color="black",
        label="original data",
    )
    plt.title(f"Bootstrap sample {bootstrap_idx}")
    plt.legend()
    plt.show()
```

Sampling is with replacement, so some original observations repeat and some are absent.

## Fit trees on bootstrap samples

```python
from sklearn.tree import DecisionTreeRegressor

bag_of_trees = []
for seed in range(3):
    data_bootstrap, target_bootstrap = bootstrap_sample(
        data_train,
        target_train,
        seed=seed,
    )
    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(data_bootstrap, target_bootstrap)
    bag_of_trees.append(tree)
```

```python
for tree_idx, tree in enumerate(bag_of_trees):
    prediction = tree.predict(data_test)
    plt.plot(
        data_test["Feature"],
        prediction,
        linestyle="--",
        label=f"tree {tree_idx + 1}",
    )

plt.scatter(data_train["Feature"], target_train, color="black")
plt.legend()
plt.title("Trees trained on different bootstraps")
plt.show()
```

The individual trees have different piecewise-constant functions because they see different samples.

## Average the predictions

```python
individual_predictions = np.array([
    tree.predict(data_test)
    for tree in bag_of_trees
])
averaged_prediction = individual_predictions.mean(axis=0)

plt.scatter(data_train["Feature"], target_train, color="black")
for prediction in individual_predictions:
    plt.plot(data_test["Feature"], prediction, "--", alpha=0.5)
plt.plot(
    data_test["Feature"],
    averaged_prediction,
    color="tab:orange",
    linewidth=3,
    label="averaged prediction",
)
plt.legend()
plt.title("Bagging reduces prediction variability")
plt.show()
```

## Use scikit-learn BaggingRegressor

```python
from sklearn.ensemble import BaggingRegressor

bagged_trees = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
bagged_trees.fit(data_train, target_train)
bagged_prediction = bagged_trees.predict(data_test)
```

The estimator's built-in bootstrap logic is preferred over manually managing models in a production workflow.

## Related pages

- [Bagging](../07-Models/Ensembles/Bagging.md)
- [Random Forest](../07-Models/Ensembles/Random-Forest.md)
- [DecisionTreeRegressor](../07-Models/Decision-Trees/DecisionTreeRegressor.md)

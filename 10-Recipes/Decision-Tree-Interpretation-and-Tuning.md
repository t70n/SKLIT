# Decision Tree Interpretation and Tuning

[Home](../README.md) / [Recipes](README.md)

## Goal

This recipe follows a complete decision-tree workflow:

- fit a shallow classification tree
- visualize its decision regions
- inspect the learned rules
- interpret class probabilities inside leaves
- compare linear and tree regression predictions
- tune tree complexity with cross-validation

## Datasets

The classification sections use two numerical features of the penguins classification data (`penguins_classification.csv` from the scikit-learn MOOC repository):

```python
import pandas as pd
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

data, target = penguins[culmen_columns], penguins[target_column]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0
)
```

The regression sections use the penguins regression data, stored in `data_regression` and `target_regression`. For a reproducible project, put preprocessing and the estimator inside a pipeline.

## Classification tree

```python
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(
    max_depth=1,
    random_state=42,
)
classifier.fit(data_train, target_train)

pred = classifier.predict(data_test)
probabilities = classifier.predict_proba(data_test)
print("accuracy:", classifier.score(data_test, target_test))
```

## Visualize decision regions

For two numerical features, `DecisionBoundaryDisplay` shows the axis-aligned regions produced by the tree:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    classifier,
    data_train,
    response_method="predict",
    multiclass_colors="tab10",
    alpha=0.5,
)

sns.scatterplot(
    data=data_train,
    x=data_train.columns[0],
    y=data_train.columns[1],
    hue=target_train,
    hue_order=classifier.classes_,
    palette="tab10",
)
plt.title("Decision regions of a decision tree")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
```

A tree creates axis-aligned boundaries because each split tests one feature at a time. Increasing depth creates more rectangular regions.

`multiclass_colors` (scikit-learn 1.7 or later) sets the color of each class when there are more than two classes; `hue_order` gives the points the same class order and therefore the same colors. With older versions, replace `multiclass_colors="tab10"` by `cmap="tab10"` together with `norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=n_classes - 0.5)`.

## Plot the learned tree

```python
from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(
    classifier,
    feature_names=data_train.columns,
    class_names=[str(value) for value in classifier.classes_],
    impurity=False,
    proportion=True,
    filled=True,
    ax=ax,
)
plt.tight_layout()
plt.show()
```

A leaf predicts the most common class in its partition. With `proportion=True`, the displayed values help interpret the class composition of each leaf.

## Interpret class probabilities

The probability predicted by a classification tree is the class proportion among training samples in the reached leaf:

$$
P(y=k\mid x)=\frac{n_{k,leaf}}{n_{leaf}}
$$

```python
import pandas as pd

sample = data_test.iloc[[0]]
probability = classifier.predict_proba(sample)[0]

probability_series = pd.Series(
    probability,
    index=classifier.classes_,
)
probability_series.plot.bar()
plt.ylabel("Estimated class probability")
plt.title("Probability for one observation")
plt.show()
```

These are empirical leaf proportions. A leaf with few samples can produce unstable or overconfident probabilities, so use `min_samples_leaf`, calibration, and validation when probabilities drive decisions.

## Regression tree prediction function

A regression tree predicts the mean target in each leaf. It therefore produces a piecewise-constant function, unlike linear regression's line or hyperplane. The regression examples use `data_regression` and `target_regression`, a dataset with one feature (flipper length) and a continuous target (body mass):

```python
penguins_regression = pd.read_csv("../datasets/penguins_regression.csv")
data_regression = penguins_regression[["Flipper Length (mm)"]]
target_regression = penguins_regression["Body Mass (g)"]
```

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

feature_name = data_regression.columns[0]
values = pd.DataFrame(
    np.linspace(
        data_regression[feature_name].min(),
        data_regression[feature_name].max(),
        300,
    ),
    columns=[feature_name],
)

linear_model = LinearRegression().fit(
    data_regression[[feature_name]],
    target_regression,
)
tree_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42,
).fit(data_regression[[feature_name]], target_regression)

linear_prediction = linear_model.predict(values)
tree_prediction = tree_model.predict(values)
```

Plot both functions:

```python
plt.scatter(
    data_regression[feature_name],
    target_regression,
    color="black",
    alpha=0.5,
    label="observations",
)
plt.plot(values[feature_name], linear_prediction, label="linear regression")
plt.plot(values[feature_name], tree_prediction, label="decision tree")
plt.legend()
plt.title("Linear versus piecewise-constant regression")
plt.show()
```

## Extrapolation warning

A regression tree cannot extrapolate a smooth trend beyond the range of training values. Its prediction outside the observed feature range follows the leaf reached by the tree and remains one of the learned leaf averages. A linear model can extrapolate, but its extrapolation is only credible when the linear trend is justified.

```python
extended_values = pd.DataFrame(
    np.linspace(
        data_regression[feature_name].min() - 20,
        data_regression[feature_name].max() + 20,
        300,
    ),
    columns=[feature_name],
)

plt.plot(
    extended_values[feature_name],
    tree_model.predict(extended_values),
    label="tree extrapolation",
)
plt.plot(
    extended_values[feature_name],
    linear_model.predict(extended_values),
    label="linear extrapolation",
)
plt.axvline(data_regression[feature_name].min(), color="grey", linestyle="--")
plt.axvline(data_regression[feature_name].max(), color="grey", linestyle="--")
plt.legend()
plt.show()
```

## Tune tree complexity

A deep tree can memorize training observations. Tune complexity using cross-validation:

```python
import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": np.arange(1, 10),
    "min_samples_leaf": [1, 5, 10, 30, 60],
}

tree_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True,
)
tree_search.fit(data_train, target_train)

print(tree_search.best_params_)
print(tree_search.best_score_)
```

Inspect train and validation scores to see underfitting and overfitting. The grid has two parameters, so fix `min_samples_leaf` at its best value before plotting the scores against the depth:

```python
results = pd.DataFrame(tree_search.cv_results_)
best_leaf = tree_search.best_params_["min_samples_leaf"]
subset = results[results["param_min_samples_leaf"] == best_leaf].copy()
subset["param_max_depth"] = subset["param_max_depth"].astype(int)

subset.plot(
    x="param_max_depth",
    y=["mean_train_score", "mean_test_score"],
    marker="o",
)
plt.ylabel("cross-validation score")
plt.title(f"min_samples_leaf={best_leaf}")
plt.show()
```

`min_samples_leaf` is an alternative to only limiting depth. Requiring larger leaves smooths the model and stabilizes class probabilities, but values that are too large can underfit.

## Classification and regression tuning

For regression, change the estimator and scoring metric:

```python
from sklearn.tree import DecisionTreeRegressor

regression_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid={
        "max_depth": np.arange(1, 10),
        "min_samples_leaf": [1, 5, 10, 30],
    },
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
regression_search.fit(data_regression, target_regression)

mae = -regression_search.best_score_
print("best cross-validated MAE:", mae)
```

The negative scoring value follows scikit-learn's convention that larger scores are better.

## Practical interpretation checklist

- A classification leaf predicts its majority class.
- A classification probability is the class fraction in that leaf.
- A regression leaf predicts a target average, or a robust alternative for some criteria.
- Depth controls the number of sequential rules.
- `min_samples_leaf` controls how much data supports each prediction.
- Small training changes can produce different trees.
- Feature importance is not causal and can favor high-cardinality features.
- Evaluate the tuned tree on untouched test data only after model selection.

## Related pages

- [DecisionTreeClassifier](../07-Models/Decision-Trees/DecisionTreeClassifier.md)
- [DecisionTreeRegressor](../07-Models/Decision-Trees/DecisionTreeRegressor.md)
- [Validation Curves](../08-Model-Evaluation/Validation-Curves.md)
- [GridSearchCV](../09-Hyperparameter-Tuning/GridSearchCV.md)

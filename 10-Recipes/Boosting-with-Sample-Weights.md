# Boosting with Sample Weights

[Home](../README.md) / [Recipes](README.md)

## Goal

Understand the core AdaBoost idea by identifying misclassified samples, increasing their training weight, and comparing the next tree with the original tree.

## Dataset

Two numerical features of the penguins classification data (`penguins_classification.csv` from the scikit-learn MOOC repository), so that the decision regions can be plotted:

```python
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

data, target = penguins[culmen_columns], penguins[target_column]
```

## Fit an initial weak tree

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

first_tree = DecisionTreeClassifier(max_depth=2, random_state=0)
first_tree.fit(data, target)

predicted = first_tree.predict(data)
misclassified_index = np.flatnonzero(target != predicted)
data_misclassified = data.iloc[misclassified_index]
```

## Visualize mistakes

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    first_tree,
    data,
    response_method="predict",
    alpha=0.5,
)
sns.scatterplot(
    data=data,
    x=data.columns[0],
    y=data.columns[1],
    hue=target,
)
sns.scatterplot(
    data=data_misclassified,
    x=data.columns[0],
    y=data.columns[1],
    marker="+",
    s=150,
    color="black",
    label="misclassified",
)
plt.show()
```

## Reweight the difficult samples

```python
sample_weight = np.zeros_like(target, dtype=float)
sample_weight[misclassified_index] = 1.0

weighted_tree = DecisionTreeClassifier(max_depth=2, random_state=0)
weighted_tree.fit(data, target, sample_weight=sample_weight)
```

This is a teaching example: it gives all previously correct samples weight zero and focuses entirely on errors. Real AdaBoost uses positive weights for every observation and updates them gradually.

The weighted training objective makes mistakes more costly. The next learner is therefore encouraged to correct regions where the previous learner failed.

## Compare remaining mistakes

```python
new_predicted = weighted_tree.predict(data)
new_misclassified_index = np.flatnonzero(target != new_predicted)
remaining = np.intersect1d(
    misclassified_index,
    new_misclassified_index,
)
print("remaining previous errors:", len(remaining))
```

## Use AdaBoost for the complete algorithm

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
    n_estimators=3,
    learning_rate=1.0,
    random_state=0,
)
adaboost.fit(data, target)
```

Each estimator is trained sequentially. Unlike bagging, boosting does not independently bootstrap the dataset; it changes the emphasis placed on observations according to previous errors and combines learners with different weights.

## Inspect each boosting round

```python
for round_index, tree in enumerate(adaboost.estimators_):
    DecisionBoundaryDisplay.from_estimator(
        tree,
        data.to_numpy(),  # the internal trees were fitted without feature names
        response_method="predict",
        alpha=0.5,
        xlabel=data.columns[0],
        ylabel=data.columns[1],
    )
    sns.scatterplot(
        data=data,
        x=data.columns[0],
        y=data.columns[1],
        hue=target,
    )
    plt.title(f"Weak learner at boosting round {round_index}")
    plt.show()
```

The individual learners can look weak or focus on different regions while the weighted ensemble becomes stronger.

## Bagging versus boosting

- Bagging fits independent learners on bootstrap samples and averages them.
- Boosting fits learners sequentially and emphasizes previous errors.
- Bagging mainly reduces variance.
- Boosting mainly reduces bias but can be sensitive to noisy labels and outliers.

## Related pages

- [AdaBoost](../07-Models/Ensembles/AdaBoost.md)
- [Bagging](../07-Models/Ensembles/Bagging.md)
- [Ensemble Models](../07-Models/Ensembles/Ensemble-Models.md)

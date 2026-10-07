# Comparing a Classifier with Dummy Baselines

[Home](../README.md) / [Recipes](README.md)

## Goal

Measure how much a logistic regression really learns on the Adult Census data by comparing it, with exactly the same cross-validation splits, to classifiers that ignore the input features:

- `DummyClassifier(strategy="most_frequent")`: always predicts the majority class
- `DummyClassifier(strategy="stratified")`: random predictions that follow the training class distribution
- `DummyClassifier(strategy="uniform")`: random predictions with equal probability for each class
- `DummyClassifier(strategy="constant")`: always predicts the minority class

The recipe also shows why balanced accuracy is easier to interpret than accuracy on imbalanced data.

## Dataset

`adult-census-numeric-all.csv` from the scikit-learn MOOC repository (numerical features of the Adult Census data and the binary target `class`). About 76% of the samples belong to the class `<=50K`.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census-numeric-all.csv")
data = adult_census.drop(columns="class")
target = adult_census["class"]
target.value_counts(normalize=True)
```

## Define the shared cross-validation strategy

All models must be evaluated on the same splits so that their scores can be compared fold by fold:

```python
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
```

## Evaluate the real model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scoring = ["accuracy", "balanced_accuracy"]

classifier = make_pipeline(StandardScaler(), LogisticRegression())
cv_results_logistic_regression = cross_validate(
    classifier, data, target, cv=cv, scoring=scoring, n_jobs=2
)
```

## Evaluate the dummy classifiers

```python
from sklearn.dummy import DummyClassifier

minority_class = target.value_counts().idxmin()

dummies = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=0),
    "Dummy (uniform)": DummyClassifier(strategy="uniform", random_state=0),
    "Dummy (constant minority)": DummyClassifier(
        strategy="constant", constant=minority_class
    ),
}

cv_results = {"Logistic regression": cv_results_logistic_regression}
for name, dummy in dummies.items():
    cv_results[name] = cross_validate(
        dummy, data, target, cv=cv, scoring=scoring, n_jobs=2
    )
```

The minority label is read from the data rather than typed by hand: in this dataset, the labels contain a leading space (`" >50K"`), which is easy to miss.

A dummy classifier ignores the features, so it does not need the `StandardScaler`. Wrapping it in the same pipeline as the real model is harmless but changes nothing.

## Collect the scores

```python
def scores_of(metric):
    """One column per model, one row per cross-validation split."""
    return pd.DataFrame(
        {name: result[f"test_{metric}"] for name, result in cv_results.items()}
    )


accuracy = scores_of("accuracy")
balanced_accuracy = scores_of("balanced_accuracy")

summary = pd.DataFrame({
    "accuracy (mean)": accuracy.mean(),
    "accuracy (std)": accuracy.std(),
    "balanced accuracy (mean)": balanced_accuracy.mean(),
})
print(summary.round(3))
```

These Series hold scores (higher is better), not errors.

## Plot the distributions of test scores

```python
import matplotlib.pyplot as plt
import numpy as np

bins = np.linspace(start=0.2, stop=0.9, num=80)
accuracy.plot.hist(bins=bins, alpha=0.6, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Accuracy")
_ = plt.title("Distribution of the test accuracy")
```

## Interpretation

- **Most frequent class**: about 0.76 accuracy, the proportion of the majority class. This is the strongest dummy baseline for accuracy on imbalanced data and the reference that the real model must beat.
- **Stratified**: worse than the most-frequent strategy; its expected accuracy is $p^2+(1-p)^2$, about 0.63 here.
- **Uniform**: about 0.5 accuracy, whatever the class proportions.
- **Constant minority class**: about 0.24 accuracy, the proportion of the minority class.
- **Logistic regression**: about 0.81 accuracy. Its histogram is well separated from the one of the most-frequent baseline, so the model uses information from the features beyond the class distribution.

The balanced accuracy of every dummy classifier is about 0.5 (the chance level $1/K$ for $K=2$ classes), whatever the strategy. A model is useful only if its balanced accuracy is clearly above 0.5; this reading does not depend on the class proportions, unlike accuracy.

## Variations

- Replace `"accuracy"` by the metric that matters for the application (for example the recall of the minority class with `make_scorer(recall_score, pos_label=minority_class)`).
- For regression, the same comparison uses `DummyRegressor(strategy="mean")` and an error metric; see [DummyRegressor](../07-Models/Baselines/DummyRegressor.md).

## Related pages

- [DummyClassifier](../07-Models/Baselines/DummyClassifier.md)
- [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md)
- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)
- [Blood Transfusion: Evaluating an Imbalanced Classifier](Blood-Transfusion-Classification-Metrics.md)

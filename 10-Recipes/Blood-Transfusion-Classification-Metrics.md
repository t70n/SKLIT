# Blood Transfusion: Evaluating an Imbalanced Classifier

[Home](../README.md) / [Recipes](README.md)

## Goal

Evaluate a classifier on an imbalanced binary problem and show why accuracy alone is misleading. The recipe covers:

- accuracy, the confusion matrix, precision, and recall
- the comparison with a dummy classifier
- balanced accuracy
- precision-recall and ROC curves with their chance levels
- cross-validated metrics with string labels, including a custom precision scorer

## Dataset

`blood_transfusion.csv` from the scikit-learn MOOC repository (UCI Blood Transfusion Service Center data). Each row describes a donor (`Recency`, `Frequency`, `Monetary`, `Time`); the target `Class` indicates whether the person gave blood (`"donated"`) or not (`"not donated"`). About 76% of the samples are `"not donated"`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]

data_train, data_test, target_train, target_test = train_test_split(
    data, target, shuffle=True, random_state=0, test_size=0.5
)
```

## Part 1: metrics on a held-out test set

### Fit a classifier

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(data_train, target_train)
target_predicted = classifier.predict(data_test)
```

### Accuracy

Accuracy counts how many times the classifier was right and divides by the number of samples:

```python
import numpy as np
from sklearn.metrics import accuracy_score

print((target_test == target_predicted)[:5])
print(f"Manual accuracy: {np.mean(target_test == target_predicted):.3f}")
print(f"Accuracy: {accuracy_score(target_test, target_predicted):.3f}")
print(f"Accuracy from score: {classifier.score(data_test, target_test):.3f}")
```

The `score` method of a classifier, part of the standard scikit-learn API, computes the same accuracy. In the course notebook, the logistic regression reaches an accuracy of about 0.78.

### Confusion matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

_ = ConfusionMatrixDisplay.from_estimator(classifier, data_test, target_test)
```

Rows are true labels and columns are predicted labels, in sorted order (`"donated"` first). Therefore:

- top left: true positives (TP), people who gave blood and were predicted as such
- bottom right: true negatives (TN), people who did not give blood and were predicted as such
- top right: false negatives (FN), people who gave blood but were predicted not to have given blood
- bottom left: false positives (FP), people who did not give blood but were predicted to have given blood

### Precision and recall

Precision, $TP/(TP+FP)$, is how likely a person actually gave blood when the classifier predicted that they did. Recall, $TP/(TP+FN)$, measures how well the classifier identifies the people who did give blood. The positive label must be given explicitly because the labels are strings:

```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(target_test, target_predicted, pos_label="donated")
recall = recall_score(target_test, target_predicted, pos_label="donated")

print(f"Precision score: {precision:.3f}")
print(f"Recall score: {recall:.3f}")
```

### Compare with a dummy classifier

```python
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(data_train, target_train)
print(
    "Accuracy of the dummy classifier: "
    f"{dummy_classifier.score(data_test, target_test):.3f}"
)
```

The dummy classifier always predicts the negative class `"not donated"` and obtains an accuracy of about 0.76. Without learning anything from the data, it is almost as accurate as the logistic regression. This is the class imbalance problem: when the classes are imbalanced, accuracy should not be used alone. Use precision and recall, or the balanced accuracy.

### Balanced accuracy

```python
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy = balanced_accuracy_score(target_test, target_predicted)
print(f"Balanced accuracy: {balanced_accuracy:.3f}")
```

The balanced accuracy is the average recall obtained on each class; it equals the accuracy when the classes are balanced. Here it is only about 0.55: the model barely detects the donors, which the accuracy concealed.

### Precision-recall and ROC curves

Both curves evaluate all the decision thresholds at once, using the predicted probabilities:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))

PrecisionRecallDisplay.from_estimator(
    classifier,
    data_test,
    target_test,
    pos_label="donated",
    name="Logistic regression",
    plot_chance_level=True,
    ax=axs[0],
)
RocCurveDisplay.from_estimator(
    classifier,
    data_test,
    target_test,
    pos_label="donated",
    name="Logistic regression",
    plot_chance_level=True,
    ax=axs[1],
)

axs[0].set_xlabel("Recall (also known as TPR or sensitivity)")
axs[0].set_ylabel("Precision (also known as PPV)")
axs[1].set_xlabel("False positive rate")
axs[1].set_ylabel("True positive rate\n(also known as sensitivity or recall)")
_ = fig.suptitle("PR and ROC curves")
```

- On the precision-recall curve, the chance level is a horizontal line at the proportion of donors; the average precision (AP) shown in the legend summarizes the curve.
- On the ROC curve, the chance level is the diagonal (AUC = 0.5); the dummy classifier lies on it. Its curve can be added with `RocCurveDisplay.from_estimator(dummy_classifier, data_test, target_test, pos_label="donated", name="Dummy", ax=axs[1])`.

Because the positive class is rare, the precision-recall curve is the more informative of the two here. `plot_chance_level` requires scikit-learn 1.3 or later.

## Part 2: cross-validated metrics with a decision tree

### Balanced accuracy with stratified folds

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
cv = StratifiedKFold()

test_score = cross_val_score(
    tree, data, target, cv=cv, n_jobs=2, scoring="balanced_accuracy"
)
print(f"The average balanced accuracy is {test_score.mean():.3f} +/- {test_score.std():.3f}")
```

### Precision with string labels

The string scorer `"precision"` assumes that the positive label is `1`, which does not exist here:

```python
try:
    scores = cross_val_score(
        tree, data, target, cv=10, scoring="precision", error_score="raise"
    )
except ValueError as exc:
    print(exc)  # pos_label=1 is not a valid label: It should be one of ['donated' 'not donated']
```

Create a scorer with the correct positive label:

```python
from sklearn.metrics import make_scorer

precision_donated = make_scorer(precision_score, pos_label="donated")
scores = cross_val_score(tree, data, target, cv=10, scoring=precision_donated)
print(f"Precision for 'donated': {scores.mean():.3f} +/- {scores.std():.3f}")
```

### Accuracy versus balanced accuracy across folds

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    tree, data, target, scoring=["accuracy", "balanced_accuracy"], cv=5
)

plt.figure(figsize=(8, 6))
plt.boxplot(
    [cv_results["test_accuracy"], cv_results["test_balanced_accuracy"]],
    tick_labels=["Accuracy", "Balanced accuracy"],
)
plt.ylabel("Score")
plt.title("Cross-validation scores: accuracy vs. balanced accuracy")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
```

`tick_labels` requires matplotlib 3.9 or later; older versions use `labels`.

## Interpretation

- The balanced accuracy is clearly lower than the accuracy: the model is biased toward the majority class `"not donated"` and misses many donors, which accuracy hides.
- The box plot shows the median score of each metric (central line) and its variability across folds (box and whiskers); wide boxes indicate a performance that depends strongly on the split.
- If the balanced accuracy is much lower than the accuracy, consider `class_weight="balanced"`, threshold tuning, resampling of the training data, or a metric focused on the class of interest (precision or recall for `"donated"`).
- If both metrics are low, the model underfits: try a more flexible model or more informative features.

## Related pages

- [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md)
- [Metrics and Scoring Overview](../08-Model-Evaluation/Metrics-and-Scoring.md)
- [DummyClassifier](../07-Models/Baselines/DummyClassifier.md)
- [Comparing a Classifier with Dummy Baselines](Dummy-Classifier-Baselines.md)

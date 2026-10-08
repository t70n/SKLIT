# Digits: Group-Aware Cross-Validation

[Home](../README.md) / [Recipes](README.md)

## Goal

Show that ignoring the grouping structure of a dataset produces an over-optimistic estimate of generalization. The digits dataset contains several samples written by the same person; the recipe compares:

1. K-fold cross-validation without shuffling
2. K-fold cross-validation with shuffling
3. group-aware cross-validation with `GroupKFold`, where all the samples of a writer stay on the same side of the split

## Dataset

`load_digits` ships with scikit-learn. It contains 1,797 images of 8x8 pixels, with gray levels from 0 to 16. According to its description (`print(digits.DESCR)`), it is a copy of the test set of the UCI handwritten digits dataset, written by 13 different people; each writer wrote the same digits several times.

```python
from sklearn.datasets import load_digits

digits = load_digits()
data, target = digits.data, digits.target
print(digits.DESCR)
```

## Define the model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

model = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1_000))
```

A `MinMaxScaler` is used because the gray level of each pixel is strictly bounded between 0 and 16. It is better suited than `StandardScaler` here: some pixels have a very low variance (pixels at the borders are almost always 0 when digits are centered), and dividing by their tiny standard deviation would produce very large scaled values.

## K-fold with and without shuffling

```python
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(shuffle=False)
test_score_no_shuffling = cross_val_score(model, data, target, cv=cv, n_jobs=2)

cv = KFold(shuffle=True, random_state=0)
test_score_with_shuffling = cross_val_score(model, data, target, cv=cv, n_jobs=2)
```

```python
import matplotlib.pyplot as plt
import pandas as pd

all_scores = pd.DataFrame(
    [test_score_no_shuffling, test_score_with_shuffling],
    index=["KFold without shuffling", "KFold with shuffling"],
).T

all_scores.plot.hist(bins=16, edgecolor="black", alpha=0.7)
plt.xlim([0.8, 1.0])
plt.xlabel("Accuracy score")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = plt.title("Distribution of the test scores")
```

Shuffling increases the mean accuracy (about 0.97 instead of 0.93) and reduces the spread across folds. The shuffled estimate is not the better one; it is the optimistic one.

## Why: the samples are grouped by writer

Assume the dataset is ordered by writer. Without shuffling, the samples of each writer stay mostly together, in either the training or the testing set. Shuffling breaks this structure: digits written by the same writer become available in both the training and the testing sets.

A writer tends to write digits in the same manner, so the model can learn to identify a writer's pattern for each digit instead of recognizing the digit itself. When the same writer appears in the test fold, this shortcut works and inflates the score; for a new writer, it does not.

## Recover the groups

The dataset does not provide writer identifiers, but the groups can be recovered approximately by looking at the target variable. Each block of samples starts with the digits written in order (`0, 1, 2, ..., 9, 0, 1, ...`), which marks the beginning of a new preprinted form; plotting `target` against the sample index makes these blocks visible. This gives the following boundaries (lower and upper sample indices of each block):

```python
from itertools import count

import numpy as np

writer_boundaries = [
    0, 130, 256, 386, 516, 646, 776, 915,
    1029, 1157, 1287, 1415, 1545, 1667, 1797,
]
groups = np.zeros_like(target)
lower_bounds = writer_boundaries[:-1]
upper_bounds = writer_boundaries[1:]

for group_id, lb, up in zip(count(), lower_bounds, upper_bounds):
    groups[lb:up] = group_id
```

These boundaries define 14 blocks for 13 writers, so they are an approximation of the true writer structure; they are nevertheless sufficient to keep samples of the same writer together.

## Group-aware cross-validation

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold()
test_score = cross_val_score(
    model, data, target, groups=groups, cv=cv, n_jobs=2
)
print(f"The average accuracy is {test_score.mean():.3f} +/- {test_score.std():.3f}")
```

The group-aware estimate is about 0.92, lower than both K-fold estimates and in particular much lower than the shuffled one.

## Interpretation

| Strategy | Mean accuracy (approximate) | Writers in both training and test sets |
| --- | ---: | --- |
| KFold without shuffling | 0.93 | Partly (fold boundaries do not match writer boundaries) |
| KFold with shuffling | 0.97 | Yes |
| GroupKFold | 0.92 | No |

Accounting for any sample grouping pattern is crucial when assessing a model's ability to generalize to new groups. Without this consideration, the results may appear overly optimistic compared to the actual performance. The same issue arises with several measurements per patient, several transactions per customer, or several images per device. Other group-aware strategies (`GroupShuffleSplit`, `LeaveOneGroupOut`, `StratifiedGroupKFold`) are described in the scikit-learn user guide.

## Related pages

- [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md)
- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)
- [MinMaxScaler](../05-Preprocessing/MinMaxScaler.md)
- [Identifiers, Leakage, Shuffling, and Sampling](../03-EDA/Identifiers-Leakage-and-Sampling.md)

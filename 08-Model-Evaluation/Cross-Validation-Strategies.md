# Cross-Validation Strategies

[Home](../README.md) / [Model Evaluation](README.md)

## Why the splitting strategy matters

Cross-validation estimates how a model will perform on new data. The estimate is only meaningful if the test folds resemble the data that the model will face in production. Standard K-fold assumes that the samples are **i.i.d.**: independent and identically distributed. When the data has an order, groups, or a time dimension, a naive split can produce misleading, usually over-optimistic, scores.

| Data structure | Risk with a naive split | Recommended splitter |
| --- | --- | --- |
| Independent samples, regression | None in particular | `KFold(shuffle=True)`, `ShuffleSplit` |
| Independent samples, classification | Folds with different class proportions; missing classes in a fold | `StratifiedKFold`, `StratifiedShuffleSplit` |
| Samples sorted by target or by source | Folds that do not represent the whole distribution | Shuffle (`KFold(shuffle=True)`) or stratify |
| Several samples per entity (patient, customer, writer) | The model recognizes the entity instead of learning the task | `GroupKFold`, `GroupShuffleSplit`, `LeaveOneGroupOut`, `StratifiedGroupKFold` |
| Time series or time-ordered events | Training on the future to predict the past | `TimeSeriesSplit` |

All splitters are passed through the `cv` argument of `cross_validate`, `cross_val_score`, `GridSearchCV`, and similar tools.

## KFold

`KFold` splits the samples into `n_splits` consecutive folds. Without shuffling, the folds follow the order of the rows:

```python
import numpy as np
from sklearn.model_selection import KFold

data_random = np.random.randn(9, 1)
cv = KFold(n_splits=3)
for train_index, test_index in cv.split(data_random):
    print("TRAIN:", train_index, "TEST:", test_index)
```

```text
TRAIN: [3 4 5 6 7 8] TEST: [0 1 2]
TRAIN: [0 1 2 6 7 8] TEST: [3 4 5]
TRAIN: [0 1 2 3 4 5] TEST: [6 7 8]
```

Use `KFold(n_splits=5, shuffle=True, random_state=0)` to shuffle the rows before splitting.

## Stratification

### The problem: ordered classes

The iris dataset is sorted by class: the first 50 rows are one species, the next 50 another, and so on. With `KFold(n_splits=3)` and no shuffling, each fold contains a single class:

- in each fold, only two of the three classes are present in the training set
- all the samples of the remaining class are used as the test set

The model is therefore unable to predict a class that it never saw during training, and the accuracy is 0 in every fold. Plotting the target against the sample index reveals the problem immediately:

```python
import matplotlib.pyplot as plt

target.plot()
plt.xlabel("Sample index")
plt.ylabel("Class")
plt.yticks(target.unique())
_ = plt.title("Class value in target y")
```

Shuffling fixes this issue, but the class frequencies still vary slightly from fold to fold, so neither the training nor the testing sets have exactly the class frequencies of the original dataset.

### The solution: `StratifiedKFold`

To preserve the original class frequencies in every fold, stratify the data by class. In scikit-learn, the cross-validation strategies that implement stratification contain `Stratified` in their names:

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=3)
results = cross_validate(model, data, target, cv=cv)
test_score = results["test_score"]
print(f"The average accuracy is {test_score.mean():.3f} +/- {test_score.std():.3f}")
```

When `cv` is an integer and the estimator is a classifier, scikit-learn already uses `StratifiedKFold`.

Stratification is especially useful for ensuring that rare classes are represented in every cross-validation split. If a class is absent from one or more splits, some classification metrics become undefined, and metrics such as precision or average precision depend on the proportion of the positive class.

### Caveat

As noted in the scikit-learn user guide, stratification makes the folds more homogeneous. In the presence of severe class imbalance, this can artificially reduce the variability of the performance metrics across folds, so the observed variability may underestimate the true uncertainty in model performance.

## Repeated and random splits

- `ShuffleSplit(n_splits, test_size)` draws independent random splits; the number of splits and the test size are chosen independently.
- `StratifiedShuffleSplit` does the same while preserving the class proportions.
- `RepeatedKFold` and `RepeatedStratifiedKFold` repeat K-fold several times with different shuffles, which gives a more stable estimate of the mean score and of its variability.
- `LeaveOneOut` uses each sample once as a test set of size one. It is expensive and its estimate has a high variance; K-fold with 5 or 10 folds is usually preferred.

## Sample grouping

### The problem: correlated samples

The handwritten digits loaded by `load_digits` were written by 13 people, and each writer wrote several samples of every digit. A writer tends to write digits in the same manner. If samples of the same writer appear in both the training and the test sets, the model can learn to recognize the writer's style instead of the digit itself, and the test score becomes optimistic.

Assuming that the dataset is ordered by writer, not shuffling the data keeps the samples of each writer mostly together in either the training or the testing sets, while shuffling breaks this structure and spreads the samples of each writer across both sets. With a logistic regression on the digits, shuffling indeed increases the mean accuracy and reduces its spread across folds: the shuffled estimate is the optimistic one.

### The solution: group-aware splitters

Group-aware splitters ensure that all the samples associated with a group (a writer) belong either to the training set or to the testing set. The group of each sample is passed with the `groups` argument:

```python
from sklearn.model_selection import GroupKFold, cross_val_score

cv = GroupKFold()
test_score = cross_val_score(model, data, target, groups=groups, cv=cv, n_jobs=2)
print(f"The average accuracy is {test_score.mean():.3f} +/- {test_score.std():.3f}")
```

Other group-aware strategies include:

- `GroupShuffleSplit`: random splits of groups
- `LeaveOneGroupOut`: each group is used once as the test set
- `StratifiedGroupKFold`: groups kept together while approximately preserving class proportions

Accounting for any sample grouping pattern is crucial when assessing a model's ability to generalize to new groups (new writers, new patients, new customers). Without this consideration, the results may appear overly optimistic compared to the actual performance. The complete workflow, including how the writer groups are approximated, is given in [Digits: Group-Aware Cross-Validation](../10-Recipes/Digits-Group-Aware-Cross-Validation.md).

## Non-i.i.d. data: time series

### The problem: the future leaks into the past

In time series, consecutive observations are correlated and the distribution can drift over time. A shuffled K-fold trains the model on observations that occur after the test observations, and close neighbors in time end up on both sides of the split. The model then appears to predict well because it effectively interpolates between known values, which is impossible when forecasting the future.

### The solution: `TimeSeriesSplit`

`TimeSeriesSplit` always trains on the past and tests on the observations that immediately follow:

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in cv.split(X):
    print("TRAIN:", train_index[[0, -1]], "TEST:", test_index[[0, -1]])
```

- the training window grows with each split (expanding window); `max_train_size` turns it into a sliding window
- `test_size` fixes the length of each test window
- `gap` leaves a number of samples out between the training and the test windows, which avoids leakage through lagged features or autocorrelation

The rows must be sorted chronologically before splitting, and features must only use information available at prediction time.

It is really important not to carelessly use a cross-validation strategy that does not respect assumptions such as having i.i.d. data. It might lead to misleading outcomes, creating the false impression that a predictive model performs well when it may not be the case in the intended real-world scenario. Beyond `TimeSeriesSplit`, scikit-learn offers useful tools for time-related feature engineering (see the corresponding example in the documentation), and scikit-learn models can be combined with specialized time-series libraries.

## Related pages

- [Cross-Validation](Cross-Validation.md)
- [Train/Test Split](../01-ML-Basics/Train-Test-Split.md)
- [Identifiers, Leakage, Shuffling, and Sampling](../03-EDA/Identifiers-Leakage-and-Sampling.md)
- [Digits: Group-Aware Cross-Validation](../10-Recipes/Digits-Group-Aware-Cross-Validation.md)
- [scikit-learn user guide: Cross-validation iterators](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators)
- [scikit-learn example: Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)

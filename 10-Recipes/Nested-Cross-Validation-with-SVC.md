# Nested Cross-Validation with SVC

[Home](../README.md) / [Recipes](README.md)

## Goal

Tune the hyperparameters `C` and `gamma` of a support vector classifier and compare two estimates of its generalization performance:

- the **non-nested** estimate: the `best_score_` of the grid search, computed on the folds used to select the hyperparameters
- the **nested** estimate: the score of the whole tuning procedure, evaluated on outer test folds that were never used for the selection

## Dataset

`load_breast_cancer` ships with scikit-learn: 569 tumors described by 30 numerical features, with a binary target (malignant or benign). About 63% of the samples belong to the majority class.

```python
from sklearn.datasets import load_breast_cancer

data, target = load_breast_cancer(return_X_y=True)
```

## Step 1: tune the model with a grid search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model_to_tune = make_pipeline(StandardScaler(), SVC())
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, 1],
}

search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid, n_jobs=2)
search.fit(data, target)

print(f"The best parameters found are: {search.best_params_}")
print(f"The mean CV score of the best model is: {search.best_score_:.3f}")
```

The features have very different scales, so the `StandardScaler` is essential for the RBF kernel. Without it, a search over `C` in `[0.1, 1, 10]` and `gamma` in `[0.01, 0.1]` finds a best score of only about 0.63, the proportion of the majority class, that is, the score of a dummy classifier.

At this stage, one should be extremely careful when using `best_score_`. It was computed on cross-validation test sets, so it is tempting to use it as the generalization performance of the model trained with the best hyperparameters. However, these scores were also used to pick the best model: knowledge from the test folds was used to select the hyperparameters. The score can therefore be too optimistic, in particular for large grids with many hyperparameters and many values per hyperparameter.

## Step 2: nested cross-validation

An inner cross-validation performs the search on the training part of each outer split; the outer cross-validation evaluates the tuned model on completely independent samples:

```python
from sklearn.model_selection import KFold, cross_val_score

# Declare the inner and outer cross-validation strategies
inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

# Inner cross-validation for parameter search
model = GridSearchCV(
    estimator=model_to_tune, param_grid=param_grid, cv=inner_cv, n_jobs=2
)

# Outer cross-validation to compute the testing score
test_score = cross_val_score(model, data, target, cv=outer_cv, n_jobs=2)
print(
    "The mean score using nested cross-validation is: "
    f"{test_score.mean():.3f} +/- {test_score.std():.3f}"
)
```

## Step 3: compare both estimates over repeated trials

A single comparison can be dominated by the randomness of the splits. Repeating it with different shuffles shows the systematic difference:

```python
import numpy as np
import pandas as pd

test_score_not_nested = []
test_score_nested = []

N_TRIALS = 20
for i in range(N_TRIALS):
    # For each trial, use cross-validation splits independent from the other trials
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=i)

    # Non-nested parameter search and scoring
    model = GridSearchCV(
        estimator=model_to_tune, param_grid=param_grid, cv=inner_cv, n_jobs=2
    )
    model.fit(data, target)
    test_score_not_nested.append(model.best_score_)

    # Nested cross-validation with parameter optimization
    test_score = cross_val_score(model, data, target, cv=outer_cv, n_jobs=2)
    test_score_nested.append(test_score.mean())

all_scores = pd.DataFrame({
    "Not nested CV": test_score_not_nested,
    "Nested CV": test_score_nested,
})
print(all_scores.describe().round(4))

n_higher = (all_scores["Not nested CV"] > all_scores["Nested CV"]).sum()
print(f"Non-nested estimate higher in {n_higher} of {N_TRIALS} trials")
```

```python
import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
all_scores.plot.box(color=color, vert=False)
plt.xlabel("Accuracy")
_ = plt.title("Comparison of mean accuracy obtained on the test sets with\nand without nested cross-validation")
```

The trials take some time: each one runs 16 candidates times 5 inner folds, three times (outer folds), plus the non-nested search.

## Interpretation

With the configuration above, a run with scikit-learn 1.9 gave:

| Estimate | Mean accuracy over 20 trials | Standard deviation |
| --- | ---: | ---: |
| Not nested (`best_score_`) | 0.978 | 0.004 |
| Nested | 0.971 | 0.005 |

The non-nested estimate was higher in 17 of the 20 trials. The difference is small here because the grid is small and the problem is easy, but it is systematic: the non-nested score includes the optimism of the selection. It grows with the size of the search space and with the noise of the data.

Report the nested score as the estimate of generalization performance. To obtain the model to deploy, fit the search once on all the data and use `best_estimator_`.

## Related pages

- [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md)
- [GridSearchCV](../09-Hyperparameter-Tuning/GridSearchCV.md)
- [SVC](../07-Models/Support-Vector-Machines/SVC.md)
- [Histogram Gradient Boosting with Nested Cross-Validation](Hist-Gradient-Boosting-Nested-CV.md)

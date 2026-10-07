# Nested Cross-Validation

[Home](../README.md) / [Hyperparameter Tuning](README.md)

## Purpose

Nested cross-validation combines two levels of cross-validation: an inner loop for hyperparameter tuning and an outer loop for an unbiased estimate of the generalization performance. It prevents the selection of hyperparameters from inflating the reported score.

## Why the score of a search is optimistic

After a search, `best_score_` is the mean cross-validated score of the best candidate:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data, target = load_breast_cancer(return_X_y=True)

model_to_tune = make_pipeline(StandardScaler(), SVC())
param_grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]}
search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid, n_jobs=2)
search.fit(data, target)

print(f"The best parameters found are: {search.best_params_}")
print(f"The mean CV score of the best model is: {search.best_score_:.3f}")
```

The scaler matters: without it, an RBF SVC on these unscaled features only reaches an accuracy of about 0.63, which is the proportion of the majority class, that is, the score of a dummy classifier.

One should be extremely careful when using this score. Since it was computed on cross-validation test sets, it is tempting to use it to assess the generalization performance of the model trained with the best hyperparameters. However, the same scores were used to pick the best model: knowledge from the test folds was used to select the hyperparameters of the model itself.

This mean score is therefore not a fair estimate of the testing error. It can be too optimistic, in particular when running a parameter search on a large grid with many hyperparameters and many possible values per hyperparameter.

## The nested procedure

- The **inner** cross-validation performs the search: it only optimizes the hyperparameters.
- The **outer** cross-validation is dedicated to estimating the testing error of the tuned model.

The inner cross-validation always receives the training set of the current outer split, so the final testing scores are always computed on completely independent sets of samples.

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

Repeated over 20 random splits, the non-nested `best_score_` of this example is higher than the nested estimate in most trials, by about 0.007 on average. The difference is small here because the grid is small and the dataset is easy, but it grows with the size of the search space. The full comparison is developed in [Nested Cross-Validation with SVC](../10-Recipes/Nested-Cross-Validation-with-SVC.md).

## Complete example: tuning the preprocessing and the model

A search can also select the preprocessing step itself. Here, the outer loop also returns the fitted searches, so that the hyperparameters chosen in each outer fold can be inspected:

```python
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)

# Define the pipeline; the preprocessor is replaced by the grid search
pipeline = Pipeline([
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier()),
])

# Define the parameter grid
param_grid = {
    "preprocessor": [
        None,
        StandardScaler(),
        MinMaxScaler(),
        QuantileTransformer(n_quantiles=100),
        PowerTransformer(method="yeo-johnson"),
    ],
    "classifier__n_neighbors": [5, 51, 101],
}

# Inner loop: GridSearchCV
inner_cv = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="balanced_accuracy",
)

# Outer loop: cross_validate
outer_cv = StratifiedKFold(n_splits=10)
nested_scores = cross_validate(
    grid_search,
    data,
    target,
    cv=outer_cv,
    scoring="balanced_accuracy",
    return_estimator=True,
)

# Generalization performance estimated by the outer loop
generalization_performance = nested_scores["test_score"]
print(f"Mean generalization performance: {np.mean(generalization_performance):.4f}")

# Best parameters selected in each outer fold
for i, fitted_search in enumerate(nested_scores["estimator"], start=1):
    print(f"Fold {i}: {fitted_search.best_params_}")
```

`PowerTransformer(method="box-cox")` is an alternative only when every feature is strictly positive; the breast cancer features include zeros, so the example uses `method="yeo-johnson"` (the default), which accepts any real value. Setting a step to `None` (or `"passthrough"`) skips it.

## What it does

The outer `cross_validate` splits the data into folds. For each outer fold, the inner `GridSearchCV` finds the best hyperparameters using only the outer training data, refits the best model on it, and this tuned model is evaluated on the held-out outer test fold. Hyperparameter selection and performance evaluation therefore use independent data.

## Why use nested validation

- **Prevents selection bias**: tuning hyperparameters on the folds used for evaluation inflates performance estimates. Nested validation separates these concerns.
- **Realistic performance estimate**: the outer loop gives an unbiased estimate of how the complete tuning procedure will perform on unseen data.
- **Stability check**: if the best parameters change a lot from one outer fold to another, the selection is unstable and the reported best parameters should be interpreted with caution.

## What nested cross-validation estimates

Nested cross-validation evaluates the **procedure** "search the grid, then refit the best model", not one particular model. Each outer fold may select different hyperparameters. To obtain the model to deploy, run the search once on the complete training data (`grid_search.fit(data, target)`) and use `best_estimator_`; the nested score is the estimate of its performance.

## Key components

- **Inner CV (`GridSearchCV`)**: searches for the best hyperparameters using cross-validation
- **Outer CV (`cross_validate`)**: provides independent folds for evaluating the tuned models
- **Two-level split**: hyperparameter tuning happens within the outer training folds, performance is measured on the outer test folds

## Advantages and cost

- provides an unbiased estimate of the generalization performance of the tuning procedure
- avoids the selection bias of reporting `best_score_`
- suitable for rigorous model evaluation and model comparison
- computationally expensive: the number of fits is roughly the number of candidates times the number of inner folds times the number of outer folds

## Important notes

- **Computational cost**: with 15 candidates, 10 inner folds, and 10 outer folds, the example above fits about 1,500 models; use `n_jobs` and smaller grids when needed
- **Return estimator**: use `return_estimator=True` to access the fitted search of each outer fold and its `best_params_`
- **Stratified folds**: use `StratifiedKFold` for classification to ensure class balance in each fold
- **Grouped or temporal data**: both loops must respect the structure of the data; see [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md)

## Related pages

- [GridSearchCV](GridSearchCV.md)
- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)
- [Nested Cross-Validation with SVC](../10-Recipes/Nested-Cross-Validation-with-SVC.md)
- [Histogram Gradient Boosting with Nested Cross-Validation](../10-Recipes/Hist-Gradient-Boosting-Nested-CV.md)
- [scikit-learn example: Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

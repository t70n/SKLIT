# Cross-Validation

[Home](../README.md) / [Model Evaluation](README.md)

## What it is

Cross-validation repeatedly splits the data into training and test subsets, fits the model multiple times, and averages the scores.

This gives a more reliable estimate of generalization performance than a single train/test split.

## Why it helps

- It reduces the risk that a lucky or unlucky split distorts the result.
- Every sample is used for testing (in K-fold), so the estimate uses the data more efficiently.
- The spread of the fold scores gives an indication of the uncertainty of the estimate.

## Typical example

```python
from sklearn.model_selection import cross_validate

cv_result = cross_validate(model, X, y, cv=5)
print(cv_result)
```

When only the test scores are needed, `cross_val_score` returns them directly as an array:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
```

Always pass the complete pipeline (preprocessing and model) so that every fold fits its own preprocessing; see [Pipeline](../05-Preprocessing/Pipeline.md).

## The `cv` argument

- an integer `k`: `StratifiedKFold(k)` for classifiers and `KFold(k)` for regressors, both without shuffling; the default is 5
- a splitter object, such as `KFold(n_splits=5, shuffle=True, random_state=0)` or `GroupKFold()`
- an iterable of `(train_indices, test_indices)` pairs

The choice of splitter matters as much as the choice of model; see [Cross-Validation Strategies](Cross-Validation-Strategies.md).

## Output structure

`cross_validate` returns a dictionary with:

- `fit_time`: time to fit the model on each training fold
- `score_time`: time to score the model on each test fold
- `test_score`: one score per fold (or `test_<name>` for each metric when `scoring` lists several metrics)
- `train_score`: with `return_train_score=True`, useful to diagnose overfitting
- `estimator`: with `return_estimator=True`, the fitted model of each fold
- `indices`: with `return_indices=True` (scikit-learn 1.3 or later), the train and test indices of each fold

```python
cv_result = cross_validate(
    model, X, y, cv=5,
    scoring=["accuracy", "balanced_accuracy"],
    return_train_score=True,
)
print(cv_result["test_balanced_accuracy"], cv_result["train_balanced_accuracy"])
```

## K-fold cross-validation

The dataset is split into `K` folds.

- one fold is used as test set
- the others are used as training set
- the process is repeated `K` times, so that each fold is used once as test set

```text
Fold 1: [test ] [train] [train] [train] [train]
Fold 2: [train] [test ] [train] [train] [train]
Fold 3: [train] [train] [test ] [train] [train]
Fold 4: [train] [train] [train] [test ] [train]
Fold 5: [train] [train] [train] [train] [test ]
```

## Example with `KFold`

```python
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
```

## ShuffleSplit

Useful when you want random repeated splits, with a number of splits and a test size chosen independently:

```python
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
```

Unlike K-fold, test sets of different splits can overlap and some samples may never be used for testing.

## Report and compare results

Report the mean and the standard deviation of the test scores, and look at their distribution rather than only the mean. A model is useful only if it clearly outperforms a dummy baseline evaluated on the same splits:

```python
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeRegressor

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

errors_tree = pd.Series(
    -cross_validate(DecisionTreeRegressor(), data, target, cv=cv,
                    scoring="neg_mean_absolute_error", n_jobs=2)["test_score"],
    name="Decision tree regressor",
)
errors_dummy = pd.Series(
    -cross_validate(DummyRegressor(strategy="mean"), data, target, cv=cv,
                    scoring="neg_mean_absolute_error", n_jobs=2)["test_score"],
    name="Dummy regressor",
)

all_errors = pd.concat([errors_tree, errors_dummy], axis=1)
all_errors.describe()
all_errors.plot.hist(bins=50, edgecolor="black")
```

If the two histograms are clearly separated, the difference is not an artifact of the particular splits. Using the same `cv` object for both models makes the comparison paired, fold by fold. See [DummyRegressor](../07-Models/Baselines/DummyRegressor.md) and [Dummy Classifier Baselines](../10-Recipes/Dummy-Classifier-Baselines.md).

## Score versus error convention

scikit-learn expects scores where higher is better. For error metrics, such as MAE, pass the negated version and negate the result:

```python
scoring="neg_mean_absolute_error"
```

See [Metrics and Scoring Overview](Metrics-and-Scoring.md#the-higher-is-better-convention).

## Inspect coefficients across folds

Use `return_estimator=True` when you need to study fitted parameters rather than only scores:

```python
cv_result = cross_validate(
    model,
    X,
    y,
    cv=10,
    return_estimator=True,
)

coefficients = []
for fitted_pipeline in cv_result["estimator"]:
    estimator = fitted_pipeline[-1]
    coefficients.append(estimator.coef_)
```

For a pipeline containing `PolynomialFeatures`, recover the transformed feature names from the fitted transformer:

```python
polynomial = cv_result["estimator"][0]["polynomialfeatures"]
feature_names = polynomial.get_feature_names_out()
```

The exact step name depends on whether `Pipeline` or `make_pipeline` was used. With a custom `Pipeline`, prefer explicit names such as `pipeline.named_steps["features"]`.

Coefficient magnitudes should be compared only after considering scaling and the feature representation. Variation across folds reveals instability that a single fitted model hides. See [Ridge Regularization and Coefficient Stability](../10-Recipes/Ridge-Regularization-and-Stability.md) for a complete workflow.

## Cross-validation and model selection

When cross-validation scores are used to choose hyperparameters, the best score becomes optimistic, because it was selected among many candidates. An unbiased estimate then requires [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md).

## Important takeaway

Cross-validation estimates the expected performance of a model on unseen data more robustly than a single benchmark, provided that the splitting strategy respects the structure of the data.

## Related pages

- [Cross-Validation Strategies](Cross-Validation-Strategies.md)
- [Metrics and Scoring Overview](Metrics-and-Scoring.md)
- [Train/Test Split](../01-ML-Basics/Train-Test-Split.md)
- [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md)
- [scikit-learn user guide: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)

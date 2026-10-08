# Manual Tuning

[Home](../README.md) / [Hyperparameter Tuning](README.md)

## What are hyperparameters?

Hyperparameters are settings that control the learning process of a model. They are chosen before training, unlike the parameters (coefficients, split thresholds) that are learned by `fit`.

Examples:

- `n_neighbors` for k-NN
- `C` for logistic regression
- `max_depth` for decision trees
- `gamma` for SVMs

## Why tuning matters

A model can perform very differently depending on the value of its hyperparameters. Default values are a reasonable starting point, not a guarantee of good performance on a specific dataset.

## Basic workflow

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression()),
])

for C in [1e-3, 1e-2, 1e-1, 1, 10]:
    model.set_params(classifier__C=C)
    cv_results = cross_validate(model, X, y)
    scores = cv_results["test_score"]
    print(f"Accuracy with C={C}: {scores.mean():.3f} +/- {scores.std():.3f}")
```

`set_params` changes a hyperparameter of a pipeline step, addressed as `<step>__<parameter>`. Cross-validation, rather than a single split, makes the comparison between values more reliable.

## How to inspect hyperparameters

```python
for param_name, param_value in model.get_params().items():
    print(param_name, param_value)
```

## Key idea

Manual tuning is useful when you want to understand a parameter and its effect, but it becomes inefficient when many parameters need to be searched. Automated searches ([GridSearchCV](GridSearchCV.md), [RandomizedSearchCV](RandomizedSearchCV.md)) implement the same loop systematically.

Selecting the best value from these cross-validation scores makes the best score optimistic; an independent estimate requires a test set or [Nested Cross-Validation](Nested-Cross-Validation.md).

## Good practice

- Start by tuning only a few parameters with a clear intuition.
- Explore positive parameters that span several orders of magnitude (`C`, `alpha`, `learning_rate`, `gamma`) on a logarithmic scale.
- Plot the training and test scores against the parameter value: this is a [validation curve](../08-Model-Evaluation/Validation-Curves.md).

## Related pages

- [GridSearchCV](GridSearchCV.md)
- [Validation Curves](../08-Model-Evaluation/Validation-Curves.md)
- [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#parameters-versus-hyperparameters)

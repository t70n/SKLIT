# Pipeline

[Home](../README.md) / [Preprocessing](README.md)

## Purpose

A `Pipeline` chains transformers and a final estimator in one object. It is the main scikit-learn abstraction for expressing a complete modeling workflow: preprocessing, feature engineering, and prediction travel together.

Example:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression())
```

The explicit equivalent is:

```python
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])
```

## Why pipelines are useful

- cleaner code
- avoids leakage between train and test data
- preprocesses data consistently during fit and predict
- easier to use with cross-validation

The leakage protection is especially important. During cross-validation, each transformer is fitted only on the training portion of each fold. If scaling, imputation, feature selection, or feature engineering is performed before cross-validation, information from the validation fold can influence the transformation and make the score optimistic.

## Pipeline behavior

During `fit`:

1. each transformer learns from the training data
2. the transformed data is passed to the final estimator
3. the model is trained

During `predict`:

1. each transformer applies its transformation
2. the model predicts

During `score`, the same transformation chain is applied before the final estimator's score method is called.

## Important property

A pipeline behaves like the last estimator, meaning:

- it has `fit`
- it has `predict`
- it has `score`

Intermediate steps must implement `fit` and `transform`. The final step must implement `fit` and can be a predictor or another estimator.

## `Pipeline` versus `make_pipeline`

Both create the same kind of object. The difference is how steps are named:

| Feature | `Pipeline` | `make_pipeline` |
| --- | --- | --- |
| Step names | Explicit names supplied by you | Generated from class names in lowercase |
| Syntax | List of `(name, estimator)` tuples | Estimators passed directly |
| Readability | More verbose but self-documenting | Compact and convenient |
| Nested parameter names | Stable custom names such as `classifier__C` | Generated names such as `logisticregression__C` |
| Best use | Reusable workflows, inspection, and tuning | Quick prototypes and simple chains |

Use `Pipeline` when you need predictable names, want to access steps programmatically, or will tune nested parameters. Use `make_pipeline` when automatic names are clear enough.

```python
from sklearn.pipeline import Pipeline, make_pipeline

explicit = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression()),
])

compact = make_pipeline(StandardScaler(), LogisticRegression())

explicit.set_params(classifier__C=0.1)
compact.set_params(logisticregression__C=0.1)
```

## Naming and accessing steps

```python
pipeline.named_steps["classifier"]
pipeline.steps
pipeline[-1]
```

`make_pipeline` generates names from estimator class names, for example `StandardScaler` becomes `standardscaler` and `LogisticRegression` becomes `logisticregression`. Explicit names are often safer in larger projects because they remain meaningful if implementation details change.

After fitting, the final estimator's learned attributes are available through the step:

```python
pipeline.fit(X_train, y_train)
coefficients = pipeline.named_steps["classifier"].coef_
```

`get_params()` exposes all nested parameters, named `<step>__<parameter>`, whether or not the pipeline is fitted. It is useful when constructing a `GridSearchCV` parameter grid.

## Inspecting intermediate results

A pipeline can be indexed and sliced like a list:

```python
pipeline[0]                          # first step
pipeline[-1]                         # final estimator
X_transformed = pipeline[:-1].transform(X_test)  # output of all preprocessing steps
feature_names = pipeline[:-1].get_feature_names_out()
```

Slicing returns a new pipeline that shares the fitted steps, which is convenient for checking what the final estimator actually receives.

## Example with mixed data

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_columns),
    (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
)

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
```

## Pipeline with feature engineering

For the complete feature-engineering map, see [Feature Engineering Overview](../06-Feature-Engineering/Feature-Engineering-Overview.md).

Transformers can be chained before the estimator:

```python
from sklearn.preprocessing import PolynomialFeatures

model = Pipeline([
    ("scale", StandardScaler()),
    ("features", PolynomialFeatures(degree=2, include_bias=False)),
    ("classifier", LogisticRegression(C=0.1, max_iter=1000)),
])
```

For mixed columns, place a `ColumnTransformer` first, then add global feature engineering only when the transformed representation is appropriate. Feature expansion can become very large, so tune its complexity with cross-validation.

## Cross-validation and model selection

Pass the complete pipeline to `cross_validate` or `GridSearchCV`, never just the final estimator:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    model,
    param_grid={"classifier__C": [0.01, 0.1, 1.0]},
    cv=5,
    scoring="accuracy",
)
search.fit(X, y)
```

This makes every fold fit its own preprocessing state and allows preprocessing and model parameters to be tuned as one workflow.

## Caching expensive transformations

When a costly transformer is refitted many times during a hyperparameter search, the `memory` parameter caches its fitted state:

```python
from tempfile import mkdtemp

model = make_pipeline(StandardScaler(), LogisticRegression(), memory=mkdtemp())
```

Caching only helps when the cached steps do not change between candidates.

## Related pages

- [ColumnTransformer](ColumnTransformer.md)
- [Why Preprocessing Matters](Why-Preprocessing-Matters.md)
- [GridSearchCV](../09-Hyperparameter-Tuning/GridSearchCV.md)
- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)
- [scikit-learn user guide: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)

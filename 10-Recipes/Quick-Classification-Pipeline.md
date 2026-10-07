# Quick Classification Pipeline

[Home](../README.md) / [Recipes](README.md)

## Typical template

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

numerical_columns = [...]
categorical_columns = [...]

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_columns),
    (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
)

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

cv_results = cross_validate(model, X, y, cv=5)
print(cv_results["test_score"].mean())
```

## What this does

- scales numerical features
- encodes categorical features
- fits a logistic regression model
- evaluates with cross-validation (stratified 5-fold for a classifier)

## Variant with missing values

```python
from sklearn.impute import SimpleImputer

numerical_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    (numerical_pipeline, numerical_columns),
    (categorical_pipeline, categorical_columns),
)
```

## Check against a baseline and choose the metric

```python
from sklearn.dummy import DummyClassifier

scoring = ["accuracy", "balanced_accuracy"]
dummy = DummyClassifier(strategy="most_frequent")

model_scores = cross_validate(model, X, y, cv=5, scoring=scoring)
dummy_scores = cross_validate(dummy, X, y, cv=5, scoring=scoring)

for name in scoring:
    model_mean = model_scores[f"test_{name}"].mean()
    dummy_mean = dummy_scores[f"test_{name}"].mean()
    print(f"{name}: model {model_mean:.3f} | dummy {dummy_mean:.3f}")
```

With imbalanced classes, prefer balanced accuracy or the precision and recall of the class of interest; see [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md).

## Use when

- you need a solid baseline quickly
- data has both numerical and categorical variables
- you want a clean, reusable, moderate-complexity pipeline

For a stronger nonlinear model on tabular data, replace the logistic regression by `HistGradientBoostingClassifier` (which does not need scaling) with an `OrdinalEncoder` for the categorical columns.

## Related pages

- [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)
- [LogisticRegression](../07-Models/Linear-Models/LogisticRegression.md)
- [Adult Census Classification](Adult-Census-Classification.md)

# Adult Census Classification

[Home](../README.md) / [Recipes](README.md)

## Goal

Build and compare logistic-regression pipelines on the Adult Census dataset. The recipe shows how to:

- evaluate a complete workflow with stratified 10-fold cross-validation
- compare numerical-only and mixed numerical/categorical data
- inspect coefficients after preprocessing
- add multiplicative feature interactions
- compare models fold by fold without leaking information

The dataset is expected at `../datasets/adult-census.csv` relative to the notebook or script that runs the example.

## Load the data

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census["class"]
```

`education-num` is removed because it duplicates the information represented by the education feature in this exercise.

## Model 1: numerical features only

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

numerical_data = adult_census.select_dtypes(["integer", "floating"])
numerical_data = numerical_data.drop(columns=["education-num"])

numerical_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

numerical_cv = cross_validate(
    numerical_pipeline,
    numerical_data,
    target,
    cv=10,
    scoring="accuracy",
    return_estimator=True,
)

numerical_scores = numerical_cv["test_score"]
print(f"Accuracy: {numerical_scores.mean():.3f} +/- {numerical_scores.std():.3f}")
```

`cross_validate` fits a fresh scaler and classifier within every fold. `return_estimator=True` keeps the ten fitted pipelines for later inspection.

## Inspect numerical coefficients

The final classifier is the last step of each fitted pipeline:

```python
import numpy as np
import pandas as pd

coefficients = []
for estimator in numerical_cv["estimator"]:
    logistic_regression = estimator[-1]
    coefficients.append(logistic_regression.coef_[0])

absolute_coefficients = pd.DataFrame(
    np.abs(coefficients),
    columns=numerical_data.columns,
)

most_important = absolute_coefficients.median().idxmax()
print("Largest median absolute coefficient:", most_important)
absolute_coefficients.boxplot()
```

Because the numerical features are standardized, absolute coefficient magnitudes are more comparable than they would be in the original units. They indicate association within this fitted model, not causation. Examining the distribution across folds shows whether the apparent importance is stable.

## Model 2: numerical and categorical features

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_columns = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data = adult_census.drop(columns=["class", "education-num"])

preprocessor = make_column_transformer(
    (
        StandardScaler(),
        numerical_columns,
    ),
    (
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=0.01,
        ),
        categorical_columns,
    ),
)

full_pipeline = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000),
)

full_cv = cross_validate(
    full_pipeline,
    data,
    target,
    cv=10,
    scoring="accuracy",
    return_estimator=True,
)

full_scores = full_cv["test_score"]
print(f"Accuracy: {full_scores.mean():.3f} +/- {full_scores.std():.3f}")
print("Full model wins on", (full_scores > numerical_scores).sum(), "of 10 folds")
```

The comparison is paired because both models use the same fold assignments. A higher mean is useful, but also inspect the standard deviation and the fold-by-fold differences.

## Recover transformed feature names

`OneHotEncoder` creates several columns from one categorical column. Retrieve those names from the fitted preprocessor rather than guessing them:

```python
fitted_preprocessor = full_cv["estimator"][0].named_steps["columntransformer"]
feature_names = fitted_preprocessor.get_feature_names_out().tolist()
```

`get_feature_names_out()` preserves the `ColumnTransformer` output order and includes the transformer prefixes, such as `standardscaler__age` and `onehotencoder__workclass_Private`. This is safer than manually concatenating names because the coefficient array must align exactly with the transformed matrix.

## Inspect mixed-data coefficients

```python
fitted_classifier = full_cv["estimator"][0].named_steps["logisticregression"]
coefficients = fitted_classifier.coef_[0]

feature_coefficients = pd.Series(coefficients, index=feature_names)
print(feature_coefficients.abs().sort_values(ascending=False).head(10))
```

This identifies the largest coefficient magnitudes for one fold. For a more stable conclusion, repeat this across all returned estimators and summarize medians or intervals. With the default `drop=None`, the encoder keeps one column per category, so the coefficients of a categorical feature are only defined up to a common shift and are made unique by the regularization: compare them with each other within a feature rather than reading each one in isolation.

## Model 3: add feature interactions

After preprocessing, add pairwise interaction features. Do not add a bias column because logistic regression already learns an intercept:

```python
from sklearn.preprocessing import PolynomialFeatures

interaction_pipeline = make_pipeline(
    preprocessor,
    PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False,
    ),
    LogisticRegression(C=0.01, max_iter=1000),
)

interaction_cv = cross_validate(
    interaction_pipeline,
    data,
    target,
    cv=10,
    scoring="accuracy",
)

interaction_scores = interaction_cv["test_score"]
print(f"Accuracy: {interaction_scores.mean():.3f} +/- {interaction_scores.std():.3f}")
print("Interaction model wins on", (interaction_scores > full_scores).sum(), "of 10 folds")
```

The interaction-only expansion creates products such as a numerical feature multiplied by a one-hot category. This lets the effect of one feature depend on another, but the number of columns can grow rapidly. A smaller `C` adds stronger regularization to control the expanded model.

## Interpretation and cautions

- A cross-validation score estimates generalization; it is not a final unbiased test result if the same folds were used repeatedly to choose the workflow.
- Reserve a final test set when the pipeline or hyperparameters are selected.
- Accuracy can hide class imbalance; also inspect the classification metrics in the evaluation wiki.
- Coefficient magnitude is only comparable after considering scaling, encoding, regularization, and the feature representation.
- For a pipeline built with `make_pipeline`, generated step names can be inspected with `pipeline.named_steps` or `pipeline.steps`.
- If categories are unseen at prediction time, `handle_unknown="ignore"` prevents the encoder from failing.

## Related pages

- [LogisticRegression](../07-Models/Linear-Models/LogisticRegression.md)
- [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md)
- [Polynomial Features](../06-Feature-Engineering/Polynomial-Features.md)
- [Comparing a Classifier with Dummy Baselines](Dummy-Classifier-Baselines.md)

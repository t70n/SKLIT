# Quick Regression Pipeline

[Home](../README.md) / [Recipes](README.md)

## Typical template

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

numerical_columns = [...]
categorical_columns = [...]

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_columns),
    (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
)

model = make_pipeline(preprocessor, LinearRegression())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

`score` returns the $R^2$ of the regressor.

## Recommended refinements

- Replace `LinearRegression` by `RidgeCV()`, which adds a regularization strength selected by internal cross-validation. One-hot encoding with an intercept creates collinear columns, and many one-hot columns can make an unregularized fit unstable.
- Add imputers when the data has missing values:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV

numerical_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    (numerical_pipeline, numerical_columns),
    (categorical_pipeline, categorical_columns),
)
model = make_pipeline(preprocessor, RidgeCV())
```

## Evaluate with cross-validation and a baseline

```python
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate

model_mae = -cross_validate(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)["test_score"]
dummy_mae = -cross_validate(
    DummyRegressor(strategy="median"), X, y, cv=5, scoring="neg_mean_absolute_error"
)["test_score"]

print(f"Model MAE: {model_mae.mean():.2f} +/- {model_mae.std():.2f}")
print(f"Dummy MAE: {dummy_mae.mean():.2f} +/- {dummy_mae.std():.2f}")
```

The median is the best constant prediction for the MAE, which makes `strategy="median"` the fair baseline for this metric. See [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md).

## Use when

- target is continuous
- you want a baseline regression model
- the data contains mixed data types

For nonlinear relationships, `HistGradientBoostingRegressor` is a strong next step; for a skewed target, see [Regression Error Analysis and Target Transformation](Regression-Error-Analysis-and-Target-Transformation.md).

## Related pages

- [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md)
- [Linear Regression](../07-Models/Linear-Models/Linear-Regression.md)
- [DummyRegressor](../07-Models/Baselines/DummyRegressor.md)
- [Ridge Regularization and Coefficient Stability](Ridge-Regularization-and-Stability.md)

# ColumnTransformer

[Home](../README.md) / [Preprocessing](README.md)

## Purpose

`ColumnTransformer` applies different transformations to different subsets of columns and concatenates the results into a single feature matrix.

This is essential when a dataset contains:

- numerical columns
- categorical columns
- both types together

Combined with a [Pipeline](Pipeline.md), it lets preprocessing and the model be fitted, cross-validated, and tuned as a single estimator.

## Example

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_columns),
    (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
)
```

## Key idea

It splits the original input data into column subsets, applies the appropriate transformer to each subset, and concatenates the outputs side by side.

## Explicit names with `ColumnTransformer`

`make_column_transformer` generates the transformer names automatically (`standardscaler`, `onehotencoder`). The explicit class takes `(name, transformer, columns)` triples, which gives stable names for inspection and hyperparameter tuning:

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numerical_columns),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
])
```

A nested parameter is then addressed as `preprocessor__categorical__min_frequency` when the column transformer is the step named `preprocessor` of a pipeline.

## Selecting the columns

The `columns` element of each triple can be:

- a list of column names, such as `["age", "hours-per-week"]`
- a `make_column_selector`, which selects columns by dtype or name pattern when the transformer is fitted
- a list of integer positions or a boolean mask, for NumPy arrays

```python
from sklearn.compose import make_column_selector

numerical_selector = make_column_selector(dtype_include="number")
categorical_selector = make_column_selector(dtype_exclude="number")

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_selector),
    (OneHotEncoder(handle_unknown="ignore"), categorical_selector),
)
```

## Columns that are not listed: `remainder`

By default, columns that are not selected by any transformer are dropped (`remainder="drop"`). Use `remainder="passthrough"` to keep them unchanged, or pass a transformer to apply it to all remaining columns:

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

preprocessor = ColumnTransformer(
    [("categorical", ordinal_encoder, categorical_columns)],
    remainder="passthrough",
)
```

This pattern is common for tree-based models: categorical columns are ordinal-encoded and numerical columns pass through unscaled.

## Inspecting the result

```python
preprocessor.fit(X_train)
preprocessor.get_feature_names_out()          # output column names, in output order
preprocessor.named_transformers_["categorical"]  # fitted transformer by name
```

Output names are prefixed with the transformer name, such as `numeric__age` or `categorical__workclass_Private`. Set `verbose_feature_names_out=False` to drop the prefixes when the names are unique. Use `set_output(transform="pandas")` to obtain a DataFrame instead of an array, provided every transformer produces dense output.

## Why it is useful

- applies different transformers to different subsets of the data
- combines numerical and categorical preprocessing in one step
- produces a single transformed dataset for the model
- is itself a transformer, so it can be used inside a pipeline and tuned with `GridSearchCV`

## Typical mixed-data workflow

```python
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_columns),
    (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
)

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

## Important principle

The preprocessing is fitted on the training data only, then reused for validation or test data. Passing the complete pipeline to `cross_validate` or `GridSearchCV` guarantees that every fold fits its own scaler and encoder.

## Related pages

- [Pipeline](Pipeline.md)
- [OrdinalEncoder vs OneHotEncoder](OrdinalEncoder-vs-OneHotEncoder.md)
- [Quick Classification Pipeline](../10-Recipes/Quick-Classification-Pipeline.md)
- [scikit-learn API reference: ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)

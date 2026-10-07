# Missing Values

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Inspect missingness

```python
missing = pd.DataFrame({
    "count": df.isna().sum(),
    "fraction": df.isna().mean(),
}).sort_values("fraction", ascending=False)
print(missing)

rows_with_missing = df[df.isna().any(axis=1)]
```

Missingness can be represented by `NaN`, `None`, empty strings, whitespace, `-999`, `"NA"`, or `"Unknown"`. Only convert sentinel values when the data definition confirms they mean missing.

```python
df["income"] = df["income"].replace(
    [-999, "-999", "NA", ""],
    pd.NA,
)
```

## Drop missing values

```python
df = df.dropna()                         # any missing cell
df = df.dropna(subset=["age", "target"])
df = df.dropna(axis="columns", how="all")
df = df.dropna(thresh=5)                 # at least 5 non-missing cells
```

Dropping rows can change the population and target distribution. Compare before and after counts and class proportions.

## Fill missing values

```python
df["age"] = df["age"].fillna(df["age"].median())
df["country"] = df["country"].fillna("Unknown")
df = df.fillna({"age": df["age"].median(), "sex": "Unknown"})
```

Mean, median, mode, a constant category, and model-based imputation make different assumptions. Median is often more robust for skewed numeric variables.

## Forward and backward fill

These are appropriate for ordered measurements, not arbitrary tabular rows:

```python
df["sensor"] = df["sensor"].ffill()
df["sensor"] = df["sensor"].bfill()
```

For time series, sort by time and use only information available at the prediction time.

## Leakage-safe imputation

For machine learning, learn imputation statistics on training data only:

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    StandardScaler(),
)
```

Put the imputer inside the full pipeline passed to cross-validation. Calculating a median from all rows before splitting lets validation data influence training.

`SimpleImputer` supports four strategies:

| `strategy` | Replacement value | Typical use |
| --- | --- | --- |
| `"mean"` | Column mean | Roughly symmetric numerical features |
| `"median"` | Column median | Skewed numerical features or features with outliers |
| `"most_frequent"` | Mode | Categorical features, or discrete numerical features |
| `"constant"` | `fill_value` | Explicit placeholder such as `"missing"` for categories |

`KNNImputer` and `IterativeImputer` use the other features to estimate a missing value; they are compared in [MICE and Iterative Imputation](MICE-and-Iterative-Imputation.md).

Some estimators handle missing numerical values natively, so an imputer is optional for them: `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, and, in recent scikit-learn releases, decision trees and random forests. See [Model Overview](../07-Models/README.md).

## Missingness indicators

Missingness itself can carry information. `add_indicator=True` adds binary flags for columns that contained missing values during fitting. You can also create one explicitly:

```python
df["income_was_missing"] = df["income"].isna()
```

Do not assume missingness is random. Compare outcomes and other variables between missing and observed groups.

## Questions before choosing an action

- Is the value truly missing or a valid category such as `Unknown`?
- Is the missingness caused by a measurement process?
- Would the value be available at prediction time?
- Does dropping rows remove a meaningful subgroup?
- Should a missingness indicator be retained?
- Is the operation fitted inside each training fold?

For relationship-aware iterative imputation and multiple imputation, see [MICE and Iterative Imputation](MICE-and-Iterative-Imputation.md).

## Related pages

- [MICE and Iterative Imputation](MICE-and-Iterative-Imputation.md)
- [Data Cleaning Overview](Data-Cleaning-Overview.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)
- [scikit-learn user guide: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)

# Pandas DataFrame Manipulation Cheat Sheet

[Home](../README.md) / [Pandas](README.md)

This page is a practical reference for the most common DataFrame operations used in data science and machine learning. Most operations return a new DataFrame; assign the result explicitly unless you intentionally use an inplace operation.

## Import and load data

```python
import numpy as np
import pandas as pd

# Common formats
df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Write results
df.to_csv("cleaned.csv", index=False)
df.to_parquet("cleaned.parquet", index=False)
```

Useful read options:

```python
df = pd.read_csv(
    "data.csv",
    na_values=["?", "NA", "unknown", ""],
    parse_dates=["date"],
)
```

Keep the raw DataFrame unchanged:

```python
raw = pd.read_csv("data.csv")
df = raw.copy()
```

## Inspect a DataFrame

```python
df.head()                 # first rows
df.tail()                 # last rows
df.sample(5, random_state=42)  # random rows
df.shape                  # (rows, columns)
df.size                   # total cells
df.index
df.columns
df.dtypes
df.info()
df.describe(include="all").T
```

Inspect a single column:

```python
df["age"].head()
df["age"].unique()
df["age"].nunique(dropna=True)
df["age"].value_counts(dropna=False)
df["age"].value_counts(normalize=True, dropna=False)
```

Check the data quality quickly:

```python
summary = pd.DataFrame({
    "dtype": df.dtypes,
    "missing": df.isna().sum(),
    "missing_fraction": df.isna().mean(),
    "unique": df.nunique(dropna=False),
}).sort_values("missing_fraction", ascending=False)
print(summary)
print("duplicate rows:", df.duplicated().sum())
```

## Select columns

### One column

```python
series = df["age"]       # Series
one_column = df[["age"]] # DataFrame
```

The double brackets matter when the result must remain a DataFrame.

### Multiple columns

```python
subset = df[["age", "income", "class"]]
```

### Include or exclude by dtype

```python
numeric = df.select_dtypes(include="number")
categorical = df.select_dtypes(include=["object", "category", "string"])
boolean = df.select_dtypes(include="bool")
non_numeric = df.select_dtypes(exclude="number")
```

A numeric-looking integer column can still be categorical in meaning. Inspect its values and domain definition before selecting it as numeric.

### Select columns by name pattern

```python
age_columns = df.filter(like="age")
measurements = df.filter(regex="^(height|weight)_")
selected = df.filter(items=["age", "income"])
```

### Select columns by position

```python
first_three = df.iloc[:, :3]
last_column = df.iloc[:, -1]
```

Use names rather than positions when the schema can change.

## Select rows

### By position with `iloc`

```python
first_ten = df.iloc[:10]
row_five = df.iloc[5]
rows_and_columns = df.iloc[:10, :3]
```

### By labels with `loc`

```python
row = df.loc["customer_001"]
subset = df.loc[:, ["age", "income"]]
```

`loc` uses labels; `iloc` uses integer positions.

### Boolean filtering

```python
adults = df[df["age"] >= 18]

subset = df[
    (df["age"] >= 18)
    & (df["hours_per_week"] > 30)
]
```

Use parentheses around each condition. Use `&`, `|`, and `~` instead of Python's `and`, `or`, and `not`.

### `isin`, `between`, and `query`

```python
selected = df[df["country"].isin(["France", "Germany"])]
working_age = df[df["age"].between(18, 67, inclusive="both")]
not_male = df[~df["sex"].isin(["Male"])]

selected = df.query("age >= 18 and hours_per_week > 30")
```

For variables, use `@` inside `query`:

```python
minimum_age = 18
adults = df.query("age >= @minimum_age")
```

### Select rows with missingness

```python
missing_income = df[df["income"].isna()]
complete_income = df[df["income"].notna()]
rows_with_any_missing = df[df.isna().any(axis=1)]
rows_without_missing = df.dropna()
```

## Set and update values safely

Use `.loc` for conditional assignment:

```python
df.loc[df["age"] < 0, "age"] = np.nan
df.loc[df["sex"].eq("M"), "sex"] = "male"
```

Avoid chained assignment such as `df[df["age"] < 0]["age"] = np.nan`. It may modify a temporary view rather than the original DataFrame.

Update several columns:

```python
df.loc[df["income"].isna(), ["income", "income_source"]] = [0, "missing"]
```

Create a filtered copy explicitly:

```python
adults = df.loc[df["age"] >= 18].copy()
adults["age_group"] = pd.cut(
    adults["age"],
    bins=[0, 18, 30, 50, np.inf],
    labels=["child", "young_adult", "adult", "senior"],
)
```

## Add columns

### Direct assignment

```python
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
df["is_adult"] = df["age"] >= 18
```

### `assign`

`assign` returns a new DataFrame and is convenient for method chains:

```python
df = (
    df.assign(
        bmi=lambda data: data["weight_kg"] / data["height_m"] ** 2,
        is_adult=lambda data: data["age"] >= 18,
    )
)
```

Later expressions can use columns created earlier in the same `assign` call.

### Insert at a specific position

```python
df.insert(
    loc=1,
    column="is_adult",
    value=df["age"].ge(18),
)
```

`insert` modifies the DataFrame and raises an error if the column name already exists unless you remove or rename it first.

## Delete columns

```python
df = df.drop(columns=["id", "raw_text"])
df = df.drop(columns="target")
```

Drop by position only when necessary:

```python
df = df.drop(df.columns[0], axis="columns")
```

Select the feature matrix and target explicitly:

```python
X = df.drop(columns="target")
y = df["target"]
```

Dropping a column is irreversible for that object, so keep `raw` or select `X` instead of destroying the source when you may need the column later.

## Add and delete rows

### Add one row

Prefer `concat` over the removed `DataFrame.append` method:

```python
new_row = pd.DataFrame([{
    "age": 35,
    "income": 50000,
    "sex": "female",
}])

df = pd.concat([df, new_row], ignore_index=True)
```

For many rows, build a list of records and concatenate once. Repeatedly concatenating inside a loop is slow.

### Delete rows

```python
df = df.drop(index=[2, 5])
df = df.drop(index=df[df["age"] < 0].index)
df = df.loc[df["age"] >= 0].copy()
```

Reset the index after filtering when the old labels no longer carry meaning:

```python
df = df.reset_index(drop=True)
```

Keep the old index as a column when it is meaningful:

```python
df = df.reset_index(names="original_index")
```

## Rename columns and indexes

### Rename selected columns

```python
df = df.rename(columns={
    "old name": "new_name",
    "hours-per-week": "hours_per_week",
})
```

### Rename all columns

```python
df.columns = [column.strip().lower() for column in df.columns]
```

Ensure the replacement list has exactly the same length as the number of columns.

### Rename index labels

```python
df = df.rename(index={0: "first", 1: "second"})
df.index.name = "record_id"
```

For ML tables, a meaningful index can help trace records, but do not accidentally pass an identifier index as a feature.

## Reorder columns

### Explicit order

```python
column_order = ["target", "age", "income", "sex"]
df = df[column_order]
```

### Move one column to the front

```python
column = "target"
df = df[[column] + [name for name in df.columns if name != column]]
```

### Put target last and split features

```python
target = "target"
feature_columns = [name for name in df.columns if name != target]
df = df[feature_columns + [target]]
X = df[feature_columns]
y = df[target]
```

Column order usually does not affect a named pandas workflow, but consistent order helps reproducibility, exports, and coefficient interpretation.

## Shuffle rows and sample data

### Shuffle the complete table

```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

This is appropriate for order-independent data. Do not shuffle time series before a chronological split.

### Sample rows or columns

```python
sample_rows = df.sample(n=100, random_state=42)
fraction = df.sample(frac=0.1, random_state=42)
sample_columns = df.sample(n=3, axis="columns", random_state=42)
```

### Reproducible train/test split

Use scikit-learn for modeling splits:

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["target"],
)
```

For time-ordered data, use a chronological split instead of random shuffling.

## Sort values

```python
df = df.sort_values("age")
df = df.sort_values(["country", "income"], ascending=[True, False])
df = df.sort_index()
```

Use a stable sort when the ordering of equal values must preserve its previous order:

```python
df = df.sort_values("date", kind="stable")
```

## Missing values

### Find missing values

```python
df.isna().sum()
df.isna().mean().sort_values(ascending=False)
df[df.isna().any(axis=1)]
```

### Drop missing values

```python
df = df.dropna()                         # any missing cell
clean_rows = df.dropna(subset=["age", "target"])
df = df.dropna(axis="columns", how="all")
df = df.dropna(thresh=5)                 # keep rows with >= 5 non-missing cells
```

Never drop missing values automatically without checking whether missingness is informative or whether dropping changes the target distribution.

### Fill missing values

```python
df["age"] = df["age"].fillna(df["age"].median())
df["country"] = df["country"].fillna("Unknown")
df = df.fillna({"age": df["age"].median(), "sex": "Unknown"})
```

For ML evaluation, fit imputation only on training data. Use `SimpleImputer` inside a scikit-learn pipeline rather than calculating statistics before cross-validation.

### Forward and backward fill

Useful for ordered measurements, but dangerous for unrelated rows:

```python
df["sensor"] = df["sensor"].ffill()
df["sensor"] = df["sensor"].bfill()
```

## Duplicates and uniqueness

```python
df.duplicated().sum()
df[df.duplicated(keep=False)]
df.drop_duplicates()
df.drop_duplicates(subset=["customer_id"], keep="last")
df["customer_id"].is_unique
df["customer_id"].duplicated().sum()
```

Before deduplicating, define what one row represents. Repeated measurements can be valid duplicates in the feature values.

## Data types

```python
df.dtypes
df.convert_dtypes()
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["category"] = df["category"].astype("category")
```

Use `errors="coerce"` only when invalid values should become missing and will be audited afterward.

## Strings and categories

```python
df["city"] = df["city"].astype("string").str.strip().str.lower()
df["email_domain"] = df["email"].str.lower().str.extract(r"@(.+)$")[0]
df["code"] = df["code"].str.replace("-", "", regex=False)
```

Useful string methods include `.str.contains()`, `.str.startswith()`, `.str.endswith()`, `.str.split()`, `.str.len()`, and `.str.replace()`.

Normalize categories only when capitalization and whitespace are not meaningful. Use `value_counts(dropna=False)` before and after.

## Map and replace values

```python
df["sex"] = df["sex"].replace({"M": "male", "F": "female"})
df["priority_code"] = df["priority"].map({
    "low": 1,
    "medium": 2,
    "high": 3,
})
```

`replace` substitutes values; `map` maps values and returns missing values for unmapped labels unless a default is handled explicitly.

## Apply functions and vectorization

Prefer vectorized operations:

```python
df["log_income"] = np.log1p(df["income"])
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 30, 50, np.inf],
    labels=["child", "young_adult", "adult", "senior"],
)
```

Use `map` for a Series and `apply` for a row-wise custom operation only when no vectorized alternative is available:

```python
df["full_name"] = df[["first_name", "last_name"]].apply(
    lambda row: f"{row['first_name']} {row['last_name']}",
    axis=1,
)
```

Row-wise `apply(axis=1)` is slower and often less clear than vectorized expressions.

## Indexing and lookup

```python
df = df.set_index("customer_id")
record = df.loc["C001"]
df = df.reset_index()
```

Use `set_index` when a column is a meaningful lookup key. Do not confuse an index with a model feature.

## Method chaining

Method chains make transformations readable when each step has one purpose:

```python
clean = (
    raw
    .rename(columns={"hours-per-week": "hours_per_week"})
    .drop_duplicates()
    .assign(
        age=lambda data: pd.to_numeric(data["age"], errors="coerce"),
        is_adult=lambda data: data["age"].ge(18),
    )
    .loc[lambda data: data["age"].notna()]
    .reset_index(drop=True)
)
```

Use `.copy()` when creating a standalone filtered object and use intermediate variables when a chain becomes difficult to audit.

## Performance basics

- Prefer vectorized pandas and NumPy operations over Python loops.
- Build rows in a list and call `pd.concat` once.
- Select only needed columns before expensive operations.
- Convert repeated strings to `category` when appropriate.
- Use `usecols`, `dtype`, and `parse_dates` while reading large files.
- Use Parquet for repeated analytical workflows.
- Inspect memory with `df.memory_usage(deep=True)`.

```python
memory_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"{memory_mb:.1f} MB")
```

## Final ML checklist

Before fitting a model:

- separate `X` and `y`
- identify numerical, categorical, datetime, and identifier columns
- remove or justify duplicates
- check missing and impossible values
- verify target labels and class balance
- prevent target leakage
- split data with a strategy appropriate to time, groups, and classes
- fit imputation, scaling, encoding, feature selection, and feature engineering inside a pipeline
- preserve column names and transformations for inference
- validate that train and test schemas match

## Related pages

- [Pandas Basics](Pandas-Basics.md)
- [Inspecting Data](Inspecting-Data.md)
- [Selecting and Filtering](Selecting-and-Filtering.md)
- [GroupBy, Joins, and Reshaping](GroupBy-Joins-and-Reshaping.md)
- [Data Types, Column Names, and Categories](../03-EDA/Data-Types-and-Categories.md)
- [Data Cleaning Overview](../03-EDA/Data-Cleaning-Overview.md)
- [pandas API reference](https://pandas.pydata.org/docs/reference/index.html)

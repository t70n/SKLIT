# Data Types, Column Names, and Categories

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Why this matters

Not every numeric-looking column is a truly numerical feature. Sometimes integer values represent categories, such as:

- `0 = male`, `1 = female`
- `0 = low`, `1 = medium`, `2 = high`
- postal codes, product identifiers, or occupation codes

If this is not recognized, the model may treat the codes as measurements and infer a meaningless order or distance between them. Inspect the meaning of every column, not only its dtype.

## Inspect types before transforming

```python
print(df.dtypes)
print(df.dtypes.unique())
print(df.head())
```

Typical interpretation:

| dtype | Usual meaning |
| --- | --- |
| `object` | Text in pandas 1.x and 2.x (often categorical), or mixed Python objects |
| `str` | Text, the default string dtype in pandas 3.0 (`StringDtype`) |
| `category` | Explicit categorical dtype with a fixed set of levels |
| `int64`, `float64` | Usually numeric, but integer codes can also represent categories |
| `bool` | Binary indicators |
| `datetime64[ns]` | Dates and timestamps |

## Clean column names

```python
import re


def clean_column_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


df.columns = [clean_column_name(column) for column in df.columns]
if df.columns.duplicated().any():
    raise ValueError("Column names collide after cleaning")
```

Keep a mapping if external systems depend on the original names.

## Convert types deliberately

```python
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["category"] = df["category"].astype("category")
df = df.convert_dtypes()
```

`errors="coerce"` turns invalid values into missing values, so inspect missing counts afterward.

## Select columns by type

```python
numeric_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

numeric = df[numeric_columns]
categorical = df[categorical_columns]
```

Listing `"object"` and `"string"` together selects text columns in every pandas version. In pandas 3.0, `include="object"` alone still selects `str` columns but emits a deprecation warning.

For scikit-learn pipelines:

```python
from sklearn.compose import make_column_selector

numeric_columns = make_column_selector(dtype_include="number")
categorical_columns = make_column_selector(dtype_exclude="number")
```

A selector is evaluated when the `ColumnTransformer` is fitted, so it adapts to the columns of the training DataFrame. Integer-coded categories are selected as numeric by these rules: list them explicitly when they must be encoded as categories.

## Normalize strings carefully

```python
for column in categorical_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )
```

Before applying this globally, confirm that capitalization or whitespace does not carry meaning.

Useful operations include:

```python
df["city"].str.contains("oslo", case=False, na=False)
df["code"].str.replace("-", "", regex=False)
df["email"].str.extract(r"@(.+)$")[0]
```

## Inspect and normalize categories

```python
for column in categorical_columns:
    print(column)
    print(df[column].value_counts(dropna=False).head(20))
```

Map equivalent labels explicitly:

```python
df["sex"] = df["sex"].replace({"M": "male", "F": "female"})
```

Use `map` for a defined mapping:

```python
df["priority_code"] = df["priority"].map({
    "low": 1,
    "medium": 2,
    "high": 3,
})
```

Unmapped values become missing, so validate the result.

## Ordinal versus nominal categories

Nominal categories have no meaningful order, such as country or occupation. Do not encode them as `0, 1, 2` and pass them to a model that interprets numeric distances, such as a linear model.

Ordinal categories have a meaningful order, such as low, medium, high. An ordered encoding can be appropriate, but state the order explicitly instead of relying on the alphabetical default:

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[["low", "medium", "high"]])
```

For nominal categories, use one-hot encoding:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=0.01,
)
```

`handle_unknown="ignore"` prevents unseen test categories from breaking prediction, and `min_frequency` groups rare categories into a single infrequent category. Fit encoders inside a pipeline. See [OrdinalEncoder vs OneHotEncoder](../05-Preprocessing/OrdinalEncoder-vs-OneHotEncoder.md).

## Check schema consistency

```python
expected = ["age", "income", "country"]
if list(df.columns) != expected:
    raise ValueError("Unexpected column order or names")
```

Before inference, verify names, dtypes, units, required columns, and category conventions. Schema errors can look like model errors.

## Practical reminder

Always inspect the content of a column before deciding whether it is numerical or categorical.

## Related pages

- [Inspecting Data](../02-Pandas/Inspecting-Data.md)
- [Selecting and Filtering](../02-Pandas/Selecting-and-Filtering.md)
- [OrdinalEncoder vs OneHotEncoder](../05-Preprocessing/OrdinalEncoder-vs-OneHotEncoder.md)
- [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md)
- [pandas user guide: Working with text data](https://pandas.pydata.org/docs/user_guide/text.html)

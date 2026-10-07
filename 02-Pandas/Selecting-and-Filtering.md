# Selecting and Filtering Data in Pandas

[Home](../README.md) / [Pandas](README.md)

This page covers the two most common subsetting operations: selecting columns (features) and filtering rows (samples).

## Selecting columns

### One or several columns

```python
subset = df[["age", "hours-per-week"]]  # DataFrame with the chosen columns
series = df["class"]                     # Series (one column)
one_column = df[["class"]]               # DataFrame with a single column
```

The double brackets matter when the result must remain a DataFrame, for example when it is passed to a scikit-learn transformer that expects a 2D input.

### Drop a column

```python
data = df.drop(columns="class")
data = df.drop(columns=["class", "education-num"])
```

### Select columns by data type

```python
numerical = df.select_dtypes(include="number")
categorical = df.select_dtypes(include=["object", "string", "category"])
```

Listing both `"object"` and `"string"` works with every pandas version. In pandas 3.0, text columns use the dedicated `str` dtype by default; selecting them with `include="object"` alone still works but emits a deprecation warning.

### Useful pattern with scikit-learn

```python
from sklearn.compose import make_column_selector as selector

numerical_columns = selector(dtype_include="number")(df)
categorical_columns = selector(dtype_exclude="number")(df)
```

This is especially helpful for building preprocessing pipelines with a `ColumnTransformer`. A selector object can also be passed directly to the `ColumnTransformer`, in which case the columns are selected when the transformer is fitted. See [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md).

## Filtering rows

### Simple condition

```python
filtered = df[df["age"] > 40]
```

This keeps only the rows satisfying the condition.

### Multiple conditions

```python
filtered = df[(df["age"] > 40) & (df["sex"] == "Male")]
```

Use `&` (and), `|` (or), and `~` (not) with parentheses around each condition, never the Python keywords `and`, `or`, and `not`.

### Membership, ranges, and queries

```python
subset = df[df["native-country"].isin(["France", "Germany"])]
working_age = df[df["age"].between(18, 67)]
selected = df.query("age > 40 and `hours-per-week` >= 50")
```

Column names containing spaces or hyphens must be wrapped in backticks inside `query`.

### Filtering missing values

```python
df_complete = df.dropna()                 # drop rows with any missing value
valid = df[df["workclass"].notna()]       # keep rows where one column is present
```

Dropping rows can change the population and the target distribution; see [Missing Values](../03-EDA/Missing-Values.md).

## Selecting by label or by position

```python
first_rows = df.iloc[0:10]            # by integer position
cell = df.iloc[0, 2]                  # first row, third column
by_label = df.loc[:, ["age", "sex"]]  # by column labels
adults = df.loc[df["age"] >= 18, ["age", "sex"]]  # condition and columns together
```

- `iloc` uses integer positions; the end of a slice is excluded.
- `loc` uses index labels and boolean masks; the end of a label slice is included.

Use `.loc` for conditional assignment, for example `df.loc[df["age"] < 0, "age"] = np.nan`, to avoid chained assignment.

## Why this matters

A model usually needs a clean separation between:

- numerical features
- categorical features
- the target column

```python
X = df.drop(columns="class")
y = df["class"]
```

Filtering is often used to:

- analyze a subgroup
- inspect a subset of the data
- remove irrelevant or invalid rows
- prepare data before a train/test split

## Related pages

- [Pandas Basics](Pandas-Basics.md)
- [DataFrame Manipulation Cheat Sheet](DataFrame-Manipulation-Cheat-Sheet.md)
- [Data Types, Column Names, and Categories](../03-EDA/Data-Types-and-Categories.md)
- [pandas user guide: Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html)

# Pandas Basics

[Home](../README.md) / [Pandas](README.md)

## Why pandas matters

Pandas is the main Python library for tabular data manipulation.

It allows you to:

- load datasets
- inspect samples and columns
- filter rows and features
- handle missing values
- perform grouping and aggregation
- prepare data for machine learning

## Import

```python
import pandas as pd
```

## Loading a dataset

```python
adult_census = pd.read_csv("../datasets/adult-census.csv")
```

The examples below use `df` as a generic name for a loaded DataFrame:

```python
df = adult_census
```

## Inspecting data

### First rows

```python
df.head()
```

### Basic shape

```python
print(df.shape)  # (n_rows, n_columns)
```

### Column names

```python
print(df.columns)
```

### Data types

```python
df.dtypes
```

### Summary statistics

```python
df.describe()
```

## Accessing a column

```python
df["class"]  # returns a Series
```

## Counting values

```python
df["sex"].value_counts()
df["class"].value_counts(normalize=True)  # proportions instead of counts
```

## Sorting values

```python
df["native-country"].value_counts().sort_index()  # alphabetical order
df.sort_values("age", ascending=False)             # rows sorted by a column
```

## Checking different data types

```python
df.dtypes.unique()
```

Useful idea:

- text columns usually hold categorical data; they have the `object` dtype in pandas 1.x and 2.x and the dedicated `str` dtype by default in pandas 3.0
- numeric columns may also be encoded as integers even when they are categorical in meaning

See [Data Types, Column Names, and Categories](../03-EDA/Data-Types-and-Categories.md).

## Selecting columns

```python
selected = df[["age", "hours-per-week"]]
```

## Dropping columns

```python
data = df.drop(columns="class")
```

`drop` returns a new DataFrame; the original `df` is unchanged unless you reassign it.

## Filtering rows

```python
filtered = df[df["age"] > 40]
```

## Combining filters

```python
filtered = df[(df["age"] > 40) & (df["sex"] == "Male")]
```

Use `&`, `|`, and `~` with parentheses around each condition. See [Selecting and Filtering](Selecting-and-Filtering.md).

## Null / missing values

```python
df.isna().sum()
```

## Keeping an eye on missing values

It is common to need to:

- remove rows with missing values
- fill missing values with a statistic
- encode missing values specially

For machine learning, learn imputation statistics on the training data only, preferably with `SimpleImputer` inside a pipeline. See [Missing Values](../03-EDA/Missing-Values.md).

## Main pandas operations you will use often

- `head()`
- `describe()`
- `value_counts()`
- `groupby()`
- `sort_values()`
- `drop()`
- `columns`
- `dtypes`
- `isna()`
- `select_dtypes()`

## Typical workflow in a notebook

```python
import pandas as pd

df = pd.read_csv("dataset.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
```

## Good practice

- Inspect the data before training a model
- Check for class imbalance
- Check whether integer columns are truly numerical or merely encoded categories
- Look for missing values and unusual values

## Related pages

- [Inspecting Data](Inspecting-Data.md)
- [Selecting and Filtering](Selecting-and-Filtering.md)
- [DataFrame Manipulation Cheat Sheet](DataFrame-Manipulation-Cheat-Sheet.md)
- [GroupBy, Joins, and Reshaping](GroupBy-Joins-and-Reshaping.md)
- [Data Cleaning Overview](../03-EDA/Data-Cleaning-Overview.md)
- [pandas user guide: 10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

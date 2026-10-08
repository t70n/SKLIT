# Inspecting Data with Pandas

[Home](../README.md) / [Pandas](README.md)

## Purpose

Before training a model, you should understand the data:

- what columns exist
- what type each column has
- how many rows there are
- how many missing values there are
- how the target is distributed

## Basic inspection commands

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
```

`df.info()` combines the column names, the number of non-missing values, the dtypes, and the memory usage in a single report.

## Summary statistics

```python
df.describe()
```

This gives statistics for numerical columns, including:

- count
- mean
- std
- min
- 25%
- 50% (median)
- 75%
- max

Use `df.describe(include="all")` to also summarize categorical columns (number of unique values, most frequent value, and its frequency).

## Distribution of categories

```python
df["species"].value_counts()
```

If you want sorted categories:

```python
df["species"].value_counts().sort_index()
```

Use `normalize=True` to obtain proportions and `dropna=False` to count missing values as a category.

## Counting missing values

```python
df.isna().sum()
```

This is extremely useful before preprocessing.

## Detecting heterogeneous data types

```python
df.dtypes
```

and

```python
df.dtypes.unique()
```

### Interpretation

- `object` (pandas 1.x and 2.x) or `str` (default string dtype in pandas 3.0) usually means text, often categorical
- `category` is an explicit categorical dtype
- `int64` and `float64` are usually numerical
- sometimes integer-coded categories are numeric but should still be treated as categorical

## Example of practical inspection

```python
for col in df.columns:
    print(f"--- {col} ---")
    print(df[col].describe())
```

## Why this matters

Data inspection is not optional. Many modeling issues start because the data is misunderstood.

Examples:

- a categorical column accidentally treated as numeric
- a rare category not seen during training
- imbalanced classes hiding poor model performance

## Related pages

- [Pandas Basics](Pandas-Basics.md)
- [EDA Overview](../03-EDA/EDA-Overview.md)
- [Data Types, Column Names, and Categories](../03-EDA/Data-Types-and-Categories.md)
- [Missing Values](../03-EDA/Missing-Values.md)

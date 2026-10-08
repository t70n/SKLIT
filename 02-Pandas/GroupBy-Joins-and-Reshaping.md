# Pandas GroupBy, Joins, and Reshaping

[Home](../README.md) / [Pandas](README.md)

This page covers the operations used to combine tables, aggregate observations, and change between wide and long formats.

## GroupBy: split, apply, combine

`groupby` divides rows into groups, computes a result per group, and combines the results.

```python
grouped = df.groupby("country")

mean_income = grouped["income"].mean()
count_rows = grouped.size()
```

### Common aggregations

```python
summary = (
    df.groupby("country", dropna=False)
    .agg(
        rows=("country", "size"),
        mean_income=("income", "mean"),
        median_age=("age", "median"),
        max_hours=("hours_per_week", "max"),
    )
    .reset_index()
)
```

Useful aggregation methods include `.count()`, `.size()`, `.sum()`, `.mean()`, `.median()`, `.min()`, `.max()`, `.std()`, `.nunique()`, and `.quantile()`.

`size()` counts rows, including missing values. `count()` counts non-missing values in a selected column.

### Multiple grouping columns

```python
summary = (
    df.groupby(["country", "education"], as_index=False)
    .agg(
        records=("income", "size"),
        average_income=("income", "mean"),
    )
)
```

### Named aggregation

Named aggregation makes output names explicit and avoids confusing multi-level column names:

```python
result = df.groupby("class", as_index=False).agg(
    average_age=("age", "mean"),
    median_income=("income", "median"),
    number_of_countries=("native_country", "nunique"),
)
```

### Multiple functions

```python
result = df.groupby("class")["income"].agg(["count", "mean", "median", "std"])
```

### Group percentages

```python
counts = df.groupby(["country", "class"]).size()
within_country = counts.groupby(level=0).transform(lambda values: values / values.sum())
```

For a simple categorical table:

```python
proportions = pd.crosstab(
    df["country"],
    df["class"],
    normalize="index",
)
```

### Filter groups

```python
large_groups = df.groupby("customer_id").filter(
    lambda group: len(group) >= 5
)
```

Use `filter` carefully on large datasets; a group-level summary or merge can be more efficient.

### Transform within groups

`transform` returns a Series aligned to the original index, which makes it useful for group features:

```python
df["country_mean_income"] = df.groupby("country")["income"].transform("mean")
df["income_minus_country_mean"] = (
    df["income"] - df["country_mean_income"]
)
```

This kind of target-related group feature can leak information. For prediction, compute it using training data and a leakage-safe strategy.

### Group ranking and cumulative features

```python
df["income_rank_in_country"] = (
    df.groupby("country")["income"]
    .rank(method="dense", ascending=False)
)

df["running_total"] = (
    df.sort_values("date")
    .groupby("customer_id")["amount"]
    .cumsum()
)
```

Sort before time-dependent operations and preserve the original order if necessary.

## `merge`: join tables by keys

Suppose `customers` contains one row per customer and `orders` contains many rows per customer:

```python
customer_orders = orders.merge(
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one",
)
```

### Join types

- `how="inner"`: keep keys present in both tables
- `how="left"`: keep every left row; unmatched right values become missing
- `how="right"`: keep every right row
- `how="outer"`: keep the union of keys from both tables
- `how="cross"`: Cartesian product; use only intentionally

```python
merged = left.merge(right, on="id", how="inner")
```

### Validate join cardinality

Use `validate` to catch accidental row multiplication:

```python
merged = facts.merge(
    lookup,
    on="product_id",
    how="left",
    validate="many_to_one",
)
```

Other useful values are `"one_to_one"`, `"one_to_many"`, and `"many_to_many"`. Avoid `many_to_many` unless multiplication is explicitly intended.

### Different key names

```python
merged = left.merge(
    right,
    left_on="customer_id",
    right_on="id",
    how="left",
)
```

### Overlapping non-key columns

```python
merged = left.merge(
    right,
    on="id",
    how="left",
    suffixes=("_left", "_right"),
)
```

Inspect suffix columns and decide which source is authoritative.

### Diagnose unmatched keys

```python
merged = left.merge(
    right,
    on="id",
    how="left",
    indicator=True,
)
print(merged["_merge"].value_counts())
```

This reveals rows found only in the left table, only in the right table, or in both.

## `concat`: stack tables

### Stack rows

```python
combined = pd.concat([january, february, march], axis=0, ignore_index=True)
```

Columns are aligned by name. Missing columns in one input become missing values.

### Stack columns

```python
wide = pd.concat([features, labels], axis=1)
```

This aligns by index. Misaligned indexes can create missing values, so verify index equality first:

```python
if not features.index.equals(labels.index):
    raise ValueError("Indexes do not align")
```

### Avoid repeated concatenation

```python
frames = []
for path in paths:
    frames.append(pd.read_csv(path))
combined = pd.concat(frames, ignore_index=True)
```

## `join`: index-based joining

```python
features = features.join(
    lookup.set_index("customer_id"),
    on="customer_id",
    how="left",
    validate="many_to_one",
)
```

`merge` is generally clearer when joining on columns; `join` is convenient when indexes are already meaningful.

## Wide and long data

### Wide format

One row contains several measurement columns:

| person | height | weight |
| --- | ---: | ---: |
| A | 170 | 70 |

### Long format

One row contains one measurement identified by a variable column:

| person | measurement | value |
| --- | --- | ---: |
| A | height | 170 |
| A | weight | 70 |

Long format is often convenient for plotting and grouped analysis.

### `melt`: wide to long

```python
long = df.melt(
    id_vars=["person", "date"],
    value_vars=["height", "weight"],
    var_name="measurement",
    value_name="value",
)
```

### `pivot`: long to wide

`pivot` requires each index/column combination to be unique:

```python
wide = long.pivot(
    index="person",
    columns="measurement",
    values="value",
).reset_index()
```

### `pivot_table`: aggregate while reshaping

Use `pivot_table` when combinations can repeat:

```python
summary = df.pivot_table(
    index="country",
    columns="class",
    values="income",
    aggfunc="median",
    fill_value=0,
)
```

### `stack` and `unstack`

These operate on index levels:

```python
stacked = wide.set_index("person").stack()
unstacked = stacked.unstack()
```

Use `reset_index()` when you want ordinary columns again.

## Crosstabs

`crosstab` counts combinations of categorical variables:

```python
counts = pd.crosstab(df["education"], df["class"])
row_percentages = pd.crosstab(
    df["education"],
    df["class"],
    normalize="index",
)
```

It is useful for EDA, class rates, and contingency analysis.

## Datetime manipulation

```python
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6])
```

For time-indexed data:

```python
daily = (
    df.set_index("timestamp")
    .sort_index()
    .resample("D")
    .agg(total_amount=("amount", "sum"), records=("amount", "size"))
)
```

Do not use future observations to construct features for past predictions. Time-based feature engineering needs chronological validation.

## Rolling and expanding features

```python
df = df.sort_values(["customer_id", "date"])
df["previous_amount"] = df.groupby("customer_id")["amount"].shift(1)
df["rolling_7"] = (
    df.groupby("customer_id")["amount"]
    .rolling(7, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
```

For prediction, use a shifted value when the current observation must not be included:

```python
df["past_7_mean"] = (
    df.groupby("customer_id")["amount"]
    .transform(lambda series: series.shift(1).rolling(7, min_periods=1).mean())
)
```

## Apply operations by row or column

```python
import numpy as np

# Column-wise summary
df.select_dtypes("number").apply(np.mean)

# Row-wise operation, use sparingly
df["total"] = df[["a", "b", "c"]].sum(axis=1)
```

Prefer built-in reductions such as `.sum(axis=1)` to row-wise `.apply(axis=1)` for speed and clarity.

## Common mistakes

- joining without checking key uniqueness, causing silent row multiplication
- merging tables with mismatched key types such as integer versus string
- using `concat(axis=1)` with misaligned indexes
- pivoting data with duplicate index/column combinations
- calculating group target means before splitting the data
- using future rows in rolling or cumulative features
- confusing `count()` with `size()` in grouped data
- forgetting `reset_index()` after an aggregation
- relying on chained assignment
- mutating the raw DataFrame before preserving a copy

## ML preparation checklist

Before passing a manipulated DataFrame to a model:

- check the row count after every merge or filter
- validate key cardinality with `validate=`
- verify index alignment after horizontal concatenation
- separate target and identifiers
- check that feature values are available at prediction time
- use a pipeline for learned preprocessing
- split by time or group when rows are not independent
- compare target distributions before and after manipulation
- preserve the final feature-column order and schema

## Related pages

- [DataFrame Manipulation Cheat Sheet](DataFrame-Manipulation-Cheat-Sheet.md)
- [Cross-Row and Local Features](../06-Feature-Engineering/Cross-Row-and-Local-Features.md)
- [Duplicates and Redundancy](../03-EDA/Duplicates-and-Redundancy.md)
- [pandas user guide: Group by](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [pandas user guide: Merge, join, concatenate](https://pandas.pydata.org/docs/user_guide/merging.html)
- [pandas user guide: Reshaping and pivot tables](https://pandas.pydata.org/docs/user_guide/reshaping.html)

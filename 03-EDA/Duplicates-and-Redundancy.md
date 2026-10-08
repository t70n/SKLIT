# Duplicates and Redundancy

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Why check duplicates?

Duplicate records can bias summaries and models by counting the same observation multiple times. But repeated rows are not automatically errors: repeated measurements, repeated purchases, or observations from different groups can be valid.

First define what one row represents and what should be unique.

## Find exact duplicate rows

```python
duplicate_mask = df.duplicated(keep=False)
duplicates = df.loc[duplicate_mask].sort_values(list(df.columns))
print(duplicates)
print("duplicate rows:", df.duplicated().sum())
```

`keep=False` marks every member of a duplicate group. The default `keep="first"` marks only later copies.

## Remove exact duplicates

```python
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("removed:", before - len(df))
```

Keep the original DataFrame or save an audit record before deleting rows. Only remove duplicates when they represent repeated records rather than valid repeated observations.

## Duplicates based on a key

If the dataset should contain one row per entity, inspect duplicates using the business key:

```python
duplicate_customers = df[
    df.duplicated(subset=["customer_id"], keep=False)
]
```

To keep the latest documented record:

```python
df = (
    df.sort_values("updated_at")
    .drop_duplicates(subset=["customer_id"], keep="last")
)
```

Never choose `keep="first"` or `keep="last"` without defining which record is authoritative.

## Duplicate columns

```python
same_columns = df.T.duplicated()
redundant_columns = df.columns[same_columns]
print(redundant_columns.tolist())
```

Exact duplicate columns are easy to find. Derived columns, duplicated identifiers, and target-derived features require domain inspection. High correlation can indicate redundancy, but low correlation does not prove independence.

## Check uniqueness

```python
print(df["customer_id"].is_unique)
print(df["customer_id"].duplicated().sum())
```

For compound keys:

```python
key = ["customer_id", "date"]
duplicate_keys = df[df.duplicated(subset=key, keep=False)]
```

## Merge-related duplication

A many-to-many merge can silently multiply rows. Validate expected cardinality:

```python
merged = facts.merge(
    lookup,
    on="product_id",
    how="left",
    validate="many_to_one",
)
```

Use `indicator=True` to inspect unmatched keys:

```python
merged = left.merge(right, on="id", how="left", indicator=True)
print(merged["_merge"].value_counts())
```

## Before deleting duplicates

Check:

- what one row represents
- whether timestamps or measurements distinguish repeated rows
- whether duplicates occur only in one data source
- whether the target distribution changes after removal
- whether duplicate entities appear across train and test
- whether the duplicate is actually caused by a faulty join

Keep a count and reason for every deletion.

## Related pages

- [Constant and Redundant Features](Constant-and-Redundant-Features.md)
- [GroupBy, Joins, and Reshaping](../02-Pandas/GroupBy-Joins-and-Reshaping.md)
- [Identifiers, Leakage, Shuffling, and Sampling](Identifiers-Leakage-and-Sampling.md)

# Data Cleaning and Manipulation for EDA

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Purpose

Data cleaning makes a dataset structurally consistent, valid, and suitable for analysis. Cleaning does not mean deleting everything unusual. An unusual value may be a valid rare observation, a measurement error, or evidence of a different subgroup.

Keep raw data unchanged, create a cleaned copy, record decisions, and preserve enough information to reproduce or reverse each transformation.

## Recommended order

1. Define what one row represents.
2. Preserve the raw data and record the source.
3. Inspect shape, columns, data types, duplicates, and missingness.
4. Standardize unambiguous column names and formatting.
5. Identify targets, identifiers, categories, and numeric features.
6. Check duplicates, constants, invalid values, and constraints.
7. Decide whether unusual values should be corrected, flagged, transformed, or removed.
8. Split data before learning target-dependent transformations.
9. Validate row counts, ranges, uniqueness, and target distribution after changes.
10. Put repeatable model-time transformations in a pipeline.

## Keep an audit trail

```python
cleaning_log = []


def log_change(operation, before, after, reason):
    cleaning_log.append({
        "operation": operation,
        "rows_before": before,
        "rows_after": after,
        "reason": reason,
    })
```

For every deletion or replacement, record the reason, affected columns, number of rows, and whether the operation was learned from training data.

## A first-pass audit

```python
import pandas as pd

raw = pd.read_csv("data.csv")
df = raw.copy()

print("shape:", df.shape)
print("duplicates:", df.duplicated().sum())
print("missing:\n", df.isna().sum().sort_values(ascending=False))
print("dtypes:\n", df.dtypes)
print(df.describe(include="all").T)
```

## Action-specific guides

- [Duplicates and Redundancy](Duplicates-and-Redundancy.md)
- [Missing Values](Missing-Values.md)
- [Data Types, Column Names, and Categories](Data-Types-and-Categories.md)
- [Constant and Redundant Features](Constant-and-Redundant-Features.md)
- [Invalid Values and Constraints](Invalid-Values-and-Constraints.md)
- [Identifiers, Leakage, Shuffling, and Sampling](Identifiers-Leakage-and-Sampling.md)
- [Outlier Detection](Outlier-Detection.md)

## Final ML checklist

Before fitting a model:

- separate `X` and `y`
- identify numerical, categorical, datetime, and identifier columns
- check duplicates, missing values, constants, and impossible values
- verify target labels and class balance
- prevent target leakage
- choose a split appropriate to time, groups, and classes
- fit imputation, encoding, scaling, feature selection, and feature engineering inside a pipeline
- preserve the feature schema for inference
- compare the data before and after cleaning

Cleaning is successful when the data is more trustworthy and the decisions are reproducible, not simply when the table becomes smaller.

## Related pages

- [EDA Overview](EDA-Overview.md)
- [DataFrame Manipulation Cheat Sheet](../02-Pandas/DataFrame-Manipulation-Cheat-Sheet.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)

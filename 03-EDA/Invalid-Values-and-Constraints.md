# Invalid Values and Data Constraints

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Statistical outliers are not automatically invalid

A rare value may be valid. An invalid value violates a physical, business, or data-collection rule. Separate these questions:

- Is the value unusual?
- Is the value impossible?
- Is it recorded in the correct unit?
- Is it available at prediction time?

For statistical outlier rules, see [Outlier Detection](Outlier-Detection.md).

## Range checks

```python
invalid_age = ~df["age"].between(0, 120)
invalid_hours = ~df["hours_per_week"].between(0, 168)

print("invalid ages:", invalid_age.sum())
print("invalid hours:", invalid_hours.sum())
```

Use domain limits, not generic standard-deviation rules alone.

## Cross-column constraints

```python
invalid_total = df["total"] < df["subtotal"]
print("invalid totals:", invalid_total.sum())

invalid_dates = df["start_date"] > df["end_date"]
print("invalid date ranges:", invalid_dates.sum())
```

Cross-column rules often find errors that univariate checks miss.

## Encode checks as assertions

```python
assert df["customer_id"].notna().all()
assert df["age"].dropna().between(0, 120).all()
assert set(df["sex"].dropna().unique()) <= {"female", "male"}
assert not df.columns.duplicated().any()
```

A failed assertion should trigger investigation, not automatic deletion.

## Possible actions

### Correct

Use a trusted source or deterministic rule. Record the original value and correction.

### Convert to missing

Use this when a value is invalid but the row remains useful:

```python
df.loc[~df["age"].between(0, 120), "age"] = pd.NA
```

Impute later with a training-only pipeline.

### Remove

Remove a row only when it is demonstrably invalid, outside the population definition, duplicated, or unusable for the analysis.

### Retain

Keep valid extreme values. Robust metrics, transformations, clipping, or robust models may be better than deletion.

## Audit changes

```python
before = len(df)
mask = df["age"].between(0, 120) | df["age"].isna()
df = df.loc[mask].copy()
print("removed:", before - len(df))
```

Always compare row count, target distribution, ranges, and missingness before and after an operation.

## Data validation libraries

For larger projects, express rules with a validation framework such as `pandera`, or keep explicit validation functions in tests. The important properties are that checks are reproducible, visible, and run before modeling.

## Related pages

- [Outlier Detection](Outlier-Detection.md)
- [Missing Values](Missing-Values.md)
- [Data Cleaning Overview](Data-Cleaning-Overview.md)

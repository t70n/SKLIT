# Exploratory Data Analysis (EDA)

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## What EDA is

Exploratory Data Analysis is the systematic process of understanding a dataset before modeling. EDA combines tabular inspection, summary statistics, visualizations, and domain questions to discover structure and problems.

The goal is not to prove a final hypothesis. The goal is to learn enough about the data to choose sensible preprocessing, features, models, and evaluation methods.

## A practical EDA workflow

1. Load the data and define the unit of observation.
2. Inspect shape, columns, data types, duplicates, and missing values.
3. Separate the target from the predictors.
4. Identify numerical, categorical, ordinal, datetime, and identifier columns.
5. Inspect distributions and unusual values.
6. Inspect relationships between predictors and the target.
7. Measure redundancy and association between variables.
8. Check class balance or target range.
9. Record hypotheses and potential data-quality issues.
10. Split the data before learning transformations or selecting features.

EDA can use the full dataset for descriptive understanding, but decisions that learn from the target or estimate performance must be validated without leaking test information.

## Automated first-pass report

This helper creates a compact numerical and categorical summary. It is a starting point, not a replacement for understanding the domain.

```python
import pandas as pd


def eda_report(df, target=None):
    report = {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().sort_values(ascending=False),
        "dtypes": df.dtypes,
        "numeric_summary": df.select_dtypes(include="number").describe().T,
        "categorical_summary": {},
    }

    for column in df.select_dtypes(exclude="number"):
        report["categorical_summary"][column] = {
            "unique": int(df[column].nunique(dropna=True)),
            "top_values": df[column].value_counts(dropna=False).head(10),
        }

    if target is not None:
        report["target_summary"] = df[target].value_counts(dropna=False)

    return report


report = eda_report(df, target="target")
print(report["shape"])
print(report["missing_values"].head(10))
print(report["numeric_summary"])
```

For a reusable project, consider a profiling tool such as `ydata-profiling`, but inspect its output critically and avoid sharing sensitive data unintentionally.

## Questions to ask for every variable

### Numerical variables

- Is the variable continuous, discrete, or actually a category encoded as a number?
- Is it strongly skewed?
- Are there impossible values or outliers?
- Is the scale appropriate for the intended model?
- Does a transformation such as log, binning, or a spline make domain sense?

### Categorical variables

- How many levels are there?
- Are some levels rare or unseen at prediction time?
- Is there a natural order?
- Should rare levels be grouped?
- Does the category identify a person, row, or data source rather than describe the phenomenon?

### Target variable

- Is the task classification or regression?
- Is the target imbalanced or heavy-tailed?
- Is there a time, group, or entity structure requiring a special split?
- Could a feature contain information from the future or from the target itself?

## Numerical and categorical visual checks

```python
import matplotlib.pyplot as plt
import seaborn as sns

numeric_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(exclude="number").columns

for column in numeric_columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(df[column], kde=True, ax=axes[0])
    sns.boxplot(x=df[column], ax=axes[1])
    fig.suptitle(column)
    plt.tight_layout()
    plt.show()

for column in categorical_columns:
    counts = df[column].value_counts(dropna=False).head(20)
    counts.sort_values().plot.barh(figsize=(7, 4), title=column)
    plt.tight_layout()
    plt.show()
```

Histograms show shape, boxplots expose extreme values and skew, and count plots reveal imbalance and rare categories.

## EDA and leakage

Do not use EDA to justify decisions based on the held-out test set after repeatedly checking test performance. In particular:

- fit imputers, scalers, encoders, binning, splines, and feature selection inside a pipeline
- use training data or training folds to learn target encodings and thresholds
- use a validation set or cross-validation for model and threshold selection
- keep the final test set for one final estimate

EDA may reveal that a random split is inappropriate. Grouped observations, repeated measurements, and time series often require group-aware or time-aware validation.

## Related pages

- [Data Cleaning Overview](Data-Cleaning-Overview.md)
- [Missing Values](Missing-Values.md)
- [MICE and Iterative Imputation](MICE-and-Iterative-Imputation.md)
- [Correlation and Distributions](Correlation-and-Distributions.md)
- [Outlier Detection](Outlier-Detection.md)
- [Inspecting Data with Pandas](../02-Pandas/Inspecting-Data.md)
- [Visualization Overview](../04-Visualization/Visualization-Overview.md)
- [Pairplots](../04-Visualization/Pairplots.md)
- [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md)

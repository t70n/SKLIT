# Outlier Detection

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## What is an outlier?

An outlier is an observation that is unusually far from the main pattern of a variable or dataset. It can represent:

- a data-entry or measurement error
- an impossible value
- a rare but valid observation
- a different population or subgroup
- a distribution with a long tail
- a real event that the model must learn

Outlier detection is therefore an investigation step, not an automatic instruction to delete rows.

## Quartile notation: `Q1`, `Q3`, `D1`, and `D3`

The standard boxplot notation is:

- $Q_1$: first quartile, the 25th percentile
- $Q_3$: third quartile, the 75th percentile
- $IQR$: interquartile range

$$
IQR=Q_3-Q_1
$$

Some course notes use `D1` and `D3` to mean the lower and upper quartiles. In that notation:

$$
IQR=D_3-D_1
$$

Check the course definition before applying the formula. In many statistics references, however, $D_1$ and $D_3$ mean the first and third **deciles**, the 10th and 30th percentiles, not quartiles. The usual boxplot rule specifically uses the 25th and 75th percentiles.

## The IQR rule

The standard Tukey rule defines the fences as:

$$
\text{Lower fence}=Q_1-1.5\times IQR
$$

$$
\text{Upper fence}=Q_3+1.5\times IQR
$$

A value $x$ is flagged when:

$$
x < Q_1-1.5\times IQR
\quad\text{or}\quad
x > Q_3+1.5\times IQR
$$

Values outside the fences are potential outliers. They are not automatically wrong.

A more conservative rule sometimes uses $3\times IQR$ to flag only extreme outliers:

$$
Q_1-3\times IQR
\quad\text{and}\quad
Q_3+3\times IQR
$$

The multiplier is a choice that should be documented and validated against domain knowledge.

## Calculate the IQR rule with pandas

```python
column = "income"
values = df[column].dropna()

q1 = values.quantile(0.25)
q3 = values.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outlier_mask = (df[column] < lower_fence) | (df[column] > upper_fence)
outliers = df.loc[outlier_mask]

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("lower fence:", lower_fence)
print("upper fence:", upper_fence)
print("number flagged:", outlier_mask.sum())
```

Use `.loc` to retrieve complete rows, not only the outlying values. The surrounding columns may explain why an observation is unusual.

## Reusable IQR function

```python
import pandas as pd


def iqr_outlier_mask(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = series.lt(lower) | series.gt(upper)
    return mask, {"q1": q1, "q3": q3, "iqr": iqr,
                  "lower": lower, "upper": upper}


mask, limits = iqr_outlier_mask(df["income"])
print(limits)
print(df.loc[mask])
```

A constant column has $IQR=0$. Handle it separately rather than assuming that every value is an outlier.

## Detect outliers in every numeric column

```python
numeric_columns = df.select_dtypes(include="number").columns

outlier_summary = []
for column in numeric_columns:
    mask, limits = iqr_outlier_mask(df[column].dropna())
    outlier_summary.append({
        "column": column,
        "q1": limits["q1"],
        "q3": limits["q3"],
        "iqr": limits["iqr"],
        "lower_fence": limits["lower"],
        "upper_fence": limits["upper"],
        "count": int(mask.sum()),
        "fraction": float(mask.mean()),
    })

outlier_summary = pd.DataFrame(outlier_summary)
print(outlier_summary.sort_values("fraction", ascending=False))
```

This analyzes each feature independently. It does not detect observations that are unusual only because of a combination of otherwise ordinary feature values.

## Add an outlier flag instead of deleting rows

```python
mask, limits = iqr_outlier_mask(df["income"])
df["income_is_iqr_outlier"] = mask
```

An indicator can preserve useful information and let a model learn whether being extreme matters. It can also be useful for auditing and investigation.

## Boxplots and distributions

```python
import matplotlib.pyplot as plt
import seaborn as sns

column = "income"

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(x=df[column], ax=axes[0])
sns.histplot(df[column], kde=True, ax=axes[1])
plt.tight_layout()
plt.show()
```

A boxplot visualizes $Q_1$, the median, $Q_3$, and the IQR fences. A histogram shows whether the apparent outliers are isolated errors or simply part of a long-tailed distribution.

## Group-specific outliers

A global fence can be misleading when groups have different scales. For example, an income that is typical in one country may be extreme in another.

```python
def group_iqr_flag(values):
    mask, _ = iqr_outlier_mask(values)
    return mask


df["income_is_group_outlier"] = (
    df.groupby("country")["income"].transform(group_iqr_flag)
)
```

`transform` returns one value per original row, aligned on the original index, so the flag can be stored directly as a new column.

Use group-specific detection only when the grouping variable is known before prediction and there are enough observations per group. Sparse groups produce unstable quartiles.

## Log transforms for right-skewed variables

Income, prices, counts, and durations often have long right tails. A log transform can make the scale more manageable:

```python
import numpy as np

df["log_income"] = np.log1p(df["income"])
```

`log1p(x)` computes $\log(1+x)$ and works when values include zero. It does not accept negative values without a justified transformation. Re-check the distribution and the domain meaning after transforming; do not use a log transform merely to hide valid large observations.

## Z-score detection

For approximately symmetric data, a standardized score measures distance from the mean:

$$
z_i=\frac{x_i-\bar{x}}{s}
$$

A common screening rule flags $|z_i|>3$:

```python
from scipy.stats import zscore

values = df["height"].astype(float)
z = pd.Series(zscore(values, nan_policy="omit"), index=values.index)
outliers = df.loc[z.abs() > 3]
```

The mean and standard deviation are themselves sensitive to outliers, and the $3$ threshold is not universally valid. Z-scores are less appropriate for strongly skewed or heavy-tailed variables.

## Robust z-scores with MAD

The median absolute deviation (MAD) is more robust:

$$
MAD=\operatorname{median}(|x_i-\operatorname{median}(X)|)
$$

A robust standardized score can be written as:

$$
z_i^{robust}=\frac{0.6745(x_i-\operatorname{median}(X))}{MAD}
$$

```python
values = df["income"].dropna()
median = values.median()
mad = (values - median).abs().median()

if mad == 0:
    robust_z = pd.Series(0.0, index=values.index)  # no spread: nothing is flagged
else:
    robust_z = 0.6745 * (values - median) / mad

flagged_index = robust_z.index[robust_z.abs() > 3.5]
outliers = df.loc[flagged_index]
```

Selecting by index labels avoids aligning a boolean mask computed on `dropna()` values with the full DataFrame, which pandas rejects when the indexes differ.

The threshold `3.5` is a common screening choice, not a law.

## Quantile and domain rules

Quantiles are useful when you want a fixed proportion of observations to receive attention:

```python
low = df["income"].quantile(0.01)
high = df["income"].quantile(0.99)
flagged = df.loc[~df["income"].between(low, high)]
```

Domain rules are often better than generic statistical rules:

```python
invalid = ~df["age"].between(0, 120)
invalid_hours = ~df["hours_per_week"].between(0, 168)
```

A domain violation is not the same as a statistical outlier. A valid but rare observation should not be deleted just because it is unusual.

## Multivariate outliers

Univariate IQR rules examine one column at a time. A row can be a multivariate outlier even when every individual feature is within its own fences.

Possible tools include:

- scatter plots and pairplots for a small number of features
- Mahalanobis distance when a covariance model is reasonable
- robust covariance estimators
- Local Outlier Factor
- Isolation Forest
- domain-specific constraints

Example of an unsupervised screening model:

```python
from sklearn.ensemble import IsolationForest

features = df[["age", "income", "hours_per_week"]].dropna()
model = IsolationForest(contamination="auto", random_state=42)
labels = model.fit_predict(features)

potential_outliers = features.loc[labels == -1]
```

These methods depend on scaling, feature selection, contamination assumptions, and random variation. Treat their output as a review list, not ground truth.

## What should you do with flagged observations?

### Investigate

Compare the row with the source system, neighboring records, units, timestamps, and other features. Check whether the value is physically or logically possible.

### Correct

Correct a value only when a trusted source or deterministic rule identifies the error. Record the original value and correction rule.

### Convert to missing

Use this when the value is invalid but the row is otherwise useful. Impute later using a training-only pipeline.

### Transform

Use a log or other domain-justified transformation for valid skewed data. Transformation changes the modeling scale; it does not delete observations.

### Clip or winsorize

Replace values beyond chosen quantiles or fences with boundary values:

```python
lower = df["income"].quantile(0.01)
upper = df["income"].quantile(0.99)
df["income_clipped"] = df["income"].clip(lower, upper)
```

Clipping can reduce the influence of extremes, but it changes the data and can hide real tail behavior. Compare model performance and report the rule.

### Remove

Remove a row only when it is demonstrably invalid, duplicated, outside the population definition, or inappropriate for the specific analysis. Recalculate target distribution and sample counts after removal.

## Avoid leakage during outlier handling

If thresholds are computed from the target or from feature values for model preparation, learn them on the training data only. For cross-validation, the operation belongs inside a transformer or pipeline.

A simple custom transformer can be used when a project requires IQR clipping, but the thresholds must be fitted in `fit` and applied in `transform`. Do not compute global test-and-train quantiles before validation.

## Outlier investigation checklist

- identify whether the value is impossible, erroneous, rare, or merely extreme
- compare IQR flags with histograms and boxplots
- inspect the complete row and relevant groups
- compare global and group-specific thresholds when appropriate
- check skew and consider a justified transformation
- inspect multivariate relationships
- preserve a flag or audit trail when possible
- avoid deleting valid tail observations automatically
- learn model-time thresholds inside training folds
- report how many observations were flagged, changed, or removed

## Related pages

- [Invalid Values and Constraints](Invalid-Values-and-Constraints.md)
- [Histograms and Distributions](../04-Visualization/Histograms-and-Distributions.md)
- [Correlation and Distributions](Correlation-and-Distributions.md)
- [scikit-learn user guide: Novelty and outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)

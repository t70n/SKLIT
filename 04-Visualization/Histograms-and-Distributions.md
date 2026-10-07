# Histograms and Distributions

[Home](../README.md) / [Visualization](README.md)

## What they are used for

Histograms visualize the distribution of a numerical variable.

They show:

- skewness
- spread
- center
- possible outliers
- multimodality (several peaks, which can reveal subgroups)

## Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

numerical = data.select_dtypes(include="number").columns

for feature in numerical:
    sns.histplot(data[feature], kde=True, bins=20)
    plt.title(f"Distribution of {feature}")
    plt.show()
```

`kde=True` overlays a kernel density estimate, a smoothed version of the histogram. The number of bins changes the visual impression: too few bins hide structure, too many bins show noise.

## Also useful for DataFrames

```python
_ = adult_census.hist(figsize=(20, 14))
```

`DataFrame.hist` draws one histogram per numerical column in a grid, which is a fast first overview.

## Countplot for categorical distributions

```python
_ = sns.countplot(
    data=data,
    x="Species",
    hue="Species",
    palette="viridis",
    legend=False,
)
plt.title("Number of Samples per Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()
```

Since seaborn 0.13, a `palette` should be combined with an explicit `hue`; `legend=False` avoids a redundant legend when `hue` repeats the `x` variable. For many categories, sort them by frequency with `order=data["Species"].value_counts().index`.

## Comparing distributions across classes

```python
sns.histplot(
    data=adult_census,
    x="age",
    hue="class",
    element="step",
    stat="density",
    common_norm=False,
)
plt.show()
```

`stat="density"` with `common_norm=False` normalizes each class separately, so classes of different sizes can be compared.

## When to use it

Use histograms when you want to inspect:

- value ranges
- concentration of samples
- class imbalance (with count plots)
- whether scaling or a transformation may help

## Important note

A distribution may suggest that a feature is heavily skewed or dominated by a few values. This can influence the choice of model and preprocessing strategy, for example a log transformation, quantile binning, or a robust scaler.

## Related pages

- [Visualization Overview](Visualization-Overview.md)
- [Correlation and Distributions](../03-EDA/Correlation-and-Distributions.md)
- [Outlier Detection](../03-EDA/Outlier-Detection.md)
- [seaborn API reference: histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html)

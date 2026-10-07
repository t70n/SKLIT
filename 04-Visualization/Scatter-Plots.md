# Scatter Plots

[Home](../README.md) / [Visualization](README.md)

## Purpose

A scatter plot shows the relationship between two numerical variables, one point per sample.

## Basic example

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load an example dataset shipped with seaborn
tips = sns.load_dataset("tips")

_ = sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="time",
    size="size",
    alpha=0.7,
)
plt.title("Total Bill vs. Tip")
plt.show()
```

`hue`, `size`, and `style` encode additional variables, which makes it possible to inspect interactions with a third variable.

## Key idea

Scatter plots are useful when you want to inspect:

- correlation between two variables
- cluster structure
- outliers
- interactions between numerical variables
- whether classes are separable in two dimensions

## Large datasets and overplotting

When thousands of points overlap, the plot can hide the density of observations. Possible remedies:

```python
sns.scatterplot(data=df, x="x", y="y", alpha=0.2, s=10)  # transparency and small markers
plt.hexbin(df["x"], df["y"], gridsize=40, cmap="viridis")  # 2D histogram
sns.scatterplot(data=df.sample(2_000, random_state=0), x="x", y="y")  # random subsample
```

## Interpretation

If two variables appear to be related, they may encode similar information (redundancy) or a model may be able to exploit the relationship. A visible pattern is not a causal relationship, and the absence of a linear trend does not rule out a nonlinear or conditional relationship.

To add a fitted trend line, use `sns.regplot` or `sns.lmplot`; see [Correlation and Distributions](../03-EDA/Correlation-and-Distributions.md#plot-the-relationship-not-just-the-coefficient).

## Related pages

- [Pairplots](Pairplots.md)
- [Correlation and Distributions](../03-EDA/Correlation-and-Distributions.md)
- [seaborn API reference: scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)

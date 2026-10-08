# Pairplots

[Home](../README.md) / [Visualization](README.md)

## What it is

A pairplot shows all pairwise relationships between numerical variables in a compact grid: scatter plots off the diagonal and the distribution of each variable on the diagonal.

## Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

_ = sns.pairplot(
    data=iris,
    hue="species",
    height=2.5,
    plot_kws={"s": 20, "alpha": 0.6},
    diag_kws={"bins": 20, "alpha": 0.7},
)
plt.show()
```

## Why it is useful

It gives a fast overview of:

- relationships between pairs of variables
- the distribution of each variable
- whether classes are separable
- whether some features are redundant

## Useful options

- `vars=[...]`: restrict the grid to a subset of columns
- `corner=True`: draw only the lower triangle, since the upper triangle repeats it
- `kind="reg"`: add a linear fit to each scatter plot
- `diag_kind="kde"`: use density curves on the diagonal

## Good use case

When you want a quick overview of several numerical features without manually creating many plots.

## Caveat

The number of panels grows quadratically with the number of features, so pairplots become dense and slow beyond roughly ten variables. Select a subset of features, or sample rows for large datasets. They are useful for exploration, not for final reporting.

## Related pages

- [Scatter Plots](Scatter-Plots.md)
- [Correlation and Distributions](../03-EDA/Correlation-and-Distributions.md)
- [seaborn API reference: pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

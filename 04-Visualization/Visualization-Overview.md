# Visualization Overview

[Home](../README.md) / [Visualization](README.md)

## Why visualization matters

Visualization helps you understand:

- the distribution of variables
- relationships between features
- class imbalance
- outliers
- whether preprocessing is needed

A model will often fail if the data is misunderstood.

## Common visualization types

| Plot | Question it answers | Page |
| --- | --- | --- |
| Histogram | How is one numerical variable distributed? | [Histograms and Distributions](Histograms-and-Distributions.md) |
| Count plot | How frequent is each category? | [Histograms and Distributions](Histograms-and-Distributions.md#countplot-for-categorical-distributions) |
| Box plot | Where are the median, the quartiles, and the extreme values? | [Outlier Detection](../03-EDA/Outlier-Detection.md#boxplots-and-distributions) |
| Scatter plot | How do two numerical variables relate? | [Scatter Plots](Scatter-Plots.md) |
| Pairplot | How do several numerical variables relate pairwise? | [Pairplots](Pairplots.md) |
| Correlation heatmap | Which variables move together? | [Correlation and Distributions](../03-EDA/Correlation-and-Distributions.md#visualize-the-matrix) |
| Learning curve | Does the model benefit from more data? | [Learning Curves](../08-Model-Evaluation/Learning-Curves.md) |
| Validation curve | How does a hyperparameter change under- and overfitting? | [Validation Curves](../08-Model-Evaluation/Validation-Curves.md) |
| Decision boundary | Which regions of feature space receive each class? | [Logistic Regression Decision Boundaries](../10-Recipes/Logistic-Regression-Decision-Boundaries.md) |
| Prediction error plot | Where does a regressor make large errors? | [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md#visual-diagnostics) |

## Libraries

- `matplotlib.pyplot`: low-level plotting and figure control
- `seaborn`: statistical plots built on matplotlib, convenient with DataFrames
- scikit-learn `Display` objects (`ConfusionMatrixDisplay`, `RocCurveDisplay`, `LearningCurveDisplay`, `DecisionBoundaryDisplay`, `PredictionErrorDisplay`): plots tied to estimators and metrics

## Typical imports

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

## Best practice

Do not visualize too early or without context. Ask:

- what feature am I looking at?
- what question do I want to answer?
- does the plot reveal class imbalance, skew, or outliers?

Label the axes with units, use a title that states the question, and keep color scales fixed when several plots must be compared.

## Symmetric log scale for coefficient plots

When coefficients contain both positive and negative values and a few magnitudes are much larger than the rest, a normal axis can hide nearly all variation near zero. Matplotlib's symmetric logarithmic scale preserves the sign while expanding the small-magnitude region:

```python
fig, ax = plt.subplots(figsize=(10, 8))
weights.plot.box(vert=False, ax=ax)
ax.set_xscale("symlog")
ax.set_xlabel("Coefficient value")
plt.tight_layout()
plt.show()
```

`symlog` is a visualization scale, not a modification of the model parameters. Use it for coefficient distributions, residuals, or other signed quantities spanning several orders of magnitude. Interpret stability across cross-validation folds rather than selecting a feature only because its coefficient is visually large. A complete example is given in [Ridge Regularization and Coefficient Stability](../10-Recipes/Ridge-Regularization-and-Stability.md).

## Related pages

- [EDA Overview](../03-EDA/EDA-Overview.md)
- [matplotlib documentation](https://matplotlib.org/stable/)
- [seaborn documentation](https://seaborn.pydata.org/)
- [scikit-learn user guide: Visualizations](https://scikit-learn.org/stable/visualizations.html)

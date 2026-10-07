# Feature Engineering Overview

[Home](../README.md) / [Feature Engineering](README.md)

## What feature engineering does

Raw columns are measurements. Feature engineering transforms them into representations that make a prediction problem easier for a chosen model and easier for a human to interpret.

A model learns a function of the available features:

$$
\hat{y}=f(X)
$$

Feature engineering creates a new representation $Z=g(X)$ and trains:

$$
\hat{y}=h(Z)=h(g(X))
$$

A good representation can turn a curved, interacting, or locally structured problem into one that a simpler model can solve.

## Feature selection versus feature generation

### Feature selection

Keep some existing columns and remove others. Reasons include:

- removing constants and duplicate information
- reducing noise and computation
- improving interpretability
- limiting overfitting
- removing leakage or unavailable-at-prediction fields

Feature selection can use domain rules, `VarianceThreshold`, model-based importance, permutation importance, or recursive selection. If the target influences selection, fit it inside cross-validation.

### Feature generation

Create new columns from existing data:

- transformations of one row, such as logs, squares, ratios, and indicators
- interactions between columns
- group statistics across rows
- temporal or spatial window statistics
- latent components such as PCA
- shared components across data sources such as CCA

Automatic feature engineering remains difficult because useful transformations depend on the domain, validation design, and model inductive bias.

## Why new features help

A linear model cannot represent every relationship in the original coordinates. Adding $x^2$ can represent curvature:

$$
\hat{y}=w_0+w_1x+w_2x^2
$$

Adding $x_1x_2$ allows the effect of one variable to depend on another:

$$
\hat{y}=w_0+w_1x_1+w_2x_2+w_3x_1x_2
$$

Feature engineering can also make interpretation clearer. A ratio, rate, age group, or change since a baseline may match the question better than the raw columns.

## A feature-engineering workflow

1. Define the prediction time and information available then.
2. Understand the row, entity, time, and target definitions.
3. Start with a baseline using raw features and a suitable preprocessing pipeline.
4. Generate a small set of domain-justified features.
5. Fit feature generation inside the pipeline.
6. Compare with cross-validation using the same folds.
7. Select features or regularize when the feature count grows.
8. Inspect feature stability, ablation results, and leakage risks.
9. Validate the final feature schema on future data.

## Feature-engineering pages

- [Row-Wise Feature Generation](Row-Wise-Feature-Generation.md)
- [Cross-Row and Local Features](Cross-Row-and-Local-Features.md)
- [Feature Selection](Feature-Selection.md)
- [Principal Component Analysis](Principal-Component-Analysis.md)
- [Canonical Correlation Analysis](Canonical-Correlation-Analysis.md)
- [Polynomial Features](Polynomial-Features.md)
- [KBinsDiscretizer](KBinsDiscretizer.md)
- [SplineTransformer](SplineTransformer.md)
- [Nystroem Kernel Approximation](Nystroem-Kernel-Approximation.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)

## Choosing a nonlinear expansion

| Transformer | Effect on each feature | Creates interactions | Smoothness | Typical downstream model |
| --- | --- | --- | --- | --- |
| [`PolynomialFeatures`](Polynomial-Features.md) | Powers $x, x^2, \ldots$ | Yes (products) | Smooth, global | Ridge, logistic regression |
| [`KBinsDiscretizer`](KBinsDiscretizer.md) | Interval indicators | No | Piecewise constant | Linear models |
| [`SplineTransformer`](SplineTransformer.md) | Local polynomial basis | No | Smooth, local | Ridge, logistic regression |
| [`Nystroem`](Nystroem-Kernel-Approximation.md) | Approximate kernel feature map | Yes (through the kernel) | Depends on the kernel | Linear models |

A comparison of these pipelines with shared cross-validation folds is given in [Comparing Nonlinear Feature-Engineering Pipelines](../10-Recipes/Nonlinear-Feature-Engineering-Comparison.md).

## The central warning: leakage

A feature is invalid if it uses information that would not be available when the prediction is made. Common examples are future values, target-derived group means, test-set statistics, or a post-outcome status.

Put learned transformations in a pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1.0),
)
```

If a feature is generated across rows, split by time or group before computing it when the prediction setting requires that separation.

## Related pages

- [Why Preprocessing Matters](../05-Preprocessing/Why-Preprocessing-Matters.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)
- [Identifiers, Leakage, Shuffling, and Sampling](../03-EDA/Identifiers-Leakage-and-Sampling.md)
- [Comparing Nonlinear Feature-Engineering Pipelines](../10-Recipes/Nonlinear-Feature-Engineering-Comparison.md)

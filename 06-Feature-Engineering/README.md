# Feature Engineering

[Home](../README.md)

This section covers the creation, expansion, selection, and reduction of features: building new columns from existing ones, adding nonlinear representations to linear models, and keeping only informative features, while preventing leakage.

## Pages

### Overview

| Page | Summary |
| --- | --- |
| [Feature Engineering Overview](Feature-Engineering-Overview.md) | Selection versus generation, workflow, choosing a nonlinear expansion, leakage |

### Feature generation

| Page | Summary |
| --- | --- |
| [Row-Wise Feature Generation](Row-Wise-Feature-Generation.md) | Transformations, ratios, differences, indicators, and interactions within a row |
| [Cross-Row and Local Features](Cross-Row-and-Local-Features.md) | Group, time-window, and neighborhood statistics, and their leakage risks |

### Nonlinear expansions

| Page | Summary |
| --- | --- |
| [Polynomial Features](Polynomial-Features.md) | Powers and interactions for linear models |
| [KBinsDiscretizer](KBinsDiscretizer.md) | Piecewise-constant effects through binning |
| [SplineTransformer](SplineTransformer.md) | Smooth piecewise-polynomial effects |
| [Nystroem Kernel Approximation](Nystroem-Kernel-Approximation.md) | Approximate kernel feature maps for linear models |

### Selection and dimensionality reduction

| Page | Summary |
| --- | --- |
| [Feature Selection](Feature-Selection.md) | Filter, wrapper, and embedded methods; leakage-safe selection |
| [Principal Component Analysis](Principal-Component-Analysis.md) | Orthogonal directions of maximal variance |
| [Canonical Correlation Analysis](Canonical-Correlation-Analysis.md) | Maximally correlated projections of two data views |

## Navigation

- Previous section: [Preprocessing](../05-Preprocessing/README.md)
- Next section: [Models](../07-Models/README.md)
- [Back to the home page](../README.md)

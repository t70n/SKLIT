# Canonical Correlation Analysis (CCA)

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

Canonical Correlation Analysis finds linear combinations of two data views that are maximally correlated. It is useful when each object has multiple sources or modalities, such as:

- article text and author metadata
- webpage text and hyperlinks
- audio and lyrics
- images and captions
- measurements from two sensor systems

PCA summarizes one feature matrix. CCA finds a shared linear subspace between two matrices.

## Mathematics

Let $X\in\mathbb{R}^{n\times p}$ and $Y\in\mathbb{R}^{n\times q}$ be centered views for the same $n$ observations. CCA finds vectors $a$ and $b$ maximizing:

$$
(a_1,b_1)=\arg\max_{a,b}\operatorname{corr}(Xa,Yb)
$$

subject to normalization constraints:

$$
 a^\top\Sigma_{XX}a=1,\qquad b^\top\Sigma_{YY}b=1
$$

The first pair gives the strongest shared linear relationship. Later pairs are constrained to capture new shared structure rather than repeat earlier canonical variates.

The cross-covariance matrices are:

$$
\Sigma_{XX}=\operatorname{cov}(X,X),\quad
\Sigma_{YY}=\operatorname{cov}(Y,Y),\quad
\Sigma_{XY}=\operatorname{cov}(X,Y)
$$

The canonical correlations are the correlations between the paired projections $Xa_k$ and $Yb_k$.

## Scikit-learn example

```python
import numpy as np
from sklearn.cross_decomposition import CCA

X = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [2.0, 2.0, 2.0],
    [3.0, 5.0, 4.0],
])
Y = np.array([
    [0.1, -0.2],
    [0.9, 1.1],
    [6.2, 5.9],
    [11.9, 12.3],
])

cca = CCA(n_components=1, max_iter=1000)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)

canonical_correlation = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
print(canonical_correlation)
```

The rows of `X` and `Y` must refer to the same observations in the same order.

CCA does not concatenate the two views. It learns one projection for `X` and one projection for `Y`, using their row-wise correspondence:

```python
cca = CCA(n_components=1)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
```

The paired transformed columns can then be compared directly, for example with `np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]`.

## Interpreting the output

- `x_weights_` and `y_weights_` define combinations of the original variables.
- `x_scores_` and `y_scores_` are projected observations.
- `x_loadings_` and `y_loadings_` describe how variables relate to the canonical scores.
- `n_components` controls the number of paired shared dimensions.

Signs are arbitrary: flipping both vectors produces the same correlation. Interpret the magnitude and pattern of weights, not the sign in isolation.

## Preprocessing and validation

CCA is sensitive to scale and multicollinearity. Standardize each view when units differ, and consider regularization for high-dimensional data:

```python
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Fit separate scaling for each view before CCA.
X_scaled = StandardScaler().fit_transform(X_train)
Y_scaled = StandardScaler().fit_transform(Y_train)
cca = CCA(n_components=2, scale=False, max_iter=1000)
```

For a predictive workflow, fit scalers and CCA on training data only. Select the number of components using validation rather than maximizing training correlation.

The basic `CCA` estimator does not regularize strongly enough for every high-dimensional setting. Regularized CCA or dimensionality reduction within each view may be more stable when features outnumber observations.

## CCA versus PCA

| Method | Data views | Objective | Output relationship |
| --- | --- | --- | --- |
| PCA | One matrix | Maximize variance | Components within one view are uncorrelated |
| CCA | Two matrices | Maximize correlation between views | Paired components across views are maximally correlated |

CCA is not a causal method and a high canonical correlation can result from confounding, leakage, or shared preprocessing artifacts.

## Related pages

- [Principal Component Analysis](Principal-Component-Analysis.md)
- [Correlation and Distributions](../03-EDA/Correlation-and-Distributions.md)
- [scikit-learn user guide: Cross decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html)

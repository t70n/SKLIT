# Principal Component Analysis (PCA)

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

PCA creates new linear features called principal components. Each component is a direction in feature space. The first component captures the largest possible variance, the second captures the largest remaining variance while being orthogonal to the first, and so on.

PCA is useful for dimensionality reduction, noise reduction, visualization, and creating less correlated features.

## Mathematics

Let centered data be $X\in\mathbb{R}^{n\times d}$. PCA finds orthonormal directions $w_k$ that maximize projected variance:

$$
 w_1=\arg\max_{\|w\|=1}\operatorname{Var}(Xw)
$$

Later directions satisfy:

$$
 w_k^\top w_j=0\quad\text{for }j<k
$$

The covariance matrix is:

$$
\Sigma=\frac{1}{n-1}X^\top X
$$

The components are eigenvectors of $\Sigma$:

$$
\Sigma w_k=\lambda_k w_k
$$

The eigenvalue $\lambda_k$ is the variance explained by component $k$. The explained-variance ratio is:

$$
 r_k=\frac{\lambda_k}{\sum_j\lambda_j}
$$

The transformed data is:

$$
Z=XW_k
$$

where $W_k$ contains the first $k$ component directions.

## Import and example

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    LogisticRegression(max_iter=1000),
)
```

`n_components=0.95` keeps enough components to explain approximately 95% of the training variance.

## Correct train/test usage

Fit PCA only on training data, then transform both datasets with the fitted object:

```python
pca = make_pipeline(StandardScaler(), PCA(n_components=5))
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

Do not call `fit_transform` separately on the test set. That creates a different coordinate system and leaks test-distribution information into the preprocessing.

## Fitting PCA on training and test features together

A pattern sometimes encountered fits PCA on the concatenation of the training and test features:

```python
import numpy as np
from sklearn.decomposition import PCA

X_all = np.concatenate([X_train, X_test])
pca = PCA(n_components=5)
pca.fit(X_all)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

This is internally consistent because both datasets are transformed by the same fitted PCA coordinate system. However, it uses the **feature distribution of the test set** to choose the PCA directions and explained-variance structure. Therefore it is appropriate for exploratory visualization, compression, or a deliberately transductive setting where unlabeled test features are allowed during preprocessing.

For an honest supervised train/test evaluation, do not fit PCA on `X_all`. Fit it on `X_train` only, as shown above, and transform `X_test` afterward. In cross-validation, put PCA inside a pipeline so every fold learns its own components from its training fold.

## Choosing components

```python
import matplotlib.pyplot as plt

pca = PCA().fit(StandardScaler().fit_transform(X_train))
plt.plot(pca.explained_variance_ratio_.cumsum(), marker="o")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.show()
```

Common choices include a fixed number, a variance threshold, or a value selected by cross-validation based on downstream predictive performance. High explained variance is not the same as high predictive usefulness because PCA is unsupervised.

## Scaling matters

PCA maximizes variance, so a feature measured in large units can dominate. Standardize numerical variables when units are not comparable:

```python
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
)
```

Do not standardize blindly when absolute scale has a meaningful scientific interpretation; make the choice explicit.

## Interpretation

- `components_` contains the directions; each row is a component.
- Large absolute loadings show which original variables define a component.
- `explained_variance_` gives variance in each component.
- `explained_variance_ratio_` gives the fraction of total variance.
- Components are orthogonal and uncorrelated on the centered training data.

A component is a weighted combination, not an original feature. Naming it requires domain interpretation of its loadings.

## Reconstruction and information loss

The inverse transform approximates the original data:

$$
\hat{X}=ZW_k^\top
$$

Discarding components loses information. The reconstruction error usually decreases as more components are retained, but a lower-dimensional representation can improve model generalization and computation.

## When to use PCA

Use PCA when:

- numeric features are strongly correlated
- dimensionality is high
- visualization or compression is needed
- a downstream model benefits from fewer, less correlated features
- privacy or storage requires a transformed representation

Avoid or carefully assess PCA when interpretability of original features is essential, the data is mostly categorical, or low-variance directions contain important predictive signal.

## Related pages

- [Canonical Correlation Analysis](Canonical-Correlation-Analysis.md)
- [Feature Selection](Feature-Selection.md)
- [StandardScaler](../05-Preprocessing/StandardScaler.md)
- [scikit-learn user guide: Decomposing signals in components](https://scikit-learn.org/stable/modules/decomposition.html)

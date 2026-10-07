# Polynomial Features

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

Polynomial feature expansion lets a linear model represent nonlinear relationships by adding powers and interactions of the original features. The downstream model is still linear in the expanded features, but the resulting function is nonlinear in the original variables.

For one feature, degree three creates:

$$
[x] \longrightarrow [x, x^2, x^3]
$$

A linear regression model on these features becomes:

$$
\hat{y}=w_0+w_1x+w_2x^2+w_3x^3
$$

This is useful when domain knowledge suggests a curved relationship but a fully nonlinear model is unnecessary.

## Import

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression(),
)
```

## How it works

For two features $x_1$ and $x_2$, degree two can create:

$$
[x_1,x_2] \longrightarrow [x_1,x_2,x_1^2,x_1x_2,x_2^2]
$$

The product $x_1x_2$ represents an interaction: the effect of one feature depends on the value of the other. The linear model learns one coefficient for every generated feature.

The number of generated features grows quickly. Without `interaction_only`, the number of terms up to degree $d$ for $p$ input features is:

$$
\binom{p+d}{d}
$$

This includes the constant bias term when `include_bias=True`.

## Minimal example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 2 * X[:, 0] ** 2 - X[:, 0] + rng.normal(0, 1, 100)

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)
model.fit(X, y)
```

Always place the transformation inside a pipeline so that cross-validation fits the feature expansion consistently within each training fold.

## Important parameters

### `degree`

Maximum polynomial degree. Increasing it makes the model more flexible but increases feature count, computation, and overfitting risk.

### `include_bias`

Adds a column of ones. Keep it `False` when the regression model already fits an intercept, which avoids duplicate intercept terms.

### `interaction_only`

When `False`, include powers such as $x_1^2$ and products such as $x_1x_2$. When `True`, include only products of distinct features:

$$
[x_1,x_2,x_3]\longrightarrow[x_1,x_2,x_3,x_1x_2,x_1x_3,x_2x_3,x_1x_2x_3]
$$

The interaction-only version excludes powers like $x_1^2$. It is useful when feature combinations matter but repeated powers have no meaningful interpretation, and it reduces the number of generated columns.

### `order`

Controls the memory layout of the output array (`"C"` or `"F"`). It is mainly a performance detail and normally does not change predictions.

## Scaling and regularization

Polynomial columns can have very different scales. For example, $x^3$ can be much larger than $x$. Scaling the expanded features is often important, especially with Ridge, Lasso, or Elastic Net:

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0),
)
```

Regularization is often essential for high-degree expansions because generated features can be strongly correlated.

## When to use it

Use polynomial features when:

- the data is moderate-sized and a smooth, low-degree curve is plausible
- you need a model that remains relatively interpretable
- domain knowledge suggests specific powers or interactions
- you want to enrich a linear model before trying a more complex estimator

Prefer tree ensembles, splines, or kernel methods when the relationship is highly irregular, the feature count is large, or the polynomial expansion would be too big.

## Limitations

- feature count grows combinatorially
- high degrees can oscillate and overfit
- extrapolation can become extreme outside the training range
- coefficients of correlated polynomial terms can be difficult to interpret
- expansion must be fitted only on training data to avoid leakage

## Related concepts

- `Ridge` and `Lasso` can regularize the expanded design matrix; see [Ridge Regularization and Coefficient Stability](../10-Recipes/Ridge-Regularization-and-Stability.md).
- `SVR` can model nonlinear relationships through kernels without explicitly creating every polynomial column; see [SVR](../07-Models/Support-Vector-Machines/SVR.md).
- `Nystroem` provides an approximate kernel feature map with a controlled number of components; see [Nystroem Kernel Approximation](Nystroem-Kernel-Approximation.md).

## Related pages

- [Row-Wise Feature Generation](Row-Wise-Feature-Generation.md)
- [SplineTransformer](SplineTransformer.md)
- [Linear Regression](../07-Models/Linear-Models/Linear-Regression.md)
- [scikit-learn API reference: PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

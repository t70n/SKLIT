# SplineTransformer

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

`SplineTransformer` represents a numerical feature with smooth piecewise-polynomial basis functions. A linear model can then combine those basis functions to learn a curved relationship while remaining linear in the learned coefficients.

Splines are a middle ground between a straight-line model and a high-degree global polynomial: the curve is flexible locally but does not require one large polynomial to describe the whole range.

## Import and example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

classifier = make_pipeline(
    SplineTransformer(degree=3, n_knots=5),
    LogisticRegression(max_iter=1000),
)
```

Use the transformer in a pipeline so knots and feature ranges are learned only from training data.

## How it works mathematically

A spline represents a function as a weighted sum of basis functions:

$$
 f(x)=\sum_{j=1}^{m}w_jB_j(x)
$$

Each basis function $B_j(x)$ is nonzero over a local region and is a polynomial of degree `degree` on each interval. Neighboring polynomial pieces are joined with continuity constraints, producing a smooth curve.

For a cubic spline, `degree=3`, the basis functions are cubic polynomials joined so that the function and lower-order derivatives change smoothly at the knots. The estimator still learns the weights $w_j$ with a linear model.

## Important parameters

- `n_knots`: number of knot locations; more knots increase flexibility
- `degree`: polynomial degree of each local piece, commonly 3
- `knots`: `"uniform"`, `"quantile"`, or manually supplied knot locations
- `extrapolation`: behavior outside the fitted range, such as `"constant"`, `"linear"`, or `"continue"`
- `include_bias`: whether to include a redundant all-ones basis column
- `sparse_output`: whether to return a sparse matrix in supported versions

Quantile knots place more modeling capacity where observations are dense. Uniform knots are easier to interpret in the original scale.

## Example with regularized regression

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

model = make_pipeline(
    SplineTransformer(n_knots=6, degree=3, include_bias=False),
    StandardScaler(with_mean=False),
    Ridge(alpha=1.0),
)
```

Regularization can stabilize the coefficients when neighboring spline basis columns are correlated.

## Why splines may still fail on XOR

By default, `SplineTransformer` acts independently on each original feature. It can model smooth nonlinear effects of $x_1$ and $x_2$, but it does not automatically create an interaction term such as $B_i(x_1)B_j(x_2)$. A non-additive pattern like XOR can therefore remain inseparable.

Use `PolynomialFeatures`, an interaction-aware kernel approximation, or a model with native interactions when cross-feature effects are essential.

## When to use it

Use splines when:

- the relationship with a numerical feature is smooth but not linear
- local flexibility is useful and a global polynomial is unstable
- you want a relatively interpretable feature transformation
- extrapolation behavior should be controlled explicitly

Tree models are often better for discontinuities and complex interactions. Splines are usually not a first choice for raw categorical features.

## Limitations

- too few knots underfit; too many knots overfit
- feature-wise splines do not automatically capture interactions
- boundary behavior depends strongly on `extrapolation`
- basis coefficients are less directly interpretable than original-feature coefficients
- fitting knots outside a pipeline can leak information during validation

Use [KBinsDiscretizer](KBinsDiscretizer.md) when abrupt intervals are meaningful and [Polynomial Features](Polynomial-Features.md) when explicit powers or interactions are desired.

## Related pages

- [KBinsDiscretizer](KBinsDiscretizer.md)
- [Polynomial Features](Polynomial-Features.md)
- [Comparing Nonlinear Feature-Engineering Pipelines](../10-Recipes/Nonlinear-Feature-Engineering-Comparison.md)
- [scikit-learn API reference: SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)

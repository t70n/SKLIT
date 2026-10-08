# Nystroem Kernel Approximation

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

Kernel methods can represent rich nonlinear functions, but exact kernel algorithms may require a large matrix of pairwise similarities. `Nystroem` creates an approximate feature representation so a standard linear model can work in an explicitly transformed space.

It is useful when you want kernel-like nonlinear behavior but need to control memory and computation.

## Import

```python
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    Nystroem(kernel="poly", degree=3, n_components=100, random_state=0),
    Ridge(alpha=1.0),
)
```

## Kernel intuition

A kernel measures similarity without explicitly writing all coordinates of a high-dimensional feature map:

$$
K(x,z)=\langle\phi(x),\phi(z)\rangle
$$

For a polynomial kernel, $\phi(x)$ contains powers and interactions. For an RBF kernel, the equivalent feature space is effectively very large or infinite. Computing every pairwise similarity can cost substantial time and memory.

Nystroem samples a subset of training points and uses them to construct a low-dimensional approximation $\tilde{\phi}(x)$ such that:

$$
K(x,z)\approx\tilde{\phi}(x)^\top\tilde{\phi}(z)
$$

The downstream estimator then learns a linear function in the approximate feature space.

## Minimal example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    Nystroem(kernel="poly", degree=3, n_components=50, random_state=0),
    Ridge(alpha=1.0),
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Put Nystroem inside the pipeline so the sampled basis and transformation are fitted separately inside each cross-validation training fold.

## Important parameters

### `kernel`

The similarity function, such as `"rbf"`, `"poly"`, `"chi2"`, or `"sigmoid"`. Its supported parameters depend on the kernel.

### `n_components`

The number of approximate features. More components generally improve the approximation but increase memory, fitting time, and downstream model size. It is the main capacity/performance trade-off.

### `gamma`, `degree`, `coef0`, and `kernel_params`

These control the selected kernel. For RBF, `gamma` determines locality. For polynomial kernels, `degree` controls interaction order and `coef0` changes the offset.

### `random_state`

Controls which basis samples are selected. Fix it for reproducible experiments.

## Approximation versus exact kernels

Increasing `n_components` usually makes the approximation more accurate, but it does not guarantee better test performance: the resulting model can still overfit. Compare approximate and exact approaches with the same cross-validation splits.

Nystroem is different from `PolynomialFeatures`:

- `PolynomialFeatures` explicitly creates interpretable powers and interactions up to a degree.
- `Nystroem` creates a data-dependent approximate feature map for a chosen kernel.
- Polynomial features can be preferable when the generated terms have domain meaning.
- Nystroem can be preferable when an exact kernel is too expensive or when an RBF-like feature space is desired.

## When to use it

Use Nystroem when:

- a nonlinear kernel model is promising but too expensive at full scale
- you want a transformed matrix compatible with linear estimators
- you need to tune the feature-map size explicitly
- the dataset is moderate or large enough that exact kernel methods are a bottleneck

It is not automatically the best choice for very large sparse datasets or problems where boosted trees already provide a strong tabular solution.

## Limitations

- approximation quality depends on `n_components` and sampled basis points
- transformed features can still be expensive to store
- kernel hyperparameters remain important
- intermediate features are difficult to interpret
- fitting must remain inside a pipeline to prevent leakage

## Related pages

- [Polynomial Features](Polynomial-Features.md)
- [SVC](../07-Models/Support-Vector-Machines/SVC.md)
- [Comparing Nonlinear Feature-Engineering Pipelines](../10-Recipes/Nonlinear-Feature-Engineering-Comparison.md)
- [scikit-learn user guide: Kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html)

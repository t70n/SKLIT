# Ridge Regularization and Coefficient Stability

[Home](../README.md) / [Recipes](README.md)

## Goal

Compare an unregularized linear regression model with Ridge regression, tune `alpha` using `RidgeCV`, and inspect how coefficient magnitudes vary across cross-validation folds. Regularization is especially useful after feature engineering because polynomial and interaction features are often correlated.

## Ridge objective

Ridge minimizes:

$$
\|y-Xw\|_2^2+\alpha\|w\|_2^2
$$

The penalty shrinks coefficients toward zero. Larger `alpha` usually gives a smoother, more stable model, but excessive regularization can underfit.

## Load a numerical dataset

```python
import numpy as np
import pandas as pd

ames_housing = pd.read_csv("../datasets/ames_housing_no_missing.csv")
target = ames_housing["SalePrice"]
data = ames_housing.drop(columns="SalePrice")
data = data.select_dtypes(include="number")
```

Adapt the path when running from a different notebook directory. When the course file is not available, a comparable dataset can be fetched from OpenML (this requires network access):

```python
from sklearn.datasets import fetch_openml

ames = fetch_openml(name="house_prices", as_frame=True)
data = ames.data.select_dtypes(include="number").copy()
target = pd.to_numeric(ames.target)

# Keep only complete columns for this focused numerical recipe.
data = data.dropna(axis="columns", how="any")
```

## Compare a regularized polynomial model

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

ridge = make_pipeline(
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1.0),
)

cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
cv_results = cross_validate(
    ridge,
    data,
    target,
    cv=cv,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    return_estimator=True,
    n_jobs=2,
)

train_mse = -cv_results["train_score"]
test_mse = -cv_results["test_score"]
print(f"Train MSE: {train_mse.mean():.2e} +/- {train_mse.std():.2e}")
print(f"Test MSE: {test_mse.mean():.2e} +/- {test_mse.std():.2e}")
```

The `neg_` prefix follows scikit-learn's higher-is-better scoring convention. Negate the values to report ordinary MSE.

## Tune `alpha` with RidgeCV

```python
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-7, 5, num=100)

tuned_ridge = make_pipeline(
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    RidgeCV(alphas=alphas),
)

tuned_results = cross_validate(
    tuned_ridge,
    data,
    target,
    cv=cv,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    return_estimator=True,
    n_jobs=2,
)

best_alphas = [estimator[-1].alpha_ for estimator in tuned_results["estimator"]]
print("Selected alphas:", best_alphas)
print("Median alpha:", np.median(best_alphas))
```

`RidgeCV` performs inner selection of `alpha` inside each outer training fold. This avoids using the outer test fold to choose regularization.

## Inspect coefficient stability

```python
feature_names = data.columns
first = tuned_results["estimator"][0]
polynomial = first[1]
feature_names = polynomial.get_feature_names_out(data.columns)

coefficient_table = pd.DataFrame(
    [estimator[-1].coef_ for estimator in tuned_results["estimator"]],
    columns=feature_names,
)

summary = pd.DataFrame({
    "median": coefficient_table.median(),
    "std": coefficient_table.std(),
    "median_abs": coefficient_table.abs().median(),
})
print(summary.sort_values("median_abs", ascending=False).head(15))
```

A large coefficient is not automatically causal or stable feature importance. Compare its distribution across folds and consider correlations, scaling, and the feature-generation basis.

## Plot coefficient distributions

```python
import matplotlib.pyplot as plt

plot_data = coefficient_table.loc[:, summary.nlargest(15, "median_abs").index]
fig, ax = plt.subplots(figsize=(10, 7))
plot_data.plot.box(vert=False, ax=ax)
ax.set_title("Ridge coefficient stability across folds")
ax.set_xlabel("Coefficient")
plt.tight_layout()
plt.show()
```

If a few coefficients span a much larger range than the others, use a symmetric log scale:

```python
fig, ax = plt.subplots(figsize=(10, 7))
plot_data.plot.box(vert=False, ax=ax)
ax.set_xscale("symlog")
ax.set_title("Ridge coefficients on a symmetric log scale")
plt.tight_layout()
plt.show()
```

`symlog` handles positive and negative values while making small and large magnitudes visible. It is a visualization scale, not a transformation of fitted coefficients.

## Interpret train/test behavior

- Very low train error and much larger test error suggests overfitting.
- Similar train and test error with both high suggests underfitting or weak features.
- Increasing `alpha` generally shrinks coefficients and can reduce variance.
- Select `alpha` by validation, not by coefficient appearance.

Use the same folds when comparing unregularized and regularized pipelines.

## Related pages

- [Linear Regression](../07-Models/Linear-Models/Linear-Regression.md)
- [Polynomial Features](../06-Feature-Engineering/Polynomial-Features.md)
- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)
- [Visualization Overview](../04-Visualization/Visualization-Overview.md#symmetric-log-scale-for-coefficient-plots)

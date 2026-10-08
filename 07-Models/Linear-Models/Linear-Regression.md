# Linear Regression

[Home](../../README.md) / [Models](../README.md) / [Linear Models](../README.md#linear-models)

## Idea

Linear regression predicts a continuous target as a weighted sum of the input features. It is one of the most useful baselines because it is fast, interpretable, and gives a clear reference point for more complex models.

For one feature, the model is a line:

$$
\hat{y} = w_0 + w_1x
$$

For multiple features, it is a hyperplane:

$$
\hat{y}_i = w_0 + w_1x_{i1} + \cdots + w_px_{ip}
$$

In matrix notation, with a column of ones included in $X$:

$$
\hat{\mathbf{y}} = X\mathbf{w}
$$

The coefficients describe the expected change in the prediction for a one-unit increase in a feature, while holding the other features constant.

## Import

```python
from sklearn.linear_model import LinearRegression
```

## Minimal example

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))
print("Coefficients:", model.coef_)
```

## How it learns

The usual objective is ordinary least squares (OLS): choose the coefficients that minimize the sum of squared residuals.

$$
\operatorname{SSE}(\mathbf{w}) = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
= \|\mathbf{y} - X\mathbf{w}\|_2^2
$$

Squaring makes large errors especially costly and gives a smooth, differentiable objective. The residual is $e_i = y_i - \hat{y}_i$.

When $X^TX$ is invertible, the optimum has the closed-form solution:

$$
\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}
$$

scikit-learn uses numerically stable linear algebra rather than explicitly computing the inverse. Iterative gradient descent is another way to reach the same minimum. Each update moves the coefficients in the direction opposite to the gradient:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla_{\mathbf{w}}\operatorname{SSE}
$$

where $\eta$ is the learning rate.

## Interpreting the fit

- `coef_` contains the feature weights.
- `intercept_` is the prediction when all features are zero.
- Positive coefficients increase the prediction as the feature increases.
- A coefficient is not automatically causal or comparable across features with different units.
- Strongly correlated features can make individual coefficients unstable even when predictions are good.

## Important assumptions

The model does not require every assumption below for prediction to work, but they matter for classical statistical interpretation:

- linearity: the conditional mean is approximately linear in the features
- independent observations
- errors with constant variance (homoscedasticity)
- limited multicollinearity when interpreting individual coefficients
- approximately normal errors for small-sample confidence intervals and hypothesis tests

Inspect residuals rather than trusting a high training score. Curved residual patterns suggest missing nonlinear structure; changing residual spread suggests non-constant variance.

## Key parameters and alternatives

`LinearRegression` has few predictive hyperparameters:

- `fit_intercept=False` is appropriate only when the features and target are known to have a zero intercept
- `positive=True` constrains coefficients to be non-negative
- `n_jobs` can parallelize some multi-target fits

Regularized linear models are often better when there are many features or collinearity:

- `Ridge`: adds $\alpha\|\mathbf{w}\|_2^2$ and shrinks coefficients
- `Lasso`: adds $\alpha\|\mathbf{w}\|_1$ and can set coefficients exactly to zero
- `ElasticNet`: combines L1 and L2 penalties

For example, Ridge minimizes:

$$
\|\mathbf{y} - X\mathbf{w}\|_2^2 + \alpha\|\mathbf{w}\|_2^2
$$

`RidgeCV`, `LassoCV`, and `ElasticNetCV` select the regularization strength `alpha` by internal cross-validation. The intercept is not penalized. Because the penalty depends on the coefficient magnitudes, scale the features before fitting a regularized model.

## Preprocessing

- numerical features do not need scaling for the OLS solution, but scaling is useful for regularized models and coefficient comparison
- categorical features need encoding, usually one-hot encoding
- put imputation, encoding, scaling, and the model in a `Pipeline` to prevent data leakage
- polynomial features can represent curved relationships while keeping a linear model in its parameters

Polynomial expansion is one way to add nonlinear structure without changing the final estimator:

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression(),
)
```

`PolynomialFeatures` can generate powers such as $x^2$ and interactions such as $x_1x_2$. Use `interaction_only=True` when you want products of distinct features but not repeated powers. See [Polynomial Features](../../06-Feature-Engineering/Polynomial-Features.md) for the feature-count and regularization trade-offs.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_columns),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_columns),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0)),
])
```

## Evaluation

Useful regression metrics include:

- MAE: average absolute error, easy to interpret and less affected by outliers
- MSE: average squared error, strongly penalizes large errors
- RMSE: square root of MSE, in the target's units
- $R^2$: fraction of variance explained relative to predicting the mean

Always compare with a [`DummyRegressor`](../Baselines/DummyRegressor.md) baseline and evaluate on held-out data or cross-validation. See [Regression Metrics](../../08-Model-Evaluation/Regression-Metrics.md).

## Cross-validation scoring

When the goal is to minimize mean absolute error, pass the scorer explicitly:

```python
from sklearn.model_selection import cross_validate

result = cross_validate(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error",
)

mae = -result["test_score"].mean()
```

scikit-learn uses the `neg_` prefix because its model-selection API assumes that larger scores are better. The returned values are negative MAE, so negate them to report the usual positive error. Keep the transformation and estimator together in a pipeline during cross-validation.

## When to use it

Use linear regression when:

- the relationship is approximately additive and linear
- speed and interpretability matter
- you need a strong, transparent baseline
- the sample size is large relative to the number of features
- extrapolation is required and a domain-justified linear trend is plausible

Consider a tree ensemble, generalized additive model, or feature transformations when interactions and nonlinearities dominate. Linear regression can extrapolate, but its extrapolation is only as credible as the assumed trend.

## Limitations

- outliers can have a large effect because residuals are squared
- a linear model misses nonlinear effects unless features are transformed
- correlated predictors make coefficient interpretation difficult
- high-dimensional unregularized fits can overfit or become numerically unstable
- predictions outside the training range may be unreliable

## Related pages

- [LogisticRegression](LogisticRegression.md)
- [Polynomial Features](../../06-Feature-Engineering/Polynomial-Features.md)
- [Ridge Regularization and Coefficient Stability](../../10-Recipes/Ridge-Regularization-and-Stability.md)
- [Ames Housing: Linear Model vs Decision Tree](../../10-Recipes/Ames-Housing-Linear-vs-Tree.md)
- [scikit-learn user guide: Linear models](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn API reference: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

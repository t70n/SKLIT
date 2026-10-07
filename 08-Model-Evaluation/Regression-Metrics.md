# Regression Metrics

[Home](../README.md) / [Model Evaluation](README.md)

## Why metrics matter

Regression models are evaluated differently from classifiers. Instead of counting correct labels, we measure how close the predictions $\hat{y}_i$ are to the true targets $y_i$, usually through the residuals:

$$
e_i=y_i-\hat{y}_i
$$

Each metric aggregates the residuals differently and therefore emphasizes different kinds of errors. For the mechanics of scorers and the `neg_` convention, see [Metrics and Scoring Overview](Metrics-and-Scoring.md).

## Start with a baseline

A regressor that always predicts the mean of the training target is the natural reference:

```python
from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(data_train, target_train)
print(f"R2 score for a regressor predicting the mean: {dummy_regressor.score(data_test, target_test):.3f}")
```

Its $R^2$ is close to 0 on the test set. Report the error of the dummy regressor next to the error of the model, with the same splits; see [DummyRegressor](../07-Models/Baselines/DummyRegressor.md).

## Mean Absolute Error (MAE)

$$
\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

```python
from sklearn.metrics import mean_absolute_error

target_predicted = regressor.predict(data_test)
print(f"Mean absolute error: {mean_absolute_error(target_test, target_predicted):.3f} k$")
```

### Interpretation

MAE measures the average absolute difference between predicted and true values. It is expressed in the unit of the target, which makes it easy to communicate ("on average, the predicted price is off by 22 k$"). It is less sensitive to large errors than the MSE. The best constant prediction for the MAE is the median of the target.

## Mean Squared Error (MSE)

$$
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

### Interpretation

MSE penalizes larger errors more strongly because it squares the differences: one error of 10 costs as much as one hundred errors of 1. It is the loss minimized by ordinary least squares, but its unit is the square of the target unit, which makes the raw value difficult to interpret. The best constant prediction for the MSE is the mean of the target.

## Root Mean Squared Error (RMSE)

$$
\operatorname{RMSE}=\sqrt{\operatorname{MSE}}
$$

```python
from sklearn.metrics import root_mean_squared_error  # scikit-learn 1.4 or later

rmse = root_mean_squared_error(y_test, y_pred)
```

RMSE is back in the unit of the target while keeping the strong penalty on large errors. With older versions, use `np.sqrt(mean_squared_error(y_test, y_pred))`. RMSE is always greater than or equal to MAE; a large gap between them signals a few large errors.

## Median Absolute Error

$$
\operatorname{MedAE}=\operatorname{median}\left(|y_1-\hat{y}_1|,\ldots,|y_n-\hat{y}_n|\right)
$$

```python
from sklearn.metrics import median_absolute_error

print(f"Median absolute error: {median_absolute_error(target_test, target_predicted):.3f} k$")
```

The median is robust to outliers: it describes the typical error and ignores the size of the largest errors. Report it together with MAE or RMSE, not instead of them.

## Mean Absolute Percentage Error (MAPE)

$$
\operatorname{MAPE}=\frac{1}{n}\sum_{i=1}^{n}\frac{|y_i-\hat{y}_i|}{|y_i|}
$$

```python
from sklearn.metrics import mean_absolute_percentage_error

print(f"Mean absolute percentage error: {mean_absolute_percentage_error(target_test, target_predicted):.2%}")
```

MAPE measures relative errors, which is useful when the target spans several orders of magnitude and a 10 k$ error matters more for a cheap house than for an expensive one. scikit-learn returns a fraction (0.15 means 15%), hence the `:.2%` format. MAPE becomes very large or unstable when some true values are close to zero, and it penalizes over-predictions and under-predictions asymmetrically.

## Coefficient of determination (R-squared)

The raw MSE is difficult to interpret. One way is to rescale it by the variance of the target. This score is the $R^2$, also called the coefficient of determination:

$$
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

```python
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
regressor.score(data_test, target_test)  # same value: R2 is the default score of regressors
```

### Interpretation

$R^2$ represents the proportion of the variance of the target that is explained by the model:

- $R^2=1$: perfect predictions
- $R^2=0$: as good as always predicting the mean of the evaluated targets
- $R^2<0$: worse than predicting the mean; there is no lower bound

$R^2$ is a normalized metric, which makes it independent of the physical unit of the target, unlike MAE. However, the value obtained cannot be compared from one dataset to another, and it has no direct interpretation in the unit of the target.

It is only possible to reach 1.0 if the target is a deterministic function of the available input features. In practice, external factors often introduce variability in the target that cannot be explained by the features. The $R^2$ of an optimal model is therefore typically less than 1.0, not because of a limitation of the learning algorithm, but because the available features are not informative enough to predict the target deterministically.

### R-squared or MAE?

- $R^2$ summarizes the quality of the fit relative to a constant prediction and is convenient for comparing models on the same dataset.
- MAE (or RMSE) keeps the unit of the target and is the right choice to report errors to stakeholders ("the typical error is 22 k$").

Report both when possible.

## Deviance metrics for counts and positive targets

When the target is a count (number of claims, visits) or a strictly positive, right-skewed quantity (amounts, durations), squared-error metrics and least-squares models are often not appropriate. Generalized linear models and their deviance metrics are designed for these targets:

| Target | Model | Metric |
| --- | --- | --- |
| Counts (non-negative, variance growing with the mean) | `PoissonRegressor` | `mean_poisson_deviance`, `scoring="neg_mean_poisson_deviance"` |
| Positive continuous, right-skewed | `GammaRegressor` | `mean_gamma_deviance`, `scoring="neg_mean_gamma_deviance"` |
| Zero-inflated amounts or a family to choose | `TweedieRegressor(power=...)` | `mean_tweedie_deviance(power=...)` |

These models predict strictly positive values; read their docstrings to choose `power`, `link`, and the regularization `alpha`. Histogram gradient boosting also supports `loss="poisson"` and `loss="gamma"`.

## Scorer names

| Metric | `scoring` string | Higher is better |
| --- | --- | --- |
| $R^2$ | `"r2"` | Yes |
| MAE | `"neg_mean_absolute_error"` | Yes (negated) |
| MSE | `"neg_mean_squared_error"` | Yes (negated) |
| RMSE | `"neg_root_mean_squared_error"` | Yes (negated) |
| Median absolute error | `"neg_median_absolute_error"` | Yes (negated) |
| MAPE | `"neg_mean_absolute_percentage_error"` | Yes (negated) |
| Poisson deviance | `"neg_mean_poisson_deviance"` | Yes (negated) |

## Visual diagnostics

A single number hides where the errors occur. `PredictionErrorDisplay` (scikit-learn 1.2 or later) draws two complementary plots:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

PredictionErrorDisplay.from_predictions(
    y_true=target_test,
    y_pred=target_predicted,
    kind="actual_vs_predicted",
    scatter_kwargs={"alpha": 0.5},
    ax=axs[0],
)
axs[0].axis("square")
axs[0].set_xlabel("Predicted values (k$)")
axs[0].set_ylabel("True values (k$)")

PredictionErrorDisplay.from_predictions(
    y_true=target_test,
    y_pred=target_predicted,
    kind="residual_vs_predicted",
    scatter_kwargs={"alpha": 0.5},
    ax=axs[1],
)
axs[1].axis("square")
axs[1].set_xlabel("Predicted values (k$)")
axs[1].set_ylabel("Residual values (k$)")

_ = fig.suptitle("Regression using a model\nwithout target transformation", y=1.1)
```

- **Actual versus predicted**: a perfect model puts every point on the diagonal.
- **Residuals versus predicted**: residuals should be centered on zero with a constant spread and no structure.

When the residuals still hold some structure, typically visible as a "banana" or "smile" shape, the model can probably be improved by transforming the features or the target, or by changing the model or its parameters. A spread that grows with the predicted value (a funnel shape) indicates that errors are proportional to the target, a situation where a log or quantile transformation of the target often helps. By default, the display subsamples 1,000 points for readability.

## Transforming the target

`TransformedTargetRegressor` fits the regressor on a transformed target and automatically applies the inverse transformation to the predictions, so the metrics are computed in the original unit:

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(n_quantiles=900, output_distribution="normal")
model_transformed_target = TransformedTargetRegressor(
    regressor=regressor,
    transformer=transformer,
)
model_transformed_target.fit(data_train, target_train)
target_predicted = model_transformed_target.predict(data_test)
```

The quantile transformation monotonically reshapes the target to follow a normal distribution. A simpler alternative for positive, right-skewed targets is a logarithm:

```python
import numpy as np

model_log_target = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1,
)
```

Compare the residual plots and the metrics before and after the transformation; a complete workflow is given in [Regression Error Analysis and Target Transformation](../10-Recipes/Regression-Error-Analysis-and-Target-Transformation.md).

## Which metric should you choose?

| Situation | Useful metric |
| --- | --- |
| Errors must be reported in the target unit | MAE, or RMSE when large errors are especially costly |
| Large errors are much worse than small ones | RMSE or MSE |
| Outliers in the target should not dominate | MAE or median absolute error |
| Relative errors matter (wide target range) | MAPE, or MAE on a log-transformed target |
| Compare models on the same dataset with a normalized score | $R^2$ |
| Counts or positive skewed targets | Poisson, gamma, or Tweedie deviance |

## Important note

In cross-validation, scikit-learn expects a score where higher is better, so errors such as MAE are exposed as negative values (`"neg_mean_absolute_error"`). Negate them to report the usual positive error.

## Common mistakes

- reporting only $R^2$, which hides the size of the errors in the target unit
- comparing $R^2$ values computed on different datasets
- forgetting to negate `neg_` scores before reporting them
- using MAPE when some targets are close to zero
- evaluating a model trained on a transformed target on the transformed scale by mistake; `TransformedTargetRegressor` avoids this
- not comparing with a `DummyRegressor` baseline

## Related pages

- [Metrics and Scoring Overview](Metrics-and-Scoring.md)
- [Classification Metrics](Classification-Metrics.md)
- [DummyRegressor](../07-Models/Baselines/DummyRegressor.md)
- [Regression Error Analysis and Target Transformation](../10-Recipes/Regression-Error-Analysis-and-Target-Transformation.md)
- [scikit-learn user guide: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

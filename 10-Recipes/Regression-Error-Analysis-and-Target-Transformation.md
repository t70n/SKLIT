# Regression Error Analysis and Target Transformation

[Home](../README.md) / [Recipes](README.md)

## Goal

Evaluate a linear regression on house prices with several complementary metrics, diagnose its errors visually with `PredictionErrorDisplay`, and improve it by transforming the target with `TransformedTargetRegressor`.

## Dataset

`house_prices.csv` from the scikit-learn MOOC repository (Ames housing). The target `SalePrice` is converted to thousands of dollars (k$).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ames_housing = pd.read_csv("../datasets/house_prices.csv")
data = ames_housing.drop(columns="SalePrice")
target = ames_housing["SalePrice"]
data = data.select_dtypes(np.number)
target /= 1000

data_train, data_test, target_train, target_test = train_test_split(
    data, target, shuffle=True, random_state=0
)
```

In this file, missing values are encoded with the character `"?"`. Read without `na_values="?"`, the columns containing `"?"` are parsed as text, so `select_dtypes(np.number)` keeps only the complete numerical columns. To use every numerical column, read the file with `na_values="?"` and add a `SimpleImputer` before the regressor in a pipeline.

## Fit a baseline and a linear model

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(data_train, target_train)

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(data_train, target_train)

print(f"R2 of the linear regression: {regressor.score(data_test, target_test):.3f}")
print(f"R2 of a regressor predicting the mean: {dummy_regressor.score(data_test, target_test):.3f}")
```

The $R^2$ is the default score of regressors. The dummy regressor obtains an $R^2$ close to 0, the reference for "no better than predicting the mean".

## Report errors in the unit of the target

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)


def report_errors(target_true, target_pred):
    print(f"Mean absolute error: {mean_absolute_error(target_true, target_pred):.3f} k$")
    print(f"Median absolute error: {median_absolute_error(target_true, target_pred):.3f} k$")
    print(
        "Mean absolute percentage error: "
        f"{mean_absolute_percentage_error(target_true, target_pred):.2%}"
    )


target_predicted = regressor.predict(data_test)
report_errors(target_test, target_predicted)
```

- The MAE is the average error in k$, easy to communicate.
- The median absolute error describes a typical error and is robust to a few very large errors.
- The MAPE measures relative errors, which matters when prices span a wide range.

## Diagnose the errors visually

```python
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


def plot_prediction_errors(target_true, target_pred, title):
    fig, axs = plt.subplots(ncols=2, figsize=(13, 5))
    PredictionErrorDisplay.from_predictions(
        y_true=target_true,
        y_pred=target_pred,
        kind="actual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
        ax=axs[0],
    )
    axs[0].axis("square")
    axs[0].set_xlabel("Predicted values (k$)")
    axs[0].set_ylabel("True values (k$)")

    PredictionErrorDisplay.from_predictions(
        y_true=target_true,
        y_pred=target_pred,
        kind="residual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
        ax=axs[1],
    )
    axs[1].axis("square")
    axs[1].set_xlabel("Predicted values (k$)")
    axs[1].set_ylabel("Residual values (k$)")
    _ = fig.suptitle(title, y=1.1)


plot_prediction_errors(
    target_test, target_predicted,
    "Regression using a model\nwithout target transformation",
)
```

The residuals still hold some structure, typically visible as a "banana" or "smile" shape in the residual plot: their average is not zero over the whole range of predictions but depends on the predicted value, and their spread grows for expensive houses. This is a clue that the model could be improved, either by transforming the features or the target, or by changing the model type or its parameters.

## Transform the target

The target is right-skewed. A `QuantileTransformer` monotonically reshapes it to follow a normal distribution; `TransformedTargetRegressor` fits the regressor on the transformed target and applies the inverse transformation to the predictions, so the errors are still computed in k$:

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(n_quantiles=900, output_distribution="normal")
model_transformed_target = TransformedTargetRegressor(
    regressor=regressor, transformer=transformer
)
model_transformed_target.fit(data_train, target_train)
target_predicted = model_transformed_target.predict(data_test)

report_errors(target_test, target_predicted)
plot_prediction_errors(
    target_test, target_predicted,
    "Regression using a model that\ntransforms the target before fitting",
)
```

`n_quantiles` must not exceed the number of training samples.

## Results

In the course notebook, the target transformation reduces all three errors (absolute errors in thousands of dollars):

| Metric | Without transformation | With quantile transformation |
| --- | ---: | ---: |
| Mean absolute error | 22.6 | 17.4 |
| Median absolute error | 14.1 | 10.3 |
| Mean absolute percentage error | 13.6% | 9.9% |

The residual plot of the transformed model is also more balanced around zero. The exact values depend on the split and on the library versions.

## Variation: a model adapted to positive, skewed targets

Instead of transforming the target, a generalized linear model can model it directly. Read the docstrings of `PoissonRegressor`, `GammaRegressor`, and `TweedieRegressor` to choose the distribution (`power`) and the link function:

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

glm = make_pipeline(
    StandardScaler(),
    TweedieRegressor(power=2, link="log", alpha=1e-4, max_iter=1000),  # gamma distribution
)
glm.fit(data_train, target_train)
report_errors(target_test, glm.predict(data_test))
```

These models use a penalized solver, so the features are standardized and the default regularization `alpha=1.0` is reduced.

## Related pages

- [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md)
- [DummyRegressor](../07-Models/Baselines/DummyRegressor.md)
- [Linear Regression](../07-Models/Linear-Models/Linear-Regression.md)
- [Quick Regression Pipeline](Quick-Regression-Pipeline.md)

# MinMaxScaler

[Home](../README.md) / [Preprocessing](README.md)

## Idea

`MinMaxScaler` rescales each numerical feature independently to a chosen interval, usually $[0,1]$:

$$
x' = \frac{x-x_{min}}{x_{max}-x_{min}}
$$

For a general interval $[a,b]$:

$$
x' = a + \frac{(x-x_{min})(b-a)}{x_{max}-x_{min}}
$$

The minimum and maximum are learned from the training data during `fit`.

## Basic usage

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Never fit the scaler independently on the test data. The test values must be transformed using the training range.

## Pipeline usage

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

model = make_pipeline(
    MinMaxScaler(),
    Ridge(alpha=1.0),
)
model.fit(X_train, y_train)
```

Putting the scaler in the pipeline ensures that cross-validation learns a separate range for every training fold.

## Important parameters

### `feature_range`

Choose another output interval:

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
```

### `clip`

If `clip=True`, transformed test values outside the training range are clipped to the requested interval:

```python
scaler = MinMaxScaler(clip=True)
```

Without clipping, a future value below the training minimum or above the training maximum can transform below 0 or above 1. This preserves information about being outside the training range but may violate assumptions of some downstream models.

## What MinMaxScaler does not do

- It does not remove outliers.
- It does not make a skewed distribution normal.
- It does not make features independent.
- It does not protect against leakage if fitted before splitting.

Because the minimum and maximum are sensitive to extreme values, StandardScaler or RobustScaler may be preferable when outliers dominate the range.

## When it is useful

Min-max scaling is useful for:

- neural networks and models expecting bounded inputs
- distance-based models such as k-NN
- algorithms where comparable feature ranges improve optimization
- polynomial feature workflows when input magnitudes should be controlled

Tree-based models usually do not need scaling.

## Constant features

If a feature has zero range on the training data, scikit-learn avoids the division by zero by using a range of 1, so the constant training value is mapped to the lower bound of `feature_range` (0 by default). Constant features are usually uninformative and should be investigated separately, for example with `VarianceThreshold`; see [Constant and Redundant Features](../03-EDA/Constant-and-Redundant-Features.md).

## MinMaxScaler versus other scalers

| Scaler | Transformation | Outlier sensitivity |
| --- | --- | --- |
| `MinMaxScaler` | Maps training min/max to a range | High |
| [`StandardScaler`](StandardScaler.md) | Centers and divides by standard deviation | Moderate |
| `RobustScaler` | Uses median and IQR | Lower |
| `MaxAbsScaler` | Divides by maximum absolute value | High |

Choose based on the model, distribution, and meaning of the feature values.

## Related pages

- [StandardScaler](StandardScaler.md)
- [Why Preprocessing Matters](Why-Preprocessing-Matters.md)
- [scikit-learn API reference: MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

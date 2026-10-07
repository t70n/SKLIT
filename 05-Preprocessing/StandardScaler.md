# StandardScaler

[Home](../README.md) / [Preprocessing](README.md)

## Why scaling matters

Some models are sensitive to the magnitude of feature values.

Examples:

- k-nearest neighbors
- logistic regression
- SVMs

These models use distances or optimization steps, so features with large numeric ranges can dominate.

## Core idea

`StandardScaler` standardizes each feature independently so that, on the training data:

- the mean is 0
- the standard deviation is 1

For a feature with training mean $\mu$ and standard deviation $\sigma$:

$$
z=\frac{x-\mu}{\sigma}
$$

The statistics are learned during `fit` and stored in `mean_` and `scale_`. `scale_` is the population standard deviation (computed with `ddof=0`). A constant feature has $\sigma=0$; scikit-learn then uses a scale of 1 to avoid a division by zero.

## Import

```python
from sklearn.preprocessing import StandardScaler
```

## Example

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The test set is transformed with the training mean and standard deviation, so its transformed values are not exactly centered. This is expected and correct.

## Keep pandas output

```python
scaler = StandardScaler().set_output(transform="pandas")
X_train_scaled = scaler.fit_transform(X_train)
```

With `set_output(transform="pandas")`, the result is a DataFrame that keeps the column names. The option is also available on pipelines and column transformers.

## In a pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
```

During cross-validation, the scaler is refitted on every training fold, which prevents information from the validation fold leaking into the preprocessing.

## Important parameters

- `with_mean=True`: subtract the mean. Set it to `False` for sparse matrices, such as the output of `OneHotEncoder`, because centering would make them dense.
- `with_std=True`: divide by the standard deviation.

## Why this helps

- faster convergence for optimization-based models
- fairer contribution of each feature in distance-based models
- regularization that penalizes all coefficients on a comparable scale
- coefficients that can be compared across features (in standard-deviation units)

## What StandardScaler does not do

- It does not make a skewed distribution normal; use a log transform, `PowerTransformer`, or `QuantileTransformer` for that.
- It is sensitive to outliers, because the mean and the standard deviation are; `RobustScaler` uses the median and the interquartile range instead.
- It does not bound the values to a fixed interval; use [MinMaxScaler](MinMaxScaler.md) for that.

## Important note

Tree-based models usually do not require scaling, because they split on thresholds and are not based on distances.

## Related pages

- [MinMaxScaler](MinMaxScaler.md)
- [Why Preprocessing Matters](Why-Preprocessing-Matters.md)
- [Pipeline](Pipeline.md)
- [scikit-learn API reference: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

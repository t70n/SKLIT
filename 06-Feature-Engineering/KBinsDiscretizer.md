# KBinsDiscretizer

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

`KBinsDiscretizer` converts a continuous numerical feature into a set of intervals, or bins. Each value is replaced by the bin to which it belongs. A linear model can then learn a different effect for each interval instead of assuming one global slope.

For example, age can be represented as ranges such as `0-25`, `25-40`, and `40+`. With one-hot encoding, the model receives one indicator column per interval.

## Import and example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

classifier = make_pipeline(
    KBinsDiscretizer(n_bins=5, encode="onehot"),
    LogisticRegression(max_iter=1000),
)
```

Keep the transformer inside the pipeline so bin boundaries are learned only from each training fold.

## How it works

For one feature, the training values are partitioned by thresholds:

$$
(-\infty,t_1], (t_1,t_2],\ldots,(t_{K-1},\infty)
$$

With `encode="onehot"`, a value belongs to exactly one bin and is represented by a one-hot vector. A linear model can therefore learn a separate weight for every interval. This produces a piecewise-constant effect.

## Binning strategies

### `strategy="uniform"`

Bins have equal width between the minimum and maximum training values. It is simple, but bins can be empty when the data is unevenly distributed.

### `strategy="quantile"`

Bin boundaries are chosen from empirical quantiles, so bins contain approximately equal numbers of training samples. This is often useful for skewed features, but repeated values can make exact equal-frequency bins impossible.

### `strategy="kmeans"`

One-dimensional k-means chooses bin centers and assigns nearby values to the same interval. It can adapt to clusters in the feature distribution.

## Important parameters

- `n_bins`: number of intervals per feature; higher values increase flexibility
- `encode`: `"onehot"`, `"onehot-dense"`, or `"ordinal"`
- `strategy`: `"uniform"`, `"quantile"`, or `"kmeans"`
- `subsample`: limits samples used to estimate thresholds for large data
- `quantile_method`: controls quantile computation in supported scikit-learn versions

`encode="ordinal"` outputs integer bin indices, but those integers can imply an artificial numeric distance. One-hot encoding is usually safer for linear models.

## Why it can help linear models

A linear model cannot solve patterns such as XOR using the original features if the relationship is not linearly separable. Binning can make some nonlinear one-dimensional effects expressible by allowing a different coefficient in each range.

However, binning is feature-wise. It transforms each feature independently and does not create products between features. It therefore cannot capture interactions such as "high $x_1$ and low $x_2$" unless interaction features are added separately.

## Limitations

- predictions can change abruptly at bin boundaries
- information is lost within each interval
- too many bins can overfit; too few can underfit
- thresholds learned from all data before cross-validation cause leakage
- it does not automatically model interactions between original features

Use [SplineTransformer](SplineTransformer.md) for smoother one-dimensional effects and [Polynomial Features](Polynomial-Features.md) for explicit powers and interactions.

## Related pages

- [SplineTransformer](SplineTransformer.md)
- [Polynomial Features](Polynomial-Features.md)
- [Feature Engineering Overview](Feature-Engineering-Overview.md)
- [scikit-learn API reference: KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)

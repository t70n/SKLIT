# Row-Wise Feature Generation

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

Row-wise features are generated independently for each observation from the values already present in that row. They include transformations, ratios, indicators, and interactions.

For a row $x=(x_1,\ldots,x_d)$, create:

$$
z_j=g_j(x)
$$

without using other rows. This is different from group or neighborhood features, which use information across observations.

## Individual transformations

### Powers and roots

```python
df["age_squared"] = df["age"] ** 2
df["income_sqrt"] = np.sqrt(df["income"].clip(lower=0))
```

Powers can represent curvature. Roots can reduce the influence of large values.

### Log transformations

```python
df["log_income"] = np.log1p(df["income"])
```

`log1p(x)` computes $\log(1+x)$ and handles zero. It requires non-negative input. Log transforms often help with positive, right-skewed measurements.

### Ratios and rates

```python
df["price_per_unit"] = df["price"] / df["units"].replace(0, np.nan)
df["conversion_rate"] = df["conversions"] / df["visits"].replace(0, np.nan)
```

Handle zero denominators explicitly and decide whether the result should become missing, zero, or a separate indicator.

### Differences and changes

```python
df["income_minus_cost"] = df["income"] - df["cost"]
df["relative_change"] = (
    df["new_value"] - df["old_value"]
) / df["old_value"].replace(0, np.nan)
```

Differences can encode a meaningful baseline relationship more directly than two raw values.

### Indicators and bins

```python
df["is_weekend"] = df["day_of_week"].isin([5, 6])
df["is_high_income"] = df["income"] > df["income"].median()
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 30, 50, np.inf],
    labels=["child", "young_adult", "adult", "senior"],
)
```

Use domain thresholds when possible. A threshold learned from the full dataset must be learned inside a training workflow when it affects evaluation.

## Pairwise interactions

Useful operations include:

$$
x+y,\quad x-y,\quad x\times y,\quad x/y
$$

```python
df["total_score"] = df["score_a"] + df["score_b"]
df["score_gap"] = df["score_a"] - df["score_b"]
df["exposure"] = df["dose"] * df["duration"]
```

An interaction $x_1x_2$ means the effect of one feature depends on the other. With a linear model:

$$
\hat{y}=w_0+w_1x_1+w_2x_2+w_3x_1x_2
$$

The marginal effect of $x_1$ is $w_1+w_3x_2$, so it changes with $x_2$.

## Automated polynomial interactions

```python
from sklearn.preprocessing import PolynomialFeatures

features = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False,
)
```

`interaction_only=True` creates products of distinct features without powers such as $x_1^2$. The number of features can grow quickly, so use regularization and cross-validation. See [Polynomial Features](Polynomial-Features.md).

## Group summaries are not row-wise

A mean, standard deviation, maximum, or percentile computed over a customer, session, time window, or neighborhood uses other rows. These are cross-row features and require special leakage checks. See [Cross-Row and Local Features](Cross-Row-and-Local-Features.md).

## Safe implementation rules

- use vectorized pandas or NumPy operations instead of row-wise loops
- handle zeros, negative values, and missing values explicitly
- preserve units and document transformations
- avoid using future or target-derived values
- put learned thresholds and transformations inside a pipeline
- test that generated features are finite and have the expected shape

```python
assert np.isfinite(df["price_per_unit"].dropna()).all()
```

## Related pages

- [Cross-Row and Local Features](Cross-Row-and-Local-Features.md)
- [Polynomial Features](Polynomial-Features.md)
- [Feature Engineering Overview](Feature-Engineering-Overview.md)

# Cross-Row and Local Features

[Home](../README.md) / [Feature Engineering](README.md)

## Idea

Cross-row features summarize other observations related to the current row. The relation can be a group, a time window, a spatial neighborhood, or the nearest observations in feature space.

These features are often powerful because they describe context, but they create a serious leakage risk. Compute only information that would have been available at prediction time.

## Group statistics

```python
summary = (
    df.groupby(["user_id", "page_id"], as_index=False)
    .agg(
        max_price=("ad_price", "max"),
        min_price=("ad_price", "min"),
        mean_price=("ad_price", "mean"),
        price_std=("ad_price", "std"),
        number_of_ads=("ad_price", "size"),
    )
)

df = df.merge(
    summary,
    on=["user_id", "page_id"],
    how="left",
    validate="many_to_one",
)
```

The resulting columns describe the group context for every row.

## Common group features

- count of events or records
- mean, median, minimum, maximum, and standard deviation
- percentiles and range
- number of unique categories
- most frequent category
- rank within the group
- difference from group mean or median
- ratio to group total

```python
grouped = df.groupby("user_id")["amount"]
df["user_mean_amount"] = grouped.transform("mean")
df["user_count"] = grouped.transform("size")
df["amount_minus_user_mean"] = (
    df["amount"] - df["user_mean_amount"]
)
```

## Prevent target leakage in group features

A group mean of the target is leakage if it includes the current row's target or validation rows. For historical prediction, use only past rows:

```python
df = df.sort_values(["user_id", "timestamp"])
df["past_user_mean"] = (
    df.groupby("user_id")["target"]
    .transform(lambda values: values.shift(1).expanding().mean())
)
```

For cross-validation, compute target-derived group features separately inside each training fold. A plain `groupby().transform("mean")` on the full dataset is not safe for target encoding.

## Time-window features

```python
df = df.sort_values(["user_id", "timestamp"])
df["previous_amount"] = (
    df.groupby("user_id")["amount"].shift(1)
)
df["past_7_mean"] = (
    df.groupby("user_id")["amount"]
    .transform(lambda values: values.shift(1).rolling(7, min_periods=1).mean())
)
```

Use `shift(1)` when the current event must not influence its own feature. For time durations, use time-based windows and check timezone and sorting behavior.

## Neighborhood features

For spatial or feature-space data, define a neighborhood and summarize it:

- nearest-neighbor mean target
- mean distance to the closest points
- number of neighbors in a radius
- local class proportions
- local mean, standard deviation, minimum, and maximum

For a point $x$, a neighborhood statistic can be written:

$$
\phi(x)=\frac{1}{|N_k(x)|}\sum_{i\in N_k(x)}g(x_i,y_i)
$$

If the statistic uses $y_i$, exclude the current row and prevent validation targets from entering the training features.

## Bray-Curtis distance

For non-negative feature vectors $u$ and $v$, Bray-Curtis distance is:

$$
 d(u,v)=\frac{\sum_j|u_j-v_j|}{\sum_j|u_j+v_j|}
$$

It emphasizes relative composition differences and is useful for some abundance or profile data. Choose a distance that matches the domain and scale the features appropriately.

## Feature generation across rows with pandas

The common pattern is:

1. group or locate neighbors
2. aggregate the contextual data
3. rename generated columns
4. merge them back using a validated key

```python
group_features = (
    df.groupby(["user_id", "page_id"], as_index=False)
    .agg(
        max_price=("ad_price", "max"),
        min_price=("ad_price", "min"),
    )
)

df = df.merge(
    group_features,
    on=["user_id", "page_id"],
    how="left",
    validate="many_to_one",
)
```

Check that the merge does not change the number of rows unexpectedly.

## Validation checklist

- Is the group or neighborhood known at prediction time?
- Are validation and test rows excluded from training-derived statistics?
- Is the current row excluded when its target or future value is used?
- Are groups large enough for stable estimates?
- Are unseen groups handled with a fallback value?
- Does the feature exist for future production records?
- Is the merge cardinality validated?
- Does cross-validation show improvement over the baseline?

## Related pages

- [Row-Wise Feature Generation](Row-Wise-Feature-Generation.md)
- [GroupBy, Joins, and Reshaping](../02-Pandas/GroupBy-Joins-and-Reshaping.md)
- [Identifiers, Leakage, Shuffling, and Sampling](../03-EDA/Identifiers-Leakage-and-Sampling.md)

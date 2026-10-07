# Identifiers, Leakage, Shuffling, and Sampling

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Identify identifier-like columns

Identifiers often have almost one unique value per row but are not meaningful predictors:

```python
identifier_like = [
    column for column in df.columns
    if df[column].nunique(dropna=False) / len(df) > 0.95
]
print(identifier_like)
```

This is only a screening rule. An identifier may encode time, location, batch, or entity structure.

Ask:

- Is it available when predictions are made?
- Does it identify the same entity across train and test?
- Does it encode the target or split membership?
- Should it be used for grouping rather than prediction?

Keep identifiers for joins, deduplication, grouped splits, and traceability when useful. Do not pass arbitrary row numbers to a model.

## Target leakage

Leakage occurs when training receives information that would not be available at prediction time. Warning signs include:

- a status recorded after the target event
- a feature derived directly from the target
- train/test membership
- duplicated entities across splits
- target-derived group statistics calculated before splitting
- future observations used in time-based features

Separate the target and split the data before learning target-dependent transformations. Use `Pipeline`, `ColumnTransformer`, and group- or time-aware validation.

## Shuffle rows

For independent, order-insensitive observations:

```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

Or:

```python
from sklearn.utils import shuffle

df = shuffle(df, random_state=42).reset_index(drop=True)
```

Shuffling changes row order, not the observations. It can prevent a file sorted by class from producing a biased split.

Do not shuffle time series, sequential events, or grouped observations before validation. Use chronological or group-aware splitting instead.

## Sample rows and columns

```python
sample_rows = df.sample(n=100, random_state=42)
sample_fraction = df.sample(frac=0.1, random_state=42)
sample_columns = df.sample(n=3, axis="columns", random_state=42)
```

Sampling is useful for visual exploration and debugging. It is not a substitute for a representative validation split.

## Train/test splitting

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["target"],
)
```

Use `stratify` for classification when class proportions should be preserved. Use group-based splitting when rows from the same person, device, or site must not occur in both sets. Use chronological splitting for future prediction.

## Permutation importance

Permutation importance is an evaluation technique, not data cleaning. It shuffles one feature in a validation set and measures the performance drop:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    fitted_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy",
)
```

Correlated features can substitute for one another, making an important feature appear unimportant when shuffled.

## Permutations and feature interactions

`itertools.permutations` generates arrangements and is rarely needed to clean a table. For feature interactions, use a documented transformer such as `PolynomialFeatures` inside a pipeline. Avoid manually generating many columns without tracking their meaning and preventing leakage.

## Related pages

- [Train/Test Split](../01-ML-Basics/Train-Test-Split.md)
- [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md)
- [Feature Selection](../06-Feature-Engineering/Feature-Selection.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)

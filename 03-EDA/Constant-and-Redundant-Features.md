# Constant and Redundant Features

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Constant features

A constant feature has the same value in every row and cannot distinguish observations:

```python
constant_columns = [
    column for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]
print(constant_columns)
```

Do not drop a constant column automatically if it is required by a schema or signals a broken data source. Check the training data and future batches.

## Near-constant features

A near-constant feature has one dominant value. Choose the threshold from the problem rather than treating `0.99` as universal:

```python
dominant_fraction = {}
for column in df.columns:
    proportions = df[column].value_counts(
        normalize=True,
        dropna=False,
    )
    dominant_fraction[column] = proportions.iloc[0]

near_constant = [
    column for column, fraction in dominant_fraction.items()
    if fraction >= 0.99
]
```

A rare indicator may be valuable even when it is near-constant, especially for safety or fraud detection.

## Variance threshold

For numeric features, scikit-learn provides a pipeline-compatible selector:

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_without_constants = selector.fit_transform(X_train)
```

Fit feature selection on training data only. A threshold chosen from the full dataset can leak information into validation.

## Duplicate columns

```python
same_columns = df.T.duplicated()
print(df.columns[same_columns].tolist())
```

Also investigate derived duplicates, such as a total that exactly reconstructs other columns, and target-derived features.

## Correlated and redundant predictors

```python
import numpy as np

corr = df.select_dtypes(include="number").corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
highly_correlated = [
    column for column in upper.columns
    if any(upper[column] > 0.95)
]
```

High correlation can indicate redundancy, but dropping a feature based only on correlation may remove useful information. Correlated features can have different missingness, interactions, costs, or behavior on future data.

## When redundancy is a problem

Redundant features can:

- make linear coefficients unstable
- increase computation and memory use
- make explanations harder
- amplify leakage risk
- create duplicate information after one-hot encoding

Tree ensembles may tolerate redundancy better than unregularized linear models, but redundant columns still complicate interpretation and data quality.

## Safer workflow

1. identify constant and duplicate columns
2. check whether columns are required by the data contract
3. measure redundancy using domain knowledge and correlation
4. select features using training folds only
5. compare validation performance and stability
6. retain a record of removed columns and reasons

## Related pages

- [Duplicates and Redundancy](Duplicates-and-Redundancy.md)
- [Correlation and Distributions](Correlation-and-Distributions.md)
- [Feature Selection](../06-Feature-Engineering/Feature-Selection.md)

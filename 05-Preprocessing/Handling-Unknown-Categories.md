# Handling Unknown Categories

[Home](../README.md) / [Preprocessing](README.md)

## Problem

At prediction time, a model may encounter a category that was not present in the training data.

Example:

- training data contains categories `A`, `B`
- test data contains category `C`

By default, scikit-learn encoders raise an error (`handle_unknown="error"`) because they do not know how to encode `C`. The same problem appears within cross-validation when a rare category is present only in a validation fold.

## OneHotEncoder: `handle_unknown="ignore"`

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

An unknown category is encoded as all zeros in the one-hot columns of that feature: the model receives no information from that feature for the sample.

## OneHotEncoder: grouping rare and unknown categories

```python
encoder = OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    min_frequency=10,
)
```

Categories seen fewer than `min_frequency` times during `fit` (an integer count, or a fraction of the samples when given as a float) are grouped into a single infrequent column. With `handle_unknown="infrequent_if_exist"`, an unknown category is mapped to that infrequent column when it exists, otherwise it is encoded as all zeros. `max_categories` gives another way to limit the number of output columns.

## OrdinalEncoder variant

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```

This maps unknown categories to a fixed value such as `-1`. The value must not collide with the codes of known categories (`0` to `n_categories - 1`); scikit-learn raises an error if it does. `unknown_value=np.nan` is also allowed, which lets estimators with native missing-value support treat the unknown category as missing. Missing values present during `fit` are encoded with `encoded_missing_value` (`np.nan` by default).

## Native categorical support

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` support categorical features natively through `categorical_features`; categories unseen during training are treated like missing values. CatBoost also handles categorical columns natively. See [HistGradientBoostingClassifier](../07-Models/Ensembles/HistGradientBoostingClassifier.md) and [CatBoost](../07-Models/Ensembles/CatBoost.md).

## Why this matters

Without handling unknown categories, you risk runtime errors or incorrect predictions when a deployment dataset contains unseen categories. Monitor how often unknown categories appear in production: a sudden increase can signal a change in the data source.

## Related pages

- [OrdinalEncoder vs OneHotEncoder](OrdinalEncoder-vs-OneHotEncoder.md)
- [ColumnTransformer](ColumnTransformer.md)
- [scikit-learn user guide: Infrequent categories](https://scikit-learn.org/stable/modules/preprocessing.html#infrequent-categories)

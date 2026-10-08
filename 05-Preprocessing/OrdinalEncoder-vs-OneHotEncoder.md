# OrdinalEncoder vs OneHotEncoder

[Home](../README.md) / [Preprocessing](README.md)

## Data type issue

Categorical features may be stored as strings such as:

```python
["male", "female", "male"]
```

Most scikit-learn models need numeric input, so we convert them.

## OrdinalEncoder

### Idea

Each category is mapped to an integer value.

Example:

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
encoded = encoder.fit_transform(df[["education"]])
```

By default, categories are sorted alphabetically and numbered `0, 1, 2, ...`. The learned mapping is stored in `encoder.categories_`.

### Explicit order for ordinal features

When the categories have a natural order, state it explicitly so that the codes follow the domain order rather than the alphabetical one:

```python
encoder = OrdinalEncoder(
    categories=[["Preschool", "HS-grad", "Bachelors", "Masters", "Doctorate"]],
)
```

`categories` takes one list per encoded column.

### Advantage

- compact representation (one column per feature)
- works well for tree-based models
- easy to use

### Risk

It introduces an order between categories, which may be artificial.

This can be misleading for linear models and models that assume ordered numeric values: a linear model would assume that the effect of category `2` is twice the effect of category `1`.

## OneHotEncoder

### Idea

Each category becomes a binary column.

Example:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["education"]])
```

`sparse_output=False` returns a dense array; the default sparse matrix is more memory-efficient when there are many categories. The generated column names are available through `encoder.get_feature_names_out()`.

### Useful parameters

- `handle_unknown="ignore"`: encode a category unseen during `fit` as all zeros instead of raising an error; see [Handling Unknown Categories](Handling-Unknown-Categories.md)
- `min_frequency` and `max_categories`: group rare categories into a single "infrequent" column
- `drop="if_binary"`: keep a single column for binary features; `drop="first"` removes one category per feature, which avoids perfect collinearity for unregularized linear models

### Advantage

- no artificial ordering assumed between categories
- often the right choice for linear models

### Risk

- creates many columns
- can be inefficient for high-cardinality features
- may be slow for tree-based models

## TargetEncoder for high-cardinality features

For categorical features with many levels (postal codes, product identifiers), one-hot encoding creates too many columns and ordinal encoding imposes an arbitrary order. `TargetEncoder` (scikit-learn 1.3 or later) replaces each category with a smoothed average of the target for that category:

```python
from sklearn.preprocessing import TargetEncoder

encoder = TargetEncoder(target_type="binary")
```

Because the encoding uses the target, `fit_transform` relies on internal cross-fitting to avoid leakage: the training data is encoded with statistics computed on other folds. Always keep it inside a pipeline evaluated with cross-validation.

## Rule of thumb

| Situation | Recommended encoder |
| --- | --- |
| Linear models, SVMs, k-nearest neighbors, nominal categories | `OneHotEncoder(handle_unknown="ignore")` |
| Tree-based models, low or moderate cardinality | `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` |
| Truly ordered categories | `OrdinalEncoder(categories=[...])` with the domain order |
| High-cardinality categories | `TargetEncoder`, or `OneHotEncoder` with `min_frequency` or `max_categories` |
| `HistGradientBoosting` models | Native categorical support (`categorical_features`) or `OrdinalEncoder` |

## Note on tree-based models

Tree-based models can often work well with integer-coded categories, so one-hot encoding is not always necessary. Each split still uses the order of the codes (a threshold groups contiguous codes), but successive splits can isolate any individual category. An arbitrary order is therefore much less harmful for trees than for linear models, and the compact encoding keeps training fast.

A complete comparison of a linear model and a decision tree on the Ames housing data, including ordinal encoding of the categorical features, is given in [Ames Housing: Linear Model vs Decision Tree](../10-Recipes/Ames-Housing-Linear-vs-Tree.md).

## Related pages

- [Handling Unknown Categories](Handling-Unknown-Categories.md)
- [ColumnTransformer](ColumnTransformer.md)
- [Data Types, Column Names, and Categories](../03-EDA/Data-Types-and-Categories.md)
- [scikit-learn user guide: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)

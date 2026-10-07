# Preprocessing

[Home](../README.md)

This section explains how to turn raw columns into a numerical matrix that a model can use: scaling numerical features, encoding categorical features, and combining these steps with the model in a single pipeline, fitted on training data only.

## Pages

### Principles

| Page | Summary |
| --- | --- |
| [Why Preprocessing Matters](Why-Preprocessing-Matters.md) | Typical data issues and the scikit-learn tool for each of them |

### Scaling

| Page | Summary |
| --- | --- |
| [StandardScaler](StandardScaler.md) | Centering and scaling to unit variance; when scaling matters |
| [MinMaxScaler](MinMaxScaler.md) | Rescaling to a fixed interval; comparison of scalers |

### Categorical encoding

| Page | Summary |
| --- | --- |
| [OrdinalEncoder vs OneHotEncoder](OrdinalEncoder-vs-OneHotEncoder.md) | Integer codes or binary columns, explicit category order, `TargetEncoder`, rule of thumb |
| [Handling Unknown Categories](Handling-Unknown-Categories.md) | Categories unseen during training, infrequent categories, native support |

### Composition

| Page | Summary |
| --- | --- |
| [ColumnTransformer](ColumnTransformer.md) | Different transformations per column subset, column selectors, `remainder` |
| [Pipeline](Pipeline.md) | Chaining preprocessing and a model; step names, inspection, leakage prevention |

## Suggested reading order

Read [Why Preprocessing Matters](Why-Preprocessing-Matters.md) first, then [Pipeline](Pipeline.md) and [ColumnTransformer](ColumnTransformer.md): every other preprocessing step should be used inside them.

## Navigation

- Previous section: [Visualization](../04-Visualization/README.md)
- Next section: [Feature Engineering](../06-Feature-Engineering/README.md)
- [Back to the home page](../README.md)

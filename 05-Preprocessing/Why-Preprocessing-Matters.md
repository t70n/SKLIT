# Why Preprocessing Matters

[Home](../README.md) / [Preprocessing](README.md)

## Basic idea

Raw data is rarely directly usable by machine learning models.

Typical issues:

- features have different scales
- categorical variables are strings
- missing values need handling
- unknown categories may appear in test data
- some features should be transformed or normalized

## Common preprocessing operations

| Issue | Typical scikit-learn tool | Page |
| --- | --- | --- |
| Features on different scales | `StandardScaler`, `MinMaxScaler`, `RobustScaler` | [StandardScaler](StandardScaler.md), [MinMaxScaler](MinMaxScaler.md) |
| Categorical strings | `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder` | [OrdinalEncoder vs OneHotEncoder](OrdinalEncoder-vs-OneHotEncoder.md) |
| Categories unseen during training | `handle_unknown` options of the encoders | [Handling Unknown Categories](Handling-Unknown-Categories.md) |
| Missing values | `SimpleImputer`, `KNNImputer`, `IterativeImputer` | [Missing Values](../03-EDA/Missing-Values.md) |
| Different processing per column type | `ColumnTransformer` | [ColumnTransformer](ColumnTransformer.md) |
| Chaining preprocessing and a model | `Pipeline`, `make_pipeline` | [Pipeline](Pipeline.md) |
| Skewed features or targets | `QuantileTransformer`, `PowerTransformer`, `TransformedTargetRegressor` | [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md#transforming-the-target) |
| Nonlinear relationships | `PolynomialFeatures`, `SplineTransformer`, `KBinsDiscretizer` | [Feature Engineering](../06-Feature-Engineering/README.md) |

## Why scaling is useful

A feature like `age` measured in years and a feature like `income` measured in thousands may have very different magnitudes. Without scaling, the larger-magnitude feature can dominate optimization or distance computations.

Scaling matters for distance-based models (k-nearest neighbors, SVMs with RBF kernels), for gradient-based optimization (logistic regression, neural networks), and for regularized models, whose penalty depends on coefficient magnitudes. Tree-based models are insensitive to monotonic feature transformations and do not need scaling.

## Why encoding is useful

Machine learning models usually expect numeric input. Categorical labels such as `"red"`, `"blue"`, or `"male"` must be converted into numbers in a mathematically meaningful way.

## Key concept: transformations

In scikit-learn, preprocessing is done via transformers with:

- `fit`: learn the statistics (means, categories, quantiles) from the training data
- `transform`: apply the learned transformation to any dataset
- `fit_transform`: both steps at once, on the training data

A transformer learns from training data and then applies the same transformation to new data.

## Good habit

Always preprocess using information learned only from the training data, not from the test data. The simplest way to guarantee this is to put every preprocessing step in a [Pipeline](Pipeline.md) and to pass the whole pipeline to cross-validation.

## Related pages

- [StandardScaler](StandardScaler.md)
- [OrdinalEncoder vs OneHotEncoder](OrdinalEncoder-vs-OneHotEncoder.md)
- [ColumnTransformer](ColumnTransformer.md)
- [Pipeline](Pipeline.md)
- [scikit-learn user guide: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)

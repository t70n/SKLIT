# Machine Learning Concepts

[Home](../README.md) / [Machine Learning Basics](README.md)

## What is machine learning?

Machine learning is the process of learning patterns from data so that a model can make predictions on, or extract structure from, new data it has not seen before.

We usually work with:

- `X`: the feature matrix, of shape `(n_samples, n_features)`
- `y`: the target vector, with one value per sample

## Data matrix interpretation

Each row is a sample (also called an observation or instance). Each column is a feature (also called a variable or predictor).

If there are `n` rows and `m` columns, the data matrix has:

- `n_samples = n`
- `n_features = m`

## Features, targets, and samples

- A feature is one column or variable describing a sample.
- The target is the quantity to predict.
- A sample is one row of the dataset.

Example with the Adult Census dataset:

```python
data = adult_census.drop(columns="class")
target = adult_census["class"]
```

A dataset can therefore be seen as:

- input features: `X` (here `data`)
- output target: `y` (here `target`)

## Supervised learning

In supervised learning, the target `y` is known for the training samples.

Two main cases:

- Classification: `y` contains discrete labels
- Regression: `y` contains continuous numerical values

Examples:

- predict whether a person earns more than a given income threshold (classification)
- predict the price of a house (regression)
- predict the species of an iris flower (classification)

## Unsupervised learning

In unsupervised learning, there is no target `y`.

The objective is to find structure in the data, such as:

- clusters
- hidden patterns
- lower-dimensional representations (dimensionality reduction)
- unusual observations (anomaly detection)

## Parameters versus hyperparameters

- **Parameters** are learned from the data during `fit`, for example the coefficients of a linear model or the split thresholds of a decision tree.
- **Hyperparameters** are chosen before training and control the learning process, for example `C` in logistic regression, `max_depth` in a decision tree, or `n_neighbors` in k-nearest neighbors.

Hyperparameters are passed to the constructor and can be inspected with `get_params()` and modified with `set_params()`. Their values are usually selected with cross-validation; see [Hyperparameter Tuning](../09-Hyperparameter-Tuning/README.md).

## The scikit-learn estimator API

In scikit-learn, every learning object follows the same interface:

| Method or attribute | Purpose | Available on |
| --- | --- | --- |
| `fit(X, y)` | Learn parameters from the training data | All estimators |
| `predict(X)` | Predict a label or a value for new samples | Predictors (classifiers, regressors) |
| `predict_proba(X)` | Estimate class probabilities | Most classifiers |
| `decision_function(X)` | Return a continuous confidence score | Some classifiers (linear models, SVMs) |
| `transform(X)` | Convert data into a new representation | Transformers (scalers, encoders) |
| `fit_transform(X)` | Fit and transform in one step | Transformers |
| `score(X, y)` | Compute a default metric | Predictors (accuracy for classifiers, R² for regressors) |

Fitted attributes end with a trailing underscore, such as `coef_`, `classes_`, or `feature_importances_`. They exist only after `fit` has been called.

Examples:

- `KNeighborsClassifier`, `LogisticRegression`: predictors
- `StandardScaler`, `OneHotEncoder`: transformers
- `Pipeline`: chains transformers and a final estimator into a single estimator

## Training and generalization

The model must be trained on a training set, then evaluated on data that was not used for training.

This avoids a misleading evaluation in which the model memorizes the data instead of learning a rule.

The key quantity is the generalization performance: how well the model predicts on new data. It is estimated with a held-out test set or, more reliably, with cross-validation.

## Important caution: class imbalance

If one class is much more frequent than another, accuracy alone can be misleading. A model can look good by always predicting the majority class.

This is why you should inspect the class distribution before modeling:

```python
target.value_counts(normalize=True)
```

Compare every model with a dummy baseline; see [DummyClassifier](../07-Models/Baselines/DummyClassifier.md) and [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md).

## Typical machine learning workflow

1. Load the data ([Pandas](../02-Pandas/README.md))
2. Explore, clean, and visualize it ([EDA](../03-EDA/README.md), [Visualization](../04-Visualization/README.md))
3. Split into training and test sets ([Train/Test Split](Train-Test-Split.md))
4. Build preprocessing and a model inside a pipeline ([Preprocessing](../05-Preprocessing/README.md), [Models](../07-Models/README.md))
5. Compare with a baseline and evaluate with cross-validation ([Model Evaluation](../08-Model-Evaluation/README.md))
6. Tune hyperparameters ([Hyperparameter Tuning](../09-Hyperparameter-Tuning/README.md))
7. Evaluate the final model once on the untouched test set
8. Iterate if necessary

## Useful reminder

The goal is not to memorize the training set, but to learn a pattern that transfers to new observations.

## Related pages

- [Train/Test Split](Train-Test-Split.md)
- [Overfitting vs Underfitting](Overfitting-vs-Underfitting.md)
- [Glossary](../11-Glossary/README.md)
- [scikit-learn user guide: Getting started](https://scikit-learn.org/stable/getting_started.html)

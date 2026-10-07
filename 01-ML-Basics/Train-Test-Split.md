# Train/Test Split

[Home](../README.md) / [Machine Learning Basics](README.md)

## Why it matters

If you train and evaluate the model on the same data, you often get a falsely optimistic result. The model may simply memorize the training samples instead of learning a general rule.

This is why we separate the dataset into:

- training set: used to fit the model
- test set: used to estimate generalization performance

## Basic usage

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.25
)
```

This creates a split where:

- 75% of the data is used for training
- 25% is used for testing

`test_size=0.25` is also the default value when neither `test_size` nor `train_size` is given.

## Stratified split for classification

With imbalanced classes, a purely random split can produce a test set whose class proportions differ from the full dataset. Use `stratify` to preserve them:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
```

## Why randomness matters

The split is randomized (`shuffle=True` by default), but `random_state` ensures the same split each time. This is useful for comparing experiments and reproducing results.

Do not shuffle time-ordered data: for forecasting problems use `shuffle=False` or a time-aware splitter such as `TimeSeriesSplit`. Rows that belong to the same group (patient, customer, writer) should not be spread across training and test sets. See [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md).

## Example interpretation

```python
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
```

## Model evaluation after split

```python
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)
```

This gives the test accuracy for a classifier or the R² score for a regressor, because `score` uses the default metric of the estimator. Use an explicit metric when another criterion matters; see [Metrics and Scoring](../08-Model-Evaluation/Metrics-and-Scoring.md).

## Training, validation, and test sets

When hyperparameters or preprocessing choices are selected, a third role appears:

| Set | Used for | Rule |
| --- | --- | --- |
| Training set | Fitting model parameters | Can be reused freely |
| Validation set (or inner cross-validation) | Comparing models and selecting hyperparameters | Its score becomes optimistic once used for selection |
| Test set | Final estimate of generalization | Used once, at the very end |

In practice, the validation role is usually played by cross-validation on the training set, for example inside `GridSearchCV`. See [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md).

## Recommended practice

- Keep the test set untouched until the end
- Fit every preprocessing step on the training set only, ideally inside a `Pipeline`
- Use cross-validation for more robust estimates
- Set `random_state` for reproducibility
- Use `stratify=y` for classification

## Main limitation

A single train/test split can be unstable. A different random split may give a noticeably different performance, especially on small datasets. This is why cross-validation is often preferred.

## Rule of thumb

- Use a single split for a quick baseline
- Use cross-validation for a more reliable estimate
- Keep the final test set for the final model evaluation

## When to use it

Use a train/test split when:

- you want a quick benchmark
- you need a simple evaluation framework
- you are developing a first baseline model
- you need a final held-out set after model selection

## Related pages

- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)
- [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md)
- [Identifiers, Leakage, Shuffling, and Sampling](../03-EDA/Identifiers-Leakage-and-Sampling.md)
- [scikit-learn API reference: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

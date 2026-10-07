# DummyClassifier

[Home](../../README.md) / [Models](../README.md) / [Baselines](../README.md#baselines)

## Idea

`DummyClassifier` makes predictions that ignore the input features. It applies a simple rule, such as always predicting the most frequent class, and therefore learns nothing about the relationship between `X` and `y`.

It is not meant to be deployed. It provides the reference score that any useful classifier must beat: if a model does not clearly outperform a dummy baseline, it has not learned anything useful, regardless of how high its accuracy looks.

## Import

```python
from sklearn.dummy import DummyClassifier
```

## Minimal example

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print(f"Accuracy of the dummy classifier: {dummy.score(X_test, y_test):.3f}")
```

`fit` only records the class distribution of `y_train`; the features are ignored. `score` returns the accuracy, like any classifier.

## Strategies

| `strategy` | Prediction rule | `predict_proba` |
| --- | --- | --- |
| `"prior"` (default) | Always the most frequent class | The class distribution of the training set |
| `"most_frequent"` | Always the most frequent class | 1 for the most frequent class, 0 otherwise |
| `"stratified"` | Random draws following the training class distribution | One-hot vector of the random draw |
| `"uniform"` | Random draws with equal probability for each class | Uniform probabilities |
| `"constant"` | Always the class given by `constant` | 1 for that class, 0 otherwise |

The random strategies (`"stratified"` and `"uniform"`) use `random_state` for reproducibility. The `"constant"` strategy is useful to predict a minority class of interest, for example `DummyClassifier(strategy="constant", constant=">50K")`.

## Expected scores of each strategy

For a binary problem where the positive class has proportion $p$ and the majority class has proportion $1-p$ (with $p<0.5$), the expected scores are:

| Strategy | Expected accuracy | Expected balanced accuracy |
| --- | --- | --- |
| `"most_frequent"` / `"prior"` | $1-p$ | $0.5$ |
| `"stratified"` | $p^2+(1-p)^2$ | $0.5$ |
| `"uniform"` | $0.5$ | $0.5$ |
| `"constant"` (minority class) | $p$ | $0.5$ |

More generally, with $K$ classes every dummy strategy has an expected balanced accuracy of $1/K$, the chance level. This is one reason why balanced accuracy is easier to interpret than accuracy on imbalanced data: its chance level does not depend on the class proportions.

For example, with 76% of negative samples, always predicting the negative class gives an accuracy of about 0.76 without learning anything. A classifier with an accuracy of 0.77 on the same data is barely better than this baseline.

## Use it with cross-validation

Evaluate the dummy baseline with exactly the same splits and metric as the real model:

```python
from sklearn.model_selection import ShuffleSplit, cross_validate

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
dummy_results = cross_validate(
    DummyClassifier(strategy="most_frequent"),
    data,
    target,
    cv=cv,
    scoring=["accuracy", "balanced_accuracy"],
)
print(dummy_results["test_accuracy"].mean())
print(dummy_results["test_balanced_accuracy"].mean())
```

Comparing the distributions of fold scores, rather than only their means, shows whether the improvement of the real model is consistent. A complete comparison of all strategies is given in [Dummy Classifier Baselines](../../10-Recipes/Dummy-Classifier-Baselines.md).

## Preprocessing is not needed

Because the features are ignored, scaling or encoding has no effect on a dummy classifier. Wrapping it in the same pipeline as the real model is harmless and can be convenient when the pipeline handles column selection, but it is not required.

## When to use it

- as the first model of every classification project
- to quantify how much of a score is explained by class imbalance alone
- to detect leakage or bugs: a real model that scores like a dummy has probably not learned anything, and a dummy that scores surprisingly well reveals an imbalanced or trivial target
- to obtain the chance level of a metric on a specific dataset

## Limitations

- it only gives a lower reference point, not a strong baseline; a simple linear model is the next natural baseline
- the random strategies produce scores that vary between runs and folds

## Related pages

- [DummyRegressor](DummyRegressor.md)
- [Classification Metrics](../../08-Model-Evaluation/Classification-Metrics.md)
- [Dummy Classifier Baselines](../../10-Recipes/Dummy-Classifier-Baselines.md)
- [scikit-learn API reference: DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)

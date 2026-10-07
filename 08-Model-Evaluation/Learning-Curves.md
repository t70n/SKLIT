# Learning Curves

[Home](../README.md) / [Model Evaluation](README.md)

## Purpose

Learning curves show how model performance changes as the size of the training set increases. For each training size, the model is fitted several times with cross-validation, and both the training score and the test score are recorded.

They help answer:

- does the model benefit from more data?
- is it underfitting or overfitting?
- is performance plateauing?

## Typical usage

```python
import numpy as np
from sklearn.model_selection import LearningCurveDisplay

train_sizes = np.linspace(0.1, 1.0, num=30, endpoint=True)

display = LearningCurveDisplay.from_estimator(
    model,
    X,
    y,
    train_sizes=train_sizes,
    cv=cv,
    score_type="both",
    scoring="accuracy",
    n_jobs=2,
)
```

`train_sizes` given as fractions are relative to the size of the training folds. `score_type="both"` plots the training and the test scores. For an error metric such as `scoring="neg_mean_absolute_error"`, set `negate_score=True` to display positive errors. `LearningCurveDisplay` is available since scikit-learn 1.2.

## Computing the values without plotting

```python
from sklearn.model_selection import learning_curve

train_sizes_abs, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=train_sizes, cv=cv, scoring="accuracy", n_jobs=2,
)
print(train_scores.mean(axis=1), test_scores.mean(axis=1))
```

`train_scores` and `test_scores` have one row per training size and one column per cross-validation split.

## Interpretation

- the training score is usually high, especially with small training sets that are easy to memorize
- the test score usually improves as more data is added
- a large gap between training and test scores indicates overfitting (high variance)
- as more data is added, the gap may shrink
- a plateau of both curves indicates that more data will not help much unless the model or the features change

## Diagnosing with learning curves

| Pattern | Diagnosis | What may help |
| --- | --- | --- |
| Both scores low and close to each other | Underfitting (high bias) | A more flexible model, better features, less regularization; more data will not help |
| High training score, much lower test score, gap shrinking with more data | Overfitting (high variance) | More data, regularization, a simpler model |
| Both scores high and converged | Good fit | Additional data brings diminishing returns |
| Test score still increasing at the largest size | The model is data-limited | Collecting more data is likely to help |

The shaded bands or error bars show the variability across cross-validation splits; a wide band means that the estimate depends strongly on the split.

## Cost

A learning curve fits the model `len(train_sizes) * n_splits` times. Reduce the number of training sizes or splits, or use `n_jobs`, for expensive models.

## Related pages

- [Validation Curves](Validation-Curves.md)
- [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md)
- [Cross-Validation](Cross-Validation.md)
- [scikit-learn user guide: Validation curves and learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)

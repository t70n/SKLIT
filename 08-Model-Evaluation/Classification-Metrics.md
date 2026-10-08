# Classification Metrics

[Home](../README.md) / [Model Evaluation](README.md)

## What these scores measure

Classification metrics compare the true labels with the model's predicted labels or predicted scores. They answer different questions, so there is no universally best metric.

Two families exist:

- **label-based metrics** (accuracy, balanced accuracy, precision, recall, F1) use hard predictions, obtained by applying a decision threshold to the scores
- **score-based metrics** (ROC-AUC, average precision, log loss, Brier score) use `predict_proba` or `decision_function` and evaluate the ranking or the quality of probabilities over all thresholds

For the mechanics of scorers and the `scoring` parameter, see [Metrics and Scoring Overview](Metrics-and-Scoring.md).

## Start with a baseline

Before interpreting any score, compute the score of a classifier that ignores the features:

```python
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(data_train, target_train)
print(f"Accuracy of the dummy classifier: {dummy_classifier.score(data_test, target_test):.3f}")
```

On the blood transfusion dataset, where about 76% of the people did not give blood during the target period, this dummy classifier always predicts `"not donated"` and reaches an accuracy of about 0.76 without learning anything, which is as accurate as a logistic regression trained on the same data. This is the class imbalance problem: accuracy alone cannot reveal that the model is useless for the minority class. See [DummyClassifier](../07-Models/Baselines/DummyClassifier.md).

## The confusion matrix

For binary classification, choose one class as positive. Every prediction falls into one of four groups:

|  | Predicted positive | Predicted negative |
| --- | ---: | ---: |
| Actually positive | True positive (TP) | False negative (FN) |
| Actually negative | False positive (FP) | True negative (TN) |

- TP: correctly found a positive case
- TN: correctly rejected a negative case
- FP: a false alarm
- FN: a missed positive case

Most classification metrics are calculated from these four counts. The counts are more informative than one score alone because they show which type of mistake the model makes.

### Layout used by scikit-learn

`confusion_matrix(y_true, y_pred)` puts the true classes in rows and the predicted classes in columns, both in sorted label order. With labels `0` and `1`, the matrix is therefore:

$$
\begin{bmatrix}
TN & FP\\
FN & TP
\end{bmatrix}
$$

```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```

The position of TP depends on the label order: with the string labels `"donated"` and `"not donated"`, `"donated"` comes first and the true positives are in the top-left cell. Pass `labels=[...]` to fix the order explicitly.

### Plot the confusion matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

_ = ConfusionMatrixDisplay.from_estimator(classifier, data_test, target_test)
```

`ConfusionMatrixDisplay.from_predictions(y_true, y_pred)` works from stored predictions, and `normalize="true"` shows the proportions within each true class (the per-class recalls on the diagonal).

## Accuracy

Accuracy is the fraction of all predictions that are correct:

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

```python
import numpy as np
from sklearn.metrics import accuracy_score

np.mean(target_test == target_predicted)               # manual computation
accuracy_score(target_test, target_predicted)           # same value
classifier.score(data_test, target_test)                # default score of classifiers
```

Accuracy is an aggregate of the errors made by the classifier: it does not tell which classes are confused.

### When accuracy is useful

Accuracy is useful when:

- false positives and false negatives have similar costs
- the classes are reasonably balanced
- every example matters equally

### Main limitation

Accuracy can be misleading with imbalanced classes. If 95% of cases are negative, a model that always predicts negative has 95% accuracy but finds no positive cases. Always compare accuracy with the class distribution, a dummy baseline, and a confusion matrix.

## Balanced accuracy

Balanced accuracy is the average of the recall obtained on each class:

$$
\text{Balanced accuracy}=\frac{1}{K}\sum_{k=1}^{K}\operatorname{Recall}_k
$$

For binary classification, it is the mean of the sensitivity and the specificity:

$$
\text{Balanced accuracy}=\frac{1}{2}\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)
$$

```python
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy = balanced_accuracy_score(target_test, target_predicted)
print(f"Balanced accuracy: {balanced_accuracy:.3f}")
```

Properties:

- it equals the accuracy when the classes are balanced
- every dummy classifier obtains about $1/K$ (0.5 for binary problems), whatever the class proportions, so the chance level is easy to read
- a large gap between accuracy and balanced accuracy indicates a model biased toward the majority class

In the blood transfusion example, a logistic regression with an accuracy close to the dummy baseline obtains a balanced accuracy of only about 0.55: it barely detects the donors.

## Precision

Precision answers: among the examples predicted positive, how many really were positive?

$$
\operatorname{Precision}=\frac{TP}{TP+FP}
$$

It represents how likely a positive prediction is to be correct; it is also called the positive predictive value (PPV).

Precision is important when false alarms are expensive. Examples include sending a costly human investigation, blocking a legitimate transaction, or contacting a customer unnecessarily.

Precision is undefined if the model predicts no positive examples. scikit-learn can control this behavior with the `zero_division` argument.

## Recall (sensitivity)

Recall answers: among all truly positive examples, how many did the model find?

$$
\operatorname{Recall}=\frac{TP}{TP+FN}
$$

Recall is important when false negatives are expensive. Examples include detecting disease, identifying safety failures, or finding fraudulent transactions that should be investigated.

Recall is also called sensitivity or the true positive rate (TPR):

$$
\operatorname{TPR}=\frac{TP}{TP+FN}
$$

The false negative rate is $1-\operatorname{Recall}$.

```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(target_test, target_predicted, pos_label="donated")
recall = recall_score(target_test, target_predicted, pos_label="donated")
```

## Specificity and false positive rate

Specificity (the true negative rate) answers: among all truly negative examples, how many were correctly rejected?

$$
\operatorname{Specificity}=\operatorname{TNR}=\frac{TN}{TN+FP}
\qquad
\operatorname{FPR}=\frac{FP}{FP+TN}=1-\operatorname{Specificity}
$$

Sensitivity and specificity are the standard pair in medical screening. scikit-learn has no dedicated function: specificity is the recall of the negative class, `recall_score(y_true, y_pred, pos_label=<negative label>)`.

## Precision-recall trade-off and the decision threshold

For a probabilistic classifier, `predict` returns the positive class when its predicted probability exceeds a threshold, $0.5$ by default. Increasing the threshold requires more evidence before predicting positive. This usually:

- increases precision because fewer borderline cases are called positive
- decreases recall because more actual positives are missed

Decreasing the threshold reverses the pattern. The default $0.5$ implicitly gives equal importance to both kinds of errors; real applications rarely do:

| Goal | Threshold direction | Example |
| --- | --- | --- |
| Minimize false positives | Higher threshold (for example 0.7) | Spam filtering: legitimate emails must not be blocked |
| Capture as many positives as possible | Lower threshold (for example 0.3) | Medical screening: a missed disease is very costly |

The threshold should be chosen from the real cost of FP versus FN, on validation data:

```python
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV

# Apply a chosen threshold to the probability of the "donated" class
model_03 = FixedThresholdClassifier(classifier, threshold=0.3, pos_label="donated")

# Learn the threshold that maximizes a metric by internal cross-validation
tuned = TunedThresholdClassifierCV(classifier, scoring="balanced_accuracy")
tuned.fit(data_train, target_train)
print(tuned.best_threshold_)
```

`FixedThresholdClassifier` and `TunedThresholdClassifierCV` are available since scikit-learn 1.5. With string labels, set `pos_label`: by default, the threshold applies to the probability of `classifier.classes_[1]`, the second label in sorted order, which is `"not donated"` here. The `best_threshold_` of `TunedThresholdClassifierCV` also refers to `classes_[1]`; to optimize a metric for a specific class, pass a scorer built with `make_scorer(..., pos_label=...)`.

## F1-score

F1-score combines precision and recall using their harmonic mean:

$$
F_1=2\frac{\operatorname{Precision}\cdot\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}
$$

The harmonic mean is pulled toward the smaller value, so F1 is high only when both precision and recall are reasonably high. F1 is useful when positive-class performance matters and you want one threshold-dependent balance between false alarms and missed positives.

### When F1 is not enough

F1 ignores true negatives and treats precision and recall as equally important. It may be a poor choice when:

- the negative class is also important
- false positives and false negatives have different costs
- predicted probabilities need to be well calibrated
- the class prevalence changes substantially

Use an $F_\beta$ score when recall or precision should receive more weight:

$$
F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}
$$

Here $P$ is precision and $R$ is recall. $\beta>1$ emphasizes recall; $\beta<1$ emphasizes precision.

## Matthews correlation coefficient

The Matthews correlation coefficient (MCC) uses all four cells of the confusion matrix:

$$
\operatorname{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

It ranges from $-1$ (total disagreement) through $0$ (chance level) to $1$ (perfect prediction) and remains informative on imbalanced data. It is available as `matthews_corrcoef` and through `scoring="matthews_corrcoef"`.

## ROC curve and ROC-AUC

A classifier often produces a score $s(x)$ rather than only a class label. The ROC curve (receiver operating characteristic) evaluates every possible threshold. For each threshold, plot:

- true positive rate: $\operatorname{TPR}=TP/(TP+FN)$
- false positive rate: $\operatorname{FPR}=FP/(FP+TN)$

The ROC curve is the collection of $(\operatorname{FPR},\operatorname{TPR})$ points as the threshold changes.

ROC-AUC is the area under this curve:

$$
\operatorname{AUC}=\int_0^1 \operatorname{TPR}(\operatorname{FPR})\,d(\operatorname{FPR})
$$

A useful probabilistic interpretation is:

$$
\operatorname{AUC}=P(s(x^+)>s(x^-))
$$

It is the probability that a randomly selected positive example receives a higher score than a randomly selected negative example. It measures ranking quality, not one fixed-threshold accuracy.

A random ranking has AUC about `0.5`; perfect ranking has AUC `1.0`; values below `0.5` mean the ranking is worse than random and may indicate reversed labels or scores. The full range is therefore $[0, 1]$, with $0.5$ as the chance level.

```python
from sklearn.metrics import RocCurveDisplay

disp = RocCurveDisplay.from_estimator(
    classifier, data_test, target_test,
    pos_label="donated", plot_chance_level=True,
)
_ = disp.ax_.set_title("Receiver Operating Characteristic curve")
```

`plot_chance_level=True` (scikit-learn 1.3 or later) draws the diagonal obtained by a classifier that ranks samples at random; a dummy classifier lies on this diagonal.

## Precision-recall curve and average precision

The precision-recall curve plots precision against recall for every threshold. It focuses on the positive class and ignores true negatives.

```python
from sklearn.metrics import PrecisionRecallDisplay

disp = PrecisionRecallDisplay.from_estimator(
    classifier, data_test, target_test,
    pos_label="donated", plot_chance_level=True,
)
_ = disp.ax_.set_title("Precision-recall curve")
```

The chance level is a horizontal line at the prevalence of the positive class: a random classifier has a precision equal to the proportion of positives at every recall. The curve is summarized by the **average precision** (AP), a weighted mean of the precisions at each threshold, often called PR-AUC:

$$
\operatorname{AP}=\sum_n (R_n-R_{n-1})P_n
$$

It is available as `average_precision_score(y_true, scores)` and `scoring="average_precision"`. A perfect classifier has $\operatorname{AP}=1$; a random one has an AP close to the positive prevalence.

## ROC-AUC versus average precision

| Aspect | ROC-AUC | Average precision (PR-AUC) |
| --- | --- | --- |
| Uses true negatives | Yes (through the FPR) | No |
| Chance level | 0.5, whatever the prevalence | The positive prevalence |
| Sensitivity to class imbalance | Can look optimistic when negatives are abundant | Reflects the difficulty of finding rare positives |
| Typical use | Balanced classes, overall ranking quality | Rare positive class: fraud, rare diseases, defects |

Both are threshold-independent. Neither replaces the evaluation of the operating point that will actually be deployed.

## Probabilistic metrics and calibration

When predicted probabilities drive decisions (expected costs, risk scores), evaluate their quality directly:

- **log loss** (`log_loss`, `scoring="neg_log_loss"`): heavily penalizes confident wrong predictions
- **Brier score** (`brier_score_loss`, `scoring="neg_brier_score"`): mean squared difference between the predicted probability and the outcome
- **calibration curve** (`CalibrationDisplay`): compares predicted probabilities with observed frequencies

A model can rank well (high ROC-AUC) and still produce poorly calibrated probabilities. `CalibratedClassifierCV` can recalibrate a classifier.

## Calculating the metrics in scikit-learn

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, pred))
```

Use predicted probabilities or decision scores for ROC-AUC, not hard class predictions. With hard predictions, there is only one operating point and the result is not a meaningful full ROC-AUC evaluation. The column `[:, 1]` of `predict_proba` corresponds to `model.classes_[1]`, the second label in sorted order.

## Positive label with string targets

The metric functions assume that label `1` is the positive class by default. Set `pos_label` when another label is positive, for example `precision_score(y_test, pred, pos_label="donated")`.

The string scorers used in cross-validation make the same assumption. With string labels, `scoring="precision"` fails:

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
try:
    scores = cross_val_score(
        tree, data, target, cv=10, scoring="precision", error_score="raise"
    )
except ValueError as exc:
    print(exc)  # pos_label=1 is not a valid label: It should be one of [...]
```

Build a scorer with an explicit positive label instead:

```python
from sklearn.metrics import make_scorer, precision_score

precision_donated = make_scorer(precision_score, pos_label="donated")
scores = cross_val_score(tree, data, target, cv=10, scoring=precision_donated)
```

Metrics that do not depend on a positive label, such as `"accuracy"` and `"balanced_accuracy"`, work directly with string labels.

## Multiclass metrics

For more than two classes, precision, recall, and F1 can be averaged in different ways:

- `average="macro"`: calculate each class separately, then give every class equal weight
- `average="weighted"`: weight each class by its support, so common classes matter more
- `average="micro"`: aggregate all TP, FP, and FN before calculating; common in multilabel settings
- `average=None`: return one score per class

Always report the averaging method. A weighted score can be high while a rare class performs poorly; macro scores reveal that imbalance more clearly. The corresponding scorer names are `"f1_macro"`, `"precision_weighted"`, `"recall_micro"`, and so on.

ROC-AUC for multiclass classification also needs a strategy such as one-vs-rest or one-vs-one and an averaging choice. Use the API's `multi_class` and `average` arguments consistently, or the scorers `"roc_auc_ovr"` and `"roc_auc_ovo"`.

## Class imbalance

### Reading accuracy and balanced accuracy together

| Accuracy | Balanced accuracy | Interpretation |
| --- | --- | --- |
| High | Low | The model is biased toward the majority class and misses the minority class |
| High | High | The model performs well on both classes |
| Low | Low | The model performs poorly overall (underfitting or uninformative features) |

For example, an accuracy of 0.85 with a balanced accuracy of 0.60 means that the model is good at predicting the majority class but poor at predicting the minority class. Comparing both metrics across cross-validation folds, for instance with a box plot, also shows how stable each metric is.

### Remedies

- use metrics that account for imbalance: balanced accuracy, precision and recall of the minority class, F1, average precision, MCC
- use stratified splits (`stratify=y`, `StratifiedKFold`) so that every fold contains the rare class; see [Cross-Validation Strategies](Cross-Validation-Strategies.md)
- reweight the classes with `class_weight="balanced"` (logistic regression, SVMs, trees, forests, histogram gradient boosting)
- tune the decision threshold instead of using 0.5
- resample the training data (oversampling the minority class or undersampling the majority class), for example with the separate `imbalanced-learn` library, always inside the cross-validation loop

## Which metric should you choose?

| Situation | Useful primary metric |
| --- | --- |
| Balanced classes and equal error costs | Accuracy, with the confusion matrix |
| Imbalanced classes, both classes matter | Balanced accuracy or MCC |
| False positives are expensive | Precision or $F_{\beta}$ with $\beta<1$ |
| False negatives are expensive | Recall or $F_{\beta}$ with $\beta>1$ |
| Need one balance for positive-class performance | F1-score |
| Need threshold-independent ranking quality | ROC-AUC |
| Rare positive class | Precision-recall curve and average precision, plus recall/precision at a chosen threshold |
| Probabilities drive decisions or expected costs | Log loss or calibration analysis, plus an operating-point metric |

The final metric should reflect the consequence of mistakes. Report several metrics when one number would hide an important failure mode, and always inspect the confusion matrix and the ROC or PR curves to understand the trade-offs visually.

## Worked example: interpreting a metric profile

Suppose a binary classifier evaluated on a test set obtains the following results:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8268 |
| Precision | 0.8644 |
| Recall | 0.6892 |
| F1-score | 0.7669 |
| ROC-AUC | 0.9019 |

- Accuracy `0.8268`: approximately 82.68% of the evaluated examples received the correct class.
- Precision `0.8644`: about 86.44% of positive predictions were correct, so false alarms are relatively limited at the chosen threshold.
- Recall `0.6892`: the model found about 68.92% of the actual positives and missed about 31.08% of them, which may be unacceptable when positives are safety-critical.
- F1 `0.7669`: the lower recall pulls the balanced positive-class score below precision. Indeed, $F_1\approx 2\frac{0.8644\cdot0.6892}{0.8644+0.6892}=0.7669$, which agrees with the reported value.
- ROC-AUC `0.9019`: the model ranks a randomly selected positive above a randomly selected negative about 90.19% of the time, which is strong ranking performance.

The high AUC alongside a recall of `0.6892` suggests that the model distinguishes the classes well but uses a threshold that is conservative for positive predictions. It does not prove that changing the threshold will solve the problem: inspect the ROC and precision-recall curves on validation data, then select the threshold there.

These values do not uniquely determine TP, TN, FP, and FN. The sample size, class prevalence, rounding, averaging method, and positive-label definition are required to reconstruct the exact confusion matrix. They also cannot establish whether the model is well calibrated, and they must be compared with a dummy baseline before concluding that the model is useful.

## Common mistakes

- evaluating on the training set and calling it test performance
- reporting accuracy alone on imbalanced data
- forgetting to compare with a dummy baseline
- calculating AUC from hard predictions instead of probabilities or decision scores
- comparing scores produced with different positive labels or averaging methods
- tuning the threshold on the final test set
- interpreting AUC as the precision or recall at the deployed threshold
- assuming a high AUC guarantees useful calibrated probabilities

Use a stratified split or `StratifiedKFold` for classification when appropriate, keep preprocessing inside a pipeline, and report uncertainty across cross-validation folds rather than treating a rounded single split as exact.

## Related pages

- [Metrics and Scoring Overview](Metrics-and-Scoring.md)
- [Regression Metrics](Regression-Metrics.md)
- [DummyClassifier](../07-Models/Baselines/DummyClassifier.md)
- [Blood Transfusion: Evaluating an Imbalanced Classifier](../10-Recipes/Blood-Transfusion-Classification-Metrics.md)
- [scikit-learn user guide: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [scikit-learn user guide: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)

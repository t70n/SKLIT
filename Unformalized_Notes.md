# Evaluation model performance

from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate

regressor = DecisionTreeRegressor()
cv_results_tree_regressor = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error", n_jobs=2
)

errors_tree_regressor = pd.Series(
    -cv_results_tree_regressor["test_score"], name="Decision tree regressor"
)
errors_tree_regressor.describe()

from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
result_dummy = cross_validate(
    dummy, data, target, cv=cv, scoring="neg_mean_absolute_error", n_jobs=2
)
errors_dummy_regressor = pd.Series(
    -result_dummy["test_score"], name="Dummy regressor"
)
errors_dummy_regressor.describe()


all_errors = pd.concat(
    [errors_tree_regressor, errors_dummy_regressor],
    axis=1,
)
all_errors

import matplotlib.pyplot as plt
import numpy as np

bins = np.linspace(start=0, stop=100, num=80)
all_errors.plot.hist(bins=bins, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Cross-validation testing errors")

result_dummy = cross_validate(
    dummy, data, target, cv=cv, scoring="r2", return_train_score=True, n_jobs=2
)
r2_train_score_dummy_regressor = pd.Series(
    result_dummy["train_score"], name="Dummy regressor train score"
)
r2_train_score_dummy_regressor.describe()



PLEASE MAKE A WIKI ABOUT THE DIFFERENT METRICS TO SCORE A MODEL: IF THERE ARE DIFFERENT BASED ON THE TYPE OF THE MODEL, MAYBE CREATE A WIKI FOR EACH TYPE.

In conclusion, $R^2$ is a normalized metric, which makes it independent of the
physical unit of the target variable, unlike MAE. A $R^2$ score of 0.0 is the
performance of a model that always predicts the mean observed value of the
target, while 1.0 corresponds to a model that predicts exactly the observed
target variable for each given input observation. Notice that it is only
possible to reach 1.0 if the target variable is a deterministic function of
the available input features. In practice, external factors often introduce
variability in the target that cannot be explained by the available features.
Therefore, the $R^2$ score of an optimal model is typically less than 1.0, not
due to a limitation of the machine learning algorithm itself, but because
the chosen input features are fundamentally not informative enough to
deterministically predict the target.

Overall, $R^2$ represents the proportion of the target's variability explained
by the model, while MAE, which retains the physical units of the target, can
be helpful for reporting errors in those units.

DO A WIKI AND A RECEIPE ABOUT DUMMY CLASSIFIER WITH THIS: 

# Write your code here.
from sklearn.dummy import DummyClassifier

# Stratégie : toujours prédire la classe la plus fréquente
dummy_most_frequent = make_pipeline(
    StandardScaler(),
    DummyClassifier(strategy="most_frequent")
)

# Stratégie : prédire en respectant la distribution des classes
dummy_stratified = make_pipeline(
    StandardScaler(),
    DummyClassifier(strategy="stratified")
)

# Stratégie : prédire uniformément au hasard
dummy_uniform = make_pipeline(
    StandardScaler(),
    DummyClassifier(strategy="uniform")
)

cv_dummy_most_frequent = cross_validate(
    dummy_most_frequent,
    data,
    target,
    cv=cv,
    n_jobs=2
)

cv_dummy_stratified = cross_validate(
    dummy_stratified,
    data,
    target,
    cv=cv,
    n_jobs=2
)

cv_dummy_uniform = cross_validate(
    dummy_uniform,
    data,
    target,
    cv=cv,
    n_jobs=2
)

# Write your code here.
errors_dummy_most_frequent = pd.Series(
    cv_dummy_most_frequent["test_score"],
    name="Dummy Classifier (most_frequent)"
)

errors_dummy_stratified = pd.Series(
    cv_dummy_stratified["test_score"],
    name="Dummy Classifier (stratified)"
)

errors_dummy_uniform = pd.Series(
    cv_dummy_uniform["test_score"],
    name="Dummy Classifier (uniform)"
)

# Write your code here.
results_df = pd.concat(
    [
        test_score_model1,
        errors_dummy_most_frequent,
        errors_dummy_stratified,
        errors_dummy_uniform
    ],
    axis=1
)

#print(results_df)

results_df.plot.hist(bins=20, alpha=0.5)

## STRATIFICATION 

import numpy as np
from sklearn.model_selection import KFold

data_random = np.random.randn(9, 1)
cv = KFold(n_splits=3)
for train_index, test_index in cv.split(data_random):
    print("TRAIN:", train_index, "TEST:", test_index)

import matplotlib.pyplot as plt

target.plot()
plt.xlabel("Sample index")
plt.ylabel("Class")
plt.yticks(target.unique())
_ = plt.title("Class value in target y")

train_cv_counts.plot.bar()
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.ylabel("Count")
_ = plt.title("Training set class counts")We can confirm that in each fold, only two of the three classes are present in the training set and all samples of the remaining class is used as a test set. So our model is unable to predict this class that was unseen during the training stage.


We see that neither the training and testing sets have the same class frequencies as our original dataset because the count for each class is varying a little.

However, one might want to split our data by preserving the original class frequencies: we want to stratify our data by class. In scikit-learn, some cross-validation strategies implement the stratification; they contain Stratified in their names.

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)

results = cross_validate(model, data, target, cv=cv)
test_score = results["test_score"]
print(
    f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)
The difference is due to the small number of samples in the iris dataset.

Stratification is especially useful for ensuring that rare classes are represented in every cross validation split. In particular, if a class is absent from one or more splits, some classification metrics may become undefined. It is also the case that some performance metrics depend on the proportion of the positive class, as we will see in a future notebook.

However, as noted in the scikit-learn user guide, stratification makes the folds more homogeneous. In the presence of severe class imbalance, this can artificially reduce the variability of performance metrics across folds, causing the observed variability to underestimate the true uncertainty in model performance.

## SAMPLE GROUPING

Note

Here we use a MinMaxScaler as we know that each pixel's gray-scale is strictly bounded between 0 (white) and 16 (black). This makes MinMaxScaler more suited in this case than StandardScaler, as some pixels consistently have low variance (pixels at the borders might almost always be zero if most digits are centered in the image). Then, using StandardScaler can result in a very high scaled value due to division by a small number.

import pandas as pd

all_scores = pd.DataFrame(
    [test_score_no_shuffling, test_score_with_shuffling],
    index=["KFold without shuffling", "KFold with shuffling"],
).T

import matplotlib.pyplot as plt

all_scores.plot.hist(bins=16, edgecolor="black", alpha=0.7)
plt.xlim([0.8, 1.0])
plt.xlabel("Accuracy score")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = plt.title("Distribution of the test scores")

print(digits.DESCR)


If we read carefully, load_digits loads a copy of the test set of the UCI ML hand-written digits dataset, which consists of 1797 images by 13 different writers. Thus, each writer wrote several times the same numbers. Let's suppose the dataset is ordered by writer. Subsequently, not shuffling the data will keep all writer samples together either in the training or the testing sets. Mixing the data will break this structure, and therefore digits written by the same writer will be available in both the training and testing sets.

Besides, a writer will usually tend to write digits in the same manner. Thus, our model will learn to identify a writer's pattern for each digit instead of recognizing the digit itself.

We can solve this problem by ensuring that the data associated with a writer should either belong to the training or the testing set. Thus, we want to group samples for each writer.

Indeed, we can recover the groups by looking at the target variable.

from itertools import count
import numpy as np

# defines the lower and upper bounds of sample indices
# for each writer
writer_boundaries = [
    0,
    130,
    256,
    386,
    516,
    646,
    776,
    915,
    1029,
    1157,
    1287,
    1415,
    1545,
    1667,
    1797,
]
groups = np.zeros_like(target)
lower_bounds = writer_boundaries[:-1]
upper_bounds = writer_boundaries[1:]

for group_id, lb, up in zip(count(), lower_bounds, upper_bounds):
    groups[lb:up] = group_id

from sklearn.model_selection import GroupKFold

cv = GroupKFold()
test_score = cross_val_score(
    model, data, target, groups=groups, cv=cv, n_jobs=2
)
print(
    f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)

In conclusion, accounting for any sample grouping patterns is crucial when assessing a model’s ability to generalize to new groups. Without this consideration, the results may appear overly optimistic compared to the actual performance.

The interested reader can learn about other group-aware cross-validation techniques in the scikit-learn user guide.

## NON i.i.d Data

Note

i.i.d is the acronym of "independent and identically distributed" (as in "independent and identically distributed random variables").

In conclusion, it is really important not to carelessly use a cross-validation strategy which do not respect some assumptions such as having i.i.d data. It might lead to misleading outcomes, creating the false impression that a predictive model performs well when it may not be the case in the intended real-world scenario.

scikit-learn offers useful tools for time-series analysis apart from TimeSeriesSplit, (see for instance the Time-related feature engineering example in the documentation), and scikit-learn models can yield even better results when combined with other specialized libraries.


## NESTED CROSS-VALIDATION

This notebook highlights nested cross-validation and its impact on the estimated generalization performance compared to naively using a single level of cross-validation, both for hyperparameter tuning and evaluation of the generalization performance.

MAKE A RECEIPE : 

from sklearn.datasets import load_breast_cancer

data, target = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1]}
model_to_tune = SVC()

search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid, n_jobs=2)
search.fit(data, target)

print(f"The best parameters found are: {search.best_params_}")
print(f"The mean CV score of the best model is: {search.best_score_:.3f}")



At this stage, one should be extremely careful using this score. The misinterpretation would be the following: since this mean score was computed using cross-validation test sets, we could use it to assess the generalization performance of the model trained with the best hyper-parameters.

However, we should not forget that we used this score to pick-up the best model. It means that we used knowledge from the test sets (i.e. test scores) to select the hyper-parameter of the model it-self.

Thus, this mean score is not a fair estimate of our testing error. Indeed, it can be too optimistic, in particular when running a parameter search on a large grid with many hyper-parameters and many possible values per hyper-parameter. A way to avoid this pitfall is to use a "nested" cross-validation.

In the following, we will use an inner cross-validation corresponding to the previous procedure above to only optimize the hyperparameters. We will also embed this tuning procedure within an outer cross-validation, which is dedicated to estimate the testing error of our tuned model.

In this case, our inner cross-validation always gets the training set of the outer cross-validation, making it possible to always compute the final testing scores on completely independent sets of samples.

Let us do this in one go as follows:


from sklearn.model_selection import cross_val_score, KFold

# Declare the inner and outer cross-validation strategies
inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

# Inner cross-validation for parameter search
model = GridSearchCV(
    estimator=model_to_tune, param_grid=param_grid, cv=inner_cv, n_jobs=2
)

# Outer cross-validation to compute the testing score
test_score = cross_val_score(model, data, target, cv=outer_cv, n_jobs=2)
print(
    "The mean score using nested cross-validation is: "
    f"{test_score.mean():.3f} ± {test_score.std():.3f}"
)


### Classification Evaluation Metrics

target_predicted = classifier.predict(data_test)
target_predicted[:5]

target_test == target_predicted


import numpy as np

np.mean(target_test == target_predicted)

Accuracy : compute how many times our classifier was right and divide it by the number of samples in our set.Accuracy is an aggregate of the errors made by the classifier.

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, target_predicted)
print(f"Accuracy: {accuracy:.3f}")

LogisticRegression also has a method named score (part of the standard scikit-learn API), which computes the accuracy score. 

from sklearn.metrics import ConfusionMatrixDisplay

_ = ConfusionMatrixDisplay.from_estimator(classifier, data_test, target_test)

the top left corner are true positives (TP) and corresponds to people who gave blood and were predicted as such by the classifier;
the bottom right corner are true negatives (TN) and correspond to people who did not give blood and were predicted as such by the classifier;
the top right corner are false negatives (FN) and correspond to people who gave blood but were predicted to not have given blood;
the bottom left corner are false positives (FP) and correspond to people who did not give blood but were predicted to have given blood.

The former metric, known as the precision, is defined as TP / (TP + FP) and represents how likely the person actually gave blood when the classifier predicted that they did. The latter, known as the recall, defined as TP / (TP + FN) and assesses how well the classifier is able to correctly identify people who did give blood. 

from sklearn.metrics import precision_score, recall_score

precision = precision_score(target_test, target_predicted, pos_label="donated")
recall = recall_score(target_test, target_predicted, pos_label="donated")

print(f"Precision score: {precision:.3f}")
print(f"Recall score: {recall:.3f}")

from sklearn.dummy import DummyClassifier
​
dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(data_train, target_train)
print(
    "Accuracy of the dummy classifier: "
    f"{dummy_classifier.score(data_test, target_test):.3f}"
)
Accuracy of the dummy classifier: 0.762
With the dummy classifier, which always predicts the negative class 'not donated', we obtain an accuracy score of 76%. Therefore, it means that this classifier, without learning anything from the data data, is capable of predicting as accurately as our logistic regression model.

The problem illustrated above is also known as the class imbalance problem. When the classes are imbalanced, accuracy should not be used. In this case, one should either use the precision and recall as presented above or the balanced accuracy score instead of accuracy.

from sklearn.metrics import balanced_accuracy_score
​
balanced_accuracy = balanced_accuracy_score(target_test, target_predicted)
print(f"Balanced accuracy: {balanced_accuracy:.3f}")
Balanced accuracy: 0.553
The balanced accuracy is equivalent to accuracy in the context of balanced classes. It is defined as the average recall obtained on each class.

The default threshold (0.5) assumes equal importance for precision (avoiding false positives) and recall (avoiding false negatives).
However, in real-world scenarios, you might prioritize one over the other:High precision: Use a higher threshold (e.g., 0.7). This reduces false positives but may increase false negatives.
High recall: Use a lower threshold (e.g., 0.3). This captures more positives but may increase false positives.

Precision-Recall Tradeoff:
If your goal is to minimize false positives (e.g., spam detection), use a higher threshold.
If your goal is to capture all positives (e.g., medical diagnosis), use a lower threshold.

ROC-AUC vs. PR-AUC:
ROC-AUC is useful when classes are balanced.
PR-AUC is more informative for imbalanced datasets (e.g., rare diseases or fraud detection).

The statistics used for this are sensitivity (recall) and specificity.

Here’s a **clear, concise, and actionable** breakdown of the key classification metrics, their definitions, formulas, and when to use them. I’ll also recommend which one(s) to prioritize based on your goals.

---

---

## **1. Core Metrics Definitions**
All metrics are derived from the **confusion matrix** for binary classification:


Confusion Matrix


|                | **Actual Positive (P)** | **Actual Negative (N)** |
|----------------|--------------------------|--------------------------|
| **Predicted Positive** | True Positive (TP)      | False Positive (FP)     |
| **Predicted Negative** | False Negative (FN)      | True Negative (TN)      |

---

### **A. Recall (Sensitivity, True Positive Rate - TPR)**
- **Definition**: Proportion of **actual positives** correctly identified.
- **Formula**:
  \( \text{Recall} = \frac{TP}{TP + FN} \)
- **Focus**: **Avoiding false negatives** (e.g., missing a disease diagnosis).
- **Use when**: False negatives are costly (e.g., medical testing, fraud detection).

---

### **B. Precision (Positive Predictive Value - PPV)**
- **Definition**: Proportion of **predicted positives** that are truly positive.
- **Formula**:
  \( \text{Precision} = \frac{TP}{TP + FP} \)
- **Focus**: **Avoiding false positives** (e.g., spam emails marked as important).
- **Use when**: False positives are costly (e.g., legal decisions, spam filtering).

---

### **C. Specificity (True Negative Rate - TNR)**
- **Definition**: Proportion of **actual negatives** correctly identified.
- **Formula**:
  \( \text{Specificity} = \frac{TN}{TN + FP} \)
- **Focus**: **Avoiding false positives** (complementary to recall).
- **Use when**: You care about correctly identifying negatives (e.g., screening tests).

---
---
### **D. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
- **Definition**: Measures the classifier’s ability to **distinguish between classes** across all thresholds.
- **Key Points**:
  - Plots **TPR (Recall)** vs. **FPR (1 - Specificity)**.
  - **AUC = 1**: Perfect classifier.
  - **AUC = 0.5**: Random guessing (baseline).
- **Use when**:
  - Classes are **balanced**.
  - You want a **threshold-independent** metric.
  - You need to compare models **overall performance**.

---
---
### **E. Average Precision (AP) / PR-AUC**
- **Definition**: Summarizes the **precision-recall curve** as a single score.
- **Key Points**:
  - Focuses on **positive class performance**.
  - **AP = 1**: Perfect precision and recall.
  - More informative than ROC-AUC for **imbalanced datasets**.
- **Use when**:
  - The positive class is **rare** (e.g., fraud, rare diseases).
  - You care more about **precision-recall tradeoffs** than overall accuracy.

---

---
---
## **2. Which Metric to Choose?**
Use this decision tree to pick the right metric for your problem:

```mermaid
graph TD
    A[Is your dataset balanced?] -->|Yes| B[Use ROC-AUC]
    A -->|No| C[Is the positive class rare?]
    C -->|Yes| D[Use PR-AUC (Average Precision)]
    C -->|No| E[Use F1-Score or Precision/Recall]
    B --> F[Also consider Accuracy]
    D --> G[Focus on Precision-Recall Curve]
    E --> H[Choose based on cost:]
    H -->|False positives costly| I[Prioritize Precision]
    H -->|False negatives costly| J[Prioritize Recall]
```

---

### **Recommendations by Use Case**
| **Goal**                          | **Primary Metric**       | **Secondary Metric**       | **Avoid**               |
|-----------------------------------|--------------------------|----------------------------|-------------------------|
| Minimize false negatives (e.g., medical diagnosis) | **Recall (Sensitivity)** | Specificity, PR-AUC       | Precision               |
| Minimize false positives (e.g., spam filtering)   | **Precision**            | Specificity, ROC-AUC      | Recall                  |
| Balanced classes, overall performance              | **ROC-AUC**              | Accuracy, F1-Score        | None                    |
| Imbalanced classes (rare positives)               | **PR-AUC**               | F1-Score, Recall          | ROC-AUC (can be misleading) |
| Need a single threshold-independent score         | **ROC-AUC or PR-AUC**    | F1-Score                  | Accuracy (if imbalanced) |

---

---
## **3. Practical Example**
Suppose you’re building a model to detect **fraudulent transactions** (rare positive class, ~1% of data):
- **Avoid**: ROC-AUC (can be overly optimistic due to class imbalance).
- **Use**:
  - **PR-AUC**: Captures performance on the rare positive class.
  - **Precision**: Minimize false alarms (legitimate transactions flagged as fraud).
  - **Recall**: Ensure most fraud cases are caught (even if it means more false positives).

---
---
## **4. Summary Table**

Metric Summary


| **Metric**       | **Formula**               | **Focus**                          | **Best for**                          | **Range**       |
|------------------|---------------------------|------------------------------------|---------------------------------------|-----------------|
| Recall           | TP / (TP + FN)            | Avoid false negatives              | Medical diagnosis, fraud detection   | [0, 1]          |
| Precision        | TP / (TP + FP)            | Avoid false positives              | Spam filtering, legal decisions        | [0, 1]          |
| Specificity      | TN / (TN + FP)            | Avoid false positives (negatives)  | Screening tests                       | [0, 1]          |
| ROC-AUC          | Area under ROC curve      | Overall class separation           | Balanced datasets                     | [0.5, 1]        |
| PR-AUC           | Area under PR curve       | Positive class performance         | Imbalanced datasets                   | [0, 1]          |

---
---
### **Final Advice**
- **Default choice**: Start with **ROC-AUC** (if balanced) or **PR-AUC** (if imbalanced).
- **Fine-tune**: Adjust thresholds based on **precision-recall tradeoffs** for your specific cost structure.
- **Always check**: The confusion matrix and curves (ROC/PR) to understand tradeoffs visually.


from sklearn.metrics import PrecisionRecallDisplay

disp = PrecisionRecallDisplay.from_estimator(
    classifier, data_test, target_test, pos_label="donated", marker="+"
)
disp = PrecisionRecallDisplay.from_estimator(
    dummy_classifier,
    data_test,
    target_test,
    pos_label="donated",
    color="tab:orange",
    linestyle="--",
    ax=disp.ax_,
)
plt.xlabel("Recall (also known as TPR or sensitivity)")
plt.ylabel("Precision (also known as PPV)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = disp.ax_.set_title("Precision-recall curve")


from sklearn.metrics import RocCurveDisplay

disp = RocCurveDisplay.from_estimator(
    classifier, data_test, target_test, pos_label="donated", marker="+"
)
disp = RocCurveDisplay.from_estimator(
    dummy_classifier,
    data_test,
    target_test,
    pos_label="donated",
    color="tab:orange",
    linestyle="--",
    ax=disp.ax_,
)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate\n(also known as sensitivity or recall)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
_ = disp.ax_.set_title("Receiver Operating Characteristic curve")


fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))

PrecisionRecallDisplay.from_estimator(
    classifier,
    data_test,
    target_test,
    pos_label="donated",
    marker="+",
    plot_chance_level=True,
    chance_level_kw={"color": "tab:orange", "linestyle": "--"},
    ax=axs[0],
)
RocCurveDisplay.from_estimator(
    classifier,
    data_test,
    target_test,
    pos_label="donated",
    marker="+",
    plot_chance_level=True,
    chance_level_kw={"color": "tab:orange", "linestyle": "--"},
    ax=axs[1],
)

_ = fig.suptitle("PR and ROC curves")

# Write your code here.
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

cv = StratifiedKFold()
test_score = cross_val_score(
    tree, data, target, cv=cv, n_jobs=2, scoring='balanced_accuracy'
)
print(
    f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
try:
    scores = cross_val_score(
        tree, data, target, cv=10, scoring="precision", error_score="raise"
    )
except ValueError as exc:
    print(exc)





# Write your code here.
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

cv = StratifiedKFold()
test_score = cross_val_score(
    tree, data, target, cv=cv, n_jobs=2, scoring='balanced_accuracy'
)
print(
    f"The average accuracy is {test_score.mean():.3f} ± {test_score.std():.3f}"
)

# Write your code here.
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np


scoring = ['accuracy', 'balanced_accuracy']


cv_results = cross_validate(
    tree, 
    data, 
    target,
    scoring=scoring,
    cv=5,       
    return_train_score=False
)


accuracy_scores = cv_results['test_accuracy']
balanced_accuracy_scores = cv_results['test_balanced_accuracy']

plt.figure(figsize=(8, 6))
plt.boxplot([accuracy_scores, balanced_accuracy_scores]) #, labels=['Accuracy', 'Balanced Accuracy'])
plt.ylabel('Score')
plt.title('Cross-Validation Scores: Accuracy vs. Balanced Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

Let’s break down the **conclusions** you can draw from your analysis, focusing on the relationship between **accuracy** and **balanced accuracy**, as well as the broader implications of your results.

---

---

## **1. Accuracy vs. Balanced Accuracy: Why the Difference?**
### **Key Observations from Your Code**
- **Accuracy**: Measures the overall correctness of the model, i.e., `(TP + TN) / (TP + TN + FP + FN)`.
- **Balanced Accuracy**: The average of recall for each class, i.e., `(Recall_class1 + Recall_class2) / 2`.
  - This metric is **insensitive to class imbalance** because it treats both classes equally, regardless of their size.

### **Why Balanced Accuracy is Lower Than Accuracy**
- **Imbalanced Dataset**: If your dataset has an **uneven class distribution** (e.g., far more "not donated" than "donated" samples), accuracy can be **misleadingly high**.
  - Example: If 90% of samples are "not donated" and the model always predicts "not donated," the accuracy would be 90%, but this is useless in practice.
  - **Balanced accuracy** accounts for this by averaging recall for both classes, so it **penalizes poor performance on the minority class**.

- **Interpretation**:
  - If **balanced accuracy < accuracy**, it suggests the model is **biased toward the majority class** (e.g., "not donated").
  - The model may be **missing many positive cases** (low recall for "donated"), which balanced accuracy captures but accuracy hides.

---

---
## **2. What Should You Understand from This Exercise?**
### **A. The Problem with Accuracy Alone**
- Accuracy can be **overly optimistic** for imbalanced datasets.
- **Example**: In your blood transfusion dataset, if "not donated" is the majority class, a model that always predicts "not donated" could have high accuracy but **zero recall for "donated"**.

### **B. Why Use Balanced Accuracy?**
- **Balanced accuracy** gives equal weight to both classes, making it a **better metric for imbalanced datasets**.
- If balanced accuracy is much lower than accuracy, your model is **not performing well on the minority class**.

### **C. Precision and Custom Scorers**
- You learned how to **create custom scorers** (e.g., precision for the "donated" class) using `make_scorer`.
- This is critical when:
  - The positive class is **not labeled as `1`** (e.g., "donated" vs. "not donated").
  - You need to **focus on a specific metric** (e.g., precision for fraud detection).

### **D. Cross-Validation Insights**
- Cross-validation helps you **assess the stability** of your model’s performance.
- The **box plot** of accuracy and balanced accuracy shows:
  - **Median performance**: The central line in the box.
  - **Variability**: The spread of the box (interquartile range) and whiskers (min/max).
  - If the boxes are **wide**, the model’s performance is **inconsistent across folds** (high variance).
  - If the boxes are **narrow**, the model is **stable**.

---

---
## **3. Practical Conclusions for Your Analysis**
### **What Your Results Likely Show**
| **Scenario**                          | **Accuracy** | **Balanced Accuracy** | **Interpretation**                                                                 |
|---------------------------------------|--------------|------------------------|-----------------------------------------------------------------------------------|
| High accuracy, low balanced accuracy | High         | Low                    | Model is biased toward the majority class ("not donated").                        |
| Similar accuracy and balanced accuracy | High      | High                   | Model performs well on **both classes**.                                          |
| Low accuracy and balanced accuracy   | Low          | Low                    | Model performs poorly **overall**.                                                |

### **Actionable Takeaways**
1. **If balanced accuracy is much lower than accuracy**:
   - Your model is **ignoring the minority class** ("donated").
   - **Solutions**:
     - Use **class weights** (e.g., `class_weight="balanced"` in `DecisionTreeClassifier`).
     - Try **resampling techniques** (oversampling "donated" or undersampling "not donated").
     - Focus on **precision/recall for the "donated" class** (since it’s likely the class of interest).

2. **If both metrics are low**:
   - The model is **underfitting** (not capturing patterns in the data).
   - **Solutions**:
     - Increase model complexity (e.g., deeper decision trees).
     - Feature engineering (add more informative features).

3. **If the box plot shows high variability**:
   - The model’s performance is **inconsistent across folds**.
   - **Solutions**:
     - Use more data or **simpler models** to reduce variance.
     - Check for **data leakage** or inconsistent preprocessing.

---

---
## **4. Example Interpretation for Your Blood Transfusion Dataset**
Assume your results look like this:
- **Accuracy**: 0.85 ± 0.02
- **Balanced Accuracy**: 0.60 ± 0.05

**Conclusion**:
- The model is **85% accurate overall**, but its **balanced accuracy is only 60%**.
- This means the model is **good at predicting the majority class ("not donated")** but **poor at predicting the minority class ("donated")**.
- **Action**: Prioritize improving recall/precision for the "donated" class (e.g., by tuning the classifier or addressing class imbalance).

---
---
## **5. General Rules of Thumb**
| **Goal**                          | **Metric to Prioritize**       | **Why?**                                                                 |
|-----------------------------------|--------------------------------|--------------------------------------------------------------------------|
| Maximize overall correctness      | Accuracy                       | Only if classes are **balanced**.                                      |
| Handle imbalanced classes        | Balanced Accuracy, F1-Score    | Avoids bias toward the majority class.                                 |
| Minimize false positives          | Precision                      | Critical for applications like spam filtering.                         |
| Minimize false negatives          | Recall (Sensitivity)           | Critical for applications like disease detection.                     |
| Compare models (balanced data)    | ROC-AUC                        | Threshold-independent measure of separability.                       |
| Compare models (imbalanced data)  | PR-AUC                         | Focuses on the positive class performance.                            |

---
---
### **Final Answer to "What Am I Supposed to Understand?"**
You’re supposed to understand:
1. **Accuracy alone can be misleading** for imbalanced datasets.
2. **Balanced accuracy** reveals how well the model performs on **both classes**, not just the majority.
3. **Custom scorers** (like precision for "donated") are necessary when the positive class isn’t labeled as `1`.
4. **Cross-validation** helps assess **stability** and **generalizability** of your model.
5. **Always pair metrics** (e.g., accuracy + balanced accuracy) to get a **complete picture** of performance.

**Next Steps**:
- Try **rebalancing your dataset** and see how it affects the metrics.
- Experiment with **different classifiers** (e.g., Random Forest) and compare their balanced accuracy.
- Plot **precision-recall curves** to further diagnose class imbalance issues.


### METRICS FOR REGRESSION




The raw MSE can be difficult to interpret. One way is to rescale the MSE by the variance of the target. This score is known as the  𝑅2 also called the coefficient of determination. Indeed, this is the default score used in scikit-learn by calling the method score.

regressor.score(data_test, target_test)

The  𝑅2
  score represents the proportion of variance of the target that is explained by the independent variables in the model. The best score possible is 1 but there is no lower bound. However, a model that predicts the expected value of the target would get a score of 0.

from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(data_train, target_train)
print(
    "R2 score for a regressor predicting the mean:"
    f"{dummy_regressor.score(data_test, target_test):.3f}"
)

The  𝑅2
  score gives insight into the quality of the model's fit. However, this score cannot be compared from one dataset to another and the value obtained does not have a meaningful interpretation relative the original unit of the target. If we wanted to get an interpretable score, we would be interested in the median or mean absolute error.

from sklearn.metrics import mean_absolute_error
​
target_predicted = regressor.predict(data_test)
print(
    "Mean absolute error: "
    f"{mean_absolute_error(target_test, target_predicted):.3f} k$"
)


import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

PredictionErrorDisplay.from_predictions(
    y_true=target_test,
    y_pred=target_predicted,
    kind="actual_vs_predicted",
    scatter_kwargs={"alpha": 0.5},
    ax=axs[0],
)
axs[0].axis("square")
axs[0].set_xlabel("Predicted values (k$)")
axs[0].set_ylabel("True values (k$)")

PredictionErrorDisplay.from_predictions(
    y_true=target_test,
    y_pred=target_predicted,
    kind="residual_vs_predicted",
    scatter_kwargs={"alpha": 0.5},
    ax=axs[1],
)
axs[1].axis("square")
axs[1].set_xlabel("Predicted values (k$)")
axs[1].set_ylabel("Residual values (k$)")

_ = fig.suptitle(
    "Regression using a model\nwithout target transformation", y=1.1
)

This means that the residuals still hold some structure typically visible as the "banana" or "smile" shape of the residual plot. This is often a clue that our model could be improved, either by transforming the features, the target or sometimes changing the model type or its parameters. In this case let's try to see if the model would benefit from a target transformation that monotonically reshapes the target variable to follow a normal distribution.

from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

transformer = QuantileTransformer(
    n_quantiles=900, output_distribution="normal"
)
model_transformed_target = TransformedTargetRegressor(
    regressor=regressor, transformer=transformer
)
model_transformed_target.fit(data_train, target_train)
target_predicted = model_transformed_target.predict(data_test)

fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

PredictionErrorDisplay.from_predictions(
    y_true=target_test,
    y_pred=target_predicted,
    kind="actual_vs_predicted",
    scatter_kwargs={"alpha": 0.5},
    ax=axs[0],
)
axs[0].axis("square")
axs[0].set_xlabel("Predicted values (k$)")
axs[0].set_ylabel("True values (k$)")

PredictionErrorDisplay.from_predictions(
    y_true=target_test,
    y_pred=target_predicted,
    kind="residual_vs_predicted",
    scatter_kwargs={"alpha": 0.5},
    ax=axs[1],
)
axs[1].axis("square")
axs[1].set_xlabel("Predicted values (k$)")
axs[1].set_ylabel("Residual values (k$)")

_ = fig.suptitle(
    "Regression using a model that\ntransforms the target before fitting",
    y=1.1,
)

print(
    "Mean absolute error: "
    f"{mean_absolute_error(target_test, target_predicted):.3f} k$"
)
print(
    "Median absolute error: "
    f"{median_absolute_error(target_test, target_predicted):.3f} k$"
)
print(
    "Mean absolute percentage error: "
    f"{mean_absolute_percentage_error(target_test, target_predicted):.2%}"
)

PoissonRegressor or a TweedieRegressor model instead of LinearRegression. READ THE DOCSTRINGS


Question 3 (1 point possible)
If all the values returned by cross_val_score(model_A, X, y, scoring="neg_mean_squared_error") are strictly lower than those returned by cross_val_score(model_B, X, y, scoring="neg_mean_squared_error"), it means that model_B generalizes:

 a) better than model_A  b) worse than model_A b) worse than <code>model_A</code> - incorrect
Hint: Remember that "neg_mean_squared_error" is an alias for the negative of the Mean Squared Error.

Explanation

solution: a)

Lower error values means a better model. Considering the negative error (i.e. multiplying by -1) reverses this relationship. In general, the scoring parameter passed to scikit-learn model selection utilities expects the convention that "higher is better", hence the slightly counter intuitive use of "neg_mean_squared_error" instead of "mean_squared_error" in scikit-learn.
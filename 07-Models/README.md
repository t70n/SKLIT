# Models

[Home](../README.md)

This section describes the estimators used in the wiki, grouped by family. Each page explains the intuition, the mathematics, the important hyperparameters, the preprocessing requirements, and the typical use cases of one model.

## What is a model in scikit-learn?

A model is an estimator that learns rules from data.

Important distinction:

- estimator: has `fit`
- predictor: has `fit` and `predict`
- transformer: has `fit` and `transform`

See [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#the-scikit-learn-estimator-api) for the complete estimator API.

## Pages by family

### Baselines

| Page | Task | Summary |
| --- | --- | --- |
| [DummyClassifier](Baselines/DummyClassifier.md) | Classification | Constant or random predictions that ignore the features; the reference score to beat |
| [DummyRegressor](Baselines/DummyRegressor.md) | Regression | Constant prediction (mean, median, quantile); the reference error to beat |

### Linear models

| Page | Task | Summary |
| --- | --- | --- |
| [Linear Regression](Linear-Models/Linear-Regression.md) | Regression | Weighted sum of features fitted by least squares; Ridge, Lasso, and Elastic Net variants |
| [LogisticRegression](Linear-Models/LogisticRegression.md) | Classification | Linear score converted to a probability by the sigmoid; regularized by `C` |

### Nearest neighbors

| Page | Task | Summary |
| --- | --- | --- |
| [KNeighborsClassifier](Nearest-Neighbors/KNeighborsClassifier.md) | Classification | Majority vote among the closest training samples |

### Support vector machines

| Page | Task | Summary |
| --- | --- | --- |
| [SVC](Support-Vector-Machines/SVC.md) | Classification | Maximum-margin boundary, nonlinear with kernels |
| [SVR](Support-Vector-Machines/SVR.md) | Regression | Epsilon-insensitive tube around a regression function |

### Decision trees

| Page | Task | Summary |
| --- | --- | --- |
| [Tree-Based Models](Decision-Trees/Tree-Based-Models.md) | Both | Overview of tree splits, bagging versus boosting, categorical and missing values |
| [DecisionTreeClassifier](Decision-Trees/DecisionTreeClassifier.md) | Classification | Recursive axis-aligned splits that minimize impurity |
| [DecisionTreeRegressor](Decision-Trees/DecisionTreeRegressor.md) | Regression | Piecewise-constant predictions that minimize squared error |

### Ensembles

| Page | Task | Summary |
| --- | --- | --- |
| [Ensemble Models](Ensembles/Ensemble-Models.md) | Both | Choosing between bagging, boosting, voting, and stacking |
| [Bagging](Ensembles/Bagging.md) | Both | Models fitted on bootstrap samples and averaged |
| [Random Forest](Ensembles/Random-Forest.md) | Both | Bagged trees with random feature subsets at each split |
| [AdaBoost](Ensembles/AdaBoost.md) | Both | Sequential weak learners that reweight difficult samples |
| [GradientBoostingClassifier](Ensembles/GradientBoostingClassifier.md) | Classification | Classic gradient boosting of shallow trees |
| [GradientBoostingRegressor](Ensembles/GradientBoostingRegressor.md) | Regression | Classic gradient boosting of shallow trees |
| [HistGradientBoostingClassifier](Ensembles/HistGradientBoostingClassifier.md) | Classification | Fast histogram-based boosting with native missing and categorical support |
| [HistGradientBoostingRegressor](Ensembles/HistGradientBoostingRegressor.md) | Regression | Fast histogram-based boosting with native missing and categorical support |
| [XGBoost](Ensembles/XGBoost.md) | Both | External library: regularized second-order gradient boosting |
| [CatBoost](Ensembles/CatBoost.md) | Both | External library: boosting with ordered target statistics for categories |

## Model selection guide

| Model | Needs scaling | Native categorical support | Native missing values | Interpretability | Typical role |
| --- | --- | --- | --- | --- | --- |
| Dummy models | No | Not applicable | Not applicable | Not applicable | Reference baseline |
| Linear and logistic regression | Recommended (required with regularization) | No (one-hot encode) | No | High (coefficients) | Fast, transparent baseline |
| k-nearest neighbors | Yes | No | No | Medium (similar samples) | Small datasets, local structure |
| SVC / SVR | Yes | No | No | Low with kernels | Small to medium datasets, nonlinear boundaries |
| Decision tree | No | No (ordinal encode) | Yes (scikit-learn 1.3 or later) | High when shallow | Interpretable rules |
| Random forest | No | No (ordinal encode) | Yes (scikit-learn 1.4 or later) | Medium (importances) | Robust tabular baseline |
| Classic gradient boosting | No | No | No | Medium | Small to medium tabular data |
| Histogram gradient boosting | No | Yes | Yes | Medium | Strong default for tabular data |
| XGBoost / CatBoost | No | Yes (CatBoost natively) | Yes | Medium | High-performance tabular models |

## Typical workflow

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = ...
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

## What determines model choice?

- type of task: classification or regression
- data shape and size
- presence of categorical features and missing values
- feature scaling needs
- desired speed and interpretability

## Baseline first

A good workflow is usually:

1. start with a dummy baseline ([DummyClassifier](Baselines/DummyClassifier.md), [DummyRegressor](Baselines/DummyRegressor.md))
2. fit a simple model, such as a linear model, inside a pipeline
3. evaluate both with the same cross-validation splits
4. improve preprocessing or the model family if needed
5. tune hyperparameters later

## Important model trade-offs

- linear models: interpretable, fast, often good baselines
- tree models: can handle mixed data well, sometimes very strong on tabular data
- k-NN: intuitive, distance based, sensitive to scaling
- SVM: powerful, but sensitive to the choice of hyperparameters and to scaling
- boosting ensembles: often the most accurate on tabular data, at the cost of more tuning

## Related sections

- [Preprocessing](../05-Preprocessing/README.md)
- [Model Evaluation](../08-Model-Evaluation/README.md)
- [Hyperparameter Tuning](../09-Hyperparameter-Tuning/README.md)
- [Recipes](../10-Recipes/README.md)
- [scikit-learn user guide: Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)

## Navigation

- Previous section: [Feature Engineering](../06-Feature-Engineering/README.md)
- Next section: [Model Evaluation](../08-Model-Evaluation/README.md)
- [Back to the home page](../README.md)

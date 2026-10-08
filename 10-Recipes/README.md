# Recipes

[Home](../README.md)

Recipes are complete, end-to-end workflows that combine the concepts of the other sections on a concrete dataset. Each recipe states its goal and dataset, gives runnable code, and interprets the results.

## Catalog

### Templates

| Recipe | Task | Dataset | Key concepts |
| --- | --- | --- | --- |
| [Quick Classification Pipeline](Quick-Classification-Pipeline.md) | Classification | Any mixed tabular data | `ColumnTransformer`, `LogisticRegression`, baseline check |
| [Quick Regression Pipeline](Quick-Regression-Pipeline.md) | Regression | Any mixed tabular data | `ColumnTransformer`, `RidgeCV`, baseline check |

### Classification

| Recipe | Task | Dataset | Key concepts |
| --- | --- | --- | --- |
| [Comparing a Classifier with Dummy Baselines](Dummy-Classifier-Baselines.md) | Classification | Adult Census (numerical) | `DummyClassifier` strategies, accuracy versus balanced accuracy |
| [Blood Transfusion: Evaluating an Imbalanced Classifier](Blood-Transfusion-Classification-Metrics.md) | Classification | Blood transfusion | Confusion matrix, precision, recall, ROC and PR curves, custom scorer |
| [Adult Census Classification](Adult-Census-Classification.md) | Classification | Adult Census | Mixed preprocessing, coefficient inspection, interactions |
| [Logistic Regression Decision Boundaries](Logistic-Regression-Decision-Boundaries.md) | Classification | Two-feature data | Regularization strength `C`, probability surface |

### Regression

| Recipe | Task | Dataset | Key concepts |
| --- | --- | --- | --- |
| [Regression Error Analysis and Target Transformation](Regression-Error-Analysis-and-Target-Transformation.md) | Regression | Ames housing | MAE, median absolute error, MAPE, residual plots, `TransformedTargetRegressor` |
| [Ames Housing: Linear Model vs Decision Tree](Ames-Housing-Linear-vs-Tree.md) | Regression | Ames housing | Paired fold comparison, nested tuning, ordinal encoding |
| [Ridge Regularization and Coefficient Stability](Ridge-Regularization-and-Stability.md) | Regression | Ames housing | `RidgeCV`, polynomial features, coefficient stability |
| [Comparing Nonlinear Feature-Engineering Pipelines](Nonlinear-Feature-Engineering-Comparison.md) | Regression | Any mixed tabular data | Splines, Nystroem, shared folds |

### Trees and ensembles

| Recipe | Task | Dataset | Key concepts |
| --- | --- | --- | --- |
| [Decision Tree Interpretation and Tuning](Decision-Tree-Interpretation-and-Tuning.md) | Both | Penguins | Decision regions, tree plots, leaf probabilities, extrapolation, tuning |
| [Synthetic Regression and Bootstrap Trees](Synthetic-Regression-and-Bootstrap.md) | Regression | Synthetic data | Bootstrap samples, bagging by hand, `BaggingRegressor` |
| [Penguins Random Forest Regression](Penguins-Random-Forest-Regression.md) | Regression | Penguins | Individual trees versus forest average |
| [Boosting with Sample Weights](Boosting-with-Sample-Weights.md) | Classification | Penguins | Reweighting errors, `AdaBoostClassifier` rounds |
| [Gradient Boosting Regression Comparison](Gradient-Boosting-Regression-Comparison.md) | Regression | California housing | Gradient boosting versus random forest, early stopping |
| [Histogram Gradient Boosting with Nested Cross-Validation](Hist-Gradient-Boosting-Nested-CV.md) | Regression | California housing | Nested tuning of `HistGradientBoostingRegressor` |

### Validation strategies

| Recipe | Task | Dataset | Key concepts |
| --- | --- | --- | --- |
| [Digits: Group-Aware Cross-Validation](Digits-Group-Aware-Cross-Validation.md) | Classification | Digits | Shuffling bias, sample groups, `GroupKFold` |
| [Nested Cross-Validation with SVC](Nested-Cross-Validation-with-SVC.md) | Classification | Breast cancer | Optimism of `best_score_`, nested versus non-nested estimates |

## Datasets

Recipes load the course datasets with paths such as `../datasets/adult-census.csv`, relative to the notebook that runs them. The files are available in the [`datasets` folder of the scikit-learn MOOC repository](https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets). Other recipes use datasets bundled with scikit-learn (`load_digits`, `load_breast_cancer`) or downloaded on first use (`fetch_california_housing`, `fetch_openml`).

## Navigation

- Previous section: [Hyperparameter Tuning](../09-Hyperparameter-Tuning/README.md)
- Next section: [Glossary](../11-Glossary/README.md)
- [Back to the home page](../README.md)

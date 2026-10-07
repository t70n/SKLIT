# Glossary

[Home](../README.md)

Short definitions of the terms used throughout the wiki, in alphabetical order. Each entry links to the page that develops the concept.

## Accuracy

Fraction of correct predictions. Default score of classifiers; misleading on imbalanced data. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#accuracy).

## Average precision

Summary of the precision-recall curve, often called PR-AUC; its chance level is the proportion of positives. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#precision-recall-curve-and-average-precision).

## Bagging

Bootstrap aggregating: models fitted independently on bootstrap samples and averaged, which mainly reduces variance. See [Bagging](../07-Models/Ensembles/Bagging.md).

## Balanced accuracy

Average of the recall of each class; equal to $1/K$ for any dummy classifier with $K$ classes. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#balanced-accuracy).

## Baseline

A simple reference model, such as a dummy model that ignores the features, that any useful model must outperform. See [DummyClassifier](../07-Models/Baselines/DummyClassifier.md) and [DummyRegressor](../07-Models/Baselines/DummyRegressor.md).

## Bias

Systematic error from an overly simplistic model; high bias leads to underfitting. See [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md#bias-variance-trade-off).

## Boosting

Ensemble method that fits weak learners sequentially, each one correcting the errors of the current ensemble; it mainly reduces bias. See [Ensemble Models](../07-Models/Ensembles/Ensemble-Models.md).

## Bootstrap sample

Sample of the same size as the original data, drawn with replacement. See [Bagging](../07-Models/Ensembles/Bagging.md#bootstrap-samples).

## Calibration

Agreement between predicted probabilities and observed frequencies. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#probabilistic-metrics-and-calibration).

## Categorical feature

Feature taking values in a finite set of categories, nominal (no order) or ordinal (ordered). See [Data Types, Column Names, and Categories](../03-EDA/Data-Types-and-Categories.md).

## Class imbalance

Situation where one class is much more frequent than the others. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#class-imbalance).

## Classification

Supervised learning task with a discrete target. See [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#supervised-learning).

## Coefficient of determination (R-squared)

Proportion of the target variance explained by the model; 0 for a model predicting the mean, 1 for perfect predictions. Default score of regressors. See [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md#coefficient-of-determination-r-squared).

## ColumnTransformer

Transformer that applies different transformations to different column subsets. See [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md).

## Confusion matrix

Table of true labels against predicted labels (TP, FP, FN, TN in the binary case). See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#the-confusion-matrix).

## Cross-validation

Repeated training and testing on different splits of the data, used to estimate performance more robustly than a single split. See [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md).

## Data leakage

Use, during training, of information that would not be available at prediction time, which makes evaluation optimistic. See [Identifiers, Leakage, Shuffling, and Sampling](../03-EDA/Identifiers-Leakage-and-Sampling.md#target-leakage).

## Decision threshold

Probability above which a classifier predicts the positive class (0.5 by default). See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#precision-recall-trade-off-and-the-decision-threshold).

## Dummy model

Model that ignores the features, such as `DummyClassifier` and `DummyRegressor`; used as a baseline. See [DummyClassifier](../07-Models/Baselines/DummyClassifier.md).

## Early stopping

Stopping the addition of boosting iterations when a validation score stops improving. See [Boosting Hyperparameter Tuning](../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md).

## Encoding

Conversion of categorical values into numbers, for example with `OneHotEncoder` or `OrdinalEncoder`. See [OrdinalEncoder vs OneHotEncoder](../05-Preprocessing/OrdinalEncoder-vs-OneHotEncoder.md).

## Ensemble

Model combining several base models, for example by bagging, boosting, voting, or stacking. See [Ensemble Models](../07-Models/Ensembles/Ensemble-Models.md).

## Estimator

An object that can learn from data via `fit`. See [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#the-scikit-learn-estimator-api).

## F1-score

Harmonic mean of precision and recall. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#f1-score).

## Feature

A characteristic or variable describing a sample; a column of `X`.

## Feature engineering

Creation of new representations of the data that make the prediction problem easier for a model. See [Feature Engineering Overview](../06-Feature-Engineering/Feature-Engineering-Overview.md).

## Feature selection

Keeping a subset of the existing features. See [Feature Selection](../06-Feature-Engineering/Feature-Selection.md).

## Generalization

How well a model performs on unseen data. See [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#training-and-generalization).

## Grid search

Exhaustive evaluation of every combination of a predefined set of hyperparameter values. See [GridSearchCV](../09-Hyperparameter-Tuning/GridSearchCV.md).

## Group (sample grouping)

Set of samples that are not independent, such as several measurements of the same patient or several digits of the same writer; they must stay on the same side of a split. See [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md#sample-grouping).

## Hyperparameter

A parameter chosen before training that controls the learning process, such as `C`, `max_depth`, or `n_neighbors`. See [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#parameters-versus-hyperparameters).

## i.i.d.

Independent and identically distributed: the assumption behind standard cross-validation. See [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md).

## Imputation

Replacement of missing values by estimated values. See [Missing Values](../03-EDA/Missing-Values.md).

## Learning curve

Training and test scores as a function of the training-set size. See [Learning Curves](../08-Model-Evaluation/Learning-Curves.md).

## Mean absolute error (MAE)

Average absolute difference between predictions and true values, in the unit of the target. See [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md#mean-absolute-error-mae).

## Mean squared error (MSE)

Average squared difference between predictions and true values; RMSE is its square root. See [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md#mean-squared-error-mse).

## Nested cross-validation

Cross-validation with an inner loop for hyperparameter tuning and an outer loop for an unbiased performance estimate. See [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md).

## Overfitting

Learning noise instead of the underlying pattern: low training error, high test error. See [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md).

## Parameter

A value learned from the data during `fit`, such as a coefficient or a split threshold. Fitted attributes end with an underscore (`coef_`). See [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#parameters-versus-hyperparameters).

## Pipeline

A sequence of preprocessing steps followed by a model, fitted and applied as a single estimator. See [Pipeline](../05-Preprocessing/Pipeline.md).

## Precision

Among the predicted positives, the fraction that are truly positive. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#precision).

## Predictor

An estimator with `predict`.

## Randomized search

Evaluation of a fixed number of hyperparameter combinations sampled from distributions. See [RandomizedSearchCV](../09-Hyperparameter-Tuning/RandomizedSearchCV.md).

## Recall

Among the true positives, the fraction found by the model; also called sensitivity or true positive rate. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#recall-sensitivity).

## Regression

Supervised learning task with a continuous target. See [Machine Learning Concepts](../01-ML-Basics/Machine-Learning-Concepts.md#supervised-learning).

## Regularization

Penalty on model complexity, such as the size of the coefficients, that reduces overfitting. See [Linear Regression](../07-Models/Linear-Models/Linear-Regression.md#key-parameters-and-alternatives).

## Residual

Difference between the true and the predicted value, $y_i-\hat{y}_i$. See [Regression Metrics](../08-Model-Evaluation/Regression-Metrics.md#visual-diagnostics).

## ROC curve and ROC-AUC

Curve of the true positive rate against the false positive rate for every threshold, and the area under it; 0.5 is the chance level. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#roc-curve-and-roc-auc).

## Sample

One row of the dataset.

## Scaling

Transformation of numerical features to comparable ranges, for example with `StandardScaler`. See [StandardScaler](../05-Preprocessing/StandardScaler.md).

## Scorer

Object or string name used by the `scoring` parameter of model-selection tools; scikit-learn always maximizes it, hence the `neg_` prefix for errors. See [Metrics and Scoring Overview](../08-Model-Evaluation/Metrics-and-Scoring.md).

## Specificity

Among the true negatives, the fraction correctly rejected; also called the true negative rate. See [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md#specificity-and-false-positive-rate).

## Stratification

Splitting that preserves the class proportions in every fold. See [Cross-Validation Strategies](../08-Model-Evaluation/Cross-Validation-Strategies.md#stratification).

## Target

The quantity we want to predict; `y`.

## Test set

Data used only to evaluate the final model.

## Train set

Data used to fit the model.

## Transformer

An estimator with `transform`, used for preprocessing. See [Why Preprocessing Matters](../05-Preprocessing/Why-Preprocessing-Matters.md#key-concept-transformations).

## Underfitting

Model too simple to capture the structure of the data: high training and test errors. See [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md#underfitting).

## Validation curve

Training and test scores as a function of one hyperparameter. See [Validation Curves](../08-Model-Evaluation/Validation-Curves.md).

## Validation set

Data used to compare models or select hyperparameters, distinct from the final test set; usually replaced by cross-validation. See [Train/Test Split](../01-ML-Basics/Train-Test-Split.md#training-validation-and-test-sets).

## Variance

Sensitivity of a model to small changes in the training set; high variance leads to overfitting. See [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md#bias-variance-trade-off).

## Navigation

- Previous section: [Recipes](../10-Recipes/README.md)
- [Back to the home page](../README.md)

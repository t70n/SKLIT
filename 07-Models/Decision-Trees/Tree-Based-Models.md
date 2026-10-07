# Tree-Based Models

[Home](../../README.md) / [Models](../README.md) / [Decision Trees](../README.md#decision-trees)

## Core idea

Tree-based models split the feature space using thresholds to create decision rules.

For a candidate split, a tree compares parent impurity with the weighted impurity of its children:

$$
\Delta I=I(P)-\frac{n_L}{n_P}I(L)-\frac{n_R}{n_P}I(R)
$$

Classification commonly uses Gini impurity or entropy. Regression commonly uses squared-error reduction. Because splits are axis-aligned, the resulting prediction surface is a collection of rectangular regions.

Examples:

- single trees: [DecisionTreeClassifier](DecisionTreeClassifier.md), [DecisionTreeRegressor](DecisionTreeRegressor.md)
- bagging ensembles: [Random Forest](../Ensembles/Random-Forest.md) (`RandomForestClassifier`, `RandomForestRegressor`), [Bagging](../Ensembles/Bagging.md)
- boosting ensembles: [AdaBoost](../Ensembles/AdaBoost.md), `GradientBoostingClassifier`/`GradientBoostingRegressor`, `HistGradientBoostingClassifier`/`HistGradientBoostingRegressor`, [XGBoost](../Ensembles/XGBoost.md), [CatBoost](../Ensembles/CatBoost.md)

For the complete ensemble comparison, see [Ensemble Models](../Ensembles/Ensemble-Models.md).

The main boosting implementations are [GradientBoostingClassifier](../Ensembles/GradientBoostingClassifier.md), [GradientBoostingRegressor](../Ensembles/GradientBoostingRegressor.md), [HistGradientBoostingClassifier](../Ensembles/HistGradientBoostingClassifier.md), and [HistGradientBoostingRegressor](../Ensembles/HistGradientBoostingRegressor.md).

## Single trees, bagging, and boosting

A single tree repeatedly makes greedy corrections within one hierarchy. It can have low bias but high variance when deep.

Bagging methods such as Random Forest fit trees on resampled data and average them:

$$
\hat{f}(x)=\frac{1}{B}\sum_{b=1}^{B}f_b(x)
$$

Random feature selection and averaging reduce variance because independent errors partially cancel. Boosting instead fits trees sequentially, adding corrections to the current model:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

This mainly reduces bias by building a strong additive predictor, but it can overfit noisy labels and requires more careful tuning.

## Why they are popular

- handle mixed data reasonably well
- do not require scaling for numerical features
- can capture nonlinear relationships
- often strong on tabular data

## Important property

A tree-based model does not assume a linear relationship between features and target.

Decision boundaries are axis-aligned and can be quite flexible.

## Categorical variables

Tree-based models can work with integer-coded categories, so `OrdinalEncoder` is often adequate: successive splits can isolate individual categories even when the order of the codes is arbitrary.

Ordinal encoding can still create an artificial order, so it is not universally safe. Use an estimator's documented native categorical support when available (for example `categorical_features` in histogram gradient boosting, or CatBoost), or choose an encoding that matches the estimator and validation setup. See [OrdinalEncoder vs OneHotEncoder](../../05-Preprocessing/OrdinalEncoder-vs-OneHotEncoder.md).

## Missing values

Native support for missing numerical values depends on the estimator and the scikit-learn version:

| Estimator | Native missing-value support |
| --- | --- |
| `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` | Yes |
| `DecisionTreeClassifier`, `DecisionTreeRegressor` | Yes, since scikit-learn 1.3 |
| `RandomForestClassifier`, `RandomForestRegressor` | Yes, since scikit-learn 1.4 |
| `GradientBoostingClassifier`, `GradientBoostingRegressor` | No: impute first |
| XGBoost, CatBoost | Yes |

When a split is learned, samples with missing values are sent to the child that gives the best improvement. If the training data had no missing values, missing values at prediction time are sent to the child with the most samples.

## Why one-hot encoding can be suboptimal

One-hot encoding creates many binary columns. For trees, this can slow training and adds dimensionality without adding much value.

## Typical use cases

- tabular data with both numerical and categorical variables
- datasets with many nonlinear interactions
- baseline models for structured tasks

## Practical limitations

- small data changes can produce very different single trees
- piecewise-constant trees do not extrapolate smoothly
- impurity feature importance can favor high-cardinality or continuous features
- smooth diagonal relationships may require many axis-aligned splits
- feature importance describes predictive association, not causation

## Related pages

- [DecisionTreeClassifier](DecisionTreeClassifier.md)
- [DecisionTreeRegressor](DecisionTreeRegressor.md)
- [Ensemble Models](../Ensembles/Ensemble-Models.md)
- [scikit-learn user guide: Decision trees](https://scikit-learn.org/stable/modules/tree.html)

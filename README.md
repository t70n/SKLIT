# SKLIT: Machine Learning Wiki with scikit-learn

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/stable/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

SKLIT is a structured, practical knowledge base on machine learning with scikit-learn. It covers the complete workflow, from loading and exploring tabular data to preprocessing, modeling, evaluation, and hyperparameter tuning, with short theoretical explanations, runnable code, and end-to-end recipes.

## About this wiki

- **Scope**: supervised learning on tabular data with pandas and scikit-learn, from first principles to model selection.
- **Format**: one concept per page, written in Markdown with LaTeX formulas, readable directly on GitHub.
- **Audience**: students and practitioners who want a concise reference and reusable code templates.
- **Versions**: code follows the scikit-learn 1.9 API and pandas 3; differences with older versions are pointed out where they matter. Most pages apply to scikit-learn 1.3 and later.

## How to use this wiki

| Goal | Where to start |
| --- | --- |
| Learn machine learning step by step | Follow the [learning path](#learning-path) below, section by section |
| Look up a concept or an estimator | Use the [mindmap](#mindmap) or the [glossary](11-Glossary/README.md) |
| Solve a concrete problem quickly | Browse the [recipes](10-Recipes/README.md) and adapt a template |
| Choose a model | Read the [model selection guide](07-Models/README.md#model-selection-guide) |
| Choose a metric | Read the [metrics overview](08-Model-Evaluation/Metrics-and-Scoring.md) |

Every section has an index page (`README.md`). Every page starts with a navigation line and ends with links to related pages or to the neighboring sections.

## Mindmap

The outline below lists every page of the wiki. On the GitHub Pages site, it is rendered as an interactive [markmap](https://markmap.js.org/) mindmap: branches can be folded and unfolded, the view can be zoomed and moved, and every node is a link. On GitHub, it is displayed as a table of contents.

<div class="markmap" markdown="0">

- SKLIT
  - [1. Machine Learning Basics](01-ML-Basics/README.md)
    - [Machine Learning Concepts](01-ML-Basics/Machine-Learning-Concepts.md)
    - [Train/Test Split](01-ML-Basics/Train-Test-Split.md)
    - [Overfitting vs Underfitting](01-ML-Basics/Overfitting-vs-Underfitting.md)
  - [2. Pandas](02-Pandas/README.md)
    - Fundamentals
      - [Pandas Basics](02-Pandas/Pandas-Basics.md)
      - [Inspecting Data](02-Pandas/Inspecting-Data.md)
      - [Selecting and Filtering](02-Pandas/Selecting-and-Filtering.md)
    - [GroupBy, Joins, and Reshaping](02-Pandas/GroupBy-Joins-and-Reshaping.md)
    - [DataFrame Cheat Sheet](02-Pandas/DataFrame-Manipulation-Cheat-Sheet.md)
  - [3. Exploratory Data Analysis](03-EDA/README.md)
    - Overview
      - [EDA Overview](03-EDA/EDA-Overview.md)
      - [Data Cleaning Overview](03-EDA/Data-Cleaning-Overview.md)
    - Data quality
      - [Data Types and Categories](03-EDA/Data-Types-and-Categories.md)
      - [Duplicates and Redundancy](03-EDA/Duplicates-and-Redundancy.md)
      - [Constant and Redundant Features](03-EDA/Constant-and-Redundant-Features.md)
      - [Invalid Values and Constraints](03-EDA/Invalid-Values-and-Constraints.md)
      - [Identifiers, Leakage, and Sampling](03-EDA/Identifiers-Leakage-and-Sampling.md)
    - Missing data
      - [Missing Values](03-EDA/Missing-Values.md)
      - [MICE and Iterative Imputation](03-EDA/MICE-and-Iterative-Imputation.md)
    - Distributions and anomalies
      - [Correlation and Distributions](03-EDA/Correlation-and-Distributions.md)
      - [Outlier Detection](03-EDA/Outlier-Detection.md)
  - [4. Visualization](04-Visualization/README.md)
    - [Visualization Overview](04-Visualization/Visualization-Overview.md)
    - [Histograms and Distributions](04-Visualization/Histograms-and-Distributions.md)
    - [Scatter Plots](04-Visualization/Scatter-Plots.md)
    - [Pairplots](04-Visualization/Pairplots.md)
  - [5. Preprocessing](05-Preprocessing/README.md)
    - [Why Preprocessing Matters](05-Preprocessing/Why-Preprocessing-Matters.md)
    - Scaling
      - [StandardScaler](05-Preprocessing/StandardScaler.md)
      - [MinMaxScaler](05-Preprocessing/MinMaxScaler.md)
    - Categorical encoding
      - [OrdinalEncoder vs OneHotEncoder](05-Preprocessing/OrdinalEncoder-vs-OneHotEncoder.md)
      - [Handling Unknown Categories](05-Preprocessing/Handling-Unknown-Categories.md)
    - Composition
      - [ColumnTransformer](05-Preprocessing/ColumnTransformer.md)
      - [Pipeline](05-Preprocessing/Pipeline.md)
  - [6. Feature Engineering](06-Feature-Engineering/README.md)
    - [Feature Engineering Overview](06-Feature-Engineering/Feature-Engineering-Overview.md)
    - Feature generation
      - [Row-Wise Feature Generation](06-Feature-Engineering/Row-Wise-Feature-Generation.md)
      - [Cross-Row and Local Features](06-Feature-Engineering/Cross-Row-and-Local-Features.md)
    - Nonlinear expansions
      - [Polynomial Features](06-Feature-Engineering/Polynomial-Features.md)
      - [KBinsDiscretizer](06-Feature-Engineering/KBinsDiscretizer.md)
      - [SplineTransformer](06-Feature-Engineering/SplineTransformer.md)
      - [Nystroem Kernel Approximation](06-Feature-Engineering/Nystroem-Kernel-Approximation.md)
    - Selection and reduction
      - [Feature Selection](06-Feature-Engineering/Feature-Selection.md)
      - [Principal Component Analysis](06-Feature-Engineering/Principal-Component-Analysis.md)
      - [Canonical Correlation Analysis](06-Feature-Engineering/Canonical-Correlation-Analysis.md)
  - [7. Models](07-Models/README.md)
    - Baselines
      - [DummyClassifier](07-Models/Baselines/DummyClassifier.md)
      - [DummyRegressor](07-Models/Baselines/DummyRegressor.md)
    - Linear models
      - [Linear Regression](07-Models/Linear-Models/Linear-Regression.md)
      - [LogisticRegression](07-Models/Linear-Models/LogisticRegression.md)
    - Nearest neighbors
      - [KNeighborsClassifier](07-Models/Nearest-Neighbors/KNeighborsClassifier.md)
    - Support vector machines
      - [SVC](07-Models/Support-Vector-Machines/SVC.md)
      - [SVR](07-Models/Support-Vector-Machines/SVR.md)
    - Decision trees
      - [Tree-Based Models](07-Models/Decision-Trees/Tree-Based-Models.md)
      - [DecisionTreeClassifier](07-Models/Decision-Trees/DecisionTreeClassifier.md)
      - [DecisionTreeRegressor](07-Models/Decision-Trees/DecisionTreeRegressor.md)
    - Ensembles
      - [Ensemble Models](07-Models/Ensembles/Ensemble-Models.md)
      - [Bagging](07-Models/Ensembles/Bagging.md)
      - [Random Forest](07-Models/Ensembles/Random-Forest.md)
      - [AdaBoost](07-Models/Ensembles/AdaBoost.md)
      - [GradientBoostingClassifier](07-Models/Ensembles/GradientBoostingClassifier.md)
      - [GradientBoostingRegressor](07-Models/Ensembles/GradientBoostingRegressor.md)
      - [HistGradientBoostingClassifier](07-Models/Ensembles/HistGradientBoostingClassifier.md)
      - [HistGradientBoostingRegressor](07-Models/Ensembles/HistGradientBoostingRegressor.md)
      - [XGBoost](07-Models/Ensembles/XGBoost.md)
      - [CatBoost](07-Models/Ensembles/CatBoost.md)
  - [8. Model Evaluation](08-Model-Evaluation/README.md)
    - Metrics
      - [Metrics and Scoring Overview](08-Model-Evaluation/Metrics-and-Scoring.md)
      - [Classification Metrics](08-Model-Evaluation/Classification-Metrics.md)
      - [Regression Metrics](08-Model-Evaluation/Regression-Metrics.md)
    - Validation
      - [Cross-Validation](08-Model-Evaluation/Cross-Validation.md)
      - [Cross-Validation Strategies](08-Model-Evaluation/Cross-Validation-Strategies.md)
    - Diagnostics
      - [Learning Curves](08-Model-Evaluation/Learning-Curves.md)
      - [Validation Curves](08-Model-Evaluation/Validation-Curves.md)
  - [9. Hyperparameter Tuning](09-Hyperparameter-Tuning/README.md)
    - Search methods
      - [Manual Tuning](09-Hyperparameter-Tuning/Manual-Tuning.md)
      - [GridSearchCV](09-Hyperparameter-Tuning/GridSearchCV.md)
      - [RandomizedSearchCV](09-Hyperparameter-Tuning/RandomizedSearchCV.md)
    - [Nested Cross-Validation](09-Hyperparameter-Tuning/Nested-Cross-Validation.md)
    - [Boosting Hyperparameter Tuning](09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md)
    - [Parallel Coordinates Visualization](09-Hyperparameter-Tuning/Parallel-Coordinates-Visualization.md)
  - [10. Recipes](10-Recipes/README.md)
    - Templates
      - [Quick Classification Pipeline](10-Recipes/Quick-Classification-Pipeline.md)
      - [Quick Regression Pipeline](10-Recipes/Quick-Regression-Pipeline.md)
    - Classification
      - [Dummy Classifier Baselines](10-Recipes/Dummy-Classifier-Baselines.md)
      - [Blood Transfusion Metrics](10-Recipes/Blood-Transfusion-Classification-Metrics.md)
      - [Adult Census Classification](10-Recipes/Adult-Census-Classification.md)
      - [Logistic Regression Decision Boundaries](10-Recipes/Logistic-Regression-Decision-Boundaries.md)
    - Regression
      - [Regression Error Analysis](10-Recipes/Regression-Error-Analysis-and-Target-Transformation.md)
      - [Ames Housing: Linear vs Tree](10-Recipes/Ames-Housing-Linear-vs-Tree.md)
      - [Ridge Regularization and Stability](10-Recipes/Ridge-Regularization-and-Stability.md)
      - [Nonlinear Feature Engineering](10-Recipes/Nonlinear-Feature-Engineering-Comparison.md)
    - Trees and ensembles
      - [Decision Tree Interpretation](10-Recipes/Decision-Tree-Interpretation-and-Tuning.md)
      - [Synthetic Regression and Bootstrap](10-Recipes/Synthetic-Regression-and-Bootstrap.md)
      - [Penguins Random Forest](10-Recipes/Penguins-Random-Forest-Regression.md)
      - [Boosting with Sample Weights](10-Recipes/Boosting-with-Sample-Weights.md)
      - [Gradient Boosting Comparison](10-Recipes/Gradient-Boosting-Regression-Comparison.md)
      - [Hist Gradient Boosting Nested CV](10-Recipes/Hist-Gradient-Boosting-Nested-CV.md)
    - Validation strategies
      - [Digits Group-Aware CV](10-Recipes/Digits-Group-Aware-Cross-Validation.md)
      - [Nested CV with SVC](10-Recipes/Nested-Cross-Validation-with-SVC.md)
  - [11. Glossary](11-Glossary/README.md)

</div>

## Learning path

The sections are numbered in the order of a typical machine learning workflow.

| Step | Section | What you will learn |
| --- | --- | --- |
| 1 | [Machine Learning Basics](01-ML-Basics/README.md) | Vocabulary, estimator API, train/test split, overfitting and underfitting |
| 2 | [Pandas](02-Pandas/README.md) | Loading, inspecting, selecting, combining, and reshaping tabular data |
| 3 | [Exploratory Data Analysis](03-EDA/README.md) | Data quality, missing values, distributions, correlations, outliers |
| 4 | [Visualization](04-Visualization/README.md) | Histograms, scatter plots, pairplots, and plot selection |
| 5 | [Preprocessing](05-Preprocessing/README.md) | Scaling, categorical encoding, `ColumnTransformer`, `Pipeline` |
| 6 | [Feature Engineering](06-Feature-Engineering/README.md) | Feature generation, nonlinear expansions, selection, PCA |
| 7 | [Models](07-Models/README.md) | Baselines, linear models, neighbors, SVMs, trees, ensembles |
| 8 | [Model Evaluation](08-Model-Evaluation/README.md) | Metrics, cross-validation strategies, learning and validation curves |
| 9 | [Hyperparameter Tuning](09-Hyperparameter-Tuning/README.md) | Grid and randomized search, nested cross-validation |
| 10 | [Recipes](10-Recipes/README.md) | End-to-end workflows on real datasets |
| 11 | [Glossary](11-Glossary/README.md) | Definitions of the main terms |

## Repository structure

```text
SKLIT/
├── README.md                     Home page, mindmap, and learning path
├── CONTRIBUTING.md               Writing conventions and page templates
├── LICENSE                       MIT license
├── 01-ML-Basics/                 Concepts, train/test split, overfitting
├── 02-Pandas/                    Data manipulation with pandas
├── 03-EDA/                       Exploratory data analysis and cleaning
├── 04-Visualization/             Exploratory plots
├── 05-Preprocessing/             Scaling, encoding, ColumnTransformer, Pipeline
├── 06-Feature-Engineering/       Feature generation, selection, and reduction
├── 07-Models/                    Estimators grouped by family
│   ├── Baselines/
│   ├── Linear-Models/
│   ├── Nearest-Neighbors/
│   ├── Support-Vector-Machines/
│   ├── Decision-Trees/
│   └── Ensembles/
├── 08-Model-Evaluation/          Metrics, cross-validation, diagnostic curves
├── 09-Hyperparameter-Tuning/     Search methods and nested cross-validation
├── 10-Recipes/                   End-to-end workflows
├── 11-Glossary/                  Definitions of the main terms
├── tools/check_links.py          Internal link checker
├── _config.yml                   GitHub Pages (Jekyll) configuration
├── _includes/head-custom.html    Mindmap and formula rendering on GitHub Pages
└── .github/workflows/            Continuous integration (link check)
```

Each numbered folder contains a `README.md` index page.

## Datasets

| Dataset | Used in | Source |
| --- | --- | --- |
| Adult Census (`adult-census.csv`, `adult-census-numeric-all.csv`) | Pandas examples, classification recipes | [scikit-learn MOOC datasets](https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets) |
| Ames housing (`house_prices.csv`, `ames_housing_no_missing.csv`) | Regression recipes | [scikit-learn MOOC datasets](https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets) |
| Blood transfusion (`blood_transfusion.csv`) | Classification metrics | [scikit-learn MOOC datasets](https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets) |
| Penguins (`penguins_classification.csv`, `penguins_regression.csv`) | Tree and ensemble recipes | [scikit-learn MOOC datasets](https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets) |
| Digits, breast cancer, iris, diabetes | Cross-validation and model examples | Bundled with scikit-learn (`sklearn.datasets`) |
| California housing | Ensemble recipes | Downloaded by `sklearn.datasets.fetch_california_housing` |

Code examples read the course files from `../datasets/`, relative to the notebook that runs them. Download the files from the MOOC repository and adapt the path if needed.

## Requirements

The code examples require Python 3.10 or later with:

```bash
pip install scikit-learn pandas numpy scipy matplotlib seaborn
```

Some pages use optional libraries: `xgboost`, `catboost`, and `plotly`.

## Publishing on GitHub Pages

The repository is ready to be published with GitHub Pages (Settings, Pages, deploy from the `main` branch, root folder). The default Jekyll build converts every page to HTML, and `_includes/head-custom.html` renders the mindmap above with markmap and the LaTeX formulas with MathJax. No build step is required on GitHub itself, where Markdown and formulas are rendered natively.

## Contributing

Writing conventions, page and recipe templates, and the checklist for adding a page are described in [CONTRIBUTING.md](CONTRIBUTING.md). Internal links are checked automatically on every push:

```bash
python tools/check_links.py
```

## License and acknowledgements

This wiki is distributed under the [MIT License](LICENSE).

The notes were written while following the Inria MOOC [Machine learning in Python with scikit-learn](https://inria.github.io/scikit-learn-mooc/). Some explanations and code examples are adapted from the course material, which is distributed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The datasets belong to their respective authors.

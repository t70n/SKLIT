# Exploratory Data Analysis and Data Cleaning

[Home](../README.md)

This section explains how to understand a dataset before modeling and how to make it trustworthy: auditing data quality, handling missing values, studying distributions and correlations, and investigating outliers, without leaking information into the evaluation.

## Pages

### Overview

| Page | Summary |
| --- | --- |
| [EDA Overview](EDA-Overview.md) | A practical EDA workflow, questions to ask for every variable, first-pass reports |
| [Data Cleaning Overview](Data-Cleaning-Overview.md) | Recommended cleaning order, audit trail, final checklist |

### Data quality

| Page | Summary |
| --- | --- |
| [Data Types, Column Names, and Categories](Data-Types-and-Categories.md) | Inspecting and converting dtypes, cleaning names, nominal and ordinal categories |
| [Duplicates and Redundancy](Duplicates-and-Redundancy.md) | Exact and key-based duplicates, duplicate columns, merge-related duplication |
| [Constant and Redundant Features](Constant-and-Redundant-Features.md) | Constant, near-constant, duplicated, and highly correlated features |
| [Invalid Values and Constraints](Invalid-Values-and-Constraints.md) | Range checks, cross-column constraints, assertions, possible actions |
| [Identifiers, Leakage, Shuffling, and Sampling](Identifiers-Leakage-and-Sampling.md) | Identifier-like columns, target leakage, shuffling, sampling, splitting |

### Missing data

| Page | Summary |
| --- | --- |
| [Missing Values](Missing-Values.md) | Inspecting, dropping, filling, leakage-safe imputation, missingness indicators |
| [MICE and Iterative Imputation](MICE-and-Iterative-Imputation.md) | Chained equations, missingness assumptions, `IterativeImputer`, multiple imputation |

### Distributions and anomalies

| Page | Summary |
| --- | --- |
| [Correlation and Distributions](Correlation-and-Distributions.md) | Common distributions, Pearson, Spearman, and Kendall correlations, categorical associations |
| [Outlier Detection](Outlier-Detection.md) | IQR rule, z-scores, robust z-scores, multivariate detection, what to do with flagged values |

## Suggested reading order

Start with the two overview pages, then consult the other pages when the corresponding issue appears in your data.

## Navigation

- Previous section: [Pandas for Machine Learning](../02-Pandas/README.md)
- Next section: [Visualization](../04-Visualization/README.md)
- [Back to the home page](../README.md)

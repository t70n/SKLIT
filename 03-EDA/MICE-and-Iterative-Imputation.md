# MICE and Iterative Imputation

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## What MICE is

**MICE** means **Multiple Imputation by Chained Equations**. It fills missing values by modeling each incomplete variable from the other variables, cycling through the incomplete variables repeatedly.

MICE is useful when missing values are related to observed variables and a simple mean or median would destroy important relationships. It is an imputation framework rather than one single model: each variable can use a suitable predictive model.

The method has two ideas:

1. **Chained equations:** estimate one incomplete variable at a time using the others as predictors.
2. **Multiple imputation:** repeat the process with different plausible draws to represent uncertainty about the missing values.

A missing value is not known exactly. A good imputation method should represent that uncertainty instead of pretending that one guessed value is certain.

## The MICE algorithm

Suppose the dataset has variables $X_1,\ldots,X_p$, and some values are missing.

### 1. Preliminary imputation

Replace missing entries with simple placeholders, such as a mean, median, mode, or random draw. These values are only starting points and will be updated.

### 2. Cycle through incomplete variables

For each variable $X_j$ containing missing values:

- treat the current variable as the response $y=X_j$
- use the other variables as predictors $X_{-j}$
- use rows where $X_j$ is observed as training data
- fit a conditional model $p(X_j\mid X_{-j})$
- predict or draw replacement values for rows where $X_j$ is missing
- replace the current placeholders with the new values

Repeat this cycle for several iterations. The imputation for one variable becomes a predictor for the next variable in the same cycle.

### 3. Generate multiple datasets

Run the chained procedure with different random seeds, posterior draws, or model samples. This creates datasets $D_1,\ldots,D_m$, each containing a plausible version of the missing values.

## Mathematical intuition

For variable $X_j$, separate the rows into observed and missing parts:

$$
X_j=(X_{j,obs},X_{j,mis})
$$

At a step of the chain, estimate the conditional distribution:

$$
X_{j,mis}\sim p(X_j\mid X_{-j,obs},X_{-j,mis})
$$

The model is trained on rows where $X_j$ is observed. The other variables can include their current imputed values. For a continuous variable, the model may produce:

$$
X_j=\hat{f}_j(X_{-j})+\epsilon_j
$$

where $\epsilon_j$ represents residual uncertainty. A deterministic prediction uses only $\hat{f}_j$; a stochastic imputation draws from the predictive distribution and includes uncertainty.

For a binary variable, use a conditional classification model:

$$
P(X_j=1\mid X_{-j})=\sigma(f_j(X_{-j}))
$$

then draw a Bernoulli value. For a categorical variable, estimate class probabilities and draw a category from them.

After enough iterations, the chain can approach a stable distribution of plausible completed datasets. In practice, inspect convergence and compare the imputed values with observed values; convergence is not guaranteed for every dataset or model.

## Missingness assumptions

MICE is commonly discussed with these assumptions:

### Missing completely at random (MCAR)

The probability of missingness does not depend on observed or unobserved values:

$$
P(M\mid X_{obs},X_{mis})=P(M)
$$

This is strong and uncommon in real datasets.

### Missing at random (MAR)

After conditioning on observed variables, missingness does not depend on the missing value itself:

$$
P(M\mid X_{obs},X_{mis})=P(M\mid X_{obs})
$$

MICE is often justified under MAR when the variables related to missingness are included in the imputation models.

### Missing not at random (MNAR)

Missingness still depends on the unobserved value after conditioning on observed data. Standard MICE cannot identify this mechanism from the observed data alone. Sensitivity analysis or a domain-specific missingness model is needed.

These are assumptions about the data-generating process, not properties that the algorithm can prove.

## MICE versus simple imputation

| Method | Main idea | Strength | Limitation |
| --- | --- | --- | --- |
| Mean/median | Replace each column with one statistic | Fast and robust baseline | Shrinks variation and can break relationships |
| KNN imputation | Use nearby complete values | Uses local similarity | Sensitive to scaling and expensive for large data |
| Iterative imputation | Predict each incomplete feature from the others | Models relationships between variables | More computation and model assumptions |
| MICE | Iterative imputation with multiple plausible datasets | Represents imputation uncertainty | Requires multiple analyses and careful pooling |

Always compare MICE or iterative imputation with a simple baseline. More complexity does not guarantee better downstream performance.

## scikit-learn `IterativeImputer`

`IterativeImputer` is scikit-learn's Python implementation of iterative chained prediction. It is a plausible MICE-like equivalent, but important distinctions matter:

- by default it returns one completed dataset, not several pooled imputations
- by default it uses `BayesianRidge` as the estimator for numeric data
- `sample_posterior=True` enables stochastic posterior sampling for estimators that support it
- `max_iter` controls the number of imputation rounds
- the order in which features are imputed can be configured

The estimator is still flagged as experimental (including in scikit-learn 1.9), so enable it explicitly before importing:

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
```

## Basic numerical example

```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

X = np.array([
    [1.0, 10.0, np.nan],
    [2.0, 11.0, 20.0],
    [3.0, np.nan, 30.0],
    [4.0, 14.0, 40.0],
])

imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
)
X_imputed = imputer.fit_transform(X)
print(X_imputed)
```

`IterativeImputer` uses `np.nan` to represent missing values. A NumPy array cannot use a special `NA` value in the same way as pandas; use floating-point `NaN` or a pandas DataFrame with `pd.NA` converted appropriately.

## Important parameters

### `estimator`

The model used to predict each incomplete feature. The default Bayesian ridge model is appropriate for many numeric relationships, but alternatives can model nonlinear structure:

```python
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    ),
    max_iter=10,
    random_state=42,
)
```

The estimator must support the required prediction interface. More flexible estimators can improve fit but increase runtime and overfitting risk.

### `max_iter`

Maximum number of complete rounds through the features. Use `imputer.n_iter_` after fitting to inspect how many rounds were used.

### `sample_posterior`

When `False`, predictions are deterministic for a fixed fitted estimator. When `True`, draw from the predictive posterior. This requires an estimator that implements `predict(..., return_std=True)`, such as the default Bayesian ridge estimator.

```python
imputer = IterativeImputer(
    max_iter=20,
    sample_posterior=True,
    random_state=42,
)
```

Different random seeds can then create different plausible imputations.

### `initial_strategy`

The placeholder strategy before iterative updates: `"mean"`, `"median"`, `"most_frequent"`, or `"constant"`.

### `imputation_order`

Controls the order of features: `"ascending"`, `"descending"`, `"roman"`, `"arabic"`, or `"random"`. The order can affect the result when the chain has not converged.

### `min_value` and `max_value`

Constrain imputed values to valid ranges:

```python
imputer = IterativeImputer(
    min_value=0,
    max_value=np.inf,
    random_state=42,
)
```

Use domain constraints carefully. A bound prevents impossible outputs but does not fix a misspecified model.

### `add_indicator`

Add binary columns indicating which features had missing values:

```python
imputer = IterativeImputer(
    add_indicator=True,
    random_state=42,
)
```

Missingness can itself carry predictive information.

## Use it in a pipeline

Fit imputation inside the modeling pipeline so each cross-validation fold learns its own imputation models:

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    IterativeImputer(
        max_iter=10,
        random_state=42,
    ),
    StandardScaler(),
    Ridge(alpha=1.0),
)
```

For a classification model:

```python
from sklearn.linear_model import LogisticRegression

classifier = make_pipeline(
    IterativeImputer(max_iter=10, random_state=42),
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)
```

Do not call `fit_transform` on the complete dataset before `train_test_split` or cross-validation. That lets validation values influence the imputation models.

## Mixed numerical and categorical data

Iterative imputation is easiest for numerical variables. For mixed data, a practical approach is often:

- use `IterativeImputer` for numerical columns
- use `SimpleImputer(strategy="most_frequent")` or a constant category for categorical columns
- encode categories after imputation
- keep all branches in a `ColumnTransformer`

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_branch = make_pipeline(
    IterativeImputer(max_iter=10, random_state=42),
    StandardScaler(),
)
categorical_branch = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = ColumnTransformer([
    ("numeric", numeric_branch, numeric_columns),
    ("categorical", categorical_branch, categorical_columns),
])
```

One-hot encoded categories create many columns and can be awkward to impute iteratively. Do not treat arbitrary category codes as continuous numeric values. For categorical MICE, use a model that predicts categories or a specialized method.

## Multiple imputation in Python

To approximate multiple imputation with scikit-learn, fit several stochastic imputers with different seeds:

```python
imputers = [
    IterativeImputer(
        max_iter=20,
        sample_posterior=True,
        random_state=seed,
    )
    for seed in range(5)
]

imputed_datasets = [imputer.fit_transform(X) for imputer in imputers]
```

Then fit the analysis model separately on every completed dataset:

```python
from sklearn.base import clone

scores = []
for X_completed in imputed_datasets:
    fitted_model = clone(base_model).fit(X_completed, y)
    scores.append(fitted_model.score(X_completed, y))

print("mean:", np.mean(scores))
print("between-imputation std:", np.std(scores, ddof=1))
```

The example score is only illustrative. For honest evaluation, split or cross-validate each imputation/model workflow correctly, and do not fit imputers on the full dataset before evaluation.

For prediction, one practical pooling strategy is to average predictions from models trained on the separate imputations:

$$
\hat{p}(y\mid x)=\frac{1}{m}\sum_{k=1}^{m}\hat{p}_k(y\mid x)
$$

For statistical inference, use Rubin's rules to pool parameter estimates and uncertainty rather than simply averaging coefficients.

## Analysis and pooling

Multiple imputation has three stages:

1. **Imputation:** generate $m$ plausible completed datasets.
2. **Analysis:** fit the same statistical model or learner to every dataset.
3. **Pooling:** combine estimates, predictions, or uncertainty across analyses.

For an estimated parameter $\hat{\theta}_k$ with within-imputation variance $U_k$:

$$
\bar{\theta}=\frac{1}{m}\sum_{k=1}^{m}\hat{\theta}_k
$$

$$
\bar{U}=\frac{1}{m}\sum_{k=1}^{m}U_k
\qquad
B=\frac{1}{m-1}\sum_{k=1}^{m}(\hat{\theta}_k-\bar{\theta})^2
$$

The total variance is commonly estimated as:

$$
T=\bar{U}+\left(1+\frac{1}{m}\right)B
$$

$\bar{U}$ represents uncertainty within each completed dataset; $B$ represents variation between imputations. A single deterministic imputation hides the between-imputation component.

## Diagnostics

Compare observed and imputed values. A useful check is to temporarily mask observed values, impute them, and compare predictions with the known values:

```python
from sklearn.metrics import mean_absolute_error

# Conceptual validation: create artificial missingness only in a copy.
# Keep the original observed values as ground truth.
mae = mean_absolute_error(known_values, imputed_values)
print(mae)
```

Also inspect:

- distributions of observed and imputed values
- min/max and domain constraints
- relationships between imputed and observed features
- convergence across iterations
- sensitivity to estimator, order, and random seed
- downstream cross-validation performance
- whether imputed values become implausibly smooth or concentrated

`IterativeImputer` exposes information such as `imputation_sequence_`, `n_iter_`, and `initial_imputer_` after fitting.

## R MICE from Python

The R `mice` package is a mature option when you need its methods, diagnostics, or pooling conventions. Two common integration approaches are possible.

### Exchange files

1. Python writes a file containing missing values.
2. Python calls an R script from the command line.
3. R reads the file and runs `mice`.
4. R writes completed datasets or model results.
5. Python reads the output.

This is simple to debug and separates environments, but it adds file management, serialization, and process overhead.

### Use `rpy2`

1. Install R and the R `mice` package.
2. Install `rpy2` in the Python environment.
3. Use `numpy2ri` or pandas conversion helpers for arrays and DataFrames.
4. Load the R package with `importr`.
5. Convert the completed R objects back to NumPy or pandas.

The exact conversion code depends on R, Python, pandas, and `rpy2` versions. Keep the R and Python package versions documented and test missing-value types carefully.

NumPy uses `np.nan` for missing floating-point values; it does not have a universal `NA` value like R. Pandas supports nullable dtypes and `pd.NA`, but convert explicitly when passing arrays between Python and R.

## When MICE is useful

Consider MICE or iterative imputation when:

- missingness is substantial but predictors contain information about missing values
- preserving relationships among variables matters
- a simple median/mode baseline is inadequate
- you can afford repeated model fitting and diagnostics
- uncertainty from missing values matters to the analysis

Simple imputation may be preferable when missingness is very low, the model is tree-based and robust, or a transparent production pipeline is more important than a complex imputation model.

## Common mistakes

- fitting the imputer before the train/test split
- treating one deterministic `IterativeImputer` result as full multiple imputation
- using arbitrary integer category codes as continuous values
- ignoring impossible imputed values
- using future data in time-dependent imputation
- evaluating imputation only by reconstruction instead of downstream performance
- assuming MAR without considering why values are missing
- pooling coefficients by simple averaging when inferential uncertainty is required

## Related pages

- [Missing Values](Missing-Values.md)
- [Data Cleaning Overview](Data-Cleaning-Overview.md)
- [Pipeline](../05-Preprocessing/Pipeline.md)
- [ColumnTransformer](../05-Preprocessing/ColumnTransformer.md)
- [Classification Metrics](../08-Model-Evaluation/Classification-Metrics.md)

Official scikit-learn references:

- [Iterative imputer user guide](https://scikit-learn.org/stable/modules/impute.html)
- [Iterative imputer variants example](https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html)
- [Missing values example](https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html)

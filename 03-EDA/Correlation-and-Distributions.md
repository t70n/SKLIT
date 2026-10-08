# Correlation and Distributions

[Home](../README.md) / [Exploratory Data Analysis](README.md)

## Why study distributions and correlation together?

A distribution describes one variable at a time. Correlation describes how two variables vary together. They answer different questions:

- **marginal distribution:** what values does one variable take, and how often?
- **joint distribution:** which pairs of values occur together?
- **conditional distribution:** how does one variable behave for a given value or group of another?
- **correlation:** how strongly do two variables follow a selected type of association?

A correlation value is not determined by the two marginal distributions alone. Two datasets can have identical individual distributions but very different relationships. Always inspect a scatter plot or grouped plot alongside a coefficient.

## Basic distributions

### Bernoulli distribution

A Bernoulli variable has two outcomes, usually encoded as 0 and 1:

$$
X\sim\operatorname{Bernoulli}(p),\qquad P(X=1)=p
$$

Its mean is $p$ and variance is $p(1-p)$. Examples include a purchase flag, a disease indicator, or a binary prediction.

For two binary variables, Pearson correlation is the **phi coefficient** and can be calculated with `df.corr()`. It is mathematically the same as the Pearson correlation of the 0/1 encodings, but do not encode arbitrary nominal categories as numbers and interpret their Pearson correlation.

### Categorical distribution

A categorical variable takes one value from a finite set of categories with probabilities $p_1,\ldots,p_K$:

$$
P(X=k)=p_k,\qquad\sum_{k=1}^{K}p_k=1
$$

Nominal categories have no numeric order. Use counts, proportions, contingency tables, grouped target rates, mutual information, or an appropriate categorical association measure rather than `df.corr()` on arbitrary integer labels.

For two categorical variables, common association measures include Cramer's $V$ and Theil's $U$. A chi-square test can assess whether independence is plausible, but statistical significance is not the same as a strong practical association.

### Uniform distribution

A continuous uniform variable gives equal density between $a$ and $b$:

$$
f(x)=\frac{1}{b-a},\qquad a\leq x\leq b
$$

Its mean is $(a+b)/2$ and variance is $(b-a)^2/12$. A histogram should look approximately flat for a large random sample, subject to sampling noise.

### Normal distribution

The normal distribution is symmetric and bell-shaped:

$$
X\sim\mathcal{N}(\mu,\sigma^2),\qquad
f(x)=\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

The mean, median, and mode coincide at $\mu$. The standard deviation $\sigma$ controls spread. Pearson correlation is especially natural for linear relationships, but normal-looking marginals do not guarantee a linear relationship or a meaningful Pearson coefficient.

### Lognormal distribution

If $\log(X)$ is normal, then $X$ is lognormal:

$$
\log(X)\sim\mathcal{N}(\mu,\sigma^2),\qquad X>0
$$

It is positive and right-skewed, making it useful for incomes, prices, and durations. A log transformation can make the distribution easier to model and can reduce the influence of extreme values. Compare correlations before and after transformation when the scientific question supports it.

### Exponential distribution

The exponential distribution models a non-negative waiting time:

$$
f(x)=\lambda e^{-\lambda x},\qquad x\geq0
$$

Its mean and standard deviation are both $1/\lambda$. It is strongly right-skewed. Pearson correlation can be sensitive to large values, so inspect Spearman correlation and scatter plots as alternatives when the relationship is monotonic rather than linear.

### Poisson distribution

The Poisson distribution models a count of events in a fixed interval:

$$
P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!},\qquad k=0,1,2,\ldots
$$

Its mean and variance are both $\lambda$. Count variables are discrete and often skewed. Correlation can still be computed for numeric counts, but consider whether a count model, rate, exposure adjustment, or transformation is more appropriate.

### Binomial distribution

A binomial variable counts successes in $n$ independent Bernoulli trials:

$$
X\sim\operatorname{Binomial}(n,p),\qquad
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$

Its mean is $np$ and variance is $np(1-p)$. It is bounded between 0 and $n$, so correlation with another variable may reflect the bound as well as the underlying relationship.

## Correlation coefficients

### Pearson correlation

Pearson correlation measures linear association between numeric variables $X$ and $Y$:

$$
r_{XY}=\frac{\operatorname{cov}(X,Y)}{s_Xs_Y}
=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_i(x_i-\bar{x})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}
$$

Its value lies between $-1$ and $1$:

- $1$: perfect increasing linear relationship
- $-1$: perfect decreasing linear relationship
- $0$: no linear association, although nonlinear association may still exist

Pearson correlation is sensitive to outliers and can be near zero for a strong curved relationship.

### Spearman rank correlation

Spearman correlation is Pearson correlation applied to ranks rather than raw values:

$$
\rho_s=\operatorname{corr}(\operatorname{rank}(X),\operatorname{rank}(Y))
$$

It measures monotonic association. It can detect a consistently increasing or decreasing nonlinear relationship and is less sensitive to scale and some outliers than Pearson correlation. It is not immune to outliers or ties.

### Kendall rank correlation

Kendall's $\tau$ compares concordant and discordant pairs:

$$
\tau=\frac{C-D}{\text{number of comparable pairs}}
$$

A pair is concordant when the ordering of $X$ and $Y$ agrees, and discordant when it disagrees. Kendall's measure is often useful for small samples or ordinal data, though it can be slower for very large datasets.

## Which correlation should I use?

| Data and relationship | First choice | Why |
| --- | --- | --- |
| Numeric and approximately linear | Pearson | Directly measures linear association |
| Numeric or ordinal and monotonic but nonlinear | Spearman | Uses ranks and captures monotonic trends |
| Small sample or ordinal measurements | Kendall | Pairwise concordance interpretation |
| Two binary 0/1 variables | Pearson/phi | Valid for the binary encoding |
| Nominal categories | Cramer's $V$, mutual information, or contingency analysis | Integer labels have no numeric meaning |
| Numerical variable and binary target | Point-biserial correlation or group comparison | Makes the binary coding assumption explicit |
| Numerical variable and nominal multiclass target | Grouped plots, ANOVA, mutual information, or model-based analysis | One correlation is not enough |

There is no universal threshold for a "strong" correlation. Interpretation depends on the field, measurement noise, sample size, and consequences of the decision.

## Calculate a correlation matrix with pandas

```python
numeric_df = df.select_dtypes(include="number")

pearson_corr = numeric_df.corr(method="pearson")
spearman_corr = numeric_df.corr(method="spearman")
kendall_corr = numeric_df.corr(method="kendall")

print(pearson_corr)
```

`DataFrame.corr()` returns a symmetric matrix with 1 on the diagonal. By default it uses pairwise complete observations: each pair can be calculated using a different subset after missing values are excluded. This is convenient, but the effective sample size can differ between cells.

Use `min_periods` to require enough paired observations:

```python
corr = numeric_df.corr(method="spearman", min_periods=30)
```

For two columns, `Series.corr()` is convenient:

```python
r = df["income"].corr(df["hours_per_week"], method="pearson")
```

Correlation measures association, not causation. A confounder, selection effect, time trend, or shared measurement process can produce a high value.

## Visualize the matrix

### `plt.matshow`

```python
import matplotlib.pyplot as plt

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
plt.tight_layout()
plt.show()
```

The fixed range `[-1, 1]` makes colors comparable across plots. Without labels and a colorbar, a matrix plot is difficult to interpret.

### Seaborn heatmap

```python
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
)
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()
```

For many variables, mask the duplicate upper triangle:

```python
import numpy as np

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0)
```

## Find the strongest relationships automatically

```python
import numpy as np

corr = numeric_df.corr(method="spearman")

pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .rename("correlation")
    .reset_index()
    .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
)

pairs["absolute_correlation"] = pairs["correlation"].abs()
strongest = pairs.sort_values("absolute_correlation", ascending=False)
print(strongest.head(10))
```

The triangular mask avoids reporting both $(X,Y)$ and $(Y,X)$ and removes the diagonal self-correlations.

## Reusable correlation report

This helper produces both the matrix and a sorted table of unique feature pairs:

```python
import numpy as np


def correlation_report(df, method="spearman", min_periods=1):
    numeric_df = df.select_dtypes(include="number")
    matrix = numeric_df.corr(method=method, min_periods=min_periods)

    upper = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    pairs = (
        matrix.where(upper)
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    )
    pairs["absolute_correlation"] = pairs["correlation"].abs()
    pairs = pairs.sort_values("absolute_correlation", ascending=False)
    return matrix, pairs


corr_matrix, correlation_pairs = correlation_report(df, method="spearman")
print(correlation_pairs.head(10))
```

This function only considers numeric columns. It intentionally does not silently convert categorical strings or arbitrary category codes into numbers.

## Plot the relationship, not just the coefficient

```python
import seaborn as sns
import matplotlib.pyplot as plt

x_column = "income"
y_column = "hours_per_week"

sns.scatterplot(data=df, x=x_column, y=y_column, alpha=0.4)
sns.regplot(data=df, x=x_column, y=y_column, scatter=False, color="red")
plt.title(
    f"Pearson r = {df[x_column].corr(df[y_column]):.2f}"
)
plt.show()
```

Use `sns.pairplot()` for a small set of numerical variables. Use hexbin plots or sample the data when a scatter plot is overcrowded. A coefficient can hide clusters, curves, heteroscedasticity, or outliers.

## Correlation with a target

For a numerical regression target:

```python
target_correlations = (
    df.select_dtypes(include="number")
    .corr(method="spearman")["target"]
    .drop("target")
    .sort_values(key=lambda values: values.abs(), ascending=False)
)
print(target_correlations)
```

For a binary target encoded as 0 and 1, Pearson correlation is the point-biserial correlation under the standard coding. For a categorical target, compare target distributions by group or use mutual information rather than assigning arbitrary integers to categories.

For two nominal categorical variables, Cramer's $V$ is one option:

$$
V=\sqrt{\frac{\chi^2/n}{\min(k-1,r-1)}}
$$

where $r$ and $k$ are the numbers of rows and columns in the contingency table. A simple implementation is:

```python
from scipy.stats import chi2_contingency


def cramers_v(first, second):
    table = pd.crosstab(first, second)
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    rows, columns = table.shape
    return np.sqrt((chi2 / n) / min(columns - 1, rows - 1))


association = cramers_v(df["workclass"], df["education"])
print(association)
```

Cramer's $V$ is between 0 and 1, but its practical interpretation depends on the table size and context. It is an association measure, not a causal effect.

Correlation-based feature filtering must be performed inside the training process when it uses the target. Otherwise, the test target influences feature selection and the evaluation becomes optimistic.

## Missing values and preprocessing

Before calculating correlation, inspect missingness:

```python
missing = df.isna().sum().sort_values(ascending=False)
print(missing)
```

`df.corr()` excludes missing pairs by default, but imputation can change correlations and should be justified. Do not blindly impute before EDA if the missingness pattern itself is informative.

Do not standardize solely to make Pearson correlation possible: Pearson correlation is invariant to separate positive linear rescaling. Scaling is still important for distance-based models, regularized models, and visual comparability.

## Correlation is not causation

A high correlation can arise from:

- a direct causal relationship
- a common confounder
- reverse causality
- selection bias
- a shared time trend
- data leakage or duplicated information

A low Pearson correlation does not rule out a nonlinear or conditional relationship. Use domain knowledge, plots, stratification, experiments, or appropriate causal methods before making causal claims.

## EDA checklist

- inspect each variable's type and support
- plot numerical distributions and category counts
- calculate Pearson and rank correlations where appropriate
- visualize the strongest and most surprising pairs
- check outliers and influential observations
- check whether missingness is associated with variables or the target
- distinguish redundant predictors from useful conditional predictors
- check interactions and subgroup behavior
- avoid treating arbitrary category codes as measurements
- repeat target-based decisions inside cross-validation

## Related pages

- [Outlier Detection](Outlier-Detection.md)
- [Constant and Redundant Features](Constant-and-Redundant-Features.md)
- [Scatter Plots](../04-Visualization/Scatter-Plots.md)
- [Pairplots](../04-Visualization/Pairplots.md)
- [pandas API reference: DataFrame.corr](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)

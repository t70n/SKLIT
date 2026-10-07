# Support Vector Regression (SVR)

[Home](../../README.md) / [Models](../README.md) / [Support Vector Machines](../README.md#support-vector-machines)

## Idea

Support Vector Regression adapts the maximum-margin idea of SVMs to continuous targets. Instead of trying to pass through every training point, it fits a function with a tube of tolerance around it. Errors inside the tube are ignored; points outside the tube influence the model and become support vectors.

This is called epsilon-insensitive regression.

## Import

```python
from sklearn.svm import SVR
```

## Minimal example

```python
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"),
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(mean_absolute_error(y_test, pred))
```

## How it works mathematically

For a linear function $f(x)=w^\top x+b$, SVR uses an epsilon-insensitive loss:

$$
L_\epsilon(y,f(x))=\max(0,|y-f(x)|-\epsilon)
$$

Residuals with absolute value at most $\epsilon$ have zero loss. The optimization problem is:

$$
\min_{w,b,\xi,\xi^*}
\frac{1}{2}\|w\|^2+C\sum_i(\xi_i+\xi_i^*)
$$

subject to:

$$
\begin{aligned}
y_i-(w^\top x_i+b)&\leq\epsilon+\xi_i\\
(w^\top x_i+b)-y_i&\leq\epsilon+\xi_i^*\\
\xi_i,\xi_i^*&\geq0
\end{aligned}
$$

The norm term prefers a smoother, flatter function. The slack variables measure errors outside the epsilon tube, and `C` controls how strongly those errors are penalized.

## Kernel intuition

With a kernel, SVR can fit a linear function in an implicit feature space:

$$
f(x)=\sum_{i\in SV}(\alpha_i-\alpha_i^*)K(x_i,x)+b
$$

Only support vectors have non-zero influence. Common kernels include:

- `"linear"`: a linear relationship in the original features
- `"poly"`: polynomial interactions
- `"rbf"`: smooth local influence based on distance
- `"sigmoid"`: sigmoid-shaped similarity

The RBF kernel is:

$$
K(x_i,x_j)=\exp(-\gamma\|x_i-x_j\|^2)
$$

Large `gamma` creates short-range, highly flexible effects; small `gamma` creates smoother effects. SVR therefore has a useful but sensitive pair of complexity controls: `C` and `gamma`.

## Important parameters

### `C`

Penalty for points outside the epsilon tube. Large `C` fits training data more aggressively and can overfit; small `C` prefers a smoother function with more tolerated error.

### `epsilon`

Width of the no-penalty tube. Larger values ignore more small errors and produce fewer support vectors, but can underfit. It should reflect the target precision that matters in the application.

### `kernel`, `gamma`, and `degree`

These determine the feature-space geometry. `degree` controls polynomial kernels; `gamma` controls RBF, polynomial, and sigmoid locality. `gamma="scale"` uses feature variance to choose a data-dependent default.

### `shrinking` and `cache_size`

These are mostly optimization and memory controls. `max_iter` can impose a stopping limit.

## Preprocessing

SVR is sensitive to feature scales because kernels and distances depend on them. Scale numerical features inside a pipeline. Encode categorical variables before SVR; it does not natively understand categories.

Unlike tree models, SVR usually becomes expensive as the number of training samples grows. For large datasets, consider `LinearSVR`, kernel approximations, or tree-based regressors.

## When to use it

Use SVR when:

- the dataset is small or medium-sized
- a smooth nonlinear relationship is plausible
- accurate regression matters more than simple coefficient interpretation
- features can be scaled and encoded carefully

Use Linear Regression or Ridge for a transparent additive baseline. Use a tree ensemble when there are many categorical variables, discontinuities, or large tabular datasets.

## Evaluation and tuning

Tune `C`, `epsilon`, and `gamma` together with cross-validation. A useful scoring choice depends on the cost of errors:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    model,
    param_grid={
        "svr__C": [0.1, 1, 10],
        "svr__epsilon": [0.05, 0.1, 0.5],
        "svr__gamma": ["scale", "auto"],
    },
    scoring="neg_mean_absolute_error",
    cv=5,
)
```

The `neg_` prefix follows scikit-learn's convention that larger scores are better. The reported value is the negative MAE, so negate it to recover the usual positive error.

## Related pages

- [SVC](SVC.md)
- [Linear Regression](../Linear-Models/Linear-Regression.md)
- [Regression Metrics](../../08-Model-Evaluation/Regression-Metrics.md)
- [scikit-learn API reference: SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

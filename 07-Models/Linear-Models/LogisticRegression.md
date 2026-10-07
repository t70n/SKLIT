# LogisticRegression

[Home](../../README.md) / [Models](../README.md) / [Linear Models](../README.md#linear-models)

## Idea

Logistic regression models the probability of each class using a linear combination of the input features.

It is one of the most common baseline classifiers and is especially useful when you want a simple interpretable model.

## Import

```python
from sklearn.linear_model import LogisticRegression
```

## Minimal example

```python
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)
```

## How it works mathematically

For binary classification, logistic regression first computes a linear score:

$$
z=w^\top x+b
$$

The sigmoid converts that score into a probability:

$$
P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}
$$

The default decision threshold is usually $0.5$, equivalent to the linear boundary $w^\top x+b=0$. The model is linear in feature space even though the probability transformation is nonlinear.

It learns by minimizing binary cross-entropy (log loss), by default with an L2 penalty. scikit-learn writes the objective as:

$$
J(w,b)=\frac{1}{2}\|w\|_2^2-C\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

This is equivalent to adding a penalty $\frac{\lambda}{2}\|w\|_2^2$ to the log loss with $\lambda=1/C$: `C` is the inverse of the regularization strength. The intercept $b$ is not penalized. The penalty discourages large coefficients.

The log-odds are linear:

$$
\log\frac{P(y=1\mid x)}{1-P(y=1\mid x)}=w^\top x+b
$$

Increasing feature $j$ by one unit changes the log-odds by $w_j$, holding other features fixed.

## Key parameters

### `C`

Regularization strength.

- small `C`: stronger regularization
- large `C`: weaker regularization

### `max_iter`

Maximum number of optimization iterations (100 by default). A `ConvergenceWarning` means that the solver stopped before converging: scale the features first, then increase `max_iter` if needed.

### Penalty type: `l1_ratio` (and the former `penalty`)

- `l1_ratio=0.0` (default): L2 penalty, which shrinks coefficients without setting them to zero
- `l1_ratio=1.0`: L1 penalty, which can set coefficients exactly to zero (sparse model)
- `0 < l1_ratio < 1`: Elastic Net, a mix of both
- `C=np.inf`: no penalty

In scikit-learn 1.8 and later, the penalty type is selected with `l1_ratio`; the former `penalty` argument (`"l2"`, `"l1"`, `"elasticnet"`, `None`) is deprecated and will be removed in version 1.10. Code written for earlier versions uses `penalty="l1"` instead of `l1_ratio=1.0`.

### `solver`

The solver performs the numerical optimization. `lbfgs` is a strong general default; `liblinear` is useful for small binary problems; `saga` supports large sparse problems and L1 or Elastic Net penalties. Not every solver supports every penalty, so check their compatibility.

### `class_weight`

`class_weight="balanced"` reweights the samples inversely to the class frequencies, which increases the recall of the minority class on imbalanced data, usually at the cost of precision. See [Classification Metrics](../../08-Model-Evaluation/Classification-Metrics.md#class-imbalance).

## Why scaling matters

This is a gradient-based model. Standardizing features often speeds up convergence and improves optimization.

Scaling also makes regularization act more comparably across features. It does not make the decision boundary nonlinear.

## Outputs and multiclass classification

- `predict_proba` returns class probabilities.
- `decision_function` returns the signed linear score(s).
- `coef_` and `intercept_` expose the fitted parameters.
- Multiclass problems are fitted with the multinomial (softmax) loss by the default `lbfgs` solver. The former `multi_class` parameter has been deprecated and removed in recent versions; wrap the model in `OneVsRestClassifier` when a one-versus-rest strategy is explicitly required.

## Decision boundaries and probability visualization

With two features, the boundary $w_1x_1+w_2x_2+b=0$ is a line. `DecisionBoundaryDisplay.from_estimator` can show either the predicted class regions or the continuous probability surface:

```python
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
	model,
	X_test,
	response_method="predict_proba",
	alpha=0.5,
)
```

Use `response_method="predict"` to display hard regions or `"predict_proba"` to display how confidence changes across feature space. These probabilities are estimates, not guarantees: a model can classify accurately while being overconfident or underconfident. Assess calibration separately when probabilities drive decisions.

When the model is the final step of a pipeline, access it by name or position, for example `pipeline[-1].coef_`. For binary classification, the coefficient array commonly has shape `(1, n_features)`; `pipeline[-1].coef_.ravel()` converts it to a one-dimensional vector for plotting.

## Use cases

- baseline classifier
- fast and interpretable linear model
- when you want to understand feature importance qualitatively

## Important note

The model assumes a linear decision boundary in feature space. This can be limiting if the true relation is highly nonlinear.

## Related pages

- [Linear Regression](Linear-Regression.md)
- [Logistic Regression Decision Boundaries](../../10-Recipes/Logistic-Regression-Decision-Boundaries.md)
- [Adult Census Classification](../../10-Recipes/Adult-Census-Classification.md)
- [Classification Metrics](../../08-Model-Evaluation/Classification-Metrics.md)
- [scikit-learn API reference: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# SVC and SVMs

[Home](../../README.md) / [Models](../README.md) / [Support Vector Machines](../README.md#support-vector-machines)

## Idea

Support Vector Machines construct a decision boundary that maximizes the margin between classes.

With an RBF kernel, the model can capture nonlinear decision boundaries.

## Import

```python
from sklearn.svm import SVC
```

## Maximum-margin intuition

For labels $y_i\in\{-1,+1\}$, a linear SVM searches for a separating hyperplane $w^\top x+b=0$. The margin is the distance from the boundary to the nearest training points, equal to $2/\|w\|$ when the closest points satisfy $y_i(w^\top x_i+b)=1$. Maximizing this margin tends to improve generalization.

Real data is rarely perfectly separable, so the soft-margin formulation introduces slack variables:

$$
\min_{w,b,\xi}\frac{1}{2}\|w\|^2+C\sum_i\xi_i
$$

subject to:

$$
y_i(w^\top x_i+b)\geq 1-\xi_i,\qquad \xi_i\geq 0
$$

`C` controls the trade-off. High `C` penalizes margin violations strongly and can fit noise; low `C` permits violations in exchange for a wider margin. The points that determine the boundary are the support vectors.

## Minimal example

```python
model = SVC(kernel="rbf", gamma="scale")
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)
```

## Important parameters

- `kernel`: `"linear"`, `"poly"`, `"rbf"`, or `"sigmoid"`
- `C`: penalty for margin violations
- `gamma`: locality of RBF, polynomial, and sigmoid kernels
- `degree` and `coef0`: shape controls for polynomial and sigmoid kernels
- `class_weight`: changes the penalty for selected classes
- `probability`: enables an additional probability-calibration step

## Important parameter: `gamma`

`gamma` controls how much influence each training sample has on the decision boundary.

### High gamma

- narrow influence
- complex boundary
- high variance
- risk of overfitting

### Low gamma

- wide influence
- smoother boundary
- higher bias
- risk of underfitting

## Relationship with kernel

The RBF kernel is:

$$
K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)
$$

This means the distance between samples is converted into similarity values for the model.

For RBF, large `gamma` means a short length scale: only nearby points influence one another. Small `gamma` means a long length scale and a smoother boundary. `decision_function` exposes the signed margin score, while `support_vectors_` exposes the influential training points. `predict_proba` is only available with `probability=True`; probabilities are calibrated estimates rather than part of the basic margin objective.

SVC handles multiclass classification by combining binary classifiers (one-versus-one). Kernel SVC can be expensive in memory and fitting time as the number of training samples grows (roughly between quadratic and cubic in the number of samples), so `LinearSVC`, `SGDClassifier`, or a [Nystroem](../../06-Feature-Engineering/Nystroem-Kernel-Approximation.md) approximation followed by a linear model may be preferable for large datasets.

## Important note

SVMs are often sensitive to scaling. You usually want to standardize numerical features before fitting, inside a pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
```

`C` and `gamma` interact and should be tuned together, typically on logarithmic grids; see [Nested Cross-Validation with SVC](../../10-Recipes/Nested-Cross-Validation-with-SVC.md).

For continuous targets, the corresponding epsilon-insensitive method is [Support Vector Regression](SVR.md). It uses the same kernel ideas but fits a tolerance tube around a regression function instead of a separating boundary.

## Related pages

- [SVR](SVR.md)
- [Validation Curves](../../08-Model-Evaluation/Validation-Curves.md)
- [Nested Cross-Validation with SVC](../../10-Recipes/Nested-Cross-Validation-with-SVC.md)
- [scikit-learn user guide: Support vector machines](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn API reference: SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

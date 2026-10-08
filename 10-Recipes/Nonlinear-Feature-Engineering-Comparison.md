# Comparing Nonlinear Feature-Engineering Pipelines

[Home](../README.md) / [Recipes](README.md)

## Goal

Compare a linear Ridge pipeline with a nonlinear pipeline that combines preprocessing, splines, Nystroem kernel features, and Ridge regularization. Models with different inductive biases are compared on exactly the same cross-validation folds, so that the comparison is paired.

## Build the models

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

alphas = np.logspace(-3, 3, num=101)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_columns,
    ),
])

linear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("ridge", RidgeCV(alphas=alphas)),
])

nonlinear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("spline", SplineTransformer(n_knots=5, degree=3)),
    ("nystroem", Nystroem(
        kernel="poly",
        degree=2,
        n_components=300,
        random_state=42,
    )),
    ("ridge", RidgeCV(alphas=alphas)),
])
```

The order of transformations matters. A spline transforms each feature independently, while Nystroem can create a kernel feature map that captures interactions in the resulting representation.

## Evaluate with the same folds

```python
from sklearn.model_selection import KFold

cv = KFold(n_splits=10, shuffle=True, random_state=42)

linear_cv = cross_validate(
    linear_pipeline,
    data,
    target,
    cv=cv,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    n_jobs=2,
)

nonlinear_cv = cross_validate(
    nonlinear_pipeline,
    data,
    target,
    cv=cv,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    n_jobs=2,
)

linear_mse = -linear_cv["test_score"]
nonlinear_mse = -nonlinear_cv["test_score"]

print("Linear mean MSE:", linear_mse.mean())
print("Nonlinear mean MSE:", nonlinear_mse.mean())
print("Nonlinear wins:", (nonlinear_mse < linear_mse).sum(), "of", len(linear_mse))
```

The negative scorer is converted back to ordinary MSE. Compare mean, spread, and paired fold differences rather than only counting wins.

## Interpret the result

- If train error decreases but test error increases, the nonlinear pipeline may overfit.
- If both train and test error improve, the generated representation likely captures useful structure.
- If neither improves, the extra flexibility may be unnecessary or poorly tuned.
- A small mean difference with large fold variation is weak evidence.

Use `RidgeCV` inside each outer training fold so regularization is tuned without using the outer test fold.

## Reduce complexity when necessary

Tune these parameters:

- `spline__n_knots`
- `spline__degree`
- `nystroem__n_components`
- `nystroem__gamma`
- `ridge__alphas`

For large feature spaces, reduce `n_components`, use fewer knots, or skip the spline stage. Always measure runtime and memory as well as predictive performance.

## Leakage checks

- Keep `ColumnTransformer`, scaling, spline fitting, Nystroem sampling, and RidgeCV inside the evaluated pipeline.
- Do not fit preprocessing on the full dataset before cross-validation.
- Use time-aware or group-aware folds when rows are dependent.
- Do not compare models on different folds.
- Keep the final test set untouched until the workflow is selected.

## Related pages

- [Feature Engineering Overview](../06-Feature-Engineering/Feature-Engineering-Overview.md)
- [SplineTransformer](../06-Feature-Engineering/SplineTransformer.md)
- [Nystroem Kernel Approximation](../06-Feature-Engineering/Nystroem-Kernel-Approximation.md)
- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)

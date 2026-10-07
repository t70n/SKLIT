# Logistic Regression Regularization and Decision Boundaries

[Home](../README.md) / [Recipes](README.md)

## Goal

Compare logistic-regression regularization strengths by plotting the predicted probability surface and the $0.5$ decision boundary. This is useful for a two-feature dataset where the geometry can be visualized.

## Why `C` matters

Logistic Regression uses a penalty controlled by `C`:

- small `C`: strong regularization and a simpler boundary
- large `C`: weak regularization and a closer fit to training data

`C=1.0` is the default. Increasing `C` decreases regularization strength.

## Reusable plotting function

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay


def plot_decision_boundary(model, X_train, y_train, X_test, y_test,
                           x_columns):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    C = model.get_params()["logisticregression__C"]

    display = DecisionBoundaryDisplay.from_estimator(
        model,
        X_train,
        response_method="predict_proba",
        plot_method="pcolormesh",
        cmap="RdBu_r",
        alpha=0.8,
        vmin=0.0,
        vmax=1.0,
        xlabel=x_columns[0],
        ylabel=x_columns[1],
    )

    DecisionBoundaryDisplay.from_estimator(
        model,
        X_train,
        response_method="predict_proba",
        plot_method="contour",
        levels=[0.5],
        linestyles="--",
        linewidths=1,
        colors="black",
        ax=display.ax_,
    )

    sns.scatterplot(
        x=X_train[x_columns[0]],
        y=X_train[x_columns[1]],
        hue=y_train,
        palette=["tab:blue", "tab:red"],
        ax=display.ax_,
    )
    display.ax_.set_title(f"C={C}; test accuracy={accuracy:.2f}")
    display.ax_.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
```

The background color represents the estimated probability for the positive class. With `vmin=0` and `vmax=1`, probability `0.5` is centered in the diverging color scale. Darker colors indicate probabilities closer to 0 or 1, not guaranteed correctness.

## Compare several regularization values

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for C in [0.01, 0.1, 1.0, 100.0]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000),
    )
    plot_decision_boundary(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        x_columns=["feature_1", "feature_2"],
    )
    plt.show()
```

The pipeline must use exactly two features for a 2D boundary plot. For more features, use metrics, partial dependence, or coefficient inspection instead of pretending the full model is two-dimensional.

## Select `C` with cross-validation

```python
from sklearn.model_selection import GridSearchCV

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

search = GridSearchCV(
    model,
    param_grid={"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="accuracy",
)
search.fit(X, y)
print(search.best_params_)
print(search.best_score_)
```

Do not choose `C` from the final test set. Use cross-validation or a validation set, then evaluate once on untouched test data.

## Interpretation cautions

- A smooth boundary can still have poorly calibrated probabilities.
- Training-set visualizations can hide generalization problems.
- The $0.5$ threshold is not always the best threshold for imbalanced or cost-sensitive classification.
- Scaling is part of the model and must be fitted within validation folds.
- Feature engineering can make a boundary nonlinear in the original variables while the classifier remains linear in the transformed space.

## Related pages

- [LogisticRegression](../07-Models/Linear-Models/LogisticRegression.md)
- [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md)
- [GridSearchCV](../09-Hyperparameter-Tuning/GridSearchCV.md)

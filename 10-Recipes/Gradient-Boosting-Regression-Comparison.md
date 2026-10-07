# Gradient Boosting Regression Comparison

[Home](../README.md) / [Recipes](README.md)

## Goal

Compare classic gradient boosting and Random Forest regression on California housing using cross-validation, MAE, and timing.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
y = y * 100  # target in k$

models = {
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, random_state=42
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=200, n_jobs=2, random_state=42
    ),
}

for name, model in models.items():
    result = cross_validate(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_jobs=2,
    )
    scores = result["test_score"]
    print(name)
    print(f"MAE: {-scores.mean():.3f} +/- {scores.std():.3f} k$")
    print(f"fit time: {result['fit_time'].mean():.3f} s")
    print(f"score time: {result['score_time'].mean():.3f} s")
```

The `neg_` prefix follows scikit-learn's higher-is-better convention, so negate it to report MAE.

## Early stopping

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_iter_no_change=5,
    validation_fraction=0.1,
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("test MAE:", mean_absolute_error(y_test, pred))
print("trees used:", model.n_estimators_)
```

Keep the final test set untouched while early stopping and model selection use training data.

## Related pages

- [GradientBoostingRegressor](../07-Models/Ensembles/GradientBoostingRegressor.md)
- [Random Forest](../07-Models/Ensembles/Random-Forest.md)
- [Boosting Hyperparameter Tuning](../09-Hyperparameter-Tuning/Boosting-Hyperparameter-Tuning.md)

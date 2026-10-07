# Ames Housing: Linear Model vs Decision Tree

[Home](../README.md) / [Recipes](README.md)

## Goal

Compare a linear model and a decision tree on the Ames housing data, fold by fold:

1. a standardized linear regression versus a default decision tree on the numerical features
2. the same comparison after tuning the depth of the tree with nested cross-validation
3. the stability of the conclusion across random seeds
4. the effect of adding the categorical features to the tree with an `OrdinalEncoder`

## Dataset

`ames_housing_no_missing.csv` from the scikit-learn MOOC repository: a version of the Ames housing data without missing values. The target `SalePrice` is the sale price of a house in dollars.

```python
import pandas as pd

ames_housing = pd.read_csv(
    "../datasets/ames_housing_no_missing.csv",
    na_filter=False,  # keep "None" and "NA" as categories instead of missing values
)
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]
```

`na_filter=False` prevents pandas from interpreting category names such as `"None"` (no masonry veneer) or `"NA"` (for example no alley access) as missing values. It is required with pandas 2.0 and later, where `"None"` belongs to the default missing-value markers of `read_csv`.

## Select the numerical features

```python
numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]
```

## Step 1: linear model versus default tree

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

linear_model = make_pipeline(StandardScaler(), LinearRegression())
tree = DecisionTreeRegressor(random_state=0)

cv_linear = cross_validate(linear_model, data_numerical, target, cv=10)
cv_tree = cross_validate(tree, data_numerical, target, cv=10)

scores_linear = cv_linear["test_score"]
scores_tree = cv_tree["test_score"]

for fold, (score_linear, score_tree) in enumerate(zip(scores_linear, scores_tree), start=1):
    print(f"Fold {fold}: linear R2={score_linear:.4f} | tree R2={score_tree:.4f}")

print("Tree better than linear model on", (scores_tree > scores_linear).sum(), "of 10 folds")
```

`cv=10` uses the same `KFold(10)` splits for both models (no shuffling), so the comparison is paired. The default score of a regressor is $R^2$.

Counting with a vectorized comparison avoids a classic error in loops: `nb =+ 1` assigns `+1` to `nb` instead of incrementing it; the increment is written `nb += 1`.

## Step 2: tune the depth of the tree with nested cross-validation

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": range(1, 16)}
tree_search = GridSearchCV(tree, param_grid=param_grid, cv=10)

tree_search.fit(data_numerical, target)
print("Best depth on the full dataset:", tree_search.best_params_)

nested_cv_results = cross_validate(tree_search, data_numerical, target, cv=10)
print(f"Nested R2: {nested_cv_results['test_score'].mean():.3f}")
```

The outer `cross_validate` refits the whole search in each outer training fold, so the outer test folds are never used to choose `max_depth`. `best_params_` from the search on the full dataset is useful for inspection, but the generalization estimate comes from the nested scores. See [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md).

## Step 3: check the stability across random seeds

The tree breaks ties between equally good splits at random, so its predictions depend on `random_state`. Repeating the comparison with several seeds shows whether the conclusion is robust:

```python
for random_state in range(3):
    tree = DecisionTreeRegressor(random_state=random_state)
    tree_search = GridSearchCV(tree, param_grid={"max_depth": range(1, 16)}, cv=10)

    scores_linear = cross_validate(linear_model, data_numerical, target, cv=10)["test_score"]
    scores_tree = cross_validate(tree_search, data_numerical, target, cv=10)["test_score"]

    wins = (scores_tree > scores_linear).sum()
    print(f"random_state={random_state}: tuned tree better on {wins} of 10 folds")
```

## Step 4: add the categorical features

Trees can use integer-coded categories, so the categorical columns are ordinal-encoded and the numerical columns pass through unchanged:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

categorical_features = data.select_dtypes(include=["object", "string"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_features,
        )
    ],
    remainder="passthrough",
)

for random_state in range(3):
    tree_numerical = DecisionTreeRegressor(max_depth=7, random_state=random_state)
    tree_all_features = make_pipeline(
        preprocessor,
        DecisionTreeRegressor(max_depth=7, random_state=random_state),
    )

    scores_numerical = cross_validate(tree_numerical, data_numerical, target, cv=10)["test_score"]
    scores_all = cross_validate(tree_all_features, data, target, cv=10)["test_score"]

    print(f"random_state={random_state}")
    print(f"  numerical features only: mean R2 = {scores_numerical.mean():.4f}")
    print(f"  all features:            mean R2 = {scores_all.mean():.4f}")
    wins = (scores_all > scores_numerical).sum()
    print(f"  all features better on {wins} of 10 folds")
```

`handle_unknown="use_encoded_value"` with `unknown_value=-1` prevents a failure when a rare category appears only in a test fold. `remainder="passthrough"` keeps all the other columns, including numerical columns that are not in `numerical_features`.

## Interpretation

- Comparing fold by fold (paired comparison) is more informative than comparing two means: it shows whether one model wins consistently or only on average.
- A default, fully grown tree overfits; tuning `max_depth` usually improves it substantially, and nested cross-validation estimates the performance of the tuned tree without bias.
- Repeating the experiment with several seeds verifies that the conclusion does not depend on a particular random choice.
- Adding informative categorical features, even with a simple ordinal encoding, can improve a tree; the same encoding would be inappropriate for the linear model, which would need one-hot encoding.

## Related pages

- [OrdinalEncoder vs OneHotEncoder](../05-Preprocessing/OrdinalEncoder-vs-OneHotEncoder.md)
- [DecisionTreeRegressor](../07-Models/Decision-Trees/DecisionTreeRegressor.md)
- [Linear Regression](../07-Models/Linear-Models/Linear-Regression.md)
- [Nested Cross-Validation](../09-Hyperparameter-Tuning/Nested-Cross-Validation.md)

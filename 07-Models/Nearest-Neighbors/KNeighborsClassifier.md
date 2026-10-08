# KNeighborsClassifier

[Home](../../README.md) / [Models](../README.md) / [Nearest Neighbors](../README.md#nearest-neighbors)

## Idea

K-nearest neighbors predicts the label of a sample based on the labels of its closest neighbors in the training set.

It is a simple and intuitive model based on local similarity.

## Import

```python
from sklearn.neighbors import KNeighborsClassifier
```

## Minimal example

```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)
```

## How it works mathematically

k-NN is a non-parametric, lazy learner: fitting mainly stores the training data, while most computation happens when predicting a new point. For a query $x$, let $N_k(x)$ be the indices of its $k$ closest training samples. Uniform classification predicts the most common class:

$$
\hat{y}(x)=\operatorname{mode}\{y_i:i\in N_k(x)\}
$$

The usual Minkowski distance is:

$$
d_p(x,z)=\left(\sum_{j=1}^{p}|x_j-z_j|^p\right)^{1/p}
$$

`p=1` gives Manhattan distance and `p=2` gives Euclidean distance. With `weights="distance"`, closer neighbors have more influence, commonly through weights proportional to $1/(d(x,x_i)+\varepsilon)$.

## Main parameter

### `n_neighbors`

Controls how many nearest points are considered.

- small `k`: more flexible, may overfit
- large `k`: smoother, may underfit

## Other parameters

- `weights`: how much each neighbor contributes
- `algorithm`: method used to find neighbors
- `p`: distance metric power
- `metric`: distance function, such as `"minkowski"` or `"manhattan"`
- `leaf_size`: speed/memory trade-off for tree-based neighbor searches
- `n_jobs`: parallelism for neighbor queries

For classification, `predict_proba` returns the fraction of neighbor votes per class, and `score` is accuracy by default.

## Distance interpretation

- `p=1`: Manhattan distance
- `p=2`: Euclidean distance

## Important note

This model is sensitive to feature scaling because it relies on distances. Features with large numerical ranges can dominate the distance.

## Typical use cases

- small or medium datasets
- as a baseline model
- when intuition matters

## Statistical intuition and limitations

Small $k$ gives low bias but high variance: one noisy neighbor can change the prediction. Large $k$ smooths over local structure, increasing bias but reducing variance. The best value is usually selected with cross-validation.

In high dimensions, distances become less informative because points tend to look similarly far apart. This curse of dimensionality, plus irrelevant features, can make k-NN weak. Feature selection, dimensionality reduction, and a meaningful metric can help.

## Good practice

Use `StandardScaler` before fitting when using k-NN, inside a pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
```

Select `n_neighbors` with cross-validation, for example with a [validation curve](../../08-Model-Evaluation/Validation-Curves.md) or [GridSearchCV](../../09-Hyperparameter-Tuning/GridSearchCV.md).

## Regression counterpart

`KNeighborsRegressor` predicts the average target of the $k$ nearest neighbors (weighted by inverse distance with `weights="distance"`). It has the same parameters, the same sensitivity to scaling, and the same limitations in high dimensions.

## Related pages

- [StandardScaler](../../05-Preprocessing/StandardScaler.md)
- [Overfitting vs Underfitting](../../01-ML-Basics/Overfitting-vs-Underfitting.md)
- [Nested Cross-Validation](../../09-Hyperparameter-Tuning/Nested-Cross-Validation.md)
- [scikit-learn user guide: Nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn API reference: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

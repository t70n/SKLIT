# DecisionTreeClassifier

[Home](../../README.md) / [Models](../README.md) / [Decision Trees](../README.md#decision-trees)

## Idea

A decision tree recursively splits the data according to feature thresholds.

At each node it asks a question like:

- is `age <= 30`?
- is `income > 50k`?

This creates a tree of decision rules that eventually leads to a class prediction.

## Import

```python
from sklearn.tree import DecisionTreeClassifier
```

## Minimal example

```python
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(model.score(X_test, y_test))
```

## How it works mathematically

The tree grows greedily from the root. At each node it tests feature thresholds such as $x_j\leq t$, partitions the data into two children, and keeps the split that most improves the objective. Repeating this creates axis-aligned rectangular regions in feature space.

For a node $t$, let $p_{k\mid t}$ be the proportion of samples in class $k$. Gini impurity is:

$$
G(t)=1-\sum_k p_{k\mid t}^2
$$

Entropy is:

$$
H(t)=-\sum_k p_{k\mid t}\log p_{k\mid t}
$$

A pure node has impurity zero. For a candidate split into left and right children, the weighted impurity reduction is:

$$
\Delta I=I(t)-\frac{n_L}{n_t}I(L)-\frac{n_R}{n_t}I(R)
$$

The chosen split maximizes $\Delta I$. A leaf predicts the majority class; its class probabilities are the class proportions in that leaf.

`predict_proba` is not an intrinsic probability model like logistic regression. It is the empirical class distribution of the reached leaf. Leaves with few training samples can give unstable probabilities, so use `min_samples_leaf` and validate calibration when probabilities matter.

## Key parameters

### `max_depth`

Controls the maximum depth of the tree.

- small depth: simpler model
- large depth: more flexible and prone to overfitting

### `min_samples_leaf`

Minimum number of samples required in a leaf.

### `criterion`

Usually `gini` or `entropy` for classification.

### Other complexity controls

- `min_samples_split`: minimum samples needed to split an internal node
- `max_features`: number of features considered for each split
- `class_weight`: gives some classes more importance, useful for imbalance
- `splitter`: `"best"` searches the best split; `"random"` samples candidate splits
- `ccp_alpha`: cost-complexity pruning strength

Pruning minimizes a trade-off such as:

$$
R_\alpha(T)=R(T)+\alpha|T|
$$

where $R(T)$ measures leaf error and $|T|$ is the number of leaves. Increasing `ccp_alpha` removes branches whose predictive improvement is not worth their complexity.

## Important note

Decision trees are sensitive to depth and can overfit if allowed to grow too deep.

Trees are unstable: a small change in the training data can change an early split and therefore the entire subtree below it. Depth, minimum leaf size, pruning, and validation help control this high variance.

For a complete workflow covering decision-boundary plots, tree diagrams, leaf probabilities, regression-tree geometry, extrapolation, and tuning, see [Decision Tree Interpretation and Tuning](../../10-Recipes/Decision-Tree-Interpretation-and-Tuning.md).

## Related pages

- [Tree-Based Models](Tree-Based-Models.md)
- [DecisionTreeRegressor](DecisionTreeRegressor.md)
- [Random Forest](../Ensembles/Random-Forest.md)
- [Decision Tree Interpretation and Tuning](../../10-Recipes/Decision-Tree-Interpretation-and-Tuning.md)
- [scikit-learn API reference: DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

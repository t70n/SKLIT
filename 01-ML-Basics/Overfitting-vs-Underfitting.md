# Overfitting vs Underfitting

[Home](../README.md) / [Machine Learning Basics](README.md)

## Overfitting

A model overfits when it fits the training data too closely and learns noise rather than the underlying pattern.

Typical signs:

- very low training error
- much worse test error
- model is too flexible

### Causes

- too few training samples
- noise in the data
- too complex model family
- hyperparameters that are too permissive

## Underfitting

A model underfits when it is too simple to capture the structure of the data.

Typical signs:

- high training error
- high test error
- model is too rigid

### Causes

- model is too simple
- important features are ignored
- hyperparameters are too restrictive

## Summary table

| Situation | Training error | Test error | Gap between them | Typical remedy |
| --- | --- | --- | --- | --- |
| Underfitting | High | High | Small | Increase capacity, add or transform features, reduce regularization |
| Good fit | Low | Low | Small | Keep, then confirm on held-out data |
| Overfitting | Very low | High | Large | Regularize, simplify the model, add data, constrain hyperparameters |

## Bias-variance trade-off

This is the central compromise in machine learning.

- High bias leads to underfitting: the model makes systematic errors.
- High variance leads to overfitting: the model changes a lot when the training set changes.

For squared error, the expected prediction error at a point $x$ can be decomposed as:

$$
\mathbb{E}\left[(y-\hat{f}(x))^2\right]=\operatorname{Bias}\left[\hat{f}(x)\right]^2+\operatorname{Var}\left[\hat{f}(x)\right]+\sigma^2
$$

where the expectation is taken over training sets and noise, and $\sigma^2$ is the irreducible noise of the target. Increasing model flexibility usually decreases bias and increases variance. The goal is to find a model with a good balance between the two.

## Example intuition

A very deep decision tree might memorize the training set, giving near-perfect training error but poor generalization.

A very shallow tree may not capture the structure, producing high training and test error.

## In validation curves

We often use validation curves to assess how a hyperparameter such as `max_depth` or `gamma` changes the training and test performance.

- If the gap between training and test error grows too much, overfitting is happening.
- If both errors stay high, the model may be underfitting.

See [Validation Curves](../08-Model-Evaluation/Validation-Curves.md). To check whether more data would help, use [Learning Curves](../08-Model-Evaluation/Learning-Curves.md).

## Practical rule

- Add regularization or reduce model complexity when overfitting
- Increase model capacity or improve feature representation when underfitting
- Use cross-validation to compare models sensibly

## Related pages

- [Train/Test Split](Train-Test-Split.md)
- [Validation Curves](../08-Model-Evaluation/Validation-Curves.md)
- [Learning Curves](../08-Model-Evaluation/Learning-Curves.md)
- [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)

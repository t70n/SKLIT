# Hyperparameter Tuning

[Home](../README.md)

This section explains how to choose the hyperparameters of a model with cross-validation, and how to estimate the performance of the tuned model without bias.

## Pages

| Page | Summary |
| --- | --- |
| [Manual Tuning](Manual-Tuning.md) | Loop over values with `set_params` and cross-validation to understand one parameter |
| [GridSearchCV](GridSearchCV.md) | Exhaustive search over a predefined grid |
| [RandomizedSearchCV](RandomizedSearchCV.md) | Random sampling of candidates from distributions within a fixed budget |
| [Nested Cross-Validation](Nested-Cross-Validation.md) | Inner loop for tuning, outer loop for an unbiased performance estimate |
| [Boosting Hyperparameter Tuning](Boosting-Hyperparameter-Tuning.md) | Learning rate, number of trees, tree complexity, and early stopping |
| [Parallel Coordinates Visualization](Parallel-Coordinates-Visualization.md) | Visual analysis of many search results at once |

## Key takeaways

- **Hyperparameters matter**: they have a significant impact on model performance and should be chosen deliberately, not left at default values without checking.
- **Automation is better than manual tuning**: manual tuning helps to understand the effect of a parameter, while automated searches scale better and are more systematic.
- **Grid search versus randomized search**:
  - grid search is systematic and interpretable, but its cost grows multiplicatively with the number of hyperparameters, and it can miss good values between grid points
  - randomized search explores the space within a fixed budget, scales better with many hyperparameters, and can sample from continuous distributions
- **Nested cross-validation prevents selection bias**: when hyperparameters are tuned and performance is evaluated, an outer loop gives an unbiased generalization estimate while the inner loop handles the selection.
- **Always separate tuning from evaluation**: evaluating on the same data used for hyperparameter selection leads to overly optimistic performance estimates.

## Workflow summary

1. **Start simple**: begin with a few key hyperparameters that you understand well
2. **Choose the search method**: `GridSearchCV` for small, well-defined search spaces; `RandomizedSearchCV` for larger spaces
3. **Define the search space**: a regular grid, or distributions (log-uniform for parameters spanning orders of magnitude)
4. **Evaluate with cross-validation**: the search must use cross-validation to avoid overfitting a single split
5. **Inspect the results**: analyze `cv_results_` to understand how performance depends on the parameters
6. **Estimate the generalization performance**: on an independent test set, or with nested cross-validation
7. **Refit for deployment**: fit the search on all the training data and use `best_estimator_`

## Related concepts

- **Overfitting**: tuning hyperparameters on the test set causes overfitting to that particular split; see [Overfitting vs Underfitting](../01-ML-Basics/Overfitting-vs-Underfitting.md)
- **Pipelines**: structure the model with preprocessing and the estimator in a pipeline; nested parameters are named `<step>__<parameter>`; see [Pipeline](../05-Preprocessing/Pipeline.md)
- **Cross-validation**: the foundation of all hyperparameter search methods; use `StratifiedKFold` for classification to maintain class balance; see [Cross-Validation](../08-Model-Evaluation/Cross-Validation.md)
- **Validation curves**: the effect of one hyperparameter on the training and test scores; see [Validation Curves](../08-Model-Evaluation/Validation-Curves.md)
- **Learning curves**: the effect of the training-set size; see [Learning Curves](../08-Model-Evaluation/Learning-Curves.md)

## Related recipes

- [Nested Cross-Validation with SVC](../10-Recipes/Nested-Cross-Validation-with-SVC.md)
- [Histogram Gradient Boosting with Nested Cross-Validation](../10-Recipes/Hist-Gradient-Boosting-Nested-CV.md)
- [Ames Housing: Linear Model vs Decision Tree](../10-Recipes/Ames-Housing-Linear-vs-Tree.md)
- [Decision Tree Interpretation and Tuning](../10-Recipes/Decision-Tree-Interpretation-and-Tuning.md)

## Further resources

### scikit-learn examples

- [Grid-search example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html): practical grid-search workflow on real data
- [Randomized-search example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html): randomized search with continuous distributions
- [Nested cross-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): separating hyperparameter tuning from evaluation
- [scikit-learn user guide: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)

### Topics to explore

- successive halving (`HalvingGridSearchCV`, `HalvingRandomSearchCV`) to discard poor candidates early
- Bayesian optimization for more sophisticated search strategies
- hyperparameter importance analysis to identify which parameters matter most
- ensemble methods that combine models trained with different hyperparameter settings

## Navigation

- Previous section: [Model Evaluation](../08-Model-Evaluation/README.md)
- Next section: [Recipes](../10-Recipes/README.md)
- [Back to the home page](../README.md)

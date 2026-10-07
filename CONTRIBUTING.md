# Contributing Guide

This guide describes the conventions of the wiki, so that new pages remain consistent with the existing ones and easy to navigate on GitHub and on GitHub Pages.

## Repository organization

- Content is organized in numbered sections (`01-ML-Basics/` to `11-Glossary/`), ordered like a machine learning workflow. The number fixes the display order on GitHub.
- Each section has a `README.md` index page listing all its pages in tables, with a short summary per page and links to the previous and next sections.
- The `07-Models/` section is further divided into one subfolder per model family (`Baselines/`, `Linear-Models/`, `Nearest-Neighbors/`, `Support-Vector-Machines/`, `Decision-Trees/`, `Ensembles/`).
- `10-Recipes/` is flat: recipes are classified in the catalog of `10-Recipes/README.md`, because a recipe often belongs to several themes.
- The root `README.md` contains the mindmap, which lists every page of the wiki.

## Naming conventions

| Element | Convention | Examples |
| --- | --- | --- |
| Section folder | Two-digit number, hyphen, title words separated by hyphens | `05-Preprocessing`, `08-Model-Evaluation` |
| Concept page | Title words separated by hyphens | `Cross-Validation-Strategies.md`, `Missing-Values.md` |
| Estimator page | Exact scikit-learn class name | `DecisionTreeClassifier.md`, `KNeighborsClassifier.md` |
| Index page | `README.md` | `03-EDA/README.md` |

Avoid spaces, parentheses, and accented characters in file names. Never keep two versions of a page (for example `Page (2).md`): merge them.

## Page structure

Every page follows the same skeleton:

1. a single level-one title (`# Title`) on the first line
2. a navigation line just below the title, linking to the home page and to the section index
3. content sections with level-two headings (`##`), and level-three headings (`###`) for subsections
4. a final `## Related pages` section with links to related pages and to the official documentation

### Page template

````markdown
# Page Title

[Home](../README.md) / [Section](README.md)

## Idea

One or two paragraphs explaining the concept and when it is useful.

## Import

```python
from sklearn.module import Estimator
```

## Minimal example

```python
model = Estimator()
model.fit(X_train, y_train)
```

## How it works

Intuition first, then the mathematics.

## Important parameters

- `parameter`: effect on the model

## When to use it

## Limitations

## Related pages

- [Related Page](Related-Page.md)
- [scikit-learn API reference: Estimator](https://scikit-learn.org/stable/modules/generated/sklearn.module.Estimator.html)
````

### Recipe template

````markdown
# Recipe Title

[Home](../README.md) / [Recipes](README.md)

## Goal

What the recipe demonstrates, in a short list.

## Dataset

Name, source, target, and the loading code.

```python
import pandas as pd

data = pd.read_csv("../datasets/file.csv")
```

## Step 1: ...

Code, then a short explanation of what it does.

## Interpretation

What the results show, with approximate values when they are reproducible.

## Related pages

- [Concept Page](../08-Model-Evaluation/Concept-Page.md)
````

## Writing style

- Use a formal, neutral tone suitable for academic and professional documentation.
- Do not use emojis, emoticons, or decorative symbols.
- Use American English spelling, consistently with the existing pages (for example "modeling", "normalize", "behavior").
- Prefer short paragraphs and lists; define each abbreviation at its first use.
- Explain the intuition before the formula, and state the practical consequence of each concept.
- When a behavior depends on the library version, say so explicitly (for example "since scikit-learn 1.5").
- Do not reproduce long passages from external sources; summarize and cite them.

## Code style

- Every code block has a language tag (`python`, `bash`, `text`).
- Snippets should run once the preceding blocks of the page have been executed: include the imports where an object is first used.
- Put preprocessing inside a `Pipeline` and evaluate the complete pipeline with cross-validation, to avoid data leakage.
- Set `random_state` when a result depends on randomness.
- Avoid deprecated APIs. When the current API differs from older versions, use the current one and mention the older form in a sentence.
- Follow PEP 8 and keep lines reasonably short; prefer explicit names (`data_train`, `target_test`) to single letters in recipes.

## Mathematics

- Inline formulas use `$...$`; display formulas use `$$` on their own lines, separated from the surrounding text by blank lines.
- Do not put formulas in headings: GitHub derives the anchors of headings from their text, and math makes these anchors unpredictable.
- Do not write a currency dollar sign in prose next to another dollar sign in the same paragraph or table row, because the pair would be interpreted as a formula. Write "thousands of dollars" or put the unit in a code span such as `k$`.
- Avoid the character `|` inside formulas placed in tables; it would split the table cell.

## Links

- Use relative links with the `.md` extension, for example `[Pipeline](../05-Preprocessing/Pipeline.md)`. They work on GitHub, in local editors, and on GitHub Pages.
- Link to a section of a page with its GitHub anchor: lowercase heading, punctuation removed, spaces replaced by hyphens, for example `Classification-Metrics.md#class-imbalance`.
- Check all internal links before committing:

```bash
python tools/check_links.py
```

The same check runs automatically on GitHub for every push and pull request (`.github/workflows/check-links.yml`).

## Adding a page: checklist

1. Create the file in the right section with the page template (or the recipe template).
2. Add the navigation line below the title and the `## Related pages` section at the end.
3. Add the page to the table of the section `README.md`.
4. Add the page to the mindmap outline in the root `README.md`, under the right branch.
5. For a recipe, add a row to the catalog of `10-Recipes/README.md`.
6. Add new important terms to the [Glossary](11-Glossary/README.md), with a link to the page.
7. Link the new page from at least one related existing page.
8. Run `python tools/check_links.py`.

## Maintaining the mindmap

The mindmap is the nested list inside `<div class="markmap" markdown="0">` in the root `README.md`:

- On GitHub, the list is displayed as a table of contents.
- On GitHub Pages, `_includes/head-custom.html` renders it as an interactive [markmap](https://markmap.js.org/) mindmap and converts the `.md` links to the generated `.html` pages.

Rules for editing the outline:

- keep a single root item (`- SKLIT`) and indent each level with two spaces
- keep the blank lines just after the opening `<div>` tag and just before the closing `</div>` tag, otherwise GitHub does not render the list
- use short node labels and relative `.md` links
- do not use the characters `&` and `<` in the outline

The display options (initial depth, colors, node width) are defined in `_includes/head-custom.html`.

## Publishing on GitHub Pages

The repository is published as is with the default Jekyll build of GitHub Pages: Settings, Pages, "Deploy from a branch", branch `main`, folder `/ (root)`. `_config.yml` selects the theme and excludes the files that should not be published. `_includes/head-custom.html` adds markmap and MathJax to every page.

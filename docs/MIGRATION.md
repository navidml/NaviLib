# Migrating to NaviLib 0.5

Import the package as `import NaviLib as nv`. The distribution and runtime now
share the same version. `navdata` and `datakit` were inconsistent documentation
names, not importable packages in this repository.

## Preferred names

Existing names remain callable aliases with the same signatures and docstrings.
Use the descriptive names for new code. Most existing names already followed
snake_case and remain unchanged.

| Previous name | Preferred name |
|---|---|
| `cleaning.fix_missing` | `cleaning.impute_missing` |
| `cleaning.fix_duplicates` | `cleaning.drop_duplicates` |
| `cleaning.fix_outliers` | `cleaning.handle_outliers` |
| `cleaning.convert` | `cleaning.convert_columns` |
| `cleaning.split` | `cleaning.split_data` |
| `cleaning.balance` | `cleaning.resample_data` |
| `feature_engineering.scale` | `feature_engineering.scale_features` |
| `feature_engineering.encode` | `feature_engineering.encode_categorical` |
| `feature_engineering.add_datetime` | `feature_engineering.add_datetime_features` |
| `feature_engineering.add_cyclical` | `feature_engineering.add_cyclical_features` |
| `modeling.validate` | `modeling.cross_validate_model` |
| `modeling.tune` | `modeling.tune_model` |
| `modeling.train` | `modeling.train_model` |
| `modeling.predict` | `modeling.predict_model` |
| `Statistical_Tests` | `statistical_tests` |
| `statistical_tests.normality_test` | `statistical_tests.test_normality` |
| `statistical_tests.variance_homogeneity_test` | `statistical_tests.test_equal_variance` |
| `statistical_tests.parametric_test` | `statistical_tests.test_parametric` |
| `statistical_tests.nonparametric_test` | `statistical_tests.test_nonparametric` |
| `statistical_tests.correlation_test` | `statistical_tests.test_correlation` |
| `statistical_tests.categorical_test` | `statistical_tests.test_categorical` |

`NaviLib.Statistical_Tests` imports resolve to the same canonical module on all
supported operating systems. The lowercase filename is the maintained source.

## Behavior corrections

- `apply_state(strict=True)` checks source columns for all built-in state types.
  Previously some cleaning/binning states silently skipped missing columns.
- `ChainTransformer` requires labels in `y`, rejects a target already in `X`, and
  rejects row removal/reordering. Attached labels are protected from unsupervised
  preprocessing. `drop_target=False` with a named target is rejected.
- Binary evaluation defaults to the last sorted class, matching scikit-learn
  probability-column order. Previously it sometimes selected the minority class.
  Set `pos_label` explicitly when your positive class differs. A probability
  **vector** always denotes that positive class; a **matrix** follows class order.
- A zero classification threshold stays zero. Training prevalence refers to
  `model.classes_[1]`, not the minority class. String-label automatic threshold
  selection uses the same convention.
- `predict_model` accepts either a training artifact or a `load_model` bundle.
  For bundles, separately stored states are applied before prediction. When the
  model already contains preprocessing, do not also save the same states separately.
- `cross_validate_model` reuses the fitted fold estimators for OOF predictions,
  preserves exact validation splits, and records probability class order in attrs.
- Nested validation uses group-aware inner as well as outer folds. Default model
  comparisons preprocess mixed DataFrames within each fold; custom estimators
  should supply their own preprocessing pipelines.
- WOE encoding raises when there are too few observations for honest out-of-fold
  encoding, instead of silently using in-sample target information.
- Iterative imputation is deterministic on replay; empty numeric features retain
  their width. Nullable integer medians can promote to nullable floats, and
  categorical imputation can add a new fallback category.
- Column-name cleaning preserves Unicode and makes duplicates unique even when
  suffix-like names already exist. Empty DataFrames can have their names cleaned.
- SMAPE includes correct zero/zero predictions with zero contribution.
- Normality `method="ks"` now uses Lilliefors correction for estimated parameters.
  Anderson-Darling uses the newer SciPy p-value API when available; older SciPy
  versions retain critical-value results. Partial correlation uses residual degrees
  of freedom. Chi-square goodness-of-fit accepts proportions summing to one.

Tabular outputs remain pandas DataFrames/dictionaries, allowing existing analysis
code to inspect them. Styling is opt-in with `style_table`; HTML reports are added
through `create_report`, while the existing `eda.report` remains available.

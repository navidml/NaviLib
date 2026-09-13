# NaviLib API reference

Generated for version 0.5.0.

Use `nv.help_map(query)` to search by purpose and `help(function)` in Python. Legacy aliases are listed in [MIGRATION.md](MIGRATION.md). Examples with `df`, `X`, or `y` assume the dataset described by that function.

## NaviLib.cleaning

### overview

```python
NaviLib.cleaning.overview(df: 'Frame', max_rows: 'int' = 40) -> 'Frame'
```

```text
One-glance profile of every column: dtype, missing, cardinality, skew.

This is the very first thing to run on a new dataset.  It flags the
three problems that silently ruin models: near-constant columns,
identifier-like columns, and columns that are numeric in name only.

Parameters
----------
df : DataFrame
max_rows : int
    Only affects how many rows are printed if you display the result.

Returns
-------
DataFrame indexed by column name, one row per column, with a ``flags``
column containing short warnings such as ``"constant"``, ``"id-like"``,
``"high-missing"``, ``"numeric-as-text"``.
```

### scan_missing

```python
NaviLib.cleaning.scan_missing(df: 'Frame', columns=None, by: 'Optional[str]' = None) -> 'Frame'
```

```text
Missing-value report, optionally broken down by a grouping column.

The ``by`` argument is the important one: if missingness depends on the
target (``by="died"``), the data are *not* missing at random and simple
mean imputation will bias the model.

Parameters
----------
df : DataFrame
columns : str or list, optional
    Restrict the report to these columns.
by : str, optional
    Group column.  Adds one ``missing_pct__<value>`` column per group
    plus a ``spread`` column (max minus min across groups).  A large
    spread is a red flag for MNAR data.

Returns
-------
DataFrame, sorted by missing count, descending.
```

### scan_duplicates

```python
NaviLib.cleaning.scan_duplicates(df: 'Frame', subset=None, show: 'int' = 0) -> 'Dict[str, Any]'
```

```text
Count duplicated rows, optionally on a subset of key columns.

Parameters
----------
df : DataFrame
subset : list, optional
    Consider rows duplicated when these columns match (e.g. a patient
    id).  ``None`` means all columns must match.
show : int, default 0
    Return this many example duplicated rows under key ``"examples"``.

Returns
-------
dict with ``n_rows``, ``n_duplicated`` (extra copies only),
``n_affected`` (all rows involved), ``pct`` and optionally ``examples``.
```

### scan_outliers

```python
NaviLib.cleaning.scan_outliers(df: 'Frame', columns=None, method: "Literal['iqr', 'zscore', 'mad', 'quantile']" = 'iqr', factor: 'float' = 1.5, z: 'float' = 3.0, q: 'Tuple[float, float]' = (0.01, 0.99)) -> 'Frame'
```

```text
Count outliers per numeric column and report the cut-off bounds.

Methods
-------
``iqr``      Q1 - factor*IQR  to  Q3 + factor*IQR.  Robust, the default.
``zscore``   mean +/- z*std.  Assumes roughly normal data; the outliers
             themselves inflate the std, so it under-detects.
``mad``      median +/- z*1.4826*MAD.  The robust version of z-score;
             prefer this over ``zscore`` on skewed data.
``quantile`` fixed empirical quantiles, e.g. 1st and 99th percentile.

Returns
-------
DataFrame with ``lower``, ``upper``, ``n_outliers``, ``pct`` per column.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['iqr', 'zscore', 'mad', 'quantile'], default 'iqr'
    Algorithm to use; see the supported methods and assumptions above.
factor : float, default 1.5
    Multiplier of the interquartile range for lower/upper outlier fences.
z : float, default 3.0
    Standard-deviation or robust-z cutoff for outlier detection.
q : Tuple[float, float], default (0.01, 0.99)
    Lower and upper quantile probabilities used as clipping/outlier bounds.
```

### check_numeric

```python
NaviLib.cleaning.check_numeric(df: 'Frame', column: 'str', sample: 'int' = 5) -> 'Dict[str, Any]'
```

```text
Find the exact values that stop a column from converting to numeric.

Returns a dict with ``n_invalid``, ``pct_invalid``, ``bad_values``
(unique offenders) and ``examples`` (sample rows).  Use this before
``convert(..., to="numeric")`` so you know what you are about to
turn into NaN.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
column : str
    Name of the source column to inspect or transform.
sample : int, default 5
    Maximum number of observations sampled for plotting or expensive
    diagnostics.

Returns
-------
dict
    Conversion success counts and examples of values that cannot be parsed.
```

### scan_leakage

```python
NaviLib.cleaning.scan_leakage(df: 'Frame', target: 'str', threshold: 'float' = 0.95) -> 'Frame'
```

```text
Flag features suspiciously predictive of the target on their own.

A single feature reaching AUC >= ``threshold`` is almost never good
news: it usually means the feature is recorded *after* the outcome, is
a proxy for it, or is the outcome renamed.  This is the check that
would have caught 'intubation' predicting neonatal death.

Returns
-------
DataFrame with per-feature univariate AUC (binary target) or absolute
correlation (continuous target), sorted descending, plus a ``flag``.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
target : str
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
threshold : float, default 0.95
    Decision cutoff or inspection threshold; see the function-specific
    interpretation above.
```

### column_types

```python
NaviLib.cleaning.column_types(df: 'Frame') -> 'Dict[str, List[str]]'
```

```text
Split columns into numeric / categorical / datetime / boolean buckets.

Used internally everywhere, but exposed because it is handy on its own.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.

Returns
-------
dict
    Numeric, categorical, datetime and boolean column-name lists.
```

### scan_outliers_multivariate

```python
NaviLib.cleaning.scan_outliers_multivariate(df: 'Frame', columns=None, method: "Literal['isolation_forest', 'lof', 'elliptic', 'mahalanobis']" = 'isolation_forest', contamination: 'float' = 0.03, n_neighbors: 'int' = 20, scale: 'bool' = True, random_state: 'int' = 42, return_state: 'bool' = False)
```

```text
Score every row for how anomalous it is across several columns at once.

Univariate rules miss the interesting cases. A 45-year-old is ordinary;
a 45-year-old with a gestational age of 24 weeks and a birth weight of
4 kg is not, and no single-column bound will flag it. These four methods
all work on the joint distribution.

Methods
-------
``isolation_forest``  random splits; rows that separate in few splits
                      are outliers. Fast, handles many columns, makes no
                      distributional assumption. The default.
``lof``               local outlier factor: compares a row's local
                      density to its neighbours'. Finds outliers that
                      sit inside the overall cloud but in a sparse
                      pocket -- the ones isolation forest can miss.
``elliptic``          robust Gaussian fit; only sensible when the data
                      really are roughly elliptical.
``mahalanobis``       distance from the robust centre in covariance
                      units, with a chi-squared cut-off. Interpretable
                      and gives a per-row distance you can rank.

Parameters
----------
contamination : float, default 0.03
    Expected share of outliers. This is an *assumption you are making*,
    not something the method discovers -- set it to 0.10 and it will
    dutifully flag 10% of rows. Check ``score`` before trusting the flag.
scale : bool, default True
    Standardise first. Without it, whichever column has the largest
    units dominates every distance and the result is about your units,
    not your data.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['isolation_forest', 'lof', 'elliptic', 'mahalanobis'], default 'isolation_forest'
    Algorithm to use; see the supported methods and assumptions above.
n_neighbors : int, default 20
    Number of neighbors used by the selected imputer or anomaly detector.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame with the original index plus ``outlier_score`` (higher =
more anomalous), ``is_outlier``, and the per-column z-scores of the
flagged rows so you can see *why* each one was flagged. With
``return_state=True`` also returns a replayable state.

>>> flags, st = dp.scan_outliers_multivariate(train, return_state=True)
>>> flags[flags.is_outlier].head()
>>> test_flags = nv.apply_state(test, st)          # same fitted model
```

### clean_names

```python
NaviLib.cleaning.clean_names(df: 'Frame', style: "Literal['snake', 'upper_snake', 'kebab', 'camel', 'pascal', 'compact']" = 'snake', rename: 'Optional[Dict[str, str]]' = None, columns=None, dedupe: 'bool' = True) -> 'Frame'
```

```text
Normalise column names.

Unlike a naive ``lower().replace(' ', '_')`` this handles CamelCase,
punctuation, leading digits, accidental double underscores and
duplicate results.

Parameters
----------
style : str
    Target convention.  ``snake`` is the default and what pandas users
    expect.
rename : dict, optional
    Explicit ``{old: new}`` overrides, applied *after* the style pass.
columns : list, optional
    Only rename these columns.
dedupe : bool, default True
    Append ``_2``, ``_3`` ... when two names collide after cleaning.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.



Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### drop_missing

```python
NaviLib.cleaning.drop_missing(df: 'Frame', max_missing: 'Union[float, int]' = 0.5, axis: "Literal['columns', 'rows']" = 'columns', protect: 'Optional[Sequence[str]]' = None) -> 'Frame'
```

```text
Drop columns (or rows) whose missing rate exceeds ``max_missing``.

Note the direction: you specify how much missingness you *tolerate*,
and anything worse is removed.

Parameters
----------
max_missing : float in (0,1) or int
    Float = proportion allowed.  Int = absolute count allowed.
axis : {"columns", "rows"}
protect : list, optional
    Never drop these columns, whatever their missing rate (put your
    target here).
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.



Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### drop_constant

```python
NaviLib.cleaning.drop_constant(df: 'Frame', protect: 'Optional[Sequence[str]]' = None, max_dominance: 'float' = 1.0) -> 'Frame'
```

```text
Drop columns with a single value, or dominated by one value.

``max_dominance=0.99`` drops any column where 99% of non-missing rows
share the same value.  Such columns cost degrees of freedom and teach
the model nothing -- 'steroid therapy', present in 0.5% of rows, is the
classic example.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
protect : Optional[Sequence[str]], default None
    Columns to retain even when they satisfy the removal criterion.
max_dominance : float, default 1.0
    Largest allowed frequency share of a single value before a column is
    treated as near-constant.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### fix_outliers_multivariate

```python
NaviLib.cleaning.fix_outliers_multivariate(df: 'Frame', columns=None, method: "Literal['isolation_forest', 'lof', 'elliptic', 'mahalanobis']" = 'isolation_forest', contamination: 'float' = 0.03, action: "Literal['drop', 'flag']" = 'flag', **kwargs) -> 'Frame'
```

```text
Drop or flag multivariate outliers.

``action="flag"`` (the default) adds ``is_outlier`` and
``outlier_score`` columns and changes nothing else -- almost always the
right first move, because dropping rows is irreversible and, on
imbalanced data, disproportionately deletes the minority class. Check
what would go before you let it go.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['isolation_forest', 'lof', 'elliptic', 'mahalanobis'], default 'isolation_forest'
    Algorithm to use; see the supported methods and assumptions above.
contamination : float, default 0.03
    Expected outlier fraction used to set the anomaly decision threshold.
action : Literal['drop', 'flag'], default 'flag'
    How to treat detected observations; the supported actions are given in
    the type/signature.

Returns
-------
DataFrame
    Copy with flagged observations or anomalous rows removed, according to
    action.
```

### group_rare

```python
NaviLib.cleaning.group_rare(df: 'Frame', columns=None, min_freq: 'float' = 0.01, min_count: 'Optional[int]' = None, other_label: 'str' = 'other', return_state: 'bool' = False)
```

```text
Merge rare categories into a single ``other`` bucket.

Rare levels are the categorical twin of class imbalance: a level seen 3
times cannot support a reliable coefficient, and one-hot encoding it
just adds a near-zero column that invites overfitting.

Parameters
----------
min_freq : float
    Levels below this share of non-missing rows are merged.
min_count : int, optional
    Absolute alternative to ``min_freq``; takes priority when given.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
other_label : str, default 'other'
    Replacement category assigned to infrequent or unseen levels when
    grouping applies.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.



Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### select_features

```python
NaviLib.cleaning.select_features(df: 'Frame', target: 'str', method: "Literal['l1', 'tree', 'mi', 'corr', 'permutation']" = 'tree', task: "Literal['auto', 'classification', 'regression']" = 'auto', top_k: 'Optional[int]' = None, threshold: 'Optional[float]' = None, cv: 'int' = 5, balanced: 'bool' = True, random_state: 'int' = 42, n_jobs: 'int' = -1, **kwargs) -> 'Tuple[Frame, List[str]]'
```

```text
Rank features by importance and return the ones worth keeping.

All methods share one interface and one output shape, so switching
technique is a one-word change.  Importance of one-hot columns is
aggregated back to the original column, so the ranking is always in
terms of *your* columns.

Methods
-------
``l1``          L1-penalised linear model with the penalty chosen by CV.
                Sparse and interpretable; assumes additive effects.
``tree``        Gradient-boosted trees, importance averaged over CV
                folds.  Captures interactions.  The default.
``permutation`` Model-agnostic: shuffle a column, measure the damage.
                Slowest but the most trustworthy, and the only one that
                can return a *negative* score (feature is pure noise).
``mi``          Mutual information -- non-linear, model-free, univariate.
``corr``        Absolute correlation.  Linear and univariate only; use
                it as a sanity check, not as a selector.

Parameters
----------
top_k : int, optional
    Keep this many features.
threshold : float, optional
    Keep features scoring at or above this.  For ``l1`` the natural
    threshold is 0 (non-zero coefficients).
balanced : bool, default True
    Use class-balanced weights while *ranking* on imbalanced targets,
    so rare-class signal is not drowned out.  Ranking only; it does not
    change your data.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
target : str
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
method : Literal['l1', 'tree', 'mi', 'corr', 'permutation'], default 'tree'
    Algorithm to use; see the supported methods and assumptions above.
task : Literal['auto', 'classification', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
cv : int, default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.

Returns
-------
ranking : DataFrame  -- feature, score, plus method-specific columns
selected : list of str
```

### drop_correlated

```python
NaviLib.cleaning.drop_correlated(df: 'Frame', threshold: 'float' = 0.95, target: 'Optional[str]' = None, method: "Literal['pearson', 'spearman']" = 'spearman', return_pairs: 'bool' = False)
```

```text
Remove one column from each pair of near-duplicate numeric features.

When two features carry the same information, tree importances and
linear coefficients get split between them and both look unimportant.
Where a target is given, the member of each pair less correlated with
the target is the one dropped.

Returns the reduced frame, or ``(frame, pairs)`` with the decisions.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
threshold : float, default 0.95
    Decision cutoff or inspection threshold; see the function-specific
    interpretation above.
target : Optional[str], default None
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
method : Literal['pearson', 'spearman'], default 'spearman'
    Algorithm to use; see the supported methods and assumptions above.
return_pairs : bool, default False
    Also return the table of correlated pairs used for deciding which
    columns to drop.

Returns
-------
DataFrame or tuple
    Filtered frame; with return_pairs=True, also returns the correlated-pair
    table.
```

### balance_pipeline

```python
NaviLib.cleaning.balance_pipeline(method: 'str', model, ratio='auto', categorical=None, k_neighbors: 'int' = 5, random_state: 'int' = 42, **kwargs)
```

```text
Build a leakage-safe ``imblearn`` Pipeline: resample, then fit.

This is the correct way to combine resampling with cross-validation or
a grid search: the sampler runs on the training part of each fold only,
and is skipped entirely at predict time.

>>> pipe = dp.balance_pipeline("smote", LGBMClassifier(), ratio=0.3)
>>> cross_validate(pipe, X, y, cv=5, scoring="average_precision")

Parameters
----------
method : str
    Algorithm to use; see the supported methods and assumptions above.
model : object
    Scikit-learn-compatible estimator or pipeline. Include learned
    preprocessing inside the pipeline during cross-validation.
ratio : optional, default 'auto'
    Sampling strategy passed to imbalanced-learn: supported string, ratio,
    or class-count mapping.
categorical : optional, default None
    Categorical column names or indices expected by the selected
    preprocessing/resampling operation.
k_neighbors : int, default 5
    Neighbor count for synthetic-sample generation; must fit the smallest
    training class.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
imblearn.pipeline.Pipeline
    Unfitted resampling/estimator pipeline; resampling runs during fitting
    only.
```

### class_weights

```python
NaviLib.cleaning.class_weights(y, scheme: "Literal['balanced', 'sqrt', 'custom']" = 'balanced', custom: 'Optional[Dict[Any, float]]' = None) -> 'Dict[str, Any]'
```

```text
Compute cost-sensitive weights -- usually a better first move than SMOTE.

Reweighting the loss achieves what resampling achieves without
inventing rows, without leakage risk, and without the compute cost.

Returns
-------
dict with ``class_weight`` (for scikit-learn), ``scale_pos_weight``
(for XGBoost / LightGBM), and ``sample_weight`` (an array aligned with
``y``, for models that take per-row weights).

>>> w = dp.class_weights(y_train)
>>> RandomForestClassifier(class_weight=w["class_weight"]).fit(X, y)
>>> LGBMClassifier(scale_pos_weight=w["scale_pos_weight"]).fit(X, y)

Parameters
----------
y : object
    Observed target values, positionally aligned with X.
scheme : Literal['balanced', 'sqrt', 'custom'], default 'balanced'
    Class weighting strategy: balanced frequencies, square-root weights, or
    an explicit custom mapping.
custom : Optional[Dict[Any, float]], default None
    Explicit mapping from class labels to weights when scheme="custom".
```

### prior_correct

```python
NaviLib.cleaning.prior_correct(proba, prevalence_train: 'float', prevalence_true: 'float')
```

```text
Undo the probability inflation caused by resampling.

A model trained on rebalanced data over-predicts the minority class.
This shifts the log-odds back to the real-world base rate, restoring
calibration.  Discrimination (AUC, ranking) is unchanged.

Parameters
----------
proba : array-like
    Predicted probabilities of the **minority / positive** class.
prevalence_train : float
    Minority share the model was trained on -- take it from
    ``balance()``'s report field ``prevalence_after``.
prevalence_true : float
    Real-world minority share (``prevalence_before``, or a known
    population rate).

>>> p = model.predict_proba(X_test)[:, 1]
>>> p_fixed = dp.prior_correct(p, rep["prevalence_after"], rep["prevalence_before"])

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### tune_threshold

```python
NaviLib.cleaning.tune_threshold(y_true, proba, metric: 'Union[str, callable]' = 'f1', cost_fn: 'float' = 10.0, cost_fp: 'float' = 1.0, n_steps: 'int' = 200) -> 'Dict[str, Any]'
```

```text
Find the decision threshold that optimises what you actually care about.

The evidence is consistent: shifting the threshold buys you the same
sensitivity/specificity trade-off as resampling, without distorting the
predicted probabilities.  Try this before you reach for SMOTE.

Parameters
----------
metric : {"f1", "youden", "balanced_accuracy", "cost"} or callable
    ``cost`` minimises ``cost_fn * FN + cost_fp * FP`` -- the honest
    option, because it forces you to state how much a miss is worth.
    A callable receives ``(y_true, y_pred)`` and is maximised.
cost_fn, cost_fp : float
    Relative cost of a false negative / false positive.
y_true : object
    Observed labels or numeric outcomes, in the same row order as
    predictions.
proba : object
    One-dimensional probabilities for the positive class, aligned with
    y_true.
cost_fp : float, default 1.0
    Nonnegative cost assigned to one false positive.
n_steps : int, default 200
    Number of evenly spaced candidate decision thresholds.

Returns
-------
dict with ``threshold``, ``score``, the confusion matrix at that
threshold, and a ``curve`` DataFrame over all thresholds tried.
```

### compare_balance

```python
NaviLib.cleaning.compare_balance(X: 'Frame', y, model=None, methods: 'Optional[Sequence[str]]' = None, ratio: 'Union[float, str]' = 0.3, categorical: 'Optional[Sequence[str]]' = None, cv: 'int' = 5, random_state: 'int' = 42, include_weights: 'bool' = True, n_jobs: 'int' = -1) -> 'Frame'
```

```text
Benchmark imbalance strategies honestly, with resampling inside each fold.

Report AUPRC first: with a rare positive class, AUROC looks flattering
and accuracy is meaningless.  Brier score tracks whether the predicted
probabilities are still trustworthy -- resampling usually makes it
worse, which is exactly what you want to see before deciding.

Every row of the result is the mean +/- std across ``cv`` folds, so a
difference smaller than the std is noise, not a finding.

Parameters
----------
model : estimator, optional
    Defaults to LightGBM.  Any scikit-learn classifier works.
methods : list, optional
    Defaults to a representative spread across all families.
include_weights : bool
    Also evaluate cost-sensitive weighting (no resampling at all).
X : pandas.DataFrame
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : object
    Observed target values, positionally aligned with X.
ratio : Union[float, str], default 0.3
    Sampling strategy passed to imbalanced-learn: supported string, ratio,
    or class-count mapping.
categorical : Optional[Sequence[str]], default None
    Categorical column names or indices expected by the selected
    preprocessing/resampling operation.
cv : int, default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.

Returns
-------
DataFrame sorted by AUPRC, descending.
```

### list_methods

```python
NaviLib.cleaning.list_methods() -> 'Frame'
```

```text
Return every ``balance`` method with a one-line description.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### save_table

```python
NaviLib.cleaning.save_table(df: 'Frame', path: 'str', fmt: "Optional[Literal['csv', 'excel', 'parquet', 'json']]" = None, index: 'bool' = False, make_dirs: 'bool' = True, **kwargs) -> 'str'
```

```text
Write a DataFrame to disk, inferring the format from the extension.

Generalises ``save_outliers``: the original hardcoded csv/excel, refused
to create the directory, and rejected empty frames -- but an empty
result is often exactly what you want to record ("no outliers found"),
so here it only warns.

Returns the absolute path written.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
path : str
    Destination or source filesystem path; pathlib.Path is also accepted.
fmt : Optional[Literal['csv', 'excel', 'parquet', 'json']], default None
    Output file format; None infers it from the filename extension.
index : bool, default False
    Include the DataFrame index in the exported table when True.
make_dirs : bool, default True
    Create missing destination parent directories when True.

Returns
-------
str
    Absolute path of the exported table.
```

### impute_missing

```python
NaviLib.cleaning.impute_missing(df: 'Frame', columns=None, method: "Literal['mean', 'median', 'mode', 'constant', 'knn', 'mice', 'ffill', 'bfill', 'drop_rows']" = 'median', fill_value: 'Any' = None, n_neighbors: 'int' = 5, estimator=None, using: 'Optional[Sequence[str]]' = None, add_indicator: 'bool' = False, return_state: 'bool' = False, random_state: 'int' = 42)
```

```text
Impute missing values.

Multivariate methods are done properly here: ``knn`` and ``mice`` fit on
*all* numeric helper columns, not on the target column alone.  (Fitting
KNNImputer on a single column silently degrades to mean imputation --
a common and invisible bug.)

Parameters
----------
columns : list, optional
    Columns to impute. ``None`` fits a strategy for every input column,
    including columns that may acquire missing values in a later batch.
    Numeric strategies are applied to numeric columns and ``mode`` to
    the rest, so ``method="median"`` on a mixed frame does the sensible
    thing automatically.
method : str
    ``mean`` / ``median`` / ``mode`` / ``constant`` -- univariate.
    ``knn``   -- k-nearest-neighbour imputation on the numeric block.
    ``mice``  -- iterative (chained-equation) imputation.
    ``ffill`` / ``bfill`` -- for time-ordered data only.
    ``drop_rows`` -- drop rows with any missing value in ``columns``.
using : list, optional
    Helper columns for ``knn`` / ``mice``.  Defaults to all numeric
    columns.  **Never put the target here** -- that leaks the label.
add_indicator : bool
    Add ``<col>_was_missing`` flags.  Strongly recommended when
    missingness may itself be informative.
return_state : bool
    Return ``(df, state)`` so the identical imputation can be replayed
    on unseen data via ``apply_state``.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
fill_value : Any, default None
    Constant or fallback value used for imputation. Required for the
    constant strategy.
n_neighbors : int, default 5
    Number of neighbors used by the selected imputer or anomaly detector.
estimator : optional, default None
    Optional regression estimator for iterative imputation; None uses
    BayesianRidge.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.



Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### drop_duplicates

```python
NaviLib.cleaning.drop_duplicates(df: 'Frame', keep: "Literal['first', 'last', 'none']" = 'first', subset=None) -> 'Frame'
```

```text
Remove duplicated rows.

``keep="none"`` removes *every* copy including the original -- only use
that when a duplicate means the record is untrustworthy.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
keep : Literal['first', 'last', 'none'], default 'first'
    Which duplicate occurrence to retain: 'first', 'last', or False to
    remove every duplicate occurrence.
subset : optional, default None
    Column names used to identify duplicate rows; None compares all columns.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### handle_outliers

```python
NaviLib.cleaning.handle_outliers(df: 'Frame', columns=None, method: "Literal['iqr', 'zscore', 'mad', 'quantile']" = 'iqr', action: "Literal['clip', 'nan', 'drop']" = 'clip', factor: 'float' = 1.5, z: 'float' = 3.0, q: 'Tuple[float, float]' = (0.01, 0.99), return_state: 'bool' = False)
```

```text
Clip, blank out, or drop outlying values.

``action="clip"`` (winsorising) is usually the least destructive: it
keeps the row and its other features while removing the leverage of the
extreme value.  ``action="drop"`` removes whole rows and can silently
delete most of your minority class -- check before using it.

Set ``return_state=True`` to get the learned bounds back, then replay
them on the test set with ``apply_state``.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['iqr', 'zscore', 'mad', 'quantile'], default 'iqr'
    Algorithm to use; see the supported methods and assumptions above.
action : Literal['clip', 'nan', 'drop'], default 'clip'
    How to treat detected observations; the supported actions are given in
    the type/signature.
factor : float, default 1.5
    Multiplier of the interquartile range for lower/upper outlier fences.
z : float, default 3.0
    Standard-deviation or robust-z cutoff for outlier detection.
q : Tuple[float, float], default (0.01, 0.99)
    Lower and upper quantile probabilities used as clipping/outlier bounds.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### convert_columns

```python
NaviLib.cleaning.convert_columns(df: 'Frame', column: 'str', to: "Literal['numeric', 'category', 'string', 'datetime', 'boolean']", mapping: 'Optional[Dict[Any, Any]]' = None, bins: 'Optional[Sequence[float]]' = None, labels: 'Optional[Sequence[str]]' = None, n_bins: 'Optional[int]' = None, date_format: 'Optional[str]' = None, errors: "Literal['coerce', 'raise']" = 'coerce', verbose: 'bool' = False) -> 'Frame'
```

```text
Convert a column's dtype, with optional value mapping and binning.

Parameters
----------
to : str
    Target type.  ``boolean`` understands the usual yes/no, true/false,
    y/n, 1/0 spellings in either case.
mapping : dict, optional
    Applied *before* conversion, e.g. ``{"Yes": 1, "No": 0}``.
bins / labels : optional
    Explicit bin edges for ``to="category"``; ``labels`` must be one
    shorter than ``bins``.
n_bins : int, optional
    Equal-frequency binning instead of explicit edges.
errors : {"coerce", "raise"}
    ``coerce`` turns unparseable values into NaN (and warns how many).
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
column : str
    Name of the source column to inspect or transform.
bins : Optional[Sequence[float]], default None
    Number of bins, explicit edges, or supported automatic binning rule as
    indicated by the signature.
labels : Optional[Sequence[str]], default None
    Explicit class or bin labels, in the order expected by the operation.
date_format : Optional[str], default None
    Explicit datetime parsing format passed to pandas; None uses inference.
verbose : bool, default False
    Print a concise progress/result summary when True.



Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### split_data

```python
NaviLib.cleaning.split_data(df: 'Frame', target: 'str', test_size: 'float' = 0.2, val_size: 'float' = 0.0, stratify: 'bool' = True, group: 'Optional[str]' = None, random_state: 'int' = 42)
```

```text
Split into train/test (and optionally validation), stratified by default.

Parameters
----------
val_size : float
    Fraction of the *original* data for validation.  ``0`` returns two
    frames, otherwise three (train, val, test).
group : str, optional
    Grouping column (patient id, hospital, ...).  Rows sharing a group
    are kept together, so the same patient cannot appear on both sides
    of the split.  Stratification is then approximate.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
target : str
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
test_size : float, default 0.2
    Fraction of the original observations reserved for testing; grouped
    splits operate on groups.
stratify : bool, default True
    Preserve target class proportions when splitting, where supported.
    Grouped splitting keeps groups intact instead.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.



Returns
-------
tuple of pandas.DataFrame
    Train/test copies, or train/validation/test copies when val_size is
    positive.
```

### resample_data

```python
NaviLib.cleaning.resample_data(df: 'Frame', target: 'str', method: 'str' = 'smote', ratio: 'Union[float, str, dict]' = 'auto', categorical: 'Optional[Sequence[str]]' = None, k_neighbors: 'int' = 5, random_state: 'int' = 42, return_report: 'bool' = True, _trusted: 'bool' = False, **kwargs)
```

```text
Resample a **training set** to change its class balance.

.. warning::
   Call this on training data only, and only *after* splitting.
   Resampling before the split leaks synthetic copies of training rows
   into the test set and inflates every metric.  For cross-validation
   use :func:`balance_pipeline` instead, which resamples inside each
   fold automatically.

Parameters
----------
df : DataFrame
    Features **and** target, already split into a training set.
target : str
    Target column name.
method : str
    Any key of :data:`BALANCE_METHODS`.  Call :func:`list_methods` to
    print them with descriptions.
ratio : float, str or dict, default "auto"
    Desired minority-to-majority ratio after resampling.  ``0.3`` means
    the minority ends up at 30% of the majority.  Full 1:1 balance
    (``"auto"`` for over-samplers) is rarely optimal -- moderate ratios
    usually generalise better.
categorical : list of str, optional
    Categorical column names, required for ``smotenc``.  Converted to
    positional indices for you.
k_neighbors : int, default 5
    Neighbourhood size for the SMOTE family.  Must be smaller than the
    size of your minority class.
return_report : bool, default True
    Return ``(df_resampled, report)``; report contains before/after
    counts, how many rows are synthetic, and the true prevalence you
    will need for :func:`prior_correct`.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
_trusted : bool, default False
    Internal resampling guard. Application code should leave this at its
    default.

Returns
-------
DataFrame, or ``(DataFrame, dict)`` when ``return_report``.

Examples
--------
>>> tr, te = dp.split(df, target="died")
>>> tr_bal, rep = dp.balance(tr, "died", method="smotenc",
...                          categorical=cat_cols, ratio=0.3)
>>> rep["prevalence_before"]
0.0783
```

## NaviLib.eda

### describe_numeric

```python
NaviLib.eda.describe_numeric(df: 'Frame', columns=None, percentiles: 'Sequence[float]' = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), outlier_method: "Literal['iqr', 'mad', 'none']" = 'iqr') -> 'Frame'
```

```text
Rich numeric profile: one row per column, everything that matters.

Goes well beyond ``df.describe()``: robust spread, shape, outlier
counts, and the three facts that decide whether a log transform is
even possible (zeros, negatives, and whether the column is really an
integer count).

Columns returned
----------------
``n``, ``missing_pct``, ``unique``
    Coverage.
``mean``, ``median``, ``std``, ``iqr``, ``mad``, ``cv``
    Centre and spread.  ``cv`` (std/|mean|) is unit-free, so it lets
    you compare the spread of birth weight against gestational age.
``skew``, ``kurtosis``
    Shape.  Kurtosis is *excess* kurtosis: 0 means normal-like tails.
``n_zero``, ``n_negative``, ``is_integer``, ``is_binary``
    Transform feasibility and dtype sanity.
``n_outliers``, ``outlier_pct``
    Per ``outlier_method``.
``shape``
    Plain-language summary such as ``"right_skewed, heavy_tails"``.

Notes
-----
No normality *test* is reported here on purpose -- see
:func:`test_normality` for why a p-value is the wrong tool once you
have more than a few hundred rows.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
percentiles : Sequence[float], default (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    Quantile probabilities in [0, 1] to include in the numeric profile.
outlier_method : Literal['iqr', 'mad', 'none'], default 'iqr'
    Outlier-counting rule: IQR fences, robust MAD bounds, or none.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### describe_categorical

```python
NaviLib.eda.describe_categorical(df: 'Frame', columns=None, top: 'int' = 3, max_unique: 'int' = 50) -> 'Frame'
```

```text
Profile categorical / low-cardinality columns.

``imbalance`` is the headline number: the share of the most common
level.  Above ~0.95 the column is effectively constant and will not
support a stable coefficient, no matter how good your model is.

``entropy_ratio`` (0 to 1) is the normalised Shannon entropy -- 1 means
perfectly uniform levels, 0 means one level dominates completely.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
top : int, default 3
    Maximum number of columns, categories or findings included in the
    displayed result.
max_unique : int, default 50
    Maximum cardinality for automatically selected categorical columns.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### test_normality

```python
NaviLib.eda.test_normality(df: 'Frame', columns=None, alpha: 'float' = 0.05) -> 'Frame'
```

```text
Normality assessment that stays honest at large sample sizes.

Any normality test rejects on large n, because real data are never
*exactly* normal and the test's power grows without bound.  Reporting
"p < 0.001, not normal" for 100 000 rows tells you nothing about
whether normality is a *useful approximation*.

This function therefore reports three things side by side:

- a p-value from the appropriate test (Shapiro-Wilk under n=5000,
  D'Agostino-Pearson above),
- the shape statistics that carry the effect size (skew, excess
  kurtosis),
- a ``verdict`` that combines both, and explicitly says
  ``"test_oversensitive"`` when n is large but the shape is close to
  normal.

Trust the ``verdict`` column, not the p-value.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
alpha : float, default 0.05
    Significance level in (0, 1); confidence intervals have nominal coverage
    1 - alpha.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### suggest_transform

```python
NaviLib.eda.suggest_transform(df: 'Frame', columns=None, target_skew: 'float' = 0.5) -> 'Frame'
```

```text
Recommend a variance-stabilising transform per numeric column.

The recommendation respects the arithmetic: ``log`` needs strictly
positive values, ``log1p`` tolerates zeros, ``sqrt`` needs
non-negatives, and ``yeo-johnson`` is the only one that handles
negative values.  Each candidate is actually applied and the resulting
skew measured, so ``skew_after`` is a real number and not a promise.

Returns a table with ``skew_before``, ``recommended``, ``skew_after``
and ``reason``.  Apply the winner yourself -- transforming is a
modelling decision, not an EDA side effect.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
target_skew : float, default 0.5
    Desired absolute skewness used when recommending a numeric
    transformation.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### relate

```python
NaviLib.eda.relate(df: 'Frame', target: 'str', columns=None, task: "Literal['auto', 'classification', 'regression']" = 'auto', max_unique_cat: 'int' = 50, sort_by: 'str' = 'strength') -> 'Frame'
```

```text
Rank every feature by how strongly it relates to the target.

This is the single most useful EDA table you can produce, and the one
the original library was missing entirely.  It picks the right
statistic for each dtype pair instead of forcing everything through
correlation:

============================  ==========================================
feature x target              statistic
============================  ==========================================
numeric x binary              AUC + Cliff's delta + Cohen's d
numeric x multiclass          eta squared (variance explained)
numeric x numeric             Spearman + Pearson correlation
categorical x categorical     Cramer's V (bias-corrected)
categorical x numeric         eta squared
============================  ==========================================

Effect sizes, not p-values: with 100 000 rows everything is
"significant", and with 138 events nothing is.  ``strength`` is a
comparable 0-1 score so the ranking is meaningful across mixed dtypes,
and ``interpretation`` translates it into words.

Returns
-------
DataFrame with ``feature``, ``dtype``, ``metric``, ``value``,
``strength``, ``interpretation``, ``n_used``, plus a ``note`` column
flagging near-perfect association (a likely leak).

Examples
--------
>>> eda.relate(df, target="died").head(10)
>>> eda.relate(df, target="price", task="regression")

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
target : str
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
task : Literal['auto', 'classification', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
max_unique_cat : int, default 50
    Maximum categorical cardinality considered in feature-target
    comparisons.
sort_by : str, default 'strength'
    Result column or metric used for ranking the output table.
```

### crosstab_target

```python
NaviLib.eda.crosstab_target(df: 'Frame', column: 'str', target: 'str', normalize: "Literal['index', 'columns', 'all', 'none']" = 'index', min_count: 'int' = 10) -> 'Frame'
```

```text
Event rate per level of a categorical feature, with sample sizes.

Rates alone are misleading: a level with 3 rows can show a 67% death
rate and mean nothing.  The ``n`` column and the ``reliable`` flag keep
that visible.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
column : str
    Name of the source column to inspect or transform.
target : str
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
normalize : Literal['index', 'columns', 'all', 'none'], default 'index'
    Return relative frequencies instead of raw counts, according to the
    supported normalization option.
min_count : int, default 10
    Minimum group count required for inclusion in the table.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### correlation_table

```python
NaviLib.eda.correlation_table(df: 'Frame', columns=None, method: "Literal['pearson', 'spearman', 'kendall']" = 'spearman', min_abs: 'float' = 0.0, target: 'Optional[str]' = None) -> 'Frame'
```

```text
Long-form correlation: one row per pair, sorted by strength.

Far easier to act on than a heatmap once you have more than ~15
columns.  Spearman is the default because it is monotone-robust and
does not assume linearity.

With ``target`` given, the correlation of each member of the pair with
the target is added, so you can tell which one to drop.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['pearson', 'spearman', 'kendall'], default 'spearman'
    Algorithm to use; see the supported methods and assumptions above.
min_abs : float, default 0.0
    Minimum absolute correlation retained in the returned table.
target : Optional[str], default None
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### missing_pattern

```python
NaviLib.eda.missing_pattern(df: 'Frame', top: 'int' = 15) -> 'Frame'
```

```text
The most common *combinations* of missing columns.

Column-by-column missing rates hide structure.  If ``lab_a``, ``lab_b``
and ``lab_c`` are always missing together, that is one phenomenon (the
panel was not ordered), not three -- and it changes how you impute.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
top : int, default 15
    Maximum number of columns, categories or findings included in the
    displayed result.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### missing_correlation

```python
NaviLib.eda.missing_correlation(df: 'Frame', min_abs: 'float' = 0.3) -> 'Frame'
```

```text
Which columns go missing *together* (nullity correlation).

A high value means the two columns' missingness is driven by the same
upstream cause.  Pairs listed here should be imputed with the same
strategy, or given a shared "was_missing" indicator.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
min_abs : float, default 0.3
    Minimum absolute correlation retained in the returned table.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### psi

```python
NaviLib.eda.psi(expected, actual, bins: 'int' = 10, eps: 'float' = 1e-06) -> 'float'
```

```text
Population Stability Index between two samples of one variable.

Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large
shift.  Bin edges come from ``expected`` (your reference / training
sample) so the comparison is anchored.

Parameters
----------
expected : object
    Reference observations that define the baseline distribution.
actual : object
    Current observations to compare against the reference distribution.
bins : int, default 10
    Number of bins, explicit edges, or supported automatic binning rule as
    indicated by the signature.
eps : float, default 1e-06
    Small positive probability floor preventing division by zero in
    distribution comparisons.

Returns
-------
float
    Population Stability Index; NaN when either nonmissing sample is empty.
```

### compare_distributions

```python
NaviLib.eda.compare_distributions(reference: 'Frame', current: 'Frame', columns=None, psi_bins: 'int' = 10, label_a: 'str' = 'reference', label_b: 'str' = 'current') -> 'Frame'
```

```text
Compare two frames column by column -- drift, or a bad split.

Use it for train vs test (they should look identical -- if they do not,
your split is broken or grouped incorrectly), for train vs production,
or for any two cohorts you want to contrast.

Reports PSI for every column, plus a KS statistic for numerics and a
Cramer's V of membership for categoricals, and a plain ``verdict``.

>>> tr, te = dp.split(df, "died")
>>> eda.compare_distributions(tr, te).head()

Parameters
----------
reference : pandas.DataFrame
    Reference dataset or datetime baseline, depending on this operation.
current : pandas.DataFrame
    Current DataFrame whose distributions are compared to the reference.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
psi_bins : int, default 10
    Number of reference-quantile bins for numeric Population Stability
    Index.
label_a : str, default 'reference'
    Human-readable name of the reference sample, used in output columns and
    legends.
label_b : str, default 'current'
    Human-readable name of the current sample, used in output columns and
    legends.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### compare_groups

```python
NaviLib.eda.compare_groups(df: 'Frame', group: 'str', columns=None, max_groups: 'int' = 10) -> 'Frame'
```

```text
Summarise every feature across the levels of a grouping column.

The classic "Table 1" of a clinical paper: mean +/- sd per group for
numerics, percentages for categoricals, plus a standardised difference
so you can see which contrasts are actually large.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
group : str
    Column name or names defining groups for the operation.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
max_groups : int, default 10
    Maximum number of groups included in the comparison.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### plot_distribution

```python
NaviLib.eda.plot_distribution(df: 'Frame', columns=None, kind: "Literal['full', 'hist', 'box']" = 'full', bins: 'Union[int, str]' = 'auto', kde: 'bool' = True, log_x: "Union[bool, Literal['auto']]" = 'auto', hue: 'Optional[str]' = None, figsize: 'Optional[Tuple[float, float]]' = None, max_cols: 'int' = 12, show: 'bool' = True, color: 'Optional[str]' = None)
```

```text
Plot the distribution of one or many numeric columns.

With one column and ``kind="full"`` you get the classic four-panel
diagnostic (histogram + KDE, boxplot, violin, QQ).  With several
columns you get a compact grid of histograms instead, because four
panels times twenty columns is not a plot anyone reads.

Parameters
----------
columns : str or list, optional
    ``None`` profiles every numeric column, capped at ``max_cols``.
log_x : bool or "auto", default "auto"
    ``"auto"`` switches to a log x-axis when the column is strictly
    positive and skew exceeds 2 -- otherwise the plot is one spike and
    a long empty tail.
hue : str, optional
    Split by a categorical column (e.g. the target) to compare
    distributions between classes.
bins : int or str
    Passed to numpy; ``"auto"`` uses the Freedman-Diaconis rule.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
kind : Literal['full', 'hist', 'box'], default 'full'
    Plot layout/type; choose one of the options shown in the signature.
kde : bool, default True
    Overlay a kernel-density estimate when the sample supports it.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
max_cols : int, default 12
    Maximum number of columns shown to keep the figure readable.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.
color : Optional[str], default None
    Explicit matplotlib color override. None uses the active NaviLib
    palette.

Returns
-------
matplotlib Figure.
```

### plot_categorical

```python
NaviLib.eda.plot_categorical(df: 'Frame', column: 'str', normalize: 'bool' = False, top_n: 'Optional[int]' = 20, include_missing: 'bool' = True, horizontal: "Union[bool, Literal['auto']]" = 'auto', sort: "Literal['count', 'index']" = 'count', ohe_prefix: 'bool' = True, figsize: 'Optional[Tuple[float, float]]' = None, color: 'Optional[str]' = None, title: 'Optional[str]' = None, show: 'bool' = True, return_counts: 'bool' = False)
```

```text
Bar chart of category frequencies, with one-hot columns handled properly.

Improvements over a plain ``value_counts().plot.bar()``:

- **Missing values are a bar**, not a silent omission, so a column that
  is 40% null cannot look complete.
- **One-hot detection is exact.**  A prefix match is only accepted when
  every candidate column is 0/1 valued; otherwise ``birth`` would
  hijack ``birth_weight``.  Case is preserved when stripping prefixes.
- **Automatic orientation**: long level names get a horizontal chart
  instead of 45-degree labels that overlap.
- **Truncation is announced** -- the title says how many levels are
  hidden by ``top_n``.

Returns the Figure, or ``(fig, counts)`` when ``return_counts=True``.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
column : str
    Name of the source column to inspect or transform.
normalize : bool, default False
    Return relative frequencies instead of raw counts, according to the
    supported normalization option.
top_n : Optional[int], default 20
    Maximum number of categories displayed; remaining levels are omitted
    from the plot.
include_missing : bool, default True
    Include missing observations as a separate category when True.
horizontal : Union[bool, Literal['auto']], default 'auto'
    Use horizontal bars; auto chooses based on label length and category
    count.
sort : Literal['count', 'index'], default 'count'
    Order categories by frequency or label, as supported by the signature.
ohe_prefix : bool, default True
    Prefix identifying one-hot columns to summarize as one categorical
    variable.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
color : Optional[str], default None
    Explicit matplotlib color override. None uses the active NaviLib
    palette.
title : Optional[str], default None
    Custom title shown above the chart or report.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.
return_counts : bool, default False
    Return (figure, counts_table) instead of only the figure.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### plot_target

```python
NaviLib.eda.plot_target(df: 'Frame', target: 'str', columns=None, top_k: 'int' = 9, task: "Literal['auto', 'classification', 'regression']" = 'auto', figsize: 'Optional[Tuple[float, float]]' = None, show: 'bool' = True)
```

```text
Grid showing how the strongest features relate to the target.

Features are ranked with :func:`relate` and the top ``top_k`` plotted,
each with the display that fits its dtype: overlaid densities or
boxplots for numeric features, event-rate bars for categorical ones.
For categorical panels a dashed line marks the overall base rate, so
you can see at a glance which levels sit above or below it.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
target : str
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
top_k : int, default 9
    Maximum number of features or categories shown.
task : Literal['auto', 'classification', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### plot_correlation

```python
NaviLib.eda.plot_correlation(df: 'Frame', columns=None, method: "Literal['pearson', 'spearman', 'kendall']" = 'spearman', cluster: 'bool' = True, annot: "Union[bool, Literal['auto']]" = 'auto', mask_upper: 'bool' = True, figsize: 'Optional[Tuple[float, float]]' = None, show: 'bool' = True)
```

```text
Correlation heatmap, optionally reordered so related blocks sit together.

``cluster=True`` applies hierarchical ordering on the correlation
distance, which turns a noisy checkerboard into visible groups of
redundant features.  Annotation is switched off automatically above 15
columns, where the numbers become unreadable anyway.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
method : Literal['pearson', 'spearman', 'kendall'], default 'spearman'
    Algorithm to use; see the supported methods and assumptions above.
cluster : bool, default True
    Reorder correlated variables by hierarchical clustering for easier
    visual inspection.
annot : Union[bool, Literal['auto']], default 'auto'
    Print numeric correlation values inside heatmap cells.
mask_upper : bool, default True
    Hide the upper triangle of a symmetric correlation matrix.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### plot_missing

```python
NaviLib.eda.plot_missing(df: 'Frame', max_cols: 'int' = 40, figsize: 'Optional[Tuple[float, float]]' = None, show: 'bool' = True)
```

```text
Two-panel missingness view: rate per column, and the nullity matrix.

The matrix (right panel) is the one worth studying: horizontal stripes
mean whole rows are incomplete, vertical blocks mean a column failed
wholesale, and aligned gaps across columns mean the same event caused
all of them.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
max_cols : int, default 40
    Maximum number of columns shown to keep the figure readable.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### plot_drift

```python
NaviLib.eda.plot_drift(reference: 'Frame', current: 'Frame', columns=None, top_k: 'int' = 6, label_a: 'str' = 'reference', label_b: 'str' = 'current', figsize: 'Optional[Tuple[float, float]]' = None, show: 'bool' = True)
```

```text
Overlay the two distributions for the most-shifted columns.

Columns are ranked by PSI (see :func:`compare_distributions`) so the
panels you get are the ones actually worth looking at.

Parameters
----------
reference : pandas.DataFrame
    Reference dataset or datetime baseline, depending on this operation.
current : pandas.DataFrame
    Current DataFrame whose distributions are compared to the reference.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
top_k : int, default 6
    Maximum number of features or categories shown.
label_a : str, default 'reference'
    Human-readable name of the reference sample, used in output columns and
    legends.
label_b : str, default 'current'
    Human-readable name of the current sample, used in output columns and
    legends.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### plot_balance

```python
NaviLib.eda.plot_balance(df: 'Frame', target: 'str', figsize: 'Tuple[float, float]' = (9, 4), show: 'bool' = True)
```

```text
Class distribution of the target, with the imbalance ratio spelled out.

Prints the number that decides your whole modelling strategy: how many
minority events you actually have.  Below ~50, no resampling technique
will save you -- the constraint is information, not balance.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
target : str
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
figsize : Tuple[float, float], default (9, 4)
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### report

```python
NaviLib.eda.report(df: 'Frame', target: 'Optional[str]' = None, plots: 'bool' = False, top: 'int' = 10, verbose: 'bool' = True) -> 'Dict[str, Any]'
```

```text
Run the whole EDA sweep and return every table in one dict.

Keys: ``shape``, ``numeric``, ``categorical``, ``normality``,
``transforms``, ``missing_pattern``, ``missing_corr``, ``correlations``,
and -- when ``target`` is given -- ``relate`` and ``balance``.

With ``verbose=True`` a short prioritised list of findings is printed:
quasi-constant columns, redundant pairs, suspicious associations,
heavy skew.  That summary is the point; the tables are for the details.

>>> res = eda.report(df, target="died")
>>> res["relate"].head()

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
target : Optional[str], default None
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
plots : bool, default False
    Include diagnostic figures alongside the numerical report.
top : int, default 10
    Maximum number of columns, categories or findings included in the
    displayed result.
verbose : bool, default True
    Print a concise progress/result summary when True.

Returns
-------
dict
    Analysis tables and optional figure handles for the selected EDA
    sections.
```

## NaviLib.feature_engineering

### transform_numeric

```python
NaviLib.feature_engineering.transform_numeric(df: 'Frame', columns=None, method: "Union[str, Literal['auto']]" = 'auto', candidates: 'Optional[Sequence[str]]' = None, replace: 'bool' = False, suffix: 'Optional[str]' = None, standardize: 'bool' = False, min_skew: 'float' = 0.5, return_state: 'bool' = False, random_state: 'int' = 42, verbose: 'bool' = False)
```

```text
Apply a variance-stabilising transform, or pick the best one automatically.

``method="auto"`` tries every feasible candidate on each column and keeps
the one with the smallest absolute skew -- but only if the column is
skewed enough to be worth it (``min_skew``), because transforming an
already-symmetric feature costs interpretability for nothing.

Available transforms
--------------------
``log``            strictly positive data only
``log1p``          tolerates zeros; auto-shifts if values go below -1
``sqrt``           auto-shifts negatives
``cuberoot``       handles negatives natively, no shift needed
``reciprocal``     no zeros
``boxcox``         strictly positive; fits a lambda
``yeojohnson``     the general-purpose choice, handles any sign
``quantile_normal`` forces an exactly normal marginal; very strong, and
                   it destroys the original spacing -- use when you care
                   about rank order, not units
``rank``           maps to a uniform [0, 1] by rank

Parameters
----------
replace : bool, default False
    Overwrite the column instead of adding ``<col>_<method>``.
standardize : bool, default False
    Only affects ``yeojohnson``.  The original code used
    ``standardize=False`` in one function and the sklearn default
    (``True``) in another, so "the same" transform gave two different
    results; here it is one explicit argument.
return_state : bool
    Return ``(df, state)`` so :func:`apply_state` can reproduce the
    exact lambda / quantile map on unseen data.  **Without this the
    transform is not reproducible** -- a fresh Box-Cox on the test set
    fits a different lambda and puts it on a different scale.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Union[str, Literal['auto']], default 'auto'
    Algorithm to use; see the supported methods and assumptions above.
candidates : Optional[Sequence[str]], default None
    Transform methods evaluated by automatic transformation selection.
suffix : Optional[str], default None
    Suffix for generated column names when the original column is retained.
min_skew : float, default 0.5
    Minimum absolute skewness before automatic transformation is attempted.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
verbose : bool, default False
    Print a concise progress/result summary when True.

Returns
-------
DataFrame, or ``(DataFrame, state)``.  The state's ``report`` field
holds the before/after skew and, for ``auto``, every candidate tried.
```

### bin_numeric

```python
NaviLib.feature_engineering.bin_numeric(df: 'Frame', columns=None, method: "Literal['quantile', 'uniform', 'kmeans', 'custom', 'tree']" = 'quantile', bins: 'Union[int, Sequence[float]]' = 5, labels: 'Optional[Sequence[str]]' = None, target: 'Optional[str]' = None, as_category: 'bool' = True, replace: 'bool' = False, suffix: 'str' = 'bin', return_state: 'bool' = False, random_state: 'int' = 42)
```

```text
Discretise numeric columns into bins.

Binning buys robustness to outliers and lets a linear model express a
non-monotone effect, at the cost of throwing away within-bin detail.

Methods
-------
``quantile``  equal-frequency bins.  Every bin has the same count, so
              no bin is too small to estimate -- usually what you want.
``uniform``   equal-width bins.  Intuitive edges, but on skewed data
              most rows land in one bin.
``kmeans``    1-D k-means on the values; edges follow natural gaps.
``tree``      **supervised**: a shallow decision tree picks the splits
              that best separate ``target``.  The most powerful option
              and the only one that uses the label -- so fit it on the
              training set only and replay with :func:`apply_state`.
``custom``    your own edges via ``bins=[0, 18, 65, 120]``.

Edges are stored in the state and extended to +/- infinity at replay
time, so a test-set value outside the training range lands in the
nearest bin instead of becoming NaN.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['quantile', 'uniform', 'kmeans', 'custom', 'tree'], default 'quantile'
    Algorithm to use; see the supported methods and assumptions above.
bins : Union[int, Sequence[float]], default 5
    Number of bins, explicit edges, or supported automatic binning rule as
    indicated by the signature.
labels : Optional[Sequence[str]], default None
    Explicit class or bin labels, in the order expected by the operation.
target : Optional[str], default None
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
as_category : bool, default True
    Return categorical interval/label columns instead of numeric bin codes.
replace : bool, default False
    Overwrite selected source columns in the returned copy instead of
    creating suffixed columns.
suffix : str, default 'bin'
    Suffix for generated column names when the original column is retained.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### add_interactions

```python
NaviLib.feature_engineering.add_interactions(df: 'Frame', columns=None, degree: 'int' = 2, operations: 'Sequence[str]' = ('multiply',), max_features: 'int' = 200, target: 'Optional[str]' = None, return_state: 'bool' = False)
```

```text
Build pairwise interaction and ratio features.

Trees find interactions on their own; linear models cannot, which is
where this pays off.  The number of pairs grows quadratically, so
``max_features`` caps the output and the function tells you what it
dropped rather than silently producing 5 000 columns.

Operations
----------
``multiply``  a * b -- the classic interaction
``divide``    a / b, guarded against division by zero, plus b / a
``add``       a + b
``subtract``  a - b (useful for dates-as-numbers, prices, scores)

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
degree : int, default 2
    Maximum polynomial interaction degree.
operations : Sequence[str], default ('multiply',)
    Interaction operations to generate from the selected numeric features.
max_features : int, default 200
    Upper bound on generated interaction features to control dimensional
    growth.
target : Optional[str], default None
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### add_aggregates

```python
NaviLib.feature_engineering.add_aggregates(df: 'Frame', group: 'Union[str, Sequence[str]]', values: 'Union[str, Sequence[str]]', funcs: 'Sequence[str]' = ('mean', 'std', 'min', 'max', 'count'), add_deviation: 'bool' = True, return_state: 'bool' = False)
```

```text
Group-level statistics joined back to every row.

"How does this patient's birth weight compare to the average in their
hospital?" is often more predictive than the raw value.  With
``add_deviation`` you also get ``<value>_dev_<group>`` -- the row's
distance from its group mean in group standard deviations, which is the
feature that usually carries the signal.

The fitted group table is stored in the state, so at transform time a
test row belonging to an unseen group gets the global statistic instead
of NaN.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
group : Union[str, Sequence[str]]
    Column name or names defining groups for the operation.
values : Union[str, Sequence[str]]
    Numeric measurement columns to aggregate within groups.
funcs : Sequence[str], default ('mean', 'std', 'min', 'max', 'count')
    Aggregation functions learned on training groups, such as mean, median
    or count.
add_deviation : bool, default True
    Also add the difference between each measurement and its fitted group
    mean.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### add_text_features

```python
NaviLib.feature_engineering.add_text_features(df: 'Frame', columns=None, parts: 'Sequence[str]' = ('length', 'n_words', 'n_digits', 'n_upper', 'n_special', 'avg_word_len'), drop_original: 'bool' = False, return_state: 'bool' = False)
```

```text
Cheap structural features from free-text columns.

Not a substitute for embeddings, but these six often carry surprising
signal in tabular problems -- message length, digit density and
capitalisation are classic spam and fraud indicators, and they cost
nothing to compute.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
parts : Sequence[str], default ('length', 'n_words', 'n_digits', 'n_upper', 'n_special', 'avg_word_len')
    Names of datetime/text components to derive; see the supported
    components above.
drop_original : bool, default False
    Remove source columns from the transformed copy after deriving new
    features.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### summary

```python
NaviLib.feature_engineering.summary(state: 'Union[Dict[str, Any], Sequence[Dict[str, Any]]]') -> 'Frame'
```

```text
Human-readable log of every fitted step: what ran, on what, producing what.

Worth printing into a notebook next to the model score -- three months
later this table is the only record of how the features were built.

Parameters
----------
state : Union[Dict[str, Any], Sequence[Dict[str, Any]]]
    Fitted state dictionary, or ordered sequence of states returned with
    return_state=True.

Returns
-------
pandas.DataFrame
    One row per feature-engineering state with its generated columns and
    fitted method.
```

### chain

```python
NaviLib.feature_engineering.chain(df: 'Frame', steps: 'Sequence[Tuple[Callable, Dict[str, Any]]]', verbose: 'bool' = False) -> 'Tuple[Frame, List[Dict[str, Any]]]'
```

```text
Run several feature-engineering steps and collect their states.

>>> train, states = fe.chain(train, [
...     (fe.transform_numeric, {"columns": ["income"], "method": "auto"}),
...     (fe.encode,            {"columns": ["city"], "method": "target",
...                             "target": "y"}),
...     (fe.scale,             {"method": "robust", "target": "y"}),
... ])
>>> test = nv.apply_state(test, states)

Each callable must accept ``return_state=True``; it is injected for you.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
steps : Sequence[Tuple[Callable, Dict[str, Any]]]
    Ordered (callable, keyword_arguments) pairs; each callable must support
    return_state=True.
verbose : bool, default False
    Print a concise progress/result summary when True.

Returns
-------
tuple of DataFrame and list
    Transformed frame and ordered fitted states for apply_state.
```

### scale_features

```python
NaviLib.feature_engineering.scale_features(df: 'Frame', columns=None, method: "Literal['standard', 'minmax', 'robust', 'maxabs', 'l2', 'none']" = 'standard', exclude: 'Optional[Sequence[str]]' = None, target: 'Optional[str]' = None, feature_range: 'Tuple[float, float]' = (0, 1), quantile_range: 'Tuple[float, float]' = (25.0, 75.0), replace: 'bool' = True, suffix: 'str' = 'scaled', return_state: 'bool' = False)
```

```text
Scale numeric columns, with the target automatically kept out.

Scaling the target along with the features is one of the easiest
mistakes to make with ``columns=None``, and one of the hardest to
notice -- so ``target`` is excluded explicitly here, and boolean and
already-binary 0/1 columns are skipped too, since rescaling an
indicator only makes it harder to read.

Methods
-------
``standard``  zero mean, unit variance.  Sensitive to outliers.
``minmax``    squeeze into ``feature_range``.  Very outlier-sensitive:
              one extreme value compresses everything else.
``robust``    centre on the median, scale by the IQR.  The safe default
              when your EDA showed heavy tails.
``maxabs``    divide by the maximum absolute value; preserves sparsity
              and zeros, so it is the right one for sparse matrices.
``l2``        scale each *row* to unit norm (not each column) -- for
              when relative composition matters, not absolute level.

Notes
-----
Unlike the original, this does not raise on missing values: NaN is
preserved through the transform so you can decide when to impute.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['standard', 'minmax', 'robust', 'maxabs', 'l2', 'none'], default 'standard'
    Algorithm to use; see the supported methods and assumptions above.
exclude : Optional[Sequence[str]], default None
    Columns excluded from automatic feature selection.
target : Optional[str], default None
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
feature_range : Tuple[float, float], default (0, 1)
    Lower and upper output bounds for min-max scaling.
quantile_range : Tuple[float, float], default (25.0, 75.0)
    Lower and upper percentile bounds used by robust scaling.
replace : bool, default True
    Overwrite selected source columns in the returned copy instead of
    creating suffixed columns.
suffix : str, default 'scaled'
    Suffix for generated column names when the original column is retained.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### encode_categorical

```python
NaviLib.feature_engineering.encode_categorical(df: 'Frame', columns=None, method: "Literal['onehot', 'ordinal', 'count', 'frequency', 'target', 'woe', 'hashing', 'binary']" = 'onehot', target: 'Optional[str]' = None, order: 'Optional[Dict[str, Sequence]]' = None, min_freq: 'Optional[Union[int, float]]' = None, other_label: 'str' = '__other__', drop_first: 'bool' = False, dummy_na: 'bool' = True, separator: 'Optional[str]' = None, n_components: 'int' = 8, cv: 'int' = 5, smoothing: 'float' = 20.0, drop_original: 'bool' = True, return_state: 'bool' = False, random_state: 'int' = 42)
```

```text
Encode categorical columns, leakage-safe and replayable.

Methods
-------
``onehot``     one indicator column per level.  Rare levels can be
               merged via ``min_freq``; unseen test levels map to all
               zeros (or to the ``other`` column when one exists).
``ordinal``    integer codes.  **Only meaningful when the levels really
               are ordered** -- pass the order explicitly via
               ``order={"size": ["S", "M", "L"]}``.  Without an order a
               warning is raised, because giving nominal categories
               fake numeric distances is a real modelling error, not a
               stylistic one.
``count``      how often the level occurs.
``frequency``  the same as a share of rows.
``target``     mean of the target per level, **computed out-of-fold**
               and smoothed towards the global mean.  See the note.
``woe``        weight of evidence, ``log(P(x|y=1) / P(x|y=0))``, for
               binary targets.  Monotone with the log-odds, which is
               why credit and clinical scorecards use it.
``hashing``    hash each level into ``n_components`` columns.  Fixed
               width regardless of cardinality, no state to store, no
               unseen-level problem -- at the cost of collisions and
               interpretability.
``binary``     base-2 encoding of the ordinal code: ``ceil(log2(k))``
               columns instead of ``k``.  A middle ground between
               one-hot and ordinal for high cardinality.

Target and WOE encoding: why out-of-fold
----------------------------------------
Encoding a level by its own rows' mean target leaks the label directly
into the feature; the model then "predicts" something it was handed.
Cross-fold encoding computes each row's value from the *other* folds,
which removes the leak.  Smoothing pulls small levels towards the
global mean by ``smoothing`` pseudo-counts, so a level seen twice does
not get a confident extreme value.

Parameters
----------
min_freq : int or float, optional
    Levels below this count (int) or share (float in 0-1) are merged
    into ``other_label`` before encoding.
separator : str, optional
    For multi-label cells such as ``"pop|rock|jazz"``.  Supported by
    ``onehot``, ``count`` and ``frequency`` -- and, unlike the original
    implementation, the count/frequency values are now computed over
    the exploded tokens and summed back per row, instead of trying to
    look up the whole unsplit string and quietly returning zeros.
dummy_na : bool, default True
    Give missing values their own indicator rather than dropping them
    silently.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
method : Literal['onehot', 'ordinal', 'count', 'frequency', 'target', 'woe', 'hashing', 'binary'], default 'onehot'
    Algorithm to use; see the supported methods and assumptions above.
target : Optional[str], default None
    Target column name. Keep it out of predictor transformations and fit
    supervised operations on training data only.
order : Optional[Dict[str, Sequence]], default None
    Mapping from column name to explicitly ordered category levels for
    ordinal/binary encoding.
other_label : str, default '__other__'
    Replacement category assigned to infrequent or unseen levels when
    grouping applies.
drop_first : bool, default False
    Drop the first sorted one-hot level to avoid a redundant indicator.
n_components : int, default 8
    Number of output dimensions for hashing or the selected projection.
cv : int, default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
smoothing : float, default 20.0
    Nonnegative pseudo-count strength shrinking rare category estimates
    toward the training-fold prior.
drop_original : bool, default True
    Remove source columns from the transformed copy after deriving new
    features.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
DataFrame, or ``(DataFrame, state)``.
```

### add_datetime_features

```python
NaviLib.feature_engineering.add_datetime_features(df: 'Frame', columns=None, parts: 'Sequence[str]' = ('year', 'month', 'day', 'dayofweek', 'hour', 'quarter', 'is_weekend', 'is_month_end'), cyclical: 'bool' = True, reference: 'Optional[Union[str, pd.Timestamp]]' = None, drop_original: 'bool' = False, return_state: 'bool' = False)
```

```text
Explode datetime columns into modelling-ready parts.

``cyclical=True`` additionally emits sine/cosine pairs for month, day of
week and hour.  This matters more than it looks: as a raw integer,
December (12) and January (1) are eleven units apart, and 23:00 and
01:00 are twenty-two -- the model has to spend capacity learning that
the scale wraps.  The sin/cos pair encodes the wrap directly.

``reference`` adds a ``<col>_days_since`` column measured from a fixed
date or another datetime column, which is usually the feature that
actually carries signal (account age, time since last visit).

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
parts : Sequence[str], default ('year', 'month', 'day', 'dayofweek', 'hour', 'quarter', 'is_weekend', 'is_month_end')
    Names of datetime/text components to derive; see the supported
    components above.
cyclical : bool, default True
    Also emit sine/cosine pairs for periodic datetime components.
reference : Optional[Union[str, pd.Timestamp]], default None
    Reference dataset or datetime baseline, depending on this operation.
drop_original : bool, default False
    Remove source columns from the transformed copy after deriving new
    features.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

### add_cyclical_features

```python
NaviLib.feature_engineering.add_cyclical_features(df: 'Frame', column: 'str', period: 'float', drop_original: 'bool' = False, return_state: 'bool' = False)
```

```text
Encode any periodic numeric column as a sin/cos pair.

For angles (``period=360``), compass bearings, day-of-year
(``period=365.25``), or anything else where the largest value is
adjacent to the smallest.

Parameters
----------
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
column : str
    Name of the source column to inspect or transform.
period : float
    Positive cycle length in the same units as the numeric input.
drop_original : bool, default False
    Remove source columns from the transformed copy after deriving new
    features.
return_state : bool, default False
    If True, return (transformed_frame, fitted_state). Replay this state on
    new data with NaviLib.apply_state; do not refit on test data.

Returns
-------
DataFrame or tuple of DataFrame and dict
    Transformed copy; when return_state=True, also returns fitted parameters
    for reuse on new data.
```

## NaviLib.statistical_tests

### find_correlated_features

```python
NaviLib.statistical_tests.find_correlated_features(df: pandas.DataFrame, test: str = 'pearson', alpha: float = 0.05, min_abs_corr: float = 0.3) -> pandas.DataFrame
```

```text
Find correlations between all feature pairs using correlation_test.

Parameters
----------
df : DataFrame
    Input dataset.
test : str, default="pearson"
    pearson / spearman / kendall
alpha : float, default=0.05
    Significance level.
min_abs_corr : float, default=0.3
    Minimum absolute correlation to report.

Returns
-------
DataFrame
    Results of correlation analysis between feature pairs.
```

### feature_correlation_analysis

```python
NaviLib.statistical_tests.feature_correlation_analysis(df: pandas.DataFrame, method: str = 'spearman', corr_threshold: float = 0.8, detect_onehot: bool = True, rare_threshold: float = 0.005, drop_constant: bool = True) -> Tuple[pandas.DataFrame, Dict[str, Any]]
```

```text
Analyze feature correlations with support for one-hot encoded features.

Parameters
----------
df : pandas.DataFrame
    Input dataset.
method : str, default="spearman"
    Correlation method (pearson, spearman, kendall).
corr_threshold : float, default=0.8
    Threshold for detecting high correlations.
detect_onehot : bool, default=True
    Whether to detect one-hot encoded feature groups.
rare_threshold : float, default=0.005
    Threshold for rare binary columns to drop.
drop_constant : bool, default=True
    Whether to drop constant columns.

Returns
-------
result_df : pandas.DataFrame
    DataFrame with correlated feature pairs.
report : dict
    Summary report of the analysis.
```

### effect_size

```python
NaviLib.statistical_tests.effect_size(df: pandas.DataFrame, metric: str, value_col: Optional[str] = None, group_col: Optional[str] = None) -> Dict[str, Any]
```

```text
Effect Size Calculator

Supports:
- "cohens_d" : Cohen's d (for two groups)
- "hedges_g" : Hedges' g (bias-corrected Cohen's d)
- "eta_squared" : Eta squared (for ANOVA)
- "omega_squared" : Omega squared (unbiased effect size)
- "cliffs_delta" : Cliff's Delta (non-parametric)
- "cramers_v" : Cramer's V (for categorical association)

Parameters
----------
df : pandas.DataFrame
    Input dataset.
metric : str
    Type of effect size to calculate.
value_col : str, optional
    Numerical variable (required for most metrics).
group_col : str, optional
    Grouping variable (required for most metrics).

Returns
-------
dict
    Dictionary containing effect size name and value.
```

### tukey_hsd_posthoc

```python
NaviLib.statistical_tests.tukey_hsd_posthoc(df: pandas.DataFrame, value_col: str, group_col: str, alpha: float = 0.05) -> pandas.DataFrame
```

```text
Perform Tukey HSD post-hoc test after ANOVA.

Parameters
----------
df : pandas.DataFrame
    Input dataset.
value_col : str
    Numerical variable.
group_col : str
    Grouping variable.
alpha : float, default=0.05
    Significance level.

Returns
-------
pandas.DataFrame
    Results of Tukey HSD test.
```

### test_normality

```python
NaviLib.statistical_tests.test_normality(df: pandas.DataFrame, column: str, method: str = 'shapiro', alpha: float = 0.05, kde: bool = True, figsize: Tuple[int, int] = (8, 5), bins: int = 30, show_plot: bool = True) -> Dict[str, Any]
```

```text
Performs a normality test on a given column, visualizes the distribution,
and returns a complete statistical report.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame.
column : str
    Column to test for normality.
method : str, default="shapiro"
    Normality test method:
    - "shapiro" : Shapiro-Wilk Test
    - "dagostino" : D'Agostino K² Test
    - "anderson" : Anderson-Darling Test
    - "ks" : Kolmogorov-Smirnov Test
alpha : float, default=0.05
    Significance level for hypothesis testing.
kde : bool, default=True
    Whether to show KDE curve on histogram.
figsize : tuple, default=(8, 5)
    Figure size for visualization.
bins : int, default=30
    Number of histogram bins.
show_plot : bool, default=True
    Whether to display the plots.

Returns
-------
dict
    Full statistical report including:
    - column, method, statistic, p_value, alpha, interpretation
```

### test_equal_variance

```python
NaviLib.statistical_tests.test_equal_variance(df: pandas.DataFrame, value_col: str, group_col: str, method: str = 'levene', alpha: float = 0.05, show_plot: bool = True, figsize: Tuple[int, int] = (8, 5)) -> Dict[str, Any]
```

```text
Performs a variance homogeneity test (equal variances across groups),
visualizes group distributions, and returns a full statistical report.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame.
value_col : str
    Numerical variable to analyze.
group_col : str
    Categorical grouping variable.
method : str, default="levene"
    Test method:
    - "levene" : Levene Test
    - "bartlett" : Bartlett Test
    - "brownforsythe" : Brown–Forsythe Test
alpha : float, default=0.05
    Significance level.
show_plot : bool, default=True
    Whether to display the boxplot.
figsize : tuple, default=(8, 5)
    Figure size for the plot.

Returns
-------
dict
    Full report including:
    - method, statistic, p_value, alpha, interpretation, group_sizes
```

### test_parametric

```python
NaviLib.statistical_tests.test_parametric(df: pandas.DataFrame, test: str, value: str, group: Union[str, List[str], NoneType] = None, subject: Optional[str] = None, covariate: Optional[str] = None, mu: float = 0, alpha: float = 0.05) -> Dict[str, Any]
```

```text
Universal parametric hypothesis testing engine.

Supported tests:
    - "one_sample"      : One-sample t-test
    - "independent_t"   : Independent two-sample t-test (Welch)
    - "paired_t"        : Paired-samples t-test
    - "oneway_anova"    : One-way ANOVA
    - "rm_anova"        : Repeated-measures ANOVA
    - "twoway_anova"    : Two-way ANOVA with interaction
    - "ancova"          : ANCOVA (group + covariate)

Parameters
----------
df : pandas.DataFrame
    Input dataset containing all variables.
test : str
    Name of the statistical test.
value : str
    Column name of the dependent variable.
group : str or list, optional
    Grouping variable(s). For two-way ANOVA pass a list of two group names.
subject : str, optional
    Subject identifier column (required for repeated-measures ANOVA).
covariate : str, optional
    Covariate column name (required for ANCOVA).
mu : float, default=0
    Population mean under H0 (for one-sample t-test).
alpha : float, default=0.05
    Significance level.

Returns
-------
dict
    Dictionary containing test statistics, p-values, and decision.
```

### test_nonparametric

```python
NaviLib.statistical_tests.test_nonparametric(df: pandas.DataFrame, test: str, value_col: str, group_col: Optional[str] = None, subject_col: Optional[str] = None, mu: float = 0, alpha: float = 0.05, show_plot: bool = True, figsize: Tuple[int, int] = (7, 5)) -> Dict[str, Any]
```

```text
Universal non-parametric test engine.

Supports:
- "wilcoxon_one_sample" : Wilcoxon signed-rank (one-sample)
- "mannwhitney" : Mann–Whitney U (two independent groups)
- "wilcoxon_paired" : Wilcoxon signed-rank (paired)
- "kruskal" : Kruskal–Wallis (3+ independent groups)
- "friedman" : Friedman Test (repeated measures)

Parameters
----------
df : pandas.DataFrame
    Input dataset.
test : str
    Name of the statistical test.
value_col : str
    Column name of the dependent variable.
group_col : str, optional
    Grouping variable (required for most tests except one-sample).
subject_col : str, optional
    Subject identifier (required for Friedman test).
mu : float, default=0
    Population median under H0 (for one-sample Wilcoxon).
alpha : float, default=0.05
    Significance level.
show_plot : bool, default=True
    Whether to display visualization.
figsize : tuple, default=(7, 5)
    Figure size for plots.

Returns
-------
dict
    Dictionary containing test statistics, p-values, and interpretation.
```

### test_correlation

```python
NaviLib.statistical_tests.test_correlation(df: pandas.DataFrame, test: str, x: str, y: str, control: Optional[str] = None, alpha: float = 0.05, show_plot: bool = True, figsize: Tuple[int, int] = (6, 5)) -> Dict[str, Any]
```

```text
Correlation analysis engine.

Supports:
- "pearson" : Pearson correlation
- "spearman" : Spearman rank correlation
- "kendall" : Kendall Tau correlation
- "partial" : Partial correlation

Parameters
----------
df : pandas.DataFrame
    Input dataset.
test : str
    Type of correlation test.
x : str
    First variable name.
y : str
    Second variable name.
control : str, optional
    Control variable (required for partial correlation).
alpha : float, default=0.05
    Significance level.
show_plot : bool, default=True
    Whether to display scatter plot.
figsize : tuple, default=(6, 5)
    Figure size for the plot.

Returns
-------
dict
    Dictionary containing correlation coefficient, p-value, and interpretation.
```

### test_categorical

```python
NaviLib.statistical_tests.test_categorical(df: pandas.DataFrame, test: str, col1: Optional[str] = None, col2: Optional[str] = None, expected: Optional[List[float]] = None, alpha: float = 0.05, show_plot: bool = True, figsize: Tuple[int, int] = (6, 5)) -> Dict[str, Any]
```

```text
Categorical Data Test Engine.

Supports:
- "chi_square_independence" : Chi-square Test of Independence
- "chi_square_gof" : Chi-square Goodness-of-fit
- "fisher_exact" : Fisher's Exact Test (for 2x2 tables)
- "mcnemar" : McNemar Test (paired categorical)

Parameters
----------
df : pandas.DataFrame
    Input dataset.
test : str
    Type of categorical test.
col1 : str, optional
    First categorical variable.
col2 : str, optional
    Second categorical variable (for independence tests).
expected : list, optional
    Expected proportions or counts (for goodness-of-fit).
alpha : float, default=0.05
    Significance level.
show_plot : bool, default=True
    Whether to display heatmap of contingency table.
figsize : tuple, default=(6, 5)
    Figure size for the plot.

Returns
-------
dict
    Dictionary containing test statistics, p-values, and interpretation.
```

### adjust_pvalues

```python
NaviLib.statistical_tests.adjust_pvalues(p_values, *, method: str = 'fdr_bh', alpha: float = 0.05) -> pandas.DataFrame
```

```text
Correct a family of hypothesis tests while preserving missing results.

Parameters
----------
p_values : one-dimensional array-like or pandas.Series
    Raw p-values in [0, 1]. NaNs remain missing and are excluded from the
    correction family; a Series index is preserved.
method : {'fdr_bh', 'fdr_by', 'holm', 'bonferroni'}, default 'fdr_bh'
    Benjamini-Hochberg/BY false-discovery control, or Holm/Bonferroni
    family-wise error control. BH assumes independence or suitable positive
    dependence; BY accommodates arbitrary dependence.
alpha : float, default 0.05
    Rejection level in (0, 1).

Returns
-------
pandas.DataFrame
    Columns p_value, p_adjusted, reject. Missing tests have reject=False.
    All tests in the intended family must be supplied together.
```

## NaviLib.modeling

### ChainTransformer

```python
NaviLib.modeling.ChainTransformer(steps: 'Sequence[Tuple[Callable, Dict[str, Any]]]', target: 'Optional[str]' = None, drop_target: 'bool' = True)
```

```text
Wrap a NaviLib preprocessing chain as a scikit-learn transformer.

This is what makes leak-free cross-validation possible without giving
up the state-based API.  On ``fit`` it runs the steps and captures
their states; on ``transform`` it replays those states via
:func:`NaviLib.apply_state`.  Put it in a ``Pipeline`` and every
``cross_val_score`` refits the preprocessing on the training part of
each fold -- the only correct way to do it when any step learns from
the data, which target encoding, imputation, scaling, Box-Cox and
SMOTE all do.

Parameters
----------
steps : list of (callable, kwargs)
    Same shape as :func:`feature_engineering.chain`.  Each callable
    must accept ``return_state=True``; it is injected for you.
target : str, optional
    Target column name.  Supervised steps (target encoding, tree
    binning) need it, so ``y`` is temporarily attached to ``X`` under
    this name during ``fit`` and removed afterwards.  At ``transform``
    time no target is attached or used -- that asymmetry is exactly
    what keeps the encoding honest.
drop_target : bool, default True
    Remove the target column from the transformed output.

Attributes
----------
states_ : list of dict
    The fitted chain.  Hand it to :func:`NaviLib.save_state`.
feature_names_out_ : list of str

Examples
--------
>>> prep = ChainTransformer([
...     (cl.fix_missing, {"method": "median"}),
...     (fe.encode, {"columns": ["city"], "method": "target", "target": "y"}),
... ], target="y")
>>> prep.fit_transform(X_train, y_train).shape
>>> prep.transform(X_test).shape          # replayed, not refitted
```

### make_preprocessor

```python
NaviLib.modeling.make_preprocessor(numeric: 'Optional[Sequence[str]]' = None, categorical: 'Optional[Sequence[str]]' = None, numeric_impute: 'str' = 'median', categorical_impute: 'str' = 'most_frequent', scale: 'bool' = True, one_hot: 'bool' = True, min_frequency: 'float' = 0.01)
```

```text
A plain ColumnTransformer, for when the preprocessing really is simple.

Use this when all you need is impute + scale + one-hot.  Reach for
:class:`ChainTransformer` when you need target encoding, Box-Cox, group
aggregates or supervised binning, none of which a ColumnTransformer
expresses.

Unknown categories at predict time go to the infrequent bucket instead
of raising, and rare levels are folded together at ``min_frequency``.

Parameters
----------
numeric : Optional[Sequence[str]], default None
    Names of numeric columns passed to the numeric preprocessing branch.
categorical : Optional[Sequence[str]], default None
    Categorical column names or indices expected by the selected
    preprocessing/resampling operation.
numeric_impute : str, default 'median'
    SimpleImputer strategy for the numeric branch.
categorical_impute : str, default 'most_frequent'
    SimpleImputer strategy for the categorical branch.
scale : bool, default True
    Standardize numeric inputs before this operation when True.
one_hot : bool, default True
    One-hot encode categorical values after imputation when True.
min_frequency : float, default 0.01
    Minimum count or frequency share retained as an individual one-hot
    category.

Returns
-------
sklearn.compose.ColumnTransformer
    Unfitted numeric/categorical preprocessing transformer.
```

### make_pipeline

```python
NaviLib.modeling.make_pipeline(*steps, balance: 'Optional[str]' = None, ratio='auto', random_state: 'int' = 42)
```

```text
Assemble a pipeline, optionally resampling inside the fold.

``balance`` takes any method name from ``cleaning.BALANCE_METHODS``.
The sampler is inserted immediately before the final estimator, so it
sees only the training part of each fold and is skipped entirely at
predict time.  Resampling anywhere else leaks.

>>> pipe = md.make_pipeline(prep, LGBMClassifier(), balance="smote", ratio=0.3)

Parameters
----------
balance : Optional[str], default None
    Optional resampling method inserted before the final estimator inside
    the training fold.
ratio : optional, default 'auto'
    Sampling strategy passed to imbalanced-learn: supported string, ratio,
    or class-count mapping.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
    Unfitted pipeline with the final estimator named model.
```

### compare_algorithms

```python
NaviLib.modeling.compare_algorithms(X, y, models: 'Optional[Dict[str, Any]]' = None, cv: 'Union[int, Any]' = 5, scoring=None, task: "Literal['auto', 'classification', 'multiclass', 'regression']" = 'auto', groups=None, include_baseline: 'bool' = True, n_jobs: 'int' = -1, random_state: 'int' = 42, verbose: 'bool' = True) -> 'Frame'
```

```text
Leaderboard across several algorithms on identical folds.

A dummy predictor is included by default and should be read first.  On
imbalanced data a model can look impressive and be doing nothing: if
your gradient booster scores 0.91 accuracy and the dummy scores 0.91,
you have learned the base rate and nothing else.

``models=None`` runs a default set -- a regularised linear model, a
random forest and gradient boosting.  Every model sees the same splits,
so the comparison is paired and the ``_sd`` columns are comparable.

>>> md.compare_algorithms(X, y)

Parameters
----------
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : object
    Observed target values, positionally aligned with X.
models : Optional[Dict[str, Any]], default None
    Mapping of display names to estimators. None uses the built-in candidate
    models.
cv : Union[int, Any], default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
scoring : optional, default None
    Scikit-learn scorer name or mapping of names to scorers. None selects
    task-specific defaults. Negative loss scorers are converted to positive
    losses in comparison tables.
task : Literal['auto', 'classification', 'multiclass', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
groups : optional, default None
    Group identifiers aligned with rows; observations in one group stay in
    the same validation partition.
include_baseline : bool, default True
    Include a dummy predictor to contextualize model performance.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
verbose : bool, default True
    Print a concise progress/result summary when True.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### nested_validate

```python
NaviLib.modeling.nested_validate(model, param_space: 'Dict[str, Any]', X, y, inner_cv: 'int' = 3, outer_cv: 'int' = 5, n_iter: 'int' = 20, scoring=None, task: "Literal['auto', 'classification', 'multiclass', 'regression']" = 'auto', groups=None, n_jobs: 'int' = -1, random_state: 'int' = 42, verbose: 'bool' = True) -> 'Frame'
```

```text
Nested cross-validation: the honest score for a model you tuned.

Tuning on folds and then reporting the best score from those same folds
reuses the test data for selection, and the optimism is not small --
with a wide search on a small dataset it is routinely worth several
points.  Nested CV puts the whole search inside each outer fold, so the
outer score is measured on data no part of the procedure has seen.

Expect the nested score to be **lower** than :func:`tune`'s
``best_score``.  The gap is the selection bias you would otherwise have
reported as model quality.

Returns
-------
DataFrame of outer-fold scores, with the chosen parameters per fold in
``attrs["params_per_fold"]`` -- if those differ wildly across folds,
the search is unstable and the tuned model should not be trusted.

Parameters
----------
model : object
    Scikit-learn-compatible estimator or pipeline. Include learned
    preprocessing inside the pipeline during cross-validation.
param_space : Dict[str, Any]
    Estimator parameter distributions/lists. Pipeline parameter names use
    step__parameter syntax.
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : object
    Observed target values, positionally aligned with X.
inner_cv : int, default 3
    Number of inner folds used only for hyperparameter selection.
outer_cv : int, default 5
    Number of outer folds used for evaluating the complete tuning procedure.
n_iter : int, default 20
    Number of parameter settings sampled by randomized search.
scoring : optional, default None
    Scikit-learn scorer name or mapping of names to scorers. None selects
    task-specific defaults. Negative loss scorers are converted to positive
    losses in comparison tables.
task : Literal['auto', 'classification', 'multiclass', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
groups : optional, default None
    Group identifiers aligned with rows; observations in one group stay in
    the same validation partition.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
verbose : bool, default True
    Print a concise progress/result summary when True.
```

### explain

```python
NaviLib.modeling.explain(model, X, y=None, method: "Literal['permutation', 'builtin', 'shap', 'coef']" = 'permutation', scoring=None, n_repeats: 'int' = 10, max_display: 'Optional[int]' = None, sample: 'Optional[int]' = 2000, random_state: 'int' = 42, n_jobs: 'int' = -1) -> 'Frame'
```

```text
Rank features by importance, with the caveats each method carries.

Methods
-------
``permutation``
    Shuffle a column and measure the damage to a held-out score.  The
    default, because it is model-agnostic and measures what you
    actually care about.  Two caveats it reports for you: importances
    can be **negative** (the feature is noise), and correlated features
    share credit, so a genuinely important feature can look
    unimportant if a near-duplicate is still available to the model.
``builtin``
    The estimator's own ``feature_importances_``.  Free, but tree
    impurity importance is biased toward high-cardinality and
    continuous features.
``coef``
    Linear model coefficients.  Only comparable across features if the
    inputs were scaled -- this is checked and flagged.
``shap``
    Per-row attributions, if the ``shap`` package is installed.  The
    only one of the four that explains an individual prediction rather
    than the model as a whole.

Returns
-------
DataFrame sorted by importance, with ``sd`` where the method provides
it and a ``note`` column flagging negative or unstable values.

Parameters
----------
model : object
    Scikit-learn-compatible estimator or pipeline. Include learned
    preprocessing inside the pipeline during cross-validation.
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : optional, default None
    Observed target values, positionally aligned with X.
method : Literal['permutation', 'builtin', 'shap', 'coef'], default 'permutation'
    Algorithm to use; see the supported methods and assumptions above.
scoring : optional, default None
    Scikit-learn scorer name or mapping of names to scorers. None selects
    task-specific defaults. Negative loss scorers are converted to positive
    losses in comparison tables.
n_repeats : int, default 10
    Number of feature permutations used to estimate importance and its
    spread.
max_display : Optional[int], default None
    Maximum number of features included in an explanation plot.
sample : Optional[int], default 2000
    Maximum number of observations sampled for plotting or expensive
    diagnostics.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.
```

### plot_importance

```python
NaviLib.modeling.plot_importance(importance: 'Frame', top: 'int' = 20, figsize: 'Optional[Tuple[float, float]]' = None, show: 'bool' = True)
```

```text
Horizontal bar chart of an :func:`explain` table.

Error bars are the spread across permutation repeats; a bar whose error
bar crosses zero is not distinguishable from noise, and is drawn in
grey to say so.

Parameters
----------
importance : pandas.DataFrame
    Feature-importance table returned by explain.
top : int, default 20
    Maximum number of columns, categories or findings included in the
    displayed result.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### learning_curve

```python
NaviLib.modeling.learning_curve(model, X, y, sizes: 'Sequence[float]' = (0.1, 0.25, 0.5, 0.75, 1.0), cv: 'Union[int, Any]' = 5, scoring=None, task: "Literal['auto', 'classification', 'multiclass', 'regression']" = 'auto', n_jobs: 'int' = -1, random_state: 'int' = 42) -> 'Frame'
```

```text
Would more data help? Train and validation score against training size.

Read the two curves at the right-hand edge:

- **Converged and both low** -- the model is underfitting. More rows
  will not help; you need better features or a stronger model.
- **Still separated, validation rising** -- more data will help.
- **Wide gap, validation flat** -- overfitting. Regularise or cut
  features; more rows help only slowly.

This is the cheapest way to decide whether to spend the next week on
data collection or on modelling.

Parameters
----------
model : object
    Scikit-learn-compatible estimator or pipeline. Include learned
    preprocessing inside the pipeline during cross-validation.
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : object
    Observed target values, positionally aligned with X.
sizes : Sequence[float], default (0.1, 0.25, 0.5, 0.75, 1.0)
    Training-size fractions or absolute counts for the learning curve.
cv : Union[int, Any], default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
scoring : optional, default None
    Scikit-learn scorer name or mapping of names to scorers. None selects
    task-specific defaults. Negative loss scorers are converted to positive
    losses in comparison tables.
task : Literal['auto', 'classification', 'multiclass', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### plot_learning_curve

```python
NaviLib.modeling.plot_learning_curve(curve: 'Frame', figsize: 'Tuple[float, float]' = (7, 4.5), show: 'bool' = True)
```

```text
Plot a :func:`learning_curve` table with its verdict in the title.

Parameters
----------
curve : pandas.DataFrame
    Summary DataFrame returned by modeling.learning_curve.
figsize : Tuple[float, float], default (7, 4.5)
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### save_model

```python
NaviLib.modeling.save_model(artifact: 'Dict[str, Any]', path: 'str', states: 'Optional[Sequence[Dict[str, Any]]]' = None, metrics: 'Optional[Frame]' = None, notes: 'str' = '', compress: 'int' = 3) -> 'str'
```

```text
Save the whole pipeline: preprocessing states, model, threshold, metrics.

Saving only the estimator is the usual mistake.  Six months later the
model loads fine and produces nonsense, because nobody recorded which
median was used for imputation, which categories the encoder knew, or
what threshold the reported sensitivity assumed.  Everything needed to
reproduce a prediction goes in one file.

Unpickling executes code, so never load an artifact you did not create.

Parameters
----------
artifact : Dict[str, Any]
    Model artifact returned by train_model. predict_model also accepts a
    load_model bundle and replays any separately stored states.
path : str
    Destination or source filesystem path; pathlib.Path is also accepted.
states : Optional[Sequence[Dict[str, Any]]], default None
    Ordered fitted preprocessing states to store alongside the model.
metrics : Optional[pandas.DataFrame], default None
    Metrics to calculate, or a metrics table to store, as indicated by the
    signature.
notes : str, default ''
    Free-text provenance or context saved with the model bundle.
compress : int, default 3
    Joblib compression level; 0 disables compression and larger values trade
    speed for size.

Returns
-------
str
    Absolute path to the written joblib bundle.
```

### load_model

```python
NaviLib.modeling.load_model(path: 'str', check_versions: 'bool' = True) -> 'Dict[str, Any]'
```

```text
Load a :func:`save_model` bundle, warning on library version drift.

Parameters
----------
path : str
    Destination or source filesystem path; pathlib.Path is also accepted.
check_versions : bool, default True
    Warn when saved and installed scikit-learn versions differ. Only load
    trusted joblib files.

Returns
-------
dict
    Bundle containing artifact, optional states, metrics, notes and version
    metadata. Pass directly to predict_model, or access bundle["artifact"].
```

### cross_validate_model

```python
NaviLib.modeling.cross_validate_model(model, X, y, cv: 'Union[int, Any]' = 5, scoring=None, task: "Literal['auto', 'classification', 'multiclass', 'regression']" = 'auto', groups=None, return_oof: 'bool' = True, n_jobs: 'int' = -1, random_state: 'int' = 42, verbose: 'bool' = True) -> 'Frame'
```

```text
Cross-validate, reporting per-fold spread rather than a bare mean.

The ``sd`` and ``folds`` columns are the point.  A model at 0.81 ± 0.09
and one at 0.78 ± 0.02 are not ranked by their means: with five folds
that gap is well inside the noise, and reporting "0.81 beats 0.78"
invents a result that is not there.

The ``overfit_gap`` column (train minus test) is the second thing to
read.  A large gap means the model is memorising, and no amount of
threshold tuning will fix that.

Parameters
----------
groups : array-like, optional
    Grouping variable (patient, hospital, user).  Switches to
    ``GroupKFold`` so one group cannot straddle a split.  Without it,
    repeated measurements on the same subject leak between train and
    test and the score is optimistic by a wide margin.
return_oof : bool, default True
    Attach out-of-fold predictions to ``result.attrs["oof"]``.  These
    are what to hand to ``evaluation.tune_threshold`` and
    ``evaluation.plot_calibration``: every prediction was made by a
    model that had not seen that row.
model : object
    Scikit-learn-compatible estimator or pipeline. Include learned
    preprocessing inside the pipeline during cross-validation.
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : object
    Observed target values, positionally aligned with X.
cv : Union[int, Any], default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
scoring : optional, default None
    Scikit-learn scorer name or mapping of names to scorers. None selects
    task-specific defaults. Negative loss scorers are converted to positive
    losses in comparison tables.
task : Literal['auto', 'classification', 'multiclass', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
verbose : bool, default True
    Print a concise progress/result summary when True.

Returns
-------
DataFrame indexed by metric, with ``mean``, ``sd``, ``min``, ``max``,
``train_mean``, ``overfit_gap`` and the individual ``folds``.
```

### tune_model

```python
NaviLib.modeling.tune_model(model, param_space: 'Dict[str, Any]', X, y, search: "Literal['random', 'grid', 'halving']" = 'random', n_iter: 'int' = 40, cv: 'Union[int, Any]' = 5, scoring=None, task: "Literal['auto', 'classification', 'multiclass', 'regression']" = 'auto', groups=None, refit: 'bool' = True, n_jobs: 'int' = -1, random_state: 'int' = 42, verbose: 'bool' = True)
```

```text
Hyper-parameter search that reports how much of the gain is noise.

``search="random"`` is the default because on a budget it beats grid
search: a grid spends most of its evaluations varying parameters that
do not matter, while random sampling explores the ones that do.
``halving`` runs successive halving, which is far cheaper on large data.

The returned ``results`` table carries ``mean_test_score`` **and**
``std_test_score``.  Read them together: if the top ten configurations
sit inside one standard deviation of each other, you have not found a
better model, you have found the noise floor -- and the "best"
parameters will not reproduce on a new split.

Parameters
----------
param_space : dict
    For ``random``, scipy distributions are allowed
    (``{"model__max_depth": randint(3, 12)}``).  For ``grid``, lists.
refit : bool, default True
    Refit the best configuration on all of ``X``.
model : object
    Scikit-learn-compatible estimator or pipeline. Include learned
    preprocessing inside the pipeline during cross-validation.
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : object
    Observed target values, positionally aligned with X.
n_iter : int, default 40
    Number of parameter settings sampled by randomized search.
cv : Union[int, Any], default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
scoring : optional, default None
    Scikit-learn scorer name or mapping of names to scorers. None selects
    task-specific defaults. Negative loss scorers are converted to positive
    losses in comparison tables.
task : Literal['auto', 'classification', 'multiclass', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
groups : optional, default None
    Group identifiers aligned with rows; observations in one group stay in
    the same validation partition.
n_jobs : int, default -1
    Parallel workers; -1 uses all available processors, 1 runs serially.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
verbose : bool, default True
    Print a concise progress/result summary when True.

Returns
-------
dict with ``best_model``, ``best_params``, ``best_score``,
``results`` (all configurations, sorted) and ``search`` (the fitted
search object).

Notes
-----
``best_score`` is **optimistically biased**: it is the maximum over
many configurations evaluated on the same folds, so it partly measures
which configuration happened to suit those folds. Use
:func:`nested_validate` for an unbiased estimate.
```

### train_model

```python
NaviLib.modeling.train_model(model, X, y, threshold: "Optional[Union[float, Literal['auto']]]" = None, cost_fn: 'float' = 1.0, cost_fp: 'float' = 1.0, cv: 'int' = 5, calibrate: "Optional[Literal['sigmoid', 'isotonic']]" = None, task: "Literal['auto', 'classification', 'multiclass', 'regression']" = 'auto', random_state: 'int' = 42, verbose: 'bool' = True) -> 'Dict[str, Any]'
```

```text
Fit on all the data and return a ready-to-ship artifact.

Two optional extras that decide whether the model is usable in
practice:

``threshold="auto"``
    Choose the decision threshold from *out-of-fold* predictions using
    the cost ratio you supply, instead of leaving it at the arbitrary
    0.5.  Picking it from in-sample predictions would bias it toward
    this dataset, so cross-validated predictions are used.
``calibrate``
    Wrap the model in ``CalibratedClassifierCV``.  Worth doing whenever
    you resampled, boosted, or otherwise care about the probability
    itself rather than the ranking.

Returns
-------
dict with ``model``, ``threshold``, ``task``, ``classes``,
``feature_names``, ``trained_at`` and ``versions`` -- the shape
:func:`save_model` expects.

Parameters
----------
model : object
    Scikit-learn-compatible estimator or pipeline. Include learned
    preprocessing inside the pipeline during cross-validation.
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y : object
    Observed target values, positionally aligned with X.
threshold : Optional[Union[float, Literal['auto']]], default None
    Decision cutoff or inspection threshold; see the function-specific
    interpretation above.
cost_fn : float, default 1.0
    Nonnegative cost assigned to one false negative.
cost_fp : float, default 1.0
    Nonnegative cost assigned to one false positive.
cv : int, default 5
    Number of cross-validation folds, or a compatible splitter where the
    signature allows one. Use group/time-aware folds for dependent
    observations.
calibrate : Optional[Literal['sigmoid', 'isotonic']], default None
    Optional sigmoid or isotonic probability calibration fitted by cross-
    validation.
task : Literal['auto', 'classification', 'multiclass', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
verbose : bool, default True
    Print a concise progress/result summary when True.
```

### predict_model

```python
NaviLib.modeling.predict_model(artifact: 'Dict[str, Any]', X, correct_prior: 'bool' = False, true_prevalence: 'Optional[float]' = None) -> 'Frame'
```

```text
Predict with an artifact, honouring its stored threshold.

Returns a frame with ``prediction`` and, for classifiers,
``probability``.  This exists so the threshold chosen at training time
actually gets used -- calling ``model.predict`` directly silently
reverts to 0.5 and quietly undoes the tuning.

``correct_prior=True`` rescales probabilities back to
``true_prevalence`` using :func:`cleaning.prior_correct`, for models
trained on resampled data.

Parameters
----------
artifact : Dict[str, Any]
    Model artifact returned by train_model. predict_model also accepts a
    load_model bundle and replays any separately stored states.
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
correct_prior : bool, default False
    Adjust binary probabilities from training prevalence to true_prevalence.
    Assumes prior shift with unchanged class-conditional distributions.
true_prevalence : Optional[float], default None
    Positive-class prevalence in the deployment population, in (0, 1).

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

## NaviLib.evaluation

### bootstrap_ci

```python
NaviLib.evaluation.bootstrap_ci(metric_fn: 'Callable[..., float]', *arrays: 'ArrayLike', n_boot: 'int' = 1000, alpha: 'float' = 0.05, stratify: 'Optional[ArrayLike]' = None, random_state: 'int' = 42) -> 'Tuple[float, float, float]'
```

```text
Percentile bootstrap confidence interval for any metric.

Resamples rows with replacement ``n_boot`` times and takes the empirical
quantiles.  With ``stratify`` the class proportions are held fixed,
which matters when the positive class is small enough that some
resamples would otherwise contain no events at all.

Parameters
----------
metric_fn : callable
    Takes the arrays positionally and returns a float.
*arrays : array-like
    Passed to ``metric_fn``, resampled together row-wise.
n_boot : int, default 1000
    More is smoother; 1000 is plenty for a 95% interval.
stratify : array-like, optional
    Usually ``y_true``.  Strongly recommended for imbalanced data.
alpha : float, default 0.05
    Significance level in (0, 1); confidence intervals have nominal coverage
    1 - alpha.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
(point_estimate, lower, upper)

>>> from sklearn.metrics import roc_auc_score
>>> ev.bootstrap_ci(roc_auc_score, y_true, y_prob, stratify=y_true)
(0.918, 0.847, 0.968)
```

### score_regression

```python
NaviLib.evaluation.score_regression(y_true: 'ArrayLike', y_pred: 'ArrayLike', ci: 'bool' = False, n_boot: 'int' = 500, baseline: "Literal['mean', 'median', 'none']" = 'mean', sample_weight: 'Optional[ArrayLike]' = None, random_state: 'int' = 42) -> 'Frame'
```

```text
Regression metrics with a naive baseline for context.

Every scale-dependent error (MAE, RMSE) is meaningless on its own -- an
RMSE of 400 is excellent for house prices and catastrophic for
probabilities.  The ``vs_baseline`` column expresses each error as a
ratio against always predicting the evaluation-sample mean (or median), so
values below 1 mean the model beats the naive rule and values above 1
mean it does not.

Metrics
-------
``mae``, ``rmse``, ``medae``, ``max_error``   scale-dependent errors
``r2``, ``explained_variance``                unit-free, 1 = perfect
``mape``, ``smape``                           percentage errors
``bias``                                      mean residual; a non-zero
                                              value means the model is
                                              systematically high or low
``r_pearson``, ``r_spearman``                 correlation of pred vs true

Notes
-----
MAPE is reported as NaN when any true value is zero, rather than being
silently rescued with an epsilon -- an epsilon of 1e-8 turns a single
zero into an 8-digit percentage error and destroys the average.
``smape`` is given as the symmetric alternative that survives zeros.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : array-like
    Predicted labels or numeric outcomes, positionally aligned with y_true.
ci : bool, default False
    Request bootstrap confidence intervals for supported metrics. This adds
    repeated metric computation.
n_boot : int, default 500
    Number of paired bootstrap resamples used to estimate metric
    uncertainty.
baseline : Literal['mean', 'median', 'none'], default 'mean'
    Constant baseline derived from this evaluation sample: mean, median,
    or none. It is descriptive, not a separately trained baseline model.
sample_weight : Optional[array-like], default None
    Finite nonnegative row weights, with positive total weight. Applied to
    MAE, RMSE, median absolute error, bias, R2, explained variance, MAPE,
    SMAPE, their supported bootstrap intervals and constant baselines.
    Correlations and maximum error remain unweighted.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### plot_regression

```python
NaviLib.evaluation.plot_regression(y_true: 'ArrayLike', y_pred: 'ArrayLike', figsize: 'Tuple[float, float]' = (14, 9), sample: 'Optional[int]' = 5000, show: 'bool' = True, random_state: 'int' = 42)
```

```text
Five-panel residual diagnostic.

Panels: predicted vs actual, residual distribution, Q-Q plot,
residuals vs fitted (heteroscedasticity), and absolute error by decile
of the prediction -- the last one answers "where is my model worst?",
which a single global RMSE cannot.

Large samples are thinned to ``sample`` points for the scatter panels
so the figure stays readable and fast; the metrics annotated on it are
still computed on everything.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : array-like
    Predicted labels or numeric outcomes, positionally aligned with y_true.
figsize : Tuple[float, float], default (14, 9)
    Figure width and height in inches; None uses the function-specific
    layout.
sample : Optional[int], default 5000
    Maximum number of observations sampled for plotting or expensive
    diagnostics.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### report_regression

```python
NaviLib.evaluation.report_regression(y_true: 'ArrayLike', y_pred: 'ArrayLike', ci: 'bool' = True, plots: 'bool' = True, show: 'bool' = True) -> 'Dict[str, Any]'
```

```text
Full regression evaluation: metrics against a naive baseline, plots, verdict.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : array-like
    Predicted labels or numeric outcomes, positionally aligned with y_true.
ci : bool, default True
    Request bootstrap confidence intervals for supported metrics. This adds
    repeated metric computation.
plots : bool, default True
    Include diagnostic figures alongside the numerical report.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
dict
    Regression scores and optional diagnostic figure.
```

### score_classification

```python
NaviLib.evaluation.score_classification(y_true: 'ArrayLike', y_pred: 'Optional[ArrayLike]' = None, y_prob: 'Optional[ArrayLike]' = None, pos_label=None, threshold: 'float' = 0.5, ci: 'bool' = False, n_boot: 'int' = 500, labels: 'Optional[Sequence]' = None, random_state: 'int' = 42) -> 'Frame'
```

```text
Classification metrics, imbalance-aware and correctly baselined.

Either ``y_pred`` or ``y_prob`` is required.  When only probabilities
are given, labels are derived at ``threshold``.

What this fixes relative to the usual implementation
----------------------------------------------------
- **Average precision** is computed with ``average_precision_score``,
  not a trapezoidal ``auc`` of the PR curve.  The PR curve is not
  monotone and trapezoid interpolation is optimistically biased --
  scikit-learn documents this explicitly.
- **The PR baseline is prevalence**, reported as ``ap_baseline``.  An
  AP of 0.40 is excellent at 8% prevalence and worthless at 45%; the
  ``ap_lift`` row makes that explicit.
- **Accuracy is baselined** against always predicting the majority
  class (``accuracy_baseline``).  A model at 92% accuracy on a 92%
  majority is doing nothing.
- **Brier score and calibration slope** are reported, because a model
  can rank perfectly (high AUC) and still output probabilities that are
  badly wrong.
- **``pos_label`` is resolved explicitly**, following scikit-learn class order
  unless the positive label is supplied explicitly.

Returns
-------
DataFrame indexed by metric, with ``value``, optional ``ci_low`` /
``ci_high``, and a ``note`` column carrying the interpretation.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : Optional[array-like], default None
    Predicted labels or numeric outcomes, positionally aligned with y_true.
y_prob : Optional[array-like], default None
    Predicted probabilities. A binary vector must refer to pos_label; a
    matrix follows class order. Multiclass matrices have one column per
    class.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
threshold : float, default 0.5
    Decision cutoff or inspection threshold; see the function-specific
    interpretation above.
ci : bool, default False
    Request bootstrap confidence intervals for supported metrics. This adds
    repeated metric computation.
n_boot : int, default 500
    Number of paired bootstrap resamples used to estimate metric
    uncertainty.
labels : Optional[Sequence], default None
    Explicit class or bin labels, in the order expected by the operation.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
```

### per_class_report

```python
NaviLib.evaluation.per_class_report(y_true: 'ArrayLike', y_pred: 'ArrayLike', labels: 'Optional[Sequence]' = None, class_names: 'Optional[Sequence[str]]' = None) -> 'Frame'
```

```text
Per-class precision / recall / F1 with support and error breakdown.

Adds what ``classification_report`` leaves out: how many of each class's
errors went to which other class, so you can see *what* it is being
confused with rather than only that it is being confused.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : array-like
    Predicted labels or numeric outcomes, positionally aligned with y_true.
labels : Optional[Sequence], default None
    Explicit class or bin labels, in the order expected by the operation.
class_names : Optional[Sequence[str]], default None
    Display names for classes, following the supplied/derived class order.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### threshold_sweep

```python
NaviLib.evaluation.threshold_sweep(y_true: 'ArrayLike', y_prob: 'ArrayLike', pos_label=None, n_steps: 'int' = 200, cost_fn: 'float' = 1.0, cost_fp: 'float' = 1.0) -> 'Frame'
```

```text
Every operating point of a binary classifier in one table.

The 0.5 default threshold is an arbitrary convention, not a decision.
This sweeps the whole range and reports sensitivity, specificity,
precision, F1, Youden's J and expected cost at each, so you can pick
the point that matches what a miss actually costs you.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_prob : array-like
    Predicted probabilities. A binary vector must refer to pos_label; a
    matrix follows class order. Multiclass matrices have one column per
    class.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
n_steps : int, default 200
    Number of evenly spaced candidate decision thresholds.
cost_fn : float, default 1.0
    Nonnegative cost assigned to one false negative.
cost_fp : float, default 1.0
    Nonnegative cost assigned to one false positive.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### decision_curve

```python
NaviLib.evaluation.decision_curve(y_true: 'ArrayLike', y_prob: 'ArrayLike', pos_label=None, thresholds: 'Optional[Sequence[float]]' = None) -> 'Frame'
```

```text
Decision curve analysis: net benefit against treat-all and treat-none.

AUC tells you whether the model ranks well.  It does not tell you
whether *acting* on the model beats the two trivial policies of
intervening on everyone or on no one.  Net benefit answers that:

    NB = TP/n - (FP/n) * pt/(1-pt)

where ``pt`` is the threshold probability -- the risk level at which
you would choose to intervene, which encodes the harm ratio between
over-treating and under-treating.

The model is worth using only over the range of ``pt`` where its net
benefit sits above both reference lines.  This is the analysis that
separates a clinically useful model from one with an impressive AUC.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_prob : array-like
    Predicted probabilities. A binary vector must refer to pos_label; a
    matrix follows class order. Multiclass matrices have one column per
    class.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
thresholds : Optional[Sequence[float]], default None
    Explicit probability cutoffs for evaluating net benefit; None uses the
    default grid.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### plot_classification

```python
NaviLib.evaluation.plot_classification(y_true: 'ArrayLike', y_pred: 'Optional[ArrayLike]' = None, y_prob: 'Optional[ArrayLike]' = None, pos_label=None, threshold: 'float' = 0.5, class_names: 'Optional[Sequence[str]]' = None, figsize: 'Optional[Tuple[float, float]]' = None, show: 'bool' = True)
```

```text
Six-panel classifier diagnostic (binary) or three-panel (multiclass).

Binary panels: confusion matrix with row percentages, ROC, PR curve
against the prevalence baseline, calibration curve, score distribution
by true class, and metric-versus-threshold.

The last two are the ones usually missing and usually decisive: the
score distribution shows *why* the classes are confused, and the
threshold panel shows that the reported precision/recall trade-off was
a choice, not a property of the model.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : Optional[array-like], default None
    Predicted labels or numeric outcomes, positionally aligned with y_true.
y_prob : Optional[array-like], default None
    Predicted probabilities. A binary vector must refer to pos_label; a
    matrix follows class order. Multiclass matrices have one column per
    class.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
threshold : float, default 0.5
    Decision cutoff or inspection threshold; see the function-specific
    interpretation above.
class_names : Optional[Sequence[str]], default None
    Display names for classes, following the supplied/derived class order.
figsize : Optional[Tuple[float, float]], default None
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### plot_calibration

```python
NaviLib.evaluation.plot_calibration(y_true: 'ArrayLike', probas: 'Union[ArrayLike, Dict[str, ArrayLike]]', pos_label=None, n_bins: 'int' = 10, strategy: "Literal['quantile', 'uniform']" = 'quantile', figsize: 'Tuple[float, float]' = (11, 4.5), show: 'bool' = True)
```

```text
Reliability diagram for one or several models, with score histograms.

``quantile`` binning is the default because uniform bins leave the
top deciles almost empty on imbalanced data, producing a curve that
swings wildly on three observations.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
probas : Union[array-like, Dict[str, array-like]]
    Mapping of model names to positive-class probability vectors, all
    aligned with y_true.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
n_bins : int, default 10
    Number of calibration bins used to summarize predicted versus observed
    probabilities.
strategy : Literal['quantile', 'uniform'], default 'quantile'
    Calibration binning strategy: uniform probability widths or quantile
    bins.
figsize : Tuple[float, float], default (11, 4.5)
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### plot_decision_curve

```python
NaviLib.evaluation.plot_decision_curve(y_true: 'ArrayLike', probas: 'Union[ArrayLike, Dict[str, ArrayLike]]', pos_label=None, max_threshold: 'float' = 0.6, figsize: 'Tuple[float, float]' = (8, 5), show: 'bool' = True)
```

```text
Plot :func:`decision_curve` -- net benefit vs threshold probability.

Read it as: over the range of risk thresholds a decision-maker might
plausibly use, does the model's curve sit above both the treat-all
diagonal and the treat-none horizontal?  If not, the model is not worth
acting on however good its AUC looks.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
probas : Union[array-like, Dict[str, array-like]]
    Mapping of model names to positive-class probability vectors, all
    aligned with y_true.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
max_threshold : float, default 0.6
    Largest probability cutoff displayed on the decision curve.
figsize : Tuple[float, float], default (8, 5)
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### report_classification

```python
NaviLib.evaluation.report_classification(y_true: 'ArrayLike', y_pred: 'Optional[ArrayLike]' = None, y_prob: 'Optional[ArrayLike]' = None, pos_label=None, threshold: 'float' = 0.5, class_names: 'Optional[Sequence[str]]' = None, ci: 'bool' = True, plots: 'bool' = True, show: 'bool' = True) -> 'Dict[str, Any]'
```

```text
Full classification evaluation: metrics, per-class table, plots, verdict.

The printed verdict is the point.  It answers, in order: does the model
beat the trivial baseline, are the probabilities usable, and over what
range of decision thresholds is it worth acting on.

Returns a dict with ``metrics``, ``per_class``, ``thresholds``,
``decision_curve`` and ``figures``.

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : Optional[array-like], default None
    Predicted labels or numeric outcomes, positionally aligned with y_true.
y_prob : Optional[array-like], default None
    Predicted probabilities. A binary vector must refer to pos_label; a
    matrix follows class order. Multiclass matrices have one column per
    class.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
threshold : float, default 0.5
    Decision cutoff or inspection threshold; see the function-specific
    interpretation above.
class_names : Optional[Sequence[str]], default None
    Display names for classes, following the supplied/derived class order.
ci : bool, default True
    Request bootstrap confidence intervals for supported metrics. This adds
    repeated metric computation.
plots : bool, default True
    Include diagnostic figures alongside the numerical report.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
dict
    Classification scores, class-level details and optional diagnostic
    figure.
```

### compare_models

```python
NaviLib.evaluation.compare_models(y_true: 'ArrayLike', predictions: 'Dict[str, ArrayLike]', task: "Literal['auto', 'classification', 'regression']" = 'auto', pos_label=None, threshold: 'float' = 0.5, ci: 'bool' = False, n_boot: 'int' = 300, sort_by: 'Optional[str]' = None) -> 'Frame'
```

```text
Leaderboard across models, evaluated identically on the same data.

``predictions`` maps a model name to its predictions: probabilities for
classification (labels are derived at ``threshold``), point estimates
for regression.

Guards against the usual comparison mistakes: every model sees exactly
the same rows, the same positive label and the same threshold, and with
``ci=True`` you get intervals so you can see whether the ranking is
real or noise.  A gap smaller than the overlapping intervals is not a
result.

>>> ev.compare_models(y_test, {"rf": p_rf, "lgbm": p_lgb, "logreg": p_lr}, ci=True)

Parameters
----------
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
predictions : Dict[str, array-like]
    Mapping of model names to predictions or prediction/probability
    dictionaries; see examples above.
task : Literal['auto', 'classification', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
pos_label : optional, default None
    Positive class label. None uses the last sorted observed class, matching
    scikit-learn predict_proba column order. A 1-D probability vector must
    correspond to this label.
threshold : float, default 0.5
    Decision cutoff or inspection threshold; see the function-specific
    interpretation above.
ci : bool, default False
    Request bootstrap confidence intervals for supported metrics. This adds
    repeated metric computation.
n_boot : int, default 300
    Number of paired bootstrap resamples used to estimate metric
    uncertainty.
sort_by : Optional[str], default None
    Result column or metric used for ranking the output table.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### error_analysis

```python
NaviLib.evaluation.error_analysis(X: 'Frame', y_true: 'ArrayLike', y_pred: 'ArrayLike', task: "Literal['auto', 'classification', 'regression']" = 'auto', columns=None, bins: 'int' = 4, min_group: 'int' = 20) -> 'Frame'
```

```text
Where does the model fail? Error rate broken down by feature segment.

A single global score hides that the model may be fine on the bulk of
the data and useless on the segment you care about.  Numeric features
are split into quantile bins, categorical ones into their levels, and
the error rate (classification) or MAE (regression) is reported per
segment alongside its size.

``lift`` is the segment's error relative to the overall error: 2.0
means the model is twice as wrong there.  Segments smaller than
``min_group`` are flagged rather than trusted.

>>> ev.error_analysis(X_test, y_test, preds).head(10)

Parameters
----------
X : pandas.DataFrame
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
y_true : array-like
    Observed labels or numeric outcomes, in the same row order as
    predictions.
y_pred : array-like
    Predicted labels or numeric outcomes, positionally aligned with y_true.
task : Literal['auto', 'classification', 'regression'], default 'auto'
    Prediction task. Auto uses target dtype/cardinality heuristics; specify
    regression for low-cardinality numeric outcomes.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
bins : int, default 4
    Number of bins, explicit edges, or supported automatic binning rule as
    indicated by the signature.
min_group : int, default 20
    Minimum number of observations required for a subgroup error summary.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### score_clustering

```python
NaviLib.evaluation.score_clustering(X, labels: 'ArrayLike', true_labels: 'Optional[ArrayLike]' = None, noise_label: 'int' = -1, sample: 'Optional[int]' = 10000, random_state: 'int' = 42) -> 'Frame'
```

```text
Internal and external clustering metrics, with noise handled correctly.

Density-based algorithms (DBSCAN, HDBSCAN, OPTICS) mark outliers with
label ``-1``.  Treating that as a real cluster is a common and serious
mistake: the "noise cluster" is by construction scattered across the
whole space, which drags the silhouette score down and makes a good
clustering look bad.  Noise points are excluded from the internal
metrics here and reported separately as ``noise_pct``.

Silhouette is O(n^2) in memory, so it is computed on a random subsample
above ``sample`` points -- with a note saying so, rather than either
hanging or silently failing.

Metrics
-------
``silhouette``          -1 to 1; above ~0.5 is a strong structure
``davies_bouldin``      lower is better; 0 is perfect
``calinski_harabasz``   higher is better, scale-dependent
``ari``, ``nmi``, ``v_measure``, ``homogeneity``, ``completeness``
                        external, only when ``true_labels`` is given
``cluster_balance``     size of the smallest cluster over the largest

Parameters
----------
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
labels : array-like
    Explicit class or bin labels, in the order expected by the operation.
true_labels : Optional[array-like], default None
    Optional reference cluster labels for external agreement metrics.
noise_label : int, default -1
    Cluster label marking noise observations, excluded from internal cluster
    metrics.
sample : Optional[int], default 10000
    Maximum number of observations sampled for plotting or expensive
    diagnostics.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### plot_clustering

```python
NaviLib.evaluation.plot_clustering(X, labels: 'ArrayLike', true_labels: 'Optional[ArrayLike]' = None, noise_label: 'int' = -1, method: "Literal['pca', 'tsne', 'umap']" = 'pca', scale: 'bool' = True, sample: 'Optional[int]' = 5000, figsize: 'Tuple[float, float]' = (15, 4.6), show: 'bool' = True, random_state: 'int' = 42)
```

```text
Three-panel clustering view: projection, sizes, silhouette profile.

The projection is standardised before PCA by default -- without
scaling, whichever feature has the largest units dominates both
components and the picture says more about your units than your
clusters.  Noise points are drawn as small grey crosses so they read as
what they are.

The silhouette profile (right panel) is the most informative of the
three: a cluster whose bar dips below zero contains points that would
be better placed elsewhere.

Parameters
----------
X : object
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
labels : array-like
    Explicit class or bin labels, in the order expected by the operation.
true_labels : Optional[array-like], default None
    Optional reference cluster labels for external agreement metrics.
noise_label : int, default -1
    Cluster label marking noise observations, excluded from internal cluster
    metrics.
method : Literal['pca', 'tsne', 'umap'], default 'pca'
    Algorithm to use; see the supported methods and assumptions above.
scale : bool, default True
    Standardize numeric inputs before this operation when True.
sample : Optional[int], default 5000
    Maximum number of observations sampled for plotting or expensive
    diagnostics.
figsize : Tuple[float, float], default (15, 4.6)
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### find_best_k

```python
NaviLib.evaluation.find_best_k(X, k_range: 'Sequence[int]' = range(2, 11), algorithm: "Literal['kmeans', 'minibatch', 'gmm', 'agglomerative']" = 'kmeans', columns=None, exclude: 'Optional[Sequence[str]]' = None, scale: 'bool' = True, metrics: 'Sequence[str]' = ('inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz'), sample: 'Optional[int]' = 10000, random_state: 'int' = 42, verbose: 'bool' = True) -> 'Dict[str, Any]'
```

```text
Search for the number of clusters, reporting where the criteria disagree.

Four criteria are computed across ``k_range`` and each nominates its own
winner. They frequently disagree, and that disagreement is information:
it usually means the data have no sharply separated cluster structure,
and the honest answer is "pick k for the business reason, the data are
not deciding for you". The original hid this by silently deferring to
silhouette; here every criterion's choice is reported.

Criteria
--------
``inertia``            within-cluster sum of squares. Always decreases
                       with k, so it is read via the **elbow**: the k of
                       maximum curvature, found here with the kneedle
                       perpendicular-distance rule rather than a second
                       difference, which is unstable on short ranges.
``silhouette``         cohesion vs separation, higher is better.
``davies_bouldin``     lower is better.
``calinski_harabasz``  variance ratio, higher is better.

Parameters
----------
X : DataFrame or array
    Numeric columns are selected automatically from a DataFrame.
algorithm : str
    ``gmm`` additionally reports BIC, which -- unlike every criterion
    above -- has a genuine minimum and is the most principled way to
    choose k when the clusters may overlap.
sample : int, optional
    Silhouette is O(n^2) in memory; above this many rows it is computed
    on a random subsample rather than hanging.
k_range : Sequence[int], default range(2, 11)
    Candidate numbers of clusters to evaluate.
columns : optional, default None
    Source column name or sequence of names. None selects the eligible
    columns described above.
exclude : Optional[Sequence[str]], default None
    Columns excluded from automatic feature selection.
scale : bool, default True
    Standardize numeric inputs before this operation when True.
metrics : Sequence[str], default ('inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz')
    Metrics to calculate, or a metrics table to store, as indicated by the
    signature.
random_state : int, default 42
    Random seed for reproducible sampling, splitting or estimator fitting.
verbose : bool, default True
    Print a concise progress/result summary when True.

Returns
-------
dict with ``table`` (one row per k), ``best`` (per-criterion winners),
``consensus``, ``labels`` for the consensus k, ``model``, and
``agreement`` -- the share of criteria that picked the consensus k.

>>> res = ev.find_best_k(X, k_range=range(2, 9))
>>> res["table"]
>>> res["best"]
{'elbow': 4, 'silhouette': 4, 'davies_bouldin': 4, 'calinski_harabasz': 3}
```

### plot_k_search

```python
NaviLib.evaluation.plot_k_search(result: 'Dict[str, Any]', X=None, figsize: 'Tuple[float, float]' = (14, 9), show: 'bool' = True)
```

```text
Four-panel view of a :func:`find_best_k` result.

Elbow curve, silhouette curve, silhouette profile at the consensus k,
and a PCA scatter of the final labelling.

Centroids in the PCA panel are **projected through the same PCA** rather
than having their first two raw coordinates plotted. Taking
``cluster_centers_[:, :2]`` puts points from the original feature space
onto axes labelled PC1/PC2, so the markers land somewhere arbitrary --
a mistake that is invisible unless you check.

Pass ``X`` (the same data given to :func:`find_best_k`) to get the two
bottom panels; without it only the two curves are drawn.

Parameters
----------
result : Dict[str, Any]
    Result dictionary returned by find_best_k.
X : optional, default None
    Feature matrix in row order, without the target. Use a DataFrame when
    column names are required.
figsize : Tuple[float, float], default (14, 9)
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

### score_ranking

```python
NaviLib.evaluation.score_ranking(df: 'Frame', query_col: 'str', true_col: 'str', pred_col: 'str', k: 'Union[int, Sequence[int]]' = 10, ap_denominator: "Literal['min', 'total']" = 'min', per_query: 'bool' = False) -> 'Frame'
```

```text
Top-K retrieval metrics, evaluated at one or several cut-offs.

Pass a list to ``k`` to get the whole curve in one table -- P@1 versus
P@10 usually tells a different story, and evaluating a single arbitrary
K is how a ranker gets shipped that is great at position 1 and useless
below it.

Parameters
----------
ap_denominator : {"min", "total"}, default "min"
    Average precision at K divides by ``min(n_relevant, K)`` (the
    standard, in which a perfect ranking scores 1.0) or by the total
    number of relevant items (in which a query with 30 relevant items
    can never exceed 0.33 at K=10).  The original implementation used
    different conventions in its ranking and recommender functions,
    which made the two sets of numbers incomparable; this is explicit.
per_query : bool
    Return one row per query instead of the aggregate, so you can find
    which queries the ranker fails on.

Metrics: ``precision@k``, ``recall@k``, ``map@k``, ``mrr``, ``ndcg@k``,
``hit_rate@k``, plus ``n_queries`` and mean relevant-per-query.
df : pandas.DataFrame
    Input pandas DataFrame. Operations return new results rather than
    modifying this frame in place.
query_col : str
    Column identifying the retrieval query or user.
true_col : str
    Column containing true relevance values.
pred_col : str
    Column containing predicted ranking scores; larger scores rank first.
k : Union[int, Sequence[int]], default 10
    Top-k cutoff, or a sequence of cutoffs when supported.



Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### score_recommender

```python
NaviLib.evaluation.score_recommender(recommended: 'Dict[Any, Sequence]', ground_truth: 'Dict[Any, Sequence]', k: 'Union[int, Sequence[int]]' = 10, catalog: 'Optional[Sequence]' = None, popularity: 'Optional[Dict[Any, float]]' = None, strict_users: 'bool' = True) -> 'Frame'
```

```text
Top-K recommender metrics including coverage, novelty and diversity.

Accuracy alone rewards recommending the same few blockbusters to
everyone.  Three beyond-accuracy metrics are reported alongside:

``coverage``      share of the catalogue ever recommended.  Requires
                  ``catalog`` to be a share rather than a raw count --
                  "1 200 distinct items" means nothing without knowing
                  whether the catalogue holds 1 500 or 5 000 000.
``novelty``       mean self-information ``-log2(popularity)`` of the
                  recommended items.  Low novelty means you are just
                  re-recommending the head of the distribution.
``personalisation`` mean pairwise dissimilarity between users' lists.
                  Near 0 means everyone gets the same recommendations.

Parameters
----------
strict_users : bool, default True
    Score users who have ground truth but received no recommendation as
    zero, rather than dropping them.  Dropping them silently inflates
    every metric -- the original behaviour, and a common way to report
    a recall that the system does not actually achieve.
recommended : Dict[Any, Sequence]
    Mapping of users to ranked recommended item IDs, best first.
ground_truth : Dict[Any, Sequence]
    Mapping of users to sets/sequences of relevant item IDs.
k : Union[int, Sequence[int]], default 10
    Top-k cutoff, or a sequence of cutoffs when supported.
catalog : Optional[Sequence], default None
    Optional complete collection of catalog item IDs for coverage metrics.
popularity : Optional[Dict[Any, float]], default None
    Optional item popularity mapping for novelty metrics.



Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### plot_ranking

```python
NaviLib.evaluation.plot_ranking(results: 'Frame', title: 'str' = 'Ranking metrics', figsize: 'Tuple[float, float]' = (11, 4.5), show: 'bool' = True)
```

```text
Plot a :func:`score_ranking` / :func:`score_recommender` table.

With several cut-offs the metrics are drawn as curves against K, which
is the shape you actually need to choose an operating K.  With a single
K it falls back to a labelled bar chart.

Parameters
----------
results : pandas.DataFrame
    Ranking metric table, or named collection of ranking results.
title : str, default 'Ranking metrics'
    Custom title shown above the chart or report.
figsize : Tuple[float, float], default (11, 4.5)
    Figure width and height in inches; None uses the function-specific
    layout.
show : bool, default True
    Display the figure when True. False closes the pyplot window while
    returning a usable Figure for saving.

Returns
-------
matplotlib.figure.Figure
    Figure using the active NaviLib theme; retain it for savefig or further
    customization.
```

## NaviLib.quality

### audit_data

```python
NaviLib.quality.audit_data(df: pandas.DataFrame, *, target: str | None = None, missing_threshold: float = 0.3, high_cardinality: float = 0.9) -> pandas.DataFrame
```

```text
Find quality problems and suggest concrete next steps without changing data.

Parameters
----------
df : pandas.DataFrame
    Dataset with unique column names. Empty datasets are reported explicitly.
target : str, optional
    Target to check for missing labels and possible feature copies.
missing_threshold : float, default 0.3
    Missing fraction at which a column receives an error instead of a warning.
high_cardinality : float, default 0.9
    Unique/nonmissing fraction above which string columns may be identifiers.

Returns
-------
pandas.DataFrame
    Columns: severity, column, issue, count, fraction, recommendation.
    Fractions are in [0, 1]; per-column fractions use the total row count.
    Heuristics are prompts for investigation, never automatic deletion rules.

Examples
--------
>>> import NaviLib as nv
>>> issues = nv.audit_data(pd.DataFrame({'x': [1, 1, None]}))
>>> issues['issue'].tolist()
['missing', 'constant', 'duplicates']
```

### infer_schema

```python
NaviLib.quality.infer_schema(df: pandas.DataFrame, *, category_limit: int = 50) -> NaviLib.quality.DataSchema
```

```text
Capture column types, nullability and small categorical vocabularies.

Parameters
----------
df : pandas.DataFrame
    Reference data with unique column names.
category_limit : int, default 50
    Maximum nonnumeric cardinality to store. Zero disables category tracking.

Returns
-------
DataSchema
    Observed properties only; edit the schema's column rules if domain
    requirements differ (e.g. allow nulls absent in the reference sample).
```

### validate_schema

```python
NaviLib.quality.validate_schema(df: pandas.DataFrame, schema: NaviLib.quality.DataSchema, *, allow_extra: bool = False, check_categories: bool = True) -> pandas.DataFrame
```

```text
Compare a new batch to a reference schema without coercing its values.

Parameters
----------
df : pandas.DataFrame
    Incoming observations.
schema : DataSchema
    Snapshot returned by infer_schema.
allow_extra : bool, default False
    Ignore unexpected columns when True.
check_categories : bool, default True
    Report values absent from stored categorical vocabularies.

Returns
-------
pandas.DataFrame
    Columns column, issue, expected, actual. An empty table means all checks
    passed. Dtypes are compared exactly, including nullable/extension types.
```

### DataSchema

```python
NaviLib.quality.DataSchema(columns: dict[typing.Any, dict]) -> None
```

```text
Reference schema returned by :func:`infer_schema`.

Attributes
----------
columns : dict
    Mapping from column label to dtype string, nullable flag and optional
    observed category tuple. It is a snapshot, not a fitted preprocessor.
```

## NaviLib.timeseries

### temporal_split

```python
NaviLib.timeseries.temporal_split(df: pandas.DataFrame, time: str, *, test_size: float = 0.2, gap: int = 0) -> tuple[pandas.DataFrame, pandas.DataFrame]
```

```text
Split on distinct timestamps, reserving the newest times for testing.

Parameters
----------
df : pandas.DataFrame
    Observations; repeated timestamps across entities remain together.
time : str
    Numeric or datetime ordering column without nulls.
test_size : float, default 0.2
    Fraction of unique timestamps assigned to test, rounded up.
gap : int, default 0
    Number of distinct timestamps excluded between training and test.

Returns
-------
tuple of pandas.DataFrame
    Training and test copies, chronologically sorted with original indices.
    The gap is excluded from both. At least one training timestamp is required.
```

### add_lag_features

```python
NaviLib.timeseries.add_lag_features(df: pandas.DataFrame, columns, *, time: str, lags=(1,), group_by=None) -> pandas.DataFrame
```

```text
Add previous-observation features, independently within each entity.

Parameters
----------
df : pandas.DataFrame
    History and observations to transform; original order/index are preserved.
columns : str or sequence of str
    Numeric measurements to shift.
time : str
    Numeric/datetime order; duplicate times within an entity are rejected.
lags : sequence of positive int, default (1,)
    Offsets in observations, not elapsed time units.
group_by : str or sequence of str, optional
    Entity keys. Missing group keys are rejected.

Returns
-------
pandas.DataFrame
    Copy with ``<column>_lag_<offset>`` columns. Insufficient history is NaN.
    To transform a future batch, include its known history, then select batch
    rows. Earlier batch measurements must actually be available at prediction
    time; this function does not recursively forecast unknown targets.
```

### add_rolling_features

```python
NaviLib.timeseries.add_rolling_features(df: pandas.DataFrame, columns, *, time: str, windows=(3, 7), statistics=('mean', 'std'), group_by=None, min_periods: int = 1) -> pandas.DataFrame
```

```text
Summarize strictly earlier observations with shifted rolling windows.

Parameters
----------
df : pandas.DataFrame
    Observations and available history; original row order/index are preserved.
columns : str or sequence of str
    Numeric columns to summarize.
time : str
    Numeric/datetime ordering column, unique within each entity.
windows : sequence of positive int, default (3, 7)
    Window widths measured in observations.
statistics : sequence of str, default ('mean', 'std')
    Any of mean, std, min, max, median, sum. Std uses ddof=1.
group_by : str or sequence of str, optional
    Independent entity keys, without missing values.
min_periods : int, default 1
    Required nonmissing observations, at most the smallest window.

Returns
-------
pandas.DataFrame
    Copy with ``<column>_rolling_<stat>_<window>`` features. Current rows are
    excluded via shift(1). For fixed-origin forecasts, do not supply unknown
    future measurements as if they were observed history.
```

## NaviLib.theme

### set_theme

```python
NaviLib.theme.set_theme(name: str = 'light', *, palette=None, font_scale: float = 1.0, rc: dict | None = None) -> dict
```

```text
Choose the theme for subsequent NaviLib figures and styled tables.

Parameters
----------
name : {'light', 'dark', 'paper'}, default 'light'
    Light dashboard, dark dashboard, or print-friendly colors.
palette : sequence of matplotlib colors, optional
    At least two colors; replaces the categorical color cycle.
font_scale : float, default 1.0
    Multiplier for chart label and title sizes.
rc : dict, optional
    Matplotlib rcParams overrides, applied only while drawing.

Returns
-------
dict
    A copy of the selected configuration. Existing figures are unchanged.

Examples
--------
>>> import NaviLib as nv
>>> config = nv.set_theme('dark', font_scale=1.1)
```

### get_theme

```python
NaviLib.theme.get_theme() -> dict
```

```text
Return an independent copy of the current NaviLib theme configuration.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### theme_context

```python
NaviLib.theme.theme_context(name: str = 'light', *, palette=None, font_scale: float = 1.0, rc: dict | None = None)
```

```text
Temporarily choose a theme; restore it even if plotting raises an error.

Parameters match :func:`set_theme`. Nested contexts restore their parent's
settings. Yields a copy of the temporary configuration.

Parameters
----------
name : str, default 'light'
    Built-in theme name: light, dark or paper.
palette : optional, default None
    Sequence of at least two matplotlib-compatible colors replacing the
    categorical cycle.
font_scale : float, default 1.0
    Positive multiplier for figure title and label sizes.
rc : dict | None, default None
    Optional matplotlib rcParams overrides, scoped to NaviLib plotting
    calls.

Returns
-------
context manager
    Yields the temporary theme configuration and restores the previous one
    on exit.
```

### available_themes

```python
NaviLib.theme.available_themes() -> list[str]
```

```text
Return built-in theme names: light, dark and paper.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### style_table

```python
NaviLib.theme.style_table(df, *, precision: int = 3, caption: str | None = None)
```

```text
Return a themed pandas Styler without modifying data or rounding values.

Parameters
----------
df : pandas.DataFrame
    Table to display in a notebook or export with ``.to_html()``.
precision : int, default 3
    Display precision for floating-point values.
caption : str, optional
    Plain-text table caption, escaped before HTML rendering.

Returns
-------
pandas.io.formats.style.Styler
    Styled view; requires the ``notebook`` extra (Jinja2).
```

## NaviLib.reporting

### DataReport

```python
NaviLib.reporting.DataReport(title: str, summary: dict, tables: dict, figures: dict, theme: dict, notes: list[str] = <factory>) -> None
```

```text
Analysis tables and figures with notebook and offline HTML rendering.

Attributes
----------
title : str
    Report heading, escaped as text in HTML.
summary : dict
    Full-data row/column counts, missing percentage and memory size.
tables : dict of pandas.DataFrame
    Full-data analysis tables, available for filtering and export.
figures : dict of matplotlib.figure.Figure
    Closed figure handles; can still be saved or inspected.
theme : dict
    Configuration captured when the report was created.
notes : list of str
    Sampling and scope information shown in the report.
```

### create_report

```python
NaviLib.reporting.create_report(df: pandas.DataFrame, *, target: str | None = None, title: str = 'Dataset overview', theme: str | None = None, include_plots: bool = True, sample_size: int = 5000, max_columns: int = 12, random_state: int = 42) -> NaviLib.reporting.DataReport
```

```text
Build an actionable profile with full-data tables and bounded-size plots.

Parameters
----------
df : pandas.DataFrame
    Dataset with unique column names. Input values are never changed.
target : str, optional
    Target for label-quality checks; no model is fitted by this report.
title : str, default 'Dataset overview'
    Heading for notebook and HTML output.
theme : {'light', 'dark', 'paper'}, optional
    Temporary report theme; None preserves the current custom theme.
include_plots : bool, default True
    Generate numeric distribution and missingness figures.
sample_size : int, default 5000
    Maximum randomly sampled rows for plots. Tables use the full dataset.
max_columns : int, default 12
    Maximum numeric columns plotted/profiled, and categorical columns profiled.
    The quality audit still checks every column.
random_state : int, default 42
    Reproducible plot sampling seed.

Returns
-------
DataReport
    Reusable tables, figures and ``to_html(path)`` export. Infinite numeric
    values are reported in the audit and treated as missing in profiles/plots.

Examples
--------
>>> import NaviLib as nv
>>> report = nv.create_report(pd.DataFrame({'x': [1., 2., 3.]}))
>>> html = report.to_html()
```

## NaviLib._common

### apply_state

```python
NaviLib._common.apply_state(df: 'Frame', state: 'Union[State, Sequence[State]]', strict: 'bool' = True) -> 'Frame'
```

```text
Replay fitted preprocessing on new data -- any kind, any mix, in order.

This is the single entry point.  Cleaning states (imputation, outlier
bounds, rare-level maps, fitted anomaly detectors) and feature states
(Box-Cox lambdas, scalers, encoders, bin edges, group tables) can be
chained freely in one list; each is routed to the module that fitted it.

Order is preserved and it matters: apply the states in the order you
fitted them, because later steps often depend on columns earlier ones
created.  If a step fails, the error names its position and kind rather
than surfacing a bare ``KeyError`` from three frames down.

Parameters
----------
df : DataFrame
    New data -- test set, validation fold, or production batch.
state : dict or list of dict
    What ``return_state=True`` gave you.
strict : bool, default True
    ``False`` skips states whose source columns are absent instead of
    raising.  Useful when replaying a long chain onto a partial frame;
    dangerous as a habit, because a silently skipped step means the new
    data is no longer on the same scale as the training data.

Returns
-------
DataFrame

Examples
--------
>>> train, s1 = nv.fix_missing(train, method="median", return_state=True)
>>> train, s2 = nv.encode(train, ["city"], method="target",
...                       target="y", return_state=True)
>>> train, s3 = nv.scale(train, target="y", return_state=True)
>>> test = nv.apply_state(test, [s1, s2, s3])
```

### describe_states

```python
NaviLib._common.describe_states(state: 'Union[State, Sequence[State]]') -> 'Frame'
```

```text
Readable log of a chain: what ran, in what order, on what, producing what.

Works across modules, so a mixed cleaning + feature chain reads as one
table.  Worth printing next to your model score -- months later this is
often the only record of how the features were built.

Parameters
----------
state : Union[State, Sequence[State]]
    Fitted state dictionary, or ordered sequence of states returned with
    return_state=True.

Returns
-------
pandas.DataFrame
    One row per fitted state, including owner, source columns and
    transformation summary.
```

### register_state_handler

```python
NaviLib._common.register_state_handler(kinds: 'Iterable[str]', handler: 'Callable[[Frame, State], Frame]', owner: 'str' = '') -> 'None'
```

```text
Declare that ``handler`` can replay these state kinds.

Called once per module at package import.  Re-registering a kind is an
error rather than a silent overwrite -- two modules quietly claiming
the same kind is exactly the class of bug this registry exists to stop.

Parameters
----------
kinds : iterable of str
    The values that appear in ``state["kind"]``.
handler : callable
    ``handler(df, state) -> DataFrame``, replaying one state.
owner : str
    Module name, used only to make error messages readable.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### registered_kinds

```python
NaviLib._common.registered_kinds() -> 'Frame'
```

```text
Every state kind the package can replay, and which module owns it.

Returns
-------
pandas.DataFrame
    Structured results with named columns; see the measures and
    interpretation described above.
```

### save_state

```python
NaviLib._common.save_state(state: 'Union[State, Sequence[State]]', path: 'str', compress: 'int' = 3) -> 'str'
```

```text
Persist a fitted chain to disk.

States hold live scikit-learn objects (scalers, imputers, isolation
forests, quantile maps), so they are pickled with ``joblib`` rather
than serialised to JSON.

Two consequences worth knowing before you rely on this: the file is
only loadable by a compatible scikit-learn version, and unpickling
executes code -- never load a state file you did not create.  For
long-term storage, keep the ``describe_states`` table alongside it so
the chain can be rebuilt from scratch if the pickle ever stops loading.

Parameters
----------
state : Union[State, Sequence[State]]
    Fitted state dictionary, or ordered sequence of states returned with
    return_state=True.
path : str
    Destination or source filesystem path; pathlib.Path is also accepted.
compress : int, default 3
    Joblib compression level; 0 disables compression and larger values trade
    speed for size.

Returns
-------
str
    Absolute path to the written joblib state file.
```

### load_state

```python
NaviLib._common.load_state(path: 'str', check_versions: 'bool' = True) -> 'List[State]'
```

```text
Load a chain saved by :func:`save_state`, warning on version drift.

Parameters
----------
path : str
    Destination or source filesystem path; pathlib.Path is also accepted.
check_versions : bool, default True
    Warn when saved and installed scikit-learn versions differ. Only load
    trusted joblib files.

Returns
-------
list of dict
    Ordered fitted states ready for apply_state.
```

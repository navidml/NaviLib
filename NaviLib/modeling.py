"""
navdata.modeling
~~~~~~~~~~~~~~~~

The middle of the pipeline: training, cross-validation, tuning,
interpretation, and saving a model you can actually reload.

Where this fits
---------------
``cleaning`` and ``feature_engineering`` fit *states*; ``evaluation`` scores
*predictions*.  Between them sits the part that has to be leak-safe:
preprocessing must be refitted inside every cross-validation fold, or the
score you report belongs to a model that has already seen its test data.

:class:`ChainTransformer` is the bridge.  It wraps a navdata chain as a
scikit-learn transformer, so the same steps you ran interactively can be
dropped into a ``Pipeline`` and refit per fold automatically:

>>> from navdata import modeling as md, feature_engineering as fe
>>> prep = md.ChainTransformer([
...     (fe.transform_numeric, {"columns": ["income"], "method": "auto"}),
...     (fe.encode, {"columns": ["city"], "method": "target", "target": "y"}),
...     (fe.scale, {"method": "robust"}),
... ], target="y")
>>> pipe = md.make_pipeline(prep, LGBMClassifier())
>>> md.validate(pipe, X, y, cv=5)          # preprocessing refit 5 times

Design principles
-----------------
1. **Nothing is scored without a baseline.**  ``compare_algorithms`` always
   includes a dummy predictor; a model that cannot beat it is not a model.
2. **Fold-level spread, not just the mean.**  A mean without a standard
   deviation hides whether the gap between two models is real.
3. **Tuning and evaluating on the same folds is reported as such.**  Use
   :func:`nested_validate` for a number you would put in a paper.
4. **The saved artifact is the whole pipeline** -- preprocessing states,
   model, threshold, metrics and library versions -- not just the estimator.

Author: Navid
License: MIT
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     cross_val_predict, cross_validate as _sk_cv)

try:                                    # inside the package
    from ._common import _finish, _plt, _sns, _style, PALETTE, GOOD, BAD, NEUTRAL
    from ._common import apply_state, describe_states
except ImportError:                     # running the file standalone
    from _common import _finish, _plt, _sns, _style, PALETTE, GOOD, BAD, NEUTRAL
    from _common import apply_state, describe_states

Frame = pd.DataFrame
Series = pd.Series

#: Default scoring per task. Average precision leads for binary problems,
#: not accuracy or ROC AUC: on a rare positive class accuracy is dominated
#: by the base rate and ROC AUC stays flattering long after the model has
#: stopped being useful.
DEFAULT_SCORING = {
    "classification": {"auprc": "average_precision", "auroc": "roc_auc",
                       "f1": "f1", "brier": "neg_brier_score",
                       "balanced_accuracy": "balanced_accuracy"},
    "multiclass": {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy",
                   "accuracy": "accuracy"},
    "regression": {"rmse": "neg_root_mean_squared_error",
                   "mae": "neg_mean_absolute_error", "r2": "r2"},
}

_LOWER_IS_BETTER = {"brier", "rmse", "mae", "log_loss", "smape_pct"}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _infer_task(y) -> str:
    y = pd.Series(y)
    if y.dtype == object or str(y.dtype) in ("category", "bool", "string"):
        return "classification" if y.nunique() == 2 else "multiclass"
    if y.nunique() <= 2:
        return "classification"
    if y.nunique() <= 20 and pd.api.types.is_integer_dtype(y):
        return "multiclass"
    return "regression"


def _make_cv(task: str, cv, y=None, groups=None, random_state: int = 42):
    if hasattr(cv, "split"):
        return cv
    n = int(cv)
    if groups is not None:
        return GroupKFold(n_splits=n)
    if task in ("classification", "multiclass"):
        return StratifiedKFold(n, shuffle=True, random_state=random_state)
    return KFold(n, shuffle=True, random_state=random_state)


def _resolve_scoring(task: str, scoring):
    if scoring is None:
        return dict(DEFAULT_SCORING[task])
    if isinstance(scoring, str):
        return {scoring: scoring}
    if isinstance(scoring, (list, tuple)):
        return {s: s for s in scoring}
    return dict(scoring)


def _tidy(res: Dict[str, np.ndarray], scoring: Dict[str, Any]) -> Frame:
    rows = []
    for name in scoring:
        sign = -1 if str(scoring[name]).startswith("neg_") else 1
        v = sign * res[f"test_{name}"]
        rows.append({
            "metric": name,
            "mean": round(float(np.mean(v)), 4),
            "sd": round(float(np.std(v)), 4),
            "min": round(float(np.min(v)), 4),
            "max": round(float(np.max(v)), 4),
            "folds": ", ".join(f"{x:.3f}" for x in v),
        })
    return pd.DataFrame(rows).set_index("metric")


# ======================================================================
#  1. THE BRIDGE  --  navdata chains as scikit-learn transformers
# ======================================================================

class ChainTransformer(BaseEstimator, TransformerMixin):
    """Wrap a navdata preprocessing chain as a scikit-learn transformer.

    This is what makes leak-free cross-validation possible without giving
    up the state-based API.  On ``fit`` it runs the steps and captures
    their states; on ``transform`` it replays those states via
    :func:`navdata.apply_state`.  Put it in a ``Pipeline`` and every
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
        The fitted chain.  Hand it to :func:`navdata.save_state`.
    feature_names_out_ : list of str

    Examples
    --------
    >>> prep = ChainTransformer([
    ...     (cl.fix_missing, {"method": "median"}),
    ...     (fe.encode, {"columns": ["city"], "method": "target", "target": "y"}),
    ... ], target="y")
    >>> prep.fit_transform(X_train, y_train).shape
    >>> prep.transform(X_test).shape          # replayed, not refitted
    """

    def __init__(self, steps: Sequence[Tuple[Callable, Dict[str, Any]]],
                 target: Optional[str] = None, drop_target: bool = True):
        self.steps = steps
        self.target = target
        self.drop_target = drop_target

    def fit(self, X: Frame, y=None):
        self._run(X, y)
        return self

    def fit_transform(self, X: Frame, y=None, **fit_params) -> Frame:
        return self._run(X, y)

    def _run(self, X: Frame, y=None) -> Frame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"ChainTransformer works on DataFrames so column names survive "
                f"the chain; got {type(X).__name__}."
            )
        df = X.copy()
        attached = False
        if self.target is not None and y is not None and self.target not in df.columns:
            df[self.target] = np.asarray(y)
            attached = True

        self.states_ = []
        for fn, kwargs in self.steps:
            kw = dict(kwargs)
            kw["return_state"] = True
            df, st = fn(df, **kw)
            self.states_.append(st)

        if attached and self.drop_target and self.target in df.columns:
            df = df.drop(columns=[self.target])
        self.feature_names_out_ = list(df.columns)
        self.n_features_in_ = X.shape[1]
        return df

    def transform(self, X: Frame) -> Frame:
        if not hasattr(self, "states_"):
            raise RuntimeError("ChainTransformer is not fitted yet; call fit first.")
        out = apply_state(X.copy(), self.states_)
        if self.drop_target and self.target in out.columns:
            out = out.drop(columns=[self.target])
        missing = [c for c in self.feature_names_out_ if c not in out.columns]
        if missing:
            raise KeyError(
                f"transform produced a frame missing {len(missing)} column(s) that "
                f"fit produced, e.g. {missing[:5]}. Usually this means a category "
                f"seen in training is absent here and an encoder emitted fewer "
                f"columns."
            )
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "feature_names_out_", []), dtype=object)

    def describe(self) -> Frame:
        """The fitted chain as a readable table (see ``describe_states``)."""
        if not hasattr(self, "states_"):
            raise RuntimeError("Not fitted yet.")
        return describe_states(self.states_)


def make_preprocessor(
    numeric: Optional[Sequence[str]] = None,
    categorical: Optional[Sequence[str]] = None,
    numeric_impute: str = "median",
    categorical_impute: str = "most_frequent",
    scale: bool = True,
    one_hot: bool = True,
    min_frequency: float = 0.01,
):
    """A plain ColumnTransformer, for when the preprocessing really is simple.

    Use this when all you need is impute + scale + one-hot.  Reach for
    :class:`ChainTransformer` when you need target encoding, Box-Cox, group
    aggregates or supervised binning, none of which a ColumnTransformer
    expresses.

    Unknown categories at predict time go to the infrequent bucket instead
    of raising, and rare levels are folded together at ``min_frequency``.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    blocks = []
    if numeric:
        inner = [("impute", SimpleImputer(strategy=numeric_impute))]
        if scale:
            inner.append(("scale", StandardScaler()))
        blocks.append(("num", SkPipeline(inner), list(numeric)))
    if categorical:
        inner = [("impute", SimpleImputer(strategy=categorical_impute,
                                          fill_value="__missing__"))]
        if one_hot:
            inner.append(("ohe", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_frequency, sparse_output=False)))
        blocks.append(("cat", SkPipeline(inner), list(categorical)))
    if not blocks:
        raise ValueError("Pass at least one of numeric= or categorical=.")
    return ColumnTransformer(blocks, remainder="drop",
                             verbose_feature_names_out=False)


def make_pipeline(*steps, balance: Optional[str] = None, ratio="auto",
                  random_state: int = 42):
    """Assemble a pipeline, optionally resampling inside the fold.

    ``balance`` takes any method name from ``cleaning.BALANCE_METHODS``.
    The sampler is inserted immediately before the final estimator, so it
    sees only the training part of each fold and is skipped entirely at
    predict time.  Resampling anywhere else leaks.

    >>> pipe = md.make_pipeline(prep, LGBMClassifier(), balance="smote", ratio=0.3)
    """
    steps = list(steps)
    if not steps:
        raise ValueError("Pass at least one estimator.")
    named = [(f"step{i}", s) for i, s in enumerate(steps[:-1])] + [("model", steps[-1])]
    if balance is None or balance == "none":
        from sklearn.pipeline import Pipeline as SkPipeline
        return SkPipeline(named)
    try:
        from .cleaning import _make_sampler
    except ImportError:
        from cleaning import _make_sampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    sampler = _make_sampler(balance, ratio, random_state)
    return ImbPipeline(named[:-1] + [("balance", sampler), named[-1]])


# ======================================================================
#  2. VALIDATION
# ======================================================================

def validate(
    model,
    X,
    y,
    cv: Union[int, Any] = 5,
    scoring=None,
    task: Literal["auto", "classification", "multiclass", "regression"] = "auto",
    groups=None,
    return_oof: bool = True,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> Frame:
    """Cross-validate, reporting per-fold spread rather than a bare mean.

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

    Returns
    -------
    DataFrame indexed by metric, with ``mean``, ``sd``, ``min``, ``max``,
    ``train_mean``, ``overfit_gap`` and the individual ``folds``.
    """
    task = _infer_task(y) if task == "auto" else task
    scoring = _resolve_scoring(task, scoring)
    splitter = _make_cv(task, cv, y, groups, random_state)

    t0 = time.time()
    res = _sk_cv(model, X, y, cv=splitter, groups=groups, scoring=scoring,
                 n_jobs=n_jobs, return_train_score=True, error_score="raise")
    table = _tidy(res, scoring)

    for name in scoring:
        sign = -1 if str(scoring[name]).startswith("neg_") else 1
        table.loc[name, "train_mean"] = round(float(np.mean(sign * res[f"train_{name}"])), 4)
    gap = table["train_mean"] - table["mean"]
    table["overfit_gap"] = [round(-g, 4) if m in _LOWER_IS_BETTER else round(g, 4)
                            for m, g in zip(table.index, gap)]

    if return_oof:
        method = ("predict_proba"
                  if task in ("classification", "multiclass")
                  and hasattr(model, "predict_proba") else "predict")
        try:
            oof = cross_val_predict(model, X, y, cv=splitter, groups=groups,
                                    method=method, n_jobs=n_jobs)
            if method == "predict_proba" and task == "classification":
                oof = oof[:, 1]
            table.attrs["oof"] = oof
            table.attrs["oof_method"] = method
        except Exception as exc:
            warnings.warn(f"Out-of-fold predictions unavailable: {exc}", stacklevel=2)

    table.attrs["task"] = task
    table.attrs["n_splits"] = splitter.get_n_splits(X, y, groups)
    table.attrs["elapsed_s"] = round(time.time() - t0, 2)
    table.attrs["fit_time_s"] = round(float(np.mean(res["fit_time"])), 3)

    if verbose:
        print(f"{task} | {table.attrs['n_splits']} folds | "
              f"{table.attrs['elapsed_s']}s")
        print(table[["mean", "sd", "train_mean", "overfit_gap", "folds"]].to_string())
        primary = list(scoring)[0]
        g = table.loc[primary, "overfit_gap"]
        if g > 0.05 and g > 3 * max(table.loc[primary, "sd"], 1e-6):
            print(f"  note: train-test gap on '{primary}' is {g:.3f} — the model is "
                  f"memorising. Regularise or cut features before tuning anything else.")
    return table


def compare_algorithms(
    X,
    y,
    models: Optional[Dict[str, Any]] = None,
    cv: Union[int, Any] = 5,
    scoring=None,
    task: Literal["auto", "classification", "multiclass", "regression"] = "auto",
    groups=None,
    include_baseline: bool = True,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> Frame:
    """Leaderboard across several algorithms on identical folds.

    A dummy predictor is included by default and should be read first.  On
    imbalanced data a model can look impressive and be doing nothing: if
    your gradient booster scores 0.91 accuracy and the dummy scores 0.91,
    you have learned the base rate and nothing else.

    ``models=None`` runs a default set -- a regularised linear model, a
    random forest and gradient boosting.  Every model sees the same splits,
    so the comparison is paired and the ``_sd`` columns are comparable.

    >>> md.compare_algorithms(X, y)
    """
    task = _infer_task(y) if task == "auto" else task
    scoring = _resolve_scoring(task, scoring)
    splitter = _make_cv(task, cv, y, groups, random_state)
    clf = task in ("classification", "multiclass")

    models = dict(_default_models(task, random_state) if models is None else models)
    if include_baseline:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        models = {"__baseline__": (DummyClassifier(strategy="prior") if clf
                                   else DummyRegressor(strategy="mean")),
                  **models}

    rows = []
    for name, mdl in models.items():
        label = "baseline (prior)" if name == "__baseline__" else name
        try:
            res = _sk_cv(clone(mdl), X, y, cv=splitter, groups=groups,
                         scoring=scoring, n_jobs=n_jobs, error_score="raise")
            rec: Dict[str, Any] = {"model": label}
            for m in scoring:
                sign = -1 if str(scoring[m]).startswith("neg_") else 1
                v = sign * res[f"test_{m}"]
                rec[m] = round(float(np.mean(v)), 4)
                rec[f"{m}_sd"] = round(float(np.std(v)), 4)
            rec["fit_time_s"] = round(float(np.mean(res["fit_time"])), 3)
            rec["status"] = "ok"
        except Exception as exc:
            rec = {"model": label, "status": f"{type(exc).__name__}: {exc}"[:90]}
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("model")
    primary = list(scoring)[0]
    if primary in out.columns:
        out = out.sort_values(primary, ascending=primary in _LOWER_IS_BETTER,
                              na_position="last")
    out.attrs["primary"] = primary

    if verbose and primary in out.columns and "baseline (prior)" in out.index:
        base = out.loc["baseline (prior)", primary]
        best_name = out.index[0]
        best, sd = out.loc[best_name, primary], out.loc[best_name, f"{primary}_sd"]
        print(f"best: {best_name} — {primary}={best:.4f} ± {sd:.4f}  "
              f"(baseline {base:.4f})")
        if abs(best - base) < 2 * sd:
            print("  warning: the winner is within 2 sd of the baseline — not yet "
                  "evidence of a working model.")
    return out


def _default_models(task: str, random_state: int) -> Dict[str, Any]:
    clf = task in ("classification", "multiclass")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    out: Dict[str, Any] = {}
    if clf:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        out["logistic"] = Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        out["random_forest"] = RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1)
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import RidgeCV
        out["ridge"] = Pipeline([("sc", StandardScaler()), ("m", RidgeCV())])
        out["random_forest"] = RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1)
    try:
        import lightgbm as lgb
        Model = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
        kw = dict(n_estimators=400, learning_rate=0.05, random_state=random_state,
                  n_jobs=-1, verbose=-1)
        if clf:
            kw["class_weight"] = "balanced"
        out["lightgbm"] = Model(**kw)
    except ImportError:
        from sklearn.ensemble import (GradientBoostingClassifier,
                                      GradientBoostingRegressor)
        out["gradient_boosting"] = (GradientBoostingClassifier(random_state=random_state)
                                    if clf else
                                    GradientBoostingRegressor(random_state=random_state))
    return out


# ======================================================================
#  3. TUNING
# ======================================================================

def tune(
    model,
    param_space: Dict[str, Any],
    X,
    y,
    search: Literal["random", "grid", "halving"] = "random",
    n_iter: int = 40,
    cv: Union[int, Any] = 5,
    scoring=None,
    task: Literal["auto", "classification", "multiclass", "regression"] = "auto",
    groups=None,
    refit: bool = True,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
):
    """Hyper-parameter search that reports how much of the gain is noise.

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
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    task = _infer_task(y) if task == "auto" else task
    scoring_map = _resolve_scoring(task, scoring)
    primary = list(scoring_map)[0]
    splitter = _make_cv(task, cv, y, groups, random_state)

    common = dict(estimator=model, cv=splitter, scoring=scoring_map[primary],
                  n_jobs=n_jobs, refit=refit, return_train_score=True)
    if search == "grid":
        srch = GridSearchCV(param_grid=param_space, **common)
    elif search == "halving":
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV
        srch = HalvingRandomSearchCV(param_distributions=param_space,
                                     random_state=random_state, factor=3, **common)
    elif search == "random":
        srch = RandomizedSearchCV(param_distributions=param_space, n_iter=n_iter,
                                  random_state=random_state, **common)
    else:
        raise ValueError("search must be 'random', 'grid' or 'halving'.")

    t0 = time.time()
    srch.fit(X, y, groups=groups) if groups is not None else srch.fit(X, y)
    elapsed = round(time.time() - t0, 1)

    res = pd.DataFrame(srch.cv_results_)
    keep = ["mean_test_score", "std_test_score", "mean_train_score",
            "rank_test_score", "mean_fit_time"]
    keep = [c for c in keep if c in res.columns]
    params = pd.DataFrame(list(res["params"]))
    table = pd.concat([params, res[keep]], axis=1).sort_values("rank_test_score")
    table = table.reset_index(drop=True).round(4)

    best_sd = float(res.loc[srch.best_index_, "std_test_score"])
    within = int((res["mean_test_score"] >= srch.best_score_ - best_sd).sum())

    out = {
        "best_model": srch.best_estimator_ if refit else None,
        "best_params": srch.best_params_,
        "best_score": round(float(srch.best_score_), 4),
        "best_score_sd": round(best_sd, 4),
        "scoring": primary,
        "results": table,
        "search": srch,
        "n_candidates": len(res),
        "n_within_1sd": within,
        "elapsed_s": elapsed,
    }

    if verbose:
        print(f"{search} search | {len(res)} candidates | {elapsed}s")
        print(f"  best {primary} = {srch.best_score_:.4f} ± {best_sd:.4f}")
        print(f"  best params: {srch.best_params_}")
        if within > max(3, 0.1 * len(res)):
            print(f"  warning: {within} of {len(res)} configurations are within one "
                  f"sd of the best. The search is reading noise — the winning "
                  f"parameters are unlikely to reproduce on a new split.")
        if "mean_train_score" in res.columns:
            g = float(res.loc[srch.best_index_, "mean_train_score"]) - srch.best_score_
            if g > 0.1:
                print(f"  warning: train-test gap of {g:.3f} at the best setting — "
                      f"the search has tuned toward overfitting.")
    return out


def nested_validate(
    model,
    param_space: Dict[str, Any],
    X,
    y,
    inner_cv: int = 3,
    outer_cv: int = 5,
    n_iter: int = 20,
    scoring=None,
    task: Literal["auto", "classification", "multiclass", "regression"] = "auto",
    groups=None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> Frame:
    """Nested cross-validation: the honest score for a model you tuned.

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
    """
    from sklearn.model_selection import RandomizedSearchCV

    task = _infer_task(y) if task == "auto" else task
    scoring_map = _resolve_scoring(task, scoring)
    primary = list(scoring_map)[0]
    outer = _make_cv(task, outer_cv, y, groups, random_state)

    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_arr = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    scores, chosen = [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(outer.split(X_df, y_arr, groups), 1):
        inner = _make_cv(task, inner_cv, y_arr.iloc[tr], None, random_state)
        srch = RandomizedSearchCV(model, param_space, n_iter=n_iter, cv=inner,
                                  scoring=scoring_map[primary], n_jobs=n_jobs,
                                  random_state=random_state, refit=True)
        srch.fit(X_df.iloc[tr], y_arr.iloc[tr])
        from sklearn.metrics import get_scorer
        s = get_scorer(scoring_map[primary])(srch.best_estimator_,
                                             X_df.iloc[te], y_arr.iloc[te])
        sign = -1 if str(scoring_map[primary]).startswith("neg_") else 1
        scores.append(sign * float(s))
        chosen.append(srch.best_params_)
        if verbose:
            print(f"  outer fold {fold}: {primary}={sign * s:.4f}")

    out = pd.DataFrame({"fold": range(1, len(scores) + 1), primary: np.round(scores, 4)})
    out.attrs["params_per_fold"] = chosen
    out.attrs["mean"] = round(float(np.mean(scores)), 4)
    out.attrs["sd"] = round(float(np.std(scores)), 4)
    out.attrs["elapsed_s"] = round(time.time() - t0, 1)

    if verbose:
        print(f"nested {primary}: {out.attrs['mean']:.4f} ± {out.attrs['sd']:.4f} "
              f"({out.attrs['elapsed_s']}s)")
        uniq = {k: len({str(p.get(k)) for p in chosen}) for k in chosen[0]}
        unstable = [k for k, v in uniq.items() if v == len(chosen) and len(chosen) > 2]
        if unstable:
            print(f"  note: {unstable} was chosen differently in every fold — "
                  f"that parameter is not identifiable from this much data.")
    return out


# ======================================================================
#  4. TRAIN & PREDICT
# ======================================================================

def train(
    model,
    X,
    y,
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
    cv: int = 5,
    calibrate: Optional[Literal["sigmoid", "isotonic"]] = None,
    task: Literal["auto", "classification", "multiclass", "regression"] = "auto",
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit on all the data and return a ready-to-ship artifact.

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
    """
    task = _infer_task(y) if task == "auto" else task
    clf = task in ("classification", "multiclass")
    mdl = clone(model)

    if calibrate is not None:
        if not clf:
            raise ValueError("calibrate only applies to classifiers.")
        from sklearn.calibration import CalibratedClassifierCV
        mdl = CalibratedClassifierCV(mdl, method=calibrate, cv=cv)

    thr = 0.5 if clf else None
    if threshold == "auto":
        if task != "classification":
            raise ValueError("threshold='auto' needs a binary target.")
        splitter = _make_cv(task, cv, y, None, random_state)
        oof = cross_val_predict(clone(mdl), X, y, cv=splitter,
                                method="predict_proba")[:, 1]
        try:
            from .cleaning import tune_threshold as _tt
        except ImportError:
            from cleaning import tune_threshold as _tt
        r = _tt(pd.Series(y).to_numpy(), oof, metric="cost",
                cost_fn=cost_fn, cost_fp=cost_fp)
        thr = r["threshold"]
        flag_rate = float((oof >= thr).mean())
        if verbose:
            print(f"threshold chosen out-of-fold: {thr:.4f} "
                  f"(sens={r['sensitivity']:.3f}, spec={r['specificity']:.3f}, "
                  f"flags {flag_rate:.1%}, cost {cost_fn}:{cost_fp})")
        if flag_rate > 0.99 or flag_rate < 0.01:
            warnings.warn(
                f"The cost-optimal threshold flags {flag_rate:.1%} of rows, which "
                f"is a degenerate rule: at a {cost_fn}:{cost_fp} cost ratio with "
                f"{pd.Series(y).mean():.1%} prevalence, 'always predict the same "
                f"class' really is cheapest. The model is not being used. Lower "
                f"cost_fn, or optimise metric='f1' instead of cost.",
                stacklevel=2)
    elif isinstance(threshold, (int, float)):
        thr = float(threshold)

    t0 = time.time()
    mdl.fit(X, y)
    elapsed = round(time.time() - t0, 2)

    import sklearn
    art = {
        "model": mdl,
        "task": task,
        "threshold": thr,
        "classes": (list(getattr(mdl, "classes_", [])) if clf else None),
        "feature_names": (list(X.columns) if isinstance(X, pd.DataFrame) else None),
        "n_train": len(pd.Series(y)),
        "prevalence": (float(pd.Series(y).value_counts(normalize=True).min())
                       if task == "classification" else None),
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "fit_time_s": elapsed,
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
    }
    if verbose:
        print(f"trained {type(mdl).__name__} on {art['n_train']:,} rows in {elapsed}s")
    return art


def predict(artifact: Dict[str, Any], X, correct_prior: bool = False,
            true_prevalence: Optional[float] = None) -> Frame:
    """Predict with an artifact, honouring its stored threshold.

    Returns a frame with ``prediction`` and, for classifiers,
    ``probability``.  This exists so the threshold chosen at training time
    actually gets used -- calling ``model.predict`` directly silently
    reverts to 0.5 and quietly undoes the tuning.

    ``correct_prior=True`` rescales probabilities back to
    ``true_prevalence`` using :func:`cleaning.prior_correct`, for models
    trained on resampled data.
    """
    mdl, task = artifact["model"], artifact["task"]
    names = artifact.get("feature_names")
    if names and isinstance(X, pd.DataFrame):
        missing = [c for c in names if c not in X.columns]
        if missing:
            raise KeyError(f"X is missing {len(missing)} feature(s) the model was "
                           f"trained on, e.g. {missing[:5]}")
        X = X[names]

    if task == "regression":
        return pd.DataFrame({"prediction": mdl.predict(X)},
                            index=getattr(X, "index", None))

    proba = mdl.predict_proba(X)
    if task == "classification":
        p = proba[:, 1]
        if correct_prior:
            if true_prevalence is None:
                raise ValueError("correct_prior=True needs true_prevalence=.")
            try:
                from .cleaning import prior_correct
            except ImportError:
                from cleaning import prior_correct
            p = prior_correct(p, artifact["prevalence"], true_prevalence)
        thr = artifact.get("threshold", 0.5) or 0.5
        classes = artifact.get("classes") or [0, 1]
        return pd.DataFrame({"probability": p,
                             "prediction": np.where(p >= thr, classes[1], classes[0])},
                            index=getattr(X, "index", None))

    idx = proba.argmax(axis=1)
    classes = np.asarray(artifact.get("classes"))
    return pd.DataFrame({"prediction": classes[idx],
                         "probability": proba.max(axis=1)},
                        index=getattr(X, "index", None))


# ======================================================================
#  5. INTERPRETATION
# ======================================================================

def explain(
    model,
    X,
    y=None,
    method: Literal["permutation", "builtin", "shap", "coef"] = "permutation",
    scoring=None,
    n_repeats: int = 10,
    max_display: Optional[int] = None,
    sample: Optional[int] = 2000,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Frame:
    """Rank features by importance, with the caveats each method carries.

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
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model)
    except NotFittedError:
        raise NotFittedError(
            f"explain() needs a fitted estimator; {type(model).__name__} has not "
            f"been fit. Call model.fit(X, y) first, or use "
            f"md.train(...)['model'], or pull the fitted pipeline out of "
            f"md.tune(...)['best_model']."
        ) from None
    except (TypeError, ValueError):
        pass                                  # not an sklearn estimator; let it try

    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
    else:
        names = [f"f{i}" for i in range(np.asarray(X).shape[1])]

    Xs, ys = X, y
    if sample and len(X) > sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), sample, replace=False)
        Xs = X.iloc[idx] if isinstance(X, pd.DataFrame) else np.asarray(X)[idx]
        ys = None if y is None else pd.Series(y).reset_index(drop=True).iloc[idx]

    if method == "permutation":
        if ys is None:
            raise ValueError("permutation importance needs y.")
        from sklearn.inspection import permutation_importance
        task = _infer_task(ys)
        sc = scoring or list(_resolve_scoring(task, None))[0]
        sc = _resolve_scoring(task, None).get(sc, sc)
        r = permutation_importance(model, Xs, ys, scoring=sc, n_repeats=n_repeats,
                                   random_state=random_state, n_jobs=n_jobs)
        out = pd.DataFrame({"feature": names,
                            "importance": r.importances_mean.round(6),
                            "sd": r.importances_std.round(6)})
        out["note"] = np.where(out["importance"] <= 0, "no better than noise",
                               np.where(out["importance"] < out["sd"],
                                        "unstable (mean < sd)", ""))
        out.attrs["scoring"] = sc

    elif method == "builtin":
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            raise AttributeError(f"{type(est).__name__} has no feature_importances_; "
                                 f"use method='permutation'.")
        out = pd.DataFrame({"feature": names[:len(imp)],
                            "importance": np.round(imp, 6), "sd": np.nan})
        out["note"] = "impurity-based: biased toward high-cardinality features"

    elif method == "coef":
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        coef = getattr(est, "coef_", None)
        if coef is None:
            raise AttributeError(f"{type(est).__name__} has no coef_.")
        coef = np.ravel(coef) if np.ndim(coef) == 1 or np.shape(coef)[0] == 1 \
            else np.abs(coef).mean(axis=0)
        out = pd.DataFrame({"feature": names[:len(coef)],
                            "coefficient": np.round(np.ravel(coef), 6),
                            "importance": np.abs(np.ravel(coef)).round(6),
                            "sd": np.nan})
        scaled = (isinstance(X, pd.DataFrame)
                  and X.select_dtypes("number").std().between(0.5, 2).all())
        out["note"] = "" if scaled else ("inputs look unscaled — coefficients are "
                                         "not comparable across features")

    elif method == "shap":
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "method='shap' needs the shap package: pip install shap\n"
                "For a model-agnostic alternative with no extra dependency, "
                "use method='permutation'."
            ) from exc
        expl = shap.Explainer(model, Xs)
        sv = expl(Xs)
        vals = np.abs(sv.values)
        if vals.ndim == 3:
            vals = vals.mean(axis=2)
        out = pd.DataFrame({"feature": names,
                            "importance": vals.mean(axis=0).round(6),
                            "sd": vals.std(axis=0).round(6), "note": ""})
        out.attrs["shap_values"] = sv

    else:
        raise ValueError("method must be permutation, builtin, coef or shap.")

    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.attrs["method"] = method
    return out.head(max_display) if max_display else out


def learning_curve(
    model,
    X,
    y,
    sizes: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    cv: Union[int, Any] = 5,
    scoring=None,
    task: Literal["auto", "classification", "multiclass", "regression"] = "auto",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Frame:
    """Would more data help? Train and validation score against training size.

    Read the two curves at the right-hand edge:

    - **Converged and both low** -- the model is underfitting. More rows
      will not help; you need better features or a stronger model.
    - **Still separated, validation rising** -- more data will help.
    - **Wide gap, validation flat** -- overfitting. Regularise or cut
      features; more rows help only slowly.

    This is the cheapest way to decide whether to spend the next week on
    data collection or on modelling.
    """
    from sklearn.model_selection import learning_curve as _lc

    task = _infer_task(y) if task == "auto" else task
    scoring_map = _resolve_scoring(task, scoring)
    primary = list(scoring_map)[0]
    splitter = _make_cv(task, cv, y, None, random_state)

    n, tr, te = _lc(model, X, y, train_sizes=np.asarray(sizes), cv=splitter,
                    scoring=scoring_map[primary], n_jobs=n_jobs, shuffle=True,
                    random_state=random_state)
    sign = -1 if str(scoring_map[primary]).startswith("neg_") else 1
    out = pd.DataFrame({
        "n_train": n,
        "train_mean": (sign * tr).mean(axis=1).round(4),
        "train_sd": tr.std(axis=1).round(4),
        "val_mean": (sign * te).mean(axis=1).round(4),
        "val_sd": te.std(axis=1).round(4),
    })
    out["gap"] = (out["train_mean"] - out["val_mean"]).round(4)
    last = out.iloc[-1]
    prev = out.iloc[-2] if len(out) > 1 else last
    improving = last["val_mean"] - prev["val_mean"] > last["val_sd"]
    out.attrs["metric"] = primary
    out.attrs["verdict"] = (
        "more data should help — validation is still improving" if improving else
        "overfitting — close the train/validation gap before collecting more"
        if last["gap"] > 0.1 else
        "converged — more rows will not help; improve features or model")
    return out


def plot_learning_curve(curve: Frame, figsize: Tuple[float, float] = (7, 4.5),
                        show: bool = True):
    """Plot a :func:`learning_curve` table with its verdict in the title."""
    plt = _plt()
    m = curve.attrs.get("metric", "score")
    with _style():
        fig, ax = plt.subplots(figsize=figsize)
        for col, sd, lab, c in [("train_mean", "train_sd", "train", PALETTE[1]),
                                ("val_mean", "val_sd", "validation", PALETTE[0])]:
            ax.plot(curve["n_train"], curve[col], "o-", lw=2, color=c, label=lab)
            ax.fill_between(curve["n_train"], curve[col] - curve[sd],
                            curve[col] + curve[sd], alpha=.15, color=c)
        ax.set_xlabel("training rows")
        ax.set_ylabel(m)
        ax.set_title(f"Learning curve — {curve.attrs.get('verdict', '')}",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=9)
        return _finish(fig, show)


def plot_importance(importance: Frame, top: int = 20,
                    figsize: Optional[Tuple[float, float]] = None,
                    show: bool = True):
    """Horizontal bar chart of an :func:`explain` table.

    Error bars are the spread across permutation repeats; a bar whose error
    bar crosses zero is not distinguishable from noise, and is drawn in
    grey to say so.
    """
    plt = _plt()
    d = importance.head(top).iloc[::-1]
    has_sd = d["sd"].notna().any()
    colors = [NEUTRAL if (has_sd and pd.notna(s) and v - s <= 0) else PALETTE[0]
              for v, s in zip(d["importance"], d["sd"])]
    with _style():
        fig, ax = plt.subplots(figsize=figsize or (8, max(3, .32 * len(d))))
        ax.barh(range(len(d)), d["importance"], color=colors,
                xerr=d["sd"] if has_sd else None, error_kw={"lw": 1, "alpha": .6})
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["feature"], fontsize=9)
        ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel(f"importance ({importance.attrs.get('method', '')})")
        ax.set_title("Feature importance", fontweight="bold", fontsize=11)
        return _finish(fig, show)


# ======================================================================
#  6. PERSISTENCE
# ======================================================================

def save_model(artifact: Dict[str, Any], path: str,
               states: Optional[Sequence[Dict[str, Any]]] = None,
               metrics: Optional[Frame] = None, notes: str = "",
               compress: int = 3) -> str:
    """Save the whole pipeline: preprocessing states, model, threshold, metrics.

    Saving only the estimator is the usual mistake.  Six months later the
    model loads fine and produces nonsense, because nobody recorded which
    median was used for imputation, which categories the encoder knew, or
    what threshold the reported sensitivity assumed.  Everything needed to
    reproduce a prediction goes in one file.

    Unpickling executes code, so never load an artifact you did not create.
    """
    import os
    import joblib
    payload = {
        "artifact": artifact,
        "states": list(states) if states is not None else None,
        "metrics": metrics,
        "notes": notes,
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "navdata_version": _package_version(),
    }
    if states is not None:
        payload["state_summary"] = describe_states(list(states))
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    joblib.dump(payload, path, compress=compress)
    return os.path.abspath(path)


def load_model(path: str, check_versions: bool = True) -> Dict[str, Any]:
    """Load a :func:`save_model` bundle, warning on library version drift."""
    import joblib
    import sklearn
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "artifact" not in payload:
        raise ValueError(f"{path} is not a navdata model bundle.")
    if check_versions:
        saved = payload["artifact"].get("versions", {}).get("sklearn")
        if saved and saved != sklearn.__version__:
            warnings.warn(
                f"Model was trained with scikit-learn {saved}, you have "
                f"{sklearn.__version__}. It will usually still predict, but "
                f"verify a few known rows before trusting it.", stacklevel=2)
    return payload


def _package_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


__all__ = [
    # bridge & pipelines
    "ChainTransformer", "make_preprocessor", "make_pipeline",
    # validation
    "validate", "compare_algorithms", "nested_validate",
    # tuning
    "tune",
    # train & predict
    "train", "predict",
    # interpretation
    "explain", "plot_importance", "learning_curve", "plot_learning_curve",
    # persistence
    "save_model", "load_model",
    # constants
    "DEFAULT_SCORING",
]

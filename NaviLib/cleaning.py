"""
datakit
~~~~~~~

A practical toolkit for tabular data cleaning, feature selection and
class-imbalance handling.

Design principles
-----------------
1. **Short, verb-first names.**  ``scan_*`` inspects and never mutates,
   ``fix_*`` returns a modified copy, ``select_*`` chooses columns.
2. **One function per job, ``method=`` picks the algorithm.**  You should
   not have to remember ten function names to try ten techniques.
3. **Leakage-safe by default.**  Anything that learns from data returns a
   reusable ``state`` so the exact same transform can be replayed on the
   test set.  Resampling helpers refuse to touch a test set at all.
4. **Nothing silent.**  Every function can return a report describing what
   it did and what it changed.

Quick start
-----------
>>> import datakit as dp
>>> dp.overview(df)
>>> df = dp.clean_names(df)
>>> df, state = dp.fix_missing(df, method="median", return_state=True)
>>> ranking, keep = dp.select_features(df, target="y", method="tree")
>>> dp.compare_balance(X, y)              # which imbalance strategy actually helps?
>>> pipe = dp.balance_pipeline("smotenc", model, categorical=cat_cols)

Author: rebuilt from feature_engineering_lib
License: MIT
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, LassoCV
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__version__ = "2.0.0"

Frame = pd.DataFrame
Series = pd.Series

# ----------------------------------------------------------------------
# optional dependencies
# ----------------------------------------------------------------------

def _require(name: str, pip_name: Optional[str] = None):
    """Import an optional dependency with an actionable error message."""
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"This feature needs `{name}`. Install it with: "
            f"pip install {pip_name or name}"
        ) from exc


# ----------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------

def _split_xy(df: Frame, target: str) -> Tuple[Frame, Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(df.columns)[:10]}...")
    y = df[target]
    X = df.drop(columns=[target])
    keep = y.notna()
    if (~keep).any():
        warnings.warn(f"Dropped {(~keep).sum()} rows with a missing target.", stacklevel=3)
    return X.loc[keep], y.loc[keep]


def column_types(df: Frame) -> Dict[str, List[str]]:
    """Split columns into numeric / categorical / datetime / boolean buckets.

    Used internally everywhere, but exposed because it is handy on its own.
    """
    bools = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    nums = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in bools and c not in dates
    ]
    cats = [
        c for c in df.columns
        if c not in bools + dates + nums
    ]
    return {"numeric": nums, "categorical": cats, "boolean": bools, "datetime": dates}


def _resolve_columns(df: Frame, columns: Union[None, str, Sequence[str]]) -> List[str]:
    if columns is None:
        return list(df.columns)
    if isinstance(columns, str):
        columns = [columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return list(columns)


def _is_binary_target(y: Series) -> bool:
    return y.nunique(dropna=True) == 2


def _minority_label(y: Series):
    return y.value_counts().idxmin()


# ======================================================================
#  1. INSPECT  --  read-only diagnostics
# ======================================================================

def overview(df: Frame, max_rows: int = 40) -> Frame:
    """One-glance profile of every column: dtype, missing, cardinality, skew.

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
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        miss = s.isna().sum()
        rec: Dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(miss),
            "missing_pct": round(miss / n * 100, 2),
            "unique": int(nunique),
            "unique_pct": round(nunique / n * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = s.describe()
            rec.update(
                mean=round(float(desc.get("mean", np.nan)), 4),
                std=round(float(desc.get("std", np.nan)), 4),
                min=float(desc.get("min", np.nan)),
                max=float(desc.get("max", np.nan)),
                skew=round(float(s.skew()), 3) if nunique > 2 else np.nan,
            )
            rec["top"] = np.nan
        else:
            vc = s.value_counts(dropna=True)
            rec.update(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, skew=np.nan)
            rec["top"] = f"{vc.index[0]} ({vc.iloc[0]})" if len(vc) else np.nan

        flags = []
        if nunique <= 1:
            flags.append("constant")
        if nunique == n and n > 20:
            flags.append("id-like")
        if rec["missing_pct"] > 40:
            flags.append("high-missing")
        if 0 < nunique <= 2 and pd.api.types.is_numeric_dtype(s):
            flags.append("binary")
        if not pd.api.types.is_numeric_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() > 0.9 * s.notna().sum() and s.notna().sum() > 0:
                flags.append("numeric-as-text")
        if pd.api.types.is_object_dtype(s) and nunique > 50:
            flags.append("high-cardinality")
        rec["flags"] = ", ".join(flags)
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("column")
    return out


def scan_missing(df: Frame, columns=None, by: Optional[str] = None) -> Frame:
    """Missing-value report, optionally broken down by a grouping column.

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
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    cols = _resolve_columns(df, columns)
    if by is not None and by in cols:
        cols.remove(by)

    n = len(df)
    out = pd.DataFrame({
        "dtype": df[cols].dtypes.astype(str),
        "missing": df[cols].isna().sum(),
        "missing_pct": (df[cols].isna().mean() * 100).round(2),
        "present": df[cols].notna().sum(),
    })

    if by is not None:
        if by not in df.columns:
            raise KeyError(f"Group column '{by}' not found.")
        per_group = df.groupby(by, observed=True)[cols].apply(lambda g: g.isna().mean() * 100)
        for val in per_group.index:
            out[f"missing_pct__{val}"] = per_group.loc[val].round(2)
        gcols = [c for c in out.columns if c.startswith("missing_pct__")]
        out["spread"] = (out[gcols].max(axis=1) - out[gcols].min(axis=1)).round(2)
        out["flag"] = np.where(out["spread"] > 5, "possibly MNAR", "")

    return out.sort_values("missing", ascending=False)


def scan_duplicates(df: Frame, subset=None, show: int = 0) -> Dict[str, Any]:
    """Count duplicated rows, optionally on a subset of key columns.

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
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    dup_extra = df.duplicated(subset=subset, keep="first")
    dup_all = df.duplicated(subset=subset, keep=False)
    rep = {
        "n_rows": len(df),
        "n_duplicated": int(dup_extra.sum()),
        "n_affected": int(dup_all.sum()),
        "pct": round(dup_extra.sum() / len(df) * 100, 2),
        "subset": subset,
    }
    if show:
        rep["examples"] = df.loc[dup_all].sort_values(
            by=subset or list(df.columns)
        ).head(show)
    return rep


def scan_outliers(
    df: Frame,
    columns=None,
    method: Literal["iqr", "zscore", "mad", "quantile"] = "iqr",
    factor: float = 1.5,
    z: float = 3.0,
    q: Tuple[float, float] = (0.01, 0.99),
) -> Frame:
    """Count outliers per numeric column and report the cut-off bounds.

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
    """
    cols = _resolve_columns(df, columns)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric columns to scan.")

    rows = []
    for c in cols:
        s = df[c].dropna()
        lo, hi = _outlier_bounds(s, method, factor, z, q)
        mask = (df[c] < lo) | (df[c] > hi)
        rows.append({
            "column": c, "method": method,
            "lower": lo, "upper": hi,
            "n_outliers": int(mask.sum()),
            "pct": round(mask.sum() / len(df) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("column").sort_values("n_outliers", ascending=False)


def _outlier_bounds(s: Series, method: str, factor: float, z: float, q) -> Tuple[float, float]:
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - factor * iqr, q3 + factor * iqr
    if method == "zscore":
        m, sd = s.mean(), s.std()
        return m - z * sd, m + z * sd
    if method == "mad":
        med = s.median()
        mad = (s - med).abs().median() * 1.4826
        if mad == 0:
            mad = s.std() or 1.0
        return med - z * mad, med + z * mad
    if method == "quantile":
        return s.quantile(q[0]), s.quantile(q[1])
    raise ValueError("method must be one of: iqr, zscore, mad, quantile")


def check_numeric(df: Frame, column: str, sample: int = 5) -> Dict[str, Any]:
    """Find the exact values that stop a column from converting to numeric.

    Returns a dict with ``n_invalid``, ``pct_invalid``, ``bad_values``
    (unique offenders) and ``examples`` (sample rows).  Use this before
    ``convert(..., to="numeric")`` so you know what you are about to
    turn into NaN.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")
    coerced = pd.to_numeric(df[column], errors="coerce")
    bad = coerced.isna() & df[column].notna()
    uniq = df.loc[bad, column].drop_duplicates().tolist()
    return {
        "column": column,
        "n_rows": len(df),
        "n_invalid": int(bad.sum()),
        "pct_invalid": round(bad.sum() / len(df) * 100, 3),
        "n_distinct_bad": len(uniq),
        "bad_values": uniq[:50],
        "examples": df.loc[bad, [column]].head(sample),
    }


def scan_leakage(df: Frame, target: str, threshold: float = 0.95) -> Frame:
    """Flag features suspiciously predictive of the target on their own.

    A single feature reaching AUC >= ``threshold`` is almost never good
    news: it usually means the feature is recorded *after* the outcome, is
    a proxy for it, or is the outcome renamed.  This is the check that
    would have caught 'intubation' predicting neonatal death.

    Returns
    -------
    DataFrame with per-feature univariate AUC (binary target) or absolute
    correlation (continuous target), sorted descending, plus a ``flag``.
    """
    from sklearn.metrics import roc_auc_score

    X, y = _split_xy(df, target)
    binary = _is_binary_target(y)
    if binary:
        y_bin = (y == _minority_label(y)).astype(int)

    rows = []
    for c in X.columns:
        s = X[c]
        if not pd.api.types.is_numeric_dtype(s):
            codes = s.astype("category").cat.codes.replace(-1, np.nan)
        else:
            codes = s
        ok = codes.notna()
        if ok.sum() < 10 or codes[ok].nunique() < 2:
            continue
        if binary:
            try:
                a = roc_auc_score(y_bin[ok], codes[ok])
            except ValueError:
                continue
            score = max(a, 1 - a)
        else:
            score = abs(np.corrcoef(codes[ok], y[ok])[0, 1])
        rows.append({"feature": c, "score": round(float(score), 4)})

    out = (pd.DataFrame(rows)
           .sort_values("score", ascending=False)
           .reset_index(drop=True))
    out["metric"] = "univariate_auc" if binary else "abs_correlation"
    out["flag"] = np.where(out["score"] >= threshold, "SUSPICIOUS - check timing", "")
    return out


# ======================================================================
#  2. CLEAN  --  return a modified copy
# ======================================================================

def clean_names(
    df: Frame,
    style: Literal["snake", "upper_snake", "kebab", "camel", "pascal", "compact"] = "snake",
    rename: Optional[Dict[str, str]] = None,
    columns=None,
    dedupe: bool = True,
) -> Frame:
    """Normalise column names.

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
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    targets = _resolve_columns(df, columns)

    def to_words(name: str) -> List[str]:
        s = str(name).strip()
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)      # camelCase -> camel Case
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)   # HTTPServer -> HTTP Server
        s = re.sub(r"[^0-9A-Za-z]+", " ", s)
        return [w for w in s.split() if w]

    def apply_style(name: str) -> str:
        words = to_words(name)
        if not words:
            return str(name)
        if style == "snake":
            out = "_".join(w.lower() for w in words)
        elif style == "upper_snake":
            out = "_".join(w.upper() for w in words)
        elif style == "kebab":
            out = "-".join(w.lower() for w in words)
        elif style == "camel":
            out = words[0].lower() + "".join(w.capitalize() for w in words[1:])
        elif style == "pascal":
            out = "".join(w.capitalize() for w in words)
        elif style == "compact":
            out = "".join(w.lower() for w in words)
        else:
            raise ValueError(f"Unknown style: {style}")
        if out and out[0].isdigit():
            out = "_" + out
        return out

    mapping = {c: (apply_style(c) if c in targets else c) for c in df.columns}
    if rename:
        mapping.update({k: v for k, v in rename.items() if k in df.columns})

    if dedupe:
        seen: Dict[str, int] = {}
        for k, v in list(mapping.items()):
            if v in seen:
                seen[v] += 1
                mapping[k] = f"{v}_{seen[v]}"
            else:
                seen[v] = 1

    return df.rename(columns=mapping).copy()


def convert(
    df: Frame,
    column: str,
    to: Literal["numeric", "category", "string", "datetime", "boolean"],
    mapping: Optional[Dict[Any, Any]] = None,
    bins: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    n_bins: Optional[int] = None,
    date_format: Optional[str] = None,
    errors: Literal["coerce", "raise"] = "coerce",
    verbose: bool = False,
) -> Frame:
    """Convert a column's dtype, with optional value mapping and binning.

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
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")

    s = df[column].copy()
    if mapping is not None:
        s = s.replace(mapping)

    before_na = s.isna().sum()

    if to == "numeric":
        out = pd.to_numeric(s, errors=errors)
    elif to == "datetime":
        out = pd.to_datetime(s, format=date_format, errors=errors)
    elif to == "string":
        out = s.astype("string")
    elif to == "boolean":
        truthy = {"yes", "y", "true", "t", "1", 1, 1.0, True}
        falsy = {"no", "n", "false", "f", "0", 0, 0.0, False}
        def _b(v):
            if pd.isna(v):
                return pd.NA
            k = v.strip().lower() if isinstance(v, str) else v
            if k in truthy:
                return True
            if k in falsy:
                return False
            return pd.NA
        out = s.map(_b).astype("boolean")
    elif to == "category":
        if bins is not None:
            if labels is not None and len(labels) != len(bins) - 1:
                raise ValueError("len(labels) must equal len(bins) - 1.")
            out = pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=labels)
        elif n_bins is not None:
            out = pd.qcut(pd.to_numeric(s, errors="coerce"),
                          q=n_bins, labels=labels, duplicates="drop")
        else:
            out = s.astype("category")
    else:
        raise ValueError("`to` must be numeric, category, string, datetime or boolean.")

    lost = int(out.isna().sum() - before_na)
    if lost > 0:
        warnings.warn(
            f"convert('{column}' -> {to}): {lost} value(s) became NaN. "
            f"Run check_numeric() first to see which.",
            stacklevel=2,
        )
    if verbose:
        print(f"{column}: {s.dtype} -> {out.dtype}  ({lost} new NaN)")

    res = df.copy()
    res[column] = out
    return res


def drop_missing(
    df: Frame,
    max_missing: Union[float, int] = 0.5,
    axis: Literal["columns", "rows"] = "columns",
    protect: Optional[Sequence[str]] = None,
) -> Frame:
    """Drop columns (or rows) whose missing rate exceeds ``max_missing``.

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
    """
    if axis not in ("columns", "rows"):
        raise ValueError("axis must be 'columns' or 'rows'")
    protect = set(protect or [])

    ax = 0 if axis == "columns" else 1
    if isinstance(max_missing, float) and not isinstance(max_missing, bool):
        if not 0 <= max_missing <= 1:
            raise ValueError("Float max_missing must be within [0, 1].")
        score = df.isna().mean(axis=ax)
    elif isinstance(max_missing, (int, np.integer)) and not isinstance(max_missing, bool):
        if max_missing < 0:
            raise ValueError("Integer max_missing must be >= 0.")
        score = df.isna().sum(axis=ax)
    else:
        raise TypeError("max_missing must be a float in [0,1] or a non-negative int.")

    drop_mask = score > max_missing
    if axis == "columns":
        drop_mask = drop_mask & ~drop_mask.index.isin(protect)
        return df.loc[:, ~drop_mask].copy()
    return df.loc[~drop_mask, :].copy()


def drop_constant(df: Frame, protect: Optional[Sequence[str]] = None,
                  max_dominance: float = 1.0) -> Frame:
    """Drop columns with a single value, or dominated by one value.

    ``max_dominance=0.99`` drops any column where 99% of non-missing rows
    share the same value.  Such columns cost degrees of freedom and teach
    the model nothing -- 'steroid therapy', present in 0.5% of rows, is the
    classic example.
    """
    protect = set(protect or [])
    drop = []
    for c in df.columns:
        if c in protect:
            continue
        s = df[c].dropna()
        if s.nunique() <= 1:
            drop.append(c)
        elif max_dominance < 1.0 and len(s) and s.value_counts(normalize=True).iloc[0] >= max_dominance:
            drop.append(c)
    return df.drop(columns=drop).copy()


def fix_duplicates(
    df: Frame,
    keep: Literal["first", "last", "none"] = "first",
    subset=None,
) -> Frame:
    """Remove duplicated rows.

    ``keep="none"`` removes *every* copy including the original -- only use
    that when a duplicate means the record is untrustworthy.
    """
    keep_arg: Any = False if keep == "none" else keep
    return df.drop_duplicates(subset=subset, keep=keep_arg).copy()


def fix_outliers(
    df: Frame,
    columns=None,
    method: Literal["iqr", "zscore", "mad", "quantile"] = "iqr",
    action: Literal["clip", "nan", "drop"] = "clip",
    factor: float = 1.5,
    z: float = 3.0,
    q: Tuple[float, float] = (0.01, 0.99),
    return_state: bool = False,
):
    """Clip, blank out, or drop outlying values.

    ``action="clip"`` (winsorising) is usually the least destructive: it
    keeps the row and its other features while removing the leverage of the
    extreme value.  ``action="drop"`` removes whole rows and can silently
    delete most of your minority class -- check before using it.

    Set ``return_state=True`` to get the learned bounds back, then replay
    them on the test set with ``apply_state``.
    """
    cols = _resolve_columns(df, columns)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]

    bounds = {c: _outlier_bounds(df[c].dropna(), method, factor, z, q) for c in cols}
    out = df.copy()

    if action == "drop":
        mask = pd.Series(False, index=df.index)
        for c, (lo, hi) in bounds.items():
            mask |= (df[c] < lo) | (df[c] > hi)
        out = out.loc[~mask]
    else:
        for c, (lo, hi) in bounds.items():
            if action == "clip":
                out[c] = out[c].clip(lo, hi)
            else:  # nan
                out.loc[(out[c] < lo) | (out[c] > hi), c] = np.nan

    state = {"kind": "outliers", "bounds": bounds, "action": action}
    return (out, state) if return_state else out


def scan_outliers_multivariate(
    df: Frame,
    columns=None,
    method: Literal["isolation_forest", "lof", "elliptic", "mahalanobis"] = "isolation_forest",
    contamination: float = 0.03,
    n_neighbors: int = 20,
    scale: bool = True,
    random_state: int = 42,
    return_state: bool = False,
):
    """Score every row for how anomalous it is across several columns at once.

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

    Returns
    -------
    DataFrame with the original index plus ``outlier_score`` (higher =
    more anomalous), ``is_outlier``, and the per-column z-scores of the
    flagged rows so you can see *why* each one was flagged. With
    ``return_state=True`` also returns a replayable state.

    >>> flags, st = dp.scan_outliers_multivariate(train, return_state=True)
    >>> flags[flags.is_outlier].head()
    >>> test_flags = dp.apply_state(test, st)          # same fitted model
    """
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in (columns or df.columns)
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]
    if len(cols) < 2:
        raise ValueError(
            f"Multivariate detection needs at least 2 numeric columns, got {cols}. "
            f"For a single column use fix_outliers(method='iqr'|'mad'), which is "
            f"faster and exactly equivalent in one dimension."
        )

    X = df[cols].to_numpy(dtype=float)
    nan_rows = np.isnan(X).any(axis=1)
    if nan_rows.any():
        warnings.warn(f"{int(nan_rows.sum())} row(s) contain NaN and cannot be "
                      f"scored; they are returned with score=NaN, is_outlier=False. "
                      f"Impute first with fix_missing() to include them.",
                      stacklevel=2)
    Xc = X[~nan_rows]
    if len(Xc) < 10:
        raise ValueError("Fewer than 10 complete rows; nothing to fit.")

    scaler = StandardScaler().fit(Xc) if scale else None
    Xs = scaler.transform(Xc) if scale else Xc

    score = np.full(len(df), np.nan)
    flag = np.zeros(len(df), dtype=bool)
    model = None

    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=contamination, n_estimators=300,
                                random_state=random_state).fit(Xs)
        score[~nan_rows] = -model.score_samples(Xs)
        flag[~nan_rows] = model.predict(Xs) == -1

    elif method == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(Xs) - 1),
                                   contamination=contamination, novelty=True).fit(Xs)
        score[~nan_rows] = -model.score_samples(Xs)
        flag[~nan_rows] = model.predict(Xs) == -1

    elif method == "elliptic":
        from sklearn.covariance import EllipticEnvelope
        model = EllipticEnvelope(contamination=contamination,
                                 random_state=random_state).fit(Xs)
        score[~nan_rows] = -model.score_samples(Xs)
        flag[~nan_rows] = model.predict(Xs) == -1

    elif method == "mahalanobis":
        from scipy.stats import chi2
        from sklearn.covariance import MinCovDet
        model = MinCovDet(random_state=random_state).fit(Xs)
        d2 = model.mahalanobis(Xs)
        score[~nan_rows] = d2
        cutoff = chi2.ppf(1 - contamination, df=len(cols))
        flag[~nan_rows] = d2 > cutoff

    else:
        raise ValueError("method must be isolation_forest, lof, elliptic or mahalanobis.")

    out = pd.DataFrame({"outlier_score": score, "is_outlier": flag}, index=df.index)
    z = pd.DataFrame(np.nan, index=df.index, columns=[f"z_{c}" for c in cols])
    if scale:
        z.loc[~nan_rows, :] = Xs
    else:
        z.loc[~nan_rows, :] = StandardScaler().fit_transform(Xc)
    out = pd.concat([out, z.round(2)], axis=1)

    driver = z.abs().idxmax(axis=1)
    out["main_driver"] = driver.str[2:].where(out["is_outlier"], "")
    out.attrs["n_flagged"] = int(flag.sum())
    out.attrs["pct_flagged"] = round(float(flag.mean() * 100), 2)
    out.attrs["columns"] = cols

    state = {"kind": "outliers_mv", "columns": cols, "method": method,
             "model": model, "scaler": scaler, "contamination": contamination}
    return (out, state) if return_state else out


def fix_outliers_multivariate(
    df: Frame,
    columns=None,
    method: Literal["isolation_forest", "lof", "elliptic", "mahalanobis"] = "isolation_forest",
    contamination: float = 0.03,
    action: Literal["drop", "flag"] = "flag",
    **kwargs,
) -> Frame:
    """Drop or flag multivariate outliers.

    ``action="flag"`` (the default) adds ``is_outlier`` and
    ``outlier_score`` columns and changes nothing else -- almost always the
    right first move, because dropping rows is irreversible and, on
    imbalanced data, disproportionately deletes the minority class. Check
    what would go before you let it go.
    """
    flags = scan_outliers_multivariate(df, columns, method=method,
                                       contamination=contamination, **kwargs)
    if action == "flag":
        out = df.copy()
        out["is_outlier"] = flags["is_outlier"].to_numpy()
        out["outlier_score"] = flags["outlier_score"].to_numpy()
        return out
    if action == "drop":
        return df.loc[~flags["is_outlier"].to_numpy()].copy()
    raise ValueError("action must be 'drop' or 'flag'.")


def fix_missing(
    df: Frame,
    columns=None,
    method: Literal["mean", "median", "mode", "constant",
                    "knn", "mice", "ffill", "bfill", "drop_rows"] = "median",
    fill_value: Any = None,
    n_neighbors: int = 5,
    estimator=None,
    using: Optional[Sequence[str]] = None,
    add_indicator: bool = False,
    return_state: bool = False,
    random_state: int = 42,
):
    """Impute missing values.

    Multivariate methods are done properly here: ``knn`` and ``mice`` fit on
    *all* numeric helper columns, not on the target column alone.  (Fitting
    KNNImputer on a single column silently degrades to mean imputation --
    a common and invisible bug.)

    Parameters
    ----------
    columns : list, optional
        Columns to impute.  ``None`` = every column with missing values.
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
    """
    cols = _resolve_columns(df, columns)
    out = df.copy()

    indicators: List[str] = []
    if add_indicator:
        for c in cols:
            if out[c].isna().any():
                out[f"{c}_was_missing"] = out[c].isna().astype(int)
                indicators.append(c)

    if method == "drop_rows":
        out = out.dropna(subset=cols)
        state = {"kind": "missing", "method": method, "columns": cols,
                 "indicators": indicators}
        return (out, state) if return_state else out

    if method in ("ffill", "bfill"):
        out[cols] = out[cols].ffill() if method == "ffill" else out[cols].bfill()
        state = {"kind": "missing", "method": method, "columns": cols,
                 "indicators": indicators}
        return (out, state) if return_state else out

    types = column_types(out)
    num_cols = [c for c in cols if c in types["numeric"]]
    other_cols = [c for c in cols if c not in num_cols]

    state: Dict[str, Any] = {"kind": "missing", "method": method,
                             "numeric": {}, "other": {}, "columns": cols,
                             "indicators": indicators}

    # ---- multivariate ------------------------------------------------
    if method in ("knn", "mice"):
        helpers = list(using) if using is not None else types["numeric"]
        block = sorted(set(num_cols) | set(helpers))
        if len(block) < 2:
            raise ValueError(
                f"method='{method}' needs at least 2 numeric columns to be useful; "
                f"found {block}. Use method='median' instead."
            )
        if method == "knn":
            imp = KNNImputer(n_neighbors=n_neighbors)
        else:
            imp = IterativeImputer(
                estimator=estimator or BayesianRidge(),
                max_iter=10, sample_posterior=True, random_state=random_state,
            )
        out[block] = imp.fit_transform(out[block])
        state["multivariate"] = {"imputer": imp, "block": block}
        # non-numeric columns still need a strategy
        for c in other_cols:
            mode = out[c].mode(dropna=True)
            val = mode.iloc[0] if len(mode) else fill_value
            out[c] = out[c].fillna(val)
            state["other"][c] = val
        return (out, state) if return_state else out

    # ---- univariate --------------------------------------------------
    for c in num_cols:
        if method == "mean":
            val = out[c].mean()
        elif method == "median":
            val = out[c].median()
        elif method == "mode":
            m = out[c].mode(dropna=True)
            val = m.iloc[0] if len(m) else fill_value
        elif method == "constant":
            if fill_value is None:
                raise ValueError("fill_value is required when method='constant'.")
            val = fill_value
        else:
            raise ValueError(f"Unknown method: {method}")
        out[c] = out[c].fillna(val)
        state["numeric"][c] = val

    for c in other_cols:
        if method == "constant":
            val = fill_value
        else:
            m = out[c].mode(dropna=True)
            val = m.iloc[0] if len(m) else fill_value
        if val is not None:
            out[c] = out[c].fillna(val)
        state["other"][c] = val

    return (out, state) if return_state else out


def apply_state(df: Frame, state: Dict[str, Any]) -> Frame:
    """Replay a fitted cleaning step on new data (test set, production).

    Accepts a single ``state`` dict or a list of them, applied in order.
    This is what keeps your test set honest: the medians, bounds and
    imputers all come from the training data.

    >>> train, s1 = dp.fix_missing(train, method="median", return_state=True)
    >>> test = dp.apply_state(test, s1)
    """
    if isinstance(state, (list, tuple)):
        for st in state:
            df = apply_state(df, st)
        return df

    out = df.copy()
    kind = state.get("kind")

    if kind == "outliers":
        for c, (lo, hi) in state["bounds"].items():
            if c not in out.columns:
                continue
            if state["action"] == "clip":
                out[c] = out[c].clip(lo, hi)
            elif state["action"] == "nan":
                out.loc[(out[c] < lo) | (out[c] > hi), c] = np.nan
        return out

    if kind == "missing":
        for c in state.get("indicators", []):
            if c in out.columns:
                out[f"{c}_was_missing"] = out[c].isna().astype(int)
        method = state["method"]
        if method == "drop_rows":
            return out.dropna(subset=[c for c in state["columns"] if c in out.columns])
        if method in ("ffill", "bfill"):
            cols = [c for c in state["columns"] if c in out.columns]
            out[cols] = out[cols].ffill() if method == "ffill" else out[cols].bfill()
            return out
        if "multivariate" in state:
            block = state["multivariate"]["block"]
            missing_cols = [c for c in block if c not in out.columns]
            if missing_cols:
                raise KeyError(f"apply_state: new data is missing columns {missing_cols}")
            out[block] = state["multivariate"]["imputer"].transform(out[block])
        for c, val in state.get("numeric", {}).items():
            if c in out.columns:
                out[c] = out[c].fillna(val)
        for c, val in state.get("other", {}).items():
            if c in out.columns and val is not None:
                out[c] = out[c].fillna(val)
        return out

    if kind == "outliers_mv":
        cols = state["columns"]
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise KeyError(f"apply_state: outlier detector needs columns {missing}")
        X = out[cols].to_numpy(dtype=float)
        ok = ~np.isnan(X).any(axis=1)
        flag = np.zeros(len(out), dtype=bool)
        score = np.full(len(out), np.nan)
        if ok.any():
            Xs = state["scaler"].transform(X[ok]) if state["scaler"] is not None else X[ok]
            if state["method"] == "mahalanobis":
                from scipy.stats import chi2
                d2 = state["model"].mahalanobis(Xs)
                score[ok] = d2
                flag[ok] = d2 > chi2.ppf(1 - state["contamination"], df=len(cols))
            else:
                score[ok] = -state["model"].score_samples(Xs)
                flag[ok] = state["model"].predict(Xs) == -1
        out["is_outlier"] = flag
        out["outlier_score"] = score
        return out

    if kind == "rare":
        for c, keep in state["keep"].items():
            if c in out.columns:
                out[c] = out[c].where(out[c].isin(keep), state["other_label"])
        return out

    raise ValueError(f"Unknown state kind: {kind}")


def group_rare(
    df: Frame,
    columns=None,
    min_freq: float = 0.01,
    min_count: Optional[int] = None,
    other_label: str = "other",
    return_state: bool = False,
):
    """Merge rare categories into a single ``other`` bucket.

    Rare levels are the categorical twin of class imbalance: a level seen 3
    times cannot support a reliable coefficient, and one-hot encoding it
    just adds a near-zero column that invites overfitting.

    Parameters
    ----------
    min_freq : float
        Levels below this share of non-missing rows are merged.
    min_count : int, optional
        Absolute alternative to ``min_freq``; takes priority when given.
    """
    cols = _resolve_columns(df, columns)
    cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])
            or df[c].dtype.name == "category"]

    out = df.copy()
    keep_map: Dict[str, List[Any]] = {}
    for c in cols:
        vc = out[c].value_counts(dropna=True)
        if min_count is not None:
            keep = vc[vc >= min_count].index.tolist()
        else:
            keep = vc[vc / vc.sum() >= min_freq].index.tolist()
        keep_map[c] = keep
        if out[c].dtype.name == "category":
            out[c] = out[c].astype(object)
        out[c] = out[c].where(out[c].isin(keep) | out[c].isna(), other_label)

    state = {"kind": "rare", "keep": keep_map, "other_label": other_label}
    return (out, state) if return_state else out


# ======================================================================
#  3. FEATURES  --  selection and redundancy
# ======================================================================

def _build_preprocessor(X: Frame, scale: bool = True) -> Tuple[ColumnTransformer, Dict[str, str]]:
    """ColumnTransformer + a reliable output-name -> source-column map.

    The map is built from the transformer's own metadata rather than by
    splitting strings on '_', which breaks on names like ``birth_weight``.
    """
    t = column_types(X)
    num = t["numeric"] + t["boolean"]
    cat = t["categorical"]

    steps = []
    if num:
        inner = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            inner.append(("sc", StandardScaler()))
        steps.append(("num", SkPipeline(inner), num))
    if cat:
        steps.append((
            "cat",
            SkPipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=0.01, sparse_output=False)),
            ]),
            cat,
        ))
    if not steps:
        raise ValueError("No usable feature columns found.")

    pre = ColumnTransformer(steps, remainder="drop")
    return pre, {"num": num, "cat": cat}


def _source_map(pre: ColumnTransformer, groups: Dict[str, List[str]]) -> Dict[str, str]:
    """Map each engineered column name back to the original column."""
    names = pre.get_feature_names_out()
    mapping = {}
    for n in names:
        prefix, rest = n.split("__", 1)
        if prefix == "num":
            mapping[n] = rest
        else:
            # OneHotEncoder emits '<column>_<category>'; match the longest
            # source column that is a prefix, so 'birth_weight_low' resolves
            # to 'birth_weight' and not to 'birth'.
            candidates = [c for c in groups["cat"] if rest == c or rest.startswith(c + "_")]
            mapping[n] = max(candidates, key=len) if candidates else rest
    return mapping


def select_features(
    df: Frame,
    target: str,
    method: Literal["l1", "tree", "mi", "corr", "permutation"] = "tree",
    task: Literal["auto", "classification", "regression"] = "auto",
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    cv: int = 5,
    balanced: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs,
) -> Tuple[Frame, List[str]]:
    """Rank features by importance and return the ones worth keeping.

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

    Returns
    -------
    ranking : DataFrame  -- feature, score, plus method-specific columns
    selected : list of str
    """
    X, y = _split_xy(df, target)

    if task == "auto":
        task = ("classification"
                if (y.dtype == object or str(y.dtype) in ("category", "bool")
                    or y.nunique() <= 20)
                else "regression")
    is_clf = task == "classification"

    pre, groups = _build_preprocessor(X, scale=(method in ("l1", "corr")))
    Xt = pre.fit_transform(X)
    names = list(pre.get_feature_names_out())
    src = _source_map(pre, groups)

    cw = "balanced" if (balanced and is_clf) else None
    splitter = (StratifiedKFold(cv, shuffle=True, random_state=random_state) if is_clf
                else KFold(cv, shuffle=True, random_state=random_state))
    extra = pd.DataFrame(index=names)

    # ---------------- L1 ----------------
    if method == "l1":
        if is_clf:
            from sklearn.linear_model import LogisticRegressionCV
            Cs = kwargs.pop("Cs", 10)
            mdl = LogisticRegressionCV(
                Cs=Cs, penalty="l1", solver="saga", cv=splitter,
                class_weight=cw, max_iter=kwargs.pop("max_iter", 5000),
                scoring=kwargs.pop("scoring", "average_precision"),
                random_state=random_state, n_jobs=n_jobs,
            )
        else:
            mdl = LassoCV(cv=splitter, random_state=random_state,
                          n_jobs=n_jobs, max_iter=10000)
        mdl.fit(Xt, y)
        coef = np.ravel(mdl.coef_)
        extra["coefficient"] = coef
        raw = np.abs(coef)
        if threshold is None and top_k is None:
            threshold = 1e-12

    # ---------------- tree ----------------
    elif method == "tree":
        lgb = _require("lightgbm")
        Model = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
        params = dict(n_estimators=kwargs.pop("n_estimators", 400),
                      learning_rate=kwargs.pop("learning_rate", 0.05),
                      num_leaves=kwargs.pop("num_leaves", 31),
                      random_state=random_state, n_jobs=n_jobs, verbose=-1)
        if is_clf and balanced:
            params["class_weight"] = "balanced"
        acc = np.zeros(len(names))
        for tr, _ in splitter.split(Xt, y):
            m = Model(**params).fit(Xt[tr], y.iloc[tr])
            imp = np.asarray(m.booster_.feature_importance(importance_type="gain"), dtype=float)
            total = imp.sum()
            acc += imp / total if total > 0 else imp
        raw = acc / cv * 100  # percent of total gain, averaged over folds

    # ---------------- permutation ----------------
    elif method == "permutation":
        from sklearn.inspection import permutation_importance
        lgb = _require("lightgbm")
        Model = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
        params = dict(n_estimators=300, learning_rate=0.05,
                      random_state=random_state, n_jobs=n_jobs, verbose=-1)
        if is_clf and balanced:
            params["class_weight"] = "balanced"
        scoring = kwargs.pop("scoring", "average_precision" if is_clf else "r2")
        acc = np.zeros(len(names))
        for tr, te in splitter.split(Xt, y):
            m = Model(**params).fit(Xt[tr], y.iloc[tr])
            r = permutation_importance(m, Xt[te], y.iloc[te], scoring=scoring,
                                       n_repeats=kwargs.pop("n_repeats", 5),
                                       random_state=random_state, n_jobs=n_jobs)
            acc += r.importances_mean
        raw = acc / cv
        if threshold is None and top_k is None:
            threshold = 0.0  # anything not beating noise is dropped

    # ---------------- mutual information ----------------
    elif method == "mi":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        fn = mutual_info_classif if is_clf else mutual_info_regression
        raw = fn(Xt, y, random_state=random_state)

    # ---------------- correlation ----------------
    elif method == "corr":
        yv = y.astype("category").cat.codes.to_numpy() if (is_clf and y.dtype == object) else y.to_numpy()
        Xdf = pd.DataFrame(Xt, columns=names)
        raw = Xdf.apply(lambda col: abs(np.corrcoef(col, yv)[0, 1])
                        if col.std() > 0 else 0.0).to_numpy()
        raw = np.nan_to_num(raw)

    else:
        raise ValueError("method must be one of: l1, tree, mi, corr, permutation")

    # -------- aggregate engineered columns back to source columns -------
    per_out = pd.DataFrame({"encoded": names, "score": raw})
    per_out["feature"] = per_out["encoded"].map(src)
    for c in extra.columns:
        per_out[c] = extra[c].to_numpy()

    agg: Dict[str, Any] = {"score": "sum"}
    for c in extra.columns:
        agg[c] = (lambda s: s.iloc[np.argmax(np.abs(s.to_numpy()))])
    ranking = per_out.groupby("feature", as_index=False).agg(agg)

    ranking["score"] = ranking["score"].astype(float).round(6)
    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["method"] = method
    ranking["n_encoded"] = ranking["feature"].map(per_out["feature"].value_counts())

    if top_k is not None:
        selected = ranking.head(top_k)["feature"].tolist()
    elif threshold is not None:
        selected = ranking.loc[ranking["score"] > threshold, "feature"].tolist()
    else:
        selected = ranking["feature"].tolist()

    return ranking, selected


def drop_correlated(
    df: Frame,
    threshold: float = 0.95,
    target: Optional[str] = None,
    method: Literal["pearson", "spearman"] = "spearman",
    return_pairs: bool = False,
):
    """Remove one column from each pair of near-duplicate numeric features.

    When two features carry the same information, tree importances and
    linear coefficients get split between them and both look unimportant.
    Where a target is given, the member of each pair less correlated with
    the target is the one dropped.

    Returns the reduced frame, or ``(frame, pairs)`` with the decisions.
    """
    t = column_types(df)
    num = [c for c in t["numeric"] if c != target]
    if len(num) < 2:
        return (df.copy(), pd.DataFrame()) if return_pairs else df.copy()

    corr = df[num].corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    if target is not None:
        tcorr = df[num].corrwith(df[target], method=method).abs().fillna(0)
    else:
        tcorr = pd.Series(0.0, index=num)

    pairs, drop = [], set()
    for a in upper.index:
        for b in upper.columns:
            v = upper.loc[a, b]
            if pd.notna(v) and v >= threshold:
                loser = a if tcorr[a] < tcorr[b] else b
                pairs.append({"kept": b if loser == a else a, "dropped": loser,
                              "corr": round(float(v), 4)})
                drop.add(loser)

    out = df.drop(columns=list(drop))
    return (out, pd.DataFrame(pairs)) if return_pairs else out


# ======================================================================
#  4. BALANCE  --  class-imbalance handling
# ======================================================================

#: Every supported resampler, keyed by the short name you pass as ``method``.
BALANCE_METHODS: Dict[str, str] = {
    # --- do nothing -------------------------------------------------
    "none":       "No resampling. Baseline -- always compare against this.",
    # --- naive over/under -------------------------------------------
    "ros":        "Random over-sampling: duplicate minority rows.",
    "rus":        "Random under-sampling: discard majority rows.",
    # --- SMOTE family (numeric) -------------------------------------
    "smote":      "SMOTE: interpolate between minority neighbours. Numeric only.",
    "borderline": "Borderline-SMOTE: synthesise only near the decision boundary.",
    "svmsmote":   "SVM-SMOTE: synthesise around SVM support vectors.",
    "kmeans":     "KMeans-SMOTE: cluster first, then oversample sparse clusters. Fussy -- lower `cluster_balance_threshold` if it finds no valid clusters.",
    "adasyn":     "ADASYN: more synthetic points where learning is hardest.",
    # --- SMOTE family (mixed / categorical) -------------------------
    "smotenc":    "SMOTE-NC: mixed numeric + categorical. Needs `categorical`.",
    "smoten":     "SMOTE-N: all-categorical data (VDM distance).",
    # --- cleaning under-samplers ------------------------------------
    "tomek":      "Tomek links: remove majority rows sitting on the boundary.",
    "enn":        "Edited NN: remove rows disagreeing with their neighbours.",
    "ncr":        "Neighbourhood cleaning rule: gentler ENN variant.",
    "oss":        "One-sided selection: condensed NN + Tomek.",
    "nearmiss":   "NearMiss: keep majority rows closest to the minority.",
    "centroids":  "Cluster centroids: replace majority with cluster centres.",
    # --- hybrids ----------------------------------------------------
    "smotetomek": "SMOTE then Tomek cleaning.",
    "smoteenn":   "SMOTE then ENN cleaning. Aggressive.",
}


#: Samplers that compute Euclidean distances and therefore need an
#: all-numeric matrix.  Passing raw strings gives an unhelpful
#: "could not convert string to float" from deep inside scikit-learn.
NUMERIC_ONLY_METHODS = {
    "smote", "borderline", "svmsmote", "kmeans", "adasyn",
    "tomek", "enn", "ncr", "oss", "nearmiss", "centroids",
    "smotetomek", "smoteenn",
}


def _check_numeric_ready(X: Frame, method: str) -> None:
    """Fail early and legibly when a numeric-only sampler meets text columns."""
    if method not in NUMERIC_ONLY_METHODS:
        return
    bad = [c for c in X.columns
           if not pd.api.types.is_numeric_dtype(X[c])
           and not pd.api.types.is_bool_dtype(X[c])]
    if bad:
        raise TypeError(
            f"method='{method}' works on numeric data only, but these columns are "
            f"not numeric: {bad[:8]}{' ...' if len(bad) > 8 else ''}\n"
            f"  Pick one:\n"
            f"    1. method='smotenc', categorical={bad[:3]}...   <- keeps categories intact\n"
            f"    2. one-hot encode first: pd.get_dummies(df, columns={bad[:3]}...)\n"
            f"    3. method='ros' / 'rus' / class_weights()       <- dtype-agnostic\n"
            f"  Note: plain SMOTE on encoded binary columns invents values like 0.43, "
            f"which are not valid categories."
        )


def _make_sampler(method: str, ratio, random_state: int,
                  categorical=None, k_neighbors: int = 5, **kw):
    """Instantiate the imbalanced-learn sampler behind a short name."""
    _require("imblearn", "imbalanced-learn")
    from imblearn import over_sampling as ov, under_sampling as un, combine as cb

    m = method.lower()
    if m == "none":
        return None

    over = {
        "ros":        lambda: ov.RandomOverSampler(sampling_strategy=ratio, random_state=random_state, **kw),
        "smote":      lambda: ov.SMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, **kw),
        "borderline": lambda: ov.BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, **kw),
        "svmsmote":   lambda: ov.SVMSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, **kw),
        "kmeans":     lambda: ov.KMeansSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, **kw),
        "adasyn":     lambda: ov.ADASYN(sampling_strategy=ratio, n_neighbors=k_neighbors, random_state=random_state, **kw),
        "smotenc":    lambda: ov.SMOTENC(categorical_features=categorical, sampling_strategy=ratio,
                                         k_neighbors=k_neighbors, random_state=random_state, **kw),
        "smoten":     lambda: ov.SMOTEN(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=random_state, **kw),
    }
    under = {
        "rus":       lambda: un.RandomUnderSampler(sampling_strategy=ratio, random_state=random_state, **kw),
        "tomek":     lambda: un.TomekLinks(**kw),
        "enn":       lambda: un.EditedNearestNeighbours(**kw),
        "ncr":       lambda: un.NeighbourhoodCleaningRule(**kw),
        "oss":       lambda: un.OneSidedSelection(random_state=random_state, **kw),
        "nearmiss":  lambda: un.NearMiss(sampling_strategy=ratio, **kw),
        "centroids": lambda: un.ClusterCentroids(sampling_strategy=ratio, random_state=random_state, **kw),
    }
    hybrid = {
        "smotetomek": lambda: cb.SMOTETomek(sampling_strategy=ratio, random_state=random_state, **kw),
        "smoteenn":   lambda: cb.SMOTEENN(sampling_strategy=ratio, random_state=random_state, **kw),
    }

    for table in (over, under, hybrid):
        if m in table:
            if m == "smotenc" and not categorical:
                raise ValueError(
                    "method='smotenc' requires `categorical=` (column names or indices). "
                    "Plain 'smote' on binary columns produces meaningless values like 0.43."
                )
            return table[m]()

    raise ValueError(
        f"Unknown method '{method}'. Available: {', '.join(sorted(BALANCE_METHODS))}"
    )


def balance(
    df: Frame,
    target: str,
    method: str = "smote",
    ratio: Union[float, str, dict] = "auto",
    categorical: Optional[Sequence[str]] = None,
    k_neighbors: int = 5,
    random_state: int = 42,
    return_report: bool = True,
    _trusted: bool = False,
    **kwargs,
):
    """Resample a **training set** to change its class balance.

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
    """
    X, y = _split_xy(df, target)
    counts_before = y.value_counts().to_dict()
    minority = _minority_label(y)
    prevalence = float((y == minority).mean())

    if not _trusted and len(df) > 0:
        warnings.warn(
            "balance() is for TRAINING data only. If this frame has not been "
            "split yet, stop: resampling before the split leaks data. "
            "Use balance_pipeline() for cross-validation.",
            UserWarning, stacklevel=2,
        )

    cat_idx = None
    if categorical:
        missing = [c for c in categorical if c not in X.columns]
        if missing:
            raise KeyError(f"Categorical columns not in frame: {missing}")
        cat_idx = [X.columns.get_loc(c) for c in categorical]

    _check_numeric_ready(X, method.lower())
    sampler = _make_sampler(method, ratio, random_state, cat_idx, k_neighbors, **kwargs)

    if sampler is None:
        Xr, yr = X, y
    else:
        n_min = int((y == minority).sum())
        if method in ("smote", "borderline", "svmsmote", "kmeans", "adasyn",
                      "smotenc", "smoten", "smotetomek", "smoteenn") and n_min <= k_neighbors:
            raise ValueError(
                f"Minority class has {n_min} rows but k_neighbors={k_neighbors}. "
                f"Lower k_neighbors to at most {max(n_min - 1, 1)}, or use "
                f"method='ros' / class_weights() instead."
            )
        Xr, yr = sampler.fit_resample(X, y)
        Xr = pd.DataFrame(Xr, columns=X.columns)
        yr = pd.Series(yr, name=target)

    out = Xr.copy()
    out[target] = yr.to_numpy()

    if not return_report:
        return out

    counts_after = pd.Series(yr).value_counts().to_dict()
    n_added = max(len(out) - len(X), 0)
    n_min_after = counts_after.get(minority, 0)
    report = {
        "method": method,
        "n_before": len(X),
        "n_after": len(out),
        "counts_before": counts_before,
        "counts_after": counts_after,
        "minority_label": minority,
        "prevalence_before": round(prevalence, 6),
        "prevalence_after": round(float((yr == minority).mean()), 6),
        "rows_added": n_added,
        "pct_minority_synthetic": (round(n_added / n_min_after * 100, 1)
                                   if n_min_after else 0.0),
    }
    return out, report


def list_methods() -> Frame:
    """Return every ``balance`` method with a one-line description."""
    return (pd.DataFrame(
        [{"method": k, "description": v} for k, v in BALANCE_METHODS.items()]
    ).set_index("method"))


def balance_pipeline(method: str, model, ratio="auto", categorical=None,
                     k_neighbors: int = 5, random_state: int = 42, **kwargs):
    """Build a leakage-safe ``imblearn`` Pipeline: resample, then fit.

    This is the correct way to combine resampling with cross-validation or
    a grid search: the sampler runs on the training part of each fold only,
    and is skipped entirely at predict time.

    >>> pipe = dp.balance_pipeline("smote", LGBMClassifier(), ratio=0.3)
    >>> cross_validate(pipe, X, y, cv=5, scoring="average_precision")
    """
    _require("imblearn", "imbalanced-learn")
    from imblearn.pipeline import Pipeline as ImbPipeline

    cat_idx = list(categorical) if categorical else None
    sampler = _make_sampler(method, ratio, random_state, cat_idx, k_neighbors, **kwargs)
    steps = ([] if sampler is None else [("balance", sampler)]) + [("model", model)]
    return ImbPipeline(steps)


def class_weights(y, scheme: Literal["balanced", "sqrt", "custom"] = "balanced",
                  custom: Optional[Dict[Any, float]] = None) -> Dict[str, Any]:
    """Compute cost-sensitive weights -- usually a better first move than SMOTE.

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
    """
    y = pd.Series(y)
    counts = y.value_counts()
    n, k = len(y), len(counts)

    if scheme == "balanced":
        w = {c: n / (k * cnt) for c, cnt in counts.items()}
    elif scheme == "sqrt":
        w = {c: float(np.sqrt(n / (k * cnt))) for c, cnt in counts.items()}
    elif scheme == "custom":
        if not custom:
            raise ValueError("scheme='custom' requires `custom={label: weight}`.")
        w = dict(custom)
    else:
        raise ValueError("scheme must be balanced, sqrt or custom")

    spw = None
    if k == 2:
        minority = counts.idxmin()
        majority = counts.idxmax()
        spw = float(counts[majority] / counts[minority])

    return {
        "class_weight": w,
        "scale_pos_weight": spw,
        "sample_weight": y.map(w).to_numpy(),
        "counts": counts.to_dict(),
    }


def prior_correct(proba, prevalence_train: float, prevalence_true: float):
    """Undo the probability inflation caused by resampling.

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
    """
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
    for name, v in (("prevalence_train", prevalence_train), ("prevalence_true", prevalence_true)):
        if not 0 < v < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {v}.")
    logit = np.log(p / (1 - p))
    shift = (np.log(prevalence_train / (1 - prevalence_train))
             - np.log(prevalence_true / (1 - prevalence_true)))
    return 1.0 / (1.0 + np.exp(-(logit - shift)))


def tune_threshold(
    y_true,
    proba,
    metric: Union[str, callable] = "f1",
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    n_steps: int = 200,
) -> Dict[str, Any]:
    """Find the decision threshold that optimises what you actually care about.

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

    Returns
    -------
    dict with ``threshold``, ``score``, the confusion matrix at that
    threshold, and a ``curve`` DataFrame over all thresholds tried.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    grid = np.linspace(proba.min(), proba.max(), n_steps)

    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0

        if callable(metric):
            score = metric(y_true, pred)
        elif metric == "f1":
            score = f1
        elif metric == "youden":
            score = sens + spec - 1
        elif metric == "balanced_accuracy":
            score = (sens + spec) / 2
        elif metric == "cost":
            score = -(cost_fn * fn + cost_fp * fp)
        else:
            raise ValueError("metric must be f1, youden, balanced_accuracy, cost or a callable")

        rows.append({"threshold": t, "score": score, "sensitivity": sens,
                     "specificity": spec, "precision": prec, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn})

    curve = pd.DataFrame(rows)
    best = curve.loc[curve["score"].idxmax()]
    return {
        "threshold": float(best["threshold"]),
        "score": float(best["score"]),
        "metric": metric if isinstance(metric, str) else "custom",
        "sensitivity": float(best["sensitivity"]),
        "specificity": float(best["specificity"]),
        "precision": float(best["precision"]),
        "confusion": {"tp": int(best["tp"]), "fp": int(best["fp"]),
                      "fn": int(best["fn"]), "tn": int(best["tn"])},
        "curve": curve,
    }


def compare_balance(
    X: Frame,
    y,
    model=None,
    methods: Optional[Sequence[str]] = None,
    ratio: Union[float, str] = 0.3,
    categorical: Optional[Sequence[str]] = None,
    cv: int = 5,
    random_state: int = 42,
    include_weights: bool = True,
    n_jobs: int = -1,
) -> Frame:
    """Benchmark imbalance strategies honestly, with resampling inside each fold.

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

    Returns
    -------
    DataFrame sorted by AUPRC, descending.
    """
    from sklearn.model_selection import cross_validate

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if model is None:
        lgb = _require("lightgbm")
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   random_state=random_state, verbose=-1)
    if methods is None:
        methods = ["none", "ros", "rus", "smote", "borderline",
                   "adasyn", "smotetomek", "smoteenn"]

    cat_idx = [X.columns.get_loc(c) for c in categorical] if categorical else None
    splitter = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    scoring = {"auprc": "average_precision", "auroc": "roc_auc",
               "brier": "neg_brier_score", "bal_acc": "balanced_accuracy"}

    rows = []
    candidates = [(m, m) for m in methods]
    if include_weights:
        candidates.append(("class_weight", None))

    for label, m in candidates:
        try:
            if label == "class_weight":
                mdl = clone(model)
                if hasattr(mdl, "class_weight"):
                    mdl.set_params(class_weight="balanced")
                elif hasattr(mdl, "scale_pos_weight"):
                    mdl.set_params(scale_pos_weight=class_weights(y)["scale_pos_weight"])
                pipe = mdl
            else:
                pipe = balance_pipeline(
                    m, clone(model), ratio=(ratio if m not in ("none",) else "auto"),
                    categorical=cat_idx, random_state=random_state,
                )
            res = cross_validate(pipe, X, y, cv=splitter, scoring=scoring,
                                 n_jobs=n_jobs, error_score="raise")
            row = {"strategy": label}
            for key in scoring:
                v = res[f"test_{key}"]
                sign = -1 if key == "brier" else 1
                row[key] = round(float(sign * v.mean()), 4)
                row[f"{key}_sd"] = round(float(v.std()), 4)
            row["fit_time_s"] = round(float(res["fit_time"].mean()), 2)
            row["status"] = "ok"
        except Exception as exc:  # keep the table complete
            row = {"strategy": label, "status": f"failed: {type(exc).__name__}: {exc}"[:110]}
        rows.append(row)

    out = pd.DataFrame(rows)
    if "auprc" in out.columns:
        out = out.sort_values("auprc", ascending=False, na_position="last")
    return out.reset_index(drop=True)


# ======================================================================
#  5. IO
# ======================================================================

def save_table(
    df: Frame,
    path: str,
    fmt: Optional[Literal["csv", "excel", "parquet", "json"]] = None,
    index: bool = False,
    make_dirs: bool = True,
    **kwargs,
) -> str:
    """Write a DataFrame to disk, inferring the format from the extension.

    Generalises ``save_outliers``: the original hardcoded csv/excel, refused
    to create the directory, and rejected empty frames -- but an empty
    result is often exactly what you want to record ("no outliers found"),
    so here it only warns.

    Returns the absolute path written.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        warnings.warn("Saving an empty DataFrame.", stacklevel=2)

    if fmt is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        fmt = {"csv": "csv", "tsv": "csv", "xlsx": "excel", "xls": "excel",
               "parquet": "parquet", "pq": "parquet", "json": "json"}.get(ext)
        if fmt is None:
            raise ValueError(f"Cannot infer format from '{path}'. Pass fmt= "
                             f"explicitly (csv, excel, parquet, json).")

    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        if make_dirs:
            os.makedirs(directory, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory does not exist: {directory}")

    writer = {"csv": df.to_csv, "excel": df.to_excel,
              "parquet": df.to_parquet, "json": df.to_json}[fmt]
    try:
        if fmt == "json":
            writer(path, orient=kwargs.pop("orient", "records"), **kwargs)
        else:
            writer(path, index=index, **kwargs)
    except ImportError as exc:
        engine = {"parquet": "pyarrow", "excel": "openpyxl"}.get(fmt, fmt)
        raise ImportError(
            f"Writing {fmt} needs an extra package. Install it with: "
            f"pip install {engine}   (or save as .csv instead)"
        ) from exc
    return os.path.abspath(path)


# ======================================================================
#  6. SPLIT
# ======================================================================

def split(
    df: Frame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify: bool = True,
    group: Optional[str] = None,
    random_state: int = 42,
):
    """Split into train/test (and optionally validation), stratified by default.

    Parameters
    ----------
    val_size : float
        Fraction of the *original* data for validation.  ``0`` returns two
        frames, otherwise three (train, val, test).
    group : str, optional
        Grouping column (patient id, hospital, ...).  Rows sharing a group
        are kept together, so the same patient cannot appear on both sides
        of the split.  Stratification is then approximate.
    """
    y = df[target]
    strat = y if (stratify and _is_binary_target(y) or (stratify and y.nunique() <= 20)) else None

    if group is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(gss.split(df, y, groups=df[group]))
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        if val_size <= 0:
            return train.copy(), test.copy()
        rel = val_size / (1 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel, random_state=random_state)
        tr2, va = next(gss2.split(train, train[target], groups=train[group]))
        return train.iloc[tr2].copy(), train.iloc[va].copy(), test.copy()

    train, test = train_test_split(df, test_size=test_size,
                                   stratify=strat, random_state=random_state)
    if val_size <= 0:
        return train.copy(), test.copy()

    rel = val_size / (1 - test_size)
    s2 = train[target] if strat is not None else None
    train, val = train_test_split(train, test_size=rel, stratify=s2,
                                  random_state=random_state)
    return train.copy(), val.copy(), test.copy()


__all__ = [
    # inspect
    "overview", "scan_missing", "scan_duplicates", "scan_outliers",
    "check_numeric", "scan_leakage", "column_types",
    "scan_outliers_multivariate",
    # clean
    "clean_names", "convert", "drop_missing", "drop_constant",
    "fix_missing", "fix_duplicates", "fix_outliers", "fix_outliers_multivariate",
    "group_rare", "apply_state",
    # features
    "select_features", "drop_correlated",
    # balance
    "balance", "balance_pipeline", "class_weights", "prior_correct",
    "tune_threshold", "compare_balance", "list_methods", "BALANCE_METHODS",
    "NUMERIC_ONLY_METHODS",
    # io
    "save_table",
    # split
    "split",
]

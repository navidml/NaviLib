"""
features
~~~~~~~~

Feature engineering: numeric transforms, scaling, encoding, binning, and
derived features (datetime, cyclical, interactions, group aggregates, text).

Companion to ``datakit`` (cleaning + imbalance), ``eda`` (exploration) and
``evaluate`` (metrics).

The one rule this module exists to enforce
------------------------------------------
**Every fitted quantity is captured in a ``state`` and replayable.**

A Box-Cox lambda, a scaler's mean, a label mapping, a target-encoding
table -- all of these are *learned from the training set*.  If they are
recomputed on the test set, the two sets are on different scales and the
model silently degrades; if they are learned from the full data before
splitting, the test score is inflated.  Every function here returns
``(df, state)`` on request, and :func:`apply_state` replays the identical
transformation on new data.

>>> import features as fe
>>> train, s1 = fe.transform_numeric(train, "income", method="auto", return_state=True)
>>> train, s2 = fe.encode(train, ["city"], method="target", target="y", return_state=True)
>>> train, s3 = fe.scale(train, method="standard", exclude=["y"], return_state=True)
>>> test = fe.apply_state(test, [s1, s2, s3])          # exactly the same maths
>>> fe.summary([s1, s2, s3])                            # what did I actually do?

Author: rebuilt from navdata
License: MIT
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__version__ = "2.0.0"

Frame = pd.DataFrame
Series = pd.Series

TRANSFORMS = ["none", "log", "log1p", "sqrt", "cuberoot", "reciprocal",
              "boxcox", "yeojohnson", "quantile_normal", "rank"]
SCALERS = ["standard", "minmax", "robust", "maxabs", "l2", "none"]
ENCODERS = ["onehot", "ordinal", "count", "frequency", "target", "woe",
            "hashing", "binary"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _cols(df: Frame, columns, numeric_only: bool = False,
          exclude: Optional[Sequence[str]] = None) -> List[str]:
    if columns is None:
        out = list(df.columns)
    elif isinstance(columns, str):
        out = [columns]
    else:
        out = list(columns)
    missing = [c for c in out if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if numeric_only:
        out = [c for c in out
               if pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
    if exclude:
        out = [c for c in out if c not in set(exclude)]
    return out


def _new_name(col: str, suffix: str, replace: bool) -> str:
    return col if replace else f"{col}_{suffix}"


def _check_no_nan(s: Series, col: str, what: str) -> None:
    if s.isna().any():
        raise ValueError(
            f"{what} cannot handle the {int(s.isna().sum())} missing value(s) in "
            f"'{col}'. Impute first (datakit.fix_missing) or pass "
            f"skip_missing=True to leave them as NaN."
        )


# ======================================================================
#  1. NUMERIC TRANSFORMS
# ======================================================================

def _fit_transform_one(
    x: Series, method: str, standardize: bool = False, random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit one transform, returning values and everything needed to replay it."""
    from scipy.stats import boxcox
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer

    ok = x.notna()
    v = x[ok].to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    params: Dict[str, Any] = {"method": method, "standardize": standardize}

    if method == "none":
        out[ok.to_numpy()] = v

    elif method in ("log", "log1p"):
        shift = 0.0
        if method == "log":
            if (v <= 0).any():
                raise ValueError("log requires strictly positive values; "
                                 "use 'log1p' for data containing zeros, or "
                                 "'yeojohnson' for data containing negatives.")
            out[ok.to_numpy()] = np.log(v)
        else:
            if (v < -1).any():
                shift = float(-v.min() - 1 + 1e-9)
                warnings.warn(f"log1p: shifting by {shift:.6g} to make all values > -1.",
                              stacklevel=3)
            out[ok.to_numpy()] = np.log1p(v + shift)
        params["shift"] = shift

    elif method == "sqrt":
        shift = float(max(0.0, -v.min()))
        out[ok.to_numpy()] = np.sqrt(v + shift)
        params["shift"] = shift

    elif method == "cuberoot":
        out[ok.to_numpy()] = np.cbrt(v)

    elif method == "reciprocal":
        if (v == 0).any():
            raise ValueError("reciprocal cannot handle zeros.")
        out[ok.to_numpy()] = 1.0 / v

    elif method == "boxcox":
        if (v <= 0).any():
            raise ValueError("Box-Cox requires strictly positive values; "
                             "use 'yeojohnson' instead.")
        t, lam = boxcox(v)
        out[ok.to_numpy()] = t
        params["lambda"] = float(lam)

    elif method == "yeojohnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=standardize)
        out[ok.to_numpy()] = pt.fit_transform(v.reshape(-1, 1)).ravel()
        params["transformer"] = pt
        params["lambda"] = float(pt.lambdas_[0])

    elif method == "quantile_normal":
        n_q = min(1000, max(10, len(v)))
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=n_q,
                                 random_state=random_state)
        out[ok.to_numpy()] = qt.fit_transform(v.reshape(-1, 1)).ravel()
        params["transformer"] = qt

    elif method == "rank":
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, max(10, len(v))),
                                 random_state=random_state)
        out[ok.to_numpy()] = qt.fit_transform(v.reshape(-1, 1)).ravel()
        params["transformer"] = qt

    else:
        raise ValueError(f"Unknown transform '{method}'. Choose from {TRANSFORMS}.")

    return out, params


def _apply_transform_one(x: Series, params: Dict[str, Any]) -> np.ndarray:
    """Replay a fitted transform on new data."""
    method = params["method"]
    ok = x.notna()
    v = x[ok].to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    m = ok.to_numpy()

    if method == "none":
        out[m] = v
    elif method == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            out[m] = np.log(np.where(v > 0, v, np.nan))
    elif method == "log1p":
        s = params.get("shift", 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[m] = np.log1p(np.where(v + s > -1, v + s, np.nan))
    elif method == "sqrt":
        s = params.get("shift", 0.0)
        out[m] = np.sqrt(np.clip(v + s, 0, None))
    elif method == "cuberoot":
        out[m] = np.cbrt(v)
    elif method == "reciprocal":
        out[m] = np.where(v != 0, 1.0 / np.where(v == 0, 1, v), np.nan)
    elif method == "boxcox":
        lam = params["lambda"]
        vv = np.where(v > 0, v, np.nan)
        out[m] = np.log(vv) if abs(lam) < 1e-12 else (np.power(vv, lam) - 1) / lam
    elif method in ("yeojohnson", "quantile_normal", "rank"):
        out[m] = params["transformer"].transform(v.reshape(-1, 1)).ravel()
    else:
        raise ValueError(f"Unknown transform '{method}'.")
    return out


def transform_numeric(
    df: Frame,
    columns=None,
    method: Union[str, Literal["auto"]] = "auto",
    candidates: Optional[Sequence[str]] = None,
    replace: bool = False,
    suffix: Optional[str] = None,
    standardize: bool = False,
    min_skew: float = 0.5,
    return_state: bool = False,
    random_state: int = 42,
    verbose: bool = False,
):
    """Apply a variance-stabilising transform, or pick the best one automatically.

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

    Returns
    -------
    DataFrame, or ``(DataFrame, state)``.  The state's ``report`` field
    holds the before/after skew and, for ``auto``, every candidate tried.
    """
    from scipy.stats import skew as _skew

    cols = _cols(df, columns, numeric_only=True)
    if not cols:
        raise ValueError("No numeric columns selected.")
    out = df.copy()
    entries: Dict[str, Dict[str, Any]] = {}
    report_rows = []

    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        d = x.dropna()
        if len(d) < 3 or d.nunique() < 3:
            report_rows.append({"column": c, "chosen": "none",
                                "reason": "too few distinct values"})
            continue
        skew_before = float(_skew(d))

        if method == "auto":
            if abs(skew_before) < min_skew:
                report_rows.append({"column": c, "skew_before": round(skew_before, 4),
                                    "chosen": "none", "skew_after": round(skew_before, 4),
                                    "reason": f"|skew| < {min_skew}, left alone"})
                continue
            cand = list(candidates) if candidates else \
                ["log1p", "sqrt", "cuberoot", "boxcox", "yeojohnson"]
            scored: Dict[str, Tuple[float, np.ndarray, Dict]] = {}
            for m in cand:
                try:
                    vals, params = _fit_transform_one(x, m, standardize, random_state)
                except (ValueError, Exception):
                    continue
                fin = vals[np.isfinite(vals)]
                if len(fin) > 8:
                    scored[m] = (abs(float(_skew(fin))), vals, params)
            if not scored:
                report_rows.append({"column": c, "skew_before": round(skew_before, 4),
                                    "chosen": "none", "reason": "no feasible transform"})
                continue
            best = min(scored, key=lambda k: scored[k][0])
            if scored[best][0] >= abs(skew_before):
                report_rows.append({"column": c, "skew_before": round(skew_before, 4),
                                    "chosen": "none", "skew_after": round(skew_before, 4),
                                    "reason": "nothing improved on the original"})
                continue
            chosen, (sk_after, vals, params) = best, scored[best]
            tried = {k: round(v[0], 4) for k, v in sorted(scored.items(),
                                                          key=lambda kv: kv[1][0])}
        else:
            vals, params = _fit_transform_one(x, method, standardize, random_state)
            fin = vals[np.isfinite(vals)]
            sk_after = abs(float(_skew(fin))) if len(fin) > 8 else np.nan
            chosen, tried = method, None

        name = _new_name(c, suffix or chosen, replace)
        out[name] = vals
        entries[c] = {"target_column": name, "params": params}
        row = {"column": c, "new_column": name,
               "skew_before": round(skew_before, 4), "chosen": chosen,
               "skew_after": round(float(sk_after), 4) if np.isfinite(sk_after) else np.nan}
        if tried:
            row["candidates"] = tried
        report_rows.append(row)
        if verbose:
            print(f"  {c}: skew {skew_before:+.3f} -> {sk_after:+.3f} via {chosen}")

    report = pd.DataFrame(report_rows)
    state = {"kind": "transform", "entries": entries, "replace": replace,
             "report": report}
    return (out, state) if return_state else out


# ======================================================================
#  2. SCALING
# ======================================================================

def scale(
    df: Frame,
    columns=None,
    method: Literal["standard", "minmax", "robust", "maxabs", "l2", "none"] = "standard",
    exclude: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
    feature_range: Tuple[float, float] = (0, 1),
    quantile_range: Tuple[float, float] = (25.0, 75.0),
    replace: bool = True,
    suffix: str = "scaled",
    return_state: bool = False,
):
    """Scale numeric columns, with the target automatically kept out.

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
    """
    from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                       RobustScaler, StandardScaler)

    ex = set(exclude or [])
    if target:
        ex.add(target)
    cols = _cols(df, columns, numeric_only=True, exclude=ex)
    skipped = [c for c in cols
               if df[c].dropna().nunique() <= 2
               and set(pd.unique(df[c].dropna())) <= {0, 1, True, False}]
    cols = [c for c in cols if c not in skipped]
    if not cols:
        raise ValueError("No numeric columns left to scale after exclusions.")

    out = df.copy()
    if method == "none":
        state = {"kind": "scale", "method": "none", "columns": cols}
        return (out, state) if return_state else out

    makers = {
        "standard": lambda: StandardScaler(),
        "minmax": lambda: MinMaxScaler(feature_range=feature_range),
        "robust": lambda: RobustScaler(quantile_range=quantile_range),
        "maxabs": lambda: MaxAbsScaler(),
        "l2": lambda: Normalizer(norm="l2"),
    }
    if method not in makers:
        raise ValueError(f"method must be one of {SCALERS}, got '{method}'.")

    scaler = makers[method]()
    block = out[cols].to_numpy(dtype=float)
    mask = np.isnan(block)
    if mask.any() and method != "l2":
        filled = np.where(mask, np.nanmedian(block, axis=0), block)
        scaler.fit(filled)
        scaled = scaler.transform(np.where(mask, np.nanmedian(block, axis=0), block))
        scaled[mask] = np.nan
    else:
        scaled = scaler.fit_transform(np.nan_to_num(block) if mask.any() else block)

    names = cols if replace else [f"{c}_{suffix}" for c in cols]
    for i, n in enumerate(names):
        out[n] = scaled[:, i]

    state = {"kind": "scale", "method": method, "columns": cols,
             "names": names, "scaler": scaler, "skipped_binary": skipped}
    if skipped:
        state["note"] = f"skipped binary/indicator columns: {skipped}"
    return (out, state) if return_state else out


# ======================================================================
#  3. BINNING
# ======================================================================

def bin_numeric(
    df: Frame,
    columns=None,
    method: Literal["quantile", "uniform", "kmeans", "custom", "tree"] = "quantile",
    bins: Union[int, Sequence[float]] = 5,
    labels: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
    as_category: bool = True,
    replace: bool = False,
    suffix: str = "bin",
    return_state: bool = False,
    random_state: int = 42,
):
    """Discretise numeric columns into bins.

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
    """
    cols = _cols(df, columns, numeric_only=True, exclude=[target] if target else None)
    if not cols:
        raise ValueError("No numeric columns selected.")
    if method == "tree" and target is None:
        raise ValueError("method='tree' is supervised and needs target=.")

    out = df.copy()
    entries: Dict[str, Dict[str, Any]] = {}

    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        d = x.dropna()
        if d.nunique() < 2:
            continue

        if method == "custom":
            edges = list(bins)
        elif method == "quantile":
            edges = list(np.unique(np.nanquantile(d, np.linspace(0, 1, int(bins) + 1))))
        elif method == "uniform":
            edges = list(np.linspace(d.min(), d.max(), int(bins) + 1))
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=int(bins), n_init=10, random_state=random_state)
            km.fit(d.to_numpy().reshape(-1, 1))
            centres = np.sort(km.cluster_centers_.ravel())
            edges = [d.min()] + list((centres[:-1] + centres[1:]) / 2) + [d.max()]
        elif method == "tree":
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            y = out.loc[x.notna(), target]
            clf = (DecisionTreeClassifier if y.nunique() <= 20
                   else DecisionTreeRegressor)(
                max_leaf_nodes=int(bins), min_samples_leaf=max(20, len(d) // 100),
                random_state=random_state)
            clf.fit(d.to_numpy().reshape(-1, 1), y)
            thr = sorted(clf.tree_.threshold[clf.tree_.feature >= 0])
            edges = [d.min()] + list(thr) + [d.max()]
        else:
            raise ValueError("method must be quantile, uniform, kmeans, tree or custom.")

        edges = list(np.unique(edges))
        if len(edges) < 3:
            warnings.warn(f"'{c}': could not form more than one bin; skipped.",
                          stacklevel=2)
            continue
        open_edges = [-np.inf] + edges[1:-1] + [np.inf]
        lab = list(labels) if labels is not None else None
        if lab is not None and len(lab) != len(open_edges) - 1:
            raise ValueError(f"labels must have {len(open_edges) - 1} entries for "
                             f"'{c}', got {len(lab)}.")

        binned = pd.cut(x, bins=open_edges, labels=lab, include_lowest=True)
        name = _new_name(c, suffix, replace)
        out[name] = binned if as_category else binned.cat.codes.replace(-1, np.nan)
        entries[c] = {"target_column": name, "edges": open_edges,
                      "labels": lab, "as_category": as_category}

    state = {"kind": "bin", "entries": entries, "method": method}
    return (out, state) if return_state else out


# ======================================================================
#  4. CATEGORICAL ENCODING
# ======================================================================

def encode(
    df: Frame,
    columns=None,
    method: Literal["onehot", "ordinal", "count", "frequency",
                    "target", "woe", "hashing", "binary"] = "onehot",
    target: Optional[str] = None,
    order: Optional[Dict[str, Sequence]] = None,
    min_freq: Optional[Union[int, float]] = None,
    other_label: str = "__other__",
    drop_first: bool = False,
    dummy_na: bool = True,
    separator: Optional[str] = None,
    n_components: int = 8,
    cv: int = 5,
    smoothing: float = 20.0,
    drop_original: bool = True,
    return_state: bool = False,
    random_state: int = 42,
):
    """Encode categorical columns, leakage-safe and replayable.

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

    Returns
    -------
    DataFrame, or ``(DataFrame, state)``.
    """
    cols = _cols(df, columns, exclude=[target] if target else None)
    if not cols:
        raise ValueError("No columns selected for encoding.")
    if method in ("target", "woe") and target is None:
        raise ValueError(f"method='{method}' is supervised and needs target=.")
    if method not in ENCODERS:
        raise ValueError(f"method must be one of {ENCODERS}, got '{method}'.")

    out = df.copy()
    entries: Dict[str, Dict[str, Any]] = {}
    y = out[target] if target else None

    for c in cols:
        s = out[c]
        na_mask = s.isna()
        sv = s.astype("object").where(~na_mask, np.nan)

        # ---- optional rare-level merge (fitted on train) --------------
        keep_levels = None
        if min_freq is not None:
            vc = sv.value_counts(dropna=True)
            thresh = min_freq if isinstance(min_freq, (int, np.integer)) \
                else len(sv) * float(min_freq)
            keep_levels = vc[vc >= thresh].index.tolist()
            sv = sv.where(sv.isin(keep_levels) | sv.isna(), other_label)

        entry: Dict[str, Any] = {"method": method, "keep_levels": keep_levels,
                                 "other_label": other_label, "separator": separator,
                                 "dummy_na": dummy_na, "source": c}

        # ---------------- ONE-HOT ----------------
        if method == "onehot":
            tokens = _tokenise(sv, separator)
            levels = sorted({t for row in tokens for t in row})
            if drop_first and levels:
                levels = levels[1:]
            mat = np.zeros((len(sv), len(levels)), dtype=np.int8)
            pos = {lv: i for i, lv in enumerate(levels)}
            for r, row in enumerate(tokens):
                for t in row:
                    if t in pos:
                        mat[r, pos[t]] = 1
            names = [f"{c}_{lv}" for lv in levels]
            for i, n in enumerate(names):
                out[n] = mat[:, i]
            if dummy_na and na_mask.any():
                out[f"{c}_nan"] = na_mask.astype(np.int8)
                names.append(f"{c}_nan")
            entry.update(levels=levels, names=names, drop_first=drop_first)

        # ---------------- ORDINAL / BINARY ----------------
        elif method in ("ordinal", "binary"):
            if order and c in order:
                levels = list(order[c])
                unseen = set(sv.dropna().unique()) - set(levels)
                if unseen:
                    warnings.warn(f"'{c}': values not in the given order will become "
                                  f"NaN: {sorted(unseen, key=str)[:5]}", stacklevel=2)
            else:
                levels = sorted(sv.dropna().unique(), key=str)
                if method == "ordinal" and len(levels) > 2:
                    warnings.warn(
                        f"'{c}': ordinal encoding assigns 0,1,2,... which implies an "
                        f"ordering and equal spacing. If these categories are nominal, "
                        f"use method='onehot' or 'target'. Pass order={{'{c}': [...]}} "
                        f"to silence this.", stacklevel=2)
            mapping = {lv: i for i, lv in enumerate(levels)}
            codes = sv.map(mapping)
            if method == "ordinal":
                name = f"{c}_code"
                out[name] = codes.astype("Float64")
                entry.update(mapping=mapping, names=[name])
            else:
                width = max(1, int(np.ceil(np.log2(max(len(levels), 2)))))
                names = [f"{c}_b{i}" for i in range(width)]
                filled = codes.fillna(-1).astype(int).to_numpy()
                for i, n in enumerate(names):
                    bit = (filled >> i) & 1
                    out[n] = np.where(filled < 0, np.nan, bit)
                entry.update(mapping=mapping, names=names, width=width)

        # ---------------- COUNT / FREQUENCY ----------------
        elif method in ("count", "frequency"):
            tokens = _tokenise(sv, separator)
            flat = pd.Series([t for row in tokens for t in row], dtype="object")
            stats = flat.value_counts()
            if method == "frequency":
                stats = stats / max(len(sv), 1)
            lookup = stats.to_dict()
            vals = np.array([sum(lookup.get(t, 0.0) for t in row) if row else np.nan
                             for row in tokens], dtype=float)
            name = f"{c}_{method}"
            out[name] = vals
            entry.update(mapping=lookup, names=[name], default=0.0)

        # ---------------- TARGET ----------------
        elif method == "target":
            if y is None:
                raise ValueError("target encoding needs target=.")
            oof, mapping, prior = _target_encode(sv, y, cv, smoothing, random_state)
            name = f"{c}_te"
            out[name] = oof
            entry.update(mapping=mapping, prior=float(prior), names=[name],
                         smoothing=smoothing)

        # ---------------- WOE ----------------
        elif method == "woe":
            yb = _binary_target(y)
            oof, mapping, prior = _woe_encode(sv, yb, cv, smoothing, random_state)
            name = f"{c}_woe"
            out[name] = oof
            entry.update(mapping=mapping, prior=float(prior), names=[name])

        # ---------------- HASHING ----------------
        elif method == "hashing":
            names = [f"{c}_h{i}" for i in range(n_components)]
            mat = _hash_matrix(sv, n_components)
            for i, n in enumerate(names):
                out[n] = mat[:, i]
            entry.update(names=names, n_components=n_components)

        entries[c] = entry

    if drop_original:
        out = out.drop(columns=[c for c in cols if c in out.columns])

    state = {"kind": "encode", "entries": entries, "drop_original": drop_original}
    return (out, state) if return_state else out


def _tokenise(s: Series, separator: Optional[str]) -> List[List[str]]:
    """Split multi-label cells; single-label cells become one-element lists."""
    if separator is None:
        return [[] if pd.isna(v) else [str(v)] for v in s]
    out = []
    for v in s:
        if pd.isna(v):
            out.append([])
        else:
            out.append([t.strip() for t in str(v).split(separator) if t.strip()])
    return out


def _binary_target(y: Series) -> np.ndarray:
    u = pd.Series(y).dropna().unique()
    if len(u) != 2:
        raise ValueError(f"WOE encoding needs a binary target; found {len(u)} classes.")
    pos = pd.Series(y).value_counts().idxmin()
    return (pd.Series(y) == pos).astype(int).to_numpy()


def _target_encode(s: Series, y: Series, cv: int, smoothing: float,
                   random_state: int) -> Tuple[np.ndarray, Dict, float]:
    """Out-of-fold smoothed mean-target encoding."""
    from sklearn.model_selection import KFold, StratifiedKFold

    yv = pd.Series(y).to_numpy()
    prior = float(np.nanmean(pd.to_numeric(yv, errors="coerce")))
    n = len(s)
    oof = np.full(n, prior)

    stratifiable = pd.Series(yv).nunique() <= 20 and \
        pd.Series(yv).value_counts().min() >= cv
    splitter = (StratifiedKFold(cv, shuffle=True, random_state=random_state)
                if stratifiable else KFold(cv, shuffle=True, random_state=random_state))
    idx = np.arange(n)

    for tr, te in splitter.split(idx, yv if stratifiable else None):
        g = pd.DataFrame({"k": s.iloc[tr].to_numpy(),
                          "y": pd.to_numeric(yv[tr], errors="coerce")})
        agg = g.groupby("k", dropna=True)["y"].agg(["mean", "count"])
        fold_prior = float(g["y"].mean())
        sm = (agg["mean"] * agg["count"] + fold_prior * smoothing) / \
             (agg["count"] + smoothing)
        oof[te] = s.iloc[te].map(sm.to_dict()).fillna(fold_prior).to_numpy()

    # full-data map, for use at transform time on unseen rows
    g = pd.DataFrame({"k": s.to_numpy(), "y": pd.to_numeric(yv, errors="coerce")})
    agg = g.groupby("k", dropna=True)["y"].agg(["mean", "count"])
    full = ((agg["mean"] * agg["count"] + prior * smoothing) /
            (agg["count"] + smoothing)).to_dict()
    return oof, full, prior


def _woe_encode(s: Series, yb: np.ndarray, cv: int, smoothing: float,
                random_state: int) -> Tuple[np.ndarray, Dict, float]:
    """Out-of-fold weight-of-evidence encoding."""
    from sklearn.model_selection import StratifiedKFold

    n = len(s)
    oof = np.zeros(n)
    splitter = StratifiedKFold(min(cv, int(min(np.bincount(yb)))) or 2,
                               shuffle=True, random_state=random_state)

    def _fit(keys, yy):
        d = pd.DataFrame({"k": keys, "y": yy})
        tot_pos = max(d["y"].sum(), 1)
        tot_neg = max((1 - d["y"]).sum(), 1)
        g = d.groupby("k", dropna=True)["y"].agg(["sum", "count"])
        pos = (g["sum"] + smoothing * tot_pos / (tot_pos + tot_neg)) / (tot_pos + smoothing)
        neg = ((g["count"] - g["sum"]) + smoothing * tot_neg / (tot_pos + tot_neg)) / \
              (tot_neg + smoothing)
        return np.log(pos / neg).to_dict()

    try:
        for tr, te in splitter.split(np.arange(n), yb):
            m = _fit(s.iloc[tr].to_numpy(), yb[tr])
            oof[te] = s.iloc[te].map(m).fillna(0.0).to_numpy()
    except ValueError:
        oof = s.map(_fit(s.to_numpy(), yb)).fillna(0.0).to_numpy()
    return oof, _fit(s.to_numpy(), yb), 0.0


def _hash_matrix(s: Series, n_components: int) -> np.ndarray:
    """Deterministic signed hashing -- stable across processes, unlike hash()."""
    mat = np.zeros((len(s), n_components), dtype=np.int8)
    for i, v in enumerate(s):
        if pd.isna(v):
            continue
        h = hashlib.md5(str(v).encode()).digest()
        bucket = int.from_bytes(h[:4], "big") % n_components
        sign = 1 if h[4] % 2 == 0 else -1
        mat[i, bucket] = sign
    return mat


# ======================================================================
#  5. DERIVED FEATURES
# ======================================================================

def add_datetime(
    df: Frame,
    columns=None,
    parts: Sequence[str] = ("year", "month", "day", "dayofweek", "hour",
                            "quarter", "is_weekend", "is_month_end"),
    cyclical: bool = True,
    reference: Optional[Union[str, pd.Timestamp]] = None,
    drop_original: bool = False,
    return_state: bool = False,
):
    """Explode datetime columns into modelling-ready parts.

    ``cyclical=True`` additionally emits sine/cosine pairs for month, day of
    week and hour.  This matters more than it looks: as a raw integer,
    December (12) and January (1) are eleven units apart, and 23:00 and
    01:00 are twenty-two -- the model has to spend capacity learning that
    the scale wraps.  The sin/cos pair encodes the wrap directly.

    ``reference`` adds a ``<col>_days_since`` column measured from a fixed
    date or another datetime column, which is usually the feature that
    actually carries signal (account age, time since last visit).
    """
    cols = _cols(df, columns)
    dt_cols = []
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_cols.append(c)
        elif df[c].dtype == object:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                dt_cols.append(c)
    if not dt_cols:
        raise ValueError("No datetime-like columns found. Convert with "
                         "datakit.convert(df, col, to='datetime') first.")

    out = df.copy()
    created: Dict[str, List[str]] = {}
    for c in dt_cols:
        d = pd.to_datetime(out[c], errors="coerce")
        made = []
        emit = {
            "year": lambda: d.dt.year, "month": lambda: d.dt.month,
            "day": lambda: d.dt.day, "dayofweek": lambda: d.dt.dayofweek,
            "dayofyear": lambda: d.dt.dayofyear, "hour": lambda: d.dt.hour,
            "minute": lambda: d.dt.minute, "week": lambda: d.dt.isocalendar().week.astype("Float64"),
            "quarter": lambda: d.dt.quarter,
            "is_weekend": lambda: (d.dt.dayofweek >= 5).astype("Int8"),
            "is_month_end": lambda: d.dt.is_month_end.astype("Int8"),
            "is_month_start": lambda: d.dt.is_month_start.astype("Int8"),
        }
        for p in parts:
            if p not in emit:
                warnings.warn(f"Unknown datetime part '{p}', skipped.", stacklevel=2)
                continue
            name = f"{c}_{p}"
            out[name] = emit[p]()
            made.append(name)

        if cyclical:
            for p, period in (("month", 12), ("dayofweek", 7), ("hour", 24)):
                if p in parts:
                    v = emit[p]().astype(float)
                    base = 1 if p == "month" else 0
                    for fn, tag in ((np.sin, "sin"), (np.cos, "cos")):
                        name = f"{c}_{p}_{tag}"
                        out[name] = fn(2 * np.pi * (v - base) / period)
                        made.append(name)

        if reference is not None:
            ref = pd.to_datetime(out[reference], errors="coerce") \
                if isinstance(reference, str) and reference in out.columns \
                else pd.Timestamp(reference)
            name = f"{c}_days_since"
            out[name] = (d - ref).dt.total_seconds() / 86400
            made.append(name)
        created[c] = made

    if drop_original:
        out = out.drop(columns=dt_cols)
    state = {"kind": "datetime", "columns": dt_cols, "parts": list(parts),
             "cyclical": cyclical, "reference": reference,
             "drop_original": drop_original, "created": created}
    return (out, state) if return_state else out


def add_cyclical(df: Frame, column: str, period: float,
                 drop_original: bool = False, return_state: bool = False):
    """Encode any periodic numeric column as a sin/cos pair.

    For angles (``period=360``), compass bearings, day-of-year
    (``period=365.25``), or anything else where the largest value is
    adjacent to the smallest.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")
    if period <= 0:
        raise ValueError("period must be positive.")
    out = df.copy()
    v = pd.to_numeric(out[column], errors="coerce").astype(float)
    out[f"{column}_sin"] = np.sin(2 * np.pi * v / period)
    out[f"{column}_cos"] = np.cos(2 * np.pi * v / period)
    if drop_original:
        out = out.drop(columns=[column])
    state = {"kind": "cyclical", "column": column, "period": period,
             "drop_original": drop_original}
    return (out, state) if return_state else out


def add_interactions(
    df: Frame,
    columns=None,
    degree: int = 2,
    operations: Sequence[str] = ("multiply",),
    max_features: int = 200,
    target: Optional[str] = None,
    return_state: bool = False,
):
    """Build pairwise interaction and ratio features.

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
    """
    from itertools import combinations

    cols = _cols(df, columns, numeric_only=True,
                 exclude=[target] if target else None)
    if len(cols) < 2:
        raise ValueError("Need at least 2 numeric columns.")
    if degree != 2:
        raise ValueError("Only degree=2 (pairwise) is supported; higher degrees "
                         "explode combinatorially. Select columns and call twice.")

    valid = {"multiply", "divide", "add", "subtract"}
    bad = set(operations) - valid
    if bad:
        raise ValueError(f"Unknown operations: {bad}. Choose from {valid}.")

    out = df.copy()
    pairs = list(combinations(cols, 2))
    n_planned = len(pairs) * sum(2 if op == "divide" else 1 for op in operations)
    made: List[str] = []

    for a, b in pairs:
        if len(made) >= max_features:
            break
        va = pd.to_numeric(out[a], errors="coerce")
        vb = pd.to_numeric(out[b], errors="coerce")
        for op in operations:
            if op == "multiply":
                out[f"{a}_x_{b}"] = va * vb
                made.append(f"{a}_x_{b}")
            elif op == "add":
                out[f"{a}_plus_{b}"] = va + vb
                made.append(f"{a}_plus_{b}")
            elif op == "subtract":
                out[f"{a}_minus_{b}"] = va - vb
                made.append(f"{a}_minus_{b}")
            elif op == "divide":
                out[f"{a}_div_{b}"] = np.where(vb != 0, va / vb.replace(0, np.nan), np.nan)
                out[f"{b}_div_{a}"] = np.where(va != 0, vb / va.replace(0, np.nan), np.nan)
                made += [f"{a}_div_{b}", f"{b}_div_{a}"]

    if n_planned > max_features:
        warnings.warn(f"{n_planned} interactions requested, capped at {max_features}. "
                      f"Narrow `columns=` or raise `max_features=`.", stacklevel=2)
    state = {"kind": "interactions", "columns": cols, "operations": list(operations),
             "max_features": max_features, "created": made}
    return (out, state) if return_state else out


def add_aggregates(
    df: Frame,
    group: Union[str, Sequence[str]],
    values: Union[str, Sequence[str]],
    funcs: Sequence[str] = ("mean", "std", "min", "max", "count"),
    add_deviation: bool = True,
    return_state: bool = False,
):
    """Group-level statistics joined back to every row.

    "How does this patient's birth weight compare to the average in their
    hospital?" is often more predictive than the raw value.  With
    ``add_deviation`` you also get ``<value>_dev_<group>`` -- the row's
    distance from its group mean in group standard deviations, which is the
    feature that usually carries the signal.

    The fitted group table is stored in the state, so at transform time a
    test row belonging to an unseen group gets the global statistic instead
    of NaN.
    """
    groups = [group] if isinstance(group, str) else list(group)
    vals = [values] if isinstance(values, str) else list(values)
    for c in groups + vals:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found.")

    out = df.copy()
    tables: Dict[str, Frame] = {}
    made: List[str] = []
    gkey = "__".join(groups)

    for v in vals:
        agg = out.groupby(groups, observed=True)[v].agg(list(funcs))
        agg.columns = [f"{v}_{f}_by_{gkey}" for f in funcs]
        tables[v] = agg
        merged = out[groups].merge(agg, left_on=groups, right_index=True, how="left")
        for col in agg.columns:
            out[col] = merged[col].to_numpy()
            made.append(col)

        if add_deviation and "mean" in funcs:
            mcol, scol = f"{v}_mean_by_{gkey}", f"{v}_std_by_{gkey}"
            if scol in out.columns:
                denom = out[scol].replace(0, np.nan)
                out[f"{v}_dev_by_{gkey}"] = (out[v] - out[mcol]) / denom
            else:
                out[f"{v}_dev_by_{gkey}"] = out[v] - out[mcol]
            made.append(f"{v}_dev_by_{gkey}")

    globals_ = {v: {f: float(pd.to_numeric(out[v], errors="coerce").agg(f))
                    for f in funcs if f != "count"} for v in vals}
    state = {"kind": "aggregates", "groups": groups, "values": vals,
             "funcs": list(funcs), "tables": tables, "globals": globals_,
             "add_deviation": add_deviation, "created": made, "gkey": gkey}
    return (out, state) if return_state else out


def add_text_features(
    df: Frame,
    columns=None,
    parts: Sequence[str] = ("length", "n_words", "n_digits", "n_upper",
                            "n_special", "avg_word_len"),
    drop_original: bool = False,
    return_state: bool = False,
):
    """Cheap structural features from free-text columns.

    Not a substitute for embeddings, but these six often carry surprising
    signal in tabular problems -- message length, digit density and
    capitalisation are classic spam and fraud indicators, and they cost
    nothing to compute.
    """
    cols = [c for c in _cols(df, columns) if not pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No text-like columns found.")
    out = df.copy()
    made: List[str] = []

    for c in cols:
        s = out[c].astype("string")
        emit = {
            "length": lambda: s.str.len(),
            "n_words": lambda: s.str.split().str.len(),
            "n_digits": lambda: s.str.count(r"\d"),
            "n_upper": lambda: s.str.count(r"[A-Z]"),
            "n_special": lambda: s.str.count(r"[^\w\s]"),
            "avg_word_len": lambda: (s.str.replace(r"\s+", "", regex=True).str.len()
                                     / s.str.split().str.len().replace(0, np.nan)),
            "n_unique_chars": lambda: s.map(lambda v: len(set(str(v)))
                                            if pd.notna(v) else np.nan),
        }
        for p in parts:
            if p not in emit:
                warnings.warn(f"Unknown text part '{p}', skipped.", stacklevel=2)
                continue
            name = f"{c}_{p}"
            out[name] = pd.to_numeric(emit[p](), errors="coerce")
            made.append(name)

    if drop_original:
        out = out.drop(columns=cols)
    state = {"kind": "text", "columns": cols, "parts": list(parts),
             "drop_original": drop_original, "created": made}
    return (out, state) if return_state else out


# ======================================================================
#  6. REPLAY  --  the whole point of the module
# ======================================================================

def apply_state(df: Frame, state: Union[Dict[str, Any], Sequence[Dict[str, Any]]]) -> Frame:
    """Replay fitted feature engineering on new data.

    Accepts a single state or a list, applied in order.  Everything learned
    from the training set -- Box-Cox lambdas, scaler means, label maps,
    target-encoding tables, bin edges, group statistics -- is reused rather
    than refitted, which is what keeps the test set an honest test set.

    Unseen categories are handled explicitly per encoder: one-hot gives all
    zeros, count/frequency give 0, target and WOE fall back to the prior,
    ordinal gives NaN.  None of these silently drops the row.

    >>> train, s = fe.transform_numeric(train, "income", method="auto", return_state=True)
    >>> test = fe.apply_state(test, s)
    """
    if isinstance(state, (list, tuple)):
        for st in state:
            df = apply_state(df, st)
        return df

    out = df.copy()
    kind = state.get("kind")

    # ---------------- transform ----------------
    if kind == "transform":
        for src, e in state["entries"].items():
            if src not in out.columns:
                warnings.warn(f"apply_state: column '{src}' missing, skipped.",
                              stacklevel=2)
                continue
            x = pd.to_numeric(out[src], errors="coerce")
            out[e["target_column"]] = _apply_transform_one(x, e["params"])
        return out

    # ---------------- scale ----------------
    if kind == "scale":
        if state["method"] == "none":
            return out
        cols, names = state["columns"], state["names"]
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise KeyError(f"apply_state: scaler needs columns {missing}")
        block = out[cols].to_numpy(dtype=float)
        mask = np.isnan(block)
        filled = np.where(mask, 0.0, block)
        scaled = state["scaler"].transform(filled)
        scaled[mask] = np.nan
        for i, n in enumerate(names):
            out[n] = scaled[:, i]
        return out

    # ---------------- bin ----------------
    if kind == "bin":
        for src, e in state["entries"].items():
            if src not in out.columns:
                continue
            x = pd.to_numeric(out[src], errors="coerce")
            b = pd.cut(x, bins=e["edges"], labels=e["labels"], include_lowest=True)
            out[e["target_column"]] = b if e["as_category"] \
                else b.cat.codes.replace(-1, np.nan)
        return out

    # ---------------- encode ----------------
    if kind == "encode":
        for src, e in state["entries"].items():
            if src not in out.columns:
                warnings.warn(f"apply_state: column '{src}' missing, skipped.",
                              stacklevel=2)
                continue
            s = out[src]
            na_mask = s.isna()
            sv = s.astype("object").where(~na_mask, np.nan)
            if e.get("keep_levels") is not None:
                sv = sv.where(sv.isin(e["keep_levels"]) | sv.isna(), e["other_label"])
            m = e["method"]

            if m == "onehot":
                tokens = _tokenise(sv, e.get("separator"))
                pos = {lv: i for i, lv in enumerate(e["levels"])}
                mat = np.zeros((len(sv), len(e["levels"])), dtype=np.int8)
                for r, row in enumerate(tokens):
                    for t in row:
                        if t in pos:                       # unseen -> all zeros
                            mat[r, pos[t]] = 1
                for i, lv in enumerate(e["levels"]):
                    out[f"{src}_{lv}"] = mat[:, i]
                if e.get("dummy_na") and f"{src}_nan" in e["names"]:
                    out[f"{src}_nan"] = na_mask.astype(np.int8)

            elif m == "ordinal":
                out[e["names"][0]] = sv.map(e["mapping"]).astype("Float64")

            elif m == "binary":
                codes = sv.map(e["mapping"]).fillna(-1).astype(int).to_numpy()
                for i, n in enumerate(e["names"]):
                    out[n] = np.where(codes < 0, np.nan, (codes >> i) & 1)

            elif m in ("count", "frequency"):
                tokens = _tokenise(sv, e.get("separator"))
                lookup = e["mapping"]
                out[e["names"][0]] = [sum(lookup.get(t, e.get("default", 0.0))
                                          for t in row) if row else np.nan
                                      for row in tokens]

            elif m in ("target", "woe"):
                out[e["names"][0]] = sv.map(e["mapping"]).astype(float).fillna(e["prior"])

            elif m == "hashing":
                mat = _hash_matrix(sv, e["n_components"])
                for i, n in enumerate(e["names"]):
                    out[n] = mat[:, i]

        if state.get("drop_original"):
            out = out.drop(columns=[c for c in state["entries"] if c in out.columns])
        return out

    # ---------------- datetime ----------------
    if kind == "datetime":
        present = [c for c in state["columns"] if c in out.columns]
        if not present:
            return out
        out = add_datetime(out, present, parts=state["parts"],
                           cyclical=state["cyclical"], reference=state["reference"],
                           drop_original=state["drop_original"])
        return out

    if kind == "cyclical":
        return add_cyclical(out, state["column"], state["period"],
                            state["drop_original"])

    if kind == "interactions":
        return add_interactions(out, state["columns"], operations=state["operations"],
                                max_features=state["max_features"])

    if kind == "text":
        present = [c for c in state["columns"] if c in out.columns]
        if not present:
            return out
        return add_text_features(out, present, parts=state["parts"],
                                 drop_original=state["drop_original"])

    # ---------------- aggregates ----------------
    if kind == "aggregates":
        groups, gkey = state["groups"], state["gkey"]
        for v in state["values"]:
            agg = state["tables"][v]
            merged = out[groups].merge(agg, left_on=groups, right_index=True, how="left")
            for col in agg.columns:
                vals = merged[col].to_numpy()
                fn = col.split("_")[-3] if "_by_" in col else None
                stat = col.replace(f"{v}_", "").replace(f"_by_{gkey}", "")
                fallback = state["globals"].get(v, {}).get(stat, np.nan)
                out[col] = pd.Series(vals).fillna(fallback).to_numpy()
            if state["add_deviation"] and "mean" in state["funcs"]:
                mcol, scol = f"{v}_mean_by_{gkey}", f"{v}_std_by_{gkey}"
                if scol in out.columns:
                    out[f"{v}_dev_by_{gkey}"] = (out[v] - out[mcol]) / \
                        out[scol].replace(0, np.nan)
                else:
                    out[f"{v}_dev_by_{gkey}"] = out[v] - out[mcol]
        return out

    raise ValueError(f"Unknown state kind: {kind!r}")


def summary(state: Union[Dict[str, Any], Sequence[Dict[str, Any]]]) -> Frame:
    """Human-readable log of every fitted step: what ran, on what, producing what.

    Worth printing into a notebook next to the model score -- three months
    later this table is the only record of how the features were built.
    """
    states = list(state) if isinstance(state, (list, tuple)) else [state]
    rows = []
    for i, st in enumerate(states, 1):
        k = st.get("kind")
        if k == "transform":
            for src, e in st["entries"].items():
                rows.append({"step": i, "kind": k, "source": src,
                             "detail": e["params"]["method"],
                             "produced": e["target_column"]})
        elif k == "scale":
            rows.append({"step": i, "kind": k, "source": ", ".join(st["columns"][:6]),
                         "detail": st["method"],
                         "produced": f"{len(st.get('names', []))} column(s)"})
        elif k == "bin":
            for src, e in st["entries"].items():
                rows.append({"step": i, "kind": k, "source": src,
                             "detail": f"{st['method']}, {len(e['edges']) - 1} bins",
                             "produced": e["target_column"]})
        elif k == "encode":
            for src, e in st["entries"].items():
                rows.append({"step": i, "kind": k, "source": src,
                             "detail": e["method"],
                             "produced": f"{len(e.get('names', []))} column(s)"})
        elif k in ("datetime", "text"):
            for src, made in (st.get("created") or {}).items() if k == "datetime" \
                    else [(", ".join(st["columns"]), st.get("created", []))]:
                rows.append({"step": i, "kind": k, "source": src,
                             "detail": ", ".join(st["parts"][:4]),
                             "produced": f"{len(made)} column(s)"})
        elif k == "aggregates":
            rows.append({"step": i, "kind": k,
                         "source": f"{st['values']} by {st['groups']}",
                         "detail": ", ".join(st["funcs"]),
                         "produced": f"{len(st['created'])} column(s)"})
        else:
            rows.append({"step": i, "kind": k, "source": st.get("column", ""),
                         "detail": "", "produced": ""})
    return pd.DataFrame(rows)


def chain(df: Frame, steps: Sequence[Tuple[Callable, Dict[str, Any]]],
          verbose: bool = False) -> Tuple[Frame, List[Dict[str, Any]]]:
    """Run several feature-engineering steps and collect their states.

    >>> train, states = fe.chain(train, [
    ...     (fe.transform_numeric, {"columns": ["income"], "method": "auto"}),
    ...     (fe.encode,            {"columns": ["city"], "method": "target",
    ...                             "target": "y"}),
    ...     (fe.scale,             {"method": "robust", "target": "y"}),
    ... ])
    >>> test = fe.apply_state(test, states)

    Each callable must accept ``return_state=True``; it is injected for you.
    """
    states = []
    for fn, kwargs in steps:
        kwargs = dict(kwargs)
        kwargs["return_state"] = True
        before = df.shape[1]
        df, st = fn(df, **kwargs)
        states.append(st)
        if verbose:
            print(f"  {fn.__name__:20s} {before:>4} -> {df.shape[1]:>4} columns")
    return df, states


__all__ = [
    # transforms & scaling
    "transform_numeric", "scale", "bin_numeric",
    # encoding
    "encode",
    # derived features
    "add_datetime", "add_cyclical", "add_interactions", "add_aggregates",
    "add_text_features",
    # replay & orchestration
    "apply_state", "summary", "chain",
    # constants
    "TRANSFORMS", "SCALERS", "ENCODERS",
]

"""
datakit.eda
~~~~~~~~~~~

Exploratory data analysis: profiling tables, distribution diagnostics,
feature-target association, correlation, missingness patterns and
train/test drift.

Companion to ``datakit`` (cleaning + imbalance).  Where ``datakit`` changes
your data, this module only looks at it.

Design principles
-----------------
1. **Tables first, plots second.**  Every plotting function has a
   table-returning sibling, because you cannot filter, sort or unit-test a
   picture.  ``describe_numeric`` before ``plot_distribution``,
   ``relate`` before ``plot_target``, ``compare_distributions`` before
   ``plot_drift``.
2. **Never mutate global state.**  Styling is applied inside a context
   manager, so your other figures keep their own theme.
3. **Nothing is shown unless you ask.**  Plot helpers return the Figure and
   only call ``plt.show()`` when ``show=True`` (the default in a notebook,
   but easy to switch off in a script or a loop).
4. **Statistics that fit the data.**  Effect sizes rather than p-values,
   the right association measure per dtype pair, and explicit warnings when
   a test is being applied outside its comfort zone.

Quick start
-----------
>>> import datakit as dp        # cleaning + imbalance
>>> import eda                  # this module
>>> eda.describe_numeric(df)
>>> eda.relate(df, target="died")          # what actually matters
>>> eda.plot_target(df, target="died")
>>> eda.compare_distributions(train, test) # did the split go wrong?

Author: rebuilt from distribution_analysis_lib
License: MIT
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__version__ = "2.0.0"

Frame = pd.DataFrame
Series = pd.Series

# thresholds used across the module, exposed so you can tune them once
SKEW_MODERATE = 0.5
SKEW_STRONG = 1.0
KURT_TOL = 0.5


# ----------------------------------------------------------------------
# infrastructure
# ----------------------------------------------------------------------

# Shared helpers live in _common so a fix lands once, not three times.
try:                                    # inside the package
    from .common import _finish, _plt, _sns, _style, PALETTE, GOOD, BAD, NEUTRAL
except ImportError:                     # running the file standalone
    from common import _finish, _plt, _sns, _style, PALETTE, GOOD, BAD, NEUTRAL

DEFAULT_PALETTE = PALETTE          # kept as an alias for existing call sites


def _numeric_cols(df: Frame, columns=None) -> List[str]:
    if columns is not None:
        cols = [columns] if isinstance(columns, str) else list(columns)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
    else:
        cols = list(df.columns)
    return [c for c in cols
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _cat_cols(df: Frame, columns=None, max_unique: int = 50) -> List[str]:
    if columns is not None:
        cols = [columns] if isinstance(columns, str) else list(columns)
    else:
        cols = list(df.columns)
    out = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or s.dtype.name == "category" \
                or not pd.api.types.is_numeric_dtype(s):
            out.append(c)
        elif s.nunique(dropna=True) <= min(max_unique, 20):
            out.append(c)          # low-cardinality integers are categorical
    return out


def _is_classification(y: Series) -> bool:
    if y.dtype == object or str(y.dtype) in ("category", "bool", "string"):
        return True
    return y.nunique(dropna=True) <= 20


# ======================================================================
#  1. PROFILING TABLES
# ======================================================================

def describe_numeric(
    df: Frame,
    columns=None,
    percentiles: Sequence[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
    outlier_method: Literal["iqr", "mad", "none"] = "iqr",
) -> Frame:
    """Rich numeric profile: one row per column, everything that matters.

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
    """
    from scipy.stats import kurtosis as _kurt, skew as _skew

    cols = _numeric_cols(df, columns)
    if not cols:
        raise ValueError("No numeric columns found.")

    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        d = s.dropna()
        n = len(d)
        rec: Dict[str, Any] = {
            "column": c,
            "n": n,
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(d.nunique()),
        }
        if n < 3:
            rows.append({**rec, "shape": "too_few_values"})
            continue
        if d.nunique() == 1:
            rows.append({**rec, "mean": float(d.iloc[0]), "median": float(d.iloc[0]),
                         "std": 0.0, "min": float(d.iloc[0]), "max": float(d.iloc[0]),
                         "skew": np.nan, "kurtosis": np.nan, "n_zero": int((d == 0).sum()),
                         "n_negative": 0, "is_integer": bool(np.allclose(d, d.round())),
                         "is_binary": False, "n_outliers": 0, "outlier_pct": 0.0,
                         "shape": "constant"})
            continue

        q = d.quantile(list(percentiles))
        mean, med, std = float(d.mean()), float(d.median()), float(d.std())
        iqr = float(d.quantile(0.75) - d.quantile(0.25))
        mad = float((d - med).abs().median() * 1.4826)
        sk = float(_skew(d)) if n >= 3 else np.nan
        ku = float(_kurt(d)) if n >= 4 else np.nan

        rec.update(
            mean=round(mean, 4), median=round(med, 4), std=round(std, 4),
            min=float(d.min()), max=float(d.max()),
            iqr=round(iqr, 4), mad=round(mad, 4),
            cv=round(std / abs(mean), 4) if mean != 0 else np.nan,
            skew=round(sk, 3), kurtosis=round(ku, 3),
            n_zero=int((d == 0).sum()),
            n_negative=int((d < 0).sum()),
            is_integer=bool(np.allclose(d, d.round())),
            is_binary=bool(d.nunique() == 2),
        )
        for p, v in zip(percentiles, q):
            rec[f"p{int(p * 100)}"] = round(float(v), 4)

        if outlier_method == "none":
            rec["n_outliers"] = np.nan
            rec["outlier_pct"] = np.nan
        else:
            if outlier_method == "iqr":
                lo, hi = d.quantile(0.25) - 1.5 * iqr, d.quantile(0.75) + 1.5 * iqr
            else:
                scale = mad if mad > 0 else (std or 1.0)
                lo, hi = med - 3 * scale, med + 3 * scale
            n_out = int(((d < lo) | (d > hi)).sum())
            rec["n_outliers"] = n_out
            rec["outlier_pct"] = round(n_out / n * 100, 2)

        tags = []
        if abs(sk) < SKEW_MODERATE:
            tags.append("symmetric")
        elif abs(sk) < SKEW_STRONG:
            tags.append("right_skewed" if sk > 0 else "left_skewed")
        else:
            tags.append("strongly_right_skewed" if sk > 0 else "strongly_left_skewed")
        if ku > KURT_TOL:
            tags.append("heavy_tails")
        elif ku < -KURT_TOL:
            tags.append("light_tails")
        if rec["is_binary"]:
            tags = ["binary"]
        elif rec["unique"] <= 10:
            tags.append("discrete")
        rec["shape"] = ", ".join(tags)
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("column")
    front = ["n", "missing_pct", "unique", "mean", "median", "std",
             "min", "max", "skew", "kurtosis", "shape"]
    rest = [c for c in out.columns if c not in front]
    return out[[c for c in front if c in out.columns] + rest]


def describe_categorical(
    df: Frame,
    columns=None,
    top: int = 3,
    max_unique: int = 50,
) -> Frame:
    """Profile categorical / low-cardinality columns.

    ``imbalance`` is the headline number: the share of the most common
    level.  Above ~0.95 the column is effectively constant and will not
    support a stable coefficient, no matter how good your model is.

    ``entropy_ratio`` (0 to 1) is the normalised Shannon entropy -- 1 means
    perfectly uniform levels, 0 means one level dominates completely.
    """
    cols = _cat_cols(df, columns, max_unique)
    if not cols:
        raise ValueError("No categorical columns found.")

    rows = []
    for c in cols:
        s = df[c]
        vc = s.value_counts(dropna=True)
        n = int(s.notna().sum())
        if n == 0:
            rows.append({"column": c, "n": 0, "unique": 0, "flags": "all_missing"})
            continue
        p = vc / vc.sum()
        ent = float(-(p * np.log(p)).sum())
        max_ent = float(np.log(len(vc))) if len(vc) > 1 else 0.0

        rec: Dict[str, Any] = {
            "column": c,
            "dtype": str(s.dtype),
            "n": n,
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(vc.size),
            "mode": vc.index[0],
            "mode_pct": round(float(p.iloc[0] * 100), 2),
            "imbalance": round(float(p.iloc[0]), 4),
            "entropy_ratio": round(ent / max_ent, 4) if max_ent > 0 else 0.0,
            "n_rare_1pct": int((p < 0.01).sum()),
            "n_singleton": int((vc == 1).sum()),
        }
        for i in range(top):
            rec[f"top{i + 1}"] = (f"{vc.index[i]} ({p.iloc[i] * 100:.1f}%)"
                                  if i < len(vc) else "")
        flags = []
        if rec["unique"] <= 1:
            flags.append("constant")
        if rec["imbalance"] >= 0.95:
            flags.append("quasi_constant")
        if rec["unique"] > 50:
            flags.append("high_cardinality")
        if rec["n_singleton"] > 0:
            flags.append(f"{rec['n_singleton']}_singleton_levels")
        if rec["unique"] == n and n > 20:
            flags.append("id_like")
        rec["flags"] = ", ".join(flags)
        rows.append(rec)

    return pd.DataFrame(rows).set_index("column")


def test_normality(
    df: Frame,
    columns=None,
    alpha: float = 0.05,
) -> Frame:
    """Normality assessment that stays honest at large sample sizes.

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
    """
    from scipy.stats import kurtosis as _kurt, normaltest, shapiro, skew as _skew

    cols = _numeric_cols(df, columns)
    rows = []
    for c in cols:
        d = pd.to_numeric(df[c], errors="coerce").dropna()
        n = len(d)
        if n < 8 or d.nunique() < 3:
            rows.append({"column": c, "n": n, "test": "none",
                         "p_value": np.nan, "verdict": "too_few_values"})
            continue

        if n <= 5000:
            test_name = "shapiro_wilk"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = shapiro(d)
        else:
            test_name = "dagostino_pearson"
            _, p = normaltest(d)

        sk, ku = float(_skew(d)), float(_kurt(d))
        near_normal = abs(sk) < SKEW_MODERATE and abs(ku) < 1.0

        if p >= alpha:
            verdict = "consistent_with_normal"
        elif near_normal and n > 1000:
            verdict = "test_oversensitive (shape is near-normal)"
        elif abs(sk) >= SKEW_STRONG:
            verdict = "clearly_skewed"
        else:
            verdict = "not_normal"

        rows.append({
            "column": c, "n": n, "test": test_name,
            "p_value": float(f"{p:.3g}"),
            "skew": round(sk, 3), "excess_kurtosis": round(ku, 3),
            "verdict": verdict,
        })
    return pd.DataFrame(rows).set_index("column")


def suggest_transform(df: Frame, columns=None, target_skew: float = 0.5) -> Frame:
    """Recommend a variance-stabilising transform per numeric column.

    The recommendation respects the arithmetic: ``log`` needs strictly
    positive values, ``log1p`` tolerates zeros, ``sqrt`` needs
    non-negatives, and ``yeo-johnson`` is the only one that handles
    negative values.  Each candidate is actually applied and the resulting
    skew measured, so ``skew_after`` is a real number and not a promise.

    Returns a table with ``skew_before``, ``recommended``, ``skew_after``
    and ``reason``.  Apply the winner yourself -- transforming is a
    modelling decision, not an EDA side effect.
    """
    from scipy.stats import skew as _skew
    from scipy.stats import yeojohnson

    cols = _numeric_cols(df, columns)
    rows = []
    for c in cols:
        d = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(d) < 10 or d.nunique() < 5:
            rows.append({"column": c, "skew_before": np.nan,
                         "recommended": "none", "reason": "too_few_distinct_values"})
            continue

        s0 = float(_skew(d))
        if abs(s0) <= target_skew:
            rows.append({"column": c, "skew_before": round(s0, 3),
                         "recommended": "none", "skew_after": round(s0, 3),
                         "reason": "already_symmetric"})
            continue

        cands: Dict[str, np.ndarray] = {}
        has_neg, has_zero = bool((d < 0).any()), bool((d == 0).any())
        if not has_neg and not has_zero:
            cands["log"] = np.log(d)
            cands["sqrt"] = np.sqrt(d)
            cands["reciprocal"] = 1.0 / d
        elif not has_neg:
            cands["log1p"] = np.log1p(d)
            cands["sqrt"] = np.sqrt(d)
        try:
            cands["yeo_johnson"] = yeojohnson(d.to_numpy())[0]
        except Exception:
            pass

        scored = {}
        for name, vals in cands.items():
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if len(v) > 8:
                scored[name] = float(_skew(v))
        if not scored:
            rows.append({"column": c, "skew_before": round(s0, 3),
                         "recommended": "none", "reason": "no_valid_transform"})
            continue

        best = min(scored, key=lambda k: abs(scored[k]))
        reason = (f"has_negatives" if has_neg else
                  "has_zeros" if has_zero else "strictly_positive")
        if abs(scored[best]) >= abs(s0):
            best, reason = "none", reason + "; no transform improved skew"

        rows.append({
            "column": c,
            "skew_before": round(s0, 3),
            "recommended": best,
            "skew_after": round(scored.get(best, s0), 3),
            "reason": reason,
            "alternatives": ", ".join(f"{k}={v:.2f}" for k, v in
                                      sorted(scored.items(), key=lambda kv: abs(kv[1]))),
        })

    out = pd.DataFrame(rows).set_index("column")
    return out.sort_values("skew_before", key=lambda s: s.abs(), ascending=False)


# ======================================================================
#  2. FEATURE <-> TARGET ASSOCIATION
# ======================================================================

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 4000,
                  rng: Optional[np.random.Generator] = None) -> float:
    """Non-parametric effect size: P(a>b) - P(a<b), in [-1, 1].

    Computed from the Mann-Whitney U statistic, so it is O(n log n) rather
    than the naive O(n*m) pairwise comparison.
    """
    from scipy.stats import mannwhitneyu
    rng = rng or np.random.default_rng(0)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return np.nan
    return float(2 * u / (len(a) * len(b)) - 1)


def _cramers_v(x: Series, y: Series, bias_correct: bool = True) -> float:
    """Association between two categoricals, in [0, 1], bias-corrected.

    The uncorrected statistic is inflated by large tables, which makes a
    high-cardinality column look important purely because it has many
    levels.  Bergsma's correction removes most of that.
    """
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2 = chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = ct.shape
    if not bias_correct:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def _interpret(metric: str, strength: float) -> str:
    """Translate a 0-1 strength score into words.

    Takes the already-normalised ``strength``, never the raw statistic --
    passing a raw AUC here would double-transform it and invert the verdict.
    """
    if pd.isna(strength):
        return ""
    a = abs(strength)
    cuts = {"cohens_d": (0.2, 0.5, 0.8), "cliffs_delta": (0.147, 0.33, 0.474),
            "cramers_v": (0.1, 0.3, 0.5), "auc": (0.1, 0.3, 0.5),
            "correlation": (0.1, 0.3, 0.5), "eta_squared": (0.01, 0.06, 0.14)}
    lo, mid, hi = cuts.get(metric, (0.1, 0.3, 0.5))
    if a < lo:
        return "negligible"
    if a < mid:
        return "small"
    if a < hi:
        return "medium"
    return "large"


def relate(
    df: Frame,
    target: str,
    columns=None,
    task: Literal["auto", "classification", "regression"] = "auto",
    max_unique_cat: int = 50,
    sort_by: str = "strength",
) -> Frame:
    """Rank every feature by how strongly it relates to the target.

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
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found.")

    y = df[target]
    keep = y.notna()
    df = df.loc[keep]
    y = y.loc[keep]

    if task == "auto":
        task = "classification" if _is_classification(y) else "regression"
    is_clf = task == "classification"
    n_classes = int(y.nunique()) if is_clf else 0
    if is_clf and n_classes < 2:
        raise ValueError(f"Target '{target}' has {n_classes} distinct value(s).")

    feats = [c for c in (columns or df.columns) if c != target]
    rows = []

    for c in feats:
        s = df[c]
        ok = s.notna()
        n_used = int(ok.sum())
        if n_used < 10:
            continue

        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        many_levels = (not is_num) and s.nunique(dropna=True) > max_unique_cat
        if many_levels:
            rows.append({"feature": c, "dtype": "categorical", "metric": "skipped",
                         "value": np.nan, "strength": np.nan, "n_used": n_used,
                         "interpretation": "", "note": f"high cardinality "
                                                       f"({s.nunique()} levels)"})
            continue

        sv, yv = s[ok], y[ok]
        rec: Dict[str, Any] = {"feature": c, "n_used": n_used, "note": ""}

        # ---- numeric feature ------------------------------------------
        if is_num:
            rec["dtype"] = "numeric"
            if is_clf and n_classes == 2:
                from sklearn.metrics import roc_auc_score
                pos = yv.value_counts().idxmin()
                yb = (yv == pos).astype(int)
                if yb.nunique() < 2 or sv.nunique() < 2:
                    continue
                auc = float(roc_auc_score(yb, sv))
                a = sv[yb == 1].to_numpy(dtype=float)
                b = sv[yb == 0].to_numpy(dtype=float)
                rec.update(metric="auc", value=round(auc, 4),
                           strength=round(abs(auc - 0.5) * 2, 4),
                           cliffs_delta=round(_cliffs_delta(a, b), 4),
                           cohens_d=round(_cohens_d(a, b), 4),
                           direction=("higher in minority" if auc > 0.5
                                      else "lower in minority"))
                if abs(auc - 0.5) * 2 >= 0.9:
                    rec["note"] = "near-perfect: check for leakage"
            elif is_clf:
                groups = [g.to_numpy(dtype=float) for _, g in sv.groupby(yv, observed=True)]
                groups = [g for g in groups if len(g) > 1]
                eta2 = _eta_squared(groups)
                rec.update(metric="eta_squared", value=round(eta2, 4),
                           strength=round(eta2, 4))
            else:
                from scipy.stats import pearsonr, spearmanr
                if sv.nunique() < 2 or yv.nunique() < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sp = float(spearmanr(sv, yv).statistic)
                    pe = float(pearsonr(sv, yv).statistic)
                rec.update(metric="spearman", value=round(sp, 4),
                           strength=round(abs(sp), 4), pearson=round(pe, 4))
                if abs(sp) >= 0.95:
                    rec["note"] = "near-perfect: check for leakage"

        # ---- categorical feature --------------------------------------
        else:
            rec["dtype"] = "categorical"
            if is_clf:
                v = _cramers_v(sv, yv)
                rec.update(metric="cramers_v", value=round(v, 4)
                           if pd.notna(v) else np.nan,
                           strength=round(v, 4) if pd.notna(v) else np.nan,
                           n_levels=int(sv.nunique()))
                if pd.notna(v) and v >= 0.9:
                    rec["note"] = "near-perfect: check for leakage"
            else:
                groups = [g.to_numpy(dtype=float) for _, g in yv.groupby(sv, observed=True)]
                groups = [g for g in groups if len(g) > 1]
                eta2 = _eta_squared(groups)
                rec.update(metric="eta_squared", value=round(eta2, 4),
                           strength=round(eta2, 4), n_levels=int(sv.nunique()))

        rec["interpretation"] = _interpret(rec.get("metric", ""), rec.get("strength", np.nan))
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = ["feature", "dtype", "metric", "value", "strength",
             "interpretation", "n_used", "note"]
    out = out[[c for c in front if c in out.columns]
              + [c for c in out.columns if c not in front]]
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=False, na_position="last")
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _eta_squared(groups: List[np.ndarray]) -> float:
    """Share of variance in a numeric variable explained by group membership."""
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((allv - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def crosstab_target(
    df: Frame,
    column: str,
    target: str,
    normalize: Literal["index", "columns", "all", "none"] = "index",
    min_count: int = 10,
) -> Frame:
    """Event rate per level of a categorical feature, with sample sizes.

    Rates alone are misleading: a level with 3 rows can show a 67% death
    rate and mean nothing.  The ``n`` column and the ``reliable`` flag keep
    that visible.
    """
    ct = pd.crosstab(df[column], df[target], dropna=False)
    counts = ct.sum(axis=1)
    if normalize != "none":
        rates = pd.crosstab(df[column], df[target], normalize=normalize, dropna=False)
        out = (rates * 100).round(2)
        out.columns = [f"pct_{c}" for c in out.columns]
        out = pd.concat([ct.add_prefix("n_"), out], axis=1)
    else:
        out = ct.add_prefix("n_")
    out["n"] = counts
    out["pct_of_data"] = (counts / counts.sum() * 100).round(2)
    out["reliable"] = np.where(counts >= min_count, "", f"n < {min_count}")
    return out.sort_values("n", ascending=False)


# ======================================================================
#  3. CORRELATION
# ======================================================================

def correlation_table(
    df: Frame,
    columns=None,
    method: Literal["pearson", "spearman", "kendall"] = "spearman",
    min_abs: float = 0.0,
    target: Optional[str] = None,
) -> Frame:
    """Long-form correlation: one row per pair, sorted by strength.

    Far easier to act on than a heatmap once you have more than ~15
    columns.  Spearman is the default because it is monotone-robust and
    does not assume linearity.

    With ``target`` given, the correlation of each member of the pair with
    the target is added, so you can tell which one to drop.
    """
    cols = _numeric_cols(df, columns)
    cols = [c for c in cols if c != target]
    dropped = [c for c in cols if df[c].nunique(dropna=True) <= 1]
    cols = [c for c in cols if c not in dropped]
    if dropped:
        warnings.warn(f"Skipped constant column(s): {dropped} "
                      f"(correlation is undefined).", stacklevel=2)
    if len(cols) < 2:
        raise ValueError("Need at least 2 non-constant numeric columns.")

    corr = df[cols].corr(method=method)
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "corr"]
    pairs["abs_corr"] = pairs["corr"].abs()

    if target is not None and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        tc = df[cols].corrwith(df[target], method=method)
        pairs["a_vs_target"] = pairs["feature_a"].map(tc).round(4)
        pairs["b_vs_target"] = pairs["feature_b"].map(tc).round(4)
        pairs["drop_suggestion"] = np.where(
            pairs["a_vs_target"].abs() < pairs["b_vs_target"].abs(),
            pairs["feature_a"], pairs["feature_b"])

    pairs["corr"] = pairs["corr"].round(4)
    pairs["abs_corr"] = pairs["abs_corr"].round(4)
    pairs = pairs[pairs["abs_corr"] >= min_abs]
    return pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)


# ======================================================================
#  4. MISSINGNESS
# ======================================================================

def missing_pattern(df: Frame, top: int = 15) -> Frame:
    """The most common *combinations* of missing columns.

    Column-by-column missing rates hide structure.  If ``lab_a``, ``lab_b``
    and ``lab_c`` are always missing together, that is one phenomenon (the
    panel was not ordered), not three -- and it changes how you impute.
    """
    na = df.isna()
    cols_with_na = na.columns[na.any()].tolist()
    if not cols_with_na:
        return pd.DataFrame({"pattern": ["<no missing values>"], "n": [len(df)],
                             "pct": [100.0], "n_missing_cols": [0]})

    key = na[cols_with_na].apply(
        lambda r: ", ".join(c for c, v in zip(cols_with_na, r) if v) or "<complete>",
        axis=1)
    vc = key.value_counts().head(top)
    return pd.DataFrame({
        "pattern": vc.index,
        "n": vc.to_numpy(),
        "pct": (vc / len(df) * 100).round(2).to_numpy(),
        "n_missing_cols": [0 if p == "<complete>" else p.count(",") + 1 for p in vc.index],
    })


def missing_correlation(df: Frame, min_abs: float = 0.3) -> Frame:
    """Which columns go missing *together* (nullity correlation).

    A high value means the two columns' missingness is driven by the same
    upstream cause.  Pairs listed here should be imputed with the same
    strategy, or given a shared "was_missing" indicator.
    """
    na = df.isna()
    na = na.loc[:, na.any() & (na.mean() < 1.0)]
    if na.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "nullity_corr"])
    corr = na.astype(int).corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "nullity_corr"]
    pairs["nullity_corr"] = pairs["nullity_corr"].round(4)
    pairs = pairs[pairs["nullity_corr"].abs() >= min_abs]
    return pairs.sort_values("nullity_corr", key=lambda s: s.abs(),
                             ascending=False).reset_index(drop=True)


# ======================================================================
#  5. DRIFT / GROUP COMPARISON
# ======================================================================

def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two samples of one variable.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large
    shift.  Bin edges come from ``expected`` (your reference / training
    sample) so the comparison is anchored.
    """
    e = pd.Series(expected).dropna()
    a = pd.Series(actual).dropna()
    if len(e) == 0 or len(a) == 0:
        return np.nan

    if pd.api.types.is_numeric_dtype(e) and e.nunique() > bins:
        edges = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e_p = np.histogram(e, bins=edges)[0] / len(e)
        a_p = np.histogram(a, bins=edges)[0] / len(a)
    else:
        levels = sorted(set(e.unique()) | set(a.unique()), key=str)
        e_p = np.array([(e == l).mean() for l in levels])
        a_p = np.array([(a == l).mean() for l in levels])

    e_p = np.clip(e_p, eps, None)
    a_p = np.clip(a_p, eps, None)
    return float(((a_p - e_p) * np.log(a_p / e_p)).sum())


def compare_distributions(
    reference: Frame,
    current: Frame,
    columns=None,
    psi_bins: int = 10,
    label_a: str = "reference",
    label_b: str = "current",
) -> Frame:
    """Compare two frames column by column -- drift, or a bad split.

    Use it for train vs test (they should look identical -- if they do not,
    your split is broken or grouped incorrectly), for train vs production,
    or for any two cohorts you want to contrast.

    Reports PSI for every column, plus a KS statistic for numerics and a
    Cramer's V of membership for categoricals, and a plain ``verdict``.

    >>> tr, te = dp.split(df, "died")
    >>> eda.compare_distributions(tr, te).head()
    """
    from scipy.stats import ks_2samp

    cols = [c for c in (columns or reference.columns) if c in current.columns]
    rows = []
    for c in cols:
        a, b = reference[c], current[c]
        is_num = pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a)
        rec: Dict[str, Any] = {
            "column": c,
            f"n_{label_a}": int(a.notna().sum()),
            f"n_{label_b}": int(b.notna().sum()),
            f"missing_pct_{label_a}": round(float(a.isna().mean() * 100), 2),
            f"missing_pct_{label_b}": round(float(b.isna().mean() * 100), 2),
        }
        rec["psi"] = round(psi(a, b, bins=psi_bins), 4)

        if is_num:
            aa, bb = a.dropna(), b.dropna()
            if len(aa) > 1 and len(bb) > 1:
                rec["ks_stat"] = round(float(ks_2samp(aa, bb).statistic), 4)
                rec["mean_shift"] = round(float(bb.mean() - aa.mean()), 4)
                pooled = np.sqrt((aa.var(ddof=1) + bb.var(ddof=1)) / 2)
                rec["std_mean_shift"] = (round(float((bb.mean() - aa.mean()) / pooled), 4)
                                         if pooled > 0 else 0.0)
        else:
            lab = pd.concat([
                pd.DataFrame({"v": a, "g": label_a}),
                pd.DataFrame({"v": b, "g": label_b}),
            ])
            rec["cramers_v"] = round(_cramers_v(lab["v"], lab["g"]), 4)
            new = set(b.dropna().unique()) - set(a.dropna().unique())
            rec["unseen_levels"] = ", ".join(map(str, sorted(new, key=str)[:5]))

        p = rec["psi"]
        rec["verdict"] = ("" if pd.isna(p) else
                          "stable" if p < 0.1 else
                          "moderate shift" if p < 0.25 else "LARGE SHIFT")
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("column")
    return out.sort_values("psi", ascending=False, na_position="last")


def compare_groups(
    df: Frame,
    group: str,
    columns=None,
    max_groups: int = 10,
) -> Frame:
    """Summarise every feature across the levels of a grouping column.

    The classic "Table 1" of a clinical paper: mean +/- sd per group for
    numerics, percentages for categoricals, plus a standardised difference
    so you can see which contrasts are actually large.
    """
    if group not in df.columns:
        raise KeyError(f"Group column '{group}' not found.")
    levels = df[group].dropna().unique()
    if len(levels) > max_groups:
        raise ValueError(f"'{group}' has {len(levels)} levels; raise max_groups "
                         f"or pick a coarser column.")
    levels = sorted(levels, key=str)
    cols = [c for c in (columns or df.columns) if c != group]

    rows = []
    for c in cols:
        s = df[c]
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rec: Dict[str, Any] = {"feature": c,
                               "type": "numeric" if is_num else "categorical"}
        if is_num:
            for lv in levels:
                d = s[df[group] == lv].dropna()
                rec[str(lv)] = f"{d.mean():.2f} ± {d.std():.2f}" if len(d) else "-"
            if len(levels) == 2:
                a = s[df[group] == levels[0]].dropna().to_numpy(dtype=float)
                b = s[df[group] == levels[1]].dropna().to_numpy(dtype=float)
                rec["std_diff"] = round(_cohens_d(b, a), 3)
                rec["effect"] = _interpret("cohens_d", rec["std_diff"])
        else:
            if s.nunique(dropna=True) > 20:
                rec["note"] = "high cardinality, skipped"
            else:
                top = s.value_counts().index[0] if s.notna().any() else None
                for lv in levels:
                    d = s[df[group] == lv]
                    pct = (d == top).mean() * 100 if len(d) else np.nan
                    rec[str(lv)] = f"{pct:.1f}% {top}" if pd.notna(pct) else "-"
                v = _cramers_v(s, df[group])
                rec["std_diff"] = round(v, 3) if pd.notna(v) else np.nan
                rec["effect"] = _interpret("cramers_v", v)
        rows.append(rec)

    counts = df[group].value_counts()
    out = pd.DataFrame(rows).set_index("feature")
    out.attrs["group_sizes"] = {str(k): int(v) for k, v in counts.items()}
    return out


# ======================================================================
#  6. PLOTS
# ======================================================================

def plot_distribution(
    df: Frame,
    columns=None,
    kind: Literal["full", "hist", "box"] = "full",
    bins: Union[int, str] = "auto",
    kde: bool = True,
    log_x: Union[bool, Literal["auto"]] = "auto",
    hue: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    max_cols: int = 12,
    show: bool = True,
    color: str = DEFAULT_PALETTE[0],
):
    """Plot the distribution of one or many numeric columns.

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

    Returns
    -------
    matplotlib Figure.
    """
    plt, sns = _plt(), _sns()
    from scipy import stats as sps

    cols = _numeric_cols(df, columns)
    if not cols:
        raise ValueError("No numeric columns to plot.")
    if len(cols) > max_cols:
        warnings.warn(f"{len(cols)} numeric columns; showing the first {max_cols}. "
                      f"Pass columns=[...] to choose.", stacklevel=2)
        cols = cols[:max_cols]

    def _use_log(d: Series) -> bool:
        if log_x is True:
            return True
        if log_x is False or d.empty:
            return False
        return bool((d > 0).all() and abs(d.skew()) > 2)

    with _style():
        # ---------- single column, full diagnostic ----------
        if len(cols) == 1 and kind == "full":
            c = cols[0]
            d = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(d) < 3:
                raise ValueError(f"Column '{c}' has fewer than 3 valid values.")
            fig, ax = plt.subplots(2, 2, figsize=figsize or (13, 8))

            if sns is not None and hue is not None and hue in df.columns:
                sub = df.loc[d.index, [c, hue]]
                sns.histplot(sub, x=c, hue=hue, bins=bins, kde=kde,
                             stat="density", common_norm=False, ax=ax[0, 0])
                sns.boxplot(data=sub, x=c, y=hue, orient="h", ax=ax[0, 1])
                sns.violinplot(data=sub, x=c, y=hue, orient="h", ax=ax[1, 0])
            elif sns is not None:
                sns.histplot(d, bins=bins, kde=kde, color=color, ax=ax[0, 0])
                sns.boxplot(x=d, color=DEFAULT_PALETTE[1], ax=ax[0, 1])
                sns.violinplot(x=d, color=DEFAULT_PALETTE[2], ax=ax[1, 0])
            else:
                ax[0, 0].hist(d, bins=30 if bins == "auto" else bins, color=color)
                ax[0, 1].boxplot(d, vert=False)
                ax[1, 0].violinplot(d, vert=False)

            ax[0, 0].set_title(f"Distribution: {c}")
            ax[0, 1].set_title("Boxplot")
            ax[1, 0].set_title("Violin")
            if _use_log(d):
                for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
                    a.set_xscale("log")
                ax[0, 0].set_title(f"Distribution: {c}  (log x)")

            sps.probplot(d, dist="norm", plot=ax[1, 1])
            ax[1, 1].set_title("Q-Q plot vs normal")
            ax[1, 1].get_lines()[0].set_markersize(3)
            ax[1, 1].get_lines()[0].set_color(color)
            ax[1, 1].get_lines()[1].set_color(DEFAULT_PALETTE[3])

            med, mean = d.median(), d.mean()
            ax[0, 0].axvline(med, color=DEFAULT_PALETTE[3], ls="--", lw=1.2, label=f"median {med:,.3g}")
            ax[0, 0].axvline(mean, color=DEFAULT_PALETTE[2], ls=":", lw=1.4, label=f"mean {mean:,.3g}")
            ax[0, 0].legend(fontsize=8)
            fig.suptitle(f"{c}   n={len(d):,}   skew={d.skew():.2f}   "
                         f"kurtosis={d.kurtosis():.2f}", fontsize=11)
            return _finish(fig, show)

        # ---------- grid ----------
        ncol = min(3, len(cols))
        nrow = int(np.ceil(len(cols) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=figsize or (5 * ncol, 3.2 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for a, c in zip(axes, cols):
            d = pd.to_numeric(df[c], errors="coerce").dropna()
            if d.empty:
                a.set_visible(False)
                continue
            if kind == "box":
                if sns is not None:
                    sns.boxplot(x=d, color=color, ax=a)
                else:
                    a.boxplot(d, vert=False)
            elif sns is not None and hue is not None and hue in df.columns:
                sns.histplot(df.loc[d.index, [c, hue]], x=c, hue=hue, bins=bins,
                             kde=kde, stat="density", common_norm=False,
                             ax=a, legend=(a is axes[0]))
            elif sns is not None:
                sns.histplot(d, bins=bins, kde=kde, color=color, ax=a)
            else:
                a.hist(d, bins=30 if bins == "auto" else bins, color=color)
            if _use_log(d) and kind != "box":
                a.set_xscale("log")
            a.set_title(f"{c}  (skew {d.skew():.2f})", fontsize=10)
            a.set_xlabel("")
        for a in axes[len(cols):]:
            a.set_visible(False)
        return _finish(fig, show)


def plot_categorical(
    df: Frame,
    column: str,
    normalize: bool = False,
    top_n: Optional[int] = 20,
    include_missing: bool = True,
    horizontal: Union[bool, Literal["auto"]] = "auto",
    sort: Literal["count", "index"] = "count",
    ohe_prefix: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = DEFAULT_PALETTE[0],
    title: Optional[str] = None,
    show: bool = True,
    return_counts: bool = False,
):
    """Bar chart of category frequencies, with one-hot columns handled properly.

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
    """
    plt = _plt()
    counts, source = _category_counts(df, column, normalize, include_missing, ohe_prefix)

    n_levels = len(counts)
    if sort == "count":
        counts = counts.sort_values(ascending=False)
    else:
        counts = counts.sort_index()
    hidden = 0
    if top_n is not None and n_levels > top_n:
        counts = counts.sort_values(ascending=False).head(top_n)
        hidden = n_levels - top_n

    if horizontal == "auto":
        horizontal = bool(max((len(str(i)) for i in counts.index), default=0) > 12
                          or len(counts) > 12)

    with _style():
        fig, ax = plt.subplots(figsize=figsize or
                               ((8, max(3, 0.35 * len(counts))) if horizontal else (9, 4.5)))
        plot_vals = counts[::-1] if horizontal else counts
        plot_vals.plot(kind="barh" if horizontal else "bar", color=color, ax=ax)

        unit = "share (%)" if normalize else "count"
        total = counts.sum()
        for i, v in enumerate(plot_vals.to_numpy()):
            label = f"{v:.1f}%" if normalize else f"{int(v):,}"
            if horizontal:
                ax.text(v, i, " " + label, va="center", fontsize=8)
            else:
                ax.text(i, v, label, ha="center", va="bottom", fontsize=8)

        ax.set_xlabel(unit if horizontal else column)
        ax.set_ylabel(column if horizontal else unit)
        t = title or f"{column} — {n_levels} level(s), n={int(df.shape[0]):,}"
        if hidden:
            t += f"  [top {top_n} shown, {hidden} hidden]"
        if source == "onehot":
            t += "  [from one-hot columns]"
        ax.set_title(t, fontsize=11)
        if not horizontal:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig = _finish(fig, show)

    return (fig, counts) if return_counts else fig


def _category_counts(df: Frame, column: str, normalize: bool,
                     include_missing: bool, ohe_prefix: bool) -> Tuple[Series, str]:
    """Resolve a column name to counts, transparently handling one-hot blocks."""
    lower_map = {c.lower(): c for c in df.columns}

    if column in df.columns or column.lower() in lower_map:
        real = column if column in df.columns else lower_map[column.lower()]
        counts = df[real].value_counts(dropna=not include_missing)
        counts.index = [("<missing>" if (isinstance(i, float) and pd.isna(i)) or i is pd.NA
                         else i) for i in counts.index]
        if normalize:
            counts = counts / counts.sum() * 100
        return counts, "direct"

    if ohe_prefix:
        pref = column + "_"
        cand = [c for c in df.columns
                if c.lower().startswith(pref.lower()) and len(c) > len(pref)]
        # accept only if every candidate really is a 0/1 indicator
        binary = [c for c in cand
                  if pd.api.types.is_numeric_dtype(df[c])
                  and set(pd.unique(df[c].dropna())) <= {0, 1, True, False}]
        if binary and len(binary) == len(cand):
            counts = df[binary].sum(axis=0)
            counts.index = [c[len(pref):] for c in binary]   # case-safe strip
            if normalize:
                counts = counts / counts.sum() * 100
            return counts, "onehot"
        if cand:
            raise ValueError(
                f"Columns starting with '{pref}' exist but are not all 0/1 "
                f"indicators ({[c for c in cand if c not in binary][:4]}). "
                f"They are probably unrelated columns, not a one-hot block. "
                f"Pass the exact column name, or ohe_prefix=False."
            )

    raise KeyError(
        f"Column '{column}' not found, and no one-hot block with prefix "
        f"'{column}_' exists. Available: {list(df.columns)[:12]}..."
    )


def plot_target(
    df: Frame,
    target: str,
    columns=None,
    top_k: int = 9,
    task: Literal["auto", "classification", "regression"] = "auto",
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """Grid showing how the strongest features relate to the target.

    Features are ranked with :func:`relate` and the top ``top_k`` plotted,
    each with the display that fits its dtype: overlaid densities or
    boxplots for numeric features, event-rate bars for categorical ones.
    For categorical panels a dashed line marks the overall base rate, so
    you can see at a glance which levels sit above or below it.
    """
    plt, sns = _plt(), _sns()
    y = df[target]
    if task == "auto":
        task = "classification" if _is_classification(y) else "regression"

    ranking = relate(df, target, columns=columns, task=task)
    ranking = ranking[ranking["metric"] != "skipped"].head(top_k)
    if ranking.empty:
        raise ValueError("No usable features to plot against the target.")

    feats = ranking["feature"].tolist()
    ncol = min(3, len(feats))
    nrow = int(np.ceil(len(feats) / ncol))

    with _style():
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (5.2 * ncol, 3.5 * nrow))
        axes = np.atleast_1d(axes).ravel()

        for ax, (_, row) in zip(axes, ranking.iterrows()):
            c = row["feature"]
            s = df[c]
            is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            sub = df[[c, target]].dropna()

            if task == "classification":
                if is_num:
                    if sns is not None:
                        sns.kdeplot(data=sub, x=c, hue=target, common_norm=False,
                                    fill=True, alpha=.35, ax=ax, warn_singular=False)
                    else:
                        for lv, g in sub.groupby(target):
                            ax.hist(g[c], bins=25, alpha=.5, density=True, label=str(lv))
                        ax.legend(fontsize=7)
                else:
                    pos = y.value_counts().idxmin()
                    rate = sub.groupby(c, observed=True)[target].apply(
                        lambda g: (g == pos).mean() * 100).sort_values(ascending=False)
                    n_lv = sub.groupby(c, observed=True).size()
                    rate = rate.head(12)
                    rate.plot(kind="bar", ax=ax, color=DEFAULT_PALETTE[0])
                    base = (y == pos).mean() * 100
                    ax.axhline(base, color=DEFAULT_PALETTE[3], ls="--", lw=1.2)
                    ax.set_ylabel(f"% {pos}")
                    for i, lv in enumerate(rate.index):
                        ax.text(i, rate.iloc[i], f"n={n_lv.get(lv, 0)}",
                                ha="center", va="bottom", fontsize=7)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            else:
                if is_num:
                    ax.scatter(sub[c], sub[target], s=8, alpha=.3,
                               color=DEFAULT_PALETTE[0])
                    ax.set_ylabel(target)
                elif sns is not None:
                    sns.boxplot(data=sub, x=c, y=target, ax=ax, color=DEFAULT_PALETTE[0])
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

            ax.set_title(f"{c}\n{row['metric']}={row['value']} ({row['interpretation']})",
                         fontsize=9)
            ax.set_xlabel("")

        for a in axes[len(feats):]:
            a.set_visible(False)
        fig.suptitle(f"Top {len(feats)} features vs '{target}'", fontsize=12)
        return _finish(fig, show)


def plot_correlation(
    df: Frame,
    columns=None,
    method: Literal["pearson", "spearman", "kendall"] = "spearman",
    cluster: bool = True,
    annot: Union[bool, Literal["auto"]] = "auto",
    mask_upper: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """Correlation heatmap, optionally reordered so related blocks sit together.

    ``cluster=True`` applies hierarchical ordering on the correlation
    distance, which turns a noisy checkerboard into visible groups of
    redundant features.  Annotation is switched off automatically above 15
    columns, where the numbers become unreadable anyway.
    """
    plt, sns = _plt(), _sns()
    cols = [c for c in _numeric_cols(df, columns) if df[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        raise ValueError("Need at least 2 non-constant numeric columns.")
    corr = df[cols].corr(method=method)

    if cluster and len(cols) > 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform
            d = squareform(np.clip(1 - corr.abs().to_numpy(), 0, None), checks=False)
            order = leaves_list(linkage(d, method="average"))
            corr = corr.iloc[order, order]
        except Exception:
            pass

    if annot == "auto":
        annot = len(cols) <= 15
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if mask_upper else None

    with _style("white"):
        size = figsize or (max(6, 0.55 * len(cols) + 3),) * 2
        fig, ax = plt.subplots(figsize=size)
        if sns is not None:
            sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                        annot=annot, fmt=".2f", square=True, linewidths=.5,
                        cbar_kws={"shrink": .7, "label": method}, ax=ax,
                        annot_kws={"size": 8})
        else:
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
            ax.set_yticks(range(len(corr)), corr.index)
            fig.colorbar(im, ax=ax, shrink=.7)
        ax.set_title(f"{method.capitalize()} correlation"
                     f"{' (clustered)' if cluster else ''}", fontsize=11)
        return _finish(fig, show)


def plot_missing(
    df: Frame,
    max_cols: int = 40,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """Two-panel missingness view: rate per column, and the nullity matrix.

    The matrix (right panel) is the one worth studying: horizontal stripes
    mean whole rows are incomplete, vertical blocks mean a column failed
    wholesale, and aligned gaps across columns mean the same event caused
    all of them.
    """
    plt = _plt()
    na = df.isna()
    rates = (na.mean() * 100).sort_values(ascending=False)
    rates = rates[rates > 0].head(max_cols)
    if rates.empty:
        warnings.warn("No missing values in this frame.", stacklevel=2)

    with _style():
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=figsize or (14, max(4, 0.3 * max(len(rates), 6) + 2)),
            gridspec_kw={"width_ratios": [1, 1.4]})

        if not rates.empty:
            rates[::-1].plot(kind="barh", ax=a1, color=DEFAULT_PALETTE[3])
            for i, v in enumerate(rates[::-1].to_numpy()):
                a1.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
        a1.set_title("Missing rate per column", fontsize=11)
        a1.set_xlabel("% missing")

        cols = rates.index.tolist() or list(df.columns)[:max_cols]
        a2.imshow(na[cols].to_numpy(), aspect="auto", interpolation="nearest",
                  cmap="binary")
        a2.set_xticks(range(len(cols)))
        a2.set_xticklabels(cols, rotation=90, fontsize=8)
        a2.set_ylabel("row index")
        a2.set_title("Nullity matrix (black = missing)", fontsize=11)
        return _finish(fig, show)


def plot_drift(
    reference: Frame,
    current: Frame,
    columns=None,
    top_k: int = 6,
    label_a: str = "reference",
    label_b: str = "current",
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """Overlay the two distributions for the most-shifted columns.

    Columns are ranked by PSI (see :func:`compare_distributions`) so the
    panels you get are the ones actually worth looking at.
    """
    plt, sns = _plt(), _sns()
    table = compare_distributions(reference, current, columns,
                                  label_a=label_a, label_b=label_b)
    cols = table.dropna(subset=["psi"]).head(top_k).index.tolist()
    if not cols:
        raise ValueError("Nothing comparable between the two frames.")

    ncol = min(3, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    with _style():
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (5.2 * ncol, 3.4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, c in zip(axes, cols):
            a, b = reference[c].dropna(), current[c].dropna()
            is_num = pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a)
            if is_num and a.nunique() > 10:
                if sns is not None:
                    sns.kdeplot(a, ax=ax, label=label_a, fill=True, alpha=.3,
                                warn_singular=False, color=DEFAULT_PALETTE[0])
                    sns.kdeplot(b, ax=ax, label=label_b, fill=True, alpha=.3,
                                warn_singular=False, color=DEFAULT_PALETTE[1])
                else:
                    ax.hist(a, bins=25, alpha=.5, density=True, label=label_a)
                    ax.hist(b, bins=25, alpha=.5, density=True, label=label_b)
            else:
                pa = a.value_counts(normalize=True)
                pb = b.value_counts(normalize=True)
                comp = pd.DataFrame({label_a: pa, label_b: pb}).fillna(0).head(10) * 100
                comp.plot(kind="bar", ax=ax,
                          color=[DEFAULT_PALETTE[0], DEFAULT_PALETTE[1]])
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{c}   PSI={table.loc[c, 'psi']:.3f} "
                         f"({table.loc[c, 'verdict']})", fontsize=9)
            ax.legend(fontsize=8)
            ax.set_xlabel("")
        for a in axes[len(cols):]:
            a.set_visible(False)
        return _finish(fig, show)


def plot_balance(
    df: Frame,
    target: str,
    figsize: Tuple[float, float] = (9, 4),
    show: bool = True,
):
    """Class distribution of the target, with the imbalance ratio spelled out.

    Prints the number that decides your whole modelling strategy: how many
    minority events you actually have.  Below ~50, no resampling technique
    will save you -- the constraint is information, not balance.
    """
    plt = _plt()
    y = df[target].dropna()
    counts = y.value_counts()
    pct = counts / counts.sum() * 100
    ratio = counts.max() / counts.min() if counts.min() > 0 else np.inf

    with _style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={"width_ratios": [1.3, 1]})
        counts.plot(kind="bar", ax=a1,
                    color=DEFAULT_PALETTE[:len(counts)])
        for i, v in enumerate(counts.to_numpy()):
            a1.text(i, v, f"{int(v):,}\n{pct.iloc[i]:.1f}%", ha="center",
                    va="bottom", fontsize=9)
        a1.set_title(f"'{target}' — imbalance ratio {ratio:.1f} : 1", fontsize=11)
        a1.set_ylabel("count")
        plt.setp(a1.get_xticklabels(), rotation=0)
        a1.margins(y=.18)

        a2.pie(counts, labels=[str(i) for i in counts.index], autopct="%1.1f%%",
               colors=DEFAULT_PALETTE[:len(counts)], startangle=90,
               textprops={"fontsize": 9})
        a2.set_title("Share", fontsize=11)

        n_min = int(counts.min())
        advice = ("collect more data / try anomaly detection" if n_min < 50 else
                  "small minority: prefer class weights + threshold tuning" if n_min < 500 else
                  "enough events for standard approaches")
        fig.suptitle(f"minority n = {n_min:,}  →  {advice}", fontsize=10, y=.02)
        return _finish(fig, show)


# ======================================================================
#  7. ONE-CALL REPORT
# ======================================================================

def report(
    df: Frame,
    target: Optional[str] = None,
    plots: bool = False,
    top: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the whole EDA sweep and return every table in one dict.

    Keys: ``shape``, ``numeric``, ``categorical``, ``normality``,
    ``transforms``, ``missing_pattern``, ``missing_corr``, ``correlations``,
    and -- when ``target`` is given -- ``relate`` and ``balance``.

    With ``verbose=True`` a short prioritised list of findings is printed:
    quasi-constant columns, redundant pairs, suspicious associations,
    heavy skew.  That summary is the point; the tables are for the details.

    >>> res = eda.report(df, target="died")
    >>> res["relate"].head()
    """
    out: Dict[str, Any] = {"shape": df.shape}
    findings: List[str] = []

    try:
        out["numeric"] = describe_numeric(df)
    except ValueError:
        out["numeric"] = pd.DataFrame()
    try:
        out["categorical"] = describe_categorical(df)
    except ValueError:
        out["categorical"] = pd.DataFrame()

    if not out["numeric"].empty:
        out["normality"] = test_normality(df)
        out["transforms"] = suggest_transform(df)
        sk = out["transforms"]
        sk = sk[(sk["recommended"] != "none") & sk["skew_before"].abs().ge(SKEW_STRONG)]
        if len(sk):
            findings.append(f"{len(sk)} strongly skewed column(s); "
                            f"suggested transforms: "
                            f"{dict(sk['recommended'].head(4))}")

    if not out["categorical"].empty:
        qc = out["categorical"]
        qc = qc[qc["flags"].str.contains("constant|id_like", na=False)]
        if len(qc):
            findings.append(f"{len(qc)} constant / quasi-constant / id-like column(s): "
                            f"{list(qc.index[:5])}")

    out["missing_pattern"] = missing_pattern(df)
    out["missing_corr"] = missing_correlation(df)
    total_na = float(df.isna().mean().mean() * 100)
    if total_na > 0:
        worst = df.isna().mean().sort_values(ascending=False)
        worst = worst[worst > 0]
        findings.append(f"{total_na:.2f}% of cells missing; worst: "
                        f"{dict(worst.head(3).round(3))}")
    if len(out["missing_corr"]):
        findings.append(f"{len(out['missing_corr'])} column pair(s) go missing together "
                        f"— impute them jointly")

    try:
        out["correlations"] = correlation_table(df, min_abs=0.8, target=target)
        if len(out["correlations"]):
            findings.append(f"{len(out['correlations'])} pair(s) with |corr| >= 0.8 "
                            f"— consider dropping one of each")
    except ValueError:
        out["correlations"] = pd.DataFrame()

    if target is not None:
        out["relate"] = relate(df, target)
        y = df[target]
        if _is_classification(y):
            vc = y.value_counts()
            out["balance"] = {"counts": vc.to_dict(),
                              "minority_n": int(vc.min()),
                              "imbalance_ratio": round(float(vc.max() / max(vc.min(), 1)), 2)}
            if out["balance"]["imbalance_ratio"] >= 3:
                findings.append(
                    f"target imbalance {out['balance']['imbalance_ratio']}:1 "
                    f"(minority n={out['balance']['minority_n']}) — "
                    f"use AUPRC, not accuracy")
        leaky = out["relate"][out["relate"]["note"].str.contains("leak", na=False)]
        if len(leaky):
            findings.append(f"POSSIBLE LEAKAGE: {list(leaky['feature'])} "
                            f"— check when these are recorded")

    out["findings"] = findings

    if verbose:
        print("=" * 68)
        print(f"  EDA REPORT   {df.shape[0]:,} rows x {df.shape[1]} columns"
              + (f"   target='{target}'" if target else ""))
        print("=" * 68)
        if findings:
            for i, f in enumerate(findings, 1):
                print(f"  {i}. {f}")
        else:
            print("  No structural issues detected.")
        if target is not None and not out["relate"].empty:
            print("-" * 68)
            print(f"  Top {top} features by association with '{target}':")
            cols = ["feature", "metric", "value", "interpretation"]
            print(out["relate"].head(top)[cols].to_string(index=False))
        print("=" * 68)

    if plots:
        figs = {}
        if not out["numeric"].empty:
            figs["distributions"] = plot_distribution(df, show=True)
        if df.isna().any().any():
            figs["missing"] = plot_missing(df, show=True)
        try:
            figs["correlation"] = plot_correlation(df, show=True)
        except ValueError:
            pass
        if target is not None:
            if _is_classification(df[target]):
                figs["balance"] = plot_balance(df, target, show=True)
            figs["target"] = plot_target(df, target, show=True)
        out["figures"] = figs

    return out


__all__ = [
    # profiling
    "describe_numeric", "describe_categorical", "test_normality", "suggest_transform",
    # association
    "relate", "crosstab_target", "correlation_table",
    # missingness
    "missing_pattern", "missing_correlation",
    # drift / groups
    "psi", "compare_distributions", "compare_groups",
    # plots
    "plot_distribution", "plot_categorical", "plot_target", "plot_correlation",
    "plot_missing", "plot_drift", "plot_balance",
    # everything
    "report",
]

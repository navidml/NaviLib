"""
evaluate
~~~~~~~~

Model evaluation for regression, classification, clustering, ranking and
recommenders.

Companion to ``datakit`` (cleaning + imbalance) and ``eda`` (exploration).

Design principles
-----------------
1. **``score_*`` returns a table, ``plot_*`` returns a Figure,
   ``report_*`` does both and interprets the result.**  You can use the
   numbers without generating a picture, and log them without parsing one.
2. **Uncertainty is not optional.**  Every ``score_*`` accepts ``ci=True``
   and returns bootstrap confidence intervals.  With 18 events in your test
   set, a metric without an interval is a decoration.
3. **The right baseline, always shown.**  R^2 against the mean predictor,
   PR curves against prevalence, accuracy against the majority-class rate.
   A metric with no reference point cannot be judged.
4. **No global state, no forced ``plt.show()``.**  Styling is scoped;
   figures are returned.

Quick start
-----------
>>> import evaluate as ev
>>> ev.score_classification(y_true, y_pred, y_prob, ci=True)
>>> ev.report_classification(y_true, y_pred, y_prob)      # numbers + plots + verdict
>>> ev.compare_models({"rf": p_rf, "lgbm": p_lgb}, y_true)
>>> ev.decision_curve(y_true, y_prob)                     # is it clinically useful?

Author: rebuilt from ML Evaluation Toolkit (Navid Bordbar)
License: MIT
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__version__ = "2.0.0"

Frame = pd.DataFrame
ArrayLike = Union[np.ndarray, pd.Series, Sequence]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
GOOD, BAD, NEUTRAL = "#55A868", "#C44E52", "#8C8C8C"


# ----------------------------------------------------------------------
# infrastructure
# ----------------------------------------------------------------------

def _plt():
    import matplotlib.pyplot as plt
    return plt


def _sns():
    try:
        import seaborn as sns
        return sns
    except ImportError:  # pragma: no cover
        return None


@contextmanager
def _style(style: str = "whitegrid"):
    """Scoped plotting theme -- never mutates the caller's rcParams."""
    sns = _sns()
    if sns is None:
        yield
        return
    with sns.axes_style(style):
        yield


def _finish(fig, show: bool):
    plt = _plt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _as_array(x: ArrayLike, name: str = "input") -> np.ndarray:
    """Coerce to a flat numpy array, dropping the index.

    Index alignment is a silent killer: subtracting two pandas Series with
    different indices produces NaN rather than an error, so every metric
    downstream is quietly wrong.  Everything is converted up front.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.to_numpy()
    a = np.asarray(x)
    if a.ndim > 1:
        if a.shape[1] == 1:
            a = a.ravel()
        else:
            raise ValueError(f"{name} must be 1-dimensional, got shape {a.shape}. "
                             f"For predict_proba output pass column 1: proba[:, 1]")
    return a


def _check_pair(a: ArrayLike, b: ArrayLike, na: str = "y_true", nb: str = "y_pred"):
    a, b = _as_array(a, na), _as_array(b, nb)
    if len(a) != len(b):
        raise ValueError(f"{na} has {len(a)} elements but {nb} has {len(b)}.")
    if len(a) == 0:
        raise ValueError("Empty input.")
    return a, b


def _table(rows: List[Dict[str, Any]]) -> Frame:
    out = pd.DataFrame(rows).set_index("metric")
    return out


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: ArrayLike,
    n_boot: int = 1000,
    alpha: float = 0.05,
    stratify: Optional[ArrayLike] = None,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap confidence interval for any metric.

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

    Returns
    -------
    (point_estimate, lower, upper)

    >>> from sklearn.metrics import roc_auc_score
    >>> ev.bootstrap_ci(roc_auc_score, y_true, y_prob, stratify=y_true)
    (0.918, 0.847, 0.968)
    """
    arrays = [_as_array(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(random_state)

    try:
        point = float(metric_fn(*arrays))
    except Exception:
        return (np.nan, np.nan, np.nan)

    if stratify is not None:
        strat = _as_array(stratify)
        groups = {v: np.flatnonzero(strat == v) for v in np.unique(strat)}
    else:
        groups = None

    vals = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(g, len(g), replace=True)
                                  for g in groups.values()])
        try:
            v = float(metric_fn(*[a[idx] for a in arrays]))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue

    if len(vals) < 20:
        return (point, np.nan, np.nan)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return (point, float(lo), float(hi))


def _maybe_ci(rows: List[Dict[str, Any]], ci: bool, specs: Dict[str, Callable],
              arrays: Tuple, stratify=None, n_boot: int = 1000,
              random_state: int = 42) -> List[Dict[str, Any]]:
    if not ci:
        return rows
    lookup = {r["metric"]: r for r in rows}
    for name, fn in specs.items():
        if name not in lookup:
            continue
        _, lo, hi = bootstrap_ci(fn, *arrays, n_boot=n_boot,
                                 stratify=stratify, random_state=random_state)
        lookup[name]["ci_low"] = round(lo, 4) if np.isfinite(lo) else np.nan
        lookup[name]["ci_high"] = round(hi, 4) if np.isfinite(hi) else np.nan
    return rows


# ======================================================================
#  1. REGRESSION
# ======================================================================

def score_regression(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    ci: bool = False,
    n_boot: int = 500,
    baseline: Literal["mean", "median", "none"] = "mean",
    sample_weight: Optional[ArrayLike] = None,
    random_state: int = 42,
) -> Frame:
    """Regression metrics with a naive baseline for context.

    Every scale-dependent error (MAE, RMSE) is meaningless on its own -- an
    RMSE of 400 is excellent for house prices and catastrophic for
    probabilities.  The ``vs_baseline`` column expresses each error as a
    ratio against always predicting the training mean (or median), so
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
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import (explained_variance_score, max_error,
                                 mean_absolute_error, mean_squared_error,
                                 median_absolute_error, r2_score)

    y_true, y_pred = _check_pair(y_true, y_pred)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.all():
        warnings.warn(f"Dropped {(~finite).sum()} non-finite pair(s).", stacklevel=2)
        y_true, y_pred = y_true[finite], y_pred[finite]
    w = _as_array(sample_weight)[finite] if sample_weight is not None else None

    resid = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred, sample_weight=w))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=w)))

    n_zero = int((y_true == 0).sum())
    if n_zero:
        mape = np.nan
    else:
        mape = float(np.mean(np.abs(resid / y_true)) * 100)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.mean(np.abs(resid[denom > 0]) / denom[denom > 0]) * 100) \
        if (denom > 0).any() else np.nan

    rows = [
        {"metric": "n", "value": len(y_true)},
        {"metric": "mae", "value": round(mae, 6)},
        {"metric": "rmse", "value": round(rmse, 6)},
        {"metric": "medae", "value": round(float(median_absolute_error(y_true, y_pred)), 6)},
        {"metric": "max_error", "value": round(float(max_error(y_true, y_pred)), 6)},
        {"metric": "bias", "value": round(float(resid.mean()), 6)},
        {"metric": "r2", "value": round(float(r2_score(y_true, y_pred, sample_weight=w)), 4)},
        {"metric": "explained_variance",
         "value": round(float(explained_variance_score(y_true, y_pred, sample_weight=w)), 4)},
        {"metric": "mape_pct", "value": round(mape, 4) if np.isfinite(mape) else np.nan},
        {"metric": "smape_pct", "value": round(smape, 4) if np.isfinite(smape) else np.nan},
    ]
    if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append({"metric": "r_pearson",
                         "value": round(float(pearsonr(y_true, y_pred).statistic), 4)})
            rows.append({"metric": "r_spearman",
                         "value": round(float(spearmanr(y_true, y_pred).statistic), 4)})

    out = _table(rows)
    out["note"] = ""
    if n_zero:
        out.loc["mape_pct", "note"] = f"undefined: {n_zero} zero(s) in y_true; use smape"
    if abs(resid.mean()) > 0.1 * (resid.std() or 1):
        out.loc["bias", "note"] = "systematic offset: residuals not centred on zero"

    if baseline != "none":
        ref = float(np.median(y_true) if baseline == "median" else np.mean(y_true))
        b_mae = float(np.mean(np.abs(y_true - ref)))
        b_rmse = float(np.sqrt(np.mean((y_true - ref) ** 2)))
        out["vs_baseline"] = np.nan
        out.loc["mae", "vs_baseline"] = round(mae / b_mae, 4) if b_mae else np.nan
        out.loc["rmse", "vs_baseline"] = round(rmse / b_rmse, 4) if b_rmse else np.nan
        out.attrs["baseline"] = {"kind": baseline, "value": ref,
                                 "mae": b_mae, "rmse": b_rmse}
        if b_mae and mae / b_mae >= 1:
            out.loc["mae", "note"] = f"NO BETTER than predicting the {baseline}"

    if ci:
        from sklearn.metrics import mean_absolute_error as _mae, r2_score as _r2
        specs = {
            "mae": lambda a, b: _mae(a, b),
            "rmse": lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))),
            "r2": lambda a, b: _r2(a, b),
            "bias": lambda a, b: float(np.mean(a - b)),
        }
        rows2 = out.reset_index().to_dict("records")
        rows2 = _maybe_ci(rows2, True, specs, (y_true, y_pred),
                          n_boot=n_boot, random_state=random_state)
        out = _table(rows2)

    cols = ["value"] + [c for c in ["ci_low", "ci_high", "vs_baseline", "note"]
                        if c in out.columns]
    return out[cols]


def plot_regression(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    figsize: Tuple[float, float] = (14, 9),
    sample: Optional[int] = 5000,
    show: bool = True,
    random_state: int = 42,
):
    """Five-panel residual diagnostic.

    Panels: predicted vs actual, residual distribution, Q-Q plot,
    residuals vs fitted (heteroscedasticity), and absolute error by decile
    of the prediction -- the last one answers "where is my model worst?",
    which a single global RMSE cannot.

    Large samples are thinned to ``sample`` points for the scatter panels
    so the figure stays readable and fast; the metrics annotated on it are
    still computed on everything.
    """
    plt, sns = _plt(), _sns()
    from scipy import stats as sps

    y_true, y_pred = _check_pair(y_true, y_pred)
    resid = y_true - y_pred

    idx = np.arange(len(y_true))
    if sample and len(y_true) > sample:
        idx = np.random.default_rng(random_state).choice(len(y_true), sample, replace=False)

    with _style():
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=.32, wspace=.28)
        a_sc = fig.add_subplot(gs[0, 0])
        a_di = fig.add_subplot(gs[0, 1])
        a_qq = fig.add_subplot(gs[0, 2])
        a_rf = fig.add_subplot(gs[1, 0])
        a_dc = fig.add_subplot(gs[1, 1:])

        # predicted vs actual
        lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        a_sc.scatter(y_true[idx], y_pred[idx], s=12, alpha=.4, color=PALETTE[0],
                     edgecolors="none")
        a_sc.plot(lim, lim, "--", color=BAD, lw=1.5, label="perfect")
        a_sc.set_xlabel("actual"); a_sc.set_ylabel("predicted")
        a_sc.set_title("Predicted vs actual", fontweight="bold", fontsize=11)
        a_sc.legend(fontsize=8)

        # residual distribution
        if sns is not None:
            sns.histplot(resid, kde=True, ax=a_di, color=PALETTE[0], alpha=.7)
        else:
            a_di.hist(resid, bins=40, color=PALETTE[0], alpha=.7)
        a_di.axvline(0, color=BAD, ls="--", lw=1.4)
        a_di.axvline(resid.mean(), color=PALETTE[2], ls=":", lw=1.6,
                     label=f"bias={resid.mean():.3g}")
        a_di.set_title("Residual distribution", fontweight="bold", fontsize=11)
        a_di.set_xlabel("residual"); a_di.legend(fontsize=8)

        # QQ
        sps.probplot(resid, dist="norm", plot=a_qq)
        a_qq.get_lines()[0].set(color=PALETTE[0], markersize=3)
        a_qq.get_lines()[1].set(color=BAD)
        a_qq.set_title("Q-Q plot of residuals", fontweight="bold", fontsize=11)

        # residuals vs fitted
        a_rf.scatter(y_pred[idx], resid[idx], s=12, alpha=.4, color=PALETTE[0],
                     edgecolors="none")
        a_rf.axhline(0, color=BAD, ls="--", lw=1.5)
        if np.std(y_pred) > 0:
            z = np.polyfit(y_pred, resid, 1)
            xs = np.linspace(y_pred.min(), y_pred.max(), 50)
            a_rf.plot(xs, np.poly1d(z)(xs), color=PALETTE[2], lw=1.8,
                      label=f"trend slope={z[0]:.3g}")
            a_rf.legend(fontsize=8)
        a_rf.set_xlabel("predicted"); a_rf.set_ylabel("residual")
        a_rf.set_title("Residuals vs fitted", fontweight="bold", fontsize=11)

        # error by decile of prediction
        try:
            dec = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
            d = pd.DataFrame({"dec": dec, "abs_err": np.abs(resid), "pred": y_pred})
            g = d.groupby("dec").agg(mae=("abs_err", "mean"), n=("abs_err", "size"),
                                     lo=("pred", "min"), hi=("pred", "max"))
            a_dc.bar(range(len(g)), g["mae"], color=PALETTE[0], alpha=.85)
            a_dc.axhline(np.abs(resid).mean(), color=BAD, ls="--", lw=1.4,
                         label=f"overall MAE={np.abs(resid).mean():.3g}")
            a_dc.set_xticks(range(len(g)))
            a_dc.set_xticklabels([f"{lo:,.3g}\n–{hi:,.3g}" for lo, hi
                                  in zip(g["lo"], g["hi"])], fontsize=7)
            a_dc.set_xlabel("decile of predicted value")
            a_dc.set_ylabel("MAE")
            a_dc.set_title("Where the model is worst — error by prediction decile",
                           fontweight="bold", fontsize=11)
            a_dc.legend(fontsize=8)
        except (ValueError, IndexError):
            a_dc.set_visible(False)

        fig.suptitle(
            f"n={len(y_true):,}   residuals: mean={resid.mean():.4g}, "
            f"sd={resid.std():.4g}, skew={sps.skew(resid):.2f}",
            fontsize=10)
        return _finish(fig, show)


# ======================================================================
#  2. CLASSIFICATION
# ======================================================================

def _resolve_pos_label(y_true: np.ndarray, pos_label=None):
    """Decide which class counts as 'positive'.

    Guessing ``np.unique(y)[1]`` is wrong as often as it is right: with
    labels ``['positive', 'negative']`` alphabetical order makes 'positive'
    the *negative* class.  Rules, in order: an explicit ``pos_label``; the
    literal 1 / True if present; otherwise the minority class, which is
    what you almost always mean by "the event".
    """
    classes = np.unique(y_true)
    if pos_label is not None:
        if pos_label not in classes:
            raise ValueError(f"pos_label={pos_label!r} not in y_true ({classes}).")
        return pos_label
    for cand in (1, True, "1"):
        if cand in classes:
            return classes[list(classes).index(cand)]
    counts = pd.Series(y_true).value_counts()
    return counts.idxmin()


def _proba_1d(y_prob: ArrayLike) -> np.ndarray:
    """Accept either a 1-D score vector or the (n, 2) predict_proba matrix."""
    p = np.asarray(y_prob)
    if p.ndim == 2:
        if p.shape[1] == 2:
            return p[:, 1]
        raise ValueError("For multiclass probabilities pass the full (n, k) matrix "
                         "to the multiclass path, not a binary metric.")
    return p.ravel()


def score_classification(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    y_prob: Optional[ArrayLike] = None,
    pos_label=None,
    threshold: float = 0.5,
    ci: bool = False,
    n_boot: int = 500,
    labels: Optional[Sequence] = None,
    random_state: int = 42,
) -> Frame:
    """Classification metrics, imbalance-aware and correctly baselined.

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
    - **``pos_label`` is resolved explicitly**, defaulting to the minority
      class rather than to alphabetical order.

    Returns
    -------
    DataFrame indexed by metric, with ``value``, optional ``ci_low`` /
    ``ci_high``, and a ``note`` column carrying the interpretation.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, brier_score_loss,
                                 cohen_kappa_score, confusion_matrix, f1_score,
                                 log_loss, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)

    y_true = _as_array(y_true, "y_true")
    if y_pred is None and y_prob is None:
        raise ValueError("Provide y_pred, y_prob, or both.")

    classes = np.unique(y_true) if labels is None else np.asarray(labels)
    n_classes = len(classes)
    binary = n_classes == 2

    p1 = None
    if y_prob is not None:
        p1 = _proba_1d(y_prob) if binary else np.asarray(y_prob)
        if binary and len(p1) != len(y_true):
            raise ValueError(f"y_prob has {len(p1)} rows, y_true has {len(y_true)}.")

    if y_pred is None:
        if not binary:
            raise ValueError("Deriving labels from probabilities is only supported "
                             "for binary targets; pass y_pred for multiclass.")
        pos = _resolve_pos_label(y_true, pos_label)
        neg = [c for c in classes if c != pos][0]
        y_pred = np.where(p1 >= threshold, pos, neg)
    y_pred = _as_array(y_pred, "y_pred")
    if len(y_pred) != len(y_true):
        raise ValueError(f"y_pred has {len(y_pred)} rows, y_true has {len(y_true)}.")

    unseen = set(np.unique(y_pred)) - set(classes)
    rows: List[Dict[str, Any]] = []
    notes: Dict[str, str] = {}

    avg = "binary" if binary else "macro"
    pos = _resolve_pos_label(y_true, pos_label) if binary else None
    kw = {"pos_label": pos} if binary else {"average": "macro"}

    acc = float(accuracy_score(y_true, y_pred))
    majority_rate = float(pd.Series(y_true).value_counts(normalize=True).max())

    rows += [
        {"metric": "n", "value": len(y_true)},
        {"metric": "n_classes", "value": n_classes},
        {"metric": "accuracy", "value": round(acc, 4)},
        {"metric": "accuracy_baseline", "value": round(majority_rate, 4)},
        {"metric": "balanced_accuracy",
         "value": round(float(balanced_accuracy_score(y_true, y_pred)), 4)},
        {"metric": "precision", "value": round(float(
            precision_score(y_true, y_pred, zero_division=0, **kw)), 4)},
        {"metric": "recall", "value": round(float(
            recall_score(y_true, y_pred, zero_division=0, **kw)), 4)},
        {"metric": "f1", "value": round(float(
            f1_score(y_true, y_pred, zero_division=0, **kw)), 4)},
        {"metric": "mcc", "value": round(float(matthews_corrcoef(y_true, y_pred)), 4)},
        {"metric": "kappa", "value": round(float(cohen_kappa_score(y_true, y_pred)), 4)},
    ]
    if acc <= majority_rate + 1e-9:
        notes["accuracy"] = "NO BETTER than always predicting the majority class"

    if binary:
        cm = confusion_matrix(y_true, y_pred, labels=[c for c in classes if c != pos] + [pos])
        tn, fp, fn, tp = cm.ravel()
        prevalence = float((y_true == pos).mean())
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        rows += [
            {"metric": "specificity", "value": round(float(spec), 4)},
            {"metric": "npv", "value": round(float(npv), 4)},
            {"metric": "prevalence", "value": round(prevalence, 4)},
            {"metric": "tp", "value": int(tp)}, {"metric": "fp", "value": int(fp)},
            {"metric": "fn", "value": int(fn)}, {"metric": "tn", "value": int(tn)},
        ]
        notes["specificity"] = f"of {int(tp + fn)} true events, {int(tp)} caught"

        if p1 is not None:
            yb = (y_true == pos).astype(int)
            auc_ = float(roc_auc_score(yb, p1))
            ap = float(average_precision_score(yb, p1))
            brier = float(brier_score_loss(yb, np.clip(p1, 0, 1)))
            rows += [
                {"metric": "roc_auc", "value": round(auc_, 4)},
                {"metric": "average_precision", "value": round(ap, 4)},
                {"metric": "ap_baseline", "value": round(prevalence, 4)},
                {"metric": "ap_lift", "value": round(ap / prevalence, 3)
                 if prevalence > 0 else np.nan},
                {"metric": "brier", "value": round(brier, 5)},
                {"metric": "brier_baseline",
                 "value": round(float(prevalence * (1 - prevalence)), 5)},
                {"metric": "log_loss", "value": round(float(
                    log_loss(yb, np.clip(p1, 1e-15, 1 - 1e-15))), 5)},
            ]
            slope, intercept = _calibration_fit(yb, p1)
            rows += [
                {"metric": "calibration_slope", "value": round(slope, 4)},
                {"metric": "calibration_intercept", "value": round(intercept, 4)},
            ]
            notes["average_precision"] = f"vs {prevalence:.3f} prevalence baseline"
            notes["brier"] = "lower is better; compare to brier_baseline"
            if np.isfinite(slope):
                notes["calibration_slope"] = (
                    "well calibrated" if 0.9 <= slope <= 1.1 else
                    "over-confident (predictions too extreme)" if slope < 0.9 else
                    "under-confident (predictions too flat)")
            if intercept > 0.5:
                notes["calibration_intercept"] = ("risks systematically OVER-estimated "
                                                  "— did you resample? use prior_correct()")
    else:
        for avg_kind in ("micro", "weighted"):
            rows.append({"metric": f"f1_{avg_kind}", "value": round(float(
                f1_score(y_true, y_pred, average=avg_kind, zero_division=0)), 4)})
        if p1 is not None and np.ndim(p1) == 2:
            try:
                rows.append({"metric": "roc_auc_ovr_macro", "value": round(float(
                    roc_auc_score(y_true, p1, multi_class="ovr", average="macro")), 4)})
                rows.append({"metric": "log_loss", "value": round(float(
                    log_loss(y_true, p1)), 5)})
            except ValueError as exc:
                warnings.warn(f"Multiclass probability metrics skipped: {exc}", stacklevel=2)

    out = _table(rows)
    out["note"] = pd.Series(notes)
    out["note"] = out["note"].fillna("")
    if unseen:
        out.attrs["warning"] = f"y_pred contains labels absent from y_true: {unseen}"
        warnings.warn(out.attrs["warning"], stacklevel=2)

    if ci and binary:
        yb = (y_true == pos).astype(int)
        specs = {
            "accuracy": lambda a, b: float((a == b).mean()),
            "recall": lambda a, b: recall_score(a, b, pos_label=pos, zero_division=0),
            "precision": lambda a, b: precision_score(a, b, pos_label=pos, zero_division=0),
            "f1": lambda a, b: f1_score(a, b, pos_label=pos, zero_division=0),
        }
        rows2 = _maybe_ci(out.reset_index().to_dict("records"), True, specs,
                          (y_true, y_pred), stratify=y_true,
                          n_boot=n_boot, random_state=random_state)
        if p1 is not None:
            prob_specs = {"roc_auc": roc_auc_score,
                          "average_precision": average_precision_score,
                          "brier": brier_score_loss}
            rows2 = _maybe_ci(rows2, True, prob_specs, (yb, p1), stratify=yb,
                              n_boot=n_boot, random_state=random_state)
        out = _table(rows2)
        out["note"] = out["note"].fillna("")

    cols = ["value"] + [c for c in ["ci_low", "ci_high", "note"] if c in out.columns]
    return out[cols]


def _calibration_fit(y_bin: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Calibration slope and intercept from a logistic recalibration fit.

    Regress the outcome on the predicted log-odds.  Slope 1 and intercept 0
    mean the probabilities can be taken at face value.  Slope < 1 means
    they are too extreme; a positive intercept means they are too high
    overall -- the classic fingerprint of training on resampled data.
    """
    from sklearn.linear_model import LogisticRegression
    p = np.clip(p, 1e-9, 1 - 1e-9)
    if len(np.unique(y_bin)) < 2:
        return (np.nan, np.nan)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if not np.isfinite(logit).all() or np.std(logit) == 0:
        return (np.nan, np.nan)
    for kwargs in ({"C": np.inf}, {"penalty": None}, {"C": 1e12}):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr = LogisticRegression(solver="lbfgs", max_iter=1000, **kwargs)
                lr.fit(logit, y_bin)
            return (float(lr.coef_[0][0]), float(lr.intercept_[0]))
        except (TypeError, ValueError):
            continue
        except Exception:
            return (np.nan, np.nan)
    return (np.nan, np.nan)


def per_class_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Frame:
    """Per-class precision / recall / F1 with support and error breakdown.

    Adds what ``classification_report`` leaves out: how many of each class's
    errors went to which other class, so you can see *what* it is being
    confused with rather than only that it is being confused.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y_true, y_pred = _check_pair(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred])) if labels is None \
        else np.asarray(labels)
    names = list(class_names) if class_names is not None else [str(c) for c in classes]

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    rows = []
    for i, c in enumerate(classes):
        errs = cm[i].copy()
        errs[i] = 0
        worst = int(np.argmax(errs)) if errs.sum() else None
        rows.append({
            "class": names[i],
            "support": int(s[i]),
            "support_pct": round(float(s[i] / len(y_true) * 100), 2),
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "n_correct": int(cm[i, i]),
            "n_missed": int(errs.sum()),
            "most_confused_with": names[worst] if worst is not None else "",
            "n_to_that_class": int(errs[worst]) if worst is not None else 0,
        })
    return pd.DataFrame(rows).set_index("class").sort_values("support", ascending=False)


def threshold_sweep(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    pos_label=None,
    n_steps: int = 200,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
) -> Frame:
    """Every operating point of a binary classifier in one table.

    The 0.5 default threshold is an arbitrary convention, not a decision.
    This sweeps the whole range and reports sensitivity, specificity,
    precision, F1, Youden's J and expected cost at each, so you can pick
    the point that matches what a miss actually costs you.
    """
    y_true = _as_array(y_true)
    p = _proba_1d(y_prob)
    pos = _resolve_pos_label(y_true, pos_label)
    yb = (y_true == pos).astype(int)
    n_pos, n_neg = int(yb.sum()), int((1 - yb).sum())

    grid = np.unique(np.quantile(p, np.linspace(0, 1, n_steps)))
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (yb == 1)).sum())
        fp = int(((pred == 1) & (yb == 0)).sum())
        fn = n_pos - tp
        tn = n_neg - fp
        sens = tp / n_pos if n_pos else np.nan
        spec = tn / n_neg if n_neg else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (prec and sens and prec + sens) else 0.0
        rows.append({"threshold": round(float(t), 6), "tp": tp, "fp": fp,
                     "fn": fn, "tn": tn,
                     "sensitivity": round(float(sens), 4),
                     "specificity": round(float(spec), 4),
                     "precision": round(float(prec), 4) if np.isfinite(prec) else np.nan,
                     "f1": round(float(f1), 4),
                     "youden_j": round(float(sens + spec - 1), 4),
                     "flag_rate": round(float((pred == 1).mean()), 4),
                     "expected_cost": round(float(cost_fn * fn + cost_fp * fp), 2)})
    out = pd.DataFrame(rows)
    out.attrs["best_f1"] = float(out.loc[out["f1"].idxmax(), "threshold"])
    out.attrs["best_youden"] = float(out.loc[out["youden_j"].idxmax(), "threshold"])
    out.attrs["best_cost"] = float(out.loc[out["expected_cost"].idxmin(), "threshold"])
    return out


def decision_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    pos_label=None,
    thresholds: Optional[Sequence[float]] = None,
) -> Frame:
    """Decision curve analysis: net benefit against treat-all and treat-none.

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
    """
    y_true = _as_array(y_true)
    p = _proba_1d(y_prob)
    pos = _resolve_pos_label(y_true, pos_label)
    yb = (y_true == pos).astype(int)
    n = len(yb)
    prevalence = yb.mean()
    pts = np.asarray(thresholds) if thresholds is not None else np.linspace(0.01, 0.99, 99)

    rows = []
    for pt in pts:
        odds = pt / (1 - pt)
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (yb == 1)).sum())
        fp = int(((pred == 1) & (yb == 0)).sum())
        nb_model = tp / n - (fp / n) * odds
        nb_all = prevalence - (1 - prevalence) * odds
        rows.append({
            "threshold_prob": round(float(pt), 4),
            "net_benefit_model": round(float(nb_model), 6),
            "net_benefit_treat_all": round(float(nb_all), 6),
            "net_benefit_treat_none": 0.0,
            "n_flagged": int(pred.sum()),
        })
    out = pd.DataFrame(rows)
    # A model that flags nobody has net benefit 0 -- identical to treat-none,
    # not better than it. Require a margin of at least one true positive
    # per 1000 patients before calling the model the winning strategy.
    margin = 1.0 / n
    out["best_strategy"] = np.where(
        (out.net_benefit_model > out.net_benefit_treat_all + margin) &
        (out.net_benefit_model > margin), "model",
        np.where(out.net_benefit_treat_all > margin, "treat_all", "treat_none"))
    useful = out[out.best_strategy == "model"]["threshold_prob"]
    out.attrs["useful_range"] = ((float(useful.min()), float(useful.max()))
                                 if len(useful) else None)
    return out


def plot_classification(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    y_prob: Optional[ArrayLike] = None,
    pos_label=None,
    threshold: float = 0.5,
    class_names: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """Six-panel classifier diagnostic (binary) or three-panel (multiclass).

    Binary panels: confusion matrix with row percentages, ROC, PR curve
    against the prevalence baseline, calibration curve, score distribution
    by true class, and metric-versus-threshold.

    The last two are the ones usually missing and usually decisive: the
    score distribution shows *why* the classes are confused, and the
    threshold panel shows that the reported precision/recall trade-off was
    a choice, not a property of the model.
    """
    plt, sns = _plt(), _sns()
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_curve, roc_auc_score, roc_curve)

    y_true = _as_array(y_true, "y_true")
    classes = np.unique(y_true)
    binary = len(classes) == 2
    names = list(class_names) if class_names is not None else [str(c) for c in classes]

    p1 = _proba_1d(y_prob) if (y_prob is not None and binary) else None
    if y_pred is None:
        if p1 is None:
            raise ValueError("Provide y_pred or y_prob.")
        pos = _resolve_pos_label(y_true, pos_label)
        neg = [c for c in classes if c != pos][0]
        y_pred = np.where(p1 >= threshold, pos, neg)
    y_pred = _as_array(y_pred, "y_pred")

    full = binary and p1 is not None
    with _style():
        if full:
            fig = plt.figure(figsize=figsize or (15, 9))
            gs = fig.add_gridspec(2, 3, hspace=.35, wspace=.3)
            axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
            a_cm, a_roc, a_pr, a_cal, a_dist, a_thr = axes
        else:
            fig = plt.figure(figsize=figsize or (14, 4.6))
            gs = fig.add_gridspec(1, 3, wspace=.3)
            a_cm = fig.add_subplot(gs[0, 0])
            a_dist = fig.add_subplot(gs[0, 1])
            a_thr = fig.add_subplot(gs[0, 2])
            a_roc = a_pr = a_cal = None

        # --- confusion matrix -----------------------------------------
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, np.where(row_sum == 0, 1, row_sum)) * 100
        annot = np.array([[f"{cm[i, j]:,}\n{cm_pct[i, j]:.1f}%"
                           for j in range(cm.shape[1])] for i in range(cm.shape[0])])
        if sns is not None:
            sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=a_cm, square=True,
                        cbar=False, linewidths=1, linecolor="white",
                        xticklabels=names, yticklabels=names,
                        annot_kws={"size": 9})
        else:
            a_cm.imshow(cm, cmap="Blues")
        a_cm.set_xlabel("predicted"); a_cm.set_ylabel("actual")
        a_cm.set_title("Confusion matrix (row %)", fontweight="bold", fontsize=11)

        if full:
            pos = _resolve_pos_label(y_true, pos_label)
            yb = (y_true == pos).astype(int)
            prev = yb.mean()

            fpr, tpr, _ = roc_curve(yb, p1)
            auc_ = roc_auc_score(yb, p1)
            a_roc.plot(fpr, tpr, lw=2.2, color=PALETTE[0], label=f"AUC = {auc_:.3f}")
            a_roc.fill_between(fpr, tpr, alpha=.12, color=PALETTE[0])
            a_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="random = 0.500")
            j = np.argmax(tpr - fpr)
            a_roc.plot(fpr[j], tpr[j], "o", color=BAD, ms=7,
                       label=f"max Youden J = {tpr[j] - fpr[j]:.3f}")
            a_roc.set_xlabel("false positive rate"); a_roc.set_ylabel("true positive rate")
            a_roc.set_title("ROC curve", fontweight="bold", fontsize=11)
            a_roc.legend(loc="lower right", fontsize=8)

            prec, rec, _ = precision_recall_curve(yb, p1)
            ap = average_precision_score(yb, p1)
            a_pr.plot(rec, prec, lw=2.2, color=PALETTE[3], label=f"AP = {ap:.3f}")
            a_pr.fill_between(rec, prec, alpha=.12, color=PALETTE[3])
            a_pr.axhline(prev, color=NEUTRAL, ls="--", lw=1.4,
                         label=f"prevalence = {prev:.3f}")
            a_pr.set_xlabel("recall"); a_pr.set_ylabel("precision")
            a_pr.set_ylim(-.02, 1.02)
            a_pr.set_title(f"PR curve  (lift {ap / prev:.2f}x)" if prev > 0 else "PR curve",
                           fontweight="bold", fontsize=11)
            a_pr.legend(loc="upper right", fontsize=8)

            # --- calibration ------------------------------------------
            from sklearn.calibration import calibration_curve
            try:
                nb = min(10, max(3, int(yb.sum() // 5)))
                frac, mean_p = calibration_curve(yb, np.clip(p1, 0, 1),
                                                 n_bins=nb, strategy="quantile")
                a_cal.plot(mean_p, frac, "o-", color=PALETTE[0], lw=2, ms=6,
                           label="model")
                slope, inter = _calibration_fit(yb, p1)
                a_cal.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect")
                a_cal.set_xlabel("mean predicted probability")
                a_cal.set_ylabel("observed frequency")
                a_cal.set_title(f"Calibration (slope {slope:.2f}, int {inter:.2f})",
                                fontweight="bold", fontsize=11)
                a_cal.legend(fontsize=8)
            except Exception:
                a_cal.set_visible(False)

            # --- score distribution by TRUE class ---------------------
            for i, (lab, col) in enumerate(zip([0, 1], [PALETTE[0], PALETTE[3]])):
                d = p1[yb == lab]
                if len(d) == 0:
                    continue
                nm = names[list(classes).index(pos)] if lab == 1 else \
                    names[list(classes).index([c for c in classes if c != pos][0])]
                a_dist.hist(d, bins=30, alpha=.55, density=True, color=col,
                            label=f"{nm} (n={len(d):,})")
            a_dist.axvline(threshold, color="black", ls="--", lw=1.4,
                           label=f"threshold = {threshold:g}")
            a_dist.set_xlabel("predicted probability")
            a_dist.set_ylabel("density")
            a_dist.set_title("Score separation by true class",
                             fontweight="bold", fontsize=11)
            a_dist.legend(fontsize=8)

            # --- metrics vs threshold ---------------------------------
            sw = threshold_sweep(y_true, p1, pos_label=pos, n_steps=120)
            for c, col, ls in [("sensitivity", PALETTE[3], "-"),
                               ("specificity", PALETTE[0], "-"),
                               ("precision", PALETTE[2], "--"),
                               ("f1", PALETTE[4], ":")]:
                a_thr.plot(sw["threshold"], sw[c], color=col, ls=ls, lw=1.8, label=c)
            a_thr.axvline(threshold, color="black", ls="--", lw=1.2)
            a_thr.axvline(sw.attrs["best_f1"], color=GOOD, ls=":", lw=1.6,
                          label=f"best F1 @ {sw.attrs['best_f1']:.3f}")
            a_thr.set_xlabel("threshold"); a_thr.set_ylabel("score")
            a_thr.set_ylim(-.02, 1.02)
            a_thr.set_title("Metrics vs threshold", fontweight="bold", fontsize=11)
            a_thr.legend(fontsize=7, ncol=2)
        else:
            vc = pd.Series(y_pred).value_counts().reindex(classes, fill_value=0)
            tc = pd.Series(y_true).value_counts().reindex(classes, fill_value=0)
            x = np.arange(len(classes))
            a_dist.bar(x - .2, tc.to_numpy(), .4, label="actual", color=PALETTE[0])
            a_dist.bar(x + .2, vc.to_numpy(), .4, label="predicted", color=PALETTE[1])
            a_dist.set_xticks(x); a_dist.set_xticklabels(names, rotation=45, ha="right")
            a_dist.set_title("Class distribution", fontweight="bold", fontsize=11)
            a_dist.legend(fontsize=8)

            pc = per_class_report(y_true, y_pred, labels=classes, class_names=names)
            pc[["precision", "recall", "f1"]].plot(kind="bar", ax=a_thr,
                                                   color=PALETTE[:3], width=.8)
            a_thr.set_ylim(0, 1.05)
            a_thr.set_title("Per-class scores", fontweight="bold", fontsize=11)
            a_thr.legend(fontsize=8)
            plt.setp(a_thr.get_xticklabels(), rotation=45, ha="right")

        return _finish(fig, show)


def plot_calibration(
    y_true: ArrayLike,
    probas: Union[ArrayLike, Dict[str, ArrayLike]],
    pos_label=None,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    figsize: Tuple[float, float] = (11, 4.5),
    show: bool = True,
):
    """Reliability diagram for one or several models, with score histograms.

    ``quantile`` binning is the default because uniform bins leave the
    top deciles almost empty on imbalanced data, producing a curve that
    swings wildly on three observations.
    """
    plt = _plt()
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    y_true = _as_array(y_true)
    pos = _resolve_pos_label(y_true, pos_label)
    yb = (y_true == pos).astype(int)
    if not isinstance(probas, dict):
        probas = {"model": probas}

    with _style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={"width_ratios": [1.1, 1]})
        a1.plot([0, 1], [0, 1], "k--", lw=1.3, label="perfectly calibrated")
        for i, (name, p) in enumerate(probas.items()):
            p = np.clip(_proba_1d(p), 0, 1)
            col = PALETTE[i % len(PALETTE)]
            nb = min(n_bins, max(3, int(yb.sum() // 5)))
            frac, mean_p = calibration_curve(yb, p, n_bins=nb, strategy=strategy)
            slope, inter = _calibration_fit(yb, p)
            a1.plot(mean_p, frac, "o-", color=col, lw=2, ms=5,
                    label=f"{name}  brier={brier_score_loss(yb, p):.4f}, "
                          f"slope={slope:.2f}")
            a2.hist(p, bins=30, alpha=.5, color=col, label=name)
        a1.set_xlabel("mean predicted probability")
        a1.set_ylabel("observed frequency")
        a1.set_title("Calibration (reliability diagram)", fontweight="bold", fontsize=11)
        a1.legend(fontsize=8, loc="upper left")
        a2.axvline(yb.mean(), color=BAD, ls="--", lw=1.4,
                   label=f"prevalence = {yb.mean():.3f}")
        a2.set_xlabel("predicted probability"); a2.set_ylabel("count")
        a2.set_title("Score distribution", fontweight="bold", fontsize=11)
        a2.legend(fontsize=8)
        return _finish(fig, show)


def plot_decision_curve(
    y_true: ArrayLike,
    probas: Union[ArrayLike, Dict[str, ArrayLike]],
    pos_label=None,
    max_threshold: float = 0.6,
    figsize: Tuple[float, float] = (8, 5),
    show: bool = True,
):
    """Plot :func:`decision_curve` -- net benefit vs threshold probability.

    Read it as: over the range of risk thresholds a decision-maker might
    plausibly use, does the model's curve sit above both the treat-all
    diagonal and the treat-none horizontal?  If not, the model is not worth
    acting on however good its AUC looks.
    """
    plt = _plt()
    y_true = _as_array(y_true)
    if not isinstance(probas, dict):
        probas = {"model": probas}

    with _style():
        fig, ax = plt.subplots(figsize=figsize)
        first = True
        for i, (name, p) in enumerate(probas.items()):
            dc = decision_curve(y_true, p, pos_label=pos_label)
            dc = dc[dc.threshold_prob <= max_threshold]
            ax.plot(dc.threshold_prob, dc.net_benefit_model, lw=2.2,
                    color=PALETTE[i % len(PALETTE)], label=name)
            if first:
                ax.plot(dc.threshold_prob, dc.net_benefit_treat_all, "--",
                        color=NEUTRAL, lw=1.6, label="treat all")
                ax.axhline(0, color="black", lw=1.2, label="treat none")
                first = False
        ax.set_xlabel("threshold probability (risk level for acting)")
        ax.set_ylabel("net benefit")
        ax.set_ylim(bottom=min(-0.02, ax.get_ylim()[0]))
        ax.set_title("Decision curve analysis", fontweight="bold", fontsize=11)
        ax.legend(fontsize=9)
        return _finish(fig, show)


# ======================================================================
#  3. MODEL COMPARISON & ERROR ANALYSIS
# ======================================================================

def compare_models(
    y_true: ArrayLike,
    predictions: Dict[str, ArrayLike],
    task: Literal["auto", "classification", "regression"] = "auto",
    pos_label=None,
    threshold: float = 0.5,
    ci: bool = False,
    n_boot: int = 300,
    sort_by: Optional[str] = None,
) -> Frame:
    """Leaderboard across models, evaluated identically on the same data.

    ``predictions`` maps a model name to its predictions: probabilities for
    classification (labels are derived at ``threshold``), point estimates
    for regression.

    Guards against the usual comparison mistakes: every model sees exactly
    the same rows, the same positive label and the same threshold, and with
    ``ci=True`` you get intervals so you can see whether the ranking is
    real or noise.  A gap smaller than the overlapping intervals is not a
    result.

    >>> ev.compare_models(y_test, {"rf": p_rf, "lgbm": p_lgb, "logreg": p_lr}, ci=True)
    """
    y_true = _as_array(y_true)
    if task == "auto":
        uniq = np.unique(y_true)
        task = "classification" if (len(uniq) <= 20 or y_true.dtype == object) \
            else "regression"

    rows = []
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        if task == "classification":
            p1 = _proba_1d(pred) if pred.ndim == 2 or np.issubdtype(pred.dtype, np.floating) \
                else None
            if p1 is not None and set(np.unique(p1)) <= {0, 1}:
                p1 = None
            tbl = score_classification(y_true, y_pred=(None if p1 is not None else pred),
                                       y_prob=(p1 if p1 is not None else None),
                                       pos_label=pos_label, threshold=threshold,
                                       ci=ci, n_boot=n_boot)
            keep = ["roc_auc", "average_precision", "brier", "f1", "recall",
                    "precision", "specificity", "balanced_accuracy",
                    "calibration_slope"]
        else:
            tbl = score_regression(y_true, pred, ci=ci, n_boot=n_boot)
            keep = ["rmse", "mae", "r2", "bias", "smape_pct"]

        rec: Dict[str, Any] = {"model": name}
        for k in keep:
            if k in tbl.index:
                rec[k] = tbl.loc[k, "value"]
                if ci and "ci_low" in tbl.columns and pd.notna(tbl.loc[k, "ci_low"]):
                    rec[f"{k}_ci"] = (f"[{tbl.loc[k, 'ci_low']:.3f}, "
                                      f"{tbl.loc[k, 'ci_high']:.3f}]")
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("model")
    default = ("average_precision" if "average_precision" in out.columns
               else "roc_auc" if "roc_auc" in out.columns
               else "rmse" if "rmse" in out.columns else None)
    key = sort_by or default
    if key in out.columns:
        ascending = key in ("rmse", "mae", "brier", "smape_pct", "log_loss")
        out = out.sort_values(key, ascending=ascending)
    out.attrs["sorted_by"] = key
    return out


def error_analysis(
    X: Frame,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    task: Literal["auto", "classification", "regression"] = "auto",
    columns=None,
    bins: int = 4,
    min_group: int = 20,
) -> Frame:
    """Where does the model fail? Error rate broken down by feature segment.

    A single global score hides that the model may be fine on the bulk of
    the data and useless on the segment you care about.  Numeric features
    are split into quantile bins, categorical ones into their levels, and
    the error rate (classification) or MAE (regression) is reported per
    segment alongside its size.

    ``lift`` is the segment's error relative to the overall error: 2.0
    means the model is twice as wrong there.  Segments smaller than
    ``min_group`` are flagged rather than trusted.

    >>> ev.error_analysis(X_test, y_test, preds).head(10)
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    if len(X) != len(y_true):
        raise ValueError(f"X has {len(X)} rows but y has {len(y_true)}.")
    X = X.reset_index(drop=True)

    if task == "auto":
        task = "classification" if len(np.unique(y_true)) <= 20 else "regression"
    err = (y_true != y_pred).astype(float) if task == "classification" \
        else np.abs(y_true - y_pred)
    overall = float(err.mean())

    cols = list(columns) if columns is not None else list(X.columns)
    rows = []
    for c in cols:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) \
                and s.nunique(dropna=True) > bins:
            try:
                seg = pd.qcut(s, bins, duplicates="drop").astype(str)
            except ValueError:
                continue
        elif s.nunique(dropna=True) <= 30:
            seg = s.astype(str)
        else:
            continue
        seg = seg.fillna("<missing>")

        for lv, idx in seg.groupby(seg).groups.items():
            pos = X.index.get_indexer(idx)
            e = err[pos]
            if len(e) == 0:
                continue
            rows.append({
                "feature": c, "segment": str(lv), "n": len(e),
                "pct_of_data": round(len(e) / len(X) * 100, 2),
                "error": round(float(e.mean()), 4),
                "lift": round(float(e.mean() / overall), 3) if overall > 0 else np.nan,
                "reliable": "" if len(e) >= min_group else f"n < {min_group}",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.attrs["overall_error"] = overall
    out.attrs["metric"] = "error_rate" if task == "classification" else "mae"
    return out.sort_values("lift", ascending=False).reset_index(drop=True)


# ======================================================================
#  4. CLUSTERING
# ======================================================================

def score_clustering(
    X,
    labels: ArrayLike,
    true_labels: Optional[ArrayLike] = None,
    noise_label: int = -1,
    sample: Optional[int] = 10000,
    random_state: int = 42,
) -> Frame:
    """Internal and external clustering metrics, with noise handled correctly.

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
    """
    from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                                 davies_bouldin_score, homogeneity_completeness_v_measure,
                                 normalized_mutual_info_score, silhouette_score)

    labels = _as_array(labels, "labels")
    Xa = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    if len(labels) != len(Xa):
        raise ValueError(f"labels has {len(labels)} entries, X has {len(Xa)} rows.")

    is_noise = labels == noise_label
    n_noise = int(is_noise.sum())
    core = ~is_noise
    lab_core, X_core = labels[core], Xa[core]
    uniq = np.unique(lab_core)

    sizes = pd.Series(lab_core).value_counts()
    rows = [
        {"metric": "n_samples", "value": len(labels)},
        {"metric": "n_clusters", "value": int(len(uniq))},
        {"metric": "noise_pct", "value": round(n_noise / len(labels) * 100, 2)},
        {"metric": "smallest_cluster", "value": int(sizes.min()) if len(sizes) else 0},
        {"metric": "largest_cluster", "value": int(sizes.max()) if len(sizes) else 0},
        {"metric": "cluster_balance",
         "value": round(float(sizes.min() / sizes.max()), 4) if len(sizes) else np.nan},
    ]
    notes: Dict[str, str] = {}
    if n_noise:
        notes["noise_pct"] = f"{n_noise:,} point(s) labelled {noise_label}, " \
                             f"excluded from internal metrics"

    if len(uniq) < 2:
        warnings.warn(f"Only {len(uniq)} non-noise cluster(s); internal metrics "
                      f"are undefined.", stacklevel=2)
    else:
        idx = np.arange(len(X_core))
        note_sil = ""
        if sample and len(idx) > sample:
            idx = np.random.default_rng(random_state).choice(len(X_core), sample,
                                                             replace=False)
            note_sil = f"computed on a {sample:,}-point subsample"
        try:
            sil = float(silhouette_score(X_core[idx], lab_core[idx]))
        except ValueError:
            sil = np.nan
        rows += [
            {"metric": "silhouette", "value": round(sil, 4)},
            {"metric": "davies_bouldin",
             "value": round(float(davies_bouldin_score(X_core, lab_core)), 4)},
            {"metric": "calinski_harabasz",
             "value": round(float(calinski_harabasz_score(X_core, lab_core)), 2)},
        ]
        if note_sil:
            notes["silhouette"] = note_sil
        elif np.isfinite(sil):
            notes["silhouette"] = ("strong structure" if sil > .5 else
                                   "reasonable structure" if sil > .25 else
                                   "weak / overlapping clusters")

    if true_labels is not None:
        t = _as_array(true_labels)[core]
        h, c, v = homogeneity_completeness_v_measure(t, lab_core)
        rows += [
            {"metric": "ari", "value": round(float(adjusted_rand_score(t, lab_core)), 4)},
            {"metric": "nmi", "value": round(float(
                normalized_mutual_info_score(t, lab_core)), 4)},
            {"metric": "homogeneity", "value": round(float(h), 4)},
            {"metric": "completeness", "value": round(float(c), 4)},
            {"metric": "v_measure", "value": round(float(v), 4)},
        ]
        notes["ari"] = "0 = random labelling, 1 = identical partitions"

    out = _table(rows)
    out["note"] = pd.Series(notes)
    out["note"] = out["note"].fillna("")
    return out


def plot_clustering(
    X,
    labels: ArrayLike,
    true_labels: Optional[ArrayLike] = None,
    noise_label: int = -1,
    method: Literal["pca", "tsne", "umap"] = "pca",
    scale: bool = True,
    sample: Optional[int] = 5000,
    figsize: Tuple[float, float] = (15, 4.6),
    show: bool = True,
    random_state: int = 42,
):
    """Three-panel clustering view: projection, sizes, silhouette profile.

    The projection is standardised before PCA by default -- without
    scaling, whichever feature has the largest units dominates both
    components and the picture says more about your units than your
    clusters.  Noise points are drawn as small grey crosses so they read as
    what they are.

    The silhouette profile (right panel) is the most informative of the
    three: a cluster whose bar dips below zero contains points that would
    be better placed elsewhere.
    """
    plt = _plt()
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples
    from sklearn.preprocessing import StandardScaler

    labels = _as_array(labels)
    Xa = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    rng = np.random.default_rng(random_state)
    if sample and len(Xa) > sample:
        sel = rng.choice(len(Xa), sample, replace=False)
        Xa, labels = Xa[sel], labels[sel]
        if true_labels is not None:
            true_labels = _as_array(true_labels)[sel]

    Xs = StandardScaler().fit_transform(Xa) if scale else Xa
    if method == "pca":
        red = PCA(n_components=2, random_state=random_state)
        emb = red.fit_transform(Xs)
        var = red.explained_variance_ratio_
        xlab, ylab = f"PC1 ({var[0]:.0%})", f"PC2 ({var[1]:.0%})"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, random_state=random_state,
                   init="pca").fit_transform(Xs)
        xlab, ylab = "t-SNE 1", "t-SNE 2"
    else:
        umap = __import__("umap")
        emb = umap.UMAP(random_state=random_state).fit_transform(Xs)
        xlab, ylab = "UMAP 1", "UMAP 2"

    is_noise = labels == noise_label
    uniq = np.unique(labels[~is_noise])

    with _style():
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=figsize)

        for i, lv in enumerate(uniq):
            m = labels == lv
            a1.scatter(emb[m, 0], emb[m, 1], s=12, alpha=.65,
                       color=PALETTE[i % len(PALETTE)], label=f"{lv} (n={m.sum():,})",
                       edgecolors="none")
        if is_noise.any():
            a1.scatter(emb[is_noise, 0], emb[is_noise, 1], s=14, marker="x",
                       color=NEUTRAL, alpha=.5, label=f"noise (n={is_noise.sum():,})")
        a1.set_xlabel(xlab); a1.set_ylabel(ylab)
        a1.set_title(f"Clusters in {method.upper()} space"
                     + ("" if scale else "  [UNSCALED]"),
                     fontweight="bold", fontsize=11)
        if len(uniq) <= 10:
            a1.legend(fontsize=7, markerscale=1.5)

        sizes = pd.Series(labels).value_counts().sort_index()
        cols = [NEUTRAL if lv == noise_label else PALETTE[i % len(PALETTE)]
                for i, lv in enumerate(sizes.index)]
        a2.bar(range(len(sizes)), sizes.to_numpy(), color=cols)
        a2.set_xticks(range(len(sizes)))
        a2.set_xticklabels([str(i) for i in sizes.index], rotation=0)
        for i, v in enumerate(sizes.to_numpy()):
            a2.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        a2.set_xlabel("cluster"); a2.set_ylabel("size")
        a2.set_title("Cluster sizes", fontweight="bold", fontsize=11)
        a2.margins(y=.15)

        if len(uniq) >= 2:
            core = ~is_noise
            try:
                sv = silhouette_samples(Xs[core], labels[core])
                avg = sv.mean()
                y0 = 0
                for i, lv in enumerate(uniq):
                    vals = np.sort(sv[labels[core] == lv])
                    a3.fill_betweenx(np.arange(y0, y0 + len(vals)), 0, vals,
                                     color=PALETTE[i % len(PALETTE)], alpha=.8)
                    a3.text(-0.05, y0 + len(vals) / 2, str(lv), fontsize=8, va="center")
                    y0 += len(vals) + max(5, len(sv) // 100)
                a3.axvline(avg, color=BAD, ls="--", lw=1.5, label=f"mean = {avg:.3f}")
                a3.axvline(0, color="black", lw=.8)
                a3.set_xlabel("silhouette coefficient")
                a3.set_yticks([])
                a3.set_title("Silhouette profile", fontweight="bold", fontsize=11)
                a3.legend(fontsize=8)
            except Exception:
                a3.set_visible(False)
        else:
            a3.set_visible(False)
        return _finish(fig, show)


# ======================================================================
#  5. RANKING
# ======================================================================

def score_ranking(
    df: Frame,
    query_col: str,
    true_col: str,
    pred_col: str,
    k: Union[int, Sequence[int]] = 10,
    ap_denominator: Literal["min", "total"] = "min",
    per_query: bool = False,
) -> Frame:
    """Top-K retrieval metrics, evaluated at one or several cut-offs.

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
    """
    missing = {query_col, true_col, pred_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    ks = [k] if isinstance(k, (int, np.integer)) else list(k)
    if any(x <= 0 for x in ks):
        raise ValueError("All k must be positive.")

    per_rows: List[Dict[str, Any]] = []
    for qid, g in df.groupby(query_col, sort=False):
        g = g.sort_values(pred_col, ascending=False, kind="mergesort")
        rel_all = g[true_col].to_numpy(dtype=float)
        n_rel = int((rel_all > 0).sum())
        if n_rel == 0:
            continue
        ideal_all = np.sort(rel_all)[::-1]

        for kk in ks:
            ke = min(kk, len(g))
            rel = rel_all[:ke]
            bin_rel = (rel > 0).astype(int)

            prec = bin_rel.sum() / ke
            rec = bin_rel.sum() / n_rel

            hits = np.cumsum(bin_rel)
            denom = min(n_rel, ke) if ap_denominator == "min" else n_rel
            ap = float((hits * bin_rel / np.arange(1, ke + 1)).sum() / denom) \
                if denom else 0.0

            first = np.flatnonzero(bin_rel)
            rr = 1.0 / (first[0] + 1) if len(first) else 0.0

            disc = np.log2(np.arange(2, ke + 2))
            dcg = float(((2.0 ** rel - 1) / disc).sum())
            idcg = float(((2.0 ** ideal_all[:ke] - 1) / disc).sum())

            per_rows.append({
                "query": qid, "k": kk, "n_relevant": n_rel, "n_retrieved": len(g),
                f"precision": round(float(prec), 6),
                f"recall": round(float(rec), 6),
                f"ap": round(ap, 6),
                f"rr": round(float(rr), 6),
                f"ndcg": round(dcg / idcg if idcg > 0 else 0.0, 6),
                f"hit": int(bin_rel.sum() > 0),
            })

    if not per_rows:
        raise ValueError("No query has any relevant item; nothing to evaluate.")
    per = pd.DataFrame(per_rows)
    if per_query:
        return per

    agg = per.groupby("k").agg(
        n_queries=("query", "nunique"),
        mean_relevant=("n_relevant", "mean"),
        **{"precision@k": ("precision", "mean"), "recall@k": ("recall", "mean"),
           "map@k": ("ap", "mean"), "mrr": ("rr", "mean"),
           "ndcg@k": ("ndcg", "mean"), "hit_rate@k": ("hit", "mean")},
    ).round(4)
    agg.attrs["ap_denominator"] = ap_denominator
    return agg


# ======================================================================
#  6. RECOMMENDER
# ======================================================================

def score_recommender(
    recommended: Dict[Any, Sequence],
    ground_truth: Dict[Any, Sequence],
    k: Union[int, Sequence[int]] = 10,
    catalog: Optional[Sequence] = None,
    popularity: Optional[Dict[Any, float]] = None,
    strict_users: bool = True,
) -> Frame:
    """Top-K recommender metrics including coverage, novelty and diversity.

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
    """
    ks = [k] if isinstance(k, (int, np.integer)) else list(k)
    if any(x <= 0 for x in ks):
        raise ValueError("All k must be positive.")

    users = set(ground_truth) | set(recommended) if strict_users else set(recommended)
    users = [u for u in users if len(set(ground_truth.get(u, []))) > 0]
    if not users:
        raise ValueError("No user has any ground-truth item.")

    n_catalog = len(set(catalog)) if catalog is not None else None
    if popularity is None:
        counts: Dict[Any, int] = {}
        for items in ground_truth.values():
            for it in items:
                counts[it] = counts.get(it, 0) + 1
        tot = sum(counts.values()) or 1
        popularity = {it: c / tot for it, c in counts.items()}

    out_rows = []
    for kk in ks:
        prec, rec, aps, ndcgs, hits, novel = [], [], [], [], [], []
        shown, lists = set(), []
        n_empty = 0

        for u in users:
            truth = set(ground_truth.get(u, []))
            recs = list(recommended.get(u, []))[:kk]
            if not recs:
                n_empty += 1
                prec.append(0.0); rec.append(0.0); aps.append(0.0)
                ndcgs.append(0.0); hits.append(0)
                continue

            hit_vec = np.array([1 if it in truth else 0 for it in recs])
            ke = len(recs)
            shown.update(recs)
            lists.append(set(recs))

            prec.append(hit_vec.sum() / ke)
            rec.append(hit_vec.sum() / len(truth))
            hits.append(int(hit_vec.sum() > 0))

            denom = min(len(truth), ke)
            cum = np.cumsum(hit_vec)
            aps.append(float((cum * hit_vec / np.arange(1, ke + 1)).sum() / denom)
                       if denom else 0.0)

            disc = np.log2(np.arange(2, ke + 2))
            dcg = float((hit_vec / disc).sum())
            idcg = float((np.ones(denom) / np.log2(np.arange(2, denom + 2))).sum()) \
                if denom else 0.0
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            novel.append(float(np.mean([-np.log2(max(popularity.get(it, 1e-6), 1e-6))
                                        for it in recs])))

        # personalisation: 1 - mean Jaccard similarity between user lists
        pers = np.nan
        if len(lists) > 1:
            rng = np.random.default_rng(0)
            pairs = min(2000, len(lists) * (len(lists) - 1) // 2)
            sims = []
            for _ in range(pairs):
                i, j = rng.integers(0, len(lists), 2)
                if i == j:
                    continue
                inter = len(lists[i] & lists[j])
                union = len(lists[i] | lists[j])
                sims.append(inter / union if union else 0.0)
            pers = 1 - float(np.mean(sims)) if sims else np.nan

        rec_row = {
            "k": kk, "n_users": len(users), "n_users_no_recs": n_empty,
            "precision@k": round(float(np.mean(prec)), 4),
            "recall@k": round(float(np.mean(rec)), 4),
            "map@k": round(float(np.mean(aps)), 4),
            "ndcg@k": round(float(np.mean(ndcgs)), 4),
            "hit_rate@k": round(float(np.mean(hits)), 4),
            "n_distinct_items": len(shown),
            "coverage": (round(len(shown) / n_catalog, 4) if n_catalog else np.nan),
            "novelty_bits": round(float(np.mean(novel)), 3) if novel else np.nan,
            "personalisation": round(pers, 4) if np.isfinite(pers) else np.nan,
        }
        out_rows.append(rec_row)

    out = pd.DataFrame(out_rows).set_index("k")
    if n_catalog is None:
        out.attrs["note"] = "coverage needs `catalog=`; only a raw item count is shown"
    return out


def plot_ranking(
    results: Frame,
    title: str = "Ranking metrics",
    figsize: Tuple[float, float] = (11, 4.5),
    show: bool = True,
):
    """Plot a :func:`score_ranking` / :func:`score_recommender` table.

    With several cut-offs the metrics are drawn as curves against K, which
    is the shape you actually need to choose an operating K.  With a single
    K it falls back to a labelled bar chart.
    """
    plt = _plt()
    cols = [c for c in results.columns
            if any(c.startswith(m) for m in
                   ("precision", "recall", "map", "mrr", "ndcg", "hit_rate"))]
    if not cols:
        raise ValueError("No recognised ranking metrics in this table.")

    with _style():
        if len(results) > 1:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
            for i, c in enumerate(cols):
                a1.plot(results.index, results[c], "o-", lw=2,
                        color=PALETTE[i % len(PALETTE)], label=c)
            a1.set_xlabel("K"); a1.set_ylabel("score"); a1.set_ylim(-.02, 1.02)
            a1.set_title(f"{title} vs K", fontweight="bold", fontsize=11)
            a1.legend(fontsize=8)
            extra = [c for c in ("coverage", "personalisation", "novelty_bits")
                     if c in results.columns and results[c].notna().any()]
            if extra:
                for i, c in enumerate(extra):
                    a2.plot(results.index, results[c], "s--", lw=2,
                            color=PALETTE[i % len(PALETTE)], label=c)
                a2.set_xlabel("K")
                a2.set_title("Beyond-accuracy metrics", fontweight="bold", fontsize=11)
                a2.legend(fontsize=8)
            else:
                a2.set_visible(False)
        else:
            fig, ax = plt.subplots(figsize=(figsize[0] * .7, figsize[1]))
            vals = results.iloc[0][cols]
            bars = ax.bar(range(len(vals)), vals.to_numpy(),
                          color=PALETTE[:len(vals)], alpha=.9)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.index, rotation=45, ha="right")
            ax.set_ylim(0, 1.05); ax.set_ylabel("score")
            for b, v in zip(bars, vals.to_numpy()):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_title(f"{title} (K = {results.index[0]})",
                         fontweight="bold", fontsize=11)
        return _finish(fig, show)


# ======================================================================
#  7. ONE-CALL REPORTS
# ======================================================================

def _print_table(title: str, tbl: Frame, cols: Optional[List[str]] = None):
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)
    view = tbl[cols] if cols else tbl
    print(view.to_string())


def report_classification(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    y_prob: Optional[ArrayLike] = None,
    pos_label=None,
    threshold: float = 0.5,
    class_names: Optional[Sequence[str]] = None,
    ci: bool = True,
    plots: bool = True,
    show: bool = True,
) -> Dict[str, Any]:
    """Full classification evaluation: metrics, per-class table, plots, verdict.

    The printed verdict is the point.  It answers, in order: does the model
    beat the trivial baseline, are the probabilities usable, and over what
    range of decision thresholds is it worth acting on.

    Returns a dict with ``metrics``, ``per_class``, ``thresholds``,
    ``decision_curve`` and ``figures``.
    """
    y_true = _as_array(y_true)
    classes = np.unique(y_true)
    binary = len(classes) == 2

    m = score_classification(y_true, y_pred, y_prob, pos_label=pos_label,
                             threshold=threshold, ci=ci)
    if y_pred is None and binary and y_prob is not None:
        pos = _resolve_pos_label(y_true, pos_label)
        neg = [c for c in classes if c != pos][0]
        y_pred = np.where(_proba_1d(y_prob) >= threshold, pos, neg)

    out: Dict[str, Any] = {"metrics": m}
    out["per_class"] = per_class_report(y_true, y_pred, class_names=class_names)

    print("=" * 70)
    print(f"  CLASSIFICATION REPORT   n={len(y_true):,}   "
          f"{len(classes)} class(es)"
          + (f"   threshold={threshold:g}" if binary and y_prob is not None else ""))
    print("=" * 70)
    _print_table("Metrics", m)
    _print_table("Per class", out["per_class"])

    verdict: List[str] = []
    acc, base = m.loc["accuracy", "value"], m.loc["accuracy_baseline", "value"]
    verdict.append(f"accuracy {acc:.3f} vs {base:.3f} majority baseline → "
                   + ("BEATS baseline" if acc > base + .01 else "NO real gain"))

    if binary and y_prob is not None:
        ap, apb = m.loc["average_precision", "value"], m.loc["ap_baseline", "value"]
        verdict.append(f"AP {ap:.3f} vs {apb:.3f} prevalence → "
                       f"{m.loc['ap_lift', 'value']:.2f}x lift")
        slope = m.loc["calibration_slope", "value"]
        if np.isfinite(slope):
            verdict.append(f"calibration slope {slope:.2f} → "
                           f"{m.loc['calibration_slope', 'note']}")
        sw = threshold_sweep(y_true, y_prob, pos_label=pos_label)
        out["thresholds"] = sw
        verdict.append(f"best F1 at threshold {sw.attrs['best_f1']:.3f} "
                       f"(you used {threshold:g})")
        dc = decision_curve(y_true, y_prob, pos_label=pos_label)
        out["decision_curve"] = dc
        rng = dc.attrs.get("useful_range")
        verdict.append(f"worth acting on for risk thresholds "
                       f"{rng[0]:.2f}–{rng[1]:.2f}" if rng else
                       "NEVER beats treat-all / treat-none — not clinically useful")

    print("-" * 70)
    print("  VERDICT")
    print("-" * 70)
    for i, v in enumerate(verdict, 1):
        print(f"  {i}. {v}")
    print("=" * 70)
    out["verdict"] = verdict

    if plots:
        figs = {"main": plot_classification(y_true, y_pred, y_prob, pos_label=pos_label,
                                            threshold=threshold, class_names=class_names,
                                            show=show)}
        if binary and y_prob is not None:
            figs["calibration"] = plot_calibration(y_true, y_prob,
                                                   pos_label=pos_label, show=show)
            figs["decision_curve"] = plot_decision_curve(y_true, y_prob,
                                                         pos_label=pos_label, show=show)
        out["figures"] = figs
    return out


def report_regression(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    ci: bool = True,
    plots: bool = True,
    show: bool = True,
) -> Dict[str, Any]:
    """Full regression evaluation: metrics against a naive baseline, plots, verdict."""
    m = score_regression(y_true, y_pred, ci=ci)
    y_true, y_pred = _check_pair(y_true, y_pred)
    resid = y_true - y_pred

    print("=" * 70)
    print(f"  REGRESSION REPORT   n={len(y_true):,}")
    print("=" * 70)
    _print_table("Metrics", m)

    verdict = []
    ratio = m.loc["mae", "vs_baseline"] if "vs_baseline" in m.columns else np.nan
    if np.isfinite(ratio):
        verdict.append(f"MAE is {ratio:.2f}x the mean-predictor baseline → "
                       + ("useful" if ratio < .9 else "NO real gain over the mean"))
    verdict.append(f"R² = {m.loc['r2', 'value']:.3f} "
                   f"({m.loc['r2', 'value'] * 100:.1f}% of variance explained)")
    b = m.loc["bias", "value"]
    verdict.append(f"bias {b:+.4g} → "
                   + ("centred" if abs(b) < .1 * (resid.std() or 1)
                      else "SYSTEMATIC offset, model is consistently "
                           + ("under" if b > 0 else "over") + "-predicting"))
    print("-" * 70); print("  VERDICT"); print("-" * 70)
    for i, v in enumerate(verdict, 1):
        print(f"  {i}. {v}")
    print("=" * 70)

    out = {"metrics": m, "verdict": verdict}
    if plots:
        out["figures"] = {"main": plot_regression(y_true, y_pred, show=show)}
    return out


__all__ = [
    # shared
    "bootstrap_ci",
    # regression
    "score_regression", "plot_regression", "report_regression",
    # classification
    "score_classification", "per_class_report", "threshold_sweep", "decision_curve",
    "plot_classification", "plot_calibration", "plot_decision_curve",
    "report_classification",
    # comparison / diagnosis
    "compare_models", "error_analysis",
    # clustering
    "score_clustering", "plot_clustering",
    # ranking & recommenders
    "score_ranking", "score_recommender", "plot_ranking",
]

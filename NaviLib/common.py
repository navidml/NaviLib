"""
navdata._common
~~~~~~~~~~~~~~~

Shared infrastructure for every module in the package:

* plotting helpers (scoped theming, no global rcParams mutation)
* the **state registry** and the single :func:`apply_state` that dispatches
  to whichever module owns a given state kind
* state persistence (:func:`save_state` / :func:`load_state`)

Why a registry
--------------
``cleaning`` and ``feature_engineering`` each fit their own kinds of state,
and each knows how to replay only its own.  Before this module they both
exported a function called ``apply_state``, so ``from cleaning import *``
followed by ``from feature_engineering import *`` silently replaced the
first with the second -- and a mixed chain then failed with
``Unknown state kind: 'missing'`` from whichever one happened to win.

Here each module registers the kinds it owns, and one dispatcher routes by
``kind``.  Adding a new kind means one ``register_state_handler`` call; no
central list to keep in sync.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

Frame = pd.DataFrame
State = Dict[str, Any]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
GOOD, BAD, NEUTRAL = "#55A868", "#C44E52", "#8C8C8C"


# ----------------------------------------------------------------------
# plotting helpers
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
def _style(style: str = "whitegrid", context: str = "notebook"):
    """Apply a plotting theme *temporarily*.

    ``sns.set_style()`` mutates matplotlib's global rcParams, so calling it
    inside a plotting function restyles every other figure the user makes
    for the rest of the session.  A context manager keeps the change local.
    """
    sns = _sns()
    if sns is None:
        yield
        return
    with sns.axes_style(style), sns.plotting_context(context):
        yield


def _finish(fig, show: bool = True, tight: bool = True):
    """Lay out, optionally display, and always return the Figure."""
    plt = _plt()
    if tight:
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


# ----------------------------------------------------------------------
# state registry
# ----------------------------------------------------------------------

_HANDLERS: Dict[str, Callable[[Frame, State], Frame]] = {}
_OWNERS: Dict[str, str] = {}


def register_state_handler(kinds: Iterable[str], handler: Callable[[Frame, State], Frame],
                           owner: str = "") -> None:
    """Declare that ``handler`` can replay these state kinds.

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
    """
    for k in kinds:
        if k in _HANDLERS and _HANDLERS[k] is not handler:
            raise ValueError(
                f"State kind '{k}' is already registered by "
                f"'{_OWNERS.get(k, '?')}'; '{owner}' cannot claim it too. "
                f"Rename one of them."
            )
        _HANDLERS[k] = handler
        _OWNERS[k] = owner


def registered_kinds() -> Frame:
    """Every state kind the package can replay, and which module owns it."""
    return (pd.DataFrame([{"kind": k, "module": _OWNERS.get(k, "")}
                          for k in sorted(_HANDLERS)])
            .set_index("kind"))


def _validate(state: Any, position: str) -> str:
    if not isinstance(state, dict):
        raise TypeError(
            f"apply_state expected a state dict (or a list of them) at {position}, "
            f"got {type(state).__name__}. States come from the `return_state=True` "
            f"argument, e.g. `df, state = nv.fix_missing(df, return_state=True)`."
        )
    kind = state.get("kind")
    if kind is None:
        raise ValueError(f"State at {position} has no 'kind' field: "
                         f"keys are {sorted(state)[:8]}")
    if kind not in _HANDLERS:
        known = ", ".join(sorted(_HANDLERS)) or "<none registered>"
        raise ValueError(
            f"Unknown state kind {kind!r} at {position}. "
            f"Registered kinds: {known}"
        )
    return kind


def apply_state(
    df: Frame,
    state: Union[State, Sequence[State]],
    strict: bool = True,
) -> Frame:
    """Replay fitted preprocessing on new data -- any kind, any mix, in order.

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
    """
    states: List[State] = list(state) if isinstance(state, (list, tuple)) else [state]
    n = len(states)

    for i, st in enumerate(states, 1):
        where = f"step {i} of {n}"
        kind = _validate(st, where)
        handler = _HANDLERS[kind]
        try:
            df = handler(df, st)
        except KeyError as exc:
            if not strict:
                warnings.warn(f"{where} (kind='{kind}') skipped: {exc}", stacklevel=2)
                continue
            raise KeyError(
                f"{where} (kind='{kind}', owner='{_OWNERS.get(kind, '?')}') failed: "
                f"{exc}. The new data is missing a column this step needs -- check "
                f"that you are replaying the states in the order they were fitted, "
                f"or pass strict=False to skip."
            ) from exc
        except Exception as exc:
            raise type(exc)(
                f"{where} (kind='{kind}', owner='{_OWNERS.get(kind, '?')}') failed: {exc}"
            ) from exc
    return df


def describe_states(state: Union[State, Sequence[State]]) -> Frame:
    """Readable log of a chain: what ran, in what order, on what, producing what.

    Works across modules, so a mixed cleaning + feature chain reads as one
    table.  Worth printing next to your model score -- months later this is
    often the only record of how the features were built.
    """
    states = list(state) if isinstance(state, (list, tuple)) else [state]
    rows = []
    for i, st in enumerate(states, 1):
        kind = st.get("kind", "?") if isinstance(st, dict) else "?"
        base = {"step": i, "kind": kind, "module": _OWNERS.get(kind, "")}

        if kind == "missing":
            n = len(st.get("numeric", {})) + len(st.get("other", {}))
            base.update(detail=st.get("method", ""),
                        source=", ".join(st.get("columns", [])[:5]),
                        produced=(f"{n} column(s) imputed"
                                  + (f", {len(st.get('indicators', []))} indicator(s)"
                                     if st.get("indicators") else "")))
        elif kind == "outliers":
            base.update(detail=st.get("action", ""),
                        source=", ".join(list(st.get("bounds", {}))[:5]),
                        produced=f"{len(st.get('bounds', {}))} column(s) bounded")
        elif kind == "outliers_mv":
            base.update(detail=st.get("method", ""),
                        source=", ".join(st.get("columns", [])[:5]),
                        produced="is_outlier, outlier_score")
        elif kind == "rare":
            base.update(detail=f"-> '{st.get('other_label', '')}'",
                        source=", ".join(list(st.get("keep", {}))[:5]),
                        produced=f"{len(st.get('keep', {}))} column(s) regrouped")
        elif kind == "transform":
            e = st.get("entries", {})
            base.update(detail=", ".join(sorted({v["params"]["method"]
                                                 for v in e.values()})),
                        source=", ".join(list(e)[:5]),
                        produced=", ".join(v["target_column"] for v in e.values())[:60])
        elif kind == "scale":
            base.update(detail=st.get("method", ""),
                        source=", ".join(st.get("columns", [])[:5]),
                        produced=f"{len(st.get('names', []))} column(s)")
        elif kind == "bin":
            e = st.get("entries", {})
            base.update(detail=st.get("method", ""), source=", ".join(list(e)[:5]),
                        produced=f"{len(e)} column(s)")
        elif kind == "encode":
            e = st.get("entries", {})
            base.update(detail=", ".join(sorted({v["method"] for v in e.values()})),
                        source=", ".join(list(e)[:5]),
                        produced=f"{sum(len(v.get('names', [])) for v in e.values())} column(s)")
        elif kind in ("datetime", "text"):
            created = st.get("created", {})
            total = (sum(len(v) for v in created.values())
                     if isinstance(created, dict) else len(created))
            base.update(detail=", ".join(st.get("parts", [])[:4]),
                        source=", ".join(st.get("columns", [])[:5]),
                        produced=f"{total} column(s)")
        elif kind == "aggregates":
            base.update(detail=", ".join(st.get("funcs", [])),
                        source=f"{st.get('values')} by {st.get('groups')}",
                        produced=f"{len(st.get('created', []))} column(s)")
        elif kind == "interactions":
            base.update(detail=", ".join(st.get("operations", [])),
                        source=", ".join(st.get("columns", [])[:5]),
                        produced=f"{len(st.get('created', []))} column(s)")
        elif kind == "cyclical":
            base.update(detail=f"period={st.get('period')}",
                        source=st.get("column", ""), produced="sin, cos")
        else:
            base.update(detail="", source="", produced="")
        rows.append(base)

    out = pd.DataFrame(rows)
    cols = ["step", "kind", "module", "source", "detail", "produced"]
    return out[[c for c in cols if c in out.columns]]


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------

def save_state(state: Union[State, Sequence[State]], path: str,
               compress: int = 3) -> str:
    """Persist a fitted chain to disk.

    States hold live scikit-learn objects (scalers, imputers, isolation
    forests, quantile maps), so they are pickled with ``joblib`` rather
    than serialised to JSON.

    Two consequences worth knowing before you rely on this: the file is
    only loadable by a compatible scikit-learn version, and unpickling
    executes code -- never load a state file you did not create.  For
    long-term storage, keep the ``describe_states`` table alongside it so
    the chain can be rebuilt from scratch if the pickle ever stops loading.
    """
    import joblib
    import sklearn
    states = list(state) if isinstance(state, (list, tuple)) else [state]
    for i, st in enumerate(states, 1):
        _validate(st, f"step {i} of {len(states)}")
    payload = {
        "states": states,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "summary": describe_states(states),
    }
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    joblib.dump(payload, path, compress=compress)
    return os.path.abspath(path)


def load_state(path: str, check_versions: bool = True) -> List[State]:
    """Load a chain saved by :func:`save_state`, warning on version drift."""
    import joblib
    import sklearn
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "states" not in payload:
        raise ValueError(f"{path} is not a navdata state file.")
    if check_versions and payload.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"State was saved with scikit-learn {payload['sklearn_version']}, "
            f"you have {sklearn.__version__}. Fitted objects usually still load, "
            f"but verify the output before trusting it.", stacklevel=2)
    states = payload["states"]
    for i, st in enumerate(states, 1):
        _validate(st, f"step {i} of {len(states)}")
    return states


__all__ = [
    "apply_state", "describe_states", "register_state_handler", "registered_kinds",
    "save_state", "load_state",
    "PALETTE", "GOOD", "BAD", "NEUTRAL",
]

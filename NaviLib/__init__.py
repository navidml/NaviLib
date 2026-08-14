"""
navdata
~~~~~~~

A practical toolkit for tabular data work, from raw file to evaluated model.

Modules
-------
``cleaning``             inspect, clean, balance, split
``eda``                  profile, relate, correlate, drift
``feature_engineering``  transform, scale, encode, bin, derive
``modeling``             train, validate, tune, explain, ship
``evaluation``           score, plot, compare, interpret

Two ways to use it
------------------
Namespaced (recommended -- 87 function names is a lot to hold in your head,
and the prefix tells you which stage of the pipeline you are in):

>>> from navdata import cleaning as cl, eda, feature_engineering as fe, evaluation as ev
>>> cl.overview(df)
>>> eda.relate(df, target="y")

Or flat, for the handful of things you use constantly:

>>> import navdata as nv
>>> nv.overview(df); nv.split(df, "y"); nv.apply_state(test, states)

The one function that lives here and nowhere else
-------------------------------------------------
:func:`apply_state` replays a fitted preprocessing chain on new data.  It is
defined at package level, not in a module, because a real chain mixes states
from ``cleaning`` and ``feature_engineering`` and no single module can own
them all:

>>> train, s1 = cl.fix_missing(train, method="median", return_state=True)
>>> train, s2 = fe.encode(train, ["city"], method="target", target="y",
...                       return_state=True)
>>> train, s3 = fe.scale(train, target="y", return_state=True)
>>> test = nv.apply_state(test, [s1, s2, s3])     # routed to the right owner
>>> nv.describe_states([s1, s2, s3])              # what did I actually do?
>>> nv.save_state([s1, s2, s3], "artifacts/prep.joblib")

Each module registers the state kinds it fits (see
:func:`registered_kinds`), so nothing here needs updating when a module
grows a new one.
"""

from __future__ import annotations

__version__ = "0.4.0"
__author__ = "Navid"

# --- submodules -------------------------------------------------------
from . import _common, cleaning, eda, evaluation, feature_engineering, modeling

# --- shared state machinery ------------------------------------------
from ._common import (
    apply_state,
    describe_states,
    load_state,
    register_state_handler,
    registered_kinds,
    save_state,
    PALETTE,
)

# --- wire each module's replay function into the dispatcher -----------
# Done once, here, at import time. A module that later gains a new state
# kind only has to add it to its own *_STATE_KINDS tuple.
register_state_handler(cleaning.CLEANING_STATE_KINDS, cleaning._replay,
                       owner="cleaning")
register_state_handler(feature_engineering.FEATURE_STATE_KINDS,
                       feature_engineering._replay, owner="feature_engineering")


def _reexport(module, names):
    """Lift selected module functions to the package namespace."""
    out = {}
    for n in names:
        obj = getattr(module, n, None)
        if obj is None:
            raise AttributeError(f"{module.__name__} has no '{n}' to re-export")
        out[n] = obj
    globals().update(out)
    return list(out)


# --- flat convenience API --------------------------------------------
# Deliberately partial. These are the functions used in almost every
# session; everything else stays behind its module prefix so the flat
# namespace does not become 87 names deep.
_FLAT = {
    cleaning: ["overview", "scan_missing", "scan_leakage", "clean_names",
               "fix_missing", "fix_outliers", "drop_constant", "split",
               "balance", "balance_pipeline", "class_weights", "prior_correct",
               "compare_balance", "save_table"],
    eda: ["describe_numeric", "describe_categorical", "relate",
          "compare_distributions", "report"],
    feature_engineering: ["transform_numeric", "scale", "encode", "bin_numeric",
                          "chain"],
    modeling: ["ChainTransformer", "make_pipeline", "validate", "tune", "train",
               "predict", "compare_algorithms", "explain", "save_model",
               "load_model"],
    evaluation: ["score_classification", "score_regression", "compare_models",
                 "report_classification", "report_regression", "error_analysis",
                 "bootstrap_ci", "find_best_k"],
}
for _mod, _names in _FLAT.items():
    _reexport(_mod, _names)


def help_map() -> "object":
    """Print what lives where -- the map of the whole package.

    Run this when you come back after three months and cannot remember
    which module holds which function.
    """
    import pandas as pd
    rows = []
    for mod in (cleaning, eda, feature_engineering, modeling, evaluation):
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            if not callable(obj):
                continue
            doc = (obj.__doc__ or "").strip().split("\n")[0]
            rows.append({
                "module": mod.__name__.rsplit(".", 1)[-1],
                "function": name,
                "flat": name in globals(),
                "summary": doc[:70],
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["module", "function"]).reset_index(drop=True)


__all__ = [
    # submodules
    "cleaning", "eda", "feature_engineering", "modeling", "evaluation",
    # state machinery -- the reason this file exists
    "apply_state", "describe_states", "save_state", "load_state",
    "registered_kinds", "register_state_handler",
    # navigation
    "help_map", "PALETTE", "__version__",
] + [n for names in _FLAT.values() for n in names]

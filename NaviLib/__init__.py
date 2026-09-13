"""NaviLib: practical tabular analysis and machine-learning workflows.

Use namespaced modules for specialist functions, and the flat namespace for
common operations. ``help_map(query)`` lists signatures, summaries and aliases.

Examples
--------
>>> import NaviLib as nv
>>> from NaviLib import cleaning, eda, modeling
>>> catalog = nv.help_map("missing")
>>> theme = nv.set_theme("light")

Fitted cleaning and feature states share a single dispatcher, ``apply_state``.
Use ``ChainTransformer`` to refit learned preprocessing inside validation folds.
The ``create_report`` function produces reusable tables, figures and offline HTML.
"""

from __future__ import annotations

from ._version import __version__
__author__ = "Navid"

# --- submodules -------------------------------------------------------
from . import _common, cleaning, eda, evaluation, feature_engineering, modeling
from . import quality, theme, timeseries, statistical_tests, reporting
from .quality import audit_data, infer_schema, validate_schema, DataSchema
from .theme import set_theme, get_theme, theme_context, available_themes, style_table
from .timeseries import temporal_split, add_lag_features, add_rolling_features
from .reporting import create_report, DataReport

# A module alias supports old imports on case-sensitive and Windows filesystems.
import sys as _sys
_sys.modules[__name__ + ".Statistical_Tests"] = statistical_tests
Statistical_Tests = statistical_tests

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

_STANDARD = {
    cleaning: ["impute_missing", "drop_duplicates", "handle_outliers", "convert_columns",
               "split_data", "resample_data"],
    feature_engineering: ["scale_features", "encode_categorical", "add_datetime_features", "add_cyclical_features"],
    modeling: ["cross_validate_model", "tune_model", "train_model", "predict_model"],
}
for _mod, _names in _STANDARD.items():
    _reexport(_mod, _names)


def help_map(query: str | None = None) -> "object":
    """Print what lives where -- the map of the whole package.

    Run this when you come back after three months and cannot remember
    which module holds which function.

    Parameters
    ----------
    query : str | None, default None
        Optional case-insensitive literal search across module, function and
        summary text.

    Returns
    -------
    pandas.DataFrame
        Searchable catalog with module, function, signature, summary, flat-
        export and alias information.
    """
    import pandas as pd
    import inspect
    rows = []
    for mod in (cleaning, eda, feature_engineering, modeling, evaluation,
                statistical_tests, quality, timeseries, theme, reporting):
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            if not callable(obj):
                continue
            doc = (obj.__doc__ or "").strip().split("\n")[0]
            rows.append({
                "module": mod.__name__.rsplit(".", 1)[-1],
                "function": name,
                "flat": name in globals(),
                "summary": doc,
                "signature": str(inspect.signature(obj)),
                "alias_of": obj.__name__ if name != obj.__name__ else "",
            })
    df = pd.DataFrame(rows)
    if query:
        mask = df[["module", "function", "summary"]].apply(
            lambda s: s.str.contains(query, case=False, regex=False)).any(axis=1)
        df = df[mask]
    return df.sort_values(["module", "function"]).reset_index(drop=True)


__all__ = [
    # submodules
    "cleaning", "eda", "feature_engineering", "modeling", "evaluation",
    # state machinery -- the reason this file exists
    "apply_state", "describe_states", "save_state", "load_state",
    "registered_kinds", "register_state_handler",
    # navigation
    "help_map", "PALETTE", "__version__",
] + [n for names in _FLAT.values() for n in names] + [
    n for names in _STANDARD.values() for n in names
] + ["quality", "theme", "timeseries", "statistical_tests", "reporting",
     "audit_data", "infer_schema", "validate_schema", "DataSchema",
     "set_theme", "get_theme", "theme_context", "available_themes", "style_table",
     "temporal_split", "add_lag_features", "add_rolling_features", "create_report", "DataReport"]

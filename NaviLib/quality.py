"""Actionable data-quality checks and reference-based schema validation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["audit_data", "infer_schema", "validate_schema", "DataSchema"]


def _check_frame(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are ambiguous; use clean_names first.")


def audit_data(df: pd.DataFrame, *, target: str | None = None,
               missing_threshold: float = .3, high_cardinality: float = .9) -> pd.DataFrame:
    """Find quality problems and suggest concrete next steps without changing data.

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
    """
    _check_frame(df)
    if not 0 <= missing_threshold <= 1 or not 0 < high_cardinality <= 1:
        raise ValueError("Thresholds must be fractions in [0, 1] (high_cardinality > 0).")
    if target is not None and target not in df:
        raise KeyError(target)
    rows = []
    n = len(df)

    def add(severity, column, issue, count, advice):
        rows.append(dict(severity=severity, column=column, issue=issue,
                         count=int(count), fraction=count / n if n else np.nan,
                         recommendation=advice))

    if not n:
        add("error", None, "empty", 0, "Load observations before analysis or fitting.")
    for col in df:
        s = df[col]
        missing = int(s.isna().sum())
        if missing:
            add("error" if col == target or missing / n >= missing_threshold else "warning",
                col, "missing", missing, "Review missing labels." if col == target else
                "Investigate missingness; fit impute_missing on training data only.")
        distinct = s.nunique(dropna=True)
        if n and distinct <= 1:
            add("warning", col, "constant", n - missing,
                "Check whether this column carries information before using drop_constant.")
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_complex_dtype(s):
            infinite = int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum())
            if infinite:
                add("error", col, "infinite", infinite, "Resolve invalid divisions or replace infinities explicitly.")
        if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
            strings = s.dropna().map(lambda x: isinstance(x, str))
            if len(strings) and strings.all():
                text = s.astype("string")
                blanks = int(text.str.strip().eq("").sum())
                padded = int(text.ne(text.str.strip()).sum())
                if blanks:
                    add("warning", col, "blank_strings", blanks, "Replace blank strings with missing values before imputation.")
                if padded:
                    add("info", col, "whitespace", padded, "Strip surrounding whitespace before grouping or encoding.")
                if len(strings) >= 20 and distinct / len(strings) >= high_cardinality:
                    add("info", col, "high_cardinality", distinct,
                        "Inspect for identifiers or free text; avoid blindly one-hot encoding.")
                convertible = pd.to_numeric(text, errors="coerce").notna().sum()
                if len(strings) and convertible / len(strings) >= .95:
                    add("info", col, "numeric_strings", convertible, "Consider convert_columns(to='numeric') after checking leading zeros.")
        if target is not None and col != target and s.equals(df[target]):
            add("error", col, "target_copy", n, "Remove the target copy from model inputs; it leaks labels.")
    duplicates = int(df.duplicated().sum())
    if duplicates:
        add("warning", None, "duplicates", duplicates, "Check record identity before calling drop_duplicates.")
    return pd.DataFrame(rows, columns=["severity", "column", "issue", "count", "fraction", "recommendation"])


@dataclass(frozen=True)
class DataSchema:
    """Reference schema returned by :func:`infer_schema`.

    Attributes
    ----------
    columns : dict
        Mapping from column label to dtype string, nullable flag and optional
        observed category tuple. It is a snapshot, not a fitted preprocessor.
    """
    columns: dict[Any, dict]


def infer_schema(df: pd.DataFrame, *, category_limit: int = 50) -> DataSchema:
    """Capture column types, nullability and small categorical vocabularies.

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
    """
    _check_frame(df)
    if not isinstance(category_limit, int) or category_limit < 0:
        raise ValueError("category_limit must be a nonnegative integer.")
    rules = {}
    for c in df:
        s = df[c]
        categories = None
        if not pd.api.types.is_numeric_dtype(s) and 0 < s.nunique() <= category_limit:
            categories = tuple(s.dropna().unique())
        rules[c] = dict(dtype=str(s.dtype), nullable=bool(s.isna().any()), categories=categories)
    return DataSchema(rules)


def validate_schema(df: pd.DataFrame, schema: DataSchema, *, allow_extra: bool = False,
                    check_categories: bool = True) -> pd.DataFrame:
    """Compare a new batch to a reference schema without coercing its values.

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
    """
    _check_frame(df)
    if not isinstance(schema, DataSchema):
        raise TypeError("schema must be a DataSchema from infer_schema.")
    rows = []
    def add(c, issue, expected, actual):
        rows.append(dict(column=c, issue=issue, expected=expected, actual=actual))
    for c, rule in schema.columns.items():
        if c not in df:
            add(c, "missing_column", rule["dtype"], None)
            continue
        s = df[c]
        if str(s.dtype) != rule["dtype"]:
            add(c, "dtype", rule["dtype"], str(s.dtype))
        if not rule["nullable"] and s.isna().any():
            add(c, "unexpected_null", 0, int(s.isna().sum()))
        if check_categories and rule["categories"] is not None:
            unseen = s.notna() & ~s.isin(rule["categories"])
            if unseen.any():
                add(c, "unseen_category", rule["categories"], tuple(s[unseen].unique()))
    if not allow_extra:
        for c in df.columns.difference(list(schema.columns), sort=False):
            add(c, "extra_column", None, str(df[c].dtype))
    return pd.DataFrame(rows, columns=["column", "issue", "expected", "actual"])

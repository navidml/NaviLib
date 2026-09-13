"""Time-ordered splits and features with explicit historical boundaries."""

from numbers import Integral

import numpy as np
import pandas as pd

from .quality import _check_frame

__all__ = ["temporal_split", "add_lag_features", "add_rolling_features"]


def _ordered(df, time, group_by=None):
    _check_frame(df)
    groups = [group_by] if isinstance(group_by, str) else list(group_by or [])
    keys = groups + [time]
    missing = [c for c in keys if c not in df]
    if missing:
        raise KeyError(f"Missing ordering columns: {missing}")
    if df[keys].isna().any().any():
        raise ValueError("Ordering and grouping columns must not contain missing values.")
    if not (pd.api.types.is_datetime64_any_dtype(df[time]) or
            pd.api.types.is_numeric_dtype(df[time])):
        raise TypeError("time must be datetime or numeric; parse strings with pd.to_datetime first.")
    if df.duplicated(keys).any():
        raise ValueError("Timestamps must be unique within each group; aggregate ties explicitly.")
    positions = pd.DataFrame({"position": np.arange(len(df))})
    for i, c in enumerate(keys):
        positions[f"key{i}"] = df[c].to_numpy()
    order = positions.sort_values([f"key{i}" for i in range(len(keys))], kind="stable")["position"].to_numpy()
    return df.iloc[order].reset_index(drop=True).copy(), order, groups


def temporal_split(df: pd.DataFrame, time: str, *, test_size: float = .2,
                    gap: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on distinct timestamps, reserving the newest times for testing.

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
    """
    _check_frame(df)
    if not 0 < test_size < 1 or not isinstance(gap, Integral) or gap < 0:
        raise ValueError("test_size must be in (0, 1) and gap a nonnegative integer.")
    if df[time].isna().any():
        raise ValueError("time must not contain missing values.")
    if not (pd.api.types.is_numeric_dtype(df[time]) or pd.api.types.is_datetime64_any_dtype(df[time])):
        raise TypeError("time must be datetime or numeric.")
    times = df[time].drop_duplicates().sort_values().to_numpy()
    n_test = int(np.ceil(len(times) * test_size))
    cut = len(times) - n_test
    if cut - gap < 1:
        raise ValueError("Not enough timestamps for a nonempty train/test split and gap.")
    train = df[df[time].isin(times[:cut - gap])].sort_values(time, kind="stable").copy()
    test = df[df[time].isin(times[cut:])].sort_values(time, kind="stable").copy()
    return train, test


def _feature_setup(df, columns, time, group_by):
    columns = [columns] if isinstance(columns, str) else list(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns must be a nonempty sequence of unique names.")
    ordered, order, groups = _ordered(df, time, group_by)
    for c in columns:
        if c not in df:
            raise KeyError(c)
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"{c!r} must be numeric.")
    return columns, ordered, order, groups


def _emit(df, ordered, order, generated):
    overlap = set(generated).intersection(df.columns)
    if overlap:
        raise ValueError(f"Generated feature names already exist: {sorted(overlap)}")
    result = df.copy()
    inverse = np.argsort(order)
    for c in generated:
        result[c] = ordered[c].iloc[inverse].to_numpy()
    return result


def add_lag_features(df: pd.DataFrame, columns, *, time: str, lags=(1,),
                     group_by=None) -> pd.DataFrame:
    """Add previous-observation features, independently within each entity.

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
    """
    columns, ordered, order, groups = _feature_setup(df, columns, time, group_by)
    lags = list(lags)
    if not lags or any(not isinstance(k, Integral) or k <= 0 for k in lags) or len(set(lags)) != len(lags):
        raise ValueError("lags must contain unique positive integers.")
    generated = []
    for c in columns:
        series = ordered.groupby(groups, sort=False, observed=True)[c] if groups else ordered[c]
        for lag in lags:
            name = f"{c}_lag_{lag}"
            ordered[name] = series.shift(lag)
            generated.append(name)
    return _emit(df, ordered, order, generated)


def add_rolling_features(df: pd.DataFrame, columns, *, time: str, windows=(3, 7),
                         statistics=("mean", "std"), group_by=None,
                         min_periods: int = 1) -> pd.DataFrame:
    """Summarize strictly earlier observations with shifted rolling windows.

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
    """
    columns, ordered, order, groups = _feature_setup(df, columns, time, group_by)
    windows, statistics = list(windows), list(statistics)
    if not windows or any(not isinstance(w, Integral) or w < 1 for w in windows):
        raise ValueError("windows must contain positive integers.")
    if len(set(windows)) != len(windows) or len(set(statistics)) != len(statistics):
        raise ValueError("windows and statistics must be unique.")
    if not isinstance(min_periods, Integral) or not 1 <= min_periods <= min(windows):
        raise ValueError("min_periods must be between 1 and the smallest window.")
    if not statistics or set(statistics) - {"mean", "std", "min", "max", "median", "sum"}:
        raise ValueError("Unsupported rolling statistic.")
    generated = []
    for c in columns:
        for window in windows:
            for stat in statistics:
                name = f"{c}_rolling_{stat}_{window}"
                def compute(s):
                    return s.shift(1).rolling(window, min_periods=min_periods).agg(stat)
                ordered[name] = (ordered.groupby(groups, sort=False, observed=True)[c].transform(compute)
                                 if groups else compute(ordered[c]))
                generated.append(name)
    return _emit(df, ordered, order, generated)

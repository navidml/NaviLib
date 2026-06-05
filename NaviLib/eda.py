"""
distribution_analysis_lib
~~~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive library for distribution analysis, numeric column profiling,
and categorical value counts visualization.

Modules included:
- analyze_numeric_distribution : Comprehensive statistical and visual analysis
- value_counts_plot : Plot value counts for categorical or one-hot encoded columns

Author: Generated
Version: 1.0
"""

from typing import Dict, Optional, Tuple, Union, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import kurtosis, normaltest, skew


# ==============================================================================
# 1. NUMERIC DISTRIBUTION ANALYSIS
# ==============================================================================

def analyze_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    density: bool = False,
    kde: bool = True,
    figsize: Tuple[int, int] = (16, 10),
    plot: bool = True,
    coerce_numeric: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Comprehensive statistical and visual analysis of a numeric column.

    Features
    --------
    - Automatic numeric coercion (optional)
    - Summary statistics
    - Skewness & kurtosis
    - Normality test (D'Agostino-Pearson)
    - Histogram + KDE
    - Boxplot
    - Violin plot
    - QQ plot

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Target numeric column.
    bins : int, default=30
        Number of histogram bins.
    density : bool, default=False
        Normalize histogram to probability density.
    kde : bool, default=True
        Overlay KDE curve.
    figsize : tuple, default=(16, 10)
        Figure size.
    plot : bool, default=True
        Whether to show plots.
    coerce_numeric : bool, default=True
        If True, attempts conversion using pd.to_numeric().

    Returns
    -------
    Dict[str, Optional[float]]
        Structured distribution report including:
        - count, mean, median, std, min, max
        - skewness, kurtosis
        - normality_p_value
        - skew_interpretation, kurtosis_interpretation, normality_interpretation
    """
    # Validation
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    series = df[column]

    if coerce_numeric:
        series = pd.to_numeric(series, errors="coerce")

    if not np.issubdtype(series.dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric.")

    data = series.dropna()

    if len(data) < 3:
        raise ValueError("Not enough valid numeric data to analyze.")

    # Core Statistics
    mean = data.mean()
    median = data.median()
    std = data.std()
    minimum = data.min()
    maximum = data.max()

    skewness = skew(data)
    kurt = kurtosis(data)

    # Normality test
    if len(data) >= 8:
        stat, p_value = normaltest(data)
    else:
        p_value = np.nan

    # Interpretation
    if abs(skewness) < 0.5:
        skew_text = "approximately_symmetric"
    elif skewness > 0:
        skew_text = "right_skewed"
    else:
        skew_text = "left_skewed"

    if kurt > 0:
        kurt_text = "heavy_tails"
    elif kurt < 0:
        kurt_text = "light_tails"
    else:
        kurt_text = "normal_like"

    if np.isnan(p_value):
        normality = "test_not_applicable"
    else:
        normality = "not_normal" if p_value < 0.05 else "possibly_normal"

    # Visualization
    if plot:
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Histogram
        sns.histplot(
            data,
            bins=bins,
            kde=kde,
            stat="density" if density else "count",
            color="steelblue",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title(f"Histogram + KDE: {column}")

        # Boxplot
        sns.boxplot(x=data, ax=axes[0, 1], color="orange")
        axes[0, 1].set_title("Boxplot")

        # Violin
        sns.violinplot(x=data, ax=axes[1, 0], color="lightgreen")
        axes[1, 0].set_title("Violin Plot")

        # QQ Plot
        stats.probplot(data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("QQ Plot")

        plt.tight_layout()
        plt.show()

    # Structured Output
    results = {
        "count": int(len(data)),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "min": float(minimum),
        "max": float(maximum),
        "skewness": float(skewness),
        "kurtosis": float(kurt),
        "normality_p_value": float(p_value) if not np.isnan(p_value) else None,
        "skew_interpretation": skew_text,
        "kurtosis_interpretation": kurt_text,
        "normality_interpretation": normality,
    }

    return results


# ==============================================================================
# 2. VALUE COUNTS PLOT
# ==============================================================================

def value_counts_plot(
    df: pd.DataFrame,
    column: str,
    normalize: bool = False,
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (8, 5),
    color: str = "#4C72B0",
    title: Optional[str] = None,
) -> pd.Series:
    """
    Plot value counts for a categorical column or its one-hot encoded (OHE) version.

    Supports:
        - Direct categorical columns
        - One-hot encoded columns with prefix (column_value)
        - Normalized percentages
        - Limiting to top N values

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name or OHE prefix to analyze.
    normalize : bool, default=False
        Whether to normalize counts (show percentages).
    top_n : int, optional
        Show only top N categories. Default is None (show all).
    figsize : tuple, default=(8, 5)
        Size of the output figure.
    color : str, default="#4C72B0"
        Color of bars.
    title : str, optional
        Custom plot title. If None, an automatic title will be generated.

    Returns
    -------
    pd.Series
        The value counts used for plotting.
    """
    col_lower = column.lower()
    columns_lower_map = {c.lower(): c for c in df.columns}

    # Case A: Direct column exists
    is_normal_column = col_lower in columns_lower_map

    # Case B: One-hot encoded columns
    ohe_cols = [
        c for c in df.columns
        if c.lower().startswith(col_lower + "_")
        and c.lower() != col_lower + "_"
    ]

    # A) Normal column mode
    if is_normal_column:
        real_col = columns_lower_map[col_lower]
        counts = df[real_col].value_counts(normalize=normalize)

    # B) OHE prefix mode
    elif len(ohe_cols) > 0:
        # Remove broken columns like 'employment_'
        ohe_cols = [c for c in ohe_cols if not c.endswith("_")]

        counts = df[ohe_cols].sum(axis=0)

        if normalize:
            counts = counts / counts.sum()

        counts.index = counts.index.str.replace(column + "_", "", regex=False)

    # C) Column does not exist
    else:
        raise ValueError(
            f"Column '{column}' not found, and no one-hot encoded columns "
            f"with prefix '{column}_' exist."
        )

    # Limit top N
    if top_n is not None:
        counts = counts.sort_values(ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    counts.plot(kind="bar", color=color)

    plt.ylabel("Percentage" if normalize else "Count")
    plt.xlabel(column)
    plt.title(title if title else f"Value Counts for '{column}'")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return counts


# ==============================================================================
# 3. MODULE EXPORTS
# ==============================================================================

__all__ = [
    "analyze_numeric_distribution",
    "value_counts_plot",
]
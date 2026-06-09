"""
navdata - A professional data preprocessing library.

This module provides utilities for:
- Feature transformation (log, box-cox, yeo-johnson, etc.)
- Outlier detection and analysis
- Categorical encoding (one-hot, count, frequency, label)
- Feature scaling (standard, minmax, robust, maxabs, log, sqrt)

Author: Navid
"""

import os
from typing import Any, Dict, Tuple, Union, Optional, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, zscore, boxcox
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)


# =============================================================================
# Auto Best Transformation
# =============================================================================

def auto_best_transformation(
    df: pd.DataFrame,
    column: str,
    methods: Optional[List[str]] = None,
    inplace: bool = False,
    return_df: bool = True,
) -> Union[Tuple[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]:
    """
    Automatically evaluate multiple numeric transformations and select the
    one that minimizes absolute skewness.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the numeric column to transform.
    methods : Optional[List[str]], default=None
        Transformations to test.
        Supported: ["none", "log", "cuberoot", "boxcox", "yeojohnson"]
        Default = all methods.
    inplace : bool, default=False
        If True, transformation is added to original df.
        If False, a copy is created.
    return_df : bool, default=True
        Whether to return the dataframe.

    Returns
    -------
    Union[Tuple[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]
        Data + report   (if return_df=True)
        Report only     (if return_df=False)

    Raises
    ------
    KeyError
        If column not found in dataframe.
    ValueError
        If methods contain unsupported values, column contains NaN,
        or no valid transformations could be applied.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe.")

    if methods is None:
        methods = ["none", "log", "cuberoot", "boxcox", "yeojohnson"]

    allowed_methods = {"none", "log", "cuberoot", "boxcox", "yeojohnson"}
    invalid_methods = set(methods) - allowed_methods
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}")

    data = df if inplace else df.copy()

    x = pd.to_numeric(data[column], errors="raise")

    if x.isna().any():
        raise ValueError(f"Column '{column}' contains NaN values. Handle them first.")

    results: Dict[str, float] = {}
    transformed_data: Dict[str, np.ndarray] = {}

    if "none" in methods:
        results["none"] = float(skew(x))
        transformed_data["none"] = x.values

    if "log" in methods:
        if (x >= 0).all():
            t = np.log1p(x)
            results["log"] = float(skew(t))
            transformed_data["log"] = t

    if "cuberoot" in methods:
        t = np.cbrt(x)
        results["cuberoot"] = float(skew(t))
        transformed_data["cuberoot"] = t

    if "boxcox" in methods:
        if (x > 0).all():
            t, lam = boxcox(x)
            results["boxcox"] = float(skew(t))
            transformed_data["boxcox"] = t

    if "yeojohnson" in methods:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        t = pt.fit_transform(x.values.reshape(-1, 1)).flatten()
        results["yeojohnson"] = float(skew(t))
        transformed_data["yeojohnson"] = t

    if not results:
        raise ValueError("No valid transformations could be applied.")

    best_method = min(results, key=lambda k: abs(results[k]))
    new_col = f"{column}_{best_method}"

    if new_col in data.columns:
        raise ValueError(f"Column '{new_col}' already exists.")

    data[new_col] = transformed_data[best_method]

    report = {
        "column": column,
        "best_method": best_method,
        "best_skew": results[best_method],
        "all_results": results,
        "new_column": new_col,
    }

    if return_df:
        return data, report
    else:
        return report


# =============================================================================
# Transformation Functions
# =============================================================================

def transform_column(
    df: pd.DataFrame,
    column: str,
    method: Literal["log", "cuberoot", "reciprocal", "boxcox", "yeojohnson"] = "log",
    new_column: Optional[str] = None,
    return_lambda: bool = False,
    inplace: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float]]:
    """
    Apply a mathematical transformation to a DataFrame column.

    Supported transformations include common skewness-reducing
    transformations used in statistical modeling and machine learning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the column to transform.
    method : Literal["log", "cuberoot", "reciprocal", "boxcox", "yeojohnson"], default="log"
        Transformation method.
    new_column : Optional[str], default=None
        Name of the output column.
        Default: "<column>_<method>".
    return_lambda : bool, default=False
        If True, return the transformation lambda parameter
        for Box-Cox or Yeo-Johnson.
    inplace : bool, default=False
        If True, modify the input dataframe in-place.
        Otherwise a copy is returned.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, float]]
        - DataFrame containing the transformed column.
        - (DataFrame, lambda_value) if return_lambda=True
          for Box-Cox or Yeo-Johnson transformations.

    Raises
    ------
    KeyError
        If the specified column does not exist.
    ValueError
        If transformation assumptions are violated.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    method = method.lower()
    valid_methods = {"log", "cuberoot", "reciprocal", "boxcox", "yeojohnson"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of: {', '.join(valid_methods)}")

    if new_column is None:
        new_column = f"{column}_{method}"

    data = df if inplace else df.copy()
    x = data[column].astype(float)

    if method == "log":
        if (x < 0).any():
            raise ValueError("Log transformation requires non-negative values")
        data[new_column] = np.log1p(x)

    elif method == "cuberoot":
        data[new_column] = np.cbrt(x)

    elif method == "reciprocal":
        if (x == 0).any():
            raise ValueError("Reciprocal transformation cannot handle zeros")
        data[new_column] = 1.0 / x

    elif method == "boxcox":
        if (x <= 0).any():
            raise ValueError("Box-Cox requires strictly positive values")
        transformed, lam = boxcox(x)
        data[new_column] = transformed
        if return_lambda:
            return data, float(lam)

    elif method == "yeojohnson":
        pt = PowerTransformer(method="yeo-johnson")
        transformed = pt.fit_transform(x.values.reshape(-1, 1)).flatten()
        data[new_column] = transformed
        if return_lambda:
            return data, float(pt.lambdas_[0])

    return data


# =============================================================================
# Outlier Detection Functions
# =============================================================================

def analyze_outliers(
    df: pd.DataFrame,
    column: str,
    z_thresh: float = 3.0,
    plot: bool = True,
    figsize: Tuple[int, int] = (16, 10),
) -> Dict[str, Union[int, float]]:
    """
    Analyze outliers in a numerical column using IQR and Z-score methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the target column.
    column : str
        Numerical column to analyze.
    z_thresh : float, default=3.0
        Threshold for Z-score outlier detection.
    plot : bool, default=True
        If True, generate visualization plots.
    figsize : Tuple[int, int], default=(16, 10)
        Figure size for the visualization.

    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing:
        - iqr_lower_bound: Lower bound from IQR method
        - iqr_upper_bound: Upper bound from IQR method
        - iqr_outliers: Number of outliers detected by IQR
        - iqr_percent: Percentage of outliers from IQR
        - z_outliers: Number of outliers detected by Z-score
        - z_percent: Percentage of outliers from Z-score

    Raises
    ------
    KeyError
        If the specified column does not exist.
    ValueError
        If the column is not numerical or contains no valid data.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    data = df[column].dropna()

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"Column '{column}' must be numerical.")

    if len(data) == 0:
        raise ValueError("Column contains no valid (non-null) values.")

    n = len(data)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr

    iqr_mask = (data < lower_iqr) | (data > upper_iqr)
    iqr_count = int(iqr_mask.sum())
    iqr_percent = float((iqr_count / n) * 100)

    z_scores = np.abs(zscore(data))
    z_mask = z_scores > z_thresh
    z_count = int(z_mask.sum())
    z_percent = float((z_count / n) * 100)

    if plot:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        sns.boxplot(x=data, ax=axes[0, 0], color="skyblue")
        axes[0, 0].set_title(f"Boxplot: {column}")

        sns.histplot(data, bins=30, kde=True, ax=axes[0, 1], color="steelblue")
        axes[0, 1].axvline(lower_iqr, color="red", linestyle="--", label="IQR Lower")
        axes[0, 1].axvline(upper_iqr, color="red", linestyle="--", label="IQR Upper")
        axes[0, 1].legend()
        axes[0, 1].set_title("Histogram with IQR Bounds")

        axes[1, 0].scatter(range(n), data, alpha=0.6)
        axes[1, 0].axhline(lower_iqr, color="red", linestyle="--")
        axes[1, 0].axhline(upper_iqr, color="red", linestyle="--")
        axes[1, 0].set_title("Scatter Plot (Index vs Value)")

        sns.histplot(z_scores, bins=30, ax=axes[1, 1], color="orange")
        axes[1, 1].axvline(z_thresh, color="red", linestyle="--", label="Z Threshold")
        axes[1, 1].legend()
        axes[1, 1].set_title("Z-score Distribution")

        plt.tight_layout()
        plt.show()

    return {
        "iqr_lower_bound": float(lower_iqr),
        "iqr_upper_bound": float(upper_iqr),
        "iqr_outliers": iqr_count,
        "iqr_percent": iqr_percent,
        "z_outliers": z_count,
        "z_percent": z_percent,
    }


def save_outliers(
    outliers_df: pd.DataFrame,
    filepath: str,
    file_format: Literal["csv", "excel"] = "csv",
) -> str:
    """
    Save detected outliers to a file in CSV or Excel format.

    Parameters
    ----------
    outliers_df : pd.DataFrame
        DataFrame containing outlier records.
    filepath : str
        Destination file path including filename.
    file_format : Literal["csv", "excel"], default="csv"
        Output format.

    Returns
    -------
    str
        Absolute path to the saved file.

    Raises
    ------
    ValueError
        If file_format is invalid or DataFrame is empty.
    FileNotFoundError
        If the directory in 'filepath' does not exist.
    """
    if not isinstance(outliers_df, pd.DataFrame):
        raise ValueError("outliers_df must be a pandas DataFrame.")

    if outliers_df.empty:
        raise ValueError("outliers_df is empty; nothing to save.")

    file_format = file_format.lower()
    if file_format not in ("csv", "excel"):
        raise ValueError("file_format must be 'csv' or 'excel'.")

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if file_format == "csv":
        outliers_df.to_csv(filepath, index=False)
    else:
        outliers_df.to_excel(filepath, index=False)

    return os.path.abspath(filepath)


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: Literal["iqr", "zscore", "percentile", "mad", "isolationforest"] = "iqr",
    *,
    iqr_factor: float = 1.5,
    z_thresh: float = 3.0,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
    mad_thresh: float = 3.5,
    iso_fraction: float = 0.03,
    return_mask: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Remove outliers from a numerical column using flexible methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Numerical column for outlier removal.
    method : Literal["iqr", "zscore", "percentile", "mad", "isolationforest"], default="iqr"
        Outlier detection method.
    iqr_factor : float, default=1.5
        IQR multiplier for 'iqr' method.
    z_thresh : float, default=3.0
        Z-score threshold for 'zscore'.
    lower_percentile : float, default=0.01
        Lower bound percentile (for 'percentile').
    upper_percentile : float, default=0.99
        Upper bound percentile.
    mad_thresh : float, default=3.5
        Threshold for Median Absolute Deviation (MAD) method.
    iso_fraction : float, default=0.03
        Contamination fraction for IsolationForest.
    return_mask : bool, default=False
        If True, return (clean_df, mask) instead of just df.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]
        Cleaned DataFrame and optional boolean mask for removed rows.

    Raises
    ------
    KeyError
        If column not in df.
    ValueError
        If column is not numeric or method is invalid.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")

    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric.")

    data = df[column]
    mask = pd.Series(True, index=df.index)

    method = method.lower()

    if method == "iqr":
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        mask = (data >= lower) & (data <= upper)

    elif method == "zscore":
        z = np.abs(zscore(data.dropna()))
        z_full = pd.Series(np.nan, index=data.index)
        z_full.loc[data.dropna().index] = z
        mask = z_full <= z_thresh

    elif method == "percentile":
        lower = data.quantile(lower_percentile)
        upper = data.quantile(upper_percentile)
        mask = (data >= lower) & (data <= upper)

    elif method == "mad":
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / (mad if mad != 0 else 1e-9)
        mask = np.abs(modified_z) <= mad_thresh

    elif method == "isolationforest":
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(
            contamination=iso_fraction,
            random_state=42,
            n_estimators=200,
        )
        preds = iso.fit_predict(data.to_frame())
        mask = preds == 1

    else:
        raise ValueError("Invalid method.")

    cleaned_df = df[mask].copy()

    if return_mask:
        return cleaned_df, mask

    return cleaned_df


# =============================================================================
# Categorical Encoding Functions
# =============================================================================

def encode_categorical_features(
    df: pd.DataFrame,
    columns: List[str],
    encoding: Literal["onehot", "count", "frequency", "label"] = "onehot",
    separator: Optional[str] = None,
    min_freq: Optional[int] = None,
    min_percent: Optional[float] = None,
    low_freq_strategy: Literal["drop", "group"] = "drop",
    other_label: str = "Other",
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Encode categorical features using various encoding methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : List[str]
        List of categorical columns to encode.
    encoding : Literal["onehot", "count", "frequency", "label"], default="onehot"
        Encoding method.
    separator : Optional[str], default=None
        If provided, splits values by this separator for multi-label encoding.
    min_freq : Optional[int], default=None
        Minimum frequency threshold for one-hot encoding.
    min_percent : Optional[float], default=None
        Minimum percentage threshold (0-1) for one-hot encoding.
    low_freq_strategy : Literal["drop", "group"], default="drop"
        Strategy for handling low-frequency categories in one-hot encoding.
    other_label : str, default="Other"
        Label for grouped low-frequency categories.
    drop_original : bool, default=True
        Whether to drop original columns after encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded features.

    Raises
    ------
    ValueError
        If DataFrame is empty, encoding is invalid, or min_percent out of range.
    KeyError
        If specified columns not found in DataFrame.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if isinstance(columns, str):
        columns = [columns]

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    encoding = encoding.lower()
    valid_encodings = {"onehot", "count", "frequency", "label"}
    if encoding not in valid_encodings:
        raise ValueError(f"encoding must be one of: {', '.join(valid_encodings)}")

    if min_percent is not None and not (0 <= min_percent <= 1):
        raise ValueError("min_percent must be between 0 and 1.")

    df_copy = df.copy()
    encoded_parts = []

    for col in columns:
        series = df_copy[col].fillna("").astype(str)

        if separator:
            exploded = series.str.split(separator).explode().str.strip()
        else:
            exploded = series

        if encoding == "onehot":
            dummies = pd.get_dummies(exploded, prefix=col)
            dummies = dummies.groupby(level=0).max()

            low_freq_cols = []

            if min_freq is not None:
                low_freq_cols += [c for c in dummies.columns if dummies[c].sum() < min_freq]

            if min_percent is not None:
                threshold = len(df_copy) * min_percent
                low_freq_cols += [c for c in dummies.columns if dummies[c].sum() < threshold]

            low_freq_cols = list(set(low_freq_cols))

            if low_freq_strategy == "drop":
                dummies = dummies.drop(columns=low_freq_cols, errors="ignore")

            elif low_freq_strategy == "group" and low_freq_cols:
                dummies_other = dummies[low_freq_cols].max(axis=1)
                dummies_other = dummies_other.rename(f"{col}_{other_label}")
                dummies = dummies.drop(columns=low_freq_cols, errors="ignore")
                dummies = pd.concat([dummies, dummies_other], axis=1)

        elif encoding == "count":
            counts = exploded.value_counts()
            mapped = series.map(counts).fillna(0)
            dummies = pd.DataFrame({f"{col}_count": mapped})

        elif encoding == "frequency":
            freq = exploded.value_counts(normalize=True)
            mapped = series.map(freq).fillna(0)
            dummies = pd.DataFrame({f"{col}_freq": mapped})

        elif encoding == "label":
            unique_vals = sorted(exploded.unique())
            label_map = {val: idx for idx, val in enumerate(unique_vals)}
            mapped = series.map(label_map)
            dummies = pd.DataFrame({f"{col}_label": mapped})

        encoded_parts.append(dummies)

    encoded_df = pd.concat(encoded_parts, axis=1)

    if drop_original:
        df_copy = df_copy.drop(columns=columns)

    return pd.concat([df_copy, encoded_df], axis=1)


# =============================================================================
# Feature Scaling Functions
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: Literal["standard", "minmax", "robust", "maxabs", "log", "sqrt", "none"] = "standard",
    fitted_scaler: Optional[
        Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]
    ] = None,
    return_scaler: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, object]]:
    """
    Scale or transform numerical features using a variety of methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : Optional[List[str]], default=None
        Columns to scale. If None, all numeric columns are selected.
    method : Literal["standard", "minmax", "robust", "maxabs", "log", "sqrt", "none"], default="standard"
        Scaling or transformation method.
    fitted_scaler : Optional[Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]], default=None
        Pre-fitted scaler for transforming new data.
        Only valid for: {'standard', 'minmax', 'robust', 'maxabs'}.
    return_scaler : bool, default=False
        Whether to return the fitted scaler.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, object]]
        - DataFrame with scaled or transformed columns.
        - (DataFrame, scaler) if return_scaler=True.

    Raises
    ------
    ValueError
        If DataFrame is empty, method is invalid, or method requires non-negative values.
    KeyError
        If specified columns do not exist.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    method = method.lower()
    valid_methods = {"standard", "minmax", "robust", "maxabs", "log", "sqrt", "none"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    missing = [c for c in columns if c not in df_copy.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    non_numeric = df_copy[columns].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    x = df_copy[columns]

    if method in {"log", "sqrt"}:
        if (x < 0).any().any():
            raise ValueError(f"Method '{method}' requires non-negative values.")

    scaler = None

    if method == "standard":
        scaler = fitted_scaler or StandardScaler()
        if fitted_scaler:
            x_scaled = scaler.transform(x)
        else:
            x_scaled = scaler.fit_transform(x)

    elif method == "minmax":
        scaler = fitted_scaler or MinMaxScaler()
        if fitted_scaler:
            x_scaled = scaler.transform(x)
        else:
            x_scaled = scaler.fit_transform(x)

    elif method == "robust":
        scaler = fitted_scaler or RobustScaler()
        if fitted_scaler:
            x_scaled = scaler.transform(x)
        else:
            x_scaled = scaler.fit_transform(x)

    elif method == "maxabs":
        scaler = fitted_scaler or MaxAbsScaler()
        if fitted_scaler:
            x_scaled = scaler.transform(x)
        else:
            x_scaled = scaler.fit_transform(x)

    elif method == "log":
        x_scaled = np.log1p(x)

    elif method == "sqrt":
        x_scaled = np.sqrt(x)

    else:
        x_scaled = x.values

    df_copy[columns] = x_scaled

    if return_scaler:
        return df_copy, scaler
    return df_copy


from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Literal, Union, Tuple


def balance_data(
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    method: Literal["oversample", "undersample", "smote"] = "oversample",
    random_state: int = 42
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Balance imbalanced datasets using oversampling, undersampling, or SMOTE.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target labels
    method : {"oversample", "undersample", "smote"}, default="oversample"
        Balancing strategy:
        - "oversample": Random oversampling of minority class
        - "undersample": Random undersampling of majority class
        - "smote": Synthetic Minority Over-sampling Technique
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    X_balanced : pd.DataFrame or np.ndarray
        Balanced feature matrix (same type as input)
    y_balanced : pd.Series or np.ndarray
        Balanced target labels (same type as input)
    
    Raises
    ------
    ValueError
        If an invalid method is provided
    
    Examples
    --------
    >>> X_bal, y_bal = balance_data(X, y, method="smote", random_state=42)
    """
    samplers = {
        "oversample": RandomOverSampler(random_state=random_state),
        "undersample": RandomUnderSampler(random_state=random_state),
        "smote": SMOTE(random_state=random_state)
    }
    
    if method not in samplers:
        raise ValueError(
            f"Invalid method: '{method}'. Choose from {list(samplers.keys())}"
        )
    
    sampler = samplers[method]
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    # Preserve column names for DataFrames
    if isinstance(X, pd.DataFrame):
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    if isinstance(y, pd.Series):
        y_balanced = pd.Series(y_balanced, name=y.name)
    
    return X_balanced, y_balanced


def find_best_k_auto(
    df, 
    features='auto',  # 'auto', 'all', or list of column names
    exclude=None,     # list of columns to exclude
    k_range=range(2, 11),
    scale=True,
    visualize_pca=True,
    random_state=42
):
    """
    Intelligent K-Means analyzer that automatically detects features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your dataframe
    features : 'auto', 'all', or list
        - 'auto': automatically select numeric columns
        - 'all': use all numeric columns
        - list: specify column names
    exclude : list
        Column names to exclude (e.g., ['ID', 'Date'])
    k_range : range
        Range of k to test
    scale : bool
        Standardize features
    visualize_pca : bool
        Show PCA visualization of clusters
    random_state : int
        Reproducibility
    
    Returns:
    --------
    dict with results and recommendations
    """
    
    # ========== 1. AUTO-DETECT FEATURES ==========
    print("\n" + "="*70)
    print(" K-MEANS CLUSTERING ANALYZER".center(70))
    print("="*70)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if features == 'auto':
        selected_features = numeric_cols
        print(f" Auto-selected {len(selected_features)} numeric features:")
    elif features == 'all':
        selected_features = df.columns.tolist()
        print(f" Using all {len(selected_features)} columns:")
    else:
        selected_features = features
        print(f" Using {len(selected_features)} specified features:")
    
    # Exclude columns if specified
    if exclude:
        selected_features = [col for col in selected_features if col not in exclude]
        print(f"   (Excluded: {exclude})")
    
    # Show feature list (first 10)
    for i, col in enumerate(selected_features[:10]):
        print(f"   {i+1}. {col}")
    if len(selected_features) > 10:
        print(f"   ... and {len(selected_features)-10} more")
    
    # Check if we have enough features
    if len(selected_features) < 2:
        raise ValueError(f"Need at least 2 features. Found only {len(selected_features)}")
    
    # Extract data
    X = df[selected_features].copy().values
    
    # ========== 2. DATA QUALITY REPORT ==========
    print("\n" + "-"*70)
    print(" DATA QUALITY REPORT")
    print("-"*70)
    print(f"Total samples    : {len(X):,}")
    print(f"Total features   : {len(selected_features)}")
    print(f"Missing values   : {np.isnan(X).sum()}")
    print(f"Feature ranges   :")
    for i, col in enumerate(selected_features[:5]):
        print(f"   {col}: [{df[col].min():.2f} - {df[col].max():.2f}]")
    
    # ========== 3. SCALE DATA ==========
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"\n Data scaled using StandardScaler")
    else:
        X_scaled = X
        print(f"\n⚠️  No scaling applied")
    
    # ========== 4. CALCULATE METRICS ==========
    print("\n" + "-"*70)
    print("🔄 CALCULATING METRICS".center(70))
    print("-"*70)
    
    inertias = []
    silhouette_scores = []
    k_list = list(k_range)
    
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        print(f"\r   Processing k={k}/{max(k_list)}", end="", flush=True)
    
    print("\n Done!")
    
    # ========== 5. FIND BEST K ==========
    # Elbow method
    if len(inertias) > 2:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmin(diffs2) + 1 if len(diffs2) > 0 else 0
        best_k_elbow = k_list[elbow_idx + 1]
    else:
        best_k_elbow = k_list[np.argmin(inertias)]
    
    # Silhouette method
    best_k_sil = k_list[np.argmax(silhouette_scores)]
    
    # Consensus (if both agree, use that; otherwise prefer silhouette)
    if best_k_elbow == best_k_sil:
        best_k_final = best_k_elbow
        consensus = "✅ PERFECT AGREEMENT"
    else:
        best_k_final = best_k_sil  # Silhouette is usually more reliable
        consensus = "DISAGREEMENT (using silhouette)"
    
    # ========== 6. FINAL MODEL ==========
    kmeans_final = KMeans(n_clusters=best_k_final, random_state=random_state, n_init=10)
    final_labels = kmeans_final.fit_predict(X_scaled)
    
    # Add cluster labels to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = final_labels
    
    # ========== 7. VISUALIZATIONS ==========
    # Plot 1: Elbow and Silhouette
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Means Clustering Analysis', fontsize=16, fontweight='bold')
    
    # Elbow plot
    axes[0, 0].plot(k_list, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axvline(best_k_elbow, color='red', linestyle='--', alpha=0.7, label=f'Elbow: k={best_k_elbow}')
    axes[0, 0].axvline(best_k_sil, color='green', linestyle=':', alpha=0.7, label=f'Silhouette: k={best_k_sil}')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette plot
    axes[0, 1].plot(k_list, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[0, 1].axvline(best_k_final, color='red', linestyle='--', label=f'Best k = {best_k_final}')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title(f'Silhouette Score (Best: {max(silhouette_scores):.3f})', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Silhouette plot for best k
    sil_samples = silhouette_samples(X_scaled, final_labels)
    y_lower = 10
    axes[1, 0].set_title(f'Silhouette Plot (k={best_k_final})', fontweight='bold')
    for i in range(best_k_final):
        cluster_sil = sil_samples[final_labels == i]
        cluster_sil.sort()
        y_upper = y_lower + len(cluster_sil)
        axes[1, 0].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7)
        axes[1, 0].text(-0.05, y_lower + len(cluster_sil)/2, str(i))
        y_lower = y_upper + 10
    axes[1, 0].axvline(np.mean(sil_samples), color='red', linestyle='--', label=f'Average: {np.mean(sil_samples):.3f}')
    axes[1, 0].set_xlabel('Silhouette Coefficient')
    axes[1, 0].set_ylabel('Cluster')
    axes[1, 0].legend()
    
    # PCA Visualization (if features >= 2)
    if visualize_pca and len(selected_features) >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='tab10', alpha=0.6, s=50)
        axes[1, 1].scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
                          c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1, 1].set_title(f'PCA Visualization (k={best_k_final})', fontweight='bold')
        axes[1, 1].legend()
        plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # ========== 8. CLUSTER ANALYSIS ==========
    print("\n" + "="*70)
    print("🎯 CLUSTER ANALYSIS RESULTS".center(70))
    print("="*70)
    
    print(f"\n📊 BEST K BY DIFFERENT METHODS:")
    print(f"   • Elbow Method      : k = {best_k_elbow}")
    print(f"   • Silhouette Method : k = {best_k_sil}")
    print(f"   • {consensus}")
    print(f"\n✨ RECOMMENDED K = {best_k_final}")
    print(f"   Silhouette Score: {np.mean(sil_samples):.4f}")
    
    # Cluster sizes
    print(f"\n📦 CLUSTER SIZES:")
    cluster_sizes = pd.Series(final_labels).value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        percentage = size / len(final_labels) * 100
        bar = '█' * int(percentage / 2)
        print(f"   Cluster {cluster}: {size:4d} samples ({percentage:5.1f}%) {bar}")
    
    # Feature importance per cluster
    print(f"\n🔍 FEATURE PROFILE BY CLUSTER (top features):")
    cluster_profile = pd.DataFrame(X_scaled, columns=selected_features)
    cluster_profile['Cluster'] = final_labels
    
    for cluster in range(best_k_final):
        cluster_data = cluster_profile[cluster_profile['Cluster'] == cluster]
        means = cluster_data[selected_features].mean()
        top_features = means.abs().nlargest(3).index.tolist()
        print(f"\n   Cluster {cluster}:")
        for feat in top_features:
            val = means[feat]
            direction = "HIGH" if val > 0 else "LOW"
            print(f"      • {feat}: {val:+.2f} ({direction})")
    
    # ========== 9. RETURN RESULTS ==========
    return {
        'best_k': best_k_final,
        'best_k_elbow': best_k_elbow,
        'best_k_silhouette': best_k_sil,
        'silhouette_score': np.mean(sil_samples),
        'silhouette_scores': silhouette_scores,
        'inertias': inertias,
        'features_used': selected_features,
        'labels': final_labels,
        'df_with_clusters': df_with_clusters,
        'cluster_sizes': cluster_sizes.to_dict(),
        'kmeans_model': kmeans_final
    }
# =============================================================================
# Module Metadata
# =============================================================================

__version__ = "1.0.0"
__author__ = "Navid"
__all__ = [
    "auto_best_transformation",
    "transform_column",
    "analyze_outliers",
    "save_outliers",
    "remove_outliers",
    "encode_categorical_features",
    "scale_features",
    "find_best_k_auto"
]

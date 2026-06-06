"""
feature_engineering_lib
~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive library for feature engineering, missing value handling,
duplicate detection, column cleaning, and feature selection.

Author: Generated
License: MIT
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor


# ================================================================
#  L1 FEATURE SELECTION (LASSO / LOGISTIC REGRESSION)
# ================================================================
def l1_feature_selection(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "classification",
    C: float = 0.1,
    alpha: float = 0.001,
    top_k: Optional[int] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    use_saga: bool = False
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Select features using L1 regularization (Lasso for regression,
    LogisticRegression with L1 penalty for classification).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str
        Target column name.
    problem_type : {"classification", "regression"}, default="classification"
        Type of problem.
    C : float, default=0.1
        Inverse regularization strength (LogisticRegression).
    alpha : float, default=0.001
        Regularization strength (Lasso).
    top_k : int, optional
        Number of top features to select.
    random_state : int, default=42
        Random seed.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all CPUs).
    use_saga : bool, default=False
        Use SAGA solver for LogisticRegression (faster for large datasets).

    Returns
    -------
    feature_importance : pd.DataFrame
        DataFrame with feature names, coefficients, and absolute coefficients.
    selected_features : List[str]
        List of selected feature names.
    report : Dict
        Summary report including hyperparameters and pipeline.
    """
    # Extract features and target efficiently
    X = df.drop(columns=[target])
    y = df[target]
    
    # Remove NaN from target and corresponding rows
    valid_mask = y.notna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    # Pre-allocate column type lists more efficiently
    numeric_cols = X_clean.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_cols = X_clean.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    
    # Build transformers with optimized settings
    transformers = []
    
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ))
    
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(
                    drop="first" if len(categorical_cols) > 1 else None,
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=0.01  # Remove rare categories to reduce dimensionality
                ))
            ]),
            categorical_cols
        ))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=n_jobs
    )
    
    # Choose optimal solver and model
    if problem_type == "classification":
        if use_saga and X_clean.shape[0] * X_clean.shape[1] > 1e6:
            solver = "saga"
        else:
            solver = "liblinear"
        
        model = LogisticRegression(
            penalty="l1",
            C=C,
            solver=solver,
            random_state=random_state,
            max_iter=2000,
            n_jobs=n_jobs,
            warm_start=False  # Disable for better performance in L1
        )
    else:
        model = Lasso(
            alpha=alpha,
            random_state=random_state,
            max_iter=5000,
            warm_start=False,
            selection='random'  # Faster convergence for large datasets
        )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ], memory=None)  # Disable memory caching for this specific case
    
    # Fit model
    pipeline.fit(X_clean, y_clean)
    
    # Extract feature names more efficiently
    pre = pipeline.named_steps["preprocessor"]
    try:
        feature_names = pre.get_feature_names_out()
        # Vectorized string replacement instead of list comprehension
        feature_names = np.char.replace(feature_names, "num__", "")
        feature_names = np.char.replace(feature_names, "cat__", "")
        feature_names = feature_names.tolist()
    except Exception:
        feature_names = X_clean.columns.tolist()
    
    # Get coefficients efficiently
    coef = pipeline.named_steps["model"].coef_
    if coef.ndim > 1:
        coef = coef.ravel()  # More efficient than coef[0]
    
    # Create feature importance DataFrame efficiently
    abs_coef = np.abs(coef)
    feature_importance = pd.DataFrame({
        "Feature Name": feature_names,
        "Coefficient": coef,
        "Abs Coef": abs_coef
    })
    
    # Sort and rank in one go (avoid reset_index overhead)
    feature_importance = feature_importance.sort_values(
        "Abs Coef", ascending=False
    ).reset_index(drop=True)
    feature_importance.insert(0, "Rank", np.arange(1, len(feature_importance) + 1))
    
    # Select top features
    if top_k:
        selected_features = feature_importance.head(top_k)["Feature Name"].tolist()
    else:
        # Use boolean indexing on numpy array for speed
        selected_features = feature_importance.loc[
            feature_importance["Abs Coef"] > 0, "Feature Name"
        ].tolist()
    
    # Create report
    report = {
        "Model Type": problem_type,
        "Total Features After Preprocessing": len(feature_names),
        "Selected Features Count": len(selected_features),
        "Non-zero Coefficients": np.sum(abs_coef > 0),
        "Hyperparameters": {
            "C": C if problem_type == "classification" else None,
            "alpha": alpha if problem_type == "regression" else None,
        },
        "Top Features": selected_features[:10],
        "Pipeline Object": pipeline,
        "Preprocessing Time": getattr(pipeline.named_steps["preprocessor"], 'n_features_in_', None)
    }
    
    return feature_importance, selected_features, report

# ================================================================
#  AGGREGATE ONE-HOT IMPORTANCE
# ================================================================
def aggregate_onehot_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate importance of one-hot encoded features at the base feature level.

    Parameters
    ----------
    feature_importance : pd.DataFrame
        DataFrame with feature names and importance scores.

    Returns
    -------
    pd.DataFrame
        Aggregated importance per base feature.
    """
    df = feature_importance.copy()

    feature_col = None
    for col in df.columns:
        if df[col].dtype == "object":
            feature_col = col
            break

    if feature_col is None:
        raise ValueError("No string feature name column detected in feature_importance")

    importance_col = None
    for candidate in df.columns:
        if candidate.lower().replace(" ", "") in ["abscoef", "abs_coefficient", "importance"]:
            importance_col = candidate
            break

    if importance_col is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c.lower() not in ["rank"]]
        if not numeric_cols:
            raise ValueError("No numeric importance column detected in feature_importance")
        importance_col = numeric_cols[-1]

    df["base_feature"] = df[feature_col].astype(str).apply(
        lambda x: x.split("_")[0] if "_" in x else x
    )

    results = []
    for base, group in df.groupby("base_feature"):
        best = group.sort_values(importance_col, ascending=False).iloc[0]

        results.append({
            "feature": base,
            "total_importance": group[importance_col].sum(),
            "max_importance": best[importance_col],
            "best_subfeature": best[feature_col]
        })

    result_df = pd.DataFrame(results).sort_values(
        "total_importance", ascending=False
    ).reset_index(drop=True)

    return result_df


# ================================================================
#  XGBOOST FEATURE IMPORTANCE
# ================================================================
def xgboost_feature_analysis(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "classification",
    top_k: Optional[int] = None,
    importance_threshold: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> Tuple[pd.DataFrame, List[str], Union[XGBClassifier, XGBRegressor]]:
    """
    Perform feature importance analysis using XGBoost Gain and Permutation importance.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features and target column.
    target : str
        Name of the target column.
    problem_type : {"classification", "regression"}, default="classification"
        Type of machine learning problem.
    top_k : int | None, default=None
        If provided, returns the top_k most important features.
    importance_threshold : float, default=0.5
        Minimum combined importance score required to select a feature.
    test_size : float, default=0.2
        Proportion of dataset to include in test split.
    random_state : int, default=42
        Random seed for reproducibility.
    n_estimators : int, default=400
        Number of boosting rounds.
    learning_rate : float, default=0.05
        Boosting learning rate.
    max_depth : int, default=5
        Maximum tree depth.
    subsample : float, default=0.8
        Subsample ratio of the training instances.
    colsample_bytree : float, default=0.8
        Subsample ratio of columns when constructing each tree.

    Returns
    -------
    report : pandas.DataFrame
        Feature importance report sorted by combined importance score.
        Columns: feature, gain_percent, perm_percent, combined_score.
    selected_features : list
        List of selected important features.
    model : xgboost model
        Trained XGBoost model.
    """
    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Handle missing values (simple approach for XGBoost)
    if X.isnull().any().any():
        print("⚠ Warning: Missing values detected. Filling with 0 for XGBoost.")
        X = X.fillna(0)
    
    if y.isnull().any():
        print("⚠ Warning: Missing values detected in target. Dropping rows with missing target.")
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Model selection
    if problem_type == "classification":
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="logloss"
        )
    elif problem_type == "regression":
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )
    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")

    # Train model
    model.fit(X_train, y_train)

    # XGBoost gain importance
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")

    # If no gain scores (all features used), get feature names from model
    if not gain_scores:
        # For newer XGBoost versions, feature names might be stored differently
        gain_df = pd.DataFrame({
            "feature": X.columns,
            "gain_importance": [0] * len(X.columns)
        })
    else:
        gain_df = pd.DataFrame({
            "feature": list(gain_scores.keys()),
            "gain_importance": list(gain_scores.values())
        })

    gain_df["gain_percent"] = (
        gain_df["gain_importance"] / gain_df["gain_importance"].sum() * 100
    )

    # Permutation importance
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": X.columns,
        "perm_importance": perm.importances_mean
    })

    perm_df["perm_percent"] = (
        perm_df["perm_importance"] / perm_df["perm_importance"].sum() * 100
    )

    # Merge importance scores
    report = (
        pd.merge(
            gain_df[["feature", "gain_percent"]],
            perm_df[["feature", "perm_percent"]],
            on="feature",
            how="outer"
        )
        .fillna(0)
    )

    # Combined importance score (60% gain, 40% permutation)
    report["combined_score"] = (
        report["gain_percent"] * 0.6 + report["perm_percent"] * 0.4
    )

    report = report.sort_values(
        "combined_score",
        ascending=False
    ).reset_index(drop=True)

    # Feature selection
    if top_k is not None:
        selected_features = report.head(top_k)["feature"].tolist()
    else:
        selected_features = report[
            report["combined_score"] >= importance_threshold
        ]["feature"].tolist()

    return report, selected_features, model


# ================================================================
#  FEATURE INFORMATION SCAN
# ================================================================
def feature_information_scan(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "classification",
    corr_threshold: float = 0.01,
    mi_threshold: float = 0.001,
    top_k: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Evaluate feature usefulness using Correlation and Mutual Information.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str
        Target column name.
    problem_type : {"classification", "regression"}, default="classification"
        Type of problem.
    corr_threshold : float, default=0.01
        Minimum absolute correlation.
    mi_threshold : float, default=0.001
        Minimum mutual information.
    top_k : int, optional
        Keep top K features.

    Returns
    -------
    report_df : pd.DataFrame
        Feature ranking table.
    selected_features : List[str]
        Recommended features.
    """
    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=np.number).columns

    corr_scores = X[numeric_cols].corrwith(y).abs()

    # Fill NaN for mutual information
    X_filled = X.fillna(0)
    
    if problem_type == "classification":
        mi_scores = mutual_info_classif(X_filled, y)
    else:
        mi_scores = mutual_info_regression(X_filled, y)

    mi_scores = pd.Series(mi_scores, index=X.columns)

    report = pd.DataFrame({
        "correlation": corr_scores,
        "mutual_information": mi_scores
    }).fillna(0)

    report["score"] = report["correlation"] + report["mutual_information"]
    report = report.sort_values("score", ascending=False)

    filtered = report[
        (report["correlation"] >= corr_threshold) |
        (report["mutual_information"] >= mi_threshold)
    ]

    if top_k:
        filtered = filtered.head(top_k)

    selected_features = filtered.index.tolist()

    return report, selected_features


# ================================================================
#  MISSING REPORT
# ================================================================
def missing_report(
    df: pd.DataFrame,
    column: Optional[str] = None
) -> Union[pd.DataFrame, Dict[str, Union[str, int, float]]]:
    """
    Generate a missing-value report for a DataFrame or a specific column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str, optional
        If provided, the report is generated for that column only.

    Returns
    -------
    pd.DataFrame or dict
        If `column` is None -> returns a DataFrame with missing statistics for all columns.
        If `column` is specified -> returns a dictionary with statistics for that column.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if column is not None:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        total_rows = len(df)
        missing_count = df[column].isna().sum()
        missing_percent = (missing_count / total_rows) * 100

        return {
            "column": column,
            "dtype": str(df[column].dtype),
            "total_rows": total_rows,
            "missing_count": int(missing_count),
            "missing_percent": round(missing_percent, 2),
        }

    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "total_rows": len(df),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().sum() / len(df)) * 100,
    })

    report["missing_percent"] = report["missing_percent"].round(2)

    return report


# ================================================================
#  DROP MISSING BELOW
# ================================================================
def drop_missing_below(
    df: pd.DataFrame,
    threshold: Union[float, int],
    axis: Literal["columns", "rows"] = "columns"
) -> pd.DataFrame:
    """
    Drop rows or columns whose number of missing values exceeds a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float or int
        - float (0 < threshold < 1): proportion of allowed missing values.
        - int: absolute count of allowed missing values.
    axis : {"columns", "rows"}, default="columns"
        Axis along which the filtering is applied.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with rows or columns removed.
    """
    if axis not in ("columns", "rows"):
        raise ValueError("axis must be 'columns' or 'rows'")

    if isinstance(threshold, float):
        if not 0 < threshold < 1:
            raise ValueError("Float threshold must be in (0, 1).")

        missing_ratio = df.isna().mean(axis=0 if axis == "columns" else 1)
        mask = missing_ratio >= threshold

    elif isinstance(threshold, int):
        if threshold < 0:
            raise ValueError("Integer threshold must be >= 0.")

        missing_count = df.isna().sum(axis=0 if axis == "columns" else 1)
        mask = missing_count >= threshold

    else:
        raise ValueError("threshold must be float or int")

    return df.loc[:, ~mask] if axis == "columns" else df.loc[~mask, :]


# ================================================================
#  CLEAN COLUMN
# ================================================================
def clean_column(
    df: pd.DataFrame,
    column: str,
    method: Literal["drop_column", "drop_rows", "simple", "knn", "iter", "mice"] = "simple",
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
    fill_value: Optional[object] = None,
    n_neighbors: int = 5,
    estimator=None,
    phase: Literal["fit", "transform", "fit_transform"] = "fit_transform",
    imputer_instance=None,  # برای انتقال imputer در مرحله transform
) -> Union[pd.DataFrame, tuple[pd.DataFrame, object]]:
    """
    Clean or impute missing values in a specific column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Column to clean.
    method : {"drop_column", "drop_rows", "simple", "knn", "iter", "mice"}
        Imputation method.
    strategy : {"mean", "median", "most_frequent", "constant"}
        Strategy for simple imputer.
    fill_value : object, optional
        Value to fill when strategy="constant".
    n_neighbors : int, default=5
        Number of neighbors for KNN.
    estimator : sklearn estimator, optional
        Estimator for iterative imputer.
    phase : {"fit", "transform", "fit_transform"}
        - "fit_transform": train and apply imputer (default, for training data)
        - "fit": only train imputer, return fitted imputer
        - "transform": apply pre-fitted imputer (for test data)
    imputer_instance : object, optional
        Pre-fitted imputer to use when phase="transform".

    Returns
    -------
    pd.DataFrame or tuple
        If phase="fit_transform" or "transform": returns cleaned DataFrame.
        If phase="fit": returns tuple (original DataFrame, fitted imputer object).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # روش‌هایی که نیاز به fit/transform ندارند
    if method in ["drop_column", "drop_rows"]:
        if phase in ["fit", "transform"]:
            raise ValueError(f"Method '{method}' does not support phase='{phase}'. Use 'fit_transform'.")
        
        df = df.copy()
        if method == "drop_column":
            return df.drop(columns=[column])
        else:  # drop_rows
            return df.dropna(subset=[column])

    # روش‌های نیازمند imputer
    df = df.copy()
    method = method.lower()

    # --- مرحله fit (فقط آموزش) ---
    if phase == "fit":
        if method == "simple":
            if strategy == "constant" and fill_value is None:
                raise ValueError("fill_value must be provided when strategy='constant'.")
            if strategy in ["mean", "median"] and not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Strategy '{strategy}' requires numeric column.")
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            imputer.fit(df[[column]])
            return df, imputer

        elif method == "knn":
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("KNN imputation requires numeric columns.")
            if n_neighbors <= 0:
                raise ValueError("n_neighbors must be greater than 0.")
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputer.fit(df[[column]])
            return df, imputer

        elif method == "iter":
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("Iterative imputation requires numeric columns.")
            imputer = IterativeImputer(estimator=estimator)
            imputer.fit(df[[column]])
            return df, imputer

        elif method == "mice":
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("MICE imputation requires numeric columns.")
            mice_estimator = estimator if estimator is not None else BayesianRidge()
            imputer = IterativeImputer(
                estimator=mice_estimator,
                max_iter=10,
                sample_posterior=True,
                random_state=42,
            )
            imputer.fit(df[[column]])
            return df, imputer

    # --- مرحله transform (فقط اعمال) ---
    elif phase == "transform":
        if imputer_instance is None:
            raise ValueError("imputer_instance must be provided when phase='transform'.")
        
        df[[column]] = imputer_instance.transform(df[[column]])
        return df

    # --- مرحله fit_transform (پیش‌فرض) ---
    else:  # fit_transform
        if method == "simple":
            if strategy == "constant" and fill_value is None:
                raise ValueError("fill_value must be provided when strategy='constant'.")
            if strategy in ["mean", "median"] and not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Strategy '{strategy}' requires numeric column.")
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            df[[column]] = imputer.fit_transform(df[[column]])
            return df

        elif method == "knn":
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("KNN imputation requires numeric columns.")
            if n_neighbors <= 0:
                raise ValueError("n_neighbors must be greater than 0.")
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[[column]] = imputer.fit_transform(df[[column]])
            return df

        elif method == "iter":
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("Iterative imputation requires numeric columns.")
            imputer = IterativeImputer(estimator=estimator)
            df[[column]] = imputer.fit_transform(df[[column]])
            return df

        elif method == "mice":
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("MICE imputation requires numeric columns.")
            mice_estimator = estimator if estimator is not None else BayesianRidge()
            imputer = IterativeImputer(
                estimator=mice_estimator,
                max_iter=10,
                sample_posterior=True,
                random_state=42,
            )
            df[[column]] = imputer.fit_transform(df[[column]])
            return df

    raise ValueError("Invalid method. Choose from: drop_column, drop_rows, simple, knn, iter, mice.")

# ================================================================
#  DUPLICATE REPORT
# ================================================================
def duplicate_report(
    df: pd.DataFrame,
    action: Literal["report", "drop", "keep_first", "keep_last"] = "report",
    return_rows: bool = False
) -> Dict[str, Union[int, float, pd.DataFrame]]:
    """
    Analyze duplicated rows in a DataFrame and optionally remove them.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    action : {"report", "drop", "keep_first", "keep_last"}, default="report"
        Action to perform on duplicates.
    return_rows : bool, default=False
        If True, duplicated rows BEFORE any removal are returned.

    Returns
    -------
    dict
        Dictionary containing duplicate statistics and optionally duplicated rows.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df_copy = df.copy()

    total_before = len(df_copy)
    dup_count_before = df_copy.duplicated().sum()
    dup_percent_before = round((dup_count_before / total_before) * 100, 2)

    duplicated_rows_df = df_copy[df_copy.duplicated(keep=False)]

    if action == "report":
        df_after = df_copy
    elif action == "drop":
        df_after = df_copy.drop_duplicates(keep=False)
    elif action == "keep_first":
        df_after = df_copy.drop_duplicates(keep="first")
    elif action == "keep_last":
        df_after = df_copy.drop_duplicates(keep="last")
    else:
        raise ValueError("Invalid action. Must be 'report', 'drop', 'keep_first', or 'keep_last'.")

    total_after = len(df_after)
    dup_count_after = df_after.duplicated().sum()
    dup_percent_after = round((dup_count_after / total_after) * 100, 2)

    report = {
        "total_rows_before": int(total_before),
        "duplicate_count_before": int(dup_count_before),
        "duplicate_percent_before": dup_percent_before,
        "total_rows_after": int(total_after),
        "duplicate_count_after": int(dup_count_after),
        "duplicate_percent_after": dup_percent_after,
    }

    if return_rows:
        report["duplicated_rows"] = duplicated_rows_df

    return report


# ================================================================
#  RENAME COLUMNS
# ================================================================
def rename_columns(
    df: pd.DataFrame,
    mode: Literal["all", "single", "manual"] = "all",
    column: Optional[str] = None,
    rename_map: Optional[Dict[str, str]] = None,
    pattern: Literal[
        "lower_snake", "upper_snake", "kebab",
        "camel", "pascal", "remove_spaces", "alphanumeric"
    ] = "lower_snake"
) -> pd.DataFrame:
    """
    Rename DataFrame columns using predefined patterns or a manual mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    mode : {"all", "single", "manual"}, default="all"
        Renaming mode.
    column : str, optional
        Target column when mode="single".
    rename_map : dict, optional
        Custom mapping for mode="manual".
    pattern : str, default="lower_snake"
        Naming convention pattern.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if mode not in ("all", "single", "manual"):
        raise ValueError("mode must be one of: 'all', 'single', 'manual'.")

    if mode == "single" and column is None:
        raise ValueError("You must provide 'column' when mode='single'.")

    if mode == "manual" and rename_map is None:
        raise ValueError("You must provide 'rename_map' when mode='manual'.")

    if mode == "single" and column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    def apply_pattern(col_name: str, p: str) -> str:
        col = col_name.strip()

        if p == "lower_snake":
            return col.lower().replace(" ", "_")
        if p == "upper_snake":
            return col.upper().replace(" ", "_")
        if p == "kebab":
            return col.lower().replace(" ", "-")
        if p == "camel":
            parts = col.lower().split()
            return parts[0] + "".join(w.capitalize() for w in parts[1:])
        if p == "pascal":
            return "".join(w.capitalize() for w in col.split())
        if p == "remove_spaces":
            return col.replace(" ", "")
        if p == "alphanumeric":
            clean = re.sub(r"[^A-Za-z0-9_ ]+", "", col)
            return clean.replace(" ", "_")
        return col

    if mode == "manual":
        return df.rename(columns=rename_map).copy()

    if mode == "single":
        new_name = apply_pattern(column, pattern)
        return df.rename(columns={column: new_name}).copy()

    new_cols = {col: apply_pattern(col, pattern) for col in df.columns}
    return df.rename(columns=new_cols).copy()


# ================================================================
#  DETECT NUMERIC ISSUES
# ================================================================
def detect_numeric_issues(
    df: pd.DataFrame,
    column: str,
    sample: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Identify non-numeric values in a column that prevent numeric conversion.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the column expected to be numeric.
    sample : int, default=5
        Number of problematic rows to return as sample.
    verbose : bool, default=True
        If True, prints a diagnostic summary.

    Returns
    -------
    Dict[str, Any]
        Dictionary with diagnostic information.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    if sample < 0:
        raise ValueError("Parameter 'sample' must be >= 0.")

    converted = pd.to_numeric(df[column], errors="coerce")
    mask_invalid = converted.isna() & df[column].notna()

    invalid_count = int(mask_invalid.sum())
    total_rows = len(df)
    invalid_ratio = invalid_count / total_rows if total_rows > 0 else 0.0

    unique_invalid_values = (
        df.loc[mask_invalid, column]
        .drop_duplicates()
        .tolist()
    )

    sample_invalid_rows = df.loc[mask_invalid, [column]].head(sample)

    result = {
        "column": column,
        "total_rows": total_rows,
        "invalid_count": invalid_count,
        "invalid_ratio": invalid_ratio,
        "unique_invalid_values": unique_invalid_values,
        "sample_invalid_rows": sample_invalid_rows
    }

    if verbose:
        print("\n========== Numeric Conversion Diagnostic ==========")
        print(f"Column: {column}")
        print(f"Total rows: {total_rows}")
        print(f"Invalid values: {invalid_count}")
        print(f"Invalid ratio: {invalid_ratio:.4f}")
        print(f"Unique invalid values: {len(unique_invalid_values)}")

        if unique_invalid_values:
            print("\nSample invalid values:")
            print(unique_invalid_values[:sample])

        print("===================================================")

    return result


# ================================================================
#  SMART DTYPE CONVERTER
# ================================================================
def smart_dtype_converter(
    df: pd.DataFrame,
    column: str,
    target_type: Literal["numeric", "category", "string"],
    value_map: Optional[Dict[Any, Any]] = None,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    quantile_bins: Optional[int] = None,
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convert column dtype intelligently with optional mapping and binning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Column name to convert.
    target_type : {"numeric", "category", "string"}
        Desired output dtype.
    value_map : dict, optional
        Mapping applied before conversion.
    bins : list of float, optional
        Custom bin edges for numeric -> category.
    labels : list of str, optional
        Labels corresponding to bins.
    quantile_bins : int, optional
        Number of quantile bins for automatic binning.
    inplace : bool, default=False
        Whether to modify DataFrame in place.
    verbose : bool, default=True
        Whether to print conversion messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with converted column.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    data = df[column].copy()

    if value_map is not None:
        data = data.replace(value_map)

    if target_type == "numeric":
        converted = pd.to_numeric(data, errors="coerce")
        invalid_count = converted.isna().sum() - data.isna().sum()
        if verbose and invalid_count > 0:
            print(f"⚠ {invalid_count} values could not be converted and became NaN.")
        final_series = converted

    elif target_type == "category":
        numeric_data = pd.to_numeric(data, errors="coerce")

        if bins is not None:
            if labels and len(labels) != len(bins) - 1:
                raise ValueError("Length of labels must be len(bins) - 1.")
            final_series = pd.cut(numeric_data, bins=bins, labels=labels)
        elif quantile_bins is not None:
            final_series = pd.qcut(
                numeric_data,
                q=quantile_bins,
                labels=labels,
                duplicates="drop"
            )
        else:
            final_series = data.astype("category")

    elif target_type == "string":
        final_series = data.astype("string")

    else:
        raise ValueError("target_type must be 'numeric', 'category', or 'string'.")

    if inplace:
        df[column] = final_series
        result_df = df
    else:
        result_df = df.copy()
        result_df[column] = final_series

    if verbose:
        print("✅ Conversion completed.")
        print("New dtype:", result_df[column].dtype)

    return result_df


__all__ = [
    "l1_feature_selection",
    "aggregate_onehot_importance",
    "xgboost_feature_analysis",
    "feature_information_scan",
    "missing_report",
    "drop_missing_below",
    "clean_column",
    "duplicate_report",
    "rename_columns",
    "detect_numeric_issues",
    "smart_dtype_converter",
]

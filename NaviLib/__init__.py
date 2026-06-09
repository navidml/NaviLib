"""
ds_toolbox
~~~~~~~~~~

A comprehensive data science toolbox for:
- Evaluation (regression, classification, clustering, ranking, recommender)
- Statistical tests (normality, variance, parametric, nonparametric, correlation, categorical)
- EDA (distribution analysis, value counts plotting)
- Data cleaning (missing values, duplicates, column renaming, numeric conversion)
- Feature engineering (transformation, encoding, scaling, outlier handling)

Author: Navid Bordbar
Version: 1.0.0
"""

# ==============================================================================
# EVALUATION MODULES
# ==============================================================================
from .evaluation import (
    evaluate_regression_model,
    evaluate_classification_model,
    evaluate_clustering_model,
    evaluate_ranking_model,
    evaluate_recommender_model,
)

# ==============================================================================
# STATISTICAL TESTS MODULES
# ==============================================================================
from .Statistical_Tests import (
    normality_test,
    variance_homogeneity_test,
    parametric_test,
    nonparametric_test,
    correlation_test,
    find_correlated_features,
    feature_correlation_analysis,
    categorical_test,
    effect_size,
    tukey_hsd_posthoc,
    _interpret_cohens_d,
    _interpret_eta_squared,
    _interpret_cramers_v,
)

# ==============================================================================
# EDA MODULES
# ==============================================================================
from .eda import (
    analyze_numeric_distribution,
    value_counts_plot,
)

# ==============================================================================
# CLEANING MODULES
# ==============================================================================
from .cleaning import (
    l1_feature_selection,
    aggregate_onehot_importance,
    xgboost_feature_analysis,  
    missing_report,
    drop_missing_below,
    clean_column,
    duplicate_report,
    rename_columns,
    detect_numeric_issues,
    smart_dtype_converter,
)

# ==============================================================================
# FEATURE ENGINEERING MODULES
# ==============================================================================
from .feature_engineering import (
    auto_best_transformation,
    transform_column,
    encode_categorical_features,
    scale_features,
    analyze_outliers,
    save_outliers,
    remove_outliers,
    balance_data,
    find_best_k_auto
)

# ==============================================================================
# PUBLIC API EXPORTS
# ==============================================================================

__all__ = [
    # Evaluation
    "evaluate_regression_model",
    "evaluate_classification_model",
    "evaluate_clustering_model",
    "evaluate_ranking_model",
    "evaluate_recommender_model",
    
    # Statistical Tests
    "normality_test",
    "variance_homogeneity_test",
    "parametric_test",
    "nonparametric_test",
    "correlation_test",
    "find_correlated_features",
    "feature_correlation_analysis",
    "categorical_test",
    "effect_size",
    "tukey_hsd_posthoc",
    
    # EDA
    "analyze_numeric_distribution",
    "value_counts_plot",
    
    # Cleaning
    "l1_feature_selection",
    "aggregate_onehot_importance",
    "feature_information_scan",
    "missing_report",
    "drop_missing_below",
    "clean_column",
    "duplicate_report",
    "rename_columns",
    "detect_numeric_issues",
    "smart_dtype_converter",
    
    # Feature Engineering
    "auto_best_transformation",
    "transform_column",
    "encode_categorical_features",
    "scale_features",
    "analyze_outliers",
    "save_outliers",
    "remove_outliers",
]

# ==============================================================================
# PACKAGE METADATA
# ==============================================================================

__version__ = "1.0.0"
__author__ = "Generated"
__license__ = "MIT"

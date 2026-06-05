# ==============================
# ML Evaluation Toolkit
# Complete evaluation functions for regression, classification,
# clustering, ranking, and recommender systems
# ==============================

from typing import Dict, Optional, Tuple, List, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    log_loss,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    explained_variance_score, 
    median_absolute_error, max_error
)
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error


# ==============================
# 1. Regression Evaluation
# ==============================

def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    show_outliers: bool = True,
) -> Dict[str, float]:
    
    residuals = y_true - y_pred
    
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Explained_Var": explained_variance_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "Max_Error": max_error(y_true, y_pred),
        "RMSPE": np.sqrt(np.mean(((y_true - y_pred) / (y_true + 1e-8)) ** 2)) * 100,
    }
    
    if show_outliers:
        outlier_threshold = np.percentile(np.abs(residuals), 95)
        metrics["Outlier_95_Ratio"] = np.mean(np.abs(residuals) > outlier_threshold)
    
    if not plot:
        return metrics
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    ax_actual_vs_pred = fig.add_subplot(gs[0, :2])
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_qq = fig.add_subplot(gs[1, 0])
    ax_resid_vs_fitted = fig.add_subplot(gs[1, 1])
    ax_metrics = fig.add_subplot(gs[2, :])
    
    # Actual vs Predicted
    ax_actual_vs_pred.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax_actual_vs_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax_actual_vs_pred.set_xlabel('Actual Values')
    ax_actual_vs_pred.set_ylabel('Predicted Values')
    ax_actual_vs_pred.set_title('Actual vs Predicted', fontweight='bold')
    ax_actual_vs_pred.legend()
    ax_actual_vs_pred.grid(True, alpha=0.3)
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=ax_dist, color='steelblue', alpha=0.7)
    ax_dist.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax_dist.set_title('Residual Distribution', fontweight='bold')
    ax_dist.set_xlabel('Residual')
    ax_dist.legend()
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_color('steelblue')
    ax_qq.get_lines()[1].set_color('red')
    ax_qq.set_title('Q-Q Plot', fontweight='bold')
    ax_qq.grid(True, alpha=0.3)
    
    # Residuals vs Fitted
    ax_resid_vs_fitted.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax_resid_vs_fitted.axhline(0, color='red', linestyle='--', lw=2)
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    ax_resid_vs_fitted.plot(np.sort(y_pred), p(np.sort(y_pred)), 'green', lw=2, label='Trend')
    ax_resid_vs_fitted.set_xlabel('Predicted Values')
    ax_resid_vs_fitted.set_ylabel('Residuals')
    ax_resid_vs_fitted.set_title('Residuals vs Fitted', fontweight='bold')
    ax_resid_vs_fitted.legend()
    ax_resid_vs_fitted.grid(True, alpha=0.3)
    
    # Metrics Bar Chart
    metric_names = list(metrics.keys())[:8]
    metric_values = [metrics[m] for m in metric_names]
    colors = ['#2ecc71' if m == 'R2' else '#3498db' for m in metric_names]
    bars = ax_metrics.bar(metric_names, metric_values, color=colors, alpha=0.8)
    ax_metrics.set_title('Performance Metrics', fontweight='bold')
    ax_metrics.grid(axis='y', linestyle='--', alpha=0.3)
    ax_metrics.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    residual_stats = f"Residuals: μ={residuals.mean():.4f}, σ={residuals.std():.4f}\nSkew={stats.skew(residuals):.3f}, Kurtosis={stats.kurtosis(residuals):.3f}"
    fig.text(0.02, 0.98, residual_stats, transform=fig.transFigure, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    if abs(residuals.mean()) > 0.1 * residuals.std():
        print(f"⚠️ Warning: Residuals not centered around zero (mean={residuals.mean():.4f})")
    
    return metrics


# ==============================
# 2. Classification Evaluation
# ==============================

def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    *,
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate classification model with comprehensive metrics and diagnostic plots.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray, optional
        Predicted probabilities (required for ROC/PR curves and log loss).
    plot : bool, default=True
        Whether to generate diagnostic plots.
    figsize : tuple, default=(14, 10)
        Figure size for plots.
    class_names : list, optional
        Names of classes for better visualization.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics.
    """
    # =====================
    #   Input Validation
    # =====================
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2
    
    if class_names is None:
        class_names = [str(c) for c in unique_classes]
    
    # =====================
    #   Core Metrics
    # =====================
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, cls in enumerate(unique_classes):
        cls_name = class_names[i] if i < len(class_names) else str(cls)
        metrics[f"precision_{cls_name}"] = per_class_precision[i]
        metrics[f"recall_{cls_name}"] = per_class_recall[i]
        metrics[f"f1_{cls_name}"] = per_class_f1[i]
    
    # Binary-specific metrics
    if is_binary:
        pos_label = unique_classes[1]
        metrics.update({
            "precision_positive": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            "recall_positive": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            "f1_positive": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            "specificity": recall_score(y_true, y_pred, pos_label=unique_classes[0], zero_division=0),
        })
        
        # Probability-based metrics
        if y_prob is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_prob)
            avg_precision = auc(recall_vals, precision_vals)
            
            metrics.update({
                "roc_auc": roc_auc,
                "average_precision": avg_precision,
                "log_loss": log_loss(y_true, y_prob),
                "brier_score": np.mean((y_prob - y_true) ** 2),
            })
            
            # Optimal threshold using Youden's J statistic
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            metrics["optimal_threshold"] = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    
    if not plot:
        return metrics
    
    # =====================
    #   Enhanced Plotting
    # =====================
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    if is_binary and y_prob is not None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        ax_cm = fig.add_subplot(gs[0, 0])
        ax_roc = fig.add_subplot(gs[0, 1])
        ax_pr = fig.add_subplot(gs[0, 2])
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_confidence = fig.add_subplot(gs[1, 1])
        ax_metrics = fig.add_subplot(gs[1, 2])
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.7))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        ax_cm = fig.add_subplot(gs[0, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_metrics = fig.add_subplot(gs[1, :])
        ax_roc = ax_pr = ax_confidence = None
    
    # =====================
    #   1. Confusion Matrix
    # =====================
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap with improved styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                linewidths=1, linecolor='white', square=True)
    ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax_cm.set_xlabel('Predicted Label', fontsize=10)
    ax_cm.set_ylabel('True Label', fontsize=10)
    
    # Add normalized percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax_cm.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})',
                          ha='center', va='center', fontsize=8, color='darkblue')
    
    # Set tick labels
    ax_cm.set_xticklabels(class_names[:n_classes], rotation=45, ha='right')
    ax_cm.set_yticklabels(class_names[:n_classes], rotation=0)
    
    # =====================
    #   2. ROC Curve
    # =====================
    if is_binary and y_prob is not None and ax_roc:
        ax_roc.plot(fpr, tpr, lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})', color='#2E86AB')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#2E86AB')
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])
        ax_roc.set_xlabel('False Positive Rate', fontsize=10)
        ax_roc.set_ylabel('True Positive Rate', fontsize=10)
        ax_roc.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax_roc.legend(loc='lower right', frameon=True, fancybox=True)
        ax_roc.grid(True, alpha=0.3)
        
        # Mark optimal threshold
        if "optimal_threshold" in metrics:
            ax_roc.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                       label=f'Threshold={metrics["optimal_threshold"]:.2f}')
            ax_roc.legend(loc='lower right')
    
    # =====================
    #   3. Precision-Recall Curve
    # =====================
    if is_binary and y_prob is not None and ax_pr:
        baseline = metrics.get("recall_positive", 0.5)
        ax_pr.plot(recall_vals, precision_vals, lw=2.5, color='#A23B72',
                  label=f'PR (AP = {avg_precision:.3f})')
        ax_pr.fill_between(recall_vals, precision_vals, alpha=0.15, color='#A23B72')
        ax_pr.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
                     label=f'Baseline (F1={baseline:.3f})')
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([-0.02, 1.02])
        ax_pr.set_xlabel('Recall', fontsize=10)
        ax_pr.set_ylabel('Precision', fontsize=10)
        ax_pr.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax_pr.legend(loc='lower left', frameon=True, fancybox=True)
        ax_pr.grid(True, alpha=0.3)
    
    # =====================
    #   4. Prediction Distribution
    # =====================
    unique_pred = np.unique(y_pred)
    value_counts = pd.Series(y_pred).value_counts().sort_index()
    
    colors_dist = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
    bars_dist = ax_dist.bar(range(len(value_counts)), value_counts.values,
                            color=colors_dist, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_dist.set_xticks(range(len(value_counts)))
    ax_dist.set_xticklabels([class_names[i] if i < len(class_names) else str(v)
                             for i, v in enumerate(value_counts.index)], rotation=45, ha='right')
    ax_dist.set_ylabel('Frequency', fontsize=10)
    ax_dist.set_xlabel('Predicted Class', fontsize=10)
    ax_dist.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax_dist.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars_dist, value_counts.values):
        ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =====================
    #   5. Confidence Distribution (Binary)
    # =====================
    if is_binary and y_prob is not None and ax_confidence:
        correct_mask = (y_pred == y_true)
        incorrect_mask = ~correct_mask
        
        ax_confidence.hist(y_prob[correct_mask], bins=20, alpha=0.6, color='green',
                          label=f'Correct (n={correct_mask.sum()})', density=True, edgecolor='darkgreen')
        ax_confidence.hist(y_prob[incorrect_mask], bins=20, alpha=0.6, color='red',
                          label=f'Incorrect (n={incorrect_mask.sum()})', density=True, edgecolor='darkred')
        ax_confidence.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')
        ax_confidence.set_xlabel('Predicted Probability (Positive Class)', fontsize=10)
        ax_confidence.set_ylabel('Density', fontsize=10)
        ax_confidence.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax_confidence.legend(loc='upper center', frameon=True, fancybox=True)
        ax_confidence.grid(True, alpha=0.3)
    
    # =====================
    #   6. Metrics Bar Chart
    # =====================
    if not is_binary:
        metric_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)', 'Kappa', 'MCC']
        metric_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'kappa', 'mcc']
    else:
        metric_names = ['Acc', 'Prec', 'Recall', 'F1', 'Spec', 'AUC', 'AP', 'Kappa']
        metric_keys = ['accuracy', 'precision_positive', 'recall_positive', 
                      'f1_positive', 'specificity', 'roc_auc', 'average_precision', 'kappa']
    
    metric_values = [metrics.get(k, 0) for k in metric_keys if k in metrics]
    metric_names_used = [metric_names[i] for i in range(len(metric_values))]
    
    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(metric_values)))
    bars_metrics = ax_metrics.bar(metric_names_used, metric_values, color=colors_bar,
                                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_metrics.set_ylim([0, 1.05])
    ax_metrics.set_ylabel('Score', fontsize=10)
    ax_metrics.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars_metrics, metric_values):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return metrics

# ==============================
# 3. Clustering Evaluation
# ==============================

def evaluate_clustering_model(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 6),
    plot: bool = True,
) -> Dict[str, float]:
    """
    Evaluate clustering performance using internal and optional external metrics
    with optional visualization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix used for clustering.
    labels : np.ndarray
        Cluster labels assigned by the clustering algorithm.
    true_labels : np.ndarray, optional
        Ground truth labels used for external evaluation metrics.
    figsize : Tuple[int, int], default=(14, 6)
        Figure size used for visualization.
    plot : bool, default=True
        If True, clustering visualizations will be generated.

    Returns
    -------
    Dict[str, float]
        Dictionary containing clustering evaluation metrics.

    Raises
    ------
    ValueError
        If number of unique clusters is less than 2.
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("Clustering evaluation requires at least two clusters.")

    silhouette = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    metrics: Dict[str, float] = {
        "n_clusters": float(len(unique_labels)),
        "silhouette_score": silhouette,
        "davies_bouldin_index": dbi,
        "calinski_harabasz_score": ch,
    }

    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        metrics["adjusted_rand_index"] = ari
        metrics["normalized_mutual_info"] = nmi

    if plot:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        scatter = ax[0].scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7
        )
        ax[0].set_title("Cluster Visualization (PCA)")
        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax[0])

        sns.countplot(x=labels, ax=ax[1])
        ax[1].set_title("Cluster Size Distribution")
        ax[1].set_xlabel("Cluster Label")
        ax[1].set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    return metrics


# ==============================
# 4. Ranking Evaluation
# ==============================

def evaluate_ranking_model(
    df: pd.DataFrame,
    query_col: str,
    true_col: str,
    pred_col: str,
    k: int = 10,
    plot: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a ranking/retrieval model using standard top-K metrics.

    This function assumes a list-wise representation where each row is a
    (query, document) pair with a predicted score and a ground-truth
    relevance label (e.g., 0/1, or graded relevance > 0).

    Metrics computed (aggregated over queries)
    -----------------------------------------
    - Precision@K
    - Recall@K
    - MAP@K  (Mean Average Precision at K)
    - MRR    (Mean Reciprocal Rank)
    - NDCG@K (Normalized Discounted Cumulative Gain at K)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least the query ID, true relevance,
        and predicted score columns.
    query_col : str
        Column name indicating query ID (used to group rows).
    true_col : str
        Column name indicating ground-truth relevance (0/1 or graded).
        Values > 0 are considered relevant for binary metrics.
    pred_col : str
        Column name indicating model's predicted score (higher is better).
    k : int, default=10
        Cutoff rank for top-K metrics. If a query has fewer than K results,
        only the available items are used for that query.
    plot : bool, default=True
        If True, a bar plot of the aggregated metrics is displayed.

    Returns
    -------
    Dict[str, float]
        Dictionary with evaluation metrics.

    Raises
    ------
    ValueError
        If k <= 0 or required columns are missing or no valid queries found.
    """
    if k <= 0:
        raise ValueError("Parameter 'k' must be a positive integer.")

    required_cols = {query_col, true_col, pred_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    ndcgs: List[float] = []
    aps: List[float] = []
    mrrs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    grouped = df.groupby(query_col, sort=False)

    for _, group in grouped:
        group_sorted = group.sort_values(pred_col, ascending=False)

        k_eff = min(k, len(group_sorted))
        if k_eff == 0:
            continue

        top_k = group_sorted.head(k_eff)

        true_relevance = top_k[true_col].to_numpy()
        binary_relevance = (true_relevance > 0).astype(int)

        total_relevant = int((group_sorted[true_col] > 0).sum())

        precision = float(binary_relevance.sum()) / k_eff
        precisions.append(precision)

        recall = (
            float(binary_relevance.sum()) / total_relevant if total_relevant > 0 else 0.0
        )
        recalls.append(recall)

        ap = 0.0
        hit_count = 0
        if total_relevant > 0:
            for i, rel in enumerate(binary_relevance):
                if rel == 1:
                    hit_count += 1
                    ap += hit_count / float(i + 1)
            ap /= float(total_relevant)
        aps.append(ap)

        rr = 0.0
        for i, rel in enumerate(binary_relevance):
            if rel == 1:
                rr = 1.0 / float(i + 1)
                break
        mrrs.append(rr)

        dcg = 0.0
        for i, rel in enumerate(true_relevance):
            gain = (2.0**rel - 1.0)
            discount = np.log2(i + 2)
            dcg += gain / discount

        ideal_relevance = (
            group_sorted[true_col].sort_values(ascending=False).to_numpy()[:k_eff]
        )

        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            gain = (2.0**rel - 1.0)
            discount = np.log2(i + 2)
            idcg += gain / discount

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    if not precisions:
        raise ValueError(
            "No valid queries found to evaluate (empty groups after filtering)."
        )

    results = {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "map_at_k": float(np.mean(aps)),
        "mrr": float(np.mean(mrrs)),
        "ndcg_at_k": float(np.mean(ndcgs)),
    }

    if plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=list(results.keys()),
            y=list(results.values()),
            palette="Blues_d",
        )
        plt.title(f"Ranking Metrics (K = {k})")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    return results


# ==============================
# 5. Recommender Evaluation
# ==============================

def evaluate_recommender_model(
    recommended: Dict[str, List[Any]],
    ground_truth: Dict[str, List[Any]],
    k: int = 10,
    plot: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a top-K recommender system using standard recommendation metrics.

    This function assumes:
    - `recommended[user]` = list of recommended item IDs for that user.
    - `ground_truth[user]` = list of relevant (true) items for that user.

    Metrics computed (averaged over users)
    --------------------------------------
    - Precision@K
    - Recall@K
    - MAP@K
    - NDCG@K
    - HitRate@K
    - Coverage (number of unique recommended items)

    Parameters
    ----------
    recommended : Dict[str, List[Any]]
        Dictionary mapping user_id -> list of recommended items.
    ground_truth : Dict[str, List[Any]]
        Dictionary mapping user_id -> list of true items.
    k : int, default=10
        Cutoff K for evaluation.
    plot : bool, default=True
        If True, generates evaluation plots.

    Returns
    -------
    Dict[str, float]
        Dictionary containing aggregated recommender metrics.

    Raises
    ------
    ValueError
        If k <= 0 or no valid users found for evaluation.
    """
    if k <= 0:
        raise ValueError("Parameter 'k' must be a positive integer.")

    precisions: List[float] = []
    recalls: List[float] = []
    aps: List[float] = []
    ndcgs: List[float] = []
    hits: List[int] = []
    rec_lengths: List[int] = []

    all_items: set = set()

    for user, rec_list in recommended.items():
        true_items = set(ground_truth.get(user, []))

        if len(true_items) == 0:
            continue

        k_eff = min(k, len(rec_list))
        rec_items = rec_list[:k_eff]

        rec_lengths.append(k_eff)
        all_items.update(rec_items)

        hit_vector = [1 if item in true_items else 0 for item in rec_items]

        precision = float(sum(hit_vector)) / k_eff
        precisions.append(precision)

        recall = float(sum(hit_vector)) / len(true_items)
        recalls.append(recall)

        hits.append(1 if sum(hit_vector) > 0 else 0)

        hit_count = 0
        ap = 0.0
        min_rel_k = min(len(true_items), k_eff)

        for i, val in enumerate(hit_vector):
            if val == 1:
                hit_count += 1
                ap += hit_count / (i + 1)

        ap /= float(min_rel_k) if min_rel_k > 0 else 1.0
        aps.append(ap)

        dcg = sum(val / np.log2(i + 2) for i, val in enumerate(hit_vector))

        ideal = [1] * min_rel_k
        idcg = sum(val / np.log2(i + 2) for i, val in enumerate(ideal))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    if len(precisions) == 0:
        raise ValueError(
            "No valid users for evaluation: all users have empty ground_truth lists."
        )

    results = {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "map_at_k": float(np.mean(aps)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "hitrate_at_k": float(np.mean(hits)),
        "coverage": float(len(all_items)),
    }

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metric_names = ["Precision", "Recall", "MAP", "NDCG", "HitRate"]
        metric_values = [
            results["precision_at_k"],
            results["recall_at_k"],
            results["map_at_k"],
            results["ndcg_at_k"],
            results["hitrate_at_k"],
        ]

        sns.barplot(x=metric_names, y=metric_values, ax=axes[0], palette="Blues_d")
        axes[0].set_title(f"Recommender Metrics (K = {k})")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1)

        sns.countplot(x=hits, ax=axes[1])
        axes[1].set_title("Hit Distribution")
        axes[1].set_xlabel("Hit (1) vs No Hit (0)")
        axes[1].set_ylabel("Number of Users")
        axes[1].set_xticklabels(["No Hit", "Hit"])

        sns.histplot(rec_lengths, bins=10, kde=True, ax=axes[2])
        axes[2].set_title("Recommendation List Length Distribution")
        axes[2].set_xlabel("Length")
        axes[2].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    return results


# ==============================
# Module Metadata
# ==============================

__version__ = "1.0.0"
__author__ = "Navid Bordbar"
__all__ = [
    "evaluate_regression_model",
    "evaluate_classification_model",
    "evaluate_clustering_model",
    "evaluate_ranking_model",
    "evaluate_recommender_model",
]
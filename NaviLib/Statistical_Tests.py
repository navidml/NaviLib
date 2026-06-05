"""
statistical_library
~~~~~~~~~~~~~~~~~~~

A comprehensive collection of statistical tests and effect size calculators.

Modules included:
- normality_test          : Tests for normal distribution
- variance_homogeneity_test : Tests for equal variances
- parametric_test         : Parametric hypothesis tests
- nonparametric_test      : Non-parametric hypothesis tests
- correlation_test        : Correlation analysis
- categorical_test        : Categorical data tests
- effect_size             : Effect size calculations

Author: Generated
Version: 1.2
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import combinations
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import (
    chi2_contingency, chisquare, fisher_exact,
    wilcoxon, mannwhitneyu, kruskal, friedmanchisquare,
    pearsonr, spearmanr, kendalltau
)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ==============================================================================
# 1. NORMALITY TEST
# ==============================================================================

def normality_test(
    df: pd.DataFrame,
    column: str,
    method: str = "shapiro",
    alpha: float = 0.05,
    kde: bool = True,
    figsize: Tuple[int, int] = (8, 5),
    bins: int = 30,
    show_plot: bool = True
) -> Dict[str, Any]:
    """
    Performs a normality test on a given column, visualizes the distribution,
    and returns a complete statistical report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Column to test for normality.
    method : str, default="shapiro"
        Normality test method:
        - "shapiro" : Shapiro-Wilk Test
        - "dagostino" : D'Agostino K² Test
        - "anderson" : Anderson-Darling Test
        - "ks" : Kolmogorov-Smirnov Test
    alpha : float, default=0.05
        Significance level for hypothesis testing.
    kde : bool, default=True
        Whether to show KDE curve on histogram.
    figsize : tuple, default=(8, 5)
        Figure size for visualization.
    bins : int, default=30
        Number of histogram bins.
    show_plot : bool, default=True
        Whether to display the plots.

    Returns
    -------
    dict
        Full statistical report including:
        - column, method, statistic, p_value, alpha, interpretation
    """
    data = df[column].dropna()

    if len(data) < 3:
        raise ValueError("Need at least 3 observations for normality test.")

    # ---- 1) Perform the selected test ----
    if method == "shapiro":
        stat, p = stats.shapiro(data)
        test_name = "Shapiro-Wilk Test"

    elif method == "dagostino":
        stat, p = stats.normaltest(data)
        test_name = "D'Agostino K² Test"

    elif method == "ks":
        data_norm = (data - data.mean()) / data.std()
        stat, p = stats.kstest(data_norm, "norm")
        test_name = "Kolmogorov-Smirnov Test"

    elif method == "anderson":
        result = stats.anderson(data, dist="norm")
        test_name = "Anderson-Darling Test"
        stat = result.statistic
        crit_vals = result.critical_values
        sig_levels = result.significance_level
        p = None

    else:
        raise ValueError("Invalid method! Choose: shapiro, dagostino, ks, anderson")

    # ---- 2) Interpretation ----
    if method != "anderson":
        interpretation = (
            "Reject H0: data is NOT normal"
            if p < alpha else
            "Fail to reject H0: data appears normal"
        )
    else:
        alpha_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
        idx = alpha_idx.get(alpha, 2)
        passed = stat < crit_vals[idx]
        interpretation = (
            "Fail to reject H0: data appears normal"
            if passed else
            "Reject H0: data is NOT normal"
        )

    # ---- 3) Visualization ----
    if show_plot:
        plt.figure(figsize=figsize)
        plt.hist(data, bins=bins, alpha=0.6, color="#4C72B0", density=True, label="Histogram")

        if kde:
            data.plot(kind="kde", color="red", label="KDE Curve")

        plt.title(f"Distribution Plot: {column}")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {column}")
        plt.tight_layout()
        plt.show()

    # ---- 4) Build the report ----
    report = {
        "column": column,
        "method": test_name,
        "statistic": float(stat),
        "p_value": float(p) if p is not None else None,
        "alpha": alpha,
        "interpretation": interpretation,
        "n_observations": int(len(data))
    }

    if method == "anderson":
        report["critical_values"] = crit_vals.tolist()
        report["significance_levels"] = sig_levels.tolist()

    return report


# ==============================================================================
# 2. VARIANCE HOMOGENEITY TEST
# ==============================================================================

def variance_homogeneity_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    method: str = "levene",
    alpha: float = 0.05,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (8, 5)
) -> Dict[str, Any]:
    """
    Performs a variance homogeneity test (equal variances across groups),
    visualizes group distributions, and returns a full statistical report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    value_col : str
        Numerical variable to analyze.
    group_col : str
        Categorical grouping variable.
    method : str, default="levene"
        Test method:
        - "levene" : Levene Test
        - "bartlett" : Bartlett Test
        - "brownforsythe" : Brown–Forsythe Test
    alpha : float, default=0.05
        Significance level.
    show_plot : bool, default=True
        Whether to display the boxplot.
    figsize : tuple, default=(8, 5)
        Figure size for the plot.

    Returns
    -------
    dict
        Full report including:
        - method, statistic, p_value, alpha, interpretation, group_sizes
    """
    groups_list = [group[value_col].dropna().values for name, group in df.groupby(group_col)]

    if len(groups_list) < 2:
        raise ValueError("Need at least 2 groups for variance homogeneity test.")

    if method == "levene":
        stat, p = stats.levene(*groups_list)
        test_name = "Levene Test"

    elif method == "bartlett":
        stat, p = stats.bartlett(*groups_list)
        test_name = "Bartlett Test"

    elif method == "brownforsythe":
        stat, p = stats.levene(*groups_list, center="median")
        test_name = "Brown–Forsythe Test (Robust Levene)"

    else:
        raise ValueError("Invalid method! Use: levene, bartlett, brownforsythe")

    interpretation = (
        "Reject H0: variances are NOT equal (heteroscedasticity present)"
        if p < alpha else
        "Fail to reject H0: variances appear equal (homoscedasticity)"
    )

    if show_plot:
        plt.figure(figsize=figsize)
        sns.boxplot(data=df, x=group_col, y=value_col, palette="Set2")
        plt.title(f"Boxplot of {value_col} across {group_col} groups")
        plt.tight_layout()
        plt.show()

    report = {
        "method": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "alpha": alpha,
        "interpretation": interpretation,
        "group_sizes": df[group_col].value_counts().to_dict(),
    }

    return report


# ==============================================================================
# 3. PARAMETRIC TEST
# ==============================================================================

def parametric_test(
    df: pd.DataFrame,
    test: str,
    value: str,
    group: Optional[Union[str, List[str]]] = None,
    subject: Optional[str] = None,
    covariate: Optional[str] = None,
    mu: float = 0,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Universal parametric hypothesis testing engine.

    Supported tests:
        - "one_sample"      : One-sample t-test
        - "independent_t"   : Independent two-sample t-test (Welch)
        - "paired_t"        : Paired-samples t-test
        - "oneway_anova"    : One-way ANOVA
        - "rm_anova"        : Repeated-measures ANOVA
        - "twoway_anova"    : Two-way ANOVA with interaction
        - "ancova"          : ANCOVA (group + covariate)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing all variables.
    test : str
        Name of the statistical test.
    value : str
        Column name of the dependent variable.
    group : str or list, optional
        Grouping variable(s). For two-way ANOVA pass a list of two group names.
    subject : str, optional
        Subject identifier column (required for repeated-measures ANOVA).
    covariate : str, optional
        Covariate column name (required for ANCOVA).
    mu : float, default=0
        Population mean under H0 (for one-sample t-test).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Dictionary containing test statistics, p-values, and decision.
    """
    data = df.copy()

    # One-sample t-test
    if test == "one_sample":
        sample = data[value].dropna()

        if len(sample) < 2:
            raise ValueError("Need at least 2 observations for t-test.")

        stat, p = stats.ttest_1samp(sample, mu)

        return {
            "test": "one-sample t-test",
            "mean": float(sample.mean()),
            "std": float(sample.std()),
            "mu0": mu,
            "t_stat": float(stat),
            "p_value": float(p),
            "reject_H0": p < alpha,
            "alpha": alpha
        }

    # Independent two-sample t-test (Welch)
    elif test == "independent_t":
        if group is None:
            raise ValueError("Independent t-test requires 'group' parameter.")

        groups = data[group].unique()

        if len(groups) != 2:
            raise ValueError("Independent t-test requires exactly 2 groups.")

        g1 = data[data[group] == groups[0]][value].dropna()
        g2 = data[data[group] == groups[1]][value].dropna()

        stat, p = stats.ttest_ind(g1, g2, equal_var=False)

        return {
            "test": "independent t-test (Welch)",
            "groups": [str(groups[0]), str(groups[1])],
            "means": {
                str(groups[0]): float(data[data[group] == groups[0]][value].mean()),
                str(groups[1]): float(data[data[group] == groups[1]][value].mean())
            },
            "t_stat": float(stat),
            "p_value": float(p),
            "reject_H0": p < alpha,
            "alpha": alpha
        }

    # Paired t-test
    elif test == "paired_t":
        if group is None or subject is None:
            raise ValueError("Paired t-test requires 'group' and 'subject' parameters.")

        pivoted = data.pivot(index=subject, columns=group, values=value).dropna()

        if pivoted.shape[1] != 2:
            raise ValueError("Paired t-test requires exactly 2 conditions.")

        a, b = pivoted.iloc[:, 0], pivoted.iloc[:, 1]

        stat, p = stats.ttest_rel(a, b)

        return {
            "test": "paired t-test",
            "conditions": [str(pivoted.columns[0]), str(pivoted.columns[1])],
            "means": {
                str(pivoted.columns[0]): float(a.mean()),
                str(pivoted.columns[1]): float(b.mean())
            },
            "mean_difference": float((b - a).mean()),
            "t_stat": float(stat),
            "p_value": float(p),
            "reject_H0": p < alpha,
            "alpha": alpha
        }

    # One-way ANOVA
    elif test == "oneway_anova":
        if group is None:
            raise ValueError("One-way ANOVA requires 'group' parameter.")

        model = ols(f"{value} ~ C({group})", data=data).fit()
        anova = anova_lm(model, typ=2)

        return {
            "test": "one-way ANOVA",
            "f_statistic": float(anova["F"].iloc[0]),
            "p_value": float(anova["PR(>F)"].iloc[0]),
            "df_between": int(anova["df"].iloc[0]),
            "df_within": int(anova["df"].iloc[1]),
            "reject_H0": anova["PR(>F)"].iloc[0] < alpha,
            "alpha": alpha
        }

    # Repeated-measures ANOVA
    elif test == "rm_anova":
        if subject is None or group is None:
            raise ValueError("Repeated-measures ANOVA requires 'subject' and 'group' parameters.")

        rm = sm.stats.AnovaRM(
            data,
            depvar=value,
            subject=subject,
            within=[group]
        ).fit()

        return {
            "test": "repeated-measures ANOVA",
            "f_statistic": float(rm.anova_table["F Value"].iloc[0]),
            "p_value": float(rm.anova_table["Pr > F"].iloc[0]),
            "df": int(rm.anova_table["Num DF"].iloc[0]),
            "reject_H0": rm.anova_table["Pr > F"].iloc[0] < alpha,
            "alpha": alpha
        }

    # Two-way ANOVA with interaction
    elif test == "twoway_anova":
        if not isinstance(group, (list, tuple)) or len(group) != 2:
            raise ValueError("Two-way ANOVA requires group=['factor1', 'factor2'].")

        formula = f"{value} ~ C({group[0]}) * C({group[1]})"
        model = ols(formula, data=data).fit()
        anova = sm.stats.anova_lm(model, typ=2)

        return {
            "test": "two-way ANOVA",
            "anova_summary": anova.to_dict(),
            "alpha": alpha
        }

    # ANCOVA (group + covariate)
    elif test == "ancova":
        if group is None or covariate is None:
            raise ValueError("ANCOVA requires 'group' and 'covariate' parameters.")

        formula = f"{value} ~ C({group}) + {covariate}"
        model = ols(formula, data=data).fit()
        anova = sm.stats.anova_lm(model, typ=2)

        return {
            "test": "ANCOVA",
            "anova_summary": anova.to_dict(),
            "alpha": alpha
        }

    else:
        raise ValueError(
            "Invalid test name. Choose from: one_sample, independent_t, paired_t, "
            "oneway_anova, rm_anova, twoway_anova, ancova"
        )


# ==============================================================================
# 4. NONPARAMETRIC TEST
# ==============================================================================

def nonparametric_test(
    df: pd.DataFrame,
    test: str,
    value_col: str,
    group_col: Optional[str] = None,
    subject_col: Optional[str] = None,
    mu: float = 0,
    alpha: float = 0.05,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (7, 5)
) -> Dict[str, Any]:
    """
    Universal non-parametric test engine.

    Supports:
    - "wilcoxon_one_sample" : Wilcoxon signed-rank (one-sample)
    - "mannwhitney" : Mann–Whitney U (two independent groups)
    - "wilcoxon_paired" : Wilcoxon signed-rank (paired)
    - "kruskal" : Kruskal–Wallis (3+ independent groups)
    - "friedman" : Friedman Test (repeated measures)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    test : str
        Name of the statistical test.
    value_col : str
        Column name of the dependent variable.
    group_col : str, optional
        Grouping variable (required for most tests except one-sample).
    subject_col : str, optional
        Subject identifier (required for Friedman test).
    mu : float, default=0
        Population median under H0 (for one-sample Wilcoxon).
    alpha : float, default=0.05
        Significance level.
    show_plot : bool, default=True
        Whether to display visualization.
    figsize : tuple, default=(7, 5)
        Figure size for plots.

    Returns
    -------
    dict
        Dictionary containing test statistics, p-values, and interpretation.
    """
    data = df.copy()

    # ONE-SAMPLE WILCOXON
    if test == "wilcoxon_one_sample":
        sample = data[value_col].dropna()

        if len(sample) < 2:
            raise ValueError("Need at least 2 observations for Wilcoxon test.")

        stat, p = wilcoxon(sample - mu)

        interpretation = "Reject H0" if p < alpha else "Fail to reject H0"

        if show_plot:
            plt.figure(figsize=figsize)
            sns.histplot(sample, kde=True)
            plt.axvline(mu, color="red", label=f"H0 Median = {mu}")
            plt.title(f"Wilcoxon One-sample Test (p={p:.4f})")
            plt.legend()
            plt.show()

        return {
            "test": "Wilcoxon Signed-Rank (one-sample)",
            "median": float(sample.median()),
            "mu0": mu,
            "statistic": float(stat),
            "p_value": float(p),
            "alpha": alpha,
            "interpretation": interpretation,
            "reject_H0": p < alpha
        }

    # MANN–WHITNEY U (two independent groups)
    elif test == "mannwhitney":
        if group_col is None:
            raise ValueError("Mann-Whitney test requires 'group_col' parameter.")

        groups = data[group_col].unique()

        if len(groups) != 2:
            raise ValueError("Mann-Whitney test requires exactly 2 groups.")

        g1 = data[data[group_col] == groups[0]][value_col].dropna()
        g2 = data[data[group_col] == groups[1]][value_col].dropna()

        stat, p = mannwhitneyu(g1, g2, alternative="two-sided")

        if show_plot:
            plt.figure(figsize=figsize)
            sns.boxplot(data=data, x=group_col, y=value_col)
            plt.title(f"Mann–Whitney U Test (p={p:.4f})")
            plt.show()

        return {
            "test": "Mann–Whitney U",
            "groups": [str(groups[0]), str(groups[1])],
            "medians": {
                str(groups[0]): float(g1.median()),
                str(groups[1]): float(g2.median())
            },
            "statistic": float(stat),
            "p_value": float(p),
            "alpha": alpha,
            "interpretation": "Reject H0" if p < alpha else "Fail to reject H0",
            "reject_H0": p < alpha
        }

    # WILCOXON SIGNED-RANK (paired)
    elif test == "wilcoxon_paired":
        if group_col is None:
            raise ValueError("Wilcoxon paired test requires 'group_col' with 'before'/'after' values.")

        before = data[data[group_col] == "before"][value_col].dropna()
        after = data[data[group_col] == "after"][value_col].dropna()

        if len(before) != len(after):
            raise ValueError("Before and after groups must have same length.")

        stat, p = wilcoxon(before, after)

        if show_plot:
            plt.figure(figsize=figsize)
            paired_data = pd.DataFrame({
                "before": before.values,
                "after": after.values
            })
            sns.boxplot(data=paired_data)
            plt.title(f"Wilcoxon Paired Test (p={p:.4f})")
            plt.show()

        return {
            "test": "Wilcoxon Signed-Rank (paired)",
            "median_before": float(before.median()),
            "median_after": float(after.median()),
            "median_change": float((after - before).median()),
            "statistic": float(stat),
            "p_value": float(p),
            "alpha": alpha,
            "interpretation": "Reject H0" if p < alpha else "Fail to reject H0",
            "reject_H0": p < alpha
        }

    # KRUSKAL–WALLIS (3+ independent groups)
    elif test == "kruskal":
        if group_col is None:
            raise ValueError("Kruskal-Wallis test requires 'group_col' parameter.")

        groups = [g[value_col].dropna().values for _, g in data.groupby(group_col)]
        names = list(data[group_col].unique())

        if len(groups) < 2:
            raise ValueError("Kruskal-Wallis test requires at least 2 groups.")

        stat, p = kruskal(*groups)

        if show_plot:
            plt.figure(figsize=figsize)
            sns.boxplot(data=data, x=group_col, y=value_col)
            plt.title(f"Kruskal–Wallis Test (p={p:.4f})")
            plt.show()

        return {
            "test": "Kruskal–Wallis",
            "groups": names,
            "statistic": float(stat),
            "p_value": float(p),
            "alpha": alpha,
            "interpretation": "Reject H0" if p < alpha else "Fail to reject H0",
            "reject_H0": p < alpha
        }

    # FRIEDMAN TEST (repeated measures)
    elif test == "friedman":
        if subject_col is None or group_col is None:
            raise ValueError("Friedman test requires 'subject_col' and 'group_col' parameters.")

        wide = data.pivot(index=subject_col, columns=group_col, values=value_col).dropna()

        if wide.shape[1] < 2:
            raise ValueError("Friedman test requires at least 2 conditions.")

        stat, p = friedmanchisquare(*[wide[c].dropna().values for c in wide.columns])

        return {
            "test": "Friedman Test",
            "conditions": list(wide.columns),
            "statistic": float(stat),
            "p_value": float(p),
            "alpha": alpha,
            "interpretation": "Reject H0" if p < alpha else "Fail to reject H0",
            "reject_H0": p < alpha
        }

    else:
        raise ValueError(
            "Invalid test name. Choose from: wilcoxon_one_sample, mannwhitney, "
            "wilcoxon_paired, kruskal, friedman"
        )


# ==============================================================================
# 5. CORRELATION TEST
# ==============================================================================

def correlation_test(
    df: pd.DataFrame,
    test: str,
    x: str,
    y: str,
    control: Optional[str] = None,
    alpha: float = 0.05,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (6, 5)
) -> Dict[str, Any]:
    """
    Correlation analysis engine.

    Supports:
    - "pearson" : Pearson correlation
    - "spearman" : Spearman rank correlation
    - "kendall" : Kendall Tau correlation
    - "partial" : Partial correlation

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    test : str
        Type of correlation test.
    x : str
        First variable name.
    y : str
        Second variable name.
    control : str, optional
        Control variable (required for partial correlation).
    alpha : float, default=0.05
        Significance level.
    show_plot : bool, default=True
        Whether to display scatter plot.
    figsize : tuple, default=(6, 5)
        Figure size for the plot.

    Returns
    -------
    dict
        Dictionary containing correlation coefficient, p-value, and interpretation.
    """
    data = df[[x, y] + ([control] if control else [])].dropna()

    if len(data) < 3:
        raise ValueError("Need at least 3 complete observations for correlation.")

    if test == "pearson":
        r, p = pearsonr(data[x], data[y])

    elif test == "spearman":
        r, p = spearmanr(data[x], data[y])

    elif test == "kendall":
        r, p = kendalltau(data[x], data[y])

    elif test == "partial":
        if control is None:
            raise ValueError("Control variable required for partial correlation.")

        X1 = sm.add_constant(data[control])
        model_x = sm.OLS(data[x], X1).fit()
        residual_x = model_x.resid

        model_y = sm.OLS(data[y], X1).fit()
        residual_y = model_y.resid

        r, p = pearsonr(residual_x, residual_y)

    else:
        raise ValueError("Invalid test type. Choose: pearson, spearman, kendall, partial")

    interpretation = "Significant correlation" if p < alpha else "Not significant"

    if show_plot and test != "partial":
        plt.figure(figsize=figsize)
        sns.regplot(data=data, x=x, y=y, scatter_kws={"alpha": 0.6})
        plt.title(f"{test.capitalize()} Correlation (r={r:.3f}, p={p:.3f})")
        plt.show()

    return {
        "test": test.capitalize(),
        "correlation_coefficient": float(r),
        "p_value": float(p),
        "alpha": alpha,
        "interpretation": interpretation,
        "n_observations": int(len(data)),
        "reject_H0": p < alpha
    }


def find_correlated_features(
    df: pd.DataFrame,
    test: str = "pearson",
    alpha: float = 0.05,
    min_abs_corr: float = 0.3
) -> pd.DataFrame:
    """
    Find correlations between all feature pairs using correlation_test.

    Parameters
    ----------
    df : DataFrame
        Input dataset.
    test : str, default="pearson"
        pearson / spearman / kendall
    alpha : float, default=0.05
        Significance level.
    min_abs_corr : float, default=0.3
        Minimum absolute correlation to report.

    Returns
    -------
    DataFrame
        Results of correlation analysis between feature pairs.
    """
    results = []
    cols = df.columns

    for x, y in combinations(cols, 2):
        try:
            res = correlation_test(
                df=df,
                test=test,
                x=x,
                y=y,
                alpha=alpha,
                show_plot=False
            )

            r = res["correlation_coefficient"]
            p = res["p_value"]

            if abs(r) >= min_abs_corr and p < alpha:
                results.append({
                    "feature_1": x,
                    "feature_2": y,
                    "correlation": r,
                    "p_value": p,
                    "n": res["n_observations"]
                })
        except Exception:
            continue

    return pd.DataFrame(results).sort_values(
        "correlation",
        key=lambda x: abs(x),
        ascending=False
    )


def feature_correlation_analysis(
    df: pd.DataFrame,
    method: str = "spearman",
    corr_threshold: float = 0.8,
    detect_onehot: bool = True,
    rare_threshold: float = 0.005,
    drop_constant: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze feature correlations with support for one-hot encoded features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    method : str, default="spearman"
        Correlation method (pearson, spearman, kendall).
    corr_threshold : float, default=0.8
        Threshold for detecting high correlations.
    detect_onehot : bool, default=True
        Whether to detect one-hot encoded feature groups.
    rare_threshold : float, default=0.005
        Threshold for rare binary columns to drop.
    drop_constant : bool, default=True
        Whether to drop constant columns.

    Returns
    -------
    result_df : pandas.DataFrame
        DataFrame with correlated feature pairs.
    report : dict
        Summary report of the analysis.
    """
    data = df.copy()

    if drop_constant:
        data = data.loc[:, data.nunique() > 1]

    binary_cols = [
        c for c in data.columns
        if set(data[c].dropna().unique()).issubset({0, 1})
    ]

    rare_cols = [
        c for c in binary_cols
        if data[c].mean() < rare_threshold
    ]

    data = data.drop(columns=rare_cols)

    groups = {}

    if detect_onehot:
        for col in data.columns:
            match = re.match(r"^([^_]+)_", col)
            if match:
                parent = match.group(1)
            else:
                parent = col
            groups.setdefault(parent, []).append(col)
    else:
        for col in data.columns:
            groups[col] = [col]

    corr = data.corr(method=method)
    results = []

    for g1, g2 in combinations(groups.keys(), 2):
        cols1 = groups[g1]
        cols2 = groups[g2]

        sub_corr = corr.loc[cols1, cols2]
        abs_corr = np.abs(sub_corr.values)
        max_corr = abs_corr.max()

        if max_corr >= corr_threshold:
            idx = np.unravel_index(abs_corr.argmax(), sub_corr.shape)

            results.append({
                "feature_1": g1,
                "feature_2": g2,
                "max_abs_corr": max_corr,
                "method": method,
                "example_col_1": cols1[idx[0]],
                "example_col_2": cols2[idx[1]]
            })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values("max_abs_corr", ascending=False)

    report = {
        "n_samples": data.shape[0],
        "n_features": data.shape[1],
        "n_feature_groups": len(groups),
        "method": method,
        "threshold": corr_threshold
    }

    return result_df, report


# ==============================================================================
# 6. CATEGORICAL TEST
# ==============================================================================

def categorical_test(
    df: pd.DataFrame,
    test: str,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    expected: Optional[List[float]] = None,
    alpha: float = 0.05,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (6, 5)
) -> Dict[str, Any]:
    """
    Categorical Data Test Engine.

    Supports:
    - "chi_square_independence" : Chi-square Test of Independence
    - "chi_square_gof" : Chi-square Goodness-of-fit
    - "fisher_exact" : Fisher's Exact Test (for 2x2 tables)
    - "mcnemar" : McNemar Test (paired categorical)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    test : str
        Type of categorical test.
    col1 : str, optional
        First categorical variable.
    col2 : str, optional
        Second categorical variable (for independence tests).
    expected : list, optional
        Expected proportions or counts (for goodness-of-fit).
    alpha : float, default=0.05
        Significance level.
    show_plot : bool, default=True
        Whether to display heatmap of contingency table.
    figsize : tuple, default=(6, 5)
        Figure size for the plot.

    Returns
    -------
    dict
        Dictionary containing test statistics, p-values, and interpretation.
    """
    # CHI-SQUARE TEST OF INDEPENDENCE
    if test == "chi_square_independence":
        if col1 is None or col2 is None:
            raise ValueError("Chi-square independence requires col1 and col2 parameters.")

        table = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected_vals = chi2_contingency(table)

        interpretation = (
            "Reject H0: variables are associated"
            if p < alpha else
            "Fail to reject H0: no significant association"
        )

        if show_plot:
            plt.figure(figsize=figsize)
            sns.heatmap(table, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Chi-Square Independence Test (p={p:.4f})")
            plt.show()

        return {
            "test": "Chi-Square Test of Independence",
            "contingency_table": table.to_dict(),
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "expected_counts": expected_vals.tolist(),
            "interpretation": interpretation,
            "reject_H0": p < alpha,
            "alpha": alpha
        }

    # CHI-SQUARE GOODNESS-OF-FIT
    elif test == "chi_square_gof":
        if col1 is None:
            raise ValueError("Chi-square goodness-of-fit requires col1 parameter.")

        if expected is None:
            raise ValueError("Expected proportions or counts must be provided.")

        observed = df[col1].value_counts().sort_index()

        if len(observed) != len(expected):
            raise ValueError("Length of observed and expected must match.")

        chi2, p = chisquare(f_obs=observed.values, f_exp=expected)

        interpretation = (
            "Reject H0: observed ≠ expected distribution"
            if p < alpha else
            "Fail to reject H0: observed follows expected distribution"
        )

        if show_plot:
            plt.figure(figsize=figsize)
            sns.barplot(x=observed.index.astype(str), y=observed.values)
            plt.title(f"Chi-Square Goodness-of-fit (p={p:.4f})")
            plt.ylabel("Observed Frequency")
            plt.show()

        return {
            "test": "Chi-Square Goodness-of-fit",
            "observed": observed.to_dict(),
            "expected": expected,
            "chi2": float(chi2),
            "p_value": float(p),
            "interpretation": interpretation,
            "reject_H0": p < alpha,
            "alpha": alpha
        }

    # FISHER'S EXACT TEST (2x2 ONLY!)
    elif test == "fisher_exact":
        if col1 is None or col2 is None:
            raise ValueError("Fisher's exact test requires col1 and col2 parameters.")

        table = pd.crosstab(df[col1], df[col2])

        if table.shape != (2, 2):
            raise ValueError("Fisher's Exact Test requires a 2x2 contingency table.")

        oddsratio, p = fisher_exact(table)

        interpretation = (
            "Reject H0: association present"
            if p < alpha else
            "Fail to reject H0: no significant association"
        )

        if show_plot:
            plt.figure(figsize=figsize)
            sns.heatmap(table, annot=True, fmt="d", cmap="Reds")
            plt.title(f"Fisher's Exact Test (p={p:.4f})")
            plt.show()

        return {
            "test": "Fisher Exact Test",
            "contingency_table": table.to_dict(),
            "odds_ratio": float(oddsratio),
            "p_value": float(p),
            "interpretation": interpretation,
            "reject_H0": p < alpha,
            "alpha": alpha
        }

    # MCNEMAR TEST (paired categorical)
    elif test == "mcnemar":
        if col1 is None or col2 is None:
            raise ValueError("McNemar test requires col1 and col2 parameters (paired observations).")

        table = pd.crosstab(df[col1], df[col2])

        result = mcnemar(table, exact=False)
        p = result.pvalue
        stat = result.statistic

        interpretation = (
            "Reject H0: significant discordance"
            if p < alpha else
            "Fail to reject H0: no significant discordance"
        )

        if show_plot:
            plt.figure(figsize=figsize)
            sns.heatmap(table, annot=True, fmt="d", cmap="Greens")
            plt.title(f"McNemar Test (p={p:.4f})")
            plt.show()

        return {
            "test": "McNemar Test",
            "contingency_table": table.to_dict(),
            "statistic": float(stat),
            "p_value": float(p),
            "interpretation": interpretation,
            "reject_H0": p < alpha,
            "alpha": alpha
        }

    else:
        raise ValueError(
            "Invalid test name. Choose from: chi_square_independence, chi_square_gof, "
            "fisher_exact, mcnemar"
        )


# ==============================================================================
# 7. EFFECT SIZE
# ==============================================================================

def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "very small"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _interpret_eta_squared(eta2: float) -> str:
    """Interpret eta squared effect size."""
    eta2 = abs(eta2)
    if eta2 < 0.01:
        return "very small"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"


def _interpret_cramers_v(v: float) -> str:
    """Interpret Cramer's V effect size (general guidelines)."""
    v = abs(v)
    if v < 0.1:
        return "very small"
    elif v < 0.3:
        return "small"
    elif v < 0.5:
        return "medium"
    else:
        return "large"


def effect_size(
    df: pd.DataFrame,
    metric: str,
    value_col: Optional[str] = None,
    group_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Effect Size Calculator

    Supports:
    - "cohens_d" : Cohen's d (for two groups)
    - "hedges_g" : Hedges' g (bias-corrected Cohen's d)
    - "eta_squared" : Eta squared (for ANOVA)
    - "omega_squared" : Omega squared (unbiased effect size)
    - "cliffs_delta" : Cliff's Delta (non-parametric)
    - "cramers_v" : Cramer's V (for categorical association)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    metric : str
        Type of effect size to calculate.
    value_col : str, optional
        Numerical variable (required for most metrics).
    group_col : str, optional
        Grouping variable (required for most metrics).

    Returns
    -------
    dict
        Dictionary containing effect size name and value.
    """
    data = df.copy()

    # COHEN'S D
    if metric == "cohens_d":
        if value_col is None or group_col is None:
            raise ValueError("Cohen's d requires value_col and group_col parameters.")

        groups = data[group_col].unique()

        if len(groups) != 2:
            raise ValueError("Cohen's d requires exactly 2 groups.")

        g1 = data[data[group_col] == groups[0]][value_col].dropna()
        g2 = data[data[group_col] == groups[1]][value_col].dropna()

        n1, n2 = len(g1), len(g2)

        pooled_sd = np.sqrt(
            ((n1 - 1) * np.var(g1, ddof=1) +
             (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
        )

        d = (np.mean(g1) - np.mean(g2)) / pooled_sd

        return {
            "effect_size": "Cohen's d",
            "value": float(d),
            "magnitude": _interpret_cohens_d(abs(d))
        }

    # HEDGES' G
    elif metric == "hedges_g":
        if value_col is None or group_col is None:
            raise ValueError("Hedges' g requires value_col and group_col parameters.")

        groups = data[group_col].unique()

        if len(groups) != 2:
            raise ValueError("Hedges' g requires exactly 2 groups.")

        g1 = data[data[group_col] == groups[0]][value_col].dropna()
        g2 = data[data[group_col] == groups[1]][value_col].dropna()

        n1, n2 = len(g1), len(g2)

        pooled_sd = np.sqrt(
            ((n1 - 1) * np.var(g1, ddof=1) +
             (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
        )

        d = (np.mean(g1) - np.mean(g2)) / pooled_sd

        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        g = d * correction

        return {
            "effect_size": "Hedges' g",
            "value": float(g),
            "magnitude": _interpret_cohens_d(abs(g))
        }

    # ETA SQUARED
    elif metric == "eta_squared":
        if value_col is None or group_col is None:
            raise ValueError("Eta squared requires value_col and group_col parameters.")

        groups = data[group_col].unique()
        grand_mean = data[value_col].mean()

        ss_between = sum(
            len(data[data[group_col] == g]) *
            (data[data[group_col] == g][value_col].mean() - grand_mean) ** 2
            for g in groups
        )

        ss_total = sum((data[value_col] - grand_mean) ** 2)

        eta2 = ss_between / ss_total if ss_total > 0 else 0

        return {
            "effect_size": "Eta squared",
            "value": float(eta2),
            "magnitude": _interpret_eta_squared(eta2)
        }

    # OMEGA SQUARED
    elif metric == "omega_squared":
        if value_col is None or group_col is None:
            raise ValueError("Omega squared requires value_col and group_col parameters.")

        groups = data[group_col].unique()
        grand_mean = data[value_col].mean()

        ss_between = sum(
            len(data[data[group_col] == g]) *
            (data[data[group_col] == g][value_col].mean() - grand_mean) ** 2
            for g in groups
        )

        ss_total = sum((data[value_col] - grand_mean) ** 2)

        df_between = len(groups) - 1
        df_within = len(data) - len(groups)

        ms_within = (ss_total - ss_between) / df_within if df_within > 0 else 0

        omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0

        return {
            "effect_size": "Omega squared",
            "value": float(omega2),
            "magnitude": _interpret_eta_squared(omega2)
        }

    # CLIFF'S DELTA
    elif metric == "cliffs_delta":
        if value_col is None or group_col is None:
            raise ValueError("Cliff's Delta requires value_col and group_col parameters.")

        groups = data[group_col].unique()

        if len(groups) != 2:
            raise ValueError("Cliff's Delta requires exactly 2 groups.")

        g1 = data[data[group_col] == groups[0]][value_col].dropna().values
        g2 = data[data[group_col] == groups[1]][value_col].dropna().values

        n1, n2 = len(g1), len(g2)

        more = 0
        less = 0

        for x in g1:
            for y in g2:
                if x > y:
                    more += 1
                elif x < y:
                    less += 1

        delta = (more - less) / (n1 * n2) if (n1 * n2) > 0 else 0

        return {
            "effect_size": "Cliff's Delta",
            "value": float(delta)
        }

    # CRAMER'S V
    elif metric == "cramers_v":
        if group_col is None or value_col is None:
            raise ValueError("Cramer's V requires group_col and value_col (as second category) parameters.")

        table = pd.crosstab(data[group_col], data[value_col])

        chi2, p, dof, exp = chi2_contingency(table)

        n = table.sum().sum()
        k = min(table.shape) - 1

        v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0

        return {
            "effect_size": "Cramer's V",
            "value": float(v),
            "magnitude": _interpret_cramers_v(v)
        }

    else:
        raise ValueError(
            "Invalid metric. Choose from: cohens_d, hedges_g, eta_squared, "
            "omega_squared, cliffs_delta, cramers_v"
        )


# ==============================================================================
# 8. ADDITIONAL UTILITY FUNCTION
# ==============================================================================

def tukey_hsd_posthoc(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform Tukey HSD post-hoc test after ANOVA.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    value_col : str
        Numerical variable.
    group_col : str
        Grouping variable.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    pandas.DataFrame
        Results of Tukey HSD test.
    """
    tukey = pairwise_tukeyhsd(df[value_col], df[group_col], alpha=alpha)
    return pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])


# ==============================================================================
# 9. MODULE EXPORTS
# ==============================================================================

__all__ = [
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
]
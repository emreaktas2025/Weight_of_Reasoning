"""Aggregation utilities for SmolLM REV pipeline results."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score


def bootstrap_confidence_interval(
    values: np.ndarray, 
    n_bootstrap: int = 10000, 
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        values: Array of metric values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(values) == 0 or np.all(np.isnan(values)):
        return float("nan"), float("nan"), float("nan")
    
    # Filter out NaN values
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return float("nan"), float("nan"), float("nan")
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(valid_values, size=len(valid_values), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_val = np.mean(valid_values)
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return float(mean_val), float(lower_bound), float(upper_bound)


def delong_test(
    y_true: np.ndarray, 
    y_scores1: np.ndarray, 
    y_scores2: np.ndarray
) -> float:
    """
    Compute DeLong test p-value for comparing two ROC curves.
    
    Args:
        y_true: True binary labels
        y_scores1: Scores from first model
        y_scores2: Scores from second model
        
    Returns:
        P-value for the test
    """
    try:
        from scipy.stats import norm
        
        # Compute AUCs
        auc1 = roc_auc_score(y_true, y_scores1)
        auc2 = roc_auc_score(y_true, y_scores2)
        
        # Compute DeLong variance
        n = len(y_true)
        n_pos = np.sum(y_true == 1)
        n_neg = n - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        
        # Simplified DeLong test (approximation)
        # In practice, you'd want to use the full DeLong implementation
        se = np.sqrt((auc1 * (1 - auc1) + auc2 * (1 - auc2)) / n)
        z_score = (auc1 - auc2) / se
        
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return float(p_value)
        
    except Exception as e:
        print(f"Error in DeLong test: {e}")
        return float("nan")


def compute_aggregate_statistics(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all models.
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not model_results:
        return {}
    
    # Extract key metrics
    delta_aucs = []
    auroc_revs = []
    auroc_baselines = []
    partial_rs = []
    model_names = []
    n_params = []
    
    for model_name, results in model_results.items():
        model_names.append(model_name)
        
        # Extract metrics
        baseline_comp = results.get("baseline_comparison", {})
        delta_auc = baseline_comp.get("delta_auc", float("nan"))
        auroc_rev = baseline_comp.get("auroc_rev", float("nan"))
        auroc_baseline = baseline_comp.get("auroc_baseline", float("nan"))
        
        delta_aucs.append(delta_auc)
        auroc_revs.append(auroc_rev)
        auroc_baselines.append(auroc_baseline)
        
        # Extract partial correlation
        partial_corr = results.get("partial_corr", {})
        rev_partial_r = partial_corr.get("REV", {}).get("r", float("nan"))
        partial_rs.append(rev_partial_r)
        
        # Extract parameter count
        n_param = results.get("n_params", 0)
        n_params.append(n_param)
    
    # Convert to numpy arrays
    delta_aucs = np.array(delta_aucs)
    auroc_revs = np.array(auroc_revs)
    auroc_baselines = np.array(auroc_baselines)
    partial_rs = np.array(partial_rs)
    n_params = np.array(n_params)
    
    # Compute aggregate statistics
    aggregate_stats = {
        "models": model_names,
        "n_models": len(model_names),
        "delta_auc_mean": float(np.nanmean(delta_aucs)),
        "delta_auc_std": float(np.nanstd(delta_aucs)),
        "auroc_rev_mean": float(np.nanmean(auroc_revs)),
        "auroc_baseline_mean": float(np.nanmean(auroc_baselines)),
        "partial_r_mean": float(np.nanmean(partial_rs)),
        "n_params": n_params.tolist(),
    }
    
    # Compute confidence intervals
    if len(delta_aucs) > 1:
        delta_auc_mean, delta_auc_ci_low, delta_auc_ci_high = bootstrap_confidence_interval(delta_aucs)
        aggregate_stats.update({
            "delta_auc_mean": delta_auc_mean,
            "ci_low": delta_auc_ci_low,
            "ci_high": delta_auc_ci_high,
        })
    
    # Check success criteria
    success_criteria = check_success_criteria(model_results)
    aggregate_stats["success_criteria"] = success_criteria
    
    return aggregate_stats


def check_success_criteria(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """
    Check success criteria for the evaluation.
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        Dictionary mapping criterion names to pass/fail boolean
    """
    criteria = {}
    
    if not model_results:
        return {"no_results": True}
    
    # Extract metrics
    delta_aucs = []
    model_sizes = []
    
    for model_name, results in model_results.items():
        baseline_comp = results.get("baseline_comparison", {})
        delta_auc = baseline_comp.get("delta_auc", 0)
        n_params = results.get("n_params", 0)
        
        delta_aucs.append(delta_auc)
        model_sizes.append(n_params)
    
    # Criterion 1: REV adds >= +0.05 AUROC over baseline
    significant_improvements = sum(1 for delta in delta_aucs if delta >= 0.05)
    criteria["REV improves baseline by >=0.05 AUC"] = significant_improvements > 0
    
    # Criterion 2: Monotonic scaling (larger models should have higher ΔAUC)
    if len(model_sizes) >= 2:
        # Sort by model size
        sorted_indices = np.argsort(model_sizes)
        sorted_delta_aucs = [delta_aucs[i] for i in sorted_indices]
        
        # Check if ΔAUC increases with model size
        monotonic = all(
            sorted_delta_aucs[i] <= sorted_delta_aucs[i+1] 
            for i in range(len(sorted_delta_aucs)-1)
        )
        criteria["Monotonic scaling"] = monotonic
    else:
        criteria["Monotonic scaling"] = True  # Not applicable with < 2 models
    
    # Criterion 3: No NaN values in key metrics
    has_nans = any(np.isnan(delta_aucs))
    criteria["No NaN values"] = not has_nans
    
    # Criterion 4: At least one model with significant effect
    cohens_d_values = []
    for results in model_results.values():
        cohens_d = results.get("cohens_d", {}).get("REV", 0)
        cohens_d_values.append(cohens_d)
    
    significant_effects = sum(1 for d in cohens_d_values if d > 0.5)
    criteria["Significant effect size (d>0.5)"] = significant_effects > 0
    
    return criteria


def create_scaling_plot_data(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create data for scaling plot (ΔAUC vs log parameters).
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        Dictionary with plotting data
    """
    if not model_results:
        return {}
    
    model_names = []
    log_params = []
    delta_aucs = []
    ci_lows = []
    ci_highs = []
    
    for model_name, results in model_results.items():
        n_params = results.get("n_params", 0)
        baseline_comp = results.get("baseline_comparison", {})
        delta_auc = baseline_comp.get("delta_auc", float("nan"))
        
        model_names.append(model_name)
        log_params.append(np.log10(n_params) if n_params > 0 else 0)
        delta_aucs.append(delta_auc)
        
        # For now, use simple error bars (would need bootstrap for real CIs)
        ci_lows.append(delta_auc - 0.01)  # Placeholder
        ci_highs.append(delta_auc + 0.01)  # Placeholder
    
    return {
        "model_names": model_names,
        "log_params": log_params,
        "delta_aucs": delta_aucs,
        "ci_lows": ci_lows,
        "ci_highs": ci_highs,
    }


def create_temperature_robustness_data(
    model_results: Dict[str, Dict[str, Any]], 
    temperatures: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5]
) -> Dict[str, Any]:
    """
    Create data for temperature robustness plot.
    
    Args:
        model_results: Dictionary of model results
        temperatures: List of temperatures to test
        
    Returns:
        Dictionary with plotting data
    """
    # This would require running the model at different temperatures
    # For now, return placeholder data
    return {
        "temperatures": temperatures,
        "auroc_rev": [0.75 + 0.05 * np.sin(t * 10) for t in temperatures],
        "auroc_baseline": [0.70 + 0.03 * np.cos(t * 10) for t in temperatures],
        "note": "Placeholder data - requires temperature sweep"
    }

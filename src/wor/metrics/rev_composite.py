"""Composite REV (Reasoning Effort Value) score computation.

REV is a composite metric that combines all six reasoning effort metrics:
- AE (Activation Energy)
- APE (Attention Process Entropy) 
- APL (Activation Path Length)
- CUD (Circuit Utilization Density)
- SIB (Stability of Intermediate Beliefs)
- FL (Feature Load)

The composite score is computed as the mean of z-scored individual metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def zscore_metric(values: np.ndarray, use_robust: bool = False) -> np.ndarray:
    """
    Compute z-scores for a metric across all samples.
    
    Args:
        values: Array of metric values
        use_robust: Whether to use robust z-scoring (median/MAD)
        
    Returns:
        Z-scored values
    """
    if len(values) == 0:
        return np.array([])
    
    # Filter out NaN values
    valid_mask = np.isfinite(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) == 0:
        return np.full_like(values, float("nan"))
    
    if use_robust:
        # Robust z-scoring using median and MAD
        median_val = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median_val))
        
        if mad == 0:
            # Fallback to standard z-scoring if MAD is zero
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            if std_val == 0:
                z_scores = np.zeros_like(valid_values)
            else:
                z_scores = (valid_values - mean_val) / std_val
        else:
            z_scores = (valid_values - median_val) / (1.4826 * mad)  # 1.4826 is consistency factor
    else:
        # Standard z-scoring
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        
        if std_val == 0:
            z_scores = np.zeros_like(valid_values)
        else:
            z_scores = (valid_values - mean_val) / std_val
    
    # Create output array with NaN preservation
    result = np.full_like(values, float("nan"))
    result[valid_mask] = z_scores
    
    return result


def compute_rev_scores(df: pd.DataFrame, use_robust: bool = False) -> np.ndarray:
    """
    Compute REV scores for all samples in the dataset.
    
    Args:
        df: DataFrame with columns AE, APE, APL, CUD, SIB, FL
        use_robust: Whether to use robust z-scoring
        
    Returns:
        Array of REV scores
    """
    # Define the six metrics
    metrics = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL']
    
    # Check that all required columns exist
    missing_cols = [col for col in metrics if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for REV computation: {missing_cols}")
    
    # Extract metric values
    metric_values = {}
    for metric in metrics:
        metric_values[metric] = df[metric].values
    
    # Compute z-scores for each metric
    z_scores = {}
    for metric in metrics:
        z_scores[metric] = zscore_metric(metric_values[metric], use_robust=use_robust)
    
    # Compute REV as mean of z-scores
    z_score_matrix = np.column_stack([z_scores[metric] for metric in metrics])
    rev_scores = np.nanmean(z_score_matrix, axis=1)
    
    return rev_scores


def compute_rev_scores_from_dict(metric_dict: Dict[str, List[float]], 
                                use_robust: bool = False) -> List[float]:
    """
    Compute REV scores from a dictionary of metric lists.
    
    Args:
        metric_dict: Dict with keys AE, APE, APL, CUD, SIB, FL and lists of values
        use_robust: Whether to use robust z-scoring
        
    Returns:
        List of REV scores
    """
    # Define the six metrics
    metrics = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL']
    
    # Check that all required keys exist
    missing_keys = [key for key in metrics if key not in metric_dict]
    if missing_keys:
        raise ValueError(f"Missing required keys for REV computation: {missing_keys}")
    
    # Convert to numpy arrays
    metric_arrays = {}
    for metric in metrics:
        metric_arrays[metric] = np.array(metric_dict[metric])
    
    # Compute z-scores for each metric
    z_scores = {}
    for metric in metrics:
        z_scores[metric] = zscore_metric(metric_arrays[metric], use_robust=use_robust)
    
    # Compute REV as mean of z-scores
    z_score_matrix = np.column_stack([z_scores[metric] for metric in metrics])
    rev_scores = np.nanmean(z_score_matrix, axis=1)
    
    return rev_scores.tolist()


def compute_individual_z_scores(df: pd.DataFrame, use_robust: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute z-scores for each individual metric.
    
    Args:
        df: DataFrame with columns AE, APE, APL, CUD, SIB, FL
        use_robust: Whether to use robust z-scoring
        
    Returns:
        Dict mapping metric names to z-score arrays
    """
    metrics = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL']
    
    # Check that all required columns exist
    missing_cols = [col for col in metrics if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    z_scores = {}
    for metric in metrics:
        values = df[metric].values
        z_scores[metric] = zscore_metric(values, use_robust=use_robust)
    
    return z_scores


def get_rev_description() -> str:
    """
    Get description of the REV metric for documentation.
    
    Returns:
        Description string
    """
    return """
    REV (Reasoning Effort Value) is a composite metric that combines all six
    reasoning effort metrics into a single score:
    
    1. AE (Activation Energy): Magnitude of hidden state activations
    2. APE (Attention Process Entropy): Diversity of attention patterns
    3. APL (Activation Path Length): Depth of reasoning pathways
    4. CUD (Circuit Utilization Density): Fraction of reasoning circuit heads engaged
    5. SIB (Stability of Intermediate Beliefs): Robustness under paraphrasing
    6. FL (Feature Load): Activation sparsity proxy
    
    REV = (z_AE + z_APE + z_APL + z_CUD + z_SIB + z_FL) / 6
    
    Where each z_i is the z-scored version of metric i across the full dataset.
    This equal-weighting approach ensures all metrics contribute equally to the
    composite score, providing a unified measure of reasoning effort.
    
    Higher REV values indicate greater reasoning effort across multiple dimensions.
    """


def validate_rev_scores(rev_scores: np.ndarray) -> bool:
    """
    Validate that REV scores are reasonable.
    
    Args:
        rev_scores: Array of REV scores
        
    Returns:
        True if scores are valid, False otherwise
    """
    if len(rev_scores) == 0:
        return False
    
    # Check for NaN values
    if np.any(np.isnan(rev_scores)):
        print("Warning: REV scores contain NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(rev_scores)):
        print("Warning: REV scores contain infinite values")
        return False
    
    # Check reasonable range (z-scores should typically be in [-3, 3])
    if np.any(np.abs(rev_scores) > 5):
        print("Warning: REV scores have extreme values (|z| > 5)")
    
    return True


def compute_rev_statistics(rev_scores: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    """
    Compute statistics for REV scores by label.
    
    Args:
        rev_scores: Array of REV scores
        labels: List of labels (e.g., 'reasoning', 'control')
        
    Returns:
        Dict with statistics
    """
    if len(rev_scores) != len(labels):
        raise ValueError("Length of REV scores and labels must match")
    
    # Filter out NaN values
    valid_mask = np.isfinite(rev_scores)
    valid_scores = rev_scores[valid_mask]
    valid_labels = [labels[i] for i in range(len(labels)) if valid_mask[i]]
    
    if len(valid_scores) == 0:
        return {"error": "No valid REV scores"}
    
    # Compute statistics by label
    unique_labels = list(set(valid_labels))
    stats = {}
    
    for label in unique_labels:
        label_scores = [valid_scores[i] for i in range(len(valid_scores)) if valid_labels[i] == label]
        if len(label_scores) > 0:
            stats[f"REV_{label}"] = {
                "mean": float(np.mean(label_scores)),
                "std": float(np.std(label_scores)),
                "count": len(label_scores),
                "min": float(np.min(label_scores)),
                "max": float(np.max(label_scores))
            }
    
    # Compute Cohen's d if we have both reasoning and control
    if "reasoning" in unique_labels and "control" in unique_labels:
        reasoning_scores = [valid_scores[i] for i in range(len(valid_scores)) if valid_labels[i] == "reasoning"]
        control_scores = [valid_scores[i] for i in range(len(valid_scores)) if valid_labels[i] == "control"]
        
        if len(reasoning_scores) > 1 and len(control_scores) > 1:
            # Compute pooled standard deviation
            reasoning_var = np.var(reasoning_scores, ddof=1)
            control_var = np.var(control_scores, ddof=1)
            pooled_std = np.sqrt(0.5 * (reasoning_var + control_var))
            
            if pooled_std > 0:
                cohens_d = (np.mean(reasoning_scores) - np.mean(control_scores)) / pooled_std
                stats["cohens_d_REV"] = float(cohens_d)
    
    return stats

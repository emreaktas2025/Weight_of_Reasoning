"""Partial correlation analysis for controlling confounding variables."""

import os
import pandas as pd
import pingouin as pg
from typing import Dict, Any
from ..core.utils import save_json


def compute_partial_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute partial correlations between metrics and label, controlling for token_len and ppl.
    
    Args:
        df: DataFrame with columns AE, APE, APL, token_len, ppl, label_num
        
    Returns:
        Dict with structure: {metric: {'r': ..., 'p': ...}}
    """
    print("Computing partial correlations...")
    
    # Ensure required columns exist
    required_cols = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL', 'REV', 'token_len', 'ppl', 'label_num']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out rows with NaN values
    df_clean = df.dropna(subset=required_cols)
    if len(df_clean) == 0:
        print("Warning: No valid data for partial correlation analysis")
        return {}
    
    print(f"Computing partial correlations on {len(df_clean)} samples")
    
    # Compute partial correlations for each metric
    results = {}
    metrics = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL', 'REV']
    
    for metric in metrics:
        try:
            # Use pingouin for robust partial correlation
            result = pg.partial_corr(
                data=df_clean,
                x=metric,
                y='label_num',
                covar=['token_len', 'ppl']
            )
            
            # Extract r and p values
            r = float(result['r'].iloc[0])
            p = float(result['p-val'].iloc[0])
            
            results[metric] = {
                'r': r,
                'p': p
            }
            
            print(f"  {metric}: r={r:.4f}, p={p:.4f}")
            
        except Exception as e:
            print(f"Error computing partial correlation for {metric}: {e}")
            results[metric] = {
                'r': float('nan'),
                'p': float('nan')
            }
    
    return results


def save_partial_correlations(results: Dict[str, Dict[str, float]], 
                            filepath: str = "reports/partial_corr.json") -> None:
    """
    Save partial correlation results to JSON file.
    
    Args:
        results: Results from compute_partial_correlations
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_json(filepath, results)
    print(f"Partial correlations saved to {filepath}")


def manual_partial_corr(df: pd.DataFrame, x: str, y: str, covar: list) -> Dict[str, float]:
    """
    Manual implementation of partial correlation using regression residuals.
    This is a backup method if pingouin is not available.
    
    Args:
        df: DataFrame with data
        x: X variable name
        y: Y variable name
        covar: List of covariate variable names
        
    Returns:
        Dict with 'r' and 'p' values
    """
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Filter out NaN values
    df_clean = df.dropna(subset=[x, y] + covar)
    
    if len(df_clean) < 3:  # Need at least 3 points for correlation
        return {'r': float('nan'), 'p': float('nan')}
    
    # Get variables
    X = df_clean[x].values.reshape(-1, 1)
    Y = df_clean[y].values.reshape(-1, 1)
    Z = df_clean[covar].values
    
    # Fit regression models
    reg_x = LinearRegression().fit(Z, X.ravel())
    reg_y = LinearRegression().fit(Z, Y.ravel())
    
    # Compute residuals
    X_resid = X.ravel() - reg_x.predict(Z)
    Y_resid = Y.ravel() - reg_y.predict(Z)
    
    # Compute correlation of residuals
    r, p = pearsonr(X_resid, Y_resid)
    
    return {'r': float(r), 'p': float(p)}

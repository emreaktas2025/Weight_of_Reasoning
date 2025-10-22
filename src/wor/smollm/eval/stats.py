"""Statistical analysis utilities for SmolLM REV pipeline."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import pearsonr
import re


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
        print(f"Warning: Missing columns: {missing_cols}")
        # Use available columns
        available_cols = [col for col in required_cols if col in df.columns]
        required_cols = available_cols
    
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
        if metric not in df_clean.columns:
            continue
            
        try:
            # Manual partial correlation using regression residuals
            result = manual_partial_corr(
                df_clean, 
                x=metric, 
                y='label_num', 
                covar=['token_len', 'ppl']
            )
            
            results[metric] = result
            print(f"  {metric}: r={result['r']:.4f}, p={result['p']:.4f}")
            
        except Exception as e:
            print(f"Error computing partial correlation for {metric}: {e}")
            results[metric] = {
                'r': float('nan'),
                'p': float('nan')
            }
    
    return results


def manual_partial_corr(df: pd.DataFrame, x: str, y: str, covar: list) -> Dict[str, float]:
    """
    Manual implementation of partial correlation using regression residuals.
    
    Args:
        df: DataFrame with data
        x: X variable name
        y: Y variable name
        covar: List of covariate variable names
        
    Returns:
        Dict with 'r' and 'p' values
    """
    from sklearn.linear_model import LinearRegression
    
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


def extract_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract baseline features for comparison with REV.
    
    Args:
        df: DataFrame with at least 'token_len', 'ppl', 'generated_text' columns
        
    Returns:
        DataFrame with baseline feature columns added
    """
    df = df.copy()
    
    # Ensure token_len exists
    if 'token_len' not in df.columns:
        df['token_len'] = df['generated_text'].apply(lambda x: len(str(x).split()))
    
    # Compute avg_logprob from perplexity: ppl = exp(-avg_logprob)
    if 'ppl' in df.columns:
        df['avg_logprob'] = -np.log(df['ppl'].replace([np.inf, -np.inf], np.nan))
    else:
        df['avg_logprob'] = 0.0
    
    # Perplexity is already in df as 'ppl', rename for clarity
    if 'ppl' in df.columns:
        df['perplexity'] = df['ppl']
    else:
        df['perplexity'] = 1.0
    
    # Compute CoT length as count of reasoning markers
    if 'generated_text' in df.columns:
        def compute_cot_length(text: str) -> int:
            if pd.isna(text):
                return 0
            text_lower = str(text).lower()
            
            # Count reasoning markers
            markers = ['because', 'therefore', 'thus', 'so ', 'step', 'first', 'second', 'then', 'finally']
            count = sum(text_lower.count(marker) for marker in markers)
            
            # Count numbered steps (e.g., "1.", "2:", "Step 1")
            numbered_steps = len(re.findall(r'\b\d+[.:]|\bstep\s+\d+', text_lower))
            
            return count + numbered_steps * 2  # Weight numbered steps more
        
        df['cot_len'] = df['generated_text'].apply(compute_cot_length)
    else:
        df['cot_len'] = 0
    
    return df


def train_baseline_classifier(
    df: pd.DataFrame,
    features: List[str] = ['token_len', 'avg_logprob', 'perplexity', 'cot_len'],
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train logistic regression on baseline features.
    
    Args:
        df: DataFrame with features and 'label_num' (0=control, 1=reasoning)
        features: List of feature column names to use
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Filter valid samples
    valid_mask = df[features + ['label_num']].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        raise ValueError("No valid samples for training baseline classifier")
    
    # Extract features and labels
    X = df_valid[features].values
    y = df_valid['label_num'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_scaled, y)
    
    return model, scaler


def evaluate_baseline_vs_rev(
    df: pd.DataFrame,
    baseline_features: List[str] = ['token_len', 'avg_logprob', 'perplexity', 'cot_len'],
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluate baseline predictors vs REV and combined model.
    
    Computes:
    - AUROC_baseline: Using only baseline features
    - AUROC_REV: Using only REV score
    - AUROC_combined: Using baseline + REV
    - delta_AUC: Improvement from adding REV to baseline
    
    Args:
        df: DataFrame with baseline features, REV, and label_num
        baseline_features: List of baseline feature names
        random_state: Random seed
        
    Returns:
        Dict with AUROC scores and delta_AUC
    """
    # Extract baseline features
    df = extract_baseline_features(df)
    
    # Filter valid samples
    required_cols = baseline_features + ['REV', 'label_num']
    valid_mask = df[required_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) < 10:
        print(f"Warning: Only {len(df_valid)} valid samples for baseline evaluation")
        return {
            "auroc_baseline": float('nan'),
            "auroc_rev": float('nan'),
            "auroc_combined": float('nan'),
            "delta_auc": float('nan'),
            "n_samples": len(df_valid)
        }
    
    y = df_valid['label_num'].values
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print("Warning: Only one class present in data")
        return {
            "auroc_baseline": float('nan'),
            "auroc_rev": float('nan'),
            "auroc_combined": float('nan'),
            "delta_auc": float('nan'),
            "n_samples": len(df_valid)
        }
    
    # 1. AUROC for baseline features only
    try:
        baseline_model, baseline_scaler = train_baseline_classifier(
            df_valid, baseline_features, random_state
        )
        X_baseline = baseline_scaler.transform(df_valid[baseline_features].values)
        baseline_probs = baseline_model.predict_proba(X_baseline)[:, 1]
        auroc_baseline = roc_auc_score(y, baseline_probs)
    except Exception as e:
        print(f"Warning: Baseline AUROC computation failed: {e}")
        auroc_baseline = float('nan')
    
    # 2. AUROC for REV only
    try:
        rev_scores = df_valid['REV'].values
        auroc_rev = roc_auc_score(y, rev_scores)
    except Exception as e:
        print(f"Warning: REV AUROC computation failed: {e}")
        auroc_rev = float('nan')
    
    # 3. AUROC for combined (baseline + REV)
    try:
        combined_features = baseline_features + ['REV']
        combined_model, combined_scaler = train_baseline_classifier(
            df_valid, combined_features, random_state
        )
        X_combined = combined_scaler.transform(df_valid[combined_features].values)
        combined_probs = combined_model.predict_proba(X_combined)[:, 1]
        auroc_combined = roc_auc_score(y, combined_probs)
    except Exception as e:
        print(f"Warning: Combined AUROC computation failed: {e}")
        auroc_combined = float('nan')
    
    # 4. Compute delta AUC
    delta_auc = auroc_combined - auroc_baseline if not np.isnan(auroc_combined) and not np.isnan(auroc_baseline) else float('nan')
    
    return {
        "auroc_baseline": float(auroc_baseline),
        "auroc_rev": float(auroc_rev),
        "auroc_combined": float(auroc_combined),
        "delta_auc": float(delta_auc),
        "n_samples": int(len(df_valid)),
        "baseline_features": baseline_features,
        "feature_importances": {
            "baseline": baseline_model.coef_[0].tolist() if not np.isnan(auroc_baseline) else [],
            "combined": combined_model.coef_[0].tolist() if not np.isnan(auroc_combined) else []
        }
    }


def get_roc_curves(
    df: pd.DataFrame,
    baseline_features: List[str] = ['token_len', 'avg_logprob', 'perplexity', 'cot_len'],
    random_state: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get ROC curve data for plotting.
    
    Args:
        df: DataFrame with features, REV, and label_num
        baseline_features: List of baseline feature names
        random_state: Random seed
        
    Returns:
        Dict with 'baseline', 'rev', 'combined' keys, each containing fpr, tpr, thresholds
    """
    df = extract_baseline_features(df)
    
    # Filter valid samples
    required_cols = baseline_features + ['REV', 'label_num']
    valid_mask = df[required_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) < 10 or len(np.unique(df_valid['label_num'])) < 2:
        return {}
    
    y = df_valid['label_num'].values
    roc_data = {}
    
    # Baseline ROC
    try:
        baseline_model, baseline_scaler = train_baseline_classifier(
            df_valid, baseline_features, random_state
        )
        X_baseline = baseline_scaler.transform(df_valid[baseline_features].values)
        baseline_probs = baseline_model.predict_proba(X_baseline)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, baseline_probs)
        roc_data['baseline'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    except:
        pass
    
    # REV ROC
    try:
        rev_scores = df_valid['REV'].values
        fpr, tpr, thresholds = roc_curve(y, rev_scores)
        roc_data['rev'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    except:
        pass
    
    # Combined ROC
    try:
        combined_features = baseline_features + ['REV']
        combined_model, combined_scaler = train_baseline_classifier(
            df_valid, combined_features, random_state
        )
        X_combined = combined_scaler.transform(df_valid[combined_features].values)
        combined_probs = combined_model.predict_proba(X_combined)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, combined_probs)
        roc_data['combined'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    except:
        pass
    
    return roc_data

"""Feature Load (FL) metric computation using activation sparsity proxy.

Note: FL is a sparsity-based proxy for feature load, not SAE-derived monosemantic features.
This implementation uses L0/L1 sparsity measures on raw activations as a proxy for
the density of reasoning-related features activated in the model.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from transformer_lens import HookedTransformer


def extract_mid_layer_activations(model: HookedTransformer, cache: Dict[str, torch.Tensor],
                                 input_tokens: torch.Tensor, reasoning_window: int = 24) -> Optional[np.ndarray]:
    """
    Extract activations from mid-layer for reasoning window.
    
    Args:
        model: HookedTransformer model
        cache: Cache from forward pass
        input_tokens: Original input tokens
        reasoning_window: Number of tokens to consider as reasoning window
        
    Returns:
        Activations for reasoning window, shape (window_size, hidden_dim)
    """
    try:
        # Get mid-layer index
        n_layers = model.cfg.n_layers
        mid_layer = n_layers // 2
        
        # Look for mid-layer activations in cache
        mid_layer_key = f"blocks.{mid_layer}.hook_resid_post"
        if mid_layer_key not in cache:
            # Fallback: look for any mid-layer activation
            for key in cache.keys():
                if f"blocks.{mid_layer}." in key and "resid" in key:
                    mid_layer_key = key
                    break
        
        if mid_layer_key not in cache:
            return None
        
        # Extract activations
        activations = cache[mid_layer_key].detach().cpu().numpy()  # (batch, seq_len, hidden_dim)
        activations = activations[0]  # Remove batch dimension
        
        # Get reasoning window (last N tokens excluding final one)
        seq_len = activations.shape[0]
        if reasoning_window + 1 >= seq_len:
            # If window is too large, use all tokens except last
            window = activations[:-1, :] if seq_len > 1 else activations
        else:
            # Take reasoning window from the end, excluding final token
            window = activations[-(reasoning_window + 1):-1, :]
        
        return window
        
    except Exception as e:
        print(f"Error extracting mid-layer activations: {e}")
        return None


def compute_l1_load(activations: np.ndarray) -> float:
    """
    Compute L1 load (mean absolute activation).
    
    Args:
        activations: Activation matrix, shape (seq_len, hidden_dim)
        
    Returns:
        L1 load value
    """
    if activations.size == 0:
        return float("nan")
    
    # Compute mean absolute activation across all dimensions
    l1_load = np.mean(np.abs(activations))
    return float(l1_load)


def compute_l0_sparsity(activations: np.ndarray) -> float:
    """
    Compute L0 sparsity (fraction of active features).
    
    Args:
        activations: Activation matrix, shape (seq_len, hidden_dim)
        
    Returns:
        L0 sparsity value (fraction of active features)
    """
    if activations.size == 0:
        return float("nan")
    
    # Compute threshold as median absolute activation per sample
    abs_activations = np.abs(activations)
    threshold = np.median(abs_activations)
    
    # Count active features (above threshold) per token
    active_features_per_token = np.sum(abs_activations > threshold, axis=1)
    
    # Compute mean fraction of active features
    hidden_dim = activations.shape[1]
    l0_sparsity = np.mean(active_features_per_token) / hidden_dim
    
    return float(l0_sparsity)


def compute_feature_load(model: HookedTransformer, cache: Dict[str, torch.Tensor],
                        input_tokens: torch.Tensor, reasoning_window: int = 24) -> float:
    """
    Compute Feature Load (FL) for a single prompt.
    
    Args:
        model: HookedTransformer model
        cache: Cache from forward pass
        input_tokens: Original input tokens
        reasoning_window: Number of tokens to consider as reasoning window
        
    Returns:
        FL value (sparsity-based proxy)
    """
    try:
        # Extract mid-layer activations
        activations = extract_mid_layer_activations(model, cache, input_tokens, reasoning_window)
        
        if activations is None or activations.size == 0:
            return float("nan")
        
        # Compute L1 load
        l1_load = compute_l1_load(activations)
        
        # Compute L0 sparsity
        l0_sparsity = compute_l0_sparsity(activations)
        
        # For single-sample computation, return mean of both measures
        # (z-scoring will be applied across the full dataset in post-processing)
        fl_raw = (l1_load + l0_sparsity) / 2.0
        
        return float(fl_raw)
        
    except Exception as e:
        print(f"Error computing FL: {e}")
        return float("nan")


def compute_feature_load_with_zscoring(fl_values: np.ndarray, use_robust: bool = False) -> np.ndarray:
    """
    Apply z-scoring to FL values across the dataset.
    
    Args:
        fl_values: Array of FL values across all samples
        use_robust: Whether to use robust z-scoring (median/MAD)
        
    Returns:
        Z-scored FL values
    """
    if len(fl_values) == 0:
        return np.array([])
    
    # Filter out NaN values
    valid_mask = np.isfinite(fl_values)
    valid_values = fl_values[valid_mask]
    
    if len(valid_values) == 0:
        return np.full_like(fl_values, float("nan"))
    
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
    result = np.full_like(fl_values, float("nan"))
    result[valid_mask] = z_scores
    
    return result


def compute_l1_l0_components(model: HookedTransformer, cache: Dict[str, torch.Tensor],
                            input_tokens: torch.Tensor, reasoning_window: int = 24) -> tuple[float, float]:
    """
    Compute L1 load and L0 sparsity components separately.
    
    Args:
        model: HookedTransformer model
        cache: Cache from forward pass
        input_tokens: Original input tokens
        reasoning_window: Number of tokens to consider as reasoning window
        
    Returns:
        Tuple of (L1_load, L0_sparsity)
    """
    try:
        # Extract mid-layer activations
        activations = extract_mid_layer_activations(model, cache, input_tokens, reasoning_window)
        
        if activations is None or activations.size == 0:
            return float("nan"), float("nan")
        
        # Compute components
        l1_load = compute_l1_load(activations)
        l0_sparsity = compute_l0_sparsity(activations)
        
        return l1_load, l0_sparsity
        
    except Exception as e:
        print(f"Error computing FL components: {e}")
        return float("nan"), float("nan")


def get_feature_load_description() -> str:
    """
    Get description of the FL metric for documentation.
    
    Returns:
        Description string
    """
    return """
    Feature Load (FL) is a sparsity-based proxy for the density of reasoning-related 
    features activated in the model. It combines:
    
    1. L1 Load: Mean absolute activation magnitude (intensity)
    2. L0 Sparsity: Fraction of features above median threshold (spread)
    
    Note: FL is NOT derived from Sparse Autoencoders (SAEs) or monosemantic features.
    It uses raw activation sparsity as a proxy for feature utilization density.
    
    Higher FL values indicate more intense and widespread feature activation,
    potentially reflecting increased computational effort during reasoning.
    """

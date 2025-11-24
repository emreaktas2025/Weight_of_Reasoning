"""Activation Energy (AE) metric computation."""

import numpy as np
from typing import Optional, Tuple


def activation_energy(hidden_states: Optional[np.ndarray], reasoning_len: int = None, 
                      token_range: Optional[Tuple[int, int]] = None) -> float:
    """
    Compute Activation Energy (AE) metric.
    
    Args:
        hidden_states: (seq_len, hidden_dim) hidden states for full prompt+generation
        reasoning_len: number of tokens considered as 'reasoning window' at the end of sequence (deprecated, use token_range)
        token_range: (start_idx, end_idx) tuple specifying token range to analyze. If None, uses reasoning_len fallback.
        
    Returns:
        AE value, length-normalized by token count
    """
    if hidden_states is None:
        return float("nan")
    
    seq_len = hidden_states.shape[0]
    
    # Use token_range if provided, otherwise fall back to reasoning_len
    if token_range is not None:
        start_idx, end_idx = token_range
        # Clamp to valid range
        start_idx = max(0, min(start_idx, seq_len - 1))
        end_idx = max(start_idx + 1, min(end_idx, seq_len))
        window = hidden_states[start_idx:end_idx, :]
    elif reasoning_len is not None and reasoning_len > 0:
        # Legacy behavior: Take last reasoning_len tokens excluding final one
        if reasoning_len + 1 >= seq_len:
            # If reasoning window is too large, use all tokens except last
            window = hidden_states[:-1, :] if seq_len > 1 else hidden_states
        else:
            # Take reasoning window from the end, excluding final token
            window = hidden_states[-(reasoning_len + 1):-1, :]
    else:
        return float("nan")
    
    if window.size == 0:
        return float("nan")
    
    # Compute feature-wise standard deviation for normalization
    std = window.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    # Compute L2 norms of normalized hidden states
    normalized_window = window / std
    norms = np.linalg.norm(normalized_window, axis=1)
    
    # Raw AE is mean of norms
    ae_raw = norms.mean()
    
    # Length normalization
    ae_normalized = ae_raw / max(1, window.shape[0])
    
    return float(ae_normalized)

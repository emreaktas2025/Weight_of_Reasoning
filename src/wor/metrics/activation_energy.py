"""Activation Energy (AE) metric computation."""

import numpy as np
from typing import Optional


def activation_energy(hidden_states: Optional[np.ndarray], reasoning_len: int) -> float:
    """
    Compute Activation Energy (AE) metric.
    
    Args:
        hidden_states: (seq_len, hidden_dim) hidden states for full prompt+generation
        reasoning_len: number of tokens considered as 'reasoning window' at the end of sequence
        
    Returns:
        AE value, length-normalized by token count
    """
    if hidden_states is None or reasoning_len <= 0:
        return float("nan")
    
    # Take last reasoning_len tokens excluding final one (assumed to be answer)
    seq_len = hidden_states.shape[0]
    if reasoning_len + 1 >= seq_len:
        # If reasoning window is too large, use all tokens except last
        window = hidden_states[:-1, :] if seq_len > 1 else hidden_states
    else:
        # Take reasoning window from the end, excluding final token
        window = hidden_states[-(reasoning_len + 1):-1, :]
    
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

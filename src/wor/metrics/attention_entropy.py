"""Attention Process Entropy (APE) metric computation."""

import numpy as np
from scipy.stats import entropy
from typing import Optional, Tuple


def attention_process_entropy(attention_probs: Optional[np.ndarray], reasoning_len: int = None,
                             token_range: Optional[Tuple[int, int]] = None) -> float:
    """
    Compute Attention Process Entropy (APE) metric.
    
    Args:
        attention_probs: (seq_len, n_heads, seq_len) attention probabilities per head
        reasoning_len: number of tokens in reasoning window (deprecated, use token_range)
        token_range: (start_idx, end_idx) tuple specifying token range to analyze. If None, uses reasoning_len fallback.
        
    Returns:
        APE value, mean entropy across heads and reasoning tokens
    """
    try:
        if attention_probs is None:
            return float("nan")
        
        # Ensure we have the right shape: (seq_len, n_heads, seq_len)
        if attention_probs.ndim != 3:
            return float("nan")
        
        seq_len, n_heads, _ = attention_probs.shape
        
        # Determine reasoning window
        if token_range is not None:
            start_idx, end_idx = token_range
            # Clamp to valid range
            start_idx = max(0, min(start_idx, seq_len - 1))
            end_idx = max(start_idx + 1, min(end_idx, seq_len))
            reasoning_tokens = list(range(start_idx, end_idx))
        elif reasoning_len is not None and reasoning_len > 0:
            # Legacy behavior: Take last reasoning_len tokens, excluding final one
            if reasoning_len + 1 >= seq_len:
                reasoning_tokens = list(range(seq_len - 1)) if seq_len > 1 else []
            else:
                reasoning_tokens = list(range(seq_len - reasoning_len - 1, seq_len - 1))
        else:
            # Use all tokens except the last one
            reasoning_tokens = list(range(seq_len - 1)) if seq_len > 1 else []
        
        if not reasoning_tokens:
            return float("nan")
        
        # Compute entropy for each (token, head) pair in reasoning window
        entropies = []
        for t in reasoning_tokens:
            for h in range(n_heads):
                # Get attention distribution for this token and head
                attn_dist = attention_probs[t, h, :]
                
                # Add small epsilon to avoid log(0)
                attn_dist = attn_dist + 1e-12
                attn_dist = attn_dist / attn_dist.sum()  # Renormalize
                
                # Compute entropy
                ent = entropy(attn_dist, base=2)
                entropies.append(ent)
        
        if not entropies:
            return float("nan")
        
        # APE is mean entropy across reasoning window
        ape = np.mean(entropies)
        
        # Length normalization if reasoning_len provided
        if reasoning_len is not None and reasoning_len > 0:
            ape = ape / max(1, len(reasoning_tokens))
        
        return float(ape)
        
    except Exception:
        # Safe fallback: return NaN if anything goes wrong
        return float("nan")

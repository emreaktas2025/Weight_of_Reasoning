"""Attention Process Entropy (APE) metric computation for SmolLM pipeline."""

import numpy as np
from scipy.stats import entropy
from typing import Optional


def attention_process_entropy(attention_probs: Optional[np.ndarray], reasoning_len: int = None) -> float:
    """
    Compute Attention Process Entropy (APE) metric.
    
    Args:
        attention_probs: (seq_len, n_heads, seq_len) attention probabilities per head
        reasoning_len: number of tokens in reasoning window (optional, for length normalization)
        
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
        if reasoning_len is not None and reasoning_len > 0:
            # Take last reasoning_len tokens, excluding final one
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
                attn_dist = attn_dist + 1e-8
                attn_dist = attn_dist / attn_dist.sum()  # Renormalize
                
                # Compute entropy
                ent = entropy(attn_dist)
                entropies.append(ent)
        
        # APE is the mean entropy across reasoning window
        ape = np.mean(entropies) if entropies else float("nan")
        
        return float(ape)
        
    except Exception as e:
        print(f"Error computing APE: {e}")
        return float("nan")


def compute_ape_from_cache(cache: dict, reasoning_len: int = None) -> float:
    """
    Compute APE from model cache.
    
    Args:
        cache: Model activation cache
        reasoning_len: Number of tokens in reasoning window
        
    Returns:
        APE value
    """
    try:
        # Look for attention patterns in cache
        attn_key = None
        for key in cache.keys():
            if "pattern" in key:
                attn_key = key
                break
        
        if attn_key is None:
            return float("nan")
        
        # Extract attention probabilities
        attention_probs = cache[attn_key].detach().cpu().numpy()
        if attention_probs.ndim == 4:  # (batch, n_heads, seq_len, seq_len)
            attention_probs = attention_probs[0]  # Remove batch dimension
            attention_probs = attention_probs.transpose(1, 0, 2)  # (seq_len, n_heads, seq_len)
        
        return attention_process_entropy(attention_probs, reasoning_len)
        
    except Exception as e:
        print(f"Error computing APE from cache: {e}")
        return float("nan")

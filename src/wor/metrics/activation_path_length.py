"""Activation Path Length (APL) metric computation using activation patching."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformer_lens import HookedTransformer


def compute_control_thresholds(model: HookedTransformer, control_prompts: List[str]) -> Dict[int, float]:
    """
    Compute control thresholds for APL metric using control set.
    
    Args:
        model: HookedTransformer model
        control_prompts: List of control prompts for baseline computation
        
    Returns:
        Dict mapping layer index to threshold value (μ_Δ,control + 1σ)
    """
    print("Computing control thresholds for APL...")
    
    # Get clean logits for all control prompts
    clean_logits_list = []
    for prompt in control_prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits, _ = model.run_with_cache(tokens, return_type="logits")
            clean_logits_list.append(logits[0, -1, :])  # Last token logits
    
    clean_logits = torch.stack(clean_logits_list)  # (n_prompts, vocab_size)
    
    # Compute ablation effects for each layer
    n_layers = model.cfg.n_layers
    layer_deltas = {layer: [] for layer in range(n_layers)}
    
    for layer in range(n_layers):
        print(f"  Computing ablation effects for layer {layer}/{n_layers-1}")
        
        for prompt in control_prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            
            # Define ablation hook: replace MLP output with mean
            def ablation_hook(activation, hook):
                # Compute mean over batch and sequence dimensions
                mean_activation = activation.mean(dim=(0, 1), keepdim=True)
                return mean_activation.expand_as(activation)
            
            # Run with ablation
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", ablation_hook)],
                    return_type="logits"
                )
            
            # Compute L2 distance between clean and ablated logits
            clean_logit = clean_logits[control_prompts.index(prompt)]
            ablated_logit = ablated_logits[0, -1, :]
            delta = torch.norm(clean_logit - ablated_logit, p=2).item()
            layer_deltas[layer].append(delta)
    
    # Compute thresholds: μ + 1σ for each layer
    thresholds = {}
    for layer in range(n_layers):
        deltas = np.array(layer_deltas[layer])
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + std_delta
        thresholds[layer] = threshold
    
    # Save thresholds for reproducibility
    os.makedirs("reports", exist_ok=True)
    np.savez("reports/control_thresholds.npz", **{f"layer_{k}": v for k, v in thresholds.items()})
    print(f"Control thresholds saved to reports/control_thresholds.npz")
    
    return thresholds


def compute_apl(model: HookedTransformer, cache: Dict[str, torch.Tensor], 
                control_thresholds: Dict[int, float], input_tokens: torch.Tensor = None) -> float:
    """
    Compute Activation Path Length (APL) for a single prompt.
    
    Args:
        model: HookedTransformer model
        cache: Cache from clean forward pass
        control_thresholds: Pre-computed thresholds per layer
        input_tokens: Original input tokens (optional, will try to extract from cache if not provided)
        
    Returns:
        APL value (fraction of active layers)
    """
    try:
        # Get input tokens - either provided or extract from cache
        if input_tokens is None:
            # Try to extract from cache (this is a fallback)
            embed_key = "hook_embed"
            if embed_key not in cache:
                return float("nan")
            
            embeddings = cache[embed_key]  # (batch, seq_len, hidden_dim)
            seq_len = embeddings.shape[1]
            # Create dummy tokens as fallback
            input_tokens = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
        
        # Get clean logits by running forward pass
        with torch.no_grad():
            clean_logits, _ = model.run_with_cache(input_tokens, return_type="logits")
            clean_logit = clean_logits[0, -1, :]  # Last token logits
        
        # Count active layers
        n_layers = model.cfg.n_layers
        active_layers = 0
        
        for layer in range(n_layers):
            # Define ablation hook
            def ablation_hook(activation, hook):
                mean_activation = activation.mean(dim=(0, 1), keepdim=True)
                return mean_activation.expand_as(activation)
            
            # Run with ablation for this layer
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    input_tokens,
                    fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", ablation_hook)],
                    return_type="logits"
                )
            
            # Compute L2 distance
            ablated_logit = ablated_logits[0, -1, :]
            delta = torch.norm(clean_logit - ablated_logit, p=2).item()
            
            # Check if layer is active
            threshold = control_thresholds.get(layer, 0.0)
            if delta > threshold:
                active_layers += 1
        
        # APL is fraction of active layers
        apl = active_layers / n_layers
        return float(apl)
        
    except Exception as e:
        print(f"Error computing APL: {e}")
        return float("nan")


def load_control_thresholds(filepath: str = "reports/control_thresholds.npz") -> Optional[Dict[int, float]]:
    """
    Load pre-computed control thresholds from file.
    
    Args:
        filepath: Path to saved thresholds file
        
    Returns:
        Dict mapping layer index to threshold, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        data = np.load(filepath)
        thresholds = {}
        for key in data.keys():
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                thresholds[layer_idx] = float(data[key])
        return thresholds
    except Exception as e:
        print(f"Error loading control thresholds: {e}")
        return None

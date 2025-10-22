"""Activation Path Length (APL) metric computation for SmolLM pipeline."""

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
                    fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", ablation_hook)],
                    return_type="logits"
                )
            
            # Compute KL divergence
            clean_logits_single = clean_logits[len(layer_deltas[layer])]
            ablated_logits_single = ablated_logits[0, -1, :]
            
            kl_div = F.kl_div(
                F.log_softmax(ablated_logits_single, dim=-1),
                F.softmax(clean_logits_single, dim=-1),
                reduction='sum'
            )
            
            layer_deltas[layer].append(kl_div.item())
    
    # Compute thresholds: mean + 1 std
    thresholds = {}
    for layer in range(n_layers):
        deltas = layer_deltas[layer]
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + std_delta
        thresholds[layer] = float(threshold)
        print(f"  Layer {layer}: threshold = {threshold:.4f} (μ={mean_delta:.4f}, σ={std_delta:.4f})")
    
    return thresholds


def load_control_thresholds(filepath: str = "cache/control_thresholds.npz") -> Optional[Dict[int, float]]:
    """
    Load pre-computed control thresholds.
    
    Args:
        filepath: Path to saved thresholds
        
    Returns:
        Dict mapping layer index to threshold, or None if not found
    """
    try:
        data = np.load(filepath)
        thresholds = {int(k): float(v) for k, v in data.items()}
        print(f"Loaded control thresholds from {filepath}")
        return thresholds
    except:
        return None


def compute_apl(model: HookedTransformer, cache: Dict[str, torch.Tensor], 
                thresholds: Dict[int, float], input_tokens: torch.Tensor) -> float:
    """
    Compute Activation Path Length (APL) metric.
    
    Args:
        model: HookedTransformer model
        cache: Activation cache from model run
        thresholds: Pre-computed control thresholds per layer
        input_tokens: Input tokens for the sequence
        
    Returns:
        APL value (number of layers with significant activation changes)
    """
    try:
        if not thresholds:
            return float("nan")
        
        # Get input length
        input_len = input_tokens.shape[1]
        
        # Compute ablation effects for each layer
        n_layers = model.cfg.n_layers
        significant_layers = 0
        
        for layer in range(n_layers):
            if layer not in thresholds:
                continue
            
            # Get clean logits (from cache if available)
            try:
                # Try to get logits from cache
                logits_key = f"blocks.{layer}.hook_resid_post"
                if logits_key in cache:
                    # Use cached activations to compute logits
                    clean_logits = model.ln_final(cache[logits_key][0, -1, :])
                    clean_logits = model.unembed(clean_logits)
                else:
                    # Fallback: run forward pass
                    with torch.no_grad():
                        clean_logits, _ = model.run_with_cache(input_tokens, return_type="logits")
                        clean_logits = clean_logits[0, -1, :]
            except:
                return float("nan")
            
            # Define ablation hook
            def ablation_hook(activation, hook):
                mean_activation = activation.mean(dim=(0, 1), keepdim=True)
                return mean_activation.expand_as(activation)
            
            # Run with ablation
            try:
                with torch.no_grad():
                    ablated_logits = model.run_with_hooks(
                        input_tokens,
                        fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", ablation_hook)],
                        return_type="logits"
                    )
                    ablated_logits = ablated_logits[0, -1, :]
            except:
                continue
            
            # Compute KL divergence
            try:
                kl_div = F.kl_div(
                    F.log_softmax(ablated_logits, dim=-1),
                    F.softmax(clean_logits, dim=-1),
                    reduction='sum'
                ).item()
                
                # Check if this layer is significant
                if kl_div > thresholds[layer]:
                    significant_layers += 1
                    
            except:
                continue
        
        # APL is the fraction of significant layers
        apl = significant_layers / n_layers if n_layers > 0 else 0.0
        
        return float(apl)
        
    except Exception as e:
        print(f"Error computing APL: {e}")
        return float("nan")

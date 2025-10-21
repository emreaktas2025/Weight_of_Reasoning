"""Circuit Utilization Density (CUD) metric computation using head-level ablation."""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from transformer_lens import HookedTransformer


def compute_circuit_heads(model: HookedTransformer, arithmetic_prompts: List[str], 
                         control_prompts: List[str], max_heads: int = 24) -> Tuple[List[Tuple[int, int]], Dict[int, float]]:
    """
    Discover circuit heads using arithmetic prompts and compute control thresholds.
    
    Args:
        model: HookedTransformer model
        arithmetic_prompts: K=4 arithmetic prompts for circuit discovery
        control_prompts: Control prompts for threshold computation
        max_heads: Maximum number of circuit heads to select (M=24)
        
    Returns:
        Tuple of (circuit_heads, control_thresholds)
        circuit_heads: List of (layer, head) tuples
        control_thresholds: Dict mapping (layer, head) to threshold value
    """
    print("Discovering circuit heads...")
    
    # Get clean logits for arithmetic prompts
    arithmetic_clean_logits = []
    for prompt in arithmetic_prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits, _ = model.run_with_cache(tokens, return_type="logits")
            arithmetic_clean_logits.append(logits[0, -1, :])  # Last token logits
    
    # Get clean logits for control prompts
    control_clean_logits = []
    for prompt in control_prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits, _ = model.run_with_cache(tokens, return_type="logits")
            control_clean_logits.append(logits[0, -1, :])
    
    # Compute head importance scores
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_deltas = {}
    
    print(f"Computing head importance for {n_layers} layers x {n_heads} heads...")
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # Compute ablation effect for arithmetic prompts
            arithmetic_deltas = []
            for i, prompt in enumerate(arithmetic_prompts):
                tokens = model.to_tokens(prompt, prepend_bos=True)
                
                # Define ablation hook: zero out this head's output
                def ablation_hook(activation, hook):
                    # Zero out the specific head
                    activation[:, :, head, :] = 0.0
                    return activation
                
                # Run with ablation
                with torch.no_grad():
                    ablated_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", ablation_hook)],
                        return_type="logits"
                    )
                
                # Compute L2 distance
                clean_logit = arithmetic_clean_logits[i]
                ablated_logit = ablated_logits[0, -1, :]
                delta = torch.norm(clean_logit - ablated_logit, p=2).item()
                arithmetic_deltas.append(delta)
            
            # Store mean delta for this head
            head_deltas[(layer, head)] = np.mean(arithmetic_deltas)
    
    # Select top heads by importance
    sorted_heads = sorted(head_deltas.items(), key=lambda x: x[1], reverse=True)
    n_total_heads = n_layers * n_heads
    top_20_percent = max(1, int(0.2 * n_total_heads))
    n_select = min(max_heads, top_20_percent)
    
    circuit_heads = [head for head, _ in sorted_heads[:n_select]]
    print(f"Selected {len(circuit_heads)} circuit heads (top {n_select} of {n_total_heads})")
    
    # Compute control thresholds for circuit heads
    print("Computing control thresholds...")
    control_thresholds = {}
    
    for layer, head in circuit_heads:
        # Compute ablation effects on control prompts
        control_deltas = []
        for i, prompt in enumerate(control_prompts):
            tokens = model.to_tokens(prompt, prepend_bos=True)
            
            # Define ablation hook
            def ablation_hook(activation, hook):
                activation[:, :, head, :] = 0.0
                return activation
            
            # Run with ablation
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", ablation_hook)],
                    return_type="logits"
                )
            
            # Compute L2 distance
            clean_logit = control_clean_logits[i]
            ablated_logit = ablated_logits[0, -1, :]
            delta = torch.norm(clean_logit - ablated_logit, p=2).item()
            control_deltas.append(delta)
        
        # Compute threshold: μ + 1σ
        mean_delta = np.mean(control_deltas)
        std_delta = np.std(control_deltas)
        threshold = mean_delta + std_delta
        control_thresholds[(layer, head)] = threshold
    
    return circuit_heads, control_thresholds


def save_circuit_heads(circuit_heads: List[Tuple[int, int]], 
                      control_thresholds: Dict[Tuple[int, int], float],
                      heads_file: str = "reports/circuits/heads.json",
                      thresholds_file: str = "reports/control_thresholds_phase3.npz") -> None:
    """
    Save circuit heads and thresholds to files.
    
    Args:
        circuit_heads: List of (layer, head) tuples
        control_thresholds: Dict mapping (layer, head) to threshold
        heads_file: Path to save circuit heads JSON
        thresholds_file: Path to save thresholds NPZ
    """
    # Create directories
    os.makedirs(os.path.dirname(heads_file), exist_ok=True)
    os.makedirs(os.path.dirname(thresholds_file), exist_ok=True)
    
    # Save circuit heads as JSON
    heads_data = [{"layer": layer, "head": head} for layer, head in circuit_heads]
    with open(heads_file, 'w') as f:
        json.dump(heads_data, f, indent=2)
    
    # Save thresholds as NPZ
    threshold_data = {}
    for (layer, head), threshold in control_thresholds.items():
        key = f"layer_{layer}_head_{head}"
        threshold_data[key] = threshold
    
    np.savez(thresholds_file, **threshold_data)
    
    print(f"Circuit heads saved to {heads_file}")
    print(f"Control thresholds saved to {thresholds_file}")


def load_circuit_heads(heads_file: str = "reports/circuits/heads.json") -> Optional[List[Tuple[int, int]]]:
    """
    Load circuit heads from JSON file.
    
    Args:
        heads_file: Path to circuit heads JSON file
        
    Returns:
        List of (layer, head) tuples, or None if file doesn't exist
    """
    if not os.path.exists(heads_file):
        return None
    
    try:
        with open(heads_file, 'r') as f:
            heads_data = json.load(f)
        
        circuit_heads = [(item["layer"], item["head"]) for item in heads_data]
        return circuit_heads
    except Exception as e:
        print(f"Error loading circuit heads: {e}")
        return None


def load_control_thresholds(thresholds_file: str = "reports/control_thresholds_phase3.npz") -> Optional[Dict[Tuple[int, int], float]]:
    """
    Load control thresholds from NPZ file.
    
    Args:
        thresholds_file: Path to thresholds NPZ file
        
    Returns:
        Dict mapping (layer, head) to threshold, or None if file doesn't exist
    """
    if not os.path.exists(thresholds_file):
        return None
    
    try:
        data = np.load(thresholds_file)
        thresholds = {}
        
        for key in data.keys():
            if key.startswith("layer_") and "_head_" in key:
                parts = key.split("_")
                layer = int(parts[1])
                head = int(parts[3])
                thresholds[(layer, head)] = float(data[key])
        
        return thresholds
    except Exception as e:
        print(f"Error loading control thresholds: {e}")
        return None


def compute_cud(model: HookedTransformer, cache: Dict[str, torch.Tensor],
                circuit_heads: List[Tuple[int, int]], 
                control_thresholds: Dict[Tuple[int, int], float],
                input_tokens: torch.Tensor) -> float:
    """
    Compute Circuit Utilization Density (CUD) for a single prompt.
    
    Args:
        model: HookedTransformer model
        cache: Cache from clean forward pass
        circuit_heads: List of (layer, head) tuples defining the circuit
        control_thresholds: Pre-computed thresholds per head
        input_tokens: Original input tokens
        
    Returns:
        CUD value (fraction of active circuit heads) ∈ [0,1]
    """
    try:
        # Get clean logits
        with torch.no_grad():
            clean_logits, _ = model.run_with_cache(input_tokens, return_type="logits")
            clean_logit = clean_logits[0, -1, :]  # Last token logits
        
        # Count active circuit heads
        active_heads = 0
        total_heads = len(circuit_heads)
        
        if total_heads == 0:
            return float("nan")
        
        for layer, head in circuit_heads:
            # Define ablation hook for this head
            def ablation_hook(activation, hook):
                activation[:, :, head, :] = 0.0
                return activation
            
            # Run with ablation
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    input_tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", ablation_hook)],
                    return_type="logits"
                )
            
            # Compute L2 distance
            ablated_logit = ablated_logits[0, -1, :]
            delta = torch.norm(clean_logit - ablated_logit, p=2).item()
            
            # Check if head is active
            threshold = control_thresholds.get((layer, head), 0.0)
            if delta > threshold:
                active_heads += 1
        
        # CUD is fraction of active heads
        cud = active_heads / total_heads
        return float(cud)
        
    except Exception as e:
        print(f"Error computing CUD: {e}")
        return float("nan")


def get_arithmetic_prompts() -> List[str]:
    """
    Get K=4 arithmetic prompts for circuit discovery.
    
    Returns:
        List of 4 arithmetic prompts
    """
    return [
        "What is 23 + 14? Show your work.",
        "Calculate 45 - 18. Show steps.",
        "What is 7 × 8? Show your work.",
        "Find 84 ÷ 12. Show steps."
    ]

"""Mechanistic validation via patch-out experiments."""

import json
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Tuple, Optional
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score

from ..core.utils import save_json, ensure_dir
from ..metrics.rev_composite import compute_rev_scores
from ..metrics.activation_energy import activation_energy


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from generated text using regex.
    
    Args:
        text: Generated text to extract answer from
        
    Returns:
        Numeric answer if found, None otherwise
    """
    import re
    # Look for numbers in the text
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if matches:
        try:
            # Return the last number found (usually the final answer)
            return float(matches[-1])
        except ValueError:
            return None
    return None


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy by comparing extracted numeric answers.
    
    Args:
        predictions: List of generated texts
        ground_truths: List of ground truth answers
        
    Returns:
        Accuracy as fraction of correct answers
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    correct = 0
    total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_num = extract_numeric_answer(pred)
        gt_num = extract_numeric_answer(gt)
        
        if pred_num is not None and gt_num is not None:
            total += 1
            if abs(pred_num - gt_num) < 1e-3:  # Allow small floating point differences
                correct += 1
    
    return correct / total if total > 0 else 0.0


def rank_attention_heads(
    model, 
    data: List[Dict[str, Any]], 
    circuit_heads: Optional[List[Tuple[int, int]]] = None
) -> List[Tuple[int, int, float]]:
    """
    Rank attention heads by importance for reasoning.
    
    Args:
        model: HookedTransformer model
        data: List of data samples
        circuit_heads: Optional pre-computed circuit heads
        
    Returns:
        List of (layer, head, importance_score) tuples, sorted by importance
    """
    print("Ranking attention heads by importance...")
    
    # Get all possible heads
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    if circuit_heads is not None:
        # Use pre-computed circuit heads as candidates
        candidate_heads = circuit_heads
    else:
        # Use all heads
        candidate_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    
    head_scores = []
    
    for layer, head in candidate_heads:
        try:
            # Compute importance score for this head
            score = compute_head_importance(model, data, layer, head)
            head_scores.append((layer, head, score))
        except Exception as e:
            print(f"Warning: Failed to score head {layer}.{head}: {e}")
            head_scores.append((layer, head, 0.0))
    
    # Sort by importance (descending)
    head_scores.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Ranked {len(head_scores)} attention heads")
    return head_scores


def compute_head_importance(
    model, 
    data: List[Dict[str, Any]], 
    layer: int, 
    head: int
) -> float:
    """
    Compute importance score for a specific attention head.
    
    Args:
        model: HookedTransformer model
        data: List of data samples
        layer: Layer index
        head: Head index
        
    Returns:
        Importance score (higher = more important)
    """
    # Simple heuristic: correlation with reasoning vs control
    reasoning_scores = []
    control_scores = []
    
    for item in data[:20]:  # Sample subset for efficiency
        try:
            # Get activations for this head
            tokens = model.to_tokens(item["prompt"], prepend_bos=True)
            
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, return_type="logits")
                
                # Get attention pattern for this head
                attn_key = f"blocks.{layer}.attn.hook_pattern"
                if attn_key in cache:
                    attn_pattern = cache[attn_key][0, head, :, :]  # [seq_len, seq_len]
                    
                    # Compute attention entropy as proxy for importance
                    attn_entropy = -torch.sum(attn_pattern * torch.log(attn_pattern + 1e-8), dim=-1)
                    avg_entropy = torch.mean(attn_entropy).item()
                    
                    if "reasoning" in item["id"]:
                        reasoning_scores.append(avg_entropy)
                    else:
                        control_scores.append(avg_entropy)
                        
        except Exception:
            continue
    
    if len(reasoning_scores) > 0 and len(control_scores) > 0:
        # Score based on difference between reasoning and control
        reasoning_mean = np.mean(reasoning_scores)
        control_mean = np.mean(control_scores)
        return abs(reasoning_mean - control_mean)
    else:
        return 0.0


def patch_out_heads(
    model, 
    heads_to_patch: List[Tuple[int, int]]
) -> None:
    """
    Add hooks to patch out (zero) specific attention heads.
    
    Args:
        model: HookedTransformer model
        heads_to_patch: List of (layer, head) tuples to patch out
    """
    def zero_head_hook(activations, hook):
        """Hook function to zero out specific heads."""
        for layer, head in heads_to_patch:
            if f"blocks.{layer}.attn.hook_z" in hook.name:
                activations[:, :, head, :] = 0
        return activations
    
    # Add hooks for each layer that has heads to patch
    layers_to_patch = set(layer for layer, _ in heads_to_patch)
    for layer in layers_to_patch:
        hook_name = f"blocks.{layer}.attn.hook_z"
        model.add_hook(hook_name, zero_head_hook, is_permanent=False)


def rank_layers_by_rev_contribution(
    model, 
    data: List[Dict[str, Any]]
) -> List[Tuple[int, float]]:
    """
    Rank layers by their contribution to REV scores.
    
    Args:
        model: HookedTransformer model
        data: List of data samples
        
    Returns:
        List of (layer, contribution_score) tuples, sorted by contribution
    """
    print("Ranking layers by REV contribution...")
    
    n_layers = model.cfg.n_layers
    layer_scores = []
    
    for layer in range(n_layers):
        try:
            # Compute REV contribution for this layer
            score = compute_layer_rev_contribution(model, data, layer)
            layer_scores.append((layer, score))
        except Exception as e:
            print(f"Warning: Failed to score layer {layer}: {e}")
            layer_scores.append((layer, 0.0))
    
    # Sort by contribution (descending)
    layer_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Ranked {len(layer_scores)} layers")
    return layer_scores


def compute_layer_rev_contribution(
    model, 
    data: List[Dict[str, Any]], 
    layer: int
) -> float:
    """
    Compute REV contribution score for a specific layer.
    
    Args:
        model: HookedTransformer model
        data: List of data samples
        layer: Layer index
        
    Returns:
        Contribution score (higher = more important)
    """
    # Simple heuristic: variance in residual stream activations
    reasoning_scores = []
    control_scores = []
    
    for item in data[:20]:  # Sample subset for efficiency
        try:
            tokens = model.to_tokens(item["prompt"], prepend_bos=True)
            
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, return_type="logits")
                
                # Get residual stream for this layer
                resid_key = f"blocks.{layer}.hook_resid_post"
                if resid_key in cache:
                    resid = cache[resid_key][0, :, :]  # [seq_len, d_model]
                    
                    # Compute activation variance as proxy for importance
                    activation_var = torch.var(resid, dim=0).mean().item()
                    
                    if "reasoning" in item["id"]:
                        reasoning_scores.append(activation_var)
                    else:
                        control_scores.append(activation_var)
                        
        except Exception:
            continue
    
    if len(reasoning_scores) > 0 and len(control_scores) > 0:
        # Score based on difference between reasoning and control
        reasoning_mean = np.mean(reasoning_scores)
        control_mean = np.mean(control_scores)
        return abs(reasoning_mean - control_mean)
    else:
        return 0.0


def patch_out_layers(
    model, 
    layers_to_patch: List[int]
) -> None:
    """
    Add hooks to patch out (zero) specific layers.
    
    Args:
        model: HookedTransformer model
        layers_to_patch: List of layer indices to patch out
    """
    def zero_layer_hook(activations, hook):
        """Hook function to zero out specific layers."""
        for layer in layers_to_patch:
            if f"blocks.{layer}.hook_resid_post" in hook.name:
                activations[:] = 0
        return activations
    
    # Add hooks for each layer to patch
    for layer in layers_to_patch:
        hook_name = f"blocks.{layer}.hook_resid_post"
        model.add_hook(hook_name, zero_layer_hook, is_permanent=False)


def run_patchout_experiment(
    model,
    data: List[Dict[str, Any]],
    runner,
    patchout_type: str,
    k_percentages: List[int],
    output_path: str = None
) -> Dict[str, Any]:
    """
    Run patch-out experiment for heads or layers.
    
    Args:
        model: HookedTransformer model
        data: List of data samples
        runner: ModelRunner instance
        patchout_type: "heads" or "layers"
        k_percentages: List of percentages to patch out
        output_path: Path to save results
        
    Returns:
        Results dictionary
    """
    print(f"Running {patchout_type} patch-out experiment...")
    
    # Get baseline metrics
    baseline_results = compute_baseline_metrics(model, data, runner)
    
    results = {
        "baseline": baseline_results,
        "patchout_results": {}
    }
    
    if patchout_type == "heads":
        # Rank heads and run patch-out
        head_rankings = rank_attention_heads(model, data)
        
        for k_percent in k_percentages:
            print(f"Patching out top {k_percent}% of heads...")
            
            # Select top K% heads
            n_heads_to_patch = max(1, int(len(head_rankings) * k_percent / 100))
            heads_to_patch = [head for head, _, _ in head_rankings[:n_heads_to_patch]]
            
            # Patch out heads
            patch_out_heads(model, heads_to_patch)
            
            # Compute metrics with patched model
            patched_results = compute_patched_metrics(model, data, runner)
            
            # Compute deltas
            delta_accuracy = patched_results["accuracy"] - baseline_results["accuracy"]
            delta_rev = patched_results["mean_rev"] - baseline_results["mean_rev"]
            
            results["patchout_results"][f"k_{k_percent}"] = {
                "heads_patched": heads_to_patch,
                "n_heads_patched": len(heads_to_patch),
                "delta_accuracy": float(delta_accuracy),
                "delta_rev": float(delta_rev),
                "patched_accuracy": float(patched_results["accuracy"]),
                "patched_mean_rev": float(patched_results["mean_rev"])
            }
            
            # Remove hooks for next iteration
            model.remove_all_hooks()
    
    elif patchout_type == "layers":
        # Rank layers and run patch-out
        layer_rankings = rank_layers_by_rev_contribution(model, data)
        
        for k_percent in k_percentages:
            print(f"Patching out top {k_percent}% of layers...")
            
            # Select top K% layers
            n_layers_to_patch = max(1, int(len(layer_rankings) * k_percent / 100))
            layers_to_patch = [layer for layer, _ in layer_rankings[:n_layers_to_patch]]
            
            # Patch out layers
            patch_out_layers(model, layers_to_patch)
            
            # Compute metrics with patched model
            patched_results = compute_patched_metrics(model, data, runner)
            
            # Compute deltas
            delta_accuracy = patched_results["accuracy"] - baseline_results["accuracy"]
            delta_rev = patched_results["mean_rev"] - baseline_results["mean_rev"]
            
            results["patchout_results"][f"k_{k_percent}"] = {
                "layers_patched": layers_to_patch,
                "n_layers_patched": len(layers_to_patch),
                "delta_accuracy": float(delta_accuracy),
                "delta_rev": float(delta_rev),
                "patched_accuracy": float(patched_results["accuracy"]),
                "patched_mean_rev": float(patched_results["mean_rev"])
            }
            
            # Remove hooks for next iteration
            model.remove_all_hooks()
    
    # Save results if output path provided
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        save_json(output_path, results)
        print(f"Saved {patchout_type} patch-out results to {output_path}")
    
    return results


def _zero_selected_heads(head_idxs):
    """Hook function to zero out selected attention heads."""
    def hook(z, hook=None):
        # z: [B, S, H, D]
        # hook parameter is optional for compatibility with TransformerLens API changes
        z[..., head_idxs, :] = 0
        return z
    return hook

def _zero_mlp(_o, _hook):
    """Hook function to zero out MLP outputs."""
    return torch.zeros_like(_o)

@contextmanager
def apply_head_hooks(model, layer_head_list):
    """Context manager to apply head ablation hooks."""
    # Group by layer -> list of heads
    by_layer = {}
    for L, H in layer_head_list: 
        by_layer.setdefault(L, []).append(H)
    handles = []
    try:
        for L, heads in by_layer.items():
            name = f"blocks.{L}.attn.hook_z"  # TL name
            handles.append(model.add_hook(name, _zero_selected_heads(heads), is_permanent=False))
        yield
    finally:
        model.reset_hooks(including_permanent=False)

@contextmanager
def apply_layer_mlp_hooks(model, layers):
    """Context manager to apply layer MLP ablation hooks."""
    handles = []
    try:
        for L in layers:
            name = f"blocks.{L}.hook_mlp_out"
            handles.append(model.add_hook(name, _zero_mlp, is_permanent=False))
        yield
    finally:
        model.reset_hooks(including_permanent=False)

def safe_patchout_heads(model, ranked_heads, K_percent_list, eval_fn):
    """Safely run head patch-out experiments with error handling."""
    results = {}
    try:
        if not ranked_heads:
            raise RuntimeError("No ranked heads provided")
        n = max(1, len(ranked_heads))
        for K in K_percent_list:
            k = max(1, int(n*K/100))
            sel = ranked_heads[:k]
            with apply_head_hooks(model, sel):
                results[str(K)] = eval_fn()
    except Exception as e:
        print(f"[patchout] Heads failed: {type(e).__name__}: {e}")
        results["__failed__"] = True
    return results

def safe_patchout_layers(model, ranked_layers, K_percent_list, eval_fn):
    """Safely run layer patch-out experiments with error handling."""
    results = {}
    if not ranked_layers:
        print("[patchout] No ranked layers; skipping")
        return results
    try:
        for K in K_percent_list:
            k = max(1, int(len(ranked_layers)*K/100))
            sel = ranked_layers[:k]
            with apply_layer_mlp_hooks(model, sel):
                results[str(K)] = eval_fn()
    except Exception as e:
        print(f"[patchout] Layers failed: {type(e).__name__}: {e}")
        results["__failed__"] = True
    return results

def auto_discover_pseudo_circuit(model, n_prompts=3):
    """Auto-discover a pseudo-circuit using simple arithmetic prompts."""
    try:
        # Simple ranking by attention entropy (placeholder)
        # In a real implementation, you'd compute actual circuit importance
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        
        # Create a simple ranking (top 20% of heads)
        ranked_heads = []
        for layer in range(n_layers):
            for head in range(n_heads):
                # Simple heuristic: prefer middle layers and heads
                importance = 1.0 - abs(layer - n_layers/2) / (n_layers/2) - abs(head - n_heads/2) / (n_heads/2)
                ranked_heads.append((layer, head, importance))
        
        # Sort by importance and return top heads
        ranked_heads.sort(key=lambda x: x[2], reverse=True)
        return [(layer, head) for layer, head, _ in ranked_heads[:max(1, n_layers * n_heads // 5)]]
        
    except Exception as e:
        print(f"[patchout] Auto-discovery failed: {e}")
        return []


def compute_baseline_metrics(
    model, 
    data: List[Dict[str, Any]], 
    runner
) -> Dict[str, Any]:
    """Compute baseline metrics without any patching."""
    print("Computing baseline metrics...")
    
    predictions = []
    ground_truths = []
    rev_scores = []
    
    for item in data:
        try:
            # Generate text
            result = runner.generate(item["prompt"])
            generated_text = result["generated_text"]
            
            predictions.append(generated_text)
            ground_truths.append(item["answer"])
            
            # Compute REV score (simplified)
            # In practice, you'd compute all metrics like in the main evaluation
            # For now, use a simple heuristic
            reasoning_len = min(32, runner.max_new_tokens - 1)
            ae = activation_energy(result["hidden_states"], reasoning_len)
            rev_scores.append(ae)  # Simplified REV
            
        except Exception as e:
            print(f"Error computing baseline for {item['id']}: {e}")
            predictions.append("")
            ground_truths.append(item["answer"])
            rev_scores.append(0.0)
    
    # Compute accuracy
    accuracy = compute_accuracy(predictions, ground_truths)
    mean_rev = np.mean(rev_scores) if rev_scores else 0.0
    
    return {
        "accuracy": float(accuracy),
        "mean_rev": float(mean_rev),
        "n_samples": len(data)
    }


def compute_patched_metrics(
    model, 
    data: List[Dict[str, Any]], 
    runner
) -> Dict[str, Any]:
    """Compute metrics with patched model."""
    print("Computing patched metrics...")
    
    predictions = []
    ground_truths = []
    rev_scores = []
    
    for item in data:
        try:
            # Generate text with patched model
            result = runner.generate(item["prompt"])
            generated_text = result["generated_text"]
            
            predictions.append(generated_text)
            ground_truths.append(item["answer"])
            
            # Compute REV score (simplified)
            reasoning_len = min(32, runner.max_new_tokens - 1)
            ae = activation_energy(result["hidden_states"], reasoning_len)
            rev_scores.append(ae)  # Simplified REV
            
        except Exception as e:
            print(f"Error computing patched metrics for {item['id']}: {e}")
            predictions.append("")
            ground_truths.append(item["answer"])
            rev_scores.append(0.0)
    
    # Compute accuracy
    accuracy = compute_accuracy(predictions, ground_truths)
    mean_rev = np.mean(rev_scores) if rev_scores else 0.0
    
    return {
        "accuracy": float(accuracy),
        "mean_rev": float(mean_rev),
        "n_samples": len(data)
    }


def compute_delta_correlation(patchout_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute correlation between ΔREV and ΔAccuracy from patch-out results.
    
    Args:
        patchout_results: Results from patch-out experiments
        
    Returns:
        Dict with correlation statistics (rho, p-value)
    """
    from scipy.stats import spearmanr
    
    delta_rev_values = []
    delta_acc_values = []
    
    # Extract delta values from all patch-out experiments
    if "patchout_results" in patchout_results:
        for k_key, k_result in patchout_results["patchout_results"].items():
            if "delta_rev" in k_result and "delta_accuracy" in k_result:
                delta_rev_values.append(k_result["delta_rev"])
                delta_acc_values.append(k_result["delta_accuracy"])
    
    if len(delta_rev_values) >= 2:
        try:
            rho, p_value = spearmanr(delta_rev_values, delta_acc_values)
            return {
                "rho": float(rho),
                "p_value": float(p_value),
                "n_points": len(delta_rev_values),
                "delta_rev": delta_rev_values,
                "delta_acc": delta_acc_values
            }
        except Exception as e:
            print(f"Warning: Correlation computation failed: {e}")
    
    return {
        "rho": float('nan'),
        "p_value": float('nan'),
        "n_points": 0,
        "delta_rev": [],
        "delta_acc": []
    }


def run_mechanistic_validation(
    model,
    data: List[Dict[str, Any]],
    runner,
    output_dir: str,
    k_percentages: List[int] = [5, 10, 20]
) -> Dict[str, Any]:
    """
    Run complete mechanistic validation with head and layer patch-out.
    
    Args:
        model: HookedTransformer model
        data: List of data samples
        runner: ModelRunner instance
        output_dir: Output directory for results
        k_percentages: List of percentages to patch out
        
    Returns:
        Combined results dictionary
    """
    print("Starting mechanistic validation...")
    
    # Run head patch-out experiment
    heads_output_path = os.path.join(output_dir, "patchout_heads.json")
    heads_results = run_patchout_experiment(
        model, data, runner, "heads", k_percentages, heads_output_path
    )
    
    # Run layer patch-out experiment
    layers_output_path = os.path.join(output_dir, "patchout_layers.json")
    layers_results = run_patchout_experiment(
        model, data, runner, "layers", k_percentages, layers_output_path
    )
    
    # Compute correlations
    heads_correlation = compute_delta_correlation(heads_results)
    layers_correlation = compute_delta_correlation(layers_results)
    
    # Combine results
    combined_results = {
        "heads_patchout": heads_results,
        "layers_patchout": layers_results,
        "correlations": {
            "heads": heads_correlation,
            "layers": layers_correlation
        },
        "experiment_summary": {
            "n_samples": len(data),
            "k_percentages": k_percentages,
            "baseline_accuracy": heads_results["baseline"]["accuracy"],
            "baseline_mean_rev": heads_results["baseline"]["mean_rev"]
        }
    }
    
    # Save combined results
    combined_output_path = os.path.join(output_dir, "mechanistic_validation.json")
    save_json(combined_output_path, combined_results)
    
    print("Mechanistic validation complete!")
    print(f"Heads ΔREV vs ΔAcc correlation: ρ={heads_correlation.get('rho', float('nan')):.4f}")
    print(f"Layers ΔREV vs ΔAcc correlation: ρ={layers_correlation.get('rho', float('nan')):.4f}")
    
    return combined_results

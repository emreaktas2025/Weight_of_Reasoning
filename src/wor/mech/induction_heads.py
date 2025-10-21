"""Mechanistic case study: Induction heads in Pythia-70M."""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from ..core.runner import ModelRunner
from ..core.utils import save_json, ensure_dir
from ..metrics.rev_composite import compute_rev_scores
from ..metrics.activation_energy import activation_energy
from ..metrics.attention_entropy import attention_process_entropy
from .patchout import apply_head_hooks, rank_attention_heads
import pandas as pd


def generate_induction_dataset(
    n_samples: int = 50,
    seq_len: int = 10,
    vocab_size: int = 26,
    seed: int = 1337
) -> List[Dict[str, Any]]:
    """
    Generate synthetic induction head dataset: [A][B][C]...[A][B] → predict [C]
    
    Args:
        n_samples: Number of samples to generate
        seq_len: Length of the sequence
        vocab_size: Size of vocabulary (default 26 for A-Z)
        seed: Random seed
        
    Returns:
        List of induction dataset samples
    """
    np.random.seed(seed)
    
    # Create alphabet
    alphabet = [chr(ord('A') + i) for i in range(min(vocab_size, 26))]
    
    data = []
    for i in range(n_samples):
        # Generate random sequence
        half_len = seq_len // 2
        sequence_a = [alphabet[np.random.randint(0, len(alphabet))] for _ in range(half_len)]
        
        # Repeat sequence (induction pattern)
        # First half: A B C D E
        # Second half: A B _ (predict C)
        prompt_tokens = sequence_a + sequence_a[:-1]  # Repeat but leave last token for prediction
        target_token = sequence_a[-1]  # The token to predict
        
        # Format as text
        prompt = " ".join(prompt_tokens)
        full_seq = " ".join(sequence_a + sequence_a)
        
        data.append({
            "id": f"induction_{i}",
            "prompt": prompt,
            "answer": target_token,
            "full_sequence": full_seq,
            "label": "reasoning",  # Induction is a form of reasoning
            "pattern_type": "induction"
        })
    
    print(f"Generated {len(data)} induction samples")
    return data


def compute_head_rev_scores(
    model,
    runner: ModelRunner,
    data: List[Dict[str, Any]]
) -> Dict[Tuple[int, int], float]:
    """
    Compute REV contribution for each attention head.
    
    Args:
        model: HookedTransformer model
        runner: ModelRunner instance
        data: Induction dataset samples
        
    Returns:
        Dict mapping (layer, head) to REV score
    """
    print("Computing REV scores for attention heads...")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    head_revs = {}
    
    for layer in range(n_layers):
        for head in range(n_heads):
            try:
                # Compute REV for this specific head by running on sample
                rev_scores = []
                
                for item in data[:10]:  # Use subset for efficiency
                    try:
                        result = runner.generate(item["prompt"])
                        
                        # Get attention pattern for this head
                        if "cache" in result:
                            attn_key = f"blocks.{layer}.attn.hook_pattern"
                            if attn_key in result["cache"]:
                                attn_pattern = result["cache"][attn_key][0, head, :, :]
                                
                                # Compute entropy as proxy for REV contribution
                                attn_entropy = -torch.sum(
                                    attn_pattern * torch.log(attn_pattern + 1e-8), dim=-1
                                )
                                rev_scores.append(torch.mean(attn_entropy).item())
                    except:
                        continue
                
                # Average REV score for this head
                head_revs[(layer, head)] = np.mean(rev_scores) if rev_scores else 0.0
                
            except Exception as e:
                print(f"Warning: Failed to compute REV for head {layer}.{head}: {e}")
                head_revs[(layer, head)] = 0.0
    
    print(f"Computed REV scores for {len(head_revs)} heads")
    return head_revs


def rank_heads_by_rev(
    head_revs: Dict[Tuple[int, int], float]
) -> List[Tuple[int, int]]:
    """
    Rank attention heads by REV score.
    
    Args:
        head_revs: Dict mapping (layer, head) to REV score
        
    Returns:
        List of (layer, head) tuples sorted by REV (descending)
    """
    # Sort heads by REV score
    sorted_heads = sorted(head_revs.items(), key=lambda x: x[1], reverse=True)
    
    # Extract (layer, head) tuples
    ranked_heads = [head for head, score in sorted_heads]
    
    return ranked_heads


def evaluate_induction_accuracy(
    runner: ModelRunner,
    data: List[Dict[str, Any]]
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluate accuracy on induction task.
    
    Args:
        runner: ModelRunner instance
        data: Induction dataset samples
        
    Returns:
        Tuple of (accuracy, predictions, ground_truths)
    """
    predictions = []
    ground_truths = []
    
    for item in data:
        try:
            result = runner.generate(item["prompt"], max_new_tokens=1)
            generated_text = result["generated_text"].strip()
            
            # Extract first token as prediction
            pred_token = generated_text.split()[0] if generated_text else ""
            predictions.append(pred_token)
            ground_truths.append(item["answer"])
            
        except Exception as e:
            print(f"Error evaluating {item['id']}: {e}")
            predictions.append("")
            ground_truths.append(item["answer"])
    
    # Compute accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / len(predictions) if predictions else 0.0
    
    return accuracy, predictions, ground_truths


def run_targeted_patchout(
    model,
    runner: ModelRunner,
    data: List[Dict[str, Any]],
    ranked_heads: List[Tuple[int, int]],
    k_percent: int = 10
) -> Dict[str, Any]:
    """
    Run targeted patch-out: ablate top-K% heads by REV.
    
    Args:
        model: HookedTransformer model
        runner: ModelRunner instance
        data: Induction dataset samples
        ranked_heads: List of (layer, head) tuples ranked by importance
        k_percent: Percentage of top heads to patch out
        
    Returns:
        Dict with accuracy and REV after patch-out
    """
    print(f"Running targeted patch-out (top {k_percent}% heads)...")
    
    # Select top K% heads
    n_heads_to_patch = max(1, int(len(ranked_heads) * k_percent / 100))
    heads_to_patch = ranked_heads[:n_heads_to_patch]
    
    print(f"Patching out {n_heads_to_patch} heads: {heads_to_patch[:5]}...")
    
    # Apply patch-out hooks
    with apply_head_hooks(model, heads_to_patch):
        accuracy, predictions, ground_truths = evaluate_induction_accuracy(runner, data)
    
    return {
        "accuracy": float(accuracy),
        "n_heads_patched": n_heads_to_patch,
        "heads_patched": heads_to_patch,
        "k_percent": k_percent
    }


def run_random_patchout(
    model,
    runner: ModelRunner,
    data: List[Dict[str, Any]],
    all_heads: List[Tuple[int, int]],
    k_percent: int = 10,
    seed: int = 1337
) -> Dict[str, Any]:
    """
    Run random patch-out: ablate random K% heads.
    
    Args:
        model: HookedTransformer model
        runner: ModelRunner instance
        data: Induction dataset samples
        all_heads: List of all (layer, head) tuples
        k_percent: Percentage of heads to patch out
        seed: Random seed
        
    Returns:
        Dict with accuracy and REV after patch-out
    """
    print(f"Running random patch-out ({k_percent}% heads)...")
    
    np.random.seed(seed)
    
    # Select random K% heads
    n_heads_to_patch = max(1, int(len(all_heads) * k_percent / 100))
    heads_to_patch = [all_heads[i] for i in np.random.choice(len(all_heads), n_heads_to_patch, replace=False)]
    
    print(f"Patching out {n_heads_to_patch} random heads...")
    
    # Apply patch-out hooks
    with apply_head_hooks(model, heads_to_patch):
        accuracy, predictions, ground_truths = evaluate_induction_accuracy(runner, data)
    
    return {
        "accuracy": float(accuracy),
        "n_heads_patched": n_heads_to_patch,
        "heads_patched": heads_to_patch,
        "k_percent": k_percent
    }


def run_induction_case_study(
    model_cfg: Dict[str, Any],
    output_dir: str = "reports",
    n_samples: int = 50,
    k_percentages: List[int] = [5, 10, 20],
    seed: int = 1337
) -> Dict[str, Any]:
    """
    Run complete induction heads case study.
    
    Args:
        model_cfg: Model configuration (should be Pythia-70M)
        output_dir: Output directory for results
        n_samples: Number of induction samples
        k_percentages: List of K percentages for patch-out
        seed: Random seed
        
    Returns:
        Complete results dictionary
    """
    print("\n" + "="*80)
    print("Induction Heads Case Study")
    print("="*80)
    
    # Generate induction dataset
    induction_data = generate_induction_dataset(n_samples, seed=seed)
    
    # Initialize model
    print("\nLoading model...")
    runner = ModelRunner(model_cfg)
    model = runner.model
    
    # Compute baseline accuracy
    print("\nComputing baseline accuracy...")
    baseline_accuracy, _, _ = evaluate_induction_accuracy(runner, induction_data)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Compute REV scores for each head
    head_revs = compute_head_rev_scores(model, runner, induction_data)
    
    # Rank heads by REV
    ranked_heads = rank_heads_by_rev(head_revs)
    
    # Get all heads for random baseline
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    
    # Run patch-out experiments
    results = {
        "baseline_accuracy": float(baseline_accuracy),
        "n_samples": n_samples,
        "n_heads_total": len(all_heads),
        "targeted_patchout": {},
        "random_patchout": {},
        "top_heads_by_rev": [
            {"layer": layer, "head": head, "rev_score": float(head_revs[(layer, head)])}
            for layer, head in ranked_heads[:20]
        ]
    }
    
    for k_percent in k_percentages:
        # Targeted patch-out (top REV heads)
        targeted_result = run_targeted_patchout(
            model, runner, induction_data, ranked_heads, k_percent
        )
        results["targeted_patchout"][f"k_{k_percent}"] = targeted_result
        
        # Random patch-out (control)
        random_result = run_random_patchout(
            model, runner, induction_data, all_heads, k_percent, seed
        )
        results["random_patchout"][f"k_{k_percent}"] = random_result
        
        # Compute deltas
        targeted_delta = targeted_result["accuracy"] - baseline_accuracy
        random_delta = random_result["accuracy"] - baseline_accuracy
        
        print(f"\nK={k_percent}%:")
        print(f"  Targeted Δ Acc: {targeted_delta:.4f} ({targeted_result['accuracy']:.4f})")
        print(f"  Random Δ Acc: {random_delta:.4f} ({random_result['accuracy']:.4f})")
        print(f"  Difference: {targeted_delta - random_delta:.4f}")
    
    # Save results
    ensure_dir(output_dir)
    output_path = f"{output_dir}/induction_case_study.json"
    save_json(output_path, results)
    print(f"\n✅ Results saved to {output_path}")
    
    return results


def create_induction_summary(results: Dict[str, Any]) -> str:
    """
    Create human-readable summary of induction case study.
    
    Args:
        results: Results from run_induction_case_study
        
    Returns:
        Summary string
    """
    summary = []
    summary.append("\n" + "="*80)
    summary.append("Induction Heads Case Study Summary")
    summary.append("="*80)
    
    baseline_acc = results["baseline_accuracy"]
    summary.append(f"\nBaseline Accuracy: {baseline_acc:.4f}")
    summary.append(f"Total Heads: {results['n_heads_total']}")
    summary.append(f"Samples: {results['n_samples']}")
    
    summary.append("\n--- Top 5 Heads by REV ---")
    for i, head_info in enumerate(results["top_heads_by_rev"][:5], 1):
        summary.append(
            f"{i}. Layer {head_info['layer']}, Head {head_info['head']}: "
            f"REV = {head_info['rev_score']:.4f}"
        )
    
    summary.append("\n--- Patch-out Results ---")
    for k_key in sorted(results["targeted_patchout"].keys()):
        targeted = results["targeted_patchout"][k_key]
        random = results["random_patchout"][k_key]
        
        targeted_delta = targeted["accuracy"] - baseline_acc
        random_delta = random["accuracy"] - baseline_acc
        
        summary.append(f"\n{k_key}:")
        summary.append(f"  Targeted: Δ Acc = {targeted_delta:.4f}")
        summary.append(f"  Random:   Δ Acc = {random_delta:.4f}")
        summary.append(f"  Effect:   {abs(targeted_delta) - abs(random_delta):.4f} stronger")
    
    summary.append("\n" + "="*80)
    
    return "\n".join(summary)


"""Phase 4 evaluation pipeline with scaling study and mechanistic validation."""

import argparse
import csv
import json
import os
import re
import time
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_auc_score

from ..core.runner import ModelRunner
from ..core.utils import set_seed, read_jsonl, ensure_dir, save_json
from ..utils.hw import detect_hw, log_hw_config, get_optimal_dataset_sizes, get_model_config_with_hw
from ..metrics.activation_energy import activation_energy
from ..metrics.attention_entropy import attention_process_entropy
from ..metrics.activation_path_length import compute_apl, compute_control_thresholds, load_control_thresholds
from ..metrics.circuit_utilization_density import (
    compute_circuit_heads, save_circuit_heads, load_circuit_heads, 
    load_control_thresholds as load_cud_thresholds, compute_cud, get_arithmetic_prompts
)
from ..metrics.stability_intermediate_beliefs import compute_sib_simple
from ..metrics.feature_load import compute_feature_load
from ..metrics.rev_composite import compute_rev_scores, compute_rev_statistics
from ..stats.partial_corr import compute_partial_correlations, save_partial_correlations


def load_reasoning_data(name: str, n: int, seed: int = 1337) -> List[Dict[str, Any]]:
    """
    Load reasoning dataset from HuggingFace with fallback to local data.
    
    Args:
        name: Dataset name ('gsm8k' or 'strategyqa')
        n: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of samples with 'question' and 'answer' fields
    """
    try:
        from datasets import load_dataset
        
        if name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="train")
        elif name == "strategyqa":
            dataset = load_dataset("strategyqa", split="train")
        else:
            raise ValueError(f"Unknown reasoning dataset: {name}")
        
        # Deterministic sampling
        dataset = dataset.shuffle(seed=seed)
        samples = dataset.select(range(min(n, len(dataset))))
        
        # Convert to standard format
        data = []
        for i, sample in enumerate(samples):
            if name == "gsm8k":
                data.append({
                    "id": f"{name}_{i}",
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": f"Solve: {sample['question']} Show steps."
                })
            elif name == "strategyqa":
                data.append({
                    "id": f"{name}_{i}",
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": f"Answer: {sample['question']} Show reasoning."
                })
        
        print(f"Loaded {len(data)} samples from {name}")
        return data
        
    except Exception as e:
        print(f"Failed to load {name} from HuggingFace: {e}")
        print("Falling back to local data...")
        return load_local_fallback_data(name, n)


def load_control_data(n: int, seed: int = 1337) -> List[Dict[str, Any]]:
    """
    Load control dataset from Wikipedia with fallback to local data.
    
    Args:
        n: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of control samples
    """
    try:
        from datasets import load_dataset
        
        # Load Wikipedia dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        dataset = dataset.shuffle(seed=seed)
        samples = dataset.select(range(min(n, len(dataset))))
        
        # Convert to standard format
        data = []
        for i, sample in enumerate(samples):
            # Take first sentence or paragraph
            text = sample["text"]
            sentences = text.split('. ')
            if sentences:
                prompt = sentences[0] + "."
            else:
                prompt = text[:200] + "..."
            
            data.append({
                "id": f"wiki_{i}",
                "question": prompt,
                "answer": "",
                "prompt": prompt
            })
        
        print(f"Loaded {len(data)} control samples from Wikipedia")
        return data
        
    except Exception as e:
        print(f"Failed to load Wikipedia: {e}")
        print("Falling back to local control data...")
        return load_local_fallback_data("control", n)


def load_local_fallback_data(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    """
    Load fallback data from local data/mini/ directory.
    
    Args:
        dataset_name: Name of dataset to load
        n: Number of samples to load
        
    Returns:
        List of samples
    """
    if dataset_name == "control":
        file_path = "data/mini/control.jsonl"
    else:
        file_path = "data/mini/reasoning.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, creating minimal fallback")
        return create_minimal_fallback_data(dataset_name, n)
    
    data = []
    for i, item in enumerate(read_jsonl(file_path)):
        if i >= n:
            break
        data.append({
            "id": f"{dataset_name}_fallback_{i}",
            "question": item["prompt"],
            "answer": "",
            "prompt": item["prompt"]
        })
    
    print(f"Loaded {len(data)} fallback samples from {file_path}")
    return data


def create_minimal_fallback_data(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    """Create minimal fallback data if no files exist."""
    data = []
    for i in range(n):
        if dataset_name == "control":
            prompt = f"This is a neutral control sentence number {i+1}."
        else:
            prompt = f"Solve this reasoning problem {i+1}: What is 2+2? Show steps."
        
        data.append({
            "id": f"{dataset_name}_minimal_{i}",
            "question": prompt,
            "answer": "4" if dataset_name != "control" else "",
            "prompt": prompt
        })
    
    print(f"Created {len(data)} minimal fallback samples for {dataset_name}")
    return data


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from generated text using regex.
    
    Args:
        text: Generated text to extract answer from
        
    Returns:
        Numeric answer if found, None otherwise
    """
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


def save_dataset_manifest(data: List[Dict[str, Any]], dataset_name: str, output_dir: str) -> None:
    """
    Save dataset manifest to CSV for reproducibility.
    
    Args:
        data: Dataset samples
        dataset_name: Name of dataset
        output_dir: Output directory
    """
    ensure_dir(output_dir)
    manifest_path = os.path.join(output_dir, f"{dataset_name}_manifest.csv")
    
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'question', 'answer', 'prompt'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {dataset_name} manifest to {manifest_path}")


def run_phase4_evaluation(cfg_path: str) -> None:
    """Run the Phase 4 evaluation pipeline."""
    start_time = time.time()
    
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Detect hardware and log configuration
    hw_config = detect_hw()
    log_hw_config(hw_config)
    
    # Set seed for reproducibility
    set_seed(cfg.get("seed", 1337))
    
    # Create output directories
    ensure_dir(cfg["output_dir"])
    ensure_dir(cfg["plots_dir"])
    ensure_dir(cfg["splits_dir"])
    
    # Get optimal dataset sizes based on hardware
    optimal_sizes = get_optimal_dataset_sizes(hw_config, cfg["datasets"])
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    all_data = []
    
    # Load reasoning datasets
    for dataset_name, n in optimal_sizes["reasoning"].items():
        data = load_reasoning_data(dataset_name, n, cfg.get("seed", 1337))
        all_data.extend(data)
        save_dataset_manifest(data, dataset_name, cfg["splits_dir"])
    
    # Load control dataset
    control_n = optimal_sizes["control"]["wiki"]
    control_data = load_control_data(control_n, cfg.get("seed", 1337))
    all_data.extend(control_data)
    save_dataset_manifest(control_data, "wiki", cfg["splits_dir"])
    
    print(f"Total samples loaded: {len(all_data)}")
    
    # Initialize results storage
    model_results = {}
    aggregate_data = {
        "n_params": [],
        "d_REV": [],
        "auroc_REV": [],
        "partial_r_REV": [],
        "runtime_sec": [],
        "model_names": []
    }
    
    # Process each model
    print("\n=== Processing Models ===")
    for model_cfg_path in cfg["models"]:
        try:
            print(f"\nProcessing model: {model_cfg_path}")
            model_start_time = time.time()
            
            # Load model configuration
            with open(model_cfg_path, 'r') as f:
                model_cfg = yaml.safe_load(f)
            
            # Update with hardware-optimized settings
            model_cfg = get_model_config_with_hw(model_cfg, hw_config)
            
            # Extract model name for file naming
            model_name = os.path.splitext(os.path.basename(model_cfg_path))[0]
            
            # Run evaluation for this model
            model_result = evaluate_single_model(
                model_cfg, all_data, cfg, model_name, hw_config
            )
            
            if model_result is not None:
                model_results[model_name] = model_result
                
                # Add to aggregate data
                aggregate_data["model_names"].append(model_name)
                aggregate_data["n_params"].append(model_result["n_params"])
                aggregate_data["d_REV"].append(model_result["cohens_d"]["REV"])
                aggregate_data["auroc_REV"].append(model_result["auroc_REV"])
                
                # Safely get partial correlation for REV
                partial_r_rev = float("nan")
                if "partial_corr" in model_result and "REV" in model_result["partial_corr"]:
                    partial_r_rev = model_result["partial_corr"]["REV"].get("r", float("nan"))
                aggregate_data["partial_r_REV"].append(partial_r_rev)
                aggregate_data["runtime_sec"].append(time.time() - model_start_time)
                
                print(f"✅ {model_name} completed successfully")
            else:
                print(f"❌ {model_name} failed")
                
        except Exception as e:
            print(f"❌ Failed to process {model_cfg_path}: {e}")
            continue
    
    # Save aggregate results
    print("\n=== Saving Aggregate Results ===")
    aggregate_path = os.path.join(cfg["output_dir"], "aggregate_scaling.json")
    save_json(aggregate_path, aggregate_data)
    
    # Compute trend analysis
    if len(aggregate_data["n_params"]) > 1:
        trend_analysis = compute_trend_analysis(aggregate_data)
        aggregate_data["trend_analysis"] = trend_analysis
        save_json(aggregate_path, aggregate_data)
    
    total_runtime = time.time() - start_time
    print(f"\n=== Phase 4 Evaluation Complete ===")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Models processed: {len(model_results)}")
    print(f"Results saved to: {cfg['output_dir']}")


def evaluate_single_model(
    model_cfg: Dict[str, Any], 
    all_data: List[Dict[str, Any]], 
    cfg: Dict[str, Any],
    model_name: str,
    hw_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single model and return results.
    
    Args:
        model_cfg: Model configuration
        all_data: All dataset samples
        cfg: Main configuration
        model_name: Name of model for file naming
        hw_config: Hardware configuration
        
    Returns:
        Model results dict or None if failed
    """
    try:
        # Initialize model runner
        print(f"Loading model: {model_cfg['model_name']}")
        runner = ModelRunner(model_cfg)
        
        # Get model parameter count (rough estimate)
        n_params = estimate_model_params(runner.model)
        
        # Load or compute circuit heads and thresholds
        circuit_heads, cud_thresholds = load_or_compute_circuits(runner, cfg)
        apl_thresholds = load_or_compute_apl_thresholds(runner, cfg)
        
        # Process all samples
        print("Processing samples...")
        rows = []
        predictions = []
        ground_truths = []
        
        for item in all_data:
            try:
                # Generate text and get activations
                result = runner.generate(item["prompt"])
                text = result["text"]
                generated_text = result["generated_text"]
                
                # Store prediction and ground truth for accuracy
                predictions.append(generated_text)
                ground_truths.append(item["answer"])
                
                # Determine label
                label = "reasoning" if "reasoning" in item["id"] else "control"
                
                # Compute reasoning window length
                reasoning_len = min(32, model_cfg.get("max_new_tokens", 64) - 1)
                
                # Compute all metrics
                ae = activation_energy(result["hidden_states"], reasoning_len)
                ape = attention_process_entropy(result["attention_probs"], reasoning_len)
                apl = compute_apl(runner.model, result["cache"], apl_thresholds, result["input_tokens"])
                cud = compute_cud(runner.model, result["cache"], circuit_heads, cud_thresholds, result["input_tokens"])
                sib = compute_sib_simple(runner.model, result["cache"], result["input_tokens"], item["prompt"], reasoning_len)
                fl = compute_feature_load(runner.model, result["cache"], result["input_tokens"], reasoning_len)
                
                # Count tokens and get perplexity
                token_count = len(text.split())
                ppl = result["perplexity"]
                
                # Store results
                rows.append({
                    "id": item["id"],
                    "label": label,
                    "token_len": token_count,
                    "ppl": ppl,
                    "AE": ae,
                    "APE": ape,
                    "APL": apl,
                    "CUD": cud,
                    "SIB": sib,
                    "FL": fl,
                    "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                })
                
            except Exception as e:
                print(f"Error processing {item['id']}: {e}")
                # Add row with NaN values
                rows.append({
                    "id": item["id"],
                    "label": "reasoning" if "reasoning" in item["id"] else "control",
                    "token_len": 0,
                    "ppl": float("nan"),
                    "AE": float("nan"),
                    "APE": float("nan"),
                    "APL": float("nan"),
                    "CUD": float("nan"),
                    "SIB": float("nan"),
                    "FL": float("nan"),
                    "generated_text": "ERROR"
                })
        
        # Create DataFrame and compute REV scores
        df = pd.DataFrame(rows)
        rev_scores = compute_rev_scores(df)
        df['REV'] = rev_scores
        
        # Add label_num column for partial correlations
        df['label_num'] = df['label'].map({'reasoning': 1, 'control': 0})
        
        # Compute accuracy
        accuracy = compute_accuracy(predictions, ground_truths)
        
        # Save detailed results
        save_model_results(df, model_name, cfg["output_dir"])
        
        # Compute summary statistics
        summary = compute_model_summary(df, accuracy, n_params)
        
        # Compute and save partial correlations
        partial_corr_results = compute_partial_correlations(df)
        partial_corr_path = os.path.join(cfg["output_dir"], f"{model_name}_partial_corr.json")
        save_partial_correlations(partial_corr_results, partial_corr_path)
        
        # Add partial correlations to summary
        summary["partial_corr"] = partial_corr_results
        
        # Save summary
        summary_path = os.path.join(cfg["output_dir"], f"{model_name}_summary.json")
        save_json(summary_path, summary)
        
        return summary
        
    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
        return None


def estimate_model_params(model) -> int:
    """Estimate model parameter count."""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
    except:
        return 0


def load_or_compute_circuits(runner: ModelRunner, cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load or compute circuit heads and thresholds."""
    circuit_heads = load_circuit_heads(cfg.get("circuit_heads_json", "reports/circuits/heads.json"))
    cud_thresholds = load_cud_thresholds(cfg.get("control_thresholds_npz", "reports/control_thresholds_phase3.npz"))
    
    if circuit_heads is None or cud_thresholds is None:
        print("Computing circuit heads and thresholds...")
        arithmetic_prompts = get_arithmetic_prompts()
        control_prompts = ["This is a neutral control sentence."] * 10  # Minimal control set
        circuit_heads, cud_thresholds = compute_circuit_heads(
            runner.model, arithmetic_prompts, control_prompts, max_heads=24
        )
    
    return circuit_heads, cud_thresholds


def load_or_compute_apl_thresholds(runner: ModelRunner, cfg: Dict[str, Any]) -> Any:
    """Load or compute APL thresholds."""
    apl_thresholds = load_control_thresholds()
    if apl_thresholds is None:
        print("Computing APL thresholds...")
        control_prompts = ["This is a neutral control sentence."] * 10
        apl_thresholds = compute_control_thresholds(runner.model, control_prompts)
    
    return apl_thresholds


def save_model_results(df: pd.DataFrame, model_name: str, output_dir: str) -> None:
    """Save detailed model results to CSV."""
    csv_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to {csv_path}")


def compute_model_summary(df: pd.DataFrame, accuracy: float, n_params: int) -> Dict[str, Any]:
    """Compute summary statistics for a model."""
    # Separate by label
    reasoning_data = df[df['label'] == 'reasoning']
    control_data = df[df['label'] == 'control']
    
    # Compute means and standard deviations
    metrics = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL', 'REV']
    means = {}
    stds = {}
    cohens_d = {}
    
    for metric in metrics:
        reasoning_values = reasoning_data[metric].dropna()
        control_values = control_data[metric].dropna()
        
        means[f"{metric}_reasoning"] = float(reasoning_values.mean()) if len(reasoning_values) > 0 else float("nan")
        means[f"{metric}_control"] = float(control_values.mean()) if len(control_values) > 0 else float("nan")
        stds[f"{metric}_reasoning"] = float(reasoning_values.std()) if len(reasoning_values) > 0 else float("nan")
        stds[f"{metric}_control"] = float(control_values.std()) if len(control_values) > 0 else float("nan")
        
        # Compute Cohen's d
        if len(reasoning_values) > 1 and len(control_values) > 1:
            pooled_std = np.sqrt(0.5 * (reasoning_values.var() + control_values.var()))
            cohens_d[metric] = float((reasoning_values.mean() - control_values.mean()) / pooled_std) if pooled_std > 0 else 0.0
        else:
            cohens_d[metric] = float("nan")
    
    # Compute AUROC for REV
    try:
        valid_mask = df['REV'].notna() & df['label_num'].notna()
        if valid_mask.sum() > 0:
            auroc_rev = roc_auc_score(df.loc[valid_mask, 'label_num'], df.loc[valid_mask, 'REV'])
        else:
            auroc_rev = float("nan")
    except:
        auroc_rev = float("nan")
    
    return {
        "n_params": n_params,
        "accuracy": accuracy,
        "n_reasoning": int(len(reasoning_data)),
        "n_control": int(len(control_data)),
        "means": means,
        "stds": stds,
        "cohens_d": cohens_d,
        "auroc_REV": float(auroc_rev),
        "partial_corr": {}  # Will be filled by partial correlation computation
    }


def compute_trend_analysis(aggregate_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute trend analysis across models."""
    from scipy.stats import pearsonr
    
    n_params = np.array(aggregate_data["n_params"])
    log_params = np.log(n_params)
    
    trends = {}
    for metric in ["d_REV", "auroc_REV", "partial_r_REV"]:
        values = np.array(aggregate_data[metric])
        valid_mask = np.isfinite(values)
        
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], values[valid_mask])
            trends[metric] = {"r": float(r), "p": float(p)}
        else:
            trends[metric] = {"r": float("nan"), "p": float("nan")}
    
    return trends


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Phase 4 evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    args = parser.parse_args()
    
    run_phase4_evaluation(args.config)


if __name__ == "__main__":
    main()

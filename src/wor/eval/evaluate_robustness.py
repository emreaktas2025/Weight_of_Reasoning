"""Robustness evaluation across seeds, temperatures, and metric ablations."""

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core.runner import ModelRunner
from ..core.utils import set_seed, ensure_dir, save_json
from ..utils.hw import detect_hw, get_model_config_with_hw
from ..data.loaders import load_reasoning_dataset, load_control_dataset
from ..metrics.activation_energy import activation_energy
from ..metrics.attention_entropy import attention_process_entropy
from ..metrics.activation_path_length import compute_apl, load_control_thresholds
from ..metrics.circuit_utilization_density import compute_cud, load_circuit_heads, load_control_thresholds as load_cud_thresholds
from ..metrics.stability_intermediate_beliefs import compute_sib_simple
from ..metrics.feature_load import compute_feature_load
from ..metrics.rev_composite import compute_rev_scores
from sklearn.metrics import roc_auc_score
from ..utils.logging import create_run_logger


def compute_metrics_with_ablation(
    runner: ModelRunner,
    data: List[Dict[str, Any]],
    circuit_heads: Any,
    cud_thresholds: Any,
    apl_thresholds: Any,
    model_cfg: Dict[str, Any],
    ablate_metric: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute metrics with optional ablation of one metric.
    
    Args:
        runner: ModelRunner instance
        data: Dataset samples
        circuit_heads: Circuit heads for CUD
        cud_thresholds: Thresholds for CUD
        apl_thresholds: Thresholds for APL
        model_cfg: Model configuration
        ablate_metric: Metric to ablate (set to 0), or None for full metrics
        
    Returns:
        DataFrame with metrics
    """
    rows = []
    
    for item in data:
        try:
            # Generate text and get activations
            result = runner.generate(item["prompt"])
            text = result["text"]
            generated_text = result["generated_text"]
            
            # Determine label
            label = item.get("label", "control")
            
            # Compute reasoning window length
            reasoning_len = min(32, model_cfg.get("max_new_tokens", 64) - 1)
            
            # Compute all metrics (even if ablating, compute for consistency)
            ae = activation_energy(result["hidden_states"], reasoning_len)
            ape = attention_process_entropy(result["attention_probs"], reasoning_len)
            apl = compute_apl(runner.model, result["cache"], apl_thresholds, result["input_tokens"])
            cud = compute_cud(runner.model, result["cache"], circuit_heads, cud_thresholds, result["input_tokens"])
            sib = compute_sib_simple(runner.model, result["cache"], result["input_tokens"], item["prompt"], reasoning_len)
            fl = compute_feature_load(runner.model, result["cache"], result["input_tokens"], reasoning_len)
            
            # Apply ablation if specified
            if ablate_metric == "AE":
                ae = 0.0
            elif ablate_metric == "APE":
                ape = 0.0
            elif ablate_metric == "APL":
                apl = 0.0
            elif ablate_metric == "CUD":
                cud = 0.0
            elif ablate_metric == "SIB":
                sib = 0.0
            elif ablate_metric == "FL":
                fl = 0.0
            
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
                "FL": fl
            })
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            rows.append({
                "id": item["id"],
                "label": item.get("label", "control"),
                "token_len": 0,
                "ppl": float("nan"),
                "AE": float("nan"),
                "APE": float("nan"),
                "APL": float("nan"),
                "CUD": float("nan"),
                "SIB": float("nan"),
                "FL": float("nan")
            })
    
    # Create DataFrame and compute REV scores
    df = pd.DataFrame(rows)
    
    # Compute REV with potentially ablated metrics
    rev_scores = compute_rev_scores(df)
    df['REV'] = rev_scores
    
    # Add label_num column for AUROC computation
    df['label_num'] = df['label'].map({'reasoning': 1, 'control': 0})
    
    return df


def evaluate_with_seed_and_temp(
    model_cfg: Dict[str, Any],
    data: List[Dict[str, Any]],
    circuit_heads: Any,
    cud_thresholds: Any,
    apl_thresholds: Any,
    seed: int,
    temperature: float,
    ablate_metric: Optional[str] = None
) -> Dict[str, float]:
    """
    Run evaluation with specific seed and temperature.
    
    Args:
        model_cfg: Model configuration
        data: Dataset samples
        circuit_heads: Circuit heads for CUD
        cud_thresholds: Thresholds for CUD
        apl_thresholds: Thresholds for APL
        seed: Random seed
        temperature: Sampling temperature
        ablate_metric: Optional metric to ablate
        
    Returns:
        Dict with AUROC and other metrics
    """
    # Set seed
    set_seed(seed)
    
    # Update model config with temperature
    model_cfg = model_cfg.copy()
    model_cfg["temperature"] = temperature
    
    # Initialize model runner
    runner = ModelRunner(model_cfg)
    
    # Compute metrics
    df = compute_metrics_with_ablation(
        runner, data, circuit_heads, cud_thresholds, apl_thresholds, 
        model_cfg, ablate_metric
    )
    
    # Compute AUROC
    valid_mask = df['REV'].notna() & df['label_num'].notna()
    if valid_mask.sum() > 0 and len(df.loc[valid_mask, 'label_num'].unique()) >= 2:
        auroc_rev = roc_auc_score(df.loc[valid_mask, 'label_num'], df.loc[valid_mask, 'REV'])
    else:
        auroc_rev = float('nan')
    
    # Compute Cohen's d
    reasoning_rev = df[df['label'] == 'reasoning']['REV'].dropna()
    control_rev = df[df['label'] == 'control']['REV'].dropna()
    
    if len(reasoning_rev) > 1 and len(control_rev) > 1:
        pooled_std = np.sqrt(0.5 * (reasoning_rev.var() + control_rev.var()))
        cohens_d = (reasoning_rev.mean() - control_rev.mean()) / pooled_std if pooled_std > 0 else 0.0
    else:
        cohens_d = float('nan')
    
    return {
        "auroc_rev": float(auroc_rev),
        "cohens_d": float(cohens_d),
        "n_samples": len(df),
        "n_reasoning": int((df['label'] == 'reasoning').sum()),
        "n_control": int((df['label'] == 'control').sum())
    }


def run_robustness_evaluation(cfg_path: str, fast: bool = False) -> None:
    """
    Run complete robustness evaluation.
    
    Args:
        cfg_path: Path to configuration file
        fast: If True, run with reduced samples for quick testing
    """
    start_time = time.time()
    
    # Create logger
    logger = create_run_logger(experiment_name="phase5_robustness")
    logger.log("="*80)
    logger.log("Phase 5 Robustness Evaluation")
    logger.log("="*80)
    
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    logger.log_config(cfg)
    
    # Detect hardware
    hw_config = detect_hw()
    logger.log_hardware(hw_config)
    
    # Create output directories
    ensure_dir(cfg["output_dir"])
    
    # Load model configuration (use smallest model for robustness tests)
    model_cfg_path = cfg["models"][0]  # Use first (smallest) model
    with open(model_cfg_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    
    # Update with hardware-optimized settings
    model_cfg = get_model_config_with_hw(model_cfg, hw_config)
    
    logger.log_model_start(model_cfg["model_name"], 0)
    
    # Load datasets (reduced size if fast mode)
    logger.log("\n=== Loading Datasets ===")
    n_samples = 10 if fast else 50
    
    all_data = []
    for dataset_name in ["gsm8k"]:  # Use single dataset for robustness
        data = load_reasoning_dataset(dataset_name, n_samples, cfg.get("seed", 1337))
        all_data.extend(data)
    
    control_data = load_control_dataset("wiki", n_samples, cfg.get("seed", 1337))
    all_data.extend(control_data)
    
    logger.log(f"Total samples: {len(all_data)}")
    
    # Load or compute circuit heads and thresholds
    circuit_heads = load_circuit_heads(cfg.get("circuit_heads_json", "reports/circuits/heads.json"))
    cud_thresholds = load_cud_thresholds(cfg.get("control_thresholds_npz", "reports/control_thresholds_phase3.npz"))
    apl_thresholds = load_control_thresholds()
    
    # If not available, compute them on-the-fly
    if circuit_heads is None or cud_thresholds is None:
        logger.log("Circuit heads or CUD thresholds not found - computing on-the-fly...")
        from ..metrics.circuit_utilization_density import compute_circuit_heads, get_arithmetic_prompts
        
        # Initialize a temporary model runner to compute thresholds
        with open(model_cfg_path, 'r') as f:
            temp_model_cfg = yaml.safe_load(f)
        temp_model_cfg = get_model_config_with_hw(temp_model_cfg, hw_config)
        
        from ..core.runner import ModelRunner
        temp_runner = ModelRunner(temp_model_cfg)
        
        # Compute circuit heads and thresholds
        arithmetic_prompts = get_arithmetic_prompts()
        control_prompts = ["This is a neutral control sentence."] * 10
        circuit_heads, cud_thresholds = compute_circuit_heads(
            temp_runner.model, arithmetic_prompts, control_prompts, max_heads=24
        )
        logger.log(f"Computed {len(circuit_heads) if circuit_heads else 0} circuit heads")
        
        # Clean up temporary runner
        del temp_runner
        
    if apl_thresholds is None:
        logger.log("APL thresholds not found - computing on-the-fly...")
        from ..metrics.activation_path_length import compute_control_thresholds
        
        # Use the same temporary model approach
        with open(model_cfg_path, 'r') as f:
            temp_model_cfg = yaml.safe_load(f)
        temp_model_cfg = get_model_config_with_hw(temp_model_cfg, hw_config)
        
        from ..core.runner import ModelRunner
        temp_runner = ModelRunner(temp_model_cfg)
        
        control_prompts = ["This is a neutral control sentence."] * 10
        apl_thresholds = compute_control_thresholds(temp_runner.model, control_prompts)
        logger.log("Computed APL thresholds")
        
        # Clean up
        del temp_runner
    
    # Initialize results
    results = {
        "seed_results": {},
        "temp_results": {},
        "ablation_results": {},
        "baseline_auroc": None
    }
    
    # Get robustness parameters
    seeds = cfg.get("robustness", {}).get("seeds", [42, 1337, 999])
    temperatures = cfg.get("robustness", {}).get("temperatures", [0.0, 0.2])
    metrics_to_ablate = cfg.get("robustness", {}).get("metric_ablations", ["AE", "APE", "APL", "CUD", "SIB", "FL"])
    
    # Run with fast mode settings if specified
    if fast:
        seeds = seeds[:2]  # Only 2 seeds
        temperatures = temperatures[:1]  # Only 1 temperature
        metrics_to_ablate = metrics_to_ablate[:3]  # Only 3 ablations
    
    # 1. Evaluate across seeds (with default temperature 0.0)
    logger.log("\n=== Evaluating Across Seeds ===")
    for seed in seeds:
        logger.log(f"Testing seed {seed}...")
        seed_result = evaluate_with_seed_and_temp(
            model_cfg, all_data, circuit_heads, cud_thresholds, apl_thresholds,
            seed, 0.0, ablate_metric=None
        )
        results["seed_results"][str(seed)] = seed_result
        logger.log_metrics(seed_result, prefix=f"Seed {seed}")
    
    # Store baseline AUROC from first seed
    results["baseline_auroc"] = results["seed_results"][str(seeds[0])]["auroc_rev"]
    
    # 2. Evaluate across temperatures (with default seed 1337)
    logger.log("\n=== Evaluating Across Temperatures ===")
    for temp in temperatures:
        logger.log(f"Testing temperature {temp}...")
        temp_result = evaluate_with_seed_and_temp(
            model_cfg, all_data, circuit_heads, cud_thresholds, apl_thresholds,
            1337, temp, ablate_metric=None
        )
        results["temp_results"][str(temp)] = temp_result
        logger.log_metrics(temp_result, prefix=f"Temp {temp}")
    
    # 3. Evaluate metric ablations (with default seed and temp)
    logger.log("\n=== Evaluating Metric Ablations ===")
    for metric in metrics_to_ablate:
        logger.log(f"Ablating {metric}...")
        ablation_result = evaluate_with_seed_and_temp(
            model_cfg, all_data, circuit_heads, cud_thresholds, apl_thresholds,
            1337, 0.0, ablate_metric=metric
        )
        
        # Compute delta AUROC
        delta_auroc = ablation_result["auroc_rev"] - results["baseline_auroc"]
        ablation_result["delta_auroc"] = float(delta_auroc)
        
        results["ablation_results"][metric] = ablation_result
        logger.log_metrics(ablation_result, prefix=f"Ablate {metric}")
    
    # Compute summary statistics
    seed_aurocs = [r["auroc_rev"] for r in results["seed_results"].values() if not np.isnan(r["auroc_rev"])]
    temp_aurocs = [r["auroc_rev"] for r in results["temp_results"].values() if not np.isnan(r["auroc_rev"])]
    
    results["summary"] = {
        "seed_mean_auroc": float(np.mean(seed_aurocs)) if seed_aurocs else float('nan'),
        "seed_std_auroc": float(np.std(seed_aurocs)) if seed_aurocs else float('nan'),
        "temp_mean_auroc": float(np.mean(temp_aurocs)) if temp_aurocs else float('nan'),
        "temp_std_auroc": float(np.std(temp_aurocs)) if temp_aurocs else float('nan'),
        "most_important_metric": max(
            results["ablation_results"].items(), 
            key=lambda x: abs(x[1].get("delta_auroc", 0))
        )[0] if results["ablation_results"] else None
    }
    
    # Save results
    output_path = os.path.join(cfg["output_dir"], "robustness_summary.json")
    save_json(output_path, results)
    logger.log(f"\n✅ Results saved to {output_path}")
    
    # Log summary
    logger.log("\n=== Robustness Summary ===")
    logger.log(f"Seed mean AUROC: {results['summary']['seed_mean_auroc']:.4f} ± {results['summary']['seed_std_auroc']:.4f}")
    logger.log(f"Temp mean AUROC: {results['summary']['temp_mean_auroc']:.4f} ± {results['summary']['temp_std_auroc']:.4f}")
    logger.log(f"Most important metric: {results['summary']['most_important_metric']}")
    
    total_runtime = time.time() - start_time
    logger.log_runtime("Robustness evaluation", total_runtime)
    
    logger.finalize(total_runtime)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Phase 5 robustness evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    parser.add_argument("--fast", action="store_true", help="Run quick test with reduced samples")
    args = parser.parse_args()
    
    run_robustness_evaluation(args.config, args.fast)


if __name__ == "__main__":
    main()


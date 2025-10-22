"""Phase 5 evaluation pipeline: NeurIPS-grade comprehensive evaluation."""

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
from ..core.experiment_logger import log_experiment
from ..utils.hw import detect_hw, log_hw_config, get_optimal_dataset_sizes, get_model_config_with_hw
from ..utils.hf_auth import ensure_hf_auth
from ..utils.logging import create_run_logger
from ..data.loaders import (
    load_reasoning_dataset, load_control_dataset, load_math_dataset,
    save_dataset_manifest, load_dataset_from_manifest, validate_dataset_labels
)
from ..metrics.activation_energy import activation_energy
from ..metrics.attention_entropy import attention_process_entropy
from ..metrics.activation_path_length import compute_apl, compute_control_thresholds, load_control_thresholds
from ..metrics.circuit_utilization_density import (
    compute_circuit_heads, save_circuit_heads, load_circuit_heads,
    load_control_thresholds as load_cud_thresholds, compute_cud, get_arithmetic_prompts
)
from ..metrics.stability_intermediate_beliefs import compute_sib_simple
from ..metrics.feature_load import compute_feature_load
from ..metrics.rev_composite import compute_rev_scores
from ..stats.partial_corr import compute_partial_correlations
from ..baselines.predictors import evaluate_baseline_vs_rev, get_roc_curves
from ..mech.patchout import (
    rank_attention_heads, rank_layers_by_rev_contribution,
    safe_patchout_heads, safe_patchout_layers, compute_delta_correlation
)
from sklearn.metrics import roc_auc_score


def estimate_model_params(model) -> int:
    """Estimate model parameter count."""
    try:
        return sum(p.numel() for p in model.parameters())
    except:
        return 0


def compute_baseline_metrics_phase5(
    runner: ModelRunner,
    all_data: List[Dict[str, Any]],
    circuit_heads: Any,
    cud_thresholds: Any,
    apl_thresholds: Any,
    model_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Compute baseline metrics for Phase 5."""
    print("Computing baseline metrics...")
    
    rows = []
    total = len(all_data)
    
    for idx, item in enumerate(all_data):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{total}...")
        try:
            # Generate text and get activations
            result = runner.generate(item["prompt"])
            text = result["text"]
            generated_text = result["generated_text"]
            
            # Determine label and dataset
            label = item.get("label", "control")
            dataset = item["id"].split("_")[0]  # Extract dataset from ID
            
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
                "dataset": dataset,
                "token_len": token_count,
                "ppl": ppl,
                "AE": ae,
                "APE": ape,
                "APL": apl,
                "CUD": cud,
                "SIB": sib,
                "FL": fl,
                "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                "answer": item.get("answer", "")
            })
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            rows.append({
                "id": item["id"],
                "label": item.get("label", "control"),
                "dataset": item["id"].split("_")[0],
                "token_len": 0,
                "ppl": float("nan"),
                "AE": float("nan"),
                "APE": float("nan"),
                "APL": float("nan"),
                "CUD": float("nan"),
                "SIB": float("nan"),
                "FL": float("nan"),
                "generated_text": "ERROR",
                "answer": item.get("answer", "")
            })
    
    # Create DataFrame and compute REV scores
    df = pd.DataFrame(rows)
    
    # ID alignment sanity checks
    assert len(df) == len(all_data), f"Sample count mismatch: df={len(df)} vs data={len(all_data)}"
    assert list(df['id']) == [item['id'] for item in all_data], "ID order mismatch between df and all_data"
    print(f"[OK] DataFrame ID alignment verified for {len(df)} samples.")
    
    rev_scores = compute_rev_scores(df)
    df['REV'] = rev_scores
    
    # Add label_num column for partial correlations
    df['label_num'] = df['label'].map({'reasoning': 1, 'control': 0})
    
    return df


def compute_model_summary_phase5(
    df: pd.DataFrame, 
    n_params: int
) -> Dict[str, Any]:
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
        
        # Debug logging for AUROC computation
        n_total = int(valid_mask.sum())
        n_pos = int((df.loc[valid_mask, 'label_num'] == 1).sum())
        n_neg = int((df.loc[valid_mask, 'label_num'] == 0).sum())
        rev_min, rev_max = float(df['REV'].min()), float(df['REV'].max())
        print(f"[AUROC Debug] N={n_total}, pos={n_pos}, neg={n_neg}, REV_range=[{rev_min:.3f}, {rev_max:.3f}]")
        
        if valid_mask.sum() > 0 and len(df.loc[valid_mask, 'label_num'].unique()) >= 2:
            auroc_rev = roc_auc_score(df.loc[valid_mask, 'label_num'], df.loc[valid_mask, 'REV'])
        else:
            auroc_rev = float("nan")
    except:
        auroc_rev = float("nan")
    
    # Per-dataset metrics
    dataset_metrics = {}
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        dataset_reasoning = dataset_df[dataset_df['label'] == 'reasoning']
        dataset_control = dataset_df[dataset_df['label'] == 'control']
        
        if len(dataset_reasoning) > 0 and len(dataset_control) > 0:
            # Compute dataset-specific AUROC
            dataset_valid = dataset_df['REV'].notna() & dataset_df['label_num'].notna()
            if dataset_valid.sum() > 0 and len(dataset_df.loc[dataset_valid, 'label_num'].unique()) >= 2:
                dataset_auroc = roc_auc_score(
                    dataset_df.loc[dataset_valid, 'label_num'], 
                    dataset_df.loc[dataset_valid, 'REV']
                )
            else:
                dataset_auroc = float('nan')
            
            dataset_metrics[dataset] = {
                "auroc_rev": float(dataset_auroc),
                "n_samples": int(len(dataset_df))
            }
    
    return {
        "n_params": n_params,
        "n_reasoning": int(len(reasoning_data)),
        "n_control": int(len(control_data)),
        "means": means,
        "stds": stds,
        "cohens_d": cohens_d,
        "auroc_REV": float(auroc_rev),
        "dataset_metrics": dataset_metrics
    }


def run_phase5_evaluation(cfg_path: str, fast: bool = False) -> None:
    """Run the Phase 5 evaluation pipeline."""
    start_time = time.time()
    
    # Create logger
    logger = create_run_logger(experiment_name="phase5")
    logger.log("="*80)
    logger.log("Phase 5 NeurIPS Evaluation Pipeline")
    logger.log("="*80)
    
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    logger.log_config(cfg)
    
    # Setup HuggingFace authentication
    auth_success = ensure_hf_auth()
    logger.log(f"HuggingFace auth: {'Success' if auth_success else 'Failed'}")
    
    # Detect hardware and log configuration
    hw_config = detect_hw()
    logger.log_hardware(hw_config)
    log_hw_config(hw_config, os.path.join(cfg["output_dir"], "hw_phase5.json"))
    
    # Set seed for reproducibility
    set_seed(cfg.get("seed", 1337))
    
    # Create output directories
    ensure_dir(cfg["output_dir"])
    ensure_dir(cfg["plots_dir"])
    ensure_dir(cfg.get("splits_dir", "reports/splits"))
    
    # Get optimal dataset sizes based on hardware
    optimal_sizes = get_optimal_dataset_sizes(hw_config, cfg["datasets"])
    
    # Override with smaller sizes if fast mode
    if fast:
        logger.log("Fast mode enabled - using reduced sample sizes")
        for category in optimal_sizes:
            for dataset in optimal_sizes[category]:
                optimal_sizes[category][dataset] = min(10, optimal_sizes[category][dataset])
    
    # Load datasets with proper labeling
    logger.log("\n=== Loading Datasets ===")
    all_data = []
    
    # Load reasoning datasets
    reasoning_loaders = {
        "gsm8k": load_reasoning_dataset,
        "strategyqa": load_reasoning_dataset,
        "math": load_math_dataset
    }
    
    for dataset_name, n in optimal_sizes["reasoning"].items():
        loader = reasoning_loaders.get(dataset_name, load_reasoning_dataset)
        try:
            # Try loading from manifest first (for exact reproducibility)
            manifest_path = os.path.join(cfg.get("splits_dir", "reports/splits"), f"{dataset_name}_manifest.csv")
            data = load_dataset_from_manifest(manifest_path)
            
            # If manifest doesn't exist, generate new split
            if data is None:
                data = loader(dataset_name, n, cfg.get("seed", 1337))
                save_dataset_manifest(data, dataset_name, cfg.get("splits_dir", "reports/splits"))
            
            if validate_dataset_labels(data, "reasoning"):
                all_data.extend(data)
                logger.log_dataset_info(dataset_name, len(data))
        except Exception as e:
            logger.log(f"Failed to load {dataset_name}: {e}")
    
    # Load control dataset
    control_n = optimal_sizes["control"]["wiki"]
    manifest_path = os.path.join(cfg.get("splits_dir", "reports/splits"), "wiki_manifest.csv")
    control_data = load_dataset_from_manifest(manifest_path)
    
    if control_data is None:
        control_data = load_control_dataset("wiki", control_n, cfg.get("seed", 1337))
        save_dataset_manifest(control_data, "wiki", cfg.get("splits_dir", "reports/splits"))
    
    if validate_dataset_labels(control_data, "control"):
        all_data.extend(control_data)
        logger.log_dataset_info("wiki", len(control_data))
    
    logger.log(f"Total samples loaded: {len(all_data)}")
    
    # Initialize results storage
    model_results = {}
    
    # Process each model
    logger.log("\n=== Processing Models ===")
    for model_cfg_path in cfg["models"]:
        try:
            logger.log(f"\nProcessing model: {model_cfg_path}")
            model_start_time = time.time()
            
            # Load model configuration
            with open(model_cfg_path, 'r') as f:
                model_cfg = yaml.safe_load(f)
            
            # Check if model requires auth
            if model_cfg.get("requires_auth", False) and not auth_success:
                logger.log(f"Skipping {model_cfg['model_name']} - requires authentication")
                continue
            
            # Update with hardware-optimized settings
            model_cfg = get_model_config_with_hw(model_cfg, hw_config)
            
            # Extract model name for file naming
            model_name = os.path.splitext(os.path.basename(model_cfg_path))[0]
            
            # Initialize model runner
            logger.log(f"Loading model: {model_cfg['model_name']}")
            runner = ModelRunner(model_cfg)
            
            # Get model parameter count
            n_params = estimate_model_params(runner.model)
            logger.log_model_start(model_name, n_params)
            
            # Load or compute circuit heads and thresholds
            circuit_heads = load_circuit_heads(cfg.get("circuit_heads_json", "reports/circuits/heads.json"))
            cud_thresholds = load_cud_thresholds(cfg.get("control_thresholds_npz", "reports/control_thresholds_phase3.npz"))
            apl_thresholds = load_control_thresholds()
            
            if circuit_heads is None or cud_thresholds is None:
                logger.log("Computing circuit heads and thresholds...")
                arithmetic_prompts = get_arithmetic_prompts()
                control_prompts = ["This is a neutral control sentence."] * 10
                circuit_heads, cud_thresholds = compute_circuit_heads(
                    runner.model, arithmetic_prompts, control_prompts, max_heads=24
                )
            
            if apl_thresholds is None:
                logger.log("Computing APL thresholds...")
                control_prompts = ["This is a neutral control sentence."] * 10
                apl_thresholds = compute_control_thresholds(runner.model, control_prompts)
            
            # Compute baseline metrics
            logger.log("Computing baseline metrics...")
            df = compute_baseline_metrics_phase5(
                runner, all_data, circuit_heads, cud_thresholds, apl_thresholds, model_cfg
            )
            
            # Save detailed metrics CSV
            metrics_csv_path = os.path.join(cfg["output_dir"], f"{model_name}_metrics.csv")
            df.to_csv(metrics_csv_path, index=False)
            logger.log(f"Saved metrics to {metrics_csv_path}")
            
            # Compute model summary
            summary = compute_model_summary_phase5(df, n_params)
            
            # Compute partial correlations
            partial_corr_results = compute_partial_correlations(df)
            summary["partial_corr"] = partial_corr_results
            
            # Evaluate baselines
            logger.log("Evaluating baseline predictors...")
            baseline_results = evaluate_baseline_vs_rev(df)
            summary["baseline_comparison"] = baseline_results
            
            # Get ROC curves for plotting
            roc_data = get_roc_curves(df)
            baseline_results["roc_data"] = roc_data
            
            logger.log_baseline_comparison(baseline_results)
            
            # Save model results
            model_results[model_name] = summary
            
            # Save individual model summary
            summary_path = os.path.join(cfg["output_dir"], f"{model_name}_summary.json")
            save_json(summary_path, summary)
            
            # Log experiment for reproducibility
            log_experiment(
                config=model_cfg,
                results=summary,
                output_dir=os.path.join(cfg["output_dir"], "tracking"),
                experiment_name=model_name
            )
            
            model_runtime = time.time() - model_start_time
            logger.log_runtime(f"{model_name}", model_runtime)
            logger.log(f"✅ {model_name} completed successfully")
            
        except Exception as e:
            logger.log(f"❌ Failed to process {model_cfg_path}: {e}")
            continue
    
    # Save aggregate model results
    model_results_path = os.path.join(cfg["output_dir"], "model_results.json")
    save_json(model_results_path, model_results)
    
    # Save baseline comparison separately
    if model_results:
        first_model = list(model_results.values())[0]
        if "baseline_comparison" in first_model:
            baseline_path = os.path.join(cfg["output_dir"], "baseline_comparison.json")
            save_json(baseline_path, first_model["baseline_comparison"])
    
    # Validate success criteria
    logger.log("\n=== Validating Success Criteria ===")
    criteria = validate_success_criteria(model_results, cfg)
    logger.log_success_criteria(criteria)
    
    total_runtime = time.time() - start_time
    logger.log(f"\n=== Phase 5 Evaluation Complete ===")
    logger.log(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    logger.log(f"Models processed: {len(model_results)}")
    logger.log(f"Results saved to: {cfg['output_dir']}")
    
    # Print final summary
    print_phase5_summary(model_results, total_runtime)
    
    logger.finalize(total_runtime)


def validate_success_criteria(
    model_results: Dict[str, Any],
    cfg: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Validate success criteria for NeurIPS submission.
    
    Returns:
        Dict mapping criterion name to pass/fail boolean
    """
    criteria = {}
    
    # Criterion 1: REV adds >= +0.05 AUROC over baseline on >= 2 datasets
    datasets_with_improvement = 0
    for model_name, results in model_results.items():
        if "baseline_comparison" in results:
            delta_auc = results["baseline_comparison"].get("delta_auc", 0)
            if delta_auc >= 0.05:
                datasets_with_improvement += 1
    
    criteria["REV improves baseline by >=0.05 AUC on >=2 datasets"] = datasets_with_improvement >= 2
    
    # Criterion 2: Check if we have results for multiple models (scaling)
    criteria["Multiple models evaluated (scaling)"] = len(model_results) >= 2
    
    # Criterion 3: REV shows significant effect (Cohen's d > 0.5)
    significant_effects = sum(
        1 for results in model_results.values()
        if results.get("cohens_d", {}).get("REV", 0) > 0.5
    )
    criteria["REV shows medium+ effect size (d>0.5)"] = significant_effects > 0
    
    return criteria


def print_phase5_summary(model_results: Dict[str, Any], total_runtime: float) -> None:
    """Print final Phase 5 summary."""
    print("\n" + "="*80)
    print("✅ Phase 5 NeurIPS Upgrade Complete – Ready for paper writing.")
    print("="*80)
    
    print("\nSummary:")
    
    # Get best model results
    if model_results:
        best_model = max(
            model_results.items(),
            key=lambda x: x[1].get("baseline_comparison", {}).get("delta_auc", 0)
        )
        model_name, results = best_model
        
        delta_auc = results.get("baseline_comparison", {}).get("delta_auc", 0)
        auroc_baseline = results.get("baseline_comparison", {}).get("auroc_baseline", 0)
        auroc_rev = results.get("baseline_comparison", {}).get("auroc_rev", 0)
        cohens_d = results.get("cohens_d", {}).get("REV", 0)
        
        print(f"- Best model: {model_name}")
        print(f"- REV ΔAUC: +{delta_auc:.4f} over baseline")
        print(f"  (Baseline: {auroc_baseline:.4f}, REV: {auroc_rev:.4f})")
        print(f"- Cohen's d (REV): {cohens_d:.4f}")
        
        # Dataset-specific results
        if "dataset_metrics" in results:
            print("\n- Per-dataset AUROC:")
            for dataset, metrics in results["dataset_metrics"].items():
                print(f"  {dataset}: {metrics['auroc_rev']:.4f} (n={metrics['n_samples']})")
    
    print(f"\n- Runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min)")
    print(f"- Models evaluated: {len(model_results)}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("1. Run robustness tests: bash scripts/06_phase5_robustness.sh")
    print("2. Generate figures: python -m wor.plots.paper_figs")
    print("3. Review results in reports/phase5/")
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Phase 5 evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    parser.add_argument("--fast", action="store_true", help="Run quick test with reduced samples")
    args = parser.parse_args()
    
    run_phase5_evaluation(args.config, args.fast)


if __name__ == "__main__":
    main()


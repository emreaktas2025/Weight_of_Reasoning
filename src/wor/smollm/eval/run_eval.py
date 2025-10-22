"""SmolLM REV evaluation pipeline with iteration management."""

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import SmolLM-specific modules
from ..loaders import SmolLMRunner, load_smollm_dataset
from ..activations import (
    extract_hidden_states_from_cache, extract_attention_from_cache,
    compute_activation_energy, compute_feature_load
)
from ..metrics.rev import compute_rev_scores, validate_rev_scores
from ..metrics.apl import compute_apl, compute_control_thresholds
from ..metrics.ape import attention_process_entropy
from ..metrics.sib import compute_sib_simple
from ..utils.io_utils import (
    create_iteration_folder, save_iteration_metadata, update_iteration_registry,
    save_results_json, create_notes_template
)
from .stats import compute_partial_correlations, evaluate_baseline_vs_rev
from .aggregate import bootstrap_confidence_interval, delong_test
from sklearn.metrics import roc_auc_score


def estimate_model_params(model) -> int:
    """Estimate model parameter count."""
    try:
        return sum(p.numel() for p in model.parameters())
    except:
        return 0


def compute_baseline_metrics(
    runner: SmolLMRunner,
    all_data: List[Dict[str, Any]],
    model_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Compute baseline metrics for SmolLM evaluation."""
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
            
            # Extract activations
            hidden_states = extract_hidden_states_from_cache(result["cache"])
            attention_probs = extract_attention_from_cache(result["cache"])
            
            # Compute individual metrics
            ae = compute_activation_energy(hidden_states, reasoning_len)
            ape = attention_process_entropy(attention_probs, reasoning_len)
            fl = compute_feature_load(hidden_states, reasoning_len)
            
            # For APL and SIB, we need control thresholds and more complex computation
            # For now, use simplified versions
            apl = float("nan")  # Will be computed with control thresholds
            sib = compute_sib_simple(runner.model, result["cache"], result["input_tokens"], 
                                   item["prompt"], reasoning_len)
            
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
                "CUD": float("nan"),  # Will be computed separately
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
    
    # Compute REV scores (will be NaN for missing metrics)
    try:
        rev_scores = compute_rev_scores(df)
        df['REV'] = rev_scores
        validate_rev_scores(rev_scores)
    except Exception as e:
        print(f"Warning: REV computation failed: {e}")
        df['REV'] = float("nan")
    
    # Add label_num column for partial correlations
    df['label_num'] = df['label'].map({'reasoning': 1, 'control': 0})
    
    return df


def compute_model_summary(
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


def run_smollm_evaluation(cfg_path: str, reproduce_iteration: str = None) -> None:
    """Run the SmolLM evaluation pipeline."""
    start_time = time.time()
    
    print("="*80)
    print("SmolLM REV Evaluation Pipeline")
    print("="*80)
    
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Configuration loaded from: {cfg_path}")
    
    # Set seed for reproducibility
    np.random.seed(cfg.get("seed", 42))
    
    # Create iteration folder
    if reproduce_iteration:
        iteration_path = os.path.join("experiments", reproduce_iteration)
        if not os.path.exists(iteration_path):
            raise ValueError(f"Reproduction iteration {reproduce_iteration} not found")
        print(f"Reproducing iteration: {reproduce_iteration}")
    else:
        iteration_path = create_iteration_folder()
        iteration_id = os.path.basename(iteration_path)
        print(f"Created new iteration: {iteration_id}")
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    all_data = []
    
    # Load reasoning datasets
    for dataset_name, n in cfg["datasets"]["reasoning"].items():
        data = load_smollm_dataset(dataset_name, n, cfg.get("seed", 42))
        all_data.extend(data)
        print(f"Loaded {len(data)} samples from {dataset_name}")
    
    # Load control dataset
    for dataset_name, n in cfg["datasets"]["control"].items():
        data = load_smollm_dataset(dataset_name, n, cfg.get("seed", 42))
        all_data.extend(data)
        print(f"Loaded {len(data)} samples from {dataset_name}")
    
    print(f"Total samples loaded: {len(all_data)}")
    
    # Initialize results storage
    model_results = {}
    model_names = []
    
    # Process each model
    print("\n=== Processing Models ===")
    for model_cfg_path in cfg["models"]:
        try:
            print(f"\nProcessing model: {model_cfg_path}")
            model_start_time = time.time()
            
            # Load model configuration
            with open(model_cfg_path, 'r') as f:
                model_cfg = yaml.safe_load(f)
            
            # Extract model name for file naming
            model_name = os.path.splitext(os.path.basename(model_cfg_path))[0]
            model_names.append(model_name)
            
            # Initialize model runner
            print(f"Loading model: {model_cfg['model_name']}")
            runner = SmolLMRunner(model_cfg)
            
            # Get model parameter count
            n_params = estimate_model_params(runner.model)
            print(f"Model parameters: {n_params:,}")
            
            # Compute baseline metrics
            print("Computing baseline metrics...")
            df = compute_baseline_metrics(runner, all_data, model_cfg)
            
            # Save detailed metrics CSV
            metrics_csv_path = os.path.join(iteration_path, "results", f"{model_name}_metrics.csv")
            df.to_csv(metrics_csv_path, index=False)
            print(f"Saved metrics to {metrics_csv_path}")
            
            # Compute model summary
            summary = compute_model_summary(df, n_params)
            
            # Compute partial correlations
            try:
                partial_corr_results = compute_partial_correlations(df)
                summary["partial_corr"] = partial_corr_results
            except Exception as e:
                print(f"Warning: Partial correlation computation failed: {e}")
                summary["partial_corr"] = {}
            
            # Evaluate baselines
            print("Evaluating baseline predictors...")
            try:
                baseline_results = evaluate_baseline_vs_rev(df)
                summary["baseline_comparison"] = baseline_results
            except Exception as e:
                print(f"Warning: Baseline evaluation failed: {e}")
                summary["baseline_comparison"] = {}
            
            # Save model results
            model_results[model_name] = summary
            
            # Save individual model summary
            summary_path = os.path.join(iteration_path, "results", f"{model_name}_summary.json")
            save_results_json(summary, summary_path)
            
            model_runtime = time.time() - model_start_time
            print(f"✅ {model_name} completed in {model_runtime:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to process {model_cfg_path}: {e}")
            continue
    
    # Save aggregate model results
    model_results_path = os.path.join(iteration_path, "results", "model_results.json")
    save_results_json(model_results, model_results_path)
    
    # Compute aggregate statistics
    print("\n=== Computing Aggregate Statistics ===")
    try:
        from .aggregate import compute_aggregate_statistics
        aggregate_stats = compute_aggregate_statistics(model_results)
        
        # Save aggregate results
        aggregate_path = os.path.join(iteration_path, "results", "aggregate_statistics.json")
        save_results_json(aggregate_stats, aggregate_path)
        
        # Update iteration registry
        if not reproduce_iteration:
            update_iteration_registry(
                os.path.basename(iteration_path),
                aggregate_stats,
                iteration_path
            )
        
    except Exception as e:
        print(f"Warning: Aggregate statistics computation failed: {e}")
        aggregate_stats = {}
    
    # Create notes template
    if not reproduce_iteration:
        create_notes_template(iteration_path, cfg, model_names, list(cfg["datasets"]["reasoning"].keys()))
    
    total_runtime = time.time() - start_time
    print(f"\n=== SmolLM Evaluation Complete ===")
    print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Models processed: {len(model_results)}")
    print(f"Results saved to: {iteration_path}")
    
    # Print final summary
    if model_results:
        best_model = max(
            model_results.items(),
            key=lambda x: x[1].get("baseline_comparison", {}).get("delta_auc", 0)
        )
        model_name, results = best_model
        
        delta_auc = results.get("baseline_comparison", {}).get("delta_auc", 0)
        auroc_baseline = results.get("baseline_comparison", {}).get("auroc_baseline", 0)
        auroc_rev = results.get("baseline_comparison", {}).get("auroc_rev", 0)
        
        print(f"\nBest model: {model_name}")
        print(f"REV ΔAUC: +{delta_auc:.4f} over baseline")
        print(f"  (Baseline: {auroc_baseline:.4f}, REV: {auroc_rev:.4f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SmolLM REV evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    parser.add_argument("--reproduce", help="Reproduce specific iteration (e.g., iteration_001)")
    args = parser.parse_args()
    
    run_smollm_evaluation(args.config, args.reproduce)


if __name__ == "__main__":
    main()

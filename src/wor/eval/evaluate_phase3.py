"""Phase 3 evaluation pipeline with CUD, SIB, FL, and REV metrics."""

import argparse
import csv
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from ..core.runner import ModelRunner
from ..core.utils import set_seed, read_jsonl, ensure_dir, save_json
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


def run_phase3_evaluation(cfg_path: str) -> None:
    """Run the Phase 3 evaluation pipeline."""
    start_time = time.time()
    
    # Load configuration
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    with open(cfg["model_cfg"], 'r') as f:
        model_cfg = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(model_cfg.get("seed", 1337))
    
    # Create output directories
    ensure_dir("reports")
    ensure_dir("reports/plots")
    ensure_dir("reports/circuits")
    
    # Initialize model runner
    print("Initializing model...")
    runner = ModelRunner(model_cfg)
    
    # Phase 3a: Circuit Discovery (if not cached)
    print("\n=== Phase 3a: Circuit Discovery ===")
    circuit_heads = load_circuit_heads(cfg.get("circuit_heads_json", "reports/circuits/heads.json"))
    cud_thresholds = load_cud_thresholds(cfg.get("control_thresholds_npz", "reports/control_thresholds_phase3.npz"))
    
    if circuit_heads is None or cud_thresholds is None:
        print("Circuit heads or thresholds not found. Computing...")
        
        # Get arithmetic prompts for circuit discovery
        arithmetic_prompts = get_arithmetic_prompts()
        
        # Load control prompts for threshold computation
        control_prompts = [item["prompt"] for item in read_jsonl(cfg["control_file"])]
        
        # Compute circuit heads and thresholds
        circuit_heads, cud_thresholds = compute_circuit_heads(
            runner.model, arithmetic_prompts, control_prompts, max_heads=24
        )
        
        # Save circuit heads and thresholds
        save_circuit_heads(
            circuit_heads, cud_thresholds,
            cfg.get("circuit_heads_json", "reports/circuits/heads.json"),
            cfg.get("control_thresholds_npz", "reports/control_thresholds_phase3.npz")
        )
    else:
        print(f"Using cached circuit heads ({len(circuit_heads)} heads)")
    
    # Load control thresholds for APL (Phase 2)
    print("Loading APL control thresholds...")
    apl_thresholds = load_control_thresholds()
    if apl_thresholds is None:
        # Load control prompts for APL threshold computation
        control_prompts = [item["prompt"] for item in read_jsonl(cfg["control_file"])]
        apl_thresholds = compute_control_thresholds(runner.model, control_prompts)
    else:
        print("Using cached APL control thresholds")
    
    # Phase 3b: Main Evaluation Loop
    print("\n=== Phase 3b: Main Evaluation ===")
    rows: List[Dict[str, Any]] = []
    
    def process_item(item: Dict[str, Any], label: str) -> None:
        """Process a single item and compute all Phase 3 metrics."""
        try:
            print(f"Processing {item['id']} ({label})...")
            
            # Generate text and get activations
            result = runner.generate(item["prompt"])
            text = result["text"]
            generated_text = result["generated_text"]
            
            # Heuristic: reasoning window = min(32, max_new_tokens - 1)
            reasoning_len = min(32, model_cfg.get("max_new_tokens", 64) - 1)
            
            # Compute existing metrics (Phase 1 & 2)
            ae = activation_energy(result["hidden_states"], reasoning_len)
            ape = attention_process_entropy(result["attention_probs"], reasoning_len)
            apl = compute_apl(runner.model, result["cache"], apl_thresholds, result["input_tokens"])
            
            # Compute new Phase 3 metrics
            cud = compute_cud(runner.model, result["cache"], circuit_heads, cud_thresholds, result["input_tokens"])
            sib = compute_sib_simple(runner.model, result["cache"], result["input_tokens"], item["prompt"], reasoning_len)
            fl = compute_feature_load(runner.model, result["cache"], result["input_tokens"], reasoning_len)
            
            # Count tokens (rough approximation)
            token_count = len(text.split())
            
            # Get perplexity
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
            
            print(f"  AE={ae:.4f}, APE={ape:.4f}, APL={apl:.4f}, CUD={cud:.4f}, SIB={sib:.4f}, FL={fl:.4f}")
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            # Add row with NaN values
            rows.append({
                "id": item["id"],
                "label": label,
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
    
    # Process reasoning prompts
    print("Processing reasoning prompts...")
    for item in read_jsonl(cfg["reasoning_file"]):
        process_item(item, "reasoning")
    
    # Process control prompts
    print("Processing control prompts...")
    for item in read_jsonl(cfg["control_file"]):
        process_item(item, "control")
    
    # Phase 3c: Post-Processing
    print("\n=== Phase 3c: Post-Processing ===")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(rows)
    
    # Compute REV scores
    print("Computing REV scores...")
    rev_scores = compute_rev_scores(df)
    df['REV'] = rev_scores
    
    # Write results to CSV
    out_csv = cfg["out_csv"]
    ensure_dir(os.path.dirname(out_csv))
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            # Add REV to the first row for fieldnames
            fieldnames = list(rows[0].keys()) + ['REV']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write rows with REV scores
            for i, row in enumerate(rows):
                row['REV'] = rev_scores[i]
                writer.writerow(row)
    
    # Compute summary statistics
    print("Computing summary statistics...")
    
    # Separate by label
    reasoning_data = df[df['label'] == 'reasoning']
    control_data = df[df['label'] == 'control']
    
    # Compute means and standard deviations
    metrics = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL', 'REV']
    means = {}
    stds = {}
    
    for metric in metrics:
        reasoning_values = reasoning_data[metric].dropna()
        control_values = control_data[metric].dropna()
        
        means[f"{metric}_reasoning"] = float(reasoning_values.mean()) if len(reasoning_values) > 0 else float("nan")
        means[f"{metric}_control"] = float(control_values.mean()) if len(control_values) > 0 else float("nan")
        stds[f"{metric}_reasoning"] = float(reasoning_values.std()) if len(reasoning_values) > 0 else float("nan")
        stds[f"{metric}_control"] = float(control_values.std()) if len(control_values) > 0 else float("nan")
    
    # Compute Cohen's d for all metrics
    cohens_d = {}
    for metric in metrics:
        reasoning_values = reasoning_data[metric].dropna()
        control_values = control_data[metric].dropna()
        
        if len(reasoning_values) > 1 and len(control_values) > 1:
            pooled_std = np.sqrt(0.5 * (reasoning_values.var() + control_values.var()))
            cohens_d[metric] = float((reasoning_values.mean() - control_values.mean()) / pooled_std) if pooled_std > 0 else 0.0
        else:
            cohens_d[metric] = float("nan")
    
    # Compute partial correlations
    print("Computing partial correlations...")
    df['label_num'] = df['label'].map({'reasoning': 1, 'control': 0})
    partial_corr_results = compute_partial_correlations(df)
    
    # Save partial correlations separately
    if 'partial_corr_json' in cfg:
        save_partial_correlations(partial_corr_results, cfg['partial_corr_json'])
    
    # Compute runtime
    runtime_sec = time.time() - start_time
    
    # Create summary
    summary = {
        "n_reasoning": int(len(reasoning_data)),
        "n_control": int(len(control_data)),
        "means": means,
        "stds": stds,
        "cohens_d": cohens_d,
        "partial_corr": partial_corr_results,
        "runtime_sec": float(runtime_sec),
        "circuit_heads_count": len(circuit_heads) if circuit_heads else 0,
    }
    
    # Write summary
    save_json(cfg["summary_json"], summary)
    
    print(f"\n=== Phase 3 Evaluation Complete ===")
    print(f"Runtime: {runtime_sec:.2f} seconds")
    print(f"Results written to: {out_csv}")
    print(f"Summary written to: {cfg['summary_json']}")
    if 'partial_corr_json' in cfg:
        print(f"Partial correlations written to: {cfg['partial_corr_json']}")
    
    # Print key results
    print(f"\nKey Results:")
    for metric in metrics:
        d = cohens_d[metric]
        if np.isfinite(d):
            print(f"  {metric}: Cohen's d = {d:.4f}")
        else:
            print(f"  {metric}: Cohen's d = NaN")
    
    # Check if REV shows significant effect
    if 'REV' in partial_corr_results:
        rev_p = partial_corr_results['REV']['p']
        if np.isfinite(rev_p) and rev_p < 0.05:
            print(f"\n✅ REV shows significant effect (p = {rev_p:.4f} < 0.05)")
        else:
            print(f"\n⚠️  REV effect not significant (p = {rev_p:.4f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Phase 3 evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    args = parser.parse_args()
    
    run_phase3_evaluation(args.config)


if __name__ == "__main__":
    main()

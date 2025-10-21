"""Evaluation pipeline for running tiny evals and computing metrics."""

import argparse
import csv
import json
import os
import yaml
import numpy as np
from typing import Dict, Any, List

from ..core.runner import ModelRunner
from ..core.utils import set_seed, read_jsonl, ensure_dir, save_json
from ..metrics.activation_energy import activation_energy
from ..metrics.attention_entropy import attention_process_entropy
from ..metrics.activation_path_length import compute_apl, compute_control_thresholds, load_control_thresholds
from ..stats.partial_corr import compute_partial_correlations, save_partial_correlations


def run_evaluation(cfg_path: str) -> None:
    """Run the evaluation pipeline."""
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
    
    # Initialize model runner
    runner = ModelRunner(model_cfg)
    
    # Compute control thresholds for APL (once at start)
    print("Computing control thresholds for APL...")
    control_thresholds = load_control_thresholds()
    if control_thresholds is None:
        # Load control prompts for threshold computation
        control_prompts = [item["prompt"] for item in read_jsonl(cfg["control_file"])]
        control_thresholds = compute_control_thresholds(runner.model, control_prompts)
    else:
        print("Using cached control thresholds")
    
    # Storage for results
    rows: List[Dict[str, Any]] = []
    
    def process_item(item: Dict[str, Any], label: str) -> None:
        """Process a single item and compute metrics."""
        try:
            # Generate text and get activations
            result = runner.generate(item["prompt"])
            text = result["text"]
            generated_text = result["generated_text"]
            
            # Heuristic: reasoning window = min(32, max_new_tokens - 1)
            reasoning_len = min(32, model_cfg.get("max_new_tokens", 64) - 1)
            
            # Compute metrics
            ae = activation_energy(result["hidden_states"], reasoning_len)
            ape = attention_process_entropy(result["attention_probs"], reasoning_len)
            apl = compute_apl(runner.model, result["cache"], control_thresholds, result["input_tokens"])
            
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
                "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
            })
            
            print(f"Processed {item['id']} ({label}): AE={ae:.4f}, APE={ape:.4f}, APL={apl:.4f}, ppl={ppl:.2f}")
            
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
    
    # Write results to CSV
    out_csv = cfg["out_csv"]
    ensure_dir(os.path.dirname(out_csv))
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    # Compute summary statistics
    reasoning_ae = np.array([r["AE"] for r in rows if r["label"] == "reasoning" and np.isfinite(r["AE"])])
    control_ae = np.array([r["AE"] for r in rows if r["label"] == "control" and np.isfinite(r["AE"])])
    
    reasoning_ape = np.array([r["APE"] for r in rows if r["label"] == "reasoning" and np.isfinite(r["APE"])])
    control_ape = np.array([r["APE"] for r in rows if r["label"] == "control" and np.isfinite(r["APE"])])
    
    reasoning_apl = np.array([r["APL"] for r in rows if r["label"] == "reasoning" and np.isfinite(r["APL"])])
    control_apl = np.array([r["APL"] for r in rows if r["label"] == "control" and np.isfinite(r["APL"])])
    
    # Compute Cohen's d for AE
    if len(reasoning_ae) > 1 and len(control_ae) > 1:
        pooled_std = np.sqrt(0.5 * (reasoning_ae.var() + control_ae.var()))
        cohens_d_ae = (reasoning_ae.mean() - control_ae.mean()) / pooled_std if pooled_std > 0 else 0.0
    else:
        cohens_d_ae = float("nan")
    
    # Compute Cohen's d for APE
    if len(reasoning_ape) > 1 and len(control_ape) > 1:
        pooled_std_ape = np.sqrt(0.5 * (reasoning_ape.var() + control_ape.var()))
        cohens_d_ape = (reasoning_ape.mean() - control_ape.mean()) / pooled_std_ape if pooled_std_ape > 0 else 0.0
    else:
        cohens_d_ape = float("nan")
    
    # Compute Cohen's d for APL
    if len(reasoning_apl) > 1 and len(control_apl) > 1:
        pooled_std_apl = np.sqrt(0.5 * (reasoning_apl.var() + control_apl.var()))
        cohens_d_apl = (reasoning_apl.mean() - control_apl.mean()) / pooled_std_apl if pooled_std_apl > 0 else 0.0
    else:
        cohens_d_apl = float("nan")
    
    # Compute partial correlations
    print("Computing partial correlations...")
    import pandas as pd
    
    # Create DataFrame for partial correlation analysis
    df = pd.DataFrame(rows)
    df['label_num'] = df['label'].map({'reasoning': 1, 'control': 0})
    
    partial_corr_results = compute_partial_correlations(df)
    
    # Save partial correlations separately
    if 'partial_corr_json' in cfg:
        save_partial_correlations(partial_corr_results, cfg['partial_corr_json'])
    
    # Create summary
    summary = {
        "n_reasoning": int(len(reasoning_ae)),
        "n_control": int(len(control_ae)),
        "AE_mean_reasoning": float(np.nanmean(reasoning_ae)) if reasoning_ae.size else float("nan"),
        "AE_mean_control": float(np.nanmean(control_ae)) if control_ae.size else float("nan"),
        "AE_std_reasoning": float(np.nanstd(reasoning_ae)) if reasoning_ae.size else float("nan"),
        "AE_std_control": float(np.nanstd(control_ae)) if control_ae.size else float("nan"),
        "cohens_d_ae": float(cohens_d_ae),
        "APE_mean_reasoning": float(np.nanmean(reasoning_ape)) if reasoning_ape.size else float("nan"),
        "APE_mean_control": float(np.nanmean(control_ape)) if control_ape.size else float("nan"),
        "APE_std_reasoning": float(np.nanstd(reasoning_ape)) if reasoning_ape.size else float("nan"),
        "APE_std_control": float(np.nanstd(control_ape)) if control_ape.size else float("nan"),
        "cohens_d_ape": float(cohens_d_ape),
        "APL_mean_reasoning": float(np.nanmean(reasoning_apl)) if reasoning_apl.size else float("nan"),
        "APL_mean_control": float(np.nanmean(control_apl)) if control_apl.size else float("nan"),
        "APL_std_reasoning": float(np.nanstd(reasoning_apl)) if reasoning_apl.size else float("nan"),
        "APL_std_control": float(np.nanstd(control_apl)) if control_apl.size else float("nan"),
        "cohens_d_apl": float(cohens_d_apl),
        "partial_corr": partial_corr_results,
    }
    
    # Write summary
    save_json(cfg["summary_json"], summary)
    
    print(f"\nEvaluation complete!")
    print(f"Results written to: {out_csv}")
    print(f"Summary written to: {cfg['summary_json']}")
    if 'partial_corr_json' in cfg:
        print(f"Partial correlations written to: {cfg['partial_corr_json']}")
    print(f"Cohen's d (AE): {cohens_d_ae:.4f}")
    print(f"Cohen's d (APE): {cohens_d_ape:.4f}")
    print(f"Cohen's d (APL): {cohens_d_apl:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tiny evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    args = parser.parse_args()
    
    run_evaluation(args.config)


if __name__ == "__main__":
    main()

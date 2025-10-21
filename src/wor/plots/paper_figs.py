"""Generate publication-ready figures for NeurIPS paper."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any, List, Optional
from pathlib import Path

# Set publication-quality style
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


def create_scaling_curve(
    model_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create scaling curve: REV AUROC vs log(parameters).
    
    Args:
        model_results: Dict mapping model names to their results
        output_path: Path to save figure
    """
    print("Creating scaling curve...")
    
    # Extract data
    models = []
    params = []
    aurocs = []
    
    for model_name, results in model_results.items():
        if "n_params" in results and "auroc_REV" in results:
            models.append(model_name)
            params.append(results["n_params"])
            aurocs.append(results["auroc_REV"])
    
    if len(models) == 0:
        print("Warning: No data for scaling curve")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot scaling curve
    ax.scatter(np.log10(params), aurocs, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Fit and plot trend line
    if len(params) >= 2:
        z = np.polyfit(np.log10(params), aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(np.log10(params)), max(np.log10(params)), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Annotate points
    for model, param, auroc in zip(models, params, aurocs):
        # Clean model name for display
        display_name = model.replace('pythia-', '').replace('llama3-', 'llama-')
        ax.annotate(display_name, (np.log10(param), auroc), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('log₁₀(Parameters)', fontweight='bold')
    ax.set_ylabel('AUROC (REV)', fontweight='bold')
    ax.set_title('Scaling: REV AUROC vs Model Size', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved scaling curve to {output_path}")


def create_roc_curves(
    baseline_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create ROC curves comparing baseline, REV, and combined predictors.
    
    Args:
        baseline_results: Results from baseline evaluation
        output_path: Path to save figure
    """
    print("Creating ROC curves...")
    
    if "roc_data" not in baseline_results:
        print("Warning: No ROC data available")
        return
    
    roc_data = baseline_results["roc_data"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot diagonal (chance level)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Chance')
    
    # Plot ROC curves
    colors = {'baseline': 'blue', 'rev': 'red', 'combined': 'green'}
    labels = {
        'baseline': f"Baseline (AUC={baseline_results.get('auroc_baseline', 0):.3f})",
        'rev': f"REV (AUC={baseline_results.get('auroc_rev', 0):.3f})",
        'combined': f"Combined (AUC={baseline_results.get('auroc_combined', 0):.3f})"
    }
    
    for key in ['baseline', 'rev', 'combined']:
        if key in roc_data:
            fpr = roc_data[key]['fpr']
            tpr = roc_data[key]['tpr']
            ax.plot(fpr, tpr, color=colors[key], linewidth=2.5, label=labels[key])
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves: Baseline vs REV vs Combined', fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved ROC curves to {output_path}")


def create_causal_scatter(
    patchout_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create scatter plot of ΔREV vs ΔAccuracy with correlation.
    
    Args:
        patchout_results: Results from patch-out experiments
        output_path: Path to save figure
    """
    print("Creating causal validation scatter plot...")
    
    # Extract delta values
    delta_rev = []
    delta_acc = []
    
    for model_name, model_data in patchout_results.items():
        if "correlations" in model_data:
            for exp_type in ["heads", "layers"]:
                if exp_type in model_data["correlations"]:
                    corr_data = model_data["correlations"][exp_type]
                    if "delta_rev" in corr_data and "delta_acc" in corr_data:
                        delta_rev.extend(corr_data["delta_rev"])
                        delta_acc.extend(corr_data["delta_acc"])
    
    if len(delta_rev) == 0:
        print("Warning: No data for causal scatter plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Scatter plot
    ax.scatter(delta_rev, delta_acc, s=80, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Compute and plot trend line
    if len(delta_rev) >= 2:
        from scipy.stats import spearmanr
        rho, p_value = spearmanr(delta_rev, delta_acc)
        
        # Fit line
        z = np.polyfit(delta_rev, delta_acc, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(delta_rev), max(delta_rev), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2)
        
        # Add correlation annotation
        ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {p_value:.4f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('ΔREV (patched - baseline)', fontweight='bold')
    ax.set_ylabel('ΔAccuracy (patched - baseline)', fontweight='bold')
    ax.set_title('Causal Validation: ΔREV vs ΔAccuracy', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved causal scatter to {output_path}")


def create_robustness_bars(
    robustness_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create bar plots showing ΔAUROC across seeds, temps, and ablations.
    
    Args:
        robustness_results: Results from robustness evaluation
        output_path: Path to save figure
    """
    print("Creating robustness bar plots...")
    
    if not robustness_results or "seed_results" not in robustness_results:
        print("Warning: No robustness data available")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Across seeds
    if "seed_results" in robustness_results:
        seed_data = robustness_results["seed_results"]
        seeds = list(seed_data.keys())
        aurocs = [seed_data[seed].get("auroc_rev", 0) for seed in seeds]
        
        axes[0].bar(range(len(seeds)), aurocs, alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(len(seeds)))
        axes[0].set_xticklabels(seeds)
        axes[0].set_ylabel('AUROC (REV)', fontweight='bold')
        axes[0].set_xlabel('Random Seed', fontweight='bold')
        axes[0].set_title('Robustness Across Seeds', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add mean line
        mean_auroc = np.mean(aurocs)
        axes[0].axhline(mean_auroc, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_auroc:.3f}')
        axes[0].legend()
    
    # 2. Across temperatures
    if "temp_results" in robustness_results:
        temp_data = robustness_results["temp_results"]
        temps = list(temp_data.keys())
        aurocs = [temp_data[temp].get("auroc_rev", 0) for temp in temps]
        
        axes[1].bar(range(len(temps)), aurocs, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xticks(range(len(temps)))
        axes[1].set_xticklabels(temps)
        axes[1].set_ylabel('AUROC (REV)', fontweight='bold')
        axes[1].set_xlabel('Temperature', fontweight='bold')
        axes[1].set_title('Robustness Across Temperatures', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 3. Metric ablations
    if "ablation_results" in robustness_results:
        ablation_data = robustness_results["ablation_results"]
        metrics = list(ablation_data.keys())
        delta_aurocs = [ablation_data[metric].get("delta_auroc", 0) for metric in metrics]
        
        colors = ['green' if d >= 0 else 'red' for d in delta_aurocs]
        axes[2].barh(range(len(metrics)), delta_aurocs, alpha=0.7, edgecolor='black', color=colors)
        axes[2].set_yticks(range(len(metrics)))
        axes[2].set_yticklabels(metrics)
        axes[2].set_xlabel('ΔAUROC (without metric)', fontweight='bold')
        axes[2].set_title('Metric Ablation Impact', fontweight='bold')
        axes[2].axvline(0, color='black', linewidth=1)
        axes[2].grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved robustness bars to {output_path}")


def create_induction_comparison(
    induction_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create bar plot comparing targeted vs random patch-out for induction heads.
    
    Args:
        induction_results: Results from induction case study
        output_path: Path to save figure
    """
    print("Creating induction comparison plot...")
    
    if "targeted_patchout" not in induction_results or "random_patchout" not in induction_results:
        print("Warning: No induction data available")
        return
    
    # Extract data
    baseline_acc = induction_results["baseline_accuracy"]
    targeted = induction_results["targeted_patchout"]
    random = induction_results["random_patchout"]
    
    k_values = sorted([int(k.split('_')[1]) for k in targeted.keys()])
    
    targeted_deltas = [targeted[f"k_{k}"]["accuracy"] - baseline_acc for k in k_values]
    random_deltas = [random[f"k_{k}"]["accuracy"] - baseline_acc for k in k_values]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(k_values))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, targeted_deltas, width, label='Targeted (Top REV heads)', 
                   alpha=0.8, edgecolor='black', color='steelblue')
    bars2 = ax.bar(x + width/2, random_deltas, width, label='Random heads', 
                   alpha=0.8, edgecolor='black', color='lightcoral')
    
    # Customization
    ax.set_xlabel('% of Heads Patched Out', fontweight='bold')
    ax.set_ylabel('ΔAccuracy (vs baseline)', fontweight='bold')
    ax.set_title('Induction Heads: Targeted vs Random Ablation', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}%' for k in k_values])
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved induction comparison to {output_path}")


def generate_all_paper_figures(
    phase5_results_dir: str = "reports/phase5",
    output_dir: str = "reports/figs_paper"
) -> None:
    """
    Generate all publication figures from Phase 5 results.
    
    Args:
        phase5_results_dir: Directory containing Phase 5 results
        output_dir: Output directory for figures
    """
    print("\n" + "="*80)
    print("Generating Publication Figures")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_files = {
        "model_results": "model_results.json",
        "baseline_comparison": "baseline_comparison.json",
        "patchout_results": "aggregate_patchout.json",
        "robustness": "robustness_summary.json",
        "induction": "induction_case_study.json"
    }
    
    loaded_results = {}
    for key, filename in results_files.items():
        filepath = os.path.join(phase5_results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_results[key] = json.load(f)
        else:
            print(f"Warning: {filepath} not found")
            loaded_results[key] = {}
    
    # Generate figures
    
    # 1. Scaling curve
    if loaded_results["model_results"]:
        create_scaling_curve(
            loaded_results["model_results"],
            os.path.join(output_dir, "scaling_curve.png")
        )
    
    # 2. ROC curves
    if loaded_results["baseline_comparison"]:
        create_roc_curves(
            loaded_results["baseline_comparison"],
            os.path.join(output_dir, "roc_curves.png")
        )
    
    # 3. Causal validation scatter
    if loaded_results["patchout_results"]:
        create_causal_scatter(
            loaded_results["patchout_results"],
            os.path.join(output_dir, "causal_scatter.png")
        )
    
    # 4. Robustness bars
    if loaded_results["robustness"]:
        create_robustness_bars(
            loaded_results["robustness"],
            os.path.join(output_dir, "robustness_bars.png")
        )
    
    # 5. Induction comparison
    if loaded_results["induction"]:
        create_induction_comparison(
            loaded_results["induction"],
            os.path.join(output_dir, "induction_comparison.png")
        )
    
    print("\n" + "="*80)
    print(f"✅ All figures saved to {output_dir}")
    print("="*80)


if __name__ == "__main__":
    import sys
    phase5_dir = sys.argv[1] if len(sys.argv) > 1 else "reports/phase5"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reports/figs_paper"
    generate_all_paper_figures(phase5_dir, output_dir)


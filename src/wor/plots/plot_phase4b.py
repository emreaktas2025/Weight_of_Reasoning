"""Phase 4b plotting utilities for causal mechanistic validation."""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from scipy.stats import spearmanr

from ..core.utils import ensure_dir


def plot_patchout_heads_delta(phase4b_dir: str, outdir: str = "reports/plots_phase4b") -> None:
    """
    Create patch-out effect plot for heads showing Δaccuracy and ΔREV vs K%.
    
    Args:
        phase4b_dir: Directory containing Phase 4b results
        outdir: Output directory for plots
    """
    ensure_dir(outdir)
    
    # Find head patch-out result files
    heads_files = []
    for filename in os.listdir(phase4b_dir):
        if filename.endswith('_patchout_heads.json'):
            heads_files.append(os.path.join(phase4b_dir, filename))
    
    if not heads_files:
        print("Warning: No head patch-out files found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Head Patch-Out Effects: Causal Validation', fontsize=14, fontweight='bold')
    
    k_percentages = [5, 10, 20]
    model_names = []
    delta_accuracies = []
    delta_revs = []
    
    for file_path in heads_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            model_name = os.path.basename(file_path).replace('_patchout_heads.json', '')
            model_names.append(model_name)
            
            model_delta_acc = []
            model_delta_rev = []
            
            for k in k_percentages:
                k_key = f"k_{k}"
                if k_key in data.get("patchout_results", {}):
                    model_delta_acc.append(data["patchout_results"][k_key]["delta_accuracy"])
                    model_delta_rev.append(data["patchout_results"][k_key]["delta_rev"])
                else:
                    model_delta_acc.append(0.0)
                    model_delta_rev.append(0.0)
            
            delta_accuracies.append(model_delta_acc)
            delta_revs.append(model_delta_rev)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Plot delta accuracy
    for i, (model_name, deltas) in enumerate(zip(model_names, delta_accuracies)):
        ax1.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('K% of Heads Patched')
    ax1.set_ylabel('Δ Accuracy')
    ax1.set_title('Accuracy Change vs Head Patch-Out')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot delta REV
    for i, (model_name, deltas) in enumerate(zip(model_names, delta_revs)):
        ax2.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), 
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('K% of Heads Patched')
    ax2.set_ylabel('Δ REV')
    ax2.set_title('REV Change vs Head Patch-Out')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'patchout_heads_delta.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Head patch-out effects plot saved to {outdir}/patchout_heads_delta.png")


def plot_patchout_layers_delta(phase4b_dir: str, outdir: str = "reports/plots_phase4b") -> None:
    """
    Create patch-out effect plot for layers showing Δaccuracy and ΔREV vs K%.
    
    Args:
        phase4b_dir: Directory containing Phase 4b results
        outdir: Output directory for plots
    """
    ensure_dir(outdir)
    
    # Find layer patch-out result files
    layers_files = []
    for filename in os.listdir(phase4b_dir):
        if filename.endswith('_patchout_layers.json'):
            layers_files.append(os.path.join(phase4b_dir, filename))
    
    if not layers_files:
        print("Warning: No layer patch-out files found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Layer Patch-Out Effects: Causal Validation', fontsize=14, fontweight='bold')
    
    k_percentages = [5, 10, 20]
    model_names = []
    delta_accuracies = []
    delta_revs = []
    
    for file_path in layers_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            model_name = os.path.basename(file_path).replace('_patchout_layers.json', '')
            model_names.append(model_name)
            
            model_delta_acc = []
            model_delta_rev = []
            
            for k in k_percentages:
                k_key = f"k_{k}"
                if k_key in data.get("patchout_results", {}):
                    model_delta_acc.append(data["patchout_results"][k_key]["delta_accuracy"])
                    model_delta_rev.append(data["patchout_results"][k_key]["delta_rev"])
                else:
                    model_delta_acc.append(0.0)
                    model_delta_rev.append(0.0)
            
            delta_accuracies.append(model_delta_acc)
            delta_revs.append(model_delta_rev)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Plot delta accuracy
    for i, (model_name, deltas) in enumerate(zip(model_names, delta_accuracies)):
        ax1.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('K% of Layers Patched')
    ax1.set_ylabel('Δ Accuracy')
    ax1.set_title('Accuracy Change vs Layer Patch-Out')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot delta REV
    for i, (model_name, deltas) in enumerate(zip(model_names, delta_revs)):
        ax2.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), 
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('K% of Layers Patched')
    ax2.set_ylabel('Δ REV')
    ax2.set_title('REV Change vs Layer Patch-Out')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'patchout_layers_delta.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Layer patch-out effects plot saved to {outdir}/patchout_layers_delta.png")


def plot_rev_accuracy_correlation(phase4b_dir: str, outdir: str = "reports/plots_phase4b") -> None:
    """
    Create scatter plot showing correlation between ΔREV and Δaccuracy.
    
    Args:
        phase4b_dir: Directory containing Phase 4b results
        outdir: Output directory for plots
    """
    ensure_dir(outdir)
    
    # Collect all ΔREV and Δaccuracy values
    delta_rev_values = []
    delta_acc_values = []
    model_labels = []
    experiment_labels = []
    
    # Process head patch-out results
    heads_files = [f for f in os.listdir(phase4b_dir) if f.endswith('_patchout_heads.json')]
    for file_path in heads_files:
        try:
            with open(os.path.join(phase4b_dir, file_path), 'r') as f:
                data = json.load(f)
            
            model_name = os.path.basename(file_path).replace('_patchout_heads.json', '')
            
            if "patchout_results" in data:
                for k_key, k_result in data["patchout_results"].items():
                    if "delta_rev" in k_result and "delta_accuracy" in k_result:
                        delta_rev_values.append(k_result["delta_rev"])
                        delta_acc_values.append(k_result["delta_accuracy"])
                        model_labels.append(model_name)
                        experiment_labels.append(f"heads_{k_key}")
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Process layer patch-out results
    layers_files = [f for f in os.listdir(phase4b_dir) if f.endswith('_patchout_layers.json')]
    for file_path in layers_files:
        try:
            with open(os.path.join(phase4b_dir, file_path), 'r') as f:
                data = json.load(f)
            
            model_name = os.path.basename(file_path).replace('_patchout_layers.json', '')
            
            if "patchout_results" in data:
                for k_key, k_result in data["patchout_results"].items():
                    if "delta_rev" in k_result and "delta_accuracy" in k_result:
                        delta_rev_values.append(k_result["delta_rev"])
                        delta_acc_values.append(k_result["delta_accuracy"])
                        model_labels.append(model_name)
                        experiment_labels.append(f"layers_{k_key}")
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(delta_rev_values) < 2:
        print("Warning: Insufficient data for correlation plot")
        return
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Create color map for models
    unique_models = list(set(model_labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_models)))
    model_colors = {model: colors[i] for i, model in enumerate(unique_models)}
    
    # Plot points
    for i, (model, exp) in enumerate(zip(model_labels, experiment_labels)):
        color = model_colors[model]
        marker = 'o' if 'heads' in exp else 's'
        plt.scatter(delta_rev_values[i], delta_acc_values[i], 
                   c=[color], marker=marker, s=100, alpha=0.7, 
                   label=model if i == model_labels.index(model) else "")
    
    # Compute correlation
    rho, p_value = spearmanr(delta_rev_values, delta_acc_values)
    
    # Add trend line
    if len(delta_rev_values) > 1:
        z = np.polyfit(delta_rev_values, delta_acc_values, 1)
        p_line = np.poly1d(z)
        x_trend = np.linspace(min(delta_rev_values), max(delta_rev_values), 100)
        plt.plot(x_trend, p_line(x_trend), "r--", alpha=0.8, 
                label=f'Trend (ρ={rho:.3f}, p={p_value:.3f})')
    
    plt.xlabel('Δ REV')
    plt.ylabel('Δ Accuracy')
    plt.title('Causal Correlation: ΔREV vs ΔAccuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add correlation text
    plt.text(0.05, 0.95, f'Spearman ρ = {rho:.4f}\np = {p_value:.4f}\nn = {len(delta_rev_values)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rev_accuracy_corr.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"REV-Accuracy correlation plot saved to {outdir}/rev_accuracy_corr.png")
    print(f"Causal correlation: ρ={rho:.4f}, p={p_value:.4f} (n={len(delta_rev_values)})")


def create_fallback_plot(outdir: str, plot_name: str, message: str) -> None:
    """Create a fallback plot with a message when data is insufficient."""
    ensure_dir(outdir)
    
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=16, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title(f'{plot_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    
    plot_path = os.path.join(outdir, f"{plot_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created fallback plot: {plot_path}")


def create_all_phase4b_plots(phase4b_dir: str = "reports/phase4b", outdir: str = "reports/plots_phase4b") -> None:
    """
    Create all Phase 4b causal validation plots with graceful fallbacks.
    
    Args:
        phase4b_dir: Directory containing Phase 4b results
        outdir: Output directory for plots
    """
    print(f"Creating Phase 4b causal validation plots...")
    
    if not os.path.exists(phase4b_dir):
        print(f"Error: Phase 4b directory {phase4b_dir} not found")
        return
    
    # Create plots with fallbacks
    heads_created = False
    layers_created = False
    corr_created = False
    
    try:
        plot_patchout_heads_delta(phase4b_dir, outdir)
        heads_created = True
    except Exception as e:
        print(f"Head patch-out plot failed: {e}")
        create_fallback_plot(outdir, "patchout_heads_delta", 
                           "Head patch-out data not available\nor insufficient for plotting")
        heads_created = True
    
    try:
        plot_patchout_layers_delta(phase4b_dir, outdir)
        layers_created = True
    except Exception as e:
        print(f"Layer patch-out plot failed: {e}")
        create_fallback_plot(outdir, "patchout_layers_delta", 
                           "Layer patch-out data not available\nor insufficient for plotting")
        layers_created = True
    
    try:
        plot_rev_accuracy_correlation(phase4b_dir, outdir)
        corr_created = True
    except Exception as e:
        print(f"Correlation plot failed: {e}")
        create_fallback_plot(outdir, "rev_accuracy_corr", 
                           "Causal correlation data not available\nor insufficient for plotting")
        corr_created = True
    
    print(f"\nAll Phase 4b plots created in {outdir}/")
    
    # List created files
    plot_files = [
        'patchout_heads_delta.png',
        'patchout_layers_delta.png',
        'rev_accuracy_corr.png'
    ]
    
    print("\nCreated plots:")
    for plot_file in plot_files:
        plot_path = os.path.join(outdir, plot_file)
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            print(f"  {plot_file} ({file_size:,} bytes)")
        else:
            print(f"  {plot_file} (NOT CREATED)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create Phase 4b causal validation plots")
    parser.add_argument("--phase4b_dir", default="reports/phase4b", 
                       help="Directory containing Phase 4b results")
    parser.add_argument("--outdir", default="reports/plots_phase4b", 
                       help="Output directory for plots")
    args = parser.parse_args()
    
    create_all_phase4b_plots(args.phase4b_dir, args.outdir)


if __name__ == "__main__":
    main()

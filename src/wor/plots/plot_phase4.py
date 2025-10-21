"""Phase 4 plotting utilities for scaling curves and patch-out figures."""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from scipy.stats import pearsonr

from ..core.utils import ensure_dir


def plot_scaling_curves(aggregate_data: Dict[str, Any], outdir: str = "reports/plots_phase4") -> None:
    """
    Create scaling curves showing how metrics change with model size.
    
    Args:
        aggregate_data: Aggregate scaling data from Phase 4
        outdir: Output directory for plots
    """
    ensure_dir(outdir)
    
    n_params = np.array(aggregate_data["n_params"])
    log_params = np.log(n_params)
    model_names = aggregate_data["model_names"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 4 Scaling Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Cohen's d vs parameters
    ax1 = axes[0, 0]
    d_rev = np.array(aggregate_data["d_REV"])
    valid_mask = np.isfinite(d_rev)
    
    if valid_mask.sum() > 0:
        ax1.scatter(log_params[valid_mask], d_rev[valid_mask], s=100, alpha=0.7)
        
        # Add trend line
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], d_rev[valid_mask])
            z = np.polyfit(log_params[valid_mask], d_rev[valid_mask], 1)
            p_line = np.poly1d(z)
            ax1.plot(log_params[valid_mask], p_line(log_params[valid_mask]), 
                    "r--", alpha=0.8, label=f'r={r:.3f}, p={p:.3f}')
            ax1.legend()
        
        # Add model labels
        for i, name in enumerate(model_names):
            if valid_mask[i]:
                ax1.annotate(name.replace('_', '-'), 
                           (log_params[i], d_rev[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Log(Parameters)')
    ax1.set_ylabel("Cohen's d (REV)")
    ax1.set_title('REV Effect Size vs Model Size')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUROC vs parameters
    ax2 = axes[0, 1]
    auroc_rev = np.array(aggregate_data["auroc_REV"])
    valid_mask = np.isfinite(auroc_rev)
    
    if valid_mask.sum() > 0:
        ax2.scatter(log_params[valid_mask], auroc_rev[valid_mask], s=100, alpha=0.7)
        
        # Add trend line
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], auroc_rev[valid_mask])
            z = np.polyfit(log_params[valid_mask], auroc_rev[valid_mask], 1)
            p_line = np.poly1d(z)
            ax2.plot(log_params[valid_mask], p_line(log_params[valid_mask]), 
                    "r--", alpha=0.8, label=f'r={r:.3f}, p={p:.3f}')
            ax2.legend()
        
        # Add model labels
        for i, name in enumerate(model_names):
            if valid_mask[i]:
                ax2.annotate(name.replace('_', '-'), 
                           (log_params[i], auroc_rev[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Log(Parameters)')
    ax2.set_ylabel('AUROC (REV)')
    ax2.set_title('REV Discriminability vs Model Size')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Partial correlation vs parameters
    ax3 = axes[1, 0]
    partial_r_rev = np.array(aggregate_data["partial_r_REV"])
    valid_mask = np.isfinite(partial_r_rev)
    
    if valid_mask.sum() > 0:
        ax3.scatter(log_params[valid_mask], partial_r_rev[valid_mask], s=100, alpha=0.7)
        
        # Add trend line
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], partial_r_rev[valid_mask])
            z = np.polyfit(log_params[valid_mask], partial_r_rev[valid_mask], 1)
            p_line = np.poly1d(z)
            ax3.plot(log_params[valid_mask], p_line(log_params[valid_mask]), 
                    "r--", alpha=0.8, label=f'r={r:.3f}, p={p:.3f}')
            ax3.legend()
        
        # Add model labels
        for i, name in enumerate(model_names):
            if valid_mask[i]:
                ax3.annotate(name.replace('_', '-'), 
                           (log_params[i], partial_r_rev[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Log(Parameters)')
    ax3.set_ylabel('Partial Correlation (REV)')
    ax3.set_title('REV Partial Correlation vs Model Size')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Runtime vs parameters
    ax4 = axes[1, 1]
    runtime_sec = np.array(aggregate_data["runtime_sec"])
    valid_mask = np.isfinite(runtime_sec)
    
    if valid_mask.sum() > 0:
        ax4.scatter(log_params[valid_mask], runtime_sec[valid_mask], s=100, alpha=0.7)
        
        # Add trend line
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], runtime_sec[valid_mask])
            z = np.polyfit(log_params[valid_mask], runtime_sec[valid_mask], 1)
            p_line = np.poly1d(z)
            ax4.plot(log_params[valid_mask], p_line(log_params[valid_mask]), 
                    "r--", alpha=0.8, label=f'r={r:.3f}, p={p:.3f}')
            ax4.legend()
        
        # Add model labels
        for i, name in enumerate(model_names):
            if valid_mask[i]:
                ax4.annotate(name.replace('_', '-'), 
                           (log_params[i], runtime_sec[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Log(Parameters)')
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Runtime vs Model Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save individual plots
    plt.savefig(os.path.join(outdir, 'scaling_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual scaling plots
    plot_individual_scaling_curves(aggregate_data, outdir)
    
    print(f"Scaling curves saved to {outdir}/")


def plot_individual_scaling_curves(aggregate_data: Dict[str, Any], outdir: str) -> None:
    """Create individual scaling curve plots."""
    n_params = np.array(aggregate_data["n_params"])
    log_params = np.log(n_params)
    model_names = aggregate_data["model_names"]
    
    # Cohen's d vs parameters
    plt.figure(figsize=(8, 6))
    d_rev = np.array(aggregate_data["d_REV"])
    valid_mask = np.isfinite(d_rev)
    
    if valid_mask.sum() > 0:
        plt.scatter(log_params[valid_mask], d_rev[valid_mask], s=100, alpha=0.7)
        
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], d_rev[valid_mask])
            z = np.polyfit(log_params[valid_mask], d_rev[valid_mask], 1)
            p_line = np.poly1d(z)
            plt.plot(log_params[valid_mask], p_line(log_params[valid_mask]), 
                    "r--", alpha=0.8, label=f'r={r:.3f}, p={p:.3f}')
            plt.legend()
        
        for i, name in enumerate(model_names):
            if valid_mask[i]:
                plt.annotate(name.replace('_', '-'), 
                           (log_params[i], d_rev[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Log(Parameters)')
    plt.ylabel("Cohen's d (REV)")
    plt.title('REV Effect Size vs Model Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'd_REV_vs_params.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # AUROC vs parameters
    plt.figure(figsize=(8, 6))
    auroc_rev = np.array(aggregate_data["auroc_REV"])
    valid_mask = np.isfinite(auroc_rev)
    
    if valid_mask.sum() > 0:
        plt.scatter(log_params[valid_mask], auroc_rev[valid_mask], s=100, alpha=0.7)
        
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], auroc_rev[valid_mask])
            z = np.polyfit(log_params[valid_mask], auroc_rev[valid_mask], 1)
            p_line = np.poly1d(z)
            plt.plot(log_params[valid_mask], p_line(log_params[valid_mask]), 
                    "r--", alpha=0.8, label=f'r={r:.3f}, p={p:.3f}')
            plt.legend()
        
        for i, name in enumerate(model_names):
            if valid_mask[i]:
                plt.annotate(name.replace('_', '-'), 
                           (log_params[i], auroc_rev[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Log(Parameters)')
    plt.ylabel('AUROC (REV)')
    plt.title('REV Discriminability vs Model Size')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'AUROC_REV_vs_params.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Partial correlation vs parameters
    plt.figure(figsize=(8, 6))
    partial_r_rev = np.array(aggregate_data["partial_r_REV"])
    valid_mask = np.isfinite(partial_r_rev)
    
    if valid_mask.sum() > 0:
        plt.scatter(log_params[valid_mask], partial_r_rev[valid_mask], s=100, alpha=0.7)
        
        if valid_mask.sum() > 1:
            r, p = pearsonr(log_params[valid_mask], partial_r_rev[valid_mask])
            z = np.polyfit(log_params[valid_mask], partial_r_rev[valid_mask], 1)
            p_line = np.poly1d(z)
            plt.plot(log_params[valid_mask], p_line(log_params[valid_mask]), 
                    "r--", alpha=0.8, label=f'r={r:.3f}, p={p:.3f}')
            plt.legend()
        
        for i, name in enumerate(model_names):
            if valid_mask[i]:
                plt.annotate(name.replace('_', '-'), 
                           (log_params[i], partial_r_rev[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Log(Parameters)')
    plt.ylabel('Partial Correlation (REV)')
    plt.title('REV Partial Correlation vs Model Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'r_partial_REV_vs_params.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_patchout_effects(phase4_dir: str, outdir: str = "reports/plots_phase4") -> None:
    """
    Create patch-out effect plots showing accuracy and REV changes.
    
    Args:
        phase4_dir: Directory containing Phase 4 results
        outdir: Output directory for plots
    """
    ensure_dir(outdir)
    
    # Find patch-out result files
    heads_files = []
    layers_files = []
    
    for filename in os.listdir(phase4_dir):
        if filename.endswith('_patchout_heads.json'):
            heads_files.append(os.path.join(phase4_dir, filename))
        elif filename.endswith('_patchout_layers.json'):
            layers_files.append(os.path.join(phase4_dir, filename))
    
    if heads_files:
        plot_heads_patchout_effects(heads_files, outdir)
    
    if layers_files:
        plot_layers_patchout_effects(layers_files, outdir)


def plot_heads_patchout_effects(heads_files: List[str], outdir: str) -> None:
    """Plot head patch-out effects."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Head Patch-Out Effects', fontsize=14, fontweight='bold')
    
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
                if k_key in data["patchout_results"]:
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
        ax1.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), linewidth=2, markersize=6)
    
    ax1.set_xlabel('K% of Heads Patched')
    ax1.set_ylabel('Δ Accuracy')
    ax1.set_title('Accuracy Change vs Head Patch-Out')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot delta REV
    for i, (model_name, deltas) in enumerate(zip(model_names, delta_revs)):
        ax2.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), linewidth=2, markersize=6)
    
    ax2.set_xlabel('K% of Heads Patched')
    ax2.set_ylabel('Δ REV')
    ax2.set_title('REV Change vs Head Patch-Out')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'patchout_heads_delta.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_layers_patchout_effects(layers_files: List[str], outdir: str) -> None:
    """Plot layer patch-out effects."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Layer Patch-Out Effects', fontsize=14, fontweight='bold')
    
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
                if k_key in data["patchout_results"]:
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
        ax1.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), linewidth=2, markersize=6)
    
    ax1.set_xlabel('K% of Layers Patched')
    ax1.set_ylabel('Δ Accuracy')
    ax1.set_title('Accuracy Change vs Layer Patch-Out')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot delta REV
    for i, (model_name, deltas) in enumerate(zip(model_names, delta_revs)):
        ax2.plot(k_percentages, deltas, 'o-', label=model_name.replace('_', '-'), linewidth=2, markersize=6)
    
    ax2.set_xlabel('K% of Layers Patched')
    ax2.set_ylabel('Δ REV')
    ax2.set_title('REV Change vs Layer Patch-Out')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'patchout_layers_delta.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_rev_violin_by_model(phase4_dir: str, outdir: str = "reports/plots_phase4") -> None:
    """
    Create violin plot showing REV distributions by model.
    
    Args:
        phase4_dir: Directory containing Phase 4 results
        outdir: Output directory for plots
    """
    ensure_dir(outdir)
    
    # Collect data from all model CSV files
    all_data = []
    
    for filename in os.listdir(phase4_dir):
        if filename.endswith('_metrics.csv'):
            model_name = filename.replace('_metrics.csv', '')
            csv_path = os.path.join(phase4_dir, filename)
            
            try:
                df = pd.read_csv(csv_path)
                df['model'] = model_name
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                continue
    
    if not all_data:
        print("No model data found for violin plot")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create violin plot
    plt.figure(figsize=(12, 8))
    
    # Filter out NaN values
    df_clean = combined_df.dropna(subset=['REV', 'label', 'model'])
    
    if len(df_clean) == 0:
        print("Warning: No valid REV data for violin plot")
        return
    
    # Create violin plot with model and label facets
    sns.violinplot(data=df_clean, x='model', y='REV', hue='label', 
                   palette=['lightblue', 'lightcoral'], split=True)
    
    plt.title('REV Distribution by Model and Label', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('REV Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rev_violin_by_model.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"REV violin plot by model saved to {outdir}/rev_violin_by_model.png")


def create_all_phase4_plots(aggregate_path: str, phase4_dir: str, outdir: str = "reports/plots_phase4") -> None:
    """
    Create all Phase 4 plots.
    
    Args:
        aggregate_path: Path to aggregate scaling JSON
        phase4_dir: Directory containing Phase 4 results
        outdir: Output directory for plots
    """
    print(f"Creating Phase 4 plots...")
    
    # Load aggregate data
    try:
        with open(aggregate_path, 'r') as f:
            aggregate_data = json.load(f)
    except Exception as e:
        print(f"Error loading aggregate data from {aggregate_path}: {e}")
        return
    
    # Create scaling curves
    plot_scaling_curves(aggregate_data, outdir)
    
    # Create patch-out effect plots
    plot_patchout_effects(phase4_dir, outdir)
    
    # Create REV violin plot by model
    plot_rev_violin_by_model(phase4_dir, outdir)
    
    print(f"\nAll Phase 4 plots created in {outdir}/")
    
    # List created files
    plot_files = [
        'scaling_curves.png',
        'd_REV_vs_params.png',
        'AUROC_REV_vs_params.png', 
        'r_partial_REV_vs_params.png',
        'patchout_heads_delta.png',
        'patchout_layers_delta.png',
        'rev_violin_by_model.png'
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
    parser = argparse.ArgumentParser(description="Create Phase 4 plots")
    parser.add_argument("--aggregate", required=True, help="Path to aggregate scaling JSON file")
    parser.add_argument("--phase4_dir", default="reports/phase4", help="Directory containing Phase 4 results")
    parser.add_argument("--outdir", default="reports/plots_phase4", help="Output directory for plots")
    args = parser.parse_args()
    
    create_all_phase4_plots(args.aggregate, args.phase4_dir, args.outdir)


if __name__ == "__main__":
    main()

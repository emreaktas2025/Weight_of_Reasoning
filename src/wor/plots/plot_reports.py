"""Generate plots and visualizations from evaluation results."""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def create_ae_violin_plot(df: pd.DataFrame, outdir: str) -> None:
    """Create violin plot for Activation Energy (AE) by label."""
    plt.figure(figsize=(8, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['AE'])
    
    if len(df_clean) == 0:
        print("Warning: No valid AE data to plot")
        return
    
    # Create box plot
    labels = df_clean['label'].unique()
    data_by_label = [df_clean[df_clean['label'] == label]['AE'].values for label in labels]
    
    plt.boxplot(data_by_label, labels=labels)
    plt.title("Activation Energy by Label")
    plt.ylabel("AE")
    plt.xlabel("Label")
    plt.grid(True, alpha=0.3)
    
    # Add sample size annotations
    for i, label in enumerate(labels):
        n_samples = len(df_clean[df_clean['label'] == label])
        plt.text(i + 1, plt.ylim()[1] * 0.95, f'n={n_samples}', 
                ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ae_violin.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AE violin plot to {os.path.join(outdir, 'ae_violin.png')}")


def create_ape_violin_plot(df: pd.DataFrame, outdir: str) -> None:
    """Create violin plot for Attention Process Entropy (APE) by label."""
    plt.figure(figsize=(8, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['APE'])
    
    if len(df_clean) == 0:
        print("Warning: No valid APE data to plot")
        return
    
    # Create box plot
    labels = df_clean['label'].unique()
    data_by_label = [df_clean[df_clean['label'] == label]['APE'].values for label in labels]
    
    plt.boxplot(data_by_label, labels=labels)
    plt.title("Attention Process Entropy by Label")
    plt.ylabel("APE")
    plt.xlabel("Label")
    plt.grid(True, alpha=0.3)
    
    # Add sample size annotations
    for i, label in enumerate(labels):
        n_samples = len(df_clean[df_clean['label'] == label])
        plt.text(i + 1, plt.ylim()[1] * 0.95, f'n={n_samples}', 
                ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ape_violin.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved APE violin plot to {os.path.join(outdir, 'ape_violin.png')}")


def create_ape_trajectory_plot(df: pd.DataFrame, outdir: str) -> None:
    """Create trajectory plot for APE over token length."""
    plt.figure(figsize=(10, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['APE', 'token_len'])
    
    if len(df_clean) == 0:
        print("Warning: No valid APE trajectory data to plot")
        return
    
    # Plot by label
    for label in df_clean['label'].unique():
        label_data = df_clean[df_clean['label'] == label]
        plt.scatter(label_data['token_len'], label_data['APE'], 
                   label=label, alpha=0.7, s=50)
    
    plt.xlabel("Token Length")
    plt.ylabel("APE")
    plt.title("Attention Process Entropy vs Token Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ape_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved APE trajectory plot to {os.path.join(outdir, 'ape_trajectory.png')}")


def create_apl_violin_plot(df: pd.DataFrame, outdir: str) -> None:
    """Create violin plot for Activation Path Length (APL) by label."""
    plt.figure(figsize=(8, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['APL'])
    
    if len(df_clean) == 0:
        print("Warning: No valid APL data to plot")
        return
    
    # Create box plot
    labels = df_clean['label'].unique()
    data_by_label = [df_clean[df_clean['label'] == label]['APL'].values for label in labels]
    
    plt.boxplot(data_by_label, labels=labels)
    plt.title("Activation Path Length by Label")
    plt.ylabel("APL")
    plt.xlabel("Label")
    plt.grid(True, alpha=0.3)
    
    # Add sample size annotations
    for i, label in enumerate(labels):
        n_samples = len(df_clean[df_clean['label'] == label])
        plt.text(i + 1, plt.ylim()[1] * 0.95, f'n={n_samples}', 
                ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "apl_violin.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved APL violin plot to {os.path.join(outdir, 'apl_violin.png')}")


def create_ae_apl_scatter(df: pd.DataFrame, outdir: str) -> None:
    """Create scatter plot for AE vs APL relationship."""
    plt.figure(figsize=(10, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['AE', 'APL'])
    
    if len(df_clean) == 0:
        print("Warning: No valid AE/APL data to plot")
        return
    
    # Plot by label
    for label in df_clean['label'].unique():
        label_data = df_clean[df_clean['label'] == label]
        plt.scatter(label_data['AE'], label_data['APL'], 
                   label=label, alpha=0.7, s=50)
    
    plt.xlabel("Activation Energy (AE)")
    plt.ylabel("Activation Path Length (APL)")
    plt.title("AE vs APL Relationship")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ae_apl_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AE vs APL scatter plot to {os.path.join(outdir, 'ae_apl_scatter.png')}")


def main():
    """Main entry point for plotting."""
    parser = argparse.ArgumentParser(description="Generate plots from evaluation results")
    parser.add_argument("--report_csv", required=True, help="Path to metrics CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(args.report_csv)
        print(f"Loaded {len(df)} rows from {args.report_csv}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Generate plots
    create_ae_violin_plot(df, args.outdir)
    create_ape_violin_plot(df, args.outdir)
    create_ape_trajectory_plot(df, args.outdir)
    create_apl_violin_plot(df, args.outdir)
    create_ae_apl_scatter(df, args.outdir)
    
    print(f"All plots saved to {args.outdir}")


if __name__ == "__main__":
    main()

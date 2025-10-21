"""Phase 3 plotting utilities for CUD, SIB, FL, and REV metrics."""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from ..core.utils import ensure_dir


def plot_rev_violin(df: pd.DataFrame, outdir: str = "reports/plots") -> None:
    """
    Create violin plot for REV distribution by label.
    
    Args:
        df: DataFrame with REV and label columns
        outdir: Output directory for plots
    """
    plt.figure(figsize=(8, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['REV', 'label'])
    
    if len(df_clean) == 0:
        print("Warning: No valid REV data for violin plot")
        return
    
    # Create violin plot
    sns.violinplot(data=df_clean, x='label', y='REV', palette=['lightblue', 'lightcoral'])
    plt.title('REV Distribution by Label', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('REV Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add sample counts
    reasoning_count = len(df_clean[df_clean['label'] == 'reasoning'])
    control_count = len(df_clean[df_clean['label'] == 'control'])
    plt.text(0, plt.ylim()[1] * 0.9, f'n={reasoning_count}', ha='center', fontsize=10)
    plt.text(1, plt.ylim()[1] * 0.9, f'n={control_count}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, 'rev_violin.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"REV violin plot saved to {outdir}/rev_violin.png")


def plot_metric_correlation_heatmap(df: pd.DataFrame, outdir: str = "reports/plots") -> None:
    """
    Create correlation heatmap for all metrics.
    
    Args:
        df: DataFrame with metric columns
        outdir: Output directory for plots
    """
    plt.figure(figsize=(10, 8))
    
    # Select metric columns
    metrics = ['AE', 'APE', 'APL', 'CUD', 'SIB', 'FL', 'REV']
    metric_data = df[metrics].dropna()
    
    if len(metric_data) == 0:
        print("Warning: No valid metric data for correlation heatmap")
        return
    
    # Compute correlation matrix
    corr_matrix = metric_data.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8})
    
    plt.title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, 'metric_corr_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metric correlation heatmap saved to {outdir}/metric_corr_heatmap.png")


def plot_cud_boxplot(df: pd.DataFrame, outdir: str = "reports/plots") -> None:
    """
    Create boxplot for CUD by label.
    
    Args:
        df: DataFrame with CUD and label columns
        outdir: Output directory for plots
    """
    plt.figure(figsize=(8, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['CUD', 'label'])
    
    if len(df_clean) == 0:
        print("Warning: No valid CUD data for boxplot")
        return
    
    # Create boxplot
    sns.boxplot(data=df_clean, x='label', y='CUD', palette=['lightblue', 'lightcoral'])
    plt.title('CUD Distribution by Label', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('CUD Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add sample counts
    reasoning_count = len(df_clean[df_clean['label'] == 'reasoning'])
    control_count = len(df_clean[df_clean['label'] == 'control'])
    plt.text(0, plt.ylim()[1] * 0.9, f'n={reasoning_count}', ha='center', fontsize=10)
    plt.text(1, plt.ylim()[1] * 0.9, f'n={control_count}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, 'cud_box.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CUD boxplot saved to {outdir}/cud_box.png")


def plot_sib_boxplot(df: pd.DataFrame, outdir: str = "reports/plots") -> None:
    """
    Create boxplot for SIB by label.
    
    Args:
        df: DataFrame with SIB and label columns
        outdir: Output directory for plots
    """
    plt.figure(figsize=(8, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['SIB', 'label'])
    
    if len(df_clean) == 0:
        print("Warning: No valid SIB data for boxplot")
        return
    
    # Create boxplot
    sns.boxplot(data=df_clean, x='label', y='SIB', palette=['lightblue', 'lightcoral'])
    plt.title('SIB Distribution by Label', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('SIB Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add sample counts
    reasoning_count = len(df_clean[df_clean['label'] == 'reasoning'])
    control_count = len(df_clean[df_clean['label'] == 'control'])
    plt.text(0, plt.ylim()[1] * 0.9, f'n={reasoning_count}', ha='center', fontsize=10)
    plt.text(1, plt.ylim()[1] * 0.9, f'n={control_count}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, 'sib_box.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SIB boxplot saved to {outdir}/sib_box.png")


def plot_fl_boxplot(df: pd.DataFrame, outdir: str = "reports/plots") -> None:
    """
    Create boxplot for FL by label.
    
    Args:
        df: DataFrame with FL and label columns
        outdir: Output directory for plots
    """
    plt.figure(figsize=(8, 6))
    
    # Filter out NaN values
    df_clean = df.dropna(subset=['FL', 'label'])
    
    if len(df_clean) == 0:
        print("Warning: No valid FL data for boxplot")
        return
    
    # Create boxplot
    sns.boxplot(data=df_clean, x='label', y='FL', palette=['lightblue', 'lightcoral'])
    plt.title('FL Distribution by Label', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('FL Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add sample counts
    reasoning_count = len(df_clean[df_clean['label'] == 'reasoning'])
    control_count = len(df_clean[df_clean['label'] == 'control'])
    plt.text(0, plt.ylim()[1] * 0.9, f'n={reasoning_count}', ha='center', fontsize=10)
    plt.text(1, plt.ylim()[1] * 0.9, f'n={control_count}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, 'fl_box.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FL boxplot saved to {outdir}/fl_box.png")


def create_all_phase3_plots(report_csv: str, outdir: str = "reports/plots") -> None:
    """
    Create all Phase 3 plots.
    
    Args:
        report_csv: Path to metrics CSV file
        outdir: Output directory for plots
    """
    print(f"Loading data from {report_csv}...")
    
    # Load data
    df = pd.read_csv(report_csv)
    
    if len(df) == 0:
        print("Error: No data found in CSV file")
        return
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Create all plots
    print("\nCreating Phase 3 plots...")
    
    plot_rev_violin(df, outdir)
    plot_metric_correlation_heatmap(df, outdir)
    plot_cud_boxplot(df, outdir)
    plot_sib_boxplot(df, outdir)
    plot_fl_boxplot(df, outdir)
    
    print(f"\nAll Phase 3 plots created in {outdir}/")
    
    # List created files
    plot_files = [
        'rev_violin.png',
        'metric_corr_heatmap.png', 
        'cud_box.png',
        'sib_box.png',
        'fl_box.png'
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
    parser = argparse.ArgumentParser(description="Create Phase 3 plots")
    parser.add_argument("--report_csv", required=True, help="Path to metrics CSV file")
    parser.add_argument("--outdir", default="reports/plots", help="Output directory for plots")
    args = parser.parse_args()
    
    create_all_phase3_plots(args.report_csv, args.outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# Phase 3 evaluation with CUD, SIB, FL, and REV
uv run python -m wor.eval.evaluate_phase3 --config configs/eval.phase3.yaml
uv run python -m wor.plots.plot_phase3 --report_csv reports/metrics_phase3.csv --outdir reports/plots

echo "Phase 3 evaluation complete."
echo "Results: reports/metrics_phase3.csv, reports/summary_phase3.json"
echo "Plots: reports/plots/rev_violin.png, reports/plots/metric_corr_heatmap.png, etc."

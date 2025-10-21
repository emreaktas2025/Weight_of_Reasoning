#!/usr/bin/env bash
set -euo pipefail

# Phase 2 evaluation with APL
uv run python -m wor.eval.evaluate --config configs/eval.phase2.yaml
uv run python -m wor.plots.plot_reports --report_csv reports/metrics_phase2.csv --outdir reports/plots

echo "Phase 2 tiny eval complete."
echo "Results: reports/metrics_phase2.csv, reports/summary_phase2.json, reports/partial_corr.json"

#!/usr/bin/env bash
set -euo pipefail

# Phase 4: Scaling Study & Mechanistic Validation
# This script runs the complete Phase 4 evaluation pipeline

echo "🚀 Starting Phase 4: Scaling Study & Mechanistic Validation"
echo "=========================================================="

# Check if config file exists
CONFIG_FILE="configs/eval.phase4.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE not found"
    exit 1
fi

echo "📋 Using config: $CONFIG_FILE"

# Run Phase 4 evaluation
echo ""
echo "🔬 Running Phase 4 evaluation pipeline..."
uv run python -m wor.eval.evaluate_phase4 --config "$CONFIG_FILE"

# Check if evaluation completed successfully
if [ $? -ne 0 ]; then
    echo "❌ Phase 4 evaluation failed"
    exit 1
fi

echo ""
echo "📊 Creating Phase 4 plots..."

# Create plots
AGGREGATE_FILE="reports/phase4/aggregate_scaling.json"
if [ -f "$AGGREGATE_FILE" ]; then
    uv run python -m wor.plots.plot_phase4 --aggregate "$AGGREGATE_FILE"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Plot generation failed, but continuing..."
    fi
else
    echo "⚠️  Warning: Aggregate file $AGGREGATE_FILE not found, skipping plots"
fi

echo ""
echo "🧪 Running acceptance tests..."

# Run tests
uv run python tests/test_phase4_scaling_patchout.py

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some tests failed, but continuing..."
fi

echo ""
echo "✅ Phase 4 scaling + patch-out complete!"
echo ""
echo "📁 Output locations:"
echo "   - reports/phase4/          (per-model results)"
echo "   - reports/plots_phase4/    (scaling curves & patch-out figures)"
echo "   - reports/hw_phase4.json   (hardware configuration)"
echo "   - reports/splits/          (dataset manifests)"
echo ""
echo "📈 Check reports/phase4/aggregate_scaling.json for runtime and scaling trends"

#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Phase 4b: Llama Integration & Mechanistic Validation"
echo "======================================================="

# Check for HuggingFace token
if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "⚠️  Warning: No Hugging Face token set. Llama-3.2-1B may be skipped."
    echo "   To enable access, set either:"
    echo "   export HUGGINGFACE_HUB_TOKEN='your_token_here'"
    echo "   or"
    echo "   export HUGGINGFACE_TOKEN='your_token_here'"
fi

# Check if config file exists
CONFIG_FILE="configs/eval.phase4b.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE not found"
    exit 1
fi

echo "📋 Using config: $CONFIG_FILE"

# Run Phase 4b evaluation with patch-out experiments
echo ""
echo "🔬 Running mechanistic validation..."
uv run python -m wor.eval.evaluate_phase4b --config "$CONFIG_FILE"

# Check if evaluation completed successfully
if [ $? -ne 0 ]; then
    echo "❌ Phase 4b evaluation failed"
    exit 1
fi

echo ""
echo "📊 Creating causality plots..."

# Generate causal plots
AGGREGATE_FILE="reports/phase4b/aggregate_patchout.json"
if [ -f "$AGGREGATE_FILE" ]; then
    uv run python -m wor.plots.plot_phase4b
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Plot generation failed, but continuing..."
    fi
else
    echo "⚠️  Warning: Aggregate file $AGGREGATE_FILE not found, skipping plots"
fi

echo ""
echo "🧪 Running validation tests..."

# Run tests
uv run python tests/test_phase4b_patchout.py

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some tests failed, but continuing..."
fi

echo ""
echo "✅ Phase 4b complete!"
echo ""
echo "📁 Output locations:"
echo "   - reports/phase4b/          (baseline + patch-out results)"
echo "   - reports/plots_phase4b/    (causal validation plots)"
echo "   - reports/hw_phase4b.json   (hardware configuration)"
echo "   - reports/splits/           (dataset manifests)"
echo ""
echo "🔬 Check reports/phase4b/aggregate_patchout.json for causal correlations"

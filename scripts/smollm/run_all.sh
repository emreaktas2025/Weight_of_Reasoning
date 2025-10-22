#!/bin/bash

# SmolLM REV Research Pipeline - Main Execution Script
# Runs all 3 SmolLM models sequentially with full evaluation

set -e  # Exit on any error

echo "=========================================="
echo "SmolLM REV Research Pipeline"
echo "=========================================="

# Configuration
CONFIG_FILE="configs/smollm/eval.default.yaml"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/smollm_run_${TIMESTAMP}.log"

echo "Starting SmolLM evaluation pipeline..."
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Check if Python environment is set up
if ! command -v python &> /dev/null; then
    echo "Error: Python not found in PATH"
    exit 1
fi

# Run the evaluation
echo "Executing: python -m src.wor.smollm.eval.run_eval --config $CONFIG_FILE"
echo ""

python -m src.wor.smollm.eval.run_eval --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SmolLM evaluation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: experiments/"
    echo "Log saved to: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Check results in experiments/iteration_XXX/"
    echo "2. Review notes.md for observations"
    echo "3. Run visualization: python -m src.wor.smollm.eval.viz"
    echo "4. Generate summary: bash scripts/smollm/summarize_iteration.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ SmolLM evaluation failed!"
    echo "=========================================="
    echo ""
    echo "Check log file: $LOG_FILE"
    echo "Common issues:"
    echo "- GPU memory insufficient (try debug config)"
    echo "- HuggingFace authentication required"
    echo "- Missing dependencies"
    echo ""
    exit 1
fi

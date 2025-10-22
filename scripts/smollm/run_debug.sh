#!/bin/bash

# SmolLM REV Research Pipeline - Debug Execution Script
# Quick test run with minimal samples for validation

set -e  # Exit on any error

echo "=========================================="
echo "SmolLM REV Debug Pipeline"
echo "=========================================="

# Configuration
CONFIG_FILE="configs/smollm/eval.debug.yaml"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/smollm_debug_${TIMESTAMP}.log"

echo "Starting SmolLM debug evaluation..."
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Run the debug evaluation
echo "Executing: python -m src.wor.smollm.eval.run_eval --config $CONFIG_FILE"
echo ""

python -m src.wor.smollm.eval.run_eval --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SmolLM debug evaluation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Debug results saved to: experiments/"
    echo "Log saved to: $LOG_FILE"
    echo ""
    echo "This was a quick test run. For full evaluation, use:"
    echo "bash scripts/smollm/run_all.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ SmolLM debug evaluation failed!"
    echo "=========================================="
    echo ""
    echo "Check log file: $LOG_FILE"
    echo "Debug issues before running full pipeline."
    echo ""
    exit 1
fi

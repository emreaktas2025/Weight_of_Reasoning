#!/bin/bash

# SmolLM REV Research Pipeline - Iteration Summary Script
# Aggregates results and updates iteration registry

set -e  # Exit on any error

echo "=========================================="
echo "SmolLM Iteration Summary"
echo "=========================================="

# Get latest iteration
LATEST_ITERATION=$(ls -t experiments/iteration_* 2>/dev/null | head -1 | xargs basename)

if [ -z "$LATEST_ITERATION" ]; then
    echo "Error: No iterations found in experiments/"
    exit 1
fi

echo "Processing iteration: $LATEST_ITERATION"

# Check if results exist
RESULTS_DIR="experiments/$LATEST_ITERATION/results"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Run aggregation script
echo "Aggregating results and updating registry..."

python -c "
import sys
import os
sys.path.append('src')

from wor.smollm.eval.aggregate import compute_aggregate_statistics
from wor.smollm.utils.io_utils import load_results_json, update_iteration_registry
import json

# Load model results
results_path = 'experiments/$LATEST_ITERATION/results/model_results.json'
if os.path.exists(results_path):
    model_results = load_results_json(results_path)
    
    # Compute aggregate statistics
    aggregate_stats = compute_aggregate_statistics(model_results)
    
    # Save aggregate results
    aggregate_path = 'experiments/$LATEST_ITERATION/results/aggregate_statistics.json'
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    # Update iteration registry
    update_iteration_registry('$LATEST_ITERATION', aggregate_stats, 'experiments/$LATEST_ITERATION')
    
    print('✅ Aggregation completed successfully!')
    print(f'ΔAUC mean: {aggregate_stats.get(\"delta_auc_mean\", \"N/A\")}')
    print(f'Success criteria: {aggregate_stats.get(\"success_criteria\", {})}')
else:
    print('Error: Model results not found')
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "✅ Iteration summary completed!"
echo "=========================================="
echo ""
echo "Updated files:"
echo "- experiments/$LATEST_ITERATION/results/aggregate_statistics.json"
echo "- experiments/iteration_registry.csv"
echo ""
echo "Check results in: experiments/$LATEST_ITERATION/"

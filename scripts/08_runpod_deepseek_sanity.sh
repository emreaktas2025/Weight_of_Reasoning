#!/usr/bin/env bash
# RunPod script for DeepSeek sanity check with GPU

set -euo pipefail

echo "🚀 RunPod DeepSeek Sanity Check"
echo "=================================="

# Detect if we're on RunPod
if [ -d "/workspace" ]; then
    echo "✅ Detected RunPod environment"
    WORKSPACE="/workspace/Weight_of_Reasoning"
    cd "$WORKSPACE" || exit 1
else
    echo "⚠️  Not on RunPod, using current directory"
    WORKSPACE="$PWD"
fi

# Check GPU
echo ""
echo "Checking GPU..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  No GPU detected!')
    exit(1)
" || {
    echo "❌ GPU check failed"
    exit 1
}

# Install bitsandbytes if needed (for 4-bit quantization)
echo ""
echo "Checking bitsandbytes..."
python3 -c "import bitsandbytes" 2>/dev/null || {
    echo "Installing bitsandbytes..."
    pip install bitsandbytes
}

# Run sanity check
echo ""
echo "Running DeepSeek sanity check..."
python3 scripts/07_deepseek_sanity.py

echo ""
echo "✅ Sanity check complete!"


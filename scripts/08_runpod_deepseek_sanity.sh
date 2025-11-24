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

# Check GPU hardware first
echo ""
echo "Checking GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi failed"
else
    echo "❌ nvidia-smi not found"
    echo ""
    echo "This means:"
    echo "  1. No GPU is attached to this RunPod pod, OR"
    echo "  2. The container doesn't have NVIDIA drivers"
    echo ""
    echo "To fix this:"
    echo "  1. Go to RunPod dashboard: https://www.runpod.io/console/pods"
    echo "  2. Find your pod and check if it has a GPU attached"
    echo "  3. If no GPU: Stop pod → Edit → Add GPU → Start"
    echo "  4. Or create a new pod with GPU support"
    echo ""
    echo "For DeepSeek-8B, you need at least:"
    echo "  - RTX 3090 (24GB) with 4-bit quantization, OR"
    echo "  - A100 (40GB) for full precision"
    echo ""
fi

# Check PyTorch CUDA support
echo ""
echo "Checking PyTorch CUDA support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  PyTorch CUDA not available')
    print('This could mean:')
    print('  1. No GPU attached to this pod')
    print('  2. PyTorch installed without CUDA support')
    print('  3. Need to install PyTorch with CUDA')
" || {
    echo "⚠️  PyTorch check failed"
}

# Check if we should continue
echo ""
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ GPU detected and available!"
    USE_GPU=true
else
    echo "⚠️  No GPU available. Options:"
    echo "  1. Continue on CPU (will be very slow for DeepSeek-8B)"
    echo "  2. Install PyTorch with CUDA support"
    echo "  3. Check RunPod dashboard to ensure GPU is attached"
    echo ""
    read -p "Continue on CPU anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please:"
        echo "  1. Check RunPod dashboard - ensure GPU is attached to pod"
        echo "  2. Or install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        exit 1
    fi
    USE_GPU=false
fi

# Install bitsandbytes if needed (for 4-bit quantization) - only if GPU available
if [ "$USE_GPU" = true ]; then
    echo ""
    echo "Checking bitsandbytes..."
    python3 -c "import bitsandbytes" 2>/dev/null || {
        echo "Installing bitsandbytes..."
        pip install bitsandbytes
    }
else
    echo ""
    echo "⚠️  Skipping bitsandbytes (no GPU available)"
fi

# Run sanity check
echo ""
echo "Running DeepSeek sanity check..."
python3 scripts/07_deepseek_sanity.py

echo ""
echo "✅ Sanity check complete!"


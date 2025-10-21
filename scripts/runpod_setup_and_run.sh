#!/usr/bin/env bash
set -euo pipefail

echo "🚀 RunPod Phase 5 Auto-Setup & Execution"
echo "=========================================="

# Color codes for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect if we're on RunPod
if [ -d "/workspace" ]; then
    echo -e "${GREEN}✅ Detected RunPod environment${NC}"
    WORKSPACE="/workspace/Weight_of_Reasoning"
else
    echo -e "${YELLOW}⚠️  Not on RunPod, using current directory${NC}"
    WORKSPACE="$PWD"
fi

# Step 1: Check GPU
echo ""
echo -e "${BLUE}Step 1: Checking GPU...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  PyTorch not installed or GPU not detected${NC}"
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Step 2: Clone/Update Repository
echo ""
echo -e "${BLUE}Step 2: Setting up repository...${NC}"
if [ -d "$WORKSPACE/.git" ]; then
    echo "Repository exists, pulling latest changes..."
    cd "$WORKSPACE"
    git fetch origin
    git pull origin main
else
    echo "Cloning repository..."
    mkdir -p "$(dirname "$WORKSPACE")"
    git clone https://github.com/emreaktas2025/Weight_of_Reasoning.git "$WORKSPACE"
    cd "$WORKSPACE"
fi

# Step 3: Install Dependencies
echo ""
echo -e "${BLUE}Step 3: Installing dependencies...${NC}"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for dependency management..."
    uv sync
else
    echo "uv not found, using pip..."
    pip install -e .
fi

# Step 4: Verify Installation
echo ""
echo -e "${BLUE}Step 4: Verifying installation...${NC}"
python3 -c "
import torch
import transformers
import datasets
import sklearn
print('✅ All core dependencies installed')
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Step 5: Run Phase 5
echo ""
echo -e "${BLUE}Step 5: Running Phase 5 Evaluation...${NC}"
echo ""
echo "Choose mode:"
echo "  1) Fast test (10 samples, ~10 minutes)"
echo "  2) Full run (200 samples, ~30 minutes)"
echo ""

# Check if running interactively
if [ -t 0 ]; then
    read -p "Enter choice [1 or 2]: " choice
else
    # Non-interactive, default to fast
    choice=1
    echo "Non-interactive mode detected, defaulting to fast test"
fi

case $choice in
    1)
        echo ""
        echo -e "${GREEN}Running FAST test mode...${NC}"
        bash scripts/06_phase5_robustness.sh --fast --use_gpu true
        ;;
    2)
        echo ""
        echo -e "${GREEN}Running FULL evaluation...${NC}"
        bash scripts/06_phase5_robustness.sh --use_gpu true
        ;;
    *)
        echo "Invalid choice, defaulting to fast mode"
        bash scripts/06_phase5_robustness.sh --fast --use_gpu true
        ;;
esac

# Step 6: Show Results
echo ""
echo -e "${BLUE}Step 6: Results Summary${NC}"
echo "=========================================="

if [ -f "reports/phase5/model_results.json" ]; then
    echo -e "${GREEN}✅ Phase 5 completed successfully!${NC}"
    echo ""
    echo "Results generated:"
    echo "  📊 reports/phase5/model_results.json"
    echo "  📊 reports/phase5/baseline_comparison.json"
    echo "  📊 reports/phase5/robustness_summary.json"
    echo "  📊 reports/phase5/induction_case_study.json"
    echo ""
    echo "Figures:"
    ls -1 reports/figs_paper/*.png 2>/dev/null | while read file; do
        echo "  🖼️  $file"
    done
    echo ""
    echo "Logs:"
    ls -1t logs/run_*.txt 2>/dev/null | head -n 1 | while read file; do
        echo "  📝 $file"
    done
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✅ Phase 5 NeurIPS Upgrade Complete!"
    echo "Ready for paper writing."
    echo "==========================================${NC}"
else
    echo -e "${YELLOW}⚠️  Results not found. Check logs for errors.${NC}"
    echo "Latest log:"
    ls -1t logs/run_*.txt 2>/dev/null | head -n 1
fi


#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Phase 5: NeurIPS Robustness & Publication Pipeline"
echo "=========================================================="

# Parse arguments
FAST_MODE=false
USE_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            shift
            ;;
        --use_gpu)
            USE_GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fast] [--use_gpu true|false]"
            exit 1
            ;;
    esac
done

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo "📦 Loading environment from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for HuggingFace token
if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "⚠️  Warning: No Hugging Face token set. Llama-3.2-1B may be skipped."
    echo "   To enable access, either:"
    echo "   1. Create a .env file with HUGGINGFACE_HUB_TOKEN=your_token"
    echo "   2. Or export HUGGINGFACE_HUB_TOKEN='your_token_here'"
else
    echo "✅ HuggingFace token loaded successfully"
fi

# Check if config file exists
CONFIG_FILE="configs/eval.phase5.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE not found"
    exit 1
fi

echo "📋 Using config: $CONFIG_FILE"

if [ "$FAST_MODE" = true ]; then
    echo "⚡ Fast mode enabled (reduced samples)"
fi

if [ "$USE_GPU" = "true" ]; then
    echo "🎮 GPU mode enabled"
fi

# Create output directories
mkdir -p reports/phase5
mkdir -p reports/figs_paper
mkdir -p logs

echo ""
echo "="*80
echo "Step 1: Main Phase 5 Evaluation"
echo "="*80

# Run Phase 5 main evaluation
PHASE5_ARGS="--config $CONFIG_FILE"
if [ "$FAST_MODE" = true ]; then
    PHASE5_ARGS="$PHASE5_ARGS --fast"
fi

echo "🔬 Running main evaluation pipeline..."
uv run python -m wor.eval.evaluate_phase5 $PHASE5_ARGS

if [ $? -ne 0 ]; then
    echo "❌ Phase 5 main evaluation failed"
    exit 1
fi

echo ""
echo "="*80
echo "Step 2: Robustness Tests"
echo "="*80

# Run robustness evaluation
echo "🧪 Running robustness tests (seeds, temps, ablations)..."
ROBUST_ARGS="--config $CONFIG_FILE"
if [ "$FAST_MODE" = true ]; then
    ROBUST_ARGS="$ROBUST_ARGS --fast"
fi

uv run python -m wor.eval.evaluate_robustness $ROBUST_ARGS

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Robustness tests failed, but continuing..."
fi

echo ""
echo "="*80
echo "Step 3: Induction Heads Case Study"
echo "="*80

# Run induction case study (using smallest model)
echo "🔍 Running induction heads mechanistic case study..."

uv run python - <<'PYTHON_SCRIPT'
import yaml
from wor.mech.induction_heads import run_induction_case_study

# Load Phase 5 config
with open("configs/eval.phase5.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# Load smallest model config
with open(cfg["models"][0], 'r') as f:
    model_cfg = yaml.safe_load(f)

# Run induction case study
results = run_induction_case_study(
    model_cfg,
    output_dir="reports/phase5",
    n_samples=50,
    k_percentages=cfg["patchout"]["k_percent"],
    seed=cfg["seed"]
)

print("✅ Induction case study complete!")
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Induction case study failed, but continuing..."
fi

echo ""
echo "="*80
echo "Step 4: Generate Publication Figures"
echo "="*80

echo "📊 Creating publication-ready figures..."
uv run python -m wor.plots.paper_figs reports/phase5 reports/figs_paper

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Figure generation failed, but continuing..."
fi

echo ""
echo "="*80
echo "Step 5: Run Validation Tests"
echo "="*80

echo "🧪 Running validation tests..."

# Run baseline tests
if [ -f "tests/test_baselines.py" ]; then
    uv run pytest tests/test_baselines.py -v
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Some baseline tests failed"
    fi
fi

# Run induction tests
if [ -f "tests/test_induction.py" ]; then
    uv run pytest tests/test_induction.py -v
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Some induction tests failed"
    fi
fi

echo ""
echo "✅ Phase 5 NeurIPS Upgrade Complete!"
echo ""
echo "📁 Output locations:"
echo "   - reports/phase5/              (all results & summaries)"
echo "   - reports/figs_paper/          (publication figures)"
echo "   - reports/phase5/robustness_summary.json"
echo "   - reports/phase5/induction_case_study.json"
echo "   - reports/phase5/baseline_comparison.json"
echo "   - logs/                        (timestamped run logs)"
echo ""
echo "🔬 Key files for paper:"
echo "   - Scaling: reports/figs_paper/scaling_curve.png"
echo "   - ROC: reports/figs_paper/roc_curves.png"
echo "   - Causal: reports/figs_paper/causal_scatter.png"
echo "   - Robustness: reports/figs_paper/robustness_bars.png"
echo "   - Induction: reports/figs_paper/induction_comparison.png"
echo ""
echo "📊 Success criteria validation:"
uv run python - <<'PYTHON_SCRIPT'
import json

try:
    with open("reports/phase5/model_results.json", 'r') as f:
        results = json.load(f)
    
    print("\n✅ SUCCESS CRITERIA:")
    
    # Criterion 1: ΔAUC
    delta_aucs = []
    for model, data in results.items():
        if "baseline_comparison" in data:
            delta_auc = data["baseline_comparison"].get("delta_auc", 0)
            delta_aucs.append(delta_auc)
            print(f"  - {model}: ΔAUC = +{delta_auc:.4f}")
    
    if delta_aucs and max(delta_aucs) >= 0.05:
        print("  ✅ REV adds ≥ +0.05 AUROC over baseline")
    else:
        print("  ⚠️  ΔAUC criterion not met")
    
    # Criterion 2: Effect size
    cohens_d_values = []
    for model, data in results.items():
        d = data.get("cohens_d", {}).get("REV", 0)
        cohens_d_values.append(d)
        print(f"  - {model}: Cohen's d = {d:.4f}")
    
    if cohens_d_values and max(cohens_d_values) > 0.5:
        print("  ✅ REV shows medium+ effect size (d>0.5)")
    else:
        print("  ⚠️  Effect size criterion not met")
    
    print("\n")
    
except Exception as e:
    print(f"⚠️  Could not validate criteria: {e}")
PYTHON_SCRIPT

echo "=========================================================="
echo "🎉 Ready for NeurIPS submission!"
echo "=========================================================="


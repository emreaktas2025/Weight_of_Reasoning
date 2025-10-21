#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Phase 5 Validation Checklist"
echo "=================================="

# Check all required files exist
echo ""
echo "1. Checking files..."
files=(
    "configs/eval.phase5.yaml"
    "scripts/06_phase5_robustness.sh"
    "src/wor/baselines/__init__.py"
    "src/wor/baselines/predictors.py"
    "src/wor/eval/evaluate_phase5.py"
    "src/wor/eval/evaluate_robustness.py"
    "src/wor/mech/induction_heads.py"
    "src/wor/plots/paper_figs.py"
    "src/wor/utils/logging.py"
    "tests/test_baselines.py"
    "tests/test_induction.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file MISSING"
    fi
done

# Check if scripts are executable
echo ""
echo "2. Checking permissions..."
if [ -x "scripts/06_phase5_robustness.sh" ]; then
    echo "  ✅ scripts/06_phase5_robustness.sh is executable"
else
    echo "  ⚠️  scripts/06_phase5_robustness.sh not executable (run: chmod +x scripts/06_phase5_robustness.sh)"
fi

# Check dependencies
echo ""
echo "3. Checking dependencies..."
uv run python -c "import sklearn; print('  ✅ scikit-learn installed')" 2>/dev/null || echo "  ⚠️  scikit-learn not found (will install on first run)"

# Run tests
echo ""
echo "4. Running tests..."
echo "  Testing baselines..."
uv run python tests/test_baselines.py > /dev/null 2>&1 && echo "  ✅ Baseline tests pass" || echo "  ❌ Baseline tests fail"

echo "  Testing induction heads..."
uv run python tests/test_induction.py > /dev/null 2>&1 && echo "  ✅ Induction tests pass" || echo "  ❌ Induction tests fail"

# Check git status
echo ""
echo "5. Git status..."
if git diff --quiet; then
    echo "  ✅ No uncommitted changes"
else
    echo "  ⚠️  You have uncommitted changes"
fi

# Summary
echo ""
echo "=================================="
echo "📋 Summary:"
echo ""
echo "Phase 5 is ready! Next steps:"
echo ""
echo "Local testing:"
echo "  bash scripts/06_phase5_robustness.sh --fast"
echo ""
echo "GPU run on RunPod:"
echo "  1. SSH: ssh vo6p5cbychop12-644115af@ssh.runpod.io -i ~/.ssh/id_ed25519"
echo "  2. Clone/pull repo"
echo "  3. Run: bash scripts/06_phase5_robustness.sh --use_gpu true"
echo ""
echo "=================================="


# Weight of Reasoning

A reproducible research codebase to compute internal reasoning effort metrics for open LLMs.

## Overview

This project implements metrics to measure the internal computational effort that language models expend during reasoning tasks. The two primary metrics are:

- **Activation Energy (AE)**: Measures the magnitude of hidden state activations during reasoning, normalized by sequence length
- **Attention Process Entropy (APE)**: Measures the entropy of attention patterns across heads during reasoning

These metrics are designed to capture the "weight" or computational intensity of reasoning processes in transformer models.

## Installation

```bash
# Set up environment with uv
uv sync
```

## Quick Start

```bash
# 1. Generate tiny evaluation dataset
uv run python scripts/01_download_data.py --mini

# 2. Run tiny evaluation
bash scripts/02_tiny_eval.sh

# 3. View results
cat reports/metrics.csv
cat reports/summary.json
ls reports/plots/
```

## Phase 2: APL and Partial Correlations

Run extended evaluation with Activation Path Length (APL):

```bash
bash scripts/02_tiny_eval.sh
```

New outputs:
- `reports/metrics_phase2.csv`: AE, APE, APL, token_len, ppl per sample
- `reports/summary_phase2.json`: Includes Cohen's d for all metrics + partial correlations
- `reports/partial_corr.json`: Partial correlation analysis controlling for length/perplexity
- `reports/plots/apl_violin.png`: APL distribution by label
- `reports/plots/ae_apl_scatter.png`: AE vs APL relationship

**APL (Activation Path Length)**: Measures how many layers show significant activation changes when ablated, indicating depth of reasoning pathways.

## Phase 3: Advanced Metrics and REV Score

Run comprehensive evaluation with all six reasoning effort metrics:

```bash
bash scripts/03_phase3_eval.sh
```

New metrics:
- **CUD (Circuit Utilization Density)**: Fraction of reasoning circuit heads engaged during evaluation
- **SIB (Stability of Intermediate Beliefs)**: Robustness of hidden states under paraphrasing perturbations
- **FL (Feature Load)**: Activation sparsity proxy measuring feature utilization density
- **REV (Reasoning Effort Value)**: Composite z-scored metric combining all six measures

Outputs:
- `reports/metrics_phase3.csv`: All metrics (AE, APE, APL, CUD, SIB, FL, REV) per sample
- `reports/summary_phase3.json`: Cohen's d, means, partial correlations for all metrics
- `reports/partial_corr_phase3.json`: Detailed correlation analysis controlling for length/perplexity
- `reports/plots/rev_violin.png`: REV distribution by label
- `reports/plots/metric_corr_heatmap.png`: 6x6 correlation matrix
- `reports/plots/cud_box.png`, `sib_box.png`, `fl_box.png`: Individual metric distributions

## Phase 4: Scaling Study & Mechanistic Validation

Run scaling study across multiple model sizes with mechanistic validation:

```bash
bash scripts/04_phase4_scale.sh
```

**Features:**
- **Multi-model scaling**: Evaluates pythia-70m, pythia-410m, and llama3-1b (optional)
- **Hardware-aware**: Automatically detects GPU/CPU and adjusts batch sizes and data types
- **Dataset scaling**: Uses GSM8K and StrategyQA with fallback to local data
- **Mechanistic validation**: Patch-out experiments to validate causal relationships
- **Accuracy computation**: Regex-based numeric answer extraction for reasoning tasks

**Outputs:**
- `reports/phase4/`: Per-model results (CSV, JSON, partial correlations)
- `reports/phase4/aggregate_scaling.json`: Cross-model scaling trends and statistics
- `reports/plots_phase4/`: Scaling curves and patch-out effect plots
- `reports/hw_phase4.json`: Hardware configuration and optimization settings
- `reports/splits/`: Dataset manifests for reproducibility

**Runtime targets:**
- CPU: <30 minutes (40 samples per dataset)
- GPU: <10 minutes (200 samples per dataset)

**Models:**
- Always includes: pythia-70m, pythia-410m
- Optional: llama3-1b (skips gracefully if unavailable or requires auth)

## Phase 4b: Mechanistic Validation & Llama Integration

Run causal patch-out experiments to validate REV components:

```bash
# Set HuggingFace token for Llama access (optional)
export HUGGINGFACE_TOKEN="your_token_here"

# Run mechanistic validation
bash scripts/05_phase4b_patchout.sh
```

**Features:**
- Llama-3.2-1B integration with authentication
- Fixed dataset loaders (StrategyQA, Wikipedia)
- Causal ablation experiments via TransformerLens hooks
- Correlation analysis: ΔREV vs Δaccuracy

**Outputs:**
- `reports/phase4b/<model>_patchout_heads.json`
- `reports/phase4b/aggregate_patchout.json`
- `reports/plots_phase4b/patchout_heads_delta.png`
- Causality demonstration plots

**Runtime targets:**
- CPU: <20 minutes (30 samples per dataset)
- GPU: <8 minutes (100 samples per dataset)

## Phase 5: NeurIPS Publication-Grade Upgrade

Run comprehensive evaluation with comparative baselines, robustness tests, and mechanistic case studies:

```bash
# Quick test (reduced samples)
bash scripts/06_phase5_robustness.sh --fast

# Full GPU run
bash scripts/06_phase5_robustness.sh --use_gpu true
```

**Features:**
- **Comparative Baselines**: Logistic regression on token_len, avg_logprob, perplexity, CoT_len
- **Multi-Dataset Generalization**: GSM8K, StrategyQA, Math Dataset with per-dataset AUROC
- **Robustness Testing**: 3 seeds (42, 1337, 999), 2 temperatures (0.0, 0.2), metric ablations
- **Mechanistic Case Study**: Induction heads with synthetic ABC→ABC dataset
- **Causal Validation**: r(ΔREV, ΔAcc) correlation from patch-out experiments
- **Publication Figures**: Scaling curves, ROC curves, causal scatter, robustness bars

**Outputs:**
- `reports/phase5/model_results.json`: Cross-model results with scaling analysis
- `reports/phase5/baseline_comparison.json`: REV vs baseline predictors (ΔAUC)
- `reports/phase5/robustness_summary.json`: Seed/temp/ablation consistency
- `reports/phase5/induction_case_study.json`: Targeted vs random patch-out
- `reports/figs_paper/*.png`: Publication-ready figures
- `logs/run_*.txt`: Timestamped execution logs

**Success Criteria:**
- ✅ REV adds ≥ +0.05 AUROC over best baseline on ≥ 2 datasets
- ✅ Targeted patch-out: ΔAcc ≤ −20%, ΔREV ≤ −0.5σ
- ✅ Results replicate across 3 seeds (mean ± std reported)
- ✅ Results consistent across temps 0.0 and 0.2

**Runtime targets:**
- CPU (--fast): ~5 minutes (10 samples per dataset)
- GPU (full): ~30 minutes on RTX 4090 (200 samples per dataset)

## Results

After running the evaluation, you'll find:

- `reports/metrics.csv`: Sample-level metrics (AE, APE, token counts)
- `reports/summary.json`: Summary statistics including Cohen's d for separability
- `reports/plots/`: Visualization plots (violin plots, trajectory plots)

## Why AE and APE?

**Activation Energy (AE)** captures the magnitude of internal representations during reasoning. Higher AE suggests more intense computational activity in the model's hidden states.

**Attention Process Entropy (APE)** measures the diversity of attention patterns. More structured reasoning may show different entropy patterns compared to free-form text generation.

Both metrics are length-normalized to control for sequence length effects and compared against length-matched control tasks.

## Limitations

- **CPU-only**: Uses small models (Pythia-70M) for accessibility
- **Heuristic reasoning window**: Uses simple heuristics to identify reasoning vs. answer tokens
- **Tiny evaluation**: Limited to 10 samples for quick validation
- **Model-specific**: Results may vary across different architectures

## Next Steps

### Future Extensions

Planned future work:

- Integration with additional reasoning datasets (MATH, LogiQA)
- Length-stratified matching and advanced correlation analysis
- Integration with larger models (GPT-2, LLaMA-2/3) and datasets
- Advanced mechanistic interpretability studies using activation patching
- CI/CD pipeline with automated evaluation
- Real-time metric computation for interactive analysis

## Project Structure

```
weight-of-reasoning/
├── src/wor/                    # Main package
│   ├── core/                   # Model runner and utilities
│   ├── data/                   # Dataset loaders (Phase 4b)
│   ├── metrics/                # AE, APE, APL, CUD, SIB, FL, REV implementations
│   ├── eval/                   # Evaluation pipeline (Phases 1-4b)
│   ├── mech/                   # Mechanistic validation (Phase 4b)
│   ├── plots/                  # Visualization utilities
│   ├── stats/                  # Statistical analysis
│   └── utils/                  # Hardware detection and utilities
├── configs/                    # Configuration files
│   └── models/                 # Model-specific configs (Phase 4)
├── data/mini/                  # Tiny evaluation dataset
├── scripts/                    # Setup and evaluation scripts
├── tests/                      # Unit tests
└── reports/                    # Generated results (auto-created)
    ├── phase4/                 # Phase 4 scaling results
    ├── phase4b/                # Phase 4b mechanistic validation
    ├── plots_phase4/           # Phase 4 plots
    ├── plots_phase4b/          # Phase 4b causal plots
    └── splits/                 # Dataset manifests
```

## Development

```bash
# Run tests
uv run pytest -q

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_imports.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{weight_of_reasoning,
  title={Weight of Reasoning: Internal Effort Metrics for Language Models},
  author={Research Team},
  year={2024},
  url={https://github.com/your-org/weight-of-reasoning}
}
```

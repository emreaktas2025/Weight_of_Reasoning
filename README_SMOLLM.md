# SmolLM REV Research Pipeline

A complete, reproducible research pipeline for evaluating Reasoning Effort in Language Models (REV) using only the SmolLM model family (135M → 350M → 1.7B).

## 🎯 Overview

This pipeline provides:
- **Automatic iteration management** with immutable experiment folders
- **SmolLM-specific evaluation** (HuggingFaceTB/SmolLM-135M, 350M, 1.7B)
- **REV metric computation** (REV, APL, APE, SIB) with activation capture
- **Statistical analysis** (bootstrap CIs, DeLong tests, partial correlations)
- **Publication-ready figures** with semantic naming conventions
- **Complete reproducibility** with git hash tracking and environment snapshots

## 📁 Project Structure

```
Weight_of_Reasoning/
├── configs/smollm/           # SmolLM-specific configurations
│   ├── eval.default.yaml    # Full evaluation config
│   ├── eval.debug.yaml      # Debug config (10 samples)
│   └── models/              # Model specifications
├── data/smollm_ids/         # Frozen dataset ID lists
├── experiments/             # Iteration-based results (immutable)
│   ├── iteration_001/       # Each run gets unique folder
│   ├── iteration_002/
│   └── iteration_registry.csv
├── src/wor/smollm/          # SmolLM-specific modules
│   ├── loaders.py           # Model and data loaders
│   ├── activations.py       # PyTorch hooks for activation capture
│   ├── metrics/             # REV, APL, APE, SIB implementations
│   ├── eval/                # Evaluation pipeline
│   └── utils/               # Iteration management
├── scripts/smollm/          # Execution scripts
└── notebooks/               # Analysis notebooks
```

## 🚀 Quick Start

### 1. Debug Run (Recommended First)
```bash
# Quick test with 10 samples per dataset
bash scripts/smollm/run_debug.sh
```

### 2. Full Evaluation
```bash
# Complete evaluation with all 3 SmolLM models
bash scripts/smollm/run_all.sh
```

### 3. Summarize Results
```bash
# Aggregate results and update registry
bash scripts/smollm/summarize_iteration.sh
```

## 📊 Iteration System

Each experiment run creates an immutable iteration folder:

```
experiments/iteration_001/
├── config_used.yaml         # Exact config snapshot
├── env_info.json           # Environment details
├── timing.json             # Runtime information
├── logs/run.log            # Complete stdout/stderr
├── results/                # Model-specific results
│   ├── smollm-135m_metrics.csv
│   ├── smollm-350m_metrics.csv
│   ├── smollm-1.7b_metrics.csv
│   ├── model_results.json
│   └── aggregate_statistics.json
├── figures/                # Generated plots
│   ├── Fig1_scaling__iteration_001.png
│   ├── Fig2_temp_robustness__iteration_001.png
│   └── Fig3_ablation_curve__iteration_001.png
└── notes.md                # Auto-generated template
```

## 🔧 Configuration

### Model Specifications
```yaml
# configs/smollm/models/smollm-135m.yaml
model_name: HuggingFaceTB/SmolLM-135M
device: cuda
dtype: bfloat16
max_new_tokens: 64
temperature: 0.2
```

### Dataset Configuration
```yaml
# configs/smollm/eval.default.yaml
datasets:
  reasoning:
    gsm8k: 800
    strategyqa: 400
  control:
    wiki: 1200
```

## 📈 Metrics Computed

### Core REV Metrics
- **REV**: Composite reasoning effort score
- **APL**: Activation Path Length
- **APE**: Attention Process Entropy  
- **SIB**: Stability of Intermediate Beliefs

### Statistical Analysis
- **AUROC**: Area under ROC curve
- **ΔAUC**: Improvement over baseline
- **Partial correlations**: Controlling for length/perplexity
- **Bootstrap CIs**: 10,000-sample confidence intervals
- **DeLong test**: Statistical significance testing

## 🎨 Figure Generation

Figures follow semantic naming conventions:

- `Fig1_scaling__{iteration_id}.png`: ΔAUC vs log(model parameters)
- `Fig2_temp_robustness__{iteration_id}.png`: AUROC vs Temperature
- `Fig3_ablation_curve__{iteration_id}.png`: Δ-accuracy vs % heads removed

## ✅ Success Criteria

Automated validation checks:
- ✅ No NaNs across temperatures
- ✅ Monotonic ΔAUC: 135M < 350M < 1.7B
- ✅ ΔAUC ≥ +0.05 @ 350M; ≥ +0.12 @ 1.7B
- ✅ Partial r(REV|length,ppl) ≥ 0.3, p < 0.01
- ✅ Temperature variance ≤ 0.03

## 🔄 Reproducibility

### Exact Reproduction
```bash
# Reproduce specific iteration
python -m src.wor.smollm.eval.run_eval --config configs/smollm/eval.default.yaml --reproduce iteration_001
```

### Environment Tracking
Each iteration saves:
- Git commit hash
- Python/CUDA/PyTorch versions
- Hardware configuration
- Exact configuration used

## 📝 Usage Examples

### Run Debug Evaluation
```bash
# Quick test with minimal samples
bash scripts/smollm/run_debug.sh
```

### Run Full Pipeline
```bash
# Complete evaluation
bash scripts/smollm/run_all.sh

# Check results
ls experiments/iteration_001/results/
cat experiments/iteration_001/notes.md
```

### Analyze Results
```bash
# Generate summary
bash scripts/smollm/summarize_iteration.sh

# Check iteration registry
cat experiments/iteration_registry.csv
```

## 🛠️ Troubleshooting

### Common Issues

1. **GPU Memory Error**
   ```bash
   # Use debug config first
   bash scripts/smollm/run_debug.sh
   ```

2. **HuggingFace Authentication**
   ```bash
   # Set up HF token
   export HUGGINGFACE_HUB_TOKEN=your_token_here
   ```

3. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install torch transformer-lens datasets scikit-learn scipy pandas
   ```

### Debug Mode
The debug configuration (`eval.debug.yaml`) uses:
- 10 samples per dataset
- Single model (135M)
- Reduced memory usage
- Faster execution

## 📊 Expected Outputs

### Per Iteration
- **Results**: `{model}_{dataset}_metrics.csv`
- **Summary**: `metrics_summary.json`
- **Figures**: Paper-ready plots with semantic naming
- **Notes**: Auto-generated template with metadata

### Global Registry
- **Chronological log**: `experiments/iteration_registry.csv`
- **Best iteration**: Highest ΔAUC across all runs
- **Reproducibility**: Complete experiment history

## 🔬 Research Workflow

1. **Initial Setup**: Run debug evaluation
2. **Full Evaluation**: Execute complete pipeline
3. **Analysis**: Review results and figures
4. **Iteration**: Modify configs and re-run
5. **Publication**: Use best iteration results

## 📚 Key Files

- `src/wor/smollm/eval/run_eval.py`: Main evaluation pipeline
- `src/wor/smollm/utils/io_utils.py`: Iteration management
- `src/wor/smollm/metrics/rev.py`: REV computation
- `scripts/smollm/run_all.sh`: Full execution script
- `configs/smollm/eval.default.yaml`: Default configuration

## 🎯 Next Steps

After running the pipeline:

1. **Review Results**: Check `experiments/iteration_XXX/notes.md`
2. **Generate Figures**: Run visualization scripts
3. **Analyze Metrics**: Examine partial correlations and effect sizes
4. **Iterate**: Modify configurations and re-run
5. **Publish**: Use best iteration for paper results

---

**Note**: This pipeline is designed for SmolLM models only. For other model families, use the main `src/wor/` evaluation modules.

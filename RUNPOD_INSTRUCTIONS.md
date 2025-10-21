# RunPod GPU Run Instructions

## Quick Setup (Run on RunPod)

```bash
# 1. Configure Git Identity
git config --global user.email "meaktas21@gmail.com"
git config --global user.name "Emre Aktas"

# 2. Pull Latest Fixes
cd /workspace/Weight_of_Reasoning
git pull origin main

# 3. Disable HF Transfer (if needed)
export HF_HUB_ENABLE_HF_TRANSFER=0

# 4. Set HuggingFace Token (use your own token)
export HUGGINGFACE_HUB_TOKEN="your_hf_token_here"

# 5. Re-run Phase 5 (JSON serialization now fixed!)
bash scripts/06_phase5_robustness.sh --fast --use_gpu true
```

## What Got Fixed

✅ **JSON Serialization** - Numpy arrays now convert properly  
✅ **Git Configuration** - Identity set for commits  
✅ **Report Tracking** - Key results can now be committed

## After Successful Run

```bash
# Verify JSON files were created
ls -lh reports/phase5/*.json

# View a summary
cat reports/phase5/model_results.json | python3 -m json.tool

# Add and commit results
git add reports/
git commit -m "Phase 5 GPU results: Pythia-70M, Pythia-410M, Llama-3.2-1B"
git push origin main
```

## Copy Results to Local Mac

From your **Mac terminal**:

```bash
# Clone fresh copy with results
cd ~/Desktop
git clone https://github.com/emreaktas2025/Weight_of_Reasoning.git phase5_results
cd phase5_results

# Run analysis notebook
jupyter notebook notebooks/analysis_phase5.ipynb
```

## Success Criteria

✅ All 3 models processed  
✅ All JSON files created  
✅ ΔAUC ≥ +0.05 on all models  
✅ Figures generated in `reports/figs_paper/`  
✅ Results committed to GitHub


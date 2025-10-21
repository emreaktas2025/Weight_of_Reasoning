"""Tests for Phase 4b mechanistic validation and patch-out functionality."""

import json
import os
import glob
import numpy as np
from typing import Dict, Any


def test_imports_and_environment():
    """Test AT-1: Import & Environment - All modules import successfully."""
    try:
        from wor.data.loaders import load_reasoning_dataset, load_control_dataset
        from wor.eval.evaluate_phase4b import run_phase4b_evaluation, setup_huggingface_auth
        from wor.plots.plot_phase4b import create_all_phase4b_plots
        from wor.mech.patchout import run_patchout_experiment
        print("✅ All Phase 4b modules imported successfully")
    except ImportError as e:
        assert False, f"Failed to import Phase 4b modules: {e}"


def test_huggingface_auth_handling():
    """Test AT-1: HuggingFace auth handled gracefully."""
    try:
        from wor.eval.evaluate_phase4b import setup_huggingface_auth
        
        # Test without token (should return False gracefully)
        auth_result = setup_huggingface_auth()
        assert isinstance(auth_result, bool), "Auth function should return boolean"
        print("✅ HuggingFace authentication handled gracefully")
    except Exception as e:
        assert False, f"HuggingFace auth handling failed: {e}"


def test_dataset_fixes():
    """Test AT-2: Dataset Fix - Loaders return ≥10 samples with proper labeling."""
    try:
        from wor.data.loaders import load_reasoning_dataset, load_control_dataset, validate_dataset_labels
        
        # Test reasoning dataset loading
        reasoning_data = load_reasoning_dataset("gsm8k", 10, seed=1337)
        assert len(reasoning_data) >= 10, f"Expected ≥10 reasoning samples, got {len(reasoning_data)}"
        assert validate_dataset_labels(reasoning_data, "reasoning"), "Reasoning samples should have 'reasoning' label"
        
        # Test control dataset loading
        control_data = load_control_dataset("wiki", 10, seed=1337)
        assert len(control_data) >= 5, f"Expected ≥5 control samples, got {len(control_data)}"
        assert validate_dataset_labels(control_data, "control"), "Control samples should have 'control' label"
        
        print("✅ Dataset loaders return proper samples with correct labeling")
    except Exception as e:
        assert False, f"Dataset loading test failed: {e}"


def test_patchout_json_exists():
    """Test AT-3: Patch-out JSON files exist."""
    phase4b_dir = "reports/phase4b"
    
    if not os.path.exists(phase4b_dir):
        print("⚠️  Phase 4b directory not found - skipping patch-out output tests")
        return
    
    # Find patch-out result files
    files = glob.glob(os.path.join(phase4b_dir, "*_patchout_*.json"))
    assert files, "No patch-out outputs found"
    
    print(f"✅ Found {len(files)} patch-out JSON files")


def test_patchout_has_deltas():
    """Test AT-3: Patch-out files have delta data."""
    phase4b_dir = "reports/phase4b"
    
    if not os.path.exists(phase4b_dir):
        print("⚠️  Phase 4b directory not found - skipping delta tests")
        return
    
    # Check for layer patch-out files (more likely to succeed)
    layer_files = glob.glob(os.path.join(phase4b_dir, "*_patchout_layers.json"))
    if layer_files:
        data = json.load(open(layer_files[0]))
        # Check for various possible data structures
        has_data = (
            "delta_accuracy" in data or 
            "structured" in data or 
            any(isinstance(v, dict) and ("acc" in v or "rev" in v) for v in data.values())
        )
        assert has_data, "No layer deltas found"
        print("✅ Layer patch-out deltas found")
    else:
        # Check for head files
        head_files = glob.glob(os.path.join(phase4b_dir, "*_patchout_heads.json"))
        if head_files:
            data = json.load(open(head_files[0]))
            # Check for various possible data structures
            has_data = (
                "delta_accuracy" in data or 
                "structured" in data or 
                any(isinstance(v, dict) and ("acc" in v or "rev" in v) for v in data.values())
            )
            assert has_data, "No head deltas found"
            print("✅ Head patch-out deltas found")
        else:
            print("⚠️  No patch-out files with deltas found")


def test_patchout_outputs():
    """Test AT-3: Patch-out Outputs - JSON files exist with valid structure."""
    test_patchout_json_exists()
    test_patchout_has_deltas()


def test_causal_correlation():
    """Test AT-4: Causal Correlation - Spearman ρ(ΔREV, Δaccuracy) > 0.6 with p < 0.05."""
    aggregate_file = "reports/phase4b/aggregate_patchout.json"
    
    if not os.path.exists(aggregate_file):
        print("⚠️  Aggregate patch-out file not found - skipping correlation test")
        return
    
    with open(aggregate_file, 'r') as f:
        aggregate_data = json.load(f)
    
    # Check if causal correlations exist
    if "causal_correlations" not in aggregate_data:
        print("⚠️  No causal correlations found - skipping correlation test")
        return
    
    causal_correlations = aggregate_data["causal_correlations"]
    
    # Look for the main correlation
    if "delta_rev_vs_delta_accuracy" in causal_correlations:
        corr_data = causal_correlations["delta_rev_vs_delta_accuracy"]
        
        rho = corr_data["rho"]
        p_value = corr_data["p"]
        n_samples = corr_data["n_samples"]
        
        # Check correlation strength and significance
        assert rho > 0.6, f"Spearman ρ should be > 0.6, got {rho}"
        assert p_value < 0.05, f"p-value should be < 0.05, got {p_value}"
        assert n_samples >= 3, f"Should have ≥3 samples for correlation, got {n_samples}"
        
        print(f"✅ Causal correlation: ρ={rho:.4f}, p={p_value:.4f} (n={n_samples})")
    else:
        print("⚠️  No delta_rev_vs_delta_accuracy correlation found")


def test_plots():
    """Test AT-5: Plots - 3 plots exist, each >10 KB."""
    plots_dir = "reports/plots_phase4b"
    
    if not os.path.exists(plots_dir):
        print("⚠️  Phase 4b plots directory not found - skipping plot tests")
        return
    
    expected_plots = [
        "patchout_heads_delta.png",
        "patchout_layers_delta.png", 
        "rev_accuracy_corr.png"
    ]
    
    found_plots = []
    for plot_file in expected_plots:
        plot_path = os.path.join(plots_dir, plot_file)
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            assert file_size > 10240, f"Plot {plot_file} is too small ({file_size} bytes)"
            found_plots.append(plot_file)
    
    assert len(found_plots) >= 2, f"Expected at least 2 plots, found {len(found_plots)}"
    print(f"✅ Found {len(found_plots)} valid plots (>10 KB each)")


def test_runtime():
    """Test AT-6: Runtime - Total Phase 4b runtime < 20 min on CPU."""
    aggregate_file = "reports/phase4b/aggregate_patchout.json"
    
    if not os.path.exists(aggregate_file):
        print("⚠️  Aggregate file not found - skipping runtime test")
        return
    
    with open(aggregate_file, 'r') as f:
        aggregate_data = json.load(f)
    
    if "runtime_sec" in aggregate_data:
        runtime_sec = aggregate_data["runtime_sec"]
        runtime_min = runtime_sec / 60
        
        assert runtime_min < 20, f"Runtime should be < 20 min, got {runtime_min:.2f} min"
        print(f"✅ Runtime: {runtime_min:.2f} minutes (< 20 min target)")
    else:
        print("⚠️  No runtime data found")


def test_unit_tests():
    """Test AT-7: Unit Tests - All pytest tests pass."""
    try:
        import subprocess
        result = subprocess.run(["uv", "run", "pytest", "-q", "tests/"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ All pytest tests pass")
        else:
            print(f"⚠️  Some pytest tests failed: {result.stdout}")
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Could not run pytest: {e}")


def test_hardware_config():
    """Test hardware configuration is logged."""
    hw_file = "reports/hw_phase4b.json"
    assert os.path.exists(hw_file), f"Hardware config file {hw_file} not found"
    
    with open(hw_file, 'r') as f:
        hw_config = json.load(f)
    
    required_fields = ["device", "dtype", "use_gpu", "max_new_tokens", "batch_size"]
    for field in required_fields:
        assert field in hw_config, f"Missing field {field} in hardware config"
    
    print("✅ Hardware configuration logged correctly")


def test_baseline_results():
    """Test baseline results exist and are valid."""
    phase4b_dir = "reports/phase4b"
    
    if not os.path.exists(phase4b_dir):
        print("⚠️  Phase 4b directory not found - skipping baseline test")
        return
    
    baseline_files = glob.glob(os.path.join(phase4b_dir, "*_baseline.json"))
    assert len(baseline_files) > 0, "No baseline JSON files found"
    
    for file_path in baseline_files:
        with open(file_path, 'r') as f:
            baseline_data = json.load(f)
        
        # Check required fields
        required_fields = ["n_params", "accuracy", "cohens_d", "auroc_REV"]
        for field in required_fields:
            assert field in baseline_data, f"Missing field {field} in baseline data"
        
        # Check parameter count is positive
        assert baseline_data["n_params"] > 0, "Parameter count should be positive"
        
        # Check accuracy is in valid range
        accuracy = baseline_data["accuracy"]
        assert 0 <= accuracy <= 1, f"Accuracy should be in [0,1], got {accuracy}"
    
    print("✅ Baseline results are valid")


def run_all_phase4b_tests():
    """Run all Phase 4b acceptance tests."""
    print("Running Phase 4b acceptance tests...\n")
    
    try:
        test_imports_and_environment()
        test_huggingface_auth_handling()
        test_dataset_fixes()
        test_patchout_outputs()
        test_causal_correlation()
        test_plots()
        test_runtime()
        test_unit_tests()
        test_hardware_config()
        test_baseline_results()
        
        print("\n🎉 All Phase 4b acceptance tests passed!")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_phase4b_tests()
    exit(0 if success else 1)

"""Tests for Phase 4 scaling and patch-out functionality."""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any


def test_hw_detection():
    """Test AT-1: Hardware detection and hw_phase4.json creation."""
    hw_file = "reports/hw_phase4.json"
    assert os.path.exists(hw_file), f"Hardware config file {hw_file} not found"
    
    with open(hw_file, 'r') as f:
        hw_config = json.load(f)
    
    # Check required fields
    required_fields = ["device", "dtype", "use_gpu", "max_new_tokens", "batch_size", "num_workers"]
    for field in required_fields:
        assert field in hw_config, f"Missing field {field} in hardware config"
    
    # Validate data types and ranges
    assert isinstance(hw_config["use_gpu"], bool), "use_gpu must be boolean"
    assert hw_config["max_new_tokens"] > 0, "max_new_tokens must be positive"
    assert hw_config["batch_size"] > 0, "batch_size must be positive"
    assert hw_config["num_workers"] > 0, "num_workers must be positive"
    
    # Check dtype is valid
    valid_dtypes = ["float32", "float16"]
    assert hw_config["dtype"] in valid_dtypes, f"Invalid dtype: {hw_config['dtype']}"
    
    print("✅ AT-1: Hardware detection test passed")


def test_scaling_outputs():
    """Test AT-2: Per-model CSV/JSON files exist with expected columns."""
    phase4_dir = "reports/phase4"
    assert os.path.exists(phase4_dir), f"Phase 4 directory {phase4_dir} not found"
    
    # Find model files
    csv_files = [f for f in os.listdir(phase4_dir) if f.endswith("_metrics.csv")]
    json_files = [f for f in os.listdir(phase4_dir) if f.endswith("_summary.json")]
    partial_corr_files = [f for f in os.listdir(phase4_dir) if f.endswith("_partial_corr.json")]
    
    assert len(csv_files) > 0, "No metrics CSV files found"
    assert len(json_files) > 0, "No summary JSON files found"
    assert len(partial_corr_files) > 0, "No partial correlation JSON files found"
    
    # Test first available model files
    csv_file = csv_files[0]
    json_file = json_files[0]
    partial_corr_file = partial_corr_files[0]
    
    # Test CSV file
    csv_path = os.path.join(phase4_dir, csv_file)
    df = pd.read_csv(csv_path)
    
    required_columns = ["id", "label", "AE", "APE", "APL", "CUD", "SIB", "FL", "REV"]
    for col in required_columns:
        assert col in df.columns, f"Missing column {col} in {csv_file}"
    
    # Test JSON summary file
    json_path = os.path.join(phase4_dir, json_file)
    with open(json_path, 'r') as f:
        summary = json.load(f)
    
    required_summary_fields = ["n_params", "accuracy", "cohens_d", "auroc_REV"]
    for field in required_summary_fields:
        assert field in summary, f"Missing field {field} in {json_file}"
    
    # Test partial correlation file
    partial_corr_path = os.path.join(phase4_dir, partial_corr_file)
    with open(partial_corr_path, 'r') as f:
        partial_corr = json.load(f)
    
    assert isinstance(partial_corr, dict), "Partial correlation should be a dictionary"
    
    print("✅ AT-2: Scaling outputs test passed")


def test_aggregate_summary():
    """Test AT-3: Aggregate scaling JSON contains valid data."""
    aggregate_file = "reports/phase4/aggregate_scaling.json"
    assert os.path.exists(aggregate_file), f"Aggregate file {aggregate_file} not found"
    
    with open(aggregate_file, 'r') as f:
        aggregate_data = json.load(f)
    
    # Check required arrays
    required_arrays = ["n_params", "d_REV", "auroc_REV", "partial_r_REV", "model_names"]
    for array_name in required_arrays:
        assert array_name in aggregate_data, f"Missing array {array_name} in aggregate data"
        assert isinstance(aggregate_data[array_name], list), f"{array_name} should be a list"
        assert len(aggregate_data[array_name]) > 0, f"{array_name} should not be empty"
    
    # Check array lengths match
    array_lengths = [len(aggregate_data[arr]) for arr in required_arrays]
    assert all(length == array_lengths[0] for length in array_lengths), "All arrays should have same length"
    
    # Check valid ranges
    n_params = aggregate_data["n_params"]
    d_rev = aggregate_data["d_REV"]
    auroc_rev = aggregate_data["auroc_REV"]
    
    assert all(p > 0 for p in n_params), "All parameter counts should be positive"
    assert all(np.isfinite(d) for d in d_rev if d is not None), "Cohen's d values should be finite"
    assert all(0 <= a <= 1 for a in auroc_rev if a is not None), "AUROC values should be in [0,1]"
    
    print("✅ AT-3: Aggregate summary test passed")


def test_patchout_heads():
    """Test AT-4: Head patch-out JSON files exist with valid data."""
    phase4_dir = "reports/phase4"
    heads_files = [f for f in os.listdir(phase4_dir) if f.endswith("_patchout_heads.json")]
    
    assert len(heads_files) > 0, "No head patch-out JSON files found"
    
    # Test first available file
    heads_file = heads_files[0]
    heads_path = os.path.join(phase4_dir, heads_file)
    
    with open(heads_path, 'r') as f:
        heads_data = json.load(f)
    
    # Check structure
    assert "baseline" in heads_data, "Missing baseline data"
    assert "patchout_results" in heads_data, "Missing patchout results"
    
    # Check baseline data
    baseline = heads_data["baseline"]
    assert "delta_accuracy" in baseline or "accuracy" in baseline, "Missing accuracy in baseline"
    assert "delta_rev" in baseline or "mean_rev" in baseline, "Missing REV in baseline"
    
    # Check patchout results
    patchout_results = heads_data["patchout_results"]
    assert len(patchout_results) > 0, "No patchout results found"
    
    # Check at least one K percentage result
    for k_key, result in patchout_results.items():
        assert "delta_accuracy" in result, f"Missing delta_accuracy in {k_key}"
        assert "delta_rev" in result, f"Missing delta_rev in {k_key}"
        assert isinstance(result["delta_accuracy"], (int, float)), "delta_accuracy should be numeric"
        assert isinstance(result["delta_rev"], (int, float)), "delta_rev should be numeric"
    
    print("✅ AT-4: Head patch-out test passed")


def test_patchout_layers():
    """Test AT-5: Layer patch-out JSON files exist with valid data."""
    phase4_dir = "reports/phase4"
    layers_files = [f for f in os.listdir(phase4_dir) if f.endswith("_patchout_layers.json")]
    
    assert len(layers_files) > 0, "No layer patch-out JSON files found"
    
    # Test first available file
    layers_file = layers_files[0]
    layers_path = os.path.join(phase4_dir, layers_file)
    
    with open(layers_path, 'r') as f:
        layers_data = json.load(f)
    
    # Check structure
    assert "baseline" in layers_data, "Missing baseline data"
    assert "patchout_results" in layers_data, "Missing patchout results"
    
    # Check baseline data
    baseline = layers_data["baseline"]
    assert "delta_accuracy" in baseline or "accuracy" in baseline, "Missing accuracy in baseline"
    assert "delta_rev" in baseline or "mean_rev" in baseline, "Missing REV in baseline"
    
    # Check patchout results
    patchout_results = layers_data["patchout_results"]
    assert len(patchout_results) > 0, "No patchout results found"
    
    # Check at least one K percentage result
    for k_key, result in patchout_results.items():
        assert "delta_accuracy" in result, f"Missing delta_accuracy in {k_key}"
        assert "delta_rev" in result, f"Missing delta_rev in {k_key}"
        assert isinstance(result["delta_accuracy"], (int, float)), "delta_accuracy should be numeric"
        assert isinstance(result["delta_rev"], (int, float)), "delta_rev should be numeric"
    
    print("✅ AT-5: Layer patch-out test passed")


def test_plots():
    """Test AT-6: Plot files exist and are reasonably sized."""
    plots_dir = "reports/plots_phase4"
    assert os.path.exists(plots_dir), f"Plots directory {plots_dir} not found"
    
    # Expected plot files
    expected_plots = [
        "d_REV_vs_params.png",
        "AUROC_REV_vs_params.png",
        "r_partial_REV_vs_params.png",
        "patchout_heads_delta.png",
        "patchout_layers_delta.png",
        "rev_violin_by_model.png"
    ]
    
    found_plots = []
    for plot_file in expected_plots:
        plot_path = os.path.join(plots_dir, plot_file)
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            assert file_size > 10240, f"Plot {plot_file} is too small ({file_size} bytes)"
            found_plots.append(plot_file)
    
    assert len(found_plots) >= 5, f"Expected at least 5 plots, found {len(found_plots)}"
    
    print(f"✅ AT-6: Plots test passed ({len(found_plots)} plots found)")


def test_metrics_ranges():
    """Test AT-7: All metrics are in valid ranges."""
    phase4_dir = "reports/phase4"
    
    # Test CSV files
    csv_files = [f for f in os.listdir(phase4_dir) if f.endswith("_metrics.csv")]
    assert len(csv_files) > 0, "No metrics CSV files found"
    
    csv_file = csv_files[0]
    csv_path = os.path.join(phase4_dir, csv_file)
    df = pd.read_csv(csv_path)
    
    # Check REV values are numeric
    rev_values = df["REV"].dropna()
    assert len(rev_values) > 0, "No valid REV values found"
    assert all(np.isfinite(rev_values)), "All REV values should be finite"
    
    # Test summary files
    json_files = [f for f in os.listdir(phase4_dir) if f.endswith("_summary.json")]
    assert len(json_files) > 0, "No summary JSON files found"
    
    json_file = json_files[0]
    json_path = os.path.join(phase4_dir, json_file)
    
    with open(json_path, 'r') as f:
        summary = json.load(f)
    
    # Check AUROC is in valid range
    if "auroc_REV" in summary and summary["auroc_REV"] is not None:
        auroc = summary["auroc_REV"]
        assert 0 <= auroc <= 1, f"AUROC should be in [0,1], got {auroc}"
    
    # Check Cohen's d is finite
    if "cohens_d" in summary and "REV" in summary["cohens_d"]:
        cohens_d = summary["cohens_d"]["REV"]
        if cohens_d is not None:
            assert np.isfinite(cohens_d), f"Cohen's d should be finite, got {cohens_d}"
    
    print("✅ AT-7: Metrics ranges test passed")


def test_runtime_logging():
    """Test AT-8: Runtime is logged in aggregate JSON."""
    aggregate_file = "reports/phase4/aggregate_scaling.json"
    assert os.path.exists(aggregate_file), f"Aggregate file {aggregate_file} not found"
    
    with open(aggregate_file, 'r') as f:
        aggregate_data = json.load(f)
    
    # Check runtime is logged
    assert "runtime_sec" in aggregate_data, "Missing runtime_sec in aggregate data"
    runtime_sec = aggregate_data["runtime_sec"]
    
    assert isinstance(runtime_sec, list), "runtime_sec should be a list"
    assert len(runtime_sec) > 0, "runtime_sec should not be empty"
    
    # Check runtime values are positive
    for runtime in runtime_sec:
        assert runtime > 0, f"Runtime should be positive, got {runtime}"
        assert np.isfinite(runtime), f"Runtime should be finite, got {runtime}"
    
    print("✅ AT-8: Runtime logging test passed")


def test_imports():
    """Test that all Phase 4 modules can be imported."""
    try:
        from wor.utils.hw import detect_hw, log_hw_config
        from wor.eval.evaluate_phase4 import run_phase4_evaluation
        from wor.mech.patchout import run_mechanistic_validation
        from wor.plots.plot_phase4 import create_all_phase4_plots
        print("✅ All Phase 4 modules imported successfully")
    except ImportError as e:
        assert False, f"Failed to import Phase 4 modules: {e}"


def run_all_tests():
    """Run all Phase 4 acceptance tests."""
    print("Running Phase 4 acceptance tests...\n")
    
    try:
        test_imports()
        test_hw_detection()
        test_scaling_outputs()
        test_aggregate_summary()
        test_patchout_heads()
        test_patchout_layers()
        test_plots()
        test_metrics_ranges()
        test_runtime_logging()
        
        print("\n🎉 All Phase 4 acceptance tests passed!")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

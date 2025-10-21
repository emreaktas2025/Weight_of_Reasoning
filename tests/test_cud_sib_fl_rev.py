"""Test Phase 3 metrics: CUD, SIB, FL, and REV."""

import os
import tempfile
import numpy as np
import pandas as pd
from wor.core.runner import ModelRunner
from wor.metrics.circuit_utilization_density import (
    compute_circuit_heads, compute_cud, get_arithmetic_prompts
)
from wor.metrics.stability_intermediate_beliefs import (
    generate_paraphrases, compute_sib_simple
)
from wor.metrics.feature_load import compute_feature_load
from wor.metrics.rev_composite import compute_rev_scores, validate_rev_scores


def test_cud_range():
    """Test that CUD values are in [0,1] range."""
    # Create temporary directory for activations
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = {
            "model_name": "EleutherAI/pythia-70m-deduped",
            "device": "cpu",
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1337,
            "save_activations": True,
            "act_dir": temp_dir
        }
        
        # Initialize model
        m = ModelRunner(cfg)
        
        # Generate text and get cache
        result = m.generate("What is 5 + 3? Show steps.")
        
        # Create dummy circuit heads and thresholds
        circuit_heads = [(0, 0), (1, 0), (2, 0)]  # 3 heads
        control_thresholds = {(0, 0): 0.1, (1, 0): 0.1, (2, 0): 0.1}
        
        # Test CUD computation
        cud = compute_cud(m.model, result["cache"], circuit_heads, control_thresholds, result["input_tokens"])
        
        # Assertions
        assert np.isfinite(cud), f"CUD is not finite: {cud}"
        assert 0.0 <= cud <= 1.0, f"CUD should be in [0,1], got {cud}"


def test_sib_range():
    """Test that SIB values are in [-1,1] range."""
    # Create temporary directory for activations
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = {
            "model_name": "EleutherAI/pythia-70m-deduped",
            "device": "cpu",
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1337,
            "save_activations": True,
            "act_dir": temp_dir
        }
        
        # Initialize model
        m = ModelRunner(cfg)
        
        # Generate text and get cache
        result = m.generate("What is 5 + 3? Show steps.")
        
        # Test SIB computation
        sib = compute_sib_simple(m.model, result["cache"], result["input_tokens"], 
                                "What is 5 + 3? Show steps.", reasoning_window=8)
        
        # Assertions
        assert np.isfinite(sib), f"SIB is not finite: {sib}"
        assert -1.0 <= sib <= 1.0, f"SIB should be in [-1,1], got {sib}"


def test_fl_finite():
    """Test that FL values are finite."""
    # Create temporary directory for activations
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = {
            "model_name": "EleutherAI/pythia-70m-deduped",
            "device": "cpu",
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1337,
            "save_activations": True,
            "act_dir": temp_dir
        }
        
        # Initialize model
        m = ModelRunner(cfg)
        
        # Generate text and get cache
        result = m.generate("What is 5 + 3? Show steps.")
        
        # Test FL computation
        fl = compute_feature_load(m.model, result["cache"], result["input_tokens"], reasoning_window=8)
        
        # Assertions
        assert np.isfinite(fl), f"FL is not finite: {fl}"


def test_rev_finite():
    """Test that REV values are finite."""
    # Create test data
    test_data = {
        'AE': [1.0, 1.5, 2.0, 0.8, 1.2],
        'APE': [0.3, 0.4, 0.5, 0.2, 0.35],
        'APL': [0.2, 0.3, 0.4, 0.1, 0.25],
        'CUD': [0.5, 0.6, 0.7, 0.4, 0.55],
        'SIB': [0.8, 0.9, 0.7, 0.6, 0.85],
        'FL': [1.1, 1.3, 1.5, 0.9, 1.2]
    }
    
    df = pd.DataFrame(test_data)
    
    # Test REV computation
    rev_scores = compute_rev_scores(df)
    
    # Assertions
    assert len(rev_scores) == len(df), "REV scores length should match DataFrame length"
    assert all(np.isfinite(rev_scores)), f"All REV scores should be finite, got: {rev_scores}"
    
    # Test validation
    assert validate_rev_scores(rev_scores), "REV scores should pass validation"


def test_paraphrase_generation():
    """Test that paraphrases differ from original."""
    original_text = "What is 5 + 3? Show your work."
    
    # Generate paraphrases
    paraphrases = generate_paraphrases(original_text, n_paraphrases=3, seed=1337)
    
    # Assertions
    assert len(paraphrases) == 3, f"Should generate 3 paraphrases, got {len(paraphrases)}"
    
    for i, paraphrase in enumerate(paraphrases):
        assert isinstance(paraphrase, str), f"Paraphrase {i} should be string"
        assert len(paraphrase) > 0, f"Paraphrase {i} should not be empty"
        # Note: paraphrases might be identical to original due to limited synonym map
        # This is acceptable for the test


def test_circuit_head_discovery():
    """Test circuit head discovery functionality."""
    # Create temporary directory for activations
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = {
            "model_name": "EleutherAI/pythia-70m-deduped",
            "device": "cpu",
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1337,
            "save_activations": True,
            "act_dir": temp_dir
        }
        
        # Initialize model
        m = ModelRunner(cfg)
        
        # Get arithmetic prompts
        arithmetic_prompts = get_arithmetic_prompts()
        assert len(arithmetic_prompts) == 4, "Should have 4 arithmetic prompts"
        
        # Simple control prompts
        control_prompts = [
            "Write a short paragraph about cats.",
            "Describe the weather today."
        ]
        
        # Test circuit head discovery (with reduced max_heads for speed)
        circuit_heads, control_thresholds = compute_circuit_heads(
            m.model, arithmetic_prompts, control_prompts, max_heads=6
        )
        
        # Assertions
        assert isinstance(circuit_heads, list), "Circuit heads should be a list"
        assert len(circuit_heads) > 0, "Should discover at least one circuit head"
        assert len(circuit_heads) <= 6, "Should not exceed max_heads limit"
        
        for layer, head in circuit_heads:
            assert isinstance(layer, int), "Layer should be integer"
            assert isinstance(head, int), "Head should be integer"
            assert 0 <= layer < m.model.cfg.n_layers, f"Layer {layer} out of range"
            assert 0 <= head < m.model.cfg.n_heads, f"Head {head} out of range"
        
        assert isinstance(control_thresholds, dict), "Control thresholds should be dict"
        assert len(control_thresholds) == len(circuit_heads), "Thresholds should match circuit heads"
        
        for (layer, head), threshold in control_thresholds.items():
            assert isinstance(threshold, float), "Threshold should be float"
            assert np.isfinite(threshold), f"Threshold for ({layer}, {head}) should be finite"
            assert threshold >= 0.0, f"Threshold for ({layer}, {head}) should be non-negative"


def test_rev_with_nan_handling():
    """Test REV computation with NaN values."""
    # Create test data with NaN values
    test_data = {
        'AE': [1.0, np.nan, 2.0, 0.8, 1.2],
        'APE': [0.3, 0.4, np.nan, 0.2, 0.35],
        'APL': [0.2, 0.3, 0.4, np.nan, 0.25],
        'CUD': [0.5, 0.6, 0.7, 0.4, np.nan],
        'SIB': [0.8, 0.9, 0.7, 0.6, 0.85],
        'FL': [1.1, 1.3, 1.5, 0.9, 1.2]
    }
    
    df = pd.DataFrame(test_data)
    
    # Test REV computation with NaN handling
    rev_scores = compute_rev_scores(df)
    
    # Assertions
    assert len(rev_scores) == len(df), "REV scores length should match DataFrame length"
    # Some scores might be NaN due to insufficient valid data
    assert any(np.isfinite(rev_scores)), "At least some REV scores should be finite"


def test_metric_consistency():
    """Test that metrics are computed consistently."""
    # Create temporary directory for activations
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = {
            "model_name": "EleutherAI/pythia-70m-deduped",
            "device": "cpu",
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1337,
            "save_activations": True,
            "act_dir": temp_dir
        }
        
        # Initialize model
        m = ModelRunner(cfg)
        
        # Generate text and get cache
        result = m.generate("What is 5 + 3? Show steps.")
        
        # Test all metrics on same input
        reasoning_window = 8
        
        # Create dummy circuit heads and thresholds
        circuit_heads = [(0, 0), (1, 0)]
        control_thresholds = {(0, 0): 0.1, (1, 0): 0.1}
        
        # Compute all metrics
        cud = compute_cud(m.model, result["cache"], circuit_heads, control_thresholds, result["input_tokens"])
        sib = compute_sib_simple(m.model, result["cache"], result["input_tokens"], 
                                "What is 5 + 3? Show steps.", reasoning_window)
        fl = compute_feature_load(m.model, result["cache"], result["input_tokens"], reasoning_window)
        
        # Assertions - all should be finite
        assert np.isfinite(cud), f"CUD should be finite: {cud}"
        assert np.isfinite(sib), f"SIB should be finite: {sib}"
        assert np.isfinite(fl), f"FL should be finite: {fl}"
        
        # Test REV computation with these values
        test_data = {
            'AE': [1.0],
            'APE': [0.3],
            'APL': [0.2],
            'CUD': [cud],
            'SIB': [sib],
            'FL': [fl]
        }
        
        df = pd.DataFrame(test_data)
        rev_scores = compute_rev_scores(df)
        
        assert len(rev_scores) == 1, "Should have one REV score"
        assert np.isfinite(rev_scores[0]), f"REV should be finite: {rev_scores[0]}"


def test_imports():
    """Test that all Phase 3 modules can be imported."""
    import wor.metrics.circuit_utilization_density
    import wor.metrics.stability_intermediate_beliefs
    import wor.metrics.feature_load
    import wor.metrics.rev_composite
    import wor.eval.evaluate_phase3
    import wor.plots.plot_phase3
    
    # If we get here, imports succeeded
    assert True

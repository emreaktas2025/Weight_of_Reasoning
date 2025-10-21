"""Test APL metric computation."""

import os
import tempfile
import numpy as np
from wor.core.runner import ModelRunner
from wor.metrics.activation_path_length import compute_apl, compute_control_thresholds


def test_apl_computation():
    """Test APL computation on a single prompt."""
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
        result = m.generate("1+1=? Show steps.")
        
        # Check that text was generated
        assert result["text"], "No text generated"
        assert len(result["text"]) > 0, "Generated text is empty"
        
        # Create dummy control thresholds (simplified for test)
        n_layers = m.model.cfg.n_layers
        control_thresholds = {i: 0.1 for i in range(n_layers)}  # Simple threshold
        
        # Test APL computation
        apl = compute_apl(m.model, result["cache"], control_thresholds, result["input_tokens"])
        
        # Assertions
        assert np.isfinite(apl), f"APL is not finite: {apl}"
        assert apl == apl, "APL is NaN"  # Check not NaN
        assert 0.0 <= apl <= 1.0, f"APL should be in [0,1], got {apl}"
        
        # Check that number of active layers is reasonable
        # (This is implicit in the APL computation, but we can verify the range)
        assert apl >= 0.0, "APL should be non-negative"
        assert apl <= 1.0, "APL should be at most 1.0 (all layers active)"


def test_control_thresholds():
    """Test control threshold computation."""
    cfg = {
        "model_name": "EleutherAI/pythia-70m-deduped",
        "device": "cpu",
        "max_new_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1337,
        "save_activations": False,
    }
    
    # Initialize model
    m = ModelRunner(cfg)
    
    # Simple control prompts for testing
    control_prompts = [
        "Write a short paragraph about cats.",
        "Describe the weather today."
    ]
    
    # Compute control thresholds
    thresholds = compute_control_thresholds(m.model, control_prompts)
    
    # Assertions
    assert isinstance(thresholds, dict), "Thresholds should be a dictionary"
    assert len(thresholds) == m.model.cfg.n_layers, "Should have one threshold per layer"
    
    for layer, threshold in thresholds.items():
        assert isinstance(layer, int), "Layer index should be integer"
        assert isinstance(threshold, float), "Threshold should be float"
        assert np.isfinite(threshold), f"Threshold for layer {layer} should be finite"
        assert threshold >= 0.0, f"Threshold for layer {layer} should be non-negative"
    
    # Check that thresholds file was created
    assert os.path.exists("reports/control_thresholds.npz"), "Control thresholds file should be created"

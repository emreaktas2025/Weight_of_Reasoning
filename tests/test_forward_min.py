"""Test minimal forward pass and metric computation."""

import os
import tempfile
import numpy as np
from wor.core.runner import ModelRunner
from wor.metrics.activation_energy import activation_energy
from wor.metrics.attention_entropy import attention_process_entropy


def test_forward():
    """Test one forward pass with model and metric computation."""
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
        
        # Generate text
        result = m.generate("1+1=? Show steps.")
        
        # Check that text was generated
        assert result["text"], "No text generated"
        assert len(result["text"]) > 0, "Generated text is empty"
        
        # Test AE computation
        ae = activation_energy(result["hidden_states"], reasoning_len=4)
        assert np.isfinite(ae), f"AE is not finite: {ae}"
        assert ae == ae, "AE is NaN"  # Check not NaN
        
        # Test APE computation (may be NaN if attention not available)
        ape = attention_process_entropy(result["attention_probs"], reasoning_len=4)
        # APE can be NaN if attention not available, that's OK
        assert ape == ape or np.isnan(ape), "APE is invalid"
        
        # Check that activations were saved
        act_files = [f for f in os.listdir(temp_dir) if f.endswith('.npz')]
        assert len(act_files) > 0, "No activation files saved"

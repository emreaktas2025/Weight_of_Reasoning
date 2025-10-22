"""Activation capture utilities for SmolLM pipeline using PyTorch hooks."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from transformer_lens import HookedTransformer


class ActivationCapture:
    """Context manager for capturing activations during model forward pass."""
    
    def __init__(self, model: HookedTransformer, layer_names: List[str] = None):
        """
        Initialize activation capture.
        
        Args:
            model: HookedTransformer model
            layer_names: List of layer names to capture (if None, captures all)
        """
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.hooks = []
    
    def __enter__(self):
        """Enter context and register hooks."""
        self._register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and remove hooks."""
        self._remove_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def create_hook(layer_name: str):
            def hook_fn(module, input, output):
                # Store activation, handling different output types
                if isinstance(output, torch.Tensor):
                    self.activations[layer_name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    # Handle tuple outputs (e.g., attention + hidden states)
                    self.activations[layer_name] = output[0].detach().cpu()
            return hook_fn
        
        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            if self.layer_names is None or name in self.layer_names:
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activations.copy()
    
    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get activation for specific layer."""
        return self.activations.get(layer_name)


def capture_reasoning_activations(model: HookedTransformer, tokens: torch.Tensor, 
                                reasoning_len: int = 32) -> Dict[str, np.ndarray]:
    """
    Capture activations specifically for reasoning window analysis.
    
    Args:
        model: HookedTransformer model
        tokens: Input tokens
        reasoning_len: Length of reasoning window to analyze
        
    Returns:
        Dict mapping layer names to activations from reasoning window
    """
    # Define key layers to capture
    layer_names = [
        "ln_final",  # Final layer norm
        "blocks.0.hook_resid_post",  # First layer residual
        "blocks.1.hook_resid_post",   # Second layer residual
        "blocks.2.hook_resid_post",   # Third layer residual
    ]
    
    # Add more layers if model has them
    n_layers = model.cfg.n_layers
    for i in range(min(3, n_layers)):  # Capture first 3 layers
        layer_names.append(f"blocks.{i}.mlp.hook_post")
        layer_names.append(f"blocks.{i}.attn.hook_result")
    
    with ActivationCapture(model, layer_names) as capturer:
        with torch.no_grad():
            _ = model(tokens)
    
    # Extract reasoning window from activations
    reasoning_activations = {}
    activations = capturer.get_activations()
    
    for layer_name, activation in activations.items():
        if activation.ndim >= 2:  # Has sequence dimension
            # Extract reasoning window (last reasoning_len tokens)
            if reasoning_len > 0 and reasoning_len < activation.shape[1]:
                reasoning_activation = activation[:, -reasoning_len:, :]
            else:
                reasoning_activation = activation
            
            # Convert to numpy
            reasoning_activations[layer_name] = reasoning_activation.numpy()
    
    return reasoning_activations


def extract_hidden_states_from_cache(cache: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
    """
    Extract hidden states from model cache.
    
    Args:
        cache: Model activation cache
        
    Returns:
        Hidden states as numpy array, or None if not found
    """
    try:
        # Look for final layer norm or residual post
        hidden_key = None
        for key in cache.keys():
            if "ln_final" in key or "resid_post" in key:
                hidden_key = key
                break
        
        if hidden_key is None:
            # Fallback: look for any hidden state
            for key in cache.keys():
                if "resid" in key or "mlp" in key:
                    hidden_key = key
                    break
        
        if hidden_key is not None:
            hidden = cache[hidden_key].detach().cpu().numpy()
            return hidden[0] if hidden.ndim == 3 else hidden  # Remove batch dimension
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting hidden states: {e}")
        return None


def extract_attention_from_cache(cache: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
    """
    Extract attention patterns from model cache.
    
    Args:
        cache: Model activation cache
        
    Returns:
        Attention patterns as numpy array, or None if not found
    """
    try:
        # Look for attention pattern
        attn_key = None
        for key in cache.keys():
            if "pattern" in key:
                attn_key = key
                break
        
        if attn_key is not None:
            attn = cache[attn_key].detach().cpu().numpy()
            return attn[0] if attn.ndim == 4 else attn  # Remove batch dimension
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting attention patterns: {e}")
        return None


def compute_activation_energy(hidden_states: Optional[np.ndarray], reasoning_len: int = 32) -> float:
    """
    Compute activation energy from hidden states.
    
    Args:
        hidden_states: Hidden states from model
        reasoning_len: Length of reasoning window
        
    Returns:
        Activation energy value
    """
    try:
        if hidden_states is None:
            return float("nan")
        
        # Extract reasoning window
        if reasoning_len > 0 and reasoning_len < hidden_states.shape[0]:
            reasoning_states = hidden_states[-reasoning_len:]
        else:
            reasoning_states = hidden_states
        
        # Compute L2 norm (activation energy)
        energy = np.linalg.norm(reasoning_states, axis=-1)
        
        # Return mean energy across reasoning window
        return float(np.mean(energy))
        
    except Exception as e:
        print(f"Error computing activation energy: {e}")
        return float("nan")


def compute_feature_load(hidden_states: Optional[np.ndarray], reasoning_len: int = 32) -> float:
    """
    Compute feature load (sparsity proxy) from hidden states.
    
    Args:
        hidden_states: Hidden states from model
        reasoning_len: Length of reasoning window
        
    Returns:
        Feature load value
    """
    try:
        if hidden_states is None:
            return float("nan")
        
        # Extract reasoning window
        if reasoning_len > 0 and reasoning_len < hidden_states.shape[0]:
            reasoning_states = hidden_states[-reasoning_len:]
        else:
            reasoning_states = hidden_states
        
        # Compute sparsity (fraction of non-zero elements)
        non_zero_elements = np.count_nonzero(reasoning_states)
        total_elements = reasoning_states.size
        
        if total_elements == 0:
            return float("nan")
        
        sparsity = non_zero_elements / total_elements
        feature_load = 1.0 - sparsity  # Higher load = more sparse
        
        return float(feature_load)
        
    except Exception as e:
        print(f"Error computing feature load: {e}")
        return float("nan")

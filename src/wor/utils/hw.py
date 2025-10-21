"""Hardware detection and configuration utilities for Phase 4."""

import os
import json
import torch
from typing import Dict, Any
from ..core.utils import ensure_dir, save_json

try:
    from huggingface_hub import login, whoami
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def detect_hw() -> Dict[str, Any]:
    """
    Detect hardware capabilities and return optimal configuration.
    
    Returns:
        Dict with device, dtype, use_gpu, max_new_tokens, batch_size, num_workers
    """
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        device = "cuda"
        dtype = "float16"
        max_new_tokens = 128
        batch_size = 8  # Larger batch for GPU
        num_workers = 4
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        device = "cpu"
        dtype = "float32"
        max_new_tokens = 64
        batch_size = 2  # Conservative for CPU
        num_workers = 2
        gpu_memory_gb = 0.0
    
    hw_config = {
        "device": device,
        "dtype": dtype,
        "use_gpu": use_gpu,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "gpu_memory_gb": gpu_memory_gb,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    return hw_config


def log_hw_config(hw_config: Dict[str, Any], output_path: str = "reports/hw_phase4.json") -> None:
    """
    Log hardware configuration to JSON file.
    
    Args:
        hw_config: Hardware configuration dict
        output_path: Path to save the configuration
    """
    ensure_dir(os.path.dirname(output_path))
    save_json(output_path, hw_config)
    print(f"Hardware configuration logged to {output_path}")


def get_optimal_dataset_sizes(hw_config: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get optimal dataset sizes based on hardware configuration.
    
    Args:
        hw_config: Hardware configuration from detect_hw()
        base_config: Base configuration with n_cpu and n_gpu settings
        
    Returns:
        Dict with actual dataset sizes to use
    """
    if hw_config["use_gpu"]:
        size_key = "n_gpu"
    else:
        size_key = "n_cpu"
    
    optimal_sizes = {}
    for dataset_type, datasets in base_config.items():
        optimal_sizes[dataset_type] = {}
        for dataset_name, sizes in datasets.items():
            optimal_sizes[dataset_type][dataset_name] = sizes[size_key]
    
    return optimal_sizes


def get_model_config_with_hw(model_config: Dict[str, Any], hw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update model configuration with hardware-optimized settings.
    
    Args:
        model_config: Base model configuration
        hw_config: Hardware configuration from detect_hw()
        
    Returns:
        Updated model configuration
    """
    updated_config = model_config.copy()
    updated_config.update({
        "device": hw_config["device"],
        "dtype": hw_config["dtype"],
        "max_new_tokens": hw_config["max_new_tokens"],
    })
    return updated_config


def ensure_hf_auth() -> bool:
    """Ensure HuggingFace authentication is working."""
    if not HF_AVAILABLE:
        print("[HF] huggingface_hub not available")
        return False
        
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
    if tok:
        try:
            login(token=tok, add_to_git_credential=True)
            who = whoami()
            print(f"[HF] Auth OK for user: {who.get('name') or who.get('email')}")
            return True
        except Exception as e:
            print(f"[HF] Auth failed despite token: {type(e).__name__}: {e}")
    else:
        print("[HF] No token in env HUGGINGFACE_HUB_TOKEN")
    return False


def pick_dtype_device():
    """Pick appropriate device and dtype based on hardware."""
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        return torch.device("cpu"), torch.float32


def load_hf_model(model_id: str):
    """Load HuggingFace model with proper authentication and device handling."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device, dtype = pick_dtype_device()
    kw = dict(torch_dtype=dtype, trust_remote_code=True)
    
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True, **kw).to(device)
        mdl.eval()
        print(f"[HF] Successfully loaded {model_id} on {device} with {dtype}")
        return tok, mdl
    except Exception as e:
        print(f"[HF] Failed to load {model_id}: {type(e).__name__}: {e}")
        raise

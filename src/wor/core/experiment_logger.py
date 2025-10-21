"""Experiment tracking and logging for reproducibility."""

import json
import os
import time
import torch
import hashlib
import sys
import platform
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_serializable(item) for item in obj)
    return obj


def log_experiment(
    config: Dict[str, Any],
    results: Dict[str, Any],
    output_dir: str = "reports/tracking",
    experiment_name: Optional[str] = None
) -> str:
    """
    Log experiment configuration and results for reproducibility.
    
    Args:
        config: Configuration dictionary (model config, hyperparameters, etc.)
        results: Results dictionary (metrics, summaries, etc.)
        output_dir: Directory to save experiment logs
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to the saved log file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate run ID with timestamp
    run_id = time.strftime("%Y%m%d_%H%M%S")
    
    # Add experiment name to run ID if provided
    if experiment_name:
        run_id = f"{run_id}_{experiment_name}"
    
    # Compute config hash for version tracking
    config_hash = hashlib.sha1(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:8]
    
    # Gather system information
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    
    # Gather GPU information
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        }
    else:
        gpu_info = {
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory_gb": None,
            "gpu_count": 0,
            "cuda_version": None,
        }
    
    # Get git commit hash if available
    git_info = {}
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_info["commit_hash"] = git_hash
        
        # Check if there are uncommitted changes
        git_status = subprocess.check_output(
            ['git', 'status', '--short'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_info["has_uncommitted_changes"] = bool(git_status)
    except:
        git_info["commit_hash"] = None
        git_info["has_uncommitted_changes"] = None
    
    # Build experiment log
    experiment_log = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_unix": int(time.time()),
        "config_hash": config_hash,
        "config": config,
        "results": results,
        "system_info": system_info,
        "gpu_info": gpu_info,
        "git_info": git_info,
    }
    
    # Convert numpy types to JSON-serializable types
    experiment_log = _convert_to_serializable(experiment_log)
    
    # Save to JSON file
    log_path = os.path.join(output_dir, f"run_{run_id}.json")
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"🧠 Experiment logged → {log_path}")
    print(f"   Config hash: {config_hash}")
    print(f"   GPU: {gpu_info['gpu_name'] or 'CPU'}")
    
    return log_path


def load_experiment(log_path: str) -> Dict[str, Any]:
    """
    Load an experiment log from disk.
    
    Args:
        log_path: Path to the experiment log JSON file
        
    Returns:
        Experiment log dictionary
    """
    with open(log_path, "r") as f:
        return json.load(f)


def list_experiments(output_dir: str = "reports/tracking") -> list:
    """
    List all experiment logs in a directory.
    
    Args:
        output_dir: Directory containing experiment logs
        
    Returns:
        List of experiment log file paths
    """
    if not os.path.exists(output_dir):
        return []
    
    log_files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("run_") and filename.endswith(".json"):
            log_files.append(os.path.join(output_dir, filename))
    
    # Sort by timestamp (newest first)
    log_files.sort(reverse=True)
    
    return log_files


def compare_experiments(log_path1: str, log_path2: str) -> Dict[str, Any]:
    """
    Compare two experiment logs.
    
    Args:
        log_path1: Path to first experiment log
        log_path2: Path to second experiment log
        
    Returns:
        Comparison dictionary
    """
    exp1 = load_experiment(log_path1)
    exp2 = load_experiment(log_path2)
    
    comparison = {
        "exp1_id": exp1["run_id"],
        "exp2_id": exp2["run_id"],
        "config_match": exp1["config_hash"] == exp2["config_hash"],
        "gpu_match": exp1["gpu_info"]["gpu_name"] == exp2["gpu_info"]["gpu_name"],
        "time_difference_hours": (exp2["timestamp_unix"] - exp1["timestamp_unix"]) / 3600,
    }
    
    # Compare key results if available
    if "results" in exp1 and "results" in exp2:
        comparison["results_comparison"] = {}
        # Add custom comparison logic here based on your results structure
    
    return comparison


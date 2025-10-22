"""Iteration management and I/O utilities for SmolLM REV pipeline."""

import os
import json
import csv
import time
import subprocess
import platform
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


def create_iteration_folder(base_dir: str = "experiments") -> str:
    """
    Create a new iteration folder with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to new iteration folder
    """
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Find next iteration number
    existing_iterations = []
    for item in os.listdir(base_dir):
        if item.startswith("iteration_"):
            try:
                num = int(item.split("_")[1])
                existing_iterations.append(num)
            except:
                continue
    
    next_num = max(existing_iterations, default=0) + 1
    iteration_id = f"iteration_{next_num:03d}"
    iteration_path = os.path.join(base_dir, iteration_id)
    
    # Create iteration folder structure
    os.makedirs(iteration_path, exist_ok=True)
    os.makedirs(os.path.join(iteration_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(iteration_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(iteration_path, "figures"), exist_ok=True)
    
    print(f"Created iteration folder: {iteration_path}")
    return iteration_path


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()[:8]  # Short hash
    except:
        return "unknown"


def get_environment_info() -> Dict[str, Any]:
    """Get environment information for reproducibility."""
    return {
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def save_iteration_metadata(iteration_path: str, config: Dict[str, Any], 
                          models: List[str], datasets: List[str]) -> None:
    """
    Save iteration metadata for reproducibility.
    
    Args:
        iteration_path: Path to iteration folder
        config: Configuration dictionary
        models: List of model names
        datasets: List of dataset names
    """
    # Save config
    config_path = os.path.join(iteration_path, "config_used.yaml")
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save environment info
    env_info = get_environment_info()
    env_path = os.path.join(iteration_path, "env_info.json")
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    # Save timing info
    timing_info = {
        "start_time": time.time(),
        "models": models,
        "datasets": datasets,
        "config_file": config.get("config_file", "unknown")
    }
    timing_path = os.path.join(iteration_path, "timing.json")
    with open(timing_path, 'w') as f:
        json.dump(timing_info, f, indent=2)


def update_iteration_registry(iteration_id: str, results: Dict[str, Any], 
                            iteration_path: str) -> None:
    """
    Update the global iteration registry with results.
    
    Args:
        iteration_id: ID of the iteration (e.g., "iteration_001")
        results: Results dictionary with metrics
        iteration_path: Path to iteration folder
    """
    registry_path = os.path.join("experiments", "iteration_registry.csv")
    
    # Create registry if it doesn't exist
    if not os.path.exists(registry_path):
        with open(registry_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration_id", "date", "code_hash", "models", "datasets",
                "delta_auc_mean", "ci_low", "ci_high", "partial_r_mean", "comments"
            ])
    
    # Extract key metrics
    delta_auc_mean = results.get("delta_auc_mean", float("nan"))
    ci_low = results.get("ci_low", float("nan"))
    ci_high = results.get("ci_high", float("nan"))
    partial_r_mean = results.get("partial_r_mean", float("nan"))
    
    # Get metadata
    env_info = get_environment_info()
    models = results.get("models", [])
    datasets = results.get("datasets", [])
    
    # Append to registry
    with open(registry_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            iteration_id,
            env_info["timestamp"],
            env_info["git_hash"],
            ",".join(models),
            ",".join(datasets),
            delta_auc_mean,
            ci_low,
            ci_high,
            partial_r_mean,
            results.get("comments", "")
        ])


def load_iteration_registry() -> List[Dict[str, Any]]:
    """
    Load the iteration registry.
    
    Returns:
        List of iteration records
    """
    registry_path = os.path.join("experiments", "iteration_registry.csv")
    
    if not os.path.exists(registry_path):
        return []
    
    records = []
    with open(registry_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    return records


def save_results_json(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to JSON file with proper formatting.
    
    Args:
        results: Results dictionary
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def load_results_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Results dictionary, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def create_notes_template(iteration_path: str, config: Dict[str, Any], 
                        models: List[str], datasets: List[str]) -> None:
    """
    Create notes.md template for the iteration.
    
    Args:
        iteration_path: Path to iteration folder
        config: Configuration dictionary
        models: List of model names
        datasets: List of dataset names
    """
    iteration_id = os.path.basename(iteration_path)
    env_info = get_environment_info()
    
    notes_content = f"""# {iteration_id}

**Date:** {env_info['timestamp']}  
**Code Hash:** {env_info['git_hash']}  
**Models:** {', '.join(models)}  
**Datasets:** {', '.join(datasets)}  
**Temperature:** {config.get('temperature', 0.2)}  
**Windowing Mode:** {config.get('windowing_mode', 'auto_no_answer')}  
**Residualization:** {config.get('residualization', True)}  
**Seed:** {config.get('seed', 42)}  

---

## 1. Summary
- Brief overview of what this iteration was testing  
  (e.g., "First clean scaling run; fixed NaNs; tested APL/APE/SIB only.")

## 2. Results Snapshot
| Model | AUROC_REV | AUROC_BASE | ΔAUC | Partial r | Notes |
|--------|------------|------------|------|------------|--------|
| 135M | | | | | |
| 350M | | | | | |
| 1.7B | | | | | |

## 3. Observations
- Key insights about performance or anomalies.
- Examples of reasoning windows checked manually.
- Notable head behaviors (if ablations done).

## 4. To-Do / Next Steps
- [ ] (example) Run temp sweep on 350M  
- [ ] (example) Try entropy-peak window  
- [ ] (example) Generate per-layer activation heatmap

---

**End of {iteration_id}**
"""
    
    notes_path = os.path.join(iteration_path, "notes.md")
    with open(notes_path, 'w') as f:
        f.write(notes_content)
    
    print(f"Created notes template: {notes_path}")


def get_best_iteration() -> Optional[str]:
    """
    Get the best iteration based on ΔAUC.
    
    Returns:
        Iteration ID of best run, or None if no iterations exist
    """
    registry = load_iteration_registry()
    
    if not registry:
        return None
    
    # Find iteration with highest ΔAUC
    best_iteration = None
    best_delta_auc = -float('inf')
    
    for record in registry:
        try:
            delta_auc = float(record.get('delta_auc_mean', 0))
            if delta_auc > best_delta_auc:
                best_delta_auc = delta_auc
                best_iteration = record['iteration_id']
        except:
            continue
    
    return best_iteration


def list_iterations() -> List[Dict[str, Any]]:
    """
    List all iterations with summary information.
    
    Returns:
        List of iteration summaries
    """
    registry = load_iteration_registry()
    
    summaries = []
    for record in registry:
        summaries.append({
            'iteration_id': record['iteration_id'],
            'date': record['date'],
            'models': record['models'],
            'delta_auc_mean': record.get('delta_auc_mean', 'N/A'),
            'comments': record.get('comments', '')
        })
    
    return summaries

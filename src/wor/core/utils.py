"""Utility functions for seeding, I/O, and directory management."""

import os
import json
import random
import numpy as np
import torch
from typing import Any, Dict, Iterator


def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # Not all PyTorch versions support this
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("[Reproducibility] Deterministic PyTorch algorithms enabled.")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Read JSONL file and yield parsed JSON objects."""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_serializable(o: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    """Save object as JSON with proper formatting and numpy type conversion."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_serializable)
    print(f"✅ Saved JSON to {path}")

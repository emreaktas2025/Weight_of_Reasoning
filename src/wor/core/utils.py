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


def save_json(path: str, obj: Dict[str, Any]) -> None:
    """Save object as JSON with proper formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

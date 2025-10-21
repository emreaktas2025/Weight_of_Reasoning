"""Fixed dataset loaders for Phase 4b with proper labeling."""

import os
import csv
from typing import Dict, Any, List, Optional
from ..core.utils import ensure_dir, read_jsonl


def load_reasoning_dataset(name: str, n: int, seed: int = 1337) -> List[Dict[str, Any]]:
    """
    Load and properly label reasoning datasets.
    
    Args:
        name: Dataset name ('gsm8k' or 'strategyqa')
        n: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of samples with 'label' set to 'reasoning'
    """
    try:
        from datasets import load_dataset
        
        if name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="train")
        elif name == "strategyqa":
            # Try alternative dataset names
            try:
                dataset = load_dataset("wics/strategy-qa", split="validation")
            except:
                try:
                    dataset = load_dataset("ChilleD/StrategyQA", split="validation")
                except:
                    raise Exception("StrategyQA not available")
        else:
            raise ValueError(f"Unknown reasoning dataset: {name}")
        
        # Deterministic sampling
        dataset = dataset.shuffle(seed=seed)
        samples = dataset.select(range(min(n, len(dataset))))
        
        # Convert to standard format with proper labeling
        data = []
        for i, sample in enumerate(samples):
            if name == "gsm8k":
                data.append({
                    "id": f"{name}_{i}",
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": f"Solve: {sample['question']} Show steps.",
                    "label": "reasoning"  # CRITICAL: Ensure proper labeling
                })
            elif name == "strategyqa":
                data.append({
                    "id": f"{name}_{i}",
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": f"Answer: {sample['question']} Show reasoning.",
                    "label": "reasoning"  # CRITICAL: Ensure proper labeling
                })
        
        print(f"Loaded {len(data)} reasoning samples from {name}")
        return data
        
    except Exception as e:
        print(f"Failed to load {name} from HuggingFace: {e}")
        print("Falling back to local reasoning data...")
        return load_local_reasoning_fallback(name, n)


def load_control_dataset(name: str, n: int, seed: int = 1337) -> List[Dict[str, Any]]:
    """
    Load and properly label control datasets.
    
    Args:
        name: Dataset name ('wiki' or 'wikipedia')
        n: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of samples with 'label' set to 'control'
    """
    try:
        from datasets import load_dataset
        
        # Try different Wikipedia configurations
        try:
            dataset = load_dataset("wikipedia", "20220301.simple", split="train")
        except:
            try:
                dataset = load_dataset("wikimedia/wikipedia", "20220301.simple", split="train")
            except:
                raise Exception("Wikipedia not available")
        
        # Deterministic sampling
        dataset = dataset.shuffle(seed=seed)
        samples = dataset.select(range(min(n, len(dataset))))
        
        # Convert to standard format with proper labeling
        data = []
        for i, sample in enumerate(samples):
            # Take first sentence or paragraph
            text = sample["text"]
            sentences = text.split('. ')
            if sentences:
                prompt = sentences[0] + "."
            else:
                prompt = text[:200] + "..."
            
            data.append({
                "id": f"wiki_{i}",
                "question": prompt,
                "answer": "",
                "prompt": prompt,
                "label": "control"  # CRITICAL: Ensure proper labeling
            })
        
        print(f"Loaded {len(data)} control samples from Wikipedia")
        return data
        
    except Exception as e:
        print(f"Failed to load Wikipedia: {e}")
        print("Falling back to local control data...")
        return load_local_control_fallback(name, n)


def load_local_reasoning_fallback(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    """
    Load fallback reasoning data from local data/mini/ directory.
    
    Args:
        dataset_name: Name of dataset to load
        n: Number of samples to load
        
    Returns:
        List of reasoning samples with proper labeling
    """
    file_path = "data/mini/reasoning.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, creating minimal reasoning fallback")
        return create_minimal_reasoning_fallback(dataset_name, n)
    
    data = []
    for i, item in enumerate(read_jsonl(file_path)):
        if i >= n:
            break
        data.append({
            "id": f"{dataset_name}_fallback_{i}",
            "question": item["prompt"],
            "answer": "",
            "prompt": item["prompt"],
            "label": "reasoning"  # CRITICAL: Ensure proper labeling
        })
    
    print(f"Loaded {len(data)} fallback reasoning samples from {file_path}")
    return data


def load_local_control_fallback(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    """
    Load fallback control data from local data/mini/ directory.
    
    Args:
        dataset_name: Name of dataset to load
        n: Number of samples to load
        
    Returns:
        List of control samples with proper labeling
    """
    file_path = "data/mini/control.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, creating minimal control fallback")
        return create_minimal_control_fallback(dataset_name, n)
    
    data = []
    for i, item in enumerate(read_jsonl(file_path)):
        if i >= n:
            break
        data.append({
            "id": f"{dataset_name}_fallback_{i}",
            "question": item["prompt"],
            "answer": "",
            "prompt": item["prompt"],
            "label": "control"  # CRITICAL: Ensure proper labeling
        })
    
    print(f"Loaded {len(data)} fallback control samples from {file_path}")
    return data


def create_minimal_reasoning_fallback(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    """Create minimal reasoning fallback data if no files exist."""
    data = []
    for i in range(n):
        prompt = f"Solve this reasoning problem {i+1}: What is 2+2? Show steps."
        
        data.append({
            "id": f"{dataset_name}_minimal_{i}",
            "question": prompt,
            "answer": "4",
            "prompt": prompt,
            "label": "reasoning"  # CRITICAL: Ensure proper labeling
        })
    
    print(f"Created {len(data)} minimal reasoning fallback samples for {dataset_name}")
    return data


def create_minimal_control_fallback(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    """Create minimal control fallback data if no files exist."""
    data = []
    for i in range(n):
        prompt = f"This is a neutral control sentence number {i+1}."
        
        data.append({
            "id": f"{dataset_name}_minimal_{i}",
            "question": prompt,
            "answer": "",
            "prompt": prompt,
            "label": "control"  # CRITICAL: Ensure proper labeling
        })
    
    print(f"Created {len(data)} minimal control fallback samples for {dataset_name}")
    return data


def save_dataset_manifest(data: List[Dict[str, Any]], dataset_name: str, output_dir: str) -> None:
    """
    Save dataset manifest to CSV for reproducibility.
    
    Args:
        data: Dataset samples
        dataset_name: Name of dataset
        output_dir: Output directory
    """
    ensure_dir(output_dir)
    manifest_path = os.path.join(output_dir, f"{dataset_name}_manifest.csv")
    
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'question', 'answer', 'prompt', 'label'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {dataset_name} manifest to {manifest_path}")


def validate_dataset_labels(data: List[Dict[str, Any]], expected_label: str) -> bool:
    """
    Validate that all samples have the expected label.
    
    Args:
        data: Dataset samples
        expected_label: Expected label ('reasoning' or 'control')
        
    Returns:
        True if all samples have correct label
    """
    for item in data:
        if item.get("label") != expected_label:
            print(f"Warning: Sample {item['id']} has label '{item.get('label')}' instead of '{expected_label}'")
            return False
    return True

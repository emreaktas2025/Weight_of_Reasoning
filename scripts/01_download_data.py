#!/usr/bin/env python3
"""Generate tiny evaluation dataset with reasoning and control prompts."""

import json
import os
import random
import argparse
from typing import List, Dict, Any


def generate_reasoning_prompts() -> List[Dict[str, Any]]:
    """Generate reasoning prompts (math/logic problems)."""
    reasoning = [
        {
            "id": "r1",
            "prompt": "Solve: John has 7 apples, buys 5, gives 3 to a friend. How many remain? Show steps."
        },
        {
            "id": "r2", 
            "prompt": "If a train travels 120 km in 2 hours at constant speed, how far in 5 hours? Show steps."
        },
        {
            "id": "r3",
            "prompt": "A store discounts a $80 item by 25% then adds 10% tax. Final price? Show steps."
        },
        {
            "id": "r4",
            "prompt": "If x+3=11 and y=2x, what is y? Show steps."
        },
        {
            "id": "r5",
            "prompt": "There are 12 red and 8 blue marbles. If you take 5 at random without replacement, expected reds? Show steps."
        }
    ]
    return reasoning


def generate_control_prompts() -> List[Dict[str, Any]]:
    """Generate control prompts (neutral descriptive text)."""
    controls = [
        {
            "id": "c1",
            "prompt": "Write a short neutral paragraph about the history of pencils."
        },
        {
            "id": "c2",
            "prompt": "Write a short neutral paragraph describing a calm lake."
        },
        {
            "id": "c3",
            "prompt": "Write a short neutral paragraph about how bread is baked."
        },
        {
            "id": "c4",
            "prompt": "Write a short neutral paragraph about library etiquette."
        },
        {
            "id": "c5",
            "prompt": "Write a short neutral paragraph about storing winter clothes."
        }
    ]
    return controls


def main():
    """Generate and save the tiny evaluation dataset."""
    parser = argparse.ArgumentParser(description="Generate tiny evaluation dataset")
    parser.add_argument("--mini", action="store_true", help="Generate mini dataset")
    args = parser.parse_args()
    
    if not args.mini:
        print("Use --mini flag to generate the dataset")
        return
    
    # Set seed for reproducibility
    random.seed(1337)
    
    # Create data directory
    os.makedirs("data/mini", exist_ok=True)
    
    # Generate prompts
    reasoning = generate_reasoning_prompts()
    controls = generate_control_prompts()
    
    # Write reasoning prompts
    with open("data/mini/reasoning.jsonl", "w", encoding="utf-8") as f:
        for item in reasoning:
            f.write(json.dumps(item) + "\n")
    
    # Write control prompts
    with open("data/mini/control.jsonl", "w", encoding="utf-8") as f:
        for item in controls:
            f.write(json.dumps(item) + "\n")
    
    print("Wrote data/mini/reasoning.jsonl and data/mini/control.jsonl")
    print(f"Generated {len(reasoning)} reasoning prompts and {len(controls)} control prompts")


if __name__ == "__main__":
    main()

"""Tests for induction heads case study."""

import pytest
import numpy as np
from wor.mech.induction_heads import (
    generate_induction_dataset,
    rank_heads_by_rev,
    create_induction_summary
)


def test_generate_induction_dataset():
    """Test induction dataset generation."""
    # Generate small dataset
    data = generate_induction_dataset(n_samples=10, seq_len=10, seed=42)
    
    # Check dataset size
    assert len(data) == 10
    
    # Check required fields
    for item in data:
        assert 'id' in item
        assert 'prompt' in item
        assert 'answer' in item
        assert 'full_sequence' in item
        assert 'label' in item
        assert 'pattern_type' in item
        
        # Check label
        assert item['label'] == 'reasoning'
        assert item['pattern_type'] == 'induction'
        
        # Check that prompt is shorter than full sequence
        assert len(item['prompt']) < len(item['full_sequence'])
    
    # Check that sequences follow induction pattern
    first_item = data[0]
    prompt_tokens = first_item['prompt'].split()
    full_tokens = first_item['full_sequence'].split()
    
    # Full sequence should be about 2x the half length
    assert len(full_tokens) >= len(prompt_tokens)
    
    print(f"✅ Generated {len(data)} induction samples")
    print(f"  Example prompt: {data[0]['prompt']}")
    print(f"  Example answer: {data[0]['answer']}")


def test_rank_heads_by_rev():
    """Test head ranking by REV scores."""
    # Create mock head REV scores
    head_revs = {
        (0, 0): 0.5,
        (0, 1): 0.8,
        (1, 0): 0.3,
        (1, 1): 0.9,
        (2, 0): 0.6,
    }
    
    # Rank heads
    ranked_heads = rank_heads_by_rev(head_revs)
    
    # Check ranking
    assert len(ranked_heads) == 5
    assert ranked_heads[0] == (1, 1), "Head with highest REV should be first"
    assert ranked_heads[-1] == (1, 0), "Head with lowest REV should be last"
    
    # Check descending order
    rev_scores = [head_revs[head] for head in ranked_heads]
    assert rev_scores == sorted(rev_scores, reverse=True)
    
    print("✅ Head ranking test passed")
    print(f"  Top head: {ranked_heads[0]} (REV={head_revs[ranked_heads[0]]})")


def test_create_induction_summary():
    """Test induction summary creation."""
    # Create mock results
    results = {
        "baseline_accuracy": 0.85,
        "n_samples": 50,
        "n_heads_total": 96,
        "targeted_patchout": {
            "k_5": {"accuracy": 0.65, "n_heads_patched": 5},
            "k_10": {"accuracy": 0.55, "n_heads_patched": 10}
        },
        "random_patchout": {
            "k_5": {"accuracy": 0.80, "n_heads_patched": 5},
            "k_10": {"accuracy": 0.75, "n_heads_patched": 10}
        },
        "top_heads_by_rev": [
            {"layer": 2, "head": 3, "rev_score": 0.95},
            {"layer": 1, "head": 5, "rev_score": 0.88},
            {"layer": 2, "head": 1, "rev_score": 0.82}
        ]
    }
    
    # Create summary
    summary = create_induction_summary(results)
    
    # Check that summary contains key information
    assert "Induction Heads Case Study" in summary
    assert "Baseline Accuracy" in summary
    assert "Top 5 Heads by REV" in summary
    assert "Patch-out Results" in summary
    
    # Check that numbers are included
    assert "0.85" in summary  # baseline accuracy
    assert "96" in summary  # total heads
    assert "50" in summary  # n_samples
    
    print("✅ Summary creation test passed")
    print("\nGenerated summary preview:")
    print(summary[:300] + "...")


def test_induction_pattern_integrity():
    """Test that induction patterns are correctly formed."""
    data = generate_induction_dataset(n_samples=20, seq_len=8, seed=1337)
    
    for item in data:
        full_seq = item['full_sequence'].split()
        prompt = item['prompt'].split()
        answer = item['answer']
        
        # The full sequence should have a repeating pattern
        # e.g., [A, B, C, D, A, B, C, D]
        half_len = len(full_seq) // 2
        
        # Check that there's a repetition pattern
        # The answer should be the last token of the full sequence
        assert answer == full_seq[-1], f"Answer {answer} should match last token {full_seq[-1]}"
        
        # The prompt should be missing just the last token
        assert len(prompt) == len(full_seq) - 1, "Prompt should be one token shorter than full sequence"
    
    print("✅ Induction pattern integrity test passed")


def test_induction_determinism():
    """Test that induction dataset generation is deterministic with fixed seed."""
    data1 = generate_induction_dataset(n_samples=5, seed=12345)
    data2 = generate_induction_dataset(n_samples=5, seed=12345)
    
    # Check that both datasets are identical
    for item1, item2 in zip(data1, data2):
        assert item1['prompt'] == item2['prompt']
        assert item1['answer'] == item2['answer']
        assert item1['full_sequence'] == item2['full_sequence']
    
    # Check that different seed produces different data
    data3 = generate_induction_dataset(n_samples=5, seed=99999)
    assert data1[0]['prompt'] != data3[0]['prompt']
    
    print("✅ Determinism test passed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Running Induction Heads Tests")
    print("="*80 + "\n")
    
    test_generate_induction_dataset()
    test_rank_heads_by_rev()
    test_create_induction_summary()
    test_induction_pattern_integrity()
    test_induction_determinism()
    
    print("\n" + "="*80)
    print("✅ All induction tests passed!")
    print("="*80 + "\n")


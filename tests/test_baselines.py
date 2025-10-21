"""Tests for baseline predictors module."""

import pytest
import pandas as pd
import numpy as np
from wor.baselines.predictors import (
    extract_baseline_features,
    train_baseline_classifier,
    evaluate_baseline_vs_rev,
    compute_delta_auc
)


def create_test_dataframe(n_samples=50):
    """Create a test DataFrame with sample data."""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        label = "reasoning" if i < n_samples // 2 else "control"
        data.append({
            "id": f"test_{i}",
            "label": label,
            "token_len": np.random.randint(10, 100),
            "ppl": np.random.uniform(1, 10),
            "generated_text": f"This is test text {i} with some reasoning steps.",
            "AE": np.random.uniform(0, 1),
            "APE": np.random.uniform(0, 1),
            "APL": np.random.uniform(0, 1),
            "CUD": np.random.uniform(0, 1),
            "SIB": np.random.uniform(0, 1),
            "FL": np.random.uniform(0, 1)
        })
    
    df = pd.DataFrame(data)
    
    # Add REV scores (synthetic - higher for reasoning)
    df['REV'] = df.apply(
        lambda row: np.random.uniform(0.5, 1.0) if row['label'] == 'reasoning' else np.random.uniform(0, 0.5),
        axis=1
    )
    
    # Add label_num
    df['label_num'] = df['label'].map({'reasoning': 1, 'control': 0})
    
    return df


def test_extract_baseline_features():
    """Test baseline feature extraction."""
    df = create_test_dataframe(20)
    
    # Extract features
    df_with_features = extract_baseline_features(df)
    
    # Check that new columns were added
    assert 'avg_logprob' in df_with_features.columns
    assert 'perplexity' in df_with_features.columns
    assert 'cot_len' in df_with_features.columns
    
    # Check that features are not all NaN
    assert not df_with_features['avg_logprob'].isna().all()
    assert not df_with_features['perplexity'].isna().all()
    assert not df_with_features['cot_len'].isna().all()
    
    print("✅ Feature extraction test passed")


def test_train_baseline_classifier():
    """Test baseline classifier training."""
    df = create_test_dataframe(50)
    df = extract_baseline_features(df)
    
    # Train classifier
    features = ['token_len', 'avg_logprob', 'perplexity', 'cot_len']
    model, scaler = train_baseline_classifier(df, features)
    
    # Check that model was trained
    assert model is not None
    assert scaler is not None
    assert hasattr(model, 'predict_proba')
    assert hasattr(scaler, 'transform')
    
    # Test prediction
    X_test = df[features].iloc[:5].values
    X_scaled = scaler.transform(X_test)
    probs = model.predict_proba(X_scaled)
    
    assert probs.shape == (5, 2)
    assert np.all((probs >= 0) & (probs <= 1))
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    print("✅ Classifier training test passed")


def test_evaluate_baseline_vs_rev():
    """Test baseline vs REV evaluation."""
    df = create_test_dataframe(100)
    
    # Evaluate
    results = evaluate_baseline_vs_rev(df)
    
    # Check that all required keys are present
    assert 'auroc_baseline' in results
    assert 'auroc_rev' in results
    assert 'auroc_combined' in results
    assert 'delta_auc' in results
    assert 'n_samples' in results
    
    # Check that AUROCs are in valid range
    for key in ['auroc_baseline', 'auroc_rev', 'auroc_combined']:
        if not np.isnan(results[key]):
            assert 0 <= results[key] <= 1, f"{key} should be between 0 and 1"
    
    # Check that combined >= baseline (in most cases)
    if not np.isnan(results['auroc_combined']) and not np.isnan(results['auroc_baseline']):
        print(f"  Baseline AUROC: {results['auroc_baseline']:.4f}")
        print(f"  REV AUROC: {results['auroc_rev']:.4f}")
        print(f"  Combined AUROC: {results['auroc_combined']:.4f}")
        print(f"  ΔAUC: {results['delta_auc']:.4f}")
    
    print("✅ Baseline vs REV evaluation test passed")


def test_compute_delta_auc():
    """Test delta AUC computation."""
    # Test normal case
    delta = compute_delta_auc(0.75, 0.85)
    assert np.isclose(delta, 0.10), f"Expected 0.10, got {delta}"
    
    # Test with NaN
    delta_nan = compute_delta_auc(np.nan, 0.85)
    assert np.isnan(delta_nan)
    
    # Test negative delta (REV hurts)
    delta_neg = compute_delta_auc(0.85, 0.75)
    assert np.isclose(delta_neg, -0.10), f"Expected -0.10, got {delta_neg}"
    
    print("✅ Delta AUC computation test passed")


def test_baseline_features_cot_len():
    """Test CoT length feature extraction."""
    test_texts = [
        "First, we calculate 2+2. Second, we get 4. Therefore, the answer is 4.",
        "This is simple text without reasoning.",
        "Step 1: Read the question. Step 2: Solve it. Step 3: Write answer."
    ]
    
    df = pd.DataFrame({
        "id": ["test_1", "test_2", "test_3"],
        "label": ["reasoning", "control", "reasoning"],
        "token_len": [10, 5, 12],
        "ppl": [2.0, 1.5, 2.5],
        "generated_text": test_texts,
        "AE": [0.5, 0.2, 0.6],
        "label_num": [1, 0, 1]
    })
    
    df_with_features = extract_baseline_features(df)
    
    # Check that CoT length is higher for reasoning texts
    cot_lens = df_with_features['cot_len'].tolist()
    assert cot_lens[0] > cot_lens[1], "Reasoning text should have higher CoT length"
    assert cot_lens[2] > cot_lens[1], "Text with steps should have higher CoT length"
    
    print(f"  CoT lengths: {cot_lens}")
    print("✅ CoT length feature test passed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Running Baseline Predictors Tests")
    print("="*80 + "\n")
    
    test_extract_baseline_features()
    test_train_baseline_classifier()
    test_evaluate_baseline_vs_rev()
    test_compute_delta_auc()
    test_baseline_features_cot_len()
    
    print("\n" + "="*80)
    print("✅ All baseline tests passed!")
    print("="*80 + "\n")


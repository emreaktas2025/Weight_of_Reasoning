#!/usr/bin/env python3
"""Sanity check script for DeepSeek-R1-Distill-Llama-8B reasoning tag parsing.

This script loads the model, generates a response to a simple math question,
parses the reasoning tags, and prints CUD metrics for reasoning vs response portions.
"""

import sys
import os
from pathlib import Path

# Disable hf_transfer early if not available (must be before any huggingface imports)
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("⚠️  hf_transfer enabled but not installed, disabling it...")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wor.utils.parser import parse_reasoning_trace, find_token_ranges_for_reasoning
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def main():
    """Run sanity check."""
    print("=" * 80)
    print("DeepSeek-R1-Distill-Llama-8B Sanity Check")
    print("=" * 80)
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit = True
    
    print(f"\n1. Loading model: {model_name}")
    print(f"   Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   4-bit quantization: {use_4bit}")
    
    # Check available memory
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"   System RAM: {ram_gb:.1f} GB total, {ram_available_gb:.1f} GB available")
        if ram_available_gb < 8:
            print("   ⚠️  Warning: Low RAM available, model loading may fail")
    except ImportError:
        pass  # psutil not available, skip check
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float16 if device == "cuda" else torch.float32,  # Use dtype instead of torch_dtype
        }
        
        if use_4bit and device == "cuda":
            try:
                import bitsandbytes as bnb
                print(f"   bitsandbytes version: {bnb.__version__}")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                print("   Using 4-bit quantization")
            except ImportError:
                print("   Warning: bitsandbytes not available, loading without quantization")
                use_4bit = False
            except Exception as e:
                print(f"   ⚠️  Error setting up quantization: {e}")
                print("   Falling back to loading without quantization...")
                use_4bit = False
                # Remove quantization config if it was added
                model_kwargs.pop("quantization_config", None)
        
        # Load model with device_map for automatic GPU placement
        # Use device_map="auto" to load directly to GPU and avoid CPU memory issues
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            # Don't set max_memory with quantization - let bitsandbytes handle it
            if not use_4bit:
                model_kwargs["max_memory"] = {0: "20GiB"}  # Reserve some GPU memory
            # Always use low_cpu_mem_usage to minimize CPU RAM usage during loading
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            model_kwargs["device_map"] = None
        
        print("   Loading model (this may take a few minutes)...")
        print("   This step downloads ~16GB if not cached...")
        
        # Clear GPU cache before loading
        if device == "cuda":
            torch.cuda.empty_cache()
            print(f"   GPU memory before loading: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Try loading with explicit error handling
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "bad_alloc" in str(e).lower():
                print(f"   ❌ Memory error during loading: {e}")
                print("   Trying alternative loading method...")
                # Try without device_map, load to CPU first then move
                if "device_map" in model_kwargs:
                    del model_kwargs["device_map"]
                if "max_memory" in model_kwargs:
                    del model_kwargs["max_memory"]
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                if device == "cuda":
                    model = model.to(device)
            else:
                raise
        model.eval()
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test prompt
    prompt = "What is 25 * 25? Show your work."
    print(f"\n2. Generating response to: '{prompt}'")
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move inputs to device (model should already be on device if device_map="auto")
        if device == "cuda" and model_kwargs.get("device_map") != "auto":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if not hasattr(model, "device") or next(model.parameters()).device.type != "cuda":
                model = model.to(device)
        elif device == "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"   ✅ Generation complete")
        print(f"\n   Full output:\n   {full_text}")
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Parse reasoning trace
    print(f"\n3. Parsing reasoning tags...")
    parsed = parse_reasoning_trace(generated_text)
    
    if parsed["has_reasoning"]:
        print(f"   ✅ Found reasoning tags")
        print(f"\n   Reasoning content (first 200 chars):")
        reasoning_preview = parsed["reasoning_content"][:200]
        print(f"   {reasoning_preview}...")
        print(f"\n   Final response:")
        print(f"   {parsed['final_response']}")
    else:
        print(f"   ⚠️  No reasoning tags found")
        print(f"   Full generated text: {generated_text[:200]}...")
        return 0
    
    # Find token ranges
    print(f"\n4. Finding token ranges...")
    input_tokens_length = inputs['input_ids'].shape[1]
    full_tokens = outputs[0].tolist()
    
    token_ranges = find_token_ranges_for_reasoning(
        generated_text,
        parsed["reasoning_content"],
        parsed["final_response"],
        tokenizer,
        input_tokens_length,
        full_tokens
    )
    
    if token_ranges:
        reasoning_range = token_ranges.get("reasoning_range")
        response_range = token_ranges.get("response_range")
        
        if reasoning_range:
            print(f"   ✅ Reasoning tokens: {reasoning_range[0]} to {reasoning_range[1]} ({reasoning_range[1] - reasoning_range[0]} tokens)")
        if response_range:
            print(f"   ✅ Response tokens: {response_range[0]} to {response_range[1]} ({response_range[1] - response_range[0]} tokens)")
    else:
        print(f"   ⚠️  Could not determine token ranges")
    
    print(f"\n   ⚠️  Note: CUD computation requires TransformerLens which doesn't support DeepSeek models.")
    print(f"   For full metric computation, we'll need to use a different approach or")
    print(f"   implement metric computation using transformers library directly.")
    
    print(f"\n" + "=" * 80)
    print("Sanity check complete!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Model loaded: ✅")
    print(f"  - Generation: ✅")
    print(f"  - Reasoning tags found: {'✅' if parsed['has_reasoning'] else '❌'}")
    if parsed['has_reasoning']:
        print(f"  - Reasoning content length: {len(parsed['reasoning_content'])} chars")
        print(f"  - Response content length: {len(parsed['final_response'])} chars")
    print(f"  - Token ranges: {'✅' if token_ranges else '❌'}")
    print("\nNext steps:")
    print("  1. Verify that reasoning tags are being parsed correctly")
    print("  2. Implement metric computation using transformers (since TransformerLens")
    print("     doesn't support DeepSeek models)")
    print("  3. Update evaluation pipeline to use transformers-based runner for DeepSeek")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


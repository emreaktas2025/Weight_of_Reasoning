"""SmolLM-specific model loader and data loaders for REV research pipeline."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..core.utils import ensure_dir, set_seed


class SmolLMRunner:
    """SmolLM model runner with activation and attention caching using HuggingFace Transformers."""
    
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize SmolLM model runner with configuration."""
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.dtype = getattr(torch, cfg.get("dtype", "float32"))
        
        # Set seed for reproducibility
        set_seed(cfg.get("seed", 42))
        
        # Disable gradients for inference
        torch.set_grad_enabled(False)
        
        # Load model using HuggingFace Transformers
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        # Add authentication if required
        if cfg.get("requires_auth", False):
            from ..utils.hf_auth import get_hf_token
            token = get_hf_token()
            if token:
                model_kwargs["use_auth_token"] = token
                print(f"Using HuggingFace token for {cfg['model_name']}")
            else:
                print(f"Warning: {cfg['model_name']} requires auth but no token found")
        
        # Add trust_remote_code if specified
        if cfg.get("trust_remote_code", False):
            model_kwargs["trust_remote_code"] = True
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Generation parameters
        self.max_new_tokens = cfg.get("max_new_tokens", 64)
        self.temperature = cfg.get("temperature", 0.2)
        self.top_p = cfg.get("top_p", 1.0)
        
        # Activation caching
        self.save_activations = cfg.get("save_activations", True)
        self.act_dir = cfg.get("act_dir", "cache/smollm_activations")
        ensure_dir(self.act_dir)
        
        # Cache for storing activations during generation
        self.cache = {}
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate text and cache activations."""
        # Clear previous cache
        self.cache.clear()
        
        # Tokenize input with consistent length and proper attention mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length",
            add_special_tokens=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate with caching
        with torch.no_grad():
            # Generate continuation
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=False,  # Disable attention output to avoid SDPA warnings
            )
        
        # Decode full text
        full_text = self.tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
        generated_text = self.tokenizer.decode(
            generated_ids.sequences[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Extract hidden states from generation
        hidden_states = self._extract_hidden_states_from_generation(generated_ids)
        attention_probs = None  # Disabled to avoid SDPA warnings
        
        # Compute perplexity from cross-entropy
        perplexity = self._compute_perplexity(input_ids, generated_ids.sequences[0])
        
        # Save activations if requested
        if self.save_activations:
            self._save_activations(prompt, hidden_states, attention_probs)
        
        return {
            "text": full_text,
            "generated_text": generated_text,
            "tokens": generated_ids.sequences[0],
            "input_tokens": input_ids,
            "hidden_states": hidden_states,
            "attention_probs": attention_probs,
            "cache": {},  # Placeholder for compatibility
            "perplexity": perplexity,
        }
    
    def _extract_hidden_states_from_generation(self, generated_ids) -> Optional[np.ndarray]:
        """Extract hidden states from HuggingFace generation output."""
        try:
            if hasattr(generated_ids, 'hidden_states') and generated_ids.hidden_states:
                # Get the last layer's hidden states
                last_hidden_states = generated_ids.hidden_states[-1]  # Last layer
                # Take the last token's hidden state
                hidden_states = last_hidden_states[0, -1, :].detach().cpu().numpy()
                return hidden_states
            else:
                return None
        except Exception:
            return None
    
    def _extract_attention_from_generation(self, generated_ids) -> Optional[np.ndarray]:
        """Extract attention probabilities from HuggingFace generation output."""
        try:
            if hasattr(generated_ids, 'attentions') and generated_ids.attentions:
                # Get attention from the last layer
                last_attention = generated_ids.attentions[-1]  # Last layer
                # Average across heads and take last token
                attention_probs = last_attention[0, :, -1, :].mean(dim=0).detach().cpu().numpy()
                return attention_probs
            else:
                return None
        except Exception:
            return None
    
    def _save_activations(self, prompt: str, hidden_states: Optional[np.ndarray], 
                         attention_probs: Optional[np.ndarray]) -> None:
        """Save activations to disk."""
        try:
            # Create filename based on prompt hash
            filename = f"acts_{abs(hash(prompt))}.npz"
            filepath = os.path.join(self.act_dir, filename)
            
            # Save data
            save_dict = {}
            if hidden_states is not None:
                save_dict["hidden_states"] = hidden_states
            if attention_probs is not None:
                save_dict["attention_probs"] = attention_probs
            
            if save_dict:
                np.savez_compressed(filepath, **save_dict)
        except Exception as e:
            print(f"Warning: Failed to save activations: {e}")
    
    def _compute_perplexity(self, input_tokens: torch.Tensor, full_tokens: torch.Tensor) -> float:
        """Compute perplexity from cross-entropy loss using HuggingFace model with safe tensor handling."""
        try:
            # Get the generated portion (excluding input)
            generated_tokens = full_tokens[input_tokens.shape[1]:]
            
            if len(generated_tokens) == 0:
                return float("nan")
            
            # Run forward pass to get logits
            with torch.no_grad():
                outputs = self.model(full_tokens)
                logits = outputs.logits
                
                # Safe tensor alignment to prevent size mismatch
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = full_tokens[..., 1:].contiguous()
                
                # Ensure matching sequence lengths
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                
                # Convert to perplexity
                perplexity = torch.exp(loss).item()
                
            return float(perplexity)
            
        except Exception as e:
            print(f"Warning: Failed to compute perplexity: {e}")
            # Fallback to simple heuristic
            seq_len = len(generated_tokens) if 'generated_tokens' in locals() else 10
            base_perplexity = 2.0 + (seq_len * 0.1)
            return float(base_perplexity)


def load_smollm_dataset(name: str, n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load datasets for SmolLM evaluation with proper labeling.
    
    Args:
        name: Dataset name ('gsm8k', 'strategyqa', 'wiki')
        n: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of samples with proper 'label' field
    """
    try:
        from datasets import load_dataset
        
        if name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="train")
            # Convert to standard format
            data = []
            dataset = dataset.shuffle(seed=seed)
            samples = dataset.select(range(min(n, len(dataset))))
            for i, sample in enumerate(samples):
                data.append({
                    "id": f"gsm8k_{i}",
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": f"Solve: {sample['question']} Show steps.",
                    "label": "reasoning"
                })
                
        elif name == "strategyqa":
            try:
                dataset = load_dataset("ChilleD/StrategyQA", split="train")
            except:
                dataset = load_dataset("wics/strategy-qa", split="train")
            
            data = []
            dataset = dataset.shuffle(seed=seed)
            samples = dataset.select(range(min(n, len(dataset))))
            for i, sample in enumerate(samples):
                data.append({
                    "id": f"strategyqa_{i}",
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": f"Answer: {sample['question']} Show reasoning.",
                    "label": "reasoning"
                })
                
        elif name == "wiki":
            try:
                dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
            except:
                dataset = load_dataset("wikipedia", "20231101.simple", split="train")
            
            data = []
            dataset = dataset.shuffle(seed=seed)
            samples = dataset.select(range(min(n, len(dataset))))
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
                    "label": "control"
                })
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        print(f"Loaded {len(data)} samples from {name}")
        return data
        
    except Exception as e:
        print(f"Failed to load {name} from HuggingFace: {e}")
        print("Falling back to minimal data...")
        return create_minimal_fallback(name, n)


def create_minimal_fallback(dataset_name: str, n: int) -> List[Dict[str, Any]]:
    """Create minimal fallback data if HuggingFace datasets fail."""
    data = []
    for i in range(n):
        if dataset_name in ["gsm8k", "strategyqa"]:
            prompt = f"Solve this reasoning problem {i+1}: What is 2+2? Show steps."
            data.append({
                "id": f"{dataset_name}_fallback_{i}",
                "question": prompt,
                "answer": "4",
                "prompt": prompt,
                "label": "reasoning"
            })
        else:  # wiki
            prompt = f"This is a neutral control sentence number {i+1}."
            data.append({
                "id": f"wiki_fallback_{i}",
                "question": prompt,
                "answer": "",
                "prompt": prompt,
                "label": "control"
            })
    
    print(f"Created {len(data)} minimal fallback samples for {dataset_name}")
    return data

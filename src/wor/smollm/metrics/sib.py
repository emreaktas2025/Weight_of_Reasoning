"""Stability of Intermediate Beliefs (SIB) metric computation for SmolLM pipeline."""

import random
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from transformer_lens import HookedTransformer
from scipy.spatial.distance import cosine


def get_synonym_map() -> Dict[str, List[str]]:
    """
    Get synonym mapping for paraphrase generation.
    
    Returns:
        Dict mapping words to lists of synonyms
    """
    return {
        # Math/numbers
        "number": ["value", "amount", "quantity", "figure"],
        "calculate": ["compute", "determine", "find", "work out"],
        "solve": ["resolve", "figure out", "determine", "find"],
        "answer": ["result", "solution", "outcome", "response"],
        "problem": ["question", "challenge", "task", "exercise"],
        "step": ["stage", "phase", "part", "process"],
        "show": ["demonstrate", "display", "present", "illustrate"],
        "work": ["calculation", "computation", "process", "method"],
        
        # Actions
        "buy": ["purchase", "acquire", "get", "obtain"],
        "give": ["provide", "offer", "hand over", "deliver"],
        "take": ["remove", "subtract", "get", "obtain"],
        "add": ["include", "combine", "sum", "total"],
        "subtract": ["remove", "take away", "deduct", "minus"],
        "multiply": ["times", "product", "scale", "increase"],
        "divide": ["split", "share", "separate", "partition"],
        
        # Objects
        "friend": ["pal", "buddy", "companion", "acquaintance"],
        "apple": ["fruit", "red fruit", "snack", "food"],
        "train": ["locomotive", "railway", "transport", "vehicle"],
        "store": ["shop", "market", "retailer", "business"],
        "item": ["product", "object", "thing", "article"],
        "price": ["cost", "value", "amount", "fee"],
        "discount": ["reduction", "savings", "markdown", "deal"],
        "tax": ["levy", "duty", "charge", "fee"],
        
        # Connectors
        "thus": ["therefore", "so", "hence", "consequently"],
        "so": ["therefore", "thus", "hence", "consequently"],
        "because": ["since", "as", "due to", "owing to"],
        "therefore": ["thus", "so", "hence", "consequently"],
        "however": ["but", "yet", "nevertheless", "nonetheless"],
        "moreover": ["furthermore", "additionally", "besides", "also"],
        "first": ["initially", "to begin", "firstly", "at first"],
        "second": ["next", "then", "secondly", "afterward"],
        "finally": ["lastly", "ultimately", "in conclusion", "at last"],
    }


def generate_paraphrase(prompt: str, n_paraphrases: int = 3) -> List[str]:
    """
    Generate paraphrases of a prompt by substituting synonyms.
    
    Args:
        prompt: Original prompt text
        n_paraphrases: Number of paraphrases to generate
        
    Returns:
        List of paraphrased prompts
    """
    synonym_map = get_synonym_map()
    words = prompt.split()
    paraphrases = []
    
    for _ in range(n_paraphrases):
        paraphrased_words = []
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in synonym_map:
                # Randomly choose synonym or keep original
                if random.random() < 0.3:  # 30% chance to substitute
                    synonym = random.choice(synonym_map[word_lower])
                    paraphrased_words.append(synonym)
                else:
                    paraphrased_words.append(word)
            else:
                paraphrased_words.append(word)
        
        paraphrase = ' '.join(paraphrased_words)
        paraphrases.append(paraphrase)
    
    return paraphrases


def compute_sib_simple(model: HookedTransformer, cache: Dict[str, torch.Tensor], 
                      input_tokens: torch.Tensor, prompt: str, reasoning_len: int = 32) -> float:
    """
    Compute simplified SIB metric using activation similarity.
    
    Args:
        model: HookedTransformer model
        cache: Activation cache from original run
        input_tokens: Input tokens
        prompt: Original prompt
        reasoning_len: Length of reasoning window
        
    Returns:
        SIB value (stability of intermediate beliefs)
    """
    try:
        # Generate paraphrases
        paraphrases = generate_paraphrase(prompt, n_paraphrases=3)
        
        # Get original activations from reasoning window
        original_activations = _extract_reasoning_activations(cache, reasoning_len)
        if original_activations is None:
            return float("nan")
        
        similarities = []
        
        for paraphrase in paraphrases:
            try:
                # Tokenize paraphrase
                paraphrase_tokens = model.to_tokens(paraphrase, prepend_bos=True)
                
                # Run paraphrase through model
                with torch.no_grad():
                    _, paraphrase_cache = model.run_with_cache(
                        paraphrase_tokens, 
                        return_type="logits"
                    )
                
                # Extract reasoning activations from paraphrase
                paraphrase_activations = _extract_reasoning_activations(paraphrase_cache, reasoning_len)
                if paraphrase_activations is None:
                    continue
                
                # Compute cosine similarity
                if original_activations.shape == paraphrase_activations.shape:
                    # Flatten for comparison
                    orig_flat = original_activations.flatten()
                    para_flat = paraphrase_activations.flatten()
                    
                    # Compute cosine similarity
                    similarity = 1 - cosine(orig_flat, para_flat)
                    similarities.append(similarity)
                
            except Exception as e:
                print(f"Error processing paraphrase: {e}")
                continue
        
        # SIB is the mean similarity across paraphrases
        if similarities:
            sib = np.mean(similarities)
            return float(sib)
        else:
            return float("nan")
            
    except Exception as e:
        print(f"Error computing SIB: {e}")
        return float("nan")


def _extract_reasoning_activations(cache: Dict[str, torch.Tensor], reasoning_len: int) -> Optional[np.ndarray]:
    """
    Extract activations from reasoning window.
    
    Args:
        cache: Model activation cache
        reasoning_len: Length of reasoning window
        
    Returns:
        Activations from reasoning window, or None if not found
    """
    try:
        # Look for hidden states in cache
        hidden_key = None
        for key in cache.keys():
            if "resid_post" in key or "ln_final" in key:
                hidden_key = key
                break
        
        if hidden_key is None:
            return None
        
        # Extract activations
        activations = cache[hidden_key].detach().cpu().numpy()
        if activations.ndim == 3:  # (batch, seq_len, hidden_dim)
            activations = activations[0]  # Remove batch dimension
        
        # Extract reasoning window (last reasoning_len tokens)
        if reasoning_len > 0 and reasoning_len < activations.shape[0]:
            reasoning_activations = activations[-reasoning_len:]
        else:
            reasoning_activations = activations
        
        return reasoning_activations
        
    except Exception as e:
        print(f"Error extracting reasoning activations: {e}")
        return None

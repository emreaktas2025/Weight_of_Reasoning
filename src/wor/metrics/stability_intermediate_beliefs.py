"""Stability of Intermediate Beliefs (SIB) metric computation via paraphrase perturbations."""

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
        "if": ["when", "provided that", "assuming", "in case"],
        "then": ["next", "afterward", "subsequently", "following"],
        "and": ["plus", "also", "additionally", "furthermore"],
        "or": ["alternatively", "otherwise", "either", "instead"],
        
        # Time/space
        "hour": ["hr", "time period", "duration", "session"],
        "minute": ["min", "moment", "brief time", "instant"],
        "distance": ["length", "space", "range", "extent"],
        "speed": ["velocity", "rate", "pace", "tempo"],
        "constant": ["steady", "fixed", "unchanging", "stable"],
        
        # Colors
        "red": ["crimson", "scarlet", "ruby", "cherry"],
        "blue": ["azure", "navy", "cobalt", "sapphire"],
        
        # Common words
        "the": ["a", "an", "this", "that"],
        "a": ["an", "one", "some", "any"],
        "is": ["are", "was", "were", "be"],
        "are": ["is", "was", "were", "be"],
        "has": ["have", "had", "possesses", "contains"],
        "have": ["has", "had", "possess", "contain"],
    }


def generate_paraphrases(text: str, n_paraphrases: int = 3, seed: int = 1337) -> List[str]:
    """
    Generate paraphrases using rule-based transformations.
    
    Args:
        text: Original text to paraphrase
        n_paraphrases: Number of paraphrases to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of paraphrased texts
    """
    random.seed(seed)
    np.random.seed(seed)
    
    synonym_map = get_synonym_map()
    paraphrases = []
    
    for i in range(n_paraphrases):
        # Start with original text
        paraphrased = text
        
        # Apply transformations with different random seeds
        random.seed(seed + i)
        np.random.seed(seed + i)
        
        # 1. Synonym replacement
        paraphrased = _apply_synonym_replacement(paraphrased, synonym_map)
        
        # 2. Clause shuffling (if multiple sentences)
        paraphrased = _apply_clause_shuffle(paraphrased)
        
        # 3. Connector substitution
        paraphrased = _apply_connector_substitution(paraphrased)
        
        paraphrases.append(paraphrased)
    
    return paraphrases


def _apply_synonym_replacement(text: str, synonym_map: Dict[str, List[str]]) -> str:
    """Apply synonym replacement to text."""
    words = text.split()
    replaced_words = []
    
    for word in words:
        # Clean word (remove punctuation for lookup)
        clean_word = word.lower().strip(".,!?;:")
        
        if clean_word in synonym_map:
            # Randomly choose a synonym
            synonyms = synonym_map[clean_word]
            chosen_synonym = random.choice(synonyms)
            
            # Preserve original capitalization and punctuation
            if word[0].isupper():
                chosen_synonym = chosen_synonym.capitalize()
            
            # Add back punctuation
            if word.endswith((".", ",", "!", "?", ";")):
                chosen_synonym += word[-1]
            
            replaced_words.append(chosen_synonym)
        else:
            replaced_words.append(word)
    
    return " ".join(replaced_words)


def _apply_clause_shuffle(text: str) -> str:
    """Apply clause shuffling if multiple sentences exist."""
    sentences = text.split(". ")
    
    if len(sentences) > 1:
        # Shuffle all but the last sentence (to preserve structure)
        if len(sentences) > 2:
            middle_sentences = sentences[1:-1]
            random.shuffle(middle_sentences)
            sentences = [sentences[0]] + middle_sentences + [sentences[-1]]
        else:
            # For exactly 2 sentences, sometimes swap them
            if random.random() < 0.3:  # 30% chance to swap
                sentences = [sentences[1], sentences[0]]
    
    return ". ".join(sentences)


def _apply_connector_substitution(text: str) -> str:
    """Apply connector word substitution."""
    connector_map = {
        "so": ["therefore", "thus", "hence"],
        "therefore": ["so", "thus", "hence"],
        "thus": ["so", "therefore", "hence"],
        "because": ["since", "as", "due to"],
        "since": ["because", "as", "due to"],
        "if": ["when", "provided that", "assuming"],
        "when": ["if", "provided that", "assuming"],
        "and": ["plus", "also", "additionally"],
        "or": ["alternatively", "otherwise", "instead"],
    }
    
    words = text.split()
    replaced_words = []
    
    for word in words:
        clean_word = word.lower().strip(".,!?;:")
        
        if clean_word in connector_map and random.random() < 0.4:  # 40% chance
            alternatives = connector_map[clean_word]
            chosen = random.choice(alternatives)
            
            # Preserve capitalization
            if word[0].isupper():
                chosen = chosen.capitalize()
            
            # Preserve punctuation
            if word.endswith((".", ",", "!", "?", ";")):
                chosen += word[-1]
            
            replaced_words.append(chosen)
        else:
            replaced_words.append(word)
    
    return " ".join(replaced_words)


def extract_mid_layer_hidden_states(model: HookedTransformer, cache: Dict[str, torch.Tensor], 
                                   input_tokens: torch.Tensor, reasoning_window: int = 24) -> Optional[np.ndarray]:
    """
    Extract hidden states from mid-layer for reasoning window.
    
    Args:
        model: HookedTransformer model
        cache: Cache from forward pass
        input_tokens: Original input tokens
        reasoning_window: Number of tokens to consider as reasoning window
        
    Returns:
        Hidden states for reasoning window, shape (window_size, hidden_dim)
    """
    try:
        # Get mid-layer index
        n_layers = model.cfg.n_layers
        mid_layer = n_layers // 2
        
        # Look for mid-layer hidden states in cache
        mid_layer_key = f"blocks.{mid_layer}.hook_resid_post"
        if mid_layer_key not in cache:
            # Fallback: look for any mid-layer activation
            for key in cache.keys():
                if f"blocks.{mid_layer}." in key and "resid" in key:
                    mid_layer_key = key
                    break
        
        if mid_layer_key not in cache:
            return None
        
        # Extract hidden states
        hidden_states = cache[mid_layer_key].detach().cpu().numpy()  # (batch, seq_len, hidden_dim)
        hidden_states = hidden_states[0]  # Remove batch dimension
        
        # Get reasoning window (last N tokens excluding final one)
        seq_len = hidden_states.shape[0]
        if reasoning_window + 1 >= seq_len:
            # If window is too large, use all tokens except last
            window = hidden_states[:-1, :] if seq_len > 1 else hidden_states
        else:
            # Take reasoning window from the end, excluding final token
            window = hidden_states[-(reasoning_window + 1):-1, :]
        
        return window
        
    except Exception as e:
        print(f"Error extracting mid-layer hidden states: {e}")
        return None


def compute_sib(model: HookedTransformer, cache: Dict[str, torch.Tensor],
                input_tokens: torch.Tensor, original_text: str,
                reasoning_window: int = 24, n_paraphrases: int = 3) -> float:
    """
    Compute Stability of Intermediate Beliefs (SIB) for a single prompt.
    
    Args:
        model: HookedTransformer model
        cache: Cache from original forward pass
        input_tokens: Original input tokens
        original_text: Original prompt text
        reasoning_window: Number of tokens to consider as reasoning window
        n_paraphrases: Number of paraphrases to generate
        
    Returns:
        SIB value (mean cosine similarity) ∈ [-1,1]
    """
    try:
        # Extract original hidden states
        original_hidden = extract_mid_layer_hidden_states(model, cache, input_tokens, reasoning_window)
        
        if original_hidden is None or original_hidden.size == 0:
            return float("nan")
        
        # Generate paraphrases
        paraphrases = generate_paraphrases(original_text, n_paraphrases)
        
        # Compute similarities
        similarities = []
        
        for paraphrase in paraphrases:
            # Generate text with paraphrase
            result = model.generate(
                paraphrase,
                max_new_tokens=model.cfg.max_new_tokens if hasattr(model.cfg, 'max_new_tokens') else 64,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                verbose=False,
            )
            
            # Get tokens for paraphrase
            paraphrase_tokens = model.to_tokens(paraphrase, prepend_bos=True)
            
            # Run forward pass to get cache
            with torch.no_grad():
                _, paraphrase_cache = model.run_with_cache(
                    paraphrase_tokens,
                    return_type="logits"
                )
            
            # Extract paraphrase hidden states
            paraphrase_hidden = extract_mid_layer_hidden_states(
                model, paraphrase_cache, paraphrase_tokens, reasoning_window
            )
            
            if paraphrase_hidden is None or paraphrase_hidden.size == 0:
                continue
            
            # Compute mean-pooled representations
            original_mean = np.mean(original_hidden, axis=0)
            paraphrase_mean = np.mean(paraphrase_hidden, axis=0)
            
            # Compute cosine similarity
            # Convert to 1D arrays and handle potential zero vectors
            orig_flat = original_mean.flatten()
            para_flat = paraphrase_mean.flatten()
            
            if np.linalg.norm(orig_flat) == 0 or np.linalg.norm(para_flat) == 0:
                similarity = 0.0
            else:
                similarity = 1 - cosine(orig_flat, para_flat)
            
            similarities.append(similarity)
        
        # SIB is mean of similarities
        if len(similarities) == 0:
            return float("nan")
        
        sib = np.mean(similarities)
        return float(sib)
        
    except Exception as e:
        print(f"Error computing SIB: {e}")
        return float("nan")


def compute_sib_simple(model: HookedTransformer, cache: Dict[str, torch.Tensor],
                      input_tokens: torch.Tensor, original_text: str,
                      reasoning_window: int = 24) -> float:
    """
    Simplified SIB computation that doesn't require model generation.
    Uses pre-computed paraphrases and focuses on input token stability.
    
    Args:
        model: HookedTransformer model
        cache: Cache from original forward pass
        input_tokens: Original input tokens
        original_text: Original prompt text
        reasoning_window: Number of tokens to consider as reasoning window
        
    Returns:
        SIB value (mean cosine similarity) ∈ [-1,1]
    """
    try:
        # Extract original hidden states
        original_hidden = extract_mid_layer_hidden_states(model, cache, input_tokens, reasoning_window)
        
        if original_hidden is None or original_hidden.size == 0:
            return float("nan")
        
        # Generate paraphrases
        paraphrases = generate_paraphrases(original_text, n_paraphrases=3)
        
        # Compute similarities using input token representations only
        similarities = []
        
        for paraphrase in paraphrases:
            # Tokenize paraphrase
            paraphrase_tokens = model.to_tokens(paraphrase, prepend_bos=True)
            
            # Run forward pass to get hidden states
            with torch.no_grad():
                _, paraphrase_cache = model.run_with_cache(
                    paraphrase_tokens,
                    return_type="logits"
                )
            
            # Extract paraphrase hidden states (same reasoning window)
            paraphrase_hidden = extract_mid_layer_hidden_states(
                model, paraphrase_cache, paraphrase_tokens, reasoning_window
            )
            
            if paraphrase_hidden is None or paraphrase_hidden.size == 0:
                continue
            
            # Compute mean-pooled representations
            original_mean = np.mean(original_hidden, axis=0)
            paraphrase_mean = np.mean(paraphrase_hidden, axis=0)
            
            # Compute cosine similarity
            orig_flat = original_mean.flatten()
            para_flat = paraphrase_mean.flatten()
            
            if np.linalg.norm(orig_flat) == 0 or np.linalg.norm(para_flat) == 0:
                similarity = 0.0
            else:
                similarity = 1 - cosine(orig_flat, para_flat)
            
            similarities.append(similarity)
        
        # SIB is mean of similarities
        if len(similarities) == 0:
            return float("nan")
        
        sib = np.mean(similarities)
        return float(sib)
        
    except Exception as e:
        print(f"Error computing SIB: {e}")
        return float("nan")

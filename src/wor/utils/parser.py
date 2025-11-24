"""Parser utilities for extracting reasoning traces from model outputs."""

import re
from typing import Dict, Optional, Tuple


def parse_reasoning_trace(text: str) -> Dict[str, str]:
    """
    Parse reasoning trace from model output with <think> tags.
    
    Args:
        text: Full model output text
        
    Returns:
        Dictionary with:
            - 'reasoning_content': Text between <think> tags (empty if not found)
            - 'final_response': Text after </think> tag (or entire text if no tags)
            - 'has_reasoning': Boolean indicating if reasoning tags were found
    """
    # Pattern to match <think>...</think> (DeepSeek-R1-Distill-Llama-8B)
    # Also support <think> for compatibility with other models
    pattern = r'<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>'
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        reasoning_content = match.group(1).strip()
        # Get everything after the closing tag
        end_pos = match.end()
        final_response = text[end_pos:].strip()
        return {
            'reasoning_content': reasoning_content,
            'final_response': final_response,
            'has_reasoning': True
        }
    else:
        # No tags found - treat entire output as final response
        return {
            'reasoning_content': '',
            'final_response': text.strip(),
            'has_reasoning': False
        }


def find_token_ranges_for_reasoning(
    full_text: str,
    reasoning_content: str,
    final_response: str,
    tokenizer,
    input_tokens_length: int,
    full_token_sequence: Optional[list] = None
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Find token index ranges for reasoning and response portions.
    
    Args:
        full_text: Full generated text
        reasoning_content: Text inside reasoning tags
        final_response: Text after reasoning tags
        tokenizer: Tokenizer to use for tokenization
        input_tokens_length: Length of input tokens (to offset generated tokens)
        full_token_sequence: Optional pre-tokenized full sequence for more accurate mapping
        
    Returns:
        Dictionary with:
            - 'reasoning_range': (start_idx, end_idx) for reasoning tokens, or None
            - 'response_range': (start_idx, end_idx) for response tokens, or None
    """
    try:
        # If no reasoning content, all generated tokens are response
        if not reasoning_content:
            if full_token_sequence is not None:
                generated_start = input_tokens_length
                generated_end = len(full_token_sequence)
            else:
                full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                generated_start = input_tokens_length
                generated_end = len(full_tokens)
            return {
                'reasoning_range': None,
                'response_range': (generated_start, generated_end)
            }
        
        # Find reasoning section in full text (try <think> first, then <think>)
        reasoning_start_tag = '<think>'
        reasoning_end_tag = '</think>'
        if reasoning_start_tag not in full_text:
            reasoning_start_tag = '<think>'
            reasoning_end_tag = '</think>'
        
        reasoning_start_pos = full_text.find(reasoning_start_tag)
        reasoning_end_pos = full_text.find(reasoning_end_tag)
        
        if reasoning_start_pos == -1 or reasoning_end_pos == -1:
            # Tags not found, treat all as response
            if full_token_sequence is not None:
                generated_start = input_tokens_length
                generated_end = len(full_token_sequence)
            else:
                full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                generated_start = input_tokens_length
                generated_end = len(full_tokens)
            return {
                'reasoning_range': None,
                'response_range': (generated_start, generated_end)
            }
        
        # Get text before reasoning, reasoning content, and after reasoning
        text_before_reasoning = full_text[:reasoning_start_pos]
        text_after_reasoning = full_text[reasoning_end_pos + len(reasoning_end_tag):]
        
        # Tokenize each section to find boundaries
        # Use incremental tokenization for better accuracy
        tokens_before = tokenizer.encode(text_before_reasoning, add_special_tokens=False)
        tokens_reasoning = tokenizer.encode(reasoning_content, add_special_tokens=False)
        tokens_after = tokenizer.encode(text_after_reasoning, add_special_tokens=False)
        
        # Calculate token ranges
        # Note: This is approximate due to tokenization differences when tokenizing separately
        # For more accuracy, we'd need to tokenize the full sequence and find positions
        reasoning_start_idx = input_tokens_length + len(tokens_before)
        reasoning_end_idx = reasoning_start_idx + len(tokens_reasoning)
        response_start_idx = reasoning_end_idx
        response_end_idx = reasoning_start_idx + len(tokens_reasoning) + len(tokens_after)
        
        return {
            'reasoning_range': (reasoning_start_idx, reasoning_end_idx) if reasoning_content else None,
            'response_range': (response_start_idx, response_end_idx) if text_after_reasoning else None
        }
        
    except Exception as e:
        print(f"Error finding token ranges: {e}")
        return None


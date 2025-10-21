"""HuggingFace authentication utilities for Weight of Reasoning."""

import os
from typing import Optional

try:
    from huggingface_hub import login, whoami
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def ensure_hf_auth() -> bool:
    """Logs into Hugging Face Hub using whichever token variable is available."""
    if not HF_AVAILABLE:
        print("⚠️  Warning: huggingface_hub not available. Install with: pip install huggingface_hub")
        return False
    
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("⚠️  Warning: No Hugging Face token found (HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN).")
        return False

    try:
        login(token=token, add_to_git_credential=True)
        who = whoami()
        user = who.get("name") or who.get("email") or "unknown"
        print(f"✅ Hugging Face authentication successful for user: {user}")
        return True
    except Exception as e:
        print(f"❌ Hugging Face authentication failed: {e}")
        return False


def get_hf_token() -> Optional[str]:
    """Get the HuggingFace token from environment variables."""
    return os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def is_hf_authenticated() -> bool:
    """Check if HuggingFace is currently authenticated."""
    if not HF_AVAILABLE:
        return False
    
    try:
        whoami()
        return True
    except:
        return False

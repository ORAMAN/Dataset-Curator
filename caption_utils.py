# -*- coding: utf-8 -*-
"""Small helpers for composing generated image prompts."""

def append_clip_category(caption: str, category: str, enabled: bool = True) -> str:
    """Append a CLIP category once, preserving the prompt's comma-separated format."""
    caption = (caption or "").strip()
    category = (category or "").strip()
    if not enabled or not category or category.casefold() in caption.casefold():
        return caption
    return f"{caption.rstrip(' .,;')} , {category}".replace(" ,", ",")

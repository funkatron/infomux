"""
Cache utilities for infomux.

Currently manages external service response caches.
OpenAI summarization is the first provider.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_OPENAI_CACHE_DIR = "INFOMUX_OPENAI_CACHE_DIR"


def get_external_cache_dir(provider: str) -> Path:
    """
    Get cache directory for an external service provider.

    Default priority:
    1) XDG_CACHE_HOME/infomux/<provider>
    2) ~/.cache/infomux/<provider>
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser().resolve() / "infomux" / provider
    return Path.home() / ".cache" / "infomux" / provider


def get_openai_cache_dir() -> Path:
    """
    Get OpenAI response cache directory.

    Priority:
    1) INFOMUX_OPENAI_CACHE_DIR
    2) external cache default for 'openai'
    """
    env_path = os.environ.get(ENV_OPENAI_CACHE_DIR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return get_external_cache_dir("openai")


def list_openai_cache_files() -> list[Path]:
    """List OpenAI cache files."""
    cache_dir = get_openai_cache_dir()
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("*.json"))


def get_openai_cache_stats() -> tuple[int, int]:
    """
    Return cache stats as (file_count, total_bytes).
    """
    files = list_openai_cache_files()
    total_bytes = sum(p.stat().st_size for p in files if p.exists())
    return len(files), total_bytes


def clear_openai_cache() -> int:
    """
    Delete all OpenAI cache files.

    Returns:
        Number of files deleted.
    """
    deleted = 0
    for path in list_openai_cache_files():
        if path.exists():
            path.unlink()
            deleted += 1
    return deleted

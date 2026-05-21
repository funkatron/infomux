"""
Bear.app x-callback-url helpers.

Shared by infomux store_bear, repocon --export-bear, and other local tools.
"""

from __future__ import annotations

import os
import subprocess
import urllib.parse


class BearError(RuntimeError):
    """Bear note creation failed."""


def parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def get_tags_from_env(env_var: str, default: str) -> list[str]:
    return parse_tags(os.environ.get(env_var, default))


def _invoke(action: str, params: dict[str, str]) -> None:
    url = f"bear://x-callback-url/{action}?" + urllib.parse.urlencode(
        params,
        quote_via=urllib.parse.quote,
    )
    try:
        subprocess.run(["open", url], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise BearError(f"Bear {action} failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise BearError("macOS 'open' command not found. Bear export requires macOS.") from exc


def create_note(
    title: str,
    text: str,
    tags: list[str],
    *,
    open_note: bool = False,
) -> None:
    """Create a Bear note via bear://x-callback-url/create (macOS only)."""
    params = {
        "title": title,
        "text": text,
        "tags": ",".join(tags),
        "open_note": "yes" if open_note else "no",
    }
    _invoke("create", params)


def replace_note(
    title: str,
    text: str,
    tags: list[str] | None = None,
    *,
    open_note: bool = False,
) -> None:
    """Replace an existing Bear note body matched by title (add-text mode=replace)."""
    params = {
        "title": title,
        "text": text,
        "mode": "replace",
        "exclude_trashed": "yes",
        "open_note": "yes" if open_note else "no",
    }
    if tags:
        params["tags"] = ",".join(tags)
    _invoke("add-text", params)

"""
OpenAI API integration helpers.

This module isolates external API request/response handling from pipeline steps.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class OpenAIError(Exception):
    """Raised when an OpenAI API request fails."""


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    timeout_seconds: int = 300,
) -> tuple[str, int]:
    """
    Generate text using OpenAI chat completions.

    Returns:
        Tuple of (content, output_tokens)
    """
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        raise OpenAIError(f"HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise OpenAIError(f"connection error: {e}")
    except json.JSONDecodeError as e:
        raise OpenAIError(f"invalid response: {e}")

    choices = result.get("choices", [])
    if not choices:
        raise OpenAIError("no choices in response")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise OpenAIError("empty completion content")

    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    if not isinstance(completion_tokens, int):
        completion_tokens = 0

    return content, completion_tokens

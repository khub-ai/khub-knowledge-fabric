"""Backends for TUTOR (Anthropic) and PUPIL (OpenRouter).

Both return a raw text reply -- caller handles JSON parsing.  Keys are
loaded from .env at KF repo root or P:/_access/Security/api_keys.env,
matching the convention used by other KF usecases.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# On-disk call cache
# ---------------------------------------------------------------------------
#
# Every call is keyed on sha256(model + system + user + image_b64 +
# max_tokens + temperature) so identical prompts are never re-billed.
# Cache is persistent across sessions; delete the dir to force refresh.

CACHE_DIR = Path(__file__).resolve().parents[1] / ".tmp" / "model_cache"


def _cache_key(
    model: str, system: str, user: str,
    image_b64: Optional[str], max_tokens: int, temperature: float,
) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x1f")
    h.update(system.encode("utf-8"))
    h.update(b"\x1f")
    h.update(user.encode("utf-8"))
    h.update(b"\x1f")
    h.update((image_b64 or "").encode("ascii"))
    h.update(b"\x1f")
    h.update(f"{max_tokens}:{temperature}".encode("ascii"))
    return h.hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["_cache_hit"] = True
        return obj
    except Exception:  # noqa: BLE001
        return None


def _cache_put(key: str, entry: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_save = {k: v for k, v in entry.items() if k != "_cache_hit"}
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(to_save, indent=2), encoding="utf-8"
    )


def _cached_call(
    *,
    model: str, system: str, user: str,
    image_b64: Optional[str], max_tokens: int, temperature: float,
    uncached_fn: Callable[[], dict],
) -> dict:
    key = _cache_key(model, system, user, image_b64, max_tokens, temperature)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    result = uncached_fn()
    result["_cache_key"] = key
    result["_cache_hit"] = False
    _cache_put(key, result)
    return result


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

KEY_FILES = [
    Path(r"C:\_backup\github\khub-knowledge-fabric\.env"),
    Path(r"P:\_access\Security\api_keys.env"),
]


def _load_keys() -> None:
    for p in KEY_FILES:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and not os.environ.get(k):
                os.environ[k] = v


_load_keys()


# ---------------------------------------------------------------------------
# Anthropic (TUTOR: claude-sonnet-4-6)
# ---------------------------------------------------------------------------

def call_anthropic(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    return _cached_call(
        model=model, system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
        uncached_fn=lambda: _call_anthropic_uncached(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature,
        ),
    )


def _call_anthropic_uncached(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    content: list = []
    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64,
            },
        })
    content.append({"type": "text", "text": user})

    body = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "system":      system,
        "messages":    [{"role": "user", "content": content}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"anthropic HTTP {e.code}: {e.read().decode('utf-8','replace')}") from e
    elapsed = time.time() - t0

    text_parts = [c.get("text", "") for c in resp.get("content", []) if c.get("type") == "text"]
    return {
        "model":       model,
        "reply":       "".join(text_parts),
        "latency_ms":  int(elapsed * 1000),
        "raw":         resp,
    }


# ---------------------------------------------------------------------------
# OpenRouter (PUPIL: google/gemma-4-26b-a4b-it)
# ---------------------------------------------------------------------------

def call_openrouter(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    return _cached_call(
        model=model, system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
        uncached_fn=lambda: _call_openrouter_uncached(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature,
        ),
    )


def _call_openrouter_uncached(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    user_content: list = []
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    user_content.append({"type": "text", "text": user})

    body = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer":  "https://github.com/khub-ai/khub-knowledge-fabric",
            "X-Title":       "ARC-AGI-3 Dialogic Distillation",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"openrouter HTTP {e.code}: {e.read().decode('utf-8','replace')}") from e
    elapsed = time.time() - t0

    text = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""
    return {
        "model":       model,
        "reply":       text,
        "latency_ms":  int(elapsed * 1000),
        "raw":         resp,
    }


def encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")

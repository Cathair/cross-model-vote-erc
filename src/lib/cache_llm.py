"""File-based LLM response cache (opt-in via MARC_CACHE_DIR env var).

Wraps any LLM object exposing .invoke(prompt)->resp with .content.
When MARC_CACHE_DIR is set, responses are persisted per (model, temperature, prompt).
When unset, behaves as transparent passthrough (no caching, no risk to existing runs).

Cache key = sha256(model + temperature + prompt). This is a faithful key: same
sample+model+temp => hit (enables free ablation replay); different sample => miss.

Per-run isolation: each run sets a DIFFERENT MARC_CACHE_DIR (e.g. cache/v32_run1),
so runs stay statistically independent (no cross-run reuse).
"""
import hashlib
import json
import os
import time
from typing import Optional


class _CachedResponse:
    """Minimal response object exposing .content (matches ChatOpenAI / _Response usage)."""

    def __init__(self, content: str):
        self.content = content


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "-").replace(":", "_")


def _cache_key(model: str, temperature: float, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(f"|temp={float(temperature):.4f}".encode("utf-8"))
    h.update(b"|")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def get_cache_dir() -> Optional[str]:
    return os.environ.get("MARC_CACHE_DIR")


class CachedLLM:
    """Wraps an inner LLM; intercepts .invoke() with a file cache."""

    def __init__(self, inner, model: str, temperature: float = 0.0, cache_dir: Optional[str] = None):
        self.inner = inner
        self.model = model
        self.temperature = float(temperature)
        self.cache_dir = cache_dir or get_cache_dir()
        self.hits = 0
        self.misses = 0

    def _path_for(self, key: str) -> str:
        sub = os.path.join(self.cache_dir, _model_slug(self.model))
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{key}.json")

    def invoke(self, prompt: str):
        # No cache dir => passthrough
        if not self.cache_dir:
            return self.inner.invoke(prompt)

        key = _cache_key(self.model, self.temperature, prompt)
        path = self._path_for(key)

        # Hit
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.hits += 1
                return _CachedResponse(payload["content"])
            except Exception:
                # Corrupt cache file => fall through to live call
                pass

        # Miss => live call
        resp = self.inner.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        self.misses += 1
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": self.model,
                        "temperature": self.temperature,
                        "prompt_hash": key,
                        "content": content,
                        "cached_at": time.time(),
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception:
            pass
        return _CachedResponse(content)

    def __getattr__(self, name):
        # Delegate any other attribute access to the inner LLM
        return getattr(self.inner, name)


def maybe_wrap(inner, model: str, temperature: float = 0.0):
    """Wrap inner LLM with cache if MARC_CACHE_DIR is set; else return inner unchanged."""
    cd = get_cache_dir()
    if not cd:
        return inner
    return CachedLLM(inner, model=model, temperature=temperature, cache_dir=cd)

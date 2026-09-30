"""LLM client: OpenAI-compatible chat completions over urllib (no SDK needed).

Configuration (environment):
    ATHENA_API_KEY   API key (required for AI features)
    ATHENA_BASE_URL  default https://integrate.api.nvidia.com/v1
    ATHENA_MODEL     default nvidia/nemotron-3-super-120b-a12b
    ATHENA_LLM_ADAPTER  optional path to a Python file exposing
                        complete(messages, model=None, **kw) -> str
                        and optionally complete_stream(messages, model=None, **kw)
                        yielding str deltas. Used for local demos with injected
                        credentials; never commit keys.
"""
from __future__ import annotations

import importlib.util
import json
import os
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("ATHENA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.environ.get("ATHENA_MODEL", "nvidia/nemotron-3-super-120b-a12b")


def _adapter():
    path = os.environ.get("ATHENA_LLM_ADAPTER")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("_athena_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def configured() -> bool:
    return bool(os.environ.get("ATHENA_API_KEY") or os.environ.get("ATHENA_LLM_ADAPTER"))


def backend_label() -> str:
    if os.environ.get("ATHENA_LLM_ADAPTER"):
        return f"adapter ({MODEL})"
    return MODEL


def complete(messages: list[dict], model: str | None = None,
             temperature: float = 0.3, max_tokens: int = 1024,
             timeout: float = 120.0) -> str:
    adapter = _adapter()
    if adapter is not None:
        return adapter.complete(messages, model=model or MODEL,
                                temperature=temperature, max_tokens=max_tokens)
    key = os.environ.get("ATHENA_API_KEY")
    if not key:
        raise RuntimeError("ATHENA_API_KEY is not set (or use ATHENA_LLM_ADAPTER).")
    payload = {"model": model or MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def complete_stream(messages: list[dict], model: str | None = None,
                    temperature: float = 0.3, max_tokens: int = 1024,
                    timeout: float = 180.0):
    """Yield text deltas (SSE from the API, or adapter generator)."""
    adapter = _adapter()
    if adapter is not None and hasattr(adapter, "complete_stream"):
        yield from adapter.complete_stream(
            messages, model=model or MODEL, temperature=temperature,
            max_tokens=max_tokens)
        return
    key = os.environ.get("ATHENA_API_KEY")
    if not key:
        raise RuntimeError("ATHENA_API_KEY is not set (or use ATHENA_LLM_ADAPTER).")
    payload = {"model": model or MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens, "stream": True}
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        for raw in resp:
            buf += raw
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, ValueError):
                    continue
                if delta:
                    yield delta

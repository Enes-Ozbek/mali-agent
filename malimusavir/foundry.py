"""Foundry Local client: endpoint discovery, model-id resolution, chat + embeddings.

Foundry Local binds a *dynamic* port, so the endpoint is discovered at runtime rather
than hardcoded. Catalog aliases ("qwen3-4b") also differ from the loaded model ids
("qwen3-4b-generic-cpu"), so ids are resolved against /v1/models.

The official foundry-local-sdk (1.2.4) exposes only execution-provider management --
no endpoint or catalog lookup -- so discovery goes through the CLI's JSON output.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess

from openai import OpenAI

DEFAULT_ENDPOINT = "http://127.0.0.1:5267"

CHAT_ALIAS = "qwen3-4b"
EMBED_ALIAS = "qwen3-embedding-0.6b"

# Qwen3 is a hybrid-reasoning model. Left enabled it emits <think>...</think> blocks
# that break JSON parsing and burn CPU cycles we cannot spare. /no_think is the
# documented switch honoured by the Qwen3 chat template.
NO_THINK = "/no_think"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class FoundryError(RuntimeError):
    """Foundry Local is unreachable, or a requested model is not loaded."""


@functools.lru_cache(maxsize=1)
def discover_endpoint() -> str:
    """Return the base URL Foundry Local is listening on."""
    override = os.environ.get("FOUNDRY_LOCAL_ENDPOINT")
    if override:
        return override.rstrip("/")

    try:
        proc = subprocess.run(
            ["foundry", "-o", "json", "server", "status"],
            capture_output=True,
            text=True,
            # The CLI prints box-drawing glyphs that cp1252 cannot decode; without an
            # explicit codec the reader threads raise UnicodeDecodeError.
            encoding="utf-8",
            errors="replace",
            timeout=60,
            shell=(os.name == "nt"),
        )
        status = json.loads(proc.stdout)
        urls = status.get("webUrls") or []
        if status.get("running") and urls:
            return str(urls[0]).rstrip("/")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        pass  # fall through to the default; the health check below gives a better error

    return DEFAULT_ENDPOINT


@functools.lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """An OpenAI client pointed at the local Foundry server. The key is unused."""
    return OpenAI(base_url=f"{discover_endpoint()}/v1", api_key="not-needed", timeout=600.0)


@functools.lru_cache(maxsize=1)
def _loaded_model_ids() -> tuple[str, ...]:
    try:
        return tuple(m.id for m in get_client().models.list().data)
    except Exception as exc:  # noqa: BLE001 - surfaced as a single actionable message
        raise FoundryError(
            f"Cannot reach Foundry Local at {discover_endpoint()}. "
            "Start it with:  foundry server start"
        ) from exc


def resolve_model(alias: str) -> str:
    """Map a catalog alias to the concrete model id."""
    ids = _loaded_model_ids()
    if alias in ids:
        return alias
    for model_id in ids:
        if model_id.startswith(alias):
            return model_id
    raise FoundryError(
        f"Model {alias!r} is not available. Downloaded: {', '.join(ids) or '(none)'}. "
        f"Get it with:  foundry model download {alias}"
    )


@functools.lru_cache(maxsize=4)
def ensure_loaded(alias: str) -> str:
    """Resolve an alias and make sure the model is resident in memory.

    /v1/models lists *downloaded* models, but the completions endpoint 400s unless the
    model is also loaded. `foundry model load` is idempotent and near-instant when the
    model is already resident, so it is safe to call on first use of each alias.
    """
    model_id = resolve_model(alias)
    try:
        subprocess.run(
            ["foundry", "model", "load", alias],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            shell=(os.name == "nt"),
        )
    except (OSError, subprocess.SubprocessError):
        pass  # if the load failed the API call below reports it precisely
    return model_id


def chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """One-shot completion with reasoning suppressed.

    Note: this endpoint ignores OpenAI `stop` sequences -- passing them has no effect
    on the generated text, so callers must bound output with max_tokens and tolerate
    trailing prose. It also still emits empty <think></think> blocks under /no_think,
    which is why the response is scrubbed below rather than trusted.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": f"{prompt}\n\n{NO_THINK}"})

    response = get_client().chat.completions.create(
        model=ensure_loaded(CHAT_ALIAS),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    # Belt and braces: strip think blocks even if the template ignored /no_think.
    return _THINK_RE.sub("", content).strip()


def chat_turns(
    messages: list[dict[str, str]],
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> str:
    """Multi-turn completion. ``messages`` is [{"role": "user"|"assistant", ...}].

    chat() above is one-shot and cannot see earlier turns, so a conversational agent
    needs this: "peki geçen yıl?" only means anything with the previous question in
    context. /no_think rides on the last user turn, the same place chat() puts it.
    """
    payload: list[dict[str, str]] = []
    if system:
        payload.append({"role": "system", "content": system})
    payload.extend({"role": m["role"], "content": m["content"]} for m in messages)
    if payload and payload[-1]["role"] == "user":
        payload[-1] = {**payload[-1], "content": f"{payload[-1]['content']}\n\n{NO_THINK}"}

    response = get_client().chat.completions.create(
        model=ensure_loaded(CHAT_ALIAS),
        messages=payload,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return _THINK_RE.sub("", content).strip()


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input, in order."""
    if not texts:
        return []
    response = get_client().embeddings.create(
        model=ensure_loaded(EMBED_ALIAS),
        input=texts,
    )
    return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

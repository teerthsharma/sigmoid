"""One adapter for every OpenAI-compatible endpoint.

OpenAI, Groq, Together, OpenRouter, DeepSeek, xAI, Gemini's compat endpoint,
Ollama and vLLM all speak ``POST /chat/completions`` with the same request and
response shape. They differ in exactly three things: base URL, environment
variable, and default model. Those three fit in a ``Spec`` row, so this file
holds one parser instead of eight copies that drift apart.

Adding an endpoint is one line in ``SPECS`` -- plus its variable in
``base.KEY_ENV_VARS``, which ``test_providers.py`` enforces so a new provider
cannot arrive without redaction coverage.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .base import (
    AuthError,
    Completion,
    Message,
    Provider,
    ProviderError,
    ToolCall,
    ToolSpec,
    _redact,
    http_transport,
)

__all__ = ["Spec", "SPECS", "OpenAICompatProvider"]


@dataclass(frozen=True)
class Spec:
    """Everything that distinguishes one OpenAI-compatible endpoint.

    ``env_var`` empty means the endpoint needs no key (a local server), in
    which case ``base_url_env`` gates availability instead: a robot with no
    ``OLLAMA_HOST`` set has no Ollama, and saying so is more useful than
    guessing localhost and timing out mid-motion.
    """

    name: str
    base_url: str
    default_model: str
    env_var: str = ""
    base_url_env: str = ""


SPECS: dict[str, Spec] = {
    s.name: s
    for s in [
        Spec("openai", "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
        Spec("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        Spec(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-2.0-flash",
            "GEMINI_API_KEY",
        ),
        Spec("deepseek", "https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
        Spec("mistral", "https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
        Spec(
            "together",
            "https://api.together.xyz/v1",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "TOGETHER_API_KEY",
        ),
        Spec(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "meta-llama/llama-3.3-70b-instruct",
            "OPENROUTER_API_KEY",
        ),
        Spec("xai", "https://api.x.ai/v1", "grok-2-latest", "XAI_API_KEY"),
        Spec("ollama", "", "llama3.2", base_url_env="OLLAMA_HOST"),
        Spec("vllm", "", "", base_url_env="VLLM_BASE_URL"),
    ]
}


class OpenAICompatProvider(Provider):
    """A chat-completions backend.

    Construction is free and touches nothing -- ``registry.available()``
    instantiates every provider just to ask whether it is configured.
    """

    def __init__(
        self,
        spec: Spec | str = "openai",
        model: str | None = None,
        timeout: float = 60.0,
        transport: Any = None,
    ) -> None:
        if isinstance(spec, str):
            if spec not in SPECS:
                raise ProviderError(f"unknown endpoint {spec!r}; known: {', '.join(SPECS)}")
            spec = SPECS[spec]
        self.spec = spec
        self.model = model or spec.default_model
        self.timeout = timeout
        # Injected in tests. There is no other way to exercise the parser
        # without a socket, and a parser nobody can test is a parser nobody
        # trusts at 3am.
        self._transport = transport or http_transport

    # -- configuration -----------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def available(self) -> bool:
        """Presence of configuration. Never inspects a key's value."""
        if self.spec.env_var:
            return bool(os.environ.get(self.spec.env_var))
        return bool(self.spec.base_url or os.environ.get(self.spec.base_url_env))

    def _base_url(self) -> str:
        """Resolved base URL, always ending in ``/v1``.

        Users set ``OLLAMA_HOST=http://localhost:11434`` (a host) but
        ``VLLM_BASE_URL=http://box:8000/v1`` (a base URL). Normalising both to
        the same suffix beats documenting which one wants what.
        """
        url = (os.environ.get(self.spec.base_url_env) or "" if self.spec.base_url_env else "") or (
            self.spec.base_url
        )
        if not url:
            raise ProviderError(f"{self.name}: set {self.spec.base_url_env} to its base URL")
        url = url.rstrip("/")
        return url if url.endswith("/v1") else url + "/v1"

    def _headers(self) -> dict[str, str]:
        """Read the key from the environment *now*. It is never stored on
        ``self``, so no attribute, repr, or pickle of this object holds it."""
        headers = {"Content-Type": "application/json"}
        if self.spec.env_var:
            key = os.environ.get(self.spec.env_var)
            if not key:
                raise AuthError(f"{self.name}: {self.spec.env_var} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    # -- requests ----------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        stream: bool,
        kw: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": kw.pop("model", None) or self.model,
            "messages": [_to_wire(m) for m in messages],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description,
                                 "parameters": t.parameters},
                }
                for t in tools
            ]
        if stream:
            payload["stream"] = True
        payload.update(kw)
        return payload

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kw: Any,
    ) -> Completion:
        payload = self._payload(messages, tools, False, kw)
        body = json.dumps(payload).encode()
        t0 = time.perf_counter()
        raw = self._transport(
            f"{self._base_url()}/chat/completions", self._headers(), body, self.timeout, False
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return _parse(raw, latency, payload["model"])

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kw: Any,
    ) -> Iterator[str]:
        payload = self._payload(messages, tools, True, kw)
        body = json.dumps(payload).encode()
        lines = self._transport(
            f"{self._base_url()}/chat/completions", self._headers(), body, self.timeout, True
        )
        for chunk in _iter_sse(lines):
            choices = chunk.get("choices") or [{}]
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta


# --------------------------------------------------------------------------
# wire translation
# --------------------------------------------------------------------------


def _to_wire(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    return out


def _parse(raw: bytes | str | dict, latency_ms: float, model: str) -> Completion:
    data = raw if isinstance(raw, dict) else json.loads(raw)
    if "error" in data and not data.get("choices"):
        # 200-with-error-body happens on proxies and gateways.
        raise ProviderError(_redact(f"{model}: {data['error']}"))
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return Completion(
        text=msg.get("content") or "",
        tool_calls=[_to_call(c) for c in (msg.get("tool_calls") or [])],
        finish_reason=choice.get("finish_reason") or "stop",
        # Normalised names: a token budget written against one provider must
        # keep working after auto() picks a different one.
        usage={
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
        },
        latency_ms=latency_ms,
        model=data.get("model") or model,
    )


def _to_call(c: dict[str, Any]) -> ToolCall:
    fn = c.get("function") or {}
    return ToolCall(id=c.get("id") or "", name=fn.get("name") or "", arguments=_args(fn))


def _args(fn: dict[str, Any]) -> dict[str, Any]:
    """Parse tool arguments, keeping malformed JSON visible.

    Models do emit invalid JSON. Returning ``{}`` would hand a robot an
    argument-free tool call that looks legitimate; keeping the raw text under
    ``_raw`` makes the failure obvious at the call site.
    """
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


def _iter_sse(lines: Any) -> Iterator[dict[str, Any]]:
    """Decode ``data:`` frames, skipping keep-alives and the ``[DONE]`` marker."""
    for line in lines:
        text = line.decode() if isinstance(line, bytes) else line
        text = text.strip()
        if not text.startswith("data:"):
            continue
        payload = text[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except ValueError:
            continue  # a partial frame is not a failed request

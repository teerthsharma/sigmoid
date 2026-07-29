"""Anthropic Messages API.

Not OpenAI-compatible, and the differences are structural rather than cosmetic,
which is why this cannot fold into ``openai_compat``:

* ``system`` is a top-level field, not a message with ``role="system"``;
* content is a list of typed blocks, not a string;
* tool calls arrive as ``tool_use`` blocks inside assistant content, and
  results go back as ``tool_result`` blocks inside a *user* message;
* auth is ``x-api-key`` plus a required ``anthropic-version`` header;
* ``max_tokens`` is mandatory.

Everything above this file still sees ``Message``/``Completion``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
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

__all__ = ["AnthropicProvider"]

BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-sonnet-4-5"

# Required by the API. A robot that forgets it gets a 400 mid-motion, so pick a
# value big enough for a plan and small enough to bound loop latency.
DEFAULT_MAX_TOKENS = 2048


class AnthropicProvider(Provider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        transport: Any = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._transport = transport or http_transport

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def available(self) -> bool:
        return bool(os.environ.get(ENV_VAR))

    def _headers(self) -> dict[str, str]:
        """Key read from the environment per request; never stored on self."""
        key = os.environ.get(ENV_VAR)
        if not key:
            raise AuthError(f"anthropic: {ENV_VAR} is not set")
        return {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": API_VERSION,
        }

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        stream: bool,
        kw: dict[str, Any],
    ) -> dict[str, Any]:
        system, turns = _split_system(messages)
        payload: dict[str, Any] = {
            "model": kw.pop("model", None) or self.model,
            "messages": turns,
            "max_tokens": kw.pop("max_tokens", DEFAULT_MAX_TOKENS),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
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
        raw = self._transport(f"{BASE_URL}/messages", self._headers(), body, self.timeout, False)
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
        lines = self._transport(f"{BASE_URL}/messages", self._headers(), body, self.timeout, True)
        for event in _iter_sse(lines):
            # Only text deltas; input_json_delta fragments are useless until
            # the whole tool call has arrived.
            if event.get("type") == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]


# --------------------------------------------------------------------------
# wire translation
# --------------------------------------------------------------------------


def _split_system(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Hoist system turns to the top-level field and block-ify the rest.

    Multiple system messages are joined rather than dropped -- a safety
    preamble silently disappearing is exactly the bug you cannot afford here.
    """
    system_parts, turns = [], []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            turns.append(_to_wire(m))
    return "\n\n".join(p for p in system_parts if p), turns


def _to_wire(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        # Tool results are user turns carrying a tool_result block.
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.content}
            ],
        }
    blocks: list[dict[str, Any]] = []
    if m.content:
        blocks.append({"type": "text", "text": m.content})
    for c in m.tool_calls or []:
        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
    return {"role": m.role, "content": blocks}


def _parse(raw: bytes | str | dict, latency_ms: float, model: str) -> Completion:
    data = raw if isinstance(raw, dict) else json.loads(raw)
    if data.get("type") == "error" or ("error" in data and "content" not in data):
        raise ProviderError(_redact(f"{model}: {data.get('error', data)}"))
    text, calls = [], []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id") or "",
                    name=block.get("name") or "",
                    arguments=block.get("input") or {},
                )
            )
    usage = data.get("usage") or {}
    return Completion(
        text="".join(text),
        tool_calls=calls,
        finish_reason=data.get("stop_reason") or "stop",
        usage={
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
        latency_ms=latency_ms,
        model=data.get("model") or model,
    )


def _iter_sse(lines: Any) -> Iterator[dict[str, Any]]:
    for line in lines:
        text = line.decode() if isinstance(line, bytes) else line
        text = text.strip()
        if not text.startswith("data:"):
            continue
        try:
            yield json.loads(text[5:].strip())
        except ValueError:
            continue

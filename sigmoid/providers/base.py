"""The one interface every model backend must satisfy.

A robot cannot assume which model it will get. On the bench it is a hosted API;
in the field it is whatever runs on the onboard accelerator. The planner should
not care, so everything -- hosted or local -- implements ``Provider``.

Two deliberate shapes here:

``Message``/``ToolSpec``/``Completion`` are provider-neutral. Wire formats
differ (OpenAI puts tool calls on the message, Anthropic puts them in content
blocks); the adapters translate, and nothing above them ever sees a wire dict.

Providers hold *no* API key. The key is read from the environment at call time,
inside the adapter. That is not defensive style -- it is the reason a key
cannot appear in ``repr()`` of a provider, in a pickle of one, or in a traceback
frame that renders locals. There is nothing to leak because nothing is stored.
"""

from __future__ import annotations

import builtins
import contextlib
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Message",
    "ToolSpec",
    "ToolCall",
    "Completion",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "AuthError",
    "TimeoutError",
    "KEY_ENV_VARS",
    "HOST_ENV_VARS",
    "http_transport",
]


# --------------------------------------------------------------------------
# secret redaction
#
# Lives here rather than in registry.py because every adapter needs it on its
# error path and registry.py imports the adapters -- a helper the whole package
# depends on cannot sit downstream of them. registry re-exports it.
# --------------------------------------------------------------------------

# Every environment variable that may hold a secret. This is also the redaction
# list: a provider whose variable is missing here would leak, so
# ``test_providers.py`` asserts every registered endpoint's variable is present.
KEY_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
)

# Host variables. Not secrets, but they gate availability for local servers.
HOST_ENV_VARS: tuple[str, ...] = ("OLLAMA_HOST", "VLLM_BASE_URL")

# Common key shapes: OpenAI/DeepSeek/Together ``sk-``, Groq ``gsk_``, xAI
# ``xai-``, Anthropic ``sk-ant-``, Google ``AIza``, plus any bearer token or
# echoed ``x-api-key`` header. Catches key material that never passed through
# *our* environment -- e.g. a key pasted into a prompt and echoed by a server.
_KEY_SHAPE = re.compile(
    r"(?:sk-ant-|sk-or-|sk-|gsk_|xai-|AIza|hf_|r8_)[A-Za-z0-9_\-]{12,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]{12,}"
    r"|(?i:x-api-key)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{8,}"
)

_REDACTED = "<redacted>"


def _redact(text: Any) -> str:
    """Strip anything that could be a secret out of ``text``.

    Every string that reaches a user -- exception messages, log lines, raw
    response bodies, urllib's URL-bearing errors -- goes through here. Two
    passes: exact match against the live environment (completely reliable, we
    know what to hide) then a shape regex (catches foreign key material).

    Cheap enough -- nine ``str.replace`` calls and one regex -- to apply
    unconditionally on every error path.
    """
    out = str(text)
    for var in KEY_ENV_VARS:
        value = os.environ.get(var)
        # Length floor only so a stray one-character env value cannot shred
        # unrelated text. Real keys are far longer than this.
        if value and len(value) >= 4:
            out = out.replace(value, _REDACTED)
    return _KEY_SHAPE.sub(_REDACTED, out)


# --------------------------------------------------------------------------
# wire-neutral types
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A model's request to run one tool.

    ``arguments`` is already-parsed JSON, not a string -- every adapter parses
    it, so parsing once at the boundary saves every caller from doing it.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn. ``role`` is user/assistant/system/tool.

    ``tool_calls`` is set on assistant turns that requested tools;
    ``tool_call_id`` is set on tool turns that answer one.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolSpec:
    """A tool offered to the model. ``parameters`` is a JSON Schema object."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class Completion:
    """What came back.

    ``latency_ms`` is measured by the adapter, not reported by the server: on a
    robot the number that matters is wall-clock to the control loop, which
    includes the network, not the server's own accounting.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    model: str = ""


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class ProviderError(Exception):
    """Base for every provider failure. Message text is always redacted."""


class RateLimitError(ProviderError):
    """429 or equivalent. Retryable after a wait."""


class AuthError(ProviderError):
    """401/403. Not retryable -- the key is missing, wrong, or revoked."""


class TimeoutError(ProviderError, builtins.TimeoutError):  # noqa: A001
    """Request exceeded its deadline.

    Subclasses the builtin too, so control code that already writes
    ``except TimeoutError`` around its deadline handling keeps working whether
    the timeout came from a socket or from us.
    """


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------


class Provider(ABC):
    """A model backend.

    Implementations must be cheap to construct and must not touch the network
    (or the environment beyond a presence check) until ``complete``/``stream``
    is called -- ``registry.available()`` builds every registered provider to
    ask whether it is configured.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name, e.g. ``"groq"``."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """True if this provider is configured -- key present, host reachable
        in principle, module importable. Never inspects a key's *value*."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kw: Any,
    ) -> Completion:
        """One blocking round trip."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kw: Any,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive.

        Text only. Streamed tool-call fragments are useless to a robot until
        complete, so callers wanting tools use ``complete``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, available={self.available})"


# --------------------------------------------------------------------------
# transport
#
# stdlib ``urllib`` on purpose. httpx and requests happen to be installed in
# this dev environment, but sigmoid's runtime dependencies are numpy and scipy
# only -- a robot image should not have to carry an HTTP stack for a feature it
# may never use, and supporting "httpx if present, urllib otherwise" would mean
# two transport paths and two sets of exceptions to map. One path, always
# available.
#
# Adapters take ``transport=`` so tests inject a fake and never touch a socket.
# --------------------------------------------------------------------------


def http_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float = 60.0,
    stream: bool = False,
) -> bytes | Iterator[bytes]:
    """POST JSON. Returns the body, or a line iterator when ``stream``.

    Maps HTTP status to the provider error hierarchy so callers can retry a
    ``RateLimitError`` and give up on an ``AuthError`` without parsing text.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 -- https literals
    except urllib.error.HTTPError as exc:
        detail = ""
        # Bodies echo request headers on some gateways, so redact before use.
        # A body we cannot read is not a second failure worth reporting.
        with contextlib.suppress(Exception):
            detail = _redact(exc.read().decode("utf-8", "replace"))[:400]
        raise _status_error(exc.code, detail) from None
    except urllib.error.URLError as exc:
        # `from None`, not `from exc`: urllib's exception text carries the full
        # request URL, and some providers accept the key as a query parameter.
        # Chaining would print the unredacted original under "During handling".
        # socket.timeout is an alias of the builtin TimeoutError, which our
        # TimeoutError subclasses -- so one arm covers both.
        if isinstance(exc.reason, builtins.TimeoutError):
            raise TimeoutError(f"request timed out after {timeout}s") from None
        raise ProviderError(_redact(f"connection failed: {exc.reason}")) from None
    except builtins.TimeoutError:
        raise TimeoutError(f"request timed out after {timeout}s") from None

    if stream:
        return _iter_lines(resp)
    with resp:
        return resp.read()


def _iter_lines(resp: Any) -> Iterator[bytes]:
    """Yield lines while holding the response open."""
    with resp:
        yield from resp


def _status_error(code: int, detail: str) -> ProviderError:
    """HTTP status -> the exception a caller can branch on.

    Redacts ``detail`` again even though the call site already did. Redaction
    is idempotent and costs a microsecond; relying on "every caller remembers"
    is how a key eventually reaches a log, and this function is reachable from
    anywhere an adapter chooses to map a status.
    """
    detail = _redact(detail)
    if code in (401, 403):
        return AuthError(f"HTTP {code}: key rejected or missing. {detail}")
    if code == 429:
        return RateLimitError(f"HTTP {code}: rate limited. {detail}")
    if code in (408, 504):
        return TimeoutError(f"HTTP {code}: upstream timeout. {detail}")
    return ProviderError(f"HTTP {code}: {detail}")

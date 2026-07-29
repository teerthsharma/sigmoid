"""Provider registry and the environment-variable key policy.

Key policy, in one sentence: **a key exists in exactly one place, the process
environment, and is read at the moment of the request.**

No constructor default, no file in the repo, no literal, no committed ``.env``.
A provider object therefore has nothing to leak -- ``repr()`` of one, a pickle
of one, or a traceback that renders its locals all contain no key material,
because the key was never assigned to an attribute in the first place.

The second half of the policy is that keys must not escape through *text*
either: servers echo the ``Authorization`` header back in some error bodies and
``urllib`` puts the full URL in its exception message. Every string that comes
from outside is routed through ``_redact`` (defined in ``base``, re-exported
here) before it reaches an exception, a log record, or a return value.

``available()`` answers presence only -- ``bool(os.environ.get(...))`` -- and
never returns, logs, or compares any part of a value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .anthropic import AnthropicProvider
from .base import HOST_ENV_VARS, KEY_ENV_VARS, Provider, ProviderError, _redact
from .local import LocalProvider
from .openai_compat import SPECS, OpenAICompatProvider

__all__ = [
    "register",
    "get",
    "available",
    "auto",
    "env_var_for",
    "PRIORITY",
    "KEY_ENV_VARS",
    "HOST_ENV_VARS",
    "_redact",
]


_REGISTRY: dict[str, Callable[..., Provider]] = {}


def register(name: str, cls: Callable[..., Provider]) -> None:
    """Register a provider factory (a class, or any callable returning one)."""
    _REGISTRY[name] = cls


def get(name: str, **kw: Any) -> Provider:
    """Construct the named provider. Raises ``ProviderError`` if unknown."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ProviderError(f"unknown provider {name!r}; registered: {known}") from None
    return factory(**kw)


def available() -> list[str]:
    """Names of configured providers, in ``PRIORITY`` order.

    Reports *presence* of configuration only. A provider whose constructor or
    availability probe raises is treated as unavailable rather than being
    allowed to break discovery -- a broken optional backend must not stop a
    robot from finding a working one.
    """
    out = []
    for name in _order():
        try:
            if get(name).available:
                out.append(name)
        except Exception:  # noqa: BLE001 -- discovery must never raise
            continue
    return out


# Deliberate for a robot, not for benchmark scores:
#
# 1. ``local``           -- the onboard Triton engine. No network, no per-token
#    cost, no rate limit, and it keeps working when the radio does not.
# 2. ``ollama``/``vllm`` -- a model server on this machine or the LAN. Still no
#    internet, still bounded latency.
# 3. ``groq``            -- fastest hosted inference; the nearest thing to a
#    real-time remote loop.
# 4. ``openai``/``anthropic`` -- strongest general tool-callers.
# 5. the rest            -- fallback, roughly cheapest first.
PRIORITY: tuple[str, ...] = (
    "local",
    "ollama",
    "vllm",
    "groq",
    "openai",
    "anthropic",
    "gemini",
    "deepseek",
    "mistral",
    "together",
    "openrouter",
    "xai",
)


def auto(**kw: Any) -> Provider:
    """First available provider by ``PRIORITY`` (see the table above)."""
    for name in available():
        return get(name, **kw)
    raise ProviderError(
        "no provider configured; set one of: " + ", ".join(KEY_ENV_VARS + HOST_ENV_VARS)
    )


def _order() -> list[str]:
    """Registered names, PRIORITY first, then anything registered later."""
    ranked = [n for n in PRIORITY if n in _REGISTRY]
    return ranked + sorted(set(_REGISTRY) - set(ranked))


def env_var_for(name: str) -> str | None:
    """The environment variable that configures ``name``, for error messages."""
    if name == "anthropic":
        return "ANTHROPIC_API_KEY"
    spec = SPECS.get(name)
    return spec.env_var if spec else None


# Built-ins. One OpenAI-compatible class covers eight endpoints -- they share a
# wire format, so a class each would be seven copies of the same parser.
for _name, _spec in SPECS.items():
    register(_name, lambda _s=_spec, **kw: OpenAICompatProvider(_s, **kw))
register("anthropic", AnthropicProvider)
register("local", LocalProvider)

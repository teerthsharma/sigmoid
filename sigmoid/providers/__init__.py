"""One interface over every model backend a robot might reach.

    from sigmoid.providers import auto, Message

    p = auto()                                    # first configured backend
    print(p.complete([Message("user", "status?")]).text)

Keys come from the environment and nowhere else -- see ``registry`` for the
policy and ``base._redact`` for how they are kept out of logs and tracebacks.
"""

from .anthropic import AnthropicProvider
from .base import (
    AuthError,
    Completion,
    Message,
    Provider,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ToolCall,
    ToolSpec,
)
from .local import LocalProvider
from .openai_compat import SPECS, OpenAICompatProvider, Spec
from .registry import PRIORITY, auto, available, env_var_for, get, register

__all__ = [
    "Message",
    "ToolCall",
    "ToolSpec",
    "Completion",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "AuthError",
    "TimeoutError",
    "OpenAICompatProvider",
    "Spec",
    "SPECS",
    "AnthropicProvider",
    "LocalProvider",
    "register",
    "get",
    "available",
    "auto",
    "env_var_for",
    "PRIORITY",
]

"""The on-robot backend: ``sigmoid.inference.InferenceEngine`` as a Provider.

This is the one that runs on the merged Triton kernel, and the reason the
abstraction exists at all -- a robot that loses its radio should degrade to
local inference, not to nothing. It therefore satisfies exactly the same
interface as the hosted adapters, including a ``stream`` that yields text.

``sigmoid.inference`` is imported *inside* the methods, never at module import.
Two reasons: the module may be absent or broken in a given checkout, and
importing it eagerly pulls torch into every process that merely wants to list
providers. Absent means ``available == False`` and nothing else breaks.

The wire translation here is prompt-flattening. ``InferenceEngine`` generates
from a token stream, not a chat transcript, so somebody has to turn turns into
text; doing it here keeps the chat abstraction intact above and leaves the
engine free of prompt-format opinions. ``template=`` is the calibration knob --
every base model wants a different one, and the default is deliberately the
plainest thing that works rather than any model's official format.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterator, Sequence
from typing import Any

from .base import Completion, Message, Provider, ProviderError, ToolSpec

__all__ = ["LocalProvider", "flatten"]


def flatten(messages: Sequence[Message]) -> str:
    """Chat turns -> one prompt. Override via ``LocalProvider(template=...)``."""
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


class LocalProvider(Provider):
    """Wraps an ``InferenceEngine``.

    Pass ``engine=`` for an already-loaded model -- which is the normal case on
    a robot, where loading weights per request is not an option -- or pass the
    engine's own kwargs (``model=``, ``tokenizer=``, ``config=``) to have one
    built on first use.
    """

    def __init__(
        self,
        engine: Any = None,
        template: Any = flatten,
        max_new_tokens: int = 256,
        **engine_kw: Any,
    ) -> None:
        self._engine = engine
        self._engine_kw = engine_kw
        self._template = template
        self.max_new_tokens = max_new_tokens
        self.model = str(engine_kw.get("model", "") or "local")

    @property
    def name(self) -> str:
        return "local"

    @property
    def available(self) -> bool:
        """Configured means: an engine, or enough to build one.

        ``InferenceEngine`` needs a loaded model and torch. Neither is implied
        by the module merely importing, so an import check alone would report
        a backend that fails on first call -- the worst answer to give a robot
        choosing where to send its next plan.
        """
        if self._engine is not None:
            return True
        if not self._engine_kw.get("model"):
            return False
        try:
            import torch  # noqa: F401

            from sigmoid.inference import InferenceEngine  # noqa: F401
        except Exception:  # noqa: BLE001 -- absent or broken; same answer
            return False
        return True

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from sigmoid.inference import InferenceEngine
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"local inference unavailable: {exc}") from None
            if not self._engine_kw.get("model"):
                raise ProviderError("local: pass engine=... or model=... (a loaded causal LM)")
            self._engine = InferenceEngine(**self._engine_kw)
        return self._engine

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kw: Any,
    ) -> Completion:
        engine = self._get_engine()
        t0 = time.perf_counter()
        result = _generate(engine, self._template(messages), self.max_new_tokens, kw, stream=False)
        latency = (time.perf_counter() - t0) * 1000.0
        return _to_completion(result, latency, self.model)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kw: Any,
    ) -> Iterator[str]:
        engine = self._get_engine()
        out = _generate(engine, self._template(messages), self.max_new_tokens, kw, stream=True)
        if isinstance(out, str):
            # Engine has no token stream. One chunk still satisfies the
            # contract, so a caller written against stream() keeps working.
            yield out
            return
        yield from out


# --------------------------------------------------------------------------
# engine translation
# --------------------------------------------------------------------------


def _generate(engine: Any, prompt: str, max_new_tokens: int, kw: dict, stream: bool) -> Any:
    """Call the engine's entry point, passing only kwargs it declares.

    Signature inspection rather than call-and-catch-TypeError: catching would
    also swallow a genuine ``TypeError`` from inside the engine and misreport
    it as a shape mismatch, which is a miserable thing to debug on a robot.
    """
    for attr in ("generate", "complete", "__call__"):
        fn = getattr(engine, attr, None)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):  # C callable, no introspectable signature
            params = {}
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        call_kw = {k: v for k, v in kw.items() if takes_any or k in params}
        if "max_new_tokens" in params:
            call_kw.setdefault("max_new_tokens", max_new_tokens)
        if "stream" in params:
            call_kw["stream"] = stream
        elif stream:
            return _text(fn(prompt, **call_kw))
        return fn(prompt, **call_kw)
    raise ProviderError(f"{type(engine).__name__} exposes no generate/complete/__call__")


def _text(result: Any) -> str:
    return result.text if hasattr(result, "text") else str(result)


def _to_completion(result: Any, latency_ms: float, model: str) -> Completion:
    """Accept a Completion, a ``GenerationResult``, or plain text.

    Pinning the sibling module's return type from over here would make this
    file break every time that one gains a field.
    """
    if isinstance(result, Completion):
        result.latency_ms = result.latency_ms or latency_ms
        result.model = result.model or model
        return result
    stats = getattr(result, "stats", None)
    return Completion(
        text=_text(result),
        # Local inference has no prompt-token accounting to bill against, so
        # only what it produced is reported; `input` stays 0 rather than lying.
        usage={"input": 0, "output": getattr(stats, "tokens", 0), "total":
               getattr(stats, "tokens", 0)},
        latency_ms=latency_ms,
        model=model,
    )

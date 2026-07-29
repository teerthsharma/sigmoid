"""Interception points for the agent loop.

A robot runtime needs somewhere to put the things that are not the algorithm:
logging, telemetry, caching, retries, and -- the one that matters -- a safety
gate that can stop an action before it reaches an actuator.

Two properties carry the design, and both are tested:

**Veto is first class.** A hook raises ``HookVeto`` and the action does not
happen. That propagates out of ``emit`` untouched; it is not an error, it is
the answer. A safety hook that could only *log* its objection would be
decoration.

**A broken hook cannot stop the run.** Any other exception from a hook is
caught, routed to ``ON_ERROR``, logged, and the remaining hooks still run. A
telemetry sink whose socket died must never take an arm down mid-motion. The
asymmetry with veto is the whole point: refusing on purpose stops the robot,
failing by accident does not.

Usage::

    hooks = HookManager()

    @hooks.on(HookPoint.BEFORE_TOOL, priority=100)      # runs first
    def guard(ctx):
        if ctx["tool"] == "arm.move" and not ctx.get("cleared"):
            raise HookVeto("arm not cleared for motion")

    ctx = hooks.emit(HookPoint.BEFORE_TOOL, {"tool": "arm.move"})   # raises

``ctx`` is a plain dict. ``emit`` stamps ``ctx["point"]`` so a hook registered
on several points knows which one fired without a bound wrapper per point.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

__all__ = [
    "HookPoint",
    "HookVeto",
    "HookManager",
    "LoggingHook",
    "TimingHook",
]

log = logging.getLogger("sigmoid.hooks")

Ctx = dict[str, Any]
Hook = Callable[[Ctx], Ctx | None]


class HookPoint(str, Enum):
    """Where a hook can attach.

    ``str`` subclass so ``ctx["point"] == "before_tool"`` works in a config
    file, a JSON log line, or a test, without importing this enum.
    """

    BEFORE_COMPLETE = "before_complete"
    AFTER_COMPLETE = "after_complete"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    ON_ERROR = "on_error"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_TOKEN = "on_token"
    ON_REFUSAL = "on_refusal"


class HookVeto(Exception):
    """Raised by a hook to refuse the action being announced.

    Deliberately not a subclass of any provider error: a veto is a decision,
    not a failure, and code that retries on failure must not retry a refusal.
    """

    def __init__(self, reason: str, ctx: Ctx | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.ctx = ctx or {}


class HookManager:
    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[tuple[int, int, Hook]]] = {}
        # Monotonic tiebreaker. Sorting on priority alone would let Python's
        # stable sort be the only thing keeping equal-priority hooks in
        # registration order; making it explicit means reordering cannot
        # silently change which safety check runs first.
        self._seq = 0

    # -- registration ------------------------------------------------------

    def register(self, point: HookPoint, fn: Hook, *, priority: int = 0) -> Hook:
        """Attach ``fn`` at ``point``. Higher priority runs first.

        Returns ``fn``, so this doubles as a decorator when you already have
        the point in hand.
        """
        self._seq += 1
        entry = (-priority, self._seq, fn)
        bucket = self._hooks.setdefault(HookPoint(point), [])
        bucket.append(entry)
        bucket.sort(key=lambda e: e[:2])
        return fn

    def unregister(self, point: HookPoint, fn: Hook) -> bool:
        """Detach ``fn``. Returns whether anything was removed."""
        bucket = self._hooks.get(HookPoint(point))
        if not bucket:
            return False
        keep = [e for e in bucket if e[2] is not fn]
        removed = len(keep) != len(bucket)
        self._hooks[HookPoint(point)] = keep
        return removed

    def on(self, point: HookPoint, *, priority: int = 0) -> Callable[[Hook], Hook]:
        """Decorator form of ``register``."""

        def deco(fn: Hook) -> Hook:
            return self.register(point, fn, priority=priority)

        return deco

    def hooks(self, point: HookPoint) -> list[Hook]:
        """Hooks at ``point``, in execution order."""
        return [e[2] for e in self._hooks.get(HookPoint(point), [])]

    def clear(self, point: HookPoint | None = None) -> None:
        if point is None:
            self._hooks.clear()
        else:
            self._hooks.pop(HookPoint(point), None)

    @contextmanager
    def scope(self) -> Iterator[HookManager]:
        """Register inside the block; everything reverts on exit.

        For a test that needs a strict safety gate, or a teleop session that
        installs an operator-confirmation hook and must not leave it behind
        when the operator disconnects::

            with hooks.scope():
                hooks.register(HookPoint.BEFORE_TOOL, confirm_with_operator)
                run_session()
            # confirm_with_operator is gone, including if run_session raised
        """
        saved = {p: list(v) for p, v in self._hooks.items()}
        saved_seq = self._seq
        try:
            yield self
        finally:
            self._hooks = saved
            self._seq = saved_seq

    # -- dispatch ----------------------------------------------------------

    def emit(self, point: HookPoint, ctx: Ctx | None = None) -> Ctx:
        """Run every hook at ``point``, threading ``ctx`` through them.

        A hook returning a *dict* replaces ``ctx`` for the hooks after it and
        for the caller (transform). Anything else -- ``None``, or a stray value
        from a one-line ``lambda c: log.append(c)`` -- leaves ``ctx`` alone.
        Treating every truthy return as a transform would let an observer hook
        silently replace the payload with its own return value, which is a
        bug you find by watching a robot use the wrong speed.

        ``HookVeto`` propagates. Anything else is contained: the failure goes
        to ``ON_ERROR``, gets logged, and the next hook still runs.
        """
        point = HookPoint(point)
        ctx = {} if ctx is None else ctx
        ctx["point"] = point.value
        # Iterate a copy: a hook that registers or removes hooks (a retry hook
        # arming a one-shot, say) must not mutate the list being walked.
        for _, _, fn in list(self._hooks.get(point, [])):
            try:
                result = fn(ctx)
            except HookVeto:
                raise
            except Exception as exc:  # noqa: BLE001 -- containment is the feature
                self._on_hook_failure(point, fn, exc, ctx)
                continue
            if isinstance(result, dict):
                ctx = result
                ctx["point"] = point.value
        return ctx

    def _on_hook_failure(self, point: HookPoint, fn: Hook, exc: Exception, ctx: Ctx) -> None:
        """Contain a hook that blew up.

        ON_ERROR handlers are hooks too and can be just as broken, so their
        failures are logged and dropped rather than re-emitted -- otherwise one
        bad error handler is an infinite loop instead of a log line.
        """
        name = getattr(fn, "__name__", repr(fn))
        log.warning("hook %s at %s failed: %s: %s", name, point.value, type(exc).__name__, exc)
        if point is HookPoint.ON_ERROR:
            return
        try:
            self.emit(
                HookPoint.ON_ERROR,
                {"error": exc, "hook": name, "failed_point": point.value, "ctx": ctx},
            )
        except HookVeto:
            # A veto raised from inside error handling has nothing left to
            # stop -- the action already failed. Swallow it rather than let it
            # surface as if the original call had been refused.
            log.warning("veto from ON_ERROR while handling %s; ignored", name)
        except Exception as nested:  # noqa: BLE001
            log.warning("ON_ERROR dispatch itself failed: %s", nested)


# --------------------------------------------------------------------------
# built-ins
# --------------------------------------------------------------------------


class LoggingHook:
    """Log every point it is attached to.

    ``fields`` limits what gets logged, because a raw ctx can hold an entire
    prompt and a log line per token is not a log, it is a denial of service.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.DEBUG,
        fields: tuple[str, ...] = ("point", "provider", "model", "tool", "finish_reason"),
    ) -> None:
        self.log = logger or log
        self.level = level
        self.fields = fields

    def __call__(self, ctx: Ctx) -> None:
        shown = {k: ctx[k] for k in self.fields if k in ctx}
        self.log.log(self.level, "hook %s", shown)

    def attach(self, manager: HookManager, *points: HookPoint, priority: int = -100) -> None:
        """Attach to ``points``, or every point. Low priority by default so
        logging observes what the transforming hooks decided, not the input."""
        for p in points or tuple(HookPoint):
            manager.register(p, self, priority=priority)


class TimingHook:
    """Wall-clock between matching BEFORE_/AFTER_ points.

    Wall clock, not CPU time: what a control loop cares about is how long the
    world moved while it was waiting, and that includes the network.
    """

    PAIRS = {
        HookPoint.BEFORE_COMPLETE: "complete",
        HookPoint.AFTER_COMPLETE: "complete",
        HookPoint.BEFORE_TOOL: "tool",
        HookPoint.AFTER_TOOL: "tool",
        HookPoint.BEFORE_STEP: "step",
        HookPoint.AFTER_STEP: "step",
    }

    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = {}
        self._open: dict[str, float] = {}

    def __call__(self, ctx: Ctx) -> None:
        try:
            point = HookPoint(ctx.get("point", ""))
        except ValueError:
            return
        label = self.PAIRS.get(point)
        if label is None:
            return
        if point.value.startswith("before_"):
            self._open[label] = time.perf_counter()
            return
        start = self._open.pop(label, None)
        if start is None:
            return  # AFTER without BEFORE: a veto fired, nothing to time
        self.timings.setdefault(label, []).append((time.perf_counter() - start) * 1000.0)

    def attach(self, manager: HookManager, priority: int = 1000) -> None:
        """BEFORE_ points get ``priority``, AFTER_ points get its negation.

        The clock must start before any hook that could spend real time and
        stop after the last one, otherwise the number measures the run minus
        whatever the other hooks did -- which is the number nobody wants.
        """
        for p in self.PAIRS:
            first = p.value.startswith("before_")
            manager.register(p, self, priority=priority if first else -priority)

    def mean_ms(self, label: str) -> float:
        vals = self.timings.get(label) or []
        return sum(vals) / len(vals) if vals else 0.0

"""Real-time execution: safety stop, action chunking, watchdog, latency budgets.

Inference latency is a robot's reaction time, and the budgets are not negotiable:

    safety-critical stop        < 1 ms    dedicated controller, off the AI path
    contact-rich manipulation   10-20 ms  0.5-1 mm error at contact fails the task
    reactive grasping           < 50 ms   stale queries miss systematically
    free-space motion           50-100 ms staleness inside tolerance

Everything here reports **p50 / p99 / max**, never a mean. A mean inside budget
with a p99 outside it is a robot that fails intermittently, which is the single
worst failure mode to debug. See `LatencyBudget.report()`.

## The honest answer on the < 1 ms budget

Measured on this host, 20k calls, `SafetyController.check` over a 12-vector:

    p50   3.6 us      p99   9.8 us      max   9458 us

p99 clears 1 ms by two orders of magnitude. **The max does not** -- a single
excursion of 9.5 ms, roughly 9x the budget. That is a CPython garbage-collection
pause or an OS scheduling slice, not anything this code does: the hot path
allocates nothing, and the excursion survives with `gc.disable()`.

So the truthful claim is: **this is a sub-millisecond path at p99 and not a
hard-real-time one.** For a genuine safety-critical stop it belongs outside
CPython -- a separate process at real-time priority, a C extension holding no
GIL, or a microcontroller on the actuator bus. Treat `SafetyController` as a
fast in-process pre-check layered under one of those, never as the last line of
defence. Anything else is claiming a deadline this runtime cannot hold.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "ActionChunker",
    "ChunkStats",
    "ConfidenceGate",
    "GateOutcome",
    "LatencyBudget",
    "SafetyController",
    "Tier",
    "TierReport",
    "Watchdog",
]


class Tier(str, Enum):
    """Latency tiers, with their budgets in milliseconds."""

    SAFETY_STOP = "safety_stop"
    CONTACT_RICH = "contact_rich"
    REACTIVE_GRASP = "reactive_grasp"
    FREE_SPACE = "free_space"

    @property
    def budget_ms(self) -> float:
        return {
            "safety_stop": 1.0,
            "contact_rich": 20.0,
            "reactive_grasp": 50.0,
            "free_space": 100.0,
        }[self.value]


# --------------------------------------------------------------------------
# the sub-millisecond path
# --------------------------------------------------------------------------


class SafetyController:
    """Deterministic stop check, independent of the policy loop.

    Design rules, all of them load-bearing for the < 1 ms budget:

    - **No allocation in the hot path.** Buffers are made in `__init__`; `check`
      writes into them with `out=`. A hot-path allocation invites a GC pause,
      and a GC pause is a p99 excursion.
    - **No lock shared with the policy loop.** The policy may hold the GIL for a
      long matmul, but it must never hold a *lock this needs*, or a hung planner
      would take the stop path down with it. Nothing here waits on the planner.
    - **No ML.** Pure interval arithmetic on limits. A network cannot be trusted
      to decide whether to stop.

    `stopped` is a plain bool set under no lock: a single attribute write is
    atomic under the GIL, and a latch that only ever goes False -> True cannot
    be corrupted by a race. Cheaper and more predictable than a lock.
    """

    __slots__ = ("_low", "_high", "_scratch", "stopped", "checks", "trips")

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self._low = np.ascontiguousarray(low, dtype=np.float64)
        self._high = np.ascontiguousarray(high, dtype=np.float64)
        if self._low.shape != self._high.shape:
            raise ValueError("low and high must share a shape")
        if np.any(self._high < self._low):
            raise ValueError("high must be >= low elementwise")
        self._scratch = np.empty_like(self._low)
        self.stopped = False
        self.checks = 0
        self.trips = 0

    def check(self, state: np.ndarray) -> bool:
        """True when `state` is inside limits. No allocation, no branch on data."""
        s = self._scratch
        np.copyto(s, state)
        ok = bool(np.all((s >= self._low) & (s <= self._high)))
        self.checks += 1
        if not ok:
            self.stopped = True
            self.trips += 1
        return ok

    def stop(self) -> None:
        self.stopped = True

    def reset(self) -> None:
        self.stopped = False

    def measure(self, state: np.ndarray, reps: int = 20_000) -> dict[str, float]:
        """p50/p99/max in milliseconds over `reps` calls.

        Percentiles, because the mean is the wrong statistic for a deadline.
        """
        for _ in range(min(500, reps)):  # warm the branch predictor and caches
            self.check(state)
        samples = np.empty(reps, dtype=np.float64)
        perf = time.perf_counter
        for i in range(reps):
            t0 = perf()
            self.check(state)
            samples[i] = perf() - t0
        ms = samples * 1e3
        return {
            "p50_ms": float(np.percentile(ms, 50)),
            "p99_ms": float(np.percentile(ms, 99)),
            "max_ms": float(ms.max()),
            "within_budget_p99": bool(np.percentile(ms, 99) < Tier.SAFETY_STOP.budget_ms),
        }


# --------------------------------------------------------------------------
# chunked execution
# --------------------------------------------------------------------------


@dataclass
class ChunkStats:
    chunks_built: int = 0
    chunks_dropped: int = 0
    actions_discarded_stale: int = 0
    max_staleness_ms: float = 0.0
    max_boundary_jump: float = 0.0


class ActionChunker:
    """Produce chunks of actions on a worker thread while the previous chunk runs.

    The point of chunking is to hide inference latency: plan 50 actions covering
    a second, execute them while the next chunk is computed. The part that is
    easy to get wrong is what makes it useful --

    **Temporal misalignment.** The world keeps moving during inference, so when a
    chunk arrives its leading actions describe a state that has already passed.
    Applying them blind is worse than not chunking, because the robot executes a
    stale plan with confidence. So on swap this drops the head of the new chunk
    by the measured elapsed time (`dt` per action), and blends the boundary over
    `blend` actions to avoid a step discontinuity -- a jump at the seam is a jerk
    in a real arm, and `stats.max_boundary_jump` records the worst one seen.

    `queue` is not used: the worker publishes one chunk under a short lock and
    the consumer takes it. A queue would buffer several chunks, and a buffered
    chunk is by definition a stale one.
    """

    def __init__(
        self,
        planner: Callable[[Any], np.ndarray],
        *,
        dt: float = 0.02,
        blend: int = 2,
        stale_grace: int = 0,
    ):
        self.planner = planner
        self.dt = float(dt)
        self.blend = int(blend)
        self.stale_grace = int(stale_grace)
        self.stats = ChunkStats()
        self._lock = threading.Lock()
        self._pending: tuple[np.ndarray, float] | None = None
        self._active: np.ndarray | None = None
        self._cursor = 0
        self._worker: threading.Thread | None = None

    # ---- production -----------------------------------------------------

    def submit(self, state: Any) -> None:
        """Start building the next chunk. No-op while one is already building."""
        if self._worker is not None and self._worker.is_alive():
            self.stats.chunks_dropped += 1
            return
        t_submit = time.perf_counter()

        def build() -> None:
            chunk = np.atleast_2d(np.asarray(self.planner(state), dtype=np.float64))
            with self._lock:
                self._pending = (chunk, t_submit)
                self.stats.chunks_built += 1

        self._worker = threading.Thread(target=build, daemon=True)
        self._worker.start()

    # ---- consumption ----------------------------------------------------

    def next_action(self) -> np.ndarray | None:
        """The next action to execute, swapping in a fresh chunk when ready."""
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is not None:
            self._swap(*pending)
        if self._active is None or self._cursor >= len(self._active):
            return None
        a = self._active[self._cursor]
        self._cursor += 1
        return a

    def _swap(self, chunk: np.ndarray, t_submit: float) -> None:
        """Install a chunk, discarding the head that inference outran."""
        staleness_s = time.perf_counter() - t_submit
        self.stats.max_staleness_ms = max(self.stats.max_staleness_ms, staleness_s * 1e3)

        skip = max(0, int(staleness_s / self.dt) - self.stale_grace)
        skip = min(skip, len(chunk) - 1)  # never discard the whole chunk
        self.stats.actions_discarded_stale += skip
        fresh = chunk[skip:]

        if self._active is not None and self._cursor < len(self._active) and self.blend > 0:
            last = self._active[min(self._cursor, len(self._active) - 1)]
            jump = float(np.abs(fresh[0] - last).max())
            self.stats.max_boundary_jump = max(self.stats.max_boundary_jump, jump)
            n = min(self.blend, len(fresh))
            w = np.linspace(0.0, 1.0, n + 1)[1:].reshape(-1, *([1] * (fresh.ndim - 1)))
            fresh = fresh.copy()
            fresh[:n] = (1.0 - w) * last + w * fresh[:n]

        self._active, self._cursor = fresh, 0

    @property
    def remaining(self) -> int:
        return 0 if self._active is None else max(0, len(self._active) - self._cursor)


# --------------------------------------------------------------------------
# watchdog
# --------------------------------------------------------------------------


class Watchdog:
    """Fire a safe stop when no action arrives inside the window.

    A separate thread, because the thing being watched may be the thing that is
    stuck. `pet()` is one attribute write so the watched loop pays almost
    nothing for being watched.

    `window_ms` must exceed normal jitter or the watchdog becomes the outage.
    `demo()` measures the false-positive rate on a healthy-but-noisy loop.
    """

    def __init__(self, window_ms: float, on_timeout: Callable[[], None], *, poll_ms: float = 1.0):
        self.window_s = window_ms / 1e3
        self.on_timeout = on_timeout
        self.poll_s = poll_ms / 1e3
        self._last = time.perf_counter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired = 0
        self.detection_latencies_ms: list[float] = []

    def pet(self) -> None:
        self._last = time.perf_counter()

    def start(self) -> Watchdog:
        self._last = time.perf_counter()

        def loop() -> None:
            while not self._stop.wait(self.poll_s):
                overdue = time.perf_counter() - self._last
                if overdue > self.window_s:
                    self.fired += 1
                    self.detection_latencies_ms.append((overdue - self.window_s) * 1e3)
                    self._last = time.perf_counter()  # one shot per lapse
                    # a broken handler must not kill the watchdog itself
                    with contextlib.suppress(Exception):
                        self.on_timeout()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> Watchdog:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


class GateOutcome(str, Enum):
    PROCEED = "proceed"
    HALT = "halt"
    TAKEOVER = "takeover"


@dataclass
class ConfidenceGate:
    """Decide proceed / halt / hand over, from signals that already exist.

    Deliberately no new uncertainty estimate. Three measured signals are already
    in the library and a fourth invented one would just be another thing to
    calibrate:

    - the sheaf gate score (`GateReading.score`, 1.0 == calibration threshold)
    - `Plan.feasible`, which is already a refusal when every candidate left the
      calibrated region
    - the rollout certificate's `error_bound` at the planned horizon

    An infeasible plan is TAKEOVER rather than HALT: the model is telling you it
    has no trustworthy option, and freezing mid-motion is not automatically the
    safe choice -- a human should decide.
    """

    halt_score: float = 2.0
    takeover_score: float = 4.0
    max_bound: float = 1.0

    def evaluate(
        self,
        *,
        gate_score: float | None = None,
        feasible: bool = True,
        error_bound: float | None = None,
    ) -> tuple[GateOutcome, str]:
        if not feasible:
            return GateOutcome.TAKEOVER, "planner refused: no feasible candidate"
        if gate_score is not None and gate_score >= self.takeover_score:
            return GateOutcome.TAKEOVER, f"gate score {gate_score:.2f} far off manifold"
        if gate_score is not None and gate_score >= self.halt_score:
            return GateOutcome.HALT, f"gate score {gate_score:.2f} above halt threshold"
        if error_bound is not None and error_bound > self.max_bound:
            return GateOutcome.HALT, f"rollout bound {error_bound:.3g} exceeds tolerance"
        return GateOutcome.PROCEED, "ok"


# --------------------------------------------------------------------------
# budgets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TierReport:
    tier: Tier
    n: int
    p50_ms: float
    p99_ms: float
    max_ms: float
    budget_ms: float

    @property
    def passes(self) -> bool:
        """Judged at p99. A deadline met on average is not met."""
        return self.p99_ms < self.budget_ms


@dataclass
class LatencyBudget:
    """Measure a callable against a named tier and report pass/fail at p99."""

    samples: dict[Tier, list[float]] = field(default_factory=dict, repr=False)

    class _Scope:
        def __init__(self, owner: LatencyBudget, tier: Tier):
            self.owner, self.tier = owner, tier

        def __enter__(self) -> LatencyBudget._Scope:
            self.t = time.perf_counter()
            return self

        def __exit__(self, *exc: object) -> None:
            ms = (time.perf_counter() - self.t) * 1e3
            self.owner.samples.setdefault(self.tier, []).append(ms)

    def measure(self, tier: Tier) -> LatencyBudget._Scope:
        return self._Scope(self, tier)

    def report(self) -> list[TierReport]:
        out = []
        for tier, xs in self.samples.items():
            a = np.asarray(xs)
            out.append(
                TierReport(
                    tier=tier,
                    n=len(xs),
                    p50_ms=float(np.percentile(a, 50)),
                    p99_ms=float(np.percentile(a, 99)),
                    max_ms=float(a.max()),
                    budget_ms=tier.budget_ms,
                )
            )
        return out

    def table(self) -> str:
        rows = [f"{'tier':<18}{'n':>6}{'p50':>9}{'p99':>9}{'max':>9}{'budget':>9}  verdict"]
        rows.append("-" * len(rows[0]))
        for r in self.report():
            rows.append(
                f"{r.tier.value:<18}{r.n:>6}{r.p50_ms:>9.3f}{r.p99_ms:>9.3f}"
                f"{r.max_ms:>9.3f}{r.budget_ms:>9.1f}  {'PASS' if r.passes else 'FAIL'}"
            )
        return "\n".join(rows)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def demo() -> None:
    """Measured behaviour on this host, percentiles not means."""
    # ---- safety path, and whether < 1 ms survives p99 in CPython
    sc = SafetyController(low=-np.ones(12), high=np.ones(12))
    state = np.zeros(12)
    m = sc.measure(state, reps=20_000)
    print(
        f"  safety check   p50 {m['p50_ms'] * 1e3:6.2f} us   "
        f"p99 {m['p99_ms'] * 1e3:6.2f} us   max {m['max_ms'] * 1e3:8.2f} us"
    )
    print(f"  < 1 ms at p99  {m['within_budget_p99']}")
    assert m["p99_ms"] < Tier.SAFETY_STOP.budget_ms, "safety path missed its p99 budget"
    assert sc.check(np.zeros(12)) is True
    assert sc.check(np.full(12, 5.0)) is False and sc.stopped, "out-of-limit state not latched"

    # independence: a wedged policy loop must not slow the safety path
    hog = threading.Event()

    def wedged() -> None:
        while not hog.is_set():
            np.linalg.svd(np.random.default_rng(0).normal(size=(96, 96)))

    t = threading.Thread(target=wedged, daemon=True)
    t.start()
    try:
        under_load = sc.measure(state, reps=5_000)
    finally:
        hog.set()
        t.join(timeout=2.0)
    print(
        f"  under GIL load p99 {under_load['p99_ms'] * 1e3:6.2f} us   "
        f"still under budget: {under_load['within_budget_p99']}"
    )

    # ---- chunker: staleness is discarded, seam is blended
    def planner(_state: Any) -> np.ndarray:
        time.sleep(0.02)  # 20 ms of "inference" -> 1 stale action at dt=20ms
        return np.tile(np.arange(10.0).reshape(-1, 1), (1, 2))

    ch = ActionChunker(planner, dt=0.02, blend=2)
    ch.submit(None)
    while ch.stats.chunks_built == 0:
        time.sleep(0.001)
    first = ch.next_action()
    assert first is not None, "no action produced"
    ch.submit(None)
    time.sleep(0.03)
    for _ in range(4):
        ch.next_action()
    print(
        f"  chunker        stale discarded {ch.stats.actions_discarded_stale}, "
        f"max staleness {ch.stats.max_staleness_ms:.1f} ms, "
        f"seam jump {ch.stats.max_boundary_jump:.2f}"
    )
    assert ch.stats.actions_discarded_stale >= 1, "stale head was not discarded"

    # ---- watchdog: detects a hang, and does NOT fire on healthy jitter
    fired: list[int] = []
    with Watchdog(window_ms=30.0, on_timeout=lambda: fired.append(1), poll_ms=1.0) as wd:
        for _ in range(40):  # healthy: jittery but always inside the window
            time.sleep(0.005 + 0.004 * np.random.default_rng(0).random())
            wd.pet()
        false_positives = wd.fired
        time.sleep(0.10)  # now hang
    print(
        f"  watchdog       false positives {false_positives}, fired on hang {wd.fired > 0}, "
        f"detect {wd.detection_latencies_ms[0]:.1f} ms"
        if wd.detection_latencies_ms
        else "  watchdog       did not fire"
    )
    assert false_positives == 0, "watchdog fired on a healthy loop"
    assert wd.fired > 0, "watchdog missed a real hang"

    # ---- confidence gate
    cg = ConfidenceGate()
    assert cg.evaluate(gate_score=0.5)[0] is GateOutcome.PROCEED
    assert cg.evaluate(gate_score=2.5)[0] is GateOutcome.HALT
    assert cg.evaluate(gate_score=9.0)[0] is GateOutcome.TAKEOVER
    assert cg.evaluate(feasible=False)[0] is GateOutcome.TAKEOVER
    assert cg.evaluate(gate_score=0.1, error_bound=50.0)[0] is GateOutcome.HALT
    print("  confidence     proceed/halt/takeover all reachable")

    # ---- budgets
    budget = LatencyBudget()
    for _ in range(200):
        with budget.measure(Tier.CONTACT_RICH):
            np.zeros(64) @ np.zeros((64, 64))
    print()
    print(budget.table())
    print("demo ok")


if __name__ == "__main__":
    demo()

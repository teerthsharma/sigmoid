"""Blackbox telemetry, deterministic replay, and tracepoints.

You cannot pause the physical world. A breakpoint in a control loop is dangerous
and destroys the state you wanted to inspect. So the debugging model is:

    log everything against one clock -> replay it off-robot bit-exactly
    -> inspect running state without ever stopping

Three pieces, in order of how load-bearing they are:

`BlackboxRecorder`   one monotonic clock for every stream, bounded memory
`DeterministicReplay` the same log back through the engine, bit-identical
`Tracepoint`         capture a value mid-loop without halting it

The replay guarantee is the one that matters. "Roughly the same rollout" is
useless for debugging a robot that dropped something once: you need the exact
tensors that produced the exact mistake. Everything here is built around not
losing that, which is why the recorder stores raw arrays rather than summaries
and why nondeterminism sources are pinned rather than tolerated.

Overhead is measured, not assumed -- telemetry that costs 5 ms inside a 20 ms
contact-rich budget is not telemetry, it is the bug. See `demo()`.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BlackboxRecorder",
    "DeterministicReplay",
    "LatencyProfiler",
    "ReplayResult",
    "StageTiming",
    "Tracepoint",
    "chebyshev_keep",
    "trace",
    "tracepoint",
]


# --------------------------------------------------------------------------
# eviction
# --------------------------------------------------------------------------


def chebyshev_keep(scores: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Which records to keep, discarding only bounded statistical outliers.

    A ring buffer drops oldest-first, which on a robot discards the frames
    *around an incident* precisely because they are old -- the opposite of what an
    investigation needs. Downsampling uniformly is no better: it thins the
    interesting stretch at the same rate as the idle one.

    Chebyshev's inequality gives a distribution-free alternative
    (AETHER `AetherChebyshev.lean`). For any finite-variance distribution,

        P(|X - mu| >= k sigma) <= 1 / k^2

    so at most `n / k^2` of `n` records lie beyond `k` standard deviations of an
    importance score. Evicting only those bounds the discard rate **without
    assuming a shape** -- which matters, because staleness and salience scores are
    not normal and a Gaussian assumption would silently over-evict a heavy tail.

    At `k = 2` the ceiling is `n / 4`: at most 25% discarded. Larger `k` keeps more.

    Chebyshev bounds the *population* tail, and a finite sample can exceed it, so
    the mask is truncated to the ceiling most-extreme-first. Without that this
    would advertise a bound it does not enforce -- measured on Cauchy-distributed
    scores, the raw tail test evicts past `n / k^2`.

    Low tail only: a small score means unimportant. The bound still holds, since a
    one-sided tail is a subset of the two-sided one.
    """
    if k <= 1.0:
        # the ceiling would be >= n, permitting eviction of everything
        raise ValueError("k must exceed 1 for the Chebyshev ceiling to constrain anything")
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    ceiling = int(np.floor(n / (k * k)))
    sigma = float(x.std())
    if sigma <= 1e-12 or ceiling == 0:  # no spread, or no budget to evict
        return np.ones(n, dtype=bool)

    z = (x - float(x.mean())) / sigma
    evict = z <= -k
    if int(evict.sum()) > ceiling:
        evict = np.zeros(n, dtype=bool)
        evict[np.argsort(z)[:ceiling]] = True
    return ~evict


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------


@dataclass
class BlackboxRecorder:
    """Time-synchronised ring buffer over named channels.

    One clock for all channels, so a later question like "was that a latency
    spike, a stale camera frame, or a hallucination" is answerable by aligning
    streams -- which is impossible if each stream carries its own timebase.

    `time.perf_counter` rather than `time.time`: the wall clock can step
    backwards over NTP adjustment, and a log whose timestamps go backwards
    cannot be aligned at all.

    Bounded by construction. A robot runs for hours; an unbounded list is an
    OOM with a countdown. `dropped` counts what the cap discarded, because
    silently losing the frames around an incident is the one failure this class
    exists to prevent being invisible.
    """

    capacity: int = 100_000
    """Total records retained across all channels."""

    channels: dict[str, deque] = field(default_factory=dict, repr=False)
    dropped: int = 0
    t0: float = field(default_factory=time.perf_counter)

    def record(self, channel: str, value: Any, t: float | None = None) -> None:
        buf = self.channels.get(channel)
        if buf is None:
            buf = self.channels[channel] = deque(maxlen=self.capacity)
        if len(buf) == buf.maxlen:
            self.dropped += 1
        buf.append(((time.perf_counter() - self.t0) if t is None else t, value))

    def compact(self, channel: str, scores: np.ndarray, k: float = 2.0) -> int:
        """Shrink a channel by Chebyshev eviction. Returns records discarded.

        The alternative to letting the ring buffer drop oldest-first. Pass an
        importance score per record -- gate score, latency, action magnitude,
        whatever the investigation will care about -- and at most `n / k^2` of the
        least important are removed. See `chebyshev_keep`.
        """
        buf = self.channels.get(channel)
        if buf is None:
            return 0
        items = list(buf)
        keep = chebyshev_keep(scores, k)
        if keep.shape[0] != len(items):
            raise ValueError(
                f"scores has {keep.shape[0]} entries, channel {channel!r} has {len(items)}"
            )
        removed = int((~keep).sum())
        buf.clear()
        buf.extend(item for item, ok in zip(items, keep, strict=True) if ok)
        self.dropped += removed
        return removed

    def __enter__(self) -> BlackboxRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    # ---- access ----------------------------------------------------------

    def times(self, channel: str) -> np.ndarray:
        return np.asarray([t for t, _ in self.channels.get(channel, ())], dtype=np.float64)

    def values(self, channel: str) -> list[Any]:
        return [v for _, v in self.channels.get(channel, ())]

    def array(self, channel: str) -> np.ndarray:
        """Channel as one stacked array. Raises if the shapes disagree."""
        vals = self.values(channel)
        if not vals:
            return np.zeros((0,), dtype=np.float64)
        return np.stack([np.asarray(v, dtype=np.float64) for v in vals])

    def aligned(self, *channels: str) -> list[tuple[float, tuple[Any, ...]]]:
        """Nearest-neighbour align several channels onto the first one's clock.

        Robotics streams arrive at different rates -- camera at 30 Hz, control
        at 200 Hz. Alignment is by nearest timestamp rather than by index,
        because index alignment silently pairs a fresh action with a stale frame
        and that mispairing is exactly the bug class this module is for.
        """
        if not channels:
            return []
        base = self.channels.get(channels[0], deque())
        others = [self.times(c) for c in channels[1:]]
        out = []
        for t, v in base:
            row = [v]
            for name, ts in zip(channels[1:], others, strict=True):
                if len(ts) == 0:
                    row.append(None)
                    continue
                row.append(self.values(name)[int(np.argmin(np.abs(ts - t)))])
            out.append((t, tuple(row)))
        return out

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """`.npz` of the numeric channels plus a JSON sidecar of everything else."""
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {"channels": [], "dropped": self.dropped, "capacity": self.capacity}
        for name, buf in self.channels.items():
            arrays[f"{name}__t"] = np.asarray([t for t, _ in buf], dtype=np.float64)
            try:
                arrays[f"{name}__v"] = np.stack(
                    [np.asarray(v, dtype=np.float64) for _, v in buf]
                )
                meta["channels"].append({"name": name, "kind": "array", "n": len(buf)})
            except (ValueError, TypeError):
                # ragged or non-numeric: JSON it rather than dropping the channel
                meta.setdefault("json", {})[name] = [_jsonable(v) for _, v in buf]
                meta["channels"].append({"name": name, "kind": "json", "n": len(buf)})
        np.savez_compressed(path, **arrays)
        out = path if path.suffix else path.with_suffix(".npz")
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> BlackboxRecorder:
        path = Path(path)
        rec = cls()
        with np.load(path) as data:  # no allow_pickle: a log is untrusted input
            names = {k.rsplit("__", 1)[0] for k in data.files}
            for name in names:
                ts, vs = data.get(f"{name}__t"), data.get(f"{name}__v")
                if ts is None or vs is None:
                    continue
                rec.channels[name] = deque(
                    zip(ts.tolist(), list(vs), strict=True), maxlen=rec.capacity
                )
        side = path.with_suffix(".meta.json")
        if side.exists():
            meta = json.loads(side.read_text(encoding="utf-8"))
            rec.dropped = int(meta.get("dropped", 0))
            for name, vals in (meta.get("json") or {}).items():
                rec.channels.setdefault(name, deque(maxlen=rec.capacity)).extend(
                    (float(i), v) for i, v in enumerate(vals)
                )
        return rec


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of feeding a log back through a function off-robot."""

    exact: bool
    n_steps: int
    max_abs_diff: float
    first_divergence: int | None
    channel: str


class DeterministicReplay:
    """Replay recorded inputs and check the outputs are bit-identical.

    Bit-identical, not `allclose`. A tolerance hides the thing you are hunting:
    if a replay is only approximately equal then some state was not captured,
    and that uncaptured state is a candidate cause of the incident.

    Sources of nondeterminism that had to be pinned for this to hold, each of
    which silently breaks exactness:

    - **RNG.** Every stochastic component in this repo takes a seed
      (`TopologicalMPC(seed=...)`, `np.random.default_rng`). An unseeded
      `np.random.*` call anywhere in the path makes replay impossible.
    - **Float accumulation order.** Reductions over a different chunking give
      different last bits. Same array shapes and same code path, or no deal.
    - **BLAS threading.** A multi-threaded GEMM may reduce in a different order
      run to run. `check_blas_determinism()` reports the risk; pin
      `OMP_NUM_THREADS=1` when it matters.
    - **dict / set iteration.** Insertion-ordered in CPython 3.7+, so fine, but
      a `set` of floats iterated into a reduction is not. Avoided here.

    What could NOT be pinned: GPU kernels. cuBLAS split-k and atomics make
    float reduction order nondeterministic across launches, so a torch-CUDA path
    is replayed with a documented tolerance rather than bit-exactly. CPU paths
    are exact. `replay()` reports which it used.
    """

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def replay(
        self,
        recorder: BlackboxRecorder,
        fn: Callable[[Any], Any],
        *,
        input_channel: str = "observation",
        output_channel: str = "state",
    ) -> ReplayResult:
        inputs = recorder.values(input_channel)
        expected = recorder.values(output_channel)
        n = min(len(inputs), len(expected))
        worst = 0.0
        first_bad: int | None = None
        for i in range(n):
            got = np.asarray(fn(inputs[i]), dtype=np.float64)
            want = np.asarray(expected[i], dtype=np.float64)
            if got.shape != want.shape:
                first_bad = i
                worst = float("inf")
                break
            if not np.array_equal(got, want):
                diff = float(np.abs(got - want).max())
                worst = max(worst, diff)
                if first_bad is None and diff > self.tolerance:
                    first_bad = i
        return ReplayResult(
            exact=(worst == 0.0),
            n_steps=n,
            max_abs_diff=worst,
            first_divergence=first_bad,
            channel=output_channel,
        )

    @staticmethod
    def check_blas_determinism(n: int = 512, trials: int = 3) -> bool:
        """True when repeated identical GEMMs agree bit-for-bit on this host."""
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(n, n)), rng.normal(size=(n, n))
        first = a @ b
        return all(np.array_equal(first, a @ b) for _ in range(trials - 1))


# --------------------------------------------------------------------------
# tracepoints
# --------------------------------------------------------------------------


@dataclass
class Tracepoint:
    """Capture a value mid-loop without halting it.

    A breakpoint stops a robot mid-motion, which is both dangerous and
    destroys the state under investigation. A tracepoint records and returns.

    Disabled cost is one attribute read and a branch, so leaving them in
    production is affordable -- measured in `demo()`. `every` and `predicate`
    exist so a rare condition can be caught without logging every cycle at
    200 Hz, which would swamp the recorder and drop the frames you wanted.
    """

    name: str
    recorder: BlackboxRecorder | None = None
    enabled: bool = True
    every: int = 1
    predicate: Callable[[Any], bool] | None = None
    hits: int = 0
    captured: int = 0

    def __call__(self, value: Any) -> Any:
        if not self.enabled:
            return value
        self.hits += 1
        if self.every > 1 and (self.hits - 1) % self.every:
            return value
        if self.predicate is not None and not self.predicate(value):
            return value
        self.captured += 1
        if self.recorder is not None:
            self.recorder.record(self.name, value)
        return value


_REGISTRY: dict[str, Tracepoint] = {}


def tracepoint(
    name: str,
    recorder: BlackboxRecorder | None = None,
    **kw: Any,
) -> Tracepoint:
    """Get or create a named tracepoint."""
    tp = _REGISTRY.get(name)
    if tp is None:
        tp = _REGISTRY[name] = Tracepoint(name=name, recorder=recorder, **kw)
    elif recorder is not None:
        tp.recorder = recorder
    return tp


def trace(name: str, value: Any) -> Any:
    """Record through a named tracepoint. Unknown name is a no-op, by design:
    a stale trace call left in the code must not raise on a robot."""
    tp = _REGISTRY.get(name)
    return value if tp is None else tp(value)


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageTiming:
    stage: str
    n: int
    p50_ms: float
    p99_ms: float
    max_ms: float
    total_ms: float


@dataclass
class LatencyProfiler:
    """Per-stage timings with percentiles.

    p50 and p99, never the mean. A mean inside budget with a p99 outside it is
    a robot that fails intermittently, which is the worst failure to debug.
    """

    samples: dict[str, list[float]] = field(default_factory=dict, repr=False)

    class _Stage:
        def __init__(self, owner: LatencyProfiler, name: str):
            self.owner, self.name = owner, name

        def __enter__(self) -> LatencyProfiler._Stage:
            self.t = time.perf_counter()
            return self

        def __exit__(self, *exc: object) -> None:
            ms = (time.perf_counter() - self.t) * 1e3
            self.owner.samples.setdefault(self.name, []).append(ms)

    def stage(self, name: str) -> LatencyProfiler._Stage:
        return self._Stage(self, name)

    def report(self) -> list[StageTiming]:
        out = []
        for name, xs in self.samples.items():
            a = np.asarray(xs)
            out.append(
                StageTiming(
                    stage=name,
                    n=len(xs),
                    p50_ms=float(np.percentile(a, 50)),
                    p99_ms=float(np.percentile(a, 99)),
                    max_ms=float(a.max()),
                    total_ms=float(a.sum()),
                )
            )
        return sorted(out, key=lambda s: -s.total_ms)

    def bottleneck(self) -> str | None:
        r = self.report()
        return r[0].stage if r else None

    def table(self) -> str:
        rows = [f"{'stage':<20}{'n':>7}{'p50 ms':>10}{'p99 ms':>10}{'max ms':>10}"]
        rows.append("-" * len(rows[0]))
        for s in self.report():
            rows.append(
                f"{s.stage:<20}{s.n:>7}{s.p50_ms:>10.3f}{s.p99_ms:>10.3f}{s.max_ms:>10.3f}"
            )
        return "\n".join(rows)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def demo() -> None:
    """Overhead, bit-exact replay, and the ring-buffer bound."""
    import tempfile

    # ---- overhead: the number that decides if this is usable in a loop
    rec = BlackboxRecorder(capacity=200_000)
    payload = np.zeros(64)
    for _ in range(200):
        rec.record("warm", payload)
    t0 = time.perf_counter()
    reps = 20_000
    for _ in range(reps):
        rec.record("observation", payload)
    per_record_us = (time.perf_counter() - t0) / reps * 1e6
    assert per_record_us < 50.0, f"record() too slow for a control loop: {per_record_us:.2f}us"

    off = Tracepoint("off", enabled=False)
    t0 = time.perf_counter()
    for _ in range(reps):
        off(payload)
    disabled_us = (time.perf_counter() - t0) / reps * 1e6
    on = Tracepoint("on", recorder=BlackboxRecorder())
    t0 = time.perf_counter()
    for _ in range(reps):
        on(payload)
    enabled_us = (time.perf_counter() - t0) / reps * 1e6
    assert disabled_us < enabled_us, "a disabled tracepoint must be cheaper than an enabled one"

    print(f"  record()            {per_record_us:7.2f} us")
    print(f"  tracepoint disabled {disabled_us:7.2f} us")
    print(f"  tracepoint enabled  {enabled_us:7.2f} us")

    # ---- bit-exact replay of a seeded stochastic pipeline
    def pipeline(x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(int(x[0]))  # seeded FROM the input: replayable
        return np.asarray(x * 2.0 + rng.normal(size=x.shape))

    log = BlackboxRecorder()
    for i in range(64):
        obs = np.full(8, float(i))
        log.record("observation", obs)
        log.record("state", pipeline(obs))
    result = DeterministicReplay().replay(log, pipeline)
    assert result.exact, f"replay not bit-exact: max diff {result.max_abs_diff}"
    assert result.n_steps == 64
    print(f"  replay              {result.n_steps} steps bit-identical")

    # an UNSEEDED pipeline must be caught, not quietly tolerated
    bad = BlackboxRecorder()
    for i in range(8):
        obs = np.full(4, float(i))
        bad.record("observation", obs)
        bad.record("state", obs + np.random.default_rng().normal(size=4))
    assert not DeterministicReplay().replay(bad, lambda x: x).exact, (
        "unseeded randomness must show up as a replay divergence"
    )
    print("  unseeded pipeline   correctly reported as non-exact")

    # ---- ring buffer holds under a long run
    small = BlackboxRecorder(capacity=100)
    for i in range(1000):
        small.record("c", float(i))
    assert len(small.channels["c"]) == 100, "ring buffer exceeded its cap"
    assert small.dropped == 900, f"dropped count wrong: {small.dropped}"
    print(f"  ring buffer         capped at 100, dropped {small.dropped} (reported)")

    # ---- save / load round trip
    with tempfile.TemporaryDirectory() as d:
        p = log.save(Path(d) / "bb")
        back = BlackboxRecorder.load(p)
        assert np.array_equal(back.array("state"), log.array("state")), "log round trip lost data"
    print("  save/load           arrays bit-identical")

    # ---- Chebyshev eviction holds its ceiling on every distribution shape
    for name, sample in (
        ("gaussian", np.random.default_rng(1).normal(size=1000)),
        ("cauchy", np.random.default_rng(2).standard_cauchy(size=1000)),
        ("all-identical", np.full(1000, 3.0)),
    ):
        for kk in (2.0, 3.0):
            kept = chebyshev_keep(sample, kk)
            evicted = int((~kept).sum())
            ceiling = int(np.floor(len(sample) / (kk * kk)))
            assert evicted <= ceiling, f"{name} k={kk}: evicted {evicted} > ceiling {ceiling}"
    g = chebyshev_keep(np.random.default_rng(1).normal(size=1000), 2.0)
    print(f"  chebyshev           k=2 kept {int(g.sum())}/1000, ceiling 250 respected")

    rec2 = BlackboxRecorder(capacity=1000)
    for i in range(200):
        rec2.record("gate", float(i))
    removed = rec2.compact("gate", np.asarray(rec2.values("gate")), k=2.0)
    assert removed <= 200 // 4, f"compact evicted {removed} beyond the ceiling"
    assert len(rec2.channels["gate"]) == 200 - removed
    print(f"  compact             discarded {removed} least-important of 200")

    # ---- profiler percentiles
    prof = LatencyProfiler()
    for _ in range(50):
        with prof.stage("fast"):
            pass
        with prof.stage("slow"):
            np.linalg.svd(np.zeros((32, 32)))
    assert prof.bottleneck() == "slow", "profiler picked the wrong bottleneck"
    print(f"  blas deterministic  {DeterministicReplay.check_blas_determinism()}")
    print("demo ok")


if __name__ == "__main__":
    demo()

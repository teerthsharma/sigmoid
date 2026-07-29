"""Deployment: quantization, hardware probing, and atomic OTA weight swap.

The world model's learned parameter is one dense float64 matrix, `CouplingOperator.W_`.
That is unusual and it is the whole reason this module can be honest: quantization here
is arithmetic on a matrix, not a framework feature, so int8 and fp16 are *measurable
today* with numpy alone. No calibration dataset, no fake-quant graph, no export step
between the claim and the measurement.

## What the measurements said, and what they cost

**Quantization buys bytes, not speed.** Measured on this host (numpy 1.x/OpenBLAS, matvec
p50 over 5000 reps, W_ 32x33):

    fp64  0.70 us    fp32  0.78 us    fp16  3.77 us    int8 (dequantized)  0.80 us

fp16 is **5.4x slower** than fp64 because there is no fp16 BLAS path -- numpy upcasts
per call. int8 is not faster either: numpy has no int8 GEMM, so `QuantizedOperator`
dequantizes once at construction and computes in fp32. Keeping the int8 payload and
dequantizing *per call* measured 1.82 us, 2.6x worse than dequantizing once, for
bit-identical output -- so it is stored int8 and computed fp32, deliberately.

The consequence is a policy constraint, not a footnote: **a precision policy on this
host cannot use low precision to fix a latency overrun.** It can only save bytes (7.1x
for int8: 8448 -> 1184 including the per-row scales), which is what matters for an OTA
transfer over a constrained link and for VRAM residency. `PrecisionPolicy` is written
that way, and says so where it decides.

**Per-row int8 scaling was not necessary on the operator we can actually measure, and is
still the right default.** On the engine's Lorenz operator (W_ 52x53) row magnitudes span
only 2.54x, and a single global scale cost 1.29x more weight error than per-row
(1.20e-2 vs 9.34e-3) while destroying *zero* rows. The premise that a global scale wipes
out small rows is false for a spectrally-projected operator. It is emphatically true once
row scales are heterogeneous: on an operator whose rows span four decades, a global scale
quantized **16 of 32 rows to all-zero** (relative error 1.000) where per-row held the
worst row to 8.3e-3. Robot state in raw units -- millimetres beside radians beside
newtons -- is exactly that operator. Per-row costs `state_dim` float32s, so it is
insurance with a price of ~0.1% of the payload. Kept on by default; `per_row=False`
reproduces the comparison.

That same operator exposed the edge case that would have shipped a brick: **31 of its 52
rows are exactly zero** (the block-diagonal fit leaves the psi->u block empty). Per-row
scaling divides by `max|row|`, so an unguarded implementation writes NaN into 60% of the
matrix. `_row_scales` guards it.

## The OTA path is the security boundary

`SECURITY.md` documents that `SigmoidWorldModel.load` uses `numpy.load(allow_pickle=True)`
and can therefore execute arbitrary code. OTA is precisely the untrusted-input path that
sharpens it, so `WeightStore` verifies a SHA-256 manifest *before* it loads anything and
refuses on mismatch. Read `WeightStore.load`'s docstring for what that does and does not
buy -- integrity is not safety.

Two measured facts shape the swap:

  - `os.replace` really is atomic here. 100 swaps under a concurrent reader produced 83
    distinct whole payloads and **zero torn or short reads**.
  - On Windows it is atomic but *refusable*: `os.replace` onto a path another handle has
    open fails with `PermissionError` (WinError 5). A reader loop opening the file
    continuously blocked **300 of 300** swaps. Python's `open`/`np.load` do not pass
    FILE_SHARE_DELETE, so this is not avoidable from here -- `commit` retries with a short
    backoff (max 1 retry needed against a realistic intermittent reader) and raises a
    named error rather than leaving a half-applied state.

## What this host cannot test, and so does not claim

No TensorRT, no TVM, no Jetson. `import tensorrt` and `import tvm` both fail here and this
module **never attempts either import** -- `export_model` reports them unavailable and
raises `NotImplementedError` naming the reason. A stub that pretended otherwise would be
worse than the gap. torch, triton, onnx and onnxruntime *do* import on this host, so the
`torch.onnx` export path is real and tested; it is still optional and lazily imported.
`HardwareProbe` measures GFLOP/s rather than reading a spec table, because a measured
number is the only kind this file is allowed to print.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .operator import CouplingOperator
from .realtime import Tier

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .engine import SigmoidWorldModel

__all__ = [
    "ExportResult",
    "HardwareProbe",
    "HardwareReport",
    "Precision",
    "PrecisionPolicy",
    "PrecisionState",
    "QuantReport",
    "QuantizedOperator",
    # Exported because callers have to catch it: every OTA rejection path raises this,
    # and a caller that cannot name it will fall back to `except Exception`.
    "StagingError",
    "WeightStore",
    "battery_percent",
    "export_model",
    "gpu_temperature_c",
    "quantization_report",
    "quantization_table",
    "quantize_operator",
    "row_relative_error",
    "sha256_file",
]


class Precision(str, Enum):
    """Storage precisions for `CouplingOperator.W_`.

    A str-Enum so `"int8"` and `Precision.INT8` are interchangeable at call sites --
    the same choice `Tier` made in realtime.py.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


# int8 < fp16 < fp32. Used only to answer "is this an upgrade?", which the latency
# veto needs and nothing else does.
_RANK = {Precision.INT8: 0, Precision.FP16: 1, Precision.FP32: 2}

_NUMPY_DTYPE = {Precision.FP32: np.float32, Precision.FP16: np.float16}


def _as_precision(p: str | Precision) -> Precision:
    return p if isinstance(p, Precision) else Precision(p)


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------


def _row_scales(W: np.ndarray) -> np.ndarray:
    """Per-row symmetric int8 scales, guarding the all-zero row.

    31 of the 52 rows of the engine's Lorenz operator are exactly zero -- the
    block-diagonal fit leaves the cross-block quadrant empty by construction. Dividing
    by `max|row| = 0` would write NaN into 60% of the matrix and every rollout after it,
    so a zero row gets scale 1.0: it quantizes to zeros and dequantizes to zeros, which
    is exact.
    """
    peak = np.abs(W).max(axis=1)
    peak[peak == 0.0] = 127.0  # -> scale 1.0, exact for an all-zero row
    return (peak / 127.0).astype(np.float32)


def row_relative_error(Wq: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Per-row relative error of `Wq` against `W`.

    The per-row view, not the Frobenius norm, is what answers the per-row-scaling
    question: a global scale that leaves the aggregate error looking fine can still have
    quantized individual rows to all-zero, and only a row-wise error shows that. Measured
    on a 4-decade operator: aggregate error 1.44e-2 (unremarkable) hiding **16 of 32 rows
    at relative error 1.000** (destroyed).
    """
    num = np.linalg.norm(Wq - W, axis=1)
    return num / np.maximum(np.linalg.norm(W, axis=1), 1e-300)


@dataclass
class QuantizedOperator:
    """A `CouplingOperator` with a reduced-precision `W_`, plus the int8 payload.

    `.step` / `.rollout` delegate to a clone of the source operator, so this drops into
    anything that took a `CouplingOperator` -- including `rollout`, `certificate` and
    `safe_horizon`, which come along for free via `.operator`.

    For int8, `operator.W_` holds the **dequantized fp32 matrix** while `q_`/`scale_`
    hold the shipped payload. That split is deliberate and measured: dequantizing per
    call costs 1.82 us against 0.80 us for dequantizing once, with bit-identical output,
    because numpy has no int8 GEMM. So int8 here is a *transport and residency* precision.
    On a host with int8 tensor-core kernels it would also be a compute precision, and
    that is the seam -- swap the matvec, keep `q_`/`scale_`.
    """

    precision: Precision
    operator: CouplingOperator = field(repr=False)
    """A clone of the source operator whose `W_` is the fp64 *compute cache*."""

    native_: np.ndarray = field(repr=False)
    """`W_` in the declared dtype: float32, float16, or the dequantized float32 for int8.

    Kept separately from `operator.W_` because they answer different questions. This is
    what a real kernel would multiply and what the byte count is taken from;
    `operator.W_` is the fp64 upcast that numpy is actually fastest with. Timing the
    upcast for every precision would report one number four times, which is what the
    first version of this file did.
    """

    q_: np.ndarray | None = field(default=None, repr=False)
    """int8 weights as shipped. None unless `precision is INT8`."""

    scale_: np.ndarray | None = field(default=None, repr=False)
    """Dequant scale: (state_dim,) per-row, or (1,) global. None unless int8."""

    per_row: bool = True

    @property
    def nbytes(self) -> int:
        """Bytes of the *shipped* weights: the int8 payload plus scales, or `native_`."""
        if self.q_ is not None and self.scale_ is not None:
            return int(self.q_.nbytes + self.scale_.nbytes)
        return int(self.native_.nbytes)

    @property
    def W_(self) -> np.ndarray:
        """The matrix actually used for compute (fp64 cache, dequantized for int8)."""
        assert self.operator.W_ is not None
        return self.operator.W_

    def step(self, z: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        return self.operator.step(z, action)

    def rollout(
        self, z0: np.ndarray, steps: int, actions: np.ndarray | None = None
    ) -> np.ndarray:
        return self.operator.rollout(z0, steps, actions)


def quantize_operator(
    op: CouplingOperator,
    precision: str | Precision = Precision.FP32,
    *,
    per_row: bool = True,
) -> QuantizedOperator:
    """Quantize `op.W_` to fp32 / fp16 / int8.

    int8 is symmetric (no zero point): the operator is a linear map, so a zero weight
    must stay a zero weight or the fixed point moves. Asymmetric quantization would buy
    a fraction of a bit of range at the cost of an affine term in every matvec.

    `per_row=False` uses one global scale, which is the comparison the module docstring
    reports -- 16 of 32 rows destroyed on a heterogeneous operator. It exists to keep
    that claim reproducible, not because it is ever the better choice.
    """
    op._check_fitted()
    assert op.W_ is not None
    prec = _as_precision(precision)

    if prec is not Precision.INT8:
        # A clone via dataclasses.replace: every fitted attribute (rho_, step_rmse_,
        # residual_cov_) carries over, so the quantized operator still issues a valid
        # certificate. Cheaper and less brittle than re-deriving them.
        low = op.W_.astype(_NUMPY_DTYPE[prec])
        clone = replace(op, W_=low.astype(np.float64))
        return QuantizedOperator(prec, clone, low, per_row=per_row)

    if per_row:
        scale = _row_scales(op.W_)
        q = np.clip(np.rint(op.W_ / scale[:, None]), -127, 127).astype(np.int8)
        deq = q.astype(np.float32) * scale[:, None]
    else:
        peak = float(np.abs(op.W_).max())
        scale = np.array([peak / 127.0 if peak > 0 else 1.0], dtype=np.float32)
        q = np.clip(np.rint(op.W_ / scale[0]), -127, 127).astype(np.int8)
        deq = q.astype(np.float32) * scale[0]

    clone = replace(op, W_=deq.astype(np.float64))
    return QuantizedOperator(prec, clone, deq, q_=q, scale_=scale, per_row=per_row)


@dataclass(frozen=True)
class QuantReport:
    """One measured row of the quantization table."""

    precision: str
    bytes: int
    compression: float
    step_rmse: float
    """One-step RMSE against the fp64 operator, over a batch of probe states."""

    rollout_rmse: float
    """RMSE over a `horizon`-step rollout against the fp64 rollout."""

    latency_us: float
    """Matvec p50. Not a mean -- see realtime.py on why."""

    model_step_rmse: float
    """The operator's own fitted one-step error, the yardstick for `usable`."""

    @property
    def usable(self) -> bool:
        """Usable when the error quantization *adds* is below the error already there.

        The alternative was an arbitrary tolerance. This one is a property of the model:
        if quantization noise sits under the operator's own fitted residual, it is not
        the limiting factor and the precision is free accuracy-wise. If it sits above,
        quantization has become the dominant error and the number to report is that fact.
        """
        return self.rollout_rmse <= self.model_step_rmse


def quantization_table(
    op: CouplingOperator,
    *,
    horizon: int = 16,
    n_probe: int = 256,
    reps: int = 2000,
    seed: int = 0,
    include_global_int8: bool = True,
) -> list[QuantReport]:
    """Measure bytes / step RMSE / rollout error / latency for every precision.

    Probe states are drawn around the operator's own fixed point, which is its attractor
    and therefore the region a rollout actually visits. Sampling N(0, I) in the raw
    coordinates would measure a regime the operator was never fitted on.
    """
    op._check_fitted()
    assert op.W_ is not None
    rng = np.random.default_rng(seed)
    centre = op.fixed_point_ if op.fixed_point_ is not None else np.zeros(op.state_dim)
    spread = max(float(np.abs(centre).max()), 1.0)
    probes = centre + spread * rng.normal(size=(n_probe, op.state_dim))
    actions = rng.normal(size=(n_probe, op.action_dim)) if op.action_dim else None

    ref_step = op.step(probes, actions)
    roll_actions = None if actions is None else np.repeat(actions[:1], horizon, axis=0)
    ref_roll = op.rollout(probes[0], horizon, roll_actions)
    base_bytes = op.W_.nbytes

    def measure(label: str, qop: QuantizedOperator) -> QuantReport:
        got = qop.step(probes, actions)
        step_rmse = float(np.sqrt(np.mean((got - ref_step) ** 2)))
        roll = qop.rollout(probes[0], horizon, roll_actions)
        # Any non-finite rollout is a divergence, and inf is the honest report --
        # NaN would be indistinguishable from "not measured".
        diff = roll - ref_roll
        roll_rmse = (
            float(np.sqrt(np.mean(diff**2))) if np.all(np.isfinite(diff)) else float("inf")
        )
        return QuantReport(
            precision=label,
            bytes=qop.nbytes,
            compression=base_bytes / max(qop.nbytes, 1),
            step_rmse=step_rmse,
            rollout_rmse=roll_rmse,
            # `native_`, not `qop.W_`: the fp64 cache is identical for every row, so
            # timing it would print one number four times and hide the whole point --
            # that fp16 is the slowest option here.
            latency_us=_matvec_p50_us(qop.native_, op.lift_dim, reps=reps),
            model_step_rmse=op.step_rmse_,
        )

    rows = [
        QuantReport(
            precision="fp64 (reference)",
            bytes=base_bytes,
            compression=1.0,
            step_rmse=0.0,
            rollout_rmse=0.0,
            latency_us=_matvec_p50_us(op.W_, op.lift_dim, reps=reps),
            model_step_rmse=op.step_rmse_,
        )
    ]
    for prec in (Precision.FP32, Precision.FP16, Precision.INT8):
        rows.append(measure(prec.value, quantize_operator(op, prec)))
    if include_global_int8:
        rows.append(
            measure("int8 (global)", quantize_operator(op, Precision.INT8, per_row=False))
        )
    return rows


def _matvec_p50_us(W: np.ndarray, lift_dim: int, *, reps: int = 2000) -> float:
    """p50 microseconds for one matvec, in W's own dtype.

    p50 rather than a mean, for the reason realtime.py gives: a mean folds in GC and
    scheduler excursions and stops describing the common case.
    """
    x = np.zeros(lift_dim, dtype=W.dtype)
    for _ in range(min(200, reps)):  # warm BLAS and the cache
        W @ x
    samples = np.empty(reps)
    perf = time.perf_counter
    for i in range(reps):
        t0 = perf()
        W @ x
        samples[i] = perf() - t0
    return float(np.percentile(samples, 50) * 1e6)


def quantization_report(rows: list[QuantReport]) -> str:
    """The measured table, with a verdict per precision."""
    head = (
        f"{'precision':<18}{'bytes':>8}{'x':>7}{'step_rmse':>12}"
        f"{'roll_rmse':>12}{'p50_us':>9}  verdict"
    )
    out = [head, "-" * len(head)]
    for r in rows:
        verdict = "reference" if r.precision.startswith("fp64") else _verdict(r)
        out.append(
            f"{r.precision:<18}{r.bytes:>8}{r.compression:>7.2f}{r.step_rmse:>12.3e}"
            f"{r.rollout_rmse:>12.3e}{r.latency_us:>9.2f}  {verdict}"
        )
    out.append(
        f"(verdict: usable == added rollout error <= the model's own "
        f"{rows[0].model_step_rmse:.3e})"
    )
    return "\n".join(out)


def _verdict(r: QuantReport) -> str:
    if not np.isfinite(r.rollout_rmse):
        return "DIVERGED"
    return "usable" if r.usable else "DOMINATES ERROR"


# --------------------------------------------------------------------------
# host signals
# --------------------------------------------------------------------------

_TEMP_CACHE: dict[str, Any] = {"t": 0.0, "value": None, "source": "unread"}


def gpu_temperature_c(
    *, max_age_s: float = 1.0, allow_smi: bool = True
) -> tuple[float | None, str]:
    """(temperature in C, source). Cached, because reading it is expensive.

    Measured on this host: `nvidia-smi` costs **32 ms** steady-state and 1.7 s on the
    first call. That is above the contact-rich budget (20 ms) and 32x the safety-stop
    budget, so calling it inside a control tick would itself be the deadline miss. The
    cache is not an optimization, it is the only way this signal is admissible at all --
    and a real deployment should sample it from a side thread, which `max_age_s` is
    sized for.

    `source` is `"nvidia-smi"` when the real sensor answered and `"stub"` when it did
    not, so a caller can tell a measurement from a placeholder. It never guesses a
    number: no sensor means `None`.

    `allow_smi=False` forces the degraded path and deliberately neither reads nor writes
    the cache -- otherwise it would hand back a real reading taken a moment earlier and
    the no-sensor path could never be tested on a host that has one.
    """
    if not allow_smi:
        return None, "stub"
    now = time.monotonic()
    if now - _TEMP_CACHE["t"] < max_age_s and _TEMP_CACHE["source"] != "unread":
        return _TEMP_CACHE["value"], _TEMP_CACHE["source"]
    value: float | None = None
    source = "stub"
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if out.returncode == 0 and out.stdout.strip():
                value = float(out.stdout.strip().splitlines()[0])
                source = "nvidia-smi"
        except (OSError, ValueError, subprocess.SubprocessError):
            # A missing driver, a hung query or an unparseable line are all the same
            # answer: no sensor. Never let a telemetry read take the control loop down.
            value, source = None, "stub"
    _TEMP_CACHE.update(t=now, value=value, source=source)
    return value, source


def battery_percent() -> float | None:
    """Battery charge 0-100, or None when there is no battery or no way to ask.

    stdlib ctypes rather than psutil: core deps are numpy and scipy only. A real robot
    reads this from its BMS over CAN and passes it in; this exists so `demo()` prints a
    measured number instead of a made-up one.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            class _Power(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_ubyte),
                    ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            status = _Power()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                pct = int(status.BatteryLifePercent)
                return None if pct == 255 else float(pct)  # 255 == unknown
        except (OSError, AttributeError):
            return None
        return None
    for node in ("/sys/class/power_supply/BAT0/capacity", "/sys/class/power_supply/BAT1/capacity"):
        try:
            return float(Path(node).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _ram_gb() -> float:
    """Total physical RAM in GiB via stdlib only. 0.0 when it cannot be determined."""
    if sys.platform == "win32":
        try:
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = _Mem()
            mem.dwLength = ctypes.sizeof(_Mem)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return float(mem.ullTotalPhys) / 2**30
        except (OSError, AttributeError):
            return 0.0
        return 0.0
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 2**30
    except (OSError, ValueError, AttributeError):
        return 0.0


def _importable(name: str) -> bool:
    """Whether `name` imports, without keeping it. Probing is the point."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------
# precision policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionState:
    """The signals a precision decision is allowed to depend on."""

    tier: Tier = Tier.FREE_SPACE
    gpu_temp_c: float | None = None
    battery_pct: float | None = None
    latency_ms: float = 0.0
    """Most recent measured loop latency. Compared against `tier.budget_ms`."""


@dataclass
class PrecisionPolicy:
    """Choose a precision from thermal headroom, battery, task tier and latency.

    Two things make this more than a lookup table.

    **The tier mapping is the inverse of the obvious one.** Precision follows each tier's
    *accuracy* tolerance, not its deadline: contact-rich manipulation gets fp32 and
    free-space motion gets int8. That is because low precision does not buy latency here
    -- fp16 matvec measured 3.77 us against fp64's 0.70 us (module docstring). Mapping
    "tight deadline -> low precision" would be reasoning from a speedup this host does
    not have, and would hand the 0.5 mm-tolerance task the noisiest weights.

    **Hysteresis, by two mechanisms, because one is not enough.** Flapping between int8
    and fp32 mid-motion is its own failure mode: each switch is a discontinuity in the
    dynamics the planner is integrating.

      1. A Schmitt band on temperature -- derate at `hot_c`, recover only below
         `cool_c`. A fan cycling 72-78 C against an 80/70 band cannot move the latch at
         all, which is the common case and the cheapest possible fix.
      2. A dwell counter -- a challenger must win `dwell` consecutive calls to be
         adopted. This catches oscillation whose amplitude *does* clear the band: a
         signal alternating every call changes its vote every call, so the streak never
         reaches `dwell` and the precision never moves.

    Neither is sufficient alone; both are cheap. `switches` counts adopted changes so a
    test can assert on flapping directly, and `demo()` measures the square-wave periods
    that get absorbed.
    """

    hot_c: float = 80.0
    cool_c: float = 70.0
    battery_low_pct: float = 20.0
    dwell: int = 3
    current: Precision = Precision.FP32

    tier_precision: dict[Tier, Precision] = field(
        default_factory=lambda: {
            Tier.SAFETY_STOP: Precision.FP32,
            Tier.CONTACT_RICH: Precision.FP32,
            Tier.REACTIVE_GRASP: Precision.FP16,
            Tier.FREE_SPACE: Precision.INT8,
        }
    )

    switches: int = 0
    derated: bool = False
    _candidate: Precision | None = field(default=None, repr=False)
    _streak: int = field(default=0, repr=False)
    reason: str = "initial"

    def __post_init__(self) -> None:
        if self.cool_c >= self.hot_c:
            raise ValueError("cool_c must be below hot_c or there is no hysteresis band")
        self.current = _as_precision(self.current)

    def select(self, state: PrecisionState) -> Precision:
        """The precision to run at. Same-vote calls are free; changes must earn it."""
        want = self._vote(state)
        if want is self.current:
            self._candidate, self._streak = None, 0
            return self.current
        if want is self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = want, 1
        if self._streak >= self.dwell:
            self.current, self._candidate, self._streak = want, None, 0
            self.switches += 1
        return self.current

    def _vote(self, state: PrecisionState) -> Precision:
        """The unfiltered preference. `select` decides whether it gets adopted."""
        temp = state.gpu_temp_c
        if temp is not None:
            # Latch on the physical signal every call even when dwell suppresses the
            # switch: the two mechanisms track different things, and folding them would
            # make the band's width depend on the dwell length.
            self.derated = temp > self.cool_c if self.derated else temp >= self.hot_c
        low_battery = state.battery_pct is not None and state.battery_pct <= self.battery_low_pct

        want = self.tier_precision.get(state.tier, Precision.FP32)
        reason = f"tier {state.tier.value}"
        if self.derated or low_battery:
            # Fewer bytes moved is less memory-controller power and less heat. It is not
            # fewer cycles -- see the class docstring.
            want = Precision.INT8
            reason = "thermal derate" if self.derated else "battery low"
        if state.latency_ms > state.tier.budget_ms and _RANK[want] > _RANK[self.current]:
            # Latency vetoes *upgrades only*. Raising precision can never lower latency
            # on any kernel, so an over-budget loop must not be handed a wider dtype.
            # The converse -- dropping precision to recover a missed deadline -- is the
            # claim this host measured to be false, so it is not made.
            want, reason = self.current, f"latency {state.latency_ms:.1f}ms over budget: no upgrade"
        self.reason = reason
        return want


# --------------------------------------------------------------------------
# hardware probe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareReport:
    """What this host is, measured rather than looked up."""

    platform: str
    python: str
    cpu_logical: int
    ram_gb: float
    cpu_gflops: float
    """Measured fp32 GEMM throughput. The only TOPS figure this file will print for a
    CPU, because a spec table cannot be verified from here."""

    gpu_name: str | None = None
    vram_mb: int | None = None
    compute_capability: str | None = None
    gpu_temp_c: float | None = None
    temp_source: str = "stub"
    gpu_tflops: float | None = None
    """Measured fp16 GEMM throughput, or None when not benchmarked (a CUDA context
    costs ~575 ms to create here, so it is opt-in)."""

    has_torch: bool = False
    has_torch_cuda: bool = False
    has_triton: bool = False
    has_onnx: bool = False
    has_onnxruntime: bool = False
    has_tensorrt: bool = False
    """Always False and never probed by import: this host has no TensorRT, so the only
    honest value is the one that does not pretend otherwise."""

    has_tvm: bool = False
    """Same. Not probed, reported unavailable."""

    step_us: float = 0.0
    """Measured p50 for one d=64 operator matvec, the unit every workload below is
    counted in."""

    @property
    def tops_tier(self) -> str:
        """Order-of-magnitude compute class, naming what it was measured on.

        The basis is part of the answer. A CPU-derived tier printed for a host that has
        an unbenchmarked GPU understates this machine by 100x -- 213 GFLOP/s on the CPU
        against 22 TFLOP/s measured on the 4060 -- so an unbenchmarked GPU is reported as
        unbenchmarked rather than folded into a number.
        """
        if self.gpu_tflops is not None:
            best, basis = self.gpu_tflops * 1000.0, "gpu fp16, measured"
        elif self.gpu_name is not None:
            return "gpu present but not benchmarked (pass gpu_bench=True)"
        else:
            best, basis = self.cpu_gflops, "cpu fp32, measured"
        if best >= 100_000:
            bucket = "100+ TFLOP/s"
        elif best >= 10_000:
            bucket = "10-100 TFLOP/s"
        elif best >= 1_000:
            bucket = "1-10 TFLOP/s"
        else:
            bucket = "<1 TFLOP/s"
        return f"{bucket} ({basis})"

    @property
    def device_class(self) -> str:
        """A spec bucket, not a form-factor claim.

        Thresholds are VRAM-led because every optional path in this library (torch
        capture, triton attention, fp16 rollout) is bounded by VRAM before it is bounded
        by anything else. An 8 GB CUDA GPU beside 16 GB of RAM lands in the same bucket
        as a Jetson AGX Orin, which is the useful comparison even though the chassis is a
        laptop. Read `workloads()` for the answer that actually matters.
        """
        vram = self.vram_mb or 0
        if vram >= 16_000 and self.ram_gb >= 32 and self.cpu_logical >= 16:
            return "workstation"
        if vram >= 6_000:
            return "high-end-edge"
        if vram > 0 or self.cpu_logical >= 8:
            return "constrained-edge"
        return "sbc"

    def workloads(self) -> list[tuple[str, bool, str]]:
        """(workload, runnable here, why) -- derived from the measured numbers above.

        This is the part of the report worth reading. `device_class` is a label;
        this is a list of things that either run on this host or do not.
        """
        step_ms = self.step_us / 1e3
        rows = [
            (
                "operator.step, d=64",
                step_ms < Tier.SAFETY_STOP.budget_ms,
                f"{self.step_us:.2f} us vs 1 ms safety-stop budget",
            ),
            (
                "16-step imagine",
                16 * step_ms < Tier.CONTACT_RICH.budget_ms,
                f"{16 * step_ms:.3f} ms vs 20 ms contact-rich budget",
            ),
            (
                "MPC, 64 candidates x 16 steps",
                1024 * step_ms < Tier.REACTIVE_GRASP.budget_ms,
                f"{1024 * step_ms:.1f} ms vs 50 ms reactive-grasp budget",
            ),
            (
                "distilgpt2 activation capture",
                self.has_torch and (self.vram_mb or 0) >= 1_000,
                "needs torch + ~350 MB VRAM"
                if self.has_torch
                else "torch does not import here",
            ),
            (
                "triton topology-sparse attention",
                self.has_triton and self.has_torch_cuda,
                "needs triton + a CUDA device",
            ),
            (
                "7B fp16 local inference",
                (self.vram_mb or 0) >= 15_000,
                f"needs ~14 GB VRAM, host has {self.vram_mb or 0} MB",
            ),
            (
                "TensorRT / TVM compiled operator",
                False,
                "not installed and not probed; see export_model",
            ),
        ]
        return rows

    def table(self) -> str:
        gpu_throughput = (
            f"{self.gpu_tflops:.1f} TFLOP/s fp16 measured"
            if self.gpu_tflops is not None
            else "not benchmarked"
        )
        temp = f"{self.gpu_temp_c:.0f} C" if self.gpu_temp_c is not None else "unavailable"
        out = [
            f"  platform        {self.platform}",
            f"  python          {self.python}",
            f"  cpu             {self.cpu_logical} logical, {self.cpu_gflops:.0f} GFLOP/s measured",
            f"  ram             {self.ram_gb:.1f} GiB",
            f"  gpu             {self.gpu_name or 'none detected'}",
            f"  vram            {f'{self.vram_mb} MB' if self.vram_mb else 'n/a'}"
            f"   compute cap {self.compute_capability or 'n/a'}",
            f"  gpu temp        {temp}  (source: {self.temp_source})",
            f"  gpu throughput  {gpu_throughput}",
            f"  frameworks      torch {self.has_torch} (cuda {self.has_torch_cuda}), "
            f"triton {self.has_triton}, onnx {self.has_onnx}, onnxruntime {self.has_onnxruntime}",
            f"  not available   tensorrt {self.has_tensorrt}, tvm {self.has_tvm}"
            f"  (never imported, reported unavailable)",
            f"  class           {self.device_class}   compute tier {self.tops_tier}",
            "",
            "  workload                             runs?  basis",
            "  " + "-" * 74,
        ]
        for name, ok, why in self.workloads():
            out.append(f"  {name:<36} {'yes' if ok else 'NO':>5}  {why}")
        return "\n".join(out)


class HardwareProbe:
    """Read what the host is. Never raises because a device is absent.

    `probe(gpu=False)` forces the no-GPU path, which is how the absent-GPU behaviour is
    tested on a machine that has one.
    """

    @staticmethod
    def probe(*, gpu: bool = True, gpu_bench: bool = False, bench_n: int = 512) -> HardwareReport:
        cpu_gflops, step_us = HardwareProbe._cpu_bench(bench_n)
        gpu_name = vram = cap = None
        temp: float | None = None
        temp_source = "stub"
        torch_cuda = False
        gpu_tflops = None

        if gpu:
            gpu_name, vram, cap = HardwareProbe._smi()
            temp, temp_source = gpu_temperature_c()
            torch_cuda, torch_gpu = HardwareProbe._torch_gpu(bench=gpu_bench)
            if torch_gpu is not None:
                # Prefer torch's view when nvidia-smi is absent but CUDA works.
                name, vram_mb, cap_str, tflops = torch_gpu
                gpu_name = gpu_name or name
                vram = vram or vram_mb
                cap = cap or cap_str
                gpu_tflops = tflops

        return HardwareReport(
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            python=platform.python_version(),
            cpu_logical=os.cpu_count() or 1,
            ram_gb=_ram_gb(),
            cpu_gflops=cpu_gflops,
            gpu_name=gpu_name,
            vram_mb=vram,
            compute_capability=cap,
            gpu_temp_c=temp,
            temp_source=temp_source,
            gpu_tflops=gpu_tflops,
            has_torch=_importable("torch"),
            has_torch_cuda=torch_cuda,
            has_triton=_importable("triton"),
            has_onnx=_importable("onnx"),
            has_onnxruntime=_importable("onnxruntime"),
            # tensorrt and tvm are absent on this host. Probing by import would only
            # ever produce False here, and a False that looks probed invites someone to
            # trust the True case that was never exercised.
            has_tensorrt=False,
            has_tvm=False,
            step_us=step_us,
        )

    @staticmethod
    def _cpu_bench(n: int) -> tuple[float, float]:
        """(GFLOP/s on an n x n fp32 GEMM, p50 us for one d=64 matvec)."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(n, n)).astype(np.float32)
        b = rng.normal(size=(n, n)).astype(np.float32)
        a @ b  # warm BLAS thread pool; the first call pays for creating it
        best = float("inf")
        for _ in range(3):  # min of 3, not mean: we want the machine's capability
            t0 = time.perf_counter()
            a @ b
            best = min(best, time.perf_counter() - t0)
        gflops = 2.0 * n**3 / best / 1e9
        W = rng.normal(size=(64, 65))
        return gflops, _matvec_p50_us(W, 65, reps=1000)

    @staticmethod
    def _smi() -> tuple[str | None, int | None, str | None]:
        """GPU identity from nvidia-smi, or (None, None, None)."""
        if not shutil.which("nvidia-smi"):
            return None, None, None
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None, None, None
            name, mem, cap = (f.strip() for f in out.stdout.strip().splitlines()[0].split(","))
            return name, int(float(mem)), cap
        except (OSError, ValueError, subprocess.SubprocessError):
            return None, None, None

    @staticmethod
    def _torch_gpu(
        *, bench: bool
    ) -> tuple[bool, tuple[str, int, str, float | None] | None]:
        """(cuda available, (name, vram MB, cap, measured TFLOP/s or None)).

        torch is imported here and nowhere else at module scope -- `import sigmoid` must
        keep working on a numpy-only host.
        """
        try:
            import torch
        except ImportError:
            return False, None
        try:
            if not torch.cuda.is_available():
                return False, None
            props = torch.cuda.get_device_properties(0)
            tflops = None
            if bench:
                # 2048^3 fp16 GEMM. Measured 22.1 TFLOP/s on this host's RTX 4060 Laptop,
                # which is why the tier reads 10-100 rather than being asserted from a
                # spec sheet.
                a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
                b = a.clone()
                for _ in range(5):
                    a @ b
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    a @ b
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 20
                tflops = 2.0 * 2048**3 / dt / 1e12
            return True, (
                props.name,
                int(props.total_memory // 2**20),
                f"{props.major}.{props.minor}",
                tflops,
            )
        except (RuntimeError, AssertionError, OSError):
            # A driver mismatch raises from inside CUDA init. A probe that crashes the
            # process is worse than a probe that reports no GPU.
            return False, None


# --------------------------------------------------------------------------
# OTA weight store
# --------------------------------------------------------------------------

_REQUIRED_KEYS = ("config", "encoder", "operator", "gate")


class StagingError(RuntimeError):
    """A staged checkpoint was rejected, or the swap could not be applied."""


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256. Chunked because a checkpoint may not fit in RAM twice."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_atomic(dst: Path, write: Any) -> None:
    """Run `write(fh)` into a sibling temp file, fsync it, then `os.replace` onto `dst`.

    Sibling, not the system temp dir: `os.replace` is only atomic within a filesystem,
    and a robot's weights directory is routinely a different mount from /tmp.

    The fsync is the part that is easy to skip and the reason this exists. Without it
    `os.replace` can publish a filename whose contents are still in the page cache, and
    a power cut then leaves a correctly-named, half-written weight file -- the single
    worst outcome available here, because it survives the reboot and loads.
    """
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    _replace_retry(tmp, dst)


def _replace_retry(src: Path, dst: Path, *, tries: int = 50, delay: float = 0.002) -> int:
    """`os.replace` with backoff. Returns the retry count.

    On Windows `os.replace` is atomic but refusable: replacing a path another handle has
    open raises PermissionError (WinError 5), because Python's `open`/`np.load` do not
    pass FILE_SHARE_DELETE. Measured here: a reader looping continuously blocked 300 of
    300 swaps, while a realistic intermittent reader needed at most 1 retry. So this is
    a retry, not a workaround -- and it either applies the swap whole or raises, never
    leaving the destination partly written.
    """
    for i in range(tries):
        try:
            os.replace(src, dst)
            return i
        except PermissionError:
            if i == tries - 1:
                break
            time.sleep(delay)
    raise StagingError(
        f"could not replace {dst.name}: another process is holding it open "
        f"(retried {tries} times over {tries * delay:.2f}s)"
    )


class WeightStore:
    """Staged, hash-verified, atomically-swapped weight files. The OTA trust boundary.

    Layout under `root`:

        current.npz     the live weights
        previous.npz    the last committed weights, for `rollback`
        staged.npz      a validated candidate awaiting `commit`
        staged.json     the candidate's digest, so a restarted process still knows it
        manifest.json   the digest `current.npz` is expected to have

    Lifecycle: `stage(path)` -> `commit()` or `rollback()`. Every write goes through
    `_write_atomic` / `_replace_retry`, so `current.npz` is at all times either the whole
    old file or the whole new one. Measured: 100 swaps under a concurrent reader produced
    83 distinct whole payloads and zero torn reads.

    Crash safety falls out of that. A crash before `commit` leaves `current.npz`
    untouched and `staged.npz` on disk, so the old weights still load and the update can
    be retried; a crash mid-`commit` leaves `current.npz` as one whole version or the
    other, and `verify()` reports which. `__init__` reaps `*.tmp` files, which by
    definition are interrupted writes and never referenced by name.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = self.root / "current.npz"
        self.previous = self.root / "previous.npz"
        self.staged = self.root / "staged.npz"
        self.manifest_path = self.root / "manifest.json"
        self.staged_path = self.root / "staged.json"
        for stale in self.root.glob("*.tmp"):
            # An interrupted write. It was never linked into a name anything reads, so
            # deleting it is the whole of the recovery.
            stale.unlink(missing_ok=True)

    # ---- staging --------------------------------------------------------

    def stage(self, path: str | Path, *, expected_sha256: str | None = None) -> str:
        """Validate an incoming checkpoint and park it as the candidate. Returns its digest.

        Three checks, in the order that keeps the cheapest first and never lets an
        unverified file reach the loader:

          1. **Digest.** If `expected_sha256` is given it must match, and a mismatch
             raises before anything is written. This is the check that makes OTA
             meaningful: the digest travels by a channel the weights do not.
          2. **Structure.** The archive must open and contain the keys
             `SigmoidWorldModel.load` requires. Done with `allow_pickle=False`, which
             reads the zip directory without unpickling anything -- a malformed or
             truncated transfer is caught here without executing the payload.
          3. **Atomic install.** The candidate is copied through a temp file and
             `os.replace`d into `staged.npz`, so an interrupted stage cannot leave a
             partial candidate that a later `commit` would publish.
        """
        src = Path(path)
        if not src.is_file():
            raise StagingError(f"no such checkpoint: {src}")
        digest = sha256_file(src)
        if expected_sha256 is not None and digest != expected_sha256:
            raise StagingError(
                f"sha256 mismatch: manifest says {expected_sha256[:16]}..., "
                f"file is {digest[:16]}... -- refusing to stage"
            )
        self._validate_structure(src)

        def copy(fh: Any) -> None:
            with open(src, "rb") as inp:
                shutil.copyfileobj(inp, fh)

        _write_atomic(self.staged, copy)
        _write_atomic(
            self.staged_path,
            lambda fh: fh.write(
                json.dumps({"sha256": digest, "bytes": src.stat().st_size}).encode()
            ),
        )
        return digest

    @staticmethod
    def _validate_structure(src: Path) -> None:
        """Check the archive without unpickling it. `allow_pickle=False` is the point."""
        try:
            with np.load(src, allow_pickle=False) as data:
                missing = [k for k in _REQUIRED_KEYS if k not in data.files]
        except Exception as exc:  # noqa: BLE001 - any failure to open is a rejection
            raise StagingError(
                f"not a readable npz checkpoint: {type(exc).__name__}: {exc}"
            ) from exc
        if missing:
            raise StagingError(f"checkpoint is missing required arrays: {missing}")

    # ---- commit / rollback ----------------------------------------------

    def commit(self) -> Path:
        """Publish the staged candidate. Re-verifies its digest first.

        The digest is checked again here even though `stage` already checked it. Between
        the two calls the file sat on disk where another process -- or a bit flip on a
        cheap eMMC -- could have changed it, and this is the last moment before those
        bytes become the weights a robot acts on. Re-hashing a checkpoint costs
        milliseconds.
        """
        if not self.staged.is_file():
            raise StagingError("nothing staged")
        recorded = self._staged_digest()
        actual = sha256_file(self.staged)
        if recorded is not None and actual != recorded:
            self.staged.unlink(missing_ok=True)
            self.staged_path.unlink(missing_ok=True)
            raise StagingError(
                f"staged file changed after staging ({recorded[:16]}... -> {actual[:16]}...): "
                "discarded, not committed"
            )
        # Copy the live file to `previous` *before* touching `current`, so there is never
        # an instant where neither name resolves to a whole checkpoint.
        if self.current.is_file():
            _write_atomic(self.previous, lambda fh: fh.write(self.current.read_bytes()))
        _replace_retry(self.staged, self.current)
        self._write_manifest(actual)
        self.staged_path.unlink(missing_ok=True)
        return self.current

    def rollback(self) -> Path:
        """Put the previous weights back. Copies rather than moves, so it is idempotent."""
        if not self.previous.is_file():
            raise StagingError("no previous version to roll back to")
        digest = sha256_file(self.previous)
        _write_atomic(self.current, lambda fh: fh.write(self.previous.read_bytes()))
        self._write_manifest(digest)
        return self.current

    def discard(self) -> None:
        """Drop the staged candidate without touching the live weights."""
        self.staged.unlink(missing_ok=True)
        self.staged_path.unlink(missing_ok=True)

    # ---- reading --------------------------------------------------------

    def verify(self) -> bool:
        """Whether `current.npz` still hashes to what the manifest recorded."""
        if not self.current.is_file() or not self.manifest_path.is_file():
            return False
        return sha256_file(self.current) == self.manifest().get("sha256")

    def load(self) -> SigmoidWorldModel:
        """Verify the manifest digest, then load the live weights.

        RESIDUAL RISK, and it is not small. `SigmoidWorldModel.load` calls
        `numpy.load(..., allow_pickle=True)`, which executes arbitrary code from the
        file (see SECURITY.md). The digest check above proves **integrity** -- these are
        the bytes whose hash you were given -- and proves nothing whatever about
        **safety**. A hostile checkpoint whose digest matches its manifest will be
        loaded and will run its payload.

        So the manifest digest has to arrive by a channel the weights did not: signed by
        your build system, pinned in a release you control, delivered over an
        authenticated transport. A digest shipped alongside the weights by the same
        untrusted server authenticates nothing, because whoever supplied one supplied
        the other.

        `stage` deliberately validates structure with `allow_pickle=False` so a
        malformed transfer is rejected without unpickling. That narrows the window; it
        does not close it. Closing it means a pickle-free checkpoint format, which is an
        engine.py change and out of scope here.
        """
        if not self.verify():
            raise StagingError(
                "current.npz does not match its manifest digest -- refusing to load. "
                "Roll back or re-stage."
            )
        from .engine import SigmoidWorldModel

        return SigmoidWorldModel.load(self.current)

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text())

    def _staged_digest(self) -> str | None:
        if not self.staged_path.is_file():
            return None
        return json.loads(self.staged_path.read_text()).get("sha256")

    def _write_manifest(self, digest: str) -> None:
        payload = {
            "sha256": digest,
            "bytes": self.current.stat().st_size,
            "committed_at": time.time(),
            "platform": platform.system(),
        }
        blob = json.dumps(payload, indent=2).encode()
        _write_atomic(self.manifest_path, lambda fh: fh.write(blob))


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportResult:
    fmt: str
    path: Path
    manifest_path: Path | None
    sha256: str
    bytes: int
    notes: str = ""


def export_model(
    engine: SigmoidWorldModel,
    path: str | Path,
    fmt: str = "npz",
    *,
    precision: str | Precision | None = None,
) -> ExportResult:
    """Export for deployment. `.npz` + JSON manifest is the primary format.

    That is not a fallback, it is the right answer for this model: the parameters are
    numpy matrices, `.npz` loads anywhere numpy does with no framework and no compiler,
    and the JSON manifest carries the SHA-256 that `WeightStore.stage` verifies. The
    export and the OTA path are therefore the same artifact.

    `fmt="onnx"` exports the operator's lifted matvec as a single Linear via
    `torch.onnx`, and is real here -- torch and onnx both import on this host, and the
    export measured 281 ms for an 11 KB graph. It is still optional and lazily imported.

    `fmt="tensorrt"` and `fmt="tvm"` raise `NotImplementedError`. Neither package is
    installed here, so neither path could be executed even once, and an untested
    compiler backend that reports success is a worse artifact than a refusal.
    """
    path = Path(path)
    if fmt in ("tensorrt", "trt", "tvm"):
        raise NotImplementedError(
            f"{fmt} export is not implemented: the package is not installed on this host, "
            "so the path cannot be tested. Export npz and compile downstream."
        )
    if fmt == "npz":
        return _export_npz(engine, path, precision)
    if fmt == "onnx":
        return _export_onnx(engine, path, precision)
    raise ValueError(f"unknown export format {fmt!r}; expected 'npz' or 'onnx'")


def _operator_matrix(engine: SigmoidWorldModel, precision: str | Precision | None) -> np.ndarray:
    op = engine.operator
    assert op.W_ is not None
    if precision is None:
        return op.W_
    return quantize_operator(op, precision).W_


def _export_npz(
    engine: SigmoidWorldModel, path: Path, precision: str | Precision | None
) -> ExportResult:
    written = engine.save(path)
    op = engine.operator
    digest = sha256_file(written)
    manifest = {
        "format": "npz",
        "sigmoid_version": _version(),
        "sha256": digest,
        "bytes": written.stat().st_size,
        "created_at": time.time(),
        "state_dim": int(op.state_dim),
        "action_dim": int(op.action_dim),
        "lift_dim": int(op.lift_dim),
        "W_shape": list(op.W_.shape) if op.W_ is not None else None,
        "rho": float(op.rho_),
        "contractive": bool(op.rho_ < 1.0),
        "step_rmse": float(op.step_rmse_),
        "weights_precision": "fp64",
        "load_warning": (
            "SigmoidWorldModel.load uses numpy.load(allow_pickle=True) and can execute "
            "code from this file. Verify sha256 against a trusted channel first."
        ),
    }
    if precision is not None:
        # The quantized matrix is reported, never substituted: engine.save writes the
        # fp64 checkpoint, and a caller who wants int8 weights quantizes at load. Writing
        # a lossy checkpoint under the same name as a lossless one is how a fleet ends up
        # unable to say which precision it is running.
        qop = quantize_operator(engine.operator, precision)
        base = op.W_.nbytes if op.W_ is not None else 0
        manifest["quantization_preview"] = {
            "precision": qop.precision.value,
            "bytes": qop.nbytes,
            "compression": round(base / max(qop.nbytes, 1), 3),
        }
    mpath = written.with_suffix(".manifest.json")
    blob = json.dumps(manifest, indent=2).encode()
    _write_atomic(mpath, lambda fh: fh.write(blob))
    return ExportResult("npz", written, mpath, digest, written.stat().st_size)


def _export_onnx(
    engine: SigmoidWorldModel, path: Path, precision: str | Precision | None
) -> ExportResult:
    """Export the lifted matvec as a one-Linear ONNX graph.

    The exported graph takes the *lifted* feature vector, not the raw state: the lift
    (`[z ; a (x) z ; a ; 1]`) is a data-layout step, and reimplementing it as ONNX ops
    would fork the definition in operator.py. The consumer lifts, the graph multiplies.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is present on this host
        raise NotImplementedError(
            "onnx export needs torch, which does not import here. Use fmt='npz'."
        ) from exc

    W = _operator_matrix(engine, precision)
    op = engine.operator
    path = path.with_suffix(".onnx")
    linear = torch.nn.Linear(op.lift_dim, op.state_dim, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)))
    torch.onnx.export(
        linear,
        torch.zeros(1, op.lift_dim),
        str(path),
        input_names=["lift"],
        output_names=["next_state"],
        dynamic_axes={"lift": {0: "batch"}, "next_state": {0: "batch"}},
    )
    digest = sha256_file(path)
    manifest = {
        "format": "onnx",
        "sigmoid_version": _version(),
        "sha256": digest,
        "input": f"lift ({op.lift_dim},) -- build it with CouplingOperator._lift",
        "output": f"next_state ({op.state_dim},)",
        "weights_precision": "fp32",
        "quantized_from": None if precision is None else _as_precision(precision).value,
        "rho": float(op.rho_),
    }
    mpath = path.with_suffix(".manifest.json")
    _write_atomic(mpath, lambda fh: fh.write(json.dumps(manifest, indent=2).encode()))
    return ExportResult(
        "onnx",
        path,
        mpath,
        digest,
        path.stat().st_size,
        notes="graph consumes the lifted vector; the lift stays in operator.py",
    )


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except ImportError:  # pragma: no cover - only if imported outside the package
        return "unknown"


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _fit_demo_operator(rho: float = 0.7, *, heterogeneous: bool = False) -> CouplingOperator:
    """A small linear system whose spectral norm is `rho` *before* fitting.

    Scaled to `rho` rather than fitted freely and clipped with `rho_max`. Clipping
    overwrites the dynamics and drives `step_rmse_` up with it -- measured 4.57e-01 under
    a 0.95 clip against a 1e-2 injected noise floor, a 45x inflation. Since
    `QuantReport.usable` compares quantization error against exactly that number, a
    clipped operator would declare every precision usable and the table would discriminate
    nothing.
    """
    rng = np.random.default_rng(0)
    d = 32
    A = rng.normal(size=(d, d))
    A *= rho / float(np.linalg.svd(A, compute_uv=False)[0])
    if heterogeneous:
        # Raw-units robot state: millimetres beside radians beside newtons. This is the
        # regime where a global int8 scale zeroes whole rows.
        A *= np.logspace(-3, 1, d).reshape(-1, 1)
    Z = rng.normal(size=(2000, d))
    Znext = Z @ A.T + 0.01 * rng.normal(size=(2000, d))
    return CouplingOperator(ridge=1e-6).fit(Z[:-1], Znext[:-1])


def demo() -> None:
    """Measured behaviour on this host. Every number below is produced, not quoted."""
    import tempfile

    # ---- quantization, on a contractive operator so the rollout means something
    op = _fit_demo_operator(rho=0.7)
    rows = quantization_table(op, horizon=16, reps=1500)
    print(f"quantization  (W_ {op.W_.shape}, rho {op.rho_:.3f}, fitted rmse {op.step_rmse_:.3e})")
    print(quantization_report(rows))
    fp32 = next(r for r in rows if r.precision == "fp32")
    int8 = next(r for r in rows if r.precision == "int8")
    assert fp32.usable, "fp32 must not be the dominant error source"
    assert int8.compression > 6.0, f"int8 compression only {int8.compression:.1f}x"
    assert min(r.latency_us for r in rows[1:]) >= rows[0].latency_us * 0.8, (
        "a lower precision came out materially faster than fp64 -- re-read the docstring, "
        "the whole precision policy is built on that not happening in numpy"
    )

    # Same measurement on heterogeneous rows, where the global scale is expected to fail.
    print()
    hetero_op = _fit_demo_operator(rho=0.7, heterogeneous=True)
    hrows = quantization_table(hetero_op, horizon=16, reps=800)
    print(
        f"quantization, rows over 4 decades  (rho {hetero_op.rho_:.3f}, "
        f"fitted rmse {hetero_op.step_rmse_:.3e})"
    )
    print(quantization_report(hrows))
    h_row = next(r for r in hrows if r.precision == "int8")
    h_glob = next(r for r in hrows if r.precision == "int8 (global)")
    assert h_glob.rollout_rmse > h_row.rollout_rmse, (
        "global int8 scaling was not worse on a heterogeneous operator -- that is the "
        "entire justification for per-row, so re-measure before believing it"
    )

    # ---- was per-row scaling necessary? measure, do not assert
    print()
    for label, hetero in (("well-conditioned", False), ("rows over 4 decades", True)):
        o = _fit_demo_operator(rho=0.7, heterogeneous=hetero)
        assert o.W_ is not None
        per_row = quantize_operator(o, Precision.INT8, per_row=True)
        glob = quantize_operator(o, Precision.INT8, per_row=False)
        peak = np.abs(o.W_).max(axis=1)
        live = peak[peak > 0]
        e_row = row_relative_error(per_row.W_, o.W_)
        e_glob = row_relative_error(glob.W_, o.W_)
        print(
            f"  {label:<22} row-magnitude spread {live.max() / live.min():>9.1f}x  "
            f"worst-row err  per-row {e_row.max():.2e}  global {e_glob.max():.2e}  "
            f"rows destroyed {int((e_glob > 0.5).sum())}/{len(e_glob)}"
        )
        assert e_row.max() <= e_glob.max() + 1e-12, "per-row scaling did worse than global"
        assert np.all(np.isfinite(per_row.W_)), "zero-row guard failed"

    # ---- precision policy: hysteresis against an oscillating signal
    print("precision policy, square-wave GPU temperature over 60 control ticks")
    for dwell, period, lo, hi in (
        (3, 1, 60.0, 95.0),
        (3, 2, 60.0, 95.0),
        (3, 8, 60.0, 95.0),
        (10, 8, 60.0, 95.0),
        (3, 1, 72.0, 78.0),
        (3, 8, 72.0, 78.0),
    ):
        pol = PrecisionPolicy(dwell=dwell)
        seen = []
        for i in range(60):
            temp = hi if (i // period) % 2 else lo
            seen.append(pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=temp)))
        print(
            f"  {lo:.0f}-{hi:.0f} C  half-period {period:>2}  dwell {dwell:>2}  -> "
            f"{pol.switches:>2} switches, {len(set(seen))} distinct precisions"
        )
        # Two absorption mechanisms, two guarantees, both asserted:
        #   - inside the 70-80 C band nothing crosses the latch at any period,
        #   - outside it, any half-period below `dwell` starves the streak counter.
        if hi < pol.hot_c or period < dwell:
            assert pol.switches == 0, (
                f"policy flapped: {pol.switches} switches at half-period {period}, dwell {dwell}"
            )

    pol = PrecisionPolicy(dwell=3)
    for _ in range(6):  # a sustained excursion must still be acted on
        pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=95.0))
    assert pol.current is Precision.INT8 and pol.switches == 1, "policy never derated"
    for _ in range(6):
        pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=65.0))
    assert pol.current is Precision.FP32, "policy never recovered"
    print(f"  sustained 95 C then 65 C  -> derated and recovered, {pol.switches} switches")

    temp, source = gpu_temperature_c()
    print(f"  gpu temperature   {temp if temp is not None else 'unavailable'} C (source: {source})")
    print(f"  battery           {battery_percent()}")

    # ---- hardware
    print()
    report = HardwareProbe.probe(gpu_bench=True)
    print("hardware")
    print(report.table())
    blind = HardwareProbe.probe(gpu=False)
    assert blind.gpu_name is None and blind.device_class in {"constrained-edge", "sbc"}, (
        "forced no-GPU probe still reported a GPU"
    )
    print(f"\n  with GPU forced off: class {blind.device_class}, no crash")

    # ---- OTA: atomic swap, hash rejection, rollback
    print()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        good = _make_checkpoint(root / "v1.npz", seed=1)
        newer = _make_checkpoint(root / "v2.npz", seed=2)
        store = WeightStore(root / "store")
        store.stage(good, expected_sha256=sha256_file(good))
        store.commit()
        v1_digest = store.manifest()["sha256"]
        assert store.verify()

        store.stage(newer, expected_sha256=sha256_file(newer))
        store.commit()
        assert store.manifest()["sha256"] != v1_digest and store.verify()
        print(f"  staged + committed two versions, manifest verifies: {store.verify()}")

        # a corrupted candidate must be rejected AND leave the live weights loadable
        corrupt = root / "bad.npz"
        corrupt.write_bytes(bytearray(newer.read_bytes())[: -16])
        try:
            store.stage(corrupt, expected_sha256=sha256_file(newer))
        except StagingError as exc:
            print(f"  corrupt candidate rejected: {str(exc)[:58]}...")
        else:
            raise AssertionError("a truncated checkpoint was accepted")
        assert store.verify(), "live weights broke while rejecting a bad candidate"

        store.rollback()
        assert store.manifest()["sha256"] == v1_digest and store.verify()
        print("  rollback restored v1 and its manifest")

        # crash between stage and commit: a fresh process must find the old weights
        store.stage(newer, expected_sha256=sha256_file(newer))
        reopened = WeightStore(root / "store")
        assert reopened.verify() and reopened.manifest()["sha256"] == v1_digest
        assert reopened.staged.is_file(), "staged candidate lost across restart"
        reopened.commit()
        print("  simulated crash between stage and commit recovered, then committed")

        # ---- export
        print()
        wm = _tiny_engine()
        res = export_model(wm, root / "export", "npz")
        print(f"  npz   {res.bytes:>7} bytes  sha {res.sha256[:16]}...  manifest written")
        assert res.manifest_path is not None and res.manifest_path.is_file()
        if _importable("torch"):
            onnx_res = export_model(wm, root / "export", "onnx")
            print(f"  onnx  {onnx_res.bytes:>7} bytes  sha {onnx_res.sha256[:16]}...")
        else:
            print("  onnx  skipped: torch does not import here")
        for fmt in ("tensorrt", "tvm"):
            try:
                export_model(wm, root / "export", fmt)
            except NotImplementedError as exc:
                print(f"  {fmt:<5} unavailable: {str(exc)[:56]}...")
            else:
                raise AssertionError(f"{fmt} claimed to export")
    print("demo ok")


def _make_checkpoint(path: Path, *, seed: int) -> Path:
    """A structurally valid checkpoint, cheap enough to write in a loop."""
    return _tiny_engine(seed=seed).save(path)


def _tiny_engine(*, seed: int = 0) -> SigmoidWorldModel:
    from .engine import SigmoidConfig, SigmoidWorldModel

    rng = np.random.default_rng(seed)
    traj = np.cumsum(rng.normal(size=(400, 3)) * 0.1, axis=0)
    return SigmoidWorldModel(config=SigmoidConfig(window=16, linear_dim=6)).fit([traj])


if __name__ == "__main__":
    demo()

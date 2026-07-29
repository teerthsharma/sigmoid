"""The falsification harness.

A topological method is worth nothing until it beats the boring thing at equal
budget. This module exists so the claim "topology helps" is a measurement and
not a preference.

Arms, all at *matched state dimension*:

    sigmoid       psi (+) u, the full hybrid state
    linear_only   PCA channel widened to the same total dimension. This is the
                  ablation that matters: same parameter count, same operator,
                  same fit, no topology.
    carry         z_{t+1} = z_t. Beats a surprising number of learned models.
    mean          predict the calibration mean. The floor.

Gates, adapted from the topology-speedup promotion rules:

    accuracy      sigmoid rollout NRMSE must beat linear_only at equal dim
    stability     rho < 1, so rollouts carry a contraction certificate
    honesty       the measured error must sit under the certificate bound
    speed         an imagined step must be materially cheaper than a real one

A run that fails a gate is reported as failed. Silent truncation of a losing
arm is how this kind of work goes wrong.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .engine import SigmoidConfig, SigmoidWorldModel

__all__ = ["ArmResult", "BenchReport", "rollout_error", "compare", "gate_report"]


@dataclass(frozen=True)
class ArmResult:
    name: str
    state_dim: int
    nrmse: dict[int, float]
    """Normalized hidden-state RMSE at each rollout horizon."""

    cosine: dict[int, float]
    rho: float
    contractive: bool
    step_rmse: float
    fit_seconds: float
    encode_ms: float
    """Milliseconds to encode one window (the runtime topology cost)."""

    imagine_us: float
    """Microseconds for one imagined step."""


@dataclass(frozen=True)
class BenchReport:
    arms: tuple[ArmResult, ...]
    horizons: tuple[int, ...]
    real_forward_ms: float | None
    gates: dict[str, bool]
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(self.gates.values())

    def table(self) -> str:
        hs = self.horizons
        width = max(20, max(len(a.name) for a in self.arms) + 2)
        head = f"{'arm':<{width}}{'dim':>5}  " + "  ".join(f"k={h:<8}" for h in hs) + f"{'rho':>8}"
        lines = [head, "-" * len(head)]
        for arm in self.arms:
            cells = "  ".join(f"{arm.nrmse.get(h, float('nan')):<10.4f}" for h in hs)
            lines.append(f"{arm.name:<{width}}{arm.state_dim:>5}  {cells}{arm.rho:>8.4f}")
        lines.append("")
        for name, ok in self.gates.items():
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _split(trajectories: Sequence[np.ndarray], holdout: float) -> tuple[list, list]:
    trajs = [np.asarray(t, dtype=np.float64) for t in trajectories]
    if len(trajs) >= 4:
        cut = max(1, int(len(trajs) * (1.0 - holdout)))
        return trajs[:cut], trajs[cut:]
    # single long trajectory: split along time
    cut = int(trajs[0].shape[0] * (1.0 - holdout))
    return [t[:cut] for t in trajs], [t[cut:] for t in trajs]


def rollout_error(
    wm: SigmoidWorldModel,
    trajectories: Sequence[np.ndarray],
    horizons: Sequence[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Normalized RMSE and cosine of predicted vs true hidden states.

    NRMSE is divided by the RMS of the true future hidden states, so a value of
    1.0 means "no better than predicting zero" and values are comparable across
    models with different activation scales.
    """
    window = wm.config.window
    max_h = max(horizons)
    sq_err = {h: [] for h in horizons}
    sq_true = {h: [] for h in horizons}
    cosines = {h: [] for h in horizons}

    for traj in trajectories:
        if traj.shape[0] < window + max_h:
            continue
        for start in range(window, traj.shape[0] - max_h + 1, max(1, max_h)):
            block = traj[start - window : start]
            pred = wm.predict_hidden(block, max_h)
            truth = traj[start : start + max_h]
            for h in horizons:
                p, t = pred[h - 1], truth[h - 1]
                sq_err[h].append(float(np.mean((p - t) ** 2)))
                sq_true[h].append(float(np.mean(t**2)))
                denom = np.linalg.norm(p) * np.linalg.norm(t)
                cosines[h].append(0.0 if denom < 1e-12 else float(p @ t / denom))

    nrmse, cos = {}, {}
    for h in horizons:
        if not sq_err[h]:
            nrmse[h], cos[h] = float("nan"), float("nan")
            continue
        nrmse[h] = float(
            np.sqrt(np.mean(sq_err[h])) / (np.sqrt(np.mean(sq_true[h])) + 1e-12)
        )
        cos[h] = float(np.mean(cosines[h]))
    return nrmse, cos


def latent_rollout_error(
    wm: SigmoidWorldModel,
    trajectories: Sequence[np.ndarray],
    horizon: int,
) -> float:
    """RMSE between imagined and truly-encoded states after `horizon` steps.

    Measured in the operator's own units, so it is directly comparable to the
    certificate bound -- which is the point. A bound that does not bound is a
    bug, not a caveat.
    """
    window = wm.config.window
    errs: list[float] = []
    for traj in trajectories:
        if traj.shape[0] < window + horizon:
            continue
        for start in range(window, traj.shape[0] - horizon + 1, max(1, horizon)):
            imagined = wm.imagine(
                traj[start - window : start], horizon, stop_on_gate=False
            ).states[-1]
            truth = wm.observe(traj[start + horizon - window : start + horizon])
            errs.append(float(np.mean((imagined - truth) ** 2)))
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


class _Carry(SigmoidWorldModel):
    """Baseline: the future equals the present."""

    def predict_hidden(self, window, steps=1, *, actions=None):  # type: ignore[override]
        last = np.asarray(window, dtype=np.float64)[-1]
        return np.repeat(last[None, :], steps, axis=0)


class _Mean(SigmoidWorldModel):
    """Baseline: the future equals the calibration mean."""

    def predict_hidden(self, window, steps=1, *, actions=None):  # type: ignore[override]
        # mean_ lives in standardized space; the comparison is in raw space.
        mean_hidden = self.encoder.denormalize(self.encoder.mean_)
        return np.repeat(mean_hidden[None, :], steps, axis=0)


def _build(config: SigmoidConfig, train, cls=SigmoidWorldModel) -> tuple[SigmoidWorldModel, float]:
    t0 = time.perf_counter()
    wm = cls(config=config).fit(train)
    return wm, time.perf_counter() - t0


def _timings(wm: SigmoidWorldModel, sample: np.ndarray) -> tuple[float, float]:
    block = sample[: wm.config.window]
    for _ in range(3):
        wm.observe(block)
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        z = wm.observe(block)
    encode_ms = (time.perf_counter() - t0) / reps * 1e3

    t0 = time.perf_counter()
    reps = 2000
    for _ in range(reps):
        wm.operator.step(z)
    imagine_us = (time.perf_counter() - t0) / reps * 1e6
    return encode_ms, imagine_us


def compare(
    trajectories: Sequence[np.ndarray],
    *,
    config: SigmoidConfig | None = None,
    horizons: Sequence[int] = (1, 4, 16),
    holdout: float = 0.3,
    real_forward_fn: Callable[[], None] | None = None,
) -> BenchReport:
    """Run every arm at matched dimension and return a gated report."""
    config = config or SigmoidConfig()
    horizons = tuple(int(h) for h in horizons)
    train, test = _split(trajectories, holdout)
    notes: list[str] = []

    full, full_secs = _build(config, train)
    matched_dim = full.state_dim

    linear_cfg = SigmoidConfig(
        **{**vars(config), "use_topology": False, "linear_dim": matched_dim}
    )
    linear, linear_secs = _build(linear_cfg, train)
    if linear.state_dim != matched_dim:
        notes.append(
            f"linear_only dim {linear.state_dim} != sigmoid dim {matched_dim}; "
            "comparison is not budget-matched"
        )

    # The topology-only ablation: same linear channel, psi deleted.
    #
    # `linear_only` above matches the TOTAL state dimension, which sounds like
    # the fair comparison and is not: it raises the PCA rank from linear_dim to
    # state_dim, and that shrinks the residual `decode_with_residual` carries
    # forward from the anchor. In carry-friendly worlds a larger residual helps,
    # so the dimension-matched arm is silently handicapped and topology looks
    # better than it is. Measured on a granular contact world: sigmoid and
    # u-matched linear agreed to four decimals (0.0072 / 0.0279 / 0.0905) while
    # the dimension-matched arm read 0.0179 / 0.0690 / 0.2349 -- the entire
    # apparent 60% "win" was the rank change. On Lorenz the same confound
    # inflated a real ~15% one-step gain into a reported ~30%.
    #
    # Holding u fixed and removing only psi co-varies nothing but topology, so
    # this is the arm the accuracy gate judges. `linear_only` is kept because
    # the budget question ("is psi the best use of those dimensions?") is also
    # worth answering -- it is just a different question.
    u_matched_cfg = SigmoidConfig(**{**vars(config), "use_topology": False})
    u_matched, u_matched_secs = _build(u_matched_cfg, train)

    arms: list[ArmResult] = []
    for name, wm, secs in (
        ("sigmoid", full, full_secs),
        ("no_topology_same_u", u_matched, u_matched_secs),
        ("linear_only", linear, linear_secs),
        ("carry", _Carry(config=config).fit(train), 0.0),
        ("mean", _Mean(config=config).fit(train), 0.0),
    ):
        nrmse, cos = rollout_error(wm, test, horizons)
        encode_ms, imagine_us = _timings(wm, test[0]) if name == "sigmoid" else (0.0, 0.0)
        arms.append(
            ArmResult(
                name=name,
                state_dim=wm.state_dim,
                nrmse=nrmse,
                cosine=cos,
                rho=wm.operator.rho_,
                contractive=wm.operator.rho_ < 1.0,
                step_rmse=wm.operator.step_rmse_,
                fit_seconds=secs,
                encode_ms=encode_ms,
                imagine_us=imagine_us,
            )
        )

    real_ms = None
    if real_forward_fn is not None:
        real_forward_fn()
        t0 = time.perf_counter()
        for _ in range(5):
            real_forward_fn()
        real_ms = (time.perf_counter() - t0) / 5 * 1e3

    long_h = max(horizons)

    # Per-horizon topology delta. The gate is judged at the longest horizon,
    # but topology can help early and hurt late -- measured on Lorenz, psi wins
    # at k=1 and k=4 and loses at k=16. A single-horizon verdict hides that,
    # and the sign flip is the interesting part.
    u_arm = next(a for a in arms if a.name == "no_topology_same_u")
    sig_arm = next(a for a in arms if a.name == "sigmoid")
    deltas = " ".join(
        f"k={h}:{sig_arm.nrmse[h] - u_arm.nrmse[h]:+.4f}" for h in horizons
    )
    notes.append(f"topology delta vs same-u ablation (negative = helps): {deltas}")

    measured = latent_rollout_error(full, test, long_h)
    bound = full.certificate(long_h).bound
    notes.append(
        f"rho={full.operator.rho_:.4f} "
        f"({'contractive' if full.operator.rho_ < 1 else 'expansive'}); "
        f"latent error at k={long_h} is {measured:.4f} vs certificate bound {bound:.4f}"
    )

    return BenchReport(
        arms=tuple(arms),
        horizons=horizons,
        real_forward_ms=real_ms,
        gates=gate_report(arms, horizons, real_ms, measured, bound, long_h),
        notes=tuple(notes),
    )


def gate_report(
    arms: Sequence[ArmResult],
    horizons: Sequence[int],
    real_forward_ms: float | None,
    measured_latent_error: float,
    certificate_bound: float,
    long_h: int,
) -> dict[str, bool]:
    """Promotion gates. Note what is *not* gated: rho < 1.

    Contraction is reported, never required. Demanding it would reward clipping
    the operator into a contraction, which improves the certificate and
    degrades the model -- exactly the wrong incentive. What is gated instead is
    that the certificate *holds*: whatever rho the data gives, the measured
    rollout error must land inside the bound derived from it.
    """
    by_name = {a.name: a for a in arms}
    sig, lin = by_name["sigmoid"], by_name["linear_only"]
    same_u = by_name["no_topology_same_u"]
    carry, mean = by_name["carry"], by_name["mean"]

    # The certificate bound only certifies anything if it is tighter than the
    # signal itself. On an expansive operator rho^k explodes and the bound
    # becomes astronomically large, at which point "the error is inside the
    # bound" is true and meaningless. Gate on informativeness, not just
    # containment, otherwise a worse operator scores a free pass.
    bound_is_informative = bool(np.isfinite(certificate_bound) and certificate_bound < 1.0)

    # A strict `<` passes on a delta of -0.0000, which is what a block-diagonal
    # operator produces: psi structurally cannot reach u, so the two arms are
    # the *same computation* and differ only in float noise. Reporting that as
    # "psi is doing work" is exactly backwards, so require a real margin.
    margin = 1e-4

    gates = {
        # THE topology gate: same linear channel, psi deleted. Nothing else
        # co-varies, so a win here is attributable to topology and only here.
        "topology: sigmoid beats the same-u ablation (psi is doing work)": bool(
            sig.nrmse[long_h] < same_u.nrmse[long_h] - margin
        ),
        "budget: sigmoid beats linear_only at matched total dim": bool(
            sig.nrmse[long_h] < lin.nrmse[long_h]
        ),
        "accuracy: sigmoid beats carry-forward": bool(
            sig.nrmse[long_h] < carry.nrmse[long_h]
        ),
        # Without this gate a model that has learned nothing but the dataset
        # mean passes everything else. It is the cheapest way to be fooled.
        "accuracy: sigmoid beats the predict-the-mean floor": bool(
            all(sig.nrmse[h] < mean.nrmse[h] for h in horizons)
        ),
        "certificate: bound holds and is informative (< 1.0)": bool(
            bound_is_informative
            and np.isfinite(measured_latent_error)
            and measured_latent_error <= certificate_bound
        ),
        "speed: imagined step under 200us": bool(sig.imagine_us < 200.0),
    }
    if real_forward_ms is not None:
        gates["speed: imagination >= 10x cheaper than a real forward"] = bool(
            sig.imagine_us / 1e3 * 10.0 < real_forward_ms
        )
    return gates

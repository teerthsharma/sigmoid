"""Sigmoid: the world-model inference engine.

    normal model  --capture-->  hidden trajectory
                  --Sigma---->  topological world states
                  --T-------->  learned contractive dynamics
                  --gate----->  when to stop imagining and look again

A "normal model" answers *what is the next output*. A world model answers *what
happens next if things keep going*, cheaply, and knows when it stopped being
right. Sigmoid supplies the missing three: a state, a dynamics on that state,
and a self-consistency test on the dynamics -- without touching the wrapped
model's weights.

The engine is useful because imagination is cheap. One imagined step is a
single (state_dim x lift_dim) matvec; one real step is a full model forward.
For distilgpt2 that is roughly four orders of magnitude of arithmetic. The gate
decides how far that trade can be pushed before it stops being honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .adapters import capture_hidden
from .operator import CouplingOperator, RolloutCertificate
from .sheaf import GateReading, SheafGate
from .state import TopoEncoder, TopoEncoderConfig

__all__ = ["SigmoidConfig", "Rollout", "SigmoidWorldModel", "wrap"]


@dataclass
class SigmoidConfig:
    window: int = 32
    hilbert_degree: int = 24
    n_radii: int = 8
    n_abs_radii: int = 8
    abs_radii: tuple[float, ...] | None = None
    linear_dim: int = 32
    use_h1: bool = False
    use_topology: bool = True

    entity_dim: int = 0
    """Set to the per-entity width for set-structured states (particles,
    bodies, agents). Leave 0 for sequence states. See TopoEncoderConfig."""

    ridge: float = 1e-3

    rho_max: float | None = None
    """None fits the dynamics as measured. Set < 1 to force a contraction and
    buy an infinite-horizon certificate, knowing it costs accuracy on any
    expansive system. See CouplingOperator.rho_max."""

    action_dim: int = 0

    block_diagonal: bool | str = "auto"
    """Whether psi and u evolve independently. True, False, or "auto".

    Neither fixed choice is defensible, because the right answer is a property
    of the wrapped model, not of this library. Measured on the same code:

        Lorenz (low-dim, smooth, deterministic)
            coupled 0.052 vs block-diagonal 0.073 at k=1 -- coupling helps a lot
        distilgpt2 residual stream (768-dim, high entropy)
            coupled 0.145 vs block-diagonal 0.168 R^2 -- coupling actively hurts

    So "auto" fits both and keeps whichever wins on a held-out slice, judged on
    the channel that actually decodes. Hard-coding either value would have
    meant tuning the library to whichever system was benchmarked last."""

    gate_quantile: float = 0.98

    def encoder_config(self) -> TopoEncoderConfig:
        return TopoEncoderConfig(
            window=self.window,
            hilbert_degree=self.hilbert_degree,
            n_radii=self.n_radii,
            n_abs_radii=self.n_abs_radii,
            abs_radii=self.abs_radii,
            linear_dim=self.linear_dim,
            use_h1=self.use_h1,
            use_topology=self.use_topology,
            entity_dim=self.entity_dim,
        )


@dataclass(frozen=True)
class Rollout:
    """The result of imagining forward."""

    states: np.ndarray
    """(k, state_dim) imagined world states."""

    hiddens: np.ndarray
    """(k, D) predicted hidden states, decoded from the linear channel."""

    readings: tuple[GateReading, ...]
    grounded_at: int | None
    """Step index where the gate fired, or None if the whole horizon held."""

    certificate: RolloutCertificate

    @property
    def trusted_steps(self) -> int:
        return len(self.states) if self.grounded_at is None else self.grounded_at

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class SigmoidWorldModel:
    """Wraps a hidden-state producer into a world model."""

    config: SigmoidConfig = field(default_factory=SigmoidConfig)

    encoder: TopoEncoder | None = field(default=None, repr=False)
    operator: CouplingOperator | None = field(default=None, repr=False)
    gate: SheafGate | None = field(default=None, repr=False)
    hidden_dim: int = 0
    n_transitions: int = 0
    block_diagonal_: bool = False
    """Which block structure the fit actually selected."""

    fitted: bool = False

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: Any,
        inputs: Iterable[Any],
        *,
        layer: int | str = -1,
        config: SigmoidConfig | None = None,
        actions: Sequence[np.ndarray] | None = None,
    ) -> "SigmoidWorldModel":
        """Convert a normal model into a world model in one call.

        `inputs` is an iterable of whatever the model consumes -- prompts,
        batches, episodes. Each one is run once to harvest a trajectory.
        """
        trajectories = [capture_hidden(model, item, layer=layer) for item in inputs]
        return cls(config=config or SigmoidConfig()).fit(trajectories, actions=actions)

    # ---- calibration ------------------------------------------------------

    def fit(
        self,
        trajectories: Sequence[np.ndarray],
        actions: Sequence[np.ndarray] | None = None,
    ) -> "SigmoidWorldModel":
        """Calibrate encoder, operator and gate on hidden-state trajectories.

        Each trajectory is (T, D). Windows never straddle a trajectory
        boundary, so transitions are always within-episode.
        """
        trajs = [np.asarray(t, dtype=np.float64) for t in trajectories]
        trajs = [t for t in trajs if t.ndim == 2 and t.shape[0] >= self.config.window + 1]
        if not trajs:
            raise ValueError(
                f"no trajectory long enough; need >= {self.config.window + 1} steps"
            )
        dims = {t.shape[1] for t in trajs}
        if len(dims) != 1:
            raise ValueError(f"trajectories disagree on hidden dim: {sorted(dims)}")
        self.hidden_dim = dims.pop()

        self.encoder = TopoEncoder(config=self.config.encoder_config())
        self.encoder.fit(np.concatenate(trajs, axis=0))

        per_traj = [self.encoder.encode_trajectory(t) for t in trajs]
        current = np.concatenate([z[:-1] for z in per_traj], axis=0)
        following = np.concatenate([z[1:] for z in per_traj], axis=0)
        self.n_transitions = current.shape[0]

        act = None
        if actions is not None:
            act = _align_actions(actions, per_traj, self.config.window)

        self.operator, self.block_diagonal_ = self._fit_operator(current, following, act)

        self.gate = SheafGate(quantile=self.config.gate_quantile).fit(
            np.concatenate([current, following[-1:]], axis=0), self.encoder.topo_dim
        )

        self.fitted = True
        return self

    # ---- use --------------------------------------------------------------

    def observe(self, window: np.ndarray) -> np.ndarray:
        """Encode the most recent `window` hidden states into a world state."""
        self._check_fitted()
        w = np.asarray(window, dtype=np.float64)
        if w.shape[0] < self.config.window:
            raise ValueError(
                f"need {self.config.window} hidden states, got {w.shape[0]}"
            )
        return self.encoder.encode(w[-self.config.window :])

    def imagine(
        self,
        start: np.ndarray,
        steps: int,
        *,
        actions: np.ndarray | None = None,
        stop_on_gate: bool = True,
    ) -> Rollout:
        """Roll the world forward without running the wrapped model.

        `start` may be a raw hidden-state window or an already-encoded state.
        Rolling stops early when the gate fires, because past that point the
        engine has no basis for its own predictions and should say so instead
        of continuing to emit them.

        When started from a real window, hidden states are decoded *as
        displacements* from the last observed activation:

            h_{t+k} = h_t + Dec(u_{t+k}) - Dec(u_t)

        The linear channel keeps only `linear_dim` principal directions, so
        Dec(u) misses a fixed residual component. Absolute decoding pays that
        residual on every prediction; differential decoding cancels it, at the
        cost of assuming the unrepresented subspace is roughly constant over
        the horizon -- the same assumption carry-forward makes, except sigmoid
        additionally moves within the represented subspace. Over long horizons
        that assumption decays, which is one of the things the gate watches.
        """
        self._check_fitted()
        z = self._as_state(start)
        anchor_hidden, anchor_state = self._anchor(start, z)

        states: list[np.ndarray] = []
        readings: list[GateReading] = []
        grounded_at: int | None = None

        for i in range(steps):
            a = None if actions is None else np.asarray(actions)[i]
            z = self.operator.step(z, a)
            reading = self.gate.read(z)
            states.append(z)
            readings.append(reading)
            if reading.fire and stop_on_gate:
                grounded_at = i
                break

        arr = np.stack(states, axis=0) if states else np.zeros((0, self.state_dim))
        if not len(arr):
            hiddens = np.zeros((0, self.hidden_dim))
        elif anchor_hidden is None:
            hiddens = np.stack([self.encoder.decode(s) for s in arr], axis=0)
        else:
            residual = self.encoder.residual_of(anchor_hidden)
            hiddens = np.stack(
                [
                    self.encoder.decode_with_residual(s, residual, k + 1)
                    for k, s in enumerate(arr)
                ],
                axis=0,
            )
        return Rollout(
            states=arr,
            hiddens=hiddens,
            readings=tuple(readings),
            grounded_at=grounded_at,
            certificate=self.operator.certificate(max(steps, 1)),
        )

    def predict_hidden(
        self,
        window: np.ndarray,
        steps: int = 1,
        *,
        actions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predicted hidden states for the next `steps`, shape (steps, D)."""
        return self.imagine(window, steps, actions=actions, stop_on_gate=False).hiddens

    # ---- properties -------------------------------------------------------

    @property
    def state_dim(self) -> int:
        self._check_fitted()
        return self.encoder.state_dim

    @property
    def attractor(self) -> np.ndarray | None:
        """The fixed point z* of the autonomous dynamics -- the world's invariant."""
        self._check_fitted()
        return self.operator.fixed_point_

    @property
    def attractor_hidden(self) -> np.ndarray | None:
        """The attractor decoded back into hidden-state space."""
        z = self.attractor
        return None if z is None else self.encoder.decode(z)

    def certificate(self, horizon: int = 16) -> RolloutCertificate:
        self._check_fitted()
        return self.operator.certificate(horizon)

    def safe_horizon(self, tolerance: float) -> int:
        self._check_fitted()
        return self.operator.safe_horizon(tolerance)

    def summary(self) -> dict:
        self._check_fitted()
        cert = self.certificate()
        return {
            "hidden_dim": self.hidden_dim,
            "state_dim": self.state_dim,
            "topo_dim": self.encoder.topo_dim,
            "linear_dim": self.config.linear_dim,
            "window": self.config.window,
            "transitions": self.n_transitions,
            "block_diagonal": self.block_diagonal_,
            "rho": round(cert.rho, 6),
            "contractive": cert.contractive,
            "step_rmse": round(cert.step_rmse, 6),
            "bound_at_16": round(cert.bound, 6),
            "fixed_point_converged": self.operator.fixed_point_converged,
        }

    # ---- internals --------------------------------------------------------

    def _fit_operator(
        self,
        current: np.ndarray,
        following: np.ndarray,
        actions: np.ndarray | None,
    ) -> tuple[CouplingOperator, bool]:
        """Fit the operator, choosing the block structure by held-out error."""
        topo_dim = self.encoder.topo_dim
        choice = self.config.block_diagonal

        def build(block: bool, fit_x, fit_y, fit_a) -> CouplingOperator:
            return CouplingOperator(
                ridge=self.config.ridge,
                rho_max=self.config.rho_max,
                action_dim=self.config.action_dim,
                block_split=topo_dim if (block and topo_dim) else None,
            ).fit(fit_x, fit_y, actions=fit_a)

        if choice != "auto" or not topo_dim:
            return build(bool(choice) and bool(topo_dim), current, following, actions), bool(choice)

        cut = int(current.shape[0] * 0.8)
        if cut < 8 or current.shape[0] - cut < 4:
            return build(False, current, following, actions), False

        # Score a multi-step rollout, not a single step. One step cannot tell
        # the structures apart -- on Lorenz the two agreed to within 0.06% --
        # because the value of letting psi inform u compounds across the
        # rollout. Judged at 16 steps the same comparison separates cleanly
        # (1.097 coupled against 1.216 block-diagonal), and 16-step rollout is
        # what a world model is actually asked to do.
        horizon = min(16, max(2, (current.shape[0] - cut) // 4))

        def held_out_error(block: bool) -> float:
            op = build(
                block,
                current[:cut],
                following[:cut],
                None if actions is None else actions[:cut],
            )
            errs: list[float] = []
            for start in range(cut, current.shape[0] - horizon, horizon):
                z = current[start]
                for j in range(horizon):
                    a = None if actions is None else actions[start + j]
                    z = op.step(z, a)
                if not np.all(np.isfinite(z)):
                    return float("inf")
                target = following[start + horizon - 1]
                # score only the channel that decodes into activations
                errs.append(float(np.mean((z[topo_dim:] - target[topo_dim:]) ** 2)))
            return float(np.mean(errs)) if errs else float("inf")

        coupled_err, block_err = held_out_error(False), held_out_error(True)
        use_block = block_err < coupled_err
        return build(use_block, current, following, actions), use_block

    def _anchor(
        self, start: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (last observed hidden state, its encoding) when available.

        Only a real window carries an anchor. Imagining onward from a bare
        state vector has no observed activation to displace from, so decoding
        falls back to absolute.
        """
        arr = np.asarray(start, dtype=np.float64)
        return (arr[-1], z) if arr.ndim == 2 else (None, None)

    def _as_state(self, start: np.ndarray) -> np.ndarray:
        arr = np.asarray(start, dtype=np.float64)
        if arr.ndim == 1 and arr.shape[0] == self.state_dim:
            return arr
        if arr.ndim == 2:
            return self.observe(arr)
        raise ValueError(
            f"start must be a (window, {self.hidden_dim}) block or a "
            f"({self.state_dim},) state vector, got shape {arr.shape}"
        )

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("SigmoidWorldModel.fit must be called before use")

    # ---- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        payload: dict[str, Any] = {
            "config": np.array(vars(self.config), dtype=object),
            "hidden_dim": self.hidden_dim,
            "n_transitions": self.n_transitions,
            "encoder": np.array(self.encoder.state_dict(), dtype=object),
            "operator": np.array(self.operator.state_dict(), dtype=object),
            "gate": np.array(self.gate.state_dict(), dtype=object),
        }
        np.savez(path, **payload)
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "SigmoidWorldModel":
        data = np.load(Path(path), allow_pickle=True)
        wm = cls(config=SigmoidConfig(**data["config"].item()))
        wm.hidden_dim = int(data["hidden_dim"])
        wm.n_transitions = int(data["n_transitions"])
        wm.encoder = TopoEncoder.from_state_dict(data["encoder"].item())
        wm.operator = CouplingOperator.from_state_dict(data["operator"].item())
        wm.gate = SheafGate.from_state_dict(data["gate"].item())
        wm.fitted = True
        return wm


def _align_actions(
    actions: Sequence[np.ndarray],
    per_traj: Sequence[np.ndarray],
    window: int,
) -> np.ndarray:
    """Drop the first window-1 actions of each episode and the last transition."""
    if len(actions) != len(per_traj):
        raise ValueError("one action array per trajectory is required")
    aligned = []
    for acts, states in zip(actions, per_traj):
        acts = np.asarray(acts, dtype=np.float64)
        trimmed = acts[window - 1 :][: len(states)]
        if len(trimmed) < len(states):
            raise ValueError("action array is shorter than its trajectory")
        aligned.append(trimmed[:-1])
    return np.concatenate(aligned, axis=0)


def wrap(
    model: Any,
    inputs: Iterable[Any],
    *,
    layer: int | str = -1,
    config: SigmoidConfig | None = None,
) -> SigmoidWorldModel:
    """Convert any normal model into a world model. The one-line entry point."""
    return SigmoidWorldModel.from_model(model, inputs, layer=layer, config=config)

"""Control: planning against a topological world model.

A world model that cannot be acted on is a forecaster. This turns one into a
controller by sampling action sequences, rolling each forward through the
learned operator, and keeping the ones that score well -- model-predictive
control with the cross-entropy method, running entirely in the latent space.

Two things here are not standard MPC.

**Topological cost.** The cost function is evaluated on world states
``z = [psi ; u]``, so it can reference the topological channel directly. That
makes objectives available that raw coordinates cannot express without a
bespoke detector:

    "never let the collision graph merge"      -> a cost on beta_0
    "keep the workspace simply connected"      -> a cost on beta_1
    "hold these bodies in one contact island"  -> a cost on merge height

For a robot whose state is a set of bodies with a fixed contact radius, that is
the natural language for the constraint, and it is exactly the regime the
measurements say the topological channel is good at (SIGMOID.md 4.3).

**Refusal.** Every candidate rollout is scored by the sheaf gate. Plans that
leave the region where the model was calibrated are discarded before they are
ranked, and if every candidate leaves it the planner reports that it has no
trustworthy plan rather than returning the least-bad extrapolation. A world
model asked to control a robot should be able to say it does not know.

The planning horizon is capped by the operator's own error certificate, so the
controller never plans further than its bound permits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .engine import SigmoidWorldModel

__all__ = ["Plan", "TopologicalMPC", "beta0_cost", "target_cost"]

CostFn = Callable[[np.ndarray], float]
"""Maps a (horizon, state_dim) imagined rollout to a scalar. Lower is better."""


@dataclass(frozen=True)
class Plan:
    """The result of planning. `actions[0]` is the one to execute."""

    actions: np.ndarray
    """(horizon, action_dim) chosen action sequence."""

    states: np.ndarray
    """(horizon, state_dim) imagined rollout under those actions."""

    cost: float
    feasible: bool
    """False when every candidate was rejected by the gate."""

    rejected: int
    """How many candidates the gate discarded."""

    considered: int
    horizon: int
    max_gate_score: float

    @property
    def action(self) -> np.ndarray:
        return self.actions[0]


@dataclass
class TopologicalMPC:
    """Cross-entropy-method MPC over a sigmoid world model."""

    world: SigmoidWorldModel
    action_low: np.ndarray
    action_high: np.ndarray

    horizon: int = 8
    samples: int = 96
    elites: int = 12
    iterations: int = 3
    gate_limit: float = 1.0
    """Reject a candidate when any imagined state scores at or above this.

    Expect most candidates to be rejected, and size `samples` accordingly.
    Measured on the swarm world: real states score a median of 0.56, while
    rollouts under *uniformly random* action sequences score a median of 26 --
    because random actions genuinely do produce implausible futures. Only ~12%
    of random candidates clear a limit of 5.0. That is the gate working, not
    misfiring, but it means a planner sampling 32 candidates can find nothing
    feasible by luck alone. `Plan.rejected` / `Plan.considered` report the rate.
    """

    init_spread: float = 0.4
    """Initial CEM std, as a fraction of the action range.

    Sampling the full range on the first iteration puts most candidates at the
    extremes, where the gate rejects them and CEM has nothing to fit. Starting
    narrower keeps the first generation near the plausible region and lets the
    elite statistics widen it if the cost calls for it."""

    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.world.fitted:
            raise RuntimeError("the world model must be fitted before planning")
        if not self.world.config.action_dim:
            raise ValueError(
                "planning needs an action-conditioned world model; fit with "
                "SigmoidConfig(action_dim=...) and pass actions to fit()"
            )
        self.action_low = np.asarray(self.action_low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(self.action_high, dtype=np.float64).reshape(-1)
        if self.action_low.shape != self.action_high.shape:
            raise ValueError("action_low and action_high must have the same shape")
        if self.action_low.shape[0] != self.world.config.action_dim:
            raise ValueError(
                f"action bounds have width {self.action_low.shape[0]}, "
                f"world model expects {self.world.config.action_dim}"
            )
        if np.any(self.action_high < self.action_low):
            raise ValueError("action_high must be >= action_low")
        if self.elites < 1 or self.elites > self.samples:
            raise ValueError("elites must satisfy 1 <= elites <= samples")
        self._rng = np.random.default_rng(self.seed)

    # ---- horizon -----------------------------------------------------------

    def effective_horizon(self, tolerance: float | None = None) -> int:
        """Requested horizon, capped by what the error certificate supports."""
        if tolerance is None:
            return self.horizon
        return max(1, min(self.horizon, self.world.safe_horizon(tolerance)))

    # ---- planning ----------------------------------------------------------

    def plan(
        self,
        observation: np.ndarray,
        cost: CostFn,
        *,
        tolerance: float | None = None,
    ) -> Plan:
        """Plan from a hidden-state window or an encoded world state."""
        horizon = self.effective_horizon(tolerance)
        z0 = self.world._as_state(np.asarray(observation, dtype=np.float64))
        dim = self.action_low.shape[0]

        mean = np.tile((self.action_low + self.action_high) / 2.0, (horizon, 1))
        spread = np.tile(
            (self.action_high - self.action_low) * self.init_spread, (horizon, 1)
        )

        best_actions = mean.copy()
        best_states = np.zeros((horizon, self.world.state_dim))
        best_cost = np.inf
        total_rejected = 0
        total_considered = 0
        best_gate = np.inf

        for _ in range(self.iterations):
            candidates = self._rng.normal(
                loc=mean, scale=np.maximum(spread, 1e-9),
                size=(self.samples, horizon, dim),
            )
            np.clip(candidates, self.action_low, self.action_high, out=candidates)

            scored: list[tuple[float, int]] = []
            for index, actions in enumerate(candidates):
                total_considered += 1
                states, gate_score = self._rollout(z0, actions)
                if gate_score >= self.gate_limit:
                    total_rejected += 1
                    continue
                value = float(cost(states))
                if not np.isfinite(value):
                    total_rejected += 1
                    continue
                scored.append((value, index))
                if value < best_cost:
                    best_cost, best_actions, best_states = value, actions.copy(), states
                    best_gate = gate_score

            if not scored:
                continue  # everything was rejected; widen and retry
            scored.sort(key=lambda item: item[0])
            elite = candidates[[i for _, i in scored[: self.elites]]]
            mean = elite.mean(axis=0)
            spread = elite.std(axis=0) + 1e-6

        feasible = np.isfinite(best_cost)
        return Plan(
            actions=best_actions,
            states=best_states,
            cost=float(best_cost) if feasible else float("inf"),
            feasible=bool(feasible),
            rejected=total_rejected,
            considered=total_considered,
            horizon=horizon,
            max_gate_score=float(best_gate) if feasible else float("inf"),
        )

    def _rollout(
        self, z0: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Imagine one action sequence; return states and the worst gate score."""
        z = z0
        states = np.empty((actions.shape[0], self.world.state_dim))
        worst = 0.0
        for step in range(actions.shape[0]):
            z = self.world.operator.step(z, actions[step])
            if not np.all(np.isfinite(z)):
                return states, np.inf
            states[step] = z
            worst = max(worst, self.world.gate.read(z).score)
        return states, worst

    # ---- closed loop -------------------------------------------------------

    def control(
        self,
        step_env: Callable[[np.ndarray], np.ndarray],
        window: np.ndarray,
        cost: CostFn,
        steps: int,
        *,
        tolerance: float | None = None,
    ) -> dict:
        """Run receding-horizon control against a live environment.

        `step_env` takes one action and returns the next observation. Only the
        first action of each plan is executed, then the whole plan is redone --
        which is what makes the short, certified horizon sufficient.
        """
        history = list(np.asarray(window, dtype=np.float64))
        executed, costs, refusals = [], [], 0

        for _ in range(steps):
            recent = np.asarray(history[-self.world.config.window :])
            plan = self.plan(recent, cost, tolerance=tolerance)
            if not plan.feasible:
                refusals += 1
                action = np.zeros(self.action_low.shape[0])  # refuse: hold still
            else:
                action = plan.action
                costs.append(plan.cost)
            executed.append(action)
            history.append(np.asarray(step_env(action), dtype=np.float64).reshape(-1))

        return {
            "observations": np.asarray(history),
            "actions": np.asarray(executed),
            "costs": costs,
            "refusals": refusals,
            "mean_cost": float(np.mean(costs)) if costs else float("nan"),
        }


# --------------------------------------------------------------------------
# cost helpers
# --------------------------------------------------------------------------


def beta0_cost(
    world: SigmoidWorldModel,
    target_components: int,
    radius: float | None = None,
    weight: float = 1.0,
) -> CostFn:
    """Penalize departure from a target component count at a physical radius.

    `radius` is the distance at which two bodies count as touching -- a contact
    radius, a safety margin, an interaction cutoff. The channel closest to it is
    used, and a warning-free silent mismatch is made impossible by refusing when
    nothing is close.

    Passing `radius=None` falls back to the coarsest calibrated radius, which is
    almost never what a control problem wants: on the robot demo the calibrated
    quantiles landed at 2.21-11.94 for a contact radius of 0.9, so the resulting
    cost constrained a grouping unrelated to collision. Prefer to say the number.
    """
    encoder = world.encoder
    if not encoder.n_abs_radii or encoder.abs_radii_ is None:
        raise ValueError(
            "beta0_cost requires absolute-radius Betti samples; set "
            "SigmoidConfig(n_abs_radii=...) or SigmoidConfig(abs_radii=(...,))"
        )

    radii = np.asarray(encoder.abs_radii_, dtype=np.float64)
    if radius is None:
        offset = len(radii) - 1
    else:
        offset = int(np.argmin(np.abs(radii - radius)))
        chosen = float(radii[offset])
        if not np.isclose(chosen, radius, rtol=0.25):
            raise ValueError(
                f"no calibrated radius near {radius}: closest is {chosen:.3f} "
                f"out of {np.round(radii, 3).tolist()}. Pass "
                f"SigmoidConfig(abs_radii=({radius}, ...)) so the constraint is "
                f"evaluated at the distance it actually refers to."
            )

    c = encoder.config
    n_dims = 2 if c.use_h1 else 1
    index = n_dims * (c.hilbert_degree + c.n_radii) + offset
    mean = encoder.psi_mean_[index]
    scale = encoder.psi_scale_[index]

    def cost(states: np.ndarray) -> float:
        standardized = states[:, index]
        beta0 = standardized * scale + mean  # undo psi standardization
        return float(weight * np.mean((beta0 - target_components) ** 2))

    return cost


def target_cost(
    world: SigmoidWorldModel,
    goal_hidden: np.ndarray,
    weight: float = 1.0,
    terminal_weight: float = 4.0,
) -> CostFn:
    """Drive the decoded observation toward a goal, weighting the endpoint."""
    goal = np.asarray(goal_hidden, dtype=np.float64).reshape(-1)

    def cost(states: np.ndarray) -> float:
        decoded = np.stack([world.encoder.decode(z) for z in states])
        errors = np.mean((decoded - goal) ** 2, axis=1)
        return float(weight * errors.mean() + terminal_weight * errors[-1])

    return cost

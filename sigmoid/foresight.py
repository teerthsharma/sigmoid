"""Visual foresight, inverse dynamics, and the interlock that sits after the network.

A world model that can imagine is a simulator the robot carries with it: propose
actions, roll them forward, score the outcome, and only then move. Three pieces:

`VisualForesight`     action-conditioned rollout + candidate ranking
`InverseDynamics`     recover the action that moved between two states
`KinematicInterlock`  deterministic veto on physically impossible commands

The interlock is the one that matters most and it is the least clever code here.
A world model can produce a rollout that is *visually* coherent and physically
impossible -- an arm passing through itself, a joint moving faster than its motor
can. Decoding such a rollout gives an inverse-dynamics model an out-of-bounds
target, and it will happily emit a command for it. So a deterministic physics
layer sits *after* the network with authority to refuse. No learned component,
because the thing being guarded against is a learned component being wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ForesightResult",
    "InterlockVerdict",
    "InverseDynamics",
    "KinematicInterlock",
    "SubGoal",
    "VisualForesight",
]


# --------------------------------------------------------------------------
# foresight
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForesightResult:
    """One imagined future, with how much of it is trustworthy."""

    states: np.ndarray
    """(k, state_dim) imagined world states."""

    observations: np.ndarray
    """(k, D) decoded observations -- pixels, coordinates, whatever was fitted."""

    grounded_at: int | None
    """Step where the gate fired, or None if the whole horizon held."""

    safe_steps: int
    """Steps the error certificate covers at the requested tolerance."""

    gate_scores: np.ndarray
    cost: float = float("nan")

    @property
    def trusted(self) -> int:
        """Steps that are BOTH ungated and inside the certificate.

        Two independent limits and the honest answer is the smaller one: the gate
        catches off-manifold drift, the certificate bounds accumulated error, and
        a rollout can fail either way first.
        """
        gated = len(self.states) if self.grounded_at is None else self.grounded_at
        return int(min(gated, self.safe_steps))


@dataclass
class VisualForesight:
    """Imagine forward under proposed actions, and rank candidates."""

    world: Any
    encoder: Any = None
    """Optional observation encoder, e.g. `sigmoid.vision.TopoImageEncoder`.
    When given, `rollout` accepts raw frames instead of pre-encoded windows."""

    tolerance: float = 1.0
    """Error budget for `safe_steps`. Units are the operator's state space."""

    def _window(self, frames: np.ndarray) -> np.ndarray:
        arr = np.asarray(frames, dtype=np.float64)
        if self.encoder is not None and arr.ndim >= 3:
            return self.encoder.encode_video(arr)
        return arr

    def rollout(
        self,
        frames: np.ndarray,
        actions: np.ndarray | None = None,
        *,
        steps: int | None = None,
        fidelity: str = "full",
    ) -> ForesightResult:
        """Predict future observations from a window plus proposed actions.

        `fidelity="semantic"` skips the decode back to observation space and
        returns only the latent states. An inverse-dynamics model consumes the
        latent directly, so when the caller only wants an action there is no
        reason to reconstruct pixels it will throw away. `demo()` measures
        whether that saving is real on this engine.
        """
        if fidelity not in ("full", "semantic"):
            raise ValueError("fidelity must be 'full' or 'semantic'")
        window = self._window(frames)
        acts = None if actions is None else np.atleast_2d(np.asarray(actions, dtype=np.float64))
        k = steps if steps is not None else (len(acts) if acts is not None else 8)

        roll = self.world.imagine(window, k, actions=acts, stop_on_gate=False)
        scores = np.asarray([r.score for r in roll.readings], dtype=np.float64)
        fired = next((i for i, r in enumerate(roll.readings) if r.fire), None)

        obs = (
            roll.hiddens
            if fidelity == "full"
            else np.zeros((len(roll.states), 0), dtype=np.float64)
        )
        return ForesightResult(
            states=roll.states,
            observations=obs,
            grounded_at=fired,
            safe_steps=self.world.safe_horizon(self.tolerance),
            gate_scores=scores,
        )

    def evaluate(
        self,
        frames: np.ndarray,
        candidates: list[np.ndarray],
        cost: Callable[[np.ndarray], float],
        *,
        fidelity: str = "semantic",
    ) -> list[ForesightResult]:
        """Score candidate action sequences, best first.

        Defaults to `semantic` fidelity: ranking N candidates means N rollouts,
        and decoding observations for the N-1 that lose is wasted work.
        """
        out = []
        for actions in candidates:
            res = self.rollout(frames, actions, fidelity=fidelity)
            out.append(
                ForesightResult(
                    states=res.states,
                    observations=res.observations,
                    grounded_at=res.grounded_at,
                    safe_steps=res.safe_steps,
                    gate_scores=res.gate_scores,
                    cost=float(cost(res.states)),
                )
            )
        return sorted(out, key=lambda r: r.cost)


# --------------------------------------------------------------------------
# inverse dynamics
# --------------------------------------------------------------------------


@dataclass
class InverseDynamics:
    """Recover the action that took the system between two states.

    Ridge, closed form, matching the rest of this library -- the forward operator
    is a ridge solve too, and a gradient-trained IDM would be the only thing here
    needing a training loop.

    **`history` is not a tuning knob, it is a correctness requirement.** For a
    second-order plant the action is not recoverable from one state pair. The
    swarm used in `demo()` integrates

        v_{t+1} = 0.85 v_t + 0.25 a_t,    p_{t+1} = p_t + 0.25 v_{t+1}

    so `p_{t+1} - p_t` depends on `v_t`, which is hidden state that no single pair
    of positions determines. Fitting on `[s_t ; s_{t+1}]` scored **R^2 = -49.6**
    -- fifty times worse than predicting the mean -- because the target genuinely
    is not a function of the inputs. Two previous states supply velocity by finite
    difference and the fit becomes well posed. Default 2 for that reason; use 1
    only for a first-order plant, and 3+ if acceleration is also hidden.

    Reported against predict-the-mean, not just held-out R^2. An IDM that cannot
    beat the mean is not recovering actions, it is reporting the average command.
    """

    history: int = 2
    """Past states fed alongside the successor. See the class docstring."""

    ridge: float = 1e-6
    """Ridge on **standardized** features.

    Standardization is not cosmetic here. The design matrix is badly conditioned
    -- measured cond(X'X) = 3.4e7 on the swarm plant, because recovering an action
    of order 1 from positions of order 1 needs coefficients of order 16. On raw
    features a ridge of 1e-3 left a residual of 0.188 on actions bounded in
    [-1, 1], which is a useless fit produced by a penalty that looked small.
    Ridge is scale-dependent; a fixed value only means something once the columns
    are comparable.
    """

    W_: np.ndarray | None = field(default=None, repr=False)
    mu_: np.ndarray | None = field(default=None, repr=False)
    sigma_: np.ndarray | None = field(default=None, repr=False)
    r2_: float = float("nan")
    r2_mean_baseline_: float = float("nan")
    fitted: bool = False

    def _lift(self, past: np.ndarray, s_next: np.ndarray) -> np.ndarray:
        """past is (n, history, d) or (history, d); s_next is (n, d) or (d,)."""
        p = np.asarray(past, dtype=np.float64)
        nx = np.atleast_2d(np.asarray(s_next, dtype=np.float64))
        if p.ndim == 2:
            p = p[None, ...]
        flat = p.reshape(len(p), -1)
        return np.concatenate([flat, nx, np.ones((len(flat), 1))], axis=1)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        holdout: float = 0.3,
    ) -> InverseDynamics:
        """Fit on consecutive state pairs. `states` is (T, d), `actions` (T-1, k)."""
        s = np.asarray(states, dtype=np.float64)
        a = np.atleast_2d(np.asarray(actions, dtype=np.float64))
        h = max(1, int(self.history))
        # index t runs over positions where h past states and one successor exist
        n = min(len(s) - h, len(a) - h + 1)
        if n < 4:
            raise ValueError(f"need at least 4 transitions with history={h}")
        past = np.stack([s[i : i + h] for i in range(n)])       # (n, h, d)
        nxt = s[h : h + n]                                      # (n, d)
        X = self._lift(past, nxt)
        Y = a[h - 1 : h - 1 + n]

        cut = max(2, int(n * (1.0 - holdout)))
        self.mu_ = X[:cut].mean(axis=0)
        sd = X[:cut].std(axis=0)
        self.sigma_ = np.where(sd > 1e-12, sd, 1.0)
        Xs = (X - self.mu_) / self.sigma_

        gram = Xs[:cut].T @ Xs[:cut] + self.ridge * np.eye(Xs.shape[1])
        self.W_ = np.linalg.solve(gram, Xs[:cut].T @ Y[:cut]).T

        held_x, held_y = Xs[cut:], Y[cut:]
        if len(held_y):
            err = held_y - held_x @ self.W_.T
            var = held_y - Y[:cut].mean(axis=0)
            ss_res = float((err**2).sum())
            ss_tot = float((var**2).sum())
            self.r2_ = 1.0 - ss_res / max(ss_tot, 1e-12)
            self.r2_mean_baseline_ = 0.0  # predicting the train mean is R^2 = 0 by construction
        self.fitted = True
        return self

    def predict(self, past: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        """`past` is the last `history` states, shape (history, d) or (d,)."""
        if not self.fitted or self.W_ is None:
            raise RuntimeError("InverseDynamics.fit must be called before predict")
        p = np.asarray(past, dtype=np.float64)
        if p.ndim == 1:
            p = np.tile(p, (self.history, 1))  # a single state repeated: zero velocity
        x = (self._lift(p, next_state) - self.mu_) / self.sigma_
        return (x @ self.W_.T)[0]

    def decode_rollout(self, states: np.ndarray) -> np.ndarray:
        """Actions implied by a whole imagined trajectory."""
        s = np.asarray(states, dtype=np.float64)
        h = self.history
        return np.stack([self.predict(s[i : i + h], s[i + h]) for i in range(len(s) - h)])


# --------------------------------------------------------------------------
# the interlock
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InterlockVerdict:
    allowed: bool
    violation: str = ""
    index: int = -1
    """Which joint/dimension tripped, or -1."""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class KinematicInterlock:
    """Deterministic veto between the network's output and the actuator.

    Pure interval arithmetic. No learned component, no probability, no
    threshold that needs calibrating -- the point is to be trustworthy while the
    thing above it is not.

    `dt` matters: a velocity limit is meaningless without knowing how long the
    step is, and getting it wrong is how a "safe" interlock passes a command that
    moves a joint ten times too fast.
    """

    position_limits: tuple[np.ndarray, np.ndarray] | None = None
    velocity_limit: np.ndarray | None = None
    accel_limit: np.ndarray | None = None
    dt: float = 0.02
    self_collision_fn: Callable[[np.ndarray], bool] | None = None

    def check(
        self,
        action: np.ndarray,
        state: np.ndarray,
        prev_velocity: np.ndarray | None = None,
    ) -> InterlockVerdict:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        s = np.asarray(state, dtype=np.float64).reshape(-1)

        if not np.all(np.isfinite(a)):
            bad_i = int(np.argmax(~np.isfinite(a)))
            return InterlockVerdict(False, "action contains NaN or inf", bad_i)

        if self.position_limits is not None:
            lo, hi = self.position_limits
            nxt = s[: len(a)] + a * self.dt
            bad = np.flatnonzero((nxt < lo[: len(a)]) | (nxt > hi[: len(a)]))
            if bad.size:
                return InterlockVerdict(False, "position limit", int(bad[0]))

        if self.velocity_limit is not None:
            bad = np.flatnonzero(np.abs(a) > self.velocity_limit[: len(a)])
            if bad.size:
                return InterlockVerdict(False, "velocity limit", int(bad[0]))

        if self.accel_limit is not None and prev_velocity is not None:
            accel = (a - np.asarray(prev_velocity).reshape(-1)[: len(a)]) / self.dt
            bad = np.flatnonzero(np.abs(accel) > self.accel_limit[: len(a)])
            if bad.size:
                return InterlockVerdict(False, "acceleration limit", int(bad[0]))

        if self.self_collision_fn is not None and self.self_collision_fn(s):
            return InterlockVerdict(False, "self collision", -1)

        return InterlockVerdict(True)

    def clamp(self, action: np.ndarray) -> np.ndarray:
        """Project into the velocity box. Position and collision are NOT clamped.

        Clamping a velocity is well defined. Clamping *out of* a self-collision
        is not -- there is no scalar projection that resolves it, and pretending
        otherwise would turn a refusal into a plausible-looking wrong command.
        Those stay vetoes.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.velocity_limit is None:
            return a
        lim = self.velocity_limit[: len(a)]
        return np.clip(a, -lim, lim)

    def validate_rollout(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[bool, list[InterlockVerdict]]:
        """Reject a visually coherent but physically impossible trajectory."""
        s = np.asarray(states, dtype=np.float64)
        a = np.atleast_2d(np.asarray(actions, dtype=np.float64))
        verdicts = []
        prev = None
        for i in range(min(len(s), len(a))):
            v = self.check(a[i], s[i], prev_velocity=prev)
            verdicts.append(v)
            prev = a[i]
        return all(v.allowed for v in verdicts), verdicts


# --------------------------------------------------------------------------
# affordances
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SubGoal:
    name: str
    target: np.ndarray
    reachable: bool
    steps_needed: int
    reason: str = ""


def decompose(
    world: Any,
    window: np.ndarray,
    affordances: dict[str, np.ndarray],
    *,
    tolerance: float = 1.0,
    max_steps: int = 32,
    actions: np.ndarray | None = None,
) -> list[SubGoal]:
    """Order affordances into sub-goals by reachability under the world model.

    Not a planner. Each affordance is a target observation; this asks the world
    model how many imagined steps it takes to get near it, and whether that
    number is inside the certified horizon. Anything outside is returned marked
    unreachable rather than silently dropped -- an agent needs to know that a
    sub-goal was excluded and why, or it will keep proposing it.

    `actions` defaults to zeros, i.e. reachability **under a hold-still policy**.
    That is a real limitation and not a neutral default: an affordance a robot
    could reach by moving may be reported unreachable. Pass a candidate action
    sequence to ask the question that policy actually faces.
    """
    safe = world.safe_horizon(tolerance)
    horizon = min(max_steps, max(safe, 1))
    if actions is None and world.config.action_dim:
        actions = np.zeros((horizon, world.config.action_dim))
    goals = []
    for name, target in affordances.items():
        tgt = np.asarray(target, dtype=np.float64).reshape(-1)
        roll = world.imagine(window, horizon, actions=actions, stop_on_gate=False)
        dists = np.linalg.norm(roll.hiddens - tgt, axis=1)
        best = int(np.argmin(dists))
        within = best + 1 <= safe
        goals.append(
            SubGoal(
                name=name,
                target=tgt,
                reachable=bool(within),
                steps_needed=best + 1,
                reason="" if within else f"needs {best + 1} steps, certified horizon is {safe}",
            )
        )
    return sorted(goals, key=lambda g: (not g.reachable, g.steps_needed))


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _swarm(n_robots: int = 4, episodes: int = 24, length: int = 80, seed: int = 0):
    """Damped double integrator, the plant used across this repo's robot tests."""
    rng = np.random.default_rng(seed)
    obs, acts = [], []
    for _ in range(episodes):
        pos = rng.normal(scale=2.0, size=(n_robots, 2))
        vel = np.zeros_like(pos)
        o, a = [], []
        drive = rng.normal(scale=0.5, size=(n_robots, 2))
        for _ in range(length):
            o.append(pos.reshape(-1).copy())
            drive = 0.85 * drive + 0.3 * rng.normal(scale=0.5, size=(n_robots, 2))
            action = np.clip(drive, -1, 1)
            a.append(action.reshape(-1).copy())
            vel = 0.85 * vel + 0.25 * action
            pos = pos + 0.25 * vel
        obs.append(np.asarray(o))
        acts.append(np.asarray(a))
    return obs, acts


def demo() -> None:
    """Measured: IDM against the mean, interlock cost, fidelity trade."""
    import time

    from .engine import SigmoidConfig, SigmoidWorldModel

    obs, acts = _swarm()
    world = SigmoidWorldModel(
        config=SigmoidConfig(window=10, linear_dim=12, entity_dim=2, action_dim=8)
    ).fit(obs, actions=acts)

    # ---- inverse dynamics, against the baseline that matters
    # history=1 is the wrong model for a second-order plant; show it, then fix it
    naive = InverseDynamics(history=1).fit(obs[0], acts[0])
    idm = InverseDynamics(history=2).fit(obs[0], acts[0])
    print(f"  IDM history=1    R^2 {naive.r2_:+.4f}   (velocity is hidden -> ill posed)")
    print(f"  IDM history=2    R^2 {idm.r2_:+.4f}   vs predict-the-mean 0.0000")
    assert idm.r2_ > 0.0, f"IDM does not beat the mean (R^2 {idm.r2_:.4f})"
    assert idm.r2_ > naive.r2_, "adding history must help on a second-order plant"
    recovered = idm.predict(obs[0][4:6], obs[0][6])
    assert recovered.shape == (8,)

    # ---- foresight, and whether semantic fidelity earns its knob
    fs = VisualForesight(world, tolerance=5.0)
    window = obs[0][:10]
    plan = np.zeros((6, 8))
    for _ in range(3):
        fs.rollout(window, plan)
    t0 = time.perf_counter()
    for _ in range(30):
        full = fs.rollout(window, plan, fidelity="full")
    full_ms = (time.perf_counter() - t0) / 30 * 1e3
    t0 = time.perf_counter()
    for _ in range(30):
        sem = fs.rollout(window, plan, fidelity="semantic")
    sem_ms = (time.perf_counter() - t0) / 30 * 1e3
    saved = (full_ms - sem_ms) / full_ms * 100 if full_ms else 0.0
    print(
        f"  rollout 6 steps  full {full_ms:.3f} ms   semantic {sem_ms:.3f} ms   "
        f"saved {saved:+.0f}%"
    )
    if saved < 10.0:
        # Reported rather than hidden: the knob does almost nothing HERE because
        # this decode is one matvec into a 8-dim observation. It would matter for
        # a real pixel decoder, and nowhere else. Do not claim a saving that was
        # not measured on the decoder actually in use.
        print("    (fidelity knob is ~free on this decoder; only pays for pixels)")
    print(f"  trusted steps    {full.trusted} of {len(full.states)} "
          f"(gate {full.grounded_at}, certificate {full.safe_steps})")
    assert full.observations.shape[1] > 0 and sem.observations.shape[1] == 0

    # ---- candidate ranking picks the cheapest
    rng = np.random.default_rng(1)
    cands = [rng.uniform(-1, 1, size=(6, 8)) for _ in range(6)]
    ranked = fs.evaluate(window, cands, cost=lambda s: float(np.abs(s).mean()))
    assert ranked[0].cost <= ranked[-1].cost, "ranking is not sorted by cost"
    print(f"  ranked 6 plans   best {ranked[0].cost:.4f}   worst {ranked[-1].cost:.4f}")

    # ---- interlock: cost, and does it actually catch things
    lock = KinematicInterlock(
        position_limits=(np.full(8, -6.0), np.full(8, 6.0)),
        velocity_limit=np.full(8, 1.0),
        accel_limit=np.full(8, 4.0),  # (1 - -1)/0.25 = 8 must trip this
        dt=0.25,
    )
    state = obs[0][10]
    ok = lock.check(np.zeros(8), state)
    assert ok.allowed, f"a zero action was refused: {ok.violation}"

    too_fast = lock.check(np.full(8, 5.0), state)
    assert not too_fast.allowed and too_fast.violation == "velocity limit"

    nan = lock.check(np.full(8, np.nan), state)
    assert not nan.allowed and "NaN" in nan.violation

    far = lock.check(np.zeros(8), np.full(8, 100.0))
    assert not far.allowed and far.violation == "position limit"

    jerk = lock.check(np.full(8, 1.0), state, prev_velocity=np.full(8, -1.0))
    assert not jerk.allowed and jerk.violation == "acceleration limit"

    for _ in range(200):
        lock.check(np.zeros(8), state)
    t0 = time.perf_counter()
    reps = 20_000
    for _ in range(reps):
        lock.check(np.zeros(8), state)
    us = (time.perf_counter() - t0) / reps * 1e6
    print(f"  interlock check  {us:.2f} us  (contact-rich budget is 10-20 ms)")
    assert us < 200.0, f"interlock too slow for the control loop: {us:.1f}us"

    # a physically impossible rollout must be rejected, not decoded
    impossible = np.stack([state + i * 50.0 for i in range(6)])  # 50 units per step
    bad_actions = idm.decode_rollout(impossible)
    accepted, verdicts = lock.validate_rollout(impossible[: len(bad_actions)], bad_actions)
    assert not accepted, "an impossible rollout was accepted"
    tripped = next(v for v in verdicts if not v.allowed)
    print(f"  impossible rollout rejected: {tripped.violation} at dim {tripped.index}")

    clamped = lock.clamp(np.full(8, 9.0))
    assert np.allclose(clamped, 1.0), "clamp did not project into the velocity box"

    # ---- affordance ordering
    goals = decompose(
        world,
        window,
        {"near": obs[0][12], "far": obs[0][70] * 3.0},
        tolerance=5.0,
    )
    assert goals and any(g.reachable for g in goals)
    for g in goals:
        print(f"  affordance {g.name:<6} reachable={g.reachable} steps={g.steps_needed} {g.reason}")
    print("demo ok")


if __name__ == "__main__":
    demo()

"""An LLM agent commanding a robot through a world model that can refuse.

This is the point of the package. The language model never touches the plant.
It calls `observe`, `imagine`, `plan_to`, `check_safety` -- all of which run in
the latent space of a fitted `SigmoidWorldModel` -- and only `execute` moves
anything. Two independent things stand between a token and an actuator:

**The gate.** `TopologicalMPC` rejects every candidate rollout whose imagined
states leave the region the model was calibrated on, and reports
`Plan.feasible == False` when that leaves nothing. Here that refusal is returned
to the model as a *result*, not raised as an exception. The distinction matters:
an exception ends the loop and loses the reason, while a refusal payload
("every one of 96 candidates was rejected by the gate") is something the model
can read and route around -- pick a nearer goal, imagine first, observe again.
A control stack whose only response to "I don't know" is a stack trace cannot
be recovered from by anything except a human.

**Confirmation.** `execute` is `dangerous=True`, so `Agent` will not run it
unless something outside the conversation confirms it. The model may ask to
move the robot; it may not authorise the move. And an infeasible plan is
refused at execute time even when confirmed, because a human clicking yes is
not evidence that the world model trusts the rollout.

    robot = RobotAgent(world, mpc, provider=provider, window=window, step_env=env.step)
    robot.run("get the team to their goals without touching")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .agent import Agent, AgentResult, Tool, ToolRegistry
from .control import Plan, TopologicalMPC, target_cost
from .engine import SigmoidWorldModel

__all__ = ["RobotAgent"]

ROBOT_SYSTEM = (
    "You are the controller for a physical robot. You act only through the "
    "tools below, which run against a calibrated world model. That model can "
    "refuse: when plan_to reports status \"refused\", every candidate motion it "
    "considered left the region where the model was calibrated, so no plan is "
    "trustworthy. Treat a refusal as information -- observe again, imagine "
    "fewer steps, or choose a nearer goal -- never as a reason to execute "
    "anyway. Nothing moves until you call execute, and execute needs an "
    "operator's confirmation that you do not have."
)


class RobotAgent:
    """An `Agent` plus a world model, an MPC, and the plant they command."""

    def __init__(
        self,
        world: SigmoidWorldModel,
        mpc: TopologicalMPC,
        *,
        provider: Any = None,
        window: np.ndarray | Sequence[np.ndarray] | None = None,
        step_env: Callable[[np.ndarray], np.ndarray] | None = None,
        goal: np.ndarray | None = None,
        hooks: Any = None,
        confirm: Callable[[Tool, dict], bool] | None = None,
        require_confirmation: bool = True,
        max_steps: int = 8,
        max_imagine_steps: int = 64,
        system: str | None = None,
    ) -> None:
        if not world.fitted:
            raise RuntimeError("RobotAgent needs a fitted SigmoidWorldModel")
        self.world = world
        self.mpc = mpc
        self.step_env = step_env
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(-1)
        self.max_imagine_steps = int(max_imagine_steps)

        self.history: list[np.ndarray] = []
        if window is not None:
            self.observe_window(window)
        self.steps_taken = 0
        self.plans: dict[str, Plan] = {}
        self._plan_count = 0

        self.tools = ToolRegistry(self._build_tools())
        self.agent = Agent(
            provider,
            tools=self.tools,
            hooks=hooks,
            max_steps=max_steps,
            system=system or ROBOT_SYSTEM,
            confirm=confirm,
            require_confirmation=require_confirmation,
        )

    # ---- plant state -------------------------------------------------------

    def observe_window(self, window: np.ndarray | Sequence[np.ndarray]) -> None:
        """Seed or replace the observation history from real measurements."""
        rows = [np.asarray(o, dtype=np.float64).reshape(-1) for o in np.asarray(window)]
        if len(rows) < self.world.config.window:
            raise ValueError(
                f"need at least {self.world.config.window} observations to encode a "
                f"world state, got {len(rows)}"
            )
        self.history = rows

    def window_array(self) -> np.ndarray:
        if not self.history:
            raise RuntimeError("no observations yet; pass window= or call observe_window()")
        return np.asarray(self.history[-self.world.config.window :])

    def state(self) -> np.ndarray:
        """The current encoded world state z = [psi ; u]."""
        return self.world.observe(self.window_array())

    def run(self, task: str) -> AgentResult:
        return self.agent.run(task)

    # ---- tools -------------------------------------------------------------

    def _build_tools(self) -> list[Tool]:
        n = int(self.world.hidden_dim)
        action_dim = int(self.world.config.action_dim)
        return [
            Tool(
                name="observe",
                description=(
                    "Read the robot's current state through the world model: latest "
                    "measurements, how far the state sits from the calibrated region, "
                    "and the model's own error numbers."
                ),
                parameters={"type": "object", "properties": {}},
                fn=self.observe,
            ),
            Tool(
                name="imagine",
                description=(
                    "Roll the world model forward from the current state, assuming "
                    "the robot holds still, without moving anything. Reports where "
                    "the grounding gate fired, i.e. how many steps ahead the "
                    "prediction is still trustworthy."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "integer",
                            "description": (
                                f"how many steps to imagine (1..{self.max_imagine_steps})"
                            ),
                        }
                    },
                    "required": ["steps"],
                },
                fn=self.imagine,
            ),
            Tool(
                name="plan_to",
                description=(
                    f"Plan a motion toward a goal observation ({n} numbers) with "
                    "model-predictive control. Returns a plan id, or status "
                    '"refused" when every candidate left the calibrated region.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "array",
                            "description": f"target observation vector of length {n}",
                            "items": {"type": "number"},
                        }
                    },
                },
                fn=self.plan_to,
            ),
            Tool(
                name="check_safety",
                description=(
                    f"Score one proposed action ({action_dim} numbers) against the "
                    "grounding gate without executing it. Higher score = further "
                    "outside the calibrated region."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "array",
                            "description": f"action vector of length {action_dim}",
                            "items": {"type": "number"},
                        }
                    },
                    "required": ["action"],
                },
                fn=self.check_safety,
            ),
            Tool(
                name="execute",
                description=(
                    "MOVES THE ROBOT. Execute the first action of a plan produced by "
                    "plan_to. Requires operator confirmation and a feasible plan."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": "plan id from plan_to; omit for the most recent",
                        }
                    },
                },
                fn=self.execute,
                dangerous=True,
            ),
        ]

    # ---- tool bodies -------------------------------------------------------

    def observe(self) -> dict:
        """Small enough to sit in a prompt: the numbers a controller acts on."""
        z = self.state()
        reading = self.world.gate.read(z)
        summary = self.world.summary()
        return {
            "step": self.steps_taken,
            "observation": _round(self.history[-1]),
            "gate_score": round(float(reading.score), 3),
            "gate_reason": reading.reason,
            "grounded": bool(reading.fire),
            "model": {
                "state_dim": summary["state_dim"],
                "topo_dim": summary["topo_dim"],
                "rho": summary["rho"],
                "step_rmse": summary["step_rmse"],
                "window": summary["window"],
            },
        }

    def imagine(self, steps: int) -> dict:
        steps = int(steps)
        if steps < 1 or steps > self.max_imagine_steps:
            raise ValueError(f"steps must be between 1 and {self.max_imagine_steps}")
        # Zero action = "what happens if I hold still", which is both the
        # honest default and the same fallback control.py executes on a refusal.
        # An action-conditioned operator refuses to step without one at all.
        actions = None
        if self.world.config.action_dim:
            actions = np.zeros((steps, self.world.config.action_dim))
        rollout = self.world.imagine(self.window_array(), steps, actions=actions)
        grounded = rollout.grounded_at
        return {
            "steps_requested": steps,
            "steps_imagined": len(rollout),
            "trusted_steps": rollout.trusted_steps,
            # grounded_at is the whole answer to "how far ahead can I plan": past
            # it the operator is extrapolating and the gate says so.
            "grounded_at": grounded,
            "gate_reason": "ok" if grounded is None else rollout.readings[grounded].reason,
            "gate_scores": [round(float(r.score), 3) for r in rollout.readings],
            "predicted_observation": _round(rollout.hiddens[-1]) if len(rollout) else [],
            "worst_case_error_bound": round(float(rollout.certificate.bound), 4),
        }

    def plan_to(self, goal: Sequence[float] | None = None) -> dict:
        target = self.goal if goal is None else np.asarray(goal, dtype=np.float64).reshape(-1)
        if target is None:
            raise ValueError("no goal given and no default goal was configured")
        if target.shape[0] != self.world.hidden_dim:
            raise ValueError(
                f"goal must have {self.world.hidden_dim} numbers, got {target.shape[0]}"
            )

        plan = self.mpc.plan(self.window_array(), target_cost(self.world, target))
        if not plan.feasible:
            # A refusal, not an exception. The model has to be able to read the
            # reason and try something else; an exception would end the loop and
            # take the reason with it.
            return {
                "status": "refused",
                "refusal": "no_feasible_plan",
                "reason": (
                    f"every one of {plan.considered} candidate motions left the region "
                    f"the world model was calibrated on ({plan.rejected} rejected by the "
                    f"grounding gate at limit {self.mpc.gate_limit}). The model has no "
                    f"trustworthy plan to this goal and will not guess."
                ),
                "rejected": plan.rejected,
                "considered": plan.considered,
                "horizon": plan.horizon,
                "options": [
                    "call observe to see how far the current state is from calibration",
                    "call imagine to find how many steps ahead are still trusted",
                    "choose a nearer goal",
                ],
            }

        self._plan_count += 1
        plan_id = f"plan-{self._plan_count}"
        self.plans[plan_id] = plan
        return {
            "status": "ok",
            "plan": plan_id,
            "feasible": True,
            "cost": round(float(plan.cost), 4),
            "horizon": plan.horizon,
            "first_action": _round(plan.action),
            "rejected": plan.rejected,
            "considered": plan.considered,
            "max_gate_score": round(float(plan.max_gate_score), 3),
        }

    def check_safety(self, action: Sequence[float]) -> dict:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        width = int(self.world.config.action_dim)
        if a.shape[0] != width:
            raise ValueError(f"action must have {width} numbers, got {a.shape[0]}")
        z_next = self.world.operator.step(self.state(), a)
        if not np.all(np.isfinite(z_next)):
            return {"safe": False, "gate_score": float("inf"), "reason": "diverged"}
        reading = self.world.gate.read(z_next)
        # Judged against the planner's own limit, not the gate's 1.0, so
        # check_safety and plan_to cannot disagree about the same action.
        score = float(reading.score)
        return {
            "safe": bool(score < self.mpc.gate_limit),
            "gate_score": round(score, 3),
            "gate_limit": self.mpc.gate_limit,
            "reason": reading.reason,
            "sheaf_score": round(float(reading.sheaf_score), 3),
            "manifold_score": round(float(reading.manifold_score), 3),
        }

    def execute(self, plan: str | None = None) -> dict:
        plan_id = plan or (f"plan-{self._plan_count}" if self._plan_count else None)
        chosen = self.plans.get(plan_id) if plan_id else None
        if chosen is None:
            raise ValueError(
                f"no plan {plan_id!r}; call plan_to first. known: "
                f"{', '.join(self.plans) or '(none)'}"
            )
        if not chosen.feasible:
            # Confirmation authorises the *intent*. It is not evidence about the
            # rollout, so an infeasible plan stays refused however loudly it was
            # approved.
            return {
                "status": "refused",
                "refusal": "infeasible_plan",
                "reason": f"{plan_id} is not feasible; the world model refuses to execute it",
            }
        if self.step_env is None:
            raise RuntimeError("RobotAgent has no step_env, so it cannot move anything")

        action = np.asarray(chosen.action, dtype=np.float64).reshape(-1)
        observation = np.asarray(self.step_env(action), dtype=np.float64).reshape(-1)
        self.history.append(observation)
        self.steps_taken += 1
        return {
            "status": "executed",
            "plan": plan_id,
            "action": _round(action),
            "observation": _round(observation),
            "step": self.steps_taken,
        }


def _round(vector: np.ndarray, places: int = 3) -> list[float]:
    """Rounded list -- these go straight into a prompt, where digits cost tokens."""
    return [round(float(v), places) for v in np.asarray(vector).reshape(-1)]

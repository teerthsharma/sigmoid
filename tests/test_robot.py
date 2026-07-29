"""Runnable checks for the agent/world-model bridge. `python tests/test_robot.py`.

The plant is the 4-robot damped double integrator from examples/robot_control.py,
rebuilt here in a dozen lines so the tests do not import an example. No network,
no LLM: the provider is scripted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sigmoid
from sigmoid.agent import Tool, ToolCall
from sigmoid.control import Plan, TopologicalMPC
from sigmoid.robot import RobotAgent

N_ROBOTS = 4
DT, DAMPING, ACTION_LIMIT = 0.25, 0.85, 1.0
CONTACT_RADIUS = 0.9
_W = 3.0
STARTS = np.array([[-_W, -_W], [_W, -_W], [_W, _W], [-_W, _W]])
GOALS = np.roll(STARTS, -2, axis=0)  # diagonal swap: everyone crosses the centre


class Swarm:
    """Damped double integrator. Positions are the observation."""

    def __init__(self, positions: np.ndarray):
        self.pos = positions.astype(np.float64).copy()
        self.vel = np.zeros_like(self.pos)

    def observe(self) -> np.ndarray:
        return self.pos.reshape(-1).copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        acc = np.clip(
            np.asarray(action, dtype=np.float64).reshape(N_ROBOTS, 2),
            -ACTION_LIMIT,
            ACTION_LIMIT,
        )
        self.vel = DAMPING * self.vel + DT * acc
        self.pos = self.pos + DT * self.vel
        return self.observe()


class FakeProvider:
    """Scripted completions. The last response repeats so loops can run out."""

    def __init__(self, *responses: str):
        self.responses = list(responses) or [""]
        self.seen: list[list] = []

    def complete(self, messages, tools=None):
        self.seen.append(list(messages))
        index = min(len(self.seen) - 1, len(self.responses) - 1)
        return SimpleNamespace(text=self.responses[index])


def _call(name: str, args: str = "{}") -> str:
    return f'<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'


# ---- world (fitted once; the fit is the expensive part of this file) --------

_WORLD: sigmoid.SigmoidWorldModel | None = None


def swarm_world() -> sigmoid.SigmoidWorldModel:
    global _WORLD
    if _WORLD is not None:
        return _WORLD
    rng = np.random.default_rng(0)
    obs, acts = [], []
    for _ in range(12):
        swarm = Swarm(STARTS + rng.normal(scale=1.2, size=STARTS.shape))
        o, a = [], []
        drive = rng.normal(scale=0.6, size=(N_ROBOTS, 2))
        for _ in range(70):
            o.append(swarm.observe())
            drive = 0.85 * drive + 0.35 * rng.normal(scale=0.6, size=(N_ROBOTS, 2))
            action = np.clip(drive, -ACTION_LIMIT, ACTION_LIMIT).reshape(-1)
            a.append(action)
            swarm.step(action)
        obs.append(np.asarray(o))
        acts.append(np.asarray(a))
    config = sigmoid.SigmoidConfig(
        window=10,
        linear_dim=12,
        hilbert_degree=12,
        n_radii=4,
        entity_dim=2,
        # state the contact radius rather than let quantiles land elsewhere
        abs_radii=(0.6, CONTACT_RADIUS, 1.4, 2.5),
        n_abs_radii=0,
        action_dim=N_ROBOTS * 2,
    )
    _WORLD = sigmoid.SigmoidWorldModel(config=config).fit(obs, actions=acts)
    return _WORLD


def make_robot(*, gate_limit: float = 5.0, seed: int = 0, **kw) -> RobotAgent:
    world = swarm_world()
    rng = np.random.default_rng(seed)
    swarm = Swarm(STARTS + rng.normal(scale=0.3, size=STARTS.shape))
    window = [swarm.observe()]
    drive = np.zeros((N_ROBOTS, 2))
    for _ in range(world.config.window - 1):
        drive = 0.85 * drive + 0.2 * rng.normal(scale=0.4, size=(N_ROBOTS, 2))
        window.append(swarm.step(np.clip(drive, -1, 1).reshape(-1)))

    mpc = TopologicalMPC(
        world=world,
        action_low=np.full(N_ROBOTS * 2, -ACTION_LIMIT),
        action_high=np.full(N_ROBOTS * 2, ACTION_LIMIT),
        horizon=4,
        samples=48,
        elites=6,
        iterations=2,
        gate_limit=gate_limit,
        seed=seed,
    )
    kw.setdefault("provider", FakeProvider("done"))
    return RobotAgent(world, mpc, window=np.asarray(window), step_env=swarm.step, **kw)


# ---- tools -----------------------------------------------------------------


def test_tools_cover_the_control_surface():
    robot = make_robot()
    assert set(robot.tools.names()) == {
        "observe",
        "imagine",
        "plan_to",
        "check_safety",
        "execute",
    }
    assert robot.tools.get("execute").dangerous
    assert not any(t.dangerous for t in robot.tools.list() if t.name != "execute")


def test_observe_is_small_enough_for_a_prompt():
    """This payload goes into every turn's context, so it is budgeted, not dumped."""
    state = make_robot().observe()
    assert state["gate_reason"] and np.isfinite(state["gate_score"])
    assert len(state["observation"]) == N_ROBOTS * 2
    rendered = json.dumps(state)
    assert len(rendered) < 700, f"observe() rendered {len(rendered)} chars"


def test_imagine_reports_the_gate():
    out = make_robot().imagine(steps=12)
    assert out["steps_imagined"] <= 12
    assert out["trusted_steps"] <= out["steps_imagined"]
    assert len(out["predicted_observation"]) == N_ROBOTS * 2
    if out["grounded_at"] is not None:
        # imagination stops at the gate, and says which stalk fired
        assert out["steps_imagined"] == out["grounded_at"] + 1
        assert out["gate_reason"] != "ok"


def test_imagine_refuses_an_absurd_horizon():
    robot = make_robot()
    try:
        robot.imagine(steps=10_000)
    except ValueError as exc:
        assert "between 1 and" in str(exc)
    else:
        raise AssertionError("expected a refusal for an out-of-range horizon")


def test_plan_to_returns_a_feasible_plan():
    result = make_robot().plan_to(GOALS.reshape(-1).tolist())
    assert result["status"] == "ok", result
    assert result["plan"] == "plan-1"
    assert len(result["first_action"]) == N_ROBOTS * 2
    assert np.isfinite(result["cost"])
    assert 0 <= result["rejected"] < result["considered"]


def test_plan_to_refuses_instead_of_raising():
    """gate_limit nothing can satisfy: the refusal is a value, not an exception."""
    result = make_robot(gate_limit=1e-9).plan_to(GOALS.reshape(-1).tolist())
    assert result["status"] == "refused"
    assert result["refusal"] == "no_feasible_plan"
    assert result["rejected"] == result["considered"] > 0
    assert "calibrated" in result["reason"]
    assert result["options"], "a refusal must suggest what to do instead"


def test_plan_to_rejects_a_wrong_width_goal():
    robot = make_robot()
    result = robot.agent.invoke(ToolCall(name="plan_to", arguments={"goal": [1.0, 2.0]}))
    assert not result.ok and "goal must have" in result.error


def test_check_safety_scores_an_action():
    robot = make_robot()
    mild = robot.check_safety([0.0] * (N_ROBOTS * 2))
    wild = robot.check_safety([500.0] * (N_ROBOTS * 2))
    assert mild["safe"] and mild["gate_score"] < robot.mpc.gate_limit
    assert not wild["safe"], "an enormous action must not read as safe"
    assert wild["gate_score"] > mild["gate_score"]
    assert wild["reason"] != "ok"


# ---- the loop --------------------------------------------------------------


def test_agent_handles_a_refusal_and_keeps_going():
    """The whole point: an infeasible goal must be reasoned about, not crashed on."""
    goal = json.dumps(GOALS.reshape(-1).round(2).tolist())
    provider = FakeProvider(
        _call("plan_to", f'{{"goal": {goal}}}'),
        "The world model refuses: every candidate left the calibrated region. "
        "I will not move the robot.",
    )
    robot = make_robot(gate_limit=1e-9, provider=provider)
    result = robot.run("swap the robots diagonally")

    assert result.stop_reason == "final_answer"
    assert result.tools_called == ("plan_to",)
    tool_result = result.results()[0]
    # the tool succeeded -- it successfully reported that the model refuses
    assert tool_result.ok and tool_result.ran
    assert tool_result.value["status"] == "refused"
    assert "no_feasible_plan" in str(provider.seen[-1]), "the model must see the refusal"
    assert robot.steps_taken == 0


def test_agent_plans_then_is_blocked_from_executing():
    provider = FakeProvider(
        _call("plan_to", "{}"),
        _call("execute", "{}"),
        "I cannot move without confirmation.",
    )
    robot = make_robot(provider=provider, goal=GOALS.reshape(-1))
    result = robot.run("get to the goals")

    blocked = [r for r in result.results() if r.name == "execute"][0]
    assert blocked.blocked == "unconfirmed"
    assert not blocked.ran
    assert robot.steps_taken == 0, "the robot moved without confirmation"
    assert result.stop_reason == "final_answer"


def test_agent_executes_once_an_operator_confirms():
    provider = FakeProvider(_call("plan_to", "{}"), _call("execute", "{}"), "moved")
    robot = make_robot(
        provider=provider,
        goal=GOALS.reshape(-1),
        confirm=lambda tool, args: tool.name == "execute",
    )
    before = robot.window_array()[-1].copy()
    result = robot.run("get to the goals")

    executed = [r for r in result.results() if r.name == "execute"][0]
    assert executed.ok and executed.ran
    assert executed.value["status"] == "executed"
    assert robot.steps_taken == 1
    assert not np.allclose(before, robot.window_array()[-1]), "the plant did not move"


def test_execute_refuses_an_infeasible_plan_even_when_confirmed():
    """Confirmation authorises the intent; it is not evidence about the rollout."""
    robot = make_robot(confirm=lambda tool, args: True)
    robot.plans["plan-9"] = Plan(
        actions=np.zeros((1, N_ROBOTS * 2)),
        states=np.zeros((1, robot.world.state_dim)),
        cost=float("inf"),
        feasible=False,
        rejected=48,
        considered=48,
        horizon=1,
        max_gate_score=float("inf"),
    )
    result = robot.agent.invoke(ToolCall(name="execute", arguments={"plan": "plan-9"}))
    assert result.ok, "a refusal is a result, not a tool failure"
    assert result.value["refusal"] == "infeasible_plan"
    assert robot.steps_taken == 0


def test_execute_without_a_plan_is_an_error_not_a_crash():
    robot = make_robot(confirm=lambda tool, args: True)
    result = robot.agent.invoke(ToolCall(name="execute", arguments={}))
    assert not result.ok and "call plan_to first" in result.error
    assert robot.steps_taken == 0


def test_hook_veto_stops_execution():
    from sigmoid.agent import HookVeto

    class Guard:
        def emit(self, point, payload):
            tool = payload.get("tool")
            if isinstance(tool, Tool) and tool.dangerous:
                raise HookVeto("the operator has the robot on hold")

    provider = FakeProvider(_call("plan_to", "{}"), _call("execute", "{}"), "held")
    robot = make_robot(
        provider=provider,
        goal=GOALS.reshape(-1),
        hooks=Guard(),
        confirm=lambda tool, args: True,  # confirmed, and still vetoed
    )
    result = robot.run("go")
    blocked = [r for r in result.results() if r.name == "execute"][0]
    assert blocked.blocked == "veto"
    assert robot.steps_taken == 0


def test_system_prompt_tells_the_model_about_refusal():
    prompt = make_robot().agent.system_prompt()
    assert "refuse" in prompt and "plan_to" in prompt
    assert "<tools>" in prompt


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all green' if not failures else f'{failures} failing'}")
    sys.exit(1 if failures else 0)

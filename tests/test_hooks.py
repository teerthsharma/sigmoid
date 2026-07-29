"""Checks for the hook system. `python tests/test_hooks.py` or pytest.

Two behaviours carry the design and both are asserted here rather than assumed:

* **veto stops the action** -- ``test_veto_*``. A safety hook that could only
  log its objection would be decoration, so the refusal has to reach the
  caller as a distinct, non-retryable exception.
* **a broken hook does not stop the run** -- ``test_hook_failure_*``. A dead
  telemetry sink must not take an arm down mid-motion, and the hooks queued
  behind it must still run.

The asymmetry is the point: refusing on purpose stops the robot, failing by
accident does not.
"""

from __future__ import annotations

import logging
import sys
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmoid.hooks import HookManager, HookPoint, HookVeto, LoggingHook, TimingHook


def move_arm(hooks, speed=0.5):
    """A stand-in actuator call, gated the way a real one would be.

    Tests assert on ``fired`` -- whether the actuator was actually reached --
    not on whether a hook ran, because "the hook ran" is not the property that
    keeps a robot from hitting something.
    """
    ctx = hooks.emit(HookPoint.BEFORE_TOOL, {"tool": "arm.move", "speed": speed})
    fired.append(ctx["speed"])
    return hooks.emit(HookPoint.AFTER_TOOL, {"tool": "arm.move", "result": "ok"})


fired: list[float] = []


def setup():
    fired.clear()
    return HookManager()


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def test_hooks_run_in_priority_then_registration_order():
    hooks = setup()
    order = []
    hooks.register(HookPoint.BEFORE_STEP, lambda c: order.append("low"), priority=-10)
    hooks.register(HookPoint.BEFORE_STEP, lambda c: order.append("first"), priority=100)
    hooks.register(HookPoint.BEFORE_STEP, lambda c: order.append("mid_a"))
    hooks.register(HookPoint.BEFORE_STEP, lambda c: order.append("mid_b"))
    hooks.emit(HookPoint.BEFORE_STEP, {})
    assert order == ["first", "mid_a", "mid_b", "low"]


def test_emit_stamps_the_point_so_one_hook_can_serve_many():
    hooks = setup()
    seen = []
    for p in (HookPoint.BEFORE_TOOL, HookPoint.ON_TOKEN):
        hooks.register(p, lambda c: seen.append(c["point"]))
    hooks.emit(HookPoint.BEFORE_TOOL, {})
    hooks.emit(HookPoint.ON_TOKEN, {})
    assert seen == ["before_tool", "on_token"]
    # str enum, so a config file or JSON log can compare without the import.
    assert HookPoint.BEFORE_TOOL == "before_tool"


def test_emit_with_no_hooks_returns_the_context_unharmed():
    hooks = setup()
    assert hooks.emit(HookPoint.AFTER_STEP, {"a": 1})["a"] == 1
    assert hooks.emit(HookPoint.AFTER_STEP)["point"] == "after_step"


def test_the_decorator_registers():
    hooks = setup()

    @hooks.on(HookPoint.BEFORE_COMPLETE, priority=5)
    def tag(ctx):
        ctx["tagged"] = True

    assert hooks.emit(HookPoint.BEFORE_COMPLETE, {})["tagged"]
    assert hooks.hooks(HookPoint.BEFORE_COMPLETE) == [tag]  # decorator returns the fn


def test_unregister_removes_exactly_one_hook():
    hooks = setup()
    a = hooks.register(HookPoint.ON_TOKEN, lambda c: None)
    b = hooks.register(HookPoint.ON_TOKEN, lambda c: None)
    assert hooks.unregister(HookPoint.ON_TOKEN, a) is True
    assert hooks.hooks(HookPoint.ON_TOKEN) == [b]
    assert hooks.unregister(HookPoint.ON_TOKEN, a) is False
    assert hooks.unregister(HookPoint.ON_REFUSAL, b) is False


def test_a_hook_may_register_another_mid_emit():
    """A retry hook arming a one-shot must not corrupt the iteration."""
    hooks = setup()
    ran = []
    hooks.register(HookPoint.ON_ERROR, lambda c: ran.append("late"))
    hooks.register(
        HookPoint.BEFORE_STEP,
        lambda c: hooks.register(HookPoint.BEFORE_STEP, lambda c2: ran.append("added")),
    )
    hooks.emit(HookPoint.BEFORE_STEP, {})
    assert ran == []  # added during the walk, runs next time
    hooks.emit(HookPoint.BEFORE_STEP, {})
    assert ran == ["added"]


# --------------------------------------------------------------------------
# transform
# --------------------------------------------------------------------------


def test_a_hook_can_transform_the_payload():
    hooks = setup()

    @hooks.on(HookPoint.BEFORE_TOOL, priority=10)
    def clamp(ctx):
        # The everyday shape of a safety hook: allow the action, bound it.
        return {**ctx, "speed": min(ctx["speed"], 0.2)}

    move_arm(hooks, speed=1.5)
    assert fired == [0.2], "the clamped value must reach the actuator, not the original"


def test_transforms_chain_in_order():
    hooks = setup()
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: {**c, "speed": c["speed"] * 2}, priority=10)
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: {**c, "speed": c["speed"] + 1}, priority=5)
    assert hooks.emit(HookPoint.BEFORE_TOOL, {"speed": 3})["speed"] == 7


def test_returning_none_leaves_the_context_alone():
    """An observer hook must not have to return ctx to avoid erasing it."""
    hooks = setup()
    hooks.register(HookPoint.AFTER_COMPLETE, lambda c: None)
    hooks.register(HookPoint.AFTER_COMPLETE, lambda c: c.get("text"))  # truthy non-dict
    out = hooks.emit(HookPoint.AFTER_COMPLETE, {"text": "hi"})
    assert out["text"] == "hi"


# --------------------------------------------------------------------------
# veto -- the reason this system exists
# --------------------------------------------------------------------------


def test_veto_stops_the_action_before_the_actuator():
    hooks = setup()

    @hooks.on(HookPoint.BEFORE_TOOL, priority=100)
    def gate(ctx):
        if ctx["speed"] > 0.3:
            raise HookVeto(f"{ctx['speed']} m/s exceeds the cleared envelope", ctx)

    try:
        move_arm(hooks, speed=0.9)
    except HookVeto as veto:
        assert "cleared envelope" in veto.reason
        assert veto.ctx["tool"] == "arm.move"
    else:
        raise AssertionError("a veto that does not reach the caller is not a veto")
    assert fired == [], "the actuator ran despite the veto"

    move_arm(hooks, speed=0.1)
    assert fired == [0.1], "the gate must still pass safe actions"


def test_veto_short_circuits_the_remaining_hooks():
    """Nothing after the refusal runs -- a veto is a decision, not advice."""
    hooks = setup()
    ran = []
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: ran.append("before"), priority=10)
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: (_ for _ in ()).throw(HookVeto("no")))
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: ran.append("after"), priority=-10)
    with suppress(HookVeto):
        hooks.emit(HookPoint.BEFORE_TOOL, {})
    assert ran == ["before"]


def test_veto_is_not_an_error_and_is_not_routed_to_on_error():
    """Retry-on-error code must not retry a refusal into a loop."""
    hooks = setup()
    errors = []
    hooks.register(HookPoint.ON_ERROR, lambda c: errors.append(c))
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: (_ for _ in ()).throw(HookVeto("no")))
    with suppress(HookVeto):
        hooks.emit(HookPoint.BEFORE_TOOL, {})
    assert errors == []
    assert not isinstance(HookVeto("x"), RuntimeError)


# --------------------------------------------------------------------------
# failure isolation -- a broken hook must never stop a robot
# --------------------------------------------------------------------------


def test_hook_failure_does_not_stop_the_run():
    hooks = setup()
    hooks.register(
        HookPoint.BEFORE_TOOL,
        lambda c: (_ for _ in ()).throw(ConnectionError("telemetry sink is gone")),
        priority=50,
    )
    move_arm(hooks, speed=0.4)
    assert fired == [0.4], "a dead telemetry sink stopped the arm"


def test_hook_failure_does_not_stop_the_hooks_behind_it():
    hooks = setup()
    ran = []
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: ran.append("first"), priority=100)
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: 1 / 0, priority=50)
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: ran.append("safety_gate"), priority=10)
    hooks.emit(HookPoint.BEFORE_TOOL, {"tool": "arm.move"})
    assert ran == ["first", "safety_gate"], "a broken hook shadowed the safety gate behind it"


def test_hook_failure_is_routed_to_on_error_with_enough_to_debug_it():
    hooks = setup()
    caught = []
    hooks.register(HookPoint.ON_ERROR, lambda c: caught.append(c))

    def broken_telemetry(ctx):
        raise ConnectionError("sink is gone")

    hooks.register(HookPoint.BEFORE_TOOL, broken_telemetry)
    hooks.emit(HookPoint.BEFORE_TOOL, {"tool": "arm.move"})

    assert len(caught) == 1
    assert isinstance(caught[0]["error"], ConnectionError)
    assert caught[0]["hook"] == "broken_telemetry"
    assert caught[0]["failed_point"] == "before_tool"
    assert caught[0]["ctx"]["tool"] == "arm.move"


def test_a_broken_on_error_hook_does_not_recurse():
    """One bad error handler must be a log line, not an infinite loop."""
    hooks = setup()
    calls = []

    def broken_handler(ctx):
        calls.append(1)
        raise RuntimeError("the error handler is broken too")

    hooks.register(HookPoint.ON_ERROR, broken_handler)
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: 1 / 0)
    hooks.emit(HookPoint.BEFORE_TOOL, {})
    assert len(calls) == 1


def test_a_veto_from_inside_error_handling_is_contained():
    """The action already failed; there is nothing left to refuse, and letting
    it surface would make a failure look like a deliberate refusal."""
    hooks = setup()
    hooks.register(HookPoint.ON_ERROR, lambda c: (_ for _ in ()).throw(HookVeto("late")))
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: 1 / 0, priority=50)
    move_arm(hooks, speed=0.4)
    assert fired == [0.4]


def test_a_failing_hook_is_logged_with_its_name(caplog=None):
    hooks = setup()
    records = []

    class Sink(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("sigmoid.hooks")
    sink = Sink()
    logger.addHandler(sink)
    try:
        hooks.register(HookPoint.ON_TOKEN, lambda c: 1 / 0)
        hooks.emit(HookPoint.ON_TOKEN, {})
    finally:
        logger.removeHandler(sink)
    assert any("on_token" in m and "ZeroDivisionError" in m for m in records)


# --------------------------------------------------------------------------
# scoping
# --------------------------------------------------------------------------


def test_scope_reverts_everything_registered_inside_it():
    hooks = setup()
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: None)
    with hooks.scope():
        hooks.register(HookPoint.BEFORE_TOOL, lambda c: (_ for _ in ()).throw(HookVeto("teleop")))
        try:
            move_arm(hooks)
        except HookVeto:
            pass
        else:  # pragma: no cover
            raise AssertionError("the scoped gate did not fire")
    move_arm(hooks, speed=0.4)
    assert fired == [0.4], "the scoped gate outlived its scope"


def test_scope_reverts_even_when_the_block_raises():
    """The operator disconnecting mid-session must not leave a hook behind."""
    hooks = setup()
    try:
        with hooks.scope():
            hooks.register(HookPoint.BEFORE_TOOL, lambda c: None)
            raise ConnectionError("operator link dropped")
    except ConnectionError:
        pass  # noqa: S110 -- the point of the test is what survives the raise
    assert hooks.hooks(HookPoint.BEFORE_TOOL) == []


def test_clear():
    hooks = setup()
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: None)
    hooks.register(HookPoint.ON_TOKEN, lambda c: None)
    hooks.clear(HookPoint.BEFORE_TOOL)
    assert hooks.hooks(HookPoint.BEFORE_TOOL) == [] and hooks.hooks(HookPoint.ON_TOKEN)
    hooks.clear()
    assert hooks.hooks(HookPoint.ON_TOKEN) == []


# --------------------------------------------------------------------------
# built-ins
# --------------------------------------------------------------------------


def test_hook_points_cover_the_agent_loop():
    assert {p.value for p in HookPoint} == {
        "before_complete",
        "after_complete",
        "before_tool",
        "after_tool",
        "on_error",
        "before_step",
        "after_step",
        "on_token",
        "on_refusal",
    }


def test_logging_hook_logs_selected_fields_only():
    """A raw ctx holds whole prompts; a log line per token is a DoS, not a log."""
    hooks = setup()
    records = []

    class Sink(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("test.hooks.logging")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(Sink())
    LoggingHook(logger, level=logging.DEBUG).attach(hooks, HookPoint.BEFORE_COMPLETE)
    hooks.emit(HookPoint.BEFORE_COMPLETE, {"model": "gpt-4o-mini", "prompt": "SECRET" * 200})

    assert len(records) == 1
    assert "gpt-4o-mini" in records[0]
    assert "SECRET" not in records[0], "the logging hook dumped the whole prompt"


def test_logging_hook_attaches_to_every_point_by_default():
    hooks = setup()
    LoggingHook().attach(hooks)
    assert all(hooks.hooks(p) for p in HookPoint)


def test_timing_hook_measures_between_matching_points():
    hooks = setup()
    timer = TimingHook()
    timer.attach(hooks)

    for _ in range(3):
        hooks.emit(HookPoint.BEFORE_TOOL, {})
        sum(range(20000))  # something with nonzero duration
        hooks.emit(HookPoint.AFTER_TOOL, {})

    assert len(timer.timings["tool"]) == 3
    assert all(ms > 0.0 for ms in timer.timings["tool"])
    assert timer.mean_ms("tool") > 0.0
    assert timer.mean_ms("step") == 0.0  # never emitted, not an error


def test_timing_hook_survives_a_vetoed_action():
    """A veto means AFTER_ never fires. The next measurement must not inherit
    the abandoned start time and report a wildly inflated duration."""
    hooks = setup()
    timer = TimingHook()
    timer.attach(hooks)
    veto = lambda c: (_ for _ in ()).throw(HookVeto("no"))  # noqa: E731
    hooks.register(HookPoint.BEFORE_TOOL, veto, priority=-1)

    with suppress(HookVeto):
        hooks.emit(HookPoint.BEFORE_TOOL, {})
    hooks.emit(HookPoint.AFTER_TOOL, {})
    assert len(timer.timings.get("tool", [])) == 1  # the timer started before the veto

    hooks.emit(HookPoint.AFTER_TOOL, {})  # AFTER with no BEFORE
    assert len(timer.timings["tool"]) == 1, "an unpaired AFTER invented a measurement"


def test_timing_and_logging_coexist_without_interfering():
    hooks = setup()
    timer = TimingHook()
    timer.attach(hooks)
    LoggingHook().attach(hooks)
    hooks.register(HookPoint.BEFORE_TOOL, lambda c: 1 / 0)  # and one broken hook
    hooks.emit(HookPoint.BEFORE_TOOL, {"tool": "arm.move"})
    hooks.emit(HookPoint.AFTER_TOOL, {"tool": "arm.move"})
    assert len(timer.timings["tool"]) == 1


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

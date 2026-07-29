"""Runnable checks for the agentic loop. `python tests/test_agent.py` or pytest.

Everything here runs against a scripted FakeProvider: no network, no weights.
The loop's job is to survive whatever a local model emits, so most of these are
malformed-input tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmoid.agent import (
    Agent,
    HermesToolParser,
    HookVeto,
    Tool,
    ToolCall,
    ToolRegistry,
    hermes_system_prompt,
)


class FakeProvider:
    """Returns scripted text. The last response repeats, so loops can run out."""

    def __init__(self, *responses: str):
        self.responses = list(responses) or [""]
        self.seen: list[list] = []

    def complete(self, messages, tools=None):
        self.seen.append(list(messages))
        index = min(len(self.seen) - 1, len(self.responses) - 1)
        return SimpleNamespace(text=self.responses[index])


class RecordingHooks:
    """Minimal stand-in for HookManager: records points, optionally vetoes."""

    def __init__(self, veto: str | None = None, explode: bool = False):
        self.points: list[str] = []
        self.veto = veto
        self.explode = explode

    def emit(self, point, payload):
        self.points.append(str(point))
        if self.explode:
            raise RuntimeError("a logging hook with a typo in it")
        tool = payload.get("tool")
        if self.veto and getattr(tool, "name", None) == self.veto:
            raise HookVeto(f"{self.veto} is not allowed here")

    def saw(self, name: str) -> bool:
        return any(name in p for p in self.points)


def _echo_tool(calls: list | None = None) -> Tool:
    def echo(text: str) -> str:
        if calls is not None:
            calls.append(text)
        return text.upper()

    return Tool(
        name="echo",
        description="echo text back in upper case",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=echo,
    )


def _call(name: str, args: str) -> str:
    return f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'


# ---- Hermes parsing --------------------------------------------------------


def test_parses_a_well_formed_call():
    parsed = HermesToolParser().parse(_call("echo", '{"text": "hi"}'))
    assert len(parsed.calls) == 1
    assert parsed.calls[0].name == "echo"
    assert parsed.calls[0].arguments == {"text": "hi"}
    assert parsed.calls[0].error is None


def test_parses_multiple_calls_in_one_message():
    text = _call("echo", '{"text": "a"}') + "\n" + _call("echo", '{"text": "b"}')
    parsed = HermesToolParser().parse(text)
    assert [c.arguments["text"] for c in parsed.calls] == ["a", "b"]
    assert not parsed.malformed


def test_keeps_text_interleaved_with_calls():
    text = (
        "Let me check the state.\n"
        + _call("echo", '{"text": "a"}')
        + "\nand then the other one\n"
        + _call("echo", '{"text": "b"}')
        + "\nDone."
    )
    parsed = HermesToolParser().parse(text)
    assert len(parsed.calls) == 2
    assert "Let me check the state." in parsed.text
    assert "Done." in parsed.text
    assert "<tool_call>" not in parsed.text


def test_malformed_json_is_a_result_not_an_exception():
    """A trailing comma is the single most common thing a local model emits."""
    parsed = HermesToolParser().parse('<tool_call>{"name": "echo", "arguments": {,}}</tool_call>')
    assert len(parsed.calls) == 1
    assert parsed.calls[0].error and "invalid JSON" in parsed.calls[0].error


def test_unterminated_tag_is_reported_and_stripped():
    """Generation cut off by the token budget leaves an open tag and half a call."""
    parsed = HermesToolParser().parse('thinking...\n<tool_call>{"name": "ec')
    assert len(parsed.calls) == 1
    assert "unterminated" in parsed.calls[0].error
    assert "<tool_call>" not in parsed.text
    assert parsed.text.strip() == "thinking..."


def test_call_missing_a_name_is_reported():
    parsed = HermesToolParser().parse('<tool_call>{"arguments": {"text": "x"}}</tool_call>')
    assert parsed.calls[0].error and "name" in parsed.calls[0].error


def test_non_object_body_is_reported():
    parsed = HermesToolParser().parse("<tool_call>[1, 2, 3]</tool_call>")
    assert parsed.calls[0].error and "JSON object" in parsed.calls[0].error


def test_double_encoded_arguments_are_decoded():
    """Some finetunes emit arguments as a JSON *string*; that is still usable."""
    parsed = HermesToolParser().parse(
        '<tool_call>{"name": "echo", "arguments": "{\\"text\\": \\"hi\\"}"}</tool_call>'
    )
    assert parsed.calls[0].error is None
    assert parsed.calls[0].arguments == {"text": "hi"}


def test_plain_text_is_a_final_answer():
    parsed = HermesToolParser().parse("The robots are at their goals.")
    assert not parsed.calls
    assert parsed.text == "The robots are at their goals."


def test_system_prompt_declares_the_schemas():
    registry = ToolRegistry([_echo_tool()])
    prompt = hermes_system_prompt(registry.to_specs())
    assert "<tools>" in prompt and "</tools>" in prompt
    assert '"echo"' in prompt
    assert "<tool_call>" in prompt


# ---- registry --------------------------------------------------------------


def test_registry_rejects_missing_required_arguments():
    registry = ToolRegistry([_echo_tool()])
    assert "missing required" in registry.validate(registry.get("echo"), {})


def test_registry_rejects_wrong_types_and_unknown_keys():
    registry = ToolRegistry([_echo_tool()])
    tool = registry.get("echo")
    assert "must be a JSON string" in registry.validate(tool, {"text": 7})
    assert "unknown argument" in registry.validate(tool, {"text": "x", "speed": 9})
    assert registry.validate(tool, {"text": "x"}) is None


def test_registry_does_not_let_a_bool_pass_as_an_integer():
    """True satisfies isinstance(int) in Python but not in JSON Schema."""
    tool = Tool(
        name="imagine",
        description="",
        parameters={"type": "object", "properties": {"steps": {"type": "integer"}}},
        fn=lambda steps: steps,
    )
    registry = ToolRegistry([tool])
    assert "must be a JSON integer" in registry.validate(tool, {"steps": True})
    assert registry.validate(tool, {"steps": 3}) is None


def test_duplicate_registration_is_refused():
    registry = ToolRegistry([_echo_tool()])
    try:
        registry.register(_echo_tool())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected a refusal for a duplicate tool name")


# ---- the loop --------------------------------------------------------------


def test_loop_calls_a_tool_then_finishes():
    calls: list[str] = []
    provider = FakeProvider(_call("echo", '{"text": "hi"}'), "all done")
    agent = Agent(provider, tools=[_echo_tool(calls)], max_steps=4)
    result = agent.run("say hi")
    assert calls == ["hi"]
    assert result.stop_reason == "final_answer"
    assert result.final == "all done"
    assert result.tools_called == ("echo",)
    assert result.steps_used == 2
    assert result.seconds >= 0.0


def test_tool_results_are_fed_back_as_tool_responses():
    provider = FakeProvider(_call("echo", '{"text": "hi"}'), "done")
    Agent(provider, tools=[_echo_tool()], max_steps=4).run("say hi")
    # the second provider call must see the result of the first tool call
    rendered = str(provider.seen[-1])
    assert "<tool_response>" in rendered and "HI" in rendered


def test_loop_terminates_on_max_steps():
    """A model that never stops calling tools must still return."""
    provider = FakeProvider(_call("echo", '{"text": "again"}'))
    result = Agent(provider, tools=[_echo_tool()], max_steps=3).run("loop forever")
    assert result.stop_reason == "max_steps"
    assert result.steps_used == 3
    assert len(result.tools_called) == 3


def test_a_throwing_tool_does_not_kill_the_run():
    """The failure being avoided: a robot that stops because one tool raised."""

    def boom() -> None:
        raise ValueError("actuator not responding")

    provider = FakeProvider(_call("boom", "{}"), "recovered")
    tool = Tool(name="boom", description="", fn=boom)
    result = Agent(provider, tools=[tool], max_steps=4).run("try it")
    assert result.stop_reason == "final_answer"
    assert result.final == "recovered"
    failed = result.results()[0]
    assert failed.ran and not failed.ok
    assert "actuator not responding" in failed.error
    assert "actuator not responding" in str(provider.seen[-1])


def test_malformed_call_is_reported_to_the_model_and_the_loop_continues():
    provider = FakeProvider("<tool_call>{not json}</tool_call>", "fixed it")
    result = Agent(provider, tools=[_echo_tool()], max_steps=4).run("go")
    assert result.stop_reason == "final_answer"
    assert not result.results()[0].ok
    assert result.tools_called == ()  # nothing ran
    assert "invalid JSON" in str(provider.seen[-1])


def test_unknown_tool_is_reported_not_raised():
    provider = FakeProvider(_call("teleport", "{}"), "ok")
    result = Agent(provider, tools=[_echo_tool()], max_steps=4).run("go")
    assert "unknown tool" in result.results()[0].error
    assert "echo" in result.results()[0].error  # tell it what does exist


def test_invalid_arguments_never_reach_the_function():
    seen: list[str] = []
    provider = FakeProvider(_call("echo", '{"text": 42}'), "ok")
    result = Agent(provider, tools=[_echo_tool(seen)], max_steps=4).run("go")
    assert seen == []
    assert not result.results()[0].ran
    assert "must be a JSON string" in result.results()[0].error


def test_provider_failure_ends_the_run_cleanly():
    class DeadProvider:
        def complete(self, messages, tools=None):
            raise ConnectionError("no local model loaded")

    result = Agent(DeadProvider(), tools=[_echo_tool()], max_steps=3).run("go")
    assert result.stop_reason == "provider_error"
    assert "no local model loaded" in result.final


# ---- hooks and danger ------------------------------------------------------


def test_hook_veto_blocks_the_call():
    calls: list[str] = []
    hooks = RecordingHooks(veto="echo")
    provider = FakeProvider(_call("echo", '{"text": "hi"}'), "understood")
    result = Agent(provider, tools=[_echo_tool(calls)], hooks=hooks, max_steps=4).run("go")
    assert calls == [], "a vetoed tool must not execute"
    blocked = result.results()[0]
    assert blocked.blocked == "veto" and not blocked.ok and not blocked.ran
    assert "not allowed here" in blocked.error
    assert hooks.saw("BEFORE_TOOL")


def test_all_four_hook_points_fire():
    hooks = RecordingHooks()
    provider = FakeProvider(_call("echo", '{"text": "hi"}'), "done")
    Agent(provider, tools=[_echo_tool()], hooks=hooks, max_steps=4).run("go")
    for point in ("BEFORE_STEP", "BEFORE_TOOL", "AFTER_TOOL", "AFTER_STEP"):
        assert hooks.saw(point), f"{point} never fired"


def test_a_broken_hook_does_not_stop_the_robot():
    calls: list[str] = []
    provider = FakeProvider(_call("echo", '{"text": "hi"}'), "done")
    agent = Agent(provider, tools=[_echo_tool(calls)], hooks=RecordingHooks(explode=True))
    result = agent.run("go")
    assert calls == ["hi"]
    assert result.stop_reason == "final_answer"


def test_dangerous_tool_does_not_run_unconfirmed():
    moved: list[int] = []
    tool = Tool(
        name="drive",
        description="moves the robot",
        fn=lambda: moved.append(1),
        dangerous=True,
    )
    provider = FakeProvider(_call("drive", "{}"), "understood")
    result = Agent(provider, tools=[tool], max_steps=4).run("drive")
    assert moved == [], "a dangerous tool ran without confirmation"
    assert result.results()[0].blocked == "unconfirmed"
    assert not result.results()[0].ran


def test_dangerous_tool_runs_when_an_operator_confirms():
    moved: list[int] = []
    tool = Tool(name="drive", description="", fn=lambda: moved.append(1), dangerous=True)
    provider = FakeProvider(_call("drive", "{}"), "done")
    agent = Agent(provider, tools=[tool], max_steps=4, confirm=lambda t, a: True)
    result = agent.run("drive")
    assert moved == [1]
    assert result.results()[0].ran and result.results()[0].ok


def test_a_hook_can_confirm_by_setting_the_payload():
    moved: list[int] = []

    class Approver:
        def emit(self, point, payload):
            if payload.get("dangerous"):
                payload["confirmed"] = True

    tool = Tool(name="drive", description="", fn=lambda: moved.append(1), dangerous=True)
    provider = FakeProvider(_call("drive", "{}"), "done")
    Agent(provider, tools=[tool], hooks=Approver(), max_steps=4).run("drive")
    assert moved == [1]


def test_a_hook_can_rewrite_arguments_and_the_rewrite_is_revalidated():
    seen: list[str] = []

    class Clamp:
        def emit(self, point, payload):
            if payload.get("arguments", {}).get("text") == "LOUD":
                return {**payload, "arguments": {"text": "quiet"}}
            return None

    provider = FakeProvider(_call("echo", '{"text": "LOUD"}'), "done")
    Agent(provider, tools=[_echo_tool(seen)], hooks=Clamp(), max_steps=3).run("go")
    assert seen == ["quiet"]

    class Wrecker:
        def emit(self, point, payload):
            if "arguments" in payload:
                return {**payload, "arguments": {"text": 0}}
            return None

    provider = FakeProvider(_call("echo", '{"text": "hi"}'), "done")
    result = Agent(provider, tools=[_echo_tool()], hooks=Wrecker(), max_steps=3).run("go")
    assert "a hook rewrote the arguments" in result.results()[0].error


def test_safe_tools_are_untouched_by_the_danger_gate():
    calls: list[str] = []
    provider = FakeProvider(_call("echo", '{"text": "hi"}'), "done")
    Agent(provider, tools=[_echo_tool(calls)], max_steps=4).run("go")
    assert calls == ["hi"]


# ---- native tool-call fields ----------------------------------------------


def test_the_real_hook_layer_vetoes_and_confirms():
    """Integration with sigmoid.hooks, which is a concurrent sibling.

    The loop is written against a duck-typed manager so it runs without that
    module; this check exists because the two ways it could silently disagree
    -- a veto class the loop does not catch, and a ctx transform the loop
    discards -- both look like working code until a robot moves.
    """
    try:
        from sigmoid.hooks import HookManager, HookPoint
        from sigmoid.hooks import HookVeto as RealVeto
    except ImportError:
        return  # sibling not landed yet

    moved: list[int] = []
    tool = Tool(name="drive", description="", fn=lambda: moved.append(1), dangerous=True)
    manager = HookManager()
    manager.register(HookPoint.BEFORE_TOOL, lambda ctx: {**ctx, "confirmed": True})
    provider = FakeProvider(_call("drive", "{}"), "done")
    Agent(provider, tools=[tool], hooks=manager, max_steps=3).run("drive")
    assert moved == [1], "a ctx transform from a real hook was dropped"

    manager.clear()

    def refuse(ctx):
        raise RealVeto("not today")

    manager.register(HookPoint.BEFORE_TOOL, refuse)
    provider = FakeProvider(_call("drive", "{}"), "done")
    result = Agent(provider, tools=[tool], hooks=manager, max_steps=3).run("drive")
    assert moved == [1], "the real HookVeto did not block the call"
    assert result.results()[0].blocked == "veto"


def test_native_tool_calls_are_used_when_the_provider_supplies_them():
    """Hosted APIs return calls in a field; the parser is not needed there."""
    calls: list[str] = []

    class NativeProvider:
        def __init__(self):
            self.n = 0

        def complete(self, messages, tools=None):
            self.n += 1
            if self.n == 1:
                return SimpleNamespace(
                    text="",
                    tool_calls=[{"name": "echo", "arguments": '{"text": "hi"}'}],
                )
            return SimpleNamespace(text="done", tool_calls=None)

    result = Agent(NativeProvider(), tools=[_echo_tool(calls)], max_steps=4).run("go")
    assert calls == ["hi"]
    assert result.tools_called == ("echo",)


def test_a_native_call_id_is_echoed_back_on_the_result():
    """Hosted APIs reject a tool result that does not name the call it answers."""

    class NativeProvider:
        def __init__(self):
            self.n = 0
            self.seen: list[list] = []

        def complete(self, messages, tools=None):
            self.seen.append(list(messages))
            self.n += 1
            if self.n == 1:
                return SimpleNamespace(
                    text="",
                    tool_calls=[
                        SimpleNamespace(id="call_1", name="echo", arguments={"text": "hi"})
                    ],
                )
            return SimpleNamespace(text="done", tool_calls=[])

    provider = NativeProvider()
    Agent(provider, tools=[_echo_tool()], max_steps=4).run("go")
    assert "call_1" in str(provider.seen[-1])


def test_invoke_is_usable_without_a_provider():
    """Direct invocation is how robot tools get tested without an LLM at all."""
    calls: list[str] = []
    agent = Agent(provider=None, tools=[_echo_tool(calls)])
    result = agent.invoke(ToolCall(name="echo", arguments={"text": "hi"}))
    assert result.ok and result.value == "HI" and calls == ["hi"]


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

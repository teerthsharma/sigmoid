"""The agentic loop: a model that calls tools, and a loop that survives them.

    provider --text--> parser --calls--> registry --results--> provider ...

Two decisions here are not the obvious ones.

**Tool calls are parsed out of text, not read off a field.** Native tool-call
fields (`message.tool_calls`) exist only on hosted APIs. A local Hermes
checkpoint running through `sigmoid.inference` emits the call *inside its
generated text*, as `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`,
and expects results back as `<tool_response>{...}</tool_response>` with the
schemas declared in the system prompt inside `<tools>`. Both paths are
supported and the native one wins when present, but the text path is the one
that has to be robust, because a local model has no server-side validator in
front of it: it will emit trailing commas, single quotes, a call cut in half by
the token budget, and three calls in one message with prose between them. Every
one of those is a parse *result*, not an exception -- the malformed call comes
back to the model as a tool error so it can correct itself, which is the only
repair mechanism a local model has.

**A tool that raises does not end the run.** This loop is meant to sit on a
robot. A world-model tool that hits an uncalibrated radius, an environment that
rejects an action, a hook that misbehaves -- each becomes a tool result the
model can read and route around. The failure mode being avoided is a robot that
stops mid-task because one tool threw a ValueError.

Danger is gated separately from correctness. `Tool(dangerous=True)` marks an
action that moves hardware; those do not execute unless something outside the
model confirms them (`confirm=`, or a hook that sets `payload["confirmed"]`).
The model can ask; it cannot authorise.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

__all__ = [
    "Agent",
    "AgentResult",
    "HermesToolParser",
    "HookVeto",
    "ParsedMessage",
    "Step",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "hermes_system_prompt",
]


class HookVeto(Exception):
    """Raised by a hook to block the action it was fired for.

    Defined here as well as in `sigmoid.hooks` so the loop runs before that
    module exists; `_veto_types()` catches both, so a veto from the real hook
    layer and one from a test double are indistinguishable to the loop.
    """


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


@dataclass
class Tool:
    """One callable the model may invoke, with the schema it is described by."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    """JSON Schema for the arguments object. Sent to the model verbatim."""

    fn: Callable[..., Any] | None = None
    dangerous: bool = False
    """True for anything that moves a robot. Requires confirmation to execute."""

    def spec(self) -> dict[str, Any]:
        """OpenAI-shaped function spec -- also what Hermes wants inside <tools>."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


class ToolRegistry:
    """Named tools plus argument validation."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def to_specs(self) -> list[Any]:
        """Tool specs for the provider layer, as `ToolSpec` when it exists."""
        specs = [t.spec() for t in self._tools.values()]
        try:
            from .providers import ToolSpec  # concurrent sibling; may not exist
        except ImportError:
            return specs
        try:
            return [
                ToolSpec(name=t.name, description=t.description, parameters=t.parameters)
                for t in self._tools.values()
            ]
        except TypeError:  # a different ToolSpec signature: plain dicts still work
            return specs

    # ---- validation --------------------------------------------------------

    def validate(self, tool: Tool, arguments: Any) -> str | None:
        """Return an error string, or None when `arguments` fits the schema.

        Deliberately strict about *unknown* keys. A hallucinated argument that
        is silently dropped turns into a tool call that looks like it honoured
        a constraint it never saw -- on a robot that is the difference between
        "move 0.1" and "move 0.1 but ignore the speed limit you asked for".
        Unknown keys come back as an error the model can fix.

        A subset of JSON Schema on purpose: types, required, enum. Adding a
        validator dependency for `minimum`/`pattern` is not worth it while
        every tool in this package hand-writes its own bounds anyway.
        """
        if not isinstance(arguments, dict):
            return f"arguments must be a JSON object, got {type(arguments).__name__}"
        schema = tool.parameters or {}
        properties: dict[str, Any] = schema.get("properties", {}) or {}
        missing = [k for k in schema.get("required", []) if k not in arguments]
        if missing:
            return f"missing required argument(s): {', '.join(sorted(missing))}"
        if properties and schema.get("additionalProperties", False) is not True:
            unknown = [k for k in arguments if k not in properties]
            if unknown:
                return (
                    f"unknown argument(s): {', '.join(sorted(unknown))}. "
                    f"expected only: {', '.join(properties) or '(none)'}"
                )
        for key, value in arguments.items():
            spec = properties.get(key)
            if not isinstance(spec, dict):
                continue
            error = _check_type(key, value, spec)
            if error:
                return error
        return None


def _check_type(key: str, value: Any, spec: dict[str, Any]) -> str | None:
    expected = spec.get("type")
    allowed = _JSON_TYPES.get(expected) if isinstance(expected, str) else None
    if allowed is not None:
        # bool is a subclass of int in Python but not in JSON Schema; letting
        # True satisfy "integer" would let `steps=True` reach a range loop.
        ok = isinstance(value, allowed) and not (
            isinstance(value, bool) and expected in ("integer", "number")
        )
        if not ok:
            return f"argument {key!r} must be a JSON {expected}, got {type(value).__name__}"
    choices = spec.get("enum")
    if choices and value not in choices:
        return f"argument {key!r} must be one of {choices}, got {value!r}"
    return None


# --------------------------------------------------------------------------
# Hermes tool-call format
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One parsed call. `error` set means the model emitted something unusable."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    error: str | None = None
    id: str = ""
    """Only hosted APIs issue call ids. Hermes text has none, so this stays "" there."""


@dataclass(frozen=True)
class ParsedMessage:
    """What the model said, split into prose and calls."""

    text: str
    calls: tuple[ToolCall, ...] = ()

    @property
    def malformed(self) -> tuple[ToolCall, ...]:
        return tuple(c for c in self.calls if c.error)


_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"


class HermesToolParser:
    """Parses `<tool_call>` blocks out of generated text; never raises.

    Every branch below corresponds to something a local checkpoint actually
    does: JSON with a trailing comma, `arguments` double-encoded as a string,
    a call truncated by the token budget so the closing tag never arrives,
    several calls in one message, and prose wrapped around all of it.
    """

    open_tag = _OPEN_TAG
    close_tag = _CLOSE_TAG

    def parse(self, text: str) -> ParsedMessage:
        text = "" if text is None else str(text)
        calls = [_parse_block(m.group(1)) for m in _CALL_RE.finditer(text)]
        remainder = _CALL_RE.sub("\n", text)

        # An unterminated tag means generation stopped mid-call (token budget,
        # stop string, crash). The tail is broken JSON, not prose: cut it, and
        # tell the model the call never closed so the retry is a whole call.
        stray = remainder.find(_OPEN_TAG)
        if stray != -1:
            truncated = remainder[stray + len(_OPEN_TAG) :].strip()
            remainder = remainder[:stray]
            calls.append(
                ToolCall(
                    name="",
                    raw=truncated,
                    error=(
                        "unterminated <tool_call>: the closing </tool_call> tag never "
                        "arrived. Re-emit the whole call between both tags."
                    ),
                )
            )
        remainder = remainder.replace(_CLOSE_TAG, "")
        return ParsedMessage(text=remainder.strip(), calls=tuple(calls))


def _parse_block(raw: str) -> ToolCall:
    raw = raw.strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return ToolCall(
            name="",
            raw=raw,
            error=(
                f"invalid JSON in <tool_call>: {exc}. "
                'Expected {"name": ..., "arguments": {...}}'
            ),
        )
    if not isinstance(payload, dict):
        return ToolCall(name="", raw=raw, error="a <tool_call> body must be a JSON object")

    name = payload.get("name") or payload.get("function") or payload.get("tool")
    if not isinstance(name, str) or not name:
        return ToolCall(name="", raw=raw, error='<tool_call> is missing a string "name"')

    args = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(args, str):
        # Some finetunes double-encode: "arguments": "{\"steps\": 4}". Decoding
        # it here costs two lines and turns a hard failure into a normal call.
        try:
            args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, ValueError) as exc:
            return ToolCall(name=name, raw=raw, error=f"could not decode arguments string: {exc}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ToolCall(name=name, raw=raw, error='"arguments" must be a JSON object')
    return ToolCall(name=name, arguments=args, raw=raw)


def hermes_system_prompt(specs: Iterable[Any], preamble: str | None = None) -> str:
    """The Hermes system prompt: instructions plus the schemas inside <tools>."""
    lines = []
    for spec in specs:
        if isinstance(spec, dict):
            lines.append(json.dumps(spec, default=str))
        else:  # a providers.ToolSpec, or anything else with the same three fields
            lines.append(
                json.dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(spec, "name", ""),
                            "description": getattr(spec, "description", ""),
                            "parameters": getattr(spec, "parameters", {}),
                        },
                    },
                    default=str,
                )
            )
    head = preamble or (
        "You are a function calling AI model. You are provided with function "
        "signatures within <tools></tools> XML tags. You may call one or more "
        "functions to assist with the user query. Do not make assumptions about "
        "what values to plug into functions."
    )
    return (
        f"{head}\n<tools>\n" + "\n".join(lines) + "\n</tools>\n"
        "For each function call return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>\n'
        "Results come back inside <tool_response></tool_response> tags. When you have "
        "the final answer, reply in plain text with no <tool_call> tags."
    )


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, in the form the model will read it."""

    name: str
    ok: bool
    value: Any = None
    error: str | None = None
    ran: bool = False
    """False when the call never reached `fn`: bad name, bad args, veto, unconfirmed."""

    blocked: str | None = None
    """"veto" or "unconfirmed" when a safety rule stopped the call."""

    seconds: float = 0.0

    def as_response(self) -> str:
        body = {"name": self.name, "content": self.value if self.ok else {"error": self.error}}
        return f"<tool_response>\n{json.dumps(body, default=str)}\n</tool_response>"


@dataclass(frozen=True)
class Step:
    """One provider call and everything it caused."""

    index: int
    text: str
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()
    seconds: float = 0.0
    provider_seconds: float = 0.0


@dataclass(frozen=True)
class AgentResult:
    final: str
    steps: tuple[Step, ...]
    tools_called: tuple[str, ...]
    stop_reason: str
    """"final_answer", "max_steps", or "provider_error"."""

    seconds: float
    messages: tuple[Any, ...] = ()

    @property
    def steps_used(self) -> int:
        return len(self.steps)

    def results(self) -> tuple[ToolResult, ...]:
        return tuple(r for s in self.steps for r in s.results)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


class Agent:
    """Observe / think / act until the model stops calling tools."""

    def __init__(
        self,
        provider: Any,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        hooks: Any = None,
        max_steps: int = 8,
        system: str | None = None,
        *,
        parser: HermesToolParser | None = None,
        confirm: Callable[[Tool, dict], bool] | None = None,
        require_confirmation: bool = True,
    ) -> None:
        self.provider = provider
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or ())
        self.hooks = hooks
        self.max_steps = int(max_steps)
        self.system = system
        self.parser = parser or HermesToolParser()
        self.confirm = confirm
        self.require_confirmation = bool(require_confirmation)

    # ---- prompt ------------------------------------------------------------

    def system_prompt(self) -> str:
        return hermes_system_prompt(self.tools.to_specs(), preamble=self.system)

    # ---- run ---------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        started = time.perf_counter()
        specs = self.tools.to_specs()
        messages: list[Any] = [
            _message("system", self.system_prompt()),
            _message("user", task),
        ]
        steps: list[Step] = []
        called: list[str] = []
        stop_reason = "max_steps"
        final = ""

        for index in range(self.max_steps):
            step_started = time.perf_counter()
            self._emit("BEFORE_STEP", {"index": index, "messages": messages})

            call_started = time.perf_counter()
            self._emit("BEFORE_COMPLETE", {"index": index, "messages": messages})
            try:
                completion = self.provider.complete(messages, tools=specs)
            except Exception as exc:  # noqa: BLE001 - a dead provider ends the run, cleanly
                self._emit("ON_ERROR", {"index": index, "error": exc, "where": "provider"})
                final = f"provider failed: {type(exc).__name__}: {exc}"
                stop_reason = "provider_error"
                steps.append(
                    Step(index=index, text=final, seconds=time.perf_counter() - step_started)
                )
                break
            provider_seconds = time.perf_counter() - call_started
            self._emit(
                "AFTER_COMPLETE",
                {"index": index, "completion": completion, "seconds": provider_seconds},
            )

            text = _completion_text(completion)
            parsed = self.parser.parse(text)
            calls = _native_calls(completion) or parsed.calls
            messages.append(_message("assistant", text))

            if not calls:
                final = parsed.text or text
                stop_reason = "final_answer"
                steps.append(
                    Step(
                        index=index,
                        text=parsed.text,
                        seconds=time.perf_counter() - step_started,
                        provider_seconds=provider_seconds,
                    )
                )
                self._emit("AFTER_STEP", {"index": index, "final": final})
                break

            results = tuple(self.invoke(call) for call in calls)
            called.extend(r.name for r in results if r.ran)
            if any(c.id for c in calls):
                # A hosted API rejects a tool result that does not name the call
                # it answers, and one message can carry one id -- so ids force
                # one message per result. Hermes has no ids, so that path keeps
                # the single combined message its template expects.
                for call, result in zip(calls, results):
                    messages.append(_message("tool", result.as_response(), call.id))
            else:
                messages.append(_message("tool", "\n".join(r.as_response() for r in results)))
            step = Step(
                index=index,
                text=parsed.text,
                calls=tuple(calls),
                results=results,
                seconds=time.perf_counter() - step_started,
                provider_seconds=provider_seconds,
            )
            steps.append(step)
            self._emit("AFTER_STEP", {"index": index, "step": step})

        return AgentResult(
            final=final,
            steps=tuple(steps),
            tools_called=tuple(called),
            stop_reason=stop_reason,
            seconds=time.perf_counter() - started,
            messages=tuple(messages),
        )

    # ---- one call ----------------------------------------------------------

    def invoke(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Returns a result for every failure mode; raises none."""
        started = time.perf_counter()

        def done(**kw: Any) -> ToolResult:
            return ToolResult(name=call.name, seconds=time.perf_counter() - started, **kw)

        if call.error:  # the parser could not read it; hand the reason back verbatim
            self._emit("ON_ERROR", {"call": call, "where": "parse"})
            return done(ok=False, error=call.error)

        tool = self.tools.get(call.name)
        if tool is None:
            return done(
                ok=False,
                error=f"unknown tool {call.name!r}. available: {', '.join(self.tools.names())}",
            )

        invalid = self.tools.validate(tool, call.arguments)
        if invalid:
            return done(ok=False, error=invalid)

        payload: dict[str, Any] = {
            "tool": tool,
            "arguments": call.arguments,
            "dangerous": tool.dangerous,
            "confirmed": not self.require_confirmation,
        }
        try:
            payload = self._emit("BEFORE_TOOL", payload, vetoable=True)
        except _veto_types() as veto:
            self._emit("ON_REFUSAL", {"tool": tool, "reason": str(veto)})
            return done(
                ok=False,
                blocked="veto",
                error=f"tool call vetoed by a hook: {veto or 'no reason given'}",
            )

        if self.confirm is not None and self.confirm(tool, dict(call.arguments)):
            payload["confirmed"] = True

        # The model may request a dangerous action; it may not authorise one.
        # Confirmation has to arrive from outside the conversation, or a prompt
        # that talks the model into "yes" also talks the robot into moving.
        if tool.dangerous and not payload.get("confirmed"):
            self._emit("ON_REFUSAL", {"tool": tool, "reason": "unconfirmed"})
            return done(
                ok=False,
                blocked="unconfirmed",
                error=(
                    f"{tool.name} is a dangerous action and was not confirmed by an "
                    f"operator, so it did not run. Ask for confirmation or choose a "
                    f"non-dangerous tool."
                ),
            )

        # A hook may rewrite the arguments (clamp a speed, pin a frame). That is
        # the reason BEFORE_TOOL threads a context at all, so honour it -- and
        # re-validate, because a hook is no more trustworthy about the schema
        # than the model was.
        arguments = payload.get("arguments", call.arguments)
        if arguments is not call.arguments:
            invalid = self.tools.validate(tool, arguments)
            if invalid:
                return done(ok=False, error=f"a hook rewrote the arguments: {invalid}")

        try:
            value = tool.fn(**arguments) if tool.fn is not None else None
        except Exception as exc:  # noqa: BLE001 - a throwing tool must not end the run
            self._emit("ON_ERROR", {"tool": tool, "error": exc, "where": "tool"})
            result = done(ok=False, ran=True, error=f"{type(exc).__name__}: {exc}")
        else:
            result = done(ok=True, ran=True, value=value)
        self._emit("AFTER_TOOL", {"tool": tool, "result": result})
        return result

    # ---- hooks -------------------------------------------------------------

    def _emit(
        self, point: str, payload: dict[str, Any], *, vetoable: bool = False
    ) -> dict[str, Any]:
        """Fire a hook point. No hook manager == no hooks.

        A HookVeto propagates only from BEFORE_TOOL, the one point where the
        loop can act on it. Vetoing a notification (AFTER_TOOL, ON_REFUSAL) is
        meaningless -- the thing already happened, or already did not -- and
        letting it escape there turned a blanket "veto everything" hook into an
        exception out of the refusal path. Every other hook exception is
        recorded and swallowed, because a logging hook with a typo in it must
        not be able to stop a robot mid-task.
        """
        if self.hooks is None:
            return payload
        resolved = getattr(_hook_points(), point, point)
        try:
            # HookManager.emit threads the context: a hook returning a dict
            # replaces it. Ignoring the return would silently drop that
            # transform, which is how a confirming hook would appear to work
            # and then not.
            returned = self.hooks.emit(resolved, payload)
            if isinstance(returned, dict):
                payload = returned
        except _veto_types() as veto:
            if vetoable:
                raise
            payload.setdefault("hook_errors", []).append(f"HookVeto ignored at {point}: {veto}")
        except Exception as exc:  # noqa: BLE001
            payload.setdefault("hook_errors", []).append(f"{type(exc).__name__}: {exc}")
        return payload


# --------------------------------------------------------------------------
# concurrent siblings: import lazily, work without them
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _veto_types() -> tuple[type[BaseException], ...]:
    try:
        from .hooks import HookVeto as RealVeto
    except ImportError:
        return (HookVeto,)
    return (HookVeto, RealVeto) if RealVeto is not HookVeto else (HookVeto,)


@lru_cache(maxsize=1)
def _hook_points() -> Any:
    try:
        from .hooks import HookPoint
    except ImportError:
        return None  # getattr(None, "BEFORE_TOOL", "BEFORE_TOOL") -> the name itself
    return HookPoint


@lru_cache(maxsize=1)
def _message_type() -> Any:
    try:
        from .providers import Message
    except ImportError:
        return None
    return Message


def _message(role: str, content: str, tool_call_id: str = "") -> Any:
    cls = _message_type()
    if cls is not None:
        try:
            if tool_call_id:
                return cls(role=role, content=content, tool_call_id=tool_call_id)
            return cls(role=role, content=content)
        except TypeError:  # a different Message signature: dicts are the lingua franca
            pass
    message = {"role": role, "content": content}
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    for attr in ("text", "content", "message"):
        value = getattr(completion, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(completion, dict):
        for key in ("text", "content"):
            if isinstance(completion.get(key), str):
                return completion[key]
    return "" if completion is None else str(completion)


def _native_calls(completion: Any) -> tuple[ToolCall, ...]:
    """Tool calls carried in a structured field, as hosted APIs return them.

    Preferred over the text path when present: the server already validated the
    JSON, so there is nothing for the parser to repair.
    """
    raw = getattr(completion, "tool_calls", None)
    if raw is None and isinstance(completion, dict):
        raw = completion.get("tool_calls")
    if not raw:
        return ()
    calls = []
    for item in raw:
        if isinstance(item, dict):
            name = item.get("name") or item.get("function", {}).get("name", "")
            args = item.get("arguments", item.get("function", {}).get("arguments", {}))
            call_id = item.get("id", "")
        else:
            name = getattr(item, "name", "")
            args = getattr(item, "arguments", {})
            call_id = getattr(item, "id", "")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (json.JSONDecodeError, ValueError) as exc:
                calls.append(
                    ToolCall(name=name, error=f"could not decode arguments: {exc}", id=call_id)
                )
                continue
        calls.append(
            ToolCall(
                name=str(name),
                arguments=args if isinstance(args, dict) else {},
                id=str(call_id or ""),
            )
        )
    return tuple(calls)

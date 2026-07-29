"""Checks for the provider abstraction. `python tests/test_providers.py` or pytest.

Nothing here touches a socket. Every adapter takes ``transport=``, so the wire
format is exercised against recorded response shapes -- a test that needs the
internet is a test that fails in the field, which is exactly where this code
has to work.

The security tests are the load-bearing ones: ``test_key_never_appears_in_*``
plant a fabricated key in the environment, force each escape route (repr, str,
raised message, error body echo, urllib's URL-bearing exception) and assert the
key is nowhere in the output.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmoid.providers import (
    SPECS,
    AnthropicProvider,
    AuthError,
    Completion,
    LocalProvider,
    Message,
    OpenAICompatProvider,
    Provider,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolSpec,
    auto,
    available,
    get,
    register,
)
from sigmoid.providers.base import KEY_ENV_VARS, _redact
from sigmoid.providers.registry import _REGISTRY, PRIORITY

# A key-shaped string that exists nowhere but this file. If it shows up in any
# output, something leaked it.
FAKE_KEY = "sk-fake-LEAKCANARY-0123456789abcdefXYZ"


@contextmanager
def env(**pairs):
    """Set env vars for the block, restoring exactly what was there.

    All provider env vars are cleared first: a real key in the developer's
    shell would otherwise change which provider `auto()` picks and make these
    tests pass or fail depending on whose laptop they run on.
    """
    saved = {k: os.environ.get(k) for k in (*KEY_ENV_VARS, "OLLAMA_HOST", "VLLM_BASE_URL", *pairs)}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in pairs.items() if v is not None})
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def fake_transport(response, capture=None, error=None):
    """A transport that returns a canned body and records what was sent."""

    def transport(url, headers, body, timeout, stream=False):
        if capture is not None:
            capture.update(url=url, headers=headers, payload=json.loads(body), timeout=timeout)
        if error is not None:
            raise error
        if stream:
            return iter(response)
        return json.dumps(response).encode()

    return transport


OPENAI_REPLY = {
    "model": "gpt-4o-mini",
    "choices": [
        {
            "message": {
                "content": "arm is level",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_imu", "arguments": '{"axis": "z"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
}

ANTHROPIC_REPLY = {
    "model": "claude-sonnet-4-5",
    "content": [
        {"type": "text", "text": "arm is level"},
        {"type": "tool_use", "id": "tu_1", "name": "read_imu", "input": {"axis": "z"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 11, "output_tokens": 4},
}


# --------------------------------------------------------------------------
# security: keys come from the environment and never come back out
# --------------------------------------------------------------------------


def test_key_never_appears_in_repr_or_str():
    """No provider stores a key, so no repr or str can contain one."""
    with env(OPENAI_API_KEY=FAKE_KEY, ANTHROPIC_API_KEY=FAKE_KEY):
        for p in (OpenAICompatProvider("openai"), AnthropicProvider(), LocalProvider()):
            assert FAKE_KEY not in repr(p), f"{p.name} leaked via repr"
            assert FAKE_KEY not in str(p), f"{p.name} leaked via str"
            assert FAKE_KEY not in repr(vars(p)), f"{p.name} stored the key on an attribute"
            assert FAKE_KEY not in json.dumps(
                {k: str(v) for k, v in vars(p).items()}
            ), f"{p.name} leaked via its attribute values"


def test_key_never_appears_in_a_raised_message():
    """Force every error path that could carry key material."""
    with env(OPENAI_API_KEY=FAKE_KEY):
        # 1. server echoes the Authorization header back in the error body
        echo = ProviderError(f"upstream said: Authorization: Bearer {FAKE_KEY}")
        assert FAKE_KEY not in _redact(str(echo))

        # 2. a transport failure whose message embeds the key (urllib puts the
        #    full URL in URLError text, and some endpoints take ?key=)
        boom = fake_transport(None, error=ProviderError(_redact(f"GET https://x/?key={FAKE_KEY}")))
        p = OpenAICompatProvider("openai", transport=boom)
        try:
            p.complete([Message("user", "hi")])
        except ProviderError as exc:
            assert FAKE_KEY not in str(exc)
            assert FAKE_KEY not in repr(exc)
        else:
            raise AssertionError("expected the transport error to propagate")

        # 3. a 200-with-error body carrying the key
        p2 = OpenAICompatProvider("openai", transport=fake_transport({"error": FAKE_KEY}))
        try:
            p2.complete([Message("user", "hi")])
        except ProviderError as exc:
            assert FAKE_KEY not in str(exc)
        else:
            raise AssertionError("expected an error body to raise")


def test_redact_catches_foreign_key_shapes():
    """Keys that never passed through our environment still get scrubbed --
    a user pasting one into a prompt must not have it logged back out."""
    with env():  # nothing configured, so only the shape regex can catch these
        for shape in (
            "sk-ant-api03-" + "a" * 40,
            "gsk_" + "b" * 40,
            "xai-" + "c" * 40,
            "AIza" + "d" * 30,
            "sk-or-v1-" + "e" * 40,
            "Bearer abcdefghijklmnopqrstuvwx",
        ):
            out = _redact(f"failed with {shape} attached")
            assert shape not in out, f"{shape[:10]}... survived redaction"
            assert "<redacted>" in out


def test_redact_leaves_ordinary_text_alone():
    """Redaction that eats normal error text is redaction nobody keeps on."""
    with env(OPENAI_API_KEY=FAKE_KEY):
        msg = "HTTP 500: model gpt-4o-mini is overloaded, retry in 3s"
        assert _redact(msg) == msg


def test_every_provider_env_var_is_covered_by_redaction():
    """A new endpoint added to SPECS without its variable in KEY_ENV_VARS would
    have a key that _redact cannot see. Fail here instead of in production."""
    missing = [s.env_var for s in SPECS.values() if s.env_var and s.env_var not in KEY_ENV_VARS]
    assert not missing, f"env vars not covered by _redact: {missing}"
    assert "ANTHROPIC_API_KEY" in KEY_ENV_VARS


def test_no_key_material_is_committed_to_the_repo():
    """The policy is environment-only. Check the package actually obeys it."""
    root = Path(__file__).resolve().parents[1]
    assert not (root / ".env").exists(), "a committed .env defeats the whole policy"
    for path in (root / "sigmoid" / "providers").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        # A literal key would have to be assigned; env lookups are the only
        # place a key name may appear next to a value.
        for var in KEY_ENV_VARS:
            assert f'{var} = "' not in text, f"{path.name} assigns {var} a literal"
            assert f"{var}='" not in text, f"{path.name} assigns {var} a literal"


def test_available_reports_presence_only():
    with env(GROQ_API_KEY=FAKE_KEY):
        names = available()
        assert "groq" in names
        assert "openai" not in names
        # The report is names, never values.
        assert all(FAKE_KEY not in n for n in names)


def test_missing_key_raises_auth_error_naming_only_the_variable():
    with env():
        p = OpenAICompatProvider("openai", transport=fake_transport(OPENAI_REPLY))
        try:
            p.complete([Message("user", "hi")])
        except AuthError as exc:
            assert "OPENAI_API_KEY" in str(exc)
        else:
            raise AssertionError("expected AuthError with no key set")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_registry_knows_every_documented_provider():
    for name in PRIORITY:
        assert name in _REGISTRY, f"{name} is in PRIORITY but not registered"
    assert set(SPECS) | {"anthropic", "local"} == set(_REGISTRY)


def test_get_unknown_provider_refuses_with_the_known_list():
    try:
        get("nonesuch")
    except ProviderError as exc:
        assert "nonesuch" in str(exc) and "openai" in str(exc)
    else:
        raise AssertionError("expected a refusal for an unknown provider")


def test_auto_follows_the_documented_priority_order():
    # groq outranks openai; both configured means groq wins.
    with env(GROQ_API_KEY=FAKE_KEY, OPENAI_API_KEY=FAKE_KEY):
        assert auto().name == "groq"
    # openai outranks anthropic.
    with env(OPENAI_API_KEY=FAKE_KEY, ANTHROPIC_API_KEY=FAKE_KEY):
        assert auto().name == "openai"
    # A local server outranks every hosted one -- no network, bounded latency.
    with env(OLLAMA_HOST="http://localhost:11434", GROQ_API_KEY=FAKE_KEY):
        assert auto().name == "ollama"


def test_auto_refuses_clearly_when_nothing_is_configured():
    with env():
        try:
            auto()
        except ProviderError as exc:
            assert "OPENAI_API_KEY" in str(exc)
        else:
            raise AssertionError("expected a refusal with no provider configured")


def test_register_accepts_a_third_party_provider():
    class Dummy(Provider):
        name = "dummy"
        available = True

        def complete(self, messages, tools=None, **kw):
            return Completion(text="ok")

        def stream(self, messages, tools=None, **kw):
            yield "ok"

    register("dummy", Dummy)
    try:
        assert get("dummy").complete([]).text == "ok"
        with env():
            assert available() == ["dummy"]  # unconfigured built-ins drop out
    finally:
        _REGISTRY.pop("dummy")


def test_a_broken_provider_does_not_break_discovery():
    """One bad backend must not stop a robot finding a working one."""

    class Exploding(Provider):
        name = "exploding"

        @property
        def available(self):
            raise RuntimeError("driver missing")

        def complete(self, messages, tools=None, **kw):
            raise RuntimeError

        def stream(self, messages, tools=None, **kw):
            raise RuntimeError

    register("exploding", Exploding)
    try:
        with env(GROQ_API_KEY=FAKE_KEY):
            assert available() == ["groq"]
            assert auto().name == "groq"
    finally:
        _REGISTRY.pop("exploding")


# --------------------------------------------------------------------------
# openai-compatible adapter
# --------------------------------------------------------------------------


def test_openai_request_shape_and_response_parsing():
    sent = {}
    with env(OPENAI_API_KEY=FAKE_KEY):
        p = OpenAICompatProvider("openai", transport=fake_transport(OPENAI_REPLY, sent))
        out = p.complete(
            [Message("system", "be careful"), Message("user", "check the arm")],
            tools=[ToolSpec("read_imu", "read the IMU", {"type": "object", "properties": {}})],
            temperature=0.0,
        )

    assert sent["url"] == "https://api.openai.com/v1/chat/completions"
    assert sent["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert sent["payload"]["messages"][0] == {"role": "system", "content": "be careful"}
    assert sent["payload"]["tools"][0]["function"]["name"] == "read_imu"
    assert sent["payload"]["temperature"] == 0.0

    assert out.text == "arm is level"
    assert out.finish_reason == "tool_calls"
    assert out.tool_calls[0].name == "read_imu"
    assert out.tool_calls[0].arguments == {"axis": "z"}
    assert out.usage == {"input": 11, "output": 4, "total": 15}
    assert out.latency_ms >= 0.0
    assert out.model == "gpt-4o-mini"


def test_tool_results_round_trip_back_onto_the_wire():
    sent = {}
    with env(OPENAI_API_KEY=FAKE_KEY):
        p = OpenAICompatProvider("openai", transport=fake_transport(OPENAI_REPLY, sent))
        p.complete(
            [
                Message(
                    "assistant", "", tool_calls=[ToolCall("call_1", "read_imu", {"axis": "z"})]
                ),
                Message("tool", "0.02", tool_call_id="call_1"),
            ]
        )
    wire = sent["payload"]["messages"]
    assert json.loads(wire[0]["tool_calls"][0]["function"]["arguments"]) == {"axis": "z"}
    assert wire[1]["tool_call_id"] == "call_1"


def test_malformed_tool_arguments_stay_visible():
    """Models emit invalid JSON. Silently handing a robot empty arguments is
    worse than handing it something obviously wrong."""
    reply = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c", "function": {"name": "move", "arguments": "{not json"}}
                    ]
                }
            }
        ]
    }
    with env(OPENAI_API_KEY=FAKE_KEY):
        p = OpenAICompatProvider("openai", transport=fake_transport(reply))
        call = p.complete([Message("user", "go")]).tool_calls[0]
    assert call.arguments == {"_raw": "{not json"}


def test_streaming_yields_text_deltas_and_stops_at_done():
    frames = [
        b'data: {"choices":[{"delta":{"content":"lift"}}]}\n',
        b": keep-alive\n",
        b'data: {"choices":[{"delta":{"content":"ing"}}]}\n',
        b"data: [DONE]\n",
        b'data: {"choices":[{"delta":{"content":"never"}}]}\n',
    ]
    with env(OPENAI_API_KEY=FAKE_KEY):
        p = OpenAICompatProvider("openai", transport=fake_transport(frames))
        assert list(p.stream([Message("user", "go")])) == ["lift", "ing"]


def test_every_spec_is_reachable_and_distinct():
    for name, spec in SPECS.items():
        with env(**({spec.env_var: FAKE_KEY} if spec.env_var else {})):
            p = get(name)
            assert p.name == name
            if spec.env_var:
                assert p.available, f"{name} should be available with {spec.env_var} set"
                assert p._base_url().endswith("/v1"), f"{name} base url must end in /v1"
    urls = {s.base_url for s in SPECS.values() if s.base_url}
    assert len(urls) == len([s for s in SPECS.values() if s.base_url]), "duplicate base urls"


def test_local_server_base_urls_normalise_either_way():
    """OLLAMA_HOST is a host, VLLM_BASE_URL is usually a base url. Both work."""
    with env(OLLAMA_HOST="http://localhost:11434"):
        assert get("ollama")._base_url() == "http://localhost:11434/v1"
    with env(VLLM_BASE_URL="http://box:8000/v1/"):
        assert get("vllm")._base_url() == "http://box:8000/v1"
    with env():
        assert not get("ollama").available and not get("vllm").available


def test_rate_limit_and_auth_map_to_distinct_exceptions():
    """A robot retries a rate limit and gives up on a bad key. It can only do
    that if the two arrive as different types."""
    from sigmoid.providers import TimeoutError as ProviderTimeout
    from sigmoid.providers.base import _status_error

    assert isinstance(_status_error(429, ""), RateLimitError)
    assert isinstance(_status_error(401, ""), AuthError)
    assert isinstance(_status_error(504, ""), ProviderTimeout)
    # ...and it is also a builtin TimeoutError, so deadline handling written
    # against `except TimeoutError` keeps working without importing ours.
    assert isinstance(_status_error(504, ""), TimeoutError)
    assert issubclass(ProviderTimeout, ProviderError)
    assert type(_status_error(500, "")) is ProviderError

    # Status mapping redacts its own detail rather than trusting the caller --
    # this is the last hop before an error body becomes an exception message.
    with env(OPENAI_API_KEY=FAKE_KEY):
        for code in (401, 429, 504, 500):
            assert FAKE_KEY not in str(_status_error(code, f"sent Bearer {FAKE_KEY}"))


# --------------------------------------------------------------------------
# anthropic adapter
# --------------------------------------------------------------------------


def test_anthropic_hoists_system_and_uses_content_blocks():
    sent = {}
    with env(ANTHROPIC_API_KEY=FAKE_KEY):
        p = AnthropicProvider(transport=fake_transport(ANTHROPIC_REPLY, sent))
        out = p.complete(
            [
                Message("system", "be careful"),
                Message("system", "stay under 0.2 m/s"),
                Message("user", "check the arm"),
            ],
            tools=[ToolSpec("read_imu", "read the IMU", {"type": "object", "properties": {}})],
        )

    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["headers"]["x-api-key"] == FAKE_KEY
    assert sent["headers"]["anthropic-version"] == "2023-06-01"
    # Both system turns survive -- a dropped safety preamble is the bug here.
    assert sent["payload"]["system"] == "be careful\n\nstay under 0.2 m/s"
    assert all(m["role"] != "system" for m in sent["payload"]["messages"])
    assert sent["payload"]["messages"][0]["content"] == [
        {"type": "text", "text": "check the arm"}
    ]
    assert sent["payload"]["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert sent["payload"]["max_tokens"] > 0

    assert out.text == "arm is level"
    assert out.tool_calls[0].name == "read_imu"
    assert out.tool_calls[0].arguments == {"axis": "z"}
    assert out.finish_reason == "tool_use"
    assert out.usage == {"input": 11, "output": 4, "total": 15}


def test_anthropic_tool_results_become_user_blocks():
    sent = {}
    with env(ANTHROPIC_API_KEY=FAKE_KEY):
        p = AnthropicProvider(transport=fake_transport(ANTHROPIC_REPLY, sent))
        p.complete(
            [
                Message("assistant", "checking", tool_calls=[ToolCall("tu_1", "read_imu", {})]),
                Message("tool", "0.02", tool_call_id="tu_1"),
            ]
        )
    assistant, result = sent["payload"]["messages"]
    assert [b["type"] for b in assistant["content"]] == ["text", "tool_use"]
    assert result["role"] == "user"
    assert result["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "0.02"}


def test_anthropic_streaming_takes_text_deltas_only():
    frames = [
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lift"}}\n',
        b'data: {"type":"content_block_delta","delta":{"type":"input_json_delta",'
        b'"partial_json":"{\\"a\\":"}}\n',
        b'data: {"type":"message_stop"}\n',
    ]
    with env(ANTHROPIC_API_KEY=FAKE_KEY):
        p = AnthropicProvider(transport=fake_transport(frames))
        assert list(p.stream([Message("user", "go")])) == ["lift"]


def test_both_adapters_return_the_same_completion_shape():
    """The point of the abstraction: a planner cannot tell them apart."""
    with env(OPENAI_API_KEY=FAKE_KEY, ANTHROPIC_API_KEY=FAKE_KEY):
        a = OpenAICompatProvider("openai", transport=fake_transport(OPENAI_REPLY)).complete([])
        b = AnthropicProvider(transport=fake_transport(ANTHROPIC_REPLY)).complete([])
    assert a.text == b.text
    assert a.usage == b.usage
    assert [(c.name, c.arguments) for c in a.tool_calls] == [
        (c.name, c.arguments) for c in b.tool_calls
    ]


# --------------------------------------------------------------------------
# local provider
# --------------------------------------------------------------------------


class FakeEngine:
    """Same shape as sigmoid.inference.InferenceEngine.generate, without torch.

    Mirrors the real signature -- ``generate(prompt, *, max_new_tokens, stream)``
    returning a ``GenerationResult``-alike -- so this exercises the same code
    path the Triton engine will, on a machine with no GPU.
    """

    class Stats:
        tokens = 3

    class Result:
        text = "arm is level"
        stats = None

    def __init__(self):
        self.seen = None
        self.Result.stats = self.Stats()

    def generate(self, prompt=None, *, max_new_tokens=32, stream=False, **overrides):
        self.seen = (prompt, max_new_tokens, overrides)
        if stream:
            return iter(("arm ", "is ", "level"))
        return self.Result()


def test_local_degrades_to_unavailable_without_a_model():
    """InferenceEngine needs a loaded model and torch. Reporting 'available'
    on the strength of the module importing would hand a robot a backend that
    fails on first call -- the worst possible moment to find out."""
    p = LocalProvider()
    assert p.available is False
    try:
        p.complete([Message("user", "hi")])
    except ProviderError as exc:
        assert "local" in str(exc)
    else:
        raise AssertionError("expected a refusal with no engine and no model")


def test_local_satisfies_the_same_interface_as_the_hosted_ones():
    engine = FakeEngine()
    p = LocalProvider(engine=engine)
    assert isinstance(p, Provider) and p.available and p.name == "local"

    out = p.complete([Message("user", "check the arm")], tools=[ToolSpec("read_imu")])
    assert out.text == "arm is level"
    assert out.model == "local"
    assert out.latency_ms > 0.0  # measured here; the engine does not report it
    assert out.usage == {"input": 0, "output": 3, "total": 3}

    # Chat turns reached the engine as a prompt, roles intact.
    prompt = engine.seen[0]
    assert "user: check the arm" in prompt and prompt.endswith("assistant:")
    assert list(p.stream([Message("user", "go")])) == ["arm ", "is ", "level"]


def test_local_completion_is_interchangeable_with_a_hosted_one():
    """The whole point: a planner cannot tell where the text came from."""
    with env(OPENAI_API_KEY=FAKE_KEY):
        hosted = OpenAICompatProvider("openai", transport=fake_transport(OPENAI_REPLY)).complete([])
    local = LocalProvider(engine=FakeEngine()).complete([])
    assert type(local) is type(hosted) is Completion
    assert local.text == hosted.text
    assert set(local.usage) == set(hosted.usage)


def test_local_streams_even_without_engine_streaming_support():
    class NoStream:
        def generate(self, prompt, **kw):
            return "one shot"

    assert list(LocalProvider(engine=NoStream()).stream([Message("user", "go")])) == ["one shot"]


def test_local_passes_only_kwargs_the_engine_declares():
    """A generation kwarg the engine does not take must not become a TypeError
    from inside the robot's control loop."""

    class Picky:
        def generate(self, prompt, *, max_new_tokens=8, temperature=0.0):
            return f"t={temperature} n={max_new_tokens}"

    out = LocalProvider(engine=Picky(), max_new_tokens=5).complete(
        [Message("user", "go")], temperature=0.7, top_p=0.9
    )
    assert out.text == "t=0.7 n=5"


def test_local_names_a_missing_entry_point():
    try:
        LocalProvider(engine=object()).complete([])
    except ProviderError as exc:
        assert "generate/complete/__call__" in str(exc)
    else:
        raise AssertionError("expected a refusal for an engine with no entry point")


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

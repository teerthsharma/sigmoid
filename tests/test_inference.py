"""Runnable checks for the prefill/decode engine.

`python tests/test_inference.py` or pytest. Standalone also prints the speed
sweep, which is the part that cannot be asserted -- it is a measurement, and it
is allowed to say topology lost.

The load-bearing test is `test_dense_reproduces_huggingface_generate`. Everything
else in this file is only interpretable once the loop itself is known to be right,
because a KV cache bug and a schedule bug both show up as "different tokens".
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SIGMOID_KERNELS_PATH", "C:/Users/seal/Documents/GitHub/kernels")

PROMPT = "The theory of topology says that the universe"

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from sigmoid.inference import (  # noqa: E402
    GenerationStats,
    InferenceConfig,
    InferenceEngine,
    load_kernel,
)
from sigmoid.triton import zero_dim_persistence_salience  # noqa: E402

_CUDA = torch.cuda.is_available()
_KERNEL = load_kernel() is not None and _CUDA
needs_cuda = pytest.mark.skipif(not _CUDA, reason="needs CUDA")
needs_kernel = pytest.mark.skipif(not _KERNEL, reason="needs CUDA + the merged Triton kernel")

_MODEL = None


def distilgpt2():
    """distilgpt2 from the local cache -- 6 layers, 12 heads, head_dim 64, ctx 1024."""
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("distilgpt2", local_files_only=True)
        model.eval()
        if _CUDA:
            model.cuda()
        _MODEL = (model, tokenizer)
    return _MODEL


def dense_config(**kw):
    """A schedule that selects every causal block -- sparse plumbing, dense result."""
    base = dict(block_size=16, local_radius_blocks=1024, sink_blocks=1024, topk=1024)
    base.update(kw)
    return InferenceConfig(**base)


def synthetic_ids(tokenizer, length, seed=0):
    """A token sequence of an exact length; real text will not land on 512 on request."""
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    return [int(t) for t in rng.integers(0, vocab, size=length)]


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


# -- the control arm -------------------------------------------------------------


def test_dense_reproduces_huggingface_generate():
    """The whole file rests on this: same tokens as HF, or the loop is broken."""
    model, tokenizer = distilgpt2()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        reference = model.generate(
            ids, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    expected = reference[0, ids.shape[1] :].tolist()

    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        result = engine.generate(PROMPT, max_new_tokens=64)
    finally:
        engine.close()
    assert result.ids == expected, (
        f"dense diverged from HF at token {first_divergence(result.ids, expected)}"
    )
    assert result.stats.sparsity == 1.0


def test_dense_backend_leaves_the_model_untouched():
    model, tokenizer = distilgpt2()
    before = model.config._attn_implementation
    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        assert model.config._attn_implementation == before
    finally:
        engine.close()


def test_dense_refuses_to_run_under_a_live_hook():
    """A leaked hook would make the control arm sparse without saying so."""
    model, tokenizer = distilgpt2()
    hooked = InferenceEngine(model, tokenizer, config=dense_config(backend="torch"))
    try:
        with pytest.raises(RuntimeError, match="still has a sigmoid attention hook"):
            InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    finally:
        hooked.close()
    assert model.config._attn_implementation != "sigmoid_topology"


# -- the scheduled path against that control -------------------------------------


@pytest.mark.parametrize("backend", ["torch", pytest.param("triton", marks=needs_kernel)])
def test_full_schedule_reproduces_dense(backend):
    """Selecting every causal block must give dense back exactly.

    This is what separates "the kernel plumbing is wrong" from "sparsity costs
    quality": with the schedule saturated the two answers are the same
    computation, so any difference here is a bug and not a trade-off.
    """
    model, tokenizer = distilgpt2()
    control = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        expected = control.generate(PROMPT, max_new_tokens=48).ids
    finally:
        control.close()

    engine = InferenceEngine(model, tokenizer, config=dense_config(backend=backend))
    try:
        got = engine.generate(PROMPT, max_new_tokens=48)
    finally:
        engine.close()
    assert got.ids == expected, (
        f"{backend} with a saturated schedule diverged at "
        f"{first_divergence(got.ids, expected)}"
    )
    assert got.stats.sparsity == 1.0


@needs_cuda
def test_a_sparse_schedule_actually_changes_the_computation():
    """Density below 1 must move the logits; if it does not, nothing is being skipped."""
    model, tokenizer = distilgpt2()
    ids = synthetic_ids(tokenizer, 960)

    saturated = _next_token_logprobs(model, tokenizer, ids, dense_config(backend="torch"))
    sparse_cfg = InferenceConfig(
        backend="torch", block_size=64, local_radius_blocks=1, sink_blocks=1, topk=2
    )
    sparse, stats = _next_token_logprobs(model, tokenizer, ids, sparse_cfg, with_stats=True)

    # a prefill can never be very sparse: query block q has only q+1 causal blocks
    # to skip, so the first few rows are dense whatever the schedule says. This
    # config caps a row at 5 blocks, giving 0.60 over 15 blocks against a measured
    # 0.54 (top-k overlaps the local window). Decode is where the density falls.
    assert stats.sparsity < 0.6, f"schedule was not sparse (density {stats.sparsity:.3f})"
    kl = _kl(saturated, sparse)
    assert kl > 0.0, "a sparse schedule produced bit-identical logits -- it ran dense"
    assert np.isfinite(kl)


@needs_cuda
def test_saturated_schedule_has_zero_kl_against_dense():
    model, tokenizer = distilgpt2()
    ids = synthetic_ids(tokenizer, 256)
    reference = _next_token_logprobs(model, tokenizer, ids, InferenceConfig(backend="dense"))
    saturated = _next_token_logprobs(model, tokenizer, ids, dense_config(backend="torch"))
    assert _kl(reference, saturated) < 1e-9


def _next_token_logprobs(model, tokenizer, ids, config, with_stats=False):
    engine = InferenceEngine(model, tokenizer, config=config)
    try:
        session = engine.prefill(ids)
        logprobs = torch.log_softmax(session.logits.float(), dim=-1).cpu().numpy()
    finally:
        engine.close()
    return (logprobs, session.stats) if with_stats else logprobs


def _kl(reference_logprobs, other_logprobs):
    """KL(reference || other) in nats, from log-probabilities."""
    p = np.exp(reference_logprobs)
    return float(np.sum(p * (reference_logprobs - other_logprobs)))


# -- the KV cache ----------------------------------------------------------------


@needs_cuda
def test_the_prefix_is_never_recomputed():
    """Total query positions must be prompt + one per step, not prompt x steps."""
    model, tokenizer = distilgpt2()
    engine = InferenceEngine(model, tokenizer, config=dense_config(backend="torch"))
    seen = []
    original = engine._attend

    def counting(module, query, key, value):
        seen.append((query.shape[-2], key.shape[-2]))
        return original(module, query, key, value)

    engine._attend = counting
    try:
        prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0].tolist()
        session = engine.prefill(prompt_ids)
        for _ in range(16):
            engine.decode_step(session)
    finally:
        engine.close()

    layers = model.config.n_layer
    prompt_len = len(prompt_ids)
    query_positions = sum(q for q, _ in seen)
    assert query_positions == layers * (prompt_len + 16), (
        "the prefix is being recomputed: "
        f"{query_positions} query positions for {prompt_len} + 16 tokens"
    )
    # the cache grows by exactly one key per step
    decode_key_lengths = [k for q, k in seen if q == 1]
    assert decode_key_lengths == sorted(decode_key_lengths)
    assert decode_key_lengths[-1] == prompt_len + 16
    assert len(session.ids) == prompt_len + 16
    assert session.stats.kv_bytes > 0


@needs_cuda
def test_kv_bytes_track_the_cache():
    model, tokenizer = distilgpt2()
    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        session = engine.prefill(synthetic_ids(tokenizer, 128))
        before = session.stats.kv_bytes
        for _ in range(8):
            engine.decode_step(session)
    finally:
        engine.close()
    # 6 layers x 12 heads x 64 dim x 4 bytes x 2 (k and v) per position
    per_token = model.config.n_layer * model.config.n_embd * 4 * 2
    assert session.stats.kv_bytes - before == 8 * per_token


# -- the schedule grows instead of being rebuilt ---------------------------------


@needs_cuda
def test_incremental_salience_matches_a_rebuild_at_every_layer():
    """`IncrementalSalience` is only worth using if it is the same answer."""
    model, tokenizer = distilgpt2()
    config = InferenceConfig(
        backend="torch", block_size=64, local_radius_blocks=1, sink_blocks=1, topk=3
    )
    engine = InferenceEngine(model, tokenizer, config=config)
    try:
        session = engine.prefill(synthetic_ids(tokenizer, 200))
        for _ in range(96):  # crosses at least one block boundary (64)
            engine.decode_step(session)
        states = list(engine._layers.values())
    finally:
        engine.close()

    assert len(states) == model.config.n_layer
    for state in states:
        assert state.n_blocks == (200 + 96) // 64
        rebuilt = zero_dim_persistence_salience(state._state.centroids)
        assert np.array_equal(state._state.salience, rebuilt), (
            "incremental salience drifted from a batch rebuild"
        )


@needs_cuda
def test_salience_updates_only_on_block_boundaries():
    """Rebuilding per token is the waste `IncrementalSalience` exists to remove."""
    model, tokenizer = distilgpt2()
    config = InferenceConfig(backend="torch", block_size=64, topk=2)
    engine = InferenceEngine(model, tokenizer, config=config)
    try:
        session = engine.prefill(synthetic_ids(tokenizer, 192))
        state = None
        counts = []
        for _ in range(70):
            engine.decode_step(session)
            state = next(iter(engine._layers.values()))
            counts.append(state.n_blocks)
    finally:
        engine.close()
    assert counts[0] == 3, counts[0]
    assert counts[-1] == 4, counts[-1]
    assert len(set(counts)) == 2, f"salience moved {len(set(counts))} times over 70 tokens"


# -- sampling --------------------------------------------------------------------


@needs_cuda
def test_sampling_is_deterministic_given_a_seed():
    model, tokenizer = distilgpt2()
    kw = dict(backend="dense", do_sample=True, temperature=0.9, top_k=50, top_p=0.95)
    runs = []
    for seed in (7, 7, 8):
        engine = InferenceEngine(model, tokenizer, config=InferenceConfig(seed=seed, **kw))
        try:
            runs.append(engine.generate(PROMPT, max_new_tokens=24).ids)
        finally:
            engine.close()
    assert runs[0] == runs[1], "same seed gave different tokens"
    assert runs[0] != runs[2], "different seeds gave identical tokens"


@needs_cuda
def test_greedy_and_zero_temperature_agree():
    model, tokenizer = distilgpt2()
    outs = []
    for kw in (dict(do_sample=False), dict(do_sample=True, temperature=0.0)):
        engine = InferenceEngine(
            model, tokenizer, config=InferenceConfig(backend="dense", **kw)
        )
        try:
            outs.append(engine.generate(PROMPT, max_new_tokens=16).ids)
        finally:
            engine.close()
    assert outs[0] == outs[1]


@needs_cuda
def test_top_k_of_one_is_greedy():
    model, tokenizer = distilgpt2()
    outs = []
    for kw in (
        dict(do_sample=False),
        dict(do_sample=True, top_k=1, temperature=1.7, seed=3),
    ):
        engine = InferenceEngine(
            model, tokenizer, config=InferenceConfig(backend="dense", **kw)
        )
        try:
            outs.append(engine.generate(PROMPT, max_new_tokens=16).ids)
        finally:
            engine.close()
    assert outs[0] == outs[1]


# -- the API an agent runtime needs ----------------------------------------------


@needs_cuda
def test_stream_yields_tokens_as_they_land():
    model, tokenizer = distilgpt2()
    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        pieces = list(engine.generate(PROMPT, max_new_tokens=12, stream=True))
        result = engine.last_result
    finally:
        engine.close()
    assert len(pieces) == 12
    assert "".join(pieces) == result.text
    assert len(result.ids) == 12


@needs_cuda
def test_prefill_and_decode_step_drive_the_loop_by_hand():
    model, tokenizer = distilgpt2()
    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        driven = engine.prefill(tokenizer(PROMPT, return_tensors="pt").input_ids)
        by_hand = [engine.decode_step(driven) for _ in range(20)]
        whole = engine.generate(PROMPT, max_new_tokens=20).ids
    finally:
        engine.close()
    assert by_hand == whole
    assert driven.generated == by_hand


def test_config_refuses_a_block_size_the_kernel_cannot_take():
    with pytest.raises(ValueError, match="power of two"):
        InferenceConfig(block_size=48)
    with pytest.raises(ValueError, match="backend must be"):
        InferenceConfig(backend="cuda")
    with pytest.raises(ValueError, match="non-negative"):
        InferenceConfig(topk=-1)


@needs_cuda
def test_prefill_refuses_a_prompt_past_the_context_limit():
    model, tokenizer = distilgpt2()
    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        assert engine.max_context == 1024
        with pytest.raises(ValueError, match="exceeds max_context"):
            engine.prefill(synthetic_ids(tokenizer, 1100))
    finally:
        engine.close()


@needs_cuda
def test_generation_stops_at_the_context_limit():
    model, tokenizer = distilgpt2()
    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        result = engine.generate(input_ids=synthetic_ids(tokenizer, 1000), max_new_tokens=64)
    finally:
        engine.close()
    assert len(result.ids) == 24


# -- the timing split ------------------------------------------------------------


def test_stats_report_schedule_apart_from_attention():
    stats = GenerationStats(
        tokens=10, decode_ms=200.0, schedule_ms=5.0, attention_ms=60.0,
        selected_blocks=30, dense_blocks=120,
    )
    assert stats.decode_ms_per_token == 20.0
    assert stats.tokens_per_sec == 50.0
    assert stats.sparsity == 0.25
    assert stats.schedule_ms != stats.attention_ms


@needs_cuda
def test_schedule_time_is_measured_and_small():
    """Schedule build is CPU/numpy work, so perf_counter around it is exact."""
    model, tokenizer = distilgpt2()
    config = InferenceConfig(
        backend="torch", block_size=64, topk=2, local_radius_blocks=1, profile=True
    )
    engine = InferenceEngine(model, tokenizer, config=config)
    try:
        session = engine.prefill(synthetic_ids(tokenizer, 512))
        for _ in range(32):
            engine.decode_step(session)
    finally:
        engine.close()
    stats = session.stats
    assert stats.schedule_ms > 0.0
    assert stats.attention_ms > 0.0
    assert stats.schedule_ms < stats.decode_ms, "schedule build outweighed the whole step"


# -- the sweep (printed, not asserted) -------------------------------------------


def _run_arm(model, tokenizer, ids, config, new_tokens, rounds=3):
    """Best of `rounds`. Laptop GPU clocks drift ~20% between back-to-back runs,
    which is larger than every difference this table is trying to show, so the
    minimum is the only estimator that survives it."""
    engine = InferenceEngine(model, tokenizer, config=config)
    best = None
    try:
        session = engine.prefill(list(ids))
        for _ in range(4):  # warmup: the first Triton call compiles
            engine.decode_step(session)
        for _ in range(rounds):
            session = engine.prefill(list(ids))
            for _ in range(new_tokens):
                engine.decode_step(session)
            if best is None or session.stats.decode_ms < best.decode_ms:
                best = session.stats
    finally:
        engine.close()
    return best


def _time(fn, repeats):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    mark = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - mark) / repeats * 1e3


def _decode_attention_cost(engine, seq, heads, dim, config, repeats=60):
    """One decode step's attention, dense against this engine's own sparse path.

    distilgpt2 stops at 1024 positions, so anything past that cannot be generated
    end to end. Timing the attention op alone is the honest substitute, and it is
    the only term that changes between the arms -- MLP, layernorm and the
    projections are identical.
    """
    from sigmoid.inference import _LayerSchedule

    device = torch.device("cuda")
    key = torch.randn(1, heads, seq, dim, device=device)
    value = torch.randn_like(key)
    query = torch.randn(1, heads, 1, dim, device=device)
    scale = 1.0 / np.sqrt(dim)

    state = _LayerSchedule(config)
    state.observe(key)
    query_block = (seq - 1) // config.block_size
    blocks = state.allowed(query_block)

    def dense():
        w = torch.softmax((torch.matmul(query, key.transpose(-1, -2)) * scale).float(), -1)
        return torch.matmul(w.type(value.dtype), value)

    def sparse():
        return engine._decode_attention(state, query, key, value, blocks, query_block, scale)

    out = {
        "dense": _time(dense, repeats),
        "topology": _time(sparse, repeats),
        "blocks": len(blocks),
        "density": sum(min((b + 1) * config.block_size, seq) - b * config.block_size
                       for b in blocks) / seq,
    }
    return out


def model_written_ids(model, tokenizer, length):
    """A long prompt the model itself wrote, so the context is in-distribution.

    Uniform random tokens are the worst case for any topology schedule -- there is
    no structure for the salience to find -- so both are reported.
    """
    config = InferenceConfig(backend="dense", do_sample=True, temperature=1.0, seed=0)
    engine = InferenceEngine(model, tokenizer, config=config)
    try:
        seed = tokenizer(PROMPT, return_tensors="pt").input_ids[0].tolist()
        return seed + engine.generate(input_ids=seed, max_new_tokens=length - len(seed)).ids
    finally:
        engine.close()


def quality(ids=None, label="uniform random tokens"):
    """What the schedule costs in answers, against the dense continuation."""
    model, tokenizer = distilgpt2()
    new_tokens = 48
    ids = synthetic_ids(tokenizer, 960) if ids is None else ids
    context = len(ids)

    control = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="dense"))
    try:
        reference = torch.log_softmax(
            control.prefill(ids).logits.float(), dim=-1
        ).cpu().numpy()
        dense_tokens = control.generate(input_ids=ids, max_new_tokens=new_tokens).ids
    finally:
        control.close()

    print(f"\nquality at ctx {context} ({label}), {new_tokens} greedy tokens vs dense")
    print(f"  {'topk':>4} {'radius':>7} {'density':>8} {'KL nats':>10} "
          f"{'first div':>10} {'diverged':>9}")
    for topk, radius in ((1024, 1024), (8, 2), (4, 1), (2, 1), (0, 1)):
        config = InferenceConfig(
            backend="torch", block_size=64, sink_blocks=1,
            local_radius_blocks=radius, topk=topk,
        )
        engine = InferenceEngine(model, tokenizer, config=config)
        try:
            session = engine.prefill(ids)
            logprobs = torch.log_softmax(session.logits.float(), dim=-1).cpu().numpy()
            result = engine.generate(input_ids=ids, max_new_tokens=new_tokens)
        finally:
            engine.close()
        diverged = sum(a != b for a, b in zip(result.ids, dense_tokens))
        print(f"  {topk:>4} {radius:>7} {session.stats.sparsity:>8.3f} "
              f"{_kl(reference, logprobs):>10.5f} "
              f"{first_divergence(result.ids, dense_tokens):>10} "
              f"{diverged:>4}/{new_tokens}")


def sweep():
    """Print the dense-vs-topology speed table. Not a test -- it may say we lost."""
    if not _CUDA:
        print("no CUDA: sweep skipped")
        return
    model, tokenizer = distilgpt2()
    quality()
    quality(model_written_ids(model, tokenizer, 960), "text the model wrote itself")
    topology = dict(block_size=64, local_radius_blocks=1, sink_blocks=1, topk=2)
    new_tokens = 64

    print("\nend-to-end decode, distilgpt2, RTX 4060 Laptop (real contexts)")
    print(f"  {'ctx':>5}  {'arm':<10} {'tok/s':>8} {'ms/tok':>8} {'sched':>8} "
          f"{'attn':>8} {'density':>8}")
    for context in (256, 512, 960):
        ids = synthetic_ids(tokenizer, context)
        arms = [("dense", InferenceConfig(backend="dense"))]
        arms.append(("torch-dense", InferenceConfig(backend="torch", block_size=64,
                                                    local_radius_blocks=1024,
                                                    sink_blocks=1024, topk=1024)))
        arms.append(("topology", InferenceConfig(backend="torch", **topology)))
        if _KERNEL:
            arms.append(("topo-triton", InferenceConfig(backend="triton", **topology)))
        for name, config in arms:
            stats = _run_arm(model, tokenizer, ids, config, new_tokens)
            print(f"  {context:>5}  {name:<10} {stats.tokens_per_sec:>8.1f} "
                  f"{stats.decode_ms_per_token:>8.2f} {stats.schedule_ms:>8.1f} "
                  f"{stats.attention_ms:>8.1f} {stats.sparsity:>8.3f}")

    print("\ndecode attention op, one layer -- where does topology start winning?")
    print("  distilgpt2 cannot run past 1024, so every longer row is synthesized:")
    print("  real key/value tensors at the model's head geometry, no token generated.")
    config = InferenceConfig(backend="torch", block_size=128, local_radius_blocks=2,
                             sink_blocks=1, topk=8)
    engine = InferenceEngine(model, tokenizer, config=config)
    try:
        for label, heads, dim, lengths in (
            ("distilgpt2 (12 x 64)", 12, 64,
             (1024, 2048, 4096, 8192, 16384, 65536, 131072)),
            ("7B-class  (32 x 128)", 32, 128, (1024, 2048, 4096, 8192, 16384, 32768)),
        ):
            print(f"\n  {label}")
            print(f"    {'seq':>7} {'dense ms':>9} {'topo ms':>9} {'speedup':>8} "
                  f"{'blocks':>7} {'density':>8}  note")
            for seq in lengths:
                row = _decode_attention_cost(engine, seq, heads, dim, config)
                torch.cuda.empty_cache()
                note = "real" if seq <= 1024 else "synthetic"
                print(f"    {seq:>7} {row['dense']:>9.3f} {row['topology']:>9.3f} "
                      f"{row['dense'] / row['topology']:>7.2f}x {row['blocks']:>7} "
                      f"{row['density']:>8.4f}  {note}")
    finally:
        engine.close()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        if any(m.name == "skipif" and m.args and m.args[0] for m in marks):
            print(f"  SKIP  {name}")
            continue
        cases = [()]
        for mark in marks:
            if mark.name == "parametrize":
                cases = [(v,) for v in mark.args[1]]
        for case in cases:
            args = tuple(
                getattr(c, "values", (c,))[0] if hasattr(c, "values") else c for c in case
            )
            label = f"{name}{list(args) if args else ''}"
            try:
                fn(*args)
                print(f"  PASS  {label}")
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ == "Skipped":
                    print(f"  SKIP  {label}")
                    continue
                failures += 1
                kind = "FAIL" if isinstance(exc, AssertionError) else "ERROR"
                print(f"  {kind}  {label}: {exc}")
    print(f"\n{'all green' if not failures else f'{failures} failing'}")
    sweep()
    sys.exit(1 if failures else 0)

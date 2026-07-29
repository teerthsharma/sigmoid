"""Runnable checks for the HuggingFace patch. `python tests/test_patch.py`.

The load-bearing test is `test_dense_schedule_matches_the_unpatched_model`. A
dense causal schedule makes the sparse kernel compute exactly the attention the
model already computes, so any difference beyond arithmetic noise is a wiring
fault -- a dropped mask, a transposed output, a wrong scale -- and not a cost of
sparsity. Only once that number is small does the topology number mean anything.

The gap between "wired wrong" and "wired right" is not subtle, and it is
measured rather than assumed. Registering the attention implementation without
registering a matching mask function leaves `create_causal_mask` returning None
and every eager fallback attending to the future: patching layer 0 alone
measured 1.342e+02 max abs logit difference that way, against 1.877e-01 wired
correctly. Patching *every* layer measures 2.750e-01 either way, so the
whole-model test alone cannot see it.

distilgpt2 must already be in the local HuggingFace cache; nothing here reaches
the network. Everything needing CUDA, triton or transformers skips rather than
fails.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# the kernels repo ships no installable distribution, so it is only importable
# if its root is on the path -- same convention as tests/test_triton_attention.py
if "SIGMOID_KERNELS_PATH" not in os.environ:
    for guess in (
        Path(__file__).resolve().parents[2] / "kernels",
        Path.home() / "Documents" / "GitHub" / "kernels",
    ):
        if (guess / "kernels" / "topology_sparse_attention.py").exists():
            os.environ["SIGMOID_KERNELS_PATH"] = str(guess)
            break

from sigmoid.triton.attention import _merged  # noqa: E402
from sigmoid.triton.patch import _IMPL, _PATCH_ATTR, patch_attention  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

try:
    import transformers  # noqa: F401

    HAVE_HF = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_HF = False

KERNEL = _merged()
HAVE_GPU = torch is not None and KERNEL is not None and torch.cuda.is_available()
NEEDS_ALL = "needs CUDA, the merged triton kernel and transformers"

BLOCK = 64
# 1000 tokens is deliberately not a multiple of 64, so the load-bearing test also
# exercises right padding and a leak cannot hide behind a conveniently sized
# input. The length matters for a second reason: at 500 tokens the default
# schedule keeps 33 of 36 causal blocks and there is barely any sparsity left to
# measure. 16 blocks brings that to 92 of 136.
TOKENS = 1000
NUM_BLOCKS = -(-TOKENS // BLOCK)

_CACHE: dict[str, object] = {}


def fixture():
    """distilgpt2 on CUDA in eval mode, plus 500 tokens of real text.

    Cached because loading dominates the runtime of this file and every test
    restores the model to exactly the state it found it in -- which is itself
    one of the things under test.
    """
    # the skipif marks do nothing when these functions are called directly by
    # the __main__ harness, so the gate has to live somewhere both paths reach
    if not (HAVE_GPU and HAVE_HF):
        pytest.skip(NEEDS_ALL)
    if "model" not in _CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
        _CACHE["model"] = model.to("cuda").eval()
        # real prose, not repeated filler: a schedule built from 0D persistence
        # over key-block centroids is only interesting if the blocks differ
        text = (
            "The history of computing is a history of abstraction. Each layer "
            "hides the one beneath it, and each hiding buys generality at the "
            "price of control. A programmer who never sees a cache line writes "
            "code that runs anywhere and fast nowhere. The machine underneath "
            "has not gone away; it has only been agreed not to mention it. "
            "Attention is the same bargain in miniature. A model that attends "
            "to everything learns everything and cannot be run, so the field "
            "spends its effort deciding what to forget. Locality is one answer, "
            "and a sink token is another, and both are guesses about where the "
            "information lives. Topology offers a third: measure how the keys "
            "cluster, keep the blocks that survive longest as the clustering "
            "threshold grows, and let the geometry of the sequence pick the "
            "budget rather than a hand-tuned window. Whether that guess is "
            "better than the others is an empirical question, and it is only "
            "answerable if the machinery underneath is exactly right. "
        ) * 8
        ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=TOKENS
        ).input_ids
        assert ids.shape[1] == TOKENS, f"corpus is too short: {ids.shape[1]} tokens"
        _CACHE["ids"] = ids.to("cuda")
    return _CACHE["model"], _CACHE["ids"]


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits.clone()


def dense_schedule():
    """The merged kernel's own lower-triangular CSR, over the padded blocks."""
    offsets, indices = KERNEL.build_dense_causal_block_schedule(NUM_BLOCKS)
    return offsets.cuda(), indices.cuda()


def max_abs(a, b):
    return (a - b).abs().max().item()


def kl_next_token(reference, other):
    """KL(reference || other) over the next-token distribution, in nats."""
    log_p = torch.log_softmax(reference[0, -1].double(), dim=-1)
    log_q = torch.log_softmax(other[0, -1].double(), dim=-1)
    return float((log_p.exp() * (log_p - log_q)).sum())


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_dense_schedule_matches_the_unpatched_model():
    """The wiring proof: dense schedule, so only arithmetic may differ.

    0.5 is the threshold and it comes from two measurements, not from taste.
    Correct wiring measures 2.750e-01, nearly all of which is TF32: `tl.dot`
    defaults to tf32 for fp32 inputs where torch's matmul does not, and rerunning
    this exact test under TRITON_F32_DEFAULT=ieee gives 3.967e-04 -- the same
    order as the 2.670e-04 separating transformers' own eager and sdpa backends.
    A wiring fault is nowhere near that band: with the mask registration dropped,
    a one-layer patch measures 1.342e+02. Nothing observed lands between 0.3 and
    100, so the threshold only has to sit in the gap.
    """
    model, ids = fixture()
    reference = logits(model, ids)

    with patch_attention(model, schedule=dense_schedule(), block_size=BLOCK) as patch:
        assert patch.active and patch.layers == list(range(model.config.n_layer))
        patched = logits(model, ids)

    difference = max_abs(patched, reference)
    print(f"    dense schedule: max abs logit diff {difference:.3e}")
    assert difference < 0.5, f"dense schedule is not the unpatched model: {difference}"


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_topology_schedule_costs_what_it_costs():
    """The sparsity cost, and a control proving it is sparsity and not plumbing.

    No threshold on the divergence itself -- dropping key blocks changes the
    output, and pretending otherwise would miss the entire point. The number
    that *is* asserted is the control: a local radius wide enough to reach block
    zero makes the topology builder emit exactly the dense causal CSR, so that
    run has to land on the dense run. If it does not, the schedule is not
    reaching the kernel the way it is meant to and the sparse number below
    measures a bug rather than a cost.
    """
    model, ids = fixture()
    reference = logits(model, ids)
    with patch_attention(model, schedule=dense_schedule(), block_size=BLOCK):
        dense = logits(model, ids)

    with patch_attention(model, block_size=BLOCK, local_radius_blocks=2,
                         sink_blocks=1, topk=4):
        sparse = logits(model, ids)
    # a radius spanning every block leaves the causal clip as the only filter,
    # which is the definition of the dense schedule
    with patch_attention(model, block_size=BLOCK, local_radius_blocks=NUM_BLOCKS,
                         sink_blocks=1, topk=0):
        control = logits(model, ids)

    control_gap = max_abs(control, dense)
    difference = max_abs(sparse, reference)
    divergence = kl_next_token(reference, sparse)
    print(
        f"    topology schedule (sink=1 local=2 topk=4 of {NUM_BLOCKS} blocks): "
        f"max abs logit diff {difference:.3e}, next-token KL {divergence:.4f} "
        f"nats; degenerate-dense control {control_gap:.3e}"
    )
    assert control_gap == 0.0, (
        f"a topology schedule that should be dense is not: {control_gap} -- the "
        f"builder's CSR is not reaching the kernel intact"
    )
    assert math.isfinite(difference) and math.isfinite(divergence)
    assert divergence >= 0.0, "KL cannot be negative"
    assert difference > 1e-3, (
        "the topology schedule is indistinguishable from dense, so this "
        "measures nothing -- check the schedule is actually dropping blocks"
    )


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_restore_is_bit_identical():
    """Not close: equal. A patch that leaves a trace invalidates every A/B after."""
    model, ids = fixture()
    before = logits(model, ids)
    implementation = model.config._attn_implementation

    patch = patch_attention(model, block_size=BLOCK)
    assert model.config._attn_implementation == _IMPL
    logits(model, ids)
    patch.restore()

    assert model.config._attn_implementation == implementation
    assert not any(hasattr(m, _PATCH_ATTR) for m in model.modules())
    after = logits(model, ids)
    assert torch.equal(after, before), f"restore left a trace: {max_abs(after, before)}"


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_a_subset_of_layers_isolates_one_layer():
    """`layers=` is how a regression gets bisected, so it has to bite per layer."""
    model, ids = fixture()
    reference = logits(model, ids)

    with patch_attention(model, block_size=BLOCK, topk=1, local_radius_blocks=0,
                         sink_blocks=1, layers=[0]) as patch:
        assert patch.layers == [0]
        patched = [m.layer_idx for m in model.modules() if hasattr(m, _PATCH_ATTR)]
        assert patched == [0], f"patched {patched}, wanted only layer 0"
        one = logits(model, ids)

    with patch_attention(model, block_size=BLOCK, topk=1, local_radius_blocks=0,
                         sink_blocks=1) as patch:
        assert len(patch.layers) == model.config.n_layer
        every = logits(model, ids)

    single, whole = max_abs(one, reference), max_abs(every, reference)
    print(f"    layers=[0] {single:.3e} vs all layers {whole:.3e}")
    assert single > 0, "patching layer 0 changed nothing"
    # deliberately not `whole > single`. At 1000 tokens that holds (1.930e+02
    # for layer 0 alone against 2.124e+02 for all six) but at 500 it inverts
    # (1.930e+02 against 1.751e+02): a badly starved layer 0 corrupts a residual
    # stream that later starved layers sometimes pull back. Max abs logit
    # difference is not monotone in how much of the model is patched, and
    # asserting that it is would be asserting an intuition over a measurement.
    assert max_abs(one, every) > 0, "layers= selected the same computation twice"


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_padding_falls_back_rather_than_ignoring_the_mask():
    """A padded batch has a mask the kernel cannot express, so it must not run.

    The kernel masks `k_pos <= offs_m` and nothing else, so pad tokens would be
    attended to as ordinary keys. `_plain_causal` sends those calls to eager,
    and what that buys is this: the patched logits stay within the eager/sdpa
    gap of the unpatched ones instead of drifting by O(1).
    """
    model, ids = fixture()
    padded = torch.cat([ids, ids], dim=0)
    mask = torch.ones_like(padded)
    mask[1, -37:] = 0  # right-pad the second row

    with torch.no_grad():
        reference = model(padded, attention_mask=mask).logits.clone()
    with (
        patch_attention(model, block_size=BLOCK, topk=1, local_radius_blocks=0),
        torch.no_grad(),
    ):
        patched = model(padded, attention_mask=mask).logits

    difference = max_abs(patched, reference)
    print(f"    padded batch: max abs logit diff {difference:.3e} (eager fallback)")
    assert difference < 1e-2, (
        f"a padded batch reached the kernel and its mask was dropped: {difference}"
    )


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_double_patch_refuses_and_leaves_the_first_one_working():
    """Saving `_previous` off a patched config would strand the model forever."""
    model, ids = fixture()
    original = model.config._attn_implementation

    patch = patch_attention(model, block_size=BLOCK)
    try:
        patch_attention(model, block_size=BLOCK)
    except ValueError as exc:
        assert "already patched" in str(exc)
    else:
        patch.restore()
        raise AssertionError("expected a refusal for a double patch")

    assert model.config._attn_implementation == _IMPL
    logits(model, ids)  # the first patch still runs
    patch.restore()
    assert model.config._attn_implementation == original


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_restore_twice_is_a_noop():
    model, ids = fixture()
    before = logits(model, ids)
    original = model.config._attn_implementation

    patch = patch_attention(model, block_size=BLOCK)
    patch.restore()
    patch.restore()

    assert not patch.active
    assert model.config._attn_implementation == original
    assert torch.equal(logits(model, ids), before)


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_context_manager_restores_through_an_exception():
    model, _ = fixture()
    original = model.config._attn_implementation
    try:
        with patch_attention(model, block_size=BLOCK) as patch:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not patch.active
    assert model.config._attn_implementation == original
    assert not any(hasattr(m, _PATCH_ATTR) for m in model.modules())


@pytest.mark.skipif(not (HAVE_GPU and HAVE_HF), reason=NEEDS_ALL)
def test_unknown_layer_indices_are_refused():
    model, _ = fixture()
    original = model.config._attn_implementation
    try:
        patch_attention(model, layers=[0, 99])
    except ValueError as exc:
        assert "99" in str(exc)
    else:
        raise AssertionError("expected a refusal for a layer that does not exist")
    # a refused patch must not have half-applied
    assert model.config._attn_implementation == original
    assert not any(hasattr(m, _PATCH_ATTR) for m in model.modules())


@pytest.mark.skipif(torch is None or not HAVE_HF, reason="needs torch and transformers")
def test_a_model_without_gpt2_attention_is_refused():
    """Silently patching nothing would make an ablation report a false null."""
    if torch is None or not HAVE_HF:
        pytest.skip("needs torch and transformers")
    try:
        patch_attention(torch.nn.Linear(4, 4))
    except ValueError as exc:
        assert "GPT2Attention" in str(exc)
    else:
        raise AssertionError("expected a refusal for a model with no GPT-2 attention")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        # pytest.skip raises from BaseException, so the clauses below miss it;
        # a skipped GPU test on a CPU box is a pass, not an error
        except pytest.skip.Exception as exc:
            print(f"  SKIP  {name}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all green' if not failures else f'{failures} failing'}")
    sys.exit(1 if failures else 0)

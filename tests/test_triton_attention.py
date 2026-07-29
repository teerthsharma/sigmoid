"""Runnable checks for the attention bridge. `python tests/test_triton_attention.py`.

Three numbers carry the whole module, and all three are measured rather than
asserted into existence:

    the kernel against the merged torch reference, over one identical schedule
    sigmoid's CSR against the merged builder's, bit for bit
    a dense causal schedule against scaled_dot_product_attention(is_causal=True)

The third is what proves this is a real causal attention rather than a
self-consistent pair of wrong implementations: it compares against something
neither half of the topology path wrote.

Skips are imperative rather than `@pytest.mark.skipif` because this file runs
under two harnesses -- pytest and the `__main__` block below -- and a mark only
fires under one of them. Anything needing CUDA or triton skips rather than
fails, so a CPU-only machine stays green. The merged kernel is found via
`SIGMOID_KERNELS_PATH`, falling back to the usual checkout locations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# the kernels repo ships no installable distribution, so it is importable only
# with its root on the path. SIGMOID_KERNELS_PATH is the portable knob; these
# two are where a checkout usually lands.
if "SIGMOID_KERNELS_PATH" not in os.environ:
    for guess in (
        Path(__file__).resolve().parents[2] / "kernels",
        Path.home() / "Documents" / "GitHub" / "kernels",
    ):
        if (guess / "kernels" / "topology_sparse_attention.py").exists():
            os.environ["SIGMOID_KERNELS_PATH"] = str(guess)
            break

from sigmoid.triton.attention import (  # noqa: E402
    _merged,
    schedule_from_keys,
    topology_attention,
)

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

# `_merged` is the module's own triton probe -- it imports the merged kernel,
# which imports triton at module scope. Reusing it gates these tests on exactly
# what the library gates on, not on a second guess about availability.
KERNEL = _merged()
HAVE_TORCH = torch is not None
HAVE_GPU = HAVE_TORCH and KERNEL is not None and torch.cuda.is_available()

NEEDS_TORCH = "PyTorch is not installed"
NEEDS_KERNEL = "the merged triton kernel is not importable"
NEEDS_GPU = "needs CUDA and the merged triton kernel"


def require(condition, reason):
    if not condition:
        pytest.skip(reason)


def keys(shape, dtype=None, device="cpu", seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype or torch.float32).to(
        device
    )


def dense_causal_csr(num_blocks):
    """Lower-triangular CSR -- the object `build_dense_causal_block_schedule`
    returns, inline so the CPU checks do not need the merged module."""
    return (
        np.cumsum(np.arange(num_blocks + 1)),
        np.concatenate([np.arange(i + 1) for i in range(num_blocks)]),
    )


# ---- the three numbers ---------------------------------------------------


def test_kernel_matches_the_torch_reference_on_one_schedule():
    """Kernel against the merged reference, same CSR, so only arithmetic differs."""
    require(HAVE_GPU, NEEDS_GPU)
    print()
    worst = {}
    for shape in ((1024, 64), (2048, 32), (2, 3, 512, 64), (1, 4, 256, 128)):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            q, k, v = (keys(shape, dtype, "cuda", seed=s) for s in (0, 1, 2))
            schedule = schedule_from_keys(k, block_size=64)
            kernel = topology_attention(
                q, k, v, block_size=64, schedule=schedule, backend="triton"
            )
            reference = topology_attention(
                q, k, v, block_size=64, schedule=schedule, backend="torch"
            )
            diff = (kernel.float() - reference.float()).abs().max().item()
            worst[dtype] = max(worst.get(dtype, 0.0), diff)
            print(f"        {str(shape):<18} {str(dtype):<16} max abs diff {diff:.3e}")
    # fp32 is looser than fp16 here, and not for the obvious reason: tl.dot
    # defaults to tf32 on Ampere and later, so the kernel carries ~10 mantissa
    # bits where the reference carries 24. Measured worst case across the
    # matrix above: 2.4e-03 fp32, 1.6e-02 bf16, 2.0e-03 fp16 -- each about one
    # ulp of its own format at these magnitudes. The bounds leave headroom
    # because a different GPU picks different tl.dot lowering.
    for dtype, bound in (
        (torch.float16, 5e-3),
        (torch.bfloat16, 4e-2),
        (torch.float32, 1e-2),
    ):
        assert worst[dtype] < bound, f"{dtype}: {worst[dtype]:.3e} exceeds {bound:.0e}"


def test_schedule_is_bit_identical_to_the_merged_builder():
    """np.array_equal on offsets and indices, not allclose.

    The CSR is what the kernel consumes, so one differing block index is a
    different attention pattern, not a rounding difference. sigmoid reaches it
    through n-1 MST edges rather than all n(n-1)/2 pairs; the bar is that both
    land on the same integers.
    """
    require(HAVE_TORCH and KERNEL is not None, NEEDS_KERNEL)
    device = "cuda" if HAVE_GPU else "cpu"
    print()
    for shape, dtype in (
        ((1024, 64), torch.float16),
        ((1024, 64), torch.float32),
        ((2048, 64), torch.float16),
        ((512, 128), torch.float32),
        ((4096, 64), torch.float16),
    ):
        k = keys(shape, dtype, device, seed=3)
        for block_size, radius, sinks, topk in (
            (64, 2, 1, 4),
            (64, 0, 0, 0),
            (128, 1, 2, 8),
            (64, 3, 1, 64),
        ):
            ours = schedule_from_keys(
                k,
                block_size=block_size,
                local_radius_blocks=radius,
                sink_blocks=sinks,
                topk=topk,
            )
            theirs = KERNEL.build_topology_block_schedule(
                k, block_size, radius, sinks, topk
            )
            for name, a, b in zip(("offsets", "indices"), ours, theirs):
                assert np.array_equal(a.cpu().numpy(), b.cpu().numpy()), (
                    f"{name} differ at {shape} {dtype} block_size={block_size} "
                    f"radius={radius} sinks={sinks} topk={topk}"
                )
        print(f"        {str(shape):<12} {str(dtype):<16} 4 configs bit-identical")


def test_dense_schedule_matches_scaled_dot_product_attention():
    """The outside check: a full causal schedule must reproduce torch's SDPA.

    Every other test here compares the topology path against itself or against
    its own reference. This one compares it against an implementation neither
    half wrote, which is the only way to catch both halves being wrong the same
    way.
    """
    require(HAVE_GPU, NEEDS_GPU)
    print()
    for shape in ((1024, 64), (2, 3, 512, 64)):
        for dtype in (torch.float16, torch.float32):
            q, k, v = (keys(shape, dtype, "cuda", seed=s) for s in (4, 5, 6))
            schedule = KERNEL.build_dense_causal_block_schedule(shape[-2] // 64)
            ours = topology_attention(
                q, k, v, block_size=64, schedule=schedule, backend="triton"
            )
            flat = [t.reshape(-1, *shape[-2:]) for t in (q, k, v)]
            sdpa = torch.nn.functional.scaled_dot_product_attention(
                *flat, is_causal=True
            ).reshape(q.shape)
            diff = (ours.float() - sdpa.float()).abs().max().item()
            print(f"        {str(shape):<18} {str(dtype):<16} vs SDPA {diff:.3e}")
            # fp32 rides the same tf32 tl.dot as above, so it is the loose one
            assert diff < (5e-3 if dtype is torch.float16 else 1e-2), (
                f"dense schedule is not causal attention at {shape} {dtype}: {diff:.3e}"
            )


# ---- the CPU-only path ---------------------------------------------------


def test_torch_backend_is_a_real_causal_attention_without_a_gpu():
    """The same SDPA check on the CPU path, verified where it actually runs.

    This is the one that matters for a machine with neither CUDA nor triton:
    the library has to stay correct there, not merely importable.
    """
    require(HAVE_TORCH, NEEDS_TORCH)
    for shape in ((256, 64), (2, 2, 128, 32)):
        q, k, v = (keys(shape, torch.float32, "cpu", seed=s) for s in (7, 8, 9))
        ours = topology_attention(
            q,
            k,
            v,
            block_size=64,
            schedule=dense_causal_csr(shape[-2] // 64),
            backend="torch",
        )
        flat = [t.reshape(-1, *shape[-2:]) for t in (q, k, v)]
        sdpa = torch.nn.functional.scaled_dot_product_attention(
            *flat, is_causal=True
        ).reshape(q.shape)
        diff = (ours - sdpa).abs().max().item()
        assert diff < 1e-5, f"cpu torch backend is not causal at {shape}: {diff:.3e}"


def test_auto_backend_falls_back_to_torch_on_cpu():
    """No CUDA means no kernel, and "auto" has to notice without being asked."""
    require(HAVE_TORCH, NEEDS_TORCH)
    q, k, v = (keys((256, 64), torch.float32, "cpu", seed=s) for s in (0, 1, 2))
    out = topology_attention(q, k, v, block_size=64)
    explicit = topology_attention(q, k, v, block_size=64, backend="torch")
    assert torch.equal(out, explicit), "auto did not pick the torch backend on cpu"
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.isfinite(out).all(), "cpu attention produced non-finite values"


def test_the_schedule_is_actually_sparse():
    """Guards every other check: a schedule that quietly went dense passes them all."""
    require(HAVE_TORCH, NEEDS_TORCH)
    k = keys((4096, 64), torch.float32, "cpu", seed=11)
    offsets, indices = schedule_from_keys(
        k, block_size=64, local_radius_blocks=2, sink_blocks=1, topk=4
    )
    offsets, indices = offsets.numpy(), indices.numpy()
    num_blocks = 4096 // 64
    density = indices.size / (num_blocks * (num_blocks + 1) / 2)
    assert density < 0.5, f"schedule is {density:.0%} of causal dense, not sparse"
    assert np.diff(offsets).min() >= 1, "a query block was left with no key blocks"
    for q_block in range(num_blocks):
        row = indices[offsets[q_block] : offsets[q_block + 1]]
        assert row.max() <= q_block, f"query block {q_block} attends to {row.max()}"


def test_schedule_from_keys_returns_the_flavour_it_was_given():
    """Torch in, torch on the keys' device; numpy in, numpy.

    An inference loop builds this once per step and hands it to every layer, so
    a forced round trip through the host would be a per-layer transfer for
    nothing.
    """
    require(HAVE_TORCH, NEEDS_TORCH)
    k = keys((512, 64), torch.float32, "cpu", seed=12)
    from_torch = schedule_from_keys(k, block_size=64)
    from_numpy = schedule_from_keys(k.numpy(), block_size=64)
    assert all(torch.is_tensor(t) for t in from_torch)
    assert all(isinstance(a, np.ndarray) and a.dtype == np.int64 for a in from_numpy)
    for t, a in zip(from_torch, from_numpy):
        assert np.array_equal(t.numpy(), a), "numpy and torch keys disagreed"
    if HAVE_GPU:
        assert all(t.is_cuda for t in schedule_from_keys(k.cuda(), block_size=64)), (
            "schedule left the keys' device"
        )


def test_four_dimensional_keys_fold_by_concatenating_heads():
    """One CSR is shared across batch and heads, so the builder folds
    [batch, heads, seq, dim] to one point per block -- by concatenation.

    Averaging heads instead would cancel here by construction: head 1 is the
    negation of head 0, so their mean is exactly zero at every position, every
    centroid coincides and the salience collapses. Concatenation gives the same
    schedule as laying the heads side by side in the feature dimension, which
    is what this asserts.
    """
    require(HAVE_TORCH, NEEDS_TORCH)
    k = keys((1, 2, 512, 8), torch.float32, "cpu", seed=13)
    k[0, 1] = -k[0, 0]
    folded = schedule_from_keys(k, block_size=64, local_radius_blocks=0, sink_blocks=0,
                                topk=3)
    side_by_side = schedule_from_keys(
        torch.cat([k[0, 0], k[0, 1]], dim=1),
        block_size=64,
        local_radius_blocks=0,
        sink_blocks=0,
        topk=3,
    )
    for name, a, b in zip(("offsets", "indices"), folded, side_by_side):
        assert np.array_equal(a.numpy(), b.numpy()), f"4D fold changed {name}"

    batched = schedule_from_keys(keys((2, 3, 512, 64), seed=14), block_size=64)
    assert batched[0].numel() == 512 // 64 + 1
    assert int(batched[0][-1]) == batched[1].numel()


# ---- refusals ------------------------------------------------------------


def test_refuses_mismatched_or_unsupported_inputs():
    """Each message names the argument at fault, because the kernel's own
    failure for most of these is either a triton CompilationError pointing at
    generated code or nothing at all."""
    require(HAVE_TORCH, NEEDS_TORCH)
    q = keys((256, 64), torch.float32, "cpu", seed=0)
    wide = keys((256, 48))
    short = keys((200, 64))
    cases = [
        ("k has shape", lambda: topology_attention(q, keys((256, 32)), q)),
        ("v has shape", lambda: topology_attention(q, q, keys((128, 64)))),
        ("k has dtype", lambda: topology_attention(q, q.half(), q)),
        ("block_size must be", lambda: topology_attention(q, q, q, block_size=48)),
        ("multiple of block_size", lambda: topology_attention(short, short, short)),
        ("head dim 48", lambda: topology_attention(wide, wide, wide)),
        ("float64", lambda: topology_attention(*(q.double(),) * 3)),
        ("backend must be", lambda: topology_attention(q, q, q, backend="cuda")),
        ("[seq, dim]", lambda: topology_attention(*(keys((2, 256, 64)),) * 3)),
    ]
    for expected, call in cases:
        try:
            call()
        except (ValueError, RuntimeError) as exc:
            assert expected in str(exc), (
                f"expected a message naming {expected!r}, got: {exc}"
            )
        else:
            raise AssertionError(f"expected a refusal mentioning {expected!r}")


def test_refuses_a_broken_caller_supplied_schedule():
    """The kernel reads out of bounds on a stray block index and returns NaN on
    an empty row, in both cases without raising, so these are caught here."""
    require(HAVE_TORCH, NEEDS_TORCH)
    q = keys((256, 64), torch.float32, "cpu", seed=0)
    offsets, indices = dense_causal_csr(4)
    cases = [
        ("expected num_blocks + 1", (offsets[:-1], indices)),
        ("outside [0, 4)", (offsets, np.full_like(indices, 9))),
        ("no key blocks", (np.array([0, 0, 3, 6, 10]), indices)),
        ("must be 1D", (offsets.reshape(1, -1), indices)),
        ("must be integer", (offsets.astype(np.float64), indices)),
        ("0..len(indices)", (np.array([0, 1, 3, 6, 11]), indices)),
    ]
    for expected, schedule in cases:
        try:
            topology_attention(
                q, q, q, block_size=64, schedule=schedule, backend="torch"
            )
        except ValueError as exc:
            assert expected in str(exc), (
                f"expected a message naming {expected!r}, got: {exc}"
            )
        else:
            raise AssertionError(f"expected a refusal mentioning {expected!r}")


def test_triton_backend_refuses_cleanly_without_a_gpu():
    """Asking for the kernel on CPU tensors is a caller error, not a fallback."""
    require(HAVE_TORCH, NEEDS_TORCH)
    q = keys((256, 64), torch.float32, "cpu", seed=0)
    try:
        topology_attention(q, q, q, block_size=64, backend="triton")
    except (ValueError, RuntimeError) as exc:
        assert "triton" in str(exc), f"unhelpful refusal: {exc}"
    else:
        raise AssertionError("expected a refusal for backend='triton' on cpu")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        # pytest.skip raises from BaseException, so the clauses below miss it;
        # an unavailable GPU is a skip, not an error
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

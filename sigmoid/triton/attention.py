"""Attention over topology block schedules, wired to the merged Triton kernel.

`triton-lang/kernels#22` ships both halves of a topology-sparse attention: a
builder that turns key-block centroids into a causal CSR schedule, and a Triton
kernel that walks it. `sigmoid.triton.schedule` already replaces the first half
-- bit-identical salience from n-1 MST edges instead of all n(n-1)/2 pairs, 13x
in batch and 8.6-21.7x per decode step -- so this module is the join: sigmoid's
schedule, the merged kernel's arithmetic.

    from sigmoid.triton.attention import topology_attention, schedule_from_keys

    out = topology_attention(q, k, v, block_size=64)          # builds a schedule
    csr = schedule_from_keys(k, block_size=64)                # or build it once
    out = topology_attention(q, k, v, schedule=csr)           # and reuse per layer

Measured on an RTX 4060 Laptop (torch 2.5.1+cu121, triton 3.7.1) over four
shapes from [1024, 64] to [2, 3, 512, 64], block_size=64, on the schedule
sigmoid builds -- see `tests/test_triton_attention.py`:

    kernel vs the merged torch reference   9.8e-04 - 2.0e-03  fp16
                                           1.7e-03 - 2.4e-03  fp32 (tf32 tl.dot)
                                           7.8e-03 - 1.6e-02  bf16
    sigmoid's CSR vs the merged builder's  bit-identical, 20 configurations
    dense causal schedule vs SDPA          9.8e-04            fp16

2.0e-03 is one fp16 ulp at these magnitudes and matches the merged kernel's own
published baseline, so the sparse path sits as close to the reference as the
reference sits to torch's dense causal attention.

The kernel is reached lazily and `sigmoid.triton.__init__` does not import this
module, so `import sigmoid` still works with neither torch nor triton present.
The `backend="torch"` path runs on a CPU-only machine with no triton at all.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from functools import lru_cache
from typing import Any

import numpy as np

from ..adapters import TorchUnavailable
from .schedule import build_topology_block_schedule

__all__ = ["topology_attention", "schedule_from_keys"]

_BACKENDS = ("auto", "triton", "torch")
# HEAD_DIM is a constexpr in the kernel and only these four are wired up; a
# head dim of 48 comes back as ValueError from the kernel itself
_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


@lru_cache(maxsize=1)
def _merged() -> Any:
    """The merged kernel module, or None when it cannot be imported.

    It does `import triton` at module scope, so this doubles as the triton
    probe: on a CPU-only box the import fails and every backend decision falls
    through to the transcription below. `SIGMOID_KERNELS_PATH` points at a
    `triton-lang/kernels` checkout, which is the normal case -- that repo ships
    no installable distribution, so the module is only importable if its root
    is already on the path.
    """
    root = os.environ.get("SIGMOID_KERNELS_PATH")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module("kernels.topology_sparse_attention")
    except ImportError:
        return None


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TorchUnavailable("topology_attention requires PyTorch") from exc
    return torch


# ---- schedule ------------------------------------------------------------


def _pool(arr: Any, block_size: int) -> Any:
    """[seq, dim] or [batch, heads, seq, dim] -> [batch, heads, num_blocks, dim].

    Indexing, reshape and mean all read the same on a torch tensor and a numpy
    array, so the shape checks live here once for both.
    """
    if arr.ndim == 2:
        arr = arr[None, None]
    if arr.ndim != 4:
        raise ValueError(
            f"keys must be [seq, dim] or [batch, heads, seq, dim], "
            f"got {tuple(arr.shape)}"
        )
    batch, heads, seq, dim = arr.shape
    if seq % block_size:
        raise ValueError(
            f"sequence length {seq} is not a multiple of block_size {block_size}"
        )
    return arr.reshape(batch, heads, seq // block_size, block_size, dim).mean(3)


def _block_centroids(keys: Any, block_size: int) -> np.ndarray:
    """Mean-pool key blocks into one float64 point per block.

    Heads are *concatenated*, not averaged. The kernel shares one CSR across
    batch and heads, so blocks have to be compared in the joint key state, and
    averaging heads first can cancel: one head separating blocks i and j along
    +x while another separates them along -x averages to coincident centroids
    and the pair merges at distance 0. Concatenating makes the distance the
    root-sum-square of the per-head distances, which cannot cancel.

    Pooling happens before the host copy, and on the torch side in the keys'
    own dtype. That is a memory decision, not a numerical one -- the flattened
    result is block_size times smaller than the keys, where converting first
    would put a float64 copy of the whole KV tensor on the host. Numerically it
    costs nothing measurable: over 300 clustered-fp16 trials a float64 mean and
    the reference's fp16 mean gave the same CSR on 298, and both disagreements
    traced to an 8.9e-16 gap between torch.cdist and scipy.cdist amplified by
    near-tied merge heights, not to the pooling.
    """
    if hasattr(keys, "detach"):  # torch tensor
        torch = _torch()
        pooled = _pool(keys.detach(), block_size)
        flat = pooled.permute(2, 0, 1, 3).reshape(pooled.shape[2], -1)
        return flat.to("cpu", torch.float64).numpy()

    pooled = _pool(np.asarray(keys, dtype=np.float64), block_size)
    return pooled.transpose(2, 0, 1, 3).reshape(pooled.shape[2], -1)


def schedule_from_keys(
    k: Any,
    *,
    block_size: int = 64,
    local_radius_blocks: int = 2,
    sink_blocks: int = 1,
    topk: int = 4,
) -> tuple[Any, Any]:
    """The causal CSR `(offsets, indices)` for `k`, without running attention.

    An inference loop builds this once and hands it to `topology_attention` for
    every layer and head, which is the whole point of the kernel sharing one
    schedule: the salience is over key *positions*, and rebuilding it per layer
    pays the MST again for the same answer.

    Torch keys give torch int64 tensors on the keys' device, numpy keys give
    numpy int64 arrays -- so the result drops straight back into whichever
    world it came from with no transfer.
    """
    centroids = _block_centroids(k, block_size)
    # block_size=1 because `centroids` is already pooled: sigmoid's builder
    # then mean-pools one row per block, which is the identity in float64
    offsets, indices = build_topology_block_schedule(
        centroids, 1, local_radius_blocks, sink_blocks, topk
    )
    if hasattr(k, "detach"):
        torch = _torch()
        return (
            torch.as_tensor(offsets, device=k.device),
            torch.as_tensor(indices, device=k.device),
        )
    return offsets, indices


# ---- validation ----------------------------------------------------------


def _validate(torch: Any, q: Any, k: Any, v: Any, block_size: int) -> None:
    """Gate both backends identically, before either gets a chance.

    The kernel checks that q, k and v share a shape and live on CUDA, but not
    that they agree on dtype or device, and its head-dim and block-size checks
    only fire on the triton path. Without the same gate on the torch path, code
    developed on a CPU-only machine is accepted there and rejected on the GPU,
    which is the failure this wrapper exists to prevent.
    """
    if q.ndim not in (2, 4):
        raise ValueError(
            f"q must be [seq, dim] or [batch, heads, seq, dim], got {tuple(q.shape)}"
        )
    for name, t in (("k", k), ("v", v)):
        if tuple(t.shape) != tuple(q.shape):
            raise ValueError(
                f"{name} has shape {tuple(t.shape)}, expected q's {tuple(q.shape)}"
            )
        if t.dtype != q.dtype:
            raise ValueError(f"{name} has dtype {t.dtype}, expected q's {q.dtype}")
        if t.device != q.device:
            raise ValueError(f"{name} is on device {t.device}, expected q's {q.device}")

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a positive power of two, got {block_size}")
    seq, head_dim = q.shape[-2:]
    if seq % block_size:
        raise ValueError(
            f"sequence length {seq} is not a multiple of block_size {block_size}"
        )
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head dim {head_dim} is not one of {_SUPPORTED_HEAD_DIMS}; the kernel "
            f"takes HEAD_DIM as a constexpr and compiles only those"
        )
    # tl.dot has no fp64 path: float64 q reaches the kernel and dies with a
    # triton CompilationError pointing at `tl.full((BLOCK_M,), -inf, fl...`,
    # which names neither the argument nor the dtype
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"q has dtype {q.dtype}; the kernel supports float16, bfloat16 and "
            f"float32 only (tl.dot has no float64 path)"
        )


def _validate_schedule(offsets: Any, indices: Any, num_blocks: int) -> None:
    """Reject a caller-supplied CSR the kernel would read past or NaN on.

    Two failures the kernel does not catch, both measured: a block index
    outside [0, num_blocks) loads memory outside k and v, because the only mask
    on the key tile is `k_pos < seq` and a negative index passes it; and a
    query block with no key blocks leaves l_i at zero, so the row comes back as
    acc/0 -- NaN, silently, with no error anywhere.

    Only caller-supplied schedules come through here. The ones this module
    builds are correct by construction, and checking them would force a device
    sync on every call.
    """
    off, idx = (np.asarray(_host(a)) for a in (offsets, indices))
    for name, arr in (("offsets", off), ("indices", idx)):
        if arr.ndim != 1:
            raise ValueError(f"schedule {name} must be 1D, got shape {arr.shape}")
        if arr.dtype.kind not in "iu":
            raise ValueError(f"schedule {name} must be integer, got {arr.dtype}")
    if off.size != num_blocks + 1:
        raise ValueError(
            f"schedule offsets has {off.size} entries, expected num_blocks + 1 "
            f"= {num_blocks + 1}"
        )
    if off[0] != 0 or off[-1] != idx.size:
        raise ValueError(
            f"schedule offsets must run 0..len(indices), got {off[0]}..{off[-1]} "
            f"with {idx.size} indices"
        )
    empty = np.flatnonzero(np.diff(off) <= 0)
    if empty.size:
        raise ValueError(
            f"schedule leaves query block {empty[0]} with no key blocks, which "
            f"the kernel returns as NaN"
        )
    if idx.size and (idx.min() < 0 or idx.max() >= num_blocks):
        raise ValueError(
            f"schedule indices span [{idx.min()}, {idx.max()}], outside "
            f"[0, {num_blocks})"
        )


def _host(x: Any) -> Any:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


def _resolve_backend(backend: str, q: Any) -> str:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend == "auto":
        return "triton" if q.is_cuda and _merged() is not None else "torch"
    if backend == "triton":
        if _merged() is None:
            raise RuntimeError(
                "backend='triton' needs the merged kernel importable: install "
                "triton and point SIGMOID_KERNELS_PATH at a triton-lang/kernels "
                "checkout"
            )
        if not q.is_cuda:
            raise ValueError(f"backend='triton' needs CUDA tensors; q is on {q.device}")
    return backend


# ---- attention -----------------------------------------------------------


def _reference_attention(q: Any, k: Any, v: Any, offsets: Any, indices: Any, block: int) -> Any:
    """`dense_masked_attention` transcribed, for machines without triton.

    The merged module imports triton at module scope, so on a CPU-only box even
    its pure-torch reference is unreachable. Twelve lines is cheaper than making
    the library's torch path conditional on a GPU compiler being installed.
    Used only when `_merged()` is None; where the real one is importable, it is
    the one that runs.
    """
    torch = _torch()
    allow = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.bool, device=q.device)
    for q_block in range(q.shape[0] // block):
        rows = slice(q_block * block, (q_block + 1) * block)
        for b in indices[int(offsets[q_block]) : int(offsets[q_block + 1])]:
            allow[rows, int(b) * block : (int(b) + 1) * block] = True
    positions = torch.arange(q.shape[0], device=q.device)
    allow &= positions[None, :] <= positions[:, None]
    logits = (q @ k.T) / math.sqrt(q.shape[1])
    return torch.softmax(logits.masked_fill(~allow, float("-inf")), dim=-1) @ v


def _dense_attention(
    torch: Any, q: Any, k: Any, v: Any, offsets: Any, indices: Any, block_size: int
) -> Any:
    """The merged torch reference, looped over batch and heads.

    `dense_masked_attention` is 2D only: it builds a [seq, seq] mask and does
    `q @ k.T`, and on a 4D tensor `.T` reverses all four dims and raises a
    broadcast error rather than doing something wrong quietly. The kernel
    shares one CSR across batch and heads, so looping is the faithful reading
    of what the triton path computes.
    """
    merged = _merged()
    reference = merged.dense_masked_attention if merged else _reference_attention
    seq, head_dim = q.shape[-2:]
    qf, kf, vf = (t.reshape(-1, seq, head_dim) for t in (q, k, v))
    rows = [
        reference(qf[i], kf[i], vf[i], offsets, indices, block_size)
        for i in range(qf.shape[0])
    ]
    return torch.stack(rows).reshape(q.shape)


def topology_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    block_size: int = 64,
    local_radius_blocks: int = 2,
    sink_blocks: int = 1,
    topk: int = 4,
    schedule: tuple[Any, Any] | None = None,
    backend: str = "auto",
) -> Any:
    """Causal attention over sink, local and topologically salient key blocks.

    Args:
        q, k, v: `[seq, dim]` or `[batch, heads, seq, dim]`, agreeing on shape,
            dtype and device. float16, bfloat16 or float32; head dim one of
            16, 32, 64, 128; seq a multiple of `block_size`.
        block_size: query and key block edge, a positive power of two.
        local_radius_blocks: how many blocks back the local window reaches. A
            radius, not a width -- the merged builder's convention.
        sink_blocks: leading blocks every query attends to.
        topk: how many of the most salient blocks to keep.
        schedule: a precomputed `(offsets, indices)`, e.g. from
            `schedule_from_keys`, reused across layers instead of rebuilt.
        backend: "triton" for the merged kernel, "torch" for the merged
            reference over the identical schedule, "auto" for triton whenever
            q is on CUDA and the kernel imports.

    Returns a tensor shaped and typed like `q`.
    """
    torch = _torch()
    _validate(torch, q, k, v, block_size)
    backend = _resolve_backend(backend, q)
    num_blocks = q.shape[-2] // block_size

    if schedule is None:
        schedule = schedule_from_keys(
            k,
            block_size=block_size,
            local_radius_blocks=local_radius_blocks,
            sink_blocks=sink_blocks,
            topk=topk,
        )
    else:
        offsets, indices = schedule
        _validate_schedule(offsets, indices, num_blocks)

    offsets, indices = (
        torch.as_tensor(a, dtype=torch.int64, device=q.device) for a in schedule
    )
    if backend == "torch":
        return _dense_attention(torch, q, k, v, offsets, indices, block_size)
    return _merged().scheduled_attention(q, k, v, offsets, indices, block_size)

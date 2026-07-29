"""Parity and speed against the merged kernel's schedule builder.

Checks sigmoid's MST-based salience against `_zero_dim_persistence_salience`
and `build_topology_block_schedule` from triton-lang/kernels#22 -- the actual
merged source, not a transcription -- and times both.

Triton itself is stubbed out: only the pure-Python schedule builders are under
test here, and they never touch the GPU. The attention kernel is untouched.

    python examples/kernel_schedule_parity.py [path/to/kernels]
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_KERNEL = Path(
    "C:/Users/seal/Documents/GitHub/kernels/kernels/topology_sparse_attention.py"
)


def load_reference(path: Path):
    """Import the merged module with triton stubbed to a permissive dummy."""

    class _Any:
        def __getattr__(self, _name):
            return _Any()

        def __call__(self, *_a, **_k):
            return _Any()

        def __getitem__(self, _i):
            return _Any()

    tl = types.ModuleType("triton.language")
    tl.__getattr__ = lambda _n: _Any()
    tr = types.ModuleType("triton")
    tr.jit = lambda f=None, **_k: (f if f else (lambda g: g))
    tr.cdiv = lambda a, b: -(-a // b)
    tr.language = tl
    tr.__getattr__ = lambda _n: _Any()
    sys.modules["triton"], sys.modules["triton.language"] = tr, tl

    spec = importlib.util.spec_from_file_location("merged_tsa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench(fn, *args, warmup: int = 2, reps: int = 5) -> float:
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    return (time.perf_counter() - t0) / reps * 1e3


def main() -> int:
    import torch

    from sigmoid.triton import (
        build_topology_block_schedule,
        zero_dim_persistence_salience,
    )

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_KERNEL
    if not path.exists():
        print(f"merged kernel not found at {path}")
        return 1
    ref = load_reference(path)
    print(f"reference: {path}\n")

    rng = np.random.default_rng(0)
    print(f"{'blocks':>7}{'max |diff|':>13}{'merged ms':>12}{'sigmoid ms':>12}{'speedup':>9}")
    print("-" * 53)
    worst = 0.0
    for num_blocks in (32, 64, 128, 256, 512):
        centroids = rng.normal(size=(num_blocks, 64))
        tensor = torch.tensor(centroids)
        theirs = ref._zero_dim_persistence_salience(tensor).numpy()
        ours = zero_dim_persistence_salience(centroids)
        diff = float(np.abs(theirs - ours).max())
        worst = max(worst, diff)
        t_ref = bench(ref._zero_dim_persistence_salience, tensor)
        t_our = bench(zero_dim_persistence_salience, centroids)
        print(
            f"{num_blocks:>7}{diff:>13.1e}{t_ref:>12.2f}{t_our:>12.2f}"
            f"{t_ref / max(t_our, 1e-9):>8.1f}x"
        )

    print(f"\nworst salience deviation: {worst:.1e} (float64 round-off)")

    print("\nfull CSR schedule parity:")
    ok = True
    for seq, block, radius, sink, topk in (
        (512, 64, 2, 1, 2),
        (1024, 64, 1, 1, 4),
        (2048, 128, 3, 2, 8),
    ):
        keys = rng.normal(size=(seq, 64))
        ref_off, ref_idx = ref.build_topology_block_schedule(
            torch.tensor(keys), block, radius, sink, topk
        )
        our_off, our_idx = build_topology_block_schedule(
            keys, block, radius, sink, topk
        )
        same = np.array_equal(ref_off.numpy(), our_off) and np.array_equal(
            ref_idx.numpy(), our_idx
        )
        ok &= same
        density = len(our_idx) / (len(our_off) - 1) ** 2 * 2
        print(
            f"  seq={seq:<5} block={block:<4} identical={str(same):<5} "
            f"selected={len(our_idx):<5} ~{density:.0%} of causal dense"
        )

    print(f"\n{'PARITY HOLDS' if ok else 'PARITY BROKEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

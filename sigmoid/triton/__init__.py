"""Bridge to the merged topology-sparse attention kernel.

`triton-lang/kernels#22` builds a causal CSR block schedule from a 0D-persistence
salience over key-block centroids. That salience is the single-linkage merge
height of each block -- the same H0 death time `sigmoid.state.h0_barcode`
computes -- so this package supplies the schedule from sigmoid's machinery and
holds bit-identical parity with the merged source.

The win is algorithmic. The reference sorts all n(n-1)/2 centroid pairs before
running union-find; single-linkage merges occur only along minimum-spanning-tree
edges, so n-1 edges carry the whole barcode. 13x in batch, and `IncrementalSalience`
grows the tree one block at a time for decode at 8.6-21.7x.

Three layers, each usable on its own:

    from sigmoid.triton import build_topology_block_schedule, IncrementalSalience
    from sigmoid.triton import topology_attention, schedule_from_keys
    from sigmoid.triton import patch_attention

    schedule    pure numpy, no torch, no GPU
    attention   sigmoid's schedule -> the merged kernel, with a torch fallback
    patch       swaps that attention into a HuggingFace model

**Importing this package pulls in no torch, no triton and no transformers.**
Every heavy dependency is imported lazily inside the function that needs it, so
`import sigmoid.triton` works on a CPU-only machine with numpy and scipy alone;
calling a GPU path without torch raises `TorchUnavailable` rather than failing
at import. Verified with torch, triton and kernels all blocked.

The merged kernel is not pip-installable, so the wrapper locates it via
`SIGMOID_KERNELS_PATH` pointing at a `triton-lang/kernels` checkout, falling
back to conventional locations.
"""

from .attention import schedule_from_keys, topology_attention
from .patch import AttentionPatch, patch_attention
from .schedule import (
    IncrementalSalience,
    block_centroids,
    build_topology_block_schedule,
    zero_dim_persistence_salience,
)

__all__ = [
    "AttentionPatch",
    "IncrementalSalience",
    "block_centroids",
    "build_topology_block_schedule",
    "patch_attention",
    "schedule_from_keys",
    "topology_attention",
    "zero_dim_persistence_salience",
]

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

    from sigmoid.triton import build_topology_block_schedule, IncrementalSalience

Nothing here imports triton. These are the pure-Python schedule builders; the
attention kernel itself lives upstream.
"""

from .schedule import (
    IncrementalSalience,
    block_centroids,
    build_topology_block_schedule,
    zero_dim_persistence_salience,
)

__all__ = [
    "IncrementalSalience",
    "block_centroids",
    "build_topology_block_schedule",
    "zero_dim_persistence_salience",
]

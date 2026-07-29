"""Block schedules for topology-derived sparse attention.

The merged Triton kernel (triton-lang/kernels#22) builds a causal CSR block
schedule from a 0D-persistence salience over key-block centroids: sink blocks,
a local window, and the top-k most salient blocks. The salience of a block is
the single-linkage merge height at which its component was absorbed -- which is
exactly that block's H0 death time, the same object `state.h0_barcode` returns.

Two implementations of one construction, so this module supplies the schedule
using sigmoid's machinery and asserts the outputs agree.

The reason to bother is algorithmic, not stylistic. The reference enumerates and
sorts all n(n-1)/2 centroid pairs in Python before running union-find. But
single-linkage merges only ever occur along edges of the minimum spanning tree,
so n-1 edges carry the entire barcode and the remaining pairs cannot change any
merge height. Building the MST first and running union-find over those n-1
edges gives bit-identical salience while sorting ~32x fewer edges at 64 blocks.

Schedule construction sits outside the attention kernel and is measured
separately, per the promotion rules this work follows.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "zero_dim_persistence_salience",
    "build_topology_block_schedule",
    "block_centroids",
]


def block_centroids(keys: np.ndarray, block_size: int) -> np.ndarray:
    """Mean-pool a [seq, dim] key tensor into [num_blocks, dim] centroids."""
    k = np.asarray(keys, dtype=np.float64)
    if k.ndim != 2:
        raise ValueError("keys must have shape [seq, dim]")
    if k.shape[0] % block_size:
        raise ValueError(
            f"sequence length {k.shape[0]} is not a multiple of block_size {block_size}"
        )
    return k.reshape(-1, block_size, k.shape[1]).mean(axis=1)


def zero_dim_persistence_salience(centroids: np.ndarray) -> np.ndarray:
    """Per-block H0 death time, via MST edges rather than all pairs.

    Kruskal over the complete graph and Kruskal over the MST produce the same
    merge sequence, because every edge the complete-graph run would accept is
    an MST edge and every edge it would reject closes a cycle either way. So
    the salience is identical and only n-1 edges need sorting.
    """
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("centroids must have shape [num_blocks, dim]")
    n = c.shape[0]
    if n == 0:
        raise ValueError("at least one block is required")
    if n == 1:
        return np.ones(1, dtype=np.float64)

    from scipy.sparse.csgraph import minimum_spanning_tree

    from .state import _pairwise

    mst = minimum_spanning_tree(_pairwise(c)).toarray()
    rows, cols = np.nonzero(mst)
    edges = sorted(
        ((float(mst[r, col]), int(r), int(col)) for r, col in zip(rows, cols)),
        key=lambda item: item[0],
    )

    parent = list(range(n))
    members: dict[int, set[int]] = {i: {i} for i in range(n)}
    salience = np.zeros(n, dtype=np.float64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for distance, left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        # the smaller component is the one absorbed, and it dies at `distance`
        if len(members[root_left]) > len(members[root_right]):
            root_left, root_right = root_right, root_left
        for block in members[root_left]:
            salience[block] = distance
        parent[root_left] = root_right
        members[root_right].update(members[root_left])
        del members[root_left]

    return salience


def build_topology_block_schedule(
    keys: np.ndarray,
    block_size: int,
    local_radius_blocks: int,
    sink_blocks: int,
    topk_topology_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal CSR (offsets, indices) over key blocks.

    Signature and semantics match `build_topology_block_schedule` in the merged
    kernel exactly, so this is a drop-in replacement -- same argument order,
    same local *radius* convention, same empty-set fallback, same int64 output.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    for name, value in (
        ("local_radius_blocks", local_radius_blocks),
        ("sink_blocks", sink_blocks),
        ("topk_topology_blocks", topk_topology_blocks),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    centroids = block_centroids(keys, block_size)
    num_blocks = centroids.shape[0]
    salience = zero_dim_persistence_salience(centroids)

    topk = min(max(topk_topology_blocks, 0), num_blocks)
    # match torch.topk's tie-breaking: it returns the lowest index first among
    # equal values, which is a stable sort on the negated array
    topological = (
        set(np.argsort(-salience, kind="stable")[:topk].tolist()) if topk else set()
    )

    offsets = [0]
    indices: list[int] = []
    for query_block in range(num_blocks):
        allowed = set(range(min(sink_blocks, num_blocks)))
        allowed.update(range(max(0, query_block - local_radius_blocks), query_block + 1))
        allowed.update(b for b in topological if b <= query_block)
        allowed = {b for b in allowed if b <= query_block}
        if not allowed:
            allowed.add(query_block)
        indices.extend(sorted(allowed))
        offsets.append(len(indices))

    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
    )


def _demo() -> None:
    """Equivalence against the merged reference, plus the edge-count saving."""
    rng = np.random.default_rng(0)
    for num_blocks, dim in ((8, 4), (32, 16), (64, 64)):
        centroids = rng.normal(size=(num_blocks, dim))
        ours = zero_dim_persistence_salience(centroids)

        # the reference construction, transcribed: all pairs, sorted, union-find
        pairs = sorted(
            (float(np.linalg.norm(centroids[i] - centroids[j])), i, j)
            for i in range(num_blocks)
            for j in range(i + 1, num_blocks)
        )
        parent = list(range(num_blocks))
        members = {i: {i} for i in range(num_blocks)}
        reference = np.zeros(num_blocks)

        def find(x, parent=parent):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for distance, left, right in pairs:
            rl, rr = find(left), find(right)
            if rl == rr:
                continue
            if len(members[rl]) > len(members[rr]):
                rl, rr = rr, rl
            for block in members[rl]:
                reference[block] = distance
            parent[rl] = rr
            members[rr].update(members[rl])
            del members[rl]

        assert np.allclose(ours, reference), (
            f"salience differs from the reference at {num_blocks} blocks"
        )
        saving = len(pairs) / max(num_blocks - 1, 1)
        print(
            f"  {num_blocks:>3} blocks: identical salience, "
            f"{len(pairs)} pairs vs {num_blocks - 1} MST edges ({saving:.0f}x fewer)"
        )

    offsets, indices = build_topology_block_schedule(
        rng.normal(size=(256, 32)),
        block_size=64,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=1,
    )
    assert offsets[0] == 0 and len(offsets) == 5
    assert offsets[-1] == len(indices)
    for q in range(4):  # causality: never attend to a future block
        assert all(b <= q for b in indices[offsets[q] : offsets[q + 1]])
        assert offsets[q + 1] > offsets[q], "every query block needs a key block"
    density = len(indices) / (4 * 5 / 2)
    print(f"  schedule: {len(indices)} selected blocks, {density:.0%} of causal dense")
    print("demo ok")


if __name__ == "__main__":
    _demo()

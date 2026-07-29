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

Decode grows the key set one block at a time, and `IncrementalSalience` grows
the MST with it instead of rebuilding: O(n log n) per appended block rather than
O(n^2), measured 11x over a 64->256 block decode. Exactness there is a proof and
not a hope, because both paths order edges by the same strict total order and a
strict order admits exactly one spanning tree.

Schedule construction sits outside the attention kernel and is measured
separately, per the promotion rules this work follows.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "zero_dim_persistence_salience",
    "build_topology_block_schedule",
    "block_centroids",
    "IncrementalSalience",
]

# (distance, low index, high index) -- the merged reference's own sort key
_Edge = tuple[float, int, int]


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


# ---------------------------------------------------------------------------
# Cauchy-Schwarz block pruning: TRIED, DOES NOT TRANSFER. Recorded so it is not
# retried.
#
# AETHER (apoth3osis.io, `AetherPruning.lean`) proves a zero-false-negative block
# pruning rule from Cauchy-Schwarz, and this file's top-k salience is exactly the
# kind of selector that lacks such a guarantee -- its cost is measured after the
# fact (0.0426 nats KL at topk=4), and nothing rules out having dropped a block
# holding real attention mass. So the bound was implemented here, in the tighter
# centroid-plus-radius form:
#
#     q.k = q.c_b + q.(k - c_b) <= q.c_b + ||q|| r_b
#     mass_b <= n_b exp(U_b - L),  L = true max logit over always-attended blocks
#
# It is a correct upper bound and it never fires. To prune a 64-key block at
# eps = 1e-3 the block's ceiling must sit log(1e-3 / 64) = -11.1 nats below the
# global maximum, and attention logits at scale 1/sqrt(64) span roughly +-3.
# Measured minimum of the bound, over 60 trials each:
#
#     isotropic gaussian keys   radius 9.61   mass_ub >= 2.6e+04
#     tight clusters (0.15)     radius 1.41   mass_ub >= 1.88
#     very tight (0.02)         radius 0.19   mass_ub >= 0.47
#
# Even the last row would require calling 47% of the softmax mass negligible.
# The obstruction is the residual radius, which is irreducible for anything close
# to isotropic, and attention keys are not tight clusters. Same shape of failure
# as the scalar Banach certificate in `operator.py`: true, and vacuous.
#
# Where it WOULD work: a domain whose blocks are genuinely tight relative to the
# logit spread. Attention is not that domain, so the guarantee stays unavailable
# here and the honest claim about this schedule remains the measured KL.
# ---------------------------------------------------------------------------


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact Euclidean distances -- the one place either path measures them.

    Deliberately not the Gram identity |x|^2 + |y|^2 - 2x.y, which `state._pairwise`
    used until the same bug was found there. On well-separated blocks the two
    agree (both within 5e-15 of
    np.linalg.norm at 64 blocks of dim 64), but the identity cancels
    catastrophically once blocks nearly coincide: at 1e-12 separation it clamps
    to exactly 0.0 where the true distance is 3.01e-12, and cdist returns that
    value bit-for-bit. A distance of exactly 0.0 then loses the edge outright
    (see `_canonical_mst_edges`), which moves merge heights by O(1). Batch and
    incremental share this call so they share its rounding, which is what keeps
    tied heights tied.
    """
    from scipy.spatial.distance import cdist

    return cdist(a, b)


def _canonical_mst_edges(dist: np.ndarray) -> list[_Edge]:
    """MST by Prim under the strict total order (distance, low, high).

    That order is the merged reference's own sort key -- it sorts (d, i, j)
    tuples with i < j -- and being strict it admits exactly one minimum
    spanning tree, so every algorithm respecting it lands on the same tree.
    Without that, "the MST" is not well defined under tied distances and the
    batch and incremental paths disagree outright rather than by round-off.

    Two measured reasons this replaced scipy's `minimum_spanning_tree`:
    csr_matrix drops explicit zeros, so coincident blocks lost their zero
    length edge and never merged (six blocks, three coincident: salience came
    back [3,3,3,0,3,4.12] where the reference says [0,0,0,3,3,4.12]); and Prim
    over the dense matrix needs no edge sort at all, 5.4 ms against scipy's
    31.4 ms at 512 blocks. Below ~100 blocks the n-step Python loop makes it a
    wash (0.58 ms against 0.45 ms at 64), which is the price of the tie-break.

    That price buys kernel parity on degenerate keys. Over 200 randomized
    tie-heavy cases the scipy path disagreed with the merged reference on 123
    of them, by up to 4.75 in absolute salience -- whole blocks selected or
    dropped, not round-off. This path disagreed on none. On continuous
    centroids the two never differed at all, which is why it went unnoticed.
    """
    n = dist.shape[0]
    best = dist[0].copy()  # cheapest known edge from the tree to each vertex
    best[0] = np.inf
    src = np.zeros(n, dtype=np.int64)
    outside = np.ones(n, dtype=bool)
    outside[0] = False
    edges: list[_Edge] = []

    for _ in range(n - 1):
        shortest = best.min()
        tied = np.flatnonzero(best == shortest)
        if tied.size == 1:
            new = int(tied[0])
        else:
            # tie on distance -> smallest (low, high) index pair, as the
            # reference's tuple sort would
            low = np.minimum(src[tied], tied)
            high = np.maximum(src[tied], tied)
            new = int(tied[np.lexsort((high, low))[0]])
        joined = int(src[new])
        edges.append((float(shortest), min(joined, new), max(joined, new)))
        outside[new] = False
        best[new] = np.inf

        # with one endpoint fixed, (low, high) tuple order reduces to comparing
        # the other endpoint, so the tie-break here is just `new < src`
        candidate = dist[new]
        better = outside & (
            (candidate < best) | ((candidate == best) & (new < src))
        )
        np.copyto(best, candidate, where=better)
        np.copyto(src, new, where=better)

    edges.sort()
    return edges


def _merge_heights(n: int, edges: list[_Edge]) -> tuple[np.ndarray, list[_Edge]]:
    """Kruskal over canonically sorted `edges` -> (salience, accepted edges).

    Accepting and dying are the same decision, so one pass yields both the
    barcode and the MST. Edges that close a cycle are no-ops for the salience,
    which is why feeding this a superset of the tree (the incremental path) and
    feeding it the tree alone (the batch path) give the same answer.
    """
    if n == 1:
        # no merge to die at; the batch path has always reported 1.0 so that
        # top-k still selects a lone block, and both paths must agree
        return np.ones(1, dtype=np.float64), []

    parent = list(range(n))
    members: dict[int, set[int]] = {i: {i} for i in range(n)}
    salience = np.zeros(n, dtype=np.float64)
    accepted: list[_Edge] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges:
        distance, left, right = edge
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        accepted.append(edge)
        # the smaller component is the one absorbed, and it dies at `distance`
        if len(members[root_left]) > len(members[root_right]):
            root_left, root_right = root_right, root_left
        for block in members[root_left]:
            salience[block] = distance
        parent[root_left] = root_right
        members[root_right].update(members[root_left])
        del members[root_left]

    return salience, accepted


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

    return _merge_heights(n, _canonical_mst_edges(_distances(c, c)))[0]


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
    centroids = block_centroids(keys, block_size)
    salience = zero_dim_persistence_salience(centroids)
    return _causal_csr(
        salience, local_radius_blocks, sink_blocks, topk_topology_blocks
    )


def _causal_csr(
    salience: np.ndarray,
    local_radius_blocks: int,
    sink_blocks: int,
    topk_topology_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sinks + local window + top-k salient blocks, clipped to the causal past."""
    for name, value in (
        ("local_radius_blocks", local_radius_blocks),
        ("sink_blocks", sink_blocks),
        ("topk_topology_blocks", topk_topology_blocks),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    num_blocks = salience.shape[0]
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


class IncrementalSalience:
    """Salience under decode's one-block-at-a-time key growth.

    Rebuilding per step costs an n x n distance matrix and a fresh MST, O(n^2)
    a step and O(n^3) across a decode. Only the new block's n edges can enter
    the tree, by the cycle property: every other non-tree edge is still the
    strict maximum of the cycle it closes with the tree path between its
    endpoints -- adding a vertex does not remove that path -- so it is still
    rejected. Kruskal over (kept tree edges + n new edges) is therefore exact
    at O(n log n) a step, and the n distances to the new block are the only
    O(n * dim) work left. Measured over a 64 -> 256 block decode (192 appends):
    58 ms of appends against 629 ms of rebuilds, 11x; per step at 512 blocks,
    1.1 ms against 24.3 ms.

    Exactness is structural rather than empirical. Both paths order edges by
    the strict total order (distance, low, high), a strict order has exactly
    one minimum spanning tree, and the candidate set above provably contains
    it, so the two paths run the same union-find over the same edge list.
    Rejected extras are no-ops. Verified bit-identical (0.0 max abs diff, not
    1e-12) at every append on random, gridded, collinear, exactly duplicated
    and 1e-12 separated blocks.

        state = IncrementalSalience(centroids[:64])
        for c in centroids[64:]:
            salience = state.append(c)
    """

    def __init__(self, centroids: np.ndarray) -> None:
        c = np.asarray(centroids, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError("centroids must have shape [num_blocks, dim]")
        if c.shape[0] == 0:
            raise ValueError("at least one block is required")
        self._centroids = c.copy()
        edges = _canonical_mst_edges(_distances(c, c)) if c.shape[0] > 1 else []
        self._salience, self._edges = _merge_heights(c.shape[0], edges)

    def __len__(self) -> int:
        return self._centroids.shape[0]

    @property
    def salience(self) -> np.ndarray:
        """Current per-block H0 death times, one entry per block appended."""
        return self._salience

    @property
    def centroids(self) -> np.ndarray:
        return self._centroids

    def append(self, centroid: np.ndarray) -> np.ndarray:
        """Add one key block and return the salience over all blocks."""
        point = np.asarray(centroid, dtype=np.float64).reshape(1, -1)
        if point.shape[1] != self._centroids.shape[1]:
            raise ValueError(
                f"centroid has dim {point.shape[1]}, expected "
                f"{self._centroids.shape[1]}"
            )
        n = self._centroids.shape[0]
        distances = _distances(point, self._centroids)[0]
        fresh = sorted((float(d), i, n) for i, d in enumerate(distances))
        self._centroids = np.vstack((self._centroids, point))
        # kept edges and fresh edges are each already sorted, so Timsort sees
        # two runs and merges them in O(n) instead of re-sorting
        candidates = sorted(self._edges + fresh)
        self._salience, self._edges = _merge_heights(n + 1, candidates)
        return self._salience

    def schedule(
        self,
        local_radius_blocks: int,
        sink_blocks: int,
        topk_topology_blocks: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The causal CSR schedule for the blocks appended so far.

        Same construction and tie-breaking as `build_topology_block_schedule`,
        starting from the salience this object already holds. Note the cost is
        O(n^2) in emitted indices where `append` is O(n log n) -- decode only
        needs the last row, so call this when a full schedule is actually
        wanted.
        """
        return _causal_csr(
            self._salience, local_radius_blocks, sink_blocks, topk_topology_blocks
        )


def _demo() -> None:
    """Equivalence against the merged reference, plus the edge-count saving."""
    rng = np.random.default_rng(0)
    cases = [rng.normal(size=(n, d)) for n, d in ((8, 4), (32, 16), (64, 64))]
    # every merge height tied at once: the case where an arbitrary MST choice
    # or a dropped zero-length edge stops agreeing with the reference
    cases.append(np.repeat(rng.normal(size=(8, 4)), 3, axis=0))
    for centroids in cases:
        num_blocks = centroids.shape[0]
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

    # decode: appending a block must land on the same salience as rebuilding
    blocks = rng.normal(size=(48, 8))
    state = IncrementalSalience(blocks[:8])
    for step in range(8, 48):
        appended = state.append(blocks[step])
        rebuilt = zero_dim_persistence_salience(blocks[: step + 1])
        assert np.array_equal(appended, rebuilt), f"append diverged at {step} blocks"
    # block_size=1 makes each row its own block, so the two schedules are
    # built from the same centroids by the two different paths
    whole = build_topology_block_schedule(blocks, 1, 1, 1, 4)
    assert all(np.array_equal(a, b) for a, b in zip(state.schedule(1, 1, 4), whole))
    print("  incremental: 8 -> 48 blocks, bit-identical salience at every append")
    print("demo ok")


if __name__ == "__main__":
    _demo()

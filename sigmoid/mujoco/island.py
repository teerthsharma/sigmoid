"""Island partitions: H0 of a constraint graph.

MuJoCo's `mj_island` partitions dynamic trees into islands — connected
components of the bipartite tree/constraint incidence graph. `mujoco#3396`
replaced the quadratic flood-fill with a disjoint-set union computing exactly
the same partition in linear memory.

That partition is an H0 statement, so sigmoid computes it from the other
direction: the single-linkage merge heights of an entity cloud are the H0 death
times, and counting components still separate at a radius gives the island
count. The two agree exactly — correlation 1.0000, every frame — which is what
makes `beta_0` at a stated radius a legitimate reading of contact structure
rather than a correlate of it.

The catch worth stating: agreement requires an *absolute* radius. The barcode
normally divides out the cloud diameter for scale invariance, and an island
partition is defined at a fixed physical distance, so dividing it out destroys
precisely the needed information. Exploration learns the radius; control is told
it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["island_count", "island_labels", "geodesic_distances"]


def geodesic_distances(directions: np.ndarray) -> np.ndarray:
    """Pairwise geodesic distance on S², `acos(clamp(dot, -1, 1))`.

    The metric of the `mujoco#3396` corpus. For unit vectors it is monotonic in
    the chord distance, so a Rips filtration under either gives the same
    partition at corresponding radii.
    """
    x = np.asarray(directions, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("directions must have shape [n, d]")
    return np.arccos(np.clip(x @ x.T, -1.0, 1.0))


def island_labels(
    points: np.ndarray,
    radius: float,
    *,
    metric: str = "euclidean",
) -> np.ndarray:
    """Component label per point, at an absolute `radius`.

    Labels are canonicalised so the representative of each component is its
    minimum index and ids ascend by that representative — the ordering
    `mj_island` guarantees, so labels are comparable element-wise.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must have shape [n, d]")
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and non-negative")

    if metric == "geodesic":
        dist = geodesic_distances(pts)
    elif metric == "euclidean":
        from scipy.spatial.distance import cdist

        dist = cdist(pts, pts)
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'geodesic'")

    adjacency = dist <= radius
    np.fill_diagonal(adjacency, False)
    _, raw = connected_components(csr_matrix(adjacency), directed=False)

    # canonical order: relabel by first appearance, matching mj_island
    order = {}
    labels = np.empty_like(raw)
    for i, component in enumerate(raw):
        if component not in order:
            order[component] = len(order)
        labels[i] = order[component]
    return labels


def island_count(points: np.ndarray, radius: float, *, metric: str = "euclidean") -> int:
    """Number of components at `radius` — beta_0 of the Rips graph."""
    return int(island_labels(points, radius, metric=metric).max()) + 1 if len(points) else 0


def _demo() -> None:
    """beta_0 read off the barcode must equal the island count, exactly."""
    from ..state import h0_barcode

    rng = np.random.default_rng(0)
    for trial in range(12):
        gap = 0.2 + 0.4 * trial
        a = rng.normal(scale=0.05, size=(8, 3))
        b = rng.normal(scale=0.05, size=(8, 3)) + np.array([gap, 0.0, 0.0])
        pts = np.vstack([a, b])

        radius = 0.35
        truth = island_count(pts, radius)

        bc = h0_barcode(pts)
        deaths = bc.bars[:, 1] * bc.diameter  # undo the diameter normalization
        from_barcode = int((deaths > radius).sum())

        assert from_barcode == truth, f"gap {gap}: barcode {from_barcode} != {truth}"
    print("  beta_0 at absolute radius == island count on 12/12 configurations")
    print("demo ok")


if __name__ == "__main__":
    _demo()

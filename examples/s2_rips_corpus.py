"""The S2-Rips corpus: a world where topology is the actual signal.

Implements the generator math from the PR #3396 topological DSU corpus design
(google-deepmind/mujoco#3396), which specifies:

    node i carries a unit direction x_i(t) on S2 and a tangent velocity v_i(t)
    d_g(x_i, x_j) = acos(clamp(dot(x_i, x_j), -1, 1))
    a Rips filtration on d_g gives constraint incidence; islands are its H0

The PR's own C++ generator emits three frames per case, which is right for
benchmarking disjoint-set union and useless for fitting dynamics, so only the
generator math is reused here -- long trajectories are produced instead.

Why this corpus is the honest test sigmoid was missing. On distilgpt2 the
topological channel measurably did not help, but token dynamics is a domain
where nothing guaranteed topology was present. Here it is present by
construction: nodes flow along great circles, clusters merge and separate, and
the island count H0 changes discretely. Crucially, "how many islands" is not a
linear function of the coordinates -- no PCA of x can represent a merge, while
a persistence barcode reports it directly. If the topological channel cannot
win here, it cannot win anywhere.

    python examples/s2_rips_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sigmoid

TAU = 2.0 * np.pi


def geodesic_step(x: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exponential map on S2: flow along great circles, transporting v.

    Speed |v| is conserved and v stays tangent, so trajectories are exact
    great circles rather than a drifting Euler approximation.
    """
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    theta = speed * dt
    unit_v = np.divide(v, np.maximum(speed, 1e-12))
    x_next = x * np.cos(theta) + unit_v * np.sin(theta)
    v_next = -x * speed * np.sin(theta) + v * np.cos(theta)
    x_next /= np.maximum(np.linalg.norm(x_next, axis=1, keepdims=True), 1e-12)
    # re-project to the tangent space to kill accumulated numerical drift
    v_next -= x_next * np.sum(v_next * x_next, axis=1, keepdims=True)
    return x_next, v_next


def geodesic_distances(x: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(x @ x.T, -1.0, 1.0))


def island_count(x: np.ndarray, radius: float) -> int:
    """H0 of the Rips graph at `radius` -- the PR's island partition."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adj = geodesic_distances(x) <= radius
    np.fill_diagonal(adj, False)
    return int(connected_components(csr_matrix(adj), directed=False)[0])


def make_corpus(
    n_caps: int = 2,
    per_cap: int = 9,
    n_bridge: int = 3,
    frames: int = 900,
    dt: float = 0.04,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Two tight caps plus bridge nodes on great circles that link them.

    Returns (observations, island_counts). Observations are flattened node
    coordinates -- exactly what a "normal model" would hand over. The island
    count is held out as a diagnostic and never shown to the world model.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])[:n_caps]

    xs, vs = [], []
    for center in centers:  # tight, slowly-jittering clusters
        noise = rng.normal(scale=0.18, size=(per_cap, 3))
        pts = center + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        tangent = rng.normal(scale=0.05, size=(per_cap, 3))
        tangent -= pts * np.sum(tangent * pts, axis=1, keepdims=True)
        xs.append(pts)
        vs.append(tangent)

    # bridges: fast movers on great circles that sweep pole to pole, so the
    # island partition genuinely merges and splits on a regular cycle
    bridge = rng.normal(size=(n_bridge, 3))
    bridge /= np.linalg.norm(bridge, axis=1, keepdims=True)
    bridge_v = rng.normal(size=(n_bridge, 3))
    bridge_v -= bridge * np.sum(bridge_v * bridge, axis=1, keepdims=True)
    bridge_v /= np.linalg.norm(bridge_v, axis=1, keepdims=True)
    bridge_v *= 0.9
    xs.append(bridge)
    vs.append(bridge_v)

    x = np.concatenate(xs, axis=0)
    v = np.concatenate(vs, axis=0)

    radius = 0.95
    obs, islands = [], []
    for _ in range(frames):
        obs.append(x.reshape(-1).copy())
        islands.append(island_count(x, radius))
        x, v = geodesic_step(x, v, dt)
    return np.asarray(obs, dtype=np.float64), np.asarray(islands)


def main() -> int:
    obs, islands = make_corpus()
    uniq, counts = np.unique(islands, return_counts=True)
    print("=" * 72)
    print("S2-Rips corpus (generator math from mujoco#3396)")
    print("=" * 72)
    print(f"  observations      {obs.shape}  ({obs.shape[1] // 3} nodes)")
    print(f"  island counts     {dict(zip(uniq.tolist(), counts.tolist()))}")
    print(f"  H0 transitions    {int((np.diff(islands) != 0).sum())} merges/splits")

    if len(uniq) < 2:
        print("\n  corpus is topologically static -- adjust radius or speeds")
        return 1

    config = sigmoid.SigmoidConfig(
        window=24,
        linear_dim=16,
        hilbert_degree=20,
        n_radii=8,
        entity_dim=3,      # the state is a set of nodes on S2, not a sequence
        n_abs_radii=8,     # islands live at a fixed radius; keep the scale
    )
    print("\n" + "=" * 72)
    print("ablation: matched state dimension, topology on vs off")
    print("=" * 72)
    report = sigmoid.compare(
        [obs], config=config, horizons=(1, 4, 16), holdout=0.35
    )
    print(report.table())

    sig = next(a for a in report.arms if a.name == "sigmoid")
    lin = next(a for a in report.arms if a.name == "linear_only")
    for h in report.horizons:
        gain = (lin.nrmse[h] - sig.nrmse[h]) / max(lin.nrmse[h], 1e-12) * 100
        print(f"  k={h:<3} topology changes error by {-gain:+.1f}%")

    print("\n" + "=" * 72)
    print("the topological observable: island count, now and k steps ahead")
    print("=" * 72)
    print("  H0 is what this world is *about*. Coordinates favour carry-forward;")
    print("  the island partition is the state a controller would actually need.")
    wm = sigmoid.SigmoidWorldModel(config=config).fit([obs])
    Z = wm.encoder.encode_trajectory(obs)
    psi, u = Z[:, : wm.encoder.topo_dim], Z[:, wm.encoder.topo_dim :]

    # The island count is a step function, so a linear R2 probe is the wrong
    # instrument -- it punishes both channels for the discreteness rather than
    # measuring what either one knows. Classification is the right question:
    # can you read the island partition off this channel at all?
    from sklearn.linear_model import LogisticRegression

    def accuracy(X, y, lag=0):
        """Multinomial logistic probe, identical treatment for both channels."""
        if lag:
            X, y = X[:-lag], y[lag:]
        n = int(len(X) * 0.7)
        mu, sd = X[:n].mean(0), X[:n].std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X[:n] - mu) / sd, y[:n])
        return float(clf.score((X[n:] - mu) / sd, y[n:]))

    labels = islands[config.window - 1 :]
    majority = float(np.bincount(labels[int(len(labels) * 0.7) :]).max()) / max(
        len(labels) - int(len(labels) * 0.7), 1
    )
    print(f"\n  {'lag':<6}{'psi (topology)':>18}{'u (linear)':>14}{'majority':>12}")
    for lag in (0, 1, 4, 16):
        print(
            f"  {lag:<6}{accuracy(psi, labels, lag):>18.3f}"
            f"{accuracy(u, labels, lag):>14.3f}{majority:>12.3f}"
        )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""The S²-Vietoris-Rips corpus generator from `mujoco#3396`.

The PR's design specifies a dynamics-driven incidence generator:

    node i carries a unit direction x_i(t) on S² and a tangent velocity v_i(t)
    d_g(x_i, x_j) = acos(clamp(dot(x_i, x_j), -1, 1))
    a Rips filtration on d_g gives constraint incidence; islands are its H0

The PR's own C++ generator emits three frames per case, which is right for
benchmarking disjoint-set union and useless for fitting dynamics, so only the
generator math is reused here — long trajectories are produced instead.

This corpus is the reference positive control for the whole library. Its island
count is H0 by construction, so "does the topological channel read structure a
PCA cannot" has a ground truth rather than a proxy.
"""

from __future__ import annotations

import numpy as np

from .island import island_count

__all__ = ["geodesic_step", "make_corpus", "CorpusConfig"]


class CorpusConfig:
    """Parameters of the two-cap-plus-bridges construction."""

    def __init__(
        self,
        n_caps: int = 2,
        per_cap: int = 9,
        n_bridge: int = 3,
        frames: int = 900,
        dt: float = 0.04,
        radius: float = 0.95,
        seed: int = 7,
    ):
        self.n_caps = n_caps
        self.per_cap = per_cap
        self.n_bridge = n_bridge
        self.frames = frames
        self.dt = dt
        self.radius = radius
        self.seed = seed


def geodesic_step(
    x: np.ndarray, v: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential map on S²: flow along great circles, transporting v.

    Speed |v| is conserved and v stays tangent, so trajectories are exact great
    circles rather than a drifting Euler approximation. The tangent is
    re-projected each step to kill accumulated numerical drift.
    """
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    theta = speed * dt
    unit_v = np.divide(v, np.maximum(speed, 1e-12))
    x_next = x * np.cos(theta) + unit_v * np.sin(theta)
    v_next = -x * speed * np.sin(theta) + v * np.cos(theta)
    x_next /= np.maximum(np.linalg.norm(x_next, axis=1, keepdims=True), 1e-12)
    v_next -= x_next * np.sum(v_next * x_next, axis=1, keepdims=True)
    return x_next, v_next


def make_corpus(config: CorpusConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate (observations, island_counts).

    Observations are flattened node coordinates — exactly what a "normal model"
    would hand over. The island count is a held-out diagnostic and is never
    shown to a world model fitted on this data.
    """
    cfg = config or CorpusConfig()
    rng = np.random.default_rng(cfg.seed)
    centers = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])[: cfg.n_caps]

    xs, vs = [], []
    for center in centers:  # tight, slowly-jittering clusters
        pts = center + rng.normal(scale=0.18, size=(cfg.per_cap, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        tangent = rng.normal(scale=0.05, size=(cfg.per_cap, 3))
        tangent -= pts * np.sum(tangent * pts, axis=1, keepdims=True)
        xs.append(pts)
        vs.append(tangent)

    # bridges: fast movers on great circles sweeping pole to pole, so the island
    # partition genuinely merges and splits on a regular cycle
    bridge = rng.normal(size=(cfg.n_bridge, 3))
    bridge /= np.linalg.norm(bridge, axis=1, keepdims=True)
    bridge_v = rng.normal(size=(cfg.n_bridge, 3))
    bridge_v -= bridge * np.sum(bridge_v * bridge, axis=1, keepdims=True)
    bridge_v /= np.linalg.norm(bridge_v, axis=1, keepdims=True)
    bridge_v *= 0.9
    xs.append(bridge)
    vs.append(bridge_v)

    x = np.concatenate(xs, axis=0)
    v = np.concatenate(vs, axis=0)

    obs, islands = [], []
    for _ in range(cfg.frames):
        obs.append(x.reshape(-1).copy())
        islands.append(island_count(x, cfg.radius, metric="geodesic"))
        x, v = geodesic_step(x, v, cfg.dt)
    return np.asarray(obs, dtype=np.float64), np.asarray(islands)


def _demo() -> None:
    """The corpus must actually change topology, or it tests nothing."""
    obs, islands = make_corpus(CorpusConfig(frames=400))
    transitions = int((np.diff(islands) != 0).sum())
    distinct = len(np.unique(islands))
    assert obs.shape[0] == 400
    assert distinct >= 2, f"topologically static corpus: only {distinct} island count"
    assert transitions > 0, "no merges or splits — nothing to predict"
    print(f"  {obs.shape[0]} frames, {obs.shape[1] // 3} nodes")
    print(f"  island counts {sorted(np.unique(islands).tolist())}, "
          f"{transitions} merges/splits")
    print("demo ok")


if __name__ == "__main__":
    _demo()

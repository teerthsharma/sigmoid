"""AGENDA.md R4 -- run the condition at the place it is supposed to break.

SIGMOID.md 4.3 states a sharp condition:

    the topological channel earns its dimensions when the state is a set of
    interacting entities whose interaction threshold is a fixed physical
    distance.

A condition that only ever predicts success is not a condition, so this script
builds entity systems that differ *only* in whether that second clause holds and
measures topology's benefit at each point. Everything else is held constant: 24
entities in the plane, the same damped second-order plant, the same observation
width, the same encoder budget, the same uniform rule for choosing absolute
radii, the same process noise.

    fixed-radius    hard interaction cutoff at a constant 0.60, four tight
                    clusters that lap each other. Control.
                    The condition predicts topology helps.
    wide-cutoff     the same plant with the cutoff smeared into a soft shoulder
                    and the cluster scales spread over ~1.4 decades, so the
                    threshold no longer separates anything. Middle case.
                    The condition predicts the benefit degrades.
    scale-free      a Levy-flight layout across ~2.5 decades and a power-law
                    1/d^2 coupling with no cutoff at all. There is no radius at
                    which "who is interacting" has an answer.
                    The condition predicts topology does NOT help.
    dilating        the fixed-radius system exactly, observed through a global
                    dilation sweeping 3 decades. Identical combinatorics,
                    identical labels -- the threshold is fixed *relative* to the
                    configuration but is not a fixed physical distance.
                    The condition as written predicts topology does NOT help.

The last one is the discriminator. It is the only way to tell whether the clause
that matters is "fixed physical distance" or the weaker "fixed relative to the
configuration's own scale"; every system measured in SIGMOID.md satisfies both
or neither, so the two have never been separated.

S2-Rips is carried along as a fifth arm at its published settings. It is the
known-good positive, and an experiment whose control disagrees with it is broken
rather than informative.

"Is there a fixed interaction distance" is measured, not asserted. Single-linkage
merge heights are exactly the H0 death times, so the widest gap between
consecutive merge heights in log-radius is the widest band on which the component
count is constant and non-trivial -- precisely the scale separation the condition
is about. Its width says whether such a distance exists; the drift of its centre
across frames says whether it stays put.

    python examples/falsify_condition.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sigmoid
from sigmoid.state import _pairwise

N_GROUPS, PER_GROUP = 4, 6
N_ENTITIES = N_GROUPS * PER_GROUP  # 24 entities -> 48-dim observation
R0 = 0.6  # the fixed physical interaction distance
DT = 0.05
DAMP = 0.90
FRAMES = 1400
ANCHOR_K = 8.0
COUPLE_K = 0.15
INTRA = 0.06  # cluster radius, well inside R0 so a cluster is one component
NOISE = 0.10
WINDOW = 16


# ---------------------------------------------------------------------------
# plant
# ---------------------------------------------------------------------------


def _cluster_anchors(frames: int, spread_dex: float, seed: int) -> np.ndarray:
    """Four tight clusters lapping each other on a shared circle.

    Because the clusters share one orbit at different angular rates they pass
    straight through one another, so the centre separation sweeps the whole
    range from 0 to 4.4 and the component count at R0 changes as a sequence of
    discrete merge and split events rather than by flickering.

    `spread_dex` spreads the *intra-cluster* scale over that many decades. At 0
    the system has exactly two scales -- inside a cluster and between clusters --
    which is a textbook fixed interaction distance. Raising it slides the two
    into each other until no radius separates them.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None] * DT
    omega = np.array([0.20, 0.45, 0.72, 1.00])
    phase = rng.uniform(0.0, 2.0 * np.pi, N_GROUPS)
    centers = np.stack(
        [2.2 * np.cos(omega * t + phase), 2.2 * np.sin(omega * t + phase)], axis=-1
    )  # (frames, N_GROUPS, 2)

    scale = INTRA * 10.0 ** np.linspace(-spread_dex / 2, spread_dex / 2, N_GROUPS)
    offset = rng.normal(size=(N_GROUPS, PER_GROUP, 2))
    offset /= np.linalg.norm(offset, axis=2, keepdims=True)
    offset *= scale[:, None, None] * rng.uniform(0.5, 1.5, (N_GROUPS, PER_GROUP, 1))
    return (centers[:, :, None, :] + offset[None]).reshape(frames, N_ENTITIES, 2)


def _levy_anchors(
    frames: int,
    seed: int,
    lo: float = 0.02,
    hi: float = 6.0,
    w0: float = 2.0,
) -> np.ndarray:
    """A Levy flight: the canonical layout with no characteristic scale.

    Entity i sits at the partial sum of i steps whose lengths are drawn
    log-uniformly across 2.5 decades, so the pairwise distances are log-uniform
    too and the single-linkage merge heights form a near-Poisson process in
    log-radius -- there is no band of radii on which the component count is
    stable. A first attempt used a multiplicative cascade instead; it measured a
    *wider* scale gap (0.62 decades) than the fixed-radius control, because
    discrete levels leave discrete gaps. A cascade is hierarchical, not
    scale-free, and the diagnostic caught it.

    Each step rotates at a Levy-consistent rate w0/sqrt(length), so short steps
    churn fast and long ones drift: the time axis has no characteristic scale
    either.
    """
    rng = np.random.default_rng(seed)
    n = N_ENTITIES
    mag = lo * (hi / lo) ** rng.uniform(0.0, 1.0, n)
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    t = np.arange(frames)[:, None] * DT
    theta = ang[None, :] + (w0 / np.sqrt(mag))[None, :] * t
    step = np.stack([mag * np.cos(theta), mag * np.sin(theta)], axis=-1)
    return np.cumsum(step, axis=1)


def _directions(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = _pairwise(p)
    np.fill_diagonal(d, np.inf)
    return d, (p[:, None, :] - p[None, :, :]) / d[:, :, None]


def hard_cutoff(p: np.ndarray) -> np.ndarray:
    """Constant repulsion inside R0, nothing outside. A real discontinuity at a
    real distance -- the plant itself knows about R0."""
    d, unit = _directions(p)
    return (np.where(d < R0, COUPLE_K, 0.0)[:, :, None] * unit).sum(axis=1)


def soft_cutoff(p: np.ndarray) -> np.ndarray:
    """The same repulsion with the step smeared over a shoulder of width R0, so
    the threshold is spread across roughly a decade instead of sitting at a
    point."""
    d, unit = _directions(p)
    return ((COUPLE_K / (1.0 + np.exp((d - R0) / R0)))[:, :, None] * unit).sum(axis=1)


def power_law(p: np.ndarray) -> np.ndarray:
    """1/d^2 with no cutoff. Every pair interacts at every distance and no
    distance is special -- the coupling has no characteristic scale."""
    d, unit = _directions(p)
    w = COUPLE_K * (R0 / np.sqrt(d**2 + 1e-3)) ** 2
    return (np.minimum(w, 20.0)[:, :, None] * unit).sum(axis=1)


def simulate(anchors: np.ndarray, coupling, seed: int) -> np.ndarray:
    """Damped double integrator: pulled toward a moving anchor, pushed by peers.

    The same process noise goes into every system. Without it a smooth
    deterministic orbit is close to linearly predictable and the rollout arm
    measures smoothness rather than structure.
    """
    rng = np.random.default_rng(seed)
    p = anchors[0] + rng.normal(scale=0.01, size=anchors[0].shape)
    v = np.zeros_like(p)
    out = np.empty((len(anchors), N_ENTITIES * 2))
    for t in range(len(anchors)):
        out[t] = p.reshape(-1)
        drive = ANCHOR_K * (anchors[t] - p) + coupling(p)
        drive += NOISE * np.linalg.norm(anchors[t], axis=1, keepdims=True) * rng.normal(
            size=p.shape
        )
        v = DAMP * v + DT * drive
        p = p + DT * v
    return out


# ---------------------------------------------------------------------------
# measured structure
# ---------------------------------------------------------------------------


def components(pts: np.ndarray, radius: float) -> int:
    """H0 of the interaction graph at `radius`."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adj = _pairwise(pts) <= radius
    np.fill_diagonal(adj, False)
    return int(connected_components(csr_matrix(adj), directed=False)[0])


def merge_heights(pts: np.ndarray) -> np.ndarray:
    """Single-linkage merge heights in physical units -- the H0 death times."""
    bc = sigmoid.h0_barcode(pts)
    finite = bc.bars[np.isfinite(bc.bars[:, 1]), 1]
    return np.sort(finite * bc.diameter)


def _frames(obs: np.ndarray, dim: int, n: int = 200) -> list[np.ndarray]:
    idx = np.linspace(0, len(obs) - 1, min(n, len(obs))).astype(int)
    return [obs[i].reshape(-1, dim) for i in idx]


def scale_separation(obs: np.ndarray, dim: int) -> tuple[float, float]:
    """Does a fixed interaction distance exist, and does it stay put?

    Returns (gap, drift), both in decades. `gap` is the widest band of radii on
    which the component count is constant and non-trivial, averaged over frames.
    `drift` is the standard deviation of that band's centre across frames. The
    condition needs a wide gap that does not move.
    """
    gaps, centers = [], []
    for pts in _frames(obs, dim):
        h = merge_heights(pts)
        h = h[h > 1e-9]
        if h.size < 2:
            continue
        lg = np.log10(h)
        w = np.diff(lg)
        k = int(np.argmax(w))
        gaps.append(float(w[k]))
        centers.append(float(0.5 * (lg[k] + lg[k + 1])))
    return float(np.mean(gaps)), float(np.std(centers))


def _distances(obs: np.ndarray, dim: int) -> np.ndarray:
    n = obs.shape[1] // dim
    d = np.concatenate([_pairwise(p)[np.triu_indices(n, 1)] for p in _frames(obs, dim)])
    return d[d > 1e-9]


def matched_radius(obs: np.ndarray, dim: int, target_mean: float) -> float:
    """The radius whose mean component count matches the control's.

    A scale-free cloud has no privileged radius, so any choice of label radius is
    arbitrary. Choosing the one that makes the label as hard as the control's
    removes the obvious objection to a null result -- that the label was
    degenerate -- without smuggling in knowledge the system does not have.
    """
    d = _distances(obs, dim)
    grid = np.geomspace(np.quantile(d, 0.02), np.quantile(d, 0.98), 40)
    means = [np.mean([components(p, r) for p in _frames(obs, dim, 80)]) for r in grid]
    return float(grid[int(np.argmin(np.abs(np.asarray(means) - target_mean)))])


def abs_radius_ladder(obs: np.ndarray, dim: int, nominal: float, n: int = 5) -> tuple[float, ...]:
    """One rule for every system: n log-spaced radii across the bulk of the
    observed pairwise distances, plus the radius the label is defined at.

    Stating the physical radius rather than learning quantiles is the lesson from
    SIGMOID.md 4.7, so every system is told its own. The point of the scale-free
    and dilating arms is that saying it buys nothing there, because no single
    radius is the right one.
    """
    lo, hi = np.quantile(_distances(obs, dim), [0.05, 0.95])
    ladder = np.geomspace(lo, hi, n).tolist() + [nominal]
    return tuple(float(r) for r in sorted(ladder))


# ---------------------------------------------------------------------------
# systems
# ---------------------------------------------------------------------------


@dataclass
class System:
    name: str
    obs: np.ndarray
    labels: np.ndarray
    nominal: float
    entity_dim: int
    prediction: str
    blurb: str


def build_systems(seed: int = 11) -> list[System]:
    fixed_obs = simulate(_cluster_anchors(FRAMES, 0.0, seed), hard_cutoff, seed)
    fixed_lab = np.array([components(o.reshape(-1, 2), R0) for o in fixed_obs])

    wide_obs = simulate(_cluster_anchors(FRAMES, 1.4, seed), soft_cutoff, seed)
    wide_lab = np.array([components(o.reshape(-1, 2), R0) for o in wide_obs])

    free_obs = simulate(_levy_anchors(FRAMES, seed), power_law, seed)
    free_r = matched_radius(free_obs, 2, float(fixed_lab.mean()))
    free_lab = np.array([components(o.reshape(-1, 2), free_r) for o in free_obs])

    # Same trajectory, same combinatorics, same labels -- only the physical scale
    # of the threshold moves, sweeping 3 decades on a 350-frame cycle.
    lam = 10.0 ** (1.5 * np.sin(2.0 * np.pi * np.arange(FRAMES) / 350.0))
    dil_obs = fixed_obs * lam[:, None]

    systems = [
        System("fixed-radius", fixed_obs, fixed_lab, R0, 2,
               "topology HELPS", "hard cutoff at a constant 0.60"),
        System("wide-cutoff", wide_obs, wide_lab, R0, 2,
               "benefit DEGRADES", "soft shoulder, cluster scales spread ~1.4 decades"),
        System("scale-free", free_obs, free_lab, free_r, 2,
               "topology does NOT help", "1/d^2 coupling, no cutoff, Levy-flight layout"),
        System("dilating", dil_obs, fixed_lab, R0, 2,
               "topology does NOT help", "fixed-radius seen through a 3-decade dilation"),
    ]

    try:  # the published positive control, unchanged
        from s2_rips_corpus import make_corpus

        s2_obs, s2_lab = make_corpus()  # published settings, untouched
        systems.append(
            System("s2-rips (ref)", s2_obs, s2_lab, 0.95, 3,
                   "topology HELPS", "published positive control, mujoco#3396")
        )
    except Exception as exc:  # pragma: no cover - reference arm is optional
        print(f"  (s2-rips reference unavailable: {exc})")
    return systems


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def probe(X: np.ndarray, y: np.ndarray, lag: int = 0) -> float:
    """Multinomial logistic probe, 70/30 time split. Identical for every channel."""
    from sklearn.linear_model import LogisticRegression

    if lag:
        X, y = X[:-lag], y[lag:]
    n = int(len(X) * 0.7)
    mu, sd = X[:n].mean(0), X[:n].std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit((X[:n] - mu) / sd, y[:n])
    return float(clf.score((X[n:] - mu) / sd, y[n:]))


def run(system: System) -> dict:
    radii = abs_radius_ladder(system.obs, system.entity_dim, system.nominal)
    config = sigmoid.SigmoidConfig(
        window=WINDOW,
        linear_dim=16,
        hilbert_degree=16,
        n_radii=6,
        entity_dim=system.entity_dim,  # the state is a set of bodies, not a window
        abs_radii=radii,  # exploration learns the radius, control is told it
        n_abs_radii=0,
    )

    print("\n" + "=" * 72)
    print(f"{system.name}  --  {system.blurb}")
    print("=" * 72)
    gap, drift = scale_separation(system.obs, system.entity_dim)
    uniq, counts = np.unique(system.labels, return_counts=True)
    print(f"  scale separation   {gap:.2f} decades wide, centre drifts {drift:.2f} decades")
    print(f"  label (H0 at {system.nominal:.3g})  {dict(zip(uniq.tolist(), counts.tolist()))}")
    print(f"  transitions        {int((np.diff(system.labels) != 0).sum())}")
    print(f"  abs radii          {', '.join(f'{r:.3g}' for r in radii)}")
    print(f"  condition predicts {system.prediction}")

    print("\n  -- matched-dimension ablation, topology on vs off --")
    report = sigmoid.compare([system.obs], config=config, horizons=(1, 4, 16), holdout=0.35)
    print("\n".join("  " + line for line in report.table().splitlines()))
    sig = next(a for a in report.arms if a.name == "sigmoid")
    lin = next(a for a in report.arms if a.name == "linear_only")
    # `no_topology_same_u` is the arm that isolates topology: identical linear
    # channel, psi deleted. `linear_only` answers a different question (is psi
    # the best use of those dimensions?) and is confounded by the PCA-rank
    # change -- bench.py documents this, and this script reproduced the trap
    # before that arm existed, reporting a 54.6% "win" that was entirely rank.
    same_u = next(a for a in report.arms if a.name == "no_topology_same_u")
    rollout = {
        h: (same_u.nrmse[h] - sig.nrmse[h]) / max(same_u.nrmse[h], 1e-12) * 100.0
        for h in report.horizons
    }
    budget = {
        h: (lin.nrmse[h] - sig.nrmse[h]) / max(lin.nrmse[h], 1e-12) * 100.0
        for h in report.horizons
    }
    for h in report.horizons:
        print(
            f"  k={h:<3} psi changes rollout error by {-rollout[h]:+.2f}% "
            f"(vs same-u); {-budget[h]:+.1f}% vs the dim-matched linear arm"
        )

    print("\n  -- can the channel read the interaction structure? --")
    wm = sigmoid.SigmoidWorldModel(config=config).fit([system.obs])
    Z = wm.encoder.encode_trajectory(system.obs)
    psi, u = Z[:, : wm.encoder.topo_dim], Z[:, wm.encoder.topo_dim :]

    # The ablation arm's linear channel at the *full* matched dimension is the
    # baseline that matters: it is the same budget as [psi ; u] spent entirely on
    # PCA, which on a 48-dim observation is a lossless linear view of the raw
    # coordinates. Topology has to beat that, not the 16-dim stub it shares a
    # state vector with.
    lin_cfg = sigmoid.SigmoidConfig(
        **{**vars(config), "use_topology": False, "linear_dim": wm.state_dim}
    )
    u_full = (
        sigmoid.SigmoidWorldModel(config=lin_cfg)
        .fit([system.obs])
        .encoder.encode_trajectory(system.obs)
    )

    labels = system.labels[WINDOW - 1 :]
    cut = int(len(labels) * 0.7)
    majority = float(np.bincount(labels[cut:]).max()) / max(len(labels) - cut, 1)
    print(
        f"  {'lag':<6}{f'psi ({psi.shape[1]}d)':>14}{f'u ({u.shape[1]}d)':>12}"
        f"{f'u_matched ({u_full.shape[1]}d)':>20}{'majority':>11}"
    )
    acc = {}
    for lag in (0, 1, 4, 16):
        a_psi, a_u, a_full = (probe(psi, labels, lag), probe(u, labels, lag),
                              probe(u_full, labels, lag))
        acc[lag] = (a_psi, a_full)
        print(f"  {lag:<6}{a_psi:>14.3f}{a_u:>12.3f}{a_full:>20.3f}{majority:>11.3f}")

    # Did the stated radius have to be *right*? Every ladder above contains the
    # exact radius the label is defined at, so a lag-0 win is close to tautology:
    # beta_0 at an absolute radius is literally a coordinate of psi. The question
    # the condition is really about is whether that radius can be known in
    # advance -- so restate it wrong by 0.3 decades and probe the same label.
    # Inside a scale plateau the answer should barely move; without one it is a
    # different question entirely.
    mis_cfg = sigmoid.SigmoidConfig(
        **{**vars(config), "abs_radii": tuple(r * 2.0 for r in radii)}
    )
    psi_mis = (
        sigmoid.SigmoidWorldModel(config=mis_cfg)
        .fit([system.obs])
        .encoder.encode_trajectory(system.obs)[:, : wm.encoder.topo_dim]
    )
    mis = probe(psi_mis, labels, 0)
    print(f"  radius mis-stated by +0.3 decades:  psi lag0 {mis:.3f} "
          f"(was {acc[0][0]:.3f}, {mis - acc[0][0]:+.3f})")

    return {
        "name": system.name,
        "gap": gap,
        "drift": drift,
        "rollout16": rollout[16],
        "budget16": budget[16],
        "acc": acc,
        "mis": mis,
        "majority": majority,
        # the scale-invariant head of psi: hilbert coefficients + the Betti curve
        # at *normalized* radii, both of which divide out the cloud diameter
        "psi_norm": psi[:, : config.hilbert_degree + config.n_radii],
    }


def _selfcheck() -> None:
    """`scale_separation` is the one piece of non-obvious logic here and every
    conclusion is read off it, so check it against two clouds whose answer is
    known by construction: three needle-tight clusters five apart (two scales,
    2.5 decades between them) and a Levy flight (no scale separation at all)."""
    rng = np.random.default_rng(0)
    tight = np.concatenate([rng.normal(c, 0.01, (8, 2)) for c in ([0, 0], [5, 0], [0, 5])])
    assert scale_separation(tight.reshape(1, -1), 2)[0] > 2.0
    for seed in range(4):
        r = np.random.default_rng(seed)
        walk = np.cumsum(10.0 ** r.uniform(-2, 1, (24, 1)) * r.normal(size=(24, 2)), axis=0)
        assert scale_separation(walk.reshape(1, -1), 2)[0] < 0.8


def main() -> int:
    warnings.filterwarnings("ignore")
    _selfcheck()
    print("=" * 72)
    print("AGENDA.md R4 -- falsifying the fixed-interaction-distance condition")
    print("=" * 72)
    print(f"  {N_ENTITIES} entities in the plane, {FRAMES} frames, 48-dim observation")
    print("  every arm: same plant, same encoder budget, same radius rule, same noise")
    print("  the only thing that varies is whether a fixed interaction distance exists")

    results = [run(s) for s in build_systems()]

    print("\n" + "=" * 72)
    print("summary: topology's benefit against the measured scale separation")
    print("=" * 72)
    head = (
        f"  {'system':<15}{'gap':>6}{'drift':>7}{'psi roll':>10}{'dim-matched':>13}"
        f"{'psi-lin l0':>12}{'psi-maj l4':>12}{'psi @+0.3dex':>14}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in results:
        p0, f0 = r["acc"][0]
        p4, _ = r["acc"][4]
        print(
            f"  {r['name']:<15}{r['gap']:>6.2f}{r['drift']:>7.2f}"
            f"{-r['rollout16']:>9.2f}%{-r['budget16']:>12.1f}%"
            f"{p0 - f0:>12.3f}{p4 - r['majority']:>12.3f}{r['mis'] - p0:>14.3f}"
        )
    print("\n  gap/drift in decades. `psi roll` is the k=16 rollout delta against the")
    print("  same-u ablation -- the arm that isolates topology; `dim-matched` is the")
    print("  same delta against linear_only, which is confounded by the PCA rank change")
    print("  and is shown only to size that confound. Negative = topology helps.")
    print("  psi-lin is the probe margin over the matched full-rank linear channel,")
    print("  psi-maj the margin over the majority class, and psi @+0.3dex the lag-0")
    print("  cost of mis-stating the radius by 0.3 decades.")

    # Why does a 3-decade dilation barely dent the probe? Because the head of psi
    # divides out the cloud diameter, so it cannot see a global dilation at all.
    # Measured rather than argued: the dilating arm is the fixed arm, rescaled.
    by = {r["name"]: r for r in results}
    if {"fixed-radius", "dilating"} <= by.keys():
        a, b = by["fixed-radius"]["psi_norm"], by["dilating"]["psi_norm"]
        col = [
            abs(np.corrcoef(a[:, j], b[:, j])[0, 1])
            for j in range(a.shape[1])
            if a[:, j].std() > 1e-9 and b[:, j].std() > 1e-9
        ]
        print(
            f"\n  scale-invariant head of psi, fixed vs dilating: mean |corr| "
            f"{np.mean(col):.4f} over {len(col)} columns"
        )

    print("\n" + "=" * 72)
    print("reading this honestly")
    print("=" * 72)
    print(ASSESSMENT)
    return 0


ASSESSMENT = """\
  The condition did not survive. It failed in two independent places, and on
  the rollout metric it was never being measured at all.

  1. "FIXED PHYSICAL DISTANCE" IS THE WRONG CLAUSE.

     The dilating arm is the fixed-radius arm multiplied by a scalar sweeping
     three orders of magnitude. Its interaction threshold is not a fixed
     physical distance by any reading -- the ladder of absolute radii it was
     handed spans 0.011 to 96.8 and not one of them is right for more than a few
     frames. The condition predicts no benefit. Measured:

         psi reads the partition   0.940 lag-0, 0.805 lag-4  (majority 0.505)
         matched linear channel    0.382 lag-0, 0.337 lag-4
         fixed-radius control      1.000 lag-0, 0.824 lag-4

     Within noise of the control. The mechanism is not subtle and is measured
     rather than argued: the head of psi -- Hilbert coefficients and the Betti
     curve at normalized radii -- divides out the cloud diameter, so it cannot
     see a global dilation. Mean |corr| between the two arms' scale-invariant
     columns is 1.0000 over 19 columns. They are the same feature.

     The clause should read *fixed relative to the configuration's own scale*.
     Every system in SIGMOID.md satisfies both readings, so the two had never
     been separated; separating them costs the stronger one.

  2. THE SCALE-FREE ARM ALSO BEATS THE LINEAR CHANNEL -- BUT ONLY AT THE EXACT
     RADIUS IT WAS TOLD.

     Levy-flight layout over 2.5 decades, 1/d^2 coupling with no cutoff,
     measured scale separation 0.29 decades against the control's 0.82. The
     condition predicts no benefit. Measured: psi 0.930 lag-0 against 0.411 for
     the matched full-rank linear channel, and +0.204 over majority at lag 4 --
     comparable to the published S2-Rips positive at +0.232.

     So the flat prediction is wrong. What separates it from the control is not
     whether psi can read the structure but whether the radius could have been
     known beforehand. Restating the radius 0.3 decades off, same label:

         fixed-radius   -0.017      dilating    -0.002
         wide-cutoff    -0.046      s2-rips     -0.004
         scale-free     -0.353

     Inside a plateau the answer barely moves; without one it collapses. beta_0
     at an absolute radius is literally a coordinate of psi, so *some* radius
     always reads *some* partition. The plateau is what makes that partition a
     property of the system rather than of the number you happened to type. On
     the scale-free arm the label radius 2.24 had to be found by matching the
     control's difficulty -- nothing in the system nominates it.

  3. ON COORDINATE ROLLOUT, PSI DOES EXACTLY NOTHING -- IN EVERY ARM.

     Against `no_topology_same_u`, the arm that deletes psi and changes nothing
     else, the k=16 delta is +0.0000 on all four constructed systems. Not small:
     zero to four decimals, at every horizon. The operator selects a
     block-diagonal structure, psi never enters the u-dynamics, and the decode
     reads u alone. On s2-rips psi is actively harmful (+0.2272 / +0.8250 /
     +3.9431).

     Against `linear_only` the same runs read -54.6%, -41.9%, +193.8%, +1.6%.
     This script measured and reported that -54.6% as a topology win before the
     same-u arm existed. It was not one. Raising the PCA rank from 16 to 48
     shrinks the residual `decode_with_residual` carries forward, and in a
     carry-friendly world that handicaps the dimension-matched arm -- the whole
     effect was rank, exactly the confound bench.py now documents. Worth stating
     plainly because the condition's other supporting evidence, Lorenz's ~30%,
     is a dimension-matched number of the same kind.

     So coordinate rollout does not discriminate anything here: it is flat zero
     across a spectrum built to span the condition. Every conclusion above rests
     on the probe, which is where SIGMOID.md 4.3's own win lives too.

  PROPOSED CORRECTED CONDITION.

     The topological channel earns its dimensions when the state is a set of
     interacting entities AND the component count has a plateau in radius --
     a band of radii, fixed relative to the configuration's own scale, on which
     H0 does not change. The centre of the plateau is the interaction radius;
     its width is how much of an error in stating it the channel will absorb.

     Dropped: the requirement that the radius be a fixed *physical* distance.
     Added: the requirement that H0 be stable in a neighbourhood of it, which is
     what makes the radius nameable in advance and the feature transferable.

  WHAT THIS IS WORTH.

     The plateau is measurable without labels, from merge heights the encoder
     already computes -- both diagnostics in this script are ~10 lines over
     sigmoid.h0_barcode. That makes the corrected condition a precondition the
     library could check and report rather than a claim a paper asserts, and it
     is the same machinery AGENDA R2 wants for cloud="auto". It also gives R5 a
     testable shape: if token-window clouds have no H0 plateau, the distilgpt2
     null is predicted rather than mysterious.

  WHAT WOULD STILL BREAK THIS.

     One plant, one seed, 24 entities, 2D, probe-only. The plateau/benefit
     relationship is five points, and s2-rips sits off it -- gap 0.30, as narrow
     as the scale-free arm, yet -0.004 under mis-statement, because its psi win
     runs through the scale-invariant head rather than the absolute radii. That
     is consistent with the correction but shows plateau width alone does not
     rank the arms. The honest next measurement is a radius-plateau sweep at
     fixed geometry, not another system."""


if __name__ == "__main__":
    sys.exit(main())

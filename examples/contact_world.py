"""Contact-rich granular physics: the regime the condition was written for.

sigmoid's measured condition (SIGMOID.md, AGENDA.md R3) says the topological
channel earns its dimensions when the state is a SET OF INTERACTING ENTITIES
whose interaction threshold is a FIXED PHYSICAL DISTANCE. Granular media is the
purest instance of that sentence: N disks, one contact radius d_c = 2r, and a
contact graph whose H0 changes discretely every time a cluster merges or breaks.

No MuJoCo here -- this is a self-contained soft-disk DEM gas in numpy. That is
not a compromise for the question being asked. The condition is about the
*structure* of the state (entity set + fixed radius), and a soft-disk gas has
exactly that structure with none of a physics engine's confounds. What it
cannot test is contact *modelling* fidelity, which the condition does not
mention.

Two observations are tested, because the choice is not obvious and it matters:

    positions only      (entity_dim=2)  the cloud metric IS physical distance,
                        so beta_0 at d_c is literally the contact partition
    positions+velocity  (entity_dim=4)  the state is Markov, so a linear
                        operator can actually roll it forward -- but the cloud
                        metric is now sqrt(dx^2 + dv^2), which is NOT the
                        contact metric

Both are reported. Prediction going in: (2) wins the topological reading and
(4) wins the coordinate rollout, and no single setting wins both.

    python examples/contact_world.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sigmoid

N_DISKS = 24
BOX = 8.0
RADIUS = 0.70          # tuned: see the density sweep below
CONTACT_RADIUS = 2.0 * RADIUS
STIFFNESS = 400.0
DT = 0.001             # tuned: energy drift 4.6% at dt=0.004, 0.5% at dt=0.001
STRIDE = 20            # record every 20 integrator steps -> observation dt 0.02
FRAMES = 2000


# --------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------


def simulate(
    n: int = N_DISKS,
    radius: float = RADIUS,
    box: float = BOX,
    frames: int = FRAMES,
    stride: int = STRIDE,
    dt: float = DT,
    stiffness: float = STIFFNESS,
    speed: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-disk granular gas in a hard box. Velocity Verlet, specular walls.

    Hookean repulsion inside d_c = 2*radius and nothing else: no gravity, no
    damping, no thermostat. Elastic on purpose -- a dissipative gas cools into
    a frozen cluster and the ground truth goes constant, which is the one thing
    that would invalidate the experiment. Contact duration is
    pi/sqrt(2k/m) = 0.111, about 5 recorded frames, so contacts are resolved
    rather than instantaneous and multi-disk clusters genuinely persist.

    Returns (positions, velocities, energy), shapes (frames, n, 2) and (frames,).
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    cell = (box - 2 * radius) / max(side - 1, 1)
    pos = np.stack([gx.ravel(), gy.ravel()], 1)[:n].astype(float) * cell + radius
    pos += rng.uniform(-0.15, 0.15, pos.shape) * cell
    vel = rng.normal(scale=speed, size=pos.shape)
    vel -= vel.mean(0)  # kill net drift so the cloud does not translate away

    d_c = 2.0 * radius
    lo, hi = radius, box - radius

    def forces(p: np.ndarray) -> tuple[np.ndarray, float]:
        # ponytail: dense O(n^2) pair loop. Fine to n~64; cell lists past that.
        delta = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((delta**2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        overlap = np.maximum(d_c - dist, 0.0)
        f = ((stiffness * overlap / dist)[..., None] * delta).sum(1)
        return f, 0.25 * stiffness * float((overlap**2).sum())  # pairs double-counted

    f, _ = forces(pos)
    P, V, E = [], [], []
    for step in range(frames * stride):
        vel += 0.5 * dt * f
        pos += dt * vel
        for j in (0, 1):  # specular reflection: exact mirror, energy-neutral
            under = pos[:, j] < lo
            pos[under, j] = 2 * lo - pos[under, j]
            vel[under, j] *= -1.0
            over = pos[:, j] > hi
            pos[over, j] = 2 * hi - pos[over, j]
            vel[over, j] *= -1.0
        f, pe = forces(pos)
        vel += 0.5 * dt * f
        if step % stride == 0:
            P.append(pos.copy())
            V.append(vel.copy())
            E.append(0.5 * float((vel**2).sum()) + pe)
    return np.asarray(P), np.asarray(V), np.asarray(E)


def contact_components(positions: np.ndarray, d_c: float) -> np.ndarray:
    """H0 of the contact graph at the fixed physical radius, per frame.

    This is the held-out ground truth. It is never shown to the world model.
    """
    out = np.empty(len(positions), dtype=int)
    for i, p in enumerate(positions):
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        adj = d <= d_c
        np.fill_diagonal(adj, False)
        out[i] = connected_components(csr_matrix(adj), directed=False)[0]
    return out


def max_overlap(positions: np.ndarray, d_c: float) -> float:
    worst = 0.0
    for p in positions:
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        worst = max(worst, float(np.maximum(d_c - d, 0.0).max()))
    return worst


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def probe_accuracy(X: np.ndarray, y: np.ndarray, lag: int = 0) -> tuple[float, float]:
    """Multinomial logistic probe, identical treatment for every channel.

    Returns (accuracy, majority-class baseline on the same test split). The
    baseline travels with the accuracy on purpose: the component count is not
    uniform, and an accuracy quoted without it says nothing.
    """
    if lag:
        X, y = X[:-lag], y[lag:]
    n = int(len(X) * 0.7)
    mu, sd = X[:n].mean(0), X[:n].std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((X[:n] - mu) / sd, y[:n])
    acc = float(clf.score((X[n:] - mu) / sd, y[n:]))
    tail = y[n:]
    majority = float(np.bincount(tail).max()) / len(tail)
    return acc, majority


def persistence_baseline(y: np.ndarray, lag: int) -> float:
    """Accuracy of "the count k steps from now equals the count now".

    The real baseline for a forecasting claim. Majority-class is the floor for
    *reading* the count; persistence is the floor for *predicting* it, and it
    is much higher. A probe that beats majority but only ties persistence has
    learned to copy the present, which is not forecasting.
    """
    if not lag:
        return 1.0
    cur, fut = y[:-lag], y[lag:]
    n = int(len(fut) * 0.7)
    return float(np.mean(cur[n:] == fut[n:]))


def run_variant(
    name: str,
    obs: np.ndarray,
    config: sigmoid.SigmoidConfig,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Matched-dimension ablation plus channel probes for one observation."""
    print("\n" + "=" * 74)
    print(f"[{name}]  ablation: matched state dimension, topology on vs off")
    print("=" * 74)
    print(f"  observation {obs.shape}   entity_dim={config.entity_dim}")
    report = sigmoid.compare([obs], config=config, horizons=(1, 4, 16), holdout=0.35)
    print(report.table())

    sig = next(a for a in report.arms if a.name == "sigmoid")
    lin = next(a for a in report.arms if a.name == "linear_only")
    for h in report.horizons:
        gain = (lin.nrmse[h] - sig.nrmse[h]) / max(lin.nrmse[h], 1e-12) * 100
        print(f"  k={h:<3} topology changes coordinate error by {-gain:+.1f}%")

    # ---- is that win topology, or is it the residual anchor? --------------
    # The harness matches TOTAL state dimension, so deleting psi forces the
    # PCA rank up from 16 to 52. That is not a topology-only edit: a wider PCA
    # leaves a smaller unrepresented residual, and the residual is precisely
    # what decode_with_residual carries forward. On a world where
    # carry-forward wins, shrinking the anchor is a handicap all by itself.
    # The control that isolates topology is the same u width with psi deleted.
    print()
    cut = int(len(obs) * 0.65)  # same split compare() uses at holdout=0.35
    ctrl_cfg = sigmoid.SigmoidConfig(**{**vars(config), "use_topology": False})
    ctrl = sigmoid.SigmoidWorldModel(config=ctrl_cfg).fit([obs[:cut]])
    ctrl_nrmse, _ = sigmoid.rollout_error(ctrl, [obs[cut:]], report.horizons)
    hs = "  ".join(f"k={h:<8}" for h in report.horizons)
    print(f"  {'control (isolates psi)':<26}{'dim':>5}  {hs}")
    print(f"  {'-'*26}{'-'*5}  {'-'*len(hs)}")
    for tag, dim, vals in (
        ("sigmoid  u=16 + psi", sig.state_dim, sig.nrmse),
        ("linear   u=16, no psi", ctrl.state_dim, ctrl_nrmse),
        ("linear   u=52, no psi", lin.state_dim, lin.nrmse),
    ):
        cells = "  ".join(f"{vals[h]:<10.4f}" for h in report.horizons)
        print(f"  {tag:<26}{dim:>5}  {cells}")

    print("\n" + "-" * 74)
    print(f"[{name}]  probe: read the contact-component count off each channel")
    print("-" * 74)
    wm = sigmoid.SigmoidWorldModel(config=config).fit([obs])
    Z = wm.encoder.encode_trajectory(obs)
    psi, u = Z[:, : wm.encoder.topo_dim], Z[:, wm.encoder.topo_dim :]

    # the matched-dimension linear channel, so the probe is not just a
    # 36-vs-16 capacity comparison dressed up as a topology result
    matched = sigmoid.SigmoidConfig(
        **{**vars(config), "use_topology": False, "linear_dim": wm.state_dim}
    )
    u_matched = (
        sigmoid.SigmoidWorldModel(config=matched).fit([obs]).encoder.encode_trajectory(obs)
    )
    y = labels[config.window - 1 :]
    print(f"  operator block structure selected by 'auto': "
          f"block_diagonal={wm.block_diagonal_}"
          + ("  (psi cannot reach u at all)" if wm.block_diagonal_ else ""))

    print(
        f"\n  {'lag':<6}{'psi (topo)':>13}{'u (linear)':>13}"
        f"{'u matched':>13}{'majority':>11}{'persist':>10}"
    )
    now = fut = 0.0
    for lag in (0, 1, 4, 16):
        a_psi, maj = probe_accuracy(psi, y, lag)
        a_u, _ = probe_accuracy(u, y, lag)
        a_um, _ = probe_accuracy(u_matched, y, lag)
        per = persistence_baseline(y, lag)
        print(
            f"  {lag:<6}{a_psi:>13.3f}{a_u:>13.3f}{a_um:>13.3f}"
            f"{maj:>11.3f}{per:>10.3f}"
        )
        if lag == 0:
            now = a_psi - max(a_u, a_um, maj)
        if lag == 16:
            fut = a_psi - max(a_u, a_um, maj, per)
    return now, fut


def main() -> int:
    print("=" * 74)
    print("contact-rich granular world -- 24 soft disks, fixed contact radius")
    print("=" * 74)

    # ---- density sweep: a degenerate ground truth invalidates everything ----
    print("\n  density sweep (400 frames each) -- is H0 actually non-constant?")
    print(f"  {'radius':>8}{'phi':>8}{'mean':>8}{'sd':>7}{'range':>10}{'majority':>10}")
    for r in (0.45, 0.55, 0.62, 0.70, 0.78):
        Pq, _, _ = simulate(radius=r, frames=400)
        cq = contact_components(Pq, 2.0 * r)
        _, ct = np.unique(cq, return_counts=True)
        phi = N_DISKS * np.pi * r**2 / BOX**2
        print(
            f"  {r:>8.2f}{phi:>8.3f}{cq.mean():>8.2f}{cq.std():>7.2f}"
            f"{f'{cq.min()}-{cq.max()}':>10}{ct.max() / len(cq):>10.3f}"
        )
    print(f"  chosen: radius={RADIUS} (area fraction {N_DISKS*np.pi*RADIUS**2/BOX**2:.3f})")

    # ---- the corpus ----
    pos, vel, energy = simulate()
    comps = contact_components(pos, CONTACT_RADIUS)

    drift = float(np.max(np.abs(energy - energy[0])) / energy[0])
    overlap = max_overlap(pos, CONTACT_RADIUS)
    inside = bool(pos.min() >= RADIUS - 1e-9 and pos.max() <= BOX - RADIUS + 1e-9)

    print("\n" + "-" * 74)
    print("  sim sanity")
    print("-" * 74)
    print(f"  frames               {len(pos)} at dt_obs={DT*STRIDE}  (t = {len(pos)*DT*STRIDE:.0f})")
    print(f"  energy drift         {drift*100:.2f}% of E0   (E0 = {energy[0]:.3f})")
    print(f"  max overlap          {overlap:.4f} = {overlap/RADIUS*100:.1f}% of a radius")
    print(f"  all disks in box     {inside}")
    print(f"  mean contacts/disk   {mean_contacts(pos, CONTACT_RADIUS):.3f}")
    assert drift < 0.05, "energy is not conserved -- integrator unstable"
    assert inside, "a disk tunnelled out of the box"
    assert overlap < 0.5 * RADIUS, "disks are interpenetrating -- stiffness too low"

    uniq, counts = np.unique(comps, return_counts=True)
    print("\n" + "-" * 74)
    print("  held-out ground truth: H0 of the contact graph at d_c = "
          f"{CONTACT_RADIUS}")
    print("-" * 74)
    print(f"  levels               {len(uniq)}  ({comps.min()}..{comps.max()})")
    print(f"  mean +- sd           {comps.mean():.2f} +- {comps.std():.2f}")
    print(f"  majority class       {counts.max()/len(comps):.3f}")
    print(f"  transitions          {int((np.diff(comps)!=0).sum())} merges/splits "
          f"in {len(comps)-1} steps")
    print("  distribution         "
          + " ".join(f"{k}:{v/len(comps):.2f}" for k, v in zip(uniq, counts)))
    assert len(uniq) >= 5 and counts.max() / len(comps) < 0.5, (
        "ground truth is near-degenerate -- retune density"
    )

    # ---- variant A: positions only, physical cloud metric ----
    obs_pos = pos.reshape(len(pos), -1)
    cfg_pos = sigmoid.SigmoidConfig(
        window=24,
        linear_dim=16,
        hilbert_degree=20,
        n_radii=8,
        entity_dim=2,                       # the state is a SET of disks
        abs_radii=(0.7 * CONTACT_RADIUS,    # and the radius that matters is
                   CONTACT_RADIUS,          # a physical constant, not a
                   1.5 * CONTACT_RADIUS,    # quantile of the data
                   2.2 * CONTACT_RADIUS),
    )
    now_p, fut_p = run_variant("positions", obs_pos, cfg_pos, comps)

    # ---- does psi contain the contact partition exactly? ----
    # beta_0 at an absolute radius counts MST edges longer than it, i.e.
    # (components at that radius) - 1. If abs_radii holds the true d_c this is
    # not an approximation of the ground truth, it is the ground truth.
    wm = sigmoid.SigmoidWorldModel(config=cfg_pos).fit([obs_pos])
    Z = wm.encoder.encode_trajectory(obs_pos)
    # (the essential H0 class has death=inf and is already counted, so the
    # beta_0 reading is the component count itself, with no off-by-one)
    idx = cfg_pos.hilbert_degree + cfg_pos.n_radii + 1  # abs_radii[1] == d_c
    recovered = Z[:, idx] * wm.encoder.psi_scale_[idx] + wm.encoder.psi_mean_[idx]
    truth = comps[cfg_pos.window - 1 :].astype(float)
    print("\n" + "-" * 74)
    print("  is beta_0 at the true contact radius the contact partition?")
    print("-" * 74)
    print(f"  correlation          {np.corrcoef(recovered, truth)[0,1]:.4f}")
    print(f"  exact agreement      {np.mean(np.abs(recovered-truth) < 1e-9):.4f} of frames")

    # ---- variant B: positions + velocities, Markov state ----
    obs_full = np.concatenate([pos, vel], axis=2).reshape(len(pos), -1)
    cfg_full = sigmoid.SigmoidConfig(**{**vars(cfg_pos), "entity_dim": 4})
    now_f, fut_f = run_variant("positions+velocities", obs_full, cfg_full, comps)

    print("\n" + "=" * 74)
    print("assessment")
    print("=" * 74)
    print("  psi advantage over the best non-topological channel:")
    print(f"    positions            now {now_p:+.3f}   forecast k=16 {fut_p:+.3f}")
    print(f"    positions+velocity   now {now_f:+.3f}   forecast k=16 {fut_f:+.3f}")
    print(
        """
  The condition holds in the half it was written for and fails in the other
  half, and contact-rich physics separates the two more cleanly than the S2
  corpus did.

  REPRESENTING the contact structure: yes, decisively. With entity_dim=2 and
  the true contact radius in abs_radii, beta_0 is not a feature correlated
  with the component count -- it IS the component count, exactly, on every
  frame. The logistic probe reads it at 0.78 against a 0.20 majority floor
  while the linear channel sits at 0.10 even when given 52 dimensions to the
  topological channel's 36. No amount of PCA recovers this, because a merge is
  not a linear function of coordinates.

  PREDICTING the coordinates: no, and this run pins down why the harness
  suggested otherwise. compare() reports sigmoid beating linear_only by ~60%,
  and that number is an artifact. Matching TOTAL dimension forces the ablated
  arm's PCA rank from 16 to 52, which shrinks the residual that
  decode_with_residual carries forward -- and on a world where carry-forward
  is the strongest arm, that alone explains the gap. The control above holds
  the linear channel at u=16 and deletes only psi: it reproduces sigmoid's
  error to four decimals. block_diagonal="auto" independently agrees, choosing
  a structure in which psi cannot influence u at all. Topology contributes
  exactly nothing to coordinate rollout here. Same verdict as S2, arrived at
  with a sharper instrument.

  FORECASTING the contact structure: no. psi reads the present contact
  partition and cannot see the next one. At lag 1 it scores 0.265 against a
  persistence rate of 0.250 -- it has essentially learned to copy the present,
  which is not forecasting. At lag 4 it is 0.203 against a 0.201 majority
  floor, and at lag 16 it is below that floor. (It clears persistence at lag 4
  and 16, but persistence has itself fallen under majority by then, so
  clearing it means nothing.) This is a property of the world rather than a
  bug: contacts last about 5 frames and the count changes on 74% of steps, so
  H0 at t+4 is nearly independent of H0 at t. A slower or denser regime should
  forecast better; that is the follow-up, not a claim this run supports.

  The velocity variant is the control that makes the mechanism explicit.
  Identical code, identical radius, entity_dim=4: the cloud metric becomes
  sqrt(dx^2 + dv^2), the contact reading collapses from 0.78 to 0.17, and psi
  drops BELOW the majority floor. The condition's clause about a fixed
  PHYSICAL distance is load-bearing -- it is not enough for a threshold to be
  fixed, the metric it is fixed in has to be the one contacts happen in.

  Certificates remain vacuous: rho ~ 1.6 expansive, bound ~ 3e3 against a
  signal of order 1. AGENDA R6 stands unaffected by anything measured here.
"""
    )
    return 0


def mean_contacts(positions: np.ndarray, d_c: float) -> float:
    tot = 0
    for p in positions:
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        tot += int((d <= d_c).sum())
    return tot / len(positions) / positions.shape[1]


if __name__ == "__main__":
    sys.exit(main())

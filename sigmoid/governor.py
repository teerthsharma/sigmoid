"""Governor: read a SigmoidConfig off the data instead of guessing it by hand.

Ported from Teerth Sharma's LAAMBA `topological_governor_full.py`, which measures
13 scalar *vitals* of a data matrix and maps them to a configuration. The vitals
port over; the target does not -- LAAMBA picks a Riemannian manifold, sigmoid
picks encoder and operator settings.

## Why sigmoid needs one

Every knob below decides whether the encoder measures anything, and every wrong
setting is silent -- healthy psi variance, sensible standardization, no
information:

    entity_dim      the wrong cloud scored 0.386 against a 0.487 majority
                    baseline on the S2-Rips corpus, i.e. below chance
    absolute radii  keeping the scale took 0.764 -> 0.855 on the same corpus
    linear_dim      a rank above min(n, d) is zero-padded, so those state
                    dimensions are exactly 0.0
    window          above the trajectory length, `TopoEncoder.fit` raises

## Three things the source does that had to change

1. **Unseeded `np.random.choice`.** Deterministic replay is a hard requirement
   here (`telemetry.DeterministicReplay`), so every subsample takes a seed.
2. **The Gram identity for pairwise distances.** `state._pairwise` documents the
   cost: for points at offset 1e6 with 1e-6 separation it cancels to *exactly*
   0.0 and two distinct points become one. Distances go through `_pairwise`.
3. **Squared distances inside the Levina-Bickel MLE.** The source feeds
   `((S-S)**2).sum(2)` to the estimator with no square root. Since
   `log(r_k^2/r_j^2) = 2 log(r_k/r_j)` the reciprocal halves, so the source
   reports exactly half the true intrinsic dimension. `demo()` prints both on a
   5-dim gaussian in R^20: 5.0 fixed against 2.5 as written.

Also: the source's O(n^2) caps are 2048 (spectral gap) and 4096 (intrinsic dim),
which means a 2048x2048 dense `eigvalsh` per call -- tens of seconds, not the
milliseconds a calibration probe can afford. The cap is 512 here and `demo()`
prints the runtime that buys.

## Seven of the thirteen vitals were dropped

A vital that drives nothing is dead weight, so it is not computed at all. Kept,
each with a named consumer: `log_n` (window), `log_d` and `aspect` (linear_dim
cap), `intrinsic_dim` (linear_dim floor warning), `spectral_gap` (confidence
contradiction check), `nan_fraction` (input scrubbing and confidence). Dropped:

    dist_mean, dist_std   cloud scale; every sigmoid decision that cares about
                          scale reads the H0 plateau instead, which is the same
                          information localized at the radius that matters
    dist_p95_p5           dynamic range; the plateau's own span covers it
    knee_clusters         a cluster count no sigmoid knob consumes -- what it
                          needs is whether beta_0 is *stable*, which is the
                          plateau -- and 10 k-means fits cost more than every
                          other vital combined. It is also the one vital that
                          cannot be made reproducible without pinning sklearn's
                          version, since `KMeans(n_init=3, random_state=42)`
                          fixes the seed but not the initialisation algorithm
    curvature_proxy       LAAMBA picks a curvature; sigmoid has no curvature knob
    small_world           path length over clustering of a kNN graph; no sigmoid
                          knob reads graph topology, and it was second-slowest
    sparsity              `mean(X == 0.0)` counts *exact* zeros, and every matrix
                          on a sigmoid path is dense float. Measured: 0.000000 on
                          the S2 corpus, 0.000000 on the Lorenz lift, 0.0025 on an
                          AR(1) sequence. It reads 0.145 on `TopoImageEncoder`
                          output, where the nonzero value is an artifact of
                          integer Betti counts being 0 rather than a statement
                          about the data

## No PolicyNet

The source maps vitals to a config through a two-layer network that is never
trained before its first prediction, then *samples* from its softmax: the first
recommendation is noise and two calls on the same data disagree. A randomly
initialised network is strictly worse than a documented heuristic, so there is
none here. `recommend` is rules with a cited measurement per field; `update`
keeps a small history and a nearest-neighbour recall over it.

## What the criterion is, and what it is not

`state.select_cloud` scores psi self-predictability minus a shuffled-order floor.
The floor exists because overlapping windows make a temporal psi nearly its own
successor -- 0.56 to 0.82 for pure artifact -- and AGENDA R5 records that even
with the floor the criterion reads 0.762 on the distilgpt2 encoder, exactly the
case it should reject. Two replacements were measured here and both failed:

    psi -> next state (AGENDA's own suggestion)
        ridge-sensitive to the point of uselessness. The S2 corpus needs
        ridge 1e-3 to rank the right width first, Lorenz needs 1e2, the vision
        encoder needs <= 1. No single value gets all three; picking one per
        system is fitting the criterion to answers already known.
    held-out rollout NRMSE (the objective itself, not a proxy)
        1 of 5. It ranks the *bogus* width first on the S2 corpus, and it is
        right to: README section 9 records that the S2 partition contributes
        exactly zero to coordinate rollout. Representation and prediction are
        different problems and the ground-truth labels come from the first.

What is shipped is the plateau delta below: an H0 plateau measured against a
*point-count-matched* temporal control. The control is the whole content of it --
see `PlateauReport.excess`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .engine import SigmoidConfig
from .state import _pairwise, _windows, h0_barcode

__all__ = [
    "ENTITY_DELTA_THRESHOLD",
    "VITAL_NAMES",
    "PlateauReport",
    "Recommendation",
    "SigmoidGovernor",
    "Vitals",
    "VitalsExtractor",
    "plateau_diagnostics",
]

VITAL_NAMES = (
    "log_n",
    "log_d",
    "aspect",
    "intrinsic_dim",
    "spectral_gap",
    "nan_fraction",
)

ENTITY_DELTA_THRESHOLD = 1.0
"""How far a spatial cloud's plateau must beat its matched temporal control.

Not fitted to a boundary. Measured over eleven systems: every non-entity system
tried peaks at or below +0.84 (four Lorenz seeds +0.04 to +0.84, a uniform cloud
+0.24, a gaussian +0.03, an AR(1) sequence +0.24, the vision encoder +0.76) and
every entity system clears +1.19 (S2-Rips at 11, 15, 21, 22 and 28 nodes: +1.19,
+3.41, +3.20, +4.41, +5.49). The gap between the two families is a factor of
about 1.4 at its narrowest, so 1.0 sits inside it rather than on either edge.
"""


# --------------------------------------------------------------------------
# vitals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Vitals:
    """Six scalars describing a data matrix. Every one is finite by construction.

    A vital returning NaN silently poisons the policy: comparisons against it are
    False in *both* directions, so the fallback branch wins and nobody is told.
    Each extractor therefore clamps rather than propagates, and
    `tests/test_governor.py` asserts finiteness on n=1, d=1, identical rows, NaN
    input and n < k.
    """

    log_n: float
    log_d: float
    aspect: float
    """n / d. Below 1 the PCA rank is limited by samples, not by width."""

    intrinsic_dim: float
    """Levina-Bickel maximum-likelihood estimate, k=5, on true distances."""

    spectral_gap: float
    """lambda_1 / lambda_max of the kNN graph Laplacian. Near 0 means the graph
    nearly disconnects -- cluster structure. 1.0 means one blob."""

    nan_fraction: float

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in VITAL_NAMES], dtype=np.float64)

    def table(self) -> str:
        return "  ".join(f"{n} {getattr(self, n):.3f}" for n in VITAL_NAMES)


def _finite(x: float, fallback: float = 0.0) -> float:
    """Degenerate input must yield a number, not a NaN that survives comparison."""
    return float(x) if np.isfinite(x) else float(fallback)


@dataclass
class VitalsExtractor:
    """Data matrix -> `Vitals`. Bit-identical across runs for a fixed seed."""

    seed: int = 0
    max_points: int = 512
    """Cap for the O(n^2) and O(n^3) vitals."""

    k_id: int = 5
    """Neighbour count for the intrinsic-dimension MLE, as in the source."""

    k_graph: int = 15
    """Neighbour count for the spectral-gap graph, as in the source -- but capped
    at n // 3. The source's fixed 15 builds a *complete* graph on any cloud of 16
    points or fewer, whose Laplacian gap is exactly 1.0 whatever the geometry.
    Measured on a 14-node S2 configuration with two tight caps: fixed k reads
    1.0000, "no cluster structure", on a cloud that is nothing but clusters."""

    def __call__(self, X: np.ndarray) -> Vitals:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2 or x.size == 0:
            raise ValueError(f"expected a non-empty 2D matrix, got shape {x.shape}")
        n, d = x.shape

        # The NaN fraction is measured on the raw matrix and the NaNs are then
        # replaced, because cdist propagates them: one non-finite cell turns the
        # whole distance matrix NaN and every geometric vital with it. Reporting
        # the fraction and continuing beats returning five NaNs.
        nan_fraction = float(np.mean(~np.isfinite(x)))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        s = self._subsample(x)
        dist = _pairwise(s) if s.shape[0] > 1 else np.zeros((1, 1))
        return Vitals(
            log_n=float(np.log1p(n)),
            log_d=float(np.log1p(d)),
            aspect=float(n / max(d, 1)),
            intrinsic_dim=self._intrinsic_dim(dist, d),
            spectral_gap=self._spectral_gap(dist),
            nan_fraction=nan_fraction,
        )

    def _subsample(self, x: np.ndarray) -> np.ndarray:
        """Seeded, so a replay reproduces the vitals bit for bit.

        The source draws from the global RNG, which means two runs of the same
        pipeline can disagree about which config to use.
        """
        if x.shape[0] <= self.max_points:
            return x
        rng = np.random.default_rng(self.seed)
        idx = np.sort(rng.choice(x.shape[0], self.max_points, replace=False))
        return x[idx]

    def _intrinsic_dim(self, dist: np.ndarray, ambient: int) -> float:
        """Levina-Bickel MLE on *true* distances, clipped to [0, ambient].

        The clip is not cosmetic. On identical rows every neighbour distance is
        0, the log ratios vanish and the reciprocal blows up to ~1e9 -- finite,
        and still poison for a nearest-neighbour recall that normalizes by
        per-vital spread.
        """
        n = dist.shape[0]
        k = min(self.k_id, n - 1)
        if n < 3 or k < 2:
            return 0.0
        # Partition, not a full sort: only the k+1 smallest matter, and at n=512
        # the full sort was 7.9 ms of a 156 ms extractor for no gain.
        near = np.sort(np.partition(dist, k, axis=1)[:, : k + 1], axis=1)[:, 1:]
        r_k = near[:, -1:]
        live = (near[:, 0] > 1e-12) & (r_k[:, 0] > 1e-12)
        if not live.any():
            return 0.0  # coincident points have no dimension to estimate
        mean_log = float(np.log(r_k[live] / near[live, :-1]).mean())
        if mean_log <= 1e-12:
            return 0.0
        return _finite(np.clip(1.0 / mean_log, 0.0, ambient))

    def _spectral_gap(self, dist: np.ndarray) -> float:
        """lambda_1 / lambda_max of the symmetrized kNN graph Laplacian.

        Returns 1.0 -- "one blob, no cluster structure" -- for anything too small
        to have a second eigenvalue, so a rule reading it sees "no evidence"
        rather than "strong evidence" on degenerate input. 0.0 would be a
        confident lie in the other direction.
        """
        n = dist.shape[0]
        k = min(self.k_graph, max(1, n // 3), n - 1)
        if n < 4 or k < 1:
            return 1.0
        # The k+1 smallest distances always include the point itself at 0, so take
        # k+1 by partition and clear the diagonal afterwards -- exactly k
        # neighbours per row, without a full O(n^2 log n) argsort.
        knn = np.argpartition(dist, k, axis=1)[:, : k + 1]
        w = np.zeros((n, n))
        w[np.repeat(np.arange(n), k + 1), knn.ravel()] = 1.0
        np.fill_diagonal(w, 0.0)
        w = np.maximum(w, w.T)
        # eigvalsh on the dense Laplacian is 84 ms of this at n=512 and dominates
        # the extractor. A partial LAPACK range solve was measured slower (132 ms
        # for two subset calls against 110 ms for the full one), and eigsh on the
        # sparse form needs a shift-invert factorization to reach lambda_1 near a
        # zero eigenvalue. Kept dense; the cap at max_points is the real control.
        lam = np.linalg.eigvalsh(np.diag(w.sum(axis=1)) - w)
        return _finite(lam[1] / lam[-1], 1.0) if lam[-1] > 1e-12 else 1.0


# --------------------------------------------------------------------------
# the plateau diagnostic
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateauReport:
    """The widest band of log-radius on which the H0 partition does not change.

    Merge heights *are* H0 death times (`state.h0_barcode`), so beta_0 is
    constant exactly between consecutive death times and the widest such
    interval is the widest band where the partition is stable. Read off the gaps
    themselves, not a sampled grid: a grid coarser than the merge spacing reports
    "no plateau" for a real one, and a finer grid costs distance matrices for
    nothing.

    Label-free, which is the point. README section 9's representation clause is
    stated in these terms -- psi reads the partition when beta_0 has a band of
    constant value in radius, the band centre is the interaction radius, and its
    width is how much error in stating that radius the encoder absorbs.
    """

    centre: float
    """Geometric centre of the band: the interaction radius, in data units."""

    width_decades: float
    """log10 width of the band. How wrong the stated radius may be."""

    components: float
    """beta_0 on the band -- how many pieces the partition has there."""

    excess: float
    """Band width over the mean gap between merge heights.

    **Only meaningful against a matched control.** A cloud of m points has m-1
    merges, and the largest of g log-gaps grows like the harmonic number H_g
    whatever the geometry is doing, so excess rises with point count on its own.
    Measured on uniform random clouds: 2.07 at m=6, 4.14 at m=21, 5.24 at m=32.
    That is why raw excess cannot rank candidate clouds -- it hands the win to
    whichever reshape has the most points -- and why `SigmoidGovernor.recommend`
    compares each spatial cloud against a *temporal* cloud of the same point
    count. See `ENTITY_DELTA_THRESHOLD` for what the controlled statistic
    separates.
    """

    drift_decades: float
    """log10 of max/min band centre across frames.

    Reported, never used to reject. A system whose threshold dilates across three
    decades is read just as well (0.940 against 0.382 linear), which is what
    falsified the "fixed physical distance" clause -- README section 9. Compared
    against `width_decades` it says which absolute-radius setting fits: drift
    inside the width means one stated `abs_radii` covers every frame, drift far
    outside it means the learned `n_abs_radii` quantiles have to span the range.
    """

    n_frames: int
    centres: tuple[float, ...] = field(default=(), repr=False)

    @property
    def readable(self) -> bool:
        """Whether a band exists at all -- excess above 1.0 on some frame.

        Deliberately weak, and stated as weak: the statistic does *not* separate
        a structured temporal cloud from noise. Lorenz reads 4.25 and a uniform
        random cloud of the same shape reads 5.37. All this refuses is a cloud
        with fewer than two distinct merges, where an absolute radius has nothing
        to sit between. The discriminating test is the matched-control delta.
        """
        return self.n_frames > 0 and self.excess > 1.0

    def table(self) -> str:
        return (
            f"centre {self.centre:>9.4f}  width {self.width_decades:>6.3f}dec  "
            f"beta_0 {self.components:>5.1f}  excess {self.excess:>6.2f}  "
            f"drift {self.drift_decades:>6.2f}dec  frames {self.n_frames:>3d}"
        )


_EMPTY_PLATEAU = PlateauReport(0.0, 0.0, 0.0, 0.0, 0.0, 0)


def _plateau_one(cloud: np.ndarray) -> tuple[float, float, float, float] | None:
    """(centre, width, beta_0, excess) for one point cloud, or None."""
    bc = h0_barcode(cloud)
    if bc.diameter <= 0.0:
        # Coincident or single points. h0_barcode reports diameter 0 and an
        # essential bar at inf, and inf * 0.0 is NaN -- guard rather than filter,
        # so the warning never fires and the intent is visible.
        return None
    deaths = bc.bars[:, 1] * bc.diameter
    heights = np.unique(deaths[np.isfinite(deaths) & (deaths > 0.0)])
    if heights.size < 2:
        return None  # fewer than two distinct merges: no band to measure
    lg = np.log10(heights)
    gaps = np.diff(lg)
    k = int(np.argmax(gaps))
    span = float(lg[-1] - lg[0])
    return (
        float(10 ** ((lg[k] + lg[k + 1]) / 2)),
        float(gaps[k]),
        float((deaths > heights[k]).sum()),
        float(gaps[k] / (span / gaps.size)) if span > 1e-12 else 1.0,
    )


def plateau_diagnostics(
    X: np.ndarray,
    *,
    entity_dim: int = 0,
    window: int = 32,
    n_frames: int = 24,
) -> PlateauReport:
    """H0 plateau of the clouds a `TopoEncoder` would actually see.

    `X` is a (T, D) trajectory. With `entity_dim=0` the frames are its sliding
    windows -- the temporal cloud -- and with `entity_dim=w` each row reshapes to
    (D/w, w), the spatial cloud at one instant. A trajectory shorter than
    `window` is treated as a single temporal frame, so `plateau_diagnostics(cloud)`
    does the obvious thing on a bare point cloud.

    Temporal frames are standardized and spatial frames are not, matching
    `TopoEncoder._topo_window`: the diagnostic has to describe the cloud psi is
    computed over, and per-coordinate standardization moves the merge heights.
    Without it the vision encoder's pixel-count column (~2300) sets the whole
    scale and the reported radius belongs to that one feature.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or x.size == 0:
        raise ValueError(f"expected a non-empty 2D trajectory, got shape {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if entity_dim:
        if x.shape[1] % entity_dim:
            raise ValueError(
                f"observation width {x.shape[1]} is not a multiple of "
                f"entity_dim={entity_dim}"
            )
        frames: list[np.ndarray] = [row.reshape(-1, entity_dim) for row in x]
    else:
        x = _robust_standardize(x)
        w = max(2, min(int(window), x.shape[0]))
        frames = [x] if x.shape[0] <= w else list(_windows(x, w))

    stride = max(1, len(frames) // max(int(n_frames), 1))
    found = [r for r in (_plateau_one(f) for f in frames[::stride]) if r is not None]
    if not found:
        return _EMPTY_PLATEAU

    arr = np.asarray(found, dtype=np.float64)
    centres = arr[:, 0]
    return PlateauReport(
        centre=float(np.median(centres)),
        width_decades=float(np.median(arr[:, 1])),
        components=float(np.median(arr[:, 2])),
        excess=float(np.median(arr[:, 3])),
        drift_decades=float(np.log10(centres.max() / max(centres.min(), 1e-300))),
        n_frames=int(arr.shape[0]),
        centres=tuple(centres.tolist()),
    )


def _robust_standardize(traj: np.ndarray) -> np.ndarray:
    """Median/MAD, matching `TopoEncoder.fit`, so the plateau is measured on the
    same numbers the encoder will see."""
    centre = np.median(traj, axis=0)
    scale = 1.4826 * np.median(np.abs(traj - centre), axis=0)
    fallback = traj.std(axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    return (traj - centre) / scale


# --------------------------------------------------------------------------
# the governor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    config: SigmoidConfig
    rationale: dict[str, str]
    """Per field: which vital or diagnostic drove it, and the measurement that
    grounds the rule. Every entry names its evidence."""

    confidence: float
    """[0, 1]. High needs one unambiguous winner, clear of the control by a
    margin, on clean input."""

    vitals: Vitals
    plateaus: dict[int, PlateauReport]
    """One per candidate entity width, including the ones that lost."""

    controls: dict[int, float]
    """Point-count-matched temporal control excess per candidate width. The
    candidate's own excess minus this is what the choice was made on."""

    recalled: bool = False
    """True when a scored history entry on similar vitals overrode the rules."""

    @property
    def deltas(self) -> dict[int, float]:
        return {
            w: self.plateaus[w].excess - self.controls.get(w, 0.0)
            for w in self.plateaus
            if w
        }

    def table(self) -> str:
        rows = [
            f"  confidence {self.confidence:.2f}"
            + ("  (recalled from history)" if self.recalled else ""),
            f"  vitals  {self.vitals.table()}",
        ]
        for name, why in self.rationale.items():
            value = f"{getattr(self.config, name)!r}" if hasattr(self.config, name) else ""
            rows.append(f"  {name:<15}{value:>8}  {why}")
        return "\n".join(rows)


@dataclass
class SigmoidGovernor:
    """Vitals plus a control-corrected H0 plateau -> a `SigmoidConfig`."""

    extractor: VitalsExtractor = field(default_factory=VitalsExtractor)
    history: list[tuple[Vitals, SigmoidConfig, float]] = field(default_factory=list)
    recall_radius: float = 0.25
    """Per-vital relative distance below which a scored history entry is reused."""

    _last_vitals: Vitals | None = field(default=None, repr=False)

    # ---- recommend --------------------------------------------------------

    def recommend(
        self,
        trajectories: Sequence[np.ndarray],
        *,
        entity_dim_candidates: Sequence[int] = (0,),
    ) -> Recommendation:
        """Pick a config from `trajectories`, label-free.

        `entity_dim_candidates` comes from the caller and is never inferred.
        `state.select_cloud` documents why: a width that happens to divide the
        observation dimension is not evidence that the state is a set of entities
        of that width. Offered a bogus 3 on a 24-dim Lorenz lift, its
        self-predictability criterion took it -- 0.42 against 0.27 on all four
        seeds tried -- and the resulting world model rolled out *worse* (0.404
        against 0.318 nrmse at k=4). Width 0 is always available as the fallback
        because it needs no width at all; it is the documented default cloud, not
        a guess.

        Known miss, measured. The delta separates entity structure from no
        entity structure cleanly (see `ENTITY_DELTA_THRESHOLD`) but among
        *several* structured widths it prefers the one with more points. On S2
        configurations with an even node count, where the observation dimension
        3n is also divisible by 2, the interleaved width-2 reshape keeps the
        cluster structure with 1.5x the points and wins: 22 nodes read +4.41 at
        width 2 against +3.65 at the correct 3. Odd node counts (11, 15, 21) are
        correct. When more than one candidate clears the control the confidence
        is halved and the rationale names the runners-up.
        """
        trajs = [np.asarray(t, dtype=np.float64) for t in trajectories]
        if not trajs:
            raise ValueError("recommend needs at least one trajectory")
        if any(t.ndim != 2 for t in trajs):
            raise ValueError("each trajectory must be 2D (T, D)")
        if len({t.shape[1] for t in trajs}) != 1:
            raise ValueError("trajectories must share an observation width")
        joined = np.concatenate(trajs, axis=0)
        n, d = joined.shape

        vitals = self.extractor(joined)
        self._last_vitals = vitals

        widths = list(dict.fromkeys(int(w) for w in entity_dim_candidates))
        if not widths:
            raise ValueError("entity_dim_candidates must not be empty")
        default = SigmoidConfig()

        # --- the plateau, per candidate, each against its own matched control --
        plateaus: dict[int, PlateauReport] = {}
        controls: dict[int, float] = {}
        plateaus[0] = plateau_diagnostics(joined, window=default.window)
        for w in widths:
            if w == 0:
                continue
            if d % w:
                plateaus[w] = _EMPTY_PLATEAU  # not a divisor: cannot be a cloud
                controls[w] = 0.0
                continue
            plateaus[w] = plateau_diagnostics(joined, entity_dim=w)
            # The control: a *temporal* cloud of exactly D/w points. Same number
            # of merge heights, therefore the same null for the widest-gap
            # statistic, therefore the difference is about arrangement and not
            # about point count. Same shape of correction as select_cloud's
            # shuffled-order floor, for the same reason.
            controls[w] = plateau_diagnostics(joined, window=d // w).excess

        deltas = {w: plateaus[w].excess - controls[w] for w in widths if w}
        passing = sorted(
            (w for w, dl in deltas.items() if dl > ENTITY_DELTA_THRESHOLD),
            key=lambda w: -deltas[w],
        )
        entity_dim = passing[0] if passing else 0
        best_delta = (
            deltas[entity_dim] if entity_dim else max(deltas.values(), default=0.0)
        )

        rationale: dict[str, str] = {}
        offered = ", ".join(f"{w}:{deltas[w]:+.2f}" for w in sorted(deltas)) or "none"
        if entity_dim:
            rationale["entity_dim"] = (
                f"plateau excess {plateaus[entity_dim].excess:.2f} beats its "
                f"point-count-matched temporal control {controls[entity_dim]:.2f} by "
                f"{best_delta:+.2f} > {ENTITY_DELTA_THRESHOLD} (all candidates "
                f"{offered}); README s9 representation clause. "
                f"spectral_gap {vitals.spectral_gap:.3f}"
                + (
                    " corroborates -- the kNN graph nearly disconnects, so there is"
                    " real cluster structure worth an entity cloud"
                    if vitals.spectral_gap < 0.5
                    else " does NOT corroborate: the kNN graph stays connected, so"
                    " the delta is the only evidence"
                )
                + (
                    f"; {len(passing)} candidates cleared the control, and the delta"
                    f" prefers point count among structured widths -- confidence halved"
                    if len(passing) > 1
                    else ""
                )
            )
        else:
            rationale["entity_dim"] = (
                f"no candidate beat its matched temporal control by "
                f"{ENTITY_DELTA_THRESHOLD} (best {best_delta:+.2f}; all {offered}), so "
                f"the temporal cloud stands. intrinsic_dim {vitals.intrinsic_dim:.1f} of "
                f"{d} ambient with no controlled plateau is the temporal case"
            )

        # --- absolute vs normalized radius ---------------------------------
        # README s9: psi reads the partition when beta_0 has a band of constant
        # value in radius. Keeping the scale was 0.764 -> 0.855 on the S2 corpus.
        chosen = plateaus[entity_dim]
        if chosen.readable:
            n_abs = default.n_abs_radii
            fits_one = chosen.drift_decades <= chosen.width_decades
            rationale["n_abs_radii"] = (
                f"band at {chosen.centre:.4f} is {chosen.width_decades:.3f} dec wide "
                f"(excess {chosen.excess:.2f}); keeping absolute scale was "
                f"0.764 -> 0.855 on S2-Rips. drift {chosen.drift_decades:.2f} dec "
                + (
                    "sits inside the band, so a single stated abs_radii would also do"
                    if fits_one
                    else "exceeds it, so learned quantiles are needed to span the range"
                    " -- a 3-decade dilation still reads 0.940, README s9"
                )
            )
        else:
            n_abs = 0
            rationale["n_abs_radii"] = (
                f"fewer than two distinct merge heights on the chosen cloud "
                f"(excess {chosen.excess:.2f}): no band for an absolute radius to sit "
                f"in. Note this test is weak -- Lorenz reads 4.25 and a uniform "
                f"random cloud of the same shape reads 5.37, so it only refuses "
                f"degenerate clouds"
            )

        # --- window ---------------------------------------------------------
        # Hard constraint, not a heuristic: TopoEncoder.fit raises below
        # window+1 samples and bench.compare holds out 30% before fitting.
        shortest = min(t.shape[0] for t in trajs)
        window = default.window
        if shortest < default.window * 2:
            window = max(4, int(shortest * 0.6) // 2 * 2)
            rationale["window"] = (
                f"log_n {vitals.log_n:.2f} (shortest trajectory {shortest}): "
                f"fit() raises below window+1 and compare() holds out 30% first, so "
                f"the default {default.window} would not fit"
            )
        else:
            rationale["window"] = (
                f"log_n {vitals.log_n:.2f}: {shortest} samples leave room for the "
                f"default {default.window} after a 30% holdout"
            )

        # --- linear_dim -----------------------------------------------------
        # PCA rank is min(n, d); above it fit() zero-pads components_, so those
        # state dimensions are exactly 0.0 and cost budget for nothing. compare()
        # matches arms on total dimension, so padding also skews the ablation.
        linear_dim = int(min(default.linear_dim, max(2, min(d, int(n * 0.7) - window))))
        rationale["linear_dim"] = (
            f"log_d {vitals.log_d:.2f} (d={d}), aspect {vitals.aspect:.2f}: PCA rank is "
            f"min(n, d)={min(n, d)}, above which fit() zero-pads components_"
            + (
                f" -- WARNING intrinsic_dim {vitals.intrinsic_dim:.1f} exceeds it, the "
                f"linear channel cannot represent the manifold"
                if vitals.intrinsic_dim > linear_dim
                else ""
            )
        )

        # --- hilbert_degree -------------------------------------------------
        # hilbert_coefficients bins births and deaths over normalized radius. A
        # cloud of m points has m-1 finite deaths, so a degree above m-1
        # guarantees empty bins -- constant features, which psi_scale_ pins to
        # 1.0 and the operator then carries as dead columns.
        cloud_points = (d // entity_dim) if entity_dim else window
        hilbert_degree = int(min(default.hilbert_degree, max(4, cloud_points - 1)))
        rationale["hilbert_degree"] = (
            f"the chosen cloud has {cloud_points} points, so {cloud_points - 1} finite "
            f"H0 deaths; a higher degree guarantees empty bins"
        )

        # --- block_diagonal -------------------------------------------------
        # Deliberately left at "auto". The engine already fits both structures
        # and keeps the held-out winner, and what it arbitrates is genuinely
        # two-sided. A vitals heuristic here would replace a measurement with a
        # guess.
        rationale["block_diagonal"] = (
            "left 'auto': the engine fits both and keeps the held-out winner. Lorenz "
            "prefers coupled (0.052 against 0.073 at k=1), distilgpt2 prefers "
            "block-diagonal (0.145 against 0.168) -- no vital decides this, a fit does"
        )

        # --- rho_max --------------------------------------------------------
        rationale["rho_max"] = (
            "left None: forcing rho < 1 cost Lorenz 0.067 -> 0.317 and the certificate "
            "stayed vacuous anyway (6.9-368 against a ~1 signal), AGENDA s3"
        )

        config = replace(
            default,
            window=window,
            hilbert_degree=hilbert_degree,
            n_abs_radii=n_abs,
            linear_dim=linear_dim,
            entity_dim=entity_dim,
        )

        # --- confidence -----------------------------------------------------
        # Distance from the decision boundary in units of the threshold, halved
        # when several widths cleared the control (the measured failure mode),
        # scaled down by the fraction of the input that was not finite.
        margin = abs(best_delta - ENTITY_DELTA_THRESHOLD) / ENTITY_DELTA_THRESHOLD
        confidence = float(np.clip(margin, 0.0, 1.0))
        if len(passing) > 1:
            confidence *= 0.5
        confidence *= 1.0 - vitals.nan_fraction
        if vitals.nan_fraction > 0.0:
            rationale["_input"] = (
                f"nan_fraction {vitals.nan_fraction:.4f}: non-finite cells were zeroed "
                f"before any distance (cdist propagates them) and confidence scaled "
                f"down by that fraction"
            )

        recalled = self._recall(vitals)
        if recalled is not None:
            config, score = recalled
            rationale["_recall"] = (
                f"a history entry within {self.recall_radius:.2f} per-vital relative "
                f"distance scored {score:.4f}, at or above the recorded median; reusing "
                f"its config over the rules"
            )

        return Recommendation(
            config=config,
            rationale=rationale,
            confidence=confidence,
            vitals=vitals,
            plateaus=plateaus,
            controls=controls,
            recalled=recalled is not None,
        )

    # ---- history ----------------------------------------------------------

    def update(self, config: SigmoidConfig, score: float) -> None:
        """Record (vitals, config, score) against the last `recommend`.

        `score` is higher-is-better, so pass an accuracy or a negated error.

        Nearest neighbour over recorded vitals, not a policy network -- see the
        module docstring on `PolicyNet`. Ridge would need more history than a
        calibration run produces; k=1 needs one row and degrades to "no advice"
        rather than to a fitted line through two points.
        """
        if self._last_vitals is None:
            raise RuntimeError("recommend() must be called before update()")
        self.history.append((self._last_vitals, config, float(score)))

    def _recall(self, vitals: Vitals) -> tuple[SigmoidConfig, float] | None:
        """Nearest recorded vitals, if close enough and if it scored well."""
        if len(self.history) < 2:
            return None  # one point defines neither a scale nor a median
        recorded = np.stack([h[0].vector() for h in self.history])
        scores = np.array([h[2] for h in self.history])
        spread = recorded.std(axis=0)
        spread = np.where(spread > 1e-9, spread, 1.0)
        dist = np.linalg.norm((recorded - vitals.vector()) / spread, axis=1)
        dist /= np.sqrt(recorded.shape[1])  # per-vital, so the radius is readable
        i = int(np.argmin(dist))
        if dist[i] > self.recall_radius or scores[i] < np.median(scores):
            return None
        return self.history[i][1], float(scores[i])


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _lorenz(n: int = 1200, dt: float = 0.01, dim: int = 24, seed: int = 0) -> np.ndarray:
    """A real nonlinear system lifted into a higher-dimensional hidden space.

    Same generator as `tests/test_sigmoid.py::lorenz`. Ground truth
    `entity_dim=0`: the state is a trajectory, not a set of entities, and the
    24-dim lift's divisors are exactly the bogus widths select_cloud loses to.
    """
    rng = np.random.default_rng(seed)
    x, y, z = 1.0, 1.0, 1.0
    pts = np.empty((n, 3))
    for i in range(n):
        x += dt * 10.0 * (y - x)
        y += dt * (x * (28.0 - z) - y)
        z += dt * (x * y - 8.0 / 3.0 * z)
        pts[i] = (x, y, z)
    pts = (pts - pts.mean(0)) / pts.std(0)
    lift = rng.normal(size=(3, dim)) / np.sqrt(3)
    return pts @ lift + 0.01 * rng.normal(size=(n, dim))


def _vision_video(frames: int = 160, size: int = 48, seed: int = 0) -> np.ndarray:
    """`TopoImageEncoder` output for a disk whose hole opens and closes.

    Ground truth `entity_dim=0`. The feature vector is a heterogeneous list of
    scalars -- euler curve, betti numbers, patch occupancy, a pixel count -- not
    a set of entities sharing a metric, so reshaping it means nothing.
    """
    from .vision import TopoImageEncoder
    from .vision.encode import _disk

    rng = np.random.default_rng(seed)
    seq = []
    for t in range(frames):
        radius = 8.0 * abs(np.sin(np.pi * t / 37.0))
        mask = _disk(size, size / 2, size / 2, size / 2.4)
        if radius >= 3.0:
            mask = mask & ~_disk(size, size / 2, size / 2, radius)
        seq.append(np.where(mask, 0.2, 0.8) + 0.01 * rng.normal(size=(size, size)))
    return TopoImageEncoder().encode_video(np.asarray(seq))


def _ar1(n: int = 400, d: int = 20, seed: int = 7) -> np.ndarray:
    """Temporal structure, no entity structure. A negative the criterion must
    reject for the right reason rather than for lack of any signal."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, d))
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(scale=0.3, size=d)
    return x


def demo() -> None:
    """Measured selection accuracy, the compare() head-to-head, and runtimes."""
    import time

    from .bench import compare
    from .mujoco.corpus import CorpusConfig, make_corpus

    obs21, islands = make_corpus(CorpusConfig(frames=400))
    obs15, _ = make_corpus(CorpusConfig(per_cap=7, n_bridge=1, frames=300))
    obs22, _ = make_corpus(CorpusConfig(per_cap=10, n_bridge=2, radius=1.0, frames=300))

    # (name, trajectory, candidates offered, truth entity_dim)
    # Every non-zero candidate divides the observation width -- that is exactly
    # the trap select_cloud documents losing to.
    systems = [
        ("s2_rips 21 nodes", obs21, (0, 3, 7, 9), 3),
        ("s2_rips 15 nodes", obs15, (0, 3, 5, 9), 3),
        ("s2_rips 22 nodes", obs22, (0, 2, 3, 6), 3),
        ("lorenz seed 0", _lorenz(seed=0), (0, 3, 4, 6), 0),
        ("lorenz seed 1", _lorenz(seed=1), (0, 3, 4, 6), 0),
        ("lorenz seed 2", _lorenz(seed=2), (0, 3, 4, 6), 0),
        ("vision topo features", _vision_video(), (0, 2, 5, 10), 0),
        ("ar(1) sequence", _ar1(), (0, 2, 4, 5), 0),
        ("uniform noise", np.random.default_rng(3).uniform(-1, 1, (400, 12)), (0, 3, 4, 6), 0),
    ]

    print("=" * 78)
    print("SELECTION ACCURACY -- systems whose right answer is known")
    print("=" * 78)
    print(f"  {'system':<22}{'got':>5}{'want':>6}{'delta':>8}{'conf':>7}   verdict")
    print("  " + "-" * 68)
    right, misses, recs = 0, [], {}
    for name, traj, cands, truth in systems:
        rec = SigmoidGovernor().recommend([traj], entity_dim_candidates=cands)
        recs[name] = rec
        got = rec.config.entity_dim
        best = max(rec.deltas.values(), default=0.0)
        ok = got == truth
        right += ok
        if not ok:
            misses.append(
                f"{name}: picked entity_dim={got}, want {truth}. deltas "
                + " ".join(f"{w}:{v:+.2f}" for w, v in sorted(rec.deltas.items()))
            )
        print(
            f"  {name:<22}{got:>5}{truth:>6}{best:>+8.2f}{rec.confidence:>7.2f}"
            f"   {'ok' if ok else 'MISS'}"
        )
    print(f"\n  {right}/{len(systems)} correct")
    for m in misses:
        print(f"  MISS  {m}")
    print(
        "  the miss is mechanistic, not noise: 3n is divisible by 2 only for even n,\n"
        "  and the interleaved width-2 reshape keeps the cluster structure with 1.5x\n"
        "  the points, so the delta -- which prefers point count among structured\n"
        "  widths -- takes it. Odd node counts (11, 15, 21) are correct."
    )
    assert right >= 8, f"selection accuracy regressed to {right}/{len(systems)}"

    # ---- plateau diagnostics on a system with a known interaction radius
    print()
    print("=" * 78)
    print("PLATEAU DIAGNOSTICS -- s2_rips 21 nodes, generator radius 0.95 rad")
    print("=" * 78)
    rec = recs["s2_rips 21 nodes"]
    for w in sorted(rec.plateaus):
        rep = rec.plateaus[w]
        ctrl = rec.controls.get(w)
        tail = f"  control {ctrl:5.2f}  delta {rep.excess - ctrl:+.2f}" if ctrl else ""
        mark = "  <- chosen" if w == rec.config.entity_dim else ""
        body = rep.table() if rep.n_frames else "no plateau"
        print(f"  entity_dim={w:<3d} {body}{tail}{mark}")
    print(f"  corpus island counts: {sorted(np.unique(islands).tolist())}")

    # Where does the generator's own radius actually sit among the bands? The
    # claim "the band centre is the interaction radius" is checkable, so check it.
    chord = 2 * np.sin(0.95 / 2)  # 0.95 rad geodesic on the unit sphere
    ranks, widths = [], []
    for frame in obs21[::16]:
        bc = h0_barcode(frame.reshape(-1, 3))
        h = np.unique((bc.bars[:, 1] * bc.diameter)[np.isfinite(bc.bars[:, 1])])
        h = h[h > 0]
        if h.size < 2 or not (h[0] <= chord < h[-1]):
            continue
        gaps = np.diff(np.log10(h))
        i = int(np.searchsorted(h, chord) - 1)
        ranks.append(int((gaps > gaps[i]).sum()) + 1)
        widths.append(float(gaps[i]))
    print(
        f"  the band centre is {rec.plateaus[3].centre:.4f}; the generator's 0.95 rad is\n"
        f"  {chord:.4f} in chord distance and lands in a band of median width "
        f"{np.median(widths):.3f} dec,\n"
        f"  ranked {np.median(ranks):.0f}th of {rec.plateaus[3].components:.0f}+ by width. So "
        f"the plateau finds the scale at\n"
        f"  which the *configuration* separates -- the cap radius -- not the threshold a\n"
        f"  labeller chose. 'The band centre is the interaction radius' holds for the\n"
        f"  geometry's own radius only, and that is the one psi can read without labels."
    )

    # ---- does the recommendation beat the default?
    print()
    print("=" * 78)
    print("RECOMMENDATION vs SigmoidConfig() -- bench.compare(), sigmoid arm")
    print("=" * 78)
    horizons = (1, 4, 16)
    head = (
        f"  {'config':<14}{'entity':>7}{'abs':>5}{'lin':>5}{'dim':>5}  "
        + "  ".join(f"k={h:<8}" for h in horizons)
    )
    for label, traj in (("s2_rips 21", obs21), ("lorenz seed 0", _lorenz(seed=0))):
        rec = recs["s2_rips 21 nodes" if label.startswith("s2") else "lorenz seed 0"]
        print(f"  {label}:")
        print(head)
        print("  " + "-" * (len(head) - 2))
        rows = []
        for arm_label, cfg in (("recommended", rec.config), ("default", SigmoidConfig())):
            t0 = time.perf_counter()
            report = compare([traj], config=cfg, horizons=horizons)
            secs = time.perf_counter() - t0
            arm = next(a for a in report.arms if a.name == "sigmoid")
            rows.append((arm_label, cfg, arm, secs))
            cells = "  ".join(f"{arm.nrmse[h]:<10.4f}" for h in horizons)
            print(
                f"  {arm_label:<14}{cfg.entity_dim:>7}{cfg.n_abs_radii:>5}"
                f"{cfg.linear_dim:>5}{arm.state_dim:>5}  {cells}({secs:.1f}s)"
            )
        rel = [rows[0][2].nrmse[h] - rows[1][2].nrmse[h] for h in horizons]
        wins = sum(x < -1e-6 for x in rel)
        ties = sum(abs(x) <= 1e-6 for x in rel)
        print(
            f"  recommendation vs default: {wins} win / {ties} tie / "
            f"{len(horizons) - wins - ties} loss   "
            + " ".join(f"k={h}:{x:+.4f}" for h, x in zip(horizons, rel))
        )
        if label.startswith("s2"):
            print(
                "  reported as measured: the recommendation LOSES here, badly. entity_dim=3\n"
                "  makes psi read the contact partition, which README s9 records as\n"
                "  contributing exactly zero to *coordinate* rollout -- and coordinate\n"
                "  rollout is what compare() reports. Both arms are above nrmse 1.0, i.e.\n"
                "  worse than predicting zero, so this corpus does not measure prediction\n"
                "  at all; it is a representation benchmark (0.855 vs 0.469 on the island\n"
                "  count). The governor optimises representation and compare() does not\n"
                "  score it. Read this as compare() being the wrong instrument here, and\n"
                "  as a reason to distrust any single-number verdict on this corpus."
            )
        elif ties == len(horizons):
            print(
                "  identical to four decimals: the only change was linear_dim 32 -> 24,\n"
                "  and Lorenz is a 3-dim system in a 24-dim lift, so PCA components past\n"
                "  the first few carry noise. The recommendation costs 8 state dimensions\n"
                "  less for the same accuracy -- it is not worse, it is cheaper."
            )

    # ---- the metric this corpus actually measures
    #
    # compare() scores coordinate rollout, which README s9 records the S2
    # partition as contributing exactly zero to. The corpus's own held-out
    # diagnostic is the island count, and reading it is what 0.386 -> 0.855
    # referred to. Reporting only the metric that makes the pick look bad would
    # be as one-sided as reporting only the one that flatters it.
    print()
    print("=" * 78)
    print("REPRESENTATION READOUT -- psi -> held-out island count, s2_rips 21 nodes")
    print("=" * 78)
    from .engine import SigmoidWorldModel

    rec = recs["s2_rips 21 nodes"]
    cut = int(len(obs21) * 0.7)
    print(f"  {'config':<14}{'entity':>7}{'abs':>5}   held-out R^2 on the island count")
    for arm_label, cfg in (("recommended", rec.config), ("default", SigmoidConfig())):
        wm = SigmoidWorldModel(config=cfg).fit([obs21[:cut]])
        psi = wm.encoder.encode_trajectory(obs21)[:, : wm.encoder.topo_dim]
        truth = islands[cfg.window - 1 :].astype(float)
        split = int(len(psi) * 0.7)
        x, y = psi[:split], truth[:split]
        xm, ym = x.mean(0), y.mean()
        a = x - xm
        w = np.linalg.solve(a.T @ a + 1e-3 * np.eye(a.shape[1]), a.T @ (y - ym))
        pred = (psi[split:] - xm) @ w + ym
        held = truth[split:]
        r2 = 1.0 - ((held - pred) ** 2).sum() / max(((held - held.mean()) ** 2).sum(), 1e-12)
        print(f"  {arm_label:<14}{cfg.entity_dim:>7}{cfg.n_abs_radii:>5}   {r2:>8.4f}")

    # ---- vitals runtime
    print()
    print("=" * 78)
    print("VITALS RUNTIME")
    print("=" * 78)
    ex = VitalsExtractor()
    for label, traj in (
        ("s2_rips 400x63", obs21),
        ("lorenz 1200x24", _lorenz()),
        ("distilgpt2-shaped 512x768", np.random.default_rng(0).normal(size=(512, 768))),
        ("long stream 20000x64", np.random.default_rng(0).normal(size=(20000, 64))),
    ):
        ex(traj)  # warm
        t0 = time.perf_counter()
        for _ in range(5):
            ex(traj)
        ms = (time.perf_counter() - t0) / 5 * 1e3
        t0 = time.perf_counter()
        plateau_diagnostics(traj)
        p_ms = (time.perf_counter() - t0) * 1e3
        print(f"  {label:<28}vitals {ms:>8.2f} ms   plateau {p_ms:>8.2f} ms")

    # the source's halved intrinsic dimension, both ways
    rng = np.random.default_rng(0)
    lift = rng.normal(size=(5, 20))
    blob = rng.normal(size=(400, 5)) @ lift
    fixed = ex(blob).intrinsic_dim
    dist = _pairwise(blob[:512])
    near = np.sort(dist**2, axis=1)[:, 1:6]
    source = 1.0 / np.log(near[:, -1:] / near[:, :-1]).mean()
    print(
        f"  intrinsic dim of a 5-dim gaussian in R^20: {fixed:.2f} on true distances, "
        f"{source:.2f} on the source's squared ones"
    )
    assert abs(fixed - 2 * source) < 0.15, "the factor-of-two claim does not hold"

    # ---- determinism, degenerate input, and the history path
    print()
    print("=" * 78)
    print("INVARIANTS")
    print("=" * 78)
    a, b = VitalsExtractor(seed=1)(obs21), VitalsExtractor(seed=1)(obs21)
    assert np.array_equal(a.vector(), b.vector()), "vitals not reproducible"
    for name, bad in (
        ("n=1", np.zeros((1, 4))),
        ("d=1", np.arange(50.0).reshape(-1, 1)),
        ("identical rows", np.ones((30, 3))),
        ("with NaNs", np.where(np.arange(120).reshape(30, 4) % 7 == 0, np.nan, 1.0)),
        ("n < k", np.arange(9.0).reshape(3, 3)),
    ):
        v = ex(bad)
        assert np.all(np.isfinite(v.vector())), f"{name} produced a non-finite vital"
    print("  vitals finite on 5 degenerate inputs, reproducible for a fixed seed")

    gov = SigmoidGovernor()
    for traj, cands in ((obs21, (0, 3)), (_lorenz(), (0, 3)), (obs15, (0, 3))):
        r = gov.recommend([traj], entity_dim_candidates=cands)
        gov.update(r.config, 0.9)
    assert len(gov.history) == 3
    again = gov.recommend([obs21], entity_dim_candidates=(0, 3))
    print(f"  history {len(gov.history)} entries, recall on a repeat: {again.recalled}")
    print()
    print("demo ok")


if __name__ == "__main__":
    demo()

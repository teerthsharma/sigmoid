"""Sigma: the topological state encoder.

Turns a window of hidden activations into a fixed-length *world state* vector.

The state is deliberately hybrid:

    z = [ psi ; u ]

    psi : topological channel. Hilbert-series coefficients of the Vietoris-Rips
          barcode of the window point cloud, plus Betti-curve samples and
          scale-invariant geometry. Carries *regime* information -- how many
          components, how many loops, how spread out, how confined.
    u   : linear channel. PCA coordinates of the window-mean hidden state.
          Carries *content* -- the part that can be decoded back to activations.

Why hybrid: psi alone is not invertible (topology forgets coordinates), so a
psi-only world model cannot emit predicted activations. u alone is a plain
linear autoencoder, which is the null model we must beat. Splitting the two
makes the topology channel *falsifiable*: ablate psi, keep total dimension
fixed, and see whether rollout error rises. See bench.py.

H0 is computed exactly via the minimum spanning tree (single-linkage
persistence), which is O(n^2 d) and fast enough for the runtime path.
H1 needs Rips and is calibration-only -- never call it in a hot loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Barcode",
    "TopoEncoderConfig",
    "TopoEncoder",
    "h0_barcode",
    "h1_barcode",
    "hilbert_coefficients",
    "betti_curve",
]


# --------------------------------------------------------------------------
# barcodes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Barcode:
    """Persistence barcode in one homology dimension.

    ``bars`` are (birth, death) pairs with death possibly ``inf``.
    Filtration values are normalized to [0, 1] by the point-cloud diameter,
    so barcodes from different windows are directly comparable.
    """

    dim: int
    bars: np.ndarray  # (k, 2) float64
    diameter: float

    @property
    def lifetimes(self) -> np.ndarray:
        finite = np.isfinite(self.bars[:, 1])
        if not finite.any():
            return np.zeros(0)
        return self.bars[finite, 1] - self.bars[finite, 0]


def _pairwise(points: np.ndarray) -> np.ndarray:
    sq = np.sum(points * points, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def h0_barcode(points: np.ndarray) -> Barcode:
    """Exact H0 persistence via the Euclidean minimum spanning tree.

    Single-linkage merge heights *are* the H0 death times of a Vietoris-Rips
    filtration -- this is exact, not an approximation, and it avoids building
    the simplicial complex at all.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return Barcode(dim=0, bars=np.array([[0.0, np.inf]]), diameter=0.0)

    dist = _pairwise(pts)
    diameter = float(dist.max())
    if diameter <= 1e-12:
        return Barcode(dim=0, bars=np.array([[0.0, np.inf]]), diameter=0.0)
    dist = dist / diameter

    from scipy.sparse.csgraph import minimum_spanning_tree

    mst = minimum_spanning_tree(dist).toarray()
    heights = np.sort(mst[mst > 0.0])
    bars = np.empty((heights.shape[0] + 1, 2), dtype=np.float64)
    bars[:-1, 0] = 0.0
    bars[:-1, 1] = heights
    bars[-1] = (0.0, np.inf)  # the essential class
    return Barcode(dim=0, bars=bars, diameter=diameter)


def h1_barcode(points: np.ndarray, max_points: int = 256) -> Barcode:
    """H1 persistence via ripser. Calibration path only -- this is not cheap.

    Returns an empty barcode when ripser is unavailable so the encoder degrades
    to H0-only rather than failing.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return Barcode(dim=1, bars=np.zeros((0, 2)), diameter=0.0)

    if pts.shape[0] > max_points:  # deterministic stride subsample
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[idx]

    dist = _pairwise(pts)
    diameter = float(dist.max())
    if diameter <= 1e-12:
        return Barcode(dim=1, bars=np.zeros((0, 2)), diameter=0.0)

    try:
        from ripser import ripser
    except ImportError:
        return Barcode(dim=1, bars=np.zeros((0, 2)), diameter=0.0)

    dgms = ripser(dist / diameter, maxdim=1, distance_matrix=True)["dgms"]
    bars = np.asarray(dgms[1], dtype=np.float64) if len(dgms) > 1 else np.zeros((0, 2))
    if bars.size == 0:
        bars = np.zeros((0, 2))
    return Barcode(dim=1, bars=bars.reshape(-1, 2), diameter=diameter)


def hilbert_coefficients(barcode: Barcode, degree: int) -> np.ndarray:
    """Hilbert-series numerator coefficients of a barcode.

        N(s) = sum_i s^{b_i} - sum_i s^{d_i}

    Coefficient k counts births in bin k minus deaths in bin k. The whole
    barcode collapses to a fixed-length vector this way, which is what makes a
    linear operator on topology possible at all.
    """
    coeffs = np.zeros(degree, dtype=np.float64)
    for birth, death in barcode.bars:
        ib = int(np.floor(birth * degree))
        if 0 <= ib < degree:
            coeffs[ib] += 1.0
        if np.isfinite(death):
            idd = int(np.floor(death * degree))
            if 0 <= idd < degree:
                coeffs[idd] -= 1.0
    return coeffs


def betti_curve(barcode: Barcode, radii: np.ndarray) -> np.ndarray:
    """Betti number of the barcode at each radius, using [birth, death)."""
    births = barcode.bars[:, 0]
    deaths = barcode.bars[:, 1]
    alive = (births[None, :] <= radii[:, None]) & (radii[:, None] < deaths[None, :])
    return alive.sum(axis=1).astype(np.float64)


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------


@dataclass
class TopoEncoderConfig:
    window: int = 32
    """Number of consecutive hidden states forming the state point cloud."""

    hilbert_degree: int = 24
    """Length of the Hilbert-series embedding per homology dimension."""

    n_radii: int = 8
    """Betti-curve sample count per homology dimension, at *normalized* radii."""

    n_abs_radii: int = 8
    """Betti-curve samples at *absolute* radii. 0 disables.

    The normalized barcode divides out the cloud diameter, which buys scale
    invariance and throws away scale. That is the wrong trade whenever the
    quantity of interest lives at a fixed physical distance -- contact
    thresholds, constraint radii, interaction cutoffs.

    Measured on the S2-Rips corpus of mujoco#3396, whose ground-truth island
    partition is H0 of a Rips graph at a fixed 0.95 rad: the normalized channel
    scored 0.386 against a 0.487 majority baseline, while beta_0 read at the
    corresponding absolute radius reproduced the island count with correlation
    1.0000 and exact agreement on every frame. Same barcode, same code -- the
    only difference was whether the diameter had been divided out.

    Radii are learned from the calibration merge heights, so no physical
    constant has to be supplied by hand -- see `abs_radii` to supply one."""

    abs_radii: tuple[float, ...] | None = None
    """Explicit absolute radii, overriding the learned quantiles.

    Learning radii from data is right when exploring: you do not know which
    scale matters, so you sample where merges actually happen. It is wrong when
    the scale is a known physical constant, because the quantiles will sit
    wherever the data happens to live and may miss it entirely.

    Measured on the robot-control demo: with a contact radius of 0.9, the
    learned quantiles came out at 2.21 through 11.94 -- every one of them too
    coarse, so a beta_0 constraint read "how many groups within 6 units", which
    is unrelated to whether anything collided. Passing the contact radius
    directly fixes it.

    Rule of thumb: exploration learns the radius, control is told it."""

    linear_dim: int = 32
    """PCA rank of the decodable linear channel."""

    use_h1: bool = False
    """Include H1 (loops). Calibration-quality but far slower; off by default."""

    entity_dim: int = 0
    """If set, treat each observation as a cloud of entities of this dimension.

    Two different topological objects hide behind the phrase "the topology of
    the state", and picking the wrong one silently measures nothing:

      temporal (entity_dim = 0, default)
          the point cloud is the window of consecutive state vectors, so the
          barcode describes the shape of a trajectory segment. Right for token
          streams, where the entities *are* the successive positions.

      spatial (entity_dim = d)
          each observation reshapes to (D/d, d) and the barcode describes the
          arrangement of entities at one instant. Right for particles, bodies,
          contacts, agents -- anything whose state is a set.

    Measured on the S2-Rips corpus, where the ground-truth island count is by
    construction the H0 of the entity cloud: the temporal encoder scored 0.386
    against a 0.487 majority baseline, i.e. it was actively uninformative,
    because the shape of a trajectory segment has nothing to do with how many
    islands the nodes currently form."""

    use_topology: bool = True
    """Ablation switch. False drops psi entirely, leaving a plain linear model.
    Set it, raise `linear_dim` to match, and the topology claim becomes a
    controlled experiment instead of an assertion. See bench.py."""

    whiten: bool = True


@dataclass
class TopoEncoder:
    """Sigma: hidden-state window -> world state vector z = [psi ; u]."""

    config: TopoEncoderConfig = field(default_factory=TopoEncoderConfig)

    h_center_: np.ndarray | None = field(default=None, repr=False)
    h_scale_: np.ndarray | None = field(default=None, repr=False)
    mean_: np.ndarray | None = field(default=None, repr=False)
    components_: np.ndarray | None = field(default=None, repr=False)
    scale_: np.ndarray | None = field(default=None, repr=False)
    psi_mean_: np.ndarray | None = field(default=None, repr=False)
    psi_scale_: np.ndarray | None = field(default=None, repr=False)
    residual_decay_: np.ndarray | None = field(default=None, repr=False)
    abs_radii_: np.ndarray | None = field(default=None, repr=False)
    fitted: bool = False

    # ---- dimensions -------------------------------------------------------

    @property
    def n_dims(self) -> int:
        return 2 if self.config.use_h1 else 1

    @property
    def topo_dim(self) -> int:
        c = self.config
        if not c.use_topology:
            return 0
        # per homology dim: hilbert coefficients + betti curve; plus 4 geometry
        return self.n_dims * (c.hilbert_degree + c.n_radii) + self.n_abs_radii + 4

    @property
    def n_abs_radii(self) -> int:
        c = self.config
        return len(c.abs_radii) if c.abs_radii else c.n_abs_radii

    @property
    def state_dim(self) -> int:
        return self.topo_dim + self.config.linear_dim

    # ---- raw topological signature ---------------------------------------

    def _topo_window(self, raw: np.ndarray, standardized: np.ndarray) -> np.ndarray:
        """Which version of the window the barcode should see.

        Robust per-coordinate standardization exists to tame transformer
        activation outliers, where coordinates are independent channels and a
        72x scale spread would otherwise dominate every distance. Entity states
        are the opposite case: the coordinates of one entity share a metric,
        and rescaling each of them separately warps the very geometry the
        barcode is meant to measure -- reshaping 63 independently-rescaled
        numbers into 21 points does not give back points on a sphere.

        So spatial clouds are read raw, temporal clouds standardized.
        """
        return raw if self.config.entity_dim else standardized

    def _cloud(self, window: np.ndarray) -> np.ndarray:
        """The point cloud the barcode is computed over.

        Temporal by default; spatial when `entity_dim` is set, in which case
        the most recent observation is unpacked into its entities.
        """
        d = self.config.entity_dim
        if not d:
            return window
        last = window[-1]
        if last.shape[0] % d:
            raise ValueError(
                f"observation width {last.shape[0]} is not a multiple of "
                f"entity_dim={d}"
            )
        return last.reshape(-1, d)

    def _raw_psi(self, window: np.ndarray) -> np.ndarray:
        c = self.config
        radii = np.linspace(0.0, 1.0, c.n_radii, endpoint=False)

        window = self._cloud(window)
        parts: list[np.ndarray] = []
        b0 = h0_barcode(window)
        parts.append(hilbert_coefficients(b0, c.hilbert_degree))
        parts.append(betti_curve(b0, radii))
        if c.use_h1:
            b1 = h1_barcode(window)
            parts.append(hilbert_coefficients(b1, c.hilbert_degree))
            parts.append(betti_curve(b1, radii))

        # Betti curve at absolute radii: undo the diameter normalization and
        # count components still separate at each physical distance. This is
        # exactly the island partition of a Rips graph at that radius.
        if self.n_abs_radii and self.abs_radii_ is not None:
            deaths = b0.bars[:, 1] * b0.diameter
            parts.append((deaths[None, :] > self.abs_radii_[:, None]).sum(axis=1).astype(float))

        # scale-aware geometry: the diameter that normalization divided out,
        # plus how the mass is distributed inside the window.
        lifetimes = b0.lifetimes
        centroid = window.mean(axis=0)
        radial = np.linalg.norm(window - centroid, axis=1)
        geometry = np.array(
            [
                np.log1p(b0.diameter),
                float(lifetimes.mean()) if lifetimes.size else 0.0,
                float(lifetimes.max()) if lifetimes.size else 0.0,
                float(radial.std() / (radial.mean() + 1e-9)),
            ],
            dtype=np.float64,
        )
        parts.append(geometry)
        return np.concatenate(parts)

    # ---- fit / encode -----------------------------------------------------

    def fit(self, trajectory: np.ndarray) -> "TopoEncoder":
        """Calibrate on a hidden-state trajectory of shape (T, D)."""
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2:
            raise ValueError("trajectory must be 2D (T, D)")
        c = self.config
        if traj.shape[0] < c.window + 1:
            raise ValueError(
                f"need at least window+1={c.window + 1} steps, got {traj.shape[0]}"
            )

        # Robust per-dimension standardization, before anything else.
        #
        # Transformer residual streams carry "massive activations": a handful of
        # channels whose scale exceeds the median channel by one to two orders
        # of magnitude. Measured on distilgpt2, max per-dim std was 22.1 against
        # a median of 0.308 -- a 72x ratio concentrated in about five channels.
        #
        # Left alone this ruins both channels at once. PCA spends its rank
        # describing the outliers, and -- worse -- Euclidean distances in the
        # window point cloud become a function of those same few channels, so
        # the persistence barcode ends up measuring activation magnitude rather
        # than the geometry of the representation. A topological feature that
        # only tracks norm is a topological feature that is not doing anything.
        #
        # Median and MAD rather than mean and std, because the outliers would
        # otherwise set the very scale meant to tame them.
        self.h_center_ = np.median(traj, axis=0)
        mad = np.median(np.abs(traj - self.h_center_), axis=0)
        scale = 1.4826 * mad  # MAD -> std for a Gaussian
        fallback = traj.std(axis=0)
        scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
        self.h_scale_ = scale
        raw_traj = traj
        traj = (traj - self.h_center_) / self.h_scale_
        topo_traj = raw_traj if c.entity_dim else traj

        # linear channel: PCA on standardized hidden states
        self.mean_ = traj.mean(axis=0)
        centered = traj - self.mean_
        rank = min(c.linear_dim, centered.shape[0], centered.shape[1])
        _, sv, vt = np.linalg.svd(centered, full_matrices=False)
        comps = np.zeros((c.linear_dim, traj.shape[1]), dtype=np.float64)
        comps[:rank] = vt[:rank]
        self.components_ = comps
        scale = np.ones(c.linear_dim, dtype=np.float64)
        if c.whiten:
            energy = sv[:rank] / np.sqrt(max(centered.shape[0] - 1, 1))
            scale[:rank] = np.where(energy > 1e-9, energy, 1.0)
        self.scale_ = scale

        # absolute radii, learned from where components actually merge during
        # calibration, so the physical scale never has to be supplied by hand
        if c.use_topology and c.abs_radii:
            self.abs_radii_ = np.asarray(c.abs_radii, dtype=np.float64)
        elif c.use_topology and c.n_abs_radii:
            heights: list[float] = []
            all_windows = list(_windows(topo_traj, c.window))
            stride = max(1, len(all_windows) // 64)
            for w in all_windows[::stride]:
                bc = h0_barcode(self._cloud(w))
                finite = bc.bars[np.isfinite(bc.bars[:, 1]), 1]
                heights.extend((finite * bc.diameter).tolist())
            self.abs_radii_ = (
                np.quantile(heights, np.linspace(0.05, 0.95, c.n_abs_radii))
                if heights
                else np.zeros(c.n_abs_radii)
            )

        # topological channel: standardize over the calibration windows
        if c.use_topology:
            raws = np.stack(
                [self._raw_psi(w) for w in _windows(topo_traj, c.window)], axis=0
            )
            self.psi_mean_ = raws.mean(axis=0)
            std = raws.std(axis=0)
            self.psi_scale_ = np.where(std > 1e-9, std, 1.0)
        else:
            self.psi_mean_ = np.zeros(0)
            self.psi_scale_ = np.ones(0)

        self.fitted = True

        # How fast does the part PCA cannot represent forget itself?
        #
        # Predictions decode as Dec(u_k) + decay^k * r_anchor, where r_anchor is
        # the unrepresented residual of the last observed activation. Two
        # extremes are both wrong: decay = 1 pins every prediction to the anchor
        # and inherits carry-forward's long-horizon error, decay = 0 throws away
        # a genuinely informative residual at short horizons. The residual is
        # well modelled as AR(1) per dimension, so its lag-1 autocorrelation is
        # the honest interpolation between the two, measured rather than chosen.
        recon = (
            (traj - self.mean_) @ self.components_.T
        ) @ self.components_ + self.mean_
        resid = traj - recon
        self.residual_decay_ = _lag1_autocorrelation(resid)
        return self

    def encode(self, window: np.ndarray) -> np.ndarray:
        """Encode one (window, D) block of hidden states into a state vector."""
        self._check_fitted()
        raw = np.asarray(window, dtype=np.float64)
        w = self.normalize(raw)
        u = (w[-1] - self.mean_) @ self.components_.T / self.scale_
        if not self.config.use_topology:
            return u
        psi = (self._raw_psi(self._topo_window(raw, w)) - self.psi_mean_) / self.psi_scale_
        return np.concatenate([psi, u])

    def encode_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Encode every sliding window of a (T, D) trajectory -> (T-W+1, state_dim)."""
        self._check_fitted()
        traj = np.asarray(trajectory, dtype=np.float64)
        return np.stack(
            [self.encode(w) for w in _windows(traj, self.config.window)], axis=0
        )

    # ---- linear channel round trip ---------------------------------------

    def normalize(self, hidden: np.ndarray) -> np.ndarray:
        """Raw hidden state(s) -> robustly standardized space."""
        self._check_fitted()
        return (np.asarray(hidden, dtype=np.float64) - self.h_center_) / self.h_scale_

    def denormalize(self, hidden: np.ndarray) -> np.ndarray:
        """Standardized space -> raw hidden state(s)."""
        self._check_fitted()
        return np.asarray(hidden, dtype=np.float64) * self.h_scale_ + self.h_center_

    def project(self, hidden: np.ndarray) -> np.ndarray:
        """Hidden state -> linear coordinates u."""
        return (self.normalize(hidden) - self.mean_) @ self.components_.T / self.scale_

    def reconstruct(self, u: np.ndarray) -> np.ndarray:
        """Linear coordinates u -> predicted hidden state. This is the decoder."""
        self._check_fitted()
        std = (np.asarray(u, dtype=np.float64) * self.scale_) @ self.components_ + self.mean_
        return self.denormalize(std)

    def split(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a state vector into (psi, u)."""
        z = np.asarray(z, dtype=np.float64)
        return z[..., : self.topo_dim], z[..., self.topo_dim :]

    def decode(self, z: np.ndarray) -> np.ndarray:
        """State vector -> predicted hidden state (uses the linear channel)."""
        _, u = self.split(z)
        return self.reconstruct(u)

    def residual_of(self, hidden: np.ndarray) -> np.ndarray:
        """The part of an activation the linear channel cannot represent."""
        self._check_fitted()
        n = self.normalize(hidden)
        recon = ((n - self.mean_) @ self.components_.T) @ self.components_ + self.mean_
        return n - recon

    def decode_with_residual(
        self, z: np.ndarray, residual: np.ndarray, step: int
    ) -> np.ndarray:
        """Decode, carrying the anchor residual forward with measured decay."""
        self._check_fitted()
        _, u = self.split(z)
        std = (np.asarray(u, dtype=np.float64) * self.scale_) @ self.components_ + self.mean_
        return self.denormalize(std + (self.residual_decay_**step) * residual)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("TopoEncoder.fit must be called before use")

    # ---- persistence ------------------------------------------------------

    def state_dict(self) -> dict:
        self._check_fitted()
        return {
            "config": vars(self.config),
            "h_center_": self.h_center_,
            "h_scale_": self.h_scale_,
            "mean_": self.mean_,
            "components_": self.components_,
            "scale_": self.scale_,
            "psi_mean_": self.psi_mean_,
            "psi_scale_": self.psi_scale_,
            "residual_decay_": self.residual_decay_,
            "abs_radii_": self.abs_radii_,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "TopoEncoder":
        enc = cls(config=TopoEncoderConfig(**payload["config"]))
        enc.h_center_ = payload["h_center_"]
        enc.h_scale_ = payload["h_scale_"]
        enc.mean_ = payload["mean_"]
        enc.components_ = payload["components_"]
        enc.scale_ = payload["scale_"]
        enc.psi_mean_ = payload["psi_mean_"]
        enc.psi_scale_ = payload["psi_scale_"]
        enc.residual_decay_ = payload["residual_decay_"]
        enc.abs_radii_ = payload["abs_radii_"]
        enc.fitted = True
        return enc


def _lag1_autocorrelation(series: np.ndarray) -> np.ndarray:
    """Per-dimension lag-1 autocorrelation, clipped to [0, 1]."""
    a, b = series[:-1], series[1:]
    a_c, b_c = a - a.mean(axis=0), b - b.mean(axis=0)
    denom = np.sqrt((a_c**2).sum(axis=0) * (b_c**2).sum(axis=0))
    corr = np.where(denom > 1e-12, (a_c * b_c).sum(axis=0) / np.maximum(denom, 1e-12), 0.0)
    return np.clip(corr, 0.0, 1.0)


def _windows(traj: np.ndarray, window: int):
    for end in range(window, traj.shape[0] + 1):
        yield traj[end - window : end]

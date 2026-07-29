"""The grounding gate: cohomological self-consistency of an imagined state.

An imagined state is a *prediction*, so we cannot check it against truth -- the
truth is exactly what we declined to compute. The gate therefore checks
something available without the real model: whether the imagined state is
internally consistent with itself.

Sheaf formulation. Over each time index put two stalks:

    F(topo)   = psi, the topological channel
    F(linear) = u,   the linear channel

Both are local views of the same underlying activation window, so a restriction
map R : F(linear) -> F(topo) exists on real data and is learned by ridge on the
calibration trajectory. On a genuine state the section glues:

    || R u - psi ||  ~  0

The operator T advances psi and u *jointly but not identically*: nothing in the
least-squares fit forces the imagined psi to remain the topological signature of
the imagined u. So the residual

    r = || R u_hat - psi_hat ||

is a first cohomology obstruction to gluing the two local sections, and it grows
precisely when the rollout has drifted off the data manifold. This is the
"sheaf consistency filter" pattern used as a safety gate over another selector,
which is where it is known to earn its keep -- it is a correctness diagnostic
turned into an execution rule.

The second term is plain support checking: a whitened Mahalanobis distance to
the calibration state cloud. Off-support states are unreliable even when
self-consistent.

Third stalk: spectral entropy (AGENDA R8). Calibrated on markdown prose from
distilgpt2 activations, the two terms above scored held-out prose 0.671 (0/9
fire), symbol soup 1.385 (4/9), one token repeated 14.596 (5/5) -- and uniform
random tokens 0.513 (0/5), *lower than real prose*. Random input produces
diffuse, high-entropy activations whose window averages sit near the centroid of
the standardized state space, so both the residual and the Mahalanobis distance
stay small. The two terms see structural degeneracy, not noise.

So add a third local section on the same window: the effective rank
exp(-sum p log p) over the window's normalized squared singular values. Noise has
anomalously HIGH effective rank, repetition anomalously LOW, and prose sits in
between -- one statistic, both tails, if the deviation is taken two-sided.
Scored as |log er - median(log er_calib)| over the calibration quantile of that
same deviation, so it lands in the same "how weird would this have been in
calibration" units as the other two.

The window is not recoverable from an encoded state z, so the caller supplies
the statistic (`effective_rank(encoder.normalize(window))`). Supply nothing and
the term is inert -- the gate is exactly the two-term gate it was before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GateReading", "SheafGate", "effective_rank"]


def effective_rank(window: np.ndarray) -> float:
    """exp(spectral entropy) of a (window, D) block: how many directions it uses.

    Ranges from 1.0 (every row the same, i.e. a repeated token) to min(window, D)
    (isotropic noise), with structured signal in between.

    Deliberately NOT column-centered. Centering looks like the right thing --
    remove the offset, keep the shape -- but it deletes the low tail: a repeated
    pattern plus any jitter centers to *pure jitter*, so its rank becomes the
    rank of the noise floor. Measured on synthetic windows (24x16, latent dim 4),
    mean effective rank centered vs uncentered:

                              centered   uncentered
        smooth signal             2.59         2.44
        repeat + 1e-3 jitter     11.24         1.00
        isotropic noise          11.17        11.36

    Centered, repetition and noise are the same number and the gate is blind to
    the tail it already caught. The input is standardized activations, where the
    offset is itself a calibrated quantity, so leaving it in is sound.
    """
    W = np.asarray(window, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] < 2:
        raise ValueError("window must be (window, D) with window >= 2")
    s = np.linalg.svd(W, compute_uv=False)
    p = s**2
    total = p.sum()
    if total <= 1e-30:  # a literally constant window: one direction, degenerately
        return 1.0
    p = p / total
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


@dataclass(frozen=True)
class GateReading:
    """Why the gate did or did not fire."""

    sheaf_residual: float
    manifold_distance: float
    sheaf_score: float
    """Residual in calibration quantile units; 1.0 == the calibration threshold."""

    manifold_score: float
    score: float
    """max of the scores. >= 1.0 means ground now."""

    fire: bool
    reason: str

    # third stalk; zero when the caller supplied no effective rank, which keeps
    # the reading identical to the two-term gate.
    effective_rank: float = 0.0
    rank_score: float = 0.0


@dataclass
class SheafGate:
    """Learned restriction map plus support model, thresholded on quantiles."""

    quantile: float = 0.98
    """Calibration quantile that defines "normal". Higher = fires less often."""

    ridge: float = 1e-3

    R_: np.ndarray | None = field(default=None, repr=False)
    mean_: np.ndarray | None = field(default=None, repr=False)
    inv_std_: np.ndarray | None = field(default=None, repr=False)
    sheaf_threshold_: float = 0.0
    manifold_threshold_: float = 0.0
    rank_center_: float | None = None
    """median log effective rank in calibration. None == the stalk is not fitted."""

    rank_threshold_: float = 0.0
    topo_dim: int = 0
    fitted: bool = False

    def fit(
        self,
        states: np.ndarray,
        topo_dim: int,
        effective_ranks: np.ndarray | None = None,
    ) -> SheafGate:
        """Learn R and the support model from calibration states (N, state_dim).

        `effective_ranks` is the optional third stalk: one `effective_rank(W)`
        per state, in the same order. Omit it and the gate is the two-term gate.
        """
        Z = np.asarray(states, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[0] < 2:
            raise ValueError("states must be (N, state_dim) with N >= 2")
        self.topo_dim = int(topo_dim)
        psi, u = Z[:, : self.topo_dim], Z[:, self.topo_dim :]

        # restriction map R : u -> psi, with intercept
        U = np.concatenate([u, np.ones((u.shape[0], 1))], axis=1)
        gram = U.T @ U + self.ridge * np.eye(U.shape[1])
        self.R_ = np.linalg.solve(gram, U.T @ psi).T

        residuals = np.linalg.norm(psi - U @ self.R_.T, axis=1)
        self.sheaf_threshold_ = float(np.quantile(residuals, self.quantile))
        if self.sheaf_threshold_ <= 1e-12:
            self.sheaf_threshold_ = 1e-12

        # support model: whitened distance to the calibration cloud
        self.mean_ = Z.mean(axis=0)
        std = Z.std(axis=0)
        self.inv_std_ = 1.0 / np.where(std > 1e-9, std, 1.0)
        dists = np.linalg.norm((Z - self.mean_) * self.inv_std_, axis=1)
        self.manifold_threshold_ = float(np.quantile(dists, self.quantile))
        if self.manifold_threshold_ <= 1e-12:
            self.manifold_threshold_ = 1e-12

        # spectral stalk, two-sided: both tails are failures, so calibrate on the
        # absolute log deviation from the calibration centre. Log because
        # effective rank is a multiplicative scale (1 -> 4 is the same distance
        # as 4 -> 16), median because a handful of degenerate calibration
        # windows must not drag the centre.
        self.rank_center_, self.rank_threshold_ = None, 0.0
        if effective_ranks is not None:
            r = np.asarray(effective_ranks, dtype=np.float64).reshape(-1)
            if r.shape[0] != Z.shape[0]:
                raise ValueError(
                    f"effective_ranks has {r.shape[0]} entries for {Z.shape[0]} states"
                )
            r = np.log(np.maximum(r, 1e-12))
            self.rank_center_ = float(np.median(r))
            deviation = np.abs(r - self.rank_center_)
            self.rank_threshold_ = max(
                float(np.quantile(deviation, self.quantile)), 1e-12
            )

        self.fitted = True
        return self

    def read(self, z: np.ndarray, effective_rank: float | None = None) -> GateReading:
        """Evaluate one imagined state, optionally with its window's effective rank."""
        if not self.fitted:
            raise RuntimeError("SheafGate.fit must be called before use")
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        psi, u = z[: self.topo_dim], z[self.topo_dim :]

        predicted_psi = self.R_ @ np.concatenate([u, [1.0]])
        sheaf_residual = float(np.linalg.norm(psi - predicted_psi))
        manifold_distance = float(np.linalg.norm((z - self.mean_) * self.inv_std_))

        sheaf_score = sheaf_residual / self.sheaf_threshold_
        manifold_score = manifold_distance / self.manifold_threshold_

        # inert unless both sides supplied a rank: an unfitted stalk must not
        # fire, and a fitted one must not judge a state it was given no rank for.
        rank, rank_score, log_rank = 0.0, 0.0, 0.0
        if effective_rank is not None and self.rank_center_ is not None:
            rank = float(effective_rank)
            log_rank = float(np.log(max(rank, 1e-12)))
            rank_score = abs(log_rank - self.rank_center_) / self.rank_threshold_

        score = max(sheaf_score, manifold_score, rank_score)
        reason = "ok"
        if score >= 1.0:
            if sheaf_score >= manifold_score and sheaf_score >= rank_score:
                reason = "sheaf_inconsistent"
            elif manifold_score >= rank_score:
                reason = "off_manifold"
            else:
                reason = "rank_inflated" if log_rank > self.rank_center_ else "rank_collapsed"
        return GateReading(
            sheaf_residual=sheaf_residual,
            manifold_distance=manifold_distance,
            sheaf_score=sheaf_score,
            manifold_score=manifold_score,
            score=score,
            fire=bool(score >= 1.0),
            reason=reason,
            effective_rank=rank,
            rank_score=rank_score,
        )

    def state_dict(self) -> dict:
        return {
            "quantile": self.quantile,
            "ridge": self.ridge,
            "R_": self.R_,
            "mean_": self.mean_,
            "inv_std_": self.inv_std_,
            "sheaf_threshold_": self.sheaf_threshold_,
            "manifold_threshold_": self.manifold_threshold_,
            "rank_center_": self.rank_center_,
            "rank_threshold_": self.rank_threshold_,
            "topo_dim": self.topo_dim,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> SheafGate:
        gate = cls(quantile=float(payload["quantile"]), ridge=float(payload["ridge"]))
        gate.R_ = payload["R_"]
        gate.mean_ = payload["mean_"]
        gate.inv_std_ = payload["inv_std_"]
        gate.sheaf_threshold_ = float(payload["sheaf_threshold_"])
        gate.manifold_threshold_ = float(payload["manifold_threshold_"])
        center = payload.get("rank_center_")  # absent in gates saved before R8
        gate.rank_center_ = None if center is None else float(center)
        gate.rank_threshold_ = float(payload.get("rank_threshold_") or 0.0)
        gate.topo_dim = int(payload["topo_dim"])
        gate.fitted = True
        return gate

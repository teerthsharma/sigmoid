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

Both are calibrated to quantiles of the *real* trajectory, so the threshold is
in units of "how weird would this have been in calibration", not raw distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GateReading", "SheafGate"]


@dataclass(frozen=True)
class GateReading:
    """Why the gate did or did not fire."""

    sheaf_residual: float
    manifold_distance: float
    sheaf_score: float
    """Residual in calibration quantile units; 1.0 == the calibration threshold."""

    manifold_score: float
    score: float
    """max of the two scores. >= 1.0 means ground now."""

    fire: bool
    reason: str


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
    topo_dim: int = 0
    fitted: bool = False

    def fit(self, states: np.ndarray, topo_dim: int) -> "SheafGate":
        """Learn R and the support model from calibration states (N, state_dim)."""
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

        self.fitted = True
        return self

    def read(self, z: np.ndarray) -> GateReading:
        """Evaluate one imagined state."""
        if not self.fitted:
            raise RuntimeError("SheafGate.fit must be called before use")
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        psi, u = z[: self.topo_dim], z[self.topo_dim :]

        predicted_psi = self.R_ @ np.concatenate([u, [1.0]])
        sheaf_residual = float(np.linalg.norm(psi - predicted_psi))
        manifold_distance = float(np.linalg.norm((z - self.mean_) * self.inv_std_))

        sheaf_score = sheaf_residual / self.sheaf_threshold_
        manifold_score = manifold_distance / self.manifold_threshold_
        score = max(sheaf_score, manifold_score)
        reason = "ok"
        if score >= 1.0:
            reason = "sheaf_inconsistent" if sheaf_score >= manifold_score else "off_manifold"
        return GateReading(
            sheaf_residual=sheaf_residual,
            manifold_distance=manifold_distance,
            sheaf_score=sheaf_score,
            manifold_score=manifold_score,
            score=score,
            fire=bool(score >= 1.0),
            reason=reason,
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
            "topo_dim": self.topo_dim,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "SheafGate":
        gate = cls(quantile=float(payload["quantile"]), ridge=float(payload["ridge"]))
        gate.R_ = payload["R_"]
        gate.mean_ = payload["mean_"]
        gate.inv_std_ = payload["inv_std_"]
        gate.sheaf_threshold_ = float(payload["sheaf_threshold_"])
        gate.manifold_threshold_ = float(payload["manifold_threshold_"])
        gate.topo_dim = int(payload["topo_dim"])
        gate.fitted = True
        return gate

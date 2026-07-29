"""Multi-entity dynamics: the Hamilton Tensor construction, applied to streams.

A single coupling operator models one thing evolving. Real worlds contain
several things that evolve *because of each other* -- agents, objects, KV-cache
shards, parallel decode streams, ensemble members.

The Hamilton Tensor handles this by learning one unified coupling H over the
tensor product of per-body topological signatures, and demanding a multi-state
fixed point H^{(x)N}(Psi) = Psi. Written out on the joint state
Psi = [psi_1 ; ... ; psi_N] in R^{Nd}, that is a block operator

    Psi_{t+1} = M Psi_t + b,    M = [[M_11 ... M_1N], ..., [M_N1 ... M_NN]]

where the off-diagonal blocks M_ij are exactly the inter-body coupling. Fitting
M directly costs O(N^2 d^2) parameters, which is the d^N blow-up in disguise
once N grows. The fix carried over from hamliton is a rank-R truncation: keep
only the top R singular directions of M, giving an O(N d R) operator that still
carries the dominant coupling modes.

The same spectral projection as the single-body case applies, so multi-body
rollouts inherit the same contraction certificate. The overlap matrix
O_ij = <psi_i | psi_j> reports which bodies actually ended up coupled, which is
the diagnostic worth reading before trusting any of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MultiBodyCoupling"]


@dataclass
class MultiBodyCoupling:
    """Joint contractive dynamics over N coupled bodies."""

    rank: int | None = None
    """Rank-R truncation of the joint operator. None keeps it full."""

    ridge: float = 1e-3
    rho_max: float = 0.995

    M_: np.ndarray | None = field(default=None, repr=False)
    b_: np.ndarray | None = field(default=None, repr=False)
    n_bodies: int = 0
    body_dim: int = 0
    rho_: float = 0.0
    step_rmse_: float = 0.0
    joint_fixed_point_: np.ndarray | None = field(default=None, repr=False)
    fixed_point_converged: bool = False
    fitted: bool = False

    # ---- fit --------------------------------------------------------------

    def fit(self, bodies: list[np.ndarray]) -> "MultiBodyCoupling":
        """Fit on per-body state trajectories, each of shape (T, d), shared T."""
        arrs = [np.asarray(b, dtype=np.float64) for b in bodies]
        if len(arrs) < 2:
            raise ValueError("need at least 2 bodies")
        lengths = {a.shape[0] for a in arrs}
        dims = {a.shape[1] for a in arrs}
        if len(lengths) != 1 or len(dims) != 1:
            raise ValueError("all bodies must share (T, d)")
        if arrs[0].shape[0] < 2:
            raise ValueError("need at least 2 time steps")

        self.n_bodies = len(arrs)
        self.body_dim = arrs[0].shape[1]

        joint = np.concatenate(arrs, axis=1)  # (T, N*d)
        X = np.concatenate([joint[:-1], np.ones((joint.shape[0] - 1, 1))], axis=1)
        Y = joint[1:]

        gram = X.T @ X + self.ridge * np.eye(X.shape[1])
        W = np.linalg.solve(gram, X.T @ Y).T
        self.M_, self.b_ = W[:, :-1], W[:, -1]

        if self.rank is not None:
            self._truncate_rank(self.rank)
        self._project_spectral()

        residual = Y - (joint[:-1] @ self.M_.T + self.b_)
        self.step_rmse_ = float(np.sqrt(np.mean(residual**2)))
        self._solve_fixed_point()
        self.fitted = True
        return self

    def _truncate_rank(self, rank: int) -> None:
        assert self.M_ is not None
        U, s, Vt = np.linalg.svd(self.M_, full_matrices=False)
        r = max(1, min(int(rank), s.shape[0]))
        self.M_ = (U[:, :r] * s[:r]) @ Vt[:r]

    def _project_spectral(self) -> None:
        assert self.M_ is not None
        U, s, Vt = np.linalg.svd(self.M_, full_matrices=False)
        rho = float(s[0]) if s.size else 0.0
        if rho > self.rho_max:
            s = s * (self.rho_max / rho)
            self.M_ = (U * s) @ Vt
            rho = self.rho_max
        self.rho_ = rho

    def _solve_fixed_point(self, iters: int = 512, tol: float = 1e-12) -> None:
        """Multi-state Banach iteration to the joint invariant Psi*."""
        assert self.M_ is not None and self.b_ is not None
        dim = self.M_.shape[0]
        try:
            psi_star = np.linalg.solve(np.eye(dim) - self.M_, self.b_)
        except np.linalg.LinAlgError:
            self.joint_fixed_point_, self.fixed_point_converged = None, False
            return

        psi = np.zeros(dim, dtype=np.float64)
        converged = False
        for _ in range(iters):
            nxt = self.M_ @ psi + self.b_
            if float(np.linalg.norm(nxt - psi)) < tol:
                psi, converged = nxt, True
                break
            psi = nxt
        self.joint_fixed_point_ = psi_star
        self.fixed_point_converged = bool(
            converged and np.linalg.norm(psi - psi_star) < 1e-6
        )

    # ---- use --------------------------------------------------------------

    def step(self, bodies: list[np.ndarray]) -> list[np.ndarray]:
        """Advance all bodies one step, coupled."""
        self._check_fitted()
        joint = np.concatenate([np.asarray(b, dtype=np.float64).reshape(-1) for b in bodies])
        if joint.shape[0] != self.M_.shape[0]:
            raise ValueError(f"expected joint dim {self.M_.shape[0]}, got {joint.shape[0]}")
        nxt = self.M_ @ joint + self.b_
        return list(np.split(nxt, self.n_bodies))

    def rollout(self, bodies: list[np.ndarray], steps: int) -> list[list[np.ndarray]]:
        """Imagine `steps` ahead for every body."""
        out = []
        current = [np.asarray(b, dtype=np.float64).reshape(-1) for b in bodies]
        for _ in range(steps):
            current = self.step(current)
            out.append([c.copy() for c in current])
        return out

    def coupling_strength(self) -> np.ndarray:
        """(N, N) matrix of block Frobenius norms. Off-diagonal == real coupling."""
        self._check_fitted()
        n, d = self.n_bodies, self.body_dim
        blocks = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                blocks[i, j] = np.linalg.norm(
                    self.M_[i * d : (i + 1) * d, j * d : (j + 1) * d]
                )
        return blocks

    @staticmethod
    def overlap_matrix(bodies: list[np.ndarray]) -> np.ndarray:
        """Pairwise normalized Hilbert overlaps <psi_i | psi_j>."""
        arrs = [np.asarray(b, dtype=np.float64).reshape(-1) for b in bodies]
        n = len(arrs)
        out = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                na, nb = np.linalg.norm(arrs[i]), np.linalg.norm(arrs[j])
                val = 0.0 if na < 1e-10 or nb < 1e-10 else float(arrs[i] @ arrs[j] / (na * nb))
                out[i, j] = out[j, i] = val
        return out

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("MultiBodyCoupling.fit must be called before use")

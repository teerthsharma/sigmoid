"""T: the topological coupling operator, and its Banach fixed point.

This is the Faraday/Hamilton construction lifted from electromagnetic cavities
to model activations. There, T was learned between the topological embeddings
of the E and H fields of one cavity mode, and iterated to a fixed point. Here
the two "fields" are the world state now and the world state next:

    T psi_t  ~  psi_{t+1}

Learned by ridge regression in closed form, then *spectrally projected* so that
sigma_max(T) <= rho < 1. That projection is the whole point. A learned neural
dynamics model rolled out 50 steps can do anything; a contraction cannot. With
rho < 1 the Banach fixed-point theorem gives, for free:

    - a unique fixed point z* (the model's attractor -- Faraday's "God Tensor"),
    - convergence z_n -> z* at rate rho^n,
    - a computable *a-priori* error bound on any rollout (see `certificate`).

That bound is what makes imagination safe to trust for k steps and what tells
the engine when to stop imagining and call the real model.

Action conditioning uses the standard bilinear lift: the operator applied to
[z ; a (x) z ; a ; 1] is exactly T_0 z + sum_k a_k T_k z + B a + c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CouplingOperator", "RolloutCertificate"]


@dataclass(frozen=True)
class RolloutCertificate:
    """A-priori guarantee for an n-step rollout."""

    rho: float
    """Spectral norm of the operator. Contractive iff < 1."""

    contractive: bool
    step_rmse: float
    """One-step residual measured on held-out calibration data."""

    horizon: int
    bound: float
    """Worst-case accumulated error after `horizon` steps."""

    def error_bound(self, steps: int) -> float:
        """Accumulated error bound after `steps` applications.

        One-step error eps propagates as eps * sum_{i<n} rho^i, a geometric
        series that converges to eps/(1-rho) when contractive and blows up
        otherwise.
        """
        if steps <= 0:
            return 0.0
        if self.rho >= 1.0 - 1e-12:
            return float(self.step_rmse * steps * max(self.rho, 1.0) ** max(steps - 1, 0))
        return float(self.step_rmse * (1.0 - self.rho**steps) / (1.0 - self.rho))


@dataclass
class CouplingOperator:
    """Linear (optionally action-conditioned) operator on world states."""

    ridge: float = 1e-3

    rho_max: float | None = None
    """Optional spectral-norm ceiling enforced after fitting.

    None (the default) fits the dynamics the data actually shows and reports
    whatever rho comes out. Setting rho_max < 1 buys an infinite-horizon
    contraction certificate by *overwriting* the measured dynamics, which is
    the right trade only when a bounded rollout matters more than a faithful
    one. It is not free: on an expansive system (anything chaotic -- Lorenz,
    turbulent control, arguably token dynamics) the true one-step operator has
    rho > 1, and clipping it to 0.995 measurably degrades one-step accuracy.
    Turning that on by default would have meant misreporting the dynamics in
    order to make the certificate look good.
    """

    action_dim: int = 0

    block_split: int | None = None
    """Index splitting the state into two independently-evolving blocks.

    Set to `topo_dim` to make the operator block-diagonal in (psi, u): psi
    predicts psi, u predicts u, and neither is allowed to contaminate the other.

    This is not a simplification, it is what the measurements demanded. On
    distilgpt2 activations, predicting u_{t+1} scored R^2 = 0.168 from u alone
    and 0.145 from [u, psi] -- the topological channel is 32 extra noisy
    regressors that only buy overfitting. Yet psi predicts *itself* at
    R^2 = 0.762 while u predicts psi at R^2 = -0.181, so psi is a real,
    stable, and genuinely independent signal.

    Fully coupling the two also wrecked conditioning: the dense operator hit
    rho = 3.16 where the u-only operator sat at 0.98. Block-diagonal keeps psi
    rolling forward for the gate to read during imagination, without letting it
    degrade the content prediction that actually decodes.
    """

    W_: np.ndarray | None = field(default=None, repr=False)
    """(state_dim, lift_dim) operator acting on the lifted feature vector."""

    state_dim: int = 0
    rho_: float = 0.0
    step_rmse_: float = 0.0
    fixed_point_: np.ndarray | None = field(default=None, repr=False)
    fixed_point_converged: bool = False
    fitted: bool = False

    # ---- feature lift -----------------------------------------------------

    @property
    def lift_dim(self) -> int:
        d, k = self.state_dim, self.action_dim
        return d + (k * d + k if k else 0) + 1

    def _lift(self, z: np.ndarray, a: np.ndarray | None) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        n = z.shape[0]
        parts = [z]
        if self.action_dim:
            if a is None:
                raise ValueError("operator was fitted with actions; pass actions")
            a = np.atleast_2d(np.asarray(a, dtype=np.float64))
            if a.shape[0] != n or a.shape[1] != self.action_dim:
                raise ValueError(f"actions must be ({n}, {self.action_dim})")
            # a (x) z, row-wise Kronecker
            parts.append((a[:, :, None] * z[:, None, :]).reshape(n, -1))
            parts.append(a)
        parts.append(np.ones((n, 1), dtype=np.float64))
        return np.concatenate(parts, axis=1)

    # ---- fit --------------------------------------------------------------

    def fit(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> "CouplingOperator":
        """Ridge-fit T on paired states.

        Closed form: W = Y^T X (X^T X + lambda I)^{-1}. No gradient descent --
        the problem is convex and small, so there is nothing to iterate.
        """
        X_states = np.asarray(states, dtype=np.float64)
        Y = np.asarray(next_states, dtype=np.float64)
        if X_states.ndim != 2 or Y.shape != X_states.shape:
            raise ValueError("states and next_states must share shape (N, state_dim)")
        if X_states.shape[0] < 2:
            raise ValueError("need at least 2 transitions")

        self.state_dim = X_states.shape[1]
        if actions is not None:
            self.action_dim = np.atleast_2d(actions).shape[1]

        X = self._lift(X_states, actions)
        if self.block_split and 0 < self.block_split < self.state_dim:
            self.W_ = self._fit_block_diagonal(X_states, Y, actions)
        else:
            gram = X.T @ X + self.ridge * np.eye(X.shape[1])
            self.W_ = np.linalg.solve(gram, X.T @ Y).T

        self._project_spectral()
        residual = Y - X @ self.W_.T
        self.step_rmse_ = float(np.sqrt(np.mean(residual**2)))
        self.fitted = True
        self._power_iterate()
        return self

    def _fit_block_diagonal(
        self,
        X_states: np.ndarray,
        Y: np.ndarray,
        actions: np.ndarray | None,
    ) -> np.ndarray:
        """Fit each block on its own inputs, then assemble a block-diagonal W.

        Two independent ridge solves. Cross-block weights are structurally
        absent rather than merely penalized, so no amount of data can
        reintroduce the coupling the ablation showed to be harmful.
        """
        split = int(self.block_split)
        W = np.zeros((self.state_dim, self.lift_dim), dtype=np.float64)
        for lo, hi in ((0, split), (split, self.state_dim)):
            X_block = self._lift_block(X_states[:, lo:hi], actions, hi - lo)
            gram = X_block.T @ X_block + self.ridge * np.eye(X_block.shape[1])
            W_block = np.linalg.solve(gram, X_block.T @ Y[:, lo:hi]).T
            W[lo:hi, lo:hi] = W_block[:, : hi - lo]
            W[lo:hi, -1] = W_block[:, -1]
            if self.action_dim:
                # a (x) z columns for this block, then the raw action columns
                k, d = self.action_dim, self.state_dim
                for j in range(k):
                    src = (hi - lo) + j * (hi - lo)
                    dst = d + j * d + lo
                    W[lo:hi, dst : dst + (hi - lo)] = W_block[:, src : src + (hi - lo)]
                W[lo:hi, d + k * d : d + k * d + k] = W_block[:, -1 - k : -1]
        return W

    def _lift_block(
        self, z: np.ndarray, a: np.ndarray | None, block_dim: int
    ) -> np.ndarray:
        parts = [z]
        if self.action_dim:
            a = np.atleast_2d(np.asarray(a, dtype=np.float64))
            parts.append((a[:, :, None] * z[:, None, :]).reshape(z.shape[0], -1))
            parts.append(a)
        parts.append(np.ones((z.shape[0], 1), dtype=np.float64))
        return np.concatenate(parts, axis=1)

    def _project_spectral(self) -> None:
        """Clip the singular values of the state block onto the rho_max ball.

        This is the gauge projection from the Hamilton Tensor, relaxed: instead
        of forcing SU(d) unitarity (which would force rho = 1 exactly and make
        rollouts marginally stable at best), we force strict contraction. Only
        the state->state block matters for stability; the action and bias blocks
        are bounded inputs, not feedback.
        """
        assert self.W_ is not None
        A = self.W_[:, : self.state_dim]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        rho = float(s[0]) if s.size else 0.0
        if self.rho_max is not None and rho > self.rho_max:
            s = s * (self.rho_max / rho)
            self.W_[:, : self.state_dim] = (U * s) @ Vt
            rho = self.rho_max
        self.rho_ = rho

    # ---- apply ------------------------------------------------------------

    def step(self, z: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        """Advance one state (or a batch of states) by one step."""
        self._check_fitted()
        single = np.asarray(z).ndim == 1
        out = self._lift(z, action) @ self.W_.T
        return out[0] if single else out

    def rollout(
        self,
        z0: np.ndarray,
        steps: int,
        actions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Imagine `steps` states ahead. Returns (steps, state_dim)."""
        self._check_fitted()
        z = np.asarray(z0, dtype=np.float64).reshape(-1)
        out = np.empty((steps, self.state_dim), dtype=np.float64)
        for i in range(steps):
            a = None if actions is None else np.asarray(actions)[i]
            z = self.step(z, a)
            out[i] = z
        return out

    # ---- fixed point ------------------------------------------------------

    def _power_iterate(self, iters: int = 512, tol: float = 1e-12) -> None:
        """Find the fixed point z* of the autonomous map z -> W [z; 0; 1].

        For a contraction with bias, the fixed point solves (I - A) z = b
        directly, which is exact and cheaper than iterating. We still record a
        Banach iteration residual so the convergence claim is *measured*, in the
        spirit of the Faraday 50k-epoch burn, rather than asserted.
        """
        assert self.W_ is not None
        A = self.W_[:, : self.state_dim]
        b = self.W_[:, -1]
        try:
            z_star = np.linalg.solve(np.eye(self.state_dim) - A, b)
        except np.linalg.LinAlgError:
            self.fixed_point_, self.fixed_point_converged = None, False
            return

        # Banach iteration. It only converges when rho < 1; on an expansive
        # operator it diverges, and that divergence is the honest report --
        # the fixed point exists algebraically but is not reachable by
        # iterating, so `fixed_point_converged` stays False.
        z = np.zeros(self.state_dim, dtype=np.float64)
        converged = False
        for _ in range(iters):
            z_next = A @ z + b
            if not np.all(np.isfinite(z_next)):
                break
            if float(np.linalg.norm(z_next - z)) < tol:
                converged = True
                z = z_next
                break
            z = z_next
        self.fixed_point_ = z_star
        self.fixed_point_converged = bool(
            converged and np.linalg.norm(z - z_star) < 1e-6
        )

    def certificate(self, horizon: int = 16) -> RolloutCertificate:
        self._check_fitted()
        cert = RolloutCertificate(
            rho=self.rho_,
            contractive=self.rho_ < 1.0,
            step_rmse=self.step_rmse_,
            horizon=horizon,
            bound=0.0,
        )
        return RolloutCertificate(
            rho=cert.rho,
            contractive=cert.contractive,
            step_rmse=cert.step_rmse,
            horizon=horizon,
            bound=cert.error_bound(horizon),
        )

    def safe_horizon(self, tolerance: float) -> int:
        """Largest k whose guaranteed error bound stays under `tolerance`.

        This is what the engine uses to decide how long it may imagine before
        it must ground itself against the real model.
        """
        self._check_fitted()
        cert = self.certificate(1)
        if cert.error_bound(1) > tolerance:
            return 0
        k = 1
        while k < 4096 and cert.error_bound(k + 1) <= tolerance:
            k += 1
        return k

    def _check_fitted(self) -> None:
        if not self.fitted or self.W_ is None:
            raise RuntimeError("CouplingOperator.fit must be called before use")

    # ---- persistence ------------------------------------------------------

    def state_dict(self) -> dict:
        self._check_fitted()
        return {
            "ridge": self.ridge,
            "rho_max": self.rho_max,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "W_": self.W_,
            "rho_": self.rho_,
            "step_rmse_": self.step_rmse_,
            "fixed_point_": self.fixed_point_,
            "fixed_point_converged": self.fixed_point_converged,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "CouplingOperator":
        rho_max = payload["rho_max"]
        op = cls(
            ridge=float(payload["ridge"]),
            rho_max=None if rho_max is None else float(rho_max),
            action_dim=int(payload["action_dim"]),
        )
        op.state_dim = int(payload["state_dim"])
        op.W_ = payload["W_"]
        op.rho_ = float(payload["rho_"])
        op.step_rmse_ = float(payload["step_rmse_"])
        op.fixed_point_ = payload["fixed_point_"]
        op.fixed_point_converged = bool(payload["fixed_point_converged"])
        op.fitted = True
        return op

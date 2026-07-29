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

from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["CouplingOperator", "RolloutCertificate"]


@dataclass(frozen=True)
class RolloutCertificate:
    """A-priori guarantee for an n-step rollout.

    Two error estimates live here and they are *not* interchangeable:

    `error_bound` is a worst-case GUARANTEE. It assumes the one-step error
    points along the worst direction and compounds maximally at every step, so
    nothing can violate it. On real data nothing comes near it either: on
    distilgpt2 activations it read 6.95 at k=16 (rho_max=0.90) against a signal
    of magnitude ~1, and on the S2-Rips corpus 82.6 against a measured latent
    error of 0.6580 -- over 100x too loose to act on. True, and useless.

    `directional_error` is an ESTIMATE. It propagates the full residual
    covariance through the operator instead of a single scalar, so residual
    energy that A contracts stops being charged at the worst-case rate. It is
    exact for the linear model *under its assumptions* (see the method) and
    silently optimistic when they fail. Do not quote it as a guarantee.
    """

    rho: float
    """Spectral norm of the operator. Contractive iff < 1."""

    contractive: bool
    step_rmse: float
    """One-step residual measured on held-out calibration data."""

    horizon: int
    bound: float
    """Worst-case accumulated error after `horizon` steps."""

    residual_cov: np.ndarray | None = field(default=None, repr=False, compare=False)
    """(d, d) second moment of the one-step residual. None for operators
    restored from a state dict written before R6."""

    state_block: np.ndarray | None = field(default=None, repr=False, compare=False)
    """The state->state block A, needed to propagate `residual_cov`."""

    directional_bound: float = float("nan")
    """Covariance-propagated RMS error estimate at `horizon`. NaN if unavailable."""

    residual_autocorr: float = 0.0
    """Measured lag-1 residual autocorrelation -- read it before believing
    `directional_bound`. Near 0 means the step-independence assumption holds
    and the estimate tracked the measured error to ~1% in test. Above ~0.5 the
    estimate under-bounds (0.81 -> 2.4x low, 0.94 -> 4.6x low), and only
    `error_bound` still means anything."""

    def error_bound(self, steps: int) -> float:
        """Accumulated error bound after `steps` applications.

        One-step error eps propagates as eps * sum_{i<n} rho^i, a geometric
        series that converges to eps/(1-rho) when contractive and blows up
        otherwise.
        """
        if steps <= 0:
            return 0.0
        if self.rho >= 1.0 - 1e-12:
            try:
                return float(
                    self.step_rmse * steps * max(self.rho, 1.0) ** max(steps - 1, 0)
                )
            except OverflowError:
                # rho^steps left float range; unbounded is the honest answer
                return float("inf")
        return float(self.step_rmse * (1.0 - self.rho**steps) / (1.0 - self.rho))

    def directional_error(self, steps: int) -> float:
        """Covariance-propagated RMS error after `steps` steps. AN ESTIMATE.

        The scalar bound above throws away everything except ||residual||: it
        charges every step the worst-case direction at rate rho. Real residuals
        are anisotropic, and the directions carrying most of their energy are
        usually not the directions A amplifies most. Tracking the covariance
        keeps that alignment. For z_{t+1} = A z_t + r with Cov(r) = Sigma, the
        error covariance obeys

            C_n = A C_{n-1} A^T + Sigma,   C_0 = 0

        i.e. C_n = sum_{i<n} A^i Sigma (A^i)^T, and the expected squared error
        is trace(C_n). Returned as sqrt(trace(C_n) / d) so it is directly
        comparable to `error_bound` and to a measured NRMSE.

        ASSUMPTIONS -- this is why it is an estimate and not a bound:
          1. residuals are zero-mean (a persistent bias is invisible to a
             covariance that never sees it accumulate coherently),
          2. residuals are uncorrelated across steps (independent errors add in
             quadrature; correlated ones add in amplitude, which is exactly
             what the scalar bound assumes),
          3. the dynamics really are the linear A (model error outside the
             residual, e.g. a state-dependent bias, is not represented).
        Under 1-3 this is the *expected* squared error, so roughly half of
        individual rollouts land above it -- it is a mean, not a tail bound.
        Violate 2 and it under-estimates badly: an AR(1) residual with
        persistence 0.9 measures ~2x this value (see tests/test_certificate.py,
        test_step_correlated_residuals_break_the_estimate). When you need
        something nothing can violate, use `error_bound`.

        Note also that even for perfectly isotropic Sigma this sits below the
        scalar bound by the quadrature-vs-amplitude gap (sqrt(sum rho^2i)
        vs sum rho^i). That gap is not extra tightness, it is assumption 2.

        O(steps * d^3). d is typically < 200, so the dense recursion is fine.
        """
        if steps <= 0:
            return 0.0
        rms = float("nan")
        for rms in self._directional_walk(steps):  # noqa: B007 -- last value wins
            pass
        return rms

    def _directional_walk(self, steps: int):
        """Yield the propagated RMS error at every step 1..`steps`.

        One shared recursion: `safe_horizon` scans up to 4096 steps, and
        restarting the O(k d^3) recursion for each k made that quadratic (28s at
        d=10, versus 7ms this way).
        """
        if self.residual_cov is None or self.state_block is None:
            return
        A, sigma = self.state_block, self.residual_cov
        d = sigma.shape[0]
        C = np.zeros_like(sigma)
        for _ in range(steps):
            # An expansive A overflows the recursion. That is not an error, it
            # is the honest answer: report inf, and keep NaN meaning strictly
            # "no covariance available".
            with np.errstate(over="ignore", invalid="ignore"):
                C = A @ C @ A.T + sigma
                total = float(np.trace(C))
            if not np.isfinite(total):
                yield float("inf")
                return
            yield float(np.sqrt(max(total, 0.0) / d))


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
    residual_cov_: np.ndarray | None = field(default=None, repr=False)
    """(state_dim, state_dim) second moment of the one-step residual.

    The scalar `step_rmse_` is its trace; keeping the whole thing is what lets
    `RolloutCertificate.directional_error` charge each residual direction the
    rate A actually applies to it instead of the worst-case rho.

    In-sample, like `step_rmse_`: both inherit the same optimism.
    """

    residual_autocorr_: float = 0.0
    """Lag-1 autocorrelation of the residual. ~0 validates the directional
    estimate's step-independence assumption; anything large invalidates it."""
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
    ) -> CouplingOperator:
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
        # Second moment about zero, not about the mean: a residual bias is a
        # real error and must be charged for. It also makes
        # trace(residual_cov_)/d == step_rmse_**2 exactly, so the scalar and
        # directional estimates agree at n = 1 by construction and any later
        # divergence is entirely the propagation, not the calibration.
        self.residual_cov_ = (residual.T @ residual) / residual.shape[0]
        # Lag-1 autocorrelation of the residual sequence: the one number that
        # says whether the directional estimate may be trusted. It assumes
        # residuals are step-uncorrelated, and that assumption is not
        # self-certifying, so measure it. Measured: 0.000 on an iid linear
        # system, 0.81 on an AR(1)-residual system where the estimate
        # under-bounds by 2.4x, 0.94 on Lorenz where it under-bounds by 4.6x.
        # Assumes rows are in temporal order (concatenated episodes contaminate
        # one row per seam, which is negligible and never in the safe direction).
        denom = float(np.sum(residual**2))
        self.residual_autocorr_ = (
            float(np.sum(residual[1:] * residual[:-1]) / denom) if denom > 0 else 0.0
        )
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
        # The state->state block is the only part that feeds back; actions and
        # bias are bounded inputs. For an action-conditioned operator the true
        # per-step map is A + sum_k a_k T_k, so the propagated covariance is the
        # a = 0 case -- same limitation rho already has.
        cert = RolloutCertificate(
            rho=self.rho_,
            contractive=self.rho_ < 1.0,
            step_rmse=self.step_rmse_,
            horizon=horizon,
            bound=0.0,
            residual_cov=self.residual_cov_,
            state_block=self.W_[:, : self.state_dim],
            residual_autocorr=self.residual_autocorr_,
        )
        return replace(
            cert,
            bound=cert.error_bound(horizon),
            directional_bound=cert.directional_error(horizon),
        )

    def safe_horizon(self, tolerance: float, directional: bool = False) -> int:
        """Largest k whose error stays under `tolerance`.

        This is what the engine uses to decide how long it may imagine before
        it must ground itself against the real model.

        `directional=False` (the default) uses the worst-case guarantee, which
        on measured data is so loose it usually answers 0. `directional=True`
        uses the covariance estimate instead: far more useful, but it inherits
        that method's assumptions and is no longer a guarantee.
        """
        self._check_fitted()
        cap = 4096
        cert = self.certificate(1)
        errors = (
            cert._directional_walk(cap)
            if directional
            else (cert.error_bound(k) for k in range(1, cap + 1))
        )
        k = 0
        for err in errors:
            if not err <= tolerance:  # NaN-safe: an unavailable estimate is unsafe
                break
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
            "residual_cov_": self.residual_cov_,
            "residual_autocorr_": self.residual_autocorr_,
            "fixed_point_": self.fixed_point_,
            "fixed_point_converged": self.fixed_point_converged,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> CouplingOperator:
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
        # .get: checkpoints written before R6 have no covariance. They keep
        # working, they just report a NaN directional estimate.
        op.residual_cov_ = payload.get("residual_cov_")
        op.residual_autocorr_ = float(payload.get("residual_autocorr_", 0.0))
        op.fixed_point_ = payload["fixed_point_"]
        op.fixed_point_converged = bool(payload["fixed_point_converged"])
        op.fitted = True
        return op

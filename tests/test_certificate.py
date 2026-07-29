"""Checks for the two rollout error estimates. `python tests/test_certificate.py`.

The scalar Banach bound (`RolloutCertificate.error_bound`) is a guarantee and is
vacuous: on distilgpt2 it read 6.95 at k=16 against a signal of magnitude ~1, and
82.6 against a measured 0.6580 on S2-Rips. The directional estimate
(`RolloutCertificate.directional_error`) propagates the residual covariance and
is an order of magnitude tighter -- but it is an ESTIMATE, and these tests exist
mostly to pin down where it stops being true. Measured here at k=16:

  system                    scalar/directional   directional/measured
  isotropic residuals             4.3                  0.97
  anisotropic residuals          12.2                  1.00
  AR(1) residuals (rho_ac .81)    4.1                  0.41  <- UNDER-bounds 2.4x
  Lorenz      (rho_ac .94)      1.5e6                  0.22  <- UNDER-bounds 4.6x

The last two are the point. The estimate is exact when its assumptions hold and
quietly optimistic when they do not, and "when they do not" includes every
nonlinear system. `residual_autocorr` is the measured discriminator: 0.00 on the
first two rows, 0.81 and 0.94 on the last two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sigmoid
from sigmoid.operator import RolloutCertificate

K = 16  # the horizon every measured number in AGENDA.md R6 is quoted at


def _rotation(d: int, seed: int) -> np.ndarray:
    """Random orthogonal matrix, so nothing is accidentally axis-aligned.

    A diagonal A with a diagonal Sigma would let a coordinate-wise method look
    directional by luck. Rotating both means only a genuine covariance
    propagation can see the structure.
    """
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def _simulate(A, chol, n=6000, seed=0, phi=0.0):
    """z_{t+1} = A z_t + r_t, r_t = phi r_{t-1} + w.

    phi=0 gives the iid residuals the directional estimate assumes. phi>0 keeps
    the *marginal* residual covariance identical (w is scaled by sqrt(1-phi^2))
    while making the sequence correlated -- so the fitted certificate barely
    changes and the real error explodes.
    """
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    out = np.empty((n, d))
    z, r = np.zeros(d), np.zeros(d)
    scale = np.sqrt(1.0 - phi**2)
    for t in range(n):
        out[t] = z
        r = phi * r + scale * (chol @ rng.normal(size=d))
        z = A @ z + r
    return out[n // 6 :]  # burn in to the stationary distribution


def _fit(A, chol, seed=0, n=20000, phi=0.0):
    """Fit on the first half of a simulated trajectory, return the second."""
    traj = _simulate(A, chol, n=n, seed=seed, phi=phi)
    cut = len(traj) // 2
    op = sigmoid.CouplingOperator(ridge=1e-8).fit(traj[:cut][:-1], traj[:cut][1:])
    return op, traj[cut:]


def _step_errors(op, traj: np.ndarray, k: int) -> np.ndarray:
    """Per-start-point mean squared error of a k-step open-loop rollout."""
    starts = np.arange(len(traj) - k)
    Z = traj[starts]
    for _ in range(k):
        Z = op.step(Z)
    return np.mean((Z - traj[starts + k]) ** 2, axis=1)


def _measured(op, traj: np.ndarray, k: int) -> float:
    return float(np.sqrt(np.mean(_step_errors(op, traj, k))))


def _anisotropic(seed: int = 0):
    """Slow directions carry almost no residual, fast directions carry it all.

    This is the shape real residuals have -- error concentrates where the
    operator has already given up -- and it is exactly what a scalar rho plus a
    scalar eps cannot represent. Both A and Sigma are rotated by the same random
    Q, so the structure is directional rather than coordinate-wise.
    """
    d = 10
    Q = _rotation(d, seed)
    a = np.array([0.95, 0.92] + [0.45] * (d - 2))
    s = np.array([0.02, 0.02] + [0.4] * (d - 2))
    return (Q * a) @ Q.T, (Q * s) @ Q.T


# ---- calibration ---------------------------------------------------------


def test_covariance_trace_reproduces_the_scalar_rmse():
    """trace(Sigma)/d == step_rmse^2, so the two estimates agree at n = 1.

    If this drifts, every later comparison is measuring a calibration bug
    rather than the propagation.
    """
    op, _ = _fit(*_anisotropic())
    cert = op.certificate(K)
    assert cert.residual_cov is not None and cert.residual_cov.shape == (10, 10)
    trace_rmse = np.trace(cert.residual_cov) / op.state_dim
    assert abs(trace_rmse / op.step_rmse_**2 - 1.0) < 1e-12
    assert abs(cert.directional_error(1) - cert.error_bound(1)) < 1e-12


def test_directional_recursion_matches_the_closed_form():
    """A = rho Q, Sigma = s^2 I: C_n = s^2 sum_i rho^2i I has an exact answer.

    Checked on a hand-built certificate rather than a fitted one so this is a
    test of the recursion alone. A tight number that misses this closed form is
    tight because it is wrong.
    """
    d, rho, s = 8, 0.9, 0.3
    cert = RolloutCertificate(
        rho=rho, contractive=True, step_rmse=s, horizon=K, bound=0.0,
        residual_cov=s**2 * np.eye(d), state_block=rho * _rotation(d, 1),
    )
    for n in (1, 4, K):
        closed = s * np.sqrt((1 - rho ** (2 * n)) / (1 - rho**2))
        assert abs(cert.directional_error(n) - closed) < 1e-12 * max(closed, 1.0), (
            f"n={n}: {cert.directional_error(n):.9f} vs closed form {closed:.9f}"
        )


def test_isotropic_bounds_roughly_agree():
    """Isotropic residuals: the only gap left is quadrature vs amplitude.

    sum rho^i / sqrt(sum rho^2i) = 3.61 at rho=0.9, k=16, and the fit's slight
    over-estimate of rho (max singular value of a noisy estimate) pushes the
    measured ratio to ~4.3. That factor is assumption 2 (step-independence), not
    anisotropy, and it is the floor of what the directional estimate can ever
    buy. Much more than that here would mean the 12x below is an artefact.
    """
    d, rho = 8, 0.9
    op, _ = _fit(rho * _rotation(d, 1), 0.3 * np.eye(d), seed=1)
    cert = op.certificate(K)
    ratio = cert.bound / cert.directional_bound
    assert 3.0 < ratio < 5.0, f"isotropic tightness ratio {ratio:.2f}, expected ~4.3"


# ---- the actual claim ----------------------------------------------------


def test_anisotropic_directional_is_much_tighter_and_tracks_the_truth():
    """R6's claim: an order of magnitude tighter, and still where the error is."""
    op, held = _fit(*_anisotropic())
    cert = op.certificate(K)
    measured = _measured(op, held, K)

    ratio = cert.bound / cert.directional_bound
    assert ratio > 8.0, f"expected ~12x tightening, got {ratio:.2f}"
    assert cert.bound > measured, f"scalar bound must hold: {cert.bound} vs {measured}"
    # Under its assumptions the estimate IS the expected error, so it has to
    # land on the measurement -- being merely below a useless number is not a
    # result. Measured 1.000 here; 5% is slack for the fit, not for the theory.
    assert abs(cert.directional_bound / measured - 1.0) < 0.05, (
        f"directional {cert.directional_bound:.4f} vs measured {measured:.4f}"
    )
    assert abs(cert.residual_autocorr) < 0.05, "iid residuals must read as iid"


def test_directional_is_a_mean_not_a_tail_bound():
    """LOUD: a large fraction of individual rollouts exceed the estimate.

    trace(C_n) is an expectation. Roughly half the rollouts of a symmetric error
    distribution sit above their root-mean-square, and that is what this
    measures. Anyone reading `directional_error` as a per-rollout guarantee is
    wrong about that often. `error_bound` is the one nothing violates.
    """
    op, held = _fit(*_anisotropic())
    cert = op.certificate(K)
    per_rollout = np.sqrt(_step_errors(op, held, K))
    over = float(np.mean(per_rollout > cert.directional_bound))
    assert 0.2 < over < 0.7, f"exceedance fraction {over:.3f}"
    assert np.max(per_rollout) > 1.5 * cert.directional_bound, (
        "worst-case rollouts must visibly exceed the estimate"
    )
    assert np.max(per_rollout) < cert.bound  # the guarantee is never violated


def test_step_correlated_residuals_break_the_estimate():
    """LOUD: violate step-independence and the estimate UNDER-bounds by 2.4x.

    Residual r_t = 0.9 r_{t-1} + w has the same one-step covariance as the iid
    case, so `directional_error` cannot see any difference -- but the errors now
    drift coherently instead of cancelling, which is the regime the scalar bound
    was built for. This is THE failure mode: the estimate is blind to anything
    the one-step residual distribution does not encode. The only warning is
    `residual_autocorr`, which reads 0.81 here against 0.00 for iid.
    """
    d = 8
    op, held = _fit(0.9 * np.eye(d), 0.3 * np.eye(d), seed=7, n=12000, phi=0.9)
    cert = op.certificate(K)
    measured = _measured(op, held, K)

    assert cert.directional_bound < measured, (
        "expected the estimate to under-bound under correlated residuals; got "
        f"{cert.directional_bound:.4f} >= {measured:.4f}"
    )
    assert measured / cert.directional_bound > 1.8, (
        f"under-estimation factor only {measured / cert.directional_bound:.2f}"
    )
    assert cert.bound > measured, "the scalar guarantee must still hold here"
    assert cert.residual_autocorr > 0.5, (
        f"the assumption was violated and the diagnostic missed it: "
        f"{cert.residual_autocorr:.3f}"
    )


def test_nonlinear_dynamics_under_bound_the_worst_and_look_the_best():
    """LOUD: on Lorenz the estimate reads 0.24 where the error is 1.10.

    Chaos makes residuals a smooth function of state, so they are strongly
    step-correlated (autocorr 0.94) and the estimate is 4.6x low -- while
    *looking* informative, since 0.24 < 1.0 and the signal has magnitude ~1.
    The scalar bound on the same operator is 1.6e6. So on the systems that
    matter you get to choose between a number that certifies nothing and a
    number that certifies the wrong thing; `residual_autocorr` is what tells
    you which one you are holding.
    """
    from test_sigmoid import lorenz

    traj = lorenz(n=2000)
    op = sigmoid.CouplingOperator(ridge=1e-6).fit(traj[:1200][:-1], traj[:1200][1:])
    cert = op.certificate(K)
    measured = _measured(op, traj[1200:], K)

    assert cert.bound > 1.0, "if the scalar bound ever became informative, say so"
    assert cert.directional_bound < cert.bound
    assert cert.directional_bound < measured, "expected an under-bound on Lorenz"
    assert measured / cert.directional_bound > 3.0, (
        f"under-estimation factor {measured / cert.directional_bound:.2f}"
    )
    assert cert.residual_autocorr > 0.5, f"missed it: {cert.residual_autocorr:.3f}"


# ---- plumbing ------------------------------------------------------------


def test_directional_survives_a_state_dict_round_trip():
    op, _ = _fit(*_anisotropic(), n=6000)
    clone = sigmoid.CouplingOperator.from_state_dict(op.state_dict())
    assert clone.certificate(K).directional_bound == op.certificate(K).directional_bound
    assert clone.certificate(K).residual_autocorr == op.certificate(K).residual_autocorr


def test_missing_covariance_degrades_to_nan_not_to_a_lie():
    """A pre-R6 checkpoint must report 'unknown', never a fabricated number."""
    op, _ = _fit(*_anisotropic(), n=6000)
    payload = op.state_dict()
    del payload["residual_cov_"]
    clone = sigmoid.CouplingOperator.from_state_dict(payload)
    assert np.isnan(clone.certificate(K).directional_bound)
    assert clone.certificate(K).bound > 0.0  # the scalar bound still works
    assert clone.safe_horizon(1e9, directional=True) == 0  # NaN is not "safe"


def test_directional_safe_horizon_is_longer_than_the_certified_one():
    """The whole point of R6: a horizon a deployment engineer would act on."""
    op, _ = _fit(*_anisotropic(), n=6000)
    tol = 1.5 * op.step_rmse_
    assert op.safe_horizon(tol, directional=True) > op.safe_horizon(tol)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all green' if not failures else f'{failures} failing'}")
    sys.exit(1 if failures else 0)

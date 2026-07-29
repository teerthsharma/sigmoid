"""Runnable checks for sigmoid. `python tests/test_sigmoid.py` or pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sigmoid
from sigmoid.state import Barcode, h0_barcode, hilbert_coefficients


def lorenz(n=1200, dt=0.01, dim=24, seed=0):
    """A real nonlinear system lifted into a higher-dimensional 'hidden' space."""
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


# ---- barcodes ------------------------------------------------------------


def test_h0_barcode_counts_two_clusters():
    """Two tight clusters must produce exactly one long H0 bar (the gap)."""
    rng = np.random.default_rng(0)
    a = rng.normal(scale=0.01, size=(20, 2))
    b = rng.normal(scale=0.01, size=(20, 2)) + np.array([10.0, 0.0])
    bc = h0_barcode(np.vstack([a, b]))
    assert bc.bars.shape[0] == 40, "n points => n H0 bars (n-1 finite + 1 essential)"
    finite = np.sort(bc.lifetimes)
    assert finite[-1] > 0.9, f"cluster gap should dominate, got {finite[-1]:.3f}"
    assert finite[-2] < 0.1, "within-cluster merges must be short"


def test_h0_barcode_merges_coincident_points_at_zero():
    """A zero-distance edge is an edge. csr drops explicit zeros, so the MST
    used to route around coincident points and report them separating at O(1)."""
    bc = h0_barcode(np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]))
    assert bc.bars.shape[0] == 3, "n points => n H0 bars"
    deaths = np.sort(bc.bars[:, 1])
    assert deaths[0] == 0.0, f"coincident pair must merge at 0, got {deaths[0]}"
    assert deaths[1] == 1.0, f"one real merge at the diameter, got {deaths[1]}"


def test_hilbert_coefficients_sum_to_essential_classes():
    """N(1) = births - finite deaths = number of essential classes."""
    bars = np.array([[0.0, 0.4], [0.0, 0.7], [0.0, np.inf]])
    coeffs = hilbert_coefficients(Barcode(0, bars, 1.0), degree=16)
    assert abs(coeffs.sum() - 1.0) < 1e-12


def test_h0_is_scale_invariant():
    """Doubling the cloud must not change the normalized barcode."""
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(40, 5))
    a = hilbert_coefficients(h0_barcode(pts), 16)
    b = hilbert_coefficients(h0_barcode(pts * 7.5), 16)
    assert np.allclose(a, b, atol=1e-9)


# ---- operator ------------------------------------------------------------


def test_operator_is_projected_into_contraction():
    """An explosive system must still yield rho < 1 after spectral projection."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=(200, 8))
    z_next = z @ (np.eye(8) * 3.0)  # spectral radius 3
    op = sigmoid.CouplingOperator(rho_max=0.9).fit(z, z_next)
    assert op.rho_ <= 0.9 + 1e-9, f"rho={op.rho_}"
    assert op.certificate(10).contractive


def test_fixed_point_is_actually_fixed():
    """Banach iteration and the direct solve must agree, and T(z*) == z*."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(300, 6))
    z_next = 0.5 * z + 0.1
    op = sigmoid.CouplingOperator().fit(z, z_next)
    star = op.fixed_point_
    assert star is not None and op.fixed_point_converged
    assert np.allclose(op.step(star), star, atol=1e-8)


def test_error_bound_is_geometric_and_finite():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(200, 5))
    op = sigmoid.CouplingOperator().fit(z[:-1], 0.5 * z[:-1] + 0.01 * z[1:])
    cert = op.certificate(8)
    assert cert.error_bound(1) <= cert.error_bound(8)
    assert cert.error_bound(10_000) < float("inf")


def test_action_conditioning_separates_dynamics():
    """Two actions driving opposite dynamics must be told apart."""
    rng = np.random.default_rng(5)
    z = rng.normal(size=(400, 4))
    a = rng.integers(0, 2, size=(400, 1)).astype(float)
    z_next = np.where(a > 0.5, 0.5 * z, -0.5 * z)
    op = sigmoid.CouplingOperator(action_dim=1, ridge=1e-8).fit(z, z_next, actions=a)
    assert op.step_rmse_ < 1e-3, f"action-conditioned fit failed: {op.step_rmse_}"


# ---- engine --------------------------------------------------------------


def test_world_model_beats_carry_forward_on_lorenz():
    """The engine must actually predict, not just echo the last state."""
    traj = lorenz()
    cfg = sigmoid.SigmoidConfig(window=24, linear_dim=12, hilbert_degree=16, n_radii=6)
    wm = sigmoid.SigmoidWorldModel(config=cfg).fit([traj[:800]])
    test = [traj[800:]]

    nrmse, _ = sigmoid.rollout_error(wm, test, [4])
    carry = np.mean(
        [
            np.mean((traj[800 + i + 3] - traj[800 + i - 1]) ** 2)
            for i in range(24, len(traj) - 800 - 4)
        ]
    )
    carry_nrmse = np.sqrt(carry) / (np.sqrt(np.mean(traj[800:] ** 2)) + 1e-12)
    assert nrmse[4] < carry_nrmse, f"sigmoid {nrmse[4]:.3f} vs carry {carry_nrmse:.3f}"


def test_imagine_stops_when_the_gate_fires():
    """A state far off the calibration manifold must trigger grounding at once."""
    traj = lorenz()
    wm = sigmoid.SigmoidWorldModel(
        config=sigmoid.SigmoidConfig(window=24, linear_dim=12)
    ).fit([traj])
    bad = np.full(wm.state_dim, 50.0)
    out = wm.imagine(bad, steps=10)
    assert out.grounded_at is not None
    assert out.readings[out.grounded_at].fire
    assert len(out) <= 10


def test_gate_accepts_real_states():
    """The gate must not fire on genuine in-distribution states."""
    traj = lorenz()
    wm = sigmoid.SigmoidWorldModel(
        config=sigmoid.SigmoidConfig(window=24, linear_dim=12, gate_quantile=0.99)
    ).fit([traj[:800]])
    fires = sum(
        wm.gate.read(wm.observe(traj[i - 24 : i])).fire for i in range(824, 1000, 4)
    )
    assert fires <= 20, f"gate fired {fires}/44 times on real states"


def test_decode_round_trip_recovers_hidden_state():
    traj = lorenz()
    wm = sigmoid.SigmoidWorldModel(
        config=sigmoid.SigmoidConfig(window=24, linear_dim=24)
    ).fit([traj])
    block = traj[100:124]
    recon = wm.encoder.decode(wm.observe(block))
    rel = np.linalg.norm(recon - block[-1]) / np.linalg.norm(block[-1])
    assert rel < 0.2, f"linear channel round trip too lossy: {rel:.3f}"


def test_save_load_round_trip(tmp_path=None):
    import tempfile

    traj = lorenz()
    wm = sigmoid.SigmoidWorldModel(
        config=sigmoid.SigmoidConfig(window=24, linear_dim=12)
    ).fit([traj])
    block = traj[200:224]
    before = wm.predict_hidden(block, 3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "wm.npz"
        wm.save(path)
        reloaded = sigmoid.SigmoidWorldModel.load(path)
    assert np.allclose(before, reloaded.predict_hidden(block, 3))


# ---- multi-body ----------------------------------------------------------


def test_multibody_recovers_coupling_structure():
    """Body 1 drives body 0; the off-diagonal block must be non-trivial."""
    rng = np.random.default_rng(6)
    t = 400
    b1 = np.cumsum(rng.normal(scale=0.1, size=(t, 4)), axis=0)
    b1 = b1 / (np.abs(b1).max() + 1e-9)
    b0 = np.zeros_like(b1)
    b0[1:] = 0.5 * b0[:-1] + 0.5 * b1[:-1]  # body 0 listens to body 1
    mb = sigmoid.MultiBodyCoupling(ridge=1e-6).fit([b0, b1])
    strength = mb.coupling_strength()
    assert strength[0, 1] > 0.1, f"missed the coupling: {strength}"
    assert mb.rho_ < 1.0
    assert len(mb.step([b0[10], b1[10]])) == 2


# ---- ablation ------------------------------------------------------------


def test_topology_channel_is_ablatable_at_matched_dim():
    """The ablation must build and must match dimensions, so the test is fair."""
    traj = lorenz()
    cfg = sigmoid.SigmoidConfig(window=24, linear_dim=12, hilbert_degree=16, n_radii=6)
    full = sigmoid.SigmoidWorldModel(config=cfg).fit([traj])
    flat_cfg = sigmoid.SigmoidConfig(
        **{**vars(cfg), "use_topology": False, "linear_dim": full.state_dim}
    )
    flat = sigmoid.SigmoidWorldModel(config=flat_cfg).fit([traj])
    assert flat.state_dim == full.state_dim
    assert flat.encoder.topo_dim == 0


def test_spatial_cloud_and_absolute_radii_recover_component_count():
    """Two entity clusters at a fixed separation: psi must see the merge.

    This locks in the two structural fixes the S2-Rips corpus forced. Encoding
    a set-structured state with the temporal cloud, or with the diameter
    divided out, both destroy the component count -- and both did so silently,
    producing a psi that looked fine and carried nothing.
    """
    rng = np.random.default_rng(11)
    n, frames = 8, 300
    obs, truth = [], []
    for t in range(frames):
        gap = 4.0 if (t // 30) % 2 == 0 else 0.15  # far apart, then merged
        a = rng.normal(scale=0.05, size=(n, 3))
        b = rng.normal(scale=0.05, size=(n, 3)) + np.array([gap, 0.0, 0.0])
        obs.append(np.vstack([a, b]).reshape(-1))
        truth.append(2 if gap > 1.0 else 1)
    obs = np.asarray(obs)
    truth = np.asarray(truth[23:])

    cfg = sigmoid.SigmoidConfig(
        window=24, linear_dim=8, hilbert_degree=16, n_radii=6,
        entity_dim=3, n_abs_radii=8,
    )
    wm = sigmoid.SigmoidWorldModel(config=cfg).fit([obs])
    Z = wm.encoder.encode_trajectory(obs)
    psi = Z[:, : wm.encoder.topo_dim]

    # a single psi feature should separate the two regimes almost perfectly
    best = max(
        abs(np.corrcoef(psi[:, j], truth)[0, 1])
        for j in range(psi.shape[1])
        if psi[:, j].std() > 1e-9
    )
    assert best > 0.9, f"no psi feature tracks the component count (best |r|={best:.3f})"


def test_entity_dim_requires_divisible_width():
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(80, 10))  # 10 is not a multiple of 3
    cfg = sigmoid.SigmoidConfig(window=24, linear_dim=4, entity_dim=3)
    try:
        sigmoid.SigmoidWorldModel(config=cfg).fit([obs])
    except ValueError as exc:
        assert "entity_dim" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a ragged entity width")


# ---- control --------------------------------------------------------------


def _swarm_data(episodes=14, length=70, n=3, seed=0):
    """Damped double integrator with 2D entities -- a miniature robot team."""
    rng = np.random.default_rng(seed)
    obs, acts = [], []
    for _ in range(episodes):
        pos = rng.normal(scale=2.0, size=(n, 2))
        vel = np.zeros_like(pos)
        o, a = [], []
        drive = rng.normal(scale=0.5, size=(n, 2))
        for _ in range(length):
            o.append(pos.reshape(-1).copy())
            drive = 0.85 * drive + 0.3 * rng.normal(scale=0.5, size=(n, 2))
            action = np.clip(drive, -1, 1)
            a.append(action.reshape(-1).copy())
            vel = 0.85 * vel + 0.25 * action
            pos = pos + 0.25 * vel
        obs.append(np.asarray(o))
        acts.append(np.asarray(a))
    return obs, acts


def _swarm_world(**overrides):
    obs, acts = _swarm_data()
    config = sigmoid.SigmoidConfig(
        window=10, linear_dim=8, hilbert_degree=12, n_radii=4,
        entity_dim=2, abs_radii=(0.5, 1.0, 2.0), n_abs_radii=0,
        action_dim=6, **overrides,
    )
    return sigmoid.SigmoidWorldModel(config=config).fit(obs, actions=acts), obs


def test_mpc_plans_a_usable_action():
    from sigmoid.control import TopologicalMPC, target_cost

    world, obs = _swarm_world()
    mpc = TopologicalMPC(
        world=world, action_low=np.full(6, -1.0), action_high=np.full(6, 1.0),
        horizon=4, samples=64, elites=6, iterations=2, gate_limit=5.0,
    )
    goal = np.array([2.0, 2.0, -2.0, 2.0, 0.0, -2.0])
    plan = mpc.plan(obs[0][:10], target_cost(world, goal))
    assert plan.feasible, (
        f"planner found nothing feasible: rejected {plan.rejected}/{plan.considered}"
    )
    assert plan.actions.shape == (4, 6)
    # the gate should reject a lot but not everything -- both extremes are bugs
    assert 0 < plan.rejected < plan.considered
    assert np.all(plan.action >= -1.0) and np.all(plan.action <= 1.0)
    assert np.isfinite(plan.cost)


def test_mpc_beats_random_actions_on_its_own_objective():
    """A planner that does not beat random sampling is not planning."""
    from sigmoid.control import TopologicalMPC, target_cost

    world, obs = _swarm_world()
    goal = np.array([2.0, 2.0, -2.0, 2.0, 0.0, -2.0])
    cost = target_cost(world, goal)
    mpc = TopologicalMPC(
        world=world, action_low=np.full(6, -1.0), action_high=np.full(6, 1.0),
        horizon=4, samples=48, elites=6, iterations=3, gate_limit=5.0,
    )
    plan = mpc.plan(obs[0][:10], cost)

    rng = np.random.default_rng(0)
    z0 = world.observe(obs[0][:10])
    random_costs = []
    for _ in range(48):
        actions = rng.uniform(-1, 1, size=(4, 6))
        states, _ = mpc._rollout(z0, actions)
        random_costs.append(cost(states))
    assert plan.cost < np.median(random_costs), (
        f"CEM cost {plan.cost:.4f} not better than random median "
        f"{np.median(random_costs):.4f}"
    )


def test_mpc_refuses_when_every_plan_leaves_the_manifold():
    """A world model asked to control a robot must be able to say it doesn't know."""
    from sigmoid.control import TopologicalMPC, target_cost

    world, obs = _swarm_world()
    mpc = TopologicalMPC(
        world=world, action_low=np.full(6, -1.0), action_high=np.full(6, 1.0),
        horizon=4, samples=16, elites=4, iterations=1,
        gate_limit=1e-9,  # nothing can satisfy this
    )
    plan = mpc.plan(obs[0][:10], target_cost(world, np.zeros(6)))
    assert not plan.feasible
    assert plan.rejected == plan.considered
    assert not np.isfinite(plan.cost)


def test_beta0_cost_refuses_a_radius_it_cannot_evaluate():
    """Silently reading the wrong radius is the bug that cost this project most."""
    from sigmoid.control import beta0_cost

    world, _ = _swarm_world()
    beta0_cost(world, target_components=3, radius=1.0)  # calibrated, fine
    try:
        beta0_cost(world, target_components=3, radius=40.0)
    except ValueError as exc:
        assert "no calibrated radius" in str(exc)
    else:
        raise AssertionError("expected a refusal for an uncalibrated radius")


def test_mpc_requires_an_action_conditioned_model():
    from sigmoid.control import TopologicalMPC

    traj = lorenz()
    world = sigmoid.SigmoidWorldModel(
        config=sigmoid.SigmoidConfig(window=24, linear_dim=8)
    ).fit([traj])
    try:
        TopologicalMPC(world=world, action_low=[-1.0], action_high=[1.0])
    except ValueError as exc:
        assert "action" in str(exc)
    else:
        raise AssertionError("expected a refusal without action conditioning")


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

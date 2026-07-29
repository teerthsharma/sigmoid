"""Checks for unsupervised (cloud, scale) selection.

`python tests/test_autoselect.py` or pytest.

Every system here is built so the *right* answer is known by construction, and
each test asserts the ground truth as well as the selection -- a selector test
whose system does not actually have the property it claims proves nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sigmoid
from sigmoid.state import TopoEncoder, TopoEncoderConfig, select_cloud

# ---- systems --------------------------------------------------------------


def two_cluster_entities(frames=300, n=8, period=30, seed=11):
    """Two entity clusters whose separation alternates between far and merged.

    The component count is a property of the *entity* cloud at one instant. The
    temporal cloud sees 24 consecutive draws from whichever regime is current,
    which is a gaussian blob either way, so it is close to blind.
    """
    rng = np.random.default_rng(seed)
    obs, truth = [], []
    for t in range(frames):
        gap = 4.0 if (t // period) % 2 == 0 else 0.15
        a = rng.normal(scale=0.05, size=(n, 3))
        b = rng.normal(scale=0.05, size=(n, 3)) + np.array([gap, 0.0, 0.0])
        obs.append(np.vstack([a, b]).reshape(-1))
        truth.append(2 if gap > 1.0 else 1)
    return np.asarray(obs), np.asarray(truth)


def breathing_scale_entities(frames=400, period=40, seed=3):
    """A fixed entity shape, uniformly rescaled by a smoothly breathing factor.

    The normalized barcode is scale invariant -- deliberately, and unit-tested
    in test_sigmoid.test_h0_is_scale_invariant -- so it is *identical* on every
    frame here up to jitter. The number of clusters at a fixed absolute radius
    is not. This is the second silent bug in its purest form: the only channel
    that can see the regime is one that kept the scale.
    """
    rng = np.random.default_rng(seed)
    shape = np.repeat(np.arange(4.0)[:, None], 4, axis=0) * np.array([1.0, 0.0, 0.0])
    shape = shape + rng.normal(scale=0.02, size=(16, 3))  # fixed, not per frame
    obs, truth, s = [], [], 1.0
    for t in range(frames):
        target = 0.3 if (t // period) % 2 else 2.0
        s = float(np.clip(0.9 * s + 0.1 * target + 0.02 * rng.normal(), 0.1, 4.0))
        obs.append((shape * s + rng.normal(scale=0.005, size=(16, 3))).reshape(-1))
        truth.append(4 if s > 0.5 else 1)  # components at absolute radius 0.5
    return np.asarray(obs), np.asarray(truth)


def lorenz(n=600, dt=0.01, dim=24, seed=0):
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


BASE = TopoEncoderConfig(window=24, linear_dim=8, hilbert_degree=16, n_radii=6)


def best_correlation(config, obs, truth):
    """Strongest |r| between any psi feature and the labels. The ground truth."""
    enc = TopoEncoder(config=config).fit(obs)
    psi = enc.encode_trajectory(obs)[:, : enc.topo_dim]
    labels = truth[config.window - 1 :]
    return max(
        (
            abs(np.corrcoef(psi[:, j], labels)[0, 1])
            for j in range(psi.shape[1])
            if psi[:, j].std() > 1e-9
        ),
        default=0.0,
    )


def pick(selection, entity_dim, absolute_radii):
    return next(
        c
        for c in selection.candidates
        if c.entity_dim == entity_dim and c.absolute_radii is absolute_radii
    )


# ---- the two bugs ---------------------------------------------------------


def test_selects_the_spatial_cloud_when_the_temporal_one_is_blind():
    """Bug 1: the barcode over the wrong point cloud. Measured 0.386 -> 0.764."""
    obs, truth = two_cluster_entities()

    spatial = best_correlation(BASE.__class__(**{**vars(BASE), "entity_dim": 3}), obs, truth)
    temporal = best_correlation(BASE, obs, truth)
    assert spatial > 0.9, f"the spatial cloud should see the merge, got |r|={spatial:.3f}"
    assert temporal < 0.6, f"the temporal cloud should be blind, got |r|={temporal:.3f}"

    sel = select_cloud(BASE, [obs], entity_dims=(0, 3))
    assert sel.chosen.entity_dim == 3, (
        f"picked the temporal cloud: "
        f"{[(c.entity_dim, c.absolute_radii, round(c.score, 3)) for c in sel.candidates]}"
    )
    assert sel.chosen.score > pick(sel, 0, True).score
    assert sel.chosen.score > pick(sel, 0, False).score


def test_selects_absolute_radii_when_the_scale_is_the_only_signal():
    """Bug 2: dividing out the diameter. Measured 0.764 -> 0.855."""
    obs, truth = breathing_scale_entities()
    spatial = {**vars(BASE), "entity_dim": 3}

    kept = best_correlation(TopoEncoderConfig(**{**spatial, "n_abs_radii": 8}), obs, truth)
    divided = best_correlation(TopoEncoderConfig(**{**spatial, "n_abs_radii": 0}), obs, truth)
    assert kept > divided + 0.05, (
        f"the system must reward keeping the scale: {kept:.3f} vs {divided:.3f}"
    )

    sel = select_cloud(BASE, [obs], entity_dims=(0, 3))
    assert sel.chosen.entity_dim == 3 and sel.chosen.absolute_radii, (
        f"picked (entity_dim={sel.chosen.entity_dim}, "
        f"abs={sel.chosen.absolute_radii}) on a fixed-radius system"
    )
    assert sel.chosen.score > pick(sel, 3, False).score


def test_the_overlap_floor_is_what_makes_the_criterion_work():
    """Raw self-predictability alone picks the wrong cloud, and here is why.

    Consecutive temporal windows share W-1 of W points, so psi_{t+1} is almost
    psi_t whatever the data says. Subtracting the shuffled-order score removes
    that free ride; without it the temporal candidate wins this system despite
    tracking nothing (|r| 0.22 against 0.87).
    """
    obs, _ = breathing_scale_entities()
    sel = select_cloud(BASE, [obs], entity_dims=(0, 3))
    temporal = max((pick(sel, 0, a) for a in (True, False)), key=lambda c: c.self_r2)
    spatial = pick(sel, 3, True)

    assert temporal.self_r2 > spatial.self_r2, (
        "expected the overlap confound to be present in the raw score: "
        f"temporal {temporal.self_r2:.3f} vs spatial {spatial.self_r2:.3f}"
    )
    assert temporal.surrogate_r2 > 0.4, (
        f"a shuffled temporal window should still self-predict, got {temporal.surrogate_r2:.3f}"
    )
    assert spatial.surrogate_r2 < 0.2, (
        f"a shuffled spatial cloud should not, got {spatial.surrogate_r2:.3f}"
    )
    assert spatial.score > temporal.score


# ---- behaviour ------------------------------------------------------------


def test_no_entity_widths_offered_means_the_temporal_cloud_survives():
    """The default candidate list must not invent an entity width."""
    sel = select_cloud(BASE, [lorenz()])
    assert all(c.entity_dim == 0 for c in sel.candidates)
    assert sel.chosen.entity_dim == 0


def test_a_degenerate_psi_is_never_selected():
    """A constant psi predicts itself perfectly and carries nothing."""
    obs = np.tile(np.arange(12.0), (200, 1)) + np.linspace(0, 1, 200)[:, None]
    sel = select_cloud(BASE, [obs], entity_dims=(0, 3))
    frozen = [c for c in sel.candidates if c.n_informative == 0]
    assert all(c.score == float("-inf") for c in frozen)
    assert sel.chosen.n_informative > 0 or not any(
        c.n_informative for c in sel.candidates
    )


def test_ragged_entity_width_is_refused_loudly():
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(120, 10))  # 10 is not a multiple of 3
    try:
        select_cloud(BASE, [obs], entity_dims=(0, 3))
    except ValueError as exc:
        assert "entity_dim" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a ragged entity width")


# ---- engine integration ---------------------------------------------------


def test_auto_is_opt_in_and_the_default_path_is_untouched():
    obs, _ = two_cluster_entities()
    cfg = sigmoid.SigmoidConfig(window=24, linear_dim=8, hilbert_degree=16, n_radii=6)
    assert cfg.cloud == "fixed" and cfg.cloud_candidates == (0,)
    wm = sigmoid.SigmoidWorldModel(config=cfg).fit([obs])
    assert wm.cloud_selection_ is None
    assert wm.encoder.config.entity_dim == 0
    assert wm.summary()["cloud_selection"] is None


def test_auto_fixes_the_encoder_and_records_the_choice():
    obs, truth = two_cluster_entities()
    cfg = sigmoid.SigmoidConfig(
        window=24, linear_dim=8, hilbert_degree=16, n_radii=6,
        cloud="auto", cloud_candidates=(3,),
    )
    wm = sigmoid.SigmoidWorldModel(config=cfg).fit([obs])

    assert wm.encoder.config.entity_dim == 3, "auto did not fix the encoder"
    assert wm.cloud_selection_ is not None
    summary = wm.summary()
    assert summary["entity_dim"] == 3
    # reported off the encoder, so it must agree with what was actually built
    assert summary["absolute_radii"] is wm.cloud_selection_.chosen.absolute_radii
    assert summary["cloud_selection"]["score"] > summary["cloud_selection"]["runner_up"]

    # the fixed encoder must be the one the selection scored: psi has to see it
    psi = wm.encoder.encode_trajectory(obs)[:, : wm.encoder.topo_dim]
    labels = truth[cfg.window - 1 :]
    best = max(
        abs(np.corrcoef(psi[:, j], labels)[0, 1])
        for j in range(psi.shape[1])
        if psi[:, j].std() > 1e-9
    )
    assert best > 0.9, f"selected encoder still cannot see the merge (|r|={best:.3f})"


def test_unknown_cloud_mode_is_refused():
    obs, _ = two_cluster_entities(frames=80)
    cfg = sigmoid.SigmoidConfig(window=24, linear_dim=8, cloud="spatial")
    try:
        sigmoid.SigmoidWorldModel(config=cfg).fit([obs])
    except ValueError as exc:
        assert "cloud" in str(exc)
    else:
        raise AssertionError('expected a refusal for cloud="spatial"')


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

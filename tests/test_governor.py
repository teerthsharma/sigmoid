"""Tests for the topological governor.

Two things need proving. First that no vital can return a NaN, because a NaN
survives every comparison as False and silently routes the policy down its
fallback branch -- that is the whole failure mode the governor exists to prevent,
so it must not have it. Second that the selection criterion survives the traps
this repo already fell into: the shuffled-order overlap artifact, and a bogus
entity width that merely divides the observation dimension.

Runs standalone with a PASS/FAIL harness and a nonzero exit, like the other test
modules here. Everything is seeded.
"""

from __future__ import annotations

import sys

import numpy as np

from sigmoid.engine import SigmoidConfig
from sigmoid.governor import (
    ENTITY_DELTA_THRESHOLD,
    VITAL_NAMES,
    SigmoidGovernor,
    Vitals,
    VitalsExtractor,
    _ar1,
    _lorenz,
    plateau_diagnostics,
)
from sigmoid.mujoco.corpus import CorpusConfig, make_corpus


def _s2(per_cap=9, n_bridge=3, radius=0.95, frames=300):
    obs, islands = make_corpus(
        CorpusConfig(per_cap=per_cap, n_bridge=n_bridge, radius=radius, frames=frames)
    )
    return obs, islands


# ---- vitals: finiteness on degenerate input -------------------------------
#
# Each case below is a real shape the calibration path can be handed, and each
# one breaks a different vital in the source: n=1 has no pairwise distance, d=1
# has no covariance, identical rows make every log ratio 0/0, NaNs propagate
# through cdist to the whole matrix, and n < k indexes past the end of the
# neighbour list.

DEGENERATE = {
    "n=1": np.zeros((1, 5)),
    "n=1,d=1": np.zeros((1, 1)),
    "d=1": np.arange(60.0).reshape(-1, 1),
    "identical_rows": np.ones((40, 4)),
    "two_identical_rows": np.ones((2, 3)),
    "all_nan": np.full((20, 3), np.nan),
    "some_nan": np.where(np.arange(200).reshape(50, 4) % 7 == 0, np.nan, 1.0),
    "inf": np.where(np.arange(200).reshape(50, 4) % 11 == 0, np.inf, 0.5),
    "n_below_k": np.arange(9.0).reshape(3, 3),
    "n_below_k_id": np.arange(4.0).reshape(2, 2),
    "huge_offset": np.array([[1e6, 1e6], [1e6 + 1e-6, 1e6], [1e6, 1e6 + 1e-6]]),
    "constant_column": np.column_stack(
        [np.arange(30.0), np.ones(30), np.zeros(30)]
    ),
}


def test_every_vital_is_finite_on_degenerate_input():
    """A vital returning NaN silently poisons every comparison downstream."""
    ex = VitalsExtractor(seed=0)
    for name, x in DEGENERATE.items():
        v = ex(x)
        vec = v.vector()
        assert vec.shape == (len(VITAL_NAMES),), f"{name}: wrong vitals length"
        bad = [n for n, val in zip(VITAL_NAMES, vec) if not np.isfinite(val)]
        assert not bad, f"{name} produced non-finite {bad}"


def test_vitals_are_bounded_not_merely_finite():
    """1/(mean log ratio) on coincident points is ~1e9: finite, still poison."""
    ex = VitalsExtractor(seed=0)
    for name, x in DEGENERATE.items():
        v = ex(x)
        ambient = x.shape[1]
        assert 0.0 <= v.intrinsic_dim <= ambient + 1e-9, (
            f"{name}: intrinsic_dim {v.intrinsic_dim} outside [0, {ambient}]"
        )
        assert 0.0 <= v.spectral_gap <= 1.0 + 1e-9, f"{name}: gap {v.spectral_gap}"
        assert 0.0 <= v.nan_fraction <= 1.0, f"{name}: nan_fraction {v.nan_fraction}"


def test_vitals_reject_shapes_that_are_not_matrices():
    ex = VitalsExtractor()
    for bad in (np.zeros((0, 3)), np.zeros((2, 0)), np.zeros((2, 2, 2))):
        try:
            ex(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted shape {bad.shape}")


def test_vitals_are_reproducible_for_a_fixed_seed():
    """The source draws from the global RNG, so two runs can disagree about the
    config to use. Deterministic replay is a hard requirement here."""
    x = np.random.default_rng(0).normal(size=(2000, 8))  # forces the subsample
    a, b = VitalsExtractor(seed=3)(x), VitalsExtractor(seed=3)(x)
    assert np.array_equal(a.vector(), b.vector())
    c = VitalsExtractor(seed=4)(x)
    assert not np.array_equal(a.vector(), c.vector()), "seed had no effect"


def test_intrinsic_dim_recovers_a_known_manifold_dimension():
    """The source feeds squared distances to the Levina-Bickel MLE and never
    un-squares, so log(r_k^2/r_j^2) = 2 log(r_k/r_j) halves the estimate."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 5)) @ rng.normal(size=(5, 20))
    got = VitalsExtractor()(x).intrinsic_dim
    assert 4.0 < got < 6.5, f"5-dim manifold in R^20 read as {got:.2f}"


def test_nan_fraction_is_measured_before_scrubbing():
    x = np.ones((10, 10))
    x[:, 0] = np.nan
    v = VitalsExtractor()(x)
    assert abs(v.nan_fraction - 0.1) < 1e-12, v.nan_fraction
    assert np.all(np.isfinite(v.vector()))


def test_spectral_gap_sees_clusters_the_source_k_cannot():
    """Fixed k=15 builds a complete graph on 16 points or fewer, whose Laplacian
    gap is 1.0 whatever the geometry. Capping k at n//3 fixes it."""
    rng = np.random.default_rng(0)
    a = rng.normal(scale=0.02, size=(7, 3))
    b = rng.normal(scale=0.02, size=(7, 3)) + 10.0
    gap = VitalsExtractor()(np.vstack([a, b])).spectral_gap
    assert gap < 0.2, f"two tight clusters of 7 read gap {gap:.4f}"
    one = VitalsExtractor()(rng.normal(size=(60, 3))).spectral_gap
    assert one > gap, f"one blob {one:.4f} should not look more clustered than {gap:.4f}"


# ---- the plateau ---------------------------------------------------------


def test_plateau_finds_a_planted_two_scale_gap():
    """Two tight clusters far apart: beta_0 is 2 across a wide band of radius."""
    rng = np.random.default_rng(0)
    cloud = np.vstack(
        [rng.normal(scale=0.01, size=(15, 3)), rng.normal(scale=0.01, size=(15, 3)) + 5.0]
    )
    rep = plateau_diagnostics(cloud, window=len(cloud))
    assert rep.n_frames == 1
    assert rep.components == 2.0, f"beta_0 on the band is {rep.components}"
    assert rep.excess > 3.0, f"a planted gap should stand out, got {rep.excess:.2f}"
    # The band must span the whole gap between the two planted scales -- the
    # within-cluster 0.01 and the 5.0 separation are 2.5 decades apart, and every
    # radius in between reads the same two components. Checked unit-free, because
    # this is the temporal path and median/MAD standardization rescales the cloud:
    # the raw 8.66 separation reads 2.33 after it, so an absolute assertion here
    # would be testing the standardizer, not the plateau.
    assert rep.width_decades > 2.0, f"band only {rep.width_decades:.2f} dec wide"
    # And the same cloud read as one spatial frame is *not* standardized, so there
    # the band top must reach the true separation of 5*sqrt(3).
    raw = plateau_diagnostics(cloud.reshape(1, -1), entity_dim=3)
    assert raw.components == 2.0
    assert raw.centre * 10 ** (raw.width_decades / 2) > 5.0, (
        f"raw band top {raw.centre * 10 ** (raw.width_decades / 2):.3f} < 8.66"
    )


def test_plateau_refuses_a_cloud_with_no_merges():
    """One point, or coincident points, has no band for a radius to sit in."""
    for bad in (np.zeros((1, 3)), np.ones((6, 3))):
        rep = plateau_diagnostics(bad, window=len(bad))
        assert rep.n_frames == 0 and not rep.readable, f"{bad.shape} claimed a plateau"


def test_plateau_rejects_a_non_divisor_entity_width():
    try:
        plateau_diagnostics(np.zeros((10, 7)), entity_dim=3)
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("accepted a width that does not divide the observation")


def test_plateau_excess_grows_with_point_count_on_pure_noise():
    """The reason a matched control is required at all: the widest of g log-gaps
    grows like the harmonic number whatever the geometry does."""
    rng = np.random.default_rng(0)
    small = plateau_diagnostics(rng.uniform(-1, 1, (8, 6)), window=8).excess
    large = plateau_diagnostics(rng.uniform(-1, 1, (48, 6)), window=48).excess
    assert large > small + 1.0, (
        f"excess {small:.2f} at m=8 vs {large:.2f} at m=48 -- if these were "
        f"comparable the control would be unnecessary"
    )


def test_plateau_standardizes_temporal_but_not_spatial_clouds():
    """`TopoEncoder._topo_window` reads spatial clouds raw and temporal clouds
    standardized, so a diagnostic describing the wrong one describes nothing.
    A single huge column must not set the temporal scale."""
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(120, 6))
    loud = traj.copy()
    loud[:, 0] *= 5000.0
    a = plateau_diagnostics(traj, window=32)
    b = plateau_diagnostics(loud, window=32)
    assert abs(a.centre - b.centre) / a.centre < 0.5, (
        f"one loud column moved the temporal band from {a.centre:.4f} to {b.centre:.4f}"
    )


def test_plateau_reports_drift_across_frames():
    """A threshold that dilates is still readable -- README s9 -- so drift is
    reported, and it has to be nonzero when the scale actually moves."""
    rng = np.random.default_rng(0)
    base = np.vstack(
        [rng.normal(scale=0.01, size=(10, 3)), rng.normal(scale=0.01, size=(10, 3)) + 1.0]
    )
    traj = np.stack([base.reshape(-1) * (10.0 ** (t / 40.0)) for t in range(120)])
    rep = plateau_diagnostics(traj, entity_dim=3)
    assert rep.drift_decades > 2.0, f"a 3-decade dilation read drift {rep.drift_decades:.2f}"
    assert rep.components == 2.0, "dilation must not change the partition"


# ---- recommend ------------------------------------------------------------


def test_recommend_picks_the_entity_cloud_on_the_s2_corpus():
    """Ground truth by construction: the island partition is H0 of the entity
    cloud. The temporal encoder scored 0.386 against a 0.487 majority baseline."""
    obs, _ = _s2()
    rec = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 3, 5, 9))
    assert rec.config.entity_dim == 3, (
        f"picked {rec.config.entity_dim}; deltas "
        + " ".join(f"{w}:{v:+.2f}" for w, v in sorted(rec.deltas.items()))
    )
    assert rec.config.n_abs_radii > 0, "a corpus with a plateau must keep absolute scale"
    assert "README s9" in rec.rationale["entity_dim"]


def test_recommend_refuses_the_bogus_width_select_cloud_takes():
    """`select_cloud` documents the miss: on a 24-dim Lorenz lift entity_dim=3
    won on all four seeds tried and the world model rolled out worse (0.404
    against 0.318 nrmse at k=4). Every candidate here divides 24."""
    for seed in (0, 1, 2):
        rec = SigmoidGovernor().recommend(
            [_lorenz(seed=seed)], entity_dim_candidates=(0, 3, 4, 6, 8)
        )
        assert rec.config.entity_dim == 0, (
            f"seed {seed} took a bogus width {rec.config.entity_dim}; deltas "
            + " ".join(f"{w}:{v:+.2f}" for w, v in sorted(rec.deltas.items()))
        )


def test_recommend_survives_the_overlap_artifact():
    """The trap `select_cloud`'s shuffled floor exists for: overlapping windows
    make a temporal psi nearly its own successor and score 0.56-0.82 on pure
    artifact. The plateau delta is computed from merge heights of the *same*
    cloud in both arms, so an AR(1) sequence and white noise -- which have very
    different self-predictability and no entity structure at all -- must both
    come out temporal."""
    for name, traj in (
        ("ar1", _ar1()),
        ("white", np.random.default_rng(1).normal(size=(400, 20))),
    ):
        rec = SigmoidGovernor().recommend([traj], entity_dim_candidates=(0, 2, 4, 5, 10))
        assert rec.config.entity_dim == 0, f"{name} invented entity structure"
        assert max(rec.deltas.values()) < ENTITY_DELTA_THRESHOLD


def test_recommend_never_returns_an_unoffered_width():
    """Candidates come from the caller. 0 is the documented default cloud and
    needs no width, so it is the only legal fallback."""
    obs, _ = _s2()
    rec = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0,))
    assert rec.config.entity_dim == 0
    rec = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 7))
    assert rec.config.entity_dim in (0, 7)


def test_recommend_skips_a_width_that_does_not_divide_the_observation():
    obs, _ = _s2()  # 63 columns; 4 is not a divisor
    rec = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 3, 4))
    assert rec.config.entity_dim == 3
    assert rec.plateaus[4].n_frames == 0, "a non-divisor cannot have a plateau"


def test_recommended_config_actually_fits():
    """A recommendation that raises on fit is worse than no recommendation."""
    import sigmoid

    obs, _ = _s2(frames=200)
    short = [obs[:80], obs[80:160]]
    rec = SigmoidGovernor().recommend(short, entity_dim_candidates=(0, 3))
    assert rec.config.window + 1 <= min(t.shape[0] for t in short)
    wm = sigmoid.SigmoidWorldModel(config=rec.config).fit(short)
    assert wm.fitted
    z = wm.observe(short[0][: rec.config.window])
    assert np.all(np.isfinite(z))


def test_linear_dim_never_exceeds_the_pca_rank():
    """Above min(n, d) `TopoEncoder.fit` zero-pads components_, so those state
    dimensions are exactly 0.0 and cost budget for nothing."""
    traj = np.random.default_rng(0).normal(size=(300, 6))
    rec = SigmoidGovernor().recommend([traj])
    assert rec.config.linear_dim <= 6, rec.config.linear_dim
    assert "zero-pads" in rec.rationale["linear_dim"]


def test_hilbert_degree_never_exceeds_the_available_bars():
    """A cloud of m points has m-1 finite H0 deaths; a higher degree guarantees
    empty bins, which become constant psi columns the operator carries as dead
    weight."""
    obs, _ = _s2(per_cap=3, n_bridge=1, frames=200)  # 7 nodes -> 7-point cloud
    rec = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 3))
    if rec.config.entity_dim == 3:
        assert rec.config.hilbert_degree <= 21 // 3 - 1 + 1e-9


def test_every_config_field_that_changed_has_a_rationale():
    obs, _ = _s2()
    rec = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 3))
    default = SigmoidConfig()
    changed = [
        f
        for f in vars(default)
        if getattr(rec.config, f) != getattr(default, f)
    ]
    for f in changed:
        assert f in rec.rationale, f"{f} changed with no rationale"
    for f in ("entity_dim", "n_abs_radii", "window", "linear_dim", "block_diagonal"):
        assert rec.rationale[f], f"{f} has an empty rationale"


def test_confidence_is_low_near_the_decision_boundary():
    """A delta just over the threshold is a coin flip and must say so."""
    obs, _ = _s2()
    strong = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 3))
    weak = SigmoidGovernor().recommend([_lorenz(seed=0)], entity_dim_candidates=(0, 4))
    assert 0.0 <= weak.confidence <= 1.0 and 0.0 <= strong.confidence <= 1.0
    assert weak.confidence < strong.confidence, (
        f"borderline {weak.confidence:.2f} not below clear-cut {strong.confidence:.2f}"
    )


def test_confidence_falls_with_nan_input():
    obs, _ = _s2()
    clean = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 3))
    dirty_obs = obs.copy()
    dirty_obs[::20, 0] = np.nan
    dirty = SigmoidGovernor().recommend([dirty_obs], entity_dim_candidates=(0, 3))
    assert dirty.vitals.nan_fraction > 0.0
    assert dirty.confidence < clean.confidence
    assert "_input" in dirty.rationale


def test_recommend_rejects_bad_inputs():
    gov = SigmoidGovernor()
    for args, kwargs in (
        (([],), {}),
        (([np.zeros((5, 5, 5))],), {}),
        (([np.zeros((10, 4)), np.zeros((10, 5))],), {}),
        (([np.zeros((10, 4))],), {"entity_dim_candidates": ()}),
    ):
        try:
            gov.recommend(*args, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args} {kwargs}")


# ---- history --------------------------------------------------------------


def test_update_requires_a_recommendation_first():
    try:
        SigmoidGovernor().update(SigmoidConfig(), 1.0)
    except RuntimeError as exc:
        assert "recommend" in str(exc)
    else:
        raise AssertionError("update accepted a score with no vitals to attach it to")


def test_history_recalls_a_config_that_scored_well():
    obs, _ = _s2()
    gov = SigmoidGovernor()
    marker = SigmoidConfig(window=17, linear_dim=5, entity_dim=3)
    for traj, score in ((obs, 0.9), (_lorenz(), 0.1), (_ar1(), 0.2)):
        rec = gov.recommend([traj], entity_dim_candidates=(0, 3))
        gov.update(marker if traj is obs else rec.config, score)
    again = gov.recommend([obs], entity_dim_candidates=(0, 3))
    assert again.recalled, "identical vitals did not hit the history"
    assert again.config == marker
    assert "_recall" in again.rationale


def test_history_ignores_a_config_that_scored_badly():
    obs, _ = _s2()
    gov = SigmoidGovernor()
    marker = SigmoidConfig(window=17, linear_dim=5, entity_dim=3)
    for traj, cfg, score in (
        (obs, marker, 0.1),
        (_lorenz(), SigmoidConfig(), 0.9),
        (_ar1(), SigmoidConfig(), 0.8),
    ):
        gov.recommend([traj], entity_dim_candidates=(0, 3))
        gov.update(cfg, score)
    again = gov.recommend([obs], entity_dim_candidates=(0, 3))
    assert not again.recalled, "a below-median score was recalled anyway"
    assert again.config != marker


def test_history_ignores_distant_vitals():
    obs, _ = _s2()
    gov = SigmoidGovernor()
    for traj in (_lorenz(), _ar1()):
        rec = gov.recommend([traj], entity_dim_candidates=(0, 3))
        gov.update(rec.config, 1.0)
    far = gov.recommend([obs], entity_dim_candidates=(0, 3))
    assert not far.recalled, "unrelated vitals matched inside the recall radius"


def test_a_single_history_entry_gives_no_advice():
    """One point defines neither a per-vital spread nor a median."""
    obs, _ = _s2()
    gov = SigmoidGovernor()
    rec = gov.recommend([obs], entity_dim_candidates=(0, 3))
    gov.update(rec.config, 1.0)
    again = gov.recommend([obs], entity_dim_candidates=(0, 3))
    assert not again.recalled


def test_vitals_vector_round_trips_the_names():
    v = Vitals(1.0, 2.0, 3.0, 4.0, 0.5, 0.0)
    assert list(v.vector()) == [1.0, 2.0, 3.0, 4.0, 0.5, 0.0]
    assert len(VITAL_NAMES) == 6, "six survivors; seven were dropped as dead weight"
    for name in VITAL_NAMES:
        assert name in v.table()


def test_recommendation_table_names_its_evidence():
    obs, _ = _s2()
    text = SigmoidGovernor().recommend([obs], entity_dim_candidates=(0, 3)).table()
    for token in ("confidence", "entity_dim", "spectral_gap", "n_abs_radii"):
        assert token in text, f"{token} missing from the rationale table"


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

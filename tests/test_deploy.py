"""Runnable checks for deployment. `python tests/test_deploy.py` or pytest.

Everything here is filesystem-local and network-free: checkpoints are written into a
`tempfile.TemporaryDirectory` and the only external process ever invoked is `nvidia-smi`,
which is probed for existence first and whose absence is itself one of the cases under
test.

The tests that matter most are the OTA ones. A corrupted staged checkpoint must be
rejected *and* leave the live weights loadable, and a crash between stage and commit must
recover -- those two are the difference between an update mechanism and a way to brick a
robot from a distance.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmoid.deploy import (
    HardwareProbe,
    HardwareReport,
    Precision,
    PrecisionPolicy,
    PrecisionState,
    QuantizedOperator,
    StagingError,
    WeightStore,
    battery_percent,
    export_model,
    gpu_temperature_c,
    quantization_table,
    quantize_operator,
    row_relative_error,
    sha256_file,
)
from sigmoid.engine import SigmoidConfig, SigmoidWorldModel
from sigmoid.operator import CouplingOperator
from sigmoid.realtime import Tier

# --------------------------------------------------------------------------
# fixtures, built rather than imported so a test failure is local
# --------------------------------------------------------------------------


def make_operator(rho=0.7, *, heterogeneous=False, action_dim=0, d=24, seed=0):
    """A fitted operator with a known spectral norm.

    Scaled to `rho` before fitting rather than clipped with `rho_max` after: clipping
    overwrites the dynamics and inflates `step_rmse_` (measured 4.57e-01 under a 0.95 clip
    against a 1e-2 noise floor), which would make every quantization verdict pass.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    A *= rho / float(np.linalg.svd(A, compute_uv=False)[0])
    if heterogeneous:
        A *= np.logspace(-3, 1, d).reshape(-1, 1)
    Z = rng.normal(size=(1200, d))
    Znext = Z @ A.T + 0.01 * rng.normal(size=(1200, d))
    if action_dim:
        acts = rng.normal(size=(1200, action_dim))
        Znext = Znext + acts @ rng.normal(size=(action_dim, d))
        return CouplingOperator(ridge=1e-6).fit(Z[:-1], Znext[:-1], acts[:-1])
    return CouplingOperator(ridge=1e-6).fit(Z[:-1], Znext[:-1])


def make_engine(seed=0, n=400):
    """A small but real fitted world model, cheap enough to write in a loop."""
    rng = np.random.default_rng(seed)
    traj = np.cumsum(rng.normal(size=(n, 3)) * 0.1, axis=0)
    return SigmoidWorldModel(config=SigmoidConfig(window=16, linear_dim=6)).fit([traj])


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------


def test_quantized_operator_drops_into_step():
    op = make_operator()
    z = np.zeros(op.state_dim)
    for prec in ("fp32", "fp16", "int8"):
        q = quantize_operator(op, prec)
        out = q.step(z)
        assert out.shape == (op.state_dim,), f"{prec} step returned {out.shape}"
        batch = q.step(np.zeros((5, op.state_dim)))
        assert batch.shape == (5, op.state_dim), f"{prec} lost the batch axis"
        # The clone keeps every fitted attribute, so a certificate still issues.
        assert q.operator.certificate(8).rho == op.rho_


def test_quantized_operator_handles_actions():
    op = make_operator(action_dim=3)
    q = quantize_operator(op, "int8")
    z, a = np.zeros(op.state_dim), np.ones(3)
    assert q.step(z, a).shape == (op.state_dim,)
    assert q.rollout(z, 4, np.ones((4, 3))).shape == (4, op.state_dim)


def test_int8_is_a_real_compression():
    op = make_operator()
    assert op.W_ is not None
    q = quantize_operator(op, "int8")
    assert q.q_ is not None and q.q_.dtype == np.int8
    assert q.scale_ is not None and q.scale_.shape == (op.state_dim,)
    ratio = op.W_.nbytes / q.nbytes
    # 8 bytes/weight -> 1 byte/weight plus one float32 scale per row, so the ratio is
    # 8L/(L+4) in the lift dimension and never 8x. Asserting a bare "7.1x" would have
    # been asserting one operator's shape: measured 6.48x at L=17 and 7.40x at L=49.
    expected = 8.0 * op.lift_dim / (op.lift_dim + 4)
    assert abs(ratio - expected) < 0.01, f"int8 compression {ratio:.3f}x, expected {expected:.3f}x"
    assert ratio > 6.0, "per-row scale overhead has eaten the compression"
    assert quantize_operator(op, "fp16").nbytes * 2 == quantize_operator(op, "fp32").nbytes


def test_zero_rows_do_not_become_nan():
    """The engine's block-diagonal fit leaves whole rows at exactly zero.

    31 of the 52 rows of the Lorenz operator are zero. Per-row scaling divides by
    `max|row|`, so an unguarded implementation writes NaN into 60% of the matrix -- and a
    NaN operator produces a NaN action, which a robot will happily execute.
    """
    op = make_operator()
    assert op.W_ is not None
    op.W_[::2] = 0.0  # every other row dead, like a block-diagonal quadrant
    q = quantize_operator(op, "int8")
    assert np.all(np.isfinite(q.W_)), "zero row produced a non-finite weight"
    assert np.all(q.W_[::2] == 0.0), "zero rows must dequantize back to exactly zero"
    assert np.all(np.isfinite(q.step(np.ones(op.state_dim))))


def test_per_row_scaling_beats_global_on_heterogeneous_rows():
    """The measurement that justifies per-row being the default.

    On a well-conditioned operator the difference is ~1.3x and nothing is destroyed. On
    one whose rows span four decades -- robot state in raw units, mm beside radians
    beside newtons -- a global scale quantizes half the rows to all-zero.
    """
    flat = make_operator()
    hetero = make_operator(heterogeneous=True)
    destroyed = {}
    for name, op in (("flat", flat), ("hetero", hetero)):
        assert op.W_ is not None
        e_row = row_relative_error(quantize_operator(op, "int8", per_row=True).W_, op.W_)
        e_glob = row_relative_error(quantize_operator(op, "int8", per_row=False).W_, op.W_)
        assert e_row.max() <= e_glob.max() + 1e-12, f"{name}: per-row did worse than global"
        destroyed[name] = (int((e_row > 0.5).sum()), int((e_glob > 0.5).sum()))
    assert destroyed["flat"] == (0, 0), (
        f"a global scale destroyed rows on a well-conditioned operator: {destroyed['flat']} "
        "-- the honest finding is that it does not, so re-measure"
    )
    assert destroyed["hetero"][0] == 0, "per-row destroyed rows on the heterogeneous operator"
    assert destroyed["hetero"][1] > 0, (
        "a global scale destroyed no rows even at 4-decade spread, which removes the "
        "entire justification for per-row scaling"
    )


def test_quantization_table_measures_every_precision():
    op = make_operator()
    rows = quantization_table(op, horizon=16, n_probe=64, reps=200)
    by = {r.precision: r for r in rows}
    assert set(by) == {"fp64 (reference)", "fp32", "fp16", "int8", "int8 (global)"}
    # Error must be monotone in precision: a wider dtype cannot be less accurate.
    assert by["fp32"].step_rmse < by["fp16"].step_rmse < by["int8"].step_rmse
    assert by["int8"].step_rmse < by["int8 (global)"].step_rmse
    assert by["fp32"].bytes > by["fp16"].bytes > by["int8"].bytes
    assert all(r.latency_us > 0 for r in rows)


def test_low_precision_is_not_faster_here():
    """The measurement the whole precision policy rests on.

    numpy has no fp16 or int8 GEMM, so lower precision does not buy latency: fp16 matvec
    measured 3.6 us against fp64's 0.8 us. If this ever stops being true the policy's
    latency reasoning needs rewriting, so it is asserted rather than assumed.
    """
    op = make_operator(d=32)
    rows = quantization_table(op, horizon=4, n_probe=16, reps=1200)
    by = {r.precision: r for r in rows}
    fp64 = by["fp64 (reference)"].latency_us
    assert by["fp16"].latency_us > fp64, (
        f"fp16 ({by['fp16'].latency_us:.2f} us) beat fp64 ({fp64:.2f} us) -- numpy grew an "
        "fp16 kernel and PrecisionPolicy's latency reasoning is now wrong"
    )


def test_int8_can_dominate_the_error_and_the_verdict_says_so():
    """If int8 destroys the rollout, that is the finding -- but check which claim is stable.

    On a heterogeneous *expansive* operator (rho 3-4 here, because scaling rows over four
    decades moves the spectrum) the 16-step rollout error is dominated by rho^16
    amplification, not by the precision. Measured across d = 16/24/32/48 the per-row int8
    verdict came out unusable / usable / unusable / usable -- it tracks how the probe
    perturbation happens to land in the expanding subspace, not anything about int8.

    So this test asserts only what held at every d: **global int8 was unusable in all four
    cases** (7.6e0, 1.4e-2, 2.9e-2, 2.1e-2 against a ~1e-2 model residual) and per-row was
    strictly better every time. Pinning the per-row verdict to one d would have been
    picking the shape that produced the headline.
    """
    unusable_seen = False
    for d in (16, 32):
        op = make_operator(heterogeneous=True, d=d)
        by = {r.precision: r for r in quantization_table(op, horizon=16, n_probe=32, reps=100)}
        assert by["fp32"].usable, f"d={d}: fp32 should never be the dominant error source"
        assert not by["int8 (global)"].usable, (
            f"d={d}: global int8 came in usable on a 4-decade operator, which removes the "
            "justification for per-row scaling"
        )
        assert by["int8 (global)"].rollout_rmse > by["int8"].rollout_rmse, f"d={d}: per-row lost"
        unusable_seen |= not by["int8"].usable
    assert unusable_seen, (
        "int8 never once dominated the error, so QuantReport.usable has never been "
        "observed to fire negatively and is decoration"
    )


def test_int8_is_usable_on_a_well_conditioned_operator():
    """The other half of the verdict: usable when the added error is below the fit's."""
    op = make_operator()
    rows = quantization_table(op, horizon=16, n_probe=32, reps=200)
    by = {r.precision: r for r in rows}
    for prec in ("fp32", "fp16", "int8"):
        assert by[prec].usable, (
            f"{prec} reported unusable on a well-conditioned contraction: "
            f"{by[prec].rollout_rmse:.3e} vs model {by[prec].model_step_rmse:.3e}"
        )


def test_quantize_rejects_an_unfitted_operator_and_a_bad_name():
    try:
        quantize_operator(CouplingOperator(), "int8")
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("quantized an unfitted operator")
    try:
        quantize_operator(make_operator(), "int4")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a precision that does not exist")


def test_quantized_operator_is_not_the_source_operator():
    """Quantizing must not mutate the operator it was given."""
    op = make_operator()
    assert op.W_ is not None
    before = op.W_.copy()
    quantize_operator(op, "int8")
    quantize_operator(op, "fp16")
    assert np.array_equal(op.W_, before), "quantize_operator mutated its input"


# --------------------------------------------------------------------------
# precision policy
# --------------------------------------------------------------------------


def test_oscillating_temperature_does_not_oscillate_precision():
    """Flapping between int8 and fp32 mid-motion is its own failure mode.

    Two absorption mechanisms, tested separately because they cover different signals:
    the Schmitt band absorbs any oscillation that stays inside 70-80 C at any period,
    and the dwell counter starves any oscillation whose half-period is below `dwell`.
    """
    # 1. inside the band: no period can move the latch
    for period in (1, 2, 5, 20):
        pol = PrecisionPolicy(dwell=3)
        for i in range(80):
            temp = 78.0 if (i // period) % 2 else 72.0
            pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=temp))
        assert pol.switches == 0, (
            f"policy flapped {pol.switches}x on a 72-78 C wave (period {period}) that "
            "never crosses the 80/70 band"
        )

    # 2. crossing the band, but faster than the dwell window
    for period in (1, 2):
        pol = PrecisionPolicy(dwell=3)
        seen = set()
        for i in range(80):
            temp = 95.0 if (i // period) % 2 else 60.0
            seen.add(pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=temp)))
        assert pol.switches == 0, f"dwell=3 admitted a half-period-{period} oscillation"
        assert len(seen) == 1, f"precision changed without a recorded switch: {seen}"

    # 3. a longer oscillation is tracked -- by design -- and a wider dwell absorbs it.
    #    This is the measured ceiling: dwell is in control ticks, so a deployment sets it
    #    from its loop rate. At 50 Hz, dwell=10 is 200 ms, well under any real thermal
    #    transient.
    fast = PrecisionPolicy(dwell=3)
    slow = PrecisionPolicy(dwell=10)
    for i in range(80):
        temp = 95.0 if (i // 8) % 2 else 60.0
        state = PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=temp)
        fast.select(state)
        slow.select(state)
    assert fast.switches > 0, "half-period 8 should clear a dwell of 3"
    assert slow.switches == 0, f"dwell=10 should absorb half-period 8, got {slow.switches}"


def test_policy_still_reacts_to_a_sustained_excursion():
    """Hysteresis must not become paralysis."""
    pol = PrecisionPolicy(dwell=3)
    for _ in range(8):
        pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=95.0))
    assert pol.current is Precision.INT8, "never derated under sustained 95 C"
    assert pol.derated and "thermal" in pol.reason
    for _ in range(8):
        pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=65.0))
    assert pol.current is Precision.FP32, "never recovered after cooling"
    assert pol.switches == 2, f"expected exactly 2 adopted switches, got {pol.switches}"


def test_recovery_needs_the_lower_threshold_not_just_the_upper():
    """Cooling to 75 C is not cooling: the latch releases below cool_c, not below hot_c."""
    pol = PrecisionPolicy(dwell=2, hot_c=80.0, cool_c=70.0)
    for _ in range(6):
        pol.select(PrecisionState(tier=Tier.FREE_SPACE, gpu_temp_c=85.0))
    assert pol.derated
    for _ in range(6):
        pol.select(PrecisionState(tier=Tier.FREE_SPACE, gpu_temp_c=75.0))
    assert pol.derated, "latch released inside the hysteresis band"
    for _ in range(6):
        pol.select(PrecisionState(tier=Tier.FREE_SPACE, gpu_temp_c=68.0))
    assert not pol.derated, "latch never released below cool_c"


def test_low_battery_derates():
    pol = PrecisionPolicy(dwell=2)
    for _ in range(5):
        pol.select(PrecisionState(tier=Tier.CONTACT_RICH, battery_pct=8.0))
    assert pol.current is Precision.INT8 and "battery" in pol.reason


def test_tier_drives_precision_by_accuracy_not_deadline():
    """Contact-rich gets fp32 and free-space gets int8 -- the inverse of the obvious map.

    Precision cannot buy latency here (see test_low_precision_is_not_faster_here), so
    mapping "tight deadline -> low precision" would hand the 0.5 mm-tolerance task the
    noisiest weights in exchange for nothing.
    """
    pol = PrecisionPolicy(dwell=1)
    got = {}
    for tier in Tier:
        for _ in range(3):
            got[tier] = pol.select(PrecisionState(tier=tier, gpu_temp_c=50.0))
    assert got[Tier.CONTACT_RICH] is Precision.FP32
    assert got[Tier.FREE_SPACE] is Precision.INT8
    assert got[Tier.REACTIVE_GRASP] is Precision.FP16


def test_latency_overrun_vetoes_upgrades_only():
    """An over-budget loop must not be handed a wider dtype; it also must not be told
    that a narrower one will save it."""
    pol = PrecisionPolicy(dwell=1, current=Precision.INT8)
    over = Tier.CONTACT_RICH.budget_ms * 3
    for _ in range(5):
        pol.select(PrecisionState(tier=Tier.CONTACT_RICH, latency_ms=over))
    assert pol.current is Precision.INT8, "latency veto let an upgrade through"
    assert "latency" in pol.reason
    # Downgrades are still allowed while over budget: thermal pressure must win.
    for _ in range(5):
        pol.select(PrecisionState(tier=Tier.CONTACT_RICH, gpu_temp_c=95.0, latency_ms=over))
    assert pol.current is Precision.INT8


def test_policy_rejects_an_inverted_hysteresis_band():
    try:
        PrecisionPolicy(hot_c=70.0, cool_c=80.0)
    except ValueError as exc:
        assert "hysteresis" in str(exc)
    else:
        raise AssertionError("accepted cool_c above hot_c, which is no hysteresis at all")


def test_gpu_temperature_reports_its_source():
    """Whichever path this host takes, the source must be labelled, never guessed."""
    temp, source = gpu_temperature_c(max_age_s=0.0)
    assert source in {"nvidia-smi", "stub"}
    if source == "nvidia-smi":
        assert temp is not None and 0.0 < temp < 130.0, f"implausible temperature {temp}"
    else:
        assert temp is None, "the stub path must report None, not a made-up number"
    # forced degradation must not hand back a cached real reading
    assert gpu_temperature_c(allow_smi=False) == (None, "stub")


def test_battery_percent_is_a_percentage_or_none():
    pct = battery_percent()
    assert pct is None or 0.0 <= pct <= 100.0, f"battery reported {pct}"


# --------------------------------------------------------------------------
# hardware probe
# --------------------------------------------------------------------------


def test_probe_reports_this_host():
    r = HardwareProbe.probe()
    assert isinstance(r, HardwareReport)
    assert r.cpu_logical >= 1 and r.cpu_gflops > 0.0
    assert r.ram_gb > 0.0, "could not determine RAM by any stdlib route"
    assert r.step_us > 0.0
    assert r.device_class in {"workstation", "high-end-edge", "constrained-edge", "sbc"}
    assert "TFLOP/s" in r.tops_tier or "not benchmarked" in r.tops_tier
    assert r.table().count("\n") > 10
    assert len(r.workloads()) >= 5


def test_probe_never_claims_tensorrt_or_tvm():
    """Not installed here, so never probed and never reported present."""
    r = HardwareProbe.probe()
    assert r.has_tensorrt is False and r.has_tvm is False
    trt = next(row for row in r.workloads() if "TensorRT" in row[0])
    assert trt[1] is False and "not probed" in trt[2]


def test_probe_survives_the_gpu_being_absent():
    """The no-GPU path, forced on a host that has one."""
    r = HardwareProbe.probe(gpu=False)
    assert r.gpu_name is None and r.vram_mb is None and r.compute_capability is None
    assert r.gpu_temp_c is None and r.temp_source == "stub"
    assert r.gpu_tflops is None and r.has_torch_cuda is False
    assert r.device_class in {"constrained-edge", "sbc"}
    assert "cpu fp32" in r.tops_tier, "a GPU-less host must not report a GPU-derived tier"
    r.table()  # must render without a GPU
    # CPU-only workloads must still be judged, not skipped
    assert any(ok for _, ok, _ in r.workloads()), "no workload survived the GPU going away"
    assert not any(
        ok for name, ok, _ in r.workloads() if "triton" in name
    ), "triton attention claimed runnable with no CUDA device"


def test_workload_verdicts_come_from_the_measured_step_time():
    r = HardwareProbe.probe(gpu=False)
    rows = {name: (ok, why) for name, ok, why in r.workloads()}
    basis = rows["operator.step, d=64"][1]
    # The verdict must cite the number it was derived from, not a remembered constant.
    assert f"{r.step_us:.2f}" in basis, f"step time {r.step_us:.2f} us not cited in {basis!r}"
    assert "safety-stop budget" in basis


# --------------------------------------------------------------------------
# OTA weight store -- the security boundary
# --------------------------------------------------------------------------


def test_stage_commit_round_trip_loads():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ckpt = make_engine(seed=1).save(root / "v1.npz")
        store = WeightStore(root / "store")
        digest = store.stage(ckpt, expected_sha256=sha256_file(ckpt))
        assert store.staged.is_file()
        store.commit()
        assert store.verify() and store.manifest()["sha256"] == digest
        wm = store.load()
        assert wm.fitted and wm.operator.W_ is not None


def test_a_corrupted_staged_file_is_rejected_and_the_old_weights_still_load():
    """The single most important test in this file.

    A truncated or tampered checkpoint must be refused, and refusing it must not disturb
    the weights the robot is currently running. Both halves are checked, twice: once
    where the digest catches it and once where the digest is absent and the structural
    check has to.
    """
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        good = make_engine(seed=1).save(root / "v1.npz")
        newer = make_engine(seed=2).save(root / "v2.npz")
        store = WeightStore(root / "store")
        store.stage(good, expected_sha256=sha256_file(good))
        store.commit()
        live = store.manifest()["sha256"]
        baseline = store.load().operator.W_
        assert baseline is not None

        # (a) tampered bytes, correct expected digest -> digest mismatch
        tampered = root / "tampered.npz"
        blob = bytearray(newer.read_bytes())
        blob[len(blob) // 2] ^= 0xFF
        tampered.write_bytes(blob)
        try:
            store.stage(tampered, expected_sha256=sha256_file(newer))
        except StagingError as exc:
            assert "mismatch" in str(exc)
        else:
            raise AssertionError("a tampered checkpoint passed the digest check")

        # (b) truncated, no expected digest -> the structural check must catch it
        truncated = root / "truncated.npz"
        truncated.write_bytes(newer.read_bytes()[:-64])
        try:
            store.stage(truncated)
        except StagingError as exc:
            assert "npz" in str(exc) or "missing" in str(exc)
        else:
            raise AssertionError("a truncated checkpoint was staged")

        # (c) a valid npz that is not a checkpoint at all
        wrong = root / "wrong.npz"
        np.savez(wrong, something_else=np.zeros(4))
        try:
            store.stage(wrong)
        except StagingError as exc:
            assert "missing required arrays" in str(exc)
        else:
            raise AssertionError("an npz without the checkpoint keys was staged")

        # every rejection above must have left the live weights untouched
        assert store.manifest()["sha256"] == live, "manifest changed while rejecting"
        assert store.verify(), "live weights failed verification after a rejection"
        assert np.array_equal(store.load().operator.W_, baseline), (
            "the previous weights no longer load after a bad candidate was rejected"
        )


def test_commit_refuses_a_staged_file_that_changed_after_staging():
    """TOCTOU: the digest is re-checked at commit, not trusted from stage time."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        good = make_engine(seed=1).save(root / "v1.npz")
        newer = make_engine(seed=2).save(root / "v2.npz")
        store = WeightStore(root / "store")
        store.stage(good, expected_sha256=sha256_file(good))
        store.commit()
        live = store.manifest()["sha256"]

        store.stage(newer, expected_sha256=sha256_file(newer))
        # something rewrites the staged file between stage and commit
        blob = bytearray(store.staged.read_bytes())
        blob[-32] ^= 0xFF
        store.staged.write_bytes(blob)
        try:
            store.commit()
        except StagingError as exc:
            assert "changed after staging" in str(exc)
        else:
            raise AssertionError("committed a staged file that had been swapped underneath")
        assert store.manifest()["sha256"] == live and store.verify()
        assert not store.staged.is_file(), "a poisoned candidate was left on disk"


def test_load_refuses_when_current_does_not_match_its_manifest():
    """Hash verification happens *before* np.load, which is where pickle would execute."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ckpt = make_engine(seed=1).save(root / "v1.npz")
        store = WeightStore(root / "store")
        store.stage(ckpt, expected_sha256=sha256_file(ckpt))
        store.commit()
        assert store.verify()

        # bit rot, or someone editing current.npz directly to bypass staging
        blob = bytearray(store.current.read_bytes())
        blob[len(blob) // 2] ^= 0xFF
        store.current.write_bytes(blob)
        assert not store.verify()
        try:
            store.load()
        except StagingError as exc:
            assert "refusing to load" in str(exc)
        else:
            raise AssertionError("loaded weights that did not match the manifest digest")


def test_crash_between_stage_and_commit_recovers():
    """A fresh process must find the old weights live and the candidate still staged."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        good = make_engine(seed=1).save(root / "v1.npz")
        newer = make_engine(seed=2).save(root / "v2.npz")
        store = WeightStore(root / "store")
        store.stage(good, expected_sha256=sha256_file(good))
        store.commit()
        v1 = store.manifest()["sha256"]

        store.stage(newer, expected_sha256=sha256_file(newer))
        del store  # the process dies here, after staging and before committing

        recovered = WeightStore(root / "store")
        assert recovered.verify(), "live weights did not verify after a mid-update crash"
        assert recovered.manifest()["sha256"] == v1, "a half-applied update became live"
        assert recovered.load().fitted, "old weights unloadable after a mid-update crash"
        assert recovered.staged.is_file(), "the staged candidate was lost"
        # and the interrupted update can simply be finished
        recovered.commit()
        assert recovered.manifest()["sha256"] == sha256_file(newer) and recovered.verify()


def test_interrupted_writes_leave_only_reapable_temp_files():
    """A half-written temp file must never be reachable by the name anything loads."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ckpt = make_engine(seed=1).save(root / "v1.npz")
        store = WeightStore(root / "store")
        store.stage(ckpt, expected_sha256=sha256_file(ckpt))
        store.commit()
        live = store.manifest()["sha256"]

        # simulate a crash mid-write: a partial temp beside every real file
        for name in ("current.npz", "staged.npz", "manifest.json"):
            (store.root / f"{name}.tmp").write_bytes(b"half a weight file")
        assert store.verify(), "a stray temp file changed what current.npz resolves to"

        reopened = WeightStore(root / "store")
        assert not list(reopened.root.glob("*.tmp")), "temp files survived a reopen"
        assert reopened.verify() and reopened.manifest()["sha256"] == live


def test_rollback_restores_the_previous_version():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        v1 = make_engine(seed=1).save(root / "v1.npz")
        v2 = make_engine(seed=2).save(root / "v2.npz")
        store = WeightStore(root / "store")
        store.stage(v1, expected_sha256=sha256_file(v1))
        store.commit()
        d1 = store.manifest()["sha256"]
        w1 = store.load().operator.W_

        store.stage(v2, expected_sha256=sha256_file(v2))
        store.commit()
        assert store.manifest()["sha256"] == sha256_file(v2)

        store.rollback()
        assert store.manifest()["sha256"] == d1 and store.verify()
        assert np.array_equal(store.load().operator.W_, w1)
        store.rollback()  # idempotent: previous is copied, never consumed
        assert store.verify() and store.manifest()["sha256"] == d1


def test_store_refuses_the_empty_cases():
    with tempfile.TemporaryDirectory() as d:
        store = WeightStore(Path(d) / "store")
        for call, expect in ((store.commit, "nothing staged"), (store.rollback, "no previous")):
            try:
                call()
            except StagingError as exc:
                assert expect in str(exc)
            else:
                raise AssertionError(f"{call.__name__} succeeded with an empty store")
        assert not store.verify(), "an empty store claimed to verify"
        try:
            store.stage(Path(d) / "does-not-exist.npz")
        except StagingError as exc:
            assert "no such checkpoint" in str(exc)
        else:
            raise AssertionError("staged a file that does not exist")


def test_discard_drops_the_candidate_and_leaves_the_live_weights():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        v1 = make_engine(seed=1).save(root / "v1.npz")
        v2 = make_engine(seed=2).save(root / "v2.npz")
        store = WeightStore(root / "store")
        store.stage(v1, expected_sha256=sha256_file(v1))
        store.commit()
        store.stage(v2, expected_sha256=sha256_file(v2))
        store.discard()
        assert not store.staged.is_file() and not store.staged_path.is_file()
        assert store.verify() and store.manifest()["sha256"] == sha256_file(v1)


def test_the_swap_is_atomic_under_a_concurrent_reader():
    """Proof, not assertion: never a torn read while `current.npz` is being replaced.

    Measured while writing this: 100 swaps under a reader loop produced 83 distinct
    *whole* payloads and zero torn or short reads. The Windows wrinkle is separate --
    `os.replace` onto an open handle raises PermissionError there, which is why both
    sides retry.
    """
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = WeightStore(root / "store")

        # Cheap stand-in payloads, every key a plain array: the swap machinery does not
        # care what is inside, and fitting 12 engines would make this test minutes long.
        # All four required keys are uniform arrays so the reader can tell a whole file
        # from a torn one by shape alone -- seeding with a real checkpoint instead put a
        # 0-d object array under "operator" and the reader called it torn.
        blobs = []
        for i in range(12):
            p = root / f"gen{i}.npz"
            np.savez(
                p,
                config=np.zeros(1),
                encoder=np.zeros(1),
                gate=np.zeros(1),
                operator=np.full(64, float(i)),
            )
            blobs.append((p, sha256_file(p)))

        # Publish the first generation before the reader starts, so "no file yet" and
        # "torn file" cannot be confused.
        store.stage(blobs[0][0], expected_sha256=blobs[0][1])
        store.commit()

        errors: list[str] = []
        whole: set[float] = set()
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    with np.load(store.current, allow_pickle=True) as data:
                        arr = data["operator"]
                        # A torn file shows up as a short array or inconsistent values,
                        # both of which this catches.
                        if arr.shape != (64,) or len(set(arr.tolist())) != 1:
                            errors.append(f"torn read: shape {arr.shape}")
                        else:
                            whole.add(float(arr[0]))
                except (PermissionError, FileNotFoundError):
                    pass  # Windows: the swap is in flight. Not a torn read.
                except Exception as exc:  # noqa: BLE001 - anything else IS the failure
                    errors.append(f"{type(exc).__name__}: {exc}")
                time.sleep(0.0005)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for path, digest in blobs[1:]:
                store.stage(path, expected_sha256=digest)
                store.commit()
                assert store.verify(), "manifest did not match right after a commit"
        finally:
            stop.set()
            t.join(timeout=5.0)

        assert not errors, f"reader saw a partial checkpoint: {errors[:3]}"
        assert len(whole) >= 2, f"reader only ever observed {len(whole)} version(s)"


def test_manifest_is_json_and_records_what_is_live():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ckpt = make_engine(seed=1).save(root / "v1.npz")
        store = WeightStore(root / "store")
        store.stage(ckpt, expected_sha256=sha256_file(ckpt))
        store.commit()
        payload = json.loads(store.manifest_path.read_text())
        assert payload["sha256"] == sha256_file(store.current)
        assert payload["bytes"] == store.current.stat().st_size
        assert payload["committed_at"] > 0


def test_load_docstring_documents_the_allow_pickle_risk():
    """SECURITY.md calls allow_pickle the most important line in the document.

    OTA is the untrusted-input path that sharpens it, so the risk must be stated at the
    method that takes it -- and stated as integrity-not-safety, because a correctly
    hashed hostile checkpoint still executes.
    """
    doc = WeightStore.load.__doc__ or ""
    assert "allow_pickle" in doc, "the pickle risk is not documented where it is taken"
    assert "integrity" in doc and "safety" in doc, (
        "the docstring must distinguish integrity from safety, or a reader will think "
        "the hash check makes an untrusted checkpoint safe to load"
    )


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def test_npz_export_writes_a_verifiable_manifest():
    with tempfile.TemporaryDirectory() as d:
        wm = make_engine()
        res = export_model(wm, Path(d) / "model", "npz")
        assert res.fmt == "npz" and res.path.is_file()
        assert res.manifest_path is not None
        manifest = json.loads(res.manifest_path.read_text())
        assert manifest["sha256"] == sha256_file(res.path) == res.sha256
        assert manifest["state_dim"] == wm.operator.state_dim
        assert manifest["rho"] == wm.operator.rho_
        assert "allow_pickle" in manifest["load_warning"]
        # the export and the OTA path must be the same artifact
        store = WeightStore(Path(d) / "store")
        store.stage(res.path, expected_sha256=manifest["sha256"])
        store.commit()
        assert store.load().fitted


def test_npz_export_reports_quantization_without_substituting_it():
    """A lossy checkpoint must never ship under the same name as a lossless one."""
    with tempfile.TemporaryDirectory() as d:
        wm = make_engine()
        res = export_model(wm, Path(d) / "model", "npz", precision="int8")
        manifest = json.loads(res.manifest_path.read_text())
        assert manifest["weights_precision"] == "fp64", "an int8 preview replaced the weights"
        preview = manifest["quantization_preview"]
        assert preview["precision"] == "int8" and preview["compression"] > 6.0
        restored = SigmoidWorldModel.load(res.path)
        assert restored.operator.W_ is not None
        assert np.array_equal(restored.operator.W_, wm.operator.W_)


def test_tensorrt_and_tvm_are_refused_not_stubbed():
    with tempfile.TemporaryDirectory() as d:
        wm = make_engine()
        for fmt in ("tensorrt", "trt", "tvm"):
            try:
                export_model(wm, Path(d) / "model", fmt)
            except NotImplementedError as exc:
                assert "not installed" in str(exc)
            else:
                raise AssertionError(f"{fmt} reported a successful export")
        for name in ("tensorrt", "tvm"):
            assert name not in sys.modules, f"{name} was imported despite being unavailable"


def test_unknown_format_is_a_value_error():
    with tempfile.TemporaryDirectory() as d:
        try:
            export_model(make_engine(), Path(d) / "m", "safetensors")
        except ValueError as exc:
            assert "unknown export format" in str(exc)
        else:
            raise AssertionError("accepted a format that is not implemented")


def test_onnx_export_when_torch_is_available():
    """Real on this host (torch and onnx both import); skipped honestly elsewhere."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as d:
        wm = make_engine()
        res = export_model(wm, Path(d) / "model", "onnx")
        assert res.path.suffix == ".onnx" and res.path.stat().st_size > 0
        assert res.sha256 == sha256_file(res.path)
        manifest = json.loads(res.manifest_path.read_text())
        assert manifest["format"] == "onnx"
        assert str(wm.operator.lift_dim) in manifest["input"]


def test_nothing_was_written_outside_the_temp_dir():
    """The repo must stay clean: everything here writes into a TemporaryDirectory.

    Scoped to the artifact names *this* module produces rather than every `*.npz` in the
    tree, so a leak from an unrelated test file cannot fail this one and send someone
    reading the wrong module.
    """
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "pyproject.toml").is_file():
        return  # this file was copied elsewhere; there is no repo here to keep clean
    names = ("current.npz", "previous.npz", "staged.npz", "staged.json", "manifest.json")
    strays = [n for n in names if (repo / n).exists() or (repo / "tests" / n).exists()]
    strays += [p.name for p in repo.glob("*.manifest.json")]
    assert not strays, f"deploy tests left files in the repo: {strays}"
    assert not (repo / "store").exists(), "a WeightStore root was created in the repo"


# --------------------------------------------------------------------------
# integration: the parts have to compose
# --------------------------------------------------------------------------


def test_policy_choice_feeds_quantize_operator():
    """The two halves must actually fit together: a selected precision must quantize."""
    op = make_operator()
    pol = PrecisionPolicy(dwell=1)
    for tier in Tier:
        prec = pol.select(PrecisionState(tier=tier, gpu_temp_c=95.0, battery_pct=50.0))
        q = quantize_operator(op, prec)
        assert isinstance(q, QuantizedOperator) and q.precision is prec
        assert np.all(np.isfinite(q.step(np.zeros(op.state_dim))))


def test_a_deployed_operator_still_certifies():
    """Quantization must not silently invalidate the contraction certificate."""
    op = make_operator(rho=0.7)
    cert = op.certificate(16)
    q = quantize_operator(op, "int8")
    qcert = q.operator.certificate(16)
    assert qcert.contractive == cert.contractive
    assert qcert.rho == cert.rho, "the certificate's rho drifted from the source operator"
    # rho_ is carried over from the fp64 fit rather than recomputed, which is only
    # honest while quantization does not measurably move the spectrum. Check that.
    assert q.W_ is not None
    A = q.W_[:, : op.state_dim]
    measured = float(np.linalg.svd(A, compute_uv=False)[0])
    assert abs(measured - op.rho_) < 0.01, (
        f"int8 moved the spectral norm from {op.rho_:.4f} to {measured:.4f}; the carried-over "
        "certificate would be describing a different operator"
    )


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

"""The spectral stalk (AGENDA R8). `python tests/test_gate_entropy.py` or pytest.

Four synthetic window populations in standardized coordinates, i.e. what
`TopoEncoder.normalize` hands back:

    normal       a smooth 8-dim latent lifted to 32 dims -- moderate rank
    degenerate   one genuine row repeated across the window -- rank ~1
    isotropic    white noise, per-coordinate variance matched to normal
    scrambled    the normal rows in random order -- rank high, everything else identical

`scrambled` is the one that carries the argument. Every row of it is a genuine
activation off the calibration trajectory, so it matches the real data in mean,
covariance, energy and support; only the *temporal* structure is destroyed. That
is what uniform random tokens do to a transformer -- each activation is a
plausible activation, the sequence of them is not a plausible sequence -- and it
is the case the measured gate scored 0.513 against 0.671 for real prose.

The stand-in encoder below builds both channels out of the window *mean*, which
is the mechanism of the blind spot stated outright: a summary that averages 24
rows cannot see how those rows were arranged, so scrambling them moves the state
nowhere. `isotropic` is the cruder control -- it fires on the old terms too,
because rows drawn off the latent subspace move the mean off-manifold.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmoid.sheaf import SheafGate, effective_rank

WINDOW, DIM, LATENT = 24, 32, 8
CALIB = 2400
"""Calibration windows. Short calibrations estimate a 0.98 quantile off ~60
independent samples and the held-out false-positive rate lands at 20%; that is
an artifact of the fixture, not of the gate, so pay for a long trajectory."""


@lru_cache(maxsize=1)
def _trajectory(seed=0, length=3000):
    """A smooth low-dimensional process lifted into a wider 'hidden' space."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(LATENT, DIM)) / np.sqrt(LATENT)
    x, rows = np.zeros(LATENT), []
    for _ in range(length):
        x = 0.9 * x + rng.normal(scale=0.4, size=LATENT)
        rows.append(x.copy())
    traj = np.asarray(rows) @ B + 0.05 * rng.normal(size=(length, DIM))
    return (traj - traj.mean(0)) / traj.std(0)  # standardized, as the encoder does


@lru_cache(maxsize=1)
def _populations():
    traj = _trajectory()
    rng = np.random.default_rng(100)
    normal = np.stack([traj[i : i + WINDOW] for i in range(len(traj) - WINDOW)])
    idx = rng.integers(0, len(traj), 60)
    return {
        "calibration": normal[:CALIB],
        "held_out": normal[CALIB:],
        # a *genuine* row repeated, so only the structure is abnormal, not the
        # location -- and with jitter, which is what makes centering the singular
        # values fail (see effective_rank).
        "degenerate": np.stack(
            [np.tile(traj[i], (WINDOW, 1)) + 1e-3 * rng.normal(size=(WINDOW, DIM))
             for i in idx]
        ),
        "isotropic": rng.normal(size=(60, WINDOW, DIM)) * traj.std(0),
        "scrambled": np.stack(
            [traj[rng.integers(0, len(traj), WINDOW)] for _ in range(60)]
        ),
    }


_rng = np.random.default_rng(7)
_P, _Q = _rng.normal(size=(DIM, 8)), _rng.normal(size=(DIM, 4))


def _encode(windows):
    """Stand-in for TopoEncoder: psi and u, both linear in the window mean."""
    m = np.asarray(windows, dtype=np.float64).mean(axis=-2)
    return np.concatenate([m @ _Q, m @ _P], axis=-1)


@lru_cache(maxsize=2)
def _gate(with_stalk=True):
    calib = _populations()["calibration"]
    ranks = [effective_rank(w) for w in calib] if with_stalk else None
    return SheafGate(quantile=0.98).fit(_encode(calib), topo_dim=4, effective_ranks=ranks)


def _read(name, with_stalk=True):
    gate, pop = _gate(with_stalk), _populations()[name]
    if not with_stalk:
        return [gate.read(z) for z in _encode(pop)]
    return [gate.read(z, effective_rank(w)) for z, w in zip(_encode(pop), pop)]


def _rate(readings, key=lambda r: r.fire):
    return float(np.mean([bool(key(r)) for r in readings]))


# ---- the statistic --------------------------------------------------------


def test_effective_rank_orders_the_populations():
    """Degenerate << normal < scrambled << isotropic, with no overlap anywhere."""
    pops = _populations()
    r = {k: np.array([effective_rank(w) for w in pops[k]])
         for k in ("held_out", "degenerate", "isotropic", "scrambled")}
    assert r["degenerate"].max() < r["held_out"].min()
    assert r["held_out"].max() < r["scrambled"].min()
    assert r["scrambled"].max() < r["isotropic"].min()


def test_effective_rank_of_a_constant_window_is_one():
    assert abs(effective_rank(np.tile(np.arange(5.0), (10, 1))) - 1.0) < 1e-9


def test_effective_rank_refuses_a_single_row():
    try:
        effective_rank(np.ones((1, 4)))
    except ValueError as exc:
        assert "window" in str(exc)
    else:
        raise AssertionError("expected a refusal for a one-row window")


# ---- the gate -------------------------------------------------------------


def test_gate_is_quiet_on_normal_windows():
    """The stalk must be a detector, not a tax on ordinary states."""
    readings = _read("held_out")
    assert _rate(readings, lambda r: r.rank_score >= 1.0) <= 0.05
    assert _rate(readings) <= 0.10, "the whole gate got noisier"


def test_gate_catches_both_tails():
    """The point of the exercise: one statistic, both tails."""
    for name in ("degenerate", "isotropic", "scrambled"):
        readings = _read(name)
        assert all(r.rank_score >= 1.0 for r in readings), (
            f"{name}: stalk fired on only "
            f"{_rate(readings, lambda r: r.rank_score >= 1.0):.0%} of windows"
        )
        assert _rate(readings) == 1.0

    # `reason` names whichever term scored highest, and on the two populations
    # the old gate could already see (degenerate 2.24, isotropic 6.86 on the
    # two-term score) the sheaf residual sometimes outscores the stalk. What
    # must never happen is the stalk naming the wrong tail.
    low = {r.reason for r in _read("degenerate")}
    high = {r.reason for r in _read("isotropic")} | {r.reason for r in _read("scrambled")}
    assert "rank_collapsed" in low and "rank_inflated" not in low
    assert "rank_inflated" in high and "rank_collapsed" not in high
    # scrambled is invisible to the other two terms, so there the stalk is alone
    assert all(r.reason == "rank_inflated" for r in _read("scrambled"))


def test_scrambled_is_the_blind_spot_the_two_term_gate_missed():
    """Reproduce R8: without the stalk, scrambled scores *below* real windows."""
    scrambled, held_out = _read("scrambled", False), _read("held_out", False)
    lo, hi = np.mean([r.score for r in scrambled]), np.mean([r.score for r in held_out])
    assert lo < hi, (
        f"blind spot not reproduced: scrambled {lo:.3f} vs held-out {hi:.3f} -- "
        "this test is no longer testing anything"
    )
    assert _rate(scrambled) <= 0.05, "the two-term gate already caught it"


# ---- backward compatibility ------------------------------------------------


def test_gate_without_ranks_is_the_old_two_term_gate():
    """Fit and read with no ranks: the stalk must be inert, not merely quiet."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 12))
    gate = SheafGate().fit(Z, topo_dim=4)
    reading = gate.read(Z[0])
    assert gate.rank_center_ is None
    assert reading.rank_score == 0.0
    assert reading.score == max(reading.sheaf_score, reading.manifold_score)
    # a rank offered to an unfitted stalk is ignored, not guessed at
    assert gate.read(Z[0], effective_rank=999.0).score == reading.score


def test_fitted_stalk_ignores_states_it_was_given_no_rank_for():
    gate = _gate()
    w = _populations()["scrambled"][0]
    z = _encode(w)
    assert gate.read(z, effective_rank(w)).fire
    assert gate.read(z).rank_score == 0.0  # no rank supplied -> no opinion


def test_rank_count_mismatch_is_refused():
    rng = np.random.default_rng(2)
    try:
        SheafGate().fit(rng.normal(size=(50, 12)), topo_dim=4, effective_ranks=np.ones(49))
    except ValueError as exc:
        assert "effective_ranks" in str(exc)
    else:
        raise AssertionError("expected a refusal for a ragged rank vector")


def test_state_dict_round_trip_keeps_the_stalk():
    gate = _gate()
    w = _populations()["scrambled"][0]
    z, er = _encode(w), effective_rank(w)
    assert SheafGate.from_state_dict(gate.state_dict()).read(z, er).score == gate.read(z, er).score
    # a gate saved before R8 has no rank keys at all and must still load
    old = {k: v for k, v in gate.state_dict().items() if not k.startswith("rank_")}
    assert SheafGate.from_state_dict(old).read(z, er).rank_score == 0.0


if __name__ == "__main__":
    g = _gate()
    print(f"  calibration centre rank {np.exp(g.rank_center_):.2f}, "
          f"log-deviation threshold {g.rank_threshold_:.3f}\n")
    print(f"  {'population':12s} {'erank':>6s} {'sheaf':>7s} {'manif':>7s} {'rank':>7s} "
          f"{'score':>7s} {'fires':>6s}  {'2-term':>7s} {'2-fires':>7s}  reasons")
    for name in ("held_out", "degenerate", "isotropic", "scrambled"):
        rs, old = _read(name), _read(name, False)
        w = _populations()[name]
        reasons = sorted({r.reason for r in rs})
        print(
            f"  {name:12s} {np.mean([effective_rank(x) for x in w]):6.2f} "
            f"{np.mean([r.sheaf_score for r in rs]):7.2f} "
            f"{np.mean([r.manifold_score for r in rs]):7.2f} "
            f"{np.mean([r.rank_score for r in rs]):7.2f} "
            f"{np.mean([r.score for r in rs]):7.2f} {_rate(rs):6.0%}  "
            f"{np.mean([r.score for r in old]):7.3f} {_rate(old):7.0%}  {','.join(reasons)}"
        )
    print()

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

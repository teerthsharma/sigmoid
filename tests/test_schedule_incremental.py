"""Runnable checks for incremental decode salience.

`python tests/test_schedule_incremental.py` or pytest. The bar is exact
equality with the batch path at every append, not closeness: a decode that
drifts from the batch schedule stops matching the kernel it feeds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmoid.schedule import (
    IncrementalSalience,
    build_topology_block_schedule,
    zero_dim_persistence_salience,
)


def merged_reference_salience(centroids: np.ndarray) -> np.ndarray:
    """The merged kernel's construction, transcribed: all pairs, sorted, union-find.

    This is the parity target (triton-lang/kernels#22), kept here rather than
    imported so the test does not check an implementation against itself.
    """
    n = centroids.shape[0]
    pairs = sorted(
        (float(np.linalg.norm(centroids[i] - centroids[j])), i, j)
        for i in range(n)
        for j in range(i + 1, n)
    )
    parent = list(range(n))
    members = {i: {i} for i in range(n)}
    salience = np.zeros(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for distance, left, right in pairs:
        rl, rr = find(left), find(right)
        if rl == rr:
            continue
        if len(members[rl]) > len(members[rr]):
            rl, rr = rr, rl
        for block in members[rl]:
            salience[block] = distance
        parent[rl] = rr
        members[rr].update(members[rl])
        del members[rl]

    return salience


def adversarial_cases() -> dict[str, np.ndarray]:
    """Inputs where merge heights tie exactly or to within 1e-12.

    Random gaussians never tie, so they cannot tell a correct tie-break from a
    lucky one. These can: each has many distinct minimum spanning trees, and
    picking a different one moves salience by O(1), not by round-off.
    """
    rng = np.random.default_rng(7)
    grid = np.stack(np.meshgrid(np.arange(5.0), np.arange(5.0)), -1).reshape(-1, 2)
    base = rng.normal(size=(6, 3))
    return {
        # every block coincident with two others -> zero-length edges
        "exact duplicates": np.repeat(rng.normal(size=(7, 4)), 3, axis=0),
        # separated by 1e-12, which the Gram-matrix distance identity rounds
        # to exactly 0.0 and the correct one does not
        "near duplicates": np.repeat(base, 3, axis=0)
        + 1e-12 * rng.normal(size=(18, 3)),
        # unit grid: every nearest-neighbour distance is exactly 1.0
        "unit grid": grid,
        # equally spaced chain, all consecutive gaps identical
        "collinear chain": np.stack([np.arange(16.0), np.zeros(16)], 1),
        # equilateral triangle repeated: all three merges at the same height
        "equilateral": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [4.0, 0.0], [5.0, 0.0]]
        ),
        # every block identical: all distances exactly zero
        "single point": np.zeros((9, 3)),
        # tight clusters, so within-cluster heights tie against each other
        "clustered": np.repeat(rng.normal(size=(4, 5)) * 10.0, 5, axis=0)
        + 1e-9 * rng.normal(size=(20, 5)),
    }


# ---- exact equivalence ---------------------------------------------------


def test_append_matches_rebuild_on_random_seeds():
    """Every append, every seed: bit-identical to a from-scratch rebuild."""
    for seed in (0, 1, 2, 3, 17):
        rng = np.random.default_rng(seed)
        blocks = rng.normal(size=(70, 12))
        state = IncrementalSalience(blocks[:5])
        assert np.array_equal(state.salience, zero_dim_persistence_salience(blocks[:5]))
        for step in range(5, 70):
            appended = state.append(blocks[step])
            rebuilt = zero_dim_persistence_salience(blocks[: step + 1])
            assert np.array_equal(appended, rebuilt), (
                f"seed {seed} diverged at {step + 1} blocks, "
                f"max abs diff {np.abs(appended - rebuilt).max():.3e}"
            )


def test_append_matches_rebuild_when_merge_heights_tie():
    """The adversarial cases, where a wrong tie-break diverges by O(1)."""
    for name, blocks in adversarial_cases().items():
        state = IncrementalSalience(blocks[:2])
        for step in range(2, blocks.shape[0]):
            appended = state.append(blocks[step])
            rebuilt = zero_dim_persistence_salience(blocks[: step + 1])
            assert np.array_equal(appended, rebuilt), (
                f"'{name}' diverged at {step + 1} blocks: "
                f"{appended} vs {rebuilt}"
            )


def test_append_matches_rebuild_from_a_single_block():
    """Decode can start from one block, where there is no merge to record."""
    rng = np.random.default_rng(4)
    blocks = rng.normal(size=(12, 3))
    state = IncrementalSalience(blocks[:1])
    assert np.array_equal(state.salience, zero_dim_persistence_salience(blocks[:1]))
    for step in range(1, 12):
        assert np.array_equal(
            state.append(blocks[step]), zero_dim_persistence_salience(blocks[: step + 1])
        )


def test_batch_and_incremental_both_match_the_merged_reference():
    """Kernel parity: neither path may drift from the reference construction."""
    rng = np.random.default_rng(11)
    cases = dict(adversarial_cases())
    cases["random"] = rng.normal(size=(24, 6))
    for name, blocks in cases.items():
        reference = merged_reference_salience(blocks)
        batch = zero_dim_persistence_salience(blocks)
        state = IncrementalSalience(blocks[:2])
        for step in range(2, blocks.shape[0]):
            state.append(blocks[step])
        for label, ours in (("batch", batch), ("incremental", state.salience)):
            assert np.allclose(ours, reference, atol=1e-12), (
                f"'{name}' {label} differs from the merged reference: "
                f"max abs diff {np.abs(ours - reference).max():.3e}"
            )


def test_schedule_matches_the_batch_schedule():
    """The CSR the class hands the kernel is the one a rebuild would hand it."""
    rng = np.random.default_rng(5)
    blocks = rng.normal(size=(33, 9))
    state = IncrementalSalience(blocks[:4])
    for step in range(4, 33):
        state.append(blocks[step])
    for radius, sinks, topk in ((0, 0, 0), (1, 1, 4), (3, 2, 8), (2, 0, 33)):
        ours = state.schedule(radius, sinks, topk)
        # block_size=1 makes each row its own block, so both paths see the
        # same centroids
        theirs = build_topology_block_schedule(blocks, 1, radius, sinks, topk)
        assert np.array_equal(ours[0], theirs[0]), "offsets differ"
        assert np.array_equal(ours[1], theirs[1]), "indices differ"


def test_append_refuses_a_mismatched_or_empty_input():
    rng = np.random.default_rng(6)
    state = IncrementalSalience(rng.normal(size=(3, 4)))
    try:
        state.append(np.zeros(5))
    except ValueError as exc:
        assert "dim" in str(exc)
    else:
        raise AssertionError("expected a refusal for a wrong-width centroid")
    try:
        IncrementalSalience(np.zeros((0, 4)))
    except ValueError as exc:
        assert "at least one block" in str(exc)
    else:
        raise AssertionError("expected a refusal for an empty block set")
    assert len(state) == 3
    state.append(np.zeros(4))
    assert len(state) == 4


# ---- cost ----------------------------------------------------------------


def test_appending_beats_rebuilding():
    """N appends against N rebuilds. Reported, not just asserted."""
    rng = np.random.default_rng(0)
    print()
    speedups = []
    for start, end, dim in ((32, 64, 64), (64, 128, 64), (64, 256, 64)):
        blocks = rng.normal(size=(end, dim))

        state = IncrementalSalience(blocks[:start])
        t0 = time.perf_counter()
        for step in range(start, end):
            state.append(blocks[step])
        incremental = time.perf_counter() - t0

        t0 = time.perf_counter()
        for step in range(start, end):
            zero_dim_persistence_salience(blocks[: step + 1])
        rebuild = time.perf_counter() - t0

        steps = end - start
        speedups.append(rebuild / incremental)
        print(
            f"        {start:>3} -> {end:<4} blocks ({steps} appends): "
            f"incremental {incremental * 1e3:7.1f} ms "
            f"({incremental / steps * 1e3:5.2f} ms/step), "
            f"rebuild {rebuild * 1e3:8.1f} ms "
            f"({rebuild / steps * 1e3:6.2f} ms/step), "
            f"{rebuild / incremental:5.1f}x"
        )
    # loose, because a shared runner is noisy: measured 8.8x-11.7x across runs
    # at 256 blocks, so 3x still fails loudly if the asymptotics regress. The
    # printed ms/step is the number to read, not this bound.
    assert min(speedups) > 3.0, f"expected a real win, got {min(speedups):.1f}x"


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

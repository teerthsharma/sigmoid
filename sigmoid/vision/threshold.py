"""Cancellation-safe threshold selection.

Teerth Sharma's rule, from `computer-vision-basics-in-microsoft-excel`, worktree
`cancellation-safe-topology`. The problem it solves:

**Equal Euler characteristic does not prove equal topology.** A component birth
and a hole birth cancel in `chi = b0 - b1`, so the obvious selector -- scan
thresholds, take the middle of the longest run with the target chi -- can land on
a "plateau" that is not a plateau at all. `euler.demo()` constructs exactly that
false plateau.

So stability is measured in the **intensity domain** instead of by counting scan
rows. Between two consecutive distinct pixel intensities the thresholded mask
*cannot* change, because no pixel value lies between them. The widest such gap is
therefore the most robust place to threshold, and it is a fact about the image
rather than an artifact of the scan step.

Four gates, all required:

1. `gap > 0`                       -- the endpoints are genuinely distinct
2. `s_k < midpoint < s_{k+1}`      -- the float64 midpoint is *representable*
                                      inside the gap. A real gap does not
                                      guarantee this: for adjacent float64
                                      values the midpoint rounds to an endpoint,
                                      and thresholding *at* a pixel value makes
                                      the mask depend on `<=` versus `<`.
3. `chi(midpoint) == target_chi`
4. occupancy within bounds         -- rejects all-black and all-white masks that
                                      trivially satisfy a chi target

No candidate means **no answer**. `certified=False` with a reason, never a
silent fallback to an uncertified threshold -- a robot acting on a guess it
believes is verified is worse than a robot that stops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .euler import euler_characteristic, to_gray

__all__ = ["ThresholdResult", "cancellation_safe_threshold"]


@dataclass(frozen=True)
class ThresholdResult:
    """A threshold and the evidence for it."""

    threshold: float
    gap: float
    chi: int
    occupancy: float
    certified: bool
    reason: str
    n_candidates: int = 0

    def __bool__(self) -> bool:
        return self.certified


def cancellation_safe_threshold(
    image: np.ndarray,
    target_chi: int | None,
    *,
    min_occupancy: float = 0.05,
    max_occupancy: float = 0.95,
    max_candidates: int | None = 4096,
) -> ThresholdResult:
    """Widest intensity gap whose midpoint yields `target_chi`.

    `target_chi=None` means "widest admissible gap, whatever chi it gives". That
    is the honest fallback when no threshold reaches a requested chi: the widest
    intensity gap is still the most stable place to cut, it is still a fact about
    the image, and it beats a median split. On a bimodal image where background
    dominates, the median *is* the background value, so `img <= median` selects
    every pixel and every topological feature collapses to a constant -- measured
    while building `encode.py`, where it silently produced a b1 channel with zero
    variance and a NaN correlation against ground truth.

    `max_candidates` subsamples the gap list for large images. Gaps are examined
    widest-first, so truncation can only discard gaps narrower than ones already
    considered -- it cannot cause a *better* candidate to be missed. Set None to
    disable.
    """
    img = to_gray(image)
    n_pixels = img.size
    if n_pixels == 0:
        return ThresholdResult(0.0, 0.0, 0, 0.0, False, "empty image")
    if not 0.0 <= min_occupancy <= max_occupancy <= 1.0:
        raise ValueError("require 0 <= min_occupancy <= max_occupancy <= 1")

    values = np.sort(img.reshape(-1))
    lo, hi = values[:-1], values[1:]
    gaps = hi - lo
    midpoints = lo + gaps / 2.0  # not (lo+hi)/2: that can overflow-round outward

    # gates 1 and 2, vectorized. The representability test is the subtle one and
    # it is a real filter, not a formality -- see demo().
    admissible = (gaps > 0.0) & (midpoints > lo) & (midpoints < hi)
    idx = np.flatnonzero(admissible)
    if idx.size == 0:
        return ThresholdResult(
            0.0, 0.0, 0, 0.0, False, "no gap with a representable interior midpoint"
        )

    order = idx[np.argsort(-gaps[idx], kind="stable")]  # widest first, stable ties
    if max_candidates is not None:
        order = order[:max_candidates]

    checked = 0
    for k in order:
        t = float(midpoints[k])
        mask = img <= t
        occupancy = float(mask.mean())
        checked += 1
        if not (min_occupancy <= occupancy <= max_occupancy):
            continue
        chi = euler_characteristic(mask)
        if target_chi is not None and chi != target_chi:
            continue
        return ThresholdResult(
            threshold=t,
            gap=float(gaps[k]),
            chi=chi,
            occupancy=occupancy,
            certified=True,
            reason=(
                "widest admissible gap matching target chi"
                if target_chi is not None
                else "widest admissible gap (no chi target requested)"
            ),
            n_candidates=checked,
        )

    return ThresholdResult(
        0.0,
        0.0,
        0,
        0.0,
        False,
        f"NO CANCELLATION-SAFE INTERVAL: no gap of {checked} examined "
        f"gave chi={target_chi} within occupancy bounds",
        n_candidates=checked,
    )


def demo() -> None:
    """The representability gate must bite, and a real plateau must certify."""
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]

    # a disk on a graded background: a wide intensity gap separates figure from
    # ground, so a chi=1 threshold should certify
    img = np.full((n, n), 0.9)
    img[(xx - 32) ** 2 + (yy - 32) ** 2 <= 400] = 0.1
    res = cancellation_safe_threshold(img, target_chi=1)
    assert res.certified, f"clean plateau failed to certify: {res.reason}"
    assert res.chi == 1
    assert 0.1 < res.threshold < 0.9, f"threshold outside the gap: {res.threshold}"
    assert res.gap > 0.5, f"expected the wide figure/ground gap, got {res.gap}"
    print(f"  certified threshold {res.threshold:.4f}  gap {res.gap:.4f}  chi {res.chi}")

    # gate 2 in isolation: adjacent float64 values have no interior midpoint
    a = 0.5
    b = np.nextafter(a, 1.0)
    assert a + (b - a) / 2.0 in (a, b), "expected the midpoint to round to an endpoint"
    tiny = np.array([[a, b], [a, b]])
    out = cancellation_safe_threshold(tiny, target_chi=1, min_occupancy=0.0)
    assert not out.certified, "a gap with no representable interior must not certify"
    assert "representable" in out.reason, out.reason
    print(f"  representability gate rejected: {out.reason}")

    # an unreachable target must refuse rather than return a guess
    flat = np.full((16, 16), 0.5)
    miss = cancellation_safe_threshold(flat, target_chi=7)
    assert not miss.certified and miss.threshold == 0.0
    print(f"  unreachable target refused: {miss.reason[:52]}...")

    # occupancy bounds must reject a mask that only trivially hits the target
    ramp = np.linspace(0.0, 1.0, n * n).reshape(n, n)
    loose = cancellation_safe_threshold(ramp, target_chi=1, min_occupancy=0.0)
    strict = cancellation_safe_threshold(
        ramp, target_chi=1, min_occupancy=0.98, max_occupancy=1.0
    )
    assert loose.certified, "a ramp should certify somewhere with no occupancy floor"
    assert not strict.certified or strict.occupancy >= 0.98, (
        "occupancy floor was not enforced"
    )
    print(f"  occupancy bounds enforced (loose {loose.occupancy:.2f}, "
          f"strict certified={strict.certified})")
    print("demo ok")


if __name__ == "__main__":
    demo()

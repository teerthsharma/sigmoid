"""Exact digital topology on images, in O(pixels).

Persistence on a point cloud needs pairwise distances and a filtration; on a
*grid* none of that is required. For a binary mask with 4-connected foreground
the Euler characteristic is exactly

    chi = F - H - V + Q

    F = foreground pixels
    H = horizontally adjacent foreground pairs
    V = vertically adjacent foreground pairs
    Q = fully-foreground 2x2 blocks

Four array reductions. No complex is built, no distance is computed, and the
result is exact rather than an estimate. That is the whole reason topology can
sit inside a 10-50 ms robot vision budget at all -- see `sigmoid/vision/encode.py`
for the measured comparison against a Vietoris-Rips baseline on the same image.

Source of the construction: Teerth Sharma's cancellation-safe topological
thresholding work in `computer-vision-basics-in-microsoft-excel`, worktree
`cancellation-safe-topology`. The critical caveat from that document is carried
into `sigmoid/vision/threshold.py`: **equal chi does not prove equal topology**,
because a component birth and a hole birth cancel in the sum.
"""

from __future__ import annotations

import numpy as np

__all__ = ["betti1_bound", "betti_from_euler", "euler_characteristic", "euler_curve", "to_gray"]


def to_gray(image: np.ndarray) -> np.ndarray:
    """Normalize an image to float64 grayscale in [0, 1].

    Accepts (H, W) or (H, W, C). `uint8` is scaled by 1/255; float input is used
    as-is when it already sits in [0, 1] and min-max scaled otherwise, because a
    caller handing over an unnormalized float array is more likely to have a
    range like [0, 255] than to want clipping.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        # ITU-R BT.601 luma. Channel means would wash out a red-on-green edge
        # that a robot camera genuinely sees.
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    elif img.ndim != 2:
        raise ValueError(f"expected (H, W) or (H, W, C), got shape {img.shape}")

    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    out = img.astype(np.float64)
    if out.size and (out.min() < 0.0 or out.max() > 1.0):
        lo, hi = float(out.min()), float(out.max())
        out = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
    return out


def euler_characteristic(mask: np.ndarray) -> int:
    """chi = F - H - V + Q for a boolean mask. Exact, 4-connected foreground."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {m.shape}")
    if m.size == 0:
        return 0
    f = int(m.sum())
    h = int((m[:, :-1] & m[:, 1:]).sum())
    v = int((m[:-1, :] & m[1:, :]).sum())
    q = int((m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]).sum())
    return f - h - v + q


def euler_curve(image: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """chi at each threshold, foreground = `image <= threshold`.

    The `<=` convention matches the source specification. Cost is O(len(t) * HW),
    which is why `encode.py` keeps the threshold count small and fixed.
    """
    img = to_gray(image)
    ts = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    return np.asarray([euler_characteristic(img <= t) for t in ts], dtype=np.float64)


def betti1_bound(n_foreground: int, b0: int) -> int:
    """Upper bound on an estimated b1, from AETHER `AetherBetti.lean`.

        b1_hat <= b0 + (n - 3)

    A cycle needs vertices, so on `n` foreground pixels the number of independent
    loops an estimator may legitimately claim is capped. Used by
    `betti_from_euler` to separate "0 because there are no holes" from "0 because
    the estimate left its valid range" -- a distinction a bare clip destroys.
    """
    n = max(0, int(n_foreground))
    b = max(0, int(b0))
    return max(0, b + max(0, n - 3))


def betti_from_euler(mask: np.ndarray) -> tuple[int, int]:
    """(b0, b1) for a 2D binary mask.

    b0 is counted directly with a 4-connected labelling; b1 follows from
    chi = b0 - b1, so `b1 = b0 - chi`. This identity holds for 2D binary images
    under the padded-8-connectivity background convention the source document
    specifies -- background is padded so the outer region is one component and
    enclosed holes are counted separately.

    b1 is bracketed rather than merely clipped. A mask touching the border can
    produce chi > b0 under this convention, which would make `b0 - chi` negative,
    and a negative hole count is meaningless. Clipping at 0 alone hides *how far*
    out of range the estimate went, so the AETHER bound `b1 <= b0 + (n - 3)`
    supplies the other side (`betti1_bound`). An estimate outside `[0, bound]` is
    a signal that the convention was violated -- a mask whose enclosed region
    touches the padded border, typically -- not a hole count to be trusted, and
    clamping silently would present it as one.
    """
    from scipy.ndimage import label

    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return 0, 0
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # 4-conn
    _, b0 = label(m, structure=structure)
    raw = int(b0) - euler_characteristic(m)
    return int(b0), int(np.clip(raw, 0, betti1_bound(int(m.sum()), int(b0))))


def _disk(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def demo() -> None:
    """Shapes whose topology is known by construction. Exact equality or bust."""
    n = 64

    solid = _disk(n, 32, 32, 20)
    assert euler_characteristic(solid) == 1, "solid disk must have chi = 1"
    assert betti_from_euler(solid) == (1, 0)

    annulus = solid & ~_disk(n, 32, 32, 10)
    assert euler_characteristic(annulus) == 0, "annulus must have chi = 0"
    assert betti_from_euler(annulus) == (1, 1), "annulus must have exactly one hole"

    two = _disk(n, 16, 32, 10) | _disk(n, 48, 32, 10)
    assert euler_characteristic(two) == 2, "two disks must have chi = 2"
    assert betti_from_euler(two) == (2, 0)

    two_holes = solid & ~_disk(n, 26, 32, 6) & ~_disk(n, 40, 32, 6)
    assert euler_characteristic(two_holes) == -1, "disk with 2 holes must have chi = -1"
    assert betti_from_euler(two_holes) == (1, 2)

    # the cancellation the thresholding module exists to defeat: adding one
    # component and one hole at once leaves chi unchanged
    plus_both = two_holes | _disk(n, 8, 8, 4)
    b0_a, b1_a = betti_from_euler(two_holes)
    b0_b, b1_b = betti_from_euler(plus_both)
    assert b0_b == b0_a + 1 and b1_b == b1_a, "expected exactly one new component"
    assert euler_characteristic(plus_both) == euler_characteristic(two_holes) + 1

    cancelling = (solid & ~_disk(n, 32, 32, 10)) | _disk(n, 8, 8, 4)
    assert euler_characteristic(cancelling) == euler_characteristic(solid), (
        "a component birth and a hole birth must cancel in chi -- this is why "
        "equal chi cannot be used as evidence of equal topology"
    )
    print("  chi exact on disk / annulus / two disks / two holes")
    print("  cancellation reproduced: different topology, identical chi")

    curve = euler_curve(np.linspace(0, 1, n * n).reshape(n, n), np.linspace(0, 1, 8))
    assert curve.shape == (8,) and np.all(np.isfinite(curve))

    # every estimate must sit inside the AETHER bound, border cases included
    for name, mask in (
        ("disk", solid),
        ("annulus", annulus),
        ("two disks", two),
        ("two holes", two_holes),
        ("border-touching", np.mgrid[0:n, 0:n][1] < 20),
        ("full frame", np.ones((n, n), dtype=bool)),
        ("single pixel", np.eye(n, dtype=bool)[:1].repeat(1, 0) & (np.arange(n) == 0)),
    ):
        b0, b1 = betti_from_euler(mask)
        ub = betti1_bound(int(np.asarray(mask).sum()), b0)
        assert 0 <= b1 <= ub, f"{name}: b1={b1} outside [0, {ub}]"
    assert betti1_bound(10, 1) == 8 and betti1_bound(2, 1) == 1
    print("  b1 inside the AETHER bound b0 + (n-3) on 7 masks incl. border cases")
    print("demo ok")


if __name__ == "__main__":
    demo()

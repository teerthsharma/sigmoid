"""Frames to world states: the piece that lets sigmoid be fitted on video.

`TopoImageEncoder.encode_video(frames)` returns the `(T, D)` array every other
part of this library consumes, so a `SigmoidWorldModel` can be fitted on a camera
stream with no change to the engine.

Why cubical rather than Vietoris-Rips. The rest of sigmoid computes H0 from a
minimum spanning tree over a point cloud, which is O(n^2 d) in the number of
points. Turning a 256x256 image into a point cloud means 65k points, and that is
not happening inside a 50 ms budget. On a grid the Euler characteristic is four
array reductions -- O(pixels), no distances, exact. `demo()` measures both and
prints the ratio.

The feature vector, per frame:

    euler curve at fixed thresholds     the workhorse; tracks components and
                                        holes appearing as the threshold sweeps
    b0, b1 at a certified threshold     absolute counts where the mask is stable
    per-patch occupancy and variance    coarse spatial layout, so two frames with
                                        identical global topology but a moved
                                        object are distinguishable
    global intensity statistics         mean, std, and the certified gap width

The thresholds are **fixed at construction, not learned per frame**. A per-frame
adaptive threshold makes consecutive feature vectors incommensurable, and the
world model's whole job is to compare them across time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .euler import betti_from_euler, euler_characteristic, to_gray
from .threshold import cancellation_safe_threshold

__all__ = ["TopoImageEncoder", "TopoVisionConfig"]


@dataclass
class TopoVisionConfig:
    n_thresholds: int = 12
    """Euler-curve sample count. Cost is linear in this; 12 fits the budget."""

    patch_grid: int = 4
    """patch_grid x patch_grid occupancy/variance cells."""

    target_chi: int = 1
    """Target for the certified threshold. 1 == one blob, no holes."""

    certify: bool = True
    """Run cancellation-safe selection. Costs a sort plus a few chi evaluations."""

    downsample: int = 1
    """Stride applied before any topology. 2 quarters the pixel count."""


@dataclass
class TopoImageEncoder:
    """Frame -> fixed-length topological feature vector."""

    config: TopoVisionConfig = field(default_factory=TopoVisionConfig)

    @property
    def n_features(self) -> int:
        c = self.config
        return c.n_thresholds + 2 + 2 * c.patch_grid * c.patch_grid + 4

    # ---- single frame ----------------------------------------------------

    def encode(self, frame: np.ndarray) -> np.ndarray:
        c = self.config
        img = to_gray(frame)
        if c.downsample > 1:
            img = img[:: c.downsample, :: c.downsample]

        # Euler curve. Interior thresholds only: chi at 0 or 1 is a constant
        # (empty or full mask) and a constant feature carries no information.
        ts = np.linspace(0.0, 1.0, c.n_thresholds + 2)[1:-1]
        curve = np.asarray([euler_characteristic(img <= t) for t in ts], dtype=np.float64)

        # Ask for the target chi first; fall back to the widest gap with no chi
        # requirement. NOT to the median: on a bimodal frame the median is the
        # background value, so `img <= median` selects every pixel and b0/b1 go
        # constant. That cost a NaN correlation against ground truth before it
        # was caught, so the fallback is still a certified-gap threshold.
        res = cancellation_safe_threshold(img, c.target_chi if c.certify else None)
        if not res.certified and c.certify:
            res = cancellation_safe_threshold(img, None)
        if res.certified:
            b0, b1 = betti_from_euler(img <= res.threshold)
            gap = res.gap
        else:
            b0, b1 = 0, 0
            gap = 0.0

        occ, var = self._patches(img)
        stats = np.array([img.mean(), img.std(), gap, float(img.size)], dtype=np.float64)
        return np.concatenate([curve, [b0, b1], occ, var, stats])

    def _patches(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = self.config.patch_grid
        h, w = img.shape
        ys = np.linspace(0, h, g + 1).astype(int)
        xs = np.linspace(0, w, g + 1).astype(int)
        thresh = float(np.median(img))
        occ = np.empty(g * g)
        var = np.empty(g * g)
        for i in range(g):
            for j in range(g):
                cell = img[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
                k = i * g + j
                if cell.size == 0:
                    occ[k] = var[k] = 0.0
                else:
                    occ[k] = float((cell <= thresh).mean())
                    var[k] = float(cell.var())
        return occ, var

    # ---- video -----------------------------------------------------------

    def encode_video(self, frames: np.ndarray) -> np.ndarray:
        """(T, H, W[, C]) -> (T, D), the contract SigmoidWorldModel.fit consumes."""
        seq = list(frames)
        if not seq:
            return np.zeros((0, self.n_features))
        return np.stack([self.encode(f) for f in seq])


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _disk(n: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _hole_sequence(n: int = 64, frames: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """A disk whose hole opens then closes. Ground truth b1 per frame."""
    seq, truth = [], []
    for t in range(frames):
        radius = 9.0 * np.sin(np.pi * t / (frames - 1))  # 0 -> 9 -> 0
        mask = _disk(n, 32, 32, 22)
        if radius >= 3.0:
            mask = mask & ~_disk(n, 32, 32, radius)
        img = np.where(mask, 0.15, 0.85)
        seq.append(img + 0.01 * np.sin(t))  # mild drift, as a camera would have
        truth.append(1 if radius >= 3.0 else 0)
    return np.asarray(seq), np.asarray(truth)


def demo() -> None:
    """Latency against the budget, the Rips comparison, and real sensitivity."""
    import time

    enc = TopoImageEncoder()

    # ---- latency, and which resolutions actually fit a robot budget
    print(f"  {'resolution':<14}{'chi ms':>9}{'encode ms':>12}   verdict (50 ms budget)")
    for size in (64, 128, 256):
        img = np.where(_disk(size, size / 2, size / 2, size / 3), 0.2, 0.8)
        mask = img <= 0.5
        for _ in range(3):
            euler_characteristic(mask)
            enc.encode(img)
        t0 = time.perf_counter()
        for _ in range(20):
            euler_characteristic(mask)
        chi_ms = (time.perf_counter() - t0) / 20 * 1e3
        t0 = time.perf_counter()
        for _ in range(10):
            enc.encode(img)
        enc_ms = (time.perf_counter() - t0) / 10 * 1e3
        verdict = "fits" if enc_ms < 50.0 else "TOO SLOW"
        print(f"  {f'{size}x{size}':<14}{chi_ms:>9.3f}{enc_ms:>12.2f}   {verdict}")

    # a real camera frame, stated honestly rather than skipped
    big = np.where(_disk(480, 320, 240, 150), 0.2, 0.8)[:, :640]
    t0 = time.perf_counter()
    enc.encode(big)
    big_ms = (time.perf_counter() - t0) * 1e3
    print(f"  {'480x640':<14}{'':>9}{big_ms:>12.2f}   "
          f"{'fits' if big_ms < 50 else 'TOO SLOW -- use downsample=2'}")

    # ---- cubical vs Vietoris-Rips on the same image
    from ..state import h0_barcode

    img = np.where(_disk(128, 64, 64, 40), 0.2, 0.8)
    mask = img <= 0.5
    t0 = time.perf_counter()
    for _ in range(20):
        euler_characteristic(mask)
    cubical = (time.perf_counter() - t0) / 20

    pts = np.argwhere(mask).astype(np.float64)
    sub = pts[:: max(1, len(pts) // 400)]  # Rips cannot take 5k points at all
    t0 = time.perf_counter()
    h0_barcode(sub)
    rips = time.perf_counter() - t0
    print(
        f"\n  cubical chi {cubical * 1e3:.3f} ms on {mask.sum()} px vs "
        f"Rips H0 {rips * 1e3:.1f} ms on {len(sub)} subsampled px "
        f"-> {rips / cubical:,.0f}x"
    )
    assert rips > cubical, "cubical must beat Rips or the premise is wrong"

    # ---- sensitivity: does the encoding actually see the hole?
    frames, truth = _hole_sequence()
    feats = enc.encode_video(frames)
    assert feats.shape == (len(frames), enc.n_features)
    b1 = feats[:, enc.config.n_thresholds + 1]
    corr = float(np.corrcoef(b1, truth)[0, 1])
    print(f"  b1 feature vs ground-truth hole presence: corr {corr:+.4f}")
    assert corr > 0.8, f"encoding does not track the hole (corr {corr:.3f})"

    best = max(
        (abs(float(np.corrcoef(feats[:, j], truth)[0, 1])), j)
        for j in range(feats.shape[1])
        if feats[:, j].std() > 1e-9
    )
    print(f"  strongest tracking feature: index {best[1]}, |corr| {best[0]:.4f}")

    # ---- the contract: SigmoidWorldModel must accept this directly
    from ..engine import SigmoidConfig, SigmoidWorldModel

    long_frames, _ = _hole_sequence(n=48, frames=120)
    obs = enc.encode_video(long_frames)
    wm = SigmoidWorldModel(config=SigmoidConfig(window=12, linear_dim=8)).fit([obs])
    roll = wm.imagine(obs[:12], steps=4, stop_on_gate=False)
    assert roll.hiddens.shape == (4, obs.shape[1]), "world model did not accept video features"
    print(f"  SigmoidWorldModel.fit accepted (T,D)={obs.shape}, imagined {len(roll)} steps")
    print("demo ok")


if __name__ == "__main__":
    demo()

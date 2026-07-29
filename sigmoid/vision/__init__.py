"""Topological vision: images to world states, in O(pixels).

A robot vision path has a 10-50 ms budget, and Vietoris-Rips persistence on a
point cloud does not fit in it -- a 256x256 frame is 65k points. On a *grid* the
Euler characteristic is exact and costs four array reductions:

    chi = F - H - V + Q

    from sigmoid.vision import TopoImageEncoder, euler_characteristic
    obs = TopoImageEncoder().encode_video(frames)   # (T, D)
    wm = sigmoid.SigmoidWorldModel().fit([obs])     # fit a world model on video

The construction and the cancellation-safe threshold rule are Teerth Sharma's,
from `computer-vision-basics-in-microsoft-excel`, worktree
`cancellation-safe-topology`. The load-bearing caveat carried from that work:
**equal chi does not prove equal topology** -- a component birth and a hole birth
cancel in the sum, so stability is measured in the intensity domain instead.

numpy + scipy only. No PIL, no OpenCV, no torch.
"""

from .encode import TopoImageEncoder, TopoVisionConfig
from .euler import betti1_bound, betti_from_euler, euler_characteristic, euler_curve, to_gray
from .threshold import ThresholdResult, cancellation_safe_threshold

__all__ = [
    "ThresholdResult",
    "TopoImageEncoder",
    "TopoVisionConfig",
    "betti1_bound",
    "betti_from_euler",
    "cancellation_safe_threshold",
    "euler_characteristic",
    "euler_curve",
    "to_gray",
]

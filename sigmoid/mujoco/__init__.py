"""Bridge to MuJoCo island partitions and the `mujoco#3396` corpus.

MuJoCo's `mj_island` partitions dynamic trees into connected components of the
tree/constraint incidence graph. That is an H0 statement, and sigmoid computes
H0 exactly from the other direction — single-linkage merge heights are the H0
death times — so the two agree on every frame, correlation 1.0000.

    from sigmoid.mujoco import island_count, make_corpus

`island_count` reads the partition at an absolute radius. `make_corpus`
generates the S²-Vietoris-Rips trajectories the PR's design specifies, which
serve as this library's reference positive control: the ground truth is H0 by
construction rather than a proxy for it.

Nothing here imports mujoco. These are the partition semantics and the corpus
generator math, in numpy.
"""

from .corpus import CorpusConfig, geodesic_step, make_corpus
from .island import geodesic_distances, island_count, island_labels

__all__ = [
    "CorpusConfig",
    "geodesic_distances",
    "geodesic_step",
    "island_count",
    "island_labels",
    "make_corpus",
]

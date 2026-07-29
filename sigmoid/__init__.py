"""sigmoid -- a world-model inference engine built on topological coupling operators.

Convert any normal model into a world model:

    import sigmoid
    wm = sigmoid.wrap(model, prompts)          # calibrate on captured activations
    z  = wm.observe(hidden_window)             # topological world state
    r  = wm.imagine(z, steps=16)               # roll forward without the model
    print(wm.certificate(16))                  # contraction bound on that rollout

The engine never touches the wrapped model's weights. It learns a contractive
operator on topological signatures of the model's own activations, and a sheaf
consistency gate that says when the imagined future stopped being trustworthy.
"""

from .state import (
    Barcode,
    TopoEncoder,
    TopoEncoderConfig,
    betti_curve,
    h0_barcode,
    h1_barcode,
    hilbert_coefficients,
)
from .operator import CouplingOperator, RolloutCertificate
from .sheaf import GateReading, SheafGate
from .nbody import MultiBodyCoupling
from .schedule import (
    block_centroids,
    build_topology_block_schedule,
    zero_dim_persistence_salience,
)
from .engine import Rollout, SigmoidConfig, SigmoidWorldModel, wrap
from .control import Plan, TopologicalMPC, beta0_cost, target_cost
from .adapters import capture_hidden, capture_from_hook, layer_names, make_capture_fn
from .bench import ArmResult, BenchReport, compare, rollout_error

__version__ = "0.1.0"

__all__ = [
    "Barcode",
    "TopoEncoder",
    "TopoEncoderConfig",
    "betti_curve",
    "h0_barcode",
    "h1_barcode",
    "hilbert_coefficients",
    "CouplingOperator",
    "RolloutCertificate",
    "GateReading",
    "SheafGate",
    "MultiBodyCoupling",
    "block_centroids",
    "build_topology_block_schedule",
    "zero_dim_persistence_salience",
    "Rollout",
    "SigmoidConfig",
    "SigmoidWorldModel",
    "wrap",
    "Plan",
    "TopologicalMPC",
    "beta0_cost",
    "target_cost",
    "capture_hidden",
    "capture_from_hook",
    "layer_names",
    "make_capture_fn",
    "ArmResult",
    "BenchReport",
    "compare",
    "rollout_error",
    "__version__",
]

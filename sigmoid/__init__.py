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

The same machinery drives an agentic runtime for robots:

    from sigmoid import InferenceEngine, RobotAgent, TopologicalMPC
    from sigmoid.providers import auto            # local, ollama, or 10 hosted APIs
    from sigmoid.triton import patch_attention    # topology-sparse attention

Importing this package pulls in numpy and scipy only. torch, triton,
transformers and every network client are imported lazily inside the function
that needs them, so `import sigmoid` works on a CPU-only machine and a missing
optional dependency surfaces at call time with a named error.

`ToolCall` is deliberately NOT exported: `sigmoid.agent.ToolCall` is a *parsed*
call and `sigmoid.providers.ToolCall` is a *wire* call. They are different
types, so import whichever you mean from its own module.
"""

from .adapters import capture_from_hook, capture_hidden, layer_names, make_capture_fn
from .agent import (
    Agent,
    AgentResult,
    HermesToolParser,
    Tool,
    ToolRegistry,
    ToolResult,
    hermes_system_prompt,
)
from .bench import ArmResult, BenchReport, compare, rollout_error
from .control import Plan, TopologicalMPC, beta0_cost, target_cost
from .engine import Rollout, SigmoidConfig, SigmoidWorldModel, wrap
from .hooks import HookManager, HookPoint, HookVeto, LoggingHook, TimingHook
from .inference import (
    GenerationResult,
    GenerationStats,
    InferenceConfig,
    InferenceEngine,
    Session,
)
from .nbody import MultiBodyCoupling
from .operator import CouplingOperator, RolloutCertificate
from .providers import Completion, Message, Provider, ProviderError, ToolSpec
from .robot import RobotAgent
from .sheaf import GateReading, SheafGate
from .state import (
    Barcode,
    TopoEncoder,
    TopoEncoderConfig,
    betti_curve,
    h0_barcode,
    h1_barcode,
    hilbert_coefficients,
)
from .triton import (
    AttentionPatch,
    IncrementalSalience,
    block_centroids,
    build_topology_block_schedule,
    patch_attention,
    schedule_from_keys,
    topology_attention,
    zero_dim_persistence_salience,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "AttentionPatch",
    "Completion",
    "GenerationResult",
    "GenerationStats",
    "HermesToolParser",
    "HookManager",
    "HookPoint",
    "HookVeto",
    "InferenceConfig",
    "InferenceEngine",
    "LoggingHook",
    "Message",
    "Provider",
    "ProviderError",
    "RobotAgent",
    "Session",
    "TimingHook",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "hermes_system_prompt",
    "patch_attention",
    "schedule_from_keys",
    "topology_attention",
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
    "IncrementalSalience",
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

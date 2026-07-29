"""Swap topology-sparse attention into a live HuggingFace GPT-2, no retraining.

    from sigmoid.triton.patch import patch_attention

    with patch_attention(model, topk=4) as patch:
        logits = model(ids).logits          # sparse
    logits = model(ids).logits              # dense again, bit-identical

The weights are never touched. Only the attention *computation* is replaced, so
whatever the model already learned is what runs -- the question this exists to
ask is how much of a trained model's output survives dropping most of its key
blocks, and that question is only meaningful if the wiring itself is exact.

Which mechanism, and why
------------------------
transformers 5.3.0 does not put attention on the module. `GPT2Attention.forward`
ends at

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward)

so the supported way in is to register an implementation and point the config at
it. Monkeypatching `forward` would have to re-implement the Conv1D projections,
the cache update and the cross-attention branch -- about 60 lines of transformers
internals copied into this repo and pinned to this release. Registering is one
dict entry, and `restore()` is one string assignment.

Registering the *mask* alongside it is not optional, and that is measured.
`masking_utils._preprocess_mask_arguments` looks the implementation up in
`ALL_MASK_ATTENTION_FUNCTIONS._global_mapping` and, when it is absent, returns no
mask at all -- the TGI/vLLM path, where the kernel is trusted to mask itself.
Ours does, so patching every layer hides the omission completely -- 2.750e-01
either way. It surfaces the moment any layer falls back to eager, which is
exactly what `layers=`, padding and decode all do. Patching layer 0 of distilgpt2
with a dense schedule measured 1.877e-01 with the mask registered and 1.342e+02
without: the five unpatched layers were attending to the future. Not a subtle
bug, but a silent one -- the model still emits plausible text, and the omission
is invisible in the whole-model test.

Measured on distilgpt2 (6 layers, 12 heads, head dim 64) on an RTX 4060 Laptop,
1000 tokens of prose, fp32, max abs logit difference against the unpatched model:

    dense causal schedule                            2.750e-01
    dense causal schedule, TRITON_F32_DEFAULT=ieee   3.967e-04
    eager vs the sdpa the model loads with           2.670e-04

Nearly all of the 2.750e-01 is TF32: `tl.dot` defaults to tf32 for fp32 inputs
while torch's matmul does not, and forcing ieee drops the residual to the gap
between two of transformers' own backends. So the sparse path's error budget
starts at one tf32 rounding per layer, not at a wiring fault. This module does
not set that knob -- it is process-global and belongs to the caller.
"""

from __future__ import annotations

from typing import Any

__all__ = ["patch_attention", "AttentionPatch"]

# One registry key for the library, not one per handle. `AttentionInterface`
# .register is a classmethod writing a class-level dict shared by every instance,
# so per-handle keys would leak an entry per patch and never be collectable. The
# per-layer state lives on the modules instead, under `_PATCH_ATTR`, which is
# also the only way to express `layers=`: `config._attn_implementation` is a
# single string on a config object every layer shares.
_IMPL = "sigmoid_topology_sparse"
_PATCH_ATTR = "_sigmoid_topology_patch"


def _register() -> None:
    """Put `_IMPL` in the attention and mask registries, once per process."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _IMPL in ALL_ATTENTION_FUNCTIONS:
        return
    ALL_ATTENTION_FUNCTIONS.register(_IMPL, _attention_forward)
    # eager's mask, not sdpa's: it is additive float rather than boolean, which
    # is what `_plain_causal` reads to notice padding, and it is what the
    # fallback layers need anyway. See the module docstring for the 1.342e+02
    # this line is worth.
    ALL_MASK_ATTENTION_FUNCTIONS.register(_IMPL, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    # Deliberately never unregistered. The entry is inert unless some config
    # names it, `restore()` is what stops it being named, and leaving it in place
    # keeps two handles over two models from racing each other's teardown.


def _plain_causal(attention_mask: Any) -> bool:
    """Is this mask exactly the causal mask the kernel already applies itself?

    The kernel takes no mask argument -- it masks `k_pos <= offs_m` inside the
    tile loop and knows nothing about padding. So a batch with pad tokens, or a
    caller-supplied 4D mask, would have its mask silently dropped. Those calls go
    to eager instead, which is slower and correct rather than fast and wrong.

    Costs one [seq, seq] bool and one device sync per layer per forward. That is
    real, and it is still nothing beside the attention it guards.
    """
    if attention_mask is None:
        # no mask created at all: causal-only is exactly what the kernel does
        return True
    import torch

    if attention_mask.ndim != 4 or attention_mask.shape[-1] != attention_mask.shape[-2]:
        return False
    # eager's mask is additive (0 attends), but a caller-supplied 4D mask early
    # exits `_preprocess_mask_arguments` as-is and may be boolean (True attends)
    allowed = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    pos = torch.arange(attention_mask.shape[-1], device=attention_mask.device)
    return bool((allowed == (pos[None, :] <= pos[:, None])).all())


def _attention_forward(module: Any, query: Any, key: Any, value: Any,
                       attention_mask: Any, **kwargs: Any) -> tuple[Any, Any]:
    """The registered implementation. Runs the kernel, or hands back to eager."""
    from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward

    patch = getattr(module, _PATCH_ATTR, None)
    if (
        patch is None            # a layer outside `layers=`
        or not patch.active      # a restored, or half-restored, handle
        or module.training       # the kernel has no dropout and no backward
        # q and k disagree only under a KV cache, where the kernel's
        # `q.shape == k.shape` contract does not hold; decode falls through
        or query.shape[-2] != key.shape[-2]
        or not _plain_causal(attention_mask)
    ):
        return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
    # the interface returns [batch, seq, heads, dim] -- GPT2Attention reshapes the
    # last two dims straight into c_proj without transposing again
    return patch._attend(query, key, value).transpose(1, 2), None


class AttentionPatch:
    """Handle for one `patch_attention` call. `restore()` puts the model back."""

    def __init__(self, model: Any, modules: list[Any], previous: str | None,
                 params: dict[str, Any]) -> None:
        self.model = model
        self.modules = modules
        self.layers = [m.layer_idx for m in modules]
        self._previous = previous
        self._params = params
        self.active = False

    def __repr__(self) -> str:
        state = "active" if self.active else "restored"
        return f"AttentionPatch({state}, layers={self.layers}, {self._params})"

    def __enter__(self) -> AttentionPatch:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.restore()

    def _apply(self) -> None:
        for module in self.modules:
            setattr(module, _PATCH_ATTR, self)
        self.model.config._attn_implementation = _IMPL
        self.active = True

    def restore(self) -> None:
        """Undo the patch. Safe to call twice, and after a failed `_apply`."""
        # flipped first, so a handle that somehow fails to unwind below is still
        # inert: `_attention_forward` checks this before anything else
        self.active = False
        for module in self.modules:
            if hasattr(module, _PATCH_ATTR):
                delattr(module, _PATCH_ATTR)
        if self.model.config._attn_implementation == _IMPL:
            self.model.config._attn_implementation = self._previous

    def _attend(self, query: Any, key: Any, value: Any) -> Any:
        import torch

        from .attention import topology_attention

        block_size = self._params["block_size"]
        seq = query.shape[-2]
        pad = -seq % block_size
        if pad:
            # right padding, and only right padding. A padded key sits at
            # position >= seq and the kernel keeps a key only where
            # k_pos <= offs_m, so no real query row can ever see one -- causality
            # already excludes the zeros, without a mask the kernel cannot take.
            # The padded query rows are computed and then sliced off.
            query, key, value = (
                torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (query, key, value)
            )
        out = topology_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            schedule=self._params["schedule"],
            backend=self._params["backend"],
            block_size=block_size,
            local_radius_blocks=self._params["local_radius_blocks"],
            sink_blocks=self._params["sink_blocks"],
            topk=self._params["topk"],
        )
        return out[..., :seq, :] if pad else out


def _attention_modules(model: Any) -> list[Any]:
    """The model's self-attention modules, in layer order."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    found = [
        m for m in model.modules()
        if isinstance(m, GPT2Attention) and not m.is_cross_attention
    ]
    if not found:
        raise ValueError(
            f"{type(model).__name__} exposes no GPT2Attention modules; "
            f"patch_attention wires the GPT-2 family only"
        )
    # `.modules()` yields registration order, which is already h.0, h.1, ...;
    # sorting makes `layers=` mean layer index rather than traversal position
    found.sort(key=lambda m: m.layer_idx)
    return found


def patch_attention(
    model: Any,
    *,
    block_size: int = 64,
    local_radius_blocks: int = 2,
    sink_blocks: int = 1,
    topk: int = 4,
    backend: str = "auto",
    schedule: tuple[Any, Any] | None = None,
    layers: list[int] | None = None,
) -> AttentionPatch:
    """Replace GPT-2's attention with the topology-sparse kernel, in place.

    Args:
        model: a loaded GPT-2 family model (`GPT2LMHeadModel`, `GPT2Model`, ...).
        block_size: query and key block edge; sequences are right-padded up to a
            multiple of it.
        local_radius_blocks: how many blocks back the local window reaches.
        sink_blocks: leading blocks every query attends to.
        topk: how many of the most salient key blocks to keep.
        backend: "triton", "torch", or "auto".
        schedule: a fixed `(offsets, indices)` used by every patched layer
            instead of one built from that layer's own keys. This is how an
            ablation pins a schedule -- passing
            `build_dense_causal_block_schedule(n)` turns the patch into plain
            causal attention and isolates a wiring fault from a sparsity cost.
            Fixed to one sequence length by construction.
        layers: layer indices to patch, or None for all of them.

    Returns an `AttentionPatch`; call `.restore()` or use it as a context
    manager. Only the forward pass is replaced -- no weight is read or written.

    Falls back to eager attention, per call, whenever the kernel's contract does
    not hold: training mode, a KV cache (q and k lengths differ), or a mask that
    is not plain causal. The patch is for prefill and scoring, which is where the
    measurement it exists to support is taken.

    Each patched layer builds its own schedule from its own keys every forward,
    which is the honest thing to measure per layer and the wrong thing for
    throughput -- `schedule_from_keys` once and `schedule=` is the fast path.
    """
    modules = _attention_modules(model)
    if layers is not None:
        wanted = sorted(set(layers))
        known = {m.layer_idx for m in modules}
        unknown = [i for i in wanted if i not in known]
        if unknown:
            raise ValueError(f"no such layers {unknown}; model has {sorted(known)}")
        modules = [m for m in modules if m.layer_idx in wanted]

    live = [m.layer_idx for m in _attention_modules(model) if hasattr(m, _PATCH_ATTR)]
    if live:
        # saving `_previous` off an already-patched config would record `_IMPL`
        # as the thing to restore to, and the model could never get back
        raise ValueError(
            f"layers {live} are already patched; call .restore() on that handle first"
        )

    for module in modules:
        # the kernel hardcodes scale = 1/sqrt(head_dim); anything else in the
        # config would be dropped without a trace in the logits
        if not module.scale_attn_weights or module.scale_attn_by_inverse_layer_idx:
            raise ValueError(
                f"layer {module.layer_idx} scales attention differently from the "
                f"kernel's fixed 1/sqrt(head_dim) (scale_attn_weights="
                f"{module.scale_attn_weights}, scale_attn_by_inverse_layer_idx="
                f"{module.scale_attn_by_inverse_layer_idx})"
            )

    _register()
    patch = AttentionPatch(
        model, modules, model.config._attn_implementation,
        {
            "block_size": block_size,
            "local_radius_blocks": local_radius_blocks,
            "sink_blocks": sink_blocks,
            "topk": topk,
            "backend": backend,
            "schedule": schedule,
        },
    )
    try:
        patch._apply()
    except Exception:
        patch.restore()  # a half-applied patch leaves the model unrunnable
        raise
    return patch

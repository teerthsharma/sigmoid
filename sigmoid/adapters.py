"""Getting a hidden-state trajectory out of "any normal model".

Three cases cover essentially everything:

1. A HuggingFace-style module whose forward accepts ``output_hidden_states``.
   Used for LLMs, ViTs, encoders -- anything transformers-shaped.
2. Any other ``torch.nn.Module``: attach a forward hook to a named submodule.
   Used for policies, CNNs, custom research code.
3. A plain callable returning arrays. Used for simulators and black boxes.

All three land on the same contract: a float64 ``(T, D)`` array, one row per
step of whatever the model's "time" is -- tokens, frames, or env steps. That
single contract is what lets the rest of the engine be model-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Sequence

import numpy as np

__all__ = ["capture_hidden", "capture_from_hook", "layer_names", "TorchUnavailable"]


class TorchUnavailable(RuntimeError):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TorchUnavailable("PyTorch is required for torch model capture") from exc
    return torch


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().to("cpu")
        if hasattr(x, "float"):
            x = x.float()
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


def layer_names(model: Any) -> list[str]:
    """List hookable submodule names -- use this to pick a capture point."""
    return [name for name, _ in model.named_modules() if name]


def capture_hidden(
    model: Any,
    inputs: Any,
    *,
    layer: int | str = -1,
    batch_index: int = 0,
) -> np.ndarray:
    """Run `model` on `inputs` and return a (T, D) hidden-state trajectory.

    Args:
        model: HF-style module, torch module, or callable.
        inputs: whatever the model consumes. A mapping is passed as kwargs.
        layer: int selects a hidden-state index for HF models (-1 = last);
            str selects a submodule name and switches to hook capture.
        batch_index: which batch row to take when the output is batched.
    """
    if isinstance(layer, str):
        return capture_from_hook(model, inputs, layer, batch_index=batch_index)

    if not hasattr(model, "forward"):
        out = model(inputs)
        return _squeeze_batch(_to_numpy(out), batch_index)

    torch = _torch()
    model.eval()
    with torch.no_grad():
        try:
            out = _call(model, inputs, output_hidden_states=True)
        except TypeError:
            out = _call(model, inputs)
            return _squeeze_batch(_to_numpy(_unwrap(out)), batch_index)

    hidden = getattr(out, "hidden_states", None)
    if hidden is None and isinstance(out, dict):
        hidden = out.get("hidden_states")
    if hidden is None:
        return _squeeze_batch(_to_numpy(_unwrap(out)), batch_index)
    return _squeeze_batch(_to_numpy(hidden[layer]), batch_index)


def capture_from_hook(
    model: Any,
    inputs: Any,
    module_name: str,
    *,
    batch_index: int = 0,
) -> np.ndarray:
    """Capture the output of one named submodule via a forward hook."""
    torch = _torch()
    target = dict(model.named_modules()).get(module_name)
    if target is None:
        raise KeyError(
            f"no submodule named {module_name!r}; try sigmoid.layer_names(model)"
        )

    grabbed: list[Any] = []
    handle = target.register_forward_hook(lambda _m, _i, output: grabbed.append(output))
    try:
        model.eval()
        with torch.no_grad():
            _call(model, inputs)
    finally:
        handle.remove()

    if not grabbed:
        raise RuntimeError(f"submodule {module_name!r} did not fire during forward")
    return _squeeze_batch(_to_numpy(_unwrap(grabbed[-1])), batch_index)


def _call(model: Any, inputs: Any, **kwargs: Any) -> Any:
    # Mapping, not dict: HuggingFace BatchEncoding is a UserDict, so an
    # isinstance(_, dict) check silently misses it and calls positionally.
    if isinstance(inputs, Mapping):
        return model(**inputs, **kwargs)
    if isinstance(inputs, (list, tuple)) and not _is_arraylike(inputs):
        return model(*inputs, **kwargs)
    return model(inputs, **kwargs)


def _is_arraylike(x: Any) -> bool:
    return hasattr(x, "shape")


def _unwrap(out: Any) -> Any:
    if isinstance(out, (list, tuple)):
        return out[0]
    for attr in ("last_hidden_state", "logits"):
        if hasattr(out, attr):
            return getattr(out, attr)
    if isinstance(out, dict):
        for key in ("last_hidden_state", "logits"):
            if key in out:
                return out[key]
    return out


def _squeeze_batch(arr: np.ndarray, batch_index: int) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[batch_index]
    elif arr.ndim > 3:
        arr = arr[batch_index].reshape(arr.shape[1], -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a (T, D) trajectory, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def concat_trajectories(trajectories: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Normalize a batch of captures into a list of (T, D) float64 arrays."""
    return [np.asarray(t, dtype=np.float64) for t in trajectories if len(t) > 1]


def make_capture_fn(
    model: Any,
    *,
    layer: int | str = -1,
) -> Callable[[Any], np.ndarray]:
    """Freeze a model + layer choice into a one-argument capture function."""
    return lambda inputs: capture_hidden(model, inputs, layer=layer)

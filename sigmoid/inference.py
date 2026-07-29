"""Prefill + decode over a KV cache, with attention restricted to a block schedule.

The loop is the ordinary one: prefill runs the prompt once and keeps every key and
value, then each decode step appends a single position and reads the cache. What is
not ordinary is which keys a decode step reads. `sigmoid.triton.schedule` turns key
blocks into a 0D-persistence salience and selects sink, local and top-k salient
blocks; attention here only visits those.

Three things had to be true for that to be worth measuring, and each one shaped the
code:

*The schedule must not be rebuilt per token.* A rebuild is an n x n distance matrix
and a fresh MST every step. `IncrementalSalience` grows the tree instead, and since
decode appends one *token* while the salience is defined over *blocks*, the update
fires only on the token that completes a block -- once every `block_size` steps.

*The control arm must be genuinely dense.* `backend="dense"` installs no attention
hook at all, so the model runs its own SDPA over the full causal mask. That makes
the dense arm reproduce `model.generate(do_sample=False)` token for token, which is
the test that separates "the loop is wrong" from "sparsity costs quality". Measured:
exact match on distilgpt2, 64/64 tokens.

*Schedule cost must be attributable.* Salience is CPU/numpy and the attention is a
GPU kernel, so they are timed separately (`GenerationStats.schedule_ms` against
`attention_ms`) rather than pooled into one "sparse attention" number. Set
`InferenceConfig.profile=True` to make the attention span sync around CUDA; it costs
throughput, so tokens/sec should be read from a non-profiled run.

Attention is swapped through `transformers`' own `AttentionInterface` registry, not
by rewriting module classes -- the model keeps its own `DynamicCache`, position
handling and sampling entry points, so the only variable between arms is which keys
the attention op reads.

What it currently buys, so nobody has to guess: nothing, at any length distilgpt2
can run. The decode attention op does not beat dense until roughly 16k positions
(12 heads x 64 dim) or 8k (32 x 128), and distilgpt2 stops at 1024, where topology
costs 0-20% of decode throughput for a density of 0.27. `tests/test_inference.py`
prints the whole table, including the rows where it loses.

    engine = InferenceEngine(model, tokenizer, config=InferenceConfig(backend="triton"))
    print(engine.generate("the theory of topology", max_new_tokens=32).text)

    session = engine.prefill(ids)          # agent runtimes drive it directly
    while not done:
        token = engine.decode_step(session)
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .triton.schedule import IncrementalSalience

__all__ = [
    "InferenceConfig",
    "GenerationStats",
    "GenerationResult",
    "Session",
    "InferenceEngine",
    "load_kernel",
]

# head dims the merged kernel compiles for (triton-lang/kernels#22)
_KERNEL_HEAD_DIMS = frozenset({16, 32, 64, 128})
_BACKENDS = ("triton", "torch", "dense", "auto")
_ATTENTION_NAME = "sigmoid_topology"

# id(model.config) -> engine. The registry hands the attention function the calling
# module, and every attention module in a model shares that model's config object,
# so config identity routes a call back to the engine that installed the hook
# without walking or tagging the module tree.
_OWNERS: dict[int, InferenceEngine] = {}


def load_kernel() -> Any:
    """The merged Triton kernel module, or None if it is not importable.

    Not pip-installable, so `SIGMOID_KERNELS_PATH` points at a checkout of
    triton-lang/kernels when it is not already on `sys.path`.
    """
    path = os.environ.get("SIGMOID_KERNELS_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        from kernels import topology_sparse_attention

        return topology_sparse_attention
    except Exception:  # noqa: BLE001 - triton import failures are not just ImportError
        return None


@dataclass
class InferenceConfig:
    """Schedule shape, backend, and sampling.

    `block_size` is both the schedule's block and the kernel's BLOCK_M/BLOCK_N, so
    the merged kernel requires it to be a power of two.
    """

    block_size: int = 64
    local_radius_blocks: int = 1
    sink_blocks: int = 1
    topk: int = 4
    backend: str = "auto"
    max_context: int = 0  # 0 -> the model's own position limit
    dtype: str | None = None
    device: str | None = None

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    # sync CUDA around the attention op so attention_ms is real rather than launch
    # time. Off by default: a sync per layer per token perturbs the throughput it
    # sits inside, so tokens/sec and the schedule/attention split want two runs.
    profile: bool = False

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {self.backend!r}")
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a positive power of two")
        for name in ("local_radius_blocks", "sink_blocks", "topk"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class GenerationStats:
    """What a run cost, split finely enough to argue about.

    `schedule_ms` and `attention_ms` are kept apart because pooling them lets a
    schedule that is cheap in the kernel but expensive to build look like a win.
    """

    backend: str = ""
    tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    schedule_ms: float = 0.0
    attention_ms: float = 0.0
    kv_bytes: int = 0
    selected_blocks: int = 0
    dense_blocks: int = 0
    context: int = 0

    @property
    def decode_ms_per_token(self) -> float:
        return self.decode_ms / self.tokens if self.tokens else float("nan")

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / (self.decode_ms / 1e3) if self.decode_ms else float("nan")

    @property
    def sparsity(self) -> float:
        """Selected blocks over causal-dense blocks; 1.0 means nothing was skipped."""
        return self.selected_blocks / self.dense_blocks if self.dense_blocks else 1.0

    def __str__(self) -> str:
        return (
            f"{self.backend:<8} ctx={self.context:<5} {self.tokens:>4} tok  "
            f"prefill {self.prefill_ms:7.1f} ms  "
            f"decode {self.decode_ms_per_token:6.2f} ms/tok  "
            f"{self.tokens_per_sec:7.1f} tok/s  "
            f"sched {self.schedule_ms:6.1f} ms  attn {self.attention_ms:7.1f} ms  "
            f"kv {self.kv_bytes / 2**20:5.1f} MiB  density {self.sparsity:.3f}"
        )


@dataclass
class GenerationResult:
    ids: list[int]  # generated tokens only, prompt excluded
    text: str
    stats: GenerationStats


@dataclass
class Session:
    """A live KV cache plus everything needed to take one more step."""

    ids: list[int]
    prompt_len: int
    past: Any
    logits: Any
    stats: GenerationStats
    generator: Any = None

    @property
    def generated(self) -> list[int]:
        return self.ids[self.prompt_len :]


class _LayerSchedule:
    """Per-layer salience and block selection, grown as the KV cache grows.

    One schedule per layer rather than one shared: each layer's keys carry their own
    topology, and the merged kernel takes a single CSR for all heads, so the natural
    unit is a layer's full key vector (heads concatenated) per position.

    Only *complete* blocks get a salience. The trailing partial block is always the
    query's own block, which the local window selects unconditionally, so it never
    needs one -- and giving it a centroid over a half-filled block would feed
    `IncrementalSalience` a point that later moves, which its append-only contract
    does not allow.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.n_blocks = 0
        self._state: IncrementalSalience | None = None
        self._salient: tuple[int, ...] = ()
        self._gather_key: tuple[int, int] | None = None
        self._gather_head: Any = None

    def observe(self, key: Any) -> None:
        """Fold any newly completed key blocks into the salience."""
        block = self.config.block_size
        complete = key.shape[-2] // block
        if complete <= self.n_blocks:
            return  # mid-block: nothing about the salience can have changed
        new = self._centroids(key, self.n_blocks, complete)
        if self._state is None:
            self._state = IncrementalSalience(new)
        else:
            for row in new:
                self._state.append(row)
        self.n_blocks = complete
        salience = self._state.salience
        topk = min(self.config.topk, salience.shape[0])
        # stable argsort on the negated array reproduces torch.topk's tie-break
        # (lowest index first), which is what `_causal_csr` matches
        self._salient = (
            tuple(np.argsort(-salience, kind="stable")[:topk].tolist()) if topk else ()
        )

    def _centroids(self, key: Any, start: int, stop: int) -> np.ndarray:
        block = self.config.block_size
        heads, dim = key.shape[1], key.shape[-1]
        window = key[0, :, start * block : stop * block, :]
        pooled = window.reshape(heads, stop - start, block, dim).mean(dim=2)
        # [heads, blocks, dim] -> [blocks, heads*dim]: one vector per position block,
        # which is what the salience is defined over
        return pooled.permute(1, 0, 2).reshape(stop - start, heads * dim).float().cpu().numpy()

    def allowed(self, query_block: int) -> list[int]:
        """Sinks + local window + salient blocks, clipped to the causal past.

        Same construction and ordering as `schedule._causal_csr`, one row at a time.
        Decode needs exactly one row per step and that builder emits all of them,
        which its own docstring flags as O(n^2) work decode does not want.
        """
        cfg = self.config
        allow = set(range(min(cfg.sink_blocks, query_block + 1)))
        allow.update(range(max(0, query_block - cfg.local_radius_blocks), query_block + 1))
        allow.update(b for b in self._salient if b <= query_block)
        return sorted(allow) if allow else [query_block]

    def gather_index(self, blocks: list[int], query_block: int, seq: int, torch: Any, device: Any):
        """Key positions for a single decode query, as a flat index tensor.

        The whole-block part only changes when the query block advances or when a
        new block enters the salience (adding a point can lower an existing block's
        merge height, so the top-k set moves), hence both in the cache key.
        """
        block = self.config.block_size
        key = (query_block, self.n_blocks)
        if self._gather_key != key:
            whole = [b for b in blocks if b < query_block]
            flat = (
                np.concatenate([np.arange(b * block, (b + 1) * block) for b in whole])
                if whole
                else np.empty(0, dtype=np.int64)
            )
            self._gather_head = torch.as_tensor(flat, dtype=torch.long, device=device)
            self._gather_key = key
        tail = torch.arange(query_block * block, seq, dtype=torch.long, device=device)
        return torch.cat([self._gather_head, tail])


def _dispatch(module, query, key, value, attention_mask, **kwargs):
    """The registered attention implementation; routes back to the owning engine."""
    config = getattr(module, "config", None)
    engine = _OWNERS.get(id(config)) if config is not None else None
    if engine is None:
        raise RuntimeError(
            f"{_ATTENTION_NAME} attention was called by a module sigmoid does not own; "
            "an InferenceEngine was closed while its model was still configured for it"
        )
    return engine._attend(module, query, key, value)


def _module_scale(module, head_dim: int) -> float:
    """The scale the model's own attention would apply to q @ k^T."""
    scaling = getattr(module, "scaling", None)
    if scaling is not None:
        return float(scaling)
    scale = 1.0 / math.sqrt(head_dim) if getattr(module, "scale_attn_weights", True) else 1.0
    if getattr(module, "scale_attn_by_inverse_layer_idx", False):
        scale /= float(module.layer_idx + 1)
    return scale


class InferenceEngine:
    """Generation over a causal LM with attention restricted to a block schedule.

    `generate` is the convenience path; `prefill` and `decode_step` are the real
    API, because an agent runtime needs to interleave its own work between tokens.
    """

    def __init__(self, model, tokenizer=None, *, config: InferenceConfig | None = None):
        import torch

        self._torch = torch
        self.config = config or InferenceConfig()
        self.tokenizer = tokenizer
        self.last_result: GenerationResult | None = None

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        if self.config.dtype:
            model = model.to(dtype=getattr(torch, self.config.dtype))
        self.model = model.to(self.device).eval()
        self._cuda = self.device.type == "cuda"

        self._kernel = None
        self.backend = self._resolve_backend()
        self._layers: dict[int, _LayerSchedule] = {}
        self._stats = GenerationStats(backend=self.backend)
        self._installed = False
        self._previous_impl: str | None = None
        if self.backend == "dense":
            # the control arm is only a control if the model really is running its
            # own attention. A leaked engine would leave the hook installed and the
            # dense numbers would silently be sparse ones -- the single most
            # expensive way for this measurement to be wrong. Any "sigmoid*" key
            # rather than just this module's: `sigmoid.triton.patch` registers
            # "sigmoid_topology_sparse" into the same registry, and a model left
            # patched by it would fail this check just as badly.
            current = self.model.config._attn_implementation
            if isinstance(current, str) and current.startswith("sigmoid"):
                raise RuntimeError(
                    f"this model still has a sigmoid attention hook installed "
                    f"({current!r}); restore it before building a dense control arm"
                )
        else:
            self._install()

    # -- setup -----------------------------------------------------------------

    def _head_dim(self) -> int:
        cfg = self.model.config
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim:
            return int(head_dim)
        return int(cfg.hidden_size // cfg.num_attention_heads)

    def _resolve_backend(self) -> str:
        want = self.config.backend
        if want == "dense":
            return "dense"
        usable = self._cuda and self._head_dim() in _KERNEL_HEAD_DIMS
        if usable:
            self._kernel = load_kernel()
        if want == "triton":
            if self._kernel is None:
                raise RuntimeError(
                    "backend='triton' needs CUDA, a head dim in "
                    f"{sorted(_KERNEL_HEAD_DIMS)}, and the merged kernel importable "
                    "(set SIGMOID_KERNELS_PATH)"
                )
            return "triton"
        if want == "torch":
            return "torch"
        return "triton" if self._kernel is not None else "torch"

    def _install(self) -> None:
        from transformers.modeling_utils import AttentionInterface

        AttentionInterface.register(_ATTENTION_NAME, _dispatch)
        _OWNERS[id(self.model.config)] = self
        self._previous_impl = self.model.config._attn_implementation
        self.model.config._attn_implementation = _ATTENTION_NAME
        self._installed = True

    def close(self) -> None:
        """Give the model its own attention back."""
        if self._installed:
            self.model.config._attn_implementation = self._previous_impl
            _OWNERS.pop(id(self.model.config), None)
            self._installed = False

    def __enter__(self) -> InferenceEngine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def max_context(self) -> int:
        if self.config.max_context:
            return self.config.max_context
        cfg = self.model.config
        return int(
            getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 2048)
        )

    # -- attention -------------------------------------------------------------

    def _attend(self, module, query, key, value):
        torch = self._torch
        stats = self._stats
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        block = self.config.block_size

        mark = time.perf_counter()
        state = self._layers.get(id(module))
        if state is None:
            state = self._layers[id(module)] = _LayerSchedule(self.config)
        state.observe(key)
        rows = None
        if seq_q == 1:
            query_block = (seq_k - 1) // block
            blocks = state.allowed(query_block)
            stats.selected_blocks += len(blocks)
            stats.dense_blocks += query_block + 1
        else:
            num_blocks = -(-seq_k // block)
            rows = [state.allowed(q) for q in range(num_blocks)]
            first = (seq_k - seq_q) // block
            stats.selected_blocks += sum(len(r) for r in rows[first:])
            stats.dense_blocks += sum(q + 1 for q in range(first, num_blocks))
        stats.schedule_ms += (time.perf_counter() - mark) * 1e3

        scale = _module_scale(module, value.shape[-1])
        if self.config.profile and self._cuda:
            torch.cuda.synchronize()
        mark = time.perf_counter()
        if seq_q == 1:
            out = self._decode_attention(state, query, key, value, blocks, query_block, scale)
        elif self.backend == "triton" and seq_q == seq_k:
            out = self._kernel_attention(query, key, value, rows, scale)
        else:
            out = self._masked_attention(query, key, value, rows, scale)
        if self.config.profile and self._cuda:
            torch.cuda.synchronize()
        stats.attention_ms += (time.perf_counter() - mark) * 1e3
        return out, None

    def _decode_attention(self, state, query, key, value, blocks, query_block, scale):
        """Gather the selected key blocks and attend over just those.

        The merged kernel cannot serve decode -- it requires q, k and v to be the
        same shape, and decode has one query row against the whole cache -- so
        this is the sparse decode path.

        ponytail: index_select materializes the selection, so a step reads the
        chosen keys and writes them again. That write is the whole cost at small
        density: at 12x64 heads and 65536 positions the gather alone measured
        2.58 ms against 0.82 ms for attention over the same slice. It still wins
        outright once density is low enough (dense 2.01 ms vs 0.48 ms total at
        that length), and the crossover against dense is ~16k positions for
        distilgpt2's geometry, ~8k for 32x128. A fused decode kernel reading the
        blocks in place would move that crossover down; nothing in the merged
        kernel does that yet, which is why this is a gather and not a kernel.
        """
        torch = self._torch
        idx = state.gather_index(blocks, query_block, key.shape[-2], torch, key.device)
        k_sel = key.index_select(-2, idx)
        v_sel = value.index_select(-2, idx)
        # every gathered position is <= the query position by construction, so the
        # causal mask is already applied and softmax needs no second one
        weights = torch.matmul(query, k_sel.transpose(-1, -2)) * scale
        weights = torch.softmax(weights.float(), dim=-1).type(value.dtype)
        return torch.matmul(weights, v_sel).transpose(1, 2)

    def _kernel_attention(self, query, key, value, rows, scale):
        """The merged Triton kernel over a CSR schedule.

        Two adaptations. The kernel folds in 1/sqrt(head_dim) itself, so q is
        pre-scaled by scale*sqrt(head_dim) -- exactly 1.0 for GPT-2, but a model
        with a layer-indexed scale would otherwise be silently wrong. And it
        requires a sequence that is a multiple of block_size, so short tails are
        zero-padded: padded keys sit at positions above every real query, and the
        kernel's own `k_pos <= offs_m` test drops them, so real rows are untouched.
        """
        torch = self._torch
        block = self.config.block_size
        seq, head_dim = query.shape[-2], query.shape[-1]
        padded = len(rows) * block
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(r) for r in rows])
        indices = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64)

        q = query * (scale * math.sqrt(head_dim))
        if padded != seq:
            pad = (0, 0, 0, padded - seq)
            q = torch.nn.functional.pad(q, pad)
            key = torch.nn.functional.pad(key, pad)
            value = torch.nn.functional.pad(value, pad)
        out = self._kernel.scheduled_attention(
            q.contiguous(),
            key.contiguous(),
            value.contiguous(),
            torch.as_tensor(offsets, device=q.device),
            torch.as_tensor(indices, device=q.device),
            block,
        )
        return out[:, :, :seq, :].transpose(1, 2)

    def _masked_attention(self, query, key, value, rows, scale):
        """Reference path: the same schedule as a boolean mask over full attention.

        Matches the merged `dense_masked_attention` semantics but batched over
        heads -- that reference is 2D only, so using it directly would mean a
        Python loop over batch*heads tiles per layer per step.
        """
        torch = self._torch
        block = self.config.block_size
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        device = query.device
        num_blocks = len(rows)
        table = torch.zeros((num_blocks, num_blocks), dtype=torch.bool, device=device)
        for q_block, allowed in enumerate(rows):
            table[q_block, torch.as_tensor(allowed, dtype=torch.long, device=device)] = True

        pos_q = torch.arange(seq_k - seq_q, seq_k, device=device)
        pos_k = torch.arange(seq_k, device=device)
        allow = table[pos_q // block][:, pos_k // block]
        allow &= pos_k[None, :] <= pos_q[:, None]

        weights = torch.matmul(query, key.transpose(-1, -2)).float() * scale
        weights = weights.masked_fill(~allow, float("-inf"))
        weights = torch.softmax(weights, dim=-1).type(value.dtype)
        return torch.matmul(weights, value).transpose(1, 2)

    # -- generation ------------------------------------------------------------

    def _sync(self) -> None:
        if self._cuda:
            self._torch.cuda.synchronize()

    def _elapsed(self, mark: float) -> float:
        self._sync()
        return (time.perf_counter() - mark) * 1e3

    def _encode(self, prompt: str):
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to generate from a string prompt")
        return self.tokenizer(prompt, return_tensors="pt").input_ids

    def _kv_bytes(self, past) -> int:
        total = 0
        for layer in getattr(past, "layers", []):
            for tensor in (layer.keys, layer.values):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return total

    def prefill(self, ids) -> Session:
        """Run the prompt once and keep its keys and values."""
        torch = self._torch
        tokens = self._as_ids(ids)
        if len(tokens) > self.max_context:
            raise ValueError(
                f"prompt of {len(tokens)} tokens exceeds max_context {self.max_context}"
            )
        self._layers.clear()
        stats = GenerationStats(backend=self.backend, context=len(tokens))
        self._stats = stats

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        self._sync()
        mark = time.perf_counter()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True)
        stats.prefill_ms = self._elapsed(mark)
        stats.kv_bytes = self._kv_bytes(out.past_key_values)

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        return Session(
            ids=list(tokens),
            prompt_len=len(tokens),
            past=out.past_key_values,
            logits=out.logits[0, -1].detach(),
            stats=stats,
            generator=generator,
        )

    def decode_step(self, session: Session, *, config: InferenceConfig | None = None) -> int:
        """Sample one token from the pending logits and append it to the cache."""
        torch = self._torch
        cfg = config or self.config
        token = self._sample(session.logits, cfg, session.generator)
        session.ids.append(token)

        self._stats = session.stats
        input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        self._sync()
        mark = time.perf_counter()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=session.past, use_cache=True)
        session.stats.decode_ms += self._elapsed(mark)
        session.stats.tokens += 1
        session.stats.context = len(session.ids)
        session.past = out.past_key_values
        session.logits = out.logits[0, -1].detach()
        session.stats.kv_bytes = self._kv_bytes(session.past)
        return token

    def _sample(self, logits, cfg: InferenceConfig, generator) -> int:
        torch = self._torch
        if not cfg.do_sample or cfg.temperature <= 0:
            return int(logits.argmax().item())
        scores = logits.float() / cfg.temperature
        if cfg.top_k:
            k = min(cfg.top_k, scores.numel())
            floor = torch.topk(scores, k).values[-1]
            scores = scores.masked_fill(scores < floor, float("-inf"))
        if cfg.top_p < 1.0:
            ordered, order = torch.sort(scores, descending=True)
            probs = torch.softmax(ordered, dim=-1)
            # keep the first token past the threshold, else top_p below the argmax
            # probability would leave nothing to sample from
            drop = (probs.cumsum(dim=-1) - probs) >= cfg.top_p
            ordered = ordered.masked_fill(drop, float("-inf"))
            scores = torch.full_like(scores, float("-inf")).scatter(0, order, ordered)
        probs = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    def _as_ids(self, ids) -> list[int]:
        if isinstance(ids, str):
            ids = self._encode(ids)
        torch = self._torch
        if isinstance(ids, torch.Tensor):
            if ids.ndim == 2:
                if ids.shape[0] != 1:
                    raise ValueError("batched generation is not supported; pass one sequence")
                ids = ids[0]
            return [int(t) for t in ids.tolist()]
        return [int(t) for t in ids]

    def generate(
        self,
        prompt: str | None = None,
        *,
        input_ids=None,
        max_new_tokens: int = 32,
        stream: bool = False,
        eos_token_id: int | None = None,
        **overrides,
    ):
        """Generate up to `max_new_tokens`; `stream=True` yields text as it lands.

        A streamed run sets `self.last_result` when the generator is exhausted,
        which is where its stats live.
        """
        source = prompt if prompt is not None else input_ids
        if source is None:
            raise ValueError("pass either a prompt or input_ids")
        cfg = replace(self.config, **overrides) if overrides else self.config
        stream_iter = self._run(self._as_ids(source), max_new_tokens, cfg, eos_token_id)
        if stream:
            return stream_iter
        for _ in stream_iter:
            pass
        return self.last_result

    def _run(self, tokens, max_new_tokens, cfg, eos_token_id) -> Iterator[str]:
        session = self.prefill(tokens)
        limit = min(max_new_tokens, self.max_context - len(tokens))
        for _ in range(max(limit, 0)):
            token = self.decode_step(session, config=cfg)
            piece = self.tokenizer.decode([token]) if self.tokenizer else ""
            yield piece
            if eos_token_id is not None and token == eos_token_id:
                break
        generated = session.generated
        text = self.tokenizer.decode(generated) if self.tokenizer else ""
        self.last_result = GenerationResult(ids=generated, text=text, stats=session.stats)

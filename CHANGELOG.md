# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Inference engine** (`sigmoid/inference.py`) — prefill + decode with a real KV
  cache, schedules grown by `IncrementalSalience` rather than rebuilt (fires once
  per 64 decode steps, bit-identical to a batch rebuild), streaming, and stats
  that report schedule time apart from attention time.
- **Kernel bridge** (`sigmoid/triton/attention.py`) — 2D/4D wrapper feeding
  sigmoid's schedule to the merged `triton-lang/kernels#22` kernel, with a torch
  fallback that runs with no GPU and no triton.
- **Model patching** (`sigmoid/triton/patch.py`) — swaps topology-sparse attention
  into HuggingFace models through the attention-interface registry; `restore()`
  is bit-identical.
- **Providers** (`sigmoid/providers/`) — 12 backends behind one interface
  (local, ollama, vllm, openai, anthropic, gemini, groq, deepseek, mistral,
  together, openrouter, xai). Keys are read from the environment per request and
  never stored; a redaction layer and six named tests check a planted canary key
  cannot reach a repr, str, or raised exception.
- **Hooks** (`sigmoid/hooks.py`) — nine hook points, veto, and failure isolation:
  a broken hook cannot stop the run or shadow a safety gate queued behind it.
- **Agent runtime** (`sigmoid/agent.py`, `sigmoid/robot.py`) — Hermes
  `<tool_call>` parsing that returns a result for every malformed case rather
  than raising, plus a robot bridge where an infeasible plan is a structured
  refusal the model can reason about. Three independent gates: schema validation
  rejecting unknown keys, `HookVeto`, and out-of-conversation confirmation for
  `dangerous=True` tools.

### Fixed
- Registering an attention implementation only in `ALL_ATTENTION_FUNCTIONS`
  makes transformers return **no mask at all**. Invisible when every layer is
  patched; with `layers=[0]` the logit error was 1.342e+02 against 1.877e-01,
  five layers attending to the future while emitting plausible text. Now also
  registered in `ALL_MASK_ATTENTION_FUNCTIONS`.
- A hook returning any truthy value replaced the context payload rather than
  only a dict — capable of handing an actuator the wrong value.

### Measured
- Dense backend reproduces HF `generate(do_sample=False)` 64/64 tokens; a
  saturated schedule through the sparse path matches token-for-token, KL < 1e-9.
- **Sparse attention does not pay** on anything distilgpt2 can run. Crossover is
  ~16k positions at 12×64, ~8k at 32×128; distilgpt2 caps at 1024, so every
  winning row is synthesized. Root cause: `index_select` gather costs 2.58 ms
  against 0.82 ms of attention at 65536.

### Fixed
- `h0_barcode` inherited the `csr_matrix` explicit-zero bug that was previously
  fixed only in the schedule path. Coincident points lost the zero-length edge
  between them, so the MST routed around it and reported them separating at a
  real distance: `[[0,0],[0,0],[1,0]]` gave merge heights `[1, 1]` instead of
  `[0, 1]`, inventing a component. Now builds the graph with
  `csgraph_from_dense(..., null_value=inf)` and restores the dropped zero
  heights by edge count. Bit-identical on clouds with no coincident points.

## [0.1.0] — 2026-07-29

First public release.

### Added
- **Engine** — `SigmoidWorldModel`: wrap any HF model, torch module or callable
  into a world model; `observe` / `imagine` / `predict_hidden` / `save` / `load`.
- **Encoder** — exact H₀ via minimum spanning tree, Hilbert-series embedding,
  Betti curves at normalized *and* absolute radii, spatial (`entity_dim`) and
  temporal clouds, robust median/MAD standardization, `cloud="auto"` selection.
- **Operator** — closed-form ridge fit, bilinear action conditioning,
  block-diagonal structure selected on held-out rollout, Banach fixed point,
  worst-case scalar certificate plus a covariance-propagated directional
  estimate with `residual_autocorr` as its validity discriminator.
- **Gate** — sheaf-consistency residual, whitened support term, and a
  spectral-entropy stalk catching both rank collapse and rank inflation.
- **Control** — `TopologicalMPC`: cross-entropy-method planning in latent space,
  topological cost functions (`beta0_cost`), and refusal when every candidate
  leaves the calibrated region.
- **Schedules** — causal CSR block schedules matching `triton-lang/kernels#22`
  bit-identically at 13×, plus `IncrementalSalience` for decode at 8.6–21.7×.
- **Multi-body** — Hamilton-tensor coupling with rank-R truncation.
- **Bench** — matched-budget ablations, promotion gates that report failures.
- Nine runnable studies in `examples/`, 57 tests, and a self-contained
  `docs/index.html` walkthrough (no build step, no external assets).

### Fixed
- Gram-identity distance cancellation in `h0_barcode` silently merged distinct
  points (offset 1e6, separation 1e-6 → merge heights `[1, 1, inf]`). Replaced
  with `cdist`; costs 23% on the encode path.
- `csr_matrix` dropped explicit zero-length edges, so coincident blocks never
  merged. Replaced with canonical Prim over a strict total order.

### Corrected
Four previously published claims, each traced to a missing control:
- Lorenz "~30%" was confounded by PCA rank; ~15% at k=1–4, and a loss at k=16.
- Kernel parity held only for random gaussians; failed 123/200 tie-heavy cases.
- The gate loses to Mahalanobis and PCA-reconstruction on OOD accuracy.
- "Fixed physical distance" was falsified and replaced by a scale-relative
  H₀-plateau clause plus a causal clause.

[Unreleased]: https://github.com/teerthsharma/sigmoid/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/teerthsharma/sigmoid/releases/tag/v0.1.0

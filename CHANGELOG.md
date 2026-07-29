# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

# Topology Speedup Map

## Candidate Construction Patterns

### Landmark Witness Sets

Object: token/key point cloud in normalized hidden space.

Construction: choose landmark blocks by farthest-point sampling on centroids, persistence-like separation, or Mapper cell representatives. Each query attends to sink tokens, local tokens, and witness landmarks that cover the historical space.

Implementation rule: produce a global sorted key index set and run dense attention over `[query_tokens, selected_key_tokens]` with a causal selected-key mask.

Risk: global landmarks can miss query-specific mass. Compare against same-count uniform and recency selectors.

### Mapper Cover Routing

Object: cover of key embeddings under scalar filters such as norm, time, entropy, or principal component.

Construction: overlapping intervals define cells; cells form a nerve graph. Select cells intersecting the query's filter neighborhood plus bridge cells with high occupancy or salience.

Implementation rule: selected keys are the union of query-relevant cells, local keys, and sinks. Batch by cells only if the resulting shape can be made kernel-friendly.

Risk: per-query unions can become ragged and slow. Favor block-level cell routing or two-stage candidate sets.

### Spectral Diffusion Gates

Object: token graph with edges from local adjacency and centroid similarity.

Construction: Laplacian diffusion scores identify stable low-frequency regions and high-drift regions. Reuse or prune only stable regions; keep high-drift regions dense.

Implementation rule: select blocks whose diffusion score exceeds a query-dependent threshold, with forced local/sink blocks.

Risk: eigensolvers are too expensive unless approximated incrementally or computed per long segment.

### Sheaf Consistency Filters

Object: heads/layers/shards as stalks, restriction maps as compatibility between local summaries.

Construction: high cohomology residual marks inconsistent regions that must remain dense; low residual regions can be compressed or shared.

Implementation rule: use as a safety gate over another sparse selector, not as the primary selector.

Risk: useful for correctness diagnostics but rarely a direct speedup alone.

### Topological Coupling Operators

Object: paired topological views of the same computation, such as query/key
blocks, E/H-like primal/dual fields, prefill/decode traces, or layer-to-layer
attention maps.

Construction: embed each view into a fixed-length topological vector, fit a
coupling operator `T` from source topology to target topology, and use the
dominant fixed point of `T` as a stable global structure. This follows the
Faraday-style pattern: barcode/Hilbert embeddings -> least-squares coupling
operator -> normalized power iteration.

Implementation rule: do expensive topology offline or per-capture, then use the
fixed-point vector as a cheap runtime gate over blocks, heads, or cache regions.
Do not compute Rips persistence in the hot attention path.

Risk: if `T` is learned from synthetic or too few captures, the fixed point may
describe the harness rather than model attention mass. Regularize `T` when the
sample count exceeds latent dimension or when eigenvalues become complex.

## Promotion Gates For TensorRT-LLM Sparse Attention

- Accuracy: relative L2 <= 0.05 and cosine >= 0.995 for screening; task-level generation quality for final evidence.
- Sparsity: selected keys, blocks, FLOPs, or bytes reduce attention work by at least 35% at long context; declare the budget unit and use the same budget for topology, locality, and random ablations.
- Speed: end-to-end speedup >= 1.15x against the optimized TensorRT-LLM/PyTorch attention path when available. A standalone PyTorch SDPA/matmul harness is screening-only unless it matches the target kernel path.
- Stability: pass at two long sequence lengths, preferably 8192 and 16384, and at head dim 64 and 128.
- Validity: include dense baseline, locality-only ablation, random same-budget ablation, and schedule-build overhead.
- Capture protocol: store real model Q/K/V captures under `.donotcommit`, record dtype, causal mask semantics, batch size, heads, head dim, sequence length, model/layer source, and whether the measurement is prefill, decode, kernel-only, or end-to-end.

## Failure Interpretation

- High speed and high error: topology selected structure, but not attention mass.
- Low error and low sparsity: selector is conservative; it needs a kernel-side win or stronger candidate compression.
- Beats dense CSR but loses to dense SDPA: reference implementation is not measuring deployable performance.
- Synthetic pass only: the theory is a screening lead, not PR evidence.
- Strong topology but slow schedule build: move topology to offline calibration
  and benchmark only the cached runtime gate.

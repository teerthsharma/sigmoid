---
name: discover-topology
description: "Automatically discover topology-first mathematical workflows for ML, sparse attention, kernel scheduling, compiler/runtime locality, graph structure, manifold geometry, persistent homology, Mapper/Reeb graphs, sheaves, spectral topology, hyperbolic geometry, and topological benchmarking. Use when Codex needs to turn topological math into falsifiable hypotheses, implementation plans, or PR-ready performance experiments."
---

# Topology Research Discovery

Use this skill when topology is not decoration but the mechanism behind a proposed speedup, compression rule, scheduler, graph transform, or approximation.

## Core Workflow

1. State the topological object:
   - Point cloud, graph, filtration, cover, nerve, sheaf, bundle, manifold, metric space, or dynamical trajectory.
2. State the invariant or construction:
   - Persistence, Mapper/Reeb graph, Betti proxy, geodesic cover, Laplacian spectrum, curvature, cohomology obstruction, or transport map.
3. Convert it into an implementable rule:
   - Candidate selection, block schedule, cache reuse decision, partition, routing index, pruning gate, or kernel layout.
4. Define the baseline and null model:
   - Dense exact baseline, random schedule with same budget, locality-only schedule, and prior best heuristic.
5. Define falsification gates before running:
   - Accuracy gate, speed gate, memory gate, build-overhead gate, and workload coverage gate.
6. Interpret failures structurally:
   - Too lossy, too slow, unstable under scale, topology proxy not correlated with attention mass, schedule overhead dominates, or kernel shape mismatch.

## Sparse Attention Pattern

For sparse attention, require every topology theory to answer:

- What is the token/key space metric?
- Which topological summary predicts attention mass?
- Is the schedule row-wise, block-wise, global subset, or two-stage?
- Can the selected shape run on a real fast kernel path?
- Does it beat dense SDPA/FlashAttention, not only a Python sparse loop?
- Does it preserve quality at long context on real Q/K/V captures?

Prefer two-stage designs when Python CSR is too slow:

1. Use topology to cheaply select a small key/value candidate set.
2. Run dense attention inside the candidate set with ordinary matmul or SDPA-friendly shapes.
3. Compare against dense causal attention and a same-budget non-topological selector.

## Topology Families

- Persistent homology: use filtrations to rank stable structures; reject if persistence only tracks norm or recency.
- Mapper/Reeb graphs: cover latent/token space and route through overlapping cells; reject if cell membership is expensive or not kernel-friendly.
- Hyperbolic/cover trees: exploit hierarchy and boundary effects for long-context retrieval; reject if representatives lose local evidence.
- Spectral topology: use graph Laplacian eigenmodes, diffusion, or drift to detect reusable structure; reject if eigen/schedule cost exceeds saved compute.
- Sheaf/cohomology: model consistency across layers, heads, workers, or KV shards; reject if it only diagnoses errors and does not produce a faster execution rule.
- Nerves/covers: turn overlapping metric balls into sparse candidate sets; reject if cover size grows like dense attention.

## Evidence Rules

- Keep scratch code, captures, logs, benchmark outputs, and temporary artifacts under the user-approved scratch directory. For TensorRT-LLM exploratory work in this workspace, use `C:\Users\seal\Documents\GitHub\TensorRT-LLM\.donotcommit` unless the user says otherwise.
- Always report speedup against the fastest relevant dense baseline available in the harness.
- Separate schedule construction time from attention execution time; for PR claims, also report end-to-end.
- Use real model Q/K/V captures when possible; synthetic results are only screening evidence.
- Include same-budget random and locality-only ablations.
- Require a Pareto frontier, not a single lucky point.
- Do not call a result PR-ready if it only beats an intentionally slow reference implementation.

## Benchmark Discipline

- Define the budget unit before testing: selected tokens, selected blocks, estimated FLOPs, bytes moved, or kernel-compatible dense shape. Use the same unit for topology, random, and locality ablations.
- For TensorRT-LLM claims, prefer the actual optimized PyTorch/TensorRT-LLM attention path. Use a standalone PyTorch SDPA/matmul harness only as a screening proxy and label it as such.
- Record dtype, device, GPU model, sequence length, head dimension, batch/head grouping, causal mask semantics, schedule-build time, attention time, and end-to-end time.
- Capture real Q/K/V from model execution when possible and store captures under `.donotcommit`; if using synthetic tensors, mark results as screening-only and avoid PR-ready language.

## Reference Loading

Read `references/topology-speedup-map.md` when the task asks for new topology theories, sparse-attention speedups, or PR-readiness planning.

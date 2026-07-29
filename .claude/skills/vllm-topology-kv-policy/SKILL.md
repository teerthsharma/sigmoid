---
name: vllm-topology-kv-policy
description: Use when working in vllm-project/vllm on topology-aware KV-cache policy, spectral or sparse attention scheduling, prefix/cache reuse, block selection, candidate pruning, or research-to-benchmark contribution ideas.
---

# Vllm Topology Kv Policy

## Overview

Use this skill to turn topology/spectral/sparse KV-cache ideas for vLLM into a narrow, falsifiable, locally testable policy. The goal is not to make a grand theory sound impressive; the goal is to produce an opt-in rule that survives vLLM review, correctness checks, and performance baselines.

## Core Contract

Every topology KV-cache idea must specify these items before implementation:

- Topology object: token stream, KV blocks, prefix graph, request cohort, cache block pool, attention candidate set, or scheduler trace.
- Invariant: spectral gap, cover/nerve membership, block centroid stability, prefix fingerprint, persistence proxy, locality scaffold, or drift score.
- Implementation rule: select, merge, retain, evict, bypass, fingerprint, or reorder concrete token/block candidates.
- Baseline: current vLLM behavior on the same path.
- Null model: same-budget random, locality-only, and learned/top-k-only selector when applicable.
- Falsification gates: correctness, latency, memory, schedule-build overhead, end-to-end serving behavior, and workload coverage.

The feature must be disabled by default. The disabled path must preserve existing vLLM behavior and must be easy to review.

Do not call a topology KV idea PR-ready unless it passes the falsification gates against vLLM-native baselines, not only a standalone Python prototype.

## Workflow

1. Read vLLM context first:
   - Read `AGENTS.md`.
   - Read the nearest design docs, usually under `docs/design/`.
   - Search current work before proposing anything:

```bash
gh issue list --repo vllm-project/vllm --state open --search "<area keywords>"
gh pr list --repo vllm-project/vllm --state open --search "<area keywords>"
```

2. Define the topology object.
   - Name the concrete vLLM surface: `vllm/v1`, `kv_offload`, prefix caching, attention backend, scheduler, connector, kernel, or benchmark.
   - State the coordinate system: token offsets, block ids, physical page ids, request-local row offsets, or prefix hash keys.

3. Define the invariant.
   - Good: stable prefix fingerprint, spectral block centroid, sink/local/witness scaffold, candidate retention ratio, drift threshold.
   - Bad: "topology says this is important" without a measured value, budget, or coordinate contract.

4. Convert the invariant into an implementation rule.
   - The rule must produce concrete tokens, blocks, or cache actions.
   - Prefer an isolated policy helper before touching a hot runtime path.
   - Keep the policy opt-in through config/env wiring and disabled by default.

5. Define baseline and null model.
   - Baseline: unmodified vLLM on the same model/path.
   - Null model: same-budget random selector.
   - Null model: same-budget locality-only selector.
   - Null model: existing learned/top-k candidate set if the path already has one.

6. Define falsification gates before coding.
   - Correctness: exact output equality where possible; otherwise accepted tolerances and task-level evals.
   - Latency: TTFT, ITL, throughput, and tail latency when serving behavior changes.
   - Memory: KV bytes, CPU/GPU transfer bytes, cache hit/read/write tokens, and block churn.
   - Overhead: separate schedule-build overhead from attention/cache execution.
   - Coverage: at least two sequence lengths or request shapes for long-context claims.

## vLLM Guardrails

- Follow `AGENTS.md` before any contribution planning.
- Use `uv` and the repo `.venv`; do not use bare `pip` or system Python.
- Avoid pure-agent PRs. The human submitter must understand and defend the change.
- Do not create low-value busywork PRs.
- Keep large architectural work behind an issue/RFC if it exceeds vLLM's normal PR size expectations.
- Keep user-facing changes documented under `docs/`.
- Do not put one-off kernel benchmarks in tests/. Put kernel performance work in `benchmarks/kernels` or the nearest existing benchmark suite; prove correctness in pytest.

## Runtime Requirements

- Disabled by default: if config/env is absent, the original candidate set or cache policy returns unchanged.
- Dense fallback: masked, unsupported, compiling, short-sequence, unknown-coordinate, and mismatch paths must fall back to the current dense or unmodified vLLM path; document this as the dense fallback contract.
- Coordinate contract: document whether returned values are absolute token indices, row-relative offsets, cache block ids, physical page ids, or prefix hashes.
- Fail closed on coordinate mismatch: raise in tests and experimental mode; use explicit fallback only when that behavior is documented and observable.
- CUDA hot path: do not introduce GPU-to-CPU copies, `.item()`, `.tolist()`, per-row Python loops, or Python sync in the CUDA hot path unless explicitly benchmarked and gated.
- Observability: counters must describe real behavior, such as learned tokens kept, learned tokens evicted, structural tokens injected, fallback count, and schedule-build time.
- Concurrency: do not store per-request fingerprints or counters in shared global/module state unless semantics are explicitly "last call only" and non-concurrent.

## Test Evidence

Use test-driven-development for implementation work. Write the failing test first, watch it fail, implement minimally, then rerun.

Minimum evidence for an experimental policy:

- Unit tests for selector math, coordinate validation, padding layout, duplicate removal, budget caps, disabled passthrough, and fallback.
- Integration or smoke test through the real vLLM surface if the policy is wired into runtime code.
- Benchmark separate from correctness tests.
- For model-affecting changes, model evals or `vllm bench` results with exact command lines.
- Local validation commands using `.venv/bin/python -m pytest ...`, `pre-commit run ...`, or the repo-approved equivalent.

Screening-only results must be labeled screening-only. Synthetic Q/K/V, Python reference attention, or standalone sparse loops do not establish PR-ready performance.

## Failure Modes

- High speed, bad output: the invariant selected structure but not attention/cache value.
- Low error, no speed: the policy is too conservative or schedule-build overhead dominates.
- Beats Python reference, loses to vLLM baseline: benchmark used the wrong baseline.
- Works on CPU tests, no GPU evidence: acceptable only for policy validation, not kernel-speed claims.
- CUDA enabled but silently no-op: add an observable fallback counter or warning.
- Topology candidates evict all learned/top-k candidates: cap structural injection and report learned retention.
- Coordinate ambiguity: stop and write the contract before adding code.
- Tail latency regresses: report it plainly; aggregate throughput alone is not enough.

## Common Mistakes

| Mistake | Correction |
| --- | --- |
| Starting with a new attention stack | Start with a small disabled policy helper and tests. |
| Vague topological language | Name the topology object, invariant, and implementation rule. |
| No null model | Add same-budget random and locality-only ablations. |
| Python loop in CUDA hot path | Keep it off the hot path or benchmark and gate it explicitly. |
| Unclear token/block coordinates | Write the coordinate contract and enforce it in tests. |
| Benchmark in `tests/` | Put performance scripts under `benchmarks/kernels` or an existing benchmark area. |
| PR-ready claim from synthetic data | Label it screening-only until real vLLM baselines and evals pass. |

## Quick Review Checklist

- [ ] Read `AGENTS.md` and relevant vLLM design docs.
- [ ] Checked duplicate issues and PRs in `vllm-project/vllm`.
- [ ] Defined topology object -> invariant -> implementation rule -> baseline -> null model -> falsification gates.
- [ ] Feature is disabled by default.
- [ ] Dense fallback or unmodified fallback is explicit.
- [ ] Coordinate contract is documented and tested.
- [ ] CUDA hot path has no unbenchmarked Python sync.
- [ ] Tests cover disabled passthrough, mismatch, budget, padding, and fallback.
- [ ] Benchmarks compare against current vLLM, same-budget random, and locality-only baselines.
- [ ] Claims distinguish screening evidence from PR-ready evidence.

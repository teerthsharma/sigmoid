---
name: "research-readme"
description: "Write or rewrite a systems-research README in the house style used across topo-asm, vec-simd, pgtable-asm, aether-link, Aether-Lang and EPSILON-PHASE — Abstract, Background with a Prior Art comparison table, Theoretical Foundation in LaTeX, a measured-numbers table with reproduction commands, Quick Start, and MSRV/architecture requirements. Use this whenever the user asks for a README, project writeup, abstract, \"make this look like a paper\", documentation for a new crate or library, or wants an existing README brought up to the standard of their other repos — including for assembly, kernel, compiler, numerics, or ML-systems projects."
---

# Research README

Write project documentation that reads like a systems paper: a claim, the mechanism behind it, and numbers that a skeptic can reproduce.

The audience is a reader who could plausibly become a user, a reviewer, or a hiring manager, and who will decide within thirty seconds whether the project is serious. What earns that decision is specificity — a named baseline, a measured number with units, an honest limitation. What loses it is adjectives.

## Structure

Follow this order. It mirrors the arc of a paper: claim → why the claim is non-obvious → how it works → proof → how to use it.

```markdown
# NAME: <One-line technical subtitle that names the domain and the method>

**<A single bold sentence a reader can quote back. States what it does and what makes
it different — not a slogan.>**
`https://github.com/teerthsharma/<repo>`

---

## Abstract

## Background
### Why <the unconventional choice>?
### Prior Art

## Theoretical Foundation

## Implementation
### <Component 1>
### <Component 2>

## Results

## Quick Start

## Requirements

## Limitations

## License
```

## Section-by-section

### Abstract

One dense paragraph, 120–200 words. It must contain, in this order: what we present, the core insight that makes it work, the method, the measured result against a *named* baseline, and how it is callable from other languages. No forward references, no "in this document we will".

The core insight sentence is the one that matters. It should be falsifiable and slightly surprising — the reason this approach beats the obvious one. "The bottleneck in persistent homology computation is pairwise distance comparison across N(N−1)/2 point pairs, and this is embarrassingly parallel" is an insight. "We use SIMD for speed" is not.

### Background

Open with `### Why <choice>?` — assembly instead of C, POVM instead of an ML model, stochastic resonance instead of a larger dtype. Justify it in mechanism terms: what the conventional approach cannot express, and what specifically is left on the table. Three or four short paragraphs, each with a bolded lead-in naming the concrete cost being paid (`**Cache layout matters.**`, `**No abstraction overhead.**`).

Then `### Prior Art` as a table. This is the highest-value table in the document and the easiest to get wrong. Every row must be a real, currently-maintained system, with its actual language, and numbers from a run you actually did or a published figure you cite. Include the column that shows where the alternatives are *better* if such a column exists — a table where the project wins every cell reads as marketing.

| System | Language | N=1k | N=10k | SIMD width |
|---|---|---|---|---|
| ripser | C++/Python | 0.4 s | 48 s | 256-bit AVX2 |
| GUDHI | C++ | 0.12 s | 14 s | 256-bit AVX2 |
| `this` | x86_64 asm | 0.018 s | 1.1 s | 512-bit AVX-512 |

### Theoretical Foundation

The mathematics the code actually implements, in LaTeX display blocks, numbered so the implementation section can reference them. Define every symbol on first use in prose immediately below the block.

```markdown
### 2) Stochastic resonance injection

The deterministic signal $x_t$ is mixed with normalized noise using adaptive gain $g_t$:

$$
x_t^{(sr)} = x_t + g_t z_t
$$

where $z_t$ is the centered, unit-variance noise from (1) and $g_t$ evolves per (3).
```

Only include equations the code implements. A derivation that does not appear in the source is a liability — a reader who checks will find the gap. Where an equation is approximated in code (a fast reciprocal, a Padé expansion, a truncated series), state the approximation and its error bound right there.

### Implementation

Per component: what it computes, the cost in the units that matter (nanoseconds, cycles, cache lines, allocations), and the one non-obvious engineering decision. For assembly projects include the actual inner loop with per-instruction comments — it is the most convincing artifact in the document.

```asm
vmovapd     zmm0, [rax]          ; load a[0..7]
vmovapd     zmm1, [rbx]          ; load b[0..7]
vfmadd231pd zmm2, zmm0, zmm1     ; zmm2 += a*b, single rounding
```

### Results

A table with explicit units and a Notes column giving the measurement condition. Every number here must be reproducible with a command stated in the document.

| Metric | Value | Notes |
|---|---|---|
| Decision latency | **~18.1 ns** | Full `process_io_cycle` loop |
| Jitter (P99 − P50) | **< 1 ns** | Criterion, 100 samples, core-pinned |
| `fast_atan` error | **≤ 1 ULP** | vs. `libm::atanf` over the full domain |

Directly beneath the table, state the substrate: CPU model, whether the core was pinned, turbo state, toolchain version, and the exact command (`cargo bench --bench io_cycle`). Then give one paragraph of *context* — the number that makes the result meaningful. "NVMe hardware latency is ~10–25 µs, so the decision overhead is ~1000× smaller than the I/O it schedules" is what converts a benchmark into an argument.

### Quick Start

A dependency line with a real published version, then the shortest program that does something true. It must compile against the current API — a README example that no longer typechecks is the single most common defect in this genre, so check it against the actual source before publishing.

### Requirements

MSRV, target architectures, required CPU features and the runtime-detection story, OS constraints, and any nightly features. Be exact: "x86_64 with AVX-512F and AVX-512DQ; falls back to an AVX2 path via `is_x86_feature_detected!`" rather than "modern x86".

### Limitations

A real section, not a disclaimer. Where the approach loses, what is unimplemented, what is untested, what would break at scale. This section is what makes the rest of the document believable — a reader who finds an unstated limitation themselves discounts every other claim.

### License

Close with the license and the attribution line used across these repos:

```markdown
*Invented by [Teerth Sharma](https://teerthsharma.vercel.app)*
```

## Standards to hold

- **Every number is measured.** If it was not run, it does not go in the table. Projected, estimated, and theoretical-peak figures must be labelled as such in the same cell.
- **Every baseline is named and versioned.** "Faster than the Python version" is unfalsifiable. "12× faster than ripser 0.6.4" is a claim someone can check — which is the point.
- **No unqualified superlatives.** "Fastest", "state of the art", and "production-ready" require either a citation or deletion.
- **Distinguish inspiration from implementation.** "Quantum-inspired adaptive measurement (POVM formalism)" is accurate and defensible. "Quantum" without the qualifier is not, and a reviewer will notice.
- **Badges must be live.** crates.io version, MSRV, license, CI status — a badge pointing at a nonexistent crate or a red build is worse than no badge.
- **Prefer prose to bullets** in Abstract, Background, and Limitations. Bullets are for enumerable facts: requirements, capabilities, metrics.

## Working method

When rewriting an existing README, read the source first — the public API surface, the actual benchmark harness, `Cargo.toml`/`pyproject.toml` for versions and MSRV, and the CI workflow. Then check every existing claim in the README against what the code does. Report any claim that the source does not support rather than silently carrying it forward; inherited overclaims are how a good project acquires a bad reputation.

Where a number is needed and no benchmark exists, say so explicitly and leave a marked placeholder rather than inventing a plausible figure. A `TODO: measure` is a small embarrassment; a fabricated benchmark is a large one.


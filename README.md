<h1 align="center">Sigmoid — Topological World-Model Inference Engine</h1>

<p align="center">
  <strong>Any model becomes a world model. Its weights are never touched.</strong><br>
  Persistent homology → Hilbert-series embedding → coupling operator → Banach fixed point.<br>
  One imagined step costs <strong>97 µs</strong> against an <strong>86 ms</strong> forward pass.
</p>

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong> ·
  <a href="https://github.com/teerthsharma/sigmoid">github.com/teerthsharma/sigmoid</a> ·
  <em>teerthsharma@outlook.com</em>
</p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/teerthsharma/sigmoid/ci.yml?branch=main&label=CI&style=flat-square" alt="CI"></a>
  <a href="https://teerthsharma.github.io/sigmoid/"><img src="https://img.shields.io/badge/docs-github.io-blue?style=flat-square" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&color=00aaff" alt="MIT"></a>
  <a href="#12-validation"><img src="https://img.shields.io/badge/tests-57%20passing-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/core%20deps-numpy%20%2B%20scipy-lightgrey?style=flat-square" alt="Deps"></a>
  <a href="#9-when-topology-pays"><img src="https://img.shields.io/badge/claims-falsification--tested-orange?style=flat-square" alt="Falsification tested"></a>
  <a href="SIGMOID.md"><img src="https://img.shields.io/badge/negatives-published-red?style=flat-square" alt="Negatives published"></a>
</p>

---

## Abstract

We present **Sigmoid**, an inference engine that converts an arbitrary trained
model into a world model without modifying, retraining, or differentiating
through it. Given any producer of hidden activations — a transformer, a policy
network, a simulator, a bare callable — Sigmoid encodes each window of
activations as a persistent-homology barcode, embeds that barcode into a
fixed-length vector through the numerator of its Hilbert series, and learns a
linear coupling operator `T` advancing the embedding one step. Rolling `T`
forward replaces the model in the prediction loop at roughly **880× lower cost
per step**. The operator admits a Banach contraction certificate when its
spectral norm falls below unity, and a sheaf-theoretic consistency gate detects
when an imagined trajectory has left the calibrated manifold — computed without
invoking the model, which is the only regime in which such a check is possible
at all.

The construction is a transplant. The pattern `barcode → Hilbert embedding →
least-squares operator → Banach fixed point` was developed for electromagnetic
cavity coupling in **Faraday** and extended to N bodies in the **Hamilton
Tensor**; here it is carried onto model activations. We validate on chaotic
attractors, an S²–Vietoris–Rips entity corpus derived from
[`google-deepmind/mujoco#3396`](https://github.com/google-deepmind/mujoco/pull/3396),
granular contact physics, closed-loop multi-robot control, and the residual
stream of a transformer language model. Sigmoid reads topological state that no
equal-dimension linear method recovers — island partitions at **0.855** against
**0.469**, contact components at **0.778** against **0.104** — and its H₀
routine reproduces the MuJoCo disjoint-set island partition **exactly** and the
merged Triton kernel salience of
[`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22)
**bit-identically at 13×** the speed.

We further report, in full, four claims of our own that adversarial review
falsified, and the two-clause condition that replaced our original hypothesis.

**Keywords:** world models, persistent homology, topological data analysis,
Banach fixed-point theorem, Hilbert series, coupling operators, sheaf
cohomology, model-predictive control, sparse attention, minimum spanning tree,
out-of-distribution detection, inference engines

---

## Table of Contents

| § | Section |
|---|---|
| [1](#1-introduction) | Introduction |
| [2](#2-architecture) | Architecture |
| [3](#3-the-topological-state-σ) | The topological state Σ |
| [4](#4-the-coupling-operator-t) | The coupling operator T |
| [5](#5-the-banach-certificate) | The Banach certificate |
| [6](#6-the-sheaf-consistency-gate) | The sheaf consistency gate |
| [7](#7-multi-body-coupling) | Multi-body coupling |
| [8](#8-results) | Results |
| [9](#9-when-topology-pays) | When topology pays |
| [10](#10-control-of-robots) | Control of robots |
| [11](#11-sparse-attention-schedules) | Sparse attention schedules |
| [12](#12-validation) | Validation |
| [13](#13-installation-and-use) | Installation and use |
| [14](#14-what-we-got-wrong) | What we got wrong |
| [15](#15-provenance) | Provenance |

---

## 1. Introduction

A trained network already computes a rich internal state. What it does not
provide is a *dynamics* over that state which can be evaluated without paying
for the network. Every forward pass reconstructs the state from scratch; nothing
in the architecture exposes "what happens next" as an object one can iterate
cheaply.

World-model research generally answers this by training a second network — a
latent dynamics model, fit by gradient descent, with the usual consequences:
training cost, hyperparameters, and rollouts that diverge without warning
because nothing constrains them.

Sigmoid takes the opposite route. The state is not learned; it is *measured*, as
the persistent homology of the activation window. The dynamics is not trained;
it is *solved*, in closed form, by ridge regression. And the trustworthiness of a
rollout is not estimated; it is *bounded*, by the Banach fixed-point theorem
where the operator contracts, and detected by a cohomological obstruction where
it does not.

Three properties follow that a trained dynamics model does not have:

1. **Calibration is convex.** One ridge solve. No gradients, no epochs, no
   learning rate. Fitting on 8226 transitions takes seconds.
2. **Rollout cost is a matvec.** 97 µs against an 86 ms forward pass.
3. **The self-check needs no model.** The gate compares an imagined state
   against *itself*, which is the only check available when the ground truth is
   precisely the thing you declined to compute.

---

## 2. Architecture

```
   model ──capture──▶ activations ──Σ──▶ z = [ψ ; u] ──T──▶ imagined rollout
                          (T, D)          │                        │
                                     barcode + PCA           sheaf gate:
                                                          still trustworthy?
```

Two tiers, following the discipline inherited from the `discover-topology`
methodology: **expensive topology at calibration, arithmetic at runtime.**
Persistent homology never enters the hot path. H₀ is exact via minimum spanning
tree; H₁ requires a Rips complex and is calibration-only.

| Module | Role |
|---|---|
| [`state.py`](sigmoid/state.py) | Σ — barcodes, Hilbert embedding, robust standardization, decoder, automatic cloud/scale selection |
| [`operator.py`](sigmoid/operator.py) | T — ridge fit, spectral projection, Banach fixed point, scalar and directional certificates |
| [`sheaf.py`](sigmoid/sheaf.py) | The grounding gate — restriction map, support term, spectral-entropy stalk |
| [`engine.py`](sigmoid/engine.py) | `SigmoidWorldModel` — fit / observe / imagine / save / load |
| [`control.py`](sigmoid/control.py) | CEM-MPC planner, topological costs, principled refusal |
| [`schedule.py`](sigmoid/schedule.py) | Causal CSR block schedules, batch and incremental |
| [`nbody.py`](sigmoid/nbody.py) | Hamilton-tensor multi-entity coupling with rank-R truncation |
| [`adapters.py`](sigmoid/adapters.py) | Capture from HuggingFace models, torch modules, callables |
| [`bench.py`](sigmoid/bench.py) | Matched-budget ablations and promotion gates |
| [`cli.py`](sigmoid/cli.py) | `python -m sigmoid fit \| roll \| bench` |

---

## 3. The topological state Σ

Given a hidden trajectory `h₁…h_T ∈ ℝᴰ` and window `W`, the state point cloud at
time `t` is `X_t = {h_{t−W+1}, …, h_t}`.

### 3.1 Robust standardization

Each coordinate is centred by its median and scaled by `1.4826 · MAD`.

This is required, not cosmetic. Transformer residual streams carry *massive
activations*: measured on distilgpt2, the largest per-dimension standard
deviation was **22.1** against a median of **0.308** — a **72× ratio**
concentrated in roughly five channels. Without this step, Euclidean distances in
`X_t`, and therefore the entire barcode, are a function of five outlier
channels. The "topological" feature would measure activation magnitude and
nothing else.

Median and MAD rather than mean and standard deviation, because the outliers
would otherwise set the very scale intended to tame them.

### 3.2 H₀ exactly, without a simplicial complex

For a Vietoris–Rips filtration the H₀ death times **are** the single-linkage
merge heights — the edge weights of the Euclidean minimum spanning tree:

```
B₀(X) = { (0, w) : w ∈ MST(X) } ∪ { (0, ∞) }
```

This is exact, not an approximation, and costs `O(W²D)` with no complex
constructed anywhere. The same identity is what makes the sparse-attention
schedule of §11 run 13× faster than its reference implementation.

### 3.3 The Hilbert-series embedding

A barcode has no fixed length, so no linear operator can act on it. The
numerator of its Hilbert series supplies one — births contribute positively,
deaths negatively:

```
N(s) = Σᵢ s^{bᵢ} − Σᵢ s^{dᵢ},    c_k = #{births in bin k} − #{deaths in bin k}
```

This step is what makes a *linear* operator on topology possible at all. A
sanity identity holds in the test suite: `N(1) = Σ c_k` equals the number of
essential classes.

### 3.4 The state is hybrid

```
z = [ ψ ; u ]
```

- **ψ** — Hilbert coefficients, Betti curves at normalized *and* absolute radii,
  scale-invariant geometry. Carries *regime*: how many components, how spread,
  how confined.
- **u** — whitened PCA coordinates of the standardized activation. Carries
  *content*: the part that can be decoded back into activations.

ψ alone is not invertible — topology forgets coordinates — so a ψ-only world
model could not emit predicted activations. u alone is a plain linear
autoencoder, which is precisely the null model any topological claim must beat.
Keeping both channels separate is what makes the claim falsifiable at matched
dimension.

### 3.5 Two specification choices that silently decide everything

Two binary choices sit upstream of every persistence pipeline, are almost never
stated in published work, and determine the outcome:

**Which cloud.** Temporal (`entity_dim=0`) treats the window of successive state
vectors as the point cloud, describing the shape of a *trajectory segment*.
Spatial (`entity_dim=d`) reshapes each observation into its entities, describing
their *arrangement at one instant*. These are different objects. On an entity
corpus whose ground truth is H₀ of the entity cloud, the temporal encoder scored
**0.386 against a 0.487 majority baseline** — actively uninformative.

**Which scale.** The barcode divides filtration values by the cloud diameter,
buying scale invariance and discarding scale. When the quantity of interest
lives at a fixed physical distance — a contact threshold, a constraint radius —
that trade is backwards. Reading β₀ at the corresponding absolute radius instead
reproduced the ground-truth component count with **correlation 1.0000 and exact
agreement on every frame**. Same barcode, same code; the only difference was
whether the diameter had been divided out.

Neither error raises an exception. Both produce a ψ with healthy variance,
sensible standardization, and no information. `SigmoidConfig(cloud="auto")`
selects between them without labels, by psi self-predictability corrected for
the overlap floor that consecutive windows induce.

---

## 4. The coupling operator T

Ridge regression in closed form on a lifted feature vector:

```
W = argmin Σ_t ‖ W·φ(z_t, a_t) − z_{t+1} ‖² + λ‖W‖²_F
  = (XᵀX + λI)⁻¹ XᵀY
```

with the bilinear action lift `φ(z, a) = [z ; a ⊗ z ; a ; 1]`, so that

```
W·φ = T₀z + Σ_k a_k T_k z + Ba + c
```

The problem is convex and small. There is nothing to iterate.

**Block structure is selected, not assumed.** Whether ψ and u evolve jointly or
independently is a property of the wrapped model, not of this library:

| System | Coupled | Block-diagonal |
|---|---|---|
| Lorenz, k=1 NRMSE | **0.052** | 0.073 |
| distilgpt2, R² for `u_{t+1}` | 0.145 | **0.168** |

So `block_diagonal="auto"` fits both and keeps whichever wins on a held-out
**16-step rollout** — not one step, because at one step the two structures agreed
to within 0.06% while at sixteen they separated cleanly. The benefit of letting
topology inform content compounds across a rollout, which is what a world model
is for.

---

## 5. The Banach certificate

Let `A` be the state→state block and `ρ = σ_max(A)`. If `ρ < 1` the rollout map
is a contraction, so by the Banach fixed-point theorem it has a unique fixed
point `z*` with `‖z_n − z*‖ ≤ ρⁿ‖z₀ − z*‖`. With one-step residual `ε`:

```
E(n) ≤ ε · (1 − ρⁿ)/(1 − ρ)   →   ε/(1−ρ)   as n → ∞
```

`safe_horizon(τ)` inverts this: the largest `k` whose guaranteed error stays
under `τ`. The fixed point is obtained by solving `(I − A)z = b` directly, with a
Banach iteration run alongside so convergence is **measured** rather than
asserted — the Faraday burn pattern.

> **Contraction is reported, never forced.** An early version clipped ρ to 0.995
> to guarantee a certificate. Measured on Lorenz, that clipping degraded
> one-step NRMSE from **0.067 to 0.317**. Chaotic systems have ρ > 1 by
> definition; a positive Lyapunov exponent *is* expansion. Clipping buys a
> presentable certificate by misreporting the dynamics, so `rho_max` defaults to
> `None` and setting it is an explicit, priced choice.

**Directional estimate.** The scalar bound assumes the worst-case error
direction compounds maximally at every step. Propagating the residual covariance
instead — `C_n = A C_{n−1} Aᵀ + Σ` — is up to **12.6× tighter** and matches
measured error to 1% when residuals are white. It is an *estimate*, not a bound:
it under-bounds by **2.4×** under AR(1) residuals and **4.6×** on Lorenz.
`residual_autocorr` reports which regime you are in (0.000 when trustworthy,
0.81–0.94 when not). It is deliberately not wired into the promotion gate,
because it would pass everywhere — including where it is wrong.

---

## 6. The sheaf consistency gate

An imagined state is a prediction, so it cannot be checked against truth: the
truth is exactly what was declined. It can be checked against **itself**.

Over each time index place two stalks:

```
F(topo)   = ψ,   the topological channel
F(linear) = u,   the linear channel
```

Both are local views of one activation window, so a restriction map
`R : F(linear) → F(topo)` exists on real data and is learned by ridge. On a
genuine state the section glues, `‖Ru − ψ‖ ≈ 0`. Nothing in the least-squares
fit forces an *imagined* ψ̂ to remain the topological signature of an imagined û,
so

```
r = ‖ R·û − ψ̂ ‖
```

is a first cohomology obstruction to gluing the two local sections, and it grows
precisely when the rollout has left the data manifold. Thresholds are set at
quantiles of the calibration residual, so a score is in units of *how unusual
this would have been during calibration*.

A **spectral-entropy stalk** covers what those two miss. Effective rank
`exp(−Σ p log p)` over the window's normalized squared singular values is
anomalously *low* under repetition and anomalously *high* under noise, so a
two-sided log deviation catches both tails. Measured on scrambled activations —
identical mean, covariance, energy and support, only temporal structure
destroyed — the two-term gate fired **0/60** while the stalk fired **60/60**.

---

## 7. Multi-body coupling

For N interacting streams, `Ψ = [ψ₁; …; ψ_N]` evolves under a block operator `M`
whose off-diagonal blocks `M_ij` are the inter-body coupling. Fitting `M` densely
costs `O(N²d²)` — the `d^N` blow-up in disguise — so a rank-R SVD truncation
gives an `O(NdR)` operator retaining the dominant coupling modes, following the
Hamilton Tensor's subspace-iteration argument. `coupling_strength()` reports
block Frobenius norms; the test suite verifies a driver/follower pair is
recovered from data alone.

---

## 8. Results

### 8.1 Representation — the central positive result

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  READING TOPOLOGICAL STATE           sigmoid   linear    baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  S²-Rips island partition             0.855     0.469      0.487
  Granular contact components          0.778     0.104      0.200
  β₀ vs true component count          1.0000        —      exact, every frame
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Forecasting the partition, S²-Rips corpus (accuracy by lag)
    lag  0    0.920   vs  0.311 linear   vs  0.398 majority
    lag  1    0.817   vs  0.308          vs  0.398
    lag  4    0.626   vs  0.305          vs  0.398
    lag 16    0.494   vs  0.201          vs  0.398
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The linear channel sits at or below chance at every lag. `linear` is given the
same total state dimension throughout.

### 8.2 Speed

| Operation | Cost |
|---|---|
| **Imagine one step** | **97 µs** |
| Encode one window | 0.9 ms |
| Real distilgpt2 forward | 86 ms |
| **Ratio** | **~880×** |

### 8.3 Exact agreement with independent implementations

| Reference | Agreement |
|---|---|
| `mujoco#3396` DSU island partition | **exact**, correlation 1.0000, every frame |
| `kernels#22` block salience | **bit-identical**, max deviation 1.8e-15 |
| `kernels#22` CSR schedules | **bit-identical** at seq 512 / 1024 / 2048 |

### 8.4 The transformer null

Topology does **not** improve prediction on distilgpt2 — at any layer, in either
normalization. This is reported because it is true, and §9 explains why.

| Layer | embed | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Δ (negative would mean topology helps) | +0.0003 | +0.0008 | +0.0010 | +0.0025 | +0.0016 | +0.0007 | +0.0000 |

What ψ encodes there is now known: **lexical repetition**. It reads "distinct
tokens in this window" at R² **0.537** against the linear channel's 0.087,
because same-token positions sit at median distance 17.6 while different-token
positions sit at 34.9 — a 1.98× gap that *is* an interaction radius, and what it
separates is token identity.

---

## 9. When topology pays

Our original hypothesis was falsified twice. The condition that survived has two
clauses, because *representing* structure and *predicting* with it are different
problems:

> **Representation.** ψ reads the partition when the component count has a
> **plateau in radius** — a band, fixed relative to the configuration's own
> scale, on which H₀ is constant. The plateau centre is the interaction radius;
> its width is how much error in stating that radius the encoder absorbs.
>
> **Prediction.** Reading it helps only when that partition is an **input to the
> dynamics**, not merely an **observable of them**.

The first clause replaced "a fixed *physical* distance", falsified by a system
whose threshold dilates across **three decades** and is read just as well
(0.940 vs 0.382 linear) — ψ's scale-invariant head divides the diameter out,
making the dilating and fixed arms literally the same feature, mean |corr|
**1.0000** across 19 columns.

The second clause explains the transformer null. Token windows satisfy the
geometry — entities are token positions, the threshold is the identity gap — and
still buy nothing, because a lexical tally is an observable of the past. On the
S²/contact worlds the partition **is** the constraint graph: what merges
determines what moves.

---

## 10. Control of robots

The operator is action-conditioned, so it can be planned against.
`TopologicalMPC` runs cross-entropy-method model-predictive control entirely in
latent space — no simulator inside the loop.

```python
from sigmoid.control import TopologicalMPC, target_cost, beta0_cost

world = sigmoid.SigmoidWorldModel(
    config=sigmoid.SigmoidConfig(
        action_dim=8,
        entity_dim=2,          # the state is a set of bodies, not a sequence
        abs_radii=(0.9,),      # the contact radius, stated rather than inferred
    )
).fit(observations, actions=actions)

mpc  = TopologicalMPC(world, action_low=-np.ones(8), action_high=np.ones(8))
plan = mpc.plan(window, target_cost(world, goal))

plan.action        # execute this
plan.feasible      # False -> the model declines to guess
```

Two departures from standard MPC:

- **Topological cost.** Costs are evaluated on world states, so `beta0_cost`
  expresses *"never let the contact graph merge"* — a constraint on component
  structure at a physical radius, which raw coordinates cannot state without a
  bespoke detector.
- **Refusal.** Every candidate rollout is gate-scored; when all of them leave
  the calibrated region the planner reports that, instead of returning its
  least-bad extrapolation. A world model asked to drive a robot should be able
  to say it does not know.

**Status.** The closed loop works — four robots reach goals, planning is pure
matvec. The `beta0_cost` term does *not* yet reduce collisions: β₀ is exact as
an observation (corr 1.000) but carries 0.11–0.26 MAE as a forecast, and the
planner optimises differences of ~1 component against it. Raising the weight
makes it worse, which is the signature of a noisy objective rather than a weak
one. The fix is an uncertainty-weighted cost, not tuning.

---

## 11. Sparse attention schedules

The merged kernel
[`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22)
builds a causal CSR block schedule from a 0D-persistence salience over key-block
centroids — the same single-linkage object §3.2 computes. `schedule.py` is a
drop-in replacement.

The reference enumerates and sorts all `n(n−1)/2` centroid pairs before running
union-find. Single-linkage merges occur **only along minimum-spanning-tree
edges**, so `n−1` edges carry the whole barcode and the rest cannot change any
merge height. This is an algorithmic reduction, not a micro-optimization.

| Blocks | Merged reference | Sigmoid | Speedup |
|---|---|---|---|
| 64 | 7.02 ms | 0.60 ms | **11.6×** |
| 128 | 36.49 ms | 2.43 ms | **15.0×** |
| 256 | 148.40 ms | 12.18 ms | **12.2×** |
| 512 | 637.23 ms | 47.28 ms | **13.5×** |

`IncrementalSalience` grows the MST one block at a time for autoregressive
decode rather than rebuilding: **8.6–21.7× per appended block**, bit-exact
across 2748 append-versus-rebuild comparisons including seven adversarial
tie-heavy families.

---

## 12. Validation

```bash
python -m pytest tests/ -q          # 57 checks; each file also runs standalone
```

`bench.py` runs every arm at matched budget and **reports failed gates rather
than dropping losing arms**:

| Arm | Question it answers |
|---|---|
| `no_topology_same_u` | Does ψ do work? *(the arm the topology claim is judged on)* |
| `linear_only` | Is ψ the best use of those dimensions? |
| `carry` | Does the model beat copying the present? |
| `mean` | Does it beat predicting the dataset average? |

Nine runnable studies, two of which falsified this project's own claims:

```bash
python examples/s2_rips_corpus.py        # entity dynamics: topology wins
python examples/contact_world.py         # granular contact physics
python examples/robot_control.py         # closed-loop control, 4 robots
python examples/falsify_condition.py     # attempts to break the central claim
python examples/why_tokens_fail.py       # what ψ encodes inside an LLM
python examples/gate_ood_benchmark.py    # gate vs Mahalanobis / PCA / MSP
python examples/layer_sweep.py           # is the LLM null a layer artifact?
python examples/kernel_schedule_parity.py  # parity vs triton-lang/kernels#22
python examples/llm_worldmodel.py        # full distilgpt2 study, offline
```

---

## 13. Installation and use

```bash
git clone https://github.com/teerthsharma/sigmoid.git && cd sigmoid
pip install -e .
```

Core requires **numpy and scipy only**. Optional: `ripser` for H₁,
`torch`/`transformers` for model capture, `scikit-learn` for the example probes.

```python
import sigmoid

wm = sigmoid.wrap(model, prompts)     # calibrate on captured activations
z  = wm.observe(window)               # encode a world state
r  = wm.imagine(z, steps=16)          # roll forward, no model in the loop

r.hiddens          # predicted activations
r.grounded_at      # where it stopped trusting itself
print(wm.summary())
```

```bash
python -m sigmoid fit   traj.npy -o world.npz
python -m sigmoid roll  world.npz traj.npy -k 16
python -m sigmoid bench traj.npy
```

---

## 14. What we got wrong

Published because an engine that only reports its wins cannot be trusted with
the ones it claims. Four previously-shipped claims were falsified under
adversarial review — **every one because a control was missing, never because
the mathematics failed.**

| Claim as shipped | What was missing | Corrected |
|---|---|---|
| "~30% better rollouts on Lorenz" | the ablation co-varied PCA rank with topology | ~15% at k=1–4, and a **loss** at k=16 |
| "bit-identical kernel parity" | only continuous inputs were sampled | failed **123/200** tie-heavy cases until fixed |
| "a useful cheap OOD detector" | no baselines had ever been run | loses to Mahalanobis (0.899) and PCA (0.888) |
| "needs a fixed *physical* distance" | no scale-varying control existed | replaced by a scale-relative H₀ plateau |

Two latent numerical faults surfaced with them: `csr_matrix` silently drops
explicit zero-length edges, so coincident blocks never merged; and the Gram
distance identity `|x|²+|y|²−2x·y` cancels to *exactly* 0.0 for points 1e-12
apart, which had been merging distinct points inside `h0_barcode`. Both are
fixed, at a 23% cost on the encode path.

The generalizable lesson is in [AGENDA.md](AGENDA.md): most published negative
results for topological features are probably specification errors of the kind
in §3.5, rather than evidence about topology.

---

## 15. Provenance

Sigmoid is not invented from nothing. It is a transplant of a construction
developed in the author's earlier work, moved from electromagnetic cavities onto
model activations.

| Source | What was taken |
|---|---|
| [**Faraday**](https://github.com/teerthsharma/faraday) | barcode → Hilbert-series embedding → least-squares coupling operator → Banach fixed point |
| [**Hamilton Tensor**](https://github.com/teerthsharma/hamliton) | N-body extension, tensor-product Hilbert space, rank-R truncation, gauge projection |
| **topological-ml-toolkit** | persistence and Betti-curve API contract, Mapper covers, sheaf consistency residual |
| [**mujoco#3396**](https://github.com/google-deepmind/mujoco/pull/3396) | the S²–Vietoris–Rips corpus: unit directions, tangent velocities, geodesic Rips, islands as H₀ |
| [**triton-lang/kernels#22**](https://github.com/triton-lang/kernels/pull/22) | 0D-persistence salience over key-block centroids → causal CSR block schedule |

---

## Documentation

| Document | Contents |
|---|---|
| [**teerthsharma.github.io/sigmoid**](https://teerthsharma.github.io/sigmoid/) | rendered walkthrough of the method |
| [SIGMOID.md](SIGMOID.md) | full mathematics and every measurement, including the negative ones |
| [AGENDA.md](AGENDA.md) | ranked research directions, and what was deliberately not funded |
| [CONTRIBUTING.md](CONTRIBUTING.md) | evidence standards for changes |
| [CHANGELOG.md](CHANGELOG.md) | release history |

---

<p align="center">
  <strong>MIT © <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong><br>
  <em>Topology is not decoration. It is the mechanism, or it is nothing.</em>
</p>

# Sigmoid

**A world-model inference engine built from topological coupling operators.**

Sigmoid converts a normal model into a world model without touching its weights.
It learns a linear operator on topological signatures of the model's own
activations, rolls that operator forward far more cheaply than the model runs,
and carries a sheaf-consistency gate that says when the imagined future stopped
being trustworthy.

Everything below was measured in-process on this machine. Numbers that came out
badly are reported as they came out; the sections marked **negative result** are
the ones that matter most for deciding what to build next.

---

## 1. Provenance: what this is assembled from

Sigmoid is not a new idea invented from nothing. It is a transplant of a
construction that already exists in this workspace, moved from electromagnetic
cavities onto model activations.

| Source | What was taken | Where it lives now |
|---|---|---|
| `faraday-main` | barcode → Hilbert-series embedding → least-squares coupling operator `T` → Banach fixed point ("God Tensor") | `state.py`, `operator.py` |
| `hamliton-main` | N-body extension: joint operator over `Ψ = ψ₁ ⊗ … ⊗ ψ_N`, rank-R truncation, overlap matrix, gauge projection | `nbody.py` |
| `topological-ml-toolkit` | persistence/Betti-curve API shape, Mapper covers, sheaf consistency residual, `PHFeaturizer` feature contract | `state.py`, `sheaf.py` |
| `discover-topology` skill | the promotion gates, same-budget ablation discipline, "do not compute Rips in the hot path" | `bench.py`, the two-tier encoder |
| `vllm-topology-kv-policy` skill | offline-topology / cheap-runtime-gate split | engine architecture |
| `lambda-topo-main` | topological invariants as the characterisation of a field, not a decoration | framing |
| `mujoco#3396` | the S²–Vietoris–Rips corpus: unit directions with tangent velocities, geodesic Rips, islands as H₀ | `examples/s2_rips_corpus.py` |
| `triton-lang/kernels#22` | 0D-persistence salience over key-block centroids → causal CSR block schedule | `schedule.py` |

The single most load-bearing inherited rule is from
`discover-topology/references/topology-speedup-map.md`:

> Do expensive topology offline or per-capture, then use the fixed-point vector
> as a cheap runtime gate. Do not compute Rips persistence in the hot attention
> path.

Sigmoid obeys this literally. H₀ is exact and cheap (MST); H₁ is calibration-only.

---

## 2. The mathematics

### 2.1 The world state Σ

Given a hidden trajectory `h₁…h_T ∈ ℝ^D` and window `W`, the state point cloud at
time `t` is `X_t = {h_{t−W+1}, …, h_t}`.

**Robust standardization (required, not cosmetic).** Each coordinate is centered
by its median and scaled by `1.4826 · MAD`. Transformer residual streams carry
*massive activations*: measured on distilgpt2, the largest per-dimension standard
deviation was **22.1** against a median of **0.308**, a 72× ratio concentrated in
about five channels. Without this step, Euclidean distances in `X_t` — and hence
the entire barcode — are a function of five outlier channels, so the "topological"
feature measures activation magnitude and nothing else.

**H₀ barcode, exactly.** For a Vietoris–Rips filtration, the H₀ death times are
exactly the single-linkage merge heights, i.e. the edge weights of the Euclidean
minimum spanning tree. So

```
B₀(X) = { (0, w) : w ∈ MST(X) } ∪ { (0, ∞) }
```

costs `O(W²D)` and involves no simplicial complex. Filtration values are divided
by the cloud diameter, making barcodes comparable across windows.

**H₁** is available via ripser but is calibration-only.

**Hilbert-series embedding.** A barcode is turned into a fixed-length vector by
the numerator of its Hilbert series,

```
N(s) = Σᵢ s^{bᵢ} − Σᵢ s^{dᵢ},     c_k = #{births in bin k} − #{deaths in bin k}
```

This is the step that makes a *linear* operator on topology possible at all.
Sanity check, verified in the test suite: `N(1) = Σ c_k` equals the number of
essential classes.

**The state is hybrid.**

```
z = [ ψ ; u ]
```

- `ψ` — Hilbert coefficients, Betti-curve samples, scale-invariant geometry.
- `u` — whitened PCA coordinates of the standardized activation.

`ψ` alone is not invertible: topology forgets coordinates, so a `ψ`-only world
model cannot emit predicted activations. `u` alone is a plain linear
autoencoder, which is exactly the null model that must be beaten. Keeping both
makes the topology claim falsifiable at matched dimension.

### 2.2 The coupling operator T

Ridge regression in closed form on the lifted feature vector:

```
W = argmin Σ_t ‖ W φ(z_t, a_t) − z_{t+1} ‖² + λ‖W‖_F²
  = (Xᵀ X + λI)⁻¹ Xᵀ Y
```

with the bilinear action lift `φ(z, a) = [z ; a ⊗ z ; a ; 1]`, so that
`W φ = T₀ z + Σ_k a_k T_k z + B a + c`. The problem is convex and small; there is
nothing to iterate.

### 2.3 The Banach certificate

Let `A` be the state→state block and `ρ = σ_max(A)`. If `ρ < 1` the rollout map
is a contraction, so by the Banach fixed-point theorem it has a unique fixed
point `z*` with `‖z_n − z*‖ ≤ ρⁿ‖z₀ − z*‖`. With one-step residual `ε`, the
accumulated error after `n` steps is bounded by

```
E(n) ≤ ε · (1 − ρⁿ)/(1 − ρ)   →   ε/(1−ρ)   as n → ∞
```

`safe_horizon(τ)` inverts this: the largest `k` whose guaranteed error stays
under `τ`. The fixed point is obtained by solving `(I − A)z = b` directly, with a
Banach iteration run alongside so that convergence is *measured* rather than
asserted (the Faraday burn pattern).

**Contraction is not forced by default, and this is a deliberate reversal.** An
early version clipped `ρ` to 0.995 to guarantee a certificate. Measured on
Lorenz, that clipping degraded one-step NRMSE from **0.067 to 0.317**. Chaotic
systems have `ρ > 1` by definition — a positive Lyapunov exponent *is* expansion.
Clipping buys a pretty certificate by misreporting the dynamics. `rho_max`
defaults to `None`; setting it is an explicit, priced choice.

### 2.4 The sheaf gate

Over each time index put two stalks, `F(topo) = ψ` and `F(linear) = u`, two local
views of the same activation window. A restriction map `R : F(linear) → F(topo)`
is learned by ridge on calibration data, where the section glues: `‖Ru − ψ‖ ≈ 0`.

Nothing in the least-squares fit forces an *imagined* `ψ̂` to remain the true
topological signature of the imagined `û`. So

```
r = ‖ R û − ψ̂ ‖
```

is a first cohomology obstruction to gluing the two local sections, and it is
computable during imagination — without the real model, which is the entire
point. Thresholds are set at quantiles of the calibration residual, so the score
is in units of "how unusual would this have been during calibration". A support
term (whitened Mahalanobis distance) is taken in `max` with it.

### 2.5 N-body coupling

For several interacting streams, `Ψ = [ψ₁; …; ψ_N]` evolves under a block
operator `M` whose off-diagonal blocks `M_ij` are the inter-body coupling. Fitting
`M` densely costs `O(N²d²)` — the `d^N` blow-up in disguise — so a rank-R
truncation via SVD gives an `O(NdR)` operator retaining the dominant coupling
modes. `coupling_strength()` reports the block Frobenius norms; the test suite
verifies that a driver/follower pair is recovered.

---

## 3. Architecture

```
normal model ──capture──> hidden trajectory (T, D)
             ──Σ────────> world states z = [ψ ; u]
             ──T────────> imagined rollout            (~100 µs/step)
             ──gate─────> ground when the section stops gluing
```

Two tiers, per the inherited rule: expensive topology at calibration, a matvec
and two norms at runtime.

| file | role |
|---|---|
| `state.py` | Σ — barcodes, Hilbert embedding, robust standardization, decoder |
| `operator.py` | T — ridge fit, spectral projection, Banach fixed point, certificate |
| `sheaf.py` | the grounding gate |
| `nbody.py` | Hamilton-tensor multi-entity coupling |
| `engine.py` | `SigmoidWorldModel`: fit / observe / imagine / save / load |
| `adapters.py` | capture from HF models, torch modules, callables |
| `bench.py` | matched-dimension ablation and promotion gates |
| `control.py` | CEM-MPC planner, topological costs, gate-based refusal |
| `schedule.py` | causal CSR block schedules for topology-sparse attention |
| `cli.py` | `python -m sigmoid fit` / `roll` / `bench` |

### Decoding

Predictions decode as

```
h_{t+k} = Dec(u_{t+k}) + γ^k ⊙ r_anchor
```

where `r_anchor` is the part of the last observed activation that PCA cannot
represent, and `γ` is its measured per-dimension lag-1 autocorrelation. Both
extremes are wrong: `γ = 1` pins predictions to the anchor and inherits
carry-forward's long-horizon error; `γ = 0` discards a genuinely informative
residual at short horizons. `γ` is measured, not chosen.

### Block structure is selected, not assumed

Whether `ψ` and `u` should evolve jointly or independently is a property of the
wrapped model, not of this library:

| system | coupled | block-diagonal |
|---|---|---|
| Lorenz, k=1 NRMSE | **0.052** | 0.073 |
| distilgpt2, R² for `u_{t+1}` | 0.145 | **0.168** |

So `block_diagonal="auto"` fits both and keeps the winner on a held-out slice.
The selection is scored on a **16-step rollout**, not one step: at one step the
two structures agreed to within 0.06% on Lorenz (0.76936 vs 0.76886), while at
16 steps they separated cleanly (1.097 vs 1.216). The benefit of letting topology
inform content compounds across the rollout, which is what a world model is for.

---

## 4. Results

### 4.1 Lorenz (24-dim lift of a chaotic attractor) — topology wins

Matched state dimension 38, 2000 steps, 30% held out.

| arm | k=1 | k=4 | k=16 |
|---|---|---|---|
| **sigmoid** | **0.0517** | **0.2139** | **0.8944** |
| linear_only (same dim, no topology) | 0.0737 | 0.2854 | 0.9887 |
| carry-forward | 0.0802 | 0.3117 | 1.0179 |
| predict-the-mean | 1.0545 | 1.0554 | 1.0598 |

**~30% relative improvement over the same-dimension linear model.** The
topological channel earns its dimensions on a low-dimensional, smooth,
deterministic system.

### 4.2 distilgpt2 residual stream — negative result on accuracy

40 local markdown passages, 9186 tokens, 8226 transitions, state dim 128.

| arm | k=1 | k=4 | k=16 |
|---|---|---|---|
| sigmoid | 0.3108 | 0.3129 | 0.3627 |
| linear_only (same dim) | 0.3105 | 0.3141 | 0.3627 |
| carry-forward | 0.4020 | 0.4139 | 0.5210 |
| predict-the-mean | 0.3799 | 0.3218 | **0.3616** |

**The topology adds nothing here, and the world model does not beat predicting
the calibration mean at long horizon.** It beats carry-forward comfortably. The
decomposition that explains it:

| question | R² |
|---|---|
| predict `u_{t+1}` from `u_t` | 0.168 |
| predict `u_{t+1}` from `[u_t, ψ_t]` | 0.145 *(worse — pure overfitting)* |
| predict `u_{t+1}` from `ψ_t` alone | −0.049 |
| predict `ψ_{t+1}` from `ψ_t` | **0.762** |
| predict `ψ_{t+1}` from `u_t` | −0.181 |

So on this model `ψ` is a **stable, strongly self-predictable signal that is
genuinely independent of the linear content and carries no information about
it**. That is the profile of a monitor, not of a dynamics feature — which is
precisely what the inherited speedup map predicted:

> Sheaf consistency filters … useful for correctness diagnostics but rarely a
> direct speedup alone. Use as a safety gate over another sparse selector.

This was reached after ruling out the plausible alternatives: robust
standardization (fixed a real 72× outlier problem), corpus size (378 → 8226
transitions, which dropped ρ from 34.5 to 1.26), residual-decay decoding, block
structure, and the selection metric. Further tuning to force a win would be
fitting the benchmark.

### 4.3 S²-Rips corpus — topology wins decisively, after two structural fixes

The distilgpt2 null left an ambiguity: token dynamics is a domain where nothing
guaranteed topological structure was present, so a null there cannot distinguish
"topology does not help" from "there was no topology to find". The S²-Rips
corpus from `mujoco#3396` removes the ambiguity — nodes flow along great
circles, clusters merge and split, and the island count is *by construction*
the H₀ of a Rips graph at a fixed radius. If topology cannot win here it cannot
win anywhere.

The first attempt scored **0.386 against a 0.487 majority baseline** — actively
uninformative. Two structural bugs, both silent:

**Bug 1: the wrong point cloud.** Σ built its barcode from the window of
consecutive *state vectors*, so it described the shape of a trajectory segment
in state space. The island count is H₀ of the *entity* cloud at one instant.
Different objects entirely. Fixed by `entity_dim`, which unpacks each
observation into its entities. → **0.764**

**Bug 2: the scale was normalized away.** The barcode divides filtration values
by the cloud diameter, which buys scale invariance — there is even a test
asserting it. But an island partition is defined at a fixed *absolute* radius,
so dividing out the diameter destroys exactly the needed information. Fixed by
`n_abs_radii`, Betti samples at absolute radii learned from calibration merge
heights. → **0.855**

A third fix was needed for Bug 1 to take effect: robust per-coordinate
standardization exists to tame transformer activation outliers, but reshaping 63
independently-rescaled numbers into 21 points does not give back points on a
sphere. Spatial clouds are now read raw, temporal clouds standardized.

Reading the island partition off each channel (5-fold, logistic probe):

| encoder | psi | u (linear) | majority |
|---|---|---|---|
| temporal cloud, standardized *(original)* | 0.386 | 0.469 | 0.487 |
| spatial cloud, raw geometry | 0.764 | 0.469 | 0.487 |
| **spatial + absolute radii** | **0.855** | 0.469 | 0.487 |

And it *forecasts* the partition, which is the actual world-model claim:

| steps ahead | psi | u (linear) | majority |
|---|---|---|---|
| 0 | **0.920** | 0.311 | 0.398 |
| 1 | **0.817** | 0.308 | 0.398 |
| 4 | **0.626** | 0.305 | 0.398 |
| 16 | **0.494** | 0.201 | 0.398 |

The linear channel sits at or below chance at every lag. Independently verified:
β₀ read at the absolute radius corresponding to the corpus threshold reproduces
the ground-truth island count with **correlation 1.0000 and exact agreement on
every frame** — sigmoid's MST-based H₀ *is* the PR's DSU island partition.

**Caveat, stated plainly.** On *coordinate* rollout this corpus still favours
carry-forward, and the coupled operator is worse than the linear one (ρ = 3.2).
The topological channel wins on the topological observable, which is the state a
controller would need, and loses on raw coordinate regression.

### 4.4 Layer sweep — the distilgpt2 null is not a layer artifact

"Wrong layer" was the leading explanation for the LLM null. It is wrong. Every
layer of distilgpt2, matched dimension, delta = sigmoid − linear at k=16
(negative would mean topology helps):

| layer | embed | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| delta | +0.0003 | +0.0008 | +0.0010 | +0.0025 | +0.0016 | +0.0007 | +0.0000 |

Topology helps at no layer. Mid-stack is also *harder* to predict (NRMSE
0.58–0.70) than the final layer (0.363), with ρ reaching 17–28 — the opposite of
the hypothesis. Direction-only normalization improves absolute accuracy
(0.363 → 0.348) but leaves every delta positive.

Combined with §4.3, the conclusion is specific rather than general: **the
topological channel needs a state that is a set of interacting entities. A token
sequence is not one.**

### 4.5 Schedule parity with the merged Triton kernel

`triton-lang/kernels#22` builds its causal CSR schedule from a 0D-persistence
salience over key-block centroids — the same single-linkage object
`state.h0_barcode` computes. `schedule.py` supplies a drop-in replacement.

The reference enumerates and sorts all n(n−1)/2 centroid pairs before running
union-find. Single-linkage merges only ever occur along minimum-spanning-tree
edges, so n−1 edges carry the whole barcode and the rest cannot change any merge
height. Building the MST first is therefore an algorithmic reduction, not a
micro-optimization.

Measured against the merged source itself (triton stubbed; only the pure-Python
builders are exercised):

| blocks | max abs diff | merged | sigmoid | speedup |
|---|---|---|---|---|
| 32 | 1.8e-15 | 1.69 ms | 0.26 ms | 6.6× |
| 64 | 3.6e-15 | 7.02 ms | 0.60 ms | 11.6× |
| 128 | 1.8e-15 | 36.49 ms | 2.43 ms | 15.0× |
| 256 | 1.8e-15 | 148.40 ms | 12.18 ms | 12.2× |
| 512 | 1.8e-15 | 637.23 ms | 47.28 ms | 13.5× |

Full CSR schedules are **bit-identical** at seq 512/1024/2048. Deviation is
float64 round-off. Since the promotion rules require schedule-build time to be
reported separately from attention time, a 13× faster builder is a real gain at
long context, where the reference's O(n²) sort would otherwise show up in
end-to-end numbers.

### 4.6 Robot control — the loop works, the topological cost does not

`control.py` plans against the action-conditioned operator with CEM-MPC, in
latent space, with the sheaf gate rejecting candidates that leave the calibrated
region. Four planar robots, contact radius 0.9, diagonal-swap benchmark, 6
episodes each:

| controller | contacts | goal error |
|---|---|---|
| topological cost (β₀ + reach) | 11.5 | 1.67 |
| reach only, topological model | 10.7 | 1.18 |
| reach only, no-topology model | 9.8 | 1.08 |

**The closed loop works.** Robots reach goals; planning is pure matvec with no
simulator inside the loop; the gate refuses off-manifold plans.

**The β₀ cost does not help.** The diagnostics locate the problem precisely, and
it is not the signal:

| quantity | error |
|---|---|
| encoded β₀ vs true component count | corr 1.000, MAE 0.000 |
| imagined β₀ at horizon 1 | MAE 0.113 |
| imagined β₀ at horizon 6 | MAE 0.261 |

The observation channel is exact. The forecast is decent. But the planner
optimises differences of ~1 component against a forecast carrying 0.11–0.26 of
error, so at weight 8.0 it chases noise into 16.8 contacts. *Raising the weight
makes it worse* — the signature of a noisy objective, not a weak one. The fix is
an uncertainty-weighted cost (AGENDA.md R6), not tuning.

Two task-design lessons, recorded because both wasted real time:

- Sending every robot to the diagonally opposite corner routes all four through
  one centre point, making the constraint *unsatisfiable* — the cost could then
  only distort the reach behaviour.
- A rotation task made contact *unreachable*, and every controller scored zero.
  A constraint is only testable when it is both violable and satisfiable.

A third instance of the scale bug appeared here and is worth stating separately.
With a contact radius of 0.9, the learned quantile radii came out at 2.21–11.94 —
every one too coarse — so `beta0_cost` was silently constraining "how many groups
within 6 units". Hence `abs_radii`: **exploration learns the radius, control is
told it.**

### 4.7 Speed — large and real

| operation | cost |
|---|---|
| encode one window (topology included) | 0.74 ms |
| **imagine one step** | **97.6 µs** |
| real distilgpt2 forward | 86.1 ms |
| **ratio** | **~880× cheaper per step** |

### 4.8 The certificate trade, priced

| `rho_max` | ρ | k=1 NRMSE | bound@16 | informative? |
|---|---|---|---|---|
| none | 1.2556 | 0.2410 | 368 | no (vacuous) |
| 0.99 | 0.9900 | 0.2576 | 12.5 | no (vacuous) |
| 0.90 | 0.9000 | 0.2602 | 6.95 | no (vacuous) |

**On distilgpt2 the certificate is never informative**, at any setting. The
one-step residual (~0.7) is too large for the geometric series to close below 1
within 16 steps. Forcing contraction costs accuracy and buys nothing here. The
`certificate` gate in `bench.py` therefore tests that the bound is *informative*
(< 1.0), not merely that it holds — an uninformative bound trivially holds, and
gating on containment alone would reward a worse operator.

### 4.9 The gate — works on structural anomaly, misses high-entropy noise

Calibrated on markdown prose, `gate_quantile=0.98`:

| probe | mean score | fires |
|---|---|---|
| held-out prose | 0.671 | 0/9 ✅ correctly quiet |
| symbol soup | 1.385 | 4/9 ✅ detected |
| **uniform random tokens** | **0.513** | **0/5 ❌ missed** |
| one token repeated | 14.596 | 5/5 ✅ strongly detected |

The random-token miss is a real limitation, not a tuning artifact. Random input
produces diffuse, high-entropy representations that sit near the *centroid* of
the standardized state space, so both the Mahalanobis term and the sheaf residual
stay small. **The gate detects structural degeneracy, not unusualness.**

Note also that the gate measures deviation from the *calibration corpus*, not
"bad text": symbol soup is only anomalous because the corpus is prose. An earlier
run with a 5-passage corpus produced a different verdict on the same probe.

---

## 5. What is established, and what is not

**Established**
- The transplant works mechanically: any HF model, torch module, or callable
  becomes a world model with `sigmoid.wrap(model, inputs)`.
- Imagination is ~880× cheaper per step than a real forward.
- On low-dimensional structured dynamics, the topological channel gives ~30%
  better rollouts at matched dimension (Lorenz).
- On entity-structured dynamics it reads the topological state at **0.855** vs
  0.469 for a matched linear channel, and forecasts it 16 steps out above
  chance where the linear channel is below chance.
- Sigmoid's H₀ reproduces the `mujoco#3396` DSU island partition exactly
  (correlation 1.0000, every frame) and the `kernels#22` block salience
  bit-identically, 13× faster.
- H₀-via-MST makes persistence cheap enough for a runtime path (0.74 ms/window).
- The sheaf residual detects structural degeneracy without invoking the model.

**Not established**
- That topology helps on transformer activations. Measured at every layer, in
  two normalizations: it does not.
- That the world model beats the mean at long horizon on an LLM. It does not.
- That the certificate is useful on an LLM. It is vacuous at every setting tried.
- That topology helps *coordinate* regression even on the S² corpus. It does
  not; carry-forward wins there.
- Anything about models larger than distilgpt2, or about a real MuJoCo
  simulation as opposed to the corpus generator's math.

**The honest summary.** The engine is real. The topological channel earns its
place exactly when the state is a **set of interacting entities** and the
quantity of interest lives at a **fixed physical scale** — then it is decisive.
On token sequences it is a monitor, not a predictor. The two structural bugs in
§4.3 are the more useful finding than either headline: both were silent, both
produced a psi that looked healthy and carried nothing, and either one alone
would have been reported as "topology does not help".

---

## 6. Open questions worth the next week

1. **Wrong layer?** Everything was measured on the last hidden layer. Mid-stack
   representations are more geometric and less token-locked; `layer=3` is one
   line to change and may be where topology has something to say.
2. **Wrong time axis.** Token-to-token is high-entropy. The natural dynamics for
   an LLM world model may be *across generation steps* or *across layers*, not
   across positions.
3. **Wrong homology.** Only H₀ is in the runtime path. H₁ (loops) is where the
   interesting structure of a trajectory manifold would live, and it is currently
   calibration-only for cost reasons.
4. **The gate's blind spot** to high-entropy input is fixable with an entropy or
   effective-rank term in the stalk set — a genuinely new stalk, not a threshold
   tweak.
5. **N-body is untested on real models.** `nbody.py` is verified on synthetic
   driver/follower pairs only. Parallel decode streams or KV shards are the
   obvious first real target.

---

## 7. Reproducing

```bash
python tests/test_sigmoid.py                # 21 checks, no framework needed
python examples/s2_rips_corpus.py           # entity dynamics: topology wins
python examples/robot_control.py            # closed-loop control, 4 robots
python examples/llm_worldmodel.py           # distilgpt2 study, offline
python examples/layer_sweep.py              # is the LLM null a layer artifact?
python examples/kernel_schedule_parity.py   # parity vs triton-lang/kernels#22
python -m sigmoid.schedule                  # salience self-check
python -m sigmoid bench traj.npy            # ablation + gates on your own capture
```

The LLM examples need a locally cached `distilgpt2`; the kernel-parity example
needs a checkout of `triton-lang/kernels`. Nothing makes a network call.

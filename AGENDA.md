# Research Agenda

*Internal note. Written after building `sigmoid`, not before — every claim below
is anchored to something measured in [SIGMOID.md](SIGMOID.md) or marked as
speculation.*

The original brief asked for a survey of all of AI and fifty research
opportunities. That document would have been worth very little: a survey written
without running anything is a list of things one has read. What follows is
narrower and, I think, more useful — the questions this build actually exposed,
ranked by how much evidence sits behind them.

---

## 1. The finding that generalizes

Two silent bugs cost more than any modelling choice in this project:

1. **The wrong topological object.** Σ computed persistence over a *temporal*
   window of state vectors when the quantity of interest was H₀ of an *entity*
   cloud at one instant. Score: 0.386 against a 0.487 majority baseline.
2. **The wrong scale.** The barcode divided out the cloud diameter — scale
   invariance, deliberately engineered and unit-tested — when the target was
   defined at a fixed absolute radius. Fixing it: 0.764 → 0.855.

Neither raised an error. Both produced a psi with healthy variance, sensible
standardization, and no information. Either alone would have been written up as
"topology does not help here", and that write-up would have been wrong.

**The generalizable claim:** most published negative results for topological
features in ML are probably specification errors of this kind rather than
evidence about topology. Two binary choices — which cloud, which scale — sit
upstream of every persistence pipeline, are almost never stated in papers, and
silently determine the outcome. A TDA-for-ML paper should be required to state
both.

**R1 — Audit the negative literature.** Take published null results for
persistent-homology features and re-run them across the (cloud, scale) matrix.
Cheap, unglamorous, high information. *Solo researcher. Weeks, not months.*

**R2 — Make the choice automatic.** Both bugs are detectable without labels.
*Shipped (§6b), with one caveat found by R5 after the fact:* raw
self-predictability is the wrong criterion — it reads 0.762 on the distilgpt2
encoder, exactly the case it should reject, because overlapping windows make any
temporal psi its own successor. The shipped version subtracts a shuffled-order
floor, which fixes the overlap artifact; the stronger criterion R5 recommends is
`psi → next state`, one ridge fit, not yet implemented.

---

## 2. When topology pays, stated precisely

The evidence supports a sharp condition, not a general endorsement:

> ~~The topological channel earns its dimensions when the state is a set of
> interacting entities whose interaction threshold is a fixed physical
> distance.~~ **Superseded — see §6b (R4, R5).** Falsified in two independent
> ways and replaced by a two-clause condition: a *scale-relative* H₀ plateau
> for representation, and *causal* (partition is an input to the dynamics, not
> an observable of them) for prediction.

| system | state | result |
|---|---|---|
| Lorenz (24-dim lift) | smooth trajectory | ~15% at k=1–4, *loses* at k=16 (was mis-stated as ~30%) |
| S²-Rips entities | set of nodes | 0.855 vs 0.469 |
| granular contact | set of disks | represents 0.778 vs 0.104; forecasts ~persistence |
| distilgpt2 residual stream | token sequence | no gain at any layer |

**R3 — Test the condition where it should pay.** Real MuJoCo contact dynamics,
multi-agent traffic, molecular systems, granular media. All are entity sets with
physical interaction radii. Predicted: topology helps. This is the single
highest-value next experiment and the corpus generator already exists.

**R4 — Test the condition where it should fail.** Deliberately construct an
entity system with *no* fixed interaction scale (scale-free coupling). Topology
should not help. A condition that only ever predicts success is not a condition.

**R5 — Why did tokens fail?** The honest answer is unknown. Candidate: a token
sequence *is* a set of entities (tokens in a window), so the condition should
have been satisfied and was not. Either the condition is incomplete, or token
representations have no fixed interaction radius. Resolving this is worth more
than another benchmark.

---

## 3. Certificates that certify something

Sigmoid can produce a Banach contraction bound. On distilgpt2 it is vacuous at
every setting tried: bounds of 6.9 to 368 where the signal has magnitude ~1.
Forcing ρ < 1 costs accuracy (Lorenz: 0.067 → 0.317) and still yields nothing
useful.

The bound is `ε(1−ρⁿ)/(1−ρ)`, and it is loose because it assumes the worst-case
error direction compounds at every step. Real errors partially cancel.

**R6 — Directional error bounds.** Track the covariance of the one-step
residual, not just its norm, and propagate it through the operator. Should be
dramatically tighter when residuals are anisotropic, which they are. This is the
difference between a certificate that is technically true and one a deployment
engineer would act on.

**R7 — Certified-safe horizon as a scheduling primitive.** If a tight bound
existed, `safe_horizon(τ)` becomes a real inference-time knob: imagine k steps,
ground, repeat. At ~880× per-step cost ratio the economics are compelling — but
only with R6.

---

## 4. Gates

The sheaf gate detects structural degeneracy (repeated tokens 14.6× threshold,
symbol runs 1.39×) but **misses uniform random tokens** (0.513, quieter than real
prose). High-entropy input produces diffuse representations near the state
centroid, so both the Mahalanobis and the sheaf term stay small.

**R8 — An entropy stalk.** Add effective rank or spectral entropy of the window
as a third local section. Random input has anomalously *high* effective rank —
the exact signal both current stalks miss. This is a new stalk, not a threshold
tweak, and it is a few lines.

**R9 — Gates as the honest product.** *Done, and it did not survive (§6b).*
Benchmarked against Mahalanobis, PCA reconstruction, k-NN, MSP and predictive
entropy, the gate ranks **behind** the two cheapest baselines on mean AUROC. The
claim that survives is applicability, not accuracy: it is the only detector that
functions on an imagined state. Encode cost is now 0.909 ms, not 0.74 ms, after
the `_pairwise` correctness fix.

---

## 5. Kernels

`schedule.py` reproduces `kernels#22`'s block salience bit-identically and 13×
faster, by exploiting the fact that single-linkage merges occur only along MST
edges. Schedule construction is measured separately from attention time under
the promotion rules, so this matters at long context.

**R10 — Upstream it.** Bit-identical output, drop-in signature, pure algorithmic
win. The obvious contribution back.

**R11 — Incremental salience for decode.** During autoregressive decode the key
set grows by one block at a time, and an MST supports incremental edge insertion
far more cheaply than a rebuild. Recomputing the schedule from scratch each step
is the current cost; it need not be.

**R12 — Close the loop.** Sigmoid's world model and the kernel's schedule
consume the same 0D-persistence object. A world model that *predicts* which
blocks will be salient could build the next schedule before the keys exist. This
is speculative and the most interesting item on the list.

---

## 6. What I would not fund

Stated because an agenda that only adds is not an agenda.

- **H₁ in the runtime path.** Loops are theoretically appealing and were
  calibration-only for cost reasons all project. No evidence yet that they carry
  anything H₀ does not. Justify with a measurement first.
- **Bigger LLMs to rescue the null.** The sweep covered every layer and two
  normalizations. Scale is not the missing variable; §2's condition is.
- **Forcing contraction by default.** It buys a pretty certificate by
  misreporting the dynamics. Priced and rejected.
- **A grand unified topological architecture.** The wins here came from getting
  two mundane specification choices right, not from more mathematics.

---

## 6b. Outcomes — eight items executed in parallel

Every item below was run. Results are what came back, not what was hoped for.

| item | outcome |
|---|---|
| **R2** auto (cloud, scale) | **Shipped**, opt-in (`cloud="auto"`). Plain self-predictability reproduces bug #1 — overlapping windows make a temporal psi its own successor (shuffled floor 0.56–0.82 vs −0.16 spatial), so the criterion subtracts a shuffled-order floor. 19/20 where the answer is structurally determined; fails when offered a bogus entity width. |
| **R8** entropy stalk | **Shipped**, inert unless fed. Catches both tails (rank collapse *and* inflation) where the two-term gate caught neither on scrambled input. **Guards ingestion, not rollout** — there is no window for an imagined state. |
| **R6** directional bound | **Shipped** as an estimate, never a guarantee. 12.6× tighter on anisotropic residuals, matches measured error to 1% — but under-bounds 2.4–4.6× when residuals are step-correlated. `residual_autocorr` is the discriminator. Deliberately **not** wired into the gate: it would pass everywhere, including where it is wrong. |
| **R11** incremental salience | **Shipped**, 8.6–21.7× per block, bit-exact over 2748 comparisons. Found the parity claim was verified on gaussians only and failed 123/200 tie-heavy cases. |
| **R9** OOD baselines | **Negative.** Gate 0.846 mean AUROC vs Mahalanobis 0.899, PCA 0.888 — both cheaper. Surviving claim is applicability (imagined states), not accuracy. |
| **R4** falsification | **Condition falsified and replaced.** "Fixed physical distance" is wrong; a 3-decade dilating threshold reads just as well. Replaced by a scale-relative H₀ plateau. |
| **R5** why tokens failed | **Answered.** psi encodes *lexical repetition* (distinct-type count, R² 0.537), not recency. Tokens satisfy the geometric condition and still fail, forcing the causal clause. |
| **R3** contact physics | **Split.** Represents contact structure decisively (0.778 vs 0.104); forecasts it barely above a persistence baseline (0.265 vs 0.250); contributes exactly zero to coordinate rollout. |

**What the parallel run bought that a serial one would not.** Three agents
independently hit the same PCA-rank confound in `compare()`, which is what made
it credible enough to fix rather than explain away. Agent R5 invalidated the
criterion agent R2 had just shipped (self-predictability reads 0.762 on exactly
the case it should reject). Agent R11 found that a parity claim I had verified
and published was true only for the input family I happened to sample. None of
those came from the agent that owned the relevant file.

**Four shipped claims were wrong**, all for the same reason — a missing control,
never failing mathematics:

- ~30% Lorenz gain → dimension-matched arm co-varied PCA rank (~15%, reverses at k=16)
- gate as a cheap OOD detector → no baselines had ever been run
- bit-identical kernel parity → only continuous inputs sampled
- "fixed physical distance" → no scale-varying control existed

## 7. Ranked

| # | item | evidence | cost | payoff |
|---|---|---|---|---|
| R3 | real physics entity systems | strong | medium | high |
| R2 | automatic cloud/scale selection | strong | low | high |
| R10 | upstream the faster salience | measured | low | medium |
| R8 | entropy stalk for the gate | diagnosed | low | medium |
| R6 | directional error bounds | measured gap | medium | high |
| R1 | audit the negative literature | inferred | low | high |
| R5 | why tokens failed | open | medium | high |
| R9 | gate vs OOD baselines | partial | low | medium |
| R11 | incremental decode salience | inferred | medium | medium |
| R4 | falsify the condition | none yet | low | medium |
| R7 | certified horizon scheduling | blocked on R6 | high | high |
| R12 | predictive schedules | speculative | high | high |

Start with R2 and R3. R2 because it would have caught this project's two worst
bugs automatically; R3 because it is the experiment the condition in §2 was
written to predict, and the corpus already exists.

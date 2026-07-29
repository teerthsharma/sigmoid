# sigmoid

Turn any normal model into a world model, without touching its weights.

```python
import sigmoid

wm = sigmoid.wrap(model, prompts)        # calibrate on captured activations
z  = wm.observe(hidden_window)           # topological world state
r  = wm.imagine(z, steps=16)             # roll forward without the model
r.hiddens                                # predicted activations
r.grounded_at                            # where the gate said "stop trusting me"
```

Sigmoid learns a linear operator on **topological signatures** of a model's own
activations — persistence barcodes embedded as Hilbert-series coefficients —
rolls it forward ~880× cheaper than a real forward pass, and carries a
**sheaf-consistency gate** that detects when its own imagination has drifted off
the data manifold, without invoking the model to find out.

The construction is a transplant of the Faraday/Hamilton coupling-operator
pattern (`barcode → embedding → least-squares operator → Banach fixed point`)
from electromagnetic cavities onto model activations.

## Install

```bash
pip install numpy scipy       # required
pip install ripser            # optional, enables H1
pip install torch transformers  # optional, for model capture
```

## What it actually does

| | |
|---|---|
| **State** | `z = [ψ ; u]` — persistence barcode features plus PCA coordinates |
| **Dynamics** | ridge-fit operator `T`, closed form, optionally action-conditioned |
| **Guarantee** | Banach contraction bound when `ρ < 1`, priced honestly when it isn't |
| **Safety** | first cohomology obstruction to gluing the two channels |
| **Multi-entity** | Hamilton-tensor block coupling with rank-R truncation |

## When topology pays

Measured, not asserted:

Two clauses, because *representing* structure and *predicting* with it turn out
to be different problems. Both were forced by falsification experiments:

> **Representation.** psi can read the partition when the component count has a
> **plateau in radius** — a band, fixed relative to the configuration's *own
> scale*, on which H₀ does not change. The plateau centre is the interaction
> radius; its width is how much error in stating it the channel absorbs.
>
> **Prediction.** Reading it helps only when that partition is an **input to the
> dynamics**, not merely an **observable of them**.

The first clause replaced "a fixed *physical* distance", which was falsified: a
system whose threshold dilates over three decades is read just as well (0.940 vs
0.382 linear), because psi's scale-invariant head divides the diameter out — the
dilating and fixed arms produce literally the same feature, mean |corr| 1.0000.

The second explains the LLM null. Token windows *do* have a plateau — same-token
positions sit ~2× closer than different-token ones — and psi duly reads
repetition at R² 0.537. It still predicts nothing, because a lexical tally is an
observable of the past. On the S²/contact worlds the partition **is** the
constraint graph: what merges determines what moves.

**S²-Rips entity corpus** (generator math from `mujoco#3396`) — reading the
island partition off each channel:

| encoder | psi | u (linear, same dim) | majority |
|---|---|---|---|
| **spatial cloud + absolute radii** | **0.855** | 0.469 | 0.487 |
| temporal cloud, standardized | 0.386 | 0.469 | 0.487 |

**Lorenz attractor** — a real but *smaller* gain than first reported, and it
reverses at long horizon:

| arm | k=1 | k=4 | k=16 |
|---|---|---|---|
| **sigmoid** | **0.0622** | **0.2662** | 1.1266 |
| no-topology, **same u** | 0.0734 | 0.2853 | **0.9889** |
| no-topology, matched total dim | 0.0737 | 0.2854 | 0.9887 |

> **Correction.** An earlier version of this README claimed ~30% here. That
> compared against a *dimension-matched* arm, which raises the PCA rank and so
> shrinks the residual the decoder carries forward — a handicap unrelated to
> topology. Against the arm that changes **only** psi, the gain is ~15% at
> k=1–4 and topology *hurts* at k=16. `bench.py` now runs both arms and gates
> on the same-u one.

**distilgpt2 residual stream** — topology does *not* help, at any layer. We now
know what psi encodes there: **lexical repetition**, not semantics. It reads
"distinct tokens in this window" at R² 0.537 (linear channel: 0.087) because
same-token positions sit ~2× closer than different-token ones. That is a real
fixed interaction radius — and a pure observable of the past, which is why it
predicts the future at ~0.

Full numbers, the two silent bugs that nearly buried the first result, and every
negative: **[SIGMOID.md](SIGMOID.md)**. What to do next: **[AGENDA.md](AGENDA.md)**.

## Run it

```bash
python -m pytest tests/ -q                    # 57 checks (each file also runs standalone)
python examples/s2_rips_corpus.py             # entity dynamics: topology wins
python examples/contact_world.py              # granular physics: represents, doesn't forecast
python examples/robot_control.py              # closed-loop control, 4 robots
python examples/why_tokens_fail.py            # what psi encodes on an LLM
python examples/falsify_condition.py          # trying to break the central claim
python examples/gate_ood_benchmark.py         # gate vs Mahalanobis / PCA / MSP
python examples/llm_worldmodel.py             # distilgpt2 study, fully offline
python examples/layer_sweep.py                # is the LLM null a layer artifact?
python examples/kernel_schedule_parity.py     # parity vs triton-lang/kernels#22
python -m sigmoid bench traj.npy              # ablation + gates on your data
```

## Controlling robots

The world model is action-conditioned, so it can be planned against. `sigmoid.control`
runs cross-entropy-method MPC entirely in latent space — no simulator in the loop.

```python
from sigmoid.control import TopologicalMPC, target_cost, beta0_cost

world = sigmoid.SigmoidWorldModel(
    config=sigmoid.SigmoidConfig(
        action_dim=8,
        entity_dim=2,              # state is a set of bodies, not a sequence
        abs_radii=(0.9,),          # the contact radius, stated not inferred
    )
).fit(observations, actions=actions)

mpc = TopologicalMPC(world, action_low=-np.ones(8), action_high=np.ones(8))
plan = mpc.plan(recent_window, target_cost(world, goal))
plan.action                        # execute this
plan.feasible                      # False => the model refuses to guess
```

Two things here are not standard MPC:

- **Topological cost.** Costs are evaluated on world states, so `beta0_cost` can
  express *"never let the contact graph merge"* — a constraint on component
  structure at a physical radius, which raw coordinates cannot state without a
  bespoke detector.
- **Refusal.** Every candidate rollout is scored by the sheaf gate; plans leaving
  the calibrated region are discarded, and if all of them do, the planner reports
  that it has no trustworthy plan instead of returning the least-bad extrapolation.

> **On the gate's accuracy, measured against real baselines.** Benchmarked on
> five OOD types, the gate's mean AUROC is **0.846** — behind Mahalanobis
> (0.899) and PCA-reconstruction error (0.888), both of which are *cheaper*.
> So "a useful cheap OOD detector" does not survive as stated. What does
> survive is narrower and is about applicability, not accuracy: it is the only
> detector here that still functions on an **imagined** state, where no
> activation and no logits exist for any baseline to consume. Notably the sheaf
> term *alone* scores 0.869, beating the full gate — `max(sheaf, manifold)` is
> dragged down by the manifold term, which lands below chance on shuffled text.

```bash
python examples/robot_control.py    # 4 planar robots, closed loop
```

**Status, honestly:** the closed loop works — robots reach goals, planning is pure
matvec. The `beta0_cost` term does *not* yet reduce collisions; the β₀ signal is
exact as an observation (corr 1.000) but carries 0.11–0.26 MAE as a forecast, and
the planner optimises differences of ~1 component against it. Raising the weight
makes it worse. That's a noisy-objective signature, and the fix is uncertainty-
weighted cost, not tuning — see [AGENDA.md](AGENDA.md) R6.

## Attention schedules

`sigmoid.schedule` reproduces the block salience of the merged topology-sparse
attention kernel (`triton-lang/kernels#22`) **bit-identically and 13× faster**,
by exploiting the fact that single-linkage merges occur only along MST edges —
so n−1 edges suffice where the reference sorts n(n−1)/2 pairs.

`IncrementalSalience` grows the MST one block at a time for autoregressive
decode instead of rebuilding: **8.6–21.7× faster per appended block**, bit-exact
against the batch path across 2748 append-vs-rebuild comparisons.

> **Correction.** The original parity claim was verified only on random
> gaussian centroids. On tie-heavy inputs the old path disagreed with the
> reference on **123 of 200 cases**, by up to 4.75 absolute salience — because
> `csr_matrix` drops explicit zeros (coincident blocks never merged) and the
> Gram distance identity cancels to exactly 0.0 for near-coincident points.
> Both are fixed (canonical Prim + `cdist`); parity now holds on all 200. The
> same cancellation was found and fixed in `h0_barcode`, where it had been
> silently merging distinct points.

## Design notes

- **Persistence is never in the hot path.** H₀ is exact via minimum spanning
  tree (single-linkage merge heights *are* the H₀ deaths); H₁ is
  calibration-only.
- **Contraction is not forced by default.** Clipping `ρ` to 0.995 buys a
  certificate by misreporting chaotic dynamics — it cost 0.067 → 0.317 NRMSE on
  Lorenz. `rho_max` is opt-in.
- **The ablation is budget-matched.** `linear_only` gets the same total state
  dimension, so "topology helps" is a measurement, not a preference.
- **Gates can fail.** `bench.py` reports failures rather than dropping losing
  arms. It gates on the **same-u** ablation, not the dimension-matched one, so a
  win is attributable to topology and nothing else.
- **Contraction is reported, never required.** Certificates come in two flavours:
  a worst-case scalar bound (true, usually vacuous) and a covariance-propagated
  directional estimate (up to 12.6x tighter, but it *under-bounds* by 2.4-4.6x
  when residuals are step-correlated). `residual_autocorr` tells you which
  regime you are in — near 0 means the directional number is trustworthy.

## Bundled skills

`.claude/skills/` ships the four skills this work was built with, so a clone
carries its own methodology:

| skill | what it enforces |
|---|---|
| `discover-topology` | topology must produce an execution rule, with falsification gates set before running |
| `vllm-topology-kv-policy` | topology-aware KV-cache and sparse-attention scheduling |
| `tda-tdd` | test-driven development for topological data analysis |
| `research-readme` | research documentation that reports negatives, not just headlines |

The first two are why this repo reports failed gates instead of dropping losing
arms, and why persistence never sits in the hot path.

## License

Built on math from `faraday`, `hamliton`, and `topological-ml-toolkit`
by Teerth Sharma.

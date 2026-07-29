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

> The topological channel earns its dimensions when the state is a **set of
> interacting entities** whose interaction threshold is a **fixed physical
> distance**.

**S²-Rips entity corpus** (generator math from `mujoco#3396`) — reading the
island partition off each channel:

| encoder | psi | u (linear, same dim) | majority |
|---|---|---|---|
| **spatial cloud + absolute radii** | **0.855** | 0.469 | 0.487 |
| temporal cloud, standardized | 0.386 | 0.469 | 0.487 |

**Lorenz attractor**, matched state dimension — ~30% better rollouts:

| arm | k=1 | k=4 | k=16 |
|---|---|---|---|
| **sigmoid** | **0.052** | **0.214** | **0.894** |
| same-dim linear, no topology | 0.074 | 0.285 | 0.989 |

**distilgpt2 residual stream** — topology does *not* help, at any layer. The
value there is the gate, not the prediction.

Full numbers, the two silent bugs that nearly buried the first result, and every
negative: **[SIGMOID.md](SIGMOID.md)**. What to do next: **[AGENDA.md](AGENDA.md)**.

## Run it

```bash
python tests/test_sigmoid.py                  # 16 checks, no framework needed
python examples/s2_rips_corpus.py             # entity dynamics: topology wins
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
  arms.

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

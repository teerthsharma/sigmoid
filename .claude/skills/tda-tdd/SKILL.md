---
name: "tda-tdd"
description: "Turn a topological-ML result into failing tests before writing the implementation — property tests for the persistent-homology invariants that a buggy implementation violates (stability under perturbation, isometry invariance, scale equivariance, permutation invariance, known-manifold ground truth, boundary-squared-is-zero), plus the correctness contracts a sparse-attention or topology-derived kernel must satisfy (dense parity, mask fidelity, causality, all-masked-row NaN guard, gradient check, determinism, tile-boundary shapes). Use whenever implementing persistent homology, Vietoris-Rips, Mapper, persistence landscapes or images, topological sparse attention, a topology-derived scheduler or KV policy, or any time theory is finished and only the coding is left. Complements discover-topology, which covers benchmark falsification but not computational correctness."
---

# TDA-TDD

Write the tests that would catch a wrong implementation before writing the implementation.

This exists because of a specific failure mode in topological ML: a persistent homology bug does not crash and rarely looks wrong. It produces a plausible diagram, the downstream classifier still trains, the sparse-attention mask still has the right density, and the benchmark still reports a speedup. Nothing surfaces until a reviewer asks whether the invariant holds — or worse, nothing surfaces at all and a paper's central claim rests on a transposed index.

Benchmark discipline (null models, ablations, budget units, Pareto frontiers) is covered by `discover-topology`. This skill covers the layer beneath it: whether the topology is computed correctly at all. A speedup measured on a wrong implementation is not a result.

## The premise for theory-first work

When the mathematics is finished and only the coding remains, the test *is* the operationalization of the theorem. Writing it first is not process ceremony — it is the step that converts a proof into a specification.

If a theorem cannot be turned into an executable assertion, the theory is not yet specified tightly enough to implement, and that gap is worth finding before the kernel is written rather than after. In practice the attempt to write the test is where the under-specification shows up: what exactly is the metric, what is the tolerance, what does the constant depend on, what happens in the degenerate case.

So the loop is:

1. **State the theorem or claim** in the form "for all inputs satisfying P, the output satisfies Q".
2. **Write the failing property test** that samples P and asserts Q. Run it. It must fail — against a stub, against `NotImplementedError`, against anything. A test that passes before the implementation exists is testing nothing.
3. **Implement** the smallest thing that turns it green.
4. **Add the adversarial case** the theorem's proof needed: the degenerate input, the boundary, the tie.
5. **Only then** measure performance.

## Persistent homology invariants

These are the properties a correct implementation satisfies and a buggy one violates. They are cheap, they run on tiny inputs, and each one localizes a different class of bug. Use `hypothesis` in Python or `proptest` in Rust so the sampler finds the adversarial case instead of you.

### 1. Permutation invariance

A persistence diagram is a multiset over a point *set*. Shuffling the input rows must not change it.

```python
@given(points=point_clouds(n=(5, 40), d=(2, 4)), perm_seed=st.integers())
def test_permutation_invariance(points, perm_seed):
    rng = np.random.default_rng(perm_seed)
    shuffled = points[rng.permutation(len(points))]
    assert diagrams_equal(ph(points), ph(shuffled), atol=1e-9)
```

Catches: implicit dependence on insertion order in the simplex ordering, unstable tie-breaking in the filtration sort.

### 2. Isometry invariance

Rotations, reflections and translations preserve pairwise distances, so they must preserve the diagram exactly.

Catches: an accidentally coordinate-dependent distance, a centering step that leaks into the filtration.

### 3. Scale equivariance

Scaling the cloud by `c > 0` scales every birth and death by exactly `c`.

Catches: a hardcoded epsilon, an absolute threshold masquerading as a relative one — the most common silent bug in filtration code, and one that makes every result dataset-dependent in a way nobody notices until a new dataset arrives.

### 4. Stability — the load-bearing test

The Cohen-Steiner–Edelsbrunner–Harer stability theorem, in the Vietoris–Rips form: perturbing the point cloud so that no point moves more than ε changes the diagram by at most 2ε in bottleneck distance.

```python
@given(points=point_clouds(n=(8, 30), d=(2, 3)), eps=st.floats(1e-4, 0.1))
def test_stability(points, eps):
    noise = sample_ball(points.shape, radius=eps)   # each row has ||noise|| <= eps
    d_b = bottleneck_distance(ph(points), ph(points + noise))
    assert d_b <= 2 * eps + 1e-9
```

This is the single most valuable test in the file. Stability is the theorem that makes persistent homology usable as a feature at all, and almost every real implementation bug — wrong pairing, dropped bar, mishandled infinite death, reduction that terminates early — shows up as a violation. If only one property test is written, write this one.

Note the constant: `2ε` for Rips via the Hausdorff bound. Using `ε` will produce spurious failures; using `4ε` will pass a broken implementation. If the code uses a different filtration, derive the constant for that filtration rather than copying this one.

### 5. Known-manifold ground truth

Sample densely from manifolds whose homology is known and assert the Betti numbers of the long bars:

| Manifold | Expected persistent features |
|---|---|
| `n` well-separated clusters | `n` H₀ bars surviving past the inter-cluster gap |
| Circle S¹, radius `r` | exactly one long H₁ bar; for Rips it dies at `√3·r` |
| Figure-eight | two long H₁ bars |
| Torus T² | two long H₁ bars, one H₂ bar |
| Sphere S² | one H₂ bar, no long H₁ |
| Gaussian blob | no long bars in any dimension above H₀ |

The `√3·r` death time for the circle is a sharp, non-obvious constant — an implementation that gets the qualitative shape right and that number wrong has a real defect. The Gaussian blob is the negative control and is worth as much as the positive cases: a pipeline that finds structure in noise will find it everywhere.

### 6. Algebraic identities

- `∂∘∂ = 0` over 𝔽₂ on the boundary matrix. A direct unit test, not a property test.
- **Filtration monotonicity**: every face of a simplex appears no later than the simplex. Assert while building; a violation means the complex is not a filtration and everything downstream is meaningless.
- **Elder rule** in H₀: when two components merge, the younger one dies. Cross-check against union-find computed independently.
- **Betti consistency**: bars alive at ε must equal the rank computed by an independent (slow, obviously-correct) reference on small complexes.

### 7. Cross-implementation parity

On inputs small enough to be tractable, assert bottleneck distance ≈ 0 against `ripser`, `gudhi`, or `giotto-tda`. Pin the reference version — parity against an unpinned dependency is a test that changes meaning silently.

This is the check that catches whole-algorithm misunderstanding, which the invariant tests can miss because a self-consistently wrong implementation can satisfy every internal property.

## Kernel correctness contracts

For a topology-derived sparse attention kernel, scheduler, or KV policy. These are ordered by how much bug they catch per line of test.

### 1. Dense parity on a full mask

If the selector returns every key, the output must equal dense SDPA within dtype tolerance.

This is the highest-value smoke test in the entire file. It catches transposed strides, wrong scale factor, off-by-one in the block index, and softmax applied over the wrong axis — the majority of kernel bugs — in a single assertion, before any topology is involved. Write it first.

### 2. Mask fidelity

The realized sparsity pattern equals the pattern the topological rule specified — asserted exactly, element-wise, not as a density statistic. A kernel that attends to approximately the right keys is a different algorithm from the one described in the paper.

### 3. Causality

For causal attention, no dependence on future positions. Test behaviourally rather than by inspecting the mask: perturb position `j`, assert outputs at all positions `i < j` are bitwise unchanged. This catches leakage the mask inspection misses.

### 4. All-masked row

A row with zero selected keys makes `softmax` over all `-inf` produce NaN. Topological selectors hit this on isolated tokens — a point that is its own connected component selects nothing. Assert the guard exists and returns a defined value. This bug reliably escapes into production because it is input-dependent and rare.

### 5. Gradient check

`torch.autograd.gradcheck` in float64 against the dense path on the same mask. A forward-correct kernel with a wrong backward trains to a plausible-looking worse optimum, which is nearly undetectable from loss curves alone.

### 6. Determinism

Same input, same output, bitwise, across repeated runs and across process restarts. Atomics and split-K reductions break this. Run under `torch.use_deterministic_algorithms(True)`. Non-determinism here invalidates every A/B benchmark downstream, because the variance being measured is partly the kernel's own.

### 7. Shape and dtype edges

`seq_len ∈ {1, BLOCK−1, BLOCK, BLOCK+1, 2·BLOCK+1}` — the tail tile is where index arithmetic fails. `head_dim` including a non-power-of-two. GQA head grouping. Batch size 1. fp16 and bf16 inputs with fp32 accumulation, checked against an fp64 reference, plus a large-logit input to verify the max-subtraction prevents overflow.

## The ablation that decides whether topology did the work

`discover-topology` requires same-budget random and locality-only baselines. Add a third:

**Same-budget oracle top-k**, selecting the keys with the genuinely largest attention weights computed densely. This is unimplementable in production — it needs the dense scores — but as a diagnostic it is decisive:

- topological selector ≈ random → the mechanism contributes nothing; any gain is sparsity regularization
- topological selector ≈ oracle → the topological summary is a good proxy for attention mass, which is the actual scientific claim
- topological selector between the two → quantify where, and report that fraction rather than the headline

Reporting the position on that axis is a stronger and more honest result than a speedup number, and it is the number a reviewer will ask for.

## Test layout

```
tests/
  test_invariants.py      # properties 1–4: permutation, isometry, scale, stability
  test_ground_truth.py    # known manifolds, including the negative control
  test_algebraic.py       # ∂∘∂=0, monotonicity, elder rule, Betti rank
  test_parity.py          # vs pinned ripser/gudhi
  test_kernel.py          # dense parity, mask fidelity, causality, NaN guard, grads
  test_determinism.py     # repeated-run bitwise equality
  conftest.py             # hypothesis strategies, manifold samplers, fixed seeds
```

Seed every stochastic test and record the seed on failure — `hypothesis` does this automatically and its shrinking is the reason to prefer it over hand-rolled random tests. When a property test fails, add the shrunk counterexample as an explicit regression case so it is checked forever after at zero cost.

## What not to do

- Do not assert on a diagram by comparing serialized floats. Compare as multisets under bottleneck or Wasserstein distance with a stated tolerance.
- Do not test only on the dataset the method was developed on. The invariant tests exist precisely because they hold for *all* inputs.
- Do not skip the negative control. A pipeline that reports structure in Gaussian noise is worse than one that reports nothing.
- Do not let a benchmark stand in for a correctness test. They fail independently, and speed on a wrong answer is the most expensive kind of wrong.


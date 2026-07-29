# Benchmarks

Every number below was produced in-process on the development machine.
Configuration, baseline and ablation are stated for each. Numbers that came out
badly appear here unchanged.

## Environment

| | |
|---|---|
| Python | 3.11.9 |
| numpy / scipy | 1.26.4 / 1.17.1 |
| torch / triton | 2.5.1+cu121 / 3.7.1 |
| GPU | NVIDIA RTX 4060 Laptop |
| Model under capture | distilgpt2 (6 layers, 768 hidden), local cache |
| Kernel | `triton-lang/kernels#22`, used as merged |

---

## 1. Representation — reading topological state

Probe: multinomial logistic regression on the channel, 5-fold, against the
majority-class floor. `linear` is the PCA channel at the **same total state
dimension**.

| System | psi | linear | majority |
|---|---|---|---|
| S²-Rips island partition | **0.855** | 0.469 | 0.487 |
| Granular contact components | **0.778** | 0.104 | 0.200 |

Ablation isolating the two specification choices of README §3.5:

| Encoder | psi | linear | majority |
|---|---|---|---|
| temporal cloud, standardized | 0.386 | 0.469 | 0.487 |
| spatial cloud, raw geometry | 0.764 | 0.469 | 0.487 |
| **spatial + absolute radii** | **0.855** | 0.469 | 0.487 |

### Forecasting the partition (S²-Rips)

| Lag | psi | linear | majority |
|---|---|---|---|
| 0 | **0.920** | 0.311 | 0.398 |
| 1 | **0.817** | 0.308 | 0.398 |
| 4 | **0.626** | 0.305 | 0.398 |
| 16 | **0.494** | 0.201 | 0.398 |

### Forecasting the partition (granular contact) — negative

Against a **persistence** baseline (predict H₀(t+k) = H₀(t)), which the table
above does not include:

| Lag | psi | linear | majority | persistence |
|---|---|---|---|---|
| 0 | **0.778** | 0.094 | 0.200 | 1.000 |
| 1 | 0.265 | 0.094 | 0.201 | **0.250** |
| 4 | 0.203 | 0.108 | 0.201 | 0.128 |
| 16 | 0.187 | 0.151 | 0.202 | 0.139 |

Lag-1 barely clears copying the present. Forecasting is not established.

---

## 2. Rollout accuracy — matched-budget ablation

Normalized RMSE of predicted against true activations. `no_topology_same_u` is
the topology-only ablation: same linear channel, psi deleted.

### Lorenz (24-dim lift, 2000 steps, 30% held out)

| Arm | dim | k=1 | k=4 | k=16 |
|---|---|---|---|---|
| sigmoid | 46 | **0.0622** | **0.2662** | 1.1266 |
| no_topology_same_u | 12 | 0.0734 | 0.2853 | **0.9889** |
| linear_only | 46 | 0.0737 | 0.2854 | 0.9887 |
| carry | 46 | 0.0802 | 0.3117 | 1.0179 |
| mean | 46 | 1.0545 | 1.0554 | 1.0598 |

Topology delta: `k=1 −0.0112, k=4 −0.0191, k=16 +0.1377`. Helps early, hurts
late. An earlier report of "~30%" compared against `linear_only`, which raises
PCA rank and shrinks the residual the decoder carries forward — a handicap
unrelated to topology.

### distilgpt2 (40 passages, 9186 tokens, 8226 transitions)

| Arm | dim | k=1 | k=4 | k=16 |
|---|---|---|---|---|
| sigmoid | 128 | 0.3108 | 0.3129 | 0.3627 |
| linear_only | 128 | 0.3105 | 0.3141 | 0.3627 |
| carry | 128 | 0.4020 | 0.4139 | 0.5210 |
| mean | 128 | 0.3799 | 0.3218 | **0.3616** |

Does not beat predicting the mean at long horizon.

### Layer sweep — the null is not a layer artifact

Δ = sigmoid − linear at k=16; negative would mean topology helps.

| Layer | embed | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Δ | +0.0003 | +0.0008 | +0.0010 | +0.0025 | +0.0016 | +0.0007 | +0.0000 |

---

## 3. Latency

| Operation | Cost |
|---|---|
| **Imagine one step** | **97 µs** |
| Encode one window (768-dim, W=24) | 0.909 ms |
| Real distilgpt2 forward (214 tokens) | 86 ms |
| **Ratio** | **~880×** |

Encode was 0.74 ms before the `_pairwise` correctness fix (Gram identity →
`cdist`). The 23% is the price of not silently merging distinct points.

---

## 4. Sparse-attention schedules

Measured against the merged `triton-lang/kernels#22` source, with triton stubbed
so only the pure-Python builders are exercised.

| Blocks | Merged | Sigmoid | Speedup | Max abs deviation |
|---|---|---|---|---|
| 32 | 1.69 ms | 0.26 ms | 6.6× | 1.8e-15 |
| 64 | 7.02 ms | 0.60 ms | 11.6× | 3.6e-15 |
| 128 | 36.49 ms | 2.43 ms | 15.0× | 1.8e-15 |
| 256 | 148.40 ms | 12.18 ms | 12.2× | 1.8e-15 |
| 512 | 637.23 ms | 47.28 ms | 13.5× | 1.8e-15 |

Full CSR schedules bit-identical at seq 512 / 1024 / 2048.

### Incremental decode

| n | Append | Rebuild | Speedup |
|---|---|---|---|
| 64 | 0.22 ms | 1.89 ms | 8.6× |
| 128 | 0.40 ms | 4.27 ms | 10.6× |
| 256 | 0.80 ms | 7.84 ms | 9.8× |
| 512 | 1.12 ms | 24.33 ms | 21.7× |

Bit-exact (`np.array_equal`, 0.0 deviation) across 2748 append-versus-rebuild
comparisons over 200 tie-heavy configurations.

### Tie-heavy correctness

The pre-fix batch path disagreed with the merged reference on **123 of 200**
tie-heavy cases, by up to **4.75** absolute salience:

| Input family | Disagreements |
|---|---|
| 1e-12 separated blocks | 40/40 |
| duplicated rows | 39/40 |
| integer lattices | 29/40 |
| **random gaussians** | **0/40** |

That last row is why the original parity check passed. Post-fix: 0/200.

---

## 5. Certificates

distilgpt2, horizon 16:

| `rho_max` | ρ | k=1 NRMSE | bound@16 | Informative (<1.0)? |
|---|---|---|---|---|
| none | 1.2556 | 0.2410 | 368 | no |
| 0.99 | 0.9900 | 0.2576 | 12.5 | no |
| 0.90 | 0.9000 | 0.2602 | 6.95 | no |

Directional (covariance-propagated) estimate:

| System | ρ | scalar | directional | measured | scalar/dir | dir/measured | autocorr |
|---|---|---|---|---|---|---|---|
| anisotropic | 0.983 | 5.065 | 0.402 | 0.401 | **12.6×** | 1.00 | 0.000 |
| isotropic | 0.912 | 2.617 | 0.678 | 0.688 | 3.9× | 0.99 | −0.003 |
| AR(1) residuals | 0.996 | 3.489 | 0.857 | 2.065 | 4.1× | **0.42** | 0.808 |
| Lorenz | 2.616 | 1.62e6 | 0.240 | 1.098 | — | **0.22** | 0.937 |

The last two under-bound. `residual_autocorr` is the discriminator: near zero
means the directional number can be acted on.

---

## 6. Out-of-distribution detection

AUROC, in-distribution = 332 held-out prose windows. `msp` and `entropy` require
a full forward pass and are therefore not on the same cost axis.

| OOD type | gate | sheaf | manifold | mahal | pca | knn | msp | entropy |
|---|---|---|---|---|---|---|---|---|
| uniform random tokens | 0.828 | 0.864 | 0.704 | **0.996** | 0.994 | 0.961 | 0.998 | 1.000 |
| one token repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| symbol soup | 0.998 | 0.998 | 0.997 | 1.000 | 1.000 | 0.997 | 0.005 | 0.007 |
| shuffled words | 0.601 | 0.672 | 0.491 | 0.560 | 0.529 | 0.473 | 0.886 | 0.932 |
| python source | 0.805 | 0.813 | 0.806 | **0.940** | 0.916 | 0.748 | 0.237 | 0.224 |
| **mean** | 0.846 | 0.869 | 0.800 | **0.899** | 0.888 | 0.836 | 0.425 | 0.433 |

The gate loses to Mahalanobis and PCA reconstruction, both cheaper. The sheaf
term alone beats the combined gate: `max(sheaf, manifold)` is dragged down by
the manifold term, which is below chance on shuffled words. MSP and entropy
invert sign on degenerate input and are not deployable at one threshold.

| Detector | Cost per window | Consumes |
|---|---|---|
| `gate.read` | 11–41 µs | a world state — real **or imagined** |
| pca | 14–54 µs | the activation |
| mahal | 160–395 µs | the activation |
| knn | 0.7–1.2 ms | activation + calibration set |
| msp + forward | ~5.7 ms | a full forward pass |

The gate's edge is applicability, not accuracy: it is the only entry that runs
on an imagined state, where the others have nothing to read.

---

## 7. Falsification of the interaction-radius clause

Five arms, 24 entities, identical plant / encoder budget / radius rule / noise;
only the scale structure varies. `gap` is the widest log-radius band on which H₀
is constant; `drift` is its centre's standard deviation across frames.

| System | gap | drift | psi rollout | dim-matched | psi lag-0 | psi @ +0.3 dex |
|---|---|---|---|---|---|---|
| fixed-radius | 0.82 | 0.20 | 0.00% | −54.6% | 0.752 | −0.017 |
| wide-cutoff | 0.58 | 0.26 | 0.00% | −41.9% | 0.541 | −0.046 |
| scale-free | 0.29 | 0.31 | −0.00% | +193.8% | 0.519 | **−0.353** |
| dilating | 0.82 | **1.08** | −0.00% | +1.6% | 0.558 | −0.002 |
| s2-rips (ref) | 0.30 | 0.45 | 587% | 587% | 0.429 | −0.004 |

The `dilating` arm sweeps its threshold across three decades and is still read
at 0.940 lag-0 against 0.382 linear. psi's scale-invariant head makes it the
same feature as the fixed arm — mean |corr| 1.0000 over 19 columns. The "fixed
physical distance" clause is therefore false.

Coordinate-rollout delta against `no_topology_same_u` is `+0.0000` on all four
constructed systems: psi contributes exactly nothing there.

---

## 8. Triton kernel path

RTX 4060 Laptop GPU, triton 3.7.1, torch 2.5.1+cu121. The kernel is
`triton-lang/kernels#22`, used as merged; sigmoid supplies the schedule.

### Kernel vs the merged torch reference, identical schedule

| dtype | max abs diff |
|---|---|
| fp16 | 9.77e-04 – 1.95e-03 |
| bf16 | 7.81e-03 – 1.56e-02 |
| fp32 | 1.69e-03 – 2.39e-03 |

fp32 looser than fp16 is `tl.dot` defaulting to tf32 on Ampere+ (~10 mantissa
bits against the reference's 24), not a defect.

### Whole path is real causal attention, not merely self-consistent

| check | result |
|---|---|
| dense-causal schedule vs `F.scaled_dot_product_attention(is_causal=True)` | 9.77e-04 fp16 |
| same, CPU torch backend | **2.68e-07** |
| schedule bit-identity vs the reference builder, 20 configurations | `np.array_equal` True |

Caveat found by stress-testing: on heavily clustered fp16 keys with near-tied
merge heights, sigmoid and the reference diverge on ~2/300 trials. Traced to an
8.9e-16 gap between `torch.cdist` (which switches to the cancellation-prone
Gram identity above 25 points) and `scipy.cdist`, amplified by ties. Sigmoid's
side is the more accurate one. Realistic keys do not reach it.

### Patching a real model (distilgpt2, 1000 tokens, fp32)

| | max abs logit diff |
|---|---|
| **dense-causal schedule — the wiring proof** | 2.750e-01 |
| same, `TRITON_F32_DEFAULT=ieee` | **3.967e-04** |
| transformers' own eager vs sdpa, for scale | 2.670e-04 |
| degenerate-dense topology schedule vs explicit dense | **0.000e+00** |
| topology (sink=1, local=2, topk=4 of 16) | 9.646e+01, KL **0.0426** nats |
| `restore()` | bit-identical (`torch.equal`) |

Nearly all of the 2.750e-01 is tf32. The error budget starts at one tf32
rounding per layer, and the bit-identical degenerate-dense row proves the CSR
reaches the kernel intact — so the topology figure is a sparsity cost, not
plumbing.

**Causality bug caught here.** Registering only in `ALL_ATTENTION_FUNCTIONS`
makes `_preprocess_mask_arguments` return **no mask** (the TGI/vLLM path).
Invisible when every layer is patched — 2.750e-01 either way. With `layers=[0]`:

| | max abs logit diff |
|---|---|
| mask registered | 1.877e-01 |
| **mask absent** | **1.342e+02** |

Five unpatched layers attending to the future while emitting plausible text.
Fixed by also registering in `ALL_MASK_ATTENTION_FUNCTIONS`.

---

## 9. Inference engine

### Correctness, before any speed claim

| control | result |
|---|---|
| `backend="dense"` vs HF `generate(do_sample=False)` | **64/64 tokens identical** |
| saturated schedule through the *sparse* path (torch and triton) | token-for-token, KL < 1e-9 |

The second separates "plumbing is wrong" from "sparsity costs quality".

### Quality, context 960, 48 greedy tokens

| topk | radius | density | KL (nats) | first div | diverged |
|---|---|---|---|---|---|
| saturated | saturated | 1.000 | 0.00000 | — | 0/48 |
| 8 | 2 | 0.721 | 0.03960 | 2 | 46/48 |
| 4 | 1 | 0.481 | 0.11923 | 0 | 46/48 |
| 2 | 1 | 0.414 | 0.09911 | 0 | 43/48 |
| **0** | 1 | 0.350 | **0.99315** | 0 | 47/48 |

Uniform-random-token context is slightly worse: 0.085 / 0.156 / 0.150 / 1.213.
The last row is load-bearing: dropping topology entirely at similar density
costs **8–14× the KL**, so the salient blocks earn their place.

### Speed — topology loses on anything distilgpt2 can run

End-to-end, best of 3: dense **102/97/95** tok/s at context 256/512/960 against
topology **51/70/78**. Laptop clock drift is ±20% run to run, the same size as
the effect, so the attention-op table is the trustworthy one.

| positions | 12×64 dense/topo | 32×128 dense/topo | density |
|---|---|---|---|
| 1024 *(real)* | 0.216 / 0.287 → 0.75× | 0.237 / 0.597 → 0.40× | 1.000 |
| 4096 | 0.183 / 0.293 → 0.62× | 0.579 / 0.891 → 0.65× | 0.34 |
| 8192 | 0.267 / 0.290 → 0.92× | **1.137 / 0.917 → 1.24×** | 0.18 |
| 16384 | **0.600 / 0.545 → 1.10×** | 2.562 / 0.924 → 2.77× | 0.094 |
| 65536 | 2.050 / 1.087 → 1.89× | — | 0.023 |
| 131072 | 3.399 / 0.545 → **6.24×** | — | 0.012 |

**Crossover ~16k positions at 12×64, ~8k at 32×128.** distilgpt2 caps at 1024,
so every winning row is **synthesized** — real K/V at the model's head geometry,
no token generated. Only the 1024 row is a real forward pass.

**Root cause of the loss, measured:** `index_select` materializes the selection,
so a decode step reads the chosen keys *and writes them back*. At 12×64 /
65536, gather alone costs **2.58 ms** against **0.82 ms** for attention over the
same slice. Sparsity pays only once density drops far enough that the write
beats a dense read. Fix is a fused decode kernel; the merged kernel cannot serve
decode, it requires `q.shape == k.shape`.

### Schedule build is not the bottleneck

Context 960, 64 tokens: schedule **44 ms** against attention **199 ms** — ~18%
of the sparse path. `IncrementalSalience` fires only on the token completing a
block, verified as exactly once per 64 decode steps, bit-identical to a batch
rebuild.

---

## 10. Upstream kernel: two silent failure modes

Found while wrapping it; neither is caught by the kernel and both are worth a
follow-up PR to `triton-lang/kernels`.

| input | behaviour |
|---|---|
| empty CSR row | `acc/0` → returns **NaN with no error** |
| out-of-range block index | out-of-bounds load; the only mask is `k_pos < seq`, which a negative index passes |

Also: the merged `dense_masked_attention` is 2D-only. On 4D input `q @ k.T`
reverses all four dims and throws a confusing broadcast error. Sigmoid's wrapper
loops it over flattened batch×heads.


---

## 11. Topological vision

Single frame, RTX 4060 host, `TopoImageEncoder` defaults (12 thresholds, 4x4 patch grid).

| resolution | `chi` only | full encode | 50 ms budget |
|---|---|---|---|
| 64x64 | 0.020 ms | 0.75 ms | fits |
| 128x128 | 0.049 ms | 1.34 ms | fits |
| 256x256 | 0.126 ms | 4.41 ms | fits |
| 480x640 | — | 15.49 ms | fits |

Cubical against Vietoris-Rips on the same 128x128 image:

| method | input | time |
|---|---|---|
| `chi = F-H-V+Q` | 5025 foreground px | **0.044 ms** |
| `h0_barcode` (MST) | 419 px, **subsampled** | 127 ms |

~2900x, and the subsampling was mandatory: Rips cannot ingest the full mask.

Sensitivity, on a synthetic sequence where a hole opens and closes:

| feature | corr vs ground-truth hole presence |
|---|---|
| `b1` channel | **+1.0000** |
| best single feature | 1.0000 (index 13) |

`SigmoidWorldModel.fit()` accepted the `(120, 50)` encoding directly and imagined
4 steps, so the `(T, D)` contract holds end to end.

**Bug found by that correlation.** The fallback threshold was the median. On a
bimodal frame the median *is* the background value, so `img <= median` selects
every pixel, b0/b1 go constant, and the correlation came back **NaN**. The
fallback is now a certified widest-gap threshold with no chi target.

---

## 12. Real-time execution

`SafetyController.check` over a 12-vector, 20k calls:

| statistic | value | budget |
|---|---|---|
| p50 | 3.6 us | — |
| p99 | 9.8 us | 1 ms — **passes** |
| **max** | **9458 us** | 1 ms — **fails by 9x** |

The excursion is a CPython GC pause or an OS scheduling slice; the hot path
allocates nothing. Under a deliberately wedged policy thread holding the GIL,
p99 measured 8.5 us — still inside budget, so the independence claim holds.

**Verdict: sub-millisecond at p99, not hard real-time.** A safety stop belongs in
a separate RT process, a C extension, or a microcontroller.

| component | measurement |
|---|---|
| watchdog false positives, healthy jittery loop | **0** |
| watchdog detection on a real hang | 4.7 ms |
| chunker stale actions discarded | 2 (35.7 ms staleness) |
| contact-rich tier, p99 | 0.049 ms against a 20 ms budget |

---

## 13. Telemetry

| operation | cost |
|---|---|
| `record()` | **0.24 us** |
| tracepoint disabled | **0.07 us** |
| tracepoint enabled | 1.43 us |

Fits a 20 ms control budget with three orders of magnitude to spare.

| property | result |
|---|---|
| replay of a seeded pipeline | **bit-identical** (`np.array_equal`, 64 steps) |
| replay of an *unseeded* pipeline | correctly reported non-exact |
| ring buffer at capacity 100, 1000 records | capped at 100, 900 dropped and counted |
| save/load round trip | arrays bit-identical |
| BLAS determinism on this host | True |


---

## Reproducing

```bash
python -m pytest tests/ -q                 # 207 checks
python -m sigmoid.vision.encode            # vision latency + Rips comparison
python -m sigmoid.realtime                 # safety p50/p99/max, watchdog
python -m sigmoid.telemetry                # overhead + bit-exact replay
python examples/s2_rips_corpus.py
python examples/contact_world.py
python examples/falsify_condition.py
python examples/gate_ood_benchmark.py
python examples/kernel_schedule_parity.py
```

The LLM studies require a locally cached `distilgpt2`; kernel parity requires a
checkout of `triton-lang/kernels`. Nothing makes a network call.

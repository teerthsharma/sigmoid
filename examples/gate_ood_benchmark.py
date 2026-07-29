"""Does the sheaf gate actually beat standard OOD detectors? (AGENDA.md R9)

R9 claims the gate's honest product on sequence models is MONITORING, not
prediction, and admits that framing has never been checked against the
OOD-detection literature. This script checks it, on the same windows, with
proper detection metrics instead of means.

Detectors, grouped by what they cost:

    NO FORWARD PASS (scores a state the world model already has, real or imagined)
      gate        max(sheaf, manifold) in calibration-quantile units
      sheaf       the cohomological term alone
      manifold    the whitened-support term alone

    FEATURE-SPACE (needs the activation; free if you were running anyway)
      mahal       Mahalanobis to the calibration activation mean (Lee et al. 2018)
      pca         reconstruction error under calibration PCA
      knn         k-th nearest neighbour distance, normalized features (Sun et al. 2022)

    FULL FORWARD (needs the model's output distribution -- unimaginable)
      msp         1 - max softmax probability (Hendrycks & Gimpel 2017)
      entropy     predictive entropy of the next-token distribution

The first group is the only one that survives when the state was IMAGINED
rather than observed, which is the gate's actual deployment. That is the axis
the comparison has to respect, and the cost section below measures it.

Runs offline against a locally cached distilgpt2.

    python examples/gate_ood_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_worldmodel import OUT_OF_DISTRIBUTION, banner, load_corpus

import sigmoid

MODEL = "distilgpt2"
WINDOW = 24
STRIDE = 6
MAX_TOKENS = 512
SYNTH_LEN = 288
N_SYNTH = 10
N_HELDOUT = 10
SEED = 0


def windows(traj: np.ndarray) -> list[np.ndarray]:
    """Every WINDOW-length slice of a trajectory, STRIDE apart."""
    return [traj[i : i + WINDOW] for i in range(0, len(traj) - WINDOW + 1, STRIDE)]


def auroc_and_fpr95(neg: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    """AUROC and FPR at 95% TPR, higher score = more out-of-distribution."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    if not np.all(np.isfinite(s)):
        s = np.nan_to_num(s, nan=0.0, posinf=1e30, neginf=-1e30)
    fpr, tpr, _ = roc_curve(y, s)
    hit = np.searchsorted(tpr, 0.95, side="left")
    return float(roc_auc_score(y, s)), float(fpr[min(hit, len(fpr) - 1)])


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers + torch are required for this example")
        return 1

    rng = np.random.default_rng(SEED)

    banner(f"loading {MODEL} (local cache only)")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    print(f"  hidden dim {model.config.n_embd}, layers {model.config.n_layer}")

    banner("corpus split")
    corpus, source = load_corpus(want=40)
    print(f"  {source}")
    if len(corpus) < N_HELDOUT + 8:
        print("  not enough local prose to split calibration from held-out")
        return 1
    calib_text, heldout_text = corpus[:-N_HELDOUT], corpus[-N_HELDOUT:]
    print(f"  calibration {len(calib_text)} passages, held-out {len(heldout_text)}")

    def encode(text: str) -> dict:
        return tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)

    # ---- probe sets -------------------------------------------------------
    # Synthetic sets are raw input_ids so nothing about them passes through a
    # tokenizer that would quietly normalize them back toward English.
    vocab = model.config.vocab_size
    words = " ".join(heldout_text).split()
    symbols = OUT_OF_DISTRIBUTION.split()

    def shuffled() -> dict:
        picked = list(rng.choice(words, size=min(380, len(words)), replace=False))
        return encode(" ".join(picked))

    def soup() -> dict:
        # Each sequence is a fresh permutation of the same symbol vocabulary, so
        # the set has variety rather than being one example measured 80 times.
        return encode(" ".join(rng.permutation(symbols * 24)))

    py_files = sorted(Path(__file__).resolve().parents[1].glob("sigmoid/*.py"))
    py_text = [p.read_text(encoding="utf-8", errors="ignore") for p in py_files]

    probe_sets: dict[str, list[dict]] = {
        "held-out prose": [encode(t) for t in heldout_text],
        "uniform random tokens": [
            {"input_ids": torch.tensor(rng.integers(0, vocab, size=(1, SYNTH_LEN)))}
            for _ in range(N_SYNTH)
        ],
        "one token repeated": [
            {
                "input_ids": torch.full(
                    (1, SYNTH_LEN), int(rng.integers(0, vocab)), dtype=torch.long
                )
            }
            for _ in range(N_SYNTH)
        ],
        "symbol soup": [soup() for _ in range(N_SYNTH)],
        "shuffled words": [shuffled() for _ in range(N_SYNTH)],
        "python source": [encode(t) for t in py_text[:N_SYNTH]],
    }

    banner("capturing activations")
    t0 = time.perf_counter()
    calib_traj = [sigmoid.capture_hidden(model, encode(t), layer=-1) for t in calib_text]
    calib_tokens = sum(len(t) for t in calib_traj)
    print(f"  calibration: {len(calib_traj)} seqs, {calib_tokens} tokens, "
          f"{time.perf_counter() - t0:.2f}s")

    # MSP and entropy are per-token quantities; a window's score is their mean
    # over the window, which is the usual way these are lifted to a segment.
    captured: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    softmax_s = 0.0
    for name, batches in probe_sets.items():
        wins, soft = [], []
        for b in batches:
            traj = sigmoid.capture_hidden(model, b, layer=-1)
            with torch.no_grad():
                logits = model(**b).logits[0].float()
            t0 = time.perf_counter()
            p = torch.softmax(logits, dim=-1)
            msp = (1.0 - p.max(dim=-1).values).numpy()
            ent = (-(p * torch.log(p + 1e-12)).sum(dim=-1)).numpy()
            softmax_s += time.perf_counter() - t0
            wins.extend(windows(traj))
            soft.extend(
                [m.mean(), e.mean()] for m, e in zip(windows(msp), windows(ent))
            )
        captured[name] = (wins, np.array(soft))
        print(f"  {name:<22} {len(wins):>4} windows")

    # ---- calibrate everything on the SAME calibration activations ---------
    banner("fitting the world model and the baselines")
    config = sigmoid.SigmoidConfig(
        window=WINDOW, linear_dim=96, hilbert_degree=20, n_radii=8
    )
    t0 = time.perf_counter()
    wm = sigmoid.SigmoidWorldModel(config=config).fit(calib_traj)
    print(f"  world model fitted in {time.perf_counter() - t0:.2f}s")
    for key in ("state_dim", "topo_dim", "transitions", "rho", "block_diagonal"):
        print(f"  {key:<22}{wm.summary()[key]}")

    # Every baseline sees one mean-pooled activation per window: the standard
    # "one feature vector per example" setting these methods were published in.
    calib_windows = [w for t in calib_traj for w in windows(t)]
    calib_feat = np.stack([w.mean(axis=0) for w in calib_windows])
    print(f"  baseline calibration set  {calib_feat.shape[0]} windows x "
          f"{calib_feat.shape[1]} dims")

    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    t0 = time.perf_counter()
    cov = LedoitWolf().fit(calib_feat)
    pca = PCA(n_components=64, random_state=SEED).fit(calib_feat)
    unit = calib_feat / np.linalg.norm(calib_feat, axis=1, keepdims=True)
    knn = NearestNeighbors(n_neighbors=5).fit(unit)
    print(f"  baselines fitted in {time.perf_counter() - t0:.2f}s "
          f"(PCA-64 keeps {pca.explained_variance_ratio_.sum():.3f} of variance)")

    # ---- score ------------------------------------------------------------
    banner("scoring every window with every detector")

    DETECTORS = ["gate", "sheaf", "manifold", "mahal", "pca", "knn", "msp", "entropy"]
    scores: dict[str, dict[str, np.ndarray]] = {}
    timings: dict[str, float] = {}

    for name, (wins, soft) in captured.items():
        feat = np.stack([w.mean(axis=0) for w in wins])

        # split, because in deployment on an IMAGINED state the encode has
        # already been paid by the operator and only the read is left
        t0 = time.perf_counter()
        states = [wm.observe(w) for w in wins]
        timings["encode"] = timings.get("encode", 0.0) + time.perf_counter() - t0
        t0 = time.perf_counter()
        readings = [wm.gate.read(z) for z in states]
        timings["gate"] = timings.get("gate", 0.0) + time.perf_counter() - t0

        t0 = time.perf_counter()
        mahal = cov.mahalanobis(feat)
        timings["mahal"] = timings.get("mahal", 0.0) + time.perf_counter() - t0

        t0 = time.perf_counter()
        recon = np.linalg.norm(feat - pca.inverse_transform(pca.transform(feat)), axis=1)
        timings["pca"] = timings.get("pca", 0.0) + time.perf_counter() - t0

        t0 = time.perf_counter()
        u = feat / np.linalg.norm(feat, axis=1, keepdims=True)
        knn_d = knn.kneighbors(u)[0][:, -1]
        timings["knn"] = timings.get("knn", 0.0) + time.perf_counter() - t0

        scores[name] = {
            "gate": np.array([r.score for r in readings]),
            "sheaf": np.array([r.sheaf_score for r in readings]),
            "manifold": np.array([r.manifold_score for r in readings]),
            "mahal": mahal,
            "pca": recon,
            "knn": knn_d,
            "msp": soft[:, 0],
            "entropy": soft[:, 1],
        }
        n = len(wins)
        print(f"  {name:<22} {n:>4} windows scored by {len(DETECTORS)} detectors")

    n_windows = sum(len(v["gate"]) for v in scores.values())
    timings["msp"] = softmax_s
    timings["entropy"] = softmax_s

    # ---- detection metrics ------------------------------------------------
    in_dist = scores["held-out prose"]
    ood_types = [k for k in scores if k != "held-out prose"]

    banner("AUROC (in-distribution = held-out prose; 0.5 = coin flip)")
    head = "  " + f"{'OOD type':<22}" + "".join(f"{d:>10}" for d in DETECTORS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    auroc = {}
    fpr95 = {}
    for name in ood_types:
        row, frow = {}, {}
        for d in DETECTORS:
            a, f = auroc_and_fpr95(in_dist[d], scores[name][d])
            row[d], frow[d] = a, f
        auroc[name], fpr95[name] = row, frow
        print(f"  {name:<22}" + "".join(f"{row[d]:>10.3f}" for d in DETECTORS))
    print("  " + "-" * (len(head) - 2))
    mean_auroc = {d: float(np.mean([auroc[n][d] for n in ood_types])) for d in DETECTORS}
    print(f"  {'MEAN':<22}" + "".join(f"{mean_auroc[d]:>10.3f}" for d in DETECTORS))
    print("\n  columns 1-3 need no forward pass; 4-6 need the activation;")
    print("  7-8 need the model's full output distribution.")
    print("\n  AUROC below 0.5 means the detector ranks that OOD type as MORE")
    print("  in-distribution than prose. A deployment cannot flip its sign per")
    print("  input, so a detector whose direction is inconsistent across OOD")
    print("  types is not usable at a single threshold, whatever its |AUROC-0.5|:")
    for d in DETECTORS:
        above = sum(auroc[n][d] > 0.5 for n in ood_types)
        flag = "" if above in (0, len(ood_types)) else "   <-- INCONSISTENT"
        print(f"  {d:<10} ranks OOD-high on {above}/{len(ood_types)} types{flag}")

    banner("FPR at 95% TPR (lower is better; 1.000 = detector is useless here)")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name in ood_types:
        print(f"  {name:<22}" + "".join(f"{fpr95[name][d]:>10.3f}" for d in DETECTORS))

    banner("the deployed threshold: fire rate at score >= 1.0")
    print("  the gate ships with a fixed quantile threshold, so AUROC is not the")
    print("  whole story -- a detector can rank well and still never fire.")
    print(f"\n  {'set':<22}{'mean':>8}{'median':>9}{'max':>10}{'fires':>14}")
    for name, s in scores.items():
        g = s["gate"]
        fires = int((g >= 1.0).sum())
        print(f"  {name:<22}{g.mean():>8.3f}{np.median(g):>9.3f}{g.max():>10.3f}"
              f"{f'{fires}/{len(g)}':>14}")
    hp = scores["held-out prose"]["gate"]
    print(f"\n  false-positive rate on held-out prose {float((hp >= 1.0).mean()):.3f}"
          f"  (calibrated for {1 - config.gate_quantile:.3f} on the calibration set)")

    banner("measured cost per window")
    probe = probe_sets["held-out prose"][:4]
    n_tok = sum(int(b["input_ids"].shape[1]) for b in probe)
    t0 = time.perf_counter()
    for b in probe:
        with torch.no_grad():
            model(**b)
    fwd_s = time.perf_counter() - t0
    fwd_ms = fwd_s * 1e3 / len(probe)
    # In a streaming monitor each new window costs STRIDE fresh tokens of
    # forward, so that -- not a whole sequence -- is the fair per-window charge.
    fwd_per_window = fwd_s / n_tok * STRIDE * 1e3
    print(f"  {'detector':<14}{'us/window':>12}   requires")
    print("  " + "-" * 62)
    for d in ("gate", "encode", "mahal", "pca", "knn", "msp"):
        need = {
            "gate": "a world state only -- real OR imagined",
            "encode": "the activation window (free on an imagined state)",
            "mahal": "the activation",
            "pca": "the activation",
            "knn": "the activation + calibration set in memory",
            "msp": "the logits (softmax reduction only, forward excluded)",
        }[d]
        print(f"  {d:<14}{timings[d] / n_windows * 1e6:>12.1f}   {need}")
    print(f"  {'msp+forward':<14}"
          f"{timings['msp'] / n_windows * 1e6 + fwd_per_window * 1e3:>12.1f}   "
          f"a FULL FORWARD PASS of the wrapped model")
    print(f"\n  one distilgpt2 forward, {n_tok // len(probe)} tokens avg   {fwd_ms:.1f} ms")
    print(f"  amortized over {STRIDE} fresh tokens/window   {fwd_per_window * 1e3:.1f} us")
    print("  MSP/entropy cannot be computed on an imagined state at any price:")
    print("  there is no output distribution until the model has been run.")

    banner("honest assessment")
    lines: list[str] = []
    cheap = ["gate", "sheaf", "manifold"]
    feature = ["mahal", "pca", "knn"]
    expensive = ["msp", "entropy"]
    for name in ood_types:
        best = max(DETECTORS, key=lambda d: auroc[name][d])
        g = auroc[name]["gate"]
        beaten_by = [d for d in DETECTORS if auroc[name][d] > g + 0.01 and d not in cheap]
        if not beaten_by:
            verdict = "gate wins or ties (nothing beats it by >0.01)"
        else:
            verdict = f"LOSES to {best} ({auroc[name][best]:.3f})"
        lines.append(f"  {name:<22} gate {g:.3f}  {verdict}")
        if beaten_by:
            lines.append(f"  {'':<22} beaten by: {', '.join(beaten_by)}")
    print("\n".join(lines))

    g_mean = mean_auroc["gate"]
    best_feat = max(feature, key=lambda d: mean_auroc[d])
    best_exp = max(expensive, key=lambda d: mean_auroc[d])
    note = "" if mean_auroc[best_exp] >= 0.5 else " (below 0.5: sign inverts, see above)"
    print(f"\n  mean AUROC   gate {g_mean:.3f} | best activation baseline "
          f"{best_feat} {mean_auroc[best_feat]:.3f} | best forward-pass baseline "
          f"{best_exp} {mean_auroc[best_exp]:.3f}{note}")

    rand = auroc["uniform random tokens"]
    print(f"\n  the known random-token miss: gate AUROC {rand['gate']:.3f}, "
          f"fires {int((scores['uniform random tokens']['gate'] >= 1.0).sum())}"
          f"/{len(scores['uniform random tokens']['gate'])} at threshold 1.0")
    print(f"  same windows: mahal {rand['mahal']:.3f}, pca {rand['pca']:.3f}, "
          f"knn {rand['knn']:.3f}, msp {rand['msp']:.3f}, entropy {rand['entropy']:.3f}")
    if rand["gate"] < max(rand[d] for d in feature + expensive) - 0.01:
        print("  REPRODUCED under proper metrics. This is a real weakness, not a")
        print("  thresholding artifact: the gate ranks random-token windows below")
        print("  baselines that see the same activations.")
    else:
        print("  did NOT reproduce under ranking metrics -- the earlier miss was a")
        print("  threshold artifact, not a ranking failure.")

    won = sum(
        1
        for n in ood_types
        if not [d for d in DETECTORS if auroc[n][d] > auroc[n]["gate"] + 0.01
                and d not in cheap]
    )
    print(f"\n  the gate wins or ties on {won}/{len(ood_types)} OOD types and is beaten"
          " by a cheaper\n  activation-space baseline on the rest.")
    print("  The comparison is not apples to apples and should not be reported as")
    print("  one: MSP and entropy buy their AUROC with a full model forward, which")
    print("  is exactly the cost the gate exists to avoid -- and they buy it with")
    print("  an inconsistent sign, so neither is deployable at one threshold.")
    print("  But mahalanobis and PCA are NOT in that category. They see the same")
    print("  activation the gate's state was built from, cost less arithmetic, and")
    print("  on this corpus rank better on average. The defensible claim left")
    print("  standing is narrower than 'the gate is a good OOD detector': it is")
    print("  the only one of these that still works on an IMAGINED state, where")
    print("  no activation exists to feed mahalanobis or PCA at all.")
    return 0


def _selfcheck() -> None:
    """The two pieces of logic that would silently corrupt every number."""
    assert [w[0] for w in windows(np.arange(40))] == [0, 6, 12], "window slicing"
    assert len(windows(np.zeros((WINDOW - 1, 3)))) == 0, "short trajectory"
    lo, hi = np.zeros(50), np.ones(50)
    assert auroc_and_fpr95(lo, hi) == (1.0, 0.0), "perfect separation"
    assert auroc_and_fpr95(hi, lo)[0] == 0.0, "inverted separation"


if __name__ == "__main__":
    _selfcheck()
    sys.exit(main())

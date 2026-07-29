"""Convert a real LLM into a world model, then measure whether it worked.

Runs entirely offline against a locally cached distilgpt2. Every number this
prints is measured in-process; nothing is quoted from elsewhere.

    python examples/llm_worldmodel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sigmoid

MODEL = "distilgpt2"

# Calibration needs far more transitions than state dimensions, and a first
# pass with five hand-written paragraphs (378 transitions against an 80-dim
# state) produced an operator with spectral radius 16 that lost to predicting
# the dataset mean. Local markdown supplies a few thousand tokens of genuinely
# varied English without touching the network.
CORPUS_GLOBS = [
    "../../topological-ml-toolkit/docs/**/*.md",
    "../../lambda-topo-main/*.md",
    "../../faraday-main/docs/**/*.md",
    "../../hamliton-main/*.md",
    "../../Epsilon-Hollow-main/*.md",
]

FALLBACK_CORPUS = [
    "The history of computation begins long before electronic machines existed. "
    "Mechanical calculators, tide predictors, and looms that read punched cards "
    "all encoded procedures into physical arrangements of matter. What changed in "
    "the twentieth century was not the idea of automatic calculation but the "
    "discovery that a single machine could be made to imitate any other, provided "
    "it was given a description of that machine as data. This is the insight that "
    "separates a calculator from a computer, and it is the reason software exists "
    "as a category distinct from hardware at all.",
    "Ocean currents move heat around the planet on timescales that dwarf weather. "
    "Warm surface water travels poleward, cools, grows dense with salt, and sinks "
    "into the abyss, where it creeps back toward the equator over centuries. This "
    "overturning circulation is why northern Europe is far milder than its latitude "
    "would suggest. Small changes in freshwater input near the poles can slow the "
    "sinking, and the climate record shows that such slowdowns have happened "
    "abruptly in the past, reorganizing regional climates within decades.",
    "A proof is not the same thing as a demonstration that something is true. "
    "A proof is a finite object that another mathematician can check step by step "
    "without trusting the person who wrote it. This is why proofs became longer and "
    "more formal as mathematics grew: not because mathematicians became less "
    "insightful, but because the community grew larger and less able to rely on "
    "shared intuition. The formalism is a social technology as much as a logical "
    "one, a way of transmitting certainty between strangers.",
    "Bird migration was a genuine mystery for most of recorded history. Aristotle "
    "believed that redstarts transformed into robins for the winter, and as late as "
    "the eighteenth century serious naturalists proposed that swallows hibernated in "
    "the mud at the bottom of ponds. The truth turned out to be stranger than the "
    "speculation: small songbirds weighing less than a bar of soap cross entire "
    "oceans, navigating by stars, by the earth's magnetic field, and by a memory of "
    "landscape they have never personally seen.",
    "Concrete is the most used material on earth after water, and its chemistry is "
    "still not fully understood. Cement hydrates into an amorphous gel whose "
    "structure resists the crystallographic methods that explain most solids. "
    "Engineers therefore design concrete largely through empirical rules refined "
    "over a century of construction. Roman concrete, which survives in harbors that "
    "have been underwater for two thousand years, achieved durability through a "
    "mechanism only recently identified: seawater reacting with volcanic ash to grow "
    "reinforcing minerals inside the cracks.",
]

OUT_OF_DISTRIBUTION = "!!! ;;; @@@ 42 42 42 zzzz qqqq !!! ;;; @@@ zzzz qqqq 42 !!!"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_corpus(min_chars: int = 900, want: int = 40) -> list[str]:
    """Collect varied local prose. Falls back to the built-in paragraphs.

    Returns (passages, source) so the caller can say which was used. An earlier
    version returned bare passages and silently fell back when the globs were
    wrong by one directory level, which quietly invalidated a whole run.
    """
    import re

    here = Path(__file__).resolve().parent
    passages: list[str] = []
    seen: set[str] = set()
    for pattern in CORPUS_GLOBS:
        for path in sorted(here.glob(pattern)):
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # strip code fences, tables and markup so this stays natural language
            raw = re.sub(r"```.*?```", " ", raw, flags=re.S)
            raw = re.sub(r"^\s*[|#>*\-].*$", " ", raw, flags=re.M)
            raw = re.sub(r"https?://\S+", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            for start in range(0, len(raw), min_chars):
                chunk = raw[start : start + min_chars]
                if len(chunk) >= min_chars and chunk[:60] not in seen:
                    seen.add(chunk[:60])
                    passages.append(chunk)
            if len(passages) >= want:
                break
        if len(passages) >= want:
            break
    if len(passages) >= 8:
        return passages[:want], f"{len(passages[:want])} local markdown passages"
    return FALLBACK_CORPUS, "BUILT-IN FALLBACK (globs matched nothing -- too little data)"


def main() -> int:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers is required for this example")
        return 1

    banner(f"loading {MODEL} (local cache only)")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    print(f"  hidden dim {model.config.n_embd}, layers {model.config.n_layer}")

    banner("capturing activation trajectories")
    corpus, source = load_corpus()
    print(f"  corpus: {source}")
    batches = [
        tok(text, return_tensors="pt", truncation=True, max_length=512)
        for text in corpus
    ]
    t0 = time.perf_counter()
    trajectories = [sigmoid.capture_hidden(model, b, layer=-1) for b in batches]
    capture_s = time.perf_counter() - t0
    total = sum(t.shape[0] for t in trajectories)
    print(f"  {len(trajectories)} sequences, {total} tokens total, {capture_s:.2f}s")

    banner("fitting the world model")
    config = sigmoid.SigmoidConfig(
        window=24, linear_dim=96, hilbert_degree=20, n_radii=8
    )
    t0 = time.perf_counter()
    wm = sigmoid.SigmoidWorldModel(config=config).fit(trajectories)
    print(f"  fitted in {time.perf_counter() - t0:.2f}s")
    for key, value in wm.summary().items():
        print(f"  {key:<24}{value}")

    banner("ablation: does topology earn its dimensions?")

    def real_forward():
        sigmoid.capture_hidden(model, batches[0], layer=-1)

    report = sigmoid.compare(
        trajectories,
        config=config,
        horizons=(1, 4, 16),
        holdout=0.4,
        real_forward_fn=real_forward,
    )
    print(report.table())

    sig = next(a for a in report.arms if a.name == "sigmoid")
    print(f"\n  encode one window   {sig.encode_ms:.3f} ms")
    print(f"  imagine one step    {sig.imagine_us:.1f} us")
    if report.real_forward_ms:
        speedup = report.real_forward_ms * 1e3 / sig.imagine_us
        print(f"  real model forward  {report.real_forward_ms:.1f} ms")
        print(f"  imagination is      {speedup:,.0f}x cheaper per step")

    banner("the certificate trade: accuracy bought against a guarantee")
    print("  rho_max   rho      k=1 NRMSE   bound@16    informative")
    for rho_max in (None, 0.99, 0.9):
        cfg = sigmoid.SigmoidConfig(**{**vars(config), "rho_max": rho_max})
        alt = sigmoid.SigmoidWorldModel(config=cfg).fit(trajectories)
        nrmse, _ = sigmoid.rollout_error(alt, trajectories[-8:], [1])
        bound = alt.certificate(16).bound
        label = "none" if rho_max is None else f"{rho_max:.2f}"
        mark = "yes" if bound < 1.0 else "no (vacuous)"
        print(
            f"  {label:<9} {alt.operator.rho_:<8.4f} {nrmse[1]:<11.4f} "
            f"{bound:<11.4g} {mark}"
        )
    print("  a contraction is not free: it is bought with one-step accuracy")

    banner("the gate: what does it actually detect?")
    print("  The gate measures deviation from the CALIBRATION corpus, not")
    print("  'bad text'. Calibrated on markdown, symbol soup is in-distribution.")
    print()
    import torch

    rng = np.random.default_rng(0)
    probes: list[tuple[str, np.ndarray]] = []
    for label, batch in (
        ("held-out prose", batches[-1]),
        ("symbol soup", tok(OUT_OF_DISTRIBUTION * 4, return_tensors="pt")),
        (
            "uniform random tokens",
            {"input_ids": torch.tensor(rng.integers(0, 50256, size=(1, 64)))},
        ),
        (
            "one token repeated",
            {"input_ids": torch.full((1, 64), 15496, dtype=torch.long)},
        ),
    ):
        traj = sigmoid.capture_hidden(model, batch, layer=-1)
        if traj.shape[0] >= config.window:
            probes.append((label, traj))

    for label, traj in probes:
        scores = [
            wm.gate.read(wm.observe(traj[i - config.window : i])).score
            for i in range(config.window, min(traj.shape[0], 96), 8)
        ]
        reading = wm.gate.read(wm.observe(traj[: config.window]))
        print(
            f"  {label:<22} mean score {np.mean(scores):.3f}  "
            f"max {np.max(scores):.3f}  fires={sum(s >= 1.0 for s in scores)}/{len(scores)}"
            f"  ({reading.reason})"
        )

    banner("imagination with grounding")
    roll = wm.imagine(trajectories[0][:24], steps=32)
    print(f"  imagined {len(roll)} steps before grounding")
    print(f"  grounded at step    {roll.grounded_at}")
    print(f"  trusted steps       {roll.trusted_steps}")
    scores = [f"{r.score:.2f}" for r in roll.readings[:12]]
    print(f"  gate score by step  {' '.join(scores)}")

    banner("the attractor")
    attractor = wm.attractor_hidden
    if attractor is not None:
        norms = [np.linalg.norm(attractor - t.mean(axis=0)) for t in trajectories]
        print(f"  ||z*|| in hidden space      {np.linalg.norm(attractor):.3f}")
        print(f"  mean distance to sequence means {np.mean(norms):.3f}")
        print(f"  Banach iteration converged  {wm.operator.fixed_point_converged}")
    else:
        print("  operator has an eigenvalue at 1; no isolated fixed point")

    banner("verdict")
    for name, ok in report.gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for note in report.notes:
        print(f"  note: {note}")
    print(f"\n  {'ALL GATES PASSED' if report.passed else 'GATES FAILED'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

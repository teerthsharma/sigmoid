"""Where in the stack, if anywhere, does topology earn its dimensions?

Section 6 of SIGMOID.md listed "wrong layer?" as the top open question: every
LLM number was measured on the last hidden layer, which is the most
token-locked and least geometric representation in the model. This sweeps every
layer and reports the matched-dimension ablation at each one.

    python examples/layer_sweep.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

from llm_worldmodel import MODEL, load_corpus

import sigmoid

HORIZONS = (1, 4, 16)


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)
    model.eval()

    corpus, source = load_corpus()
    print(f"corpus: {source}")
    batches = [
        tok(t, return_tensors="pt", truncation=True, max_length=512) for t in corpus
    ]

    n_layers = model.config.n_layer
    config = sigmoid.SigmoidConfig(
        window=24, linear_dim=96, hilbert_degree=20, n_radii=8
    )

    print(f"\n{'layer':<8}{'k=1':>10}{'k=4':>10}{'k=16':>10}   "
          f"{'lin k=16':>10}{'delta':>10}  {'block':>6}{'rho':>8}")
    print("-" * 78)

    rows = []
    for layer in range(n_layers + 1):
        trajectories = [
            sigmoid.capture_hidden(model, b, layer=layer) for b in batches
        ]
        report = sigmoid.compare(
            trajectories, config=config, horizons=HORIZONS, holdout=0.4
        )
        sig = next(a for a in report.arms if a.name == "sigmoid")
        lin = next(a for a in report.arms if a.name == "linear_only")
        # negative delta means topology helped
        delta = sig.nrmse[16] - lin.nrmse[16]
        wm = sigmoid.SigmoidWorldModel(config=config).fit(trajectories)
        label = "embed" if layer == 0 else f"{layer}"
        print(
            f"{label:<8}{sig.nrmse[1]:>10.4f}{sig.nrmse[4]:>10.4f}"
            f"{sig.nrmse[16]:>10.4f}   {lin.nrmse[16]:>10.4f}{delta:>+10.4f}"
            f"  {str(wm.block_diagonal_):>6}{sig.rho:>8.3f}"
        )
        rows.append((label, delta, sig.nrmse[16]))

    print()
    helped = [r for r in rows if r[1] < -1e-4]
    if helped:
        best = min(helped, key=lambda r: r[1])
        print(f"topology helps at layer(s): {', '.join(r[0] for r in helped)}")
        print(f"strongest at layer {best[0]} ({best[1]:+.4f} NRMSE at k=16)")
    else:
        print("topology does not help at ANY layer of this model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point: `python -m sigmoid ...`.

    sigmoid fit     TRAJ.npy -o world.npz     calibrate a world model
    sigmoid roll    world.npz TRAJ.npy -k 16  imagine forward, print gate scores
    sigmoid bench   TRAJ.npy                  run the ablation + gates

Trajectories are .npy files of shape (T, D) or .npz archives whose every array
is one (T, D) episode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .bench import compare
from .engine import SigmoidConfig, SigmoidWorldModel


def _load(path: str) -> list[np.ndarray]:
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as data:
            return [np.asarray(data[k], dtype=np.float64) for k in data.files]
    return [np.asarray(np.load(p), dtype=np.float64)]


def _config(args: argparse.Namespace) -> SigmoidConfig:
    return SigmoidConfig(
        window=args.window,
        linear_dim=args.linear_dim,
        hilbert_degree=args.degree,
        use_h1=args.h1,
        rho_max=args.rho_max,
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--linear-dim", dest="linear_dim", type=int, default=32)
    p.add_argument("--degree", type=int, default=24, help="Hilbert embedding degree")
    p.add_argument("--h1", action="store_true", help="include H1 (slow)")
    p.add_argument(
        "--rho-max",
        dest="rho_max",
        type=float,
        default=None,
        help="force a contraction (costs accuracy; buys a certificate)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sigmoid", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="calibrate a world model")
    p_fit.add_argument("trajectories")
    p_fit.add_argument("-o", "--out", default="world.npz")
    _add_common(p_fit)

    p_roll = sub.add_parser("roll", help="imagine forward from a trajectory")
    p_roll.add_argument("world")
    p_roll.add_argument("trajectories")
    p_roll.add_argument("-k", "--steps", type=int, default=16)

    p_bench = sub.add_parser("bench", help="ablation and promotion gates")
    p_bench.add_argument("trajectories")
    p_bench.add_argument("--horizons", default="1,4,16")
    _add_common(p_bench)

    args = parser.parse_args(argv)

    if args.cmd == "fit":
        wm = SigmoidWorldModel(config=_config(args)).fit(_load(args.trajectories))
        out = wm.save(args.out)
        for key, value in wm.summary().items():
            print(f"{key:<24}{value}")
        print(f"saved -> {out}")
        return 0

    if args.cmd == "roll":
        wm = SigmoidWorldModel.load(args.world)
        traj = _load(args.trajectories)[0]
        if traj.shape[0] < wm.config.window:
            print(f"trajectory shorter than window {wm.config.window}", file=sys.stderr)
            return 1
        roll = wm.imagine(traj[: wm.config.window], args.steps)
        print(f"imagined {len(roll)} of {args.steps} steps")
        print(f"grounded at {roll.grounded_at}, trusted {roll.trusted_steps}")
        for i, r in enumerate(roll.readings):
            print(f"  step {i:>3}  score {r.score:>7.3f}  {r.reason}")
        return 0

    horizons = tuple(int(h) for h in args.horizons.split(","))
    report = compare(_load(args.trajectories), config=_config(args), horizons=horizons)
    print(report.table())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

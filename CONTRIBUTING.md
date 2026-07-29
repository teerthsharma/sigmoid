# Contributing

The bar here is evidence, not style.

## The one rule

**A claim needs a control.** Every number this project got wrong, it got wrong
because a control was missing — never because the mathematics failed:

| claim | what was missing |
|---|---|
| "~30% better rollouts" | the ablation co-varied PCA rank with topology |
| "bit-identical kernel parity" | only continuous inputs were sampled |
| "a useful cheap OOD detector" | no baselines had been run |
| "needs a fixed physical distance" | no scale-varying control existed |

So a PR that claims an improvement should say what it was measured against.

## Practical

```bash
pip install -e ".[dev]"
pytest -q                      # 57 checks
ruff check . && ruff format --check .
python -m sigmoid.schedule     # salience self-check
```

- Core depends on **numpy and scipy only**. torch, transformers, ripser and
  sklearn are optional and must degrade gracefully.
- Each test file runs standalone (`python tests/test_x.py`) as well as under
  pytest. Keep that.
- Persistence stays out of the hot path. H₀ via MST is fine; Rips is
  calibration-only.
- Comments explain *why*, ideally with the measurement that forced the choice.
  There are several in the source; match them.

## Benchmarks

`sigmoid.compare()` runs every arm at matched budget and reports failed gates
rather than dropping losing arms. Two ablations, because they answer different
questions:

- `no_topology_same_u` — same linear channel, ψ deleted. **This is the arm the
  topology claim is judged on**; nothing else co-varies.
- `linear_only` — matched *total* dimension. Answers the budget question: is ψ
  the best use of those dimensions?

If a change makes a gate fail, report the failure. Silent truncation of a losing
arm reads as "covered everything" when it did not.

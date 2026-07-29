# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Email **teerthsharma@outlook.com**. Please do not open a public issue for
security matters. Expect an acknowledgement within 72 hours.

## Threat model

`sigmoid` is a numerical library. It opens no sockets, spawns no processes, and
executes no user-supplied source. The realistic concerns are:

- **Untrusted checkpoints.** `SigmoidWorldModel.load` uses
  `numpy.load(..., allow_pickle=True)`, which can execute arbitrary code when
  given a malicious file. **Load only checkpoints you produced or trust.** This
  is the most important line in this document.
- **Untrusted models.** `sigmoid.adapters` runs a forward pass on whatever model
  you hand it, and inherits everything that model does.
- **Resource exhaustion.** Calibration is `O(T·W²·D)`. A hostile trajectory
  shape can make it arbitrarily slow; validate shapes at your trust boundary.

## Out of scope

Numerical accuracy issues, convergence failures, and benchmark disagreements are
correctness bugs, not vulnerabilities — open a public issue for those. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the evidence standard.

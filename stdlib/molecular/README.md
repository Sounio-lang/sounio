# stdlib/molecular

Classical particle simulation in a periodic box.

## What is here

`lj.sio` — module `molecular::lj`, reduced Lennard-Jones units.

- Pair energy and pair virial, with the standard long-range corrections for a truncated potential.
- Periodic cubic box with minimum image. Up to 1024 particles in a caller-owned `[f64; 3072]`.
- Metropolis displacement sweeps at fixed N, V, T.
- Grand-canonical insertion and deletion at a given activity, with the tail energy in the acceptance rule.
- Widom test-particle insertion, and the activity from its mean.
- Mean and standard error over independent runs. No within-run error bar is offered.

## What it is checked against

| Check | File | Reference |
|---|---|---|
| Analytic values, energy bookkeeping, near-ideal limit of the exchange move | `tests/stdlib/molecular/test_lj_core.sio` | closed forms; second virial coefficient |
| Energy and virial of a fixed 400-particle configuration | `tests/stdlib/molecular/test_lj_oracle.sio` | `oracles/lj_config_oracle.cpp`, long double, written independently |
| Liquid and vapour energy and pressure, vapour density at the saturation activity | `examples/molecular/lj_nist_parity.sio` | NIST Lennard-Jones benchmark tables, T* = 0.85, cutoff 3 with tail corrections |
| Widom insertion: dilute gas, and two system sizes at the saturated-vapour density | `examples/molecular/lj_widom_finite_size.sio` | `oracles/lj_b2_oracle.cpp` (second virial coefficient); NIST saturation activity with two finite-size terms |

The two tests under `tests/` pass on Madaros and on lean_single. The NIST driver states its predictions and its pass rule in its header; a second driver, `examples/molecular/lj_widom_finite_size.sio`, covers Widom insertion. Measured outcomes of both are in `examples/molecular/LJ_NIST_PARITY.md`.

## What is not here

One site per particle, one species, a cubic box, an O(N) energy sum per trial move. There is no cell list, no Ewald sum, no rigid molecule, no host framework, no quantum correction and no molecular dynamics integrator. Measured cost on Madaros v0.80.0: about 80 ns per pair distance, so a 500-particle liquid sweep takes about 40 ms.

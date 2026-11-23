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

Also: `host_guest.sio` — module `molecular::host_guest`: the sII clathrate lattice (Fd-3m closed from generators, Falenty 2014 coordinates), one-site guest with Lorentz–Berthelot and quadratic Feynman–Hibbs, grand-canonical trials at a given fugacity, cage assignment, and the parameter gradients behind fluctuation sensitivities. Checked by `tests/stdlib/molecular/test_host_guest_oracle.sio` (lattice, field and gradients against `oracles/host_guest_oracle.cpp`, 1e-10) and by `examples/molecular/h2_sii_gcmc.sio` (occupancies and sensitivities against predictions written before the run; results in `examples/molecular/H2_SII_GCMC.md`).

Also: `ice_rules.sio` — module `molecular::ice_rules`: bonds of a periodic water framework, an Euler-circuit orientation that satisfies the Bernal–Fowler ice rules, loop moves that disorder the protons and drive the bond-winding (polarisation) vector to exactly zero, and rigid three-site water (SPC/E geometry and charges, or any other) placed on the result. `ewald.sio` — module `molecular::ewald`: tinfoil Ewald potential of a fixed neutral charge set at a point, and the Ewald energy of a set of rigid neutral molecules. Both are checked by `tests/stdlib/molecular/test_sii_charges_ewald.sio` against `oracles/sii_ewald_oracle.cpp` (long double, same minstd generator): the configuration fingerprint, the residual dipole, the Ewald potential, the Alavi-H2 energy in a small and a large cage at two splitting parameters, the orientational extremes and Boltzmann factor at four cage centres, and a five-molecule guest–guest energy. The oracle also checks what the test cannot (`oracles/sii_ewald_predictions.txt`, sha256 a0abd95d…, written before it ran): Ewald against the cubic direct sum, the point-quadrupole limit, and the Laplace identity that makes the orientation average vanish.

Also: `langmuir.sio` — module `molecular::langmuir`: the Langmuir constant of a light guest beyond the classical one-site picture. Feynman–Kac paths (Lévy bridges on molecular::lj's generator) give the exact quantum centre-of-mass ratio C_q/C_cl and the quantum pair and triple ratios R₂, R₃ of a multiply occupied cage; the charged (three-site) guest's orientationally averaged constant in the Ewald potential of an ice-rule host gives ln R_c. Checked by `tests/stdlib/molecular/test_langmuir_oracle.sio` against `oracles/langmuir_oracle.cpp` (same algorithms and random stream, 1e-9) and against large independent runs (within 4 standard errors; `oracles/langmuir_predictions.txt`, sha256 19fd4e0a…, written before the library). Results: `examples/molecular/H2_SII_GCMC.md`.

Also: `promoter.sio` — module `molecular::promoter`: a light guest in the small cages of a hydrate whose large cages hold a rigid promoter (THF in sII). The promoter sits at its cage centre with uniform orientations independent of the guest, so the orientation average factorises per molecule; one quadrature over the small cage gives the Langmuir constant with and without the promoter and d ln C / d(ε_H, σ_H), averaged over a deterministic orientation set (Shoemake quaternions from the Halton sequence). A factor ξ scales every guest–promoter well depth. Checked by `tests/stdlib/molecular/test_promoter_oracle.sio` against `oracles/promoter_oracle.cpp` (same quadrature and orientations, 1e-9; `oracles/promoter_predictions.txt`, sha256 2d1f6fd6…, written before the library). For H₂ next to TraPPE-UA THF at 250 K the promoter raises C of the 5¹² by 34 %, the same in SPC/E, TIP4P/Ice and TIP5P to 0.002 in ln C.

| Check | File | Reference |
|---|---|---|
| Analytic values, energy bookkeeping, near-ideal limit of the exchange move | `tests/stdlib/molecular/test_lj_core.sio` | closed forms; second virial coefficient |
| Energy and virial of a fixed 400-particle configuration | `tests/stdlib/molecular/test_lj_oracle.sio` | `oracles/lj_config_oracle.cpp`, long double, written independently |
| Liquid and vapour energy and pressure, vapour density at the saturation activity | `examples/molecular/lj_nist_parity.sio` | NIST Lennard-Jones benchmark tables, T* = 0.85, cutoff 3 with tail corrections |
| Widom insertion: dilute gas, and two system sizes at the saturated-vapour density | `examples/molecular/lj_widom_finite_size.sio` | `oracles/lj_b2_oracle.cpp` (second virial coefficient); NIST saturation activity with two finite-size terms |

The two tests under `tests/` pass on Madaros and on lean_single. The NIST driver states its predictions and its pass rule in its header; a second driver, `examples/molecular/lj_widom_finite_size.sio`, covers Widom insertion. Measured outcomes of both are in `examples/molecular/LJ_NIST_PARITY.md`.

## What is not here

One site per particle, one species, a cubic box, an O(N) energy sum per trial move. `lj.sio` itself has no cell list, no Ewald sum, no rigid molecule, no host framework, no quantum correction and no molecular dynamics integrator (the host framework, the quantum correction and the Ewald sum live in `host_guest.sio` and `ewald.sio`). Measured cost on Madaros v0.80.0: about 80 ns per pair distance, so a 500-particle liquid sweep takes about 40 ms.

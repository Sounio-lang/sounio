# H2 in sII: grand-canonical Monte Carlo against predictions

Driver: `examples/molecular/h2_sii_gcmc.sio` (module `molecular::host_guest`). Model: rigid sII
lattice (136 oxygens per cell, a = 17.31 A), one-site H2 (Buch, 34.2 K / 2.96 A), SPC/E oxygen
(78.20 K / 3.166 A), Lorentz-Berthelot, quadratic Feynman-Hibbs, 12 A cutoff with tail.

The predictions are in the driver header and were written before the run. Pass rule: each
quantity within three combined standard errors (20 block means of the run, plus the
prediction's own sampling error).

## Measured, Madaros v0.80.0 (2026-11-16)

Per cell. Sensitivities from the grand-canonical fluctuations,
d<N>/dtheta = -beta (<N U_theta> - <N><U_theta>).

| Case | Quantity | Run | Predicted | Deviation |
|---|---|---|---|---|
| A: ideal guests, 250 K, 5 MPa, 2·10⁶ trials | N_s | 2.2726 ± 0.0236 | 2.24957 | +1.0σ |
| | N_l | 3.0862 ± 0.0225 | 3.11810 | −1.4σ |
| | dN_s/dε_H | 0.0874 ± 0.0020 | 0.08849 | −0.5σ |
| | dN_l/dε_H | 0.0917 ± 0.0016 | 0.09304 | −0.8σ |
| | dN_s/dσ_H | 0.163 ± 0.060 | 0.2218 | −1.0σ |
| | dN_l/dσ_H | 1.692 ± 0.056 | 1.6234 | +1.2σ |
| B: interaction within a cage, 200 K, 300 MPa, 10⁶ trials | N_s | 15.2301 ± 0.0484 | 15.25805 | −0.6σ |
| | N_l | 16.2935 ± 0.0847 | 16.1337 ± 0.036 | +1.7σ |
| | dN_s/dε_H | 0.0371 ± 0.0031 | 0.03614 | +0.3σ |
| | dN_l/dε_H | 0.0413 ± 0.0053 | 0.0462 ± 0.0028 | −0.8σ |
| | dN_s/dσ_H | 0.84 ± 0.32 | 0.346 ± 0.005 | +1.5σ |
| | dN_l/dσ_H | −20.8 ± 1.1 | −19.31 ± 0.51 | −1.2σ |

All twelve pass (largest deviation 1.7σ). Wall time 14 min on one core.

## Where the predictions come from

- Case A: each cage is an independent Henry adsorber, <N> = (cages) · C · f exactly. C and
  d ln C come from a long-double quadrature of the guest-host field over the Voronoi domain of
  each cage (the oracle of `tests/run-pass/langmuir_h2_sii_lattice_uq.sio` at a finer rule).
- Case B: the cage grand partition function with R_n = Z_n / Z_1^n (n ≤ 4) and dR_n/dθ,
  sampled exactly from the one-guest Boltzmann density in each cage. The interaction between
  neighbouring cages is not in this model; that is why case B switches it off. With it on
  (mode 1), the occupancy rises by about 2.4% in the large cage, a measured, unpredicted term.

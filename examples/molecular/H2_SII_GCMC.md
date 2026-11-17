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

All twelve pass (largest deviation 1.7σ). Wall time 14 min on one core with the full site sum; 5 min with the cell lists of `hg_build_cells` (same trajectory, same output to the printed digits).

## Where the predictions come from

- Case A: each cage is an independent Henry adsorber, <N> = (cages) · C · f exactly. C and
  d ln C come from a long-double quadrature of the guest-host field over the Voronoi domain of
  each cage (the oracle of `tests/run-pass/langmuir_h2_sii_lattice_uq.sio` at a finer rule).
- Case B: the cage grand partition function with R_n = Z_n / Z_1^n (n ≤ 4) and dR_n/dθ,
  sampled exactly from the one-guest Boltzmann density in each cage. The interaction between
  neighbouring cages is not in this model; that is why case B switches it off. With it on
  (mode 1), the occupancy rises by about 2.4% in the large cage, a measured, unpredicted term.

# H2 content as Knowledge: `examples/molecular/h2_sii_epistemic.sio`

Full guest-guest interaction, 200 K, 300 MPa, three water models. Each occupancy is a
Knowledge value with two kinds of GUM channel: the H2 Lennard-Jones parameters (two latent
channels shared by all three models, covariance of four published one-site sets) entering
through the fluctuation sensitivities, and the Monte Carlo error of each run. The predictions
in the header come from the C++ GCMC of the same model.

## Measured, 4·10⁶ trials per model (2026-11-16)

| | N per cell | u(N), H2 and MC |
|---|---|---|
| SPC/E | 31.919 ± 0.065 (predicted 31.82 ± 0.04) | 0.703 |
| TIP4P/Ice | 33.016 ± 0.052 (33.08 ± 0.06) | 0.725 |
| TIP5P | 32.387 ± 0.050 (32.35 ± 0.06) | 0.739 |

| Difference | Value ± u | H2 part, shared channels | H2 part if independent |
|---|---|---|---|
| TIP4P/Ice − SPC/E | 1.096 ± 0.087 | 0.023 | 1.006 |
| TIP5P − SPC/E | 0.468 ± 0.090 | 0.038 | 1.017 |

Each water model alone carries about 2.2% uncertainty from the H2 parameters, the size of
the differences between models. Taken as independent, the models would look alike; with the
H2 channels shared, the H2 part of a difference falls about thirtyfold, and both differences
are resolved (12.7 and 5.2 standard uncertainties).

A first run with 10⁶ trials per model failed the pass rule: TIP5P − SPC/E was 0.50 ± 0.20,
2.5 standard uncertainties, below the threshold of 3. The H2 part had cancelled as predicted;
the Monte Carlo error of the shorter runs was too large to resolve the smaller difference.
The predictions came from 4·10⁶-trial runs, and the driver now uses the same length; the pass
rule is unchanged.

# Three water models as a contest: `examples/molecular/h2_sii_water_contest.sio`

The same C++ GCMC values (200 K, 300 MPa), with the H2 parameters as two channels shared
by all three water models and each run's Monte Carlo error as its own, go through the
ADR-016 builtins (`meet`, `contest`, `retract`). The GUM covariance comes from the channels;
the oracle is `tests/oracles/epistemic_meet/h2_water_models.cpp`.

| | Correlated (shared H2 channels) | Treated as independent |
|---|---|---|
| E_n TIP4P/Ice − SPC/E (k = 2) | 9.02 | 0.64 |
| E_n TIP5P − SPC/E | 3.61 | 0.26 |
| E_n TIP4P/Ice − TIP5P | 4.36 | 0.37 |
| GLS weights SPC/E / TIP4P/Ice / TIP5P | 2.57 / −0.52 / −1.05 | 0.34 / 0.33 / 0.33 |
| Fusion | 30.6 (outside all three) | 32.41 ± 0.41 |
| Birge ratio | 12.8 | 0.90 |

The correlation between models is 0.993 to 0.995, so the three count as n_eff = 1.004
independent sources. Treated as independent, they would look consistent and fuse to a
value narrower than any one of them. With the covariance, `meet` refuses (negative weight,
under every policy), and the models are carried as `contest(spce, tip4, tip5)`. Retracting
TIP4P/Ice, the force field fitted to the ice phases, leaves SPC/E and TIP5P, which still
disagree (E_n = 3.6) and still cannot be fused.

# A charged H2 in a proton-disordered host: `tests/stdlib/molecular/test_sii_charges_ewald.sio`

The runs above use a one-site H2, so the water hydrogens never enter. With the Alavi
three-site H2 (+0.4932 e on the atoms, −0.9864 e at the centre: quadrupole
2.17·10⁻⁴⁰ C m², the experimental value) they do, and their positions are not in
the oxygen lattice. `molecular::ice_rules` builds them: an Euler circuit on the
4-regular bond graph satisfies the ice rules, and loop moves disorder the protons and
bring the bond-winding vector to (0, 0, 0) exactly. On one sII cell (seed 20261116):
272 bonds, 0 ice-rule violations, residual SPC/E dipole 0.32 e Å per cell against
66.5 e Å if every water were aligned. `molecular::ewald` gives the tinfoil Ewald
potential of the 408 host charges. Everything below was predicted in
`stdlib/molecular/oracles/sii_ewald_predictions.txt` before the oracle ran, and matches
it in Sounio to 1e-9 or better:

| Check | Result |
|---|---|
| H2 energy independent of the Ewald splitting (α = 0.35, 0.40, 0.45 Å⁻¹) | 4e-12 relative |
| Ewald equals the cubic direct sum of host images (N = 24 shells) | 6e-5 (small cage) and 2e-4 (large) relative, falling as 1/N² |
| Point-quadrupole limit ½ Q_ab ∂_a∂_b φ | within 3 % of the orientational spread |
| Orientation average at a cage centre (Laplace: ∇²φ = 0) | below 5e-5 of the spread |
| Orientational spread (max − min energy) | 0.42–1.33 kJ/mol in small cages, 0.14–0.43 in large ones |
| Orientational Boltzmann factor at 250 K, cage centre | 1.0016–1.0141 (small), 1.0002–1.0015 (large) |

Two consequences. Because the orientation average of the quadrupole energy at a
charge-free point is zero (Laplace) and exp is convex (Jensen), classical charges on H2
can only raise the Langmuir constant over the uncharged model with the same
Lennard-Jones centre, and at the cage centres they raise it by at most 1.4 % — the
electrostatics do not explain a difference between groups that use the one-site and the
three-site H2. The second: the spread varies threefold or more from cage to cage of the same
type, because each cage sees a different proton arrangement, so "the" small-cage
orientational potential of a clathrate is a distribution over proton configurations, not
a number.

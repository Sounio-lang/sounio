# stdlib/particle_physics

Quantum Field Theory and Standard Model foundations for Sounio.

## Status

- **Phase 1 (Foundations)** — COMPLETE. Lorentz vectors, Dirac algebra,
  gauge-group representations, SM parameters with epistemic uncertainty,
  phase-space kinematics.
- **Phase 2 (Propagators, Vertices & Amplitudes)** — COMPLETE. Tree-level
  propagators, QED/QCD/weak/Yukawa vertices, and amplitude builders for
  e⁺e⁻ → μ⁺μ⁻, Møller/Bhabha scattering, Z/Higgs decay widths.
- **Phase 3 (Lattice gauge / mass-gap probe)** — SU(2) Wilson lattice with
  runtime **L∈{4,6,8}** (`lg_set_L`, flat cap 65536); SU(3) **L∈{4,6}**
  (`su3_set_L`, flat cap 93312): Metropolis/heatbath, plaquettes, Wilson loops,
  Creutz ratios, APE smear, **0++ glueball** C(τ) + spectral m_eff.
  Confinement + spectral-gap *computational* attack — **not** a Clay
  Millennium proof. Drivers:
  `su2_mass_gap_probe.sio`, `su2_glueball_mass_gap.sio`,
  `su2_glueball_smeared.sio`, `su2_glueball_L6.sio`,
  `su2_continuum_sketch.sio` (multi-β m_eff/√σ trend),
  `su3_mass_gap_probe.sio` (pure SU(3) Wilson + Creutz),
  `su3_glueball_mass_gap.sio` (SU(3) 0++ thin/APE spectral m_eff),
  `su3_continuum_sketch.sio` (SU(3) multi-β m_eff/√σ trend),
  `heatbath_vs_metropolis.sio` (Creutz SU(2) + Cabibbo–Marinari SU(3) heatbath),
  `mass_gap_heatbath_production.sio` (HB production: SU(2) L=6 m/√σ≈3.81, SU(3)≈4.45),
  `su2_continuum_heatbath.sio` (HB multi-β: ⟨m/√σ⟩≈4.13, 4/4 β ok, lit-mean PASS),
  `su2_precision_witness.sio` (f64 vs Dd64 dual-path Creutz/plaquette on L=6 HB),
  `su2_meff_plateau.sio` (multi-τ m_eff(τ) plateau diagnostic, spectral honesty),
  `su2_variational_glueball.sio` (2-smear GEVP ground-state projection),
  `su2_variational_stability.sio` (3-smear multi-pair GEVP stability),
  `su2_gevp3.sio` (full 3×3 GEVP principal mass — no pair shopping),
  `su2_scale_gevp.sio` (multi-R Creutz scale × 3×3 GEVP consistency),
  `su2_sommer_r0.sio` (static V(r)/F(r)/Sommer: path live; force-r0 FAIL_HONEST on L=6,
  r0_σ=√(1.65/χ) proxy ⇒ m·r0_σ≈3.17),
  `su2_sommer_production.sio` (**BOLD**: L=8 + multi-β; ⟨m·r0_σ⟩≈4.27, L=8 m/√χ₂≈4.15;
  spread/FS FAIL_HONEST),
  `su2_fs_gevp.sio` (cascade GEVP3→2→contact: 4/4 variational; same-op FS contact
  L6→L8 shift 12.8% PASS; multi-β spread FAIL_HONEST),
  `su2_contact_continuum.sio` (same-op contact multi-β: ⟨m/√χ⟩≈3.80 lit; √σ↓ 4/4;
  spread FAIL_HONEST),
  `su3_fs_probe.sio` (SU(3) L=6 live; thin FS L4→L6 shift **2.4%** PASS),
  `su3_l6_production.sio` (APE n∈{1,2,3} all live; m/√χ₃≈**3.46** near lit;
  multi-R Creutz),
  `su2_jackknife_continuum.sio` (jk+GUM multi-β; ⟨m/√σ⟩_w≈3.49; χ²/spread FAIL_HONEST),
  `su3_jackknife_continuum.sio` (SU(3) L=6 multi-β; honest SEM floors),
  `dual_ym_scoreboard.sio` (dual multi-β engineering; ratios are **not** gap claims),
  `mass_claim_plateau.sio` (**HARD GATE**: single-op multi-τ ⇒ MASS_GAP KILLED),
  `mass_claim_gevp_plateau.sio` (multi-τ GEVP2 earns PLATEAU_MASS on L6/L8;
  legal m/√σ≈2.49 / 1.91 — below lit; fragile two-sink).

## Modules

| File | Contents |
|------|----------|
| `lorentz.sio` | 4-vectors, boosts, Mandelstam variables, rapidity |
| `spinor.sio` | Dirac matrices (chiral basis), γ^μ, γ^5, slash(p), trace identities |
| `gauge.sio` | SU(2), SU(3), U(1) generators, structure constants, Casimirs |
| `sm_params.sio` | PDG 2024 constants with `Epistemic` uncertainty propagation |
| `kinematics.sio` | Flux factors, CM kinematics, decay widths, cross-sections |
| `propagator.sio` | Scalar, fermion, photon, gluon, W/Z/Higgs propagators |
| `vertex.sio` | QED, QCD, weak charged/neutral current, Yukawa vertices |
| `amplitude.sio` | e⁺e⁻ → μ⁺μ⁻, Møller, Bhabha, Z/Higgs decay widths |
| `lattice_gauge.sio` | Pure SU(2) Wilson L∈{4,6,8}: Metropolis **+ Creutz heatbath**, Creutz χ R≤4, 0++ C(τ), APE, GEVP, Sommer r0_σ, Epistemic m_eff + σa² |
| `lattice_su3.sio` | Pure SU(3) Wilson **L∈{4,6}**: Metropolis **+ Cabibbo–Marinari heatbath**, Creutz χ, APE, 0++ C(τ), Epistemic |
| `lattice_prec.sio` | Dual-path f64 vs **Dd64** plaquette/Wilson/Creutz precision witness |

## Tests

- `tests/stdlib/particle_physics/test_particle_physics_core.sio` — BSM, statistics, systematics (check-only)
- `stdlib/particle_physics/lib.sio` — full inline regression driver (heavy; IO effects)

## Quick Example

```sio
use particle_physics::lorentz::*;
use particle_physics::spinor::*;
use particle_physics::sm_params::*;

fn demo() -> i32 with Mut, Div, Panic {
    // Lorentz vector for an electron with E = 10 GeV, pz = 8 GeV
    let p = lorentz_new(10.0, 0.0, 0.0, 8.0)
    let m = lorentz_mass(p)

    // Dirac slash matrix p̸
    let ps = slash(p)

    // Z boson mass with PDG uncertainty
    let mz = mass_z()
    // mz.val() = 91.1876, mz.std() ≈ 0.0021

    return 0
}
```

## Epistemic Discipline

Every SM parameter returns `Epistemic` (from `epistemic::knowledge`).
Derived couplings (e, g, g′, g_s) propagate PDG uncertainties via the
GUM delta method.  Confidence decays through arithmetic chains.

## References

- Peskin & Schroeder, *An Introduction to Quantum Field Theory* (1995)
- PDG 2024: https://pdg.lbl.gov/2024/
- CODATA 2022 / SI 2019 exact constants

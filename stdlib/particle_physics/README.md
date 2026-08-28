# stdlib/particle_physics

Quantum Field Theory and Standard Model foundations for Sounio.

## Status

- **Phase 1 (Foundations)** — COMPLETE. Lorentz vectors, Dirac algebra,
  gauge-group representations, SM parameters with epistemic uncertainty,
  phase-space kinematics.
- **Phase 2 (Propagators, Vertices & Amplitudes)** — COMPLETE. Tree-level
  propagators, QED/QCD/weak/Yukawa vertices, and amplitude builders for
  e⁺e⁻ → μ⁺μ⁻, Møller/Bhabha scattering, Z/Higgs decay widths.

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

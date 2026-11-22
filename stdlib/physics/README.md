# stdlib/physics

## Checked correlated-volume entropy

isothermal_entropy_correlated_volumes propagates first-order standard
uncertainty with a supplied rho between the two volumes and independent
amount of substance. Correct opposite volume sensitivities permit common-
calibration cancellation; cancellation precedes norm scaling. rho0 delegates
exactly to the preserved independent API. Inputs/units/rho and arithmetic
availability are explicit, with no inferred correlation or coverage level.
Final main-based Forge15457:95/95;32 independent C++23 quad cases and actual
Canvas/SVG consumer with analytic geometry verification.
See [contract](ENTROPY.md) and
[receipt](../../docs/plans/COUNTS_COVARIANCE_EXECUTION_2026_10_19.md).


Epistemic-first physics umbrella: classical mechanics, electromagnetism,
thermodynamics, and special relativity on top of `units::Quantity` with
GUM-style uncertainty propagation.

## Layout

| File | Domain |
|---|---|
| `mod.sio` | Public exports (`physics::classical`, `em`, `thermo`, `sr`) |
| `lib.sio` | Umbrella runner + quantity bridge helpers |
| `classical.sio` | Newtonian mechanics, kinematics, epistemic force |
| `em.sio` | Coulomb, Lorentz, Poynting, wave propagation |
| `thermo.sio` | Ideal gas, Carnot, entropy |
| `sr.sio` | Relativistic energy, four-momentum, Compton |
| `phonon.sio` | Lattice / phonon quantities (legacy) |
| `pbpk_phonon.sio` | PBPK–phonon bridge |

`particle_physics/` is a sibling high-energy module; a `physics::particle` reexport
is planned.

## Verification

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc check stdlib/physics/lib.sio
bash scripts/run_sio_test_suite.sh physics --verbose
```

Harness: `tests/stdlib/physics/`.
## Compton scattering

The canonical scalar kernel is `physics::scattering::electron_compton_shift`.
It computes the wavelength shift in metres as
`2 * (h/(m_e*c)) * sin(theta/2)^2`, with theta in radians in [0,pi].
The half-angle form avoids subtracting nearly equal cosines at small angles.
Physical constants come from `constants::physical`.

The result records shift, status and the **first-order angular standard-uncertainty
contribution** `(h/(m_e*c))*abs(sin(theta))*u_theta`. Electron-mass uncertainty
and higher-order nonlinear propagation are not included. At stationary angles,
the first-order contribution can vanish while higher-order terms remain.

The legacy `physics::sr::compton_shift` delegates to the same kernel and
preserves its tuple interface. The free-function epistemic API and canonical
`units::lib` import are used by that module.

Kernel and legacy-adapter consumers have native fixtures. This evidence does not
certify every relativity or uncertainty routine in the module. Details:
[Multi-domain receipt](../../docs/plans/STDLIB_MULTIDOMAIN_EXECUTION_2026_10_17.md).

## Checked isothermal entropy

The scalar `physics::entropy::isothermal_entropy_independent` kernel returns
explicit statuses and independent first-order standard uncertainty in J/K.
Stable logarithms/products and scaled norm cover the documented finite domain.
The legacy thermo tuple delegates to it, with (0,-1) for unavailable results.
Typed SVG bars carry standard-uncertainty semantics without coverage probability.
Native closure:69/69, job15308; two controlled regressions retain all prior64.
Other thermodynamic uncertainty formulas retain their legacy limitations.
[Contract](ENTROPY.md);
[receipt](../../docs/plans/PHYSICS_ENTROPY_EXECUTION_2026_10_18.md).

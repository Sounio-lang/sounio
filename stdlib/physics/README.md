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
| `mechanics.sio` | Checked SI Quantity energy, spring force, work, power and gravity |
| `em.sio` | Coulomb, Lorentz, Poynting, wave propagation |
| `thermo.sio` | Ideal gas, Carnot, entropy |
| `sr.sio` | Relativistic energy, four-momentum, Compton |
| `phonon.sio` | Protected SI joint-factor lattice transport and explicit legacy import |
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

## Checked Quantity mechanics

`physics::mechanics` supplies `kinetic_energy_q`, `hookean_force_q`,
`work_done_q`, `power_q` and `gravitational_force_q`, with corresponding
`*_checked` functions. Both physics facades export this family. Inputs are SI
`units::lib::Quantity` values built with `q_mass`, `q_velocity`, `q_force`,
`q_length`, `q_energy` and `q_time`. Constructors carry the supplied standard
uncertainty and dimension without implicit magnitude conversion.

Checked results contain status, value, standard_uncertainty, uncertainty_model
and first_order. Status1 means nonfinite input,2 physical domain,3 negative
uncertainty,4 conservative arithmetic unavailability,5 unit mismatch.
Convenience Quantity wrappers assert successful status. Spring force is the
collinear applied force k*x; work preserves the signs of force/displacement,
and power uses positive elapsed time. Masses are nonnegative and gravity uses
a positive separation.

Each function declares independent first-order GUM inputs. Kinetic energy
reuses the qualified classical checked kernel, whose velocity sensitivity is
m*v; a repeated velocity is differentiated once. A first-order result does not
certify the approximation's adequacy or a coverage probability. Required
uncertainty contributions that cannot be represented return status4.

Default gravity includes the [CODATA2022 G value and standard uncertainty](https://physics.nist.gov/cuu/pdf/wall_2022.pdf):
6.67430e-11 and1.5e-15 in SI. `gravitational_force_with_g_checked` accepts an
explicit Quantity constant and combines its declared independent contribution
with those of the two masses and separation.

The reusable source/compiler/ELF guarded gate is
`scripts/stdlib/wave3_mechanics_gate.sh FRESH_ABSOLUTE_OUTPUT`. It runs the
unchanged original mechanics/classical fixtures, independent analytic C++23
value/uncertainty checks and a deliberately wrong kinetic-energy uncertainty
that the oracle must reject. Native evidence is kept separate from the
remaining typed phonon migration onto protected joint-factor transport.

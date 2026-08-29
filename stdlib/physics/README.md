# stdlib/physics

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
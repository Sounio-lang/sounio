# demos/quantum

## `h2_vqe_2q.sio`: two-qubit H2 VQE against exact diagonalisation

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
bin/souc run demos/quantum/h2_vqe_2q.sio            # Madaros (default), about 7 s including compile
SOUNIO_SOUC_ENGINE=lean_single bin/souc run demos/quantum/h2_vqe_2q.sio   # about 8 s, same output
```

The demo uses the library `stdlib/quantum/epistemic_vqe.sio` through a normal
`use quantum::epistemic_vqe::{…}` import.

| Item | What it is |
|---|---|
| Hamiltonian | `H = c0 I + c1 Z0 + c2 Z1 + c3 Z0Z1 + c4 (X0X1 + Y0Y1)`, coefficients (−0.4804, 0.3435, −0.4347, 0.5716, 0.0910) from O'Malley et al., *Phys. Rev. X* **6**, 031007 (2016) |
| Exact energy | −1.851199124123644 Ha, the lowest eigenvalue of the 2×2 block on {\|01⟩, \|10⟩}. `c0` does not include nuclear repulsion, so this is an electronic energy, not the total energy of −1.137 Ha that is often quoted |
| Ansatz | Ry(θ0) q0, Ry(θ1) q1, CNOT(q0→q1), Ry(θ2) q0, Ry(θ3) q1. This circuit can represent the exact ground state |
| Optimiser | Rotosolve (exact minimisation along one coordinate of a sinusoid): 9 sweeps, \|E_vqe − E_exact\| ≈ 4e-16 Ha |
| Uncertainty | Gate angles θ ~ N(θ\*, u²I). At a minimum the gradient vanishes, so the first-order GUM term is ≈ 0 and the band is second order: u²(E) = gᵀΣg + ½ tr((HΣ)²), with g and H computed by the parameter-shift rule |
| Validation | A seeded Monte Carlo with N = 20000. The ratio u_GUM2/σ_MC must lie in [0.8, 1.25] at u = 0.01, 0.05 and 0.1 rad. Measured: 1.008, 1.018, 1.020 |

The program prints `H2_VQE_2Q_OK` only if every check passes. Otherwise it
prints `H2_VQE_2Q_FAILED` and returns 1. `scripts/talk.sh` step 10 relies on
this sentinel.

### Independent cross-check (C++23)

```bash
g++ -std=c++23 -O2 -o /tmp/h2x demos/quantum/tools/h2_vqe_crosscheck.cpp
/tmp/h2x <theta*_0..3> <E_vqe> <MC sigma at 0.01> <MC sigma at 0.05> <MC sigma at 0.1>
```

All of these numbers come from the demo's output. The C++ program was written
from the physics rather than translated from the Sounio code:

- the Hamiltonian is a dense Kronecker-product matrix, diagonalised with Jacobi;
- the ansatz is built as a product of unitaries;
- the Monte Carlo uses `std::mt19937_64`;
- the gradient and Hessian come from finite differences.

Results measured on 2026-10-06:

| Check | Result |
|---|---|
| E_exact | agrees with the closed form to < 1e-15 |
| E(θ\*) | agrees with Sounio's E_vqe to 3.6e-13 (limited by the 12 digits printed) |
| MC σ | Sounio's N = 2e4 values are within 0.7–1.7 % of a C++ N = 1e6 run, which is the size of the N = 2e4 sampling error |
| GUM-2 | the C++ finite-difference value matches Sounio's parameter-shift value to 7 digits |

The program ends with `H2_VQE_CROSSCHECK_OK`.

## Limits

- **Linear `Qubit` simulation: at most 10 qubits.** `stdlib/quantum/linear.sio`
  stores 1024 amplitudes; the 11th `qubit_alloc` panics.
- **VQE modules: at most 4 qubits.** `epistemic_vqe` stores 16 amplitudes, and
  so does `stdlib/quantum/vqe.sio`, the general version with Pauli sums and
  gate lists. Neither uses tensor networks or sampling.
- **No noise model.** The only uncertainty is Gaussian error on the gate
  angles, and its propagation is validated by Monte Carlo. There is no
  decoherence, readout error or depolarisation.
- **No shot noise.** Expectation values are computed exactly.
- **No hardware backend.** Nothing here runs on a quantum device.
- **The QIR artefact is a text shim.** `artifacts/quantum/omega/qir_shim.json`
  holds a two-gate (H, CNOT) text shim with digests. Its intrinsic names and
  operands do not follow the QIR base profile (`__quantum__qis__*__body` on
  `%Qubit*`), so it is not a loadable QIR module, and nothing executes it.
  `quantum_conformance.json` next to it describes a 128-shot statistical check
  of that shim, not of this demo.
- **The no-cloning demos are a type-system analogue.** On branch
  `feat/talk-script` these are `demos/quantum/no_cloning_single_use.sio`,
  `no_cloning_double_use.sio` and `no_cloning_dropped.sio`. They show that a
  `linear struct Qubit` value must be used exactly once: E039 if it is used
  twice, E040 if it is dropped. This mirrors no-cloning and no-deleting in the
  type system. It is not a statement about quantum states, and the simulator
  state behind `stdlib/quantum/linear.sio` is an ordinary array.

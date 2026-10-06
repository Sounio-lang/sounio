# stdlib/quantum

Exact quantum simulation on a classical statevector. No noise model, no shot
noise and no hardware backend.

- `linear.sio`: a linear `Qubit` type. A qubit value must be used exactly once,
  which is the type-system analogue of no-cloning and no-deleting. Cap
  `QUANTUM_MAX_QUBITS = 10` (1024 amplitudes); the 11th `qubit_alloc` panics.
  Gates: H, X, Y, Z, S, T, CNOT, CZ, SWAP. Circuits: Bell, GHZ3, teleportation.
  Measurement is `measure_qubit` (not `measure`: that name is the builtin
  Knowledge constructor).
- `vqe.sio`: Pauli-sum Hamiltonians, gate-list circuits, exact expectation
  values, an exact ground-state oracle (Jacobi) and a Rotosolve VQE driver.
  Cap: 4 qubits.
- `epistemic_vqe.sio`: the two-qubit H2 VQE (O'Malley et al., PRX 6, 031007
  (2016)) with a second-order GUM band on the gate angles, validated against
  Monte Carlo. Cap: 4 qubits.

Rx/Ry/Rz live on `Statevector` in `epistemic_vqe.sio`; they are not on linear
`Qubit`.

Demo, independent C++ cross-check and the full list of limits:
[`demos/quantum/`](../../demos/quantum/README.md).

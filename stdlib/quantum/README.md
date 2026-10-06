# stdlib/quantum

Small, exact quantum simulation for at most 4 qubits. The statevector is
simulated classically, with no noise model, no shot noise and no hardware
backend.

- `linear.sio`: a linear `Qubit` type. A qubit value must be used exactly once,
  which is the type-system analogue of no-cloning and no-deleting. The module
  provides gates, `measure_qubit` and Bell pairs.
- `vqe.sio`: Pauli-sum Hamiltonians, gate-list circuits, exact expectation
  values, an exact ground-state oracle (Jacobi) and a Rotosolve VQE driver.
- `epistemic_vqe.sio`: the two-qubit H2 VQE (O'Malley et al., PRX 6, 031007
  (2016)) with a second-order GUM band on the gate angles, validated against
  Monte Carlo.

Demo, independent C++ cross-check and the full list of limits:
[`demos/quantum/`](../../demos/quantum/README.md).

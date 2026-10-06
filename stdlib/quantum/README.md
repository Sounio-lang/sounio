# stdlib/quantum

Exact quantum simulation on a classical statevector. No noise model, no shot
noise and no hardware backend.

- `linear.sio`: a linear `Qubit` type. A qubit value must be used exactly once,
  which is the type-system analogue of no-cloning and no-deleting. Cap
  `QUANTUM_MAX_QUBITS = 10` (1024 amplitudes); the 11th `qubit_alloc` panics.
  Gates: H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP, Toffoli. Circuits: Bell,
  GHZ3, teleportation.
  Measurement is `measure_qubit` (not `measure`: that name is the builtin
  Knowledge constructor).
- `vqe.sio`: Pauli-sum Hamiltonians, gate-list circuits, exact expectation
  values, an exact ground-state oracle (Jacobi) and a Rotosolve VQE driver.
  Cap: 4 qubits.
- `epistemic_vqe.sio`: the two-qubit H2 VQE (O'Malley et al., PRX 6, 031007
  (2016)) with a second-order GUM band on the gate angles, validated against
  Monte Carlo. Cap: 4 qubits.

Rx/Ry/Rz also exist on `Statevector` in `epistemic_vqe.sio`. The two simulators
are not unified.

`epistemic_linear.sio` runs parameter-shift GUM on linear `ry`. The angle
is an `Epistemic`: its variance is a struct field and survives a call.
Builtin `Knowledge<f64>` drops its uncertainty when passed as an argument;
deposit the band into `Epistemic` in the function that built it.
At θ = π the first-order band is zero. Teleportation leaves both the
probability and its derivative unchanged.

`born_channel.sio` keeps those two zeros apart. A filled channel has a numeric
read. An empty channel does not: `band_variance` panics, and the unread payload
is a poison, not a variance. `born_prepared` fills the angle channel and leaves
the shot channel empty. `born_one_shot` does the opposite, after one
`measure_qubit`. One `Epistemic` on two wires fills the covariance with
∂a ∂b σ². Two `Epistemic` values with the same numbers leave that covariance
empty, and `pair_var_sum` does not treat the hole as zero.

Demo, independent C++ cross-check and the full list of limits:
[`demos/quantum/`](../../demos/quantum/README.md).

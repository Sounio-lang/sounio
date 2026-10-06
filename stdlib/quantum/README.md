# stdlib/quantum

Quantum computing with linear types (no-cloning enforced). Classical statevector simulation; not a quantum computer.

## linear

- Simulator cap: `QUANTUM_MAX_QUBITS = 10` (1024 amplitudes). The 11th `qubit_alloc` panics.
- Gates: H, X, Y, Z, S, T, CNOT, CZ, SWAP
- Circuits: Bell pair, GHZ3, teleportation (`TeleportResult`)
- Measurement via `qubit_measure` (consumes the qubit). The name `measure` is a compiler builtin.

Rx/Ry/Rz live on `Statevector` in `epistemic_vqe.sio`; they are not on linear `Qubit`.

## VQE

VQE with epistemic gate-angle uncertainty.

# Pireus Operator Seed Kernel

Concept-ID: `SOUNIO-PIREUS-OPERATOR-SEED-KERNEL`

Status: `garden`

Canonical source planned:

`stdlib/hardware/pireus/operator_seed_kernel.sio`

Semantic boundary:

- consumes only a frozen Sounio v7 `OperatorSeed`;
- defines a 16-lane bilinear twisted-XOR operator from its residual table;
- compiles exactly 256 destination-major reduction terms;
- produces exact Sounio basis and dense-probe results before parity opens;
- emits a generated operator candidate that can become an explicit future
  parent;
- does not assert broad, historical, material, performance, or scientific
  novelty.

The registry entry is owned by the coordination lane and remains pending. This
document must not be committed before `docs/internal/concepts/registry.tsv`
contains the Concept-ID.

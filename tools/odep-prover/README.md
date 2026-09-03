# ODEP Prover — Phase 1 Stub

Oblivious Differential Epistemic Privacy (ODEP) witness generator for
Sounio ZD-based unlearning operations.

See `spec.md` for the R1CS circuit specification and the Lean soundness
reduction.

## Build

```bash
cd tools/odep-prover
cargo build --release
```

Produces:
- `target/release/odep-prove`    — CLI binary
- `target/release/libodep_prover.{so,dylib}` — FFI library for Sounio

## CLI usage

```bash
echo '{"w_pre": [...], "w_post": [...], "c1": 0.0, "c2": 0.0, "prim_a_i": 3, "prim_a_j": 10}' \
  | ODEP_REGULATION=GDPR-Art17 ./target/release/odep-prove
```

Exits 0 on acceptance (and prints the JSON envelope), non-zero on
constraint violation.

## Test

```bash
cargo test
```

Two tests ship:
- `roundtrip_accepts` — a valid unlearn witness is accepted.
- `tampered_post_rejected` — a malformed `w_post` is rejected with
  `OdepError::PostWeightMismatch`.

## Roadmap

- **Phase 1 (this commit):** numerical R1CS check + JSON envelope +
  FFI surface.
- **Phase 2:** Halo2 circuit compilation + real ZK proof generation.
- **Phase 3:** Nova recursion for batched unlearn ops.

## Lean soundness

The R1CS constraint system verified by this prover is equivalent to the
`unlearning_kernel_exact` theorem in
`formal/lean4/SounioSurgicalInterventions.lean`.  The soundness of
the reduction — ODEP circuit satisfiable ⇒ Lean theorem holds — is
stated in `formal/lean4/SounioRegulatory.lean::odep_soundness`.

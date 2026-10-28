# Linear pair and quantum concordance fixtures

let_layers.sio and let_take_field.sio are the qualified single-consumption forms. Both Token and Pair are explicitly linear. repeated_projection.sio is a semantic negative requiring E039 before native emission. The role oracle is tests/reference/wave4_quantum_pair_reference.cpp and verifies first=17, second=25, sum=42 independently.

match.sio and let.sio retain the original compiler-crash reproductions. The latter deliberately retains its take(Token) helper written with match, so its failure must not be attributed to the outer let pattern. They are diagnostic fixtures, not advertised run-pass programs.

observables.sio invokes the real public quantum/tape APIs and emits the exact f64 bits for 39 named observations. observables_reference.cpp is an independent C++23 long-double complex statevector oracle; --self-test exercises adversarial controls, otherwise pass a native stdout file. It rejects missing/extra/duplicate labels and nonfinite outputs and uses the fixed absolute tolerance 1e-10.

The exact Slurm recipes, immutable compiler hashes, source ledger, before/after results and remaining slow/match limitations are documented under docs/evidence/stdlib-wave4-quantum-pair. All execution in this qualification is Slurm; no Python, Rust or Julia is used.

# Preserved-test syntax repairs

Slurm 18119 used source-built a955 compiler
7df87ae9d9b4b3c0cefcd18b892512b6bf7e5c6fde6690ca5223669a9a1a2639.
Both original tests failed before the library-only repairs and passed check,
compile, and native execution afterward. Test sources and assertions remained
byte-identical.

- nn/epistemic_backward.sio had a math import inside a grouped nn import.
  Moving the math import before the group restores the intended module syntax.
- linalg/epistemic_matrix.sio used Rust-style implicit &self receivers.
  Explicit self: &EpistemicMatrix receivers follow the documented Sounio syntax.

Native markers were PASS: epistemic backward pass e2e and EPISTEMIC_MATRIX_OK.
Algorithms, iteration counts, tolerances, and expected outputs were unchanged.
The source, overlay, test identities, compiler guards, native ELFs, and logs are
retained in the wave3 syntax-repair packet.

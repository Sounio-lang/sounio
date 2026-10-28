# Activate144 documented run-proof programs

Each selected file already described itself as a Run-proof or Deepened run-proof program in its first-line comment and contained a main entrypoint. It lacked the runner annotation and was skipped in19073. The only change to each file is a leading //@ run-pass line. SELECTION.json binds the old body and annotated file hashes; all144 bodies are byte-identical to their previous committed sources.

Slurm19167 runs the canonical E2E runner separately for these exact144 paths, in four isolated source copies of the earlier full-suite archive, using raw bd4ba2409a0097fdc9e9b4fe32e5858e396d4d71dd3515648a6f01d451b8ace1 and the preserved runtime. The library snapshot has no new unqualified stdlib source changes. Every selected path produces one row, and all original source/copy identities are checked after execution.

Result119pass/25fail/0skip; zero selection or identity errors. The job correctly exits1. RESULT.json preserves all25 failures. They are newly exposed failures, not approved exclusions. The remaining77 unannotated files include negative controls, argument-driven fixtures and additional standalone tests that still need explicit classification.

This is an activation receipt for previously unexecuted tests, not a whole-corpus rerun or zero-failure claim.

A Lean 4 model of the x86-64 the bootstrap seed loaders run on
(`formal/Bootstrap/X86Min.lean`), with kernel-checked runs of both seeds:
short vectors in CI, and the full self-reproduction and cross-check theorems
through `bootstrap/model/seed_proofs.sh`. `bootstrap/model/x86diff_gate.sh`
tests the model against the CPU instruction by instruction.

# Draft inputs, no Sounio execution yet

manifest.json pins17 new input/oracle files and6 immutable canonical originals.
harness.mjs performs orchestration only; C++23 owns numerical expected results.

--inventory performs hashing only.
--oracle-controls is Slurm-only and invokes no Sounio compiler.
--language-components is Slurm-only and requires a root authorization envelope
bound to the exact reviewed patch, source lower file, compiler and RAW_READY.

The eight profile-only controls are not language-rejection tests. The six semantic
refusals are proposed expectations requiring check0 followed by exact lower
diagnostic, ordinary nonzero compile and no ELF. No parse failure can qualify them.
No full acceptance marker is available: CFG/global coverage and actual Lowerer
private-state controls remain separate required gates.

Wire the aarch64-linux preview backend into `souc compile` (SIO-32).

`souc compile --target aarch64-linux` typechecked and lowered fine, then died
at the native binary write with bare `rc=11`: the file-writing entry
(`compile_native_v2_preview_to_file`) only knew x86_64-linux and arm64-macos.
The aarch64-linux preview backend — machine module, tabulated AArch64
encoder, Linux entry trampoline (`bl main`, `mov x8, #93`, `svc`), ELF
finalizer with `EM_AARCH64` — already existed in `native::codegen` but was
unreachable from the `souc compile` path, which imports the stripped
`native::codegen_x86_linux` fork.

`self-hosted/native/codegen_x86_linux.sio` now carries a local twin of the
preview function (same helpers, same encoder, same finalizer; the result
narrowing honors this file's 128 KiB contract with a loud
`native_v2_aarch64_elf_too_large` refusal past it instead of a silent
truncate) plus a file-writing shell that verifies the byte count. Remaining
unsupported targets fail with a named sentence
(`target <resolved> is not supported by this backend yet`) instead of a bare
number, per the issue's requested scope.

Pinned by `scripts/ci/madaros_aarch64_linux_elf_gate.sh` (hello-world emits
an ARM aarch64 ELF; riscv64-linux refuses by name), wired into the
current-source CI job. Execution on target hardware stays manual
(`scripts/dev/a64_run.sh` on the arm64 Slurm nodes); no GHA runner can run an
aarch64-linux ELF.

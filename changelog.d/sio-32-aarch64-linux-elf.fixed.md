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

Pinned by `scripts/ci/madaros_aarch64_linux_elf_gate.sh` (hello-world plus a
wider int/compare/float/convert program emit ARM aarch64 ELFs;
riscv64-linux refuses by name), wired into the current-source CI job.
Execution on target hardware stays manual (`scripts/dev/a64_run.sh` on the
arm64 Slurm nodes); no GHA runner can run an aarch64-linux ELF.

Verified on real arm64 hardware (Nov 2026): empty main exits 0;
`print("hi")`, `print_int(42)`, int `==`/`<`, float `+`/`>`/negation,
`as f64`/`as i64` conversions, and multi-arg callees all execute correctly
(`hi / 42 / eq / lt / fgt / fneg / cv / 3`, exit 0). Three backend bugs
found and fixed along the way: the frame pointer was zeroed (`mov x29, xzr`
— ORR reads r31 as XZR; fixed to `add x29, sp, #0`), `print` builtins
emitted empty stubs (now real `write`-syscall leaves), and call args never
reached callees (the legalized stream carries CALL without ARG_MOVE, so the
preview now reads args from the IR arena like the x86 backend does).

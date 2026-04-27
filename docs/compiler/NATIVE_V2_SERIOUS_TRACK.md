# Native V2 Serious Track

Status date: 2026-04-27.

This note records the current compiler reality for the native-v2 path. It is
intentionally narrower than the long-term language vision: claims here must be
backed by executable repo gates.

## What Compiler We Have

The active host compiler is resolved through `scripts/lib/resolve_souc.sh`.
In this workspace the wrapper is `bin/souc`, and it selects
`bin/souc-linux-x86_64` on Linux x86-64. That selected binary is the runner for
Sounio implementation files.

The serious native-v2 Linux lane is:

- `self-hosted/compiler/native_compile_driver.sio`
- `self-hosted/native/codegen_x86_linux.sio`
- `self-hosted/native/elf_bulk.sio`
- `self-hosted/native/frame.sio`

The driver is a Sounio program. It lexes source through the repo lexer, reads
tokens through scalar parser rails, lowers the supported `main` subset into
core IR records, emits x86-64 code through native-v2 codegen, and writes an ELF
binary.

The current executable proof is:

```sh
bash scripts/ci/native_v2_serious_track_gate.sh
```

That gate checks the driver, compiles `examples/native/hello.sio` into a Linux
ELF, verifies `.rodata` and `.data` section witnesses when host tools are
available, runs the generated binary, and compares stdout to:

```text
Hello from self-hosted Sounio!
42
```

## What Is Self-Hosted

Self-hosted today means the native-v2 driver and backend repair live in Sounio
source and are executed by the current Sounio compiler. The generated artifact
is a standalone x86-64 Linux ELF.

This is not yet a fixed-point native compiler that compiles the full Sounio
compiler into itself. The fixed-point/bootstrap lane remains separate and must
not be blurred with the native-v2 scalar ELF proof.

## Linux

Linux x86-64 is the only native-v2 runtime lane promoted by this note.

Current proof:

- the driver checks cleanly through `souc check`
- the driver emits a standalone ELF for `examples/native/hello.sio`
- the ELF has a real `.rodata` segment for string literals
- `.data` runtime-context relocations target writable data, not `.rodata`
- the generated binary exits 0 and prints the expected output

The important repair was structural: large aggregate returns and nested
`StringTable`/`RelocationTable` mutation are not safe enough for this lane yet.
The Linux path now uses scalar token kind codes, flat relocation arrays, and
flat rodata bytes on `NativeCompiler`.

## macOS

macOS is not promoted to the same runtime status yet.

The repo has source surfaces for Apple work, including:

- `self-hosted/native/aarch64.sio`
- `self-hosted/native/macho.sio`
- `tests/native-v2/aarch64_macho_preview_emit.sio`
- `scripts/apple/apple_native_v2_ssh_gate.sh`

Those are preview and gate surfaces, not evidence that the shipped local
entrypoint is a working Apple-native self-hosted compiler. macOS should be
promoted only after the Apple gate produces and runs a Mach-O artifact on real
Apple hardware.

## Where We Are Going

The next serious track is:

1. Keep Linux native-v2 scalar ELF green under `scripts/ci/native_v2_serious_track_gate.sh`.
2. Replace remaining large aggregate by-value native-v2 boundaries with explicit
   ref/out-param or scalar rails.
3. Expand the Linux driver from the current core subset to more real language
   constructs with one executable witness per increment.
4. Bring Mach-O/AArch64 onto the same scalar discipline, then require the Apple
   SSH gate to run the produced artifact before making macOS support claims.
5. Only after that, connect this lane back to fixed-point self-hosting.

The rule is simple: no public support claim without a runnable artifact and a
repo gate that can reproduce it.

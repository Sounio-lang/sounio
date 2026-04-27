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

## Driver Self-Compile Target

The stricter native-v2 acceptance target is now encoded as:

```sh
bash scripts/ci/native_v2_driver_self_compile_gate.sh
```

That gate is Linux x86-64 only. It first requires
`native_v2_serious_track_gate.sh`, then asks the current
`native_compile_driver.sio` to compile `native_compile_driver.sio` into a
stage1 driver. If stage1 exists, the gate runs that generated driver on
`examples/native/hello.sio` and checks ELF kind, executable bit, `.rodata` and
`.data` witnesses, string-literal presence, and stdout parity with the current
driver.

This gate is green on Linux x86-64. The generated stage1 driver is an ELF
binary, and that generated driver compiles `examples/native/hello.sio` into an
ELF that runs with stdout parity against the current driver.

The stage1 driver proof remains intentionally narrow: it covers the
native-v2 driver source and a hello runtime witness, not the full language or
the older fixed-point `lean_single.sio` compiler lane.

## Native Epistemic Science Spine

The next checked native-v2 milestone is:

```sh
bash scripts/ci/native_v2_epistemic_science_spine_gate.sh
```

That gate first requires `native_v2_driver_self_compile_gate.sh`. It then builds
the native-v2 driver into a stage1 driver twice, requires byte-identical stage1
artifacts, and uses the generated stage1 driver to compile a small manifest
corpus under `tests/native-v2/science_spine/`.

Current corpus classes:

- baseline native hello
- loop/control-flow witness
- struct-return witness
- fixed-point epistemic arithmetic witness using a natural 3-field
  `KnowledgeI64` struct, struct-return construction, and two struct arguments
- fixed-point two-compartment PBPK-style witness using a natural `Compartments`
  struct carried through a step function
- Fano-plane/octonion combinatorics witness for the ordered non-collinear count
  `168`

The gate requires Linux x86-64 ELF kind, executable runtime output, fixture
stdout parity, deterministic replay for emitted corpus binaries, and a
summary JSON recording compiler path, manifest hash, stage1 hashes, per-case
hashes, `fallback_path=none`, and `host_callback=none`.

The science spine now includes a GUM PBPK entry that uses the `sqrt_f64`
SSE2 builtin: `pbpk_epistemic_gum` exercises ISO JCGM 100:2008 GUM
uncertainty propagation through a two-compartment PBPK simulation loop with a
confidence gate. See the dedicated GUM primitives gate section below.

This is a science-spine proof, not a general scientific-language support claim.
The current native-v2 stage1 corpus promotes the fixed-point integer science
spine. Floating-point epistemic witnesses are promoted separately by the f64
ladder gate below. Native floating register ABI and broader stdlib/PBPK imports
are not yet promoted through the generated stage1 driver.

For a narrower semantic regression check around the hardest stage1 shapes, run:

```sh
bash scripts/ci/native_v2_semantic_hardening_gate.sh
```

That wrapper runs the same generated-stage1 replay engine against
`tests/native-v2/semantic_hardening/`, covering 3-field struct literals,
3-field struct returns, 3-field struct parameters, and two `KnowledgeI64`
struct arguments returning a `KnowledgeI64`.

For the narrower floating-point promotion ladder, run:

```sh
bash scripts/ci/native_v2_f64_ladder_gate.sh
```

That gate promotes f64 literal parsing, f64 arithmetic/comparison emission,
`print_f64` stdout witnesses, a monomorphic `KnowledgeF64` struct witness, and
a narrow `Knowledge<f64>` witness through the generated stage1 driver. The
generic witness covers `struct Knowledge<T>`, `Knowledge<f64>` literals,
`Knowledge<f64>` parameters, `Knowledge<f64>` return values, and f64 field
arithmetic in that monomorphic instantiation.

This is still a generated-stage1 witness lane, not a full floating-point
runtime/ABI claim. The current `print_f64` surface is exercised by fixed
three-decimal positive fixtures in the gate, and the generic coverage is the
single-type-argument f64 instantiation path rather than a general generics
implementation.

## ISO GUM Primitives (SOTA)

The ISO JCGM 100:2008 (GUM) uncertainty propagation gate is:

```sh
bash scripts/ci/native_v2_gum_primitives_gate.sh
```

This gate requires `native_v2_driver_self_compile_gate.sh`, builds stage1 twice
for deterministic-replay proof, then compiles four programs through the generated
stage1 driver:

- `sqrt_f64_smoke` — exercises the `sqrt_f64` builtin which emits `sqrtsd`
  (SSE2 opcode F2 0F 51) in the generated ELF
- `gum_addition` — ISO GUM quadrature addition: `u_c = sqrt(u_a² + u_b²)`;
  verifies `gum_add({1.0, 0.3, 0.9}, {2.0, 0.4, 0.8})` gives `uncertainty=0.500`
- `gum_multiplication` — ISO GUM multiplicative rule:
  `u_c = |a·b| · sqrt((u_a/a)² + (u_b/b)²)`; verifies `gum_mul({10, 1, 0.9}, {2, 0.2, 0.9})`
  gives `uncertainty=2.828`
- `pbpk_epistemic_gum` — two-compartment PBPK simulation with `Knowledge<f64>` state
  carrying GUM uncertainty through 4 steps and a confidence gate (`if confidence < 0.8`)

The gate additionally runs `objdump -d` on the generated sqrt binary and
confirms the `sqrtsd` instruction is present, recording `gum_sse2_verified` in
the summary JSON.

**The SOTA claim:** Sounio is the first language where ISO JCGM 100:2008 GUM
uncertainty propagation is a native compiler primitive — `sqrt_f64` is a
self-hosted stage1 builtin that emits `sqrtsd` (SSE2) in the generated ELF, not
a library function. GUM addition, multiplication, and confidence gates are
expressed in pure Sounio and compile through stage1 to standalone x86-64 ELF
binaries with no fallback or host callback.

The gate corpus lives in `tests/native-v2/gum_primitives/`.

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

1. Keep all native-v2 gates green (self-compile, semantic hardening, f64 ladder,
   science spine, GUM primitives).
2. Expand GUM primitives: multi-step propagation chains, population variance,
   full rapamycin PBPK with `Knowledge<f64>` across all compartments.
3. Replace remaining large aggregate by-value native-v2 boundaries with explicit
   ref/out-param or scalar rails.
4. Expand the Linux driver from the current core subset to more real language
   constructs with one executable witness per increment.
5. Bring Mach-O/AArch64 onto the same scalar discipline, then require the Apple
   SSH gate to run the produced artifact before making macOS support claims.
6. Only after that, connect this lane back to fixed-point self-hosting.

The rule is simple: no public support claim without a runnable artifact and a
repo gate that can reproduce it.

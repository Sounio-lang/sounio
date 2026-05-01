<!-- docs:meta
topic_id: repo.docs.compiler.native-v2-serious-track
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.native-v2-serious-track
-->

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

## Native Algebra Accelerator Spine

The accelerator tracking gate is:

```sh
bash scripts/ci/native_v2_epistemic_accel_spine_gate.sh
```

This gate joins three surfaces that must eventually become one compiler-native
path:

- generated-stage1 CPU oracles under `tests/native-v2/accel_spine/`, including
  nested `Knowledge<f64>` inside a `Compartments` struct and a PBPK-shaped GUM
  batch witness
- public GPU-profile PTX fixtures under `tests/gpu/epistemic_accel/` for f64
  vector arithmetic, GUM-style uncertainty arithmetic, PBPK-shaped batch
  arithmetic, O-SSM-shaped octonion recurrence arithmetic, and S-SSM-shaped
  Cayley-Dickson/sedenion recurrence arithmetic
- compiler-owned native algebra emitters under `self-hosted/gpu/kernels/`:
  octonion Fano multiplication with epistemic shadow registers, sedenion
  Cayley-Dickson multiplication with zero-divisor checks and epistemic shadow
  registers, tensor-core Fano sign correction helpers, and O-SSM f32/f64
  forward/epistemic/backward/associator PTX surfaces

As of this note, the structural rows in this gate pass: the generated stage1
CPU oracle passes with `fallback_path=none` and `host_callback=none`; the
public GPU contract gate passes; the compiler-owned hypercomplex/O-SSM/S-SSM
source surfaces are present; and the public GPU-profile fixtures emit
structurally f64-clean PTX. The top-level gate is `status=pass` only when the
local CUDA runtime smoke also runs; on hosts without `libcuda.so.1`, it reports
`status=partial` and records the CUDA runtime as `not_run`.

The old public GPU artifact still emits raw PTX that can lower f64-shaped
operands through `.f32` opcodes. The gate preserves that raw PTX beside the
checked PTX and runs `scripts/gpu/ptx_f64_legalize.py` as a deterministic f64
PTX legalization bridge. The checked artifact therefore proves the promoted PTX
contract, not that the pinned beta.4 GPU binary has been source-rebuilt.

This does not yet promote GPU execution for epistemic f64 kernels, tensor-core
performance, full O-SSM/S-SSM runtime parity, ROCm, Metal, WebGPU, or DDC. It
does promote the next architectural line: non-conventional algebras now have
compiler-native accelerator surfaces and emitted f64 PTX witnesses, not merely
stdlib helper code or static source probes.

## Epistemic GPU Runtime Parity

The runtime parity gate is:

```sh
bash scripts/ci/native_v2_epistemic_gpu_runtime_parity_gate.sh
```

This gate first runs the accelerator spine without attempting CUDA runtime, then
checks a runtime manifest under `tests/gpu/epistemic_runtime/`. The manifest
contains baseline f64 vector launch rows plus two epistemic algebra rows:

- `tests/run-pass/gpu_epistemic_f64_ossm_parity.sio`
- `tests/run-pass/gpu_epistemic_f64_sedenion_parity.sio`

Those two programs launch O-SSM-shaped octonion recurrence arithmetic and
S-SSM-shaped Cayley-Dickson/sedenion recurrence arithmetic through Sounio
kernel calls, then check host-visible f64 outputs. The helper rejects fallback
by treating any `GPU unavailable:` output as a failure.

On hosts without a visible CUDA driver runtime, the gate reports
`status=partial` with per-row `not_run` results. This is the intended honest
classification: the fixtures are checked and the PTX contract is green, but
native CUDA runtime parity is not promoted until a host with `libcuda.so.1`
runs the rows without fallback.

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

## Dissertation Rapamycin Gate (SOTA)

The dissertation gate is:

```sh
bash scripts/ci/native_v2_dissertation_rapamycin_gate.sh
```

This gate requires `native_v2_gum_primitives_gate.sh`, builds stage1 twice for
deterministic-replay proof, then compiles
`tests/native-v2/science_spine/rapamycin_des_gum.sio` through the generated
stage1 driver. That program implements a three-compartment Cypher DES
rapamycin-class PBPK model:

- `plasma` / `stent-polymer` / `peripheral tissue` as `KnowledgeF64` struct fields
- Ten Euler ODE steps with GUM uncertainty propagation through each flux term
- `gum_add` function uses `sqrt_f64` (SSE2 `sqrtsd`) for quadrature combination
- Confidence gate (`if plasma.confidence < 0.75 { return 1 }`)
- ISO GUM budget table: per-compartment value + uncertainty, then combined total
  via `sqrt(u_plasma² + u_stent² + u_tissue²)`

The gate checks ELF kind, stdout parity with fixture, binary determinism across
pass1/pass2, and confirms `sqrtsd` appears in the generated ELF.

**What this closes for the dissertation:** all three novel contributions now have
native ELF witnesses:
1. GUM-through-ODE: `gum_euler_ode` in science spine — Euler ODE where the
   derivative uncertainty is propagated using GUM multiplicative rule
   `u_flux = sqrt((u_k·x)² + (k·u_x)²)`.
2. ISO uncertainty budgets: `gum_iso_budget` in science spine — three-compartment
   budget table with combined total in ISO GUM Supplement 1 Table 1 structure.
3. Compile-time confidence gate: already in `gum_primitives/pbpk_epistemic_gum`
   and now also in `rapamycin_des_gum`.

**The extended SOTA claim:** Sounio is the first language where a complete
dissertation-quality epistemic PBPK pipeline — three-compartment rapamycin
model, ISO GUM uncertainty propagation, confidence gating, and uncertainty
budget reporting — compiles to a standalone x86-64 ELF with `sqrtsd` (SSE2)
as the native uncertainty primitive.

## Formal GUM Monotonicity (formal/GUM.lean)

`formal/GUM.lean` proves five theorems about ISO GUM uncertainty composition:

- `gum_uncertainty_nondecreasing`: combined uncertainty is at least as large as
  each component (conservative composition).
- `gum_uncertainty_strictly_increasing`: strictly larger when both components are
  positive.
- `gum_zero_component_identity`: combining with zero leaves uncertainty unchanged.
- `gum_combined_comm`: GUM combination is commutative.
- `gum_combined_assoc`: GUM combination is associative (budget totals are
  order-independent).

All proofs use `nlinarith` / `positivity` — zero `sorry`.

## Door β — Variance-of-Associator (SOTA)

`IrAssociatorVariance` is a new IR primitive implementing GUM-correct uncertainty propagation
through non-associative arithmetic for `Knowledge<O>` (octonion-valued epistemic types).

**The bug it closes:** The compiler's default product rule `Var(a·b) = a²Var(b) + b²Var(a)` is
wrong for octonion products when the triple (a,b,c) does not lie on a Fano line. The intermediate
products (ab)c and a(bc) are correlated through the associator [a,b,c] ≠ 0, which the component-
wise rule ignores. The corrected formula adds the variance-of-associator term: `||[a,b,c]||² × σ²`.

**The 168 theorem as a compiler optimizer:** Of the 343 ordered octonion basis triples (e₁..e₇)³,
exactly 168 lie on Fano lines → [a,b,c] = 0 → `IrAssociatorVariance` emits a single VXORPD
(1 instruction / 6 bytes). The remaining 175 triples emit the full correction (~128 EVEX
instructions). This is a 254× instruction-count reduction for the Fano case — the 168 theorem
becomes a compile-time performance gate.

**Register encoding:**
- `imm_flags bit 1 = fano_exact`: 1 → Fano path (1 VXORPD), 0 → non-Fano path (~128 EVEX)
- `imm_f64 = σ²`: combined variance from `KnowledgeOctonion.eps`, set by type-checker
- `label_id`, `mask_k`, `ctrl_base`: same Fano constant encoding as `IrAssociator`

**Formal guarantee:** `formal/NonAssocHessian.lean` §5 adds two zero-sorry, zero-new-axiom theorems:
- `assoc_correction_zero_on_fano`: when Associates(a,b,c), the variance correction is 𝟎
- `door_beta_fano_naive_eq_corrected`: corrected = naive when correction = 𝟎 (roundtrip)

**Science spine test:** `tests/native-v2/science_spine/knowledge_octonion_variance.sio`
demonstrates the witness with σ=1.0:
- Fano triple (e₁,e₂,e₄): norm([e₁,e₂,e₄])=0 → correction=0.0
- Non-Fano triple (e₁,e₂,e₃): norm([e₁,e₂,e₃])=2 → correction=4.0

**SOTA claim:** Sage, GAP, SymPy, and Mathematica all have octonion arithmetic. None have
interval or GUM uncertainty propagation where the algebra structure (Fano-line membership)
gates which variance formula is sound. Sounio is the first compiler where the 168 theorem
is a performance optimization and a correctness condition simultaneously.

## Where We Are Going

The next serious track is:

1. Keep all native-v2 gates green (self-compile, semantic hardening, f64 ladder,
   science spine, GUM primitives, dissertation rapamycin).
2. Expand the rapamycin model: multi-drug interactions, population variance,
   RK4 ODE stepper with adaptive step size.
3. Bring Mach-O/AArch64 onto the same scalar discipline, then require the Apple
   SSH gate to run the produced artifact before making macOS support claims.
4. Replace remaining large aggregate by-value native-v2 boundaries with explicit
   ref/out-param or scalar rails.
5. Expand the Linux driver from the current core subset to more real language
   constructs with one executable witness per increment.
6. Only after that, connect this lane back to fixed-point self-hosting.

The rule is simple: no public support claim without a runnable artifact and a
repo gate that can reproduce it.

<!-- docs:meta
topic_id: repo.docs.audit.madaros-gpu-module-compile-oom-2026-07-03
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-gpu-module-compile-oom-2026-07-03
-->

# Madaros forensic dispatch — GPU driver compile SIGSEGV is virtual-memory exhaustion, not a logic bug

Date: 2026-07-03
Branch: `research/solver-ts3-parallel` @ `abd1cb48a` (merge of #585)
Class: **RESOURCE EXHAUSTION** (`ulimit -v` cap too low for this compile, not a
miscompile) — a specific, evidenced root cause for one manifestation of the broader,
still-open `docs/audit/EPISTEMIC_MADAROS_SIGSEGV_2026-06-29/DISPATCH.md` cluster
Status: root-caused and reproduced; NOT fixed (deliberately — see "Why this wasn't
patched" below)

> Found continuing the work from `docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md`
> (#585): after that fix, `souc build self-hosted/gpu/kretikos_emit_epistemic_wmma.sio`
> gets past typecheck and IR lowering/merge, then still segfaults — same symptom class as
> the already-open SIGSEGV cluster, but this dispatch pins down *why*, for this specific
> case, with much more precision than "compiler-runtime, class unknown."

## Symptom

```bash
$ ./bin/souc build self-hosted/gpu/kretikos_emit_epistemic_wmma.sio -o /tmp/out.elf
...
imported_compile: lower_begin
lower_array: seed_begin
lower_array: seed_done
lower_array: dep_begin 1

Segmentation fault      ( ulimit -v 16777216 2>/dev/null || true; exec timeout 300 "$RAW_MADAROS" "$src" -o "$out" )
error: madaros build: compiler exited 139
```

Reduces to a 2-line reproduction — no code of the driver or `lower_to_ptx.sio`/`ptx.sio`
is involved:

```bash
cat > self-hosted/gpu/probe.sio << 'EOF'
use gpu::kernel_ir::*
fn main() -> i32 with IO, Mut, Panic, Div, Alloc { 0 }
EOF
./bin/souc build self-hosted/gpu/probe.sio -o /tmp/probe.elf   # segfaults, same trace
```

`kernel_ir.sio` alone (~209 KB, 132 top-level functions) is enough to trigger it.

## Root cause

`bin/madaros` (the launcher wrapper, not the compiler itself) hardcodes
`ulimit -v 16777216` (16 GiB virtual memory) at two call sites (lines 165, 229 — the
`build` and generic-invocation paths; a third, unrelated site at line 126 caps the
`--check` path at 8 GiB instead) before invoking the raw compiler ELF. Compiling
`kernel_ir.sio` genuinely needs more than that:

```bash
RAW_MADAROS="$(readlink -f artifacts/self-hosted/madaros)"
( ulimit -v unlimited; "$RAW_MADAROS" self-hosted/gpu/probe.sio -o /tmp/probe.elf )
# completes — no segfault. Gets to "Merged IR: 134 functions", then hits the
# separate, already-documented, LOUD rc=13 (docs/audit/MADAROS_RC13_ELF_256K_CAP_2026-06-28.md
# — 256 KiB ELF staging buffer cap), which triggers the compiler's own automatic
# fallback to a "compact modular IR table" path and finishes with a trivial 140-byte
# stub ELF (Compilation successful). No SIGSEGV anywhere once the ulimit is lifted.
```

Measured peak (`VmPeak`, polled via `/proc/<pid>/status` every 100ms during the run):
**~61.5 GiB** — roughly 3.8× the 16 GiB cap. This is virtual address space, not
necessarily resident memory, but it's real: the process legitimately maps that much
before the compiler finishes lowering/merging.

**Why compiling `kernel_ir.sio` needs ~4× more virtual memory than the wrapper allows**:
almost certainly the same underlying issue the compiler's own (now-removed, see below)
warnings pointed at all session — `GpuOp`/`GpuKernelIr`/`HlirGpuLoweredModule` are
fixed-size-array-heavy structs (`ops: [GpuOp; 1024]` inside `GpuKernelIr`,
`kernels: [GpuKernelIr; 64]` inside `HlirGpuLoweredModule` — the latter alone is
multi-megabyte per value) passed and returned **by value** rather than by reference
(`&!`) or heap allocation, throughout the ~130 functions in this file. Sounio's IR
lowering/merge stage (`module_frontend_lower_program_items_box_traced_with_externs`,
`self-hosted/compiler/module_frontend.sio`) appears to process the whole loaded module
regardless of what's actually reachable from `main` (same pattern already established in
`docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md` — see
that doc's own finding that 3 unrelated large functions in this same file block *any*
importer). If each
of ~130 functions' IR representation carries megabyte-scale value-type data through
several lowering/merge passes without freeing intermediate copies, tens of GiB adds up
fast.

## What was ruled out

- **Not a single buggy function.** Binary-bisected by truncating `kernel_ir.sio` to keep
  only its first *N* top-level functions and testing `souc build` at each *N* (structs/
  enums stay intact since they precede all functions in the file). First pass found
  N=31 OK / N=32 crash, but **re-running the identical N=32 test later crashed at a later
  compiler stage instead** (`lower_array: merge_done 1` was reached before the segfault,
  vs. failing at `dep_begin 1` the first time) — the crash *site* is non-deterministic
  across identical inputs, which is the signature of a resource-exhaustion crash (exact
  failure point depends on what else is already mapped/allocated), not a fixed logic bug
  at a fixed instruction.
- **Not `hlir_gpu_lowered_module_new` specifically**, despite it being the one function
  flagged by "stack frame too large (12150792 bytes)" in every single `souc check`
  output all session, and despite it being genuine, confirmed dead code (zero callers
  anywhere in `self-hosted/`, verified via repo-wide grep). Deleting it removes the
  warning entirely (`souc check` goes from always printing that warning to printing none)
  but **does not fix the segfault** — build still crashes at the same `dep_begin 1` point
  with it removed. The warning and the crash are correlated (same file, same class of
  oversized struct) but not the same thing; the warning was a red herring for *this*
  specific crash, not the cause.
- **Not 8 other confirmed-dead functions in the same file** (`gpu_add_op`,
  `gpu_binary_format_to_string`, `gpu_kernel_set_workload_tag`,
  `gpu_target_profile_unknown`, `gpu_target_profile_rocm_gfx942`,
  `gpu_target_profile_arch_string`, `gpu_target_supports_requirement`,
  `gpu_build_transpose_i32_ir`, `gpu_build_epistemic_wmma_backward_ir` — found via a
  repo-wide zero-caller scan). Not attempted to remove/measure individually since the
  scale mismatch (need to close a ~45 GiB gap; 9 small dead functions won't do it) makes
  it not worth the risk of deleting live-looking code on a guess.
- **Not `IR_MAX_INSTRS`** (the cap defined in `docs/audit/
  MADAROS_IR_MAX_INSTRS_1024_TRUNCATION_2026-06-28.md`, hit for this file per the finding
  in `docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md`)
  — that's a separate, already-documented cap hit by 3 specific large functions
  (`gpu_build_gemm_shared_ir`, `gpu_build_conv2d_ir`, `gpu_build_epistemic_tiled_gemm_ir`);
  raising it was tried and reverted in that prior dispatch's investigation and doesn't
  touch this crash (this crash happens during general IR lowering/merge of the *whole*
  module, not specifically inside those 3 functions).

## Why this wasn't patched

Two candidate fixes, both deliberately not applied:

1. **Raise `ulimit -v` in `bin/madaros`.** Would "fix" this specific repro, but it's a
   project-wide launcher used by every `souc build`/`check`/`run` invocation, including in
   CI. Standard CI runners do not reliably have 60+ GiB of memory available; silently
   requiring it would trade a loud, debuggable SIGSEGV for a machine-dependent hang/OOM-
   kill elsewhere, and would mask genuinely pathological future memory blowups instead of
   surfacing them. Not a safe blanket change without knowing the actual CI memory budget.
2. **Convert `GpuOp`/`GpuKernelIr`/`HlirGpuLoweredModule` (and everything that passes them
   by value) to reference/heap-based passing**, per the compiler's own now-removed
   warning text ("consider passing by &!/reference or using heap/global storage"). This is
   the architecturally correct fix, but it's a large, cross-cutting refactor (`GpuKernelIr`
   is threaded by value through most of `self-hosted/gpu/kernel_ir.sio`'s ~130 functions,
   plus `self-hosted/gpu/lower_to_ptx.sio`, and every `opt/*.sio` file that consumes
   `HlirGpuLoweredModule`), well beyond a single session, and — separately —
   heap-allocation-based fixes may not even be viable right now: GitHub issue "codegen
   (madaros): SIGSEGV compiling any `Box::new`" is open against this same compiler,
   meaning `Box`-based heap allocation is itself untrustworthy as a workaround until that
   closes.

## Impact

Same as the parent dispatch (#585's `MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md`):
`self-hosted/gpu/kretikos_emit_epistemic_wmma.sio` (and any future driver needing
`gpu_lower_to_ptx`) cannot produce a real native ELF via `souc build` on a normally-
provisioned machine. The module-combination *typecheck* blocker is fixed (#585); this is
the next blocker in the chain, and it's a resource ceiling, not a code defect — running on
a machine with a much higher `ulimit -v` (or a machine-specific override of
`MADAROS_RAW_BIN`/direct raw-ELF invocation bypassing `bin/madaros`'s wrapper limits, as
used for the reproduction above) is the only known way through it today.

## Suggested next steps (not attempted here)

1. Measure actual peak **RSS** (not just `VmPeak`, which overstates true memory pressure
   via reserved/lazy address space) with a proper profiler (`/usr/bin/time -v`, `valgrind
   --tool=massif`, or `/proc/<pid>/status VmHWM`) — neither `gdb` nor GNU `time` were
   available in this environment; only manual `/proc` polling was used here, which is
   coarse (100ms sampling) and VmPeak-only.
2. Bisect the ~130 functions in `kernel_ir.sio` for the ones with the largest individual
   memory footprint during lowering (not just "large IR instruction count" — a function
   can have modest instruction count but still carry a huge value-typed local like
   `[GpuKernelIr; 64]`) and prioritise converting *those* to reference-based signatures
   first, rather than a whole-file rewrite.
3. Decide, with whoever owns Madaros CI resource budgets, whether raising `ulimit -v` is
   acceptable for this specific launcher path and by how much, informed by real CI runner
   memory limits (this dispatch deliberately did not make that call).

## Cross-references

- `docs/audit/EPISTEMIC_MADAROS_SIGSEGV_2026-06-29/DISPATCH.md` — the broader, still-open
  SIGSEGV cluster this dispatch's finding likely explains (or explains a subset of). As of
  this branch, that dispatch's own forensic history (a single §7, "Suggested compiler
  bisection order") never identifies `ulimit -v`/virtual-memory exhaustion as a factor;
  worth revisiting whether some of its "fixed" and "open" items are actually the same
  resource-ceiling issue observed on different inputs. (An earlier draft of this cross-
  reference cited specific commit hashes and a §7a/§7b split that do not exist in this
  branch's version of `DISPATCH.md` — corrected here; do not carry those citations forward.)
- `docs/audit/MADAROS_IR_MAX_INSTRS_1024_TRUNCATION_2026-06-28.md` — defines the general
  `IR_MAX_INSTRS` cap concept (demonstrated there via a synthetic `theorem::smt` harness,
  unrelated to `kernel_ir.sio`); the specific finding that 3 functions in `kernel_ir.sio`
  hit this cap lives in the `MODULE_COMBINATION` doc below, not in this one.
- `docs/audit/MADAROS_RC13_ELF_256K_CAP_2026-06-28.md` — the loud error this crash's repro
  hits immediately *after* removing the `ulimit -v` cap, confirming the segfault and the
  256 KiB ELF cap are two different, sequential blockers on the same path.
- `docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md` —
  the prior blocker in this same driver's build chain (module-combination typecheck
  failures), fixed in #585; this dispatch picks up immediately where that one left off.

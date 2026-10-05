<!-- docs:meta
topic_id: repo.docs.audit.madaros-gpu-kernel-ir-lower-to-ptx-ptx-module-combination-2026-07-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-gpu-kernel-ir-lower-to-ptx-ptx-module-combination-2026-07-02
-->

# Madaros forensic dispatch — `gpu::kernel_ir` + `gpu::lower_to_ptx` + `gpu::ptx` fail to check together

Date: 2026-07-02, resolved 2026-07-03
Branch: `gpu/epistemic-tensor-core-next` (base: `research/solver-ts3-parallel` @ `65f44e60c`,
merge of PR #572); fix landed on `fix/gpu-lower-to-ptx-module-combo`
Class: **PRE-EXISTING MODULE-COMBINATION BREAKAGE** (hundreds of typecheck errors,
independent of any function body or call site — reproducible with an empty `main`)
Status: **FIXED** (`souc check` verdict=0 on all three files together, and on the
`kretikos_emit_epistemic_wmma.sio` driver this was blocking). See "Resolution" below.
The three root causes were NOT what the diagnosis section originally guessed
(the diagnosis correctly named privacy-annotation drift as one factor, but there were
two more independent bugs stacked underneath it — see Resolution).

> Found while trying to write a CLI driver
> (`self-hosted/gpu/kretikos_emit_epistemic_wmma.sio`) to emit PTX for the
> compiler-generated epistemic WMMA matmul kernel
> (`self-hosted/gpu/kernel_ir.sio::gpu_build_epistemic_wmma_matmul_16x16_ir`) for
> hardware verification on the DGX Spark. This dispatch documents why that driver
> still cannot be built into a native ELF on this branch, so the blocker survives
> context loss instead of being rediscovered from scratch.

## Symptom

A `.sio` file that does nothing but import three GPU backend modules and return 0 from
`main` fails `souc check` with **hundreds of errors** — no call into any of the three
modules is required to trigger it:

```sio
use gpu::kernel_ir::*
use gpu::lower_to_ptx::*
use gpu::ptx::*

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    0
}
```

```
$ ./bin/souc check probe.sio
run_check_mode: about to check 4 modules
error[E175] ... : function is private in its defining module     (335 occurrences)
error[E177] ... : enum constructor is private in its defining module   (18 occurrences)
error[E046] ... : struct literal has wrong number of fields       (5 occurrences)
run_check_mode: verdict=1
```

None of these errors reference anything the probe file itself wrote — `main`'s body is
just `0`. They come from the transitive resolution of `kernel_ir.sio` + `lower_to_ptx.sio`
+ `ptx.sio` against each other.

## Why this matters

`self-hosted/gpu/mod.sio` — the GPU backend's own orchestrator, which re-exports
"all GPU code generation and runtime modules" per its header comment — does **not**
import `gpu::lower_to_ptx`:

```sio
module gpu::mod

use gpu::kernel_ir::*
use gpu::hlir_to_gpu::*
use gpu::epistemic_spirv::*
use gpu::ptx::*
use gpu::metal::*
use gpu::cuda_tile::*
```

`souc check self-hosted/gpu/mod.sio` passes clean (only pre-existing stack-frame-size
warnings, `verdict=0`). That's real evidence the rest of the GPU backend is healthy — but
it also means **nothing in the checked, working surface actually exercises
`lower_to_ptx.sio` together with `kernel_ir.sio` and `ptx.sio`**, even though
`lower_to_ptx.sio` is the documented, intended "phase 1" lowering path
(`GpuKernelIr → PTX`) and both `self-hosted/test_epistemic_wmma.sio` and
`self-hosted/gpu/spirv_lower.sio`'s self-checks call `gpu_lower_to_ptx` directly. Those
test files have no `use` statements themselves and are only ever compiled by
concatenation into the full bootstrap bundle (`scripts/bootstrap/bootstrap_concat.sh`,
`BOOTSTRAP_PROFILE=full`), which sidesteps normal module-boundary checking entirely (no
`use`, no cross-module privacy enforcement) and — separately — currently segfaults during
native codegen for unrelated reasons. So this 3-module combination, checked as its own
proper module graph, has apparently never been validated.

## What's actually failing

Not investigated function-by-function (335 + 18 + 5 = 358 errors is not something to
triage one at a time by hand). The three error classes:

- **E175 "function is private in its defining module"** (335×) and **E177 "enum
  constructor is private in its defining module"** (18×) — some function/module needed by
  the `lower_to_ptx.sio` ⟷ `ptx.sio` call chain lost its `pub` modifier (or was never
  `pub`) relative to what cross-module resolution now needs. This matches a pattern seen
  elsewhere on this branch this session: `self-hosted/gpu/kernel_ir.sio`'s `struct Name`
  and `struct GpuOp` field declarations were observed to have lost their `pub` prefixes at
  some point in this branch's history (compare: an earlier commit on this same GPU work
  needed to add `pub` back to `gpu_build_epistemic_wmma_matmul_16x16_ir` for exactly this
  reason). This looks like **ongoing, uncoordinated privacy-annotation drift** across the
  GPU module tree, not a one-off.
- **E046 "struct literal has wrong number of fields"** (5×, at byte offsets 80304, 82284,
  82895, 83507, 85443 in the 4-module concatenation) — a struct literal somewhere in this
  combination doesn't match its struct's current field list. Given the E175/E177 pattern
  above, the likely cause is the same class of drift: a struct gained/lost a field in one
  file without every literal constructing it (in a different file) being updated to match.

Neither class was root-caused to a specific symbol/line — that requires either a bisection
across `lower_to_ptx.sio`'s and `ptx.sio`'s own internal `use` graphs (which files, not
which functions), or fixing the compiler to emit file:line instead of raw byte offsets
into the concatenated check buffer.

## Reproduction

```bash
cat > /tmp/probe.sio << 'EOF'
use gpu::kernel_ir::*
use gpu::lower_to_ptx::*
use gpu::ptx::*

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    0
}
EOF
cp /tmp/probe.sio self-hosted/gpu/probe.sio   # must live under self-hosted/gpu/ for `use gpu::*` to resolve
./bin/souc check self-hosted/gpu/probe.sio
rm self-hosted/gpu/probe.sio
```

Expect `verdict=1` with the error counts above. `./bin/souc check self-hosted/gpu/mod.sio`
in the same tree passes (`verdict=0`) for contrast.

## What was ruled out (not the cause)

- **Not a MY-code issue.** The probe's `main` body is empty; no call into any of the three
  modules occurs.
- **Not `IR_MAX_INSTRS` (the sibling `docs/audit/MADAROS_IR_MAX_INSTRS_1024_TRUNCATION_2026-06-28.md`
  finding).** That issue is real and separate — three unrelated large functions in
  `kernel_ir.sio` (`gpu_build_gemm_shared_ir`, `gpu_build_conv2d_ir`,
  `gpu_build_epistemic_tiled_gemm_ir`) do independently exceed the 1024-instruction cap
  and block `souc build` (not `souc check`) of anything importing `kernel_ir.sio`
  wholesale — but fixing that (raising the cap; attempted and reverted this session,
  touching ~680 call sites across 7 files) does not touch the E175/E177/E046 errors here,
  which occur at `check` time, before lowering ever starts.
- **Not the known Madaros native-codegen segfault** (see `docs/audit/EPISTEMIC_MADAROS_SIGSEGV_2026-06-29/`
  and related) — this dispatch is entirely about `check`/typecheck-time failures, upstream
  of codegen.
- **Not a stale pinned binary.** `bin/souc` was rebuilt as of this branch's HEAD; the
  errors reproduce with the in-tree compiler on the in-tree source, not an old artifact.

## Impact

Any new CLI driver that needs `gpu_lower_to_ptx` (the `GpuKernelIr → PTX` path) — i.e.
anything downstream of `self-hosted/gpu/kernel_ir.sio`'s `gpu_build_*_ir()` builder
functions that isn't reachable through `bin/kretikos`'s existing K-AXI-only dispatch —
cannot currently be compiled to a native ELF via `souc build`, because `souc check`
already rejects the module combination before codegen is reached. This blocked
`self-hosted/gpu/kretikos_emit_epistemic_wmma.sio` (added 2026-07-02 to expose
`gpu_build_epistemic_wmma_matmul_16x16_ir`'s PTX for DGX Spark hardware verification) from
ever reaching the native-codegen stage on this branch.

## Proposed next step (not attempted here)

1. Bisect by importing `gpu::lower_to_ptx` and `gpu::ptx` alone (no `kernel_ir`) and vice
   versa, to narrow which pairwise combination first breaks — this dispatch only
   established that all three together break, not which pair.
2. Audit `pub`/private-modifier consistency across the whole `self-hosted/gpu/` tree in one
   pass (grep for struct/fn declarations missing `pub` that are referenced via `::*`
   imports from outside their own file) rather than patching one symbol at a time as each
   is hit — the repeated pattern (`Name`, `GpuOp`, and now this) suggests a systemic gap,
   not isolated typos.
3. Add a CI check that runs `souc check` on `gpu::kernel_ir` + `gpu::lower_to_ptx` +
   `gpu::ptx` together (e.g. via `mod.sio` gaining a `use gpu::lower_to_ptx::*` line, if
   that's semantically correct for the orchestrator to own) so this combination is never
   silently unvalidated again.

## Resolution (2026-07-03)

Bisected pairwise as step 1 above suggested: `kernel_ir + lower_to_ptx` alone reproduced
357 of the 358 errors; `lower_to_ptx` alone (which has its own internal `use` of both
other modules) reproduced 334 as **E137 "use of undeclared variable"** instead of
E175/E177/E046 — a different error class, which turned out to be the first real clue.
Three independent, stacked bugs, found in this order:

1. **`lower_to_ptx.sio` had lost its own `use gpu::kernel_ir::*` / `use gpu::ptx::*`
   lines** (and `gpu_lower_to_ptx`'s `pub`) at some point in this branch's churn — every
   reference inside it to `GpuKernelIr`, `PtxBuf`, `ptx_buf_new()`, etc. was genuinely
   undeclared in that file's own scope. `use` in Sounio is **not transitive** — a
   dependency's own `use` lines only resolve symbols for that dependency's own body; they
   don't extend the importer's visible symbol set. Restored both `use` lines.
2. **`kernel_ir.sio` had 5 struct literals with a duplicated field key** (`rhs_reg` or
   `lhs_reg` given two different `key: value` entries in the same `GpuOp { ... }`
   literal — 24 pairs for a 23-field struct). This is what E046 ("wrong number of fields")
   actually meant; the count check runs before any duplicate-key check. All 5 were
   address-computation ops (`GpuAdd`/`GpuSetpLt`/`GpuStoreSharedPred`) built from a copied
   template that kept both the template's placeholder value and the real one. Removed the
   redundant copy in each, keeping the meaningful value.
3. **`ptx.sio` called `i64_to_string`, which it never defined or imported** (it only exists
   in `lower_to_ptx.sio` and `opt/warp_vote_fastpath.sio` — importing either from `ptx.sio`
   would be circular, since `lower_to_ptx.sio` imports `gpu::ptx`). Added a local copy.
4. **Privacy-annotation drift, confirmed as real and widespread, not a one-off**: in
   `kernel_ir.sio`, the `Name`/`GpuOp`/`GpuParam`/`GpuKernelIr` structs and their fields,
   all 12 top-level enums, and 131/132 top-level functions were missing `pub`. In
   `ptx.sio`, `PtxBuf` and all 147 of its top-level functions were missing `pub`. Both
   files exist specifically to be shared libraries for the rest of the GPU backend, so
   blanket `pub` (not selective) is the correct fix, applied via a scripted pass rather
   than fixing one symbol at a time as each was hit by a different caller.

**Verified:** `souc check` now passes (`verdict=0`, no errors) on `kernel_ir.sio`,
`ptx.sio`, `lower_to_ptx.sio` individually and together, and on
`self-hosted/gpu/kretikos_emit_epistemic_wmma.sio` (the driver this was blocking).
`souc build` of that driver now gets past typecheck and IR merge — down to a 1-function
merged IR from the ~240–330 seen before this fix — and reaches native codegen, where it
hits the **separate**, already-tracked, open Madaros SIGSEGV cluster
(`docs/audit/EPISTEMIC_MADAROS_SIGSEGV_2026-06-29/`). That is out of scope for this
dispatch; getting a real native ELF (and from there, PTX for DGX Spark hardware
verification) still depends on that separate bug being fixed.

Fix commit: `fix(gpu): resolve gpu::kernel_ir+lower_to_ptx+ptx module-combination
breakage` on `fix/gpu-lower-to-ptx-module-combo`.

## Cross-references

- `docs/audit/MADAROS_IR_MAX_INSTRS_1024_TRUNCATION_2026-06-28.md` — separate, orthogonal
  compile-time cap issue in the same file (`kernel_ir.sio`), hit while chasing this one.
- `docs/audit/EPISTEMIC_MADAROS_SIGSEGV_2026-06-29/` — separate native-codegen segfault
  class, downstream of where this dispatch's errors occur.
- `self-hosted/gpu/kretikos_emit_epistemic_wmma.sio` — the driver whose build is blocked by
  this; its commit history on `gpu/epistemic-tensor-core-next` records the specific
  symptoms hit in order (segfault → `var`-without-initializer typecheck regression, fixed
  → this module-combination breakage, not fixed).

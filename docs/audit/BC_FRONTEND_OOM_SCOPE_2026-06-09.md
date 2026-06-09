# Scope: B/C front-half OOM — eliminating IrModule by-value materialization

Branch `claude/ir-heap-indirect` off `claude/codegen-largestruct-fix` @ `f73bc57a0`
(which already carries the A+D codegen fixes + the rebootstrapped `bin/souc` `d8f2a76d`).
Worktree `/workspace/sounio-ir`. This is a **scoping document**, not an implementation.

## Context

After the A+D codegen fixes landed (PR #270), the sole remaining self-host wall is the
**front-half OOM/SIGSEGV** when `--native-compile main.sio` lowers the 6642-function compiler:
`Merged IR: 64 → imported_simple_ir_missing_main → full-IR fallback → OOM (137)` at the 16 GB
worker ceiling (job 2380 PHASE 3; baseline job 2356 = SIGSEGV peak 7.9 GB at the old 1400-fn cap).

### Why it OOMs (root cause, measured)

`IrModule` (`self-hosted/ir/ir.sio:1929`) is a **~526 MB by-value struct**:
- `functions: [IrFunction; 8192]`, each `IrFunction` ≈ **64 KB** (`instrs: [IrInstr; 256]`, IrInstr ≈ 248 B)
  → `8192 × 64 KB ≈ 526 MB`. (The 1400→8192 cap bump inflated the old "96 MB" figure ~5×.)

The OOM is **NOT inherent** — ONE 526 MB module fits 16 GB easily. It comes from **multiplied
526 MB footprints**. Crucially there are TWO multipliers, and the second is the one that decides
whether the easy fix works:

1. **By-value COPIES** — `ir_merge_modules(dst: IrModule, src: IrModule) -> IrModule`
   (`module_frontend.sio:891`): `var merged = dst` (copy) + return-by-value = **TWO 526 MB copies
   per merge**, once per imported module.
2. **Per-import ALLOCATIONS (independent of copies)** — with fixed `[IrFunction; 8192]`, **every**
   `IrModule` allocation is ~526 MB regardless of how many functions it holds.
   `module_frontend_lower_imported_source_recursive(...) -> IrModule` (`:2699`) returns an
   `IrModule`, so it almost certainly **allocates a fresh full 526 MB module per import**. If the
   bootstrap runtime uses a **bump/arena allocator with no free/reuse** (common for self-hosted
   compilers), then N imports keep N × 526 MB **live even after every copy is removed**. THIS, not
   the copies, is the gating unknown — see Phase 0 and Risks.

**Corroborating evidence**: the 1400-cap baseline (job 2356) died at **FNINSTR=0 — before any
function body was lowered**, i.e. during load/merge/preseed where modules are being *constructed
and combined*. That points at construction/merge (multipliers above), not auto-deref of bodies, as
the primary locus — prioritize A1 over A2.
2. **By-value IrModule returns/params** (a handful):
   - `module_frontend_lower_imported_source_recursive(...) -> IrModule` (`:2699`)
   - `compile_multimodule_native_with_ir(..., preloaded_ir: IrModule, ...)` (`module_native_driver.sio:1093`)
   - `MultiModuleIrResult { module: IrModule, ... }` (`:2754`) — *already* mitigated by the
     `load_multimodule_ir_into(&!out, ...)` BSS-global workaround (`:3999`, used by the driver).
3. **~94 auto-deref reads `self.module.FIELD`** in `lower.sio` (vs 12 already-explicit
   `(*self.module).FIELD`): the bootstrap codegen **materializes the whole 526 MB module to read
   one field** (confirmed by in-tree comments at `lower.sio:3196,3300,3323` — "auto-deref …
   materializes the whole module and faults"). This is the same codegen family as the A fix, on
   the READ side.

### What does NOT work (ruled out)

- **Fixing the compact-IR path** (`module_frontend.sio:3491`, cap 64): it is a text-scan lookup
  table of ~20 hardcoded "simple" function shapes (4 i64s/fn), cannot represent main.sio's real
  functions, and doesn't avoid the front-half lowering materialization anyway. Dead end.
- **Naive cap-bump**: bigger `IrModule` = worse fault (more bytes per copy). Already known.

## Options

### Option A — Eliminate materializations, keep fixed arrays (RECOMMENDED, tractable)
Make the front-half **allocate exactly one** `IrModule` and thread it by pointer; never copy AND
never re-allocate a second full module.
- **A1 (source, biggest win) — SINGLE ACCUMULATOR, ZERO per-import full-`IrModule` allocation.**
  Not merely "in-place merge + box the returns" (that removes copies but can leave N × 526 MB of
  *allocations* live under a no-free allocator). Instead thread **one** `Box<IrModule>` accumulator
  through the entire recursive lowering: `lower_imported_source_recursive(dst: &! IrModule, ...)`
  appends each import's functions **directly into the shared module** (`dst.functions[dst.fn_count..]`
  by index) and **returns nothing large** — so only one full `IrModule` is ever allocated.
  `ir_merge_modules` likewise becomes `(dst: &! IrModule, src: &IrModule)` in-place (and ideally
  src is lowered straight into dst, never built standalone). Box/`&` `compile_multimodule_native_with_ir`.
- **A2 (auto-deref reads):** two sub-options, can combine:
  - **A2a (surgical codegen, preferred):** fix the bootstrap codegen so an auto-deref field
    **read** `box.field` reads *through* the pointer (computes field address + loads) instead of
    materializing the pointee — the READ analog of the A store fix. Fixes all 94 sites at once +
    future-proofs. Requires a `lean_single.sio` change + rebootstrap.
  - **A2b (source sweep, fallback):** mechanically convert the ~94 `self.module.X` →
    `(*self.module).X`. No rebootstrap, but whack-a-mole and re-introducible.
- **Memory after A (IF the allocator-free question resolves favorably):** one 526 MB module +
  lowering working set ≪ 16 GB.
- **Cap headroom is tight:** `functions[8192]` holds the **merged whole-program** function count
  (~6642 in recent runs; single-file main.sio ≈ 1418 per the 06-08 memory — Phase 0 must pin which
  number fills the array). 6642 / 8192 ≈ **only ~19 % headroom, and growing** — so Option B's
  ceiling-removal is **near-term, not "only if needed."**
- **Cost:** bounded source refactor (single-accumulator threading + in-place merge) + one codegen
  fix/rebootstrap. No new language features. The 412 `module.functions[i]` index sites are
  UNCHANGED (still `[i]`).

### Option B — Heap-indirect / variable-length IR (the memory's stated fix; big; defer)
Make `functions`/`instrs` heap-allocated growable arrays so `IrModule` is small (pointers+counts),
copies are cheap, and the 8192 cap disappears. **Blocked on a missing language feature**: Sounio
has no `Vec<T>` / owned dynamic slice / runtime-sized allocation (`IrAlloc` is compile-time-size
only; `Box<T>` wraps fixed T). Building that (owned `[T]` value type + ptr/len metadata +
`Vec::push` + runtime-sized `IrAlloc` + native malloc lowering) is a multi-sprint prerequisite.
**Promotion criteria (more near-term than "only if needed"):** pursue B if (a) Phase 0 shows the
allocator does NOT free per-import full-module allocations and A1's single-accumulator form is
infeasible to thread cleanly, OR (b) the merged fn_count headroom under 8192 is judged too thin to
land on (it is already ~19 %). Building B needs an owned `[T]` value type + ptr/len metadata +
`Vec::push` + runtime-sized `IrAlloc` + native malloc lowering — a multi-sprint prerequisite.

### Option C — Fix the compact-IR path — REJECTED (see "What does NOT work").

## Phase 0 — RESULTS (run 2026-06-09, SLURM job 2382)

Static + empirical, both decisive. The gating question is **answered: the allocator does NOT
free**, and the leak is **upstream of the documented merge loop**.

- **Allocator (static):** runtime is mmap-per-allocation (`lean_single.sio:emit_heap_alloc_x86`);
  `emit_heap_free` is emitted ONLY for the explicit `free()` builtin (2 sites) — **no drop / RAII /
  scope-free**. So every `Box::new(IrModule)` mmaps ~526 MB and **leaks until exit**.
- **Empirical (job 2382, instrumented `mc --native-compile main.sio`, 16 GB worker):**
  - `NATIVE_COMPILE_RC=137` (OOM); **PEAK 16194 MB**; first RSS sample already **15381 MB**.
  - **`MERGE_STEPS_REACHED=0`** — the instrumented merge loop (`module_frontend.sio:3938` /
    `:4187`) **never ran**. The OOM is **upstream**, in
    **`module_frontend_lower_imported_source_recursive` (`:2699`)** — the recursive lowerer that
    returns `IrModule` BY VALUE and recurses over all imports (invoked at `:4004` before the merge
    loop).
  - `15381 MB / 526 MB ≈ 29` live `IrModule`s at OOM (of 72–81 imports ⇒ dies ~40 % through);
    RSS climbs **monotonically** (no free) — confirms per-import 526 MB leak quantitatively.
  - Compact path runs first (`Merged IR: 64 → missing_main`) then falls back to the full-IR path
    which OOMs — compact is a cheap dead-end, as scoped.

**Conclusion:** A1-**strong** (single accumulator, zero per-import full-`IrModule` allocation) is
**required** (no-free allocator) and **sufficient** (collapses ~29+ allocations to one ≪ 16 GB).
**Option B is NOT forced** by memory (one 526 MB fits easily) — only by the ~19 % cap headroom,
which can be addressed later. **Refined Phase-1 target = `module_frontend_lower_imported_source_recursive:2699`**
(thread a shared `&! IrModule` accumulator through the recursion; never return/allocate a fresh
`IrModule` per import), then the two merge loops (`:3962`, `:4229`) → in-place. Harness +
SUMMARY: `slurm-jobs/ir-heap-indirect/` (job 2382).

## Phase 1 — ATTEMPT 1 RESULT (job 2383): builds + gate-clean + peak down, but NOT correct yet

The A1 single-accumulator refactor (commit on `claude/ir-heap-indirect`) is implemented:
`ir_merge_modules_into` / `ir_module_logical_reset` / `_summary_into` / `_recursive_into`,
wired into the `--native-compile` full-IR path via `&!(*out).module`.

Job 2383: **`MC_BUILD_RC=0`** (compiles), **`native_v2_multimodule 27/27`** (no regression),
peak **16194 → 12979 MB** (down, but far from the ~1–1.5 GB target). BUT
**`NATIVE_COMPILE_RC=139` (SIGSEGV)** with **`Merged IR: 6878231886984147744`** — a garbage
`fn_count` (the value decodes as ASCII bytes = it's reading `Name`/string-table data). So the
merge reads/writes `fn_count` at the **wrong offset** through the pointer threading; the corrupt
count drives `functions[garbage] = …` wild writes (the residual 13 GB) → SIGSEGV.

**Pinned next bug:** `IrModule.fn_count` sits ~526 MB into the struct (right after
`functions[8192]`). Accessing it (and `string_count`) through the freshly-threaded `&!`/`&`
pointers at that scale mis-resolves. Local-struct access to the same fields works (the original
by-value code did), and the 800 KB-scale pointer test passes — but a 520 MB **local** struct is
outright rejected ("consider *mut/heap or global arrays"), so large IrModules must be
global/BSS, and the `var scratch = ir_empty_module()` **local** (BSS-spilled) + `&! scratch` +
`&!(*out).module` interaction at 526 MB is the suspect. Candidate fixes for attempt 2:
(a) make the scratch a **global** `IrModule` (matches the proven `NATIVE_DRIVER_IR_RESULT`
pattern; large IrModules should never be locals); (b) add per-merge `fn_count` instrumentation to
localize whether the bad offset is on the read (`(*mod_ptr).fn_count`), the write
(`logical_reset`/append), or the scratch SRET init; (c) if `&!(*out).module` is implicated,
reference the global field directly (the driver already uses `&NATIVE_DRIVER_IR_RESULT.module`).
Harness: `slurm-jobs/ir-heap-indirect/submit-phase1-verify.sh`. **Phase 1 is NOT done.**

### Phase 1 — ROOT-CAUSED + FIXED (attempts 2-4 / pinpoint jobs 2385-2386); confirm BLOCKED by infra
Instrumented localization: `DBG_A1` (job 2385) → reset/scratch reads OK, garbage only `after_merge`;
`DBG_M` (job 2386) → garbage `after_fns` = the **functions-loop store**. **Root cause:**
`(*dst).functions[(*dst).fn_count] = func` uses an **inline deref-read in the array-index position**,
so the 64 KB struct-element store overshoots onto `fn_count` (the ASCII garbage = a stored `Name`).
**Fix (the proven `find_or_add_fn_id` idiom): bind the index to a local first** —
`let idx = (*dst).fn_count; (*dst).functions[idx as usize] = func; (*dst).fn_count = idx + 1` — applied
to both array stores in `ir_merge_modules_into` and the `summary_into` store. **Committed.**

**⛔ Confirm run BLOCKED by a SLURM infra outage** (node `r770-proxmox` disk-pressure cascade:
controller repeatedly evicted → killed job 2387 mid-build; login pods all Error/Evicted; cpu-ops
worker stuck `Init` 35 min+). Workaround committed (submit via controller pod:
`LOGIN_POD=slurm-pilot-controller-0 LOGIN_CTR=slurmctld`), pending a healthy worker. **Resume:**
rerun the harness, confirm `DBG_M after_fns` is small + `Merged IR ~6642` + peak ↓ ~1.5 GB + 27/27,
then drop the `DBG_*` instrumentation before merge.

## Recommended phased plan

- **Phase 0 — instrumented baseline (measures the GATING question, not just copy-count).** SLURM
  `--native-compile main.sio` under an RSS poller (reuse the `slurm-jobs/codegen-largestruct/`
  harness), instrumented to answer **three** things — this triplet, not "copy-count × 526 MB ≈
  peak," decides A1-strong vs Option B:
  1. **How many full `IrModule` allocations** happen during one run (instrument
     `ir_empty_module`/`Box::new(IrModule)`/the recursive lowering entry).
  2. **Does the allocator free/reuse?** — does RSS *drop* after a scope that built a module exits,
     or does it monotonically climb (bump/arena, no free)? This is THE determinant.
  3. **Does peak scale with import count** (vary the import set) — and would it still after copies
     are removed (i.e. is the driver allocation, not copy)?
  Also pin which fn_count fills `functions[8192]` (single-module ~1418 vs merged ~6642).
- **Phase 1 — A1 (in-place merge + Box the by-value sigs).** Source-only. Rebuild mc, re-run the
  native-compile probe under the RSS poller. Expected: peak drops from ~16 GB toward ~1–2 GB.
  This alone may clear the OOM.
- **Phase 2 — A2 (auto-deref reads).** If Phase 1's peak is still dominated by per-field
  materializations, do A2a (surgical auto-deref-read codegen fix + rebootstrap); fall back to A2b
  (source sweep) if the codegen fix proves too subtle. Re-measure.
- **Phase 3 — only if needed: Option B.** If the single 526 MB allocation + working set still
  doesn't fit, or > 8192 fns are needed, scope the `Vec`/heap-indirect language feature as its own
  effort.
- **Acceptance:** `--native-compile main.sio` completes (no OOM/SIGSEGV) and emits a **running**
  gen-N ELF; then gen2==gen3 self-host of the full pipeline; `release_gate` no-regression.

## Risks & guardrails

- **Bootstrap paradox:** any `lean_single.sio` codegen fix (A2a) is compiled by the current
  `bin/souc`; write it in the BSS-spill/`(*box)` idiom that compiles correctly (as A did).
- **In-place merge correctness (A1):** appending into `dst.functions[dst.fn_count..]` must respect
  the 8192 cap with a loud guard; string-table / export / prof-counter sub-tables must merge too
  (not just functions). Verify against the multimodule gate (27/27) — must stay green.
- **8192 cap is tight:** main.sio = 6642 now and growing; Option A buys headroom to 8192, not
  beyond. Track fn_count; Option B is the real ceiling-remover.
- **⚠️ THE GATING UNKNOWN — allocator free-behavior:** with fixed `[IrFunction; 8192]` every
  `IrModule` alloc is 526 MB. If the bootstrap allocator is bump/arena (no free) AND lowering
  allocates per-import, then **removing copies alone does NOT move the peak** — A1 MUST be the
  single-accumulator / zero-per-import-allocation form, or Option B is forced. Phase 0 measures
  this BEFORE any code change. Do not assert the optimistic "~1–2 GB" outcome until measured.
- **Measure, don't assume:** the FNINSTR=0 death (job 2356) supports merge/preseed (A1) as primary
  and auto-deref-of-bodies (A2) as secondary — but Phase 0's three-way measurement, not this
  inference, gates Phase 1→2→B.
- **One 526 MB module must fit:** verify the post-A1 peak empirically (worker is 16 GB bounded).

## Critical files

- `self-hosted/ir/ir.sio:1929` — `IrModule` (526 MB); `:709` `IrFunction` (64 KB); `:661` `IrInstr`.
- `self-hosted/compiler/module_frontend.sio` — `ir_merge_modules:891` (A1 target),
  `module_frontend_lower_imported_source_recursive:2699`, `MultiModuleIrResult:2754`,
  `load_multimodule_ir_traced_into:3999` (the existing &!out pattern to mirror).
- `self-hosted/compiler/module_native_driver.sio` — `compile_multimodule_native_advanced:1116`
  (compact→full fallback), `compile_multimodule_native_with_ir:1093`, the BSS global
  `NATIVE_DRIVER_IR_RESULT:29`.
- `self-hosted/ir/lower.sio` — ~94 auto-deref `self.module.X` reads (A2); the `Lowerer.module:
  Box<IrModule>` pattern + explicit `(*self.module)` precedent.
- `self-hosted/compiler/lean_single.sio` — bootstrap codegen for A2a (auto-deref-read lowering);
  rebootstrap via `slurm-jobs/codegen-largestruct/submit-rebootstrap.sh` (reuse).
- Prior diagnosis: memory `project_lowercodegen_oom_2026-06-08`, `project_codegen_largestruct_lvalue_2026-06-09`.

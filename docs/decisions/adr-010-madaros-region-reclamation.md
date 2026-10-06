<!-- docs:meta
topic_id: repo.docs.decisions.adr-010-madaros-region-reclamation
authority: repo_only
audience: users
last_validated: 2026-10-06
validated_by: Claude Opus 5.5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-010-madaros-region-reclamation
-->

# ADR-010: Madaros Region Reclamation at Function Return

**Status**: experimental
**Date**: 2026-10-06
**Related**: ADR-004 (capacity guards over silent corruption), PR #2737,
`self-hosted/native/gc.sio` ("Region reclamation"),
`docs/audit/MADAROS_HANDLE_182_BBB_PER_CONSTRUCTION_DISPATCH_2026-09-26.md`

---

## Context

The Madaros native runtime allocates every >16 B aggregate from one bump
heap and one bump handle table (2^22 entries) carved from a fixed 2 GiB
mapping, and never frees either: a whole-heap collector cannot run because
stack maps carry slot counts, not root bitmaps, so no point is known at which
every live root is visible. Iterative scientific code (an RK2 step building
a few temporaries per call) therefore died with `madaros: handles full`
(exit 182) after a few million constructions, although almost all of them
were dead at the end of the call that made them.

Freeing memory is the one runtime change that can turn a crash into a
silent wrong answer: an object reclaimed while still reachable is reused by
the next allocation and read back as somebody else's data. This ADR records
the lifetime contract that makes reclamation sound, what it refuses, and the
mechanisms that keep the refusals honest.

## Decision

### 1. Regions

A function that lowering judges eligible opens a **region** at entry
(`__rgn_enter`): it records the heap cursor H0 and handle count C0 in a
record on a side stack carved above `heap_limit` (65 536 records,
`native_v2_region_stack_*`). Both resources are bumped, so everything the
call allocates sits at addresses >= H0 and handle ids > C0.

**Invariant (address order).** Every live object allocated before a region
opened lies below its H0; everything allocated after lies at or above it. A
reset preserves the invariant because survivors are re-placed at the start
of the closing region, i.e. at the top of the parent's.

At return every path funnels through one exit block. If the region is not
marked escaped, the young part of the return value is copied above the
region, both bumps are reset to (H0, C0), and it is copied back down
(`rgn_emit_young_copy`, `__rgn_reset_fits`); reclaimed bytes are zeroed
because lowering assumes fresh heap bytes are zero. A scalar result is
checked by `__rgn_exit_scalar`, which keeps the region whenever the result's
bits classify as a young reference.

Eligible: not `main`, not a kernel, extern, GPU or Async function, a return
type that is unit, an integer, bool, f64, or a struct whose slots are all
non-integer word scalars, their fixed arrays, or such structs
(`rgn_copy_cost`; integer-bearing aggregates are refused because declared
integer slots can hold encoded references; f64 slots are copied but their
bits are run through the barrier first). A body that cannot allocate enough
to pay for the epilogue gets `tok = 0` instead of an enter call.

### 2. Escapes are caught dynamically

No static escape analysis is trusted. Every way a young reference can reach
memory older than its region marks the affected regions **escaped**, and an
escaped region never resets (its contents merge into the parent):

| Path | Mechanism |
|---|---|
| field / index / raw-pointer (`*p = v`) / global store | inline pre-filter + `__rgn_barrier(w, t)` before the store (core emitter) |
| `Seq.set`, `Seq.push` (element, and growth repointing the old handle) | barrier inside the builtin body |
| `write_i64` / `write_f64` | barrier inside the builtin body |
| dynlinked extern call | `__rgn_mark_all` in the GOT stub (foreign code may keep any pointer) |
| `syscall6` | `__rgn_mark_all` in the builtin (the kernel may keep a pointer) |
| `__arena_reset` | `__rgn_mark_all` (a hand-moved cursor breaks address order) |
| non-numeric value cast to a numeric type (`p as i64`, `as f64`, `as bool`, ...) | `__rgn_barrier(v, 0)` at the cast (below) |
| scalar slots of a returned struct | exit escape scan runs each slot through the barrier |

`__rgn_barrier(w, t)`: if w is a heap reference born at b and the target was
born at a < b (stack, BSS, 0 and memory outside the heap count as a = 0),
every open region with start s, a < s <= b, is marked escaped. A word that
merely looks like a reference can only retain memory, never free it.

**Casts.** The exit classifier and the barrier look at a word's final value.
A reference cast to an integer and transformed reversibly (`(p as i64) ^
(1 << 62)`) is neither a handle id nor a heap address, yet the caller can
undo the transform and cast back. So the cast itself is the escape point:
every cast whose lowering keeps the source word (identity, wide copy, exact
int -> float) runs `__rgn_barrier(v, 0)` unless the source is *proven*
numeric -- a literal, a numeric cast, arithmetic/comparison of proven
operands, a field or callee whose **declared** type is numeric, or a local
whose declared type is numeric / range induction variable / proven
initialiser (`LOWER_RGN_NUM_EPOCH`; the print-dispatch `scalar_kind` is
deliberately not trusted). An integer can only carry reference bits that
passed through such a cast, so integer arithmetic afterwards needs no hook.

### 3. Activation, opt-out and backends

- Region runtime slots (`__rgn_enter` .. `__rgn_mark_all`, builtin ids 60-69)
  are added to a module's function table on demand, and reserved up front in
  the summary lanes (box and flat), whose consumers freeze the function-id
  table before bodies are lowered. Neither happens at the 16 384-function
  ceiling (ten slots of headroom): such a module opens no region and its
  casts get no hook. Residual: a function of a module at that ceiling that
  runs inside another module's region could publish a reference through an
  unhooked cast; reaching it needs a single module with more than 16 374
  functions.
- Codegen activates the barriers, the extern/syscall mark-all and the core
  emitter routing only when some function actually calls `__rgn_enter`
  (`nc_rgn_bind_module_fns`). A main-only program, or one where no candidate
  was worth a region, compiles exactly as without reclamation.
- `SOUNIO_NO_REGION_RECLAIM=1` lowers with no region slots and no region
  calls (A/B measurement and fallback).
- Region lowering is **opt-in per driver** (`lower_rgn_backend_opt_in`).
  Only Madaros (`compiler/main.sio`), whose only native backend is
  `native/codegen_x86_linux.sio`, opts in. The legacy `native/codegen.sio`
  drivers (module_loader's thin linker, the standalone render/wide drivers)
  do not implement the region runtime, so their IR never contains region
  calls or slots.

### 4. The MIR fallback

The MIR emitter (`compile_ir_function_v2_into`, used by the single-module
streaming lane) carries no barrier pseudo-instructions. In a module where a
region can open, a function with a barrier-guarded store (field, index,
raw-pointer, global) is compiled from IR through the core emitter instead;
every function may run inside a caller's region, so this is not limited to
the functions that open one. Functions without such a store keep the MIR
register allocator. The default whole-module path
(`compile_native_v2_preview_to_file`) already compiles every function
through the core emitter, before and after this change.

`compile_ir_function_v2_into` returns false when a function could not be
emitted and the streaming lane refuses the program
(`streaming_native_v2_codegen_failed:<fn>`); an incomplete body never
reaches an ELF.

## Refusal cases (fail closed)

- Region record stack full: `__rgn_enter` returns token 0, the call runs
  without a region.
- Result too large to copy (`lower_rgn_max_copy_cost`) or the epilogue would
  overflow the per-function instruction cap: no region.
- Second copy could overrun the first (`__rgn_reset_fits`): the region is
  closed without a reset, the first copy is returned.
- Any return shape outside section 1: not eligible.

## Consequences

- Iterative code that builds temporaries inside eligible helpers runs in
  bounded heap and handle space (`madaros_region_reclaim_loop.sio`: 14.4 M
  constructions under the 2^22 table).
- Conservatism only costs memory: a scalar that looks like a reference, an
  unproven cast source, an extern or syscall call all retain regions; none
  frees a live object.
- Every store path in section 2 has a run-pass witness
  (`tests/run-pass/madaros_region_reclaim_*`), and
  `scripts/ci/madaros_region_reclaim_gate.sh` removes each barrier in turn
  (`SOUNIO_RGN_BARRIER_SABOTAGE`) to show its witness then fails, compiles
  the witnesses through the streaming lane and runs them, and checks both
  lanes refuse a function the core emitter cannot emit.
- New store-like IR operations, builtins that write memory, or ways to turn
  a reference into a number must add their barrier and witness here; that is
  the maintenance cost of a dynamic escape check.
- `RuntimeContext` slot 152 (formerly the unused `render_ctx`) now holds the
  innermost region record; the heap arena is 2 GiB minus the context, the
  handle table and the 3 MiB region stack (`nc_heap_arena_bytes`).

## Grounded in

PR #2737 (region reclamation and its review round), the run-pass witnesses
and gate named above, and the full `tests/run-pass` sweep against
`origin/main` recorded on the PR.

<!-- docs:meta
topic_id: repo.docs.decisions.adr-010-madaros-region-reclamation
authority: repo_only
audience: users
last_validated: 2026-11-14
validated_by: Claude Opus 5.5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-010-madaros-region-reclamation
-->

# ADR-010: Madaros Region Reclamation at Function Return and Loop Iteration End

**Status**: experimental
**Date**: 2026-10-06; amended 2026-11-14 (section 5, loop regions and numeric provenance)
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
closed by `__rgn_exit_scalar`, which resets the region unless it escaped
(since 2026-11-14 the result it is handed is not classified: section 5).

Eligible: not `main`, not a kernel, extern, GPU or Async function, a return
type that is unit, an integer, bool, f64, a payload-free enum, or a struct
whose slots are all declared word scalars (integers included since
2026-11-14), their fixed arrays, or such structs (`rgn_copy_cost`). No slot
is scanned for encoded references: section 5. A body that cannot allocate enough
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
| scalar slots of a returned or loop-carried value | none needed: numeric provenance (section 5) |

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

- Region runtime slots (`__rgn_enter` .. `__rgn_mark_all`, builtin ids 60-69;
  `__rgn_reset` 74, `__rgn_settle` 75, `__rgn_copy_payload` 84)
  are added to a module's function table on demand, and reserved up front in
  the summary lanes (box and flat), whose consumers freeze the function-id
  table before bodies are lowered. Neither happens at the 16 384-function
  ceiling (`lower_rgn_fn_headroom()` slots of headroom, 13 since 2026-11-14): such a module opens no region and its
  casts get no hook. Within one function table that is complete: a table
  without the hook has no region either. The default dependency lane lowers
  every module into one shared table (`into_acc`), so it is the single-table
  case. Two lanes lower modules into separate tables and merge them (the
  `SOUNIO_MADAROS_DEP_MERGE=place` lane and `ir_merge_modules_into`, the
  `--probe-machine-ir` route). There an unhooked module could run inside
  another module's region, so the merge vetoes regions
  (`lower_rgn_merge_check`): a side with no `__rgn_barrier` slot and more
  than `IR_MAX_FUNCS - lower_rgn_fn_headroom()` functions is exactly a side
  that may have skipped a hook (below that bound its first unproven cast
  would have added the slot). When one takes part, every `__rgn_enter` of
  the merged program becomes the token 0, which every region path treats as
  "no region". The veto runs after the place lane restores the user `main`
  from its pre-merge snapshot; run before it, the restore brought back 16 of
  the 19 entries the first witness vetoed. The state is durable per compile
  (`lower_merge_state_reset` at the compile boundary), because `lowerer_new()`
  clears the transient hard-error state for every module lowered after a
  merge. A merge that does not fit one table is refused
  (`LOWER_MERGE_FN_OVERFLOWED`, durable for the same reason); the merges used
  to stop at `IR_MAX_FUNCS` and drop the rest silently.
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

### 5. Loop regions and numeric provenance (amendment, 2026-11-14)

**Why.** A function region returns its garbage only when the function
returns, and `main` has none, so a loop in `main` (or any long loop) that
calls helpers keeps every temporary they hand back: 10 M three-word tuples
from one loop died with exit 182 at i = 4.19 M; the GCMC exchange move of
`stdlib/molecular/lj.sio` had been narrowed to a two-word result to stay
under the table.

**Loop regions.** One region spans each eligible loop (`while`, `loop`,
range and array `for`; ir/lower.sio `LOWER_LOOP_RGN_*`). It opens before the
first condition test and is **reset in place** at every iteration end
(`__rgn_reset`: zero `[H0, cursor)`, bumps back to (H0, C0), the region
stays open). The condition's false edge and every `break` share one exit
that pops the region **without** resetting it; `return` pops every open
loop region the same way, innermost first. A region found escaped at an
iteration end is restarted (leave + enter), so the following iterations
reclaim again.

Escape analysis over the body's IR, W = registers the body writes:

- an outer local of declared numeric type: nothing to do;
- an outer local of a type `rgn_copy_cost` accepts (struct, fixed word
  array): **carried** -- at the iteration end its young part is lifted above
  the dead iteration, the region is reset, and it is laid back down at H0
  (the function-exit protocol, with `__rgn_settle` in place of
  `__rgn_leave`); `__rgn_reset_fits` declines exactly as at a function exit;
- any other outer local (an array of structs, a string, an Option, ...):
  region refused (why 2);
- a non-local W register written before the loop (why 3) or read after it
  (why 4, checked when the function is finished): region refused.

Heap stores out of an iteration are barriered as in section 2.

A register is left out of W as a float scalar only when it provably is one:
a local that is a plain scalar (no array, struct or reference, not bound as
an f64-array register), or a temporary not bound as one. The float marker
alone does not say so -- it is also put on the handle of an f64 array
(`var w = a` copies through `emit_array_alloc_like_source`) -- and reading
it as "scalar" let such a local skip the carry decision, so the iteration
reset reclaimed the array. The corpus sweep caught it: `linalg_factorize`,
`stats_poisson_reference`, `stats_poisson_status` and
`stdlib_deep_stats_checked` printed wrong results (L and U all zero in the
LU factorisation) under the first version of loop regions; the escape
witness gained case 9 for it. Float temporaries keep the exemption because
one constant is one register per function, so a constant first materialised
in a loop body is read after it.

**Numeric provenance.** Section 1 refused integer-bearing aggregates
because "declared integer slots can hold encoded references", and the exit
paths ran every f64 / integer word through the barrier. That contradicts
section 2's own conclusion -- an integer can carry reference bits only
through a hooked conversion -- and it is not harmless conservatism: handle
ids are the integers 1..handle_count, so a counter, index or enum tag that
walks into that range classifies as a live young reference and keeps the
region from then on. Measured: a 4 M-iteration loop over
`struct { a: f64, n: i64, c: f64 }` held 402 MB with the scan and 68 kB
without it; a carried 32 KiB struct, 100 k iterations, 3.2 GB -> 180 kB.

Invariant: a word in a slot whose **declared** type is numeric (f64,
integer, bool, or a fixed array of them; `decl_word_scalar` /
`word_scalar_array_len` are set only from the declared field type) carries a
reference's bits only through a word-keeping conversion -- a numeric-target
`as`, `__ptr_of_ref`, an extern or syscall argument -- and each of those
marks the referent's region escaped at the conversion. The hook is
module-wide and bound to the same function-table headroom as every region
opening, so no region opens after an unhooked cast in its module. Region
exits and carried copies therefore scan nothing (`rgn_emit_result_escape_scan`
is gone), and `__rgn_exit_scalar` no longer classifies the result it is
handed. (It is still handed the result: opt_cleanup `ocp_center_frame` and
the HOF-transparency probe in lower.sio read that exact epilogue. The first
version passed a constant 0 instead, which silently disabled the exact
integral-center pass; `sco_exact_integral_center_gate.sh` caught it.) The cast hook now carries
encoded references alone, and the gate proves it: removing its calls
(`SOUNIO_RGN_BARRIER_SABOTAGE=call`) breaks `cast_xor` (1),
`encoded_aggregate` (3) and `encoded_scalar` (2, its Seq case still held by
the seqpush barrier) -- exit codes predicted before the run.

**Block copy.** The young copy of a word array was an element loop; per
element an `IrIndexGet`/`IrIndexSet` pair pays two handle resolves, two
bounds checks and a store barrier, and the protocol copies a carried array
twice per iteration (32 KiB carried, 100 k iterations: 10.9 s). It is now
one `rep movsq` of min(dst, src) payload words (`__rgn_copy_payload`, id 84;
0.54 s). No barrier: the destination is the youngest object, so no store into
it can make older memory point at younger. Reached only after
`__rgn_young` answered 1, i.e. never on arm64, whose stub exits 1.

**Builtin ids.** The dispatch is a flat `builtin_id == N` chain; a shared
id runs the wrong emitter (#2966; the first id picked for
`__rgn_copy_payload`, 76, was `__sounio_fx_install`'s). Region ids must be
dispatched once and returned for no other name; the loop-region gate
checks both, and was shown to catch 76.

**Witnesses and gates.**

- `madaros_loop_region_reclaim.sio`: 10 M tuples (`while`, `for`,
  `loop`/`break`), an integer-carrying struct carried 10 M times, a
  `[i64; 8]` carried 6 M times, and a 513-word f64 array checked element by
  element after 20 k carried copies (a mutant copying one word short exits 6).
- `madaros_loop_region_escape.sio`: nine routes out of an iteration, one
  bit each, printed as `LOOP_REGION_ESCAPE_MASK=<n>` (not the exit status:
  past 255 an exit status drops bit 8, and case 9 alone exited 0 before
  this was changed); `scripts/ci/madaros_loop_region_gate.sh` requires mask
  0 plain, 1 / 2 / 4 with the field / index / storeptr barrier removed, and
  with `SOUNIO_LOOP_RGN_SABOTAGE` 1 (keep refused regions) 128, 2 (exits
  reset) 376, 3 (both) 504, 4 (no carried copy-out) 328; with
  `SOUNIO_NO_REGION_RECLAIM=1` the reclaim witness must die with 182.
- `handle_table_ceiling_gate.sh` and `arena_full_diagnostic_gate.sh` now
  reach their walls with live data (a linked chain; the arena one under
  `ulimit -v`, which the runtime sizes its reservation to) and also require
  the same volume of garbage to complete. Both fail on the pre-loop-region
  compiler.
- Merge veto (section 3). The reclaim witness with one import, in the
  place lane: 0 plain, 182 with `SOUNIO_RGN_FORCE_MERGE_VETO=1` (gate only:
  vetoes at every merge), 0 in the default lane with the same switch (no
  separate-table merge there). The gate requires all three.
- Separate-table merges at the ceiling, measured once (each compile takes
  minutes): a dependency of 9 000 chained functions under a main of 7 375,
  and a dependency of 16 372 under a bare main. Before this change the place
  lane allocated a second slot for every imported function (the dependency
  body was appended and the seed's stub left dead), so the first program
  needed 25 378 slots. The rest were dropped silently. It now refuses with
  the slot count. The stub is now filled in place, so both programs fit
  (16 378 and 16 375 slots) and print 16 375 and 16 372, as the default lane
  does. Both trip the veto on their own (a side above the bound without
  `__rgn_barrier`). The place lane also ran out of instruction regions, which
  are never reused within a compile and the lane holds two tables of. That
  was reported as "invalid handle". The table is now 2 * `IR_MAX_FUNCS` + 256,
  and its exhaustion has its own violation kind.

## Refusal cases (fail closed)

- Region record stack full: `__rgn_enter` returns token 0, the call runs
  without a region.
- Result too large to copy (`lower_rgn_max_copy_cost`) or the epilogue would
  overflow the per-function instruction cap: no region.
- Second copy could overrun the first (`__rgn_reset_fits`): the region is
  closed without a reset, the first copy is returned.
- Any return shape outside section 1: not eligible.
- A loop whose body writes an outer local no copy-out supports, or a
  non-local register that lives across the loop boundary: no loop region
  (section 5, why 2-4); a loop region that escaped restarts instead of
  resetting.
- A separately lowered module that may hold unhooked casts takes part in a
  merge: every region of the merged program is vetoed (section 3).
- A merge needs more than `IR_MAX_FUNCS` slots: the program is refused.

## Consequences

- Iterative code that builds temporaries inside eligible helpers runs in
  bounded heap and handle space (`madaros_region_reclaim_loop.sio`: 14.4 M
  constructions under the 2^22 table).
- Conservatism only costs memory: an unproven cast source, an extern or
  syscall call, a store whose word merely looks like a reference all retain
  regions; none frees a live object. A declared numeric slot is no longer
  such a case (section 5).
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
  innermost region record. The heap arena was 2 GiB minus the context, the
  handle table and the 3 MiB region stack (`nc_heap_arena_bytes`); it is now
  reserved at start-up to RAM + swap, never below 2 GiB, as the arena-full
  diagnostic states.

## Grounded in

PR #2737 (region reclamation and its review round), the run-pass witnesses
and gates named above, and the full `tests/run-pass` sweep against
`origin/main` recorded on the PR. Section 5: the measurements quoted there
(2026-11-14, x86-64 Linux, both compilers built from the same tree with and
without the change) and the corpus sweep: 2 429 of the 2 631 run-pass
programs measured with the final compiler, 138 failures, exactly the 138 of
`tests/madaros_corpus_baseline.txt`, none new and none lost. The first
version of loop regions had four new failures there (the float-marker
defect of section 5). The other 202 programs (the `lorenz_i256_*` family and
others) could not be measured on that machine with either compiler: one
compile needs more than the 6.3 GB its memory cgroup allowed (the compiler
without this change was killed at 6.18 GB on the same program). They need
the larger CI pod the corpus gate names.

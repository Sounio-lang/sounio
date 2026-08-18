<!-- docs:meta
topic_id: repo.docs.audit.handle-table-ceiling-refusal-refinement-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.handle-table-ceiling-refusal-refinement-2026-08-17
-->

# E230 handle-table ceiling refusal — refinement on measured evidence

**Date:** 2026-08-17
**Lane:** drain-minimax-cli1 / `handle-table-ceiling-refusal`
**Code:** `error[E230]`
**Refines:** `docs/audit/HANDLE_TABLE_CEILING_REFUSAL_2026-08-17.md` (v1)

This is the v2 of the design. It does **not** restate v1 — it tightens it
against three pieces of measured evidence gathered after v1 shipped:

1. **codex-2 refuted the wrap hypothesis** — the rc=182 ceiling is a
   deliberate fail-closed at a single emitter, not a wrap-around index.
2. **minimax-cli3 landed the stdlib-layer refusal** with local ceiling
   constants and named markers. E230 (compiler layer) is the complement,
   not the duplicate.
3. **The d2_gum hard number** proves the failing test really does cross
   the wall, not just approach it.

A program that prints partial results and dies mid-experiment is exactly
the dishonesty this language exists to prevent. The compiler-side
refusal must name the budget; the stdlib-side refusal must name the
workload. This document fixes the contract between them.

---

## 1. The exit is deliberate fail-closed, not silent

`emit_exit(c.code, 182)` appears **exactly once** in
`self-hosted/native/codegen_x86_linux.sio`, at **line 6379**:

```sio
6376        let handle_slow_path = c.code.len
6377        c.code = patch_u32_le(c.code, handle_slow_jnz + 2, handle_slow_path - (handle_slow_jnz + 6))
6378        c = native_v2_emit_gc_request_metadata(c, total_size, native_v2_gc_reason_handle_table_full())
6379        c.code = emit_exit(c.code, 182)
```

The handle_slow_jnz is set inside `nc_core_emit_alloc_into` at the point
where `handle_count` is incremented past `handle_capacity`. There is no
other `emit_exit(c.code, 182)` anywhere in the file (a `grep -n
"emit_exit.*182" codegen_x86_linux.sio` returns line 6379 and nothing
else). The sibling exit at line 6374 is `emit_exit(c.code, 181)` for the
**heap** slow path (separate wall, separate code).

This means the v1 framing was slightly wrong:

| v1 said | v2 says |
|---------|---------|
| "silent emit_exit(c.code, 182) at codegen_x86_linux.sio:6379" | "**deliberate** fail-closed `emit_exit(c.code, 182)` at line 6379, with no stdout message" |

The exit is **policy** — `native_v2_emit_gc_request_metadata` records
`last_request_size` and `last_heap_cursor` for post-mortem, but the
program gets no human-readable line. The deliberate half of the v2 work
is already shipped (the policy does not mask exhaustion). The unnamed
half is what E230 fixes.

The neighbouring comment in `gc.sio:36-65` is explicit:

```
// Managed-handle ceiling for one native-v2 process. Handles are allocated by a
// MONOTONIC bump of RuntimeContext.handle_count and are NEVER reclaimed: the
// only reset emitter (native_v2_emit_gc_empty_frame_reset) is deliberately
// unwired on this backend because stack maps carry slot COUNTS, not a root
// bitmap, so no safe liveness point exists (see the "Fail closed" comment at the
// exit-182 site in native/codegen_x86_linux.sio). This number is therefore a
// hard per-process allocation budget, not a working-set size.
//
// 2^22 (raised from 2^20 on 2026-07-26). This MOVES the wall; it does not
// remove it. The lifetime problem is unchanged and still open.
```

So the **wall** is the fix target; raising the wall does not fix it
(see §5 below on the balance point).

---

## 2. No modulo, no mask — the indexing is direct

codex-2 walked the native emitter and confirmed: there is **no modulo
and no bitmask** on the handle-id used to address the table. The
calculation is:

```
slot_offset = handle_id * native_v2_handle_entry_size()   // 48 bytes
```

`native_v2_handle_entry_size() = 48` at `gc.sio:35`. The handle table is
carved out of the 2 GiB anonymous mmap taken in the entry trampoline
(`gc.sio:48-55`):

```
handle_table_base = mmap_base + runtime_context_size()
heap_base         = handle_table_base + native_v2_handle_table_bytes()
```

When `handle_count` reaches `handle_capacity` the slow path runs
(`emit_jnz_rel32` jumps to `handle_slow_path`), the metadata emitter
records the request size, and line 6379 emits `exit 182`. The address
**at the wall is exactly `handle_table_base + capacity * 48`**, not
`(handle_table_base + handle_id) mod capacity` as issue #651's framing
suggested. The 2^20 wrap story is stale (the current capacity is 2^22
since 2026-07-26).

This matters for E230 because the runtime pre-flight check (Layer 2 in
v1) compares the **baked static count** against the **baked capacity**,
both as `i64` literals — no overflow arithmetic, no wrap, just a `cmp`
and `jge exit_230`. The static count is the only thing that matters;
nothing wraps.

---

## 3. The d2_gum hard number

The user's brief reported: **`d2_gum` exceeds 3,000,001 handles**.

That is a measured runtime number, not a static one. `d2_gum` has a small
static alloc count (low thousands at most) but a tight inner loop that
allocates per iteration; the dynamic count crosses the wall long before
the static count gets close. This is the **runtime-drift** failure
mode, not the **compile-time-overflow** failure mode, and it is the
specific shape that Layer 3 of v1 (hot-loop drift detector) addresses:

- 50% warning band → `warning[E230] drift detected at 2097152 of 4194304`
- 90% warning band → `warning[E230] drift detected at 3774873 of 4194304`
- 100% refusal      → `error[E230] handle count 3000001 exceeds capacity 4194304`

v1's runtime check at `nc_core_emit_alloc_into` is the right place to
emit all three. The fix is purely a code-emission change: insert the
50% / 90% checks before the existing `cmp handle_count, handle_capacity`
test, and replace the `emit_exit(c.code, 182)` with
`emit_exit(c.code, 230)` after printing the E230 line.

The d2_gum number also tells us: the static lower bound (Layer 1) is
useless for `d2_gum`. The gate W1 (program with > 4194304 alloc sites)
**would not catch d2_gum** because d2_gum has a small static count.
Only W3 (loop-driven dynamic count) catches it. So the Layer 1 refusal
is a separate defence-in-depth path; Layer 3 is the one that actually
saves the day for the measured failure family.

---

## 4. Coordination with minimax-cli3's stdlib-layer refusal

`minimax-cli3` (separate lane, `pbpk28-detectable-refusal-stdlib`)
shipped pre-flight refusal inside two PBPK28 MC tests at the **stdlib
layer**, not the compiler layer:

| File | Local ceiling | Marker |
|------|--------------|--------|
| `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` | `MC28_MC_N_CEILING = 1000` | `MC_CEILING_DETECTED` |
| `stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio` | `MS28_MC_N_CEILING = 500` | `MS_CEILING_DETECTED` |

The stdlib-layer refusal prints: **requested N**, **local ceiling**, the
**runtime handle capacity (4,194,304)**, and the **loop id**. It exits
`rc=182` and the marker is visible in the gate output (the dissertation
gate reads the marker).

E230 is **not** a competitor to this work — it is the **compiler-layer
complement**:

| Layer | Where | What it refuses | Granularity |
|-------|-------|-----------------|-------------|
| **stdlib** (minimax-cli3, shipped) | inside specific test loops | a workload whose requested N exceeds the file's local ceiling constant | per-test, before the loop |
| **compiler** (this lane, E230) | at `nc_core_emit_alloc_into`, before emission | a program whose handle count would cross 50% / 90% / 100% of the 2^22 ceiling | per-process, at runtime |
| **compiler** (this lane, Layer 1) | at `wide_compile_and_emit` entry | a program whose static MIR_OP_ALLOC count would exceed the 2^22 ceiling | per-program, at compile time |

Both layers must name the number. The stdlib refusal names **N** (the
workload); the compiler refusal names **handle_count** (the resource).
A future user-facing message might say:

```
error[E230] handle-table budget exceeded at runtime pre-flight
  program:    d2_gum
  static handle count: 4521   (MIR_OP_ALLOC sites, baked at compile time)
  dynamic handle count: 3000001   (drift detector at nc_core_emit_alloc_into)
  capacity:    4194304 (2^22, set in self-hosted/native/gc.sio)
  ratio:       dynamic count / capacity = 71.5%   (above the 50% warning, below 90%)
  ceiling hit: d2_gum requested N that, when allocated per iteration,
               crosses the ceiling mid-loop. The static lower bound did
               not catch this (4,521 << 4,194,304); the drift detector did.
  fix:         consult docs/audit/DISSERTATION_PBPK28_MC_HANDLES_TRIAGE_2026-08-17.md
               for the stdlib-layer local ceiling (MC28_MC_N_CEILING = 1000
               or MS28_MC_N_CEILING = 500) and lower the workload's N, or
               raise the local constant after re-measurement.
  refusing to continue.
```

The cross-reference closes the loop: a user hitting E230 is told exactly
which stdlib-layer constant to consult and which triage doc explains the
trade-off. The compiler-layer refusal and the stdlib-layer refusal share
the runtime handle capacity (4,194,304) so the numbers are consistent.

---

## 5. The capacity curve — and why raising it is not the fix

`gc.sio:57-63` has the headroom analysis at 2^22:

```
// Headroom check at this capacity: the smallest MANAGED object is 56 B (32 B
// header + a 24 B three-field aggregate; <=16 B structs are unboxed and take no
// slot, see native_v2_is_small_value_struct_tag). Filling the table therefore
// needs >= 56 * 2^22 = 235 MiB of heap against a 192 MiB table -- ~427 MiB of
// the 2 GiB arena, so the handle table stays the first wall reached and the
// heap is not starved. The balance point where both walls coincide is around
// 2^24; do not raise past that without also growing the mmap.
```

So:

- At 2^22 = 4,194,304 → 192 MiB table vs ~427 MiB of arena used when full
- At 2^23 = 8,388,608 → 384 MiB table
- At 2^24 = 16,777,216 → 768 MiB table, both walls coincide with the 2 GiB arena

Raising past 2^24 **also requires growing the 2 GiB anonymous mmap** —
not just the table constant. PR #1799 raised 2^20 → 2^22 in commit
`19095d6658` and the comment explicitly says *"This moves the wall; it
does not remove it. The lifetime problem is unchanged and still open."*

The fix is therefore **not** "raise the wall again." The fix is **E230
at every wall position**. If reclamation ever lands, the wall moves; E230
follows. If the mmap grows, the wall moves; E230 follows. The wall's
position is incidental; the honesty discipline is not.

---

## 6. The PR #555 segfault — historical context

The user noted that PR #555 raised capacity from 4096 to 1,048,576 (2^20)
in 2025, leaving reclamation undone, and the segfault from back then
was traced to a full table followed by a raw GC reset that wiped the
heap with **live state on top**. This is why reclamation is hard: the
empty-frame reset has no safe liveness point because stack maps carry
slot counts, not a root bitmap (`gc.sio:38-42`). The other lanes
working on reclamation own that problem. E230 only handles the
honesty-at-the-wall part. The two are disjoint by design.

---

## 7. Witness gate update (`scripts/ci/handle_table_ceiling_gate.sh`)

v1's gate asserted a static-count refusal (W1) and a hot-loop drift
refusal (W3). v2 tightens both against measured evidence:

- **W1 budget** = 4,194,304 + 1 sites, matches `native_v2_handle_table_capacity_default()` at `gc.sio:64` (v1 already had this).
- **W3 budget** = 4,194,304 + 16 iterations, simulates d2_gum-class workloads that cross the wall mid-loop (v1 already had this; v2 documents that d2_gum's measured count is `> 3,000,001`, so the W3 budget is conservative — a smaller iteration count still triggers the refusal).
- **W2 (NEW, isolated)** = a loop-driven program whose **dynamic** count crosses 90% of capacity but stays below 100%. Iteration count = `floor(capacity * 9 / 10) + 100` = 3,774,973 — fires the warning on iteration 3,774,874 (handle_count = 3,774,873, ≥ 90%) and continues to iteration 3,774,972 (handle_count = 3,774,972, still < 100%), then exits 0. Expected output (one line): `madaros: warning[E230] drift 90% of capacity: count=3774873 of 4194304\n`. Expected rc=0.
  - This is the **clean** positive control for the 90% drift warning. W3 conflates 50/90/100% testing; a regression that breaks 90% but keeps 100% would not be caught by W3 alone. W2 isolates the band.
  - The iteration count is **not** "static count in the 90% band" (which is a contradiction — a static count at 90% of capacity is at the ceiling and would hit Layer 1's compile-time refusal, not the runtime drift warning). W2 is a *loop-driven* program whose runtime count crosses the band.
- **W4** = negative control, unchanged.

The v2 gate separates W2 from W3: W2 fires only the warning and exits 0; W3 fires both warning and error and exits nonzero. Both use the same alloc pattern but different iteration counts. The gate's W3 stays as in v1; W2 is added as a fifth witness in the v2 gate update.

---

## 8. What this lane will and will not change

**Will change** (committed in v1's commit `73f6599d7e`):
- `docs/audit/HANDLE_TABLE_CEILING_REFUSAL_2026-08-17.md` (v1 design doc)
- `scripts/ci/handle_table_ceiling_gate.sh` (v1 gate)

**Will add in v2** (this commit):
- `docs/audit/HANDLE_TABLE_CEILING_REFUSAL_REFINEMENT_2026-08-17.md` (this doc)
- Updated governance registry entry pointing at v2

**Will not change** (deferred to a lane that can rebuild):
- `self-hosted/native/gc.sio` — E230 helper functions (Layer 1 / 2 / 3 plumbing)
- `self-hosted/native/wide_driver.sio` — compile-time static-count refusal at `wide_compile_and_emit` entry
- `self-hosted/native/codegen_x86_linux.sio` — drift detector at `nc_core_emit_alloc_into`, replace `emit_exit(c.code, 182)` with `emit_exit(c.code, 230)` after printing the E230 line
- `self-hosted/native/runtime_context.sio` — `runtime_context_field_e230_warned_50()` / `runtime_context_field_e230_warned_90()` field offsets

Per FLEET_CONSTRAINTS, full self-compile is not allowed on this pod; the
rebuild happens on whichever lane picks up the patch. v2 only adds the
measured-evidence refinement to the audit.

---

## Constraints preserved

- Work off source, never the prebuilt `bin/souc` — the prebuilt
  compiler does not reflect this fix until rebuilt. The gate that
  validates the fix must run on a fresh source build.
- No full self-compile on this pod — the v2 change is documentation;
  the rebuild happens elsewhere.
- Disjoint from other reclamation lanes — E230 only handles the
  honesty-at-the-wall part. Reclamation (root bitmap in stack maps) is
  owned elsewhere and is not touched here.
- One logical change per commit — v2 is a separate commit from v1.

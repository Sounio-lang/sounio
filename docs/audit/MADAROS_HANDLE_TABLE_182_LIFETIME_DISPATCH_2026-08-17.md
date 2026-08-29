<!-- docs:meta
topic_id: repo.docs.audit.madaros-handle-table-182-lifetime-dispatch-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: fable-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-handle-table-182-lifetime-dispatch-2026-08-17
-->

# Madaros exit-182 "handles full" — the managed-handle lifetime wall (dispatch)

**Date:** 2026-08-17
**Scope:** `bin/souc` default Madaros engine — **native run** exit 182 in the
emitted program after it starts executing. Distinct from the exit-181 heap wall
and from the println/SIGSEGV-139 family
([`MADAROS_PRINTLN_BOOL_SCALARKIND_SEGV_2026-08-17.md`](MADAROS_PRINTLN_BOOL_SCALARKIND_SEGV_2026-08-17.md)).
**Status:** mechanism CONFIRMED with a quantified minimal witness; root cause
KNOWN and OPEN. This is a dispatch (classification + fix directions), not a fix.
**E230:** the proposed *named count/capacity diagnostic* was closed by
refutation on 2026-08-18 — the wall already prints `madaros: handles full`
and exits 182; it does not emit E230 or `count=`. See
[`HANDLE_TABLE_E230_REFUTATION_2026-08-18.md`](HANDLE_TABLE_E230_REFUTATION_2026-08-18.md).
Do not reopen E230 to invent a number for a gate to grep.
**Owner:** Madaros native GC lane (compiler). Merge/authority: codex-2.
**Prior art:** `docs/handoff/repros/handle_table_2pow20_wrap_madaros.sio` (2²⁰
era); commits `a1959e00c2` (Sprint 90 GC activation), `0b6a03c58a` (4096→2²⁰),
`b502b2a171` (unbox ≤16B + fail-closed), `3da75acd31` (2²⁰→2²²).

## 1. Symptom and separation from the 139 family

Five of the ten dissertation "lower_array SIGSEGV 139" tests are **not** 139 on
current main — they are **182**, a different bug: `rapamycin_clinical`,
`gum_vs_mc`, `d2_gum`, `d2_voi`, `rapamycin_pop_sim`. Each **compiles clean,
starts running, prints its banner**, then dies:

```
  D2 Occupancy GUM Budget — Haloperidol CNS-PBPK/PD
  ...
madaros: handles full          <- runtime, mid-execution
# exit 182
```

The `lower_array:` lines in `souc run` output are compiler status; the program
is already built and running when it hits this. This is a **runtime resource
exhaustion**, not a compile failure and not a wrong-science result.

## 2. Mechanism (confirmed)

The native-v2 runtime gives every **managed** (heap/boxed) object a slot in a
fixed **handle table**. From `self-hosted/native/gc.sio`:

- Capacity = `native_v2_handle_table_capacity_default() = 4194304` (**2²²**),
  48 B/slot, carved out of the single 2 GiB process mmap.
- Handles are allocated by a **MONOTONIC bump** of
  `RuntimeContext.handle_count` and are **NEVER reclaimed**.
- On overflow the allocator takes the slow path and **fails closed**:
  `self-hosted/native/codegen_x86_linux.sio:6379` → `emit_exit(182)` with
  `native_v2_gc_reason_handle_table_full()`. (Sibling: `emit_exit(181)` for the
  heap-limit wall.)

The source is explicit that this is a hard **per-process lifetime budget, not a
working-set size**, and that raising 2²⁰→2²² "MOVES the wall; it does not remove
it. The lifetime problem is unchanged and still open."

### Why nothing is reclaimed (the actual root)

The reset emitter **exists** — `native_v2_emit_gc_empty_frame_reset`
(`native/codegen.sio:1574`, `native/codegen_x86_linux.sio:2886`) — but on the
x86_linux backend it is **deliberately not on the allocation slow path**. The
reason: stack maps carry a `root_kind_mask` and slot **counts**
(`native_v2_stack_map_root_kind_mixed/handles_only`, `stack_maps.sio`), **not a
per-slot root bitmap**. Without knowing *which* stack slots hold live handles,
the collector cannot preserve live roots, so it cannot safely reclaim — hence
fail-closed instead of collect.

## 3. Killer evidence — the wall is cumulative lifetime, not live set

Minimal witness: allocate a **>16 B managed** struct, use it, discard it each
iteration (**live set is always 1**), N times.

```sounio
struct S { a: f64, b: f64, c: f64 }   // 24 B > 16 B -> managed (takes a handle)
fn main() -> i32 with IO {
    var acc = 0.0
    var i = 0
    while i < N {
        let s = S { a: 1.0, b: 2.0, c: 3.0 }
        acc = acc + s.a
        i = i + 1
    }
    print("acc="); println(acc); 0
}
```

| N | rc | note |
|---:|---:|---|
| 1,000,000 | 0 | acc=1000000 |
| 4,000,000 | 0 | acc=4000000 |
| **4,194,304 (=2²²)** | **182** | handles full — **exactly at capacity** |
| 5,000,000 | 182 | handles full |

The wall lands on the **cumulative allocation count**, to the exact slot, while
the live set never exceeds one object. A collector that reclaimed the dead `s`
each iteration would run this forever. This is the whole bug.

### The ≤16 B escape (confirms boxing-specificity + gives the workaround)

The same loop with a **16 B** struct (`P { a: f64, b: f64 }`) runs **5,000,000**
iterations at **rc=0**: ≤16 B value structs are unboxed
(`native_v2_is_small_value_struct_tag`) and consume **no** handle. So the wall is
boxing-specific — only managed (>16 B, tags 0/1/2) objects count.

### Why the dissertation tests hit it

`d2_gum` (9-D sensitivity, chronic 3×QD dosing, D2 trough at 72 h),
`rapamycin_pop_sim` (population loop), and the GUM budget tests allocate fresh
**managed** aggregates (`Knowledge<f64>`, `D2GUMPriors`/`D2GUMBudget`, PBPK
states) inside ODE / sampling / per-subject loops. Over a full run these cross
2²² cumulative allocations even though only a handful are ever live.

## 4. Fix directions (ranked)

**C — Source workaround, available NOW (no compiler change).** This is the same
lever the PBPK28 CN mitigation used
([`MADAROS_IMPORTED_PBPK28_CN_SIGSEGV_2026-08-16.md`](MADAROS_IMPORTED_PBPK28_CN_SIGSEGV_2026-08-16.md)):
do not allocate a fresh >16 B managed object per iteration. Either keep the
hot-loop aggregate ≤16 B (unboxed), or mutate a single state in place through
`&!` (`*_step_mut`) instead of returning/binding a fresh one each step. Unblocks
the five dissertation tests without touching the runtime; it is a lifetime
discipline, not a fix.

**B — Escape-scoped frame reclamation (highest-leverage real fix).** The repo
already has an escape analyzer (`self-hosted/analysis/escape.sio`:
`esc_analyze`, `esc_node_escapes`, `esc_mark_returned`) and the reset emitter
(`native_v2_emit_gc_empty_frame_reset`). Wire them: capture `handle_count` at
frame entry (a watermark); at frame return — or a loop back-edge — where escape
analysis proves **no handle allocated in that region escapes** (not returned, not
stored to an outliving object, not captured), reset `handle_count` to the
watermark. This reclaims the nursery-like per-frame allocations that dominate
iterative scientific code (exactly the d2_gum/pop_sim pattern) without a full
tracing collector. Soundness rests on the existing binary-provenance escape work
(PR #397 / E091). Risk: a false "does-not-escape" frees a live handle — so this
must be conservative (default to "escapes") and gated behind a witness suite that
includes a returned-handle positive control.

**A — Precise roots + tracing GC (complete but largest).** Extend stack maps
from `root_kind_mask`+counts to a real per-slot root **bitmap**, then a tracing
collector can mark from precise roots and compact the handle table. This is the
proper closure and what Sprint 90's "precise GC" was heading toward; it removes
the wall for arbitrary live-set programs, not just frame-scoped ones. Largest
surface; do it after B proves the reclamation plumbing.

Raising the ceiling again (2²²→2²⁴) is **not** on this list: the source notes 2²⁴
is where the handle and heap walls coincide, and it does not touch the lifetime
problem — it only postpones the same failure.

## 5. Reproduction

```bash
SOUC=./bin/souc ; export SOUNIO_STDLIB_PATH=$(pwd)/stdlib ; ulimit -s 524288
# dissertation instance:
$SOUC run stdlib/darwin_pbpk/pd/d2_gum.sio ; echo $?      # -> "madaros: handles full", 182
# minimal quantified witness: see §3 (N=4194304 is the exact wall).
```

Build from source before quoting runtime behaviour (`bin/souc` is prebuilt):
`bash scripts/ci/build_modular_madaros.sh <out>` (self-locks); run a specific
build with `MADAROS_RAW_BIN=<out> ./bin/souc run <file>`.

## 6. Non-goals

- Do not conflate 182 with the SIGSEGV-139 println family; they share a triage
  label only.
- Do not claim a ceiling raise fixes it.
- Do not present fix C (source discipline) as closing the compiler bug — it is a
  caller-side avoidance; the runtime still has no reclamation.

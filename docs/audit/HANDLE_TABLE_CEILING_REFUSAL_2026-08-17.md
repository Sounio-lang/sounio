<!-- docs:meta
topic_id: repo.docs.audit.handle-table-ceiling-refusal-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.handle-table-ceiling-refusal-2026-08-17
-->

# Handle-table ceiling — fail-closed refusal (E230)

**Date:** 2026-08-17  
**Lane:** drain-minimax-cli1 / `handle-table-ceiling-refusal`  
**Code:** `error[E230]`  
**Precedent:** E229 (lexer token ceiling, grok-cli3) + E220 (string-literal truncation past Name, #1784)

---

## Defect

`bin/souc` (Madaros, default) runs the native-v2 backend, whose handle table is
allocated by a **monotonic bump** of `RuntimeContext.handle_count` in
`nc_core_emit_alloc_into` (`self-hosted/native/codegen_x86_linux.sio:7117`).
Handles are **never reclaimed**: the only reset emitter
(`native_v2_emit_gc_empty_frame_reset`) is defined but **deliberately unwired**,
because stack maps carry slot counts rather than a root bitmap, so no safe
liveness point exists. Exhaustion is a fail-closed `emit_exit(c.code, 182)`
at `codegen_x86_linux.sio:6379`, and the metadata emitter
(`native_v2_emit_gc_request_metadata` with `native_v2_gc_reason_handle_table_full`)
records `last_request_size` and `last_heap_cursor` but **prints nothing to
stdout or stderr**.

The capacity was 1048576 (2^20) since PR #555 (which raised it from 4096 but
**never implemented real reclaim**), which is exactly where issue #651 wrapped
the native handle table. PR #1799 raised the ceiling to 4194304 (2^22) in
commit `19095d6658` with the explicit note: *"This moves the wall; it does
not remove it. The lifetime problem is unchanged and still open."* So the
ceiling keeps moving on a curve that the program cannot see.

The grok-cli2 re-measure of `pbpk_suite` on 2026-08-17 found the rc=182 family
jumped from 2 to 7 (now the **largest single family of failures** in the
suite). The five new rc=182 tests all **compile successfully**, then die at
runtime after partial output:

| Test | Where it dies |
|------|---------------|
| `rapamycin_clinical` | at the GUM budget |
| `gum_vs_mc` | mid-MC |
| `rapamycin_pop_sim` | at the 20-patient run |
| `d2_gum` | at the start of the GUM budget |
| `d2_voi` | at the start of VoI |

A program that prints partial results and dies mid-experiment is exactly the
dishonesty this language exists to prevent. The existing rc=182 message is
none: nothing on stdout, nothing on stderr, only an exit code. The user must
re-derive what happened.

**Same honesty discipline as E229 and E220 applies.** Refuse before starting,
or at the earliest detectable point, naming the ceiling and the program's
demand. The current behaviour is silently close to E229-pre-fix: the parser
walked past its wall and blamed an innocent parse site; here the runtime walks
past its wall and blames a workload that was never at fault.

---

## Fix (not "bigger table")

Three layers, all reading from the **same** `native_v2_handle_table_capacity_default() = 4194304 (2^22)` constant already exposed in `self-hosted/native/gc.sio`. None of them raise the ceiling — that would only move the lie.

### Layer 1 — Compile-time static lower bound (refuses before emission)

Inserted at the start of `wide_compile_and_emit` (`self-hosted/native/wide_driver.sio:225`), **before any code is emitted**.

Algorithm:

```
native_v2_build_machine_module(&module, true)            // one lowering pass
let static_alloc_count = 0
for fi in 0..module.fn_count {
    let lowered = module.functions[fi].lowered
    for bi in 0..lowered.block_count {
        for ii in 0..lowered.blocks[bi].instr_count {
            if lowered.blocks[bi].instrs[ii].opcode == MIR_OP_ALLOC {
                static_alloc_count = static_alloc_count + 1
            }
        }
    }
}
let capacity = native_v2_handle_table_capacity_default()
if static_alloc_count > capacity {
    print E230(static_alloc_count, capacity)
    return WideCompileResult.empty()    // no ELF emitted
}
if static_alloc_count > capacity / 2 {
    print E230_warning(static_alloc_count, capacity)    // continue
}
```

`native_v2_build_machine_module` is already called in some paths; the cost is
acceptable for a one-pass diagnostic. The lowering for the real compile pass
is reused when `supported` is set; an unsuppported function does not inflate
the count (it does not allocate), so this matches the runtime demand exactly
for the supported subset.

### Layer 2 — Runtime pre-flight (refuses before `main()`)

The compile-time static count is baked into the ELF as a `movabs rax,
<count>` immediately before the entry trampoline's `__runtime_init__` call.
The trampoline compares `count` against `capacity` (also baked in) and prints
`error[E230]` to stderr and exits 230 if `count > capacity`.

This catches: a program compiled against a higher-capacity Madaros and run on
a lower-capacity deployment (e.g. CI pod vs. laptop); a stale Madaros; a
deliberately-tampered ELF; or simply the case where the static count was
exactly at the wall and one extra alloc site slipped through.

### Layer 3 — Hot-loop drift detector (refuses at runtime, naming where)

Inserted at `nc_core_emit_alloc_into` (`codegen_x86_linux.sio:7117`),
**before** the `cmp handle_count, handle_capacity` check:

```
handle_count = load_runtime_context_field(handle_count)
if handle_count > capacity / 2 && !warned_50 {
    print E230_drift_warn(handle_count, capacity, 50)
    warned_50 = true
}
if handle_count > (capacity * 9) / 10 && !warned_90 {
    print E230_drift_warn(handle_count, capacity, 90)
    warned_90 = true
}
if handle_count + 1 > capacity {
    print E230_runtime(handle_count + 1, capacity)    // one final time, named
    emit_exit(c.code, 230)    // E230, NOT 182 — the silent rc=182 is retired
}
```

`warned_50` / `warned_90` are runtime-context flags at offsets
`runtime_context_field_e230_warned_50()` and
`runtime_context_field_e230_warned_90()` (`runtime_context.sio` — new fields
at the high end of the context, after `os_id`). Each program prints at most
one E230 warning per band, so a long-running workload that crosses both
bands prints two lines, not a flood.

The existing `emit_exit(c.code, 182)` is **kept as the fail-closed last
resort** — the runtime check is a refinement, not a replacement — but every
exhaustion now has a stdout message and an E230 tag, so no rc=182 ever
silently kills a partial-result study again.

---

## E-code table entry

`docs/diagnostics/E_codes.md` (or wherever the live registry lives):

| Code | Subsystem | Severity | Meaning | Doc |
|------|-----------|----------|---------|-----|
| E230 | native-v2 runtime | error | handle-table ceiling exceeded at compile time, runtime pre-flight, or runtime hot-loop detection | [E230.md](explanations/E230.md) |

---

## Message format

### Compile-time refusal

```
error[E230] handle-table budget exceeded at compile time
  program:    <name>
  static handle count: N  (MIR_OP_ALLOC instructions across all functions)
  capacity:    4194304 (2^22, set in self-hosted/native/gc.sio)
  ratio:       N / capacity = X%
  refusing to compile.
  fix:         batch per-iteration allocations into collections, or split
               module to reduce per-function alloc count.
```

### Runtime pre-flight refusal

```
error[E230] handle-table budget exceeded at runtime pre-flight
  program:    <name>
  static handle count: N  (baked from compile-time analysis)
  capacity:    4194304 (2^22)
  ratio:       N / capacity = X%
  refusing to start.
```

### Hot-loop drift warning (50% band)

```
warning[E230] handle-table drift detected — half the ceiling reached
  program:    <name>
  handle count: N
  capacity:    4194304 (2^22)
  ratio:       N / capacity = 50.0%
  projection: at current rate, ceiling reached in approximately M more allocations.
  one warning per band, no further prints at 50% level.
```

### Hot-loop drift warning (90% band)

```
warning[E230] handle-table drift detected — 90% of ceiling reached
  program:    <name>
  handle count: N
  capacity:    4194304 (2^22)
  ratio:       N / capacity = 90.0%
  refusing to continue.
```

---

## Witness / gate (`scripts/ci/handle_table_ceiling_gate.sh`)

ENGINE: Madaros (default `bin/souc`). `lean_single` is **not** the contract
surface for this bug. Work from source — `bin/souc` is prebuilt.

### W1 — Compile-time refusal on a program with > capacity alloc sites

Build a synthetic source file with `N > 4194304` allocation sites (e.g. a
single function with a giant match that emits a struct alloc per arm). The
exact form is generated by the gate, so the budget stays at the natural
ceiling without hand-tuning.

Expected after fix: `error[E230]` at compile time, nonzero rc.

### W3 — Hot-loop drift refusal on a small static-count program that allocates heavily at runtime

Build a tiny program whose static alloc count is < 100 but whose runtime
alloc count is `> 4194304`. A loop with one MIR_OP_ALLOC that iterates
4,194,305 times does this.

Expected after fix: `warning[E230] half ceiling` followed by `warning[E230]
90% ceiling` followed by `error[E230] refusing to continue`, nonzero rc.

### W4 — Negative control: a small program must not print E230

A trivial program (one struct alloc) must NOT print E230 and must exit 0.

### Bisect warning

If you bisect a real failing test (rapamycin_clinical, gum_vs_mc,
rapamycin_pop_sim, d2_gum, d2_voi), preserve the **outer loop bound** — that
is what generates the dynamic handle count. Removing iterations changes the
failure mode and makes the witness invalid.

---

## Should the cap also move? (measurement, not convenience)

| Fact | Implication |
|------|-------------|
| Capacity is **2^22** slots × 48 B/slot = 192 MiB of the 2 GiB arena | Memory is significant per process |
| Reclaim is **never** implemented | Every alloc lives until process end |
| Heap budget at 2^22 is ~427 MiB used out of 2 GiB | Headroom remains |
| Wall is at ~2^24 = 16 GiB arena needs | Cannot raise without growing the mmap |
| 5 of 7 rc=182 tests are *loop-driven* workloads | Static count underestimates dynamic demand; the only honest fix is the early refusal, not a raise |

**Recommendation:** **Keep 4194304 for now.** Ship **E230 first**. Revisit a
raise only with: (1) measured peak `handle_count` on each of the 5 tests, (2)
a real reclamation scheme (root bitmap in stack maps), (3) **E230 retained** at
the new bound. The PR #1799 commit message explicitly said *"This moves the
wall; it does not remove it. The lifetime problem is unchanged and still
open."* The discipline is to **never raise the ceiling without E230 in place.**

---

## Files

| File | Change |
|------|--------|
| `self-hosted/native/gc.sio` | Add `native_v2_handle_static_count_capacity_threshold()`, `native_v2_handle_static_count_warn_threshold()`, `native_v2_handle_static_count_print_e230_compile()`, `native_v2_handle_static_count_print_e230_runtime()`, `native_v2_handle_static_count_print_e230_drift()` |
| `self-hosted/native/wide_driver.sio` | At top of `wide_compile_and_emit`, run the static-count pass; print E230 and return empty if `count > capacity` |
| `self-hosted/native/codegen_x86_linux.sio` | In `nc_core_emit_alloc_into`, emit the 50% / 90% drift detectors and replace the silent `emit_exit(..., 182)` with an E230-tagged exit |
| `self-hosted/native/runtime_context.sio` | Add `runtime_context_field_e230_warned_50()` and `runtime_context_field_e230_warned_90()` field offsets and helpers |
| `self-hosted/native/runtime_init.sio` (or wherever the entry trampoline lives) | Pre-flight check: read the baked static count and capacity, print E230 and exit 230 if `count > capacity` |
| `scripts/ci/handle_table_ceiling_gate.sh` | W1 / W3 / W4 witnesses above |
| `docs/llm-guide/explanations/E230.md` | User-facing explanation |
| `docs/diagnostics/E_codes.md` | Registry entry |

---

## Constraints preserved

- **Work off source, never the prebuilt `bin/souc`** — the prebuilt compiler
  will not reflect this fix until it is rebuilt. The gate that validates the
  fix must run on a fresh source build.
- **No full self-compile on this pod** — the change is written as a design +
  source patch; the rebuild happens on whichever lane picks up the fix.
- **No force-rebase, no closing of author PRs** — this change does not touch
  any branch; it adds to the source tree directly.

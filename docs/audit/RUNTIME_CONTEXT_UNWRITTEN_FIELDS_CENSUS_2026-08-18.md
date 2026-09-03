<!-- docs:meta
topic_id: repo.docs.audit.runtime-context-unwritten-fields-census-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.runtime-context-unwritten-fields-census-2026-08-18
-->

# Census — runtime state that is declared, readable, and never written

**Date:** 2026-08-18  
**Lane:** grok-cli2 / `runtime-field-census`  
**Premise:** `#1830` made `pin_count` detectably UNWIRED after it spent hours looking like a live metric (`0` ≡ “zero pins”). The obvious follow-up: **how many other fields are in the same condition?**

**Method:** enumerate every `runtime_context_field_*` accessor and every `native_v2_gc_state_field_*` slot; classify writes on the **live Madaros emitter** (`self-hosted/native/codegen_x86_linux.sio`) versus init-only / dead-path / never-touched. Cross-check `codegen.sio` only as “other backend / vestige.” Grep + call-site reading; no runtime binary census required for *write* classification (writes are compile-time emission).

**Live path definition:** anything reachable from Madaros `main.sio` → `codegen_x86_linux`. `native_v2_emit_gc_empty_frame_reset` is **defined** on x86_linux but **has zero call sites** there (only `codegen.sio` calls it). Writers that exist solely inside that function are **dead on the live path**.

---

## 0. Summary counts (`runtime_context`, 31 fields)

| Class | N | Meaning |
|---|---:|---|
| **A — live-written** | 18 | Updated by trampoline and/or alloc / I/O paths that actually run |
| **B — init-only / sentinel** (read, never live-updated) | 4 | Same honesty class as pre-#1830 `pin_count` |
| **C — never touched** on x86_linux | 3 | Declared layout, zero emit load/store |
| **D — Windows-partial** | 4 | Read by Win builtins; **never stored** even on the Windows trampoline in this file |
| **E — dead-path only** | 2 | Written only inside unwired `empty_frame_reset` |

`pin_count` is already in **B** with sentinel `-1` (#1830). The rest of B–E are still lies-in-waiting.

---

## 1. Full `runtime_context` table

Offset | Field | Live writes (x86_linux) | Reads | Class | Verdict
---:|---|---|---|---|---
0 | `heap_base` | trampoline | alloc / reset | **A** | Live
8 | `heap_cursor` | trampoline + every managed alloc | alloc compare | **A** | Live
16 | `heap_limit` | trampoline | alloc compare | **A** | Live
24 | `handle_table_base` | trampoline | alloc table index | **A** | Live
32 | `handle_count` | trampoline + every managed alloc | alloc compare; mirrored to gc_state | **A** | Live (the 182 wall)
40 | `handle_capacity` | trampoline (2²²) | alloc compare | **A** | Live
48 | `descriptor_table` | trampoline data ptr | metadata | **A** | Live when descriptors embedded
56 | `descriptor_count` | trampoline | metadata | **A** | Live
64 | `pin_registry` | **imm 0 only** | mirrored to gc_state on 182 | **B** | **Dead surface** — null pointer forever; same class as pin_count before #1830
72 | `pin_count` | **UNWIRED sentinel −1 only** (#1830) | mirrored to gc_state | **B** | **Closed** — detectably absent
80 | `argc` | trampoline (Linux argv; Win 0) | builtins | **A** | Live
88 | `argv_ptr` | trampoline | builtins | **A** | Live
96 | `stdout_fd` | trampoline (=1) | print path | **A** | Live
104 | `stderr_fd` | trampoline (=2) | fail diagnostics | **A** | Live
112 | `simd_capability_mask` | AVX detect | (mostly write) | **A** | Live
120 | `gc_state` | trampoline data ptr | slow-path metadata | **A** | Live pointer; *contents* partly dead (below)
128 | `stack_maps` | trampoline data ptr | mirrored | **A** | Pointer live; maps are coarse (see honesty gate)
136 | `deopt_state` | trampoline data ptr | mirrored | **A** | Pointer embedded; deopt execution not verified here
144 | `gpu_ctx` | **none** | **none** | **C** | **Never touched** — layout reserved for heterogeneous GPU
152 | `render_ctx` | **none** | **none** | **C** | **Never touched** — layout reserved for render
160 | `provenance` | **none** | **none** | **C** | **Never touched** — name appears only in policy *group labels*, not this field
168 | `collector_epoch` | init 0; **increment only in `empty_frame_reset`** | mirrored on 182 | **E** | **Dead path** — reset never called on x86_linux → always 0 if read
176 | `last_gc_stats` | init 0; set to 1 **only in `empty_frame_reset`** | none elsewhere | **E** | **Dead path** — always 0 on live Linux runs
184 | `os_id` | Linux trampoline / Win trampoline | Win dispatch | **A** | Live
192 | `win_stdout_handle` | **Win trampoline only** | Win print | **A** (Win) / zero on Linux | Platform-gated; OK if readers check `os_id`
200 | `win_stderr_handle` | **Win trampoline only** | (limited) | **A** (Win) | Same
208 | `win_write_fn_ptr` | **Win trampoline only** | Win write builtin | **A** (Win) | Same
216 | `win_read_fn_ptr` | **never stored** | Win read builtin | **D** | **Read, never written** — even Windows trampoline omits it
224 | `win_create_file_fn_ptr` | **never stored** | Win create builtin | **D** | Same
232 | `win_close_handle_fn_ptr` | **never stored** | Win close builtin | **D** | Same
240 | `win_get_file_attrs_fn_ptr` | **never stored** | Win attrs builtin | **D** | Same

### Class B detail (pin-shaped)

| Field | Init | Live update? | Advertised? |
|---|---|---|---|
| `pin_registry` | `0` (null) | No | Was bundled under `pin_registry_ready: true` — now `false` (#1830) |
| `pin_count` | `-1` UNWIRED | No | Closed by #1830 |

`pin_registry` is still a **plausible null** (looks like “no registry”) rather than an explicit UNWIRED sentinel. Lower urgency than pin_count (null vs zero-pins), but same structural class: **readable, never populated**.

### Class C detail (never touched)

`gpu_ctx`, `render_ctx`, `provenance` — pure layout reservations. No load, no store on x86_linux. Safer than pin_count (nothing reads them yet), but they are **landmines** the moment a probe or future backend starts reading zeros.

### Class D detail (Windows holes)

The Windows trampoline stores stdout/stderr handles and `WriteFile` only. Four IAT slots used by emitted Win builtins are **never filled**. On Linux those builtins should be unreachable; on Windows a call would jump through **NULL**. That is not a silent-zero *metric* lie — it is a **latent SEGV** if the path is taken.

### Class E detail (writers exist but are unreachable)

`collector_epoch` and `last_gc_stats` are updated inside `native_v2_emit_gc_empty_frame_reset`, which x86_linux **defines and never calls** (fail-closed 182 instead). Any probe reading epoch=0 / stats=0 after an 182 is reading **absence of collection**, not “zero collections completed after a successful GC.”

---

## 2. `gc_state` fields (embedded block)

Mirrored from context or written on slow path / dead reset:

| Field | Live write? | Class |
|---|---|---|
| `magic` / `version` / `flags` | embedded init + slow-path flags | **A** (partial) |
| `collection_requests` | **yes** — incremented on every 181/182 metadata emit | **A** |
| `collections_completed` | **only** `empty_frame_reset` | **E** — always 0 on live Linux |
| `compaction_count` | **only** `empty_frame_reset` | **E** — always 0 |
| `last_request_size` / `last_reason` / `last_code_offset` | yes on 181/182 | **A** |
| `last_heap_cursor` / `last_heap_limit` | mirrored on 181/182 | **A** |
| `handle_count` / `handle_capacity` | mirrored on 181/182 | **A** |
| `pin_registry_offset` / `pin_count` | mirrored from context (null / −1) | **B** |
| `collector_epoch` | mirrored; context never live-increments | **E/B** |
| `stack_maps_offset` / `deopt_state_offset` / descriptor fields | trampoline / mirror | **A** (pointers) |
| `last_safepoint_kind` | check emit sites… | see note |

`last_safepoint_kind`: grep shows it is part of layout; live emitters set related safepoint metadata in stack maps, but the gc_state field itself is primarily cleared/mirrored — treat as **low-confidence; do not claim without a dedicated write audit** if someone builds a probe on it.

---

## 3. Related honesty surface (not a field, but the same class)

Contract JSON still advertises **models** as ready while the live path is fail-closed:

| Advertisement | Value today | Reality on x86_linux |
|---|---|---|
| `pin_registry_ready` | **false** (#1830) | Correct |
| `ffi_pinning_model` | **false** (#1830) | Correct |
| `precise_stack_maps` | **false** (honesty gate) | Correct — coarse roots |
| `tracing_gc` | **true** | **Lie** — no tracing collector runs; bump + fail-closed |
| `gc_mark_compact_model` | **true** | **Lie** — model exists in `gc.sio`, not driven by emitter |
| `gc_precise_descriptor_scanning` | **true** | **Lie** as a *runtime* claim |
| `gc_handle_relocation_model` | **true** | **Lie** as a *runtime* claim |
| `gc_runtime_retry_active` | tied to metadata flag | Retry exists only in `codegen.sio` path, **not** x86_linux 182 path |
| `gc_current_frame_root_scan` | metadata flag | Probe exists; **not** wired into x86_linux 182 fail path |

These are not “unread fields,” but they are **the same defect shape**: a plausible true where the honest value is absence.

---

## 4. Priority for follow-up (do not silently “fix” without owners)

| Priority | Item | Suggested honesty move | Owner hint |
|---|---|---|---|
| P0 done | `pin_count` | sentinel −1 + `pin_registry_ready:false` | #1830 |
| **P1** | `pin_registry` | null → documented UNWIRED / non-zero poison, or drop from mirrors | reclaim / honesty |
| **P1** | Contract lies: `tracing_gc`, `gc_mark_compact_model`, `gc_*_model`, retry/root-scan flags | Flip to `false` + honesty gates (pin / precise_stack_maps pattern) | native honesty |
| **P2** | `collector_epoch`, `last_gc_stats`, `collections_completed`, `compaction_count` | Document DEAD-UNTIL-RESET; or stop mirroring as if collections happened | reclaim (when reset is real) |
| **P2** | Win IAT holes (4 fn ptrs) | Write them on Win trampoline, or fail-closed if null before call | Windows backend |
| **P3** | `gpu_ctx` / `render_ctx` / `provenance` | Leave reserved; do **not** invent zeros in probes; optional UNWIRED if anyone starts reading | heterogeneous |

---

## 5. What this census does **not** claim

- That every Class C field must get a sentinel tomorrow — unread zeros are safer than read zeros, but still landmines.
- That flipping all GC contract `true`s is free — some omega gates assert them; same choreography as #1830.
- That `codegen.sio`’s pin/reset path “counts” as a live writer for Linux Madaros — it does not; Madaros routes through `codegen_x86_linux`.

---

## 6. One-line answer

**Besides `pin_count` (fixed):** at least **`pin_registry`**, **four Windows IAT slots read without writers**, **`collector_epoch` / `last_gc_stats` / `collections_completed` / `compaction_count` (writers only on an unwired reset)**, and **three never-touched context slots** (`gpu_ctx`, `render_ctx`, `provenance`), plus a cluster of **contract capability booleans that still advertise GC/pinning models the live path does not run**. A field that is read and never written is a lie waiting for a reader — `#1830` closed one; this list is the rest of the minefield.

---

## 7. Document control

| Date | Change |
|---|---|
| 2026-08-18 | Initial census of `runtime_context` + `gc_state` unwritten / dead-path fields after #1830. |

<!-- docs:meta
topic_id: repo.docs.handoff.compiler-651-defects-codex-dispatch-2026-07-15
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-651-defects-codex-dispatch-2026-07-15
-->

# Prompt — CODEX-2: fix the two Madaros compiler defects split out of #651

**For:** CODEX-2 (compiler-internals agent; owns `self-hosted/`)
**Authored by:** Claude (hypercomplex canonicalization + #651 root-cause investigation), 2026-07-15
**Type:** self-hosted compiler — native GC / codegen (Defect A) + multimodule thin-link / compact-IR
writer (Defect B). Both are **runtime/codegen** defects on Madaros (the default engine), not math.
**Tracking issues:** **#919** (Defect A), **#921** (Defect B). Full forensic dispatch (on main):
`docs/audit/HYPERCOMPLEX_651_ROOTCAUSE_2026-07-14.md`.

Both were isolated while attacking #651. #651 itself is **resolved as filed** (misdiagnosed — the
exact CD product over ℚ runs correctly on Madaros; proof `tests/run-pass/cd_exact_rational_concrete.sio`).
These two are the real, general-purpose compiler defects that remain. Repros are checked in on main.

---

## Defect A (#919) — native handle-table wraps at 2²⁰ → GC reset corrupts live heap

**Root fix (preferred):** do **not** heap-box ≤16-byte value-struct returns. A `Rational{num:i64,den:i64}`
is a SysV register-pair return — `abi_is_aggregate` already classifies it as register-returnable — so
it should never consume a GC handle. Eliminating the per-call box removes the trigger entirely and is
a large throughput win for every exact-arithmetic inner loop.

**Mechanism:** `self-hosted/native/gc.sio:36` sets `native_v2_handle_table_capacity_default() = 1048576`
(2²⁰). Each struct-returning call (e.g. `rat_add`) allocates one handle. At exactly 2²⁰ allocations the
table fills; the alloc slow-path (`self-hosted/native/codegen_x86_linux.sio:5407–5489`) fires
`gc_reason_handle_table_full`, checks `pin_count` + `current_frame_live_probe`, and — because the probe
under-detects live heap-boxed value-locals — runs `native_v2_emit_gc_empty_frame_reset` and retries,
wiping the heap out from under live data.

**Repro (checked in):** `docs/handoff/repros/handle_table_2pow20_wrap_madaros.sio`
```
./bin/souc compile docs/handoff/repros/handle_table_2pow20_wrap_madaros.sio -o /tmp/w && /tmp/w
# prints acc=2097150/1 (WRONG; = 2×0xFFFFF). Correct is 1048576/1. Threshold is exact at 2²⁰.
```
Scalar accumulator → wrong value (multiple of 0xFFFFF); array-of-struct → SIGSEGV (OOB ptr) — the sole
reason the #651 d8 repro segfaults at N=2048. Same family as the PBPK struct-of-array segfault (PR #555).

**Acceptance:** the demonstrator prints `acc=1048576/1`, **OR** (fallback) handle-table exhaustion
becomes a deterministic non-zero-exit STOP rather than a silent reset that corrupts live data (project
rule: "on overflow, STOP and report"). **Severity B1** — bites any Sounio program exceeding 2²⁰ boxed
value-struct allocations (training loops, large sweeps), independent of hypercomplex algebra.

## Defect B (#921) — multimodule thin-link (compact-IR ELF writer) fails rc=12

**Symptom:** importing `math::rational` alongside a **second** module fails the compact-IR ELF writer,
even though each import alone compiles. Forces single-module inlining (code duplication) for
exact-arithmetic clients.

**Repro (checked in):** `docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio` (8 lines)
```
./bin/souc compile docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio -o /tmp/x
# module_native_driver: compact IR ELF write failed; rc=1
# error: multimodule native thin-link compilation failed   (stable 3/3)
# Removing either import (single module) -> compiles green.
```

**Scope caveat (honest):** this is **not** "any 2-module import fails" — 108 run-pass tests import 2–12
modules and compile; `algebra::cd_sigma` + `algebra::fano` compiles green. The trigger involves
`math::rational` specifically in these pairings; minimal synthetic probes with other second modules
surface a *different* top-line error ("visibility preflight failed"), so the error reporting conflates
phases. Needs your instrumentation to find which module-pair property (symbol count? descriptor volume?
a `rational` type/effect signature?) trips the compact-IR ELF writer. Likely the EISA thin-link /
compact-IR write-capacity family. **Evidence level E1** for the `cd_sigma`+`rational` case.

---

## Protocol

- Both are compiler-owned (`self-hosted/`), per CLAUDE.md §8 — this is a forensic handoff, not a patch.
- Reconcile before refiling: Defect A is the **root cause** of #651's array-of-struct symptom and the
  same family as PR #555; do not refile as a new `[struct;N]` aggregate bug.
- Fixed-point + output-verified gates mandatory for any `self-hosted/` change; the checked-in
  `bin/souc-linux-x86_64` / lean_single serialized surfaces must stay coordinated.
- **Note:** the checked-in `artifacts/self-hosted/madaros` (Jul-11) is stale — build fresh
  (`scripts/ci/build_modular_madaros.sh /tmp/out`, ~2 min) before concluding behavior; a fresh binary
  already fixes some earlier-observed symptoms.

## Out of scope
- Madaros **generic** exact-CD engine (`cd_exact_generic_i64.sio` won't compile on Madaros) — that is
  the separate M1–M3 generics track, not these two defects.

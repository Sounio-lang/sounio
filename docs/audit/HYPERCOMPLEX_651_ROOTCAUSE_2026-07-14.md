<!-- docs:meta
topic_id: repo.docs.audit.hypercomplex-651-rootcause-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.hypercomplex-651-rootcause-2026-07-14
-->

# Issue #651 — true root cause on Madaros (forensic dispatch)

**Date:** 2026-07-14
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default engine — the reference, per user directive)
**Verdict:** The issue as filed is **misdiagnosed for Madaros**. It is **not** a `[struct;N]`
aggregate / struct-temporary codegen corruption. The exact CD product over ℚ — the science the issue
says is blocked — **runs correctly on Madaros today**. Two *distinct* compiler defects were conflated.

This is a **forensic dispatch** (per CLAUDE.md §8: do not patch `self-hosted/` ad hoc). The
compiler-owned fixes below are for CODEX-2.

---

## Headline

The real science target — the canonical sedenion zero-divisor `(e3+e10)(e6−e15)` annihilating
**exactly over ℚ** at k=4 — **is PROVED on Madaros**:

```
RAT-ZD PROVED     ← (e3+e10)(e6−e15) == 0 over Rational, decidable equality
RAT-SQ PASS       ← e1² == −1/1 over Rational
```

A hand-monomorphized concrete `[Rational;16]` Cayley–Dickson multiply (with `cd_sigma` inlined and
only `math::rational` imported) compiles and runs green. The 16×16 loop is **256 iterations** — it
never approaches the failure regime described below. The exact-algebra lane's "last residual" is,
on Madaros, **shippable now**.

Repro of the proof: `docs/handoff/repros/cd_exact_rational_concrete_madaros_ok.sio`.

---

## Why the issue looked like a `[struct;N]` bug — the two conflated defects

### Defect A — native handle-table wrap at 2²⁰ (the d8 SIGSEGV)

`souc`'s **native** GC (`self-hosted/native/gc.sio`) heap-boxes value-struct returns. Each
`rat_add` / `rat_mul` call allocates **one handle**. The handle table has
`native_v2_handle_table_capacity_default() = 1048576` (= 2²⁰) slots.

Measured on Madaros with a flat scalar accumulate loop (`acc = rat_add(acc, one)`, **no array of
struct at all**):

| calls N | result | note |
|---|---|---|
| 1,048,575 | `1048575` | correct |
| 1,048,576 (=2²⁰) | `2097150` | **wrong** = 2 × 0xFFFFF |
| 1,048,577 | `3145725` | wrong = 3 × 0xFFFFF |
| 1,048,600 | `27262950` | wrong = 26 × 0xFFFFF |

The corruption begins **exactly at 2²⁰ allocations** and the garbage is a multiple of
`0xFFFFF = 1048575 = capacity − 1`. This is the handle table filling and the GC firing.

**Mechanism** (`self-hosted/native/codegen_x86_linux.sio:5407–5489`): the alloc fast path guards
`handle_count + 1 >= handle_capacity` and branches to the slow path, which emits
`gc_reason_handle_table_full`, checks `pin_count` + `current_frame_live_probe`, and — if the frame
is **not** detected live — runs `native_v2_emit_gc_empty_frame_reset` and retries. The accumulator
`acc` (and the `[Rational;16]` slot in the array case) is a **heap-boxed value struct that the
liveness probe under-detects**, so the reset wipes the heap out from under live data.

- In a **scalar** accumulator → the dangling handle reads garbage → **wrong value, exit 0**.
- In an **array-of-struct** (`[Rational;N]`, large N) → the corrupted read produces an
  out-of-bounds pointer → **SIGSEGV (exit 139)**. This is the only reason the d8 repro segfaults;
  the array is incidental.

Threshold reproduced with the d8 XOR loop: array `[Rational;1024]`, loop bound **700 → correct**,
**768 → SIGSEGV** (768² × 2 calls ≈ 1.18M > 2²⁰; 700² × 2 ≈ 0.98M < 2²⁰).

**This is the same family as the PBPK struct-of-array segfault** (`project_pbpk_struct_array_segfault`
— "crude GC reset wipes heap while a live handle is held"). Not aggregate codegen.

Controls proving it is neither struct-copy nor high-offset nor loop-count *per se*:
- pass a 16 KB `[Rational;1024]` struct **by value**, no heavy loop → correct;
- **by-ref** heavy loop (no large copies) → still SIGSEGVs → not struct-copy;
- single RMW at high var index 900 (offset 14400 > 8192) → correct → not an offset-immediate bug;
- single loop reading `a.c[i]` / writing `r.c[i]` to i<1024 → correct → not high-index-in-loop.
Only the combination *>2²⁰ struct-returning allocations* corrupts.

### Defect B — multimodule thin-link write failure (rc=12), specific module pairings (issue #921)

A single-module program (`math::rational` only) compiles fine. Adding a **second** module import
can trip the multimodule native thin-link:

```
module_native_driver: compact IR ELF write failed; rc=1
Error: Failed to write native binary rc=12
error: multimodule native thin-link compilation failed
```

This is **not** "any 2-module import fails" — 108 run-pass tests import 2–12 modules, and a
minimal `algebra::cd_sigma` + `algebra::fano` two-module program compiles green. It is a **specific
pairing** sensitivity involving `math::rational`: `cd_sigma`(leaf) + `rational` and
`cayley_dickson` + `rational` both fail the thin-link, while `rational` + `math::octonion` fails a
*different* preflight ("visibility preflight failed"). The reliable workaround is to stay
**single-module** (inline the helper) — a leaf `cd_sigma` module does **NOT** dodge it, because the
client still needs the second `rational` import. EISA thin-link / compact-IR write-capacity family
(`project_eisa_thinlink_codebuffer`); **not** a math or aggregate bug. Needs CODEX-2
characterization of which module-pair property (symbol count? descriptor volume? effect set?) trips
the compact-IR ELF writer.

### Not a defect on Madaros — the "garbage @ N=16" claim

The issue's "N=16 → garbage `4206741/1`" is a **lean_single / fable5 stage2** symptom. On Madaros
the minimal d8 at N=16 prints `c0=1/1` (correct) from the first run. The generic engine
(`cd_exact_generic_i64.sio`) separately **fails to compile** on Madaros (generics gap, tracks M1–M3)
— orthogonal to #651.

---

## Recommended fixes (compiler-owned → CODEX-2)

1. **Do not heap-box value-struct returns.** A `Rational{num:i64,den:i64}` (16 B, two-int) is a
   SysV register-pair return (`abi_is_aggregate` already says ≤16 B fits in registers) — it should
   never consume a handle. Eliminating the per-call box removes Defect A's trigger entirely and is
   a large throughput win for all exact-arithmetic inner loops. **Root fix.**
2. If boxing must remain, **fix `current_frame_live_probe` to detect live value-typed locals**, or
   make handle-table exhaustion an **honest STOP** (`emit_exit` with a dedicated code) rather than a
   silent `gc_empty_frame_reset` that corrupts live data. Matches the project rule "on overflow,
   STOP and report." **Correctness landmine for any >1M-allocation program** (training loops, large
   sweeps), independent of hypercomplex algebra.
3. **Multimodule thin-link (Defect B):** raise the compact-IR ELF write capacity / diagnose rc=12
   under combined-import volume (EISA thin-link family).

## In-lane follow-ups (no compiler change; safe to land now)

- **DONE (2026-07-14, branch `work/sr651-madaros-witness`, `8d98d1b09`):**
  `tests/run-pass/cd_exact_rational_concrete.sio` — the ℚ zero-divisor proof, green on Madaros,
  single-module (cd_sigma inlined). Corrects the sr_mul witness comment's N=2048 overclaim.
- A `cd_sigma` leaf-module extraction was **tried and dropped**: it does NOT dodge Defect B (a
  `leaf + rational` two-module program still fails the thin-link), so it added ineffective code.
  Pure de-dup of the ~10 `cd_sigma` importers remains worthwhile hygiene but is orthogonal to #651
  and should wait until the crowded compiler/heap lane settles.
- Wiring into `scripts/ci/sedenion_zd168_crosscheck_gate.sh` is deferred until the run-pass lands on
  main (the test is globbed by the standard run-pass harness meanwhile).

## Reconciliation with the existing witness (`work/sr651-madaros-witness`)

There was **also** a genuine N=16 defect — struct-value-copy aliasing on aggregate ident-init —
**already fixed** by `ff7afab69` ("value semantics for aggregate ident-init — deep copy, not
alias"). That is why N=16 now prints the correct value on Madaros. Credit where due: two defects
coexisted, and the aliasing fix masked the second one.

But the witness (`tests/run-pass/sr_mul_array_of_struct_651.sio`) **overclaims** that `ff7afab69`
also fixed the N=2048 SIGSEGV, and it only exercises **N=16** (256 allocations). Verified here:
- witness N=16 → `651 PASS` on current Madaros (value semantics OK);
- **N=2048 → still SIGSEGV (exit 139)** on the same binary;
- scalar 2²⁰ accumulate (zero struct-copy) → still corrupts (`2097150`).

So the aliasing fix does **not** address Defect A. The N=2048 failure survives it via the handle
wrap. A regression guard at N=16 can never catch this — a witness that asserts the N=2048 case is
fixed needs to run at N below **and** above 2²⁰ allocations (4-cell discipline). Until Defect A
lands, the honest guard is: N=16 green, large-N documented as the handle-wrap blocker.

## Blocker record (Parallel Blocker Contract — typed dispatch to CODEX-2)

- **Tracking issue:** #919 (Defect A, root cause). **Owner:** CODEX-2 (compiler; `self-hosted/native/`).
- **Blocked lane:** exact Cayley–Dickson product over `Rational`/`BigInt` at scale (>2²⁰ struct
  allocations) — and, transitively, any long-running Sounio program exceeding 2²⁰ heap-boxed
  value-struct allocations (training loops, large sweeps).
- **Reproducible command:** `./bin/souc compile docs/handoff/repros/handle_table_2pow20_wrap_madaros.sio -o /tmp/w && /tmp/w` → prints `acc=2097150/1` (WRONG; correct `1048576/1`). Threshold is exact at 2²⁰.
- **Acceptance condition:** the demonstrator prints `acc=1048576/1` (correct), OR handle-table
  exhaustion becomes a deterministic non-zero-exit STOP rather than silent live-heap corruption.
- **Evidence level:** **E2 (classified)** — isolated to the native handle-table wrap at 2²⁰ via the
  scalar-vs-array, by-value/by-ref, high-offset, and iteration-count controls above; distinct from
  aggregate codegen and from generics.
- **Severity:** B1 (lane-blocking at scale). The 256-iteration science target is unaffected, so the
  exact-algebra lane is NOT stop-the-line.

## Repro assets (this dispatch)

- `docs/handoff/repros/cd_exact_rational_concrete_madaros_ok.sio` — the science, GREEN on Madaros.
- `docs/handoff/repros/d8_generic_struct_F_mul_segv.sio` — existing; segfaults only at large N.
- Scalar 2²⁰ wrap demonstrator: a flat `rat_add` accumulate loop; wrong value at N≥1048576.

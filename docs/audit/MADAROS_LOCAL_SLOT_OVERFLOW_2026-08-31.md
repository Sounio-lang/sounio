<!-- docs:meta
topic_id: repo.docs.audit.madaros-local-slot-overflow-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-local-slot-overflow-2026-08-31
-->

# Madaros: a function with ≳ 23 five-field struct locals miscompiles (segfault or silent wrong values)

**Found:** 2026-08-31, while running `docs/research/sounio/rq4_vanco_two_compartment_flip.sio`
(Paper A RQ4) on the committed prebuilt `bin/souc` → Madaros v0.80.0 (`md5=709acf97`,
tree `3800820b4c`, i.e. `main` at `cd325f66fc`). Owner for the fix: codex-2 (compiler). Tracked as #2318.
This note is a measurement, not a fix.

## Symptom

A single function that binds many locals of a 5-field `f64` struct compiles ("Compilation
successful!") and then the produced ELF either **segfaults** or — worse — **prints wrong
values with exit 0**. No diagnostic. The threshold is by *count of scalar slots*, not by
which operations are used.

## Bisection (same helper set, same inputs; each row is one program)

| program | struct locals in the function | result |
|---|---|---|
| t13 | 19 | exit 0, correct numbers |
| t10 | 21 | exit 0, correct numbers |
| t14 (adds one `af_mul`) | 22 | exit 0, correct |
| t15 (adds `af_mul(&s, &s)`, aliasing) | 22 | exit 0, correct |
| t17 (adds `af_mul`, `af_scale`, `af_scale`) | 24 | **SIGSEGV (139)** |
| t16 (adds four no-op `af_scale`) | 25 | **SIGSEGV (139)** |
| t4c (the same chain inline in `main`, ~25 locals) | 25 | **exit 0, prints `0` — wrong value, no crash** |

So 22 → OK, 24 → crash, and the four extra locals in t16 are *identity scalings* — nothing
numerically new. A second series with ONLY identity scalings pins the count dependence:

| params | 5-field locals | scalar slots (params + 5·locals) | result |
|---|---|---|---|
| 1 | 25 | 126 | exit 0, correct |
| 6 | 25 | 131 | exit 0, correct |
| 6 | 22 | 116 | exit 0, correct |
| 1 | 30 | 151 | **SIGSEGV** |
| 1 | 40 | 201 | **SIGSEGV** |

The real arithmetic chain crashes earlier (6 + 24×5 = 126) than the identity chain (131
passes), so struct-returning calls and borrowed temporaries evidently consume slots of their
own; the exact accounting is the compiler owner's to establish. What is established: a
per-function scalar-slot budget somewhere in the 130–150 range is overrun without a bounds
check, and the overrun is silent in the `main`-inline variant. Ruled out by the same bisection: `&f(&g(x))` call-temporary borrows
(t14/t15 use them and pass), early `return <struct>` (t9 passes), nested struct fields
(t3 passes), `f64 as i64` (t1 passes), field mutation on a `var` struct (t2 passes).

Related but distinct: a *nested* by-value aggregate return (`struct { Aff, Aff, Aff }`)
segfaults even with few locals — that is the known #1692 SRET defect (t4d).

## Minimal reproduction

`tests/run-pass/madaros_local_slot_chain_30.sio` (originally landed as
`tests/known_failures/madaros_local_slot_overflow_probe.sio`; see the correction below) — 30 locals of a 5-field
struct, all identity scalings of one measured value; expected output `4634 / 8046`
(value×10⁴, variance×10⁶); actual: SIGSEGV under the committed Madaros. Cut the chain to 25
`let` bindings and it passes.

```
bin/souc run tests/run-pass/madaros_local_slot_chain_30.sio ; echo $?   # 139 on the stale build only — see the correction
```

## Workaround used in the RQ4 program

Split the chain into stage functions with ≤ 16 struct locals each (`st_cl_*`, `st_root_*`,
`st_phase_*`), one 5-field struct per return, no nested aggregate returns. With that the
5000-patient cohort runs to completion with the algebraic identity check `AUC = A/α + B/β
= 2D/CL` holding to first order at error 0 × 10⁻⁹ — which is also the evidence that the
*math* was never the problem.

## Why this matters beyond RQ4

The silent variant (t4c: wrong value, exit 0) is the dangerous one: a scientific program
crossing the threshold does not fail, it lies. Any `stdlib` or user function with a long
straight-line chain of struct-valued lets is exposed. Suggested fix shape: bounds-check
the local slot table at lowering and fail closed with a diagnostic (an E-code), then raise
the cap; the probe above is the regression test.

## Correction (2026-08-31, same day): the compiler measured above was not the committed one

The binary that produced every row above is `md5=709acf97`. No committed
`bin/madaros-linux-x86_64` has ever had that hash. It is
`/workspace/.wt/fable-1/artifacts/self-hosted/madaros`, a **local build dated 2026-08-16
04:46** from the `lane/fable-1/p0f-ffi-takeover` source of that day (lower.sio and the
native backend were ~10k lines diverged from `main` at the time). `bin/souc` resolves
`artifacts/self-hosted/madaros` *before* the committed ELF, so the provenance line's
`tree=3800820b4c` named the checkout, not the compiler. The same trap as #2315
(8bf30533f4, "two known-failure notes I landed from a stale binary"): a stale local
build, measured as if it were the shipped one.

Re-measured against the compilers that actually exist:

| compiler | chain 30 | chain 100 | chain 400 | inline 100 | mul/add/div chain 60 |
|---|---|---|---|---|---|
| stale local build `709acf97` (2026-08-16) | **SIGSEGV** | SIGSEGV | — | **exit 0, prints nothing** | SIGSEGV (already at 8 rounds) |
| committed prebuilt `02251653` (2026-08-15, *before* that build) | 4634 / 8046 | | | | |
| committed prebuilts `518006cc` (08-17), `a76d6ec5` (08-24), `956d1dc6` (08-29) | 4634 / 8046 | | | | |
| committed prebuilt `bf1fe608` (`main` since #2302) | 4634 / 8046 | 4634 / 8046 | 4634 / 8046 | 4634 / 8046 | 4634 / 8046 |
| fresh source build of `main` 8bf30533f4 (`md5=297f956c`) | 4634 / 8046 | 4634 / 8046 | 4634 / 8046 | 4634 / 8046 | 4634 / 8046 |
| lean_single (`SOUNIO_SOUC_ENGINE=lean_single`) | 4634 / 8046 | 4634 / 8046 | | 4634 / 8046 | 4634 / 8046 |

So the defect was never on `main`: the prebuilt committed the day *before* the stale build
already passes, and nothing since regresses it. On the stale build, gdb puts the fault at
`mov (%rax,%rbx,8),%rax` with `rax=0` in `main` — the struct handle returned by `ph` came
back null, i.e. a lane-only lowering defect, not a slot table.

What `main` actually bounds per function, and how it fails: IR instructions
(`IR_MAX_INSTRS` = 16384, refused at lowering with "function `main` needs 19382 IR
instructions but IR_MAX_INSTRS is 16384; split it into smaller functions" — reached by
the inline shape at 200 structs; no E-code, but not silent) and virtual registers
(`MIR_MAX_FLOAT_SLOTS` = `NC_MAX_VREGS` = 2048, refused as `float_slot_capacity`). The
lowering-side local tables (`LowerLocalStack`, 4096 boxed slots) are not reached by any
of these shapes. There is no 130–150-slot cap to bounds-check; no compiler change was
made for this issue.

The probe was moved out of `tests/known_failures/` and pinned as four `//@ run-pass`
fixtures with `//@ expect-stdout` verdict lines, each of which FAILS on the stale build
(positive control) and passes on the committed, source-built and lean_single engines:
`tests/run-pass/madaros_local_slot_chain_30.sio`, `…_chain_100.sio`,
`…_inline_100.sio`, `…_borrowed_chain_60.sio`.

Lesson, again: `bin/souc --version` prints the ELF path on stderr — read it, and
`rm artifacts/self-hosted/madaros` (or rebuild it) before measuring a "committed" compiler
in a worktree that once built one.

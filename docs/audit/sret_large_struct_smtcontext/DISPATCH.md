<!-- docs:meta
topic_id: repo.docs.audit.sret-large-struct-smtcontext.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sret-large-struct-smtcontext.dispatch
-->

# DISPATCH — struct/array initialisation codegen: a reproducible local-array bug + a non-reproducible large-struct return observation

**Opened:** 2026-05-28
**Class:** souc compiler internals (value initialisation / SRET struct return).
Sibling of `docs/audit/r2_3_compiler_tuple_return_bug/` (same symptom family:
wrong value from struct/array materialisation, no crash).
**Priority:** P2 — neither finding blocks current work; `stdlib/theorem/smt.sio`
ships with a conservative, stable clause budget. The de Grey-scale (dense, ~510-vtx)
χ≥5 path is gated by Finding B.
**Toolchain caveat:** `bin/souc` was rebuilt mid-session (mtime 2026-05-28 23:56,
`md5 79e6e85d2fb4a7aa8787b5bb6a7fd4b3` at time of Finding A). It is a standalone ELF,
not the `artifacts/self-hosted/souc-self-hosted-x86_64` (4.9 MB) binary. Some
observations below may be specific to a particular `bin/souc` build; record the md5
when reproducing.

---

## Finding A — REPRODUCIBLE: local `var` array repeat-literal drops its fill value

**Symptom (deterministic).** A local `var a: [i64; N] = [X; N]` reads back as all
zeros for any nonzero `X`, including an integer literal. `[0; N]` "works" only because
the value already equals zeroed stack memory. Manual element assignment works.

```
z (init [0;4]):  0 0   (expect 0 0)   OK (value is 0)
f (init [5;4]):  0 0   (expect 5 5)   BUG
g (manual =5):   5 5   (expect 5 5)   OK
```

Repro: `repro/array_repeat_init.sio` (deterministic; reproduced across rebuilds).

**Scope / impact.** Struct-field array literals are NOT affected (`Big { data:
[seed; 65536] }` initialises correctly — see Finding B probes). Global `var`
array literals are not affected (e.g. `var ADJ: [i64; 262144] = [0; 262144]`). Only
**local `var`** array repeat-literals with a nonzero fill. Most stdlib code uses
`[0; N]` then fills explicitly, so the bug is largely latent — but any local
`[k; N]` with k≠0 is silently zero. Audit candidates: grep for `= \[[^0]` array
literals in local scope. (`stdlib/theorem/smt.sio` is safe: its only nonzero
repeat-fills, e.g. `lrb_interval: [1; 1024]`, `phase_conf: [0.5; 1024]`,
`lia_var_map: [0 - 1; 64]`, are STRUCT FIELDS, which initialise correctly.)

**Likely cause.** The codegen for a local array repeat-literal emits the slot but
skips the fill loop / `rep stosq` (or computes the fill value and discards it).
Compare against the struct-field path, which is correct, and the global path, which
is correct — the divergence localises the fix.

**Acceptance.** `repro/array_repeat_init.sio` prints `f -> 5 5`; bootstrap
fixed-point + native umbrella gates stay green.

---

## Finding B — NON-REPRODUCIBLE: large `SmtContext` by-value return corruption

**Context.** `SmtContext` (in `stdlib/theorem/smt.sio`) is returned by value from
`smt_new()`. At the committed size (`clause_data: [i64; 16384]`, ≈310 KB) it is
correct. Raising `clause_data` to `[i64; 65536]` (≈800 KB) once produced a spurious
**UNSAT** on a known-SAT instance (even cycle C₄₀₀, 800 vars) in one build of
`examples/erdos/native_sat_scale_demo.sio` (sequence K₁₈/C₈₀/C₈₁/K₇₀/C₄₀₀/C₅₁₁).
Deterministic for that binary (re-run twice), no crash.

**Could NOT be reproduced afterwards.** All of these passed at the 512 KB config:

1. Single 512 KB struct return (`Big { tag, [i64; 65536] }`) — fully initialised.
2. Trailing scalars after a 512 KB array — initialised (rules out "fields after big
   array unwritten").
3. Live 512 KB struct vs 40-deep recursion with 64 KB/frame — struct survives.
4. The exact demo sequence at 512 KB in a minimal source — C₄₀₀ correct (SAT).
5. Re-adding K₇₀ to the committed demo at 512 KB — correct.
6. `ulimit -s` 2048/1024 KB — correct (so NOT a stack overflow; below the struct
   size it simply fails to start).

The non-reproducibility is consistent with (a) the layout-fragility noted in R.2.3
("magnitude shifts stack layout … exposes a latent invariant violation"), and/or
(b) the `bin/souc` rebuild mid-session changing codegen between observations.

**Leading hypothesis.** Same family as R.2.3 — an SRET-pointer/register aliasing
that bites only at certain frame layouts, leaving a control field of the fresh
`SmtContext` (prime suspect `decision_limit`; `smt_search` returns 2 → caller reads
UNSAT when `n_decisions >= decision_limit`) holding stale memory.

**Next steps (gdb session).** (1) Find a deterministic repro first — vary demo
layout / `SmtContext` field order at 512 KB and pin the `bin/souc` md5. (2) `awatch`
`decision_limit`/`score_mode`/`restart_budget`/`regime_label` right after
`smt_new()`. (3) Fix at the SRET emit site, OR — lower risk — **arena/heap-allocate
`SmtContext`** (`self-hosted/collections/arena.sio`) and pass `&!ctx` instead of
returning by value; this removes the 800 KB stack local and the SRET path entirely,
unblocking de Grey-scale clause budgets (≈16384 clauses / 65536 literals).

---

## Constraints

Do not patch `self-hosted/` ad hoc. Any compiler patch MUST clear the bootstrap
fixed-point gate and the native compiler umbrella gate (see R.2.3 §3 Phase C) before
commit. Record the `bin/souc` md5 with every reproduction.

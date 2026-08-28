<!-- docs:meta
topic_id: repo.docs.audit.modular-compiler-audit-2026-06-10
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.modular-compiler-audit-2026-06-10
-->

# Modular Compiler Audit — state re-measured (2026-06-10)

**Scope.** Re-audit of the modular/multi-module compiler path
(`self-hosted/compiler/main.sio` driver + `module_frontend.sio` +
`module_native_driver.sio`) against the standing audit corpus
(`MODULAR_COMPILER_STACK_CLASH_2026-05-29.md`,
`G1_LET_SPINE_CRASH_ROOTCAUSE_2026-06-01.md`,
`MODULAR_PIPE_COVERAGE_MAP_2026-06-01.md`, `docs/audit/g1_wip/*`,
`slurm-jobs/selfhost-lower-oom/DIAGNOSIS.md`). All measurements re-run on this
workspace, branch `feat/assoc-variance-wiring`, 2026-06-10. Subject binaries:

- `artifacts/self-hosted/souc-mc-check.elf` — modular compiler ELF, built 2026-06-07, 87 MB ("mc").
- `artifacts/omega/modular_sret_diag/fresh2_modular_souc.elf` — 2026-06-02 build (stale control).
- `bin/souc` — canonical legacy single-bundle compiler.

Every claim below has a re-runnable command. Scripts preserved under `/tmp/mc_audit/`.

---

## 1. Headline: the G1 `--check` crash class is GONE on the 2026-06-07 binary

The 2026-06-01 root-cause doc measured `mc.elf --check` SIGSEGV on *any*
expression statement (`fn main() { 1 }`) with runaway recursion (~940 MB/s VmStk,
6.5 GB). Re-measured on `souc-mc-check.elf` (8 MB stack, `timeout 20`):

| Program | fresh2 (2026-06-02) | mc (2026-06-07) |
|---|---|---|
| `fn main() {}` | rc=139, 4.6 MB error spam | **rc=0, check: OK** |
| `fn main() { 1 }` | rc=139 | **rc=0** |
| `fn main() { let x = 1 }` | rc=139 | **rc=0** |
| `fn main() with IO { println("hi") }` | rc=139 | **rc=0** |

Command: `bash /tmp/mc_audit/run_audit.sh`. The stale 2026-06-02 binary still
crashes 100% — the standing crash docs describe a state that no longer
reproduces on the current artifact. They should be stamped *superseded as of
the 2026-06-07 build* (root-cause history remains valid).

## 2. `--check` corpus: pass rate 25% → ~37%, crash frontier collapsed to 1

Sample: every 5th file of `tests/run-pass/*.sio` (109 of ~545), `mc --check`,
8 MB stack, 15 s timeout (`bash /tmp/mc_audit/corpus_check.sh`):

```
TOTAL=109  PASS=40 (37%)  FAIL=68  CRASH=1  TIMEOUT=0
```

vs the 2026-06-02 backlog (124/504 = 25% pass, large rc=139 frontier). The
single crasher is `tests/run-pass/slice_fat_pointers.sio` (`&[f64]` / `&![f64]`
slice args — dies mid-typecheck of module 0). **File as a blocker with this
repro.**

### 2.1 Top-4 historical root causes: 3 of 4 FIXED (verified by direct repro)

| 2026-06-02 backlog item (count then) | Status 2026-06-10 | Evidence |
|---|---|---|
| #1 Bridge state-loss E008/E170 (~132) | **FIXED** | `fn f() -> i64 { 42 }` + call → `check: OK` |
| #2 Scientific-notation lexer (~54) | **FIXED** | `let x = 1.0e-30` → `check: OK` |
| #3 E004 literal width (~26) | **OPEN** | `let a: i32 = 5; a + 10` → `error[E004]` |
| #4 E014 `usize` index (~18) | **FIXED** | `a[i as usize] = 7` → `check: OK` |

Command: `bash /tmp/mc_audit/root_bugs.sh`.

### 2.2 Remaining failure histogram (68 fails in the 109-sample)

```
23 OTHER (silent "error: type checking failed", NO error code)
11 E004   8 E009   6 E072   4 E015   3 PARSE   3 E008(residual)
 2 E011   2 E001   1 each: E006 E007 E016 E019 E035 E039   1 SIGNAL139
```

Two actionable reads:

1. **E004 literal-width inference is now the single largest coded bucket** —
   the highest-leverage frontend fix remaining (was #3 in the old backlog).
2. **23/68 failures emit no error code at all** ("Type check failed for module
   0" and nothing else). Whatever the underlying causes (the old backlog's
   silent `&!` borrow-reject class is a candidate), the *diagnostics gap is
   itself a defect*: a third of rejections are unactionable without bisection.

## 3. Legacy `bin/souc` SRET-forwarding family: MOSTLY FIXED, one residual

The 2026-06-02 "known large-SRET miscompile / leave documented" decision
(`SEVEN_CRASHES_DIAGNOSED_2026-06-02.md`, `SRET_FORWARDING_BUG_2026-06-02.md`)
no longer matches the current `bin/souc`:

- `SRET_FORWARDING_MINIMAL_REPRO_2026-06-02.sio` (return-position
  struct-returning call, all three shapes): compiles, runs, prints
  **7.000000** (was 0.0). → pinned green in
  `tests/run-pass/sret_forwarding_minimal.sio`.
- `MODULAR_CROSS_SRET_cd_mul_repro_2026-06-02.sio` (large `CDElement` across
  stdlib import): **CD_MUL_CROSS_SRET_OK**, rc=0 (was FAIL). → pinned green in
  `tests/run-pass/sret_forwarding_cross_module_cd_mul.sio`.
- **Residual (found during promotion, 2026-06-10): forward-in-aggregate
  `return (ctor(), 1)` no longer SIGSEGVs but returns uninitialised memory**
  (`t.0.f0 = 6.95e-310`, `t.1 = 4198812`). The crash escalation is gone; the
  value bug persists for the tuple-wrapped forward. → pinned as
  `//@ known-failure` in `tests/run-pass/sret_forwarding_tuple_aggregate.sio`.

Plausibly landed by `994525a69` / `4aab38cd8` / `63d4cad09` (SRET aliasing +
store codegen fixes on this branch). Harness check on the pins:
`--filter "sret"` → 5 pass / 0 fail / 1 skip (the known-failure).

## 4. Backend paths on mc: v2 works, full-IR `--native-compile` is dead

| Path | Input | Result |
|---|---|---|
| `mc --native-v2-compile` | `fn main() -> i64 { 42 }` | **emits ELF, runs, exit 42** ✓ |
| `mc --native-compile` | same file | **SIGSEGV (rc=139), no ELF** |

This matches `slurm-jobs/selfhost-lower-oom/DIAGNOSIS.md` (2026-06-08/09):
the full-IR path has at least three orthogonal walls — (1) pervasive
gen N-1 large-struct miscompiles in the codegen/writer (`c634b38f` class,
writer fix `d731cc3ce` verified on SLURM but the emitted ELF still SIGBUSes),
(2) by-value `IrModule` merge churn OOM on self-host, (3) `IR_MAX_FUNCS=1400`
cap vs ~6,642 fns in main.sio. **The production multi-module story is
`--native-v2-compile` (source-concat bridge); the full-IR path cannot yield a
working self-host as it stands.**

⚠️ **The five fix commits from that diagnosis (`c8b1843b0` iterative BFS merge,
`fdeedf9c1` cap 8192 + resolver index, `1f5282655` compact re-enable,
`a6a0dc8dc` `_into` SRET hardening, `d731cc3ce` finalize copy-loop extraction)
live only in `/tmp/kw-demote` on branch `claude/kw-demote-module` — none are in
this tree.** A `/tmp` worktree is an eviction-loss risk; land or push them.

## 5. Uncommitted working-tree changes touching the modular path — review flags

### 5.1 `self-hosted/compiler/module_frontend.sio` (+56/−8, uncommitted)

Adds `module_frontend_source_skip_let_or_var_assign`: a **raw source-text
scanner** (byte-comparison of `l`,`e`,`t`,` `…) that skips `let p = ` prefixes
so the compact "summary IR" path recognises `let p = cd_mul(...)`-style calls.
The in-code comment states it exists to make "cross-module cases like the SRET
cd_mul repro" produce non-empty IR. Flags:

1. It extends the **compact path**, which the SLURM diagnosis classified a
   confirmed dead-end (64-entry table, `missing_main` on real programs).
2. A text-pattern heuristic tuned so a specific named repro passes is
   borderline retrofitting (CLAUDE.md principles 6/7) — the repro would go
   green without the underlying lowering being general.
3. No accompanying test; uncommitted.

Recommendation: do not land as-is. Either generalise inside the real parser
path or drop with the compact path.

### 5.2 `self-hosted/ir/egraph.sio` (+73/−5, uncommitted)

New self-test `T83 gpu_thread_rewrite` **prints** a Lean 4 theorem as stdout
text and then prints "Theorem generated and verified in-memory." Nothing is
verified — no Lean elaboration occurs; the test then unions two e-graph nodes
(which always succeeds) and passes. This is an overclaim in test output
(principle 7: plausible-looking output that does not survive forensics is worse
than honest partial output). Also replaces the `total` counter with a hardcoded
`total_tests = 83`. Recommendation: reword the claim (e.g. "Lean obligation
*emitted*, not checked") or pipe the emitted theorem through the real Lean
gate; restore a derived total.

## 6. Coverage-map status (61 broken types)

`MODULAR_PIPE_COVERAGE_MAP_2026-06-01.md` (9 COMPLETE / 19 CHECK_ONLY /
2 FRONT_ONLY / 31 ENUM_ONLY) remains the standing per-type map. Not re-traced
this audit (frontend/backend liveness above was the priority); its "what G1
unlocks" framing should be re-read against the now-working `--check` spine —
the two "cheapest wins" (`TyObservedTransition`, `TyRollbackCertificate`,
declared opcodes lacking builder+lowering) are unchanged in the tree.

## 7. Prioritised next actions

1. **Land/push the `/tmp/kw-demote` series** (or explicitly discard) — highest
   loss-risk item; contains the only verified full-IR emit fix.
2. **E004 literal-width inference** — largest coded `--check` bucket (11/68
   sample; ~26 in the full-corpus census).
3. **Error-code coverage**: make the 23 silent "type checking failed"
   rejections emit codes; then re-histogram (some may be one cause).
4. **File the `slice_fat_pointers.sio` crash** as a blocker (only rc=139 left
   in the sample).
5. ~~Stamp stale docs; promote the SRET repros to `tests/run-pass/`.~~
   **DONE 2026-06-10**: banners added to stack-clash 05-29, G1 06-01,
   SRET_FORWARDING_BUG 06-02, SEVEN_CRASHES (Cluster C) and the corpus
   backlog; three regression pins added (2 green + 1 known-failure for the
   residual aggregate case, §3).
6. **Decide the uncommitted `module_frontend.sio` / `egraph.sio` diffs**
   (§5) — revert or rework before any commit.
7. Re-run the full 504-file backlog workflow on the 2026-06-07 binary to
   refresh `MODULAR_CORPUS_FAILURE_BACKLOG` (this audit's 109-sample is a
   1-in-5 systematic sample, not the full census).

## Honest limits of this audit

- Corpus numbers are from a 1-in-5 systematic sample (109 files), not the full
  545; bucket counts are ±a few, ranking is robust.
- No heavy self-host build was run on this pod (per the concurrency directive);
  full-IR/self-host findings are inherited from the SLURM-verified DIAGNOSIS
  and only the cheap single-module probes were re-run locally.
- `souc-mc-check.elf` provenance (exact source commit) was not established —
  it post-dates the last commits touching `self-hosted/compiler/` on this
  branch; treat "fixed in the 2026-06-07 build" as binary-level evidence, not
  commit-level attribution.

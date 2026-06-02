# Modular `--check` crash count → 0 (2026-06-02)

The whole-arc goal — drive the modular `mc.elf` `--check` SIGSEGV count to zero across the
847-example corpus at the canonical 1 GB stack — is **reached**.

## Final numbers (mc_P, this session)

Sweep of all 847 `examples/**.sio` under `ulimit -s 1048576`, `timeout 15`:

```
rc dist:  144 rc=0   703 rc=1   0 rc=139   0 rc=124
```

**Zero crashes, zero timeouts.** g1 expr-recursion gate PASS (`fn main(){1}` --check rc=0,
VmStk peak 132 kB).

## This session's change — the last 7 SRET-smash stragglers

After the stack-overflow class was resolved by the 1 GB standard stack (222 → 7,
`e529e21f5`), 7 programs still crashed at 1 GB. These were the SRET-smash class: their hot
path ran `var`/assignment RHS exprs through the by-value `check_stmt` → `check_expr`, whose
multi-MB Checker self-copy trips the bin/souc large-struct return-value miscompile.

Fix (all in `self-hosted/check/check.sio`, bin/souc untouched):

1. **StmtVar → `*mut`**: `checker_check_var_stmt_inplace` (transcription of `check_var_stmt`
   @14041; binds mutable, routes the init expr through the `*mut` spine).
2. **StmtAssign → `*mut`**: `checker_check_assign_stmt_inplace` + the E003 mutability check
   `checker_check_assign_mutability_inplace` (transcription of `check_assign_stmt` @14136;
   routes target + value through the spine).
3. Wired both into `checker_check_stmt_inplace` (StmtVar/StmtAssign no longer bridge).

The var/assign migration exposed a **pre-existing gap**: `checker_lower_type_expr_mut` had no
`TypeArray` case, so `var x: [T;N] = …` fell to the silent `_ => note_type_error` default →
spurious E-mismatch (2 pass→fail). Bridging the default to by-value `lower_type_expr`
**re-introduced the SRET-smash** (12 crashes — wrong fix). Correct fix:

4. **`checker_lower_array_type_mut`** — a direct `*mut` `TypeArray` lowering (transcription of
   `lower_array_type` @10651), returning only `TypeEntry` so the recursive frame stays small
   and can never re-enter the by-value smash path. Wired as the `TypeArray` arm; other
   unhandled type-kinds keep the silent-error default (no crash).

## Verification (vs HEAD `e529e21f5`, built binary `.dbg/mc.elf`)

- 7 stragglers: **139 → 1** (crashes fixed).
- 2 programs: **1 → 0** (`conversational_ossm/test_syntax.sio`, `zd_forgettable_training.sio`)
  — these were false-positive type errors; canonical `bin/souc` compiles both rc=0, so the
  modular checker now AGREES with the oracle (strict improvement, not a dropped error).
- **0 non-crash → crash, 0 pass → fail.**

## Arc total

**481 → 0 crashes**: ~259 via the SRET-class `*mut` migration (enum + 13 expr kinds + Call
arg-checker + StmtVar/StmtAssign), ~222 via the canonical 1 GB stack. The heap-tables
structural refactor proposed mid-arc proved **unnecessary** (and advisor-flagged as unsound
via Box-aliasing of the save/restore idioms).

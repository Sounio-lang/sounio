# Layer B3 — postfix `&Type` ref + ref-field load/store (REVERTED)

**Status (2026-04-30):** B3's commit `cd203313` reverted by `d33f2822`
on `main`. WIP code preserved on the worktree branch
`worktree-agent-ac4e2dc69d9617bea` (commit `4c2265cb`) until the
re-land conditions below are met.

## What landed in B3

- Param parsing: added a branch for `&Type` (immutable ref) params
  mirroring the existing `&!Type` (TK_AMPBANG) branch. Used literal
  `141` for TK_AMP because the self-hosted `driver_const_value_tok`
  table doesn't list it.
- `parse_atom_ir` ref-local branch: for `r.field` and `r.field[idx]`
  against ref locals, emit `load_ref_field` (and `local_load` for
  `[idx]`).
- `parse_stmt_ir` field-mutation branch: added an `is_ref_lv` check
  to emit `store_ref_field` when the lvalue is a ref struct local.

## Why reverted

The driver-self-compile gate passed (the driver source itself has no
refinement-type syntax to trigger the bug), but the new stage1 binary
**hangs at 99% CPU** when compiling user programs that use
refinement-type syntax `{ NAME: TYPE | PREDICATE }`.

Confirmed hang on three run-pass files:
- `tests/run-pass/refinement_nested_arithmetic.sio`
- `tests/run-pass/refinement_medical_dose.sio`
- `tests/run-pass/ode_rk4_general.sio`

After revert, all three fail fast with concrete `unsupported_frontend`
errors (correct prior behavior — they're parse-rejected because the
driver doesn't support refinement syntax, but they don't hang).

The bisect: pre-B1 driver fails fast on the same files (correct). B1-only
driver also fails fast. Adding B3 on top → hang. So **B3 is the cause**.

## Likely root cause (hypothesis)

The new `&Type` param branch matches `TK_AMP (kind 141)` followed by
some type expectation. When the param parser instead sees `{` (the
refinement-type opener), the new branch may not advance correctly,
causing the parser to spin. The agent's note about needing literal
`141` instead of a named constant suggests there's a self-resolution
gap that may also be relevant.

## Re-land conditions

Before B3 can be cherry-picked back onto main:

1. **Add a stage1-smoke entry that triggers refinement-type syntax** —
   for example, copy `tests/run-pass/refinement_nested_arithmetic.sio`
   (or a minimal repro) into `examples/native/` and add it to the
   stage1-smoke cohort in `scripts/ci/native_v2_driver_self_compile_gate.sh`.
   This makes the gate fail loudly the next time someone tries to
   land code that hangs on refinement syntax.
2. **Trace the hang in the WIP branch** — instrument the new param
   branch with `print` calls to find which token sequence causes the
   parser not to advance. Likely fix: ensure the `&Type` branch does
   not match when the next token is `{` (refinement opener).
3. **Make the driver self-resolve `TK_AMP`** — add the entry to
   `driver_const_value_tok` so future agents can use the named
   constant safely. This is a separate hardening pass; not strictly
   required for B3 to re-land, but eliminates the gotcha class.

## How to resume

```bash
git checkout worktree-agent-ac4e2dc69d9617bea  # if not deleted
# OR cherry-pick into a fresh branch:
git checkout -b m1_2-layer-b3-retry main
git cherry-pick 4c2265cb
# diagnose with print instrumentation per section above
# re-test against the 3 trigger files before re-landing
```

## What was lost in the revert

The 6-case drop in `kind=183 text=.` punch-list (postfix chained field
access against ref-struct locals). That's recoverable when B3 is
re-landed correctly. The PUNCH_LIST.md was NOT updated to reflect the
6-case closure, so the inventory metric goes back to its pre-cd203313
baseline.

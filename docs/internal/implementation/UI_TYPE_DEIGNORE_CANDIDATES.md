<!-- docs:meta
topic_id: repo.docs.internal.implementation.ui-type-deignore-candidates
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.ui-type-deignore-candidates
-->

# UI Type De-ignore Candidates

## Scope and Method
- Date: 2026-02-28
- Sweep target: `tests/ui/type/*.sio` entries with `//@ ignore`
- Systematic command: `bash scripts/ui_type_deignore_audit.sh` (reads pinned `SOUC_BIN` via `scripts/lib/resolve_souc.sh`)
- Manual validation: reran pinned checks for epistemic candidates with `"$SOUC_BIN" check <file>`

## Findings
- Ignored files scanned: `40`
- Audit-marked safe candidates: `0`
- Manual safety-confirmed de-ignore candidates: `0`

### Why no safe de-ignore now
- Pinned behavior for both epistemic call-boundary tests is currently the same parser failure:
  - `Error: P0003`
  - `Expected a type expression`
- Error patterns were tightened to semantic strings (`confidence bound`, `invalid Knowledge metadata`) so parser-level failures no longer appear as false positives.
- Because pinned still fails before semantic epsilon checks, de-ignoring now would be brittle and misleading.

## Next Candidates
1. `tests/ui/type/epistemic_call_boundary_unknown_provided_epsilon.sio`
   - Current state: exits non-zero with `P0003`, but semantic pattern is intentionally unmet.
   - Next trigger: when pinned parser supports `Knowledge<T, epsilon < V>` syntax and semantic call-boundary checks are reachable.
2. `tests/ui/type/epistemic_call_boundary_invalid_provided_epsilon.sio`
   - Current state: exits non-zero with `P0003`, but semantic pattern `invalid Knowledge metadata` is intentionally unmet.
   - Next trigger: revisit when pinned parser supports `Knowledge<T, epsilon < V>` syntax (or if test intent is deliberately changed to parser-level coverage).
3. `tests/ui/type/unit_mismatch.sio`
   - Current state: fails with resolution errors while `//@ error-pattern: unit` does not match.
   - Next trigger: once pinned unit registry behavior is stabilized, align pattern to the stable diagnostic and re-evaluate de-ignore.

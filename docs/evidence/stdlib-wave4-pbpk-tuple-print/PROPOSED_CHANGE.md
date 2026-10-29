# Proposed checker-only primitive annotation handoff — not implemented

Root architectural direction: retain the positive primitive type already established for a parser-generated projection binding in its existing AST `Stmt.ty`. No lowerer edit, new tuple mask, print workaround or primitive guess is proposed.

## Concrete insertion points

Both checker spines retain the original `Box<StmtList>` at the point needed:

1. In `checker_check_stmts_inplace` (`check/check.sio`, around8066), immediately after `last_ty = checker_check_stmt_inplace(c, (*list).head)`, before `checker_end_statement_loans` and before block-scope pop.
2. In legacy `Checker.check_stmts` (around28204), after obtaining `pair.0` as the updated checker and before recursing to the tail. Keep both paths behaviorally identical.

Do not stamp `last_ty`/`pair.1`: let statements return Unit. The primitive type is the newly checked binding's type in the current environment. `checker_check_let_stmt_inplace` computes `bind_ty`, publishes it with `type_env_bind_mut`, then returns Unit. Do not recheck the RHS, because that can repeat effects, consume affine values or alter loan/constant state.

Intended control flow (interface sketch, not a compilable patch):

```
errors_before := checker.error_count
check original statement once
if errors_before == 0 and checker.error_count == 0:
    if original statement is an unannotated synthetic tuple projection:
        checked_binding_type := current environment's just-bound type
        if checked_binding_type is an exact canonical supported primitive:
            original_node.head.ty := owned equivalent primitive TypeExpr
```

The node written must be the original heap AST node, not the by-value `Stmt` passed into `check_stmt`. Preserve statement/expression spans, name, RHS/target, assignment operation, list links and all tuple producers/fields. Use a managed AST field store so the new `TypeExpr`/path boxes remain owned by that AST; do not bypass the write barrier through a raw cast.

## Positive evidence and exact scope

The first bounded whitelist can be the exact plain i64/f64 types exercised here. Extend other primitives only after their existing annotation consumer and rechecker behavior are qualified. Never derive integer kind from a missing float bit or from native word width.

A `TypeKind` test alone is insufficient. `TypeEntry` contains unit/refinement/ontology/noise and epistemic metadata; stamping bare `f64` must not erase those. Require the canonical bare-primitive shape, or skip the stamp. Error, Unknown, refs/slices, arrays/tuples, functions/capabilities, named aggregates, wide floats and Knowledge are excluded. Do not unwrap them into a guessed primitive.

Parser `tplet_emit_elems` supplies several structural clues: an unannotated StmtLet, RHS numeric FieldAccess, base temporary `__tupN`, and shared generated spans. A prefix is only a scope filter, not type authority. The just-checked environment type is authoritative. All element bindings from one tuple let share a span, so a span-only map cannot distinguish status from arrays. Prefer this direct node update over a new such map.

Same-name shadowing matters if any receiver/type metadata is captured: obtain it before the binding shadows the RHS name. The resulting primitive hint must come from the checked new binding. Do not reconstruct classifier locals with only scalar_kind or replace their Box storage, which can lose existing tuple/reference metadata and locals-count restoration.

## Rechecker and Worker ordering

The stamp should be idempotent: only `ty=None` changes, to a semantically identical canonical annotation. A second check must produce the same bindings, effects, diagnostics, loans/provenance and return type. Existing authored annotations are never overwritten.

Worker source declarations capture the body before checking (check.sio around4981), while `checker_finalize_worker_requests_mut` runs after all bodies (around33815). Confirm that the registered body and the AST being lowered see the same original list nodes. `worker_source_block` checks the RHS class and rejects class<=0 before reading `stmt.ty` (worker_proof.sio around286–293); an annotation must never bypass that evaluation. Preserve final IR reproof and all issuance/origin checks.

The new boxes must survive the actual frontend lifetimes. The observed seed checking path precedes the per-dependency arena mark (module_frontend around6067 versus6217), but all single/imported/specialized paths must be audited. An annotation stored into an older AST from inside a later reset interval needs correct managed ownership/promotion. Do not disable arena reclamation to make it work.

## Required component and integration controls before opening

- Demonstrate an actual original `Box<StmtList>` update visible through the parser program, checker and lowering views. Verify all fields other than the new `ty` are unchanged.
- Double-check the same AST without duplicating RHS execution, effects, errors, loan/linear consumption or const metadata; independently parse/check a fresh copy as control.
- Positive synthetic i64/f64 projection stamps; authored annotations unchanged; unknown/error/enriched/wide/ref/aggregate/capability projections untouched; no stamp after a failing statement or preexisting errors.
- Same-name and nested scopes, multiple binders sharing the span, imported and local declarations, and generic unresolved cases that must remain closed.
- Both checker spines and default single/imported compiler entrypoints.
- The current failing bare-print fixtures should pass on the repaired compiler; float and signed native/C++ controls must remain exact. Opaque aggregate print must remain refused.
- Re-run the original PBPK source unchanged with its1200-second compiler deadline. Its later SIG139 frontier remains a separate result even if print dispatch is repaired.
- Worker source/IR/forgery controls and canonical callbacks must retain their existing results; a type annotation is not a certificate.

There is no demonstrated architectural blocker to this insertion point. Actual Box mutation, canonical primitive construction, second-check invariance and lifetime behavior still need the component proof before an implementation is adopted.

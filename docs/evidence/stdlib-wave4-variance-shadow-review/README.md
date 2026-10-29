# Review proposal: conditional publication of an absent variance shadow

Status: unapplied, uncompiled review patch. No lowerer, compiler producer, root source or witness was edited. The proposed.patch file is a diff against lower.sio SHA-256 87ca50e53e3f6284db6c158eea40637e15a8a4ef6893660e49d18b1ac75bc24c at root 7f994ba7183a78389ce2541a4ee4f07338a9d4b6. SOURCE.json binds the original, in-memory candidate and patch. The patch uses zero context so its required diff-context blank lines are not mistaken for trailing spaces in a documentation file. Static git apply --unidiff-zero --check was performed; it did not apply the patch. Root must use the same --unidiff-zero switch only after source/claim approval. No candidate native or numerical gate is claimed.

## Confirmed source path

lower_expr_variance_non_call_ref returns -1 for an untracked integer parameter. The mutable local var s=seed therefore has no shadow after lower_let_stmt_type_metadata_ref. A later conditional s=1 emits a fresh canonical f64 zero for the literal's variance. lower_assign_stmt_ref currently publishes that RHS register into the shared local map when vslot<0. That compile-time publication outlives the branch, although the register's definition does not dominate later uses.

Root's qualified minimal witness 19737 reproduces this in core: check/compile/native values remain correct in both region modes, while the final-IR auditor reports three uses of the branch-only variance register. This proposal addresses that metadata-publication defect. It does not explain value-register behavior through a fabricated zero or normalize the captured failing IR.

## Positive origin, separate from representation

scalar_kind=1 only says int/bool representation. It is also set for integer Call/MethodCall and Field/Index results. It cannot certify the origin of an Ident. The patch adds a separate boxed 4096-entry untracked_int_origin table and a whole-function fence; it does not hide bits in scalar_kind or reuse region-origin flags.

The positive bit is seeded only for a direct, unqualified builtin word-integer parameter or a declaration whose initializer is already in the closed grammar. Parameter type arguments, references, Knowledge/refinement types, bool/char/f64, wide types and unknown types do not seed it. Declarations capture RHS provenance before binding a same-name local. All three local-slot writers reset/initialize the new field, including direct parameter setup, so slot reuse cannot inherit an old bit.

An Ident is accepted only with positive provenance AND scalar_kind=1 AND no local/base variance, no matching FO row, no reference/global/raw/array/aggregate/Box/Option/tuple/function-pointer/closure or Knowledge metadata. kn_value_float must be exactly -1; legacy unknown -2 is refused. Pending variance/FO/closure facts and FO inline context prevent certification.

Only integer literals, certified Idents, unary negation/bit-not and closed +,-,*,bitwise/shift trees qualify. Calls, field/index access, casts, f64, division/remainder, bool operations and unknown nodes are not certified by type or by a floating-point equality test.

The whole-body scan rejects borrow/dereference, methods, closures, match/handler/for-in/unknown forms and depth/list overflow. It walks statement targets as well as RHS, struct field initializers, arguments and all admitted blocks/branches. The fence is intentionally broader than a binding-specific escape analysis. It prevents a later address/capture from invalidating an earlier loop-prefix assumption.

## Assignment protocol in the patch

1. Existing tracked target: the original variance COPY, detach, sensitivity and scalar-kind path is retained. No residual/row/Hessian cleanup change is included.
2. Absent shadow plus a certified untracked integer RHS: produce the normal canonical zero variance without channels, require empty expression sensitivity/Hessian, and preserve the absent shadow. Do not bind a register or FO row.
3. First real tracking: permit a fresh private cell initialized by COPY of the real RHS only at the binding's own lexical scope, outside every loop, with a plain scalar target and no Hessian. Initialize FO slots through the existing helper. Never bind the RHS register directly.
4. First tracking of an outer binding in a block/loop without a prior cell: explicit refusal. The refusal prints a distinct semantic marker and latches the existing hard-error helper in a valid function context; invalid context does not dereference current_func merely to report an error. Probe errors and FO-inline rollback remain separate and require controls.
5. Assignment never promotes origin 0 to 1. If origin 1 is lost during any loop, latch refusal so an already-emitted prefix cannot be accepted for another iteration. Outside loops, invalidate before subsequent uses. Compound assignments are not certified in this initial patch.

The zero producer runs only for an absent integer target. It does not broaden lower_expr_variance globally, alter float algebra, fold 0*NaN/Inf, create eager scalar shadows, spend FO channels, or discard a Knowledge seed whose variance happens to be zero. SourceFO row/pending guards and the independent provenance bit are required even when representation is integer.

FO inline rejection uses the existing FO_INLINE_FRAME_CLASH escape rather than hard-failing the native value compilation; its rollback/isolation is a required gate. This behavior has not been qualified for the candidate.

## Approval boundary

The foreign v13 claim on lower.sio remains respected. Review the positive-origin signature, conservative scope rule, metadata cloning/reset and diagnostic allocation before any isolated application. The complete matrix in CONTROLS.md is required before a new full compiler bootstrap or elision acceptance. A basic enum-or-pattern syntax probe was checked by the user-required MCP; it is unrelated to this uncompiled patch and is not counted as a candidate gate.

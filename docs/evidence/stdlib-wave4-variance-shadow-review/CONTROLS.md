# Required controls; none executed for this candidate

All actual source/component/native/scientific gates must run under Slurm with exact source/compiler/ELF guards. Numerical references remain C++23 and the original Sounio witnesses. Do not rewrite expected outputs or increase deadlines to admit the patch.

| Group | Required discriminators |
| --- | --- |
| Confirmed minimum | Preserve root19737 core source, seed zero/nonzero, LOOP_REGIONS 0/1, optimize 0/1; C++23 oracle and assert mutant unchanged; final-IR mustdefs has zero missing uses. |
| Closed grammar | Multiple untracked integer parameter aliases and literal +,-,*,bit/shift trees; nested branch then loop; no new FO channels/rows or declaration-wide zero slots. |
| Anti-laundering | Direct and aliased f64/call/method/field/index/cast/unknown/bool/wide/ref/global inputs do not gain origin from scalar_kind=1. Existing sigma/base/FO row and kn tags 0,1,2,-2 refuse certification. |
| Knowledge | Knowledge<f64> and Knowledge<i64> parameters; extracted .value, zero-variance Knowledge with nonempty sensitivity, multi-channel/correlated cases. Old tracked results/IR behavior must remain intact. |
| Scope refusal | First real tracking of an outer binding in one branch or loop must either have a prior dominating private cell or refuse without ELF. No branch-local RHS rebind may be published. |
| Straight-line promotion | First real tracking at the binding's own scope outside loops uses a fresh private COPY cell, not the RHS/parameter register; aliases retain independent cells and original correlations. Hessian-bearing first promotion is explicitly unsupported in this first patch. |
| Loopback | After an earlier closed use, assign an opaque/call/field/Knowledge source to the origin-certified variable within any loop; require refusal even when that assignment follows the use. Assignment cannot re-certify an invalidated origin. |
| Body fence | Future borrow/closure/method/dereference, including in a struct field initializer, target or else arm, disables the proof. Unknown AST forms and depth/list bounds fail closed. |
| Names and reuse | Same-name declaration, nested shadow, sibling branches, slot pop/reuse, function-to-function reuse; newest binding identity and conservative branch invalidation. |
| Storage/constructor | Verify all new field writes through all three local binders and all four Lowerer constructors; first/last 4096-slot entries, invalid index, and object-size/cap64k boundaries. No large inline Lowerer array. |
| Inline/clone context | Deep and copied Lowerer contexts, closure lowering, FO inline rollback: no caller origin mutation survives rejected analysis and no caller whole-body certificate is reused for a callee. |
| Channel budget | 31+ deterministic integer aliases/counters alongside existing real Knowledge channels; no artificial channel creation or correlation loss, and existing 32-channel/128-row policies remain visible. |
| Existing gates | Preserve the existing checker/loans 51/21 and relevant uncertainty/GUM/covariance/Hessian gates. Do not label a new rejection as numerical correctness. |
| WIDE | Original projection and imported program, exact literal/dyadic/hoist cores and seeds; FULL schema accepted before claims; C++23 mustdefs and 14-case oracle, mutant controls, LOOP1 ELISION0/1. Slow61B remains separate and unchanged. |

Additional suspected reset/residual/declaration issues described in RISKS.md need separate repros and are not part of this patch.

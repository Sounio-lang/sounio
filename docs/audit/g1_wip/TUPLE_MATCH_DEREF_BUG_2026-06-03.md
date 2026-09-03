<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.tuple-match-deref-bug-2026-06-03
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.tuple-match-deref-bug-2026-06-03
-->

# Pre-existing canonical-compiler tuple-match codegen bug — root of the Knowledge survivors (2026-06-03)

The Knowledge deeper-crash survivors (epsilon_comparison_valid, knowledge_octonion_inner) reduce
to a PRE-EXISTING canonical-compiler codegen bug — NOT introduced by the nested-write or ontology
fixes (the OLD bin/souc also crashes the repro).

## 15-line build-independent repro (TUPLE_MATCH_DEREF_REPRO_2026-06-03.sio)
```
struct Ty { kind: i64, inner: Option<Box<Ty>> }
fn teq(a: Ty, b: Ty) -> bool {
    match (a.inner, b.inner) {
        (Some(ia), Some(ib)) => teq(*ia, *ib),
        (None, None) => true,
        _ => false
    }
}
fn main() -> i32 {
    let x = Ty { kind: 1, inner: None }
    let y = Ty { kind: 1, inner: None }
    if teq(x, y) { 0 } else { 1 }
}
```
`souc_gen2 teq.sio teq.elf && teq.elf` -> rc=139 (SIGSEGV). OLD bin/souc: also rc=139 (pre-existing).

## Bisected trigger (all cheap standalone compiles)
Crash needs ALL of: (1) `match (a.inner, b.inner)` TUPLE scrutinee, (2) an arm
`(Some(ia), Some(ib))` binding payloads, (3) a use that DEREFS a binding `*ia`. Remove any one
(Some-arm => true; recurse on a,b without deref; single non-tuple match) and it does NOT crash.
The crashing arm is NOT taken (inner=None) — it is reached by WRONG dispatch.

## Root cause (gdb disasm of teq.elf)
The match tests `a.inner`'s discriminant against 1 (Some) and, when it is NOT Some (None=0),
`jne` jumps INTO the Some-arm payload-extraction/deref path (`mov -0x48(%rbp),%rax; mov (%rax),%rax`)
-> derefs the absent Some payload (garbage) -> SIGSEGV. I.e. the tuple-pattern arm dispatch
mis-routes `(None, None)` into the `(Some, Some)` arm. (The deref load is 64-bit; the earlier
"32-bit %eax" reading was a misaligned-disasm artifact.) The page-aligned gdb backtrace was a
corrupted-stack unwind, not stack overflow.

## Scope + fix
Broadly impactful: any `match (x, y) { (Some(a), Some(b)) => …a… }` over enum/Option tuple
elements with payload binding+deref. Fix is in lean_single.sio's match-expression codegen for
NESTED tuple-of-enum patterns (the arm-discriminant test/jump is inverted/mis-targeted) — a full
codegen project (repro -> fix dispatch -> bootstrap fixed point -> run-pass + examples sweeps),
nested-write-scale, for the codegen lane. Closure (approx_propagation, needs Approx effect) and
Seq (seq_borrow, seq_struct_elems) survivors are still undiagnosed (likely separate bugs; may or
may not share this tuple-match root).

## PRECISE LOCALIZATION (2026-06-03)
The match-arm parser in lean_single.sio (the `while` loop at ~20253, "compile match arms")
recognizes pattern starts: `Some(` (tk 57), `None` (58), `Ok(` (59), `Err(` (60), ident/wildcard
(3), literal (4), and or-patterns (`A | B`). There is **NO case for `(` (tk 6) = a TUPLE
pattern**. So an arm `(Some(ia), Some(ib))` is mis-parsed: the leading `(` is not consumed as a
pattern, `disc` stays -1 / `arm_is_tagged` 0, EP advances into the sub-pattern tokens, and the
emitted discriminant test + payload-deref are wrong → mis-route + garbage deref → SIGSEGV.

=> The fix is a real CODEGEN FEATURE: implement tuple patterns in match arms — parse
`( p1 , p2 , … )`, emit a discriminant test per element against the tuple's element offsets,
bind each sub-pattern's payload, AND-combine the per-element tests for the arm. Then revalidate
(bootstrap fixed point + run-pass + examples), nested-write-scale. Best done as a focused effort,
not a tail-of-session partial (a buggy partial would break the bootstrap fixed point).

## Survivor status
- Knowledge (epsilon_comparison_valid, knowledge_octonion_inner): rooted here (types_equal uses
  `match (a.inner, b.inner)`). Fixed by implementing tuple-pattern match arms.
- closure (approx_propagation): UNDIAGNOSED — needs the Approx effect; likely separate.
- Seq (seq_borrow, seq_struct_elems): UNDIAGNOSED — may or may not share the tuple-match root.

## FULL SURVIVOR MAP (2026-06-03, after reducing all 5) — 3 distinct bugs, none a cheap guard
1. Knowledge ×2 (epsilon_comparison_valid, knowledge_octonion_inner): CANONICAL codegen bug —
   tuple patterns in match arms unimplemented (this doc). Repro: TUPLE_MATCH_DEREF_REPRO. Crashes
   OLD bin/souc too. Fix = implement tuple-pattern match arms (feature).
2. Seq ×2 (seq_borrow, seq_struct_elems): MODULAR-CHECKER bug — checker_check_method_call_inplace
   (check.sio:3495) bridges to by-value `(*c).check_method_call_with_base_ty(...)` → copies the 8MB
   Checker per method call; a struct-returning-CALL arg (v.push(mk(1))) adds nested copies → stack
   smash. Minimal repro: SEQ_PUSH_SRET_CALL_REPRO_2026-06-03.sio (v.push(mk(1))). souc_gen2 RUNS the
   program fine → checker-only. Fix = *mut transcription of check_method_call_with_base_ty (like the
   arg-boundary checks), NOT a cheap guard (the method check is needed).
3. Closure ×1 (approx_propagation): MODULAR-CHECKER bug — the in-place expr checker has NO
   ExprClosure case, so a closure literal bridges to by-value check_closure_expr (check.sio:18449)
   → 8MB copy + recursive body check (census's "runaway recursion"). Minimal repro:
   CLOSURE_BODY_CALL_REPRO_2026-06-03.sio (typed closure whose body calls a fn; no Approx/no call
   needed). Fix = add an in-place ExprClosure case (transcribe check_closure_expr to *mut).

=> The ontology guard (165 crashers) was the cheap win. The remaining 5 are 3 separate substantial
fixes (1 canonical feature + 2 *mut transcriptions), each covering only 1-2 programs.

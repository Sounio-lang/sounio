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

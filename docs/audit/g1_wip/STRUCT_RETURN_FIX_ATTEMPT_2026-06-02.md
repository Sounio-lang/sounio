# Body-check dominant crasher — root-cause (machine-level solid, source-level partial) — 2026-06-02

The dominant body-check crasher (131/170 genuine SIGSEGVs) on the modular repro
`fn f(x:i64)->i64{x} fn main()->i64{let y=f(5) 0}` → mc_fixed --check faults at
0x4c2805b. Below is what is PROVEN vs INFERRED, after gdb probing the live mc_fixed
binary (no rebuilds).

## PROVEN (machine level, gdb on mc_fixed)

Disassembling the faulting function's prologue + fault site (probe: `x/45i` back
from 0x4c2805b):
- Frame `sub $0xa4250,%rsp` = **672 KB** — a huge by-value frame.
- It copies 4 incoming arguments (passed as pointers, ABI for large by-value
  structs) into locals via `rep movsq`:
  - arg0 (rdi): **0x51ff qwords ≈ 164 KB** → a struct the size of `Checker` passed
    BY VALUE. (So this is NOT the `*mut Checker` in-place spine — it is a by-value
    Checker function.)
  - arg1 (rsi), arg2 (rdx): **34 qwords = 272 bytes** each (= the size of a
    `TypeEntry`, 29 fields).
  - arg3 (rcx): a **16-byte** struct, manual 2-qword copy → the faulting
    `mov 0x0(%rdx),%rax` with **rdx (arg3 ptr) = −1**.
- So: a **by-value Checker method crashes copying its 4th (16-byte) struct argument
  from address −1.** −1 is the classic `find()`-miss sentinel — a lookup returned −1
  and it is being used as the source ADDRESS of a by-value struct argument, unguarded.

Determinism + clustering (prior commit): 131/170 fault at this exact instruction;
per-binary deterministic. A single shared site rules out layout noise. **This is a
genuine, deterministic, single-site bug — the earlier "intractable/non-bisectable"
framing (imported from project_modular_B_repro_verdict, which described a DIFFERENT
non-deterministic crash) was wrong and is retracted.**

## NOT a struct-RETURN bug

The "struct-return" label (from the census writeup) is corrected: the fault is a
by-value struct **argument** passed from −1, not a return value. The earlier
bootstrap repros (repro/sret_norepro_attempts/) modelled struct-RETURN and all
compiled correctly under ds_fixed2 — they were testing the wrong mechanism.

## INFERRED but NOT confirmed (source level)

A plausible chain — by-value call-arg checking bridges via
`call_expr_should_bridge_by_value` to a by-value boundary checker
`(self: Checker, arg_ty: TypeEntry, param_ty: TypeEntry, span: Span) -> Checker`,
crashing on the `span` (16-byte) argument — but this is **contradicted** by
check.sio:15304's own comment ("ExprCall NEVER sets e.right; ExprIndex always
does"), since for a genuine `ExprCall` the bridge should return false and route to
the working *mut path. So the exact function and the precise reason arg3=−1 are
**unconfirmed**. The arg shape (Checker by value + 2×TypeEntry + 1×16-byte) matches
the by-value `report_*_mismatch` / call-arg-boundary family, but the dispatch path
that reaches it for a plain `f(5)` is not established.

## Why isolated bootstrap repros don't reproduce it

7 progressively-faithful repros (repro/sret_norepro_attempts/) all compile correctly
under ds_fixed2 (the same bootstrap that emits the crashing mc_fixed). The bug only
manifests in the full check.sio codegen context — consistent with a path/dispatch or
arg-passing fault specific to the real by-value function, not reproducible by a small
model of the hypothesised mechanism.

## Concrete next steps (a real fix lane, needs 2:36 rebuilds)

1. **Identify the function**: it is a by-value Checker method (returns Checker, ~672KB
   frame, args = Checker + 2×TypeEntry + 1×16-byte). gdb-instrument or add a
   distinctive marker; or rebuild mc with a symbol table if the toolchain supports it.
2. **Find why arg3 = −1**: which lookup/find returns −1 and is passed as a by-value
   struct-arg address. Likely a missing `if x < 0` guard before constructing/passing
   the argument.
3. **Two candidate fixes**: (a) guard the −1 at the call site; (b) if the dispatch to
   the by-value path is itself wrong for plain calls (the e.right/ExprIndex kind
   mismatch the comment warns about), route to the working *mut path. Either needs a
   modular rebuild + the 504-corpus census to confirm the 131 crashers clear.

The nested-store codegen fix (this branch) is unaffected and remains correct.

## Decision gate (run before opening a rebuild lane): NOT the bare-enum-pattern family

Checked whether the *mut expr-kind dispatch uses bare `ExprCall`/`ExprIndex` match
arms (the documented bin/souc bare-pattern miscompile family that could mis-route a
call to a by-value/index handler):
- **Zero bare `ExprCall`/`ExprIndex`/`ExprBinary`/`ExprUnary` match arms** in check.sio.
- The dispatch (check.sio:2674/2682) already uses **explicit if-equality**
  (`if e.kind == ExprKind::ExprIndex … else if e.kind == ExprKind::ExprCall`), with
  NOTE comments (2591/2634) saying it deliberately avoids `match e.kind` "because
  bin/souc miscompiles" it.

So the dispatch is already hardened against that miscompile → the contradiction is
NOT explained by the bare-pattern family. **The dispatch path that reaches a by-value
function for a plain `f(5)` remains genuinely open.** Without a testable hypothesis,
rebuild-iteration (2:36 each) would be exploratory guessing — so this is handed off
as its own lane, not pursued here.

## Hand-off summary

- **It is NOT a struct-return bug.** Distinct, pre-existing crash: a by-value Checker
  function (672KB frame) deref's a 16-byte struct argument from address −1
  (find()-miss sentinel used as a by-value-arg source, unguarded). 131/170 genuine
  crashers, one instruction (0x4c2805b), deterministic.
- **Open questions for the fix lane:** (1) exact faulting function (arg fingerprint:
  Checker + 2×TypeEntry + 1×16-byte, returns Checker); (2) which lookup yields the −1;
  (3) why a plain `f(5)` reaches a by-value path at all (NOT the bare-pattern family —
  already ruled out). Needs symbol-ful build or marker-instrumented rebuilds.
- **Unrelated to and unblocking-independent of the nested-store fix on this branch**,
  which stands correct + validated.

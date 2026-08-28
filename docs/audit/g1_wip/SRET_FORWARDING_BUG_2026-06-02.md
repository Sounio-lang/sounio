<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.sret-forwarding-bug-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.sret-forwarding-bug-2026-06-02
-->

# bin/souc SRET-forwarding codegen bug — minimal repro (2026-06-02)

> **✅ MOSTLY FIXED as of 2026-06-10** — re-measured in
> [`../MODULAR_COMPILER_AUDIT_2026-06-10.md`](../MODULAR_COMPILER_AUDIT_2026-06-10.md)
> on the current `bin/souc`:
> - Plain return-forwarding (`return ctor()`, `var p = ctor(); return p`,
>   tail): **FIXED** — prints 7.000000. Pinned green in
>   `tests/run-pass/sret_forwarding_minimal.sio`.
> - Cross-module large-struct forward (cd_mul / CDElement): **FIXED** —
>   pinned green in `tests/run-pass/sret_forwarding_cross_module_cd_mul.sio`.
> - **Forward-in-aggregate (`return (ctor(), 1)`): STILL BROKEN** — the
>   SIGSEGV is gone but the tuple now carries uninitialised memory
>   (t.0.f0 = 6.95e-310, t.1 = garbage). Pinned as
>   `//@ known-failure` in `tests/run-pass/sret_forwarding_tuple_aggregate.sio`.
>
> Likely fixed by the SRET/store codegen commits on `feat/assoc-variance-wiring`
> (`994525a69` / `4aab38cd8` / `63d4cad09` — plausible, not proven). The
> gdb-pinned mechanism below remains the reference for the residual aggregate
> case.

**A function that returns the struct result of another struct-returning function returns a
ZEROED struct.** Verified, minimal (6 lines, 1-field struct), size-independent. It provably
explains the olanzapine bug (commit `a3ea42082`). Whether it relates to the G1 lane's E008
work is an **open hypothesis with evidence against it** — see "Relation to E008" below;
treat this as a real, independently-worth-fixing codegen bug, not (yet) "the E008 root."

Repro: `docs/audit/g1_wip/SRET_FORWARDING_MINIMAL_REPRO_2026-06-02.sio`
```
struct S { f0: f64 }
fn ctor() -> S { var p: S = S { f0: 7.0 }; return p }
fn make() -> S { return ctor() }     // forwards a struct-returning call
fn main()       { let s = make(); print_f64(s.f0) }   // prints 0.000000; want 7.0
```
Verified on `bin/souc` md5 `9d4ef541` (the bootstrap compiler), `ulimit -s 1048576`.

## Isolation matrix (exact)
| `make()` body | result | |
|---|---|---|
| `return ctor()` | **0.0** | BUG |
| `var p = ctor(); return p` | **0.0** | BUG (= `olanzapine_pbpk_params_smoker` shape) |
| `let p = ctor(); return p` | **0.0** | BUG |
| `ctor()` (tail) | **0.0** | BUG |
| **`let p = ctor(); <read p in place>`** (no outer return) | **7.0** | **OK (control)** |

- **Size-independent**: triggers at a 1-field (8-byte) struct AND at 44 fields. Not a
  frame-size / large-struct-threshold effect.
- **Not the field write**: the no-reassign copy-and-forward (`var p = ctor(); return p`
  with ctor setting the field) already returns 0. It is the **return-forwarding** of a
  struct-returning call that loses the data.
- **Use-in-place is fine**: `let p = ctor(); p.f0` read in a non-returning caller gives 7.0.
  The bug only manifests when the caller's own return value IS the forwarded struct.

## gdb-PINNED root cause (2026-06-02, runtime addresses)
Disassembly + gdb on the failing repro (`bin/souc 9d4ef541`) pin it precisely — and it is
NOT a "make forgets to copy" bug (the copy instructions ARE emitted). The forwarded
struct-return writes to the callee's OWN LOCAL TEMP instead of the caller-provided sret
destination:

```
main:  lea -0x8(%rbp),%rdi      ; rdi = &s = 0x7fffffffe638  (correct sret dest)
       call make                ; passes 0x...e638
       reads s at 0x...e638      -> 0x0   (still zero)
make:  (entry) rdi = 0x...e638  ; the real sret dest from main
       lea -0x10(%rbp),%rdi     ; rdi = 0x...e600  = make's OWN local temp
       call ctor                ; ctor writes 7.0 to *rdi = 0x...e600  (the temp) ✓
       ; "copy temp->sret": stores 7.0 to r12, but r12 = 0x...e600 (the temp), NOT 0x...e638
```
gdb-confirmed: ctor writes 7.0 to `0x...e600`; make's "sret destination" register at the
copy is `0x...e600` (its own temp), not the `0x...e638` main passed; main reads `s` at
`0x...e638` = 0. **The caller-supplied sret pointer is dropped: make materializes the
forwarded call's result into a local temp and "returns" that temp's address region instead
of writing through main's sret pointer.** Because the doomed address is in make's frame, the
outcome is layout-sensitive — silent zero here, but a different layout can make it an invalid
deref (consistent with the G1 lane's "layout-sensitive sentinel-deref" crash).

**Fix direction (for the codegen owner):** for a return-position struct-returning call,
pass the ENCLOSING function's sret pointer directly as the inner call's destination (no local
temp + copy). The temp+copy path drops/aliases the real sret pointer. This is the call-side
return-forwarding lowering (around `emit_sret_destination_x86`:1765 + how `return <call>`
threads the destination), NOT the verified-correct return-of-local path (7196-7225).

## STOP point (honest, per time-box)
Mechanism is gdb-PINNED above. The actual codegen edit + re-bootstrap (build a candidate
bin/souc, clear repro + 5-working-variants + run-pass corpus + stage2==stage3 fixed point)
is the brick-risky, unbounded step the G1 codegen owner already stopped at, and edits their
live file. This handoff — deterministic repro + disassembly + gdb-pinned wrong-write — is
the high-value deliverable that advances the lane past "reproducer NOT FOUND"; the fix itself
is left to whoever owns the SRET lowering. Did NOT attempt the edit.

## Mechanism (earlier hypothesis — superseded by the gdb finding above)
SRET (struct-return via hidden pointer): the outer fn's hidden return-slot pointer is **not
threaded into the inner struct-returning call**. The inner `ctor` writes into a discarded
temporary (or a fresh stack slot), and the outer fn returns its own untouched (zeroed)
return slot. Use-in-place works because the inner call's result is materialized into a real
local before being read. The fix is in the SRET calling-convention lowering: when a
struct-returning call's result is itself returned (directly, via tail, or via a local that
is then returned), forward the caller's sret pointer to the callee instead of allocating a
throwaway.

## Relation to E008 — HYPOTHESIS, with evidence AGAINST a direct match
Initial guess was "this is the E008 root." A cheap discriminator weakened it; stated
honestly so the G1 lane isn't misdirected:
- **TyUnit discriminant is 3, not 0** (TyI64=0, TyF64=1, TyBool=2, TyUnit=3, check/types.sio:19).
  So a zeroed struct/discriminant reads as **TyI64 (0)**, NOT TyUnit. The clean "zeroed →
  TyUnit → E008 'expected ()'" mechanism therefore does **not** hold — E008's "expected ()"
  is TyUnit(3), which zeroing would not produce.
- **Symptom mismatch**: this bug is **non-crashing** (silent 0.0) and **size-independent**
  (fires at N=1). The G1 lane's *current* blocker is a **crash** ("layout-sensitive sentinel
  deref", commits `cd4377838`/`7f8c4dac8`). Non-crashing + size-independent vs crashing +
  layout-sensitive ⇒ **likely a different (sibling) codegen bug**, not the same one.
- It *might* relate to the earlier `3f2591eeb` "fn_sigs nested-write not persisting" thread
  (a value-persistence flavor), but that link is unproven and the discriminant evidence
  above argues against the simplest mapping. **Do not treat as E008's root without
  demonstrating the E008 path actually forwards a struct-returning call AND that the
  zeroing maps to TyUnit (it doesn't, per the discriminant).**

## What IS established
- **Verified codegen bug**: SRET return-forwarding zeroes the returned struct. Real,
  reproducible, size-independent, worth fixing on its own.
- **Explains olanzapine**: `olanzapine_pbpk_params_smoker` (`var prm = ...(); ...; return
  prm`) was this exact pattern; fixed at source with a constant literal (`a3ea42082`).
- A clean small repro for whoever works the SRET/struct-return lowering — possibly a
  sibling of the G1 codegen issues; useful regardless of the E008 mapping.

## Bug family (16-variant fan-out + adversarial verify, 2026-06-02)
A 19-agent workflow (haiku probes + opus synthesis/verify) mapped the family on
`bin/souc` `9d4ef541`. **11 buggy / 5 working** variants:

- **BUGGY** (zeroed return): `return ctor()`, `var p=ctor();return p`, tail `ctor()`,
  two-level forward, reassign-then-return, **i64 field** (not just f64), read-2nd-field,
  **nested struct** (`s.a.x`), **if-arm** `if b {return ctor()}`, 44-field, and
  **`(ctor(), 1)` tuple-wrapped → SIGSEGV** (the zeroing ESCALATES to a crash when the
  forwarded struct is tuple-wrapped).
- **WORKING**: direct literal return, use-in-place (`let p=ctor(); read p`), return a
  *derived scalar/field* (`let s=ctor(); return s.f0`), **arg-passthrough**
  (`ident(ctor())`), **method self-passthrough** (`ctor().dup()`).

Refinements from adversarial verify:
- **`with Mut` is INCIDENTAL** — the bug fires without it (same miscompile).
- **It needs the forwarding LAYER**: single-level `let s = ctor()` in a non-returning
  caller works; the second function that returns ctor()'s result is what breaks.
- **Dimensions that DON'T matter**: field type (i64 too), struct size (1↔44), nesting
  depth, nested-struct fields, which field is read.
- **The tuple→SIGSEGV case matters for the G1 crash hypothesis**: this family DOES have a
  crashing variant, so "non-crashing ⇒ unrelated to their sentinel-deref crash" is weaker
  than first stated — a tuple/aggregate-wrapped SRET-forward can crash. (Still: TyUnit
  discriminant is 3≠0, so the E008 *value* mapping remains unsupported; the crash link is
  now plausible-but-unproven.)
- **Boundary is broader than the simplest rule**: one verifier found counterexamples where
  the bug fires but "forwards a struct-returning call's result as the return value"
  predicted it should work — so the family UNDER-predicts under that rule. The CORE trigger
  is rock-solid; the exact full boundary is not yet pinned (a codegen-level question best
  answered by whoever owns the SRET lowering).

## Relation to the G1 lane's "SRET-smash" crash (the "reproducer NOT FOUND" one)
The G1 docs (`CRASH_CLASS_ZERO`, `E008_ROOT_CAUSE`) **name this family**: "bin/souc
large-struct return-value miscompile" / "SRET-smash". They:
- verified **SRET-return-of-a-local is correct** (codegen lines 7196-7225), and
- located their crash in the **arg-checker** (a call WITH ARGUMENTS, `g(5)`→139,
  stack-independent; copies a 272-byte/34-qword TypeEntry field-by-field; **relocates per
  source change**), and worked AROUND it by routing exprs off the by-value path — they
  declared **"reproducer hunt NOT FOUND — emergent at full-compiler scale."**

**This repro is a clean minimal member of that same named family, and supplies the missing
distinction**: SRET-return-of-a-local is correct (they verified), but **SRET-FORWARDING
(returning another struct-returning call's result) is broken** — silent zeroing, or SIGSEGV
when the forward is aggregate-wrapped (`SRET_FORWARDING_CRASH_REPRO_2026-06-02.sio`, 4 lines,
deterministic rc=139).

**Honest non-identity** (don't overclaim — the discriminators say sibling, not same):
- Mine is **deterministic**; theirs **relocates per source change** (layout-sensitive).
- Mine is the **forwarding-return** instance; theirs is the **arg-checker-copy** instance.
- So: same family root (large-struct value-move in the SRET path), distinct trigger. A fix
  to the codegen's large-struct-move/SRET handling should address both; this minimal,
  deterministic repro is a far easier test case for that fix than the emergent-at-scale
  arg-checker crash. **Not proven byte-identical to their crash.**

## Handoff
This is a doc + repro (no `check.sio` / codegen edit) — the SRET lowering lives in the
codegen the G1 lane is actively working. Hand this minimal repro to whoever drives the
codegen fix; it should pin the bug in minutes vs gdb-ing the full Checker. Supporting:
`FRONT_HALF_LEVERAGE_HANDOFF_2026-06-02.md`, `EPISTEMIC_DEMO_SWEEP_2026-06-02.md`.

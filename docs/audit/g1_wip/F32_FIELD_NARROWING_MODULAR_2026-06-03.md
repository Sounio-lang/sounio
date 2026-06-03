# Cross-width float compatibility in types_compatible — +3 PASS, 0 regr, units-safe (2026-06-03)

Branch `check/f32-field-narrowing-e008` (stacked on algebra/sci-notation, off integration/e008
`df8d1db36`). Commit `549b93c5d`.

## Root cause (global, not field-local)
Canonical bin/souc treats same-category numeric types as mutually compatible in EVERY position —
verified rc=0 for f64-var → f32 in arg / return / let / assign / struct field-init (and i64↔i32;
cross-category int↔float correctly rejected). The modular `types_compatible` (compat.sio:10) was
strict exact-kind (`TyF32&&TyF32`, `TyF64&&TyF64` only), no cross-width case, so f64-into-f32
field inits spuriously raised E016 ("field initializer has wrong type, expected f32") — 31 progs.
Field-local patch would be incomplete (4 other positions); fixed the root.

## Fix + units safety (advisor-guarded)
Added a cross-width float case AFTER the exact f64-f64 case (so dimensioned f64s keep their unit
check) with the unit guard replicated. UNITS are the census blind spot (no unit-mismatch tests in
run-pass) — verified BY HAND outside the census: `mg`-into-`kg` mismatch still errors (rc=1) on
the rebuilt mc AND on bin/souc; `mg`-into-`mg` and the f32-field cases behave per canonical.
Check-level compatibility only; modular-backend f64→f32 coercion-insertion is out of scope.

## Result (modular census, 504 run-pass, mc rebuilt via build lock)
| | PASS | CRASH | regr |
|---|---:|---:|---:|
| algebra baseline | 238 | 0 | — |
| + cross-width float compat | **241** | 0 | **0** |

+3 FAIL→PASS (oct_minimal, octonion_basic_ops_standalone, octonion_cayley_dickson — the
sole-blocker subset). UNITS probe: mg→kg still rc=1 (no dimensional-analysis regression).

## Fix landed even where double-blocked (E016 31→26)
The former-E016 programs that did NOT flip now fail ELSEWHERE, proving the field-init E016 is
cleared: algebra_g2_invariants→E004, linalg_factorize→E014, test_fem→E001,
probability_distributions→E004. Of 26 residual E016: only 3 are still float-typed (a DIFFERENT
path — likely array-of-f32 / Knowledge<f32>-wrapped fields, e.g. autodiff_reverse, NOT cleared by
the scalar types_compatible change); 23 are non-float field mismatches (unrelated).

## Next levers (now exposed)
E004 (int-literal width — same narrowing family, algebra_g2 + probability_distributions land here
now) and E014 (usize index). Float-only this round by design; int cross-width (i64↔i32) is the
same root in types_compatible but a separate verified step.

## Blast radius
compat.sio (MODULAR checker) only; `lean_single.sio` untouched ⇒ `bin/souc` unchanged, canonical
gate unaffected. types_compatible is core/repo-wide but the change is monotonic (accepts strictly
more, matching canonical) — census confirms 0 PASS→FAIL.

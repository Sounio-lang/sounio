# Auto-deref &T/&!T in field access (E012) — +22 PASS, 0 regr (2026-06-03)

Branch `check/field-deref-ref-e008` (stacked, off integration/e008 `df8d1db36`). Commit `30f8f70bb`.

## Root cause (exposed by the ref-param lowering fix)
`checker_check_field_access_inplace` (check.sio:3191) branched on base_ty.kind for
TyNamed/TyKnowledge/TyIntervention/TyValidated/TyHyper/TyTuple/... but had NO TyRef/TyRefMut case.
So `s.field` where `s: &S` fell through every branch to E012 "this type has no field named"
(while `(*s).field` worked). The prior ref-param lowering fix (`421a91827`) exposed this: ref
params now lower to ty_ref(S) instead of silently failing, so E012 jumped 1 -> 27 (25/27 with ref
params).

## Fix
Auto-deref base_ty via the existing `call_ref_inner_or_self` helper (check.sio:10445; unwraps
TyRef/TyRefMut to inner, else identity) before the field lookup. Covers both read (`s.x`) and
write (`s.x = 1`, same handler). Matches canonical bin/souc and the explicit `(*s).x` path.

## Result (modular census, 504 run-pass, mc rebuilt via build lock)
| | PASS | CRASH | regr |
|---|---:|---:|---:|
| ref-param baseline | 300 | 0 | — |
| + field auto-deref | **322** | 0 | **0** |

+22 FAIL->PASS (array_mut_ref_bare, dissertation_frontend_parity_ref, mcmc_integration, test_fem,
test_lie, test_mor, pbpk_rapamycin_second_order, epistemic_ode_14comp, ...). PASS 322/504 (63.9%).

## Safety (advisor's "still rejects bad code" probe)
- &S read s.x: rc=0; &!S write s.x: rc=0.
- bad field `s.nonexistent` STILL rejected: rc=1 (bin/souc agrees rc=1) -> fixed deref, not silenced.

## Campaign tally (integration line, 8 pushed branches, 0 regressions throughout)
PASS 209 -> 235 -> 238 -> 241 -> 265 -> 273 -> 300 -> **322**. lean_single UNTOUCHED.

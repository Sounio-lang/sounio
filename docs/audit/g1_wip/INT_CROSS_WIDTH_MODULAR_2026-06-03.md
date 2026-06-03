# Cross-width/cross-sign integer compatibility in types_compatible — +24 PASS, 0 regr (2026-06-03)

Branch `check/int-cross-width-e008` (stacked on f32/algebra/sci-notation, off integration/e008
`df8d1db36`). Commit `c6f49e825`.

## Root cause (the int analog of the f32 fix — same function)
E004 ("these types cannot be combined with this operator") was NOT int-literal width (label was
wrong) — it is the binary-op type-mismatch raised when `binary_result_type(op,l,r)` returns
TyError. Arithmetic there is gated on `is_numeric_type(left) && types_compatible(left,right)`, so
the strict exact-kind `types_compatible` (no cross-width int case) made `i32 + i64` (and every
mixed-int pair) fail E004 — 61 progs — plus int field-init E016. Canonical bin/souc treats ALL
integer types as mutually compatible in EVERY position (arith/cmp/field/arg/return/assign),
including signed<->unsigned — verified rc=0 for i32+i64, u32+u64, i32+u32, i8+i64, u8+u64,
i32+u64, i64+u8.

## Fix
A broad integer case in `types_compatible` (any of i8/i32/i64/u8/u32/u64 with any of the same),
mirroring the cross-width float case. Monotonic (accepts strictly more, matching canonical).

## Result (modular census, 504 run-pass, mc rebuilt via build lock)
| | PASS | CRASH | regr |
|---|---:|---:|---:|
| f32 baseline | 241 | 0 | — |
| + cross-width int compat | **265** | 0 | **0** |

**+24 FAIL→PASS** (algebra_g2_invariants, algebra_g2_null_model, autodiff_forward,
complex_arithmetic, connectome_laplacian_eigenvectors, optimization_nelder_mead,
probability_distributions, gum_reporting, g2_bridge_pipeline, ode_rk4_general, +14). Note the two
algebra_g2 programs were double-blocked by E004 after the f32 fix — the layered campaign
compounds. Biggest single win of the parser/checker campaign.

## Safety (probed by hand — census-blind dimensions)
- `mg`->`kg` units mismatch still errors (rc=1) — the float-case unit guard survives.
- `f64 + i64` cross-CATEGORY still errors (rc=1) — broadened WITHIN int, not across int<->float.
- `i32 + i64` now passes (rc=0).

## Blast radius / honesty
compat.sio (MODULAR checker) only; lean_single.sio untouched -> bin/souc unchanged, canonical
gate unaffected. types_compatible is core/repo-wide and this is the campaign's broadest change
("touches every integer comparison"), but it is monotonic — census confirms 0 PASS->FAIL across
504. Check-level compatibility only; backend width/sign coercion-insertion (e.g. i64->i32
truncation) is out of scope. Cross-category int<->float and unit mismatches remain correctly
rejected.

## Campaign tally (integration line, 5 pushed branches, 0 regressions throughout)
PASS 209 -> 235 (sci-notation) -> 238 (algebra) -> 241 (f32 field) -> **265** (int cross-width).

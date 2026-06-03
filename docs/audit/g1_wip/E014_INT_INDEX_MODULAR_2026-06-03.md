# Accept any integer array index (E014) — +8 PASS, 0 regr (2026-06-03)

Branch `check/e014-int-index-e008` (stacked on int/f32/algebra/sci-notation, off integration/e008
`df8d1db36`). Commit `503ac2d21`.

## Root cause
The 4 array-index type checks in check.sio (read/write x inplace/by-value: 3291, 3933, 15287,
15689) required `idx_ty.kind == TypeKind::TyI64` exactly, so an i32/u32/i8 index raised E014
("array indices must be integers") even though canonical bin/souc accepts any integer index
(verified i32 index rc=0). usize already passed (usize maps to TyI64 internally).

## Fix
All 4 sites: `idx_ty.kind != TyI64` -> `!is_integer_type(idx_ty)`. Accepts every int width/sign
(i8/i32/i64/u8/u32/u64); float/bool/non-integer indices STILL rejected. Consistent with the
cross-width int compat fix (`is_integer_type` already used 16x in check.sio).

## Result (modular census, 504 run-pass, mc rebuilt via build lock)
| | PASS | CRASH | regr |
|---|---:|---:|---:|
| int-cross-width baseline | 265 | 0 | — |
| + any-integer index | **273** | 0 | **0** |

+8 FAIL->PASS (autodiff_reverse, _diag_sobol, fft_spectral, crypto_hash, compress_crc32,
polynomial_ops, interpolation, g2_cohort_comparison).

## Safety (hand-probed)
- i32 index now passes (rc=0).
- f64 index STILL rejected (rc=1) — did not over-broaden to non-integers.

## Blast radius
check.sio (MODULAR checker) only; lean_single.sio untouched -> bin/souc unchanged, canonical gate
unaffected. Monotonic (accepts strictly more int indices, matching canonical).

## Campaign tally (integration line, 6 pushed branches, 0 regressions throughout)
PASS 209 -> 235 -> 238 -> 241 -> 265 -> **273**. Remaining biggest lever: the SILENT "type
checking failed" bucket (~52, `&[T;N]`/`&![T;N]` array-reference params rejected with NO
diagnostic — needs error-reporting instrumentation to pin), then parse-error tail (70) + parse_failed
(37, item-level kernel/etc.).

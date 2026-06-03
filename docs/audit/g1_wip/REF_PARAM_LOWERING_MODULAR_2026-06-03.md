# Lower &T / &!T reference types in the *mut native checker — +27 PASS, 0 regr (2026-06-03)

Branch `check/ref-param-lower-e008` (stacked, off integration/e008 `df8d1db36`). Commit `421a91827`.
This is the "make it work full" deep fix — the dominant SILENT census bucket.

## The architecture (discovered, advisor-guided reads not instrumentation)
mc has two frontend paths (module_frontend.sio):
- NO imports -> check_items_verdict_boot4 -> the FULL strict *mut Checker (check.sio).
- WITH imports -> load_multimodule_ir_counts, which only COUNTS functions and NEVER type-checks
  (so import programs "pass" unchecked — they can't surface this bug).
The ~52 "type checking failed" census bucket is no-import programs hitting the strict path.

## Root cause (found by READING, per advisor)
The strict path failed SILENTLY (had_error/error_count bumped, no diagnostic) on EVERY ref param
(&i64, &[i64;8], &Struct, &!T, &self) — verified across scalar/array/struct, shared/excl, and
confirmed no native-PASS program uses a ref param (clean discriminator). The silent setter is
`checker_note_type_error_mut` (check.sio:870 — no print). Its caller on this path:
`checker_lower_type_expr_mut`'s match had NO TypeReference/TypeRefMut case, so every reference
type fell to `_ => checker_note_type_error_mut; ty_error()`.

## Fix
Ported the by-value `lower_ref_like_type` (10649) to a *mut helper: &[T] unsized -> ty_slice /
ty_slice_mut; &[T;N] / &T / &Struct -> ty_ref / ty_ref_mut(inner). Added TypeReference/TypeRefMut
arms to the match (mutually recursive with checker_lower_type_expr_mut, same pattern as the
existing array helper).

## Result (modular census, 504 run-pass, mc rebuilt via build lock)
| | PASS | CRASH | regr |
|---|---:|---:|---:|
| E014 baseline | 273 | 0 | — |
| + ref-type lowering | **300** | 0 | **0** |

+27 FAIL->PASS (array_mut_ref, array_elem_field_store, bdf64_test, borrow_*, octonion_mul_test,
wavelet_haar_roundtrip, graph_laplacian_path, symbolic_test, ...). Crosses PASS 300 / 504 (59.5%).

## Safety (advisor's "don't just silence the checker" probe)
- &i64 param valid: rc=0; &![i64;8] write valid: rc=0.
- bad ref-param body `fn f(a:&i64)->i64 { "string" }` STILL rejected: rc=1 (bin/souc agrees rc=1).
  => the fix lowers the type correctly; the body checker still catches real errors.

## Campaign tally (integration line, 7 pushed branches, 0 regressions throughout)
PASS 209 -> 235 -> 238 -> 241 -> 265 -> 273 -> **300**. Remaining: parse-error tail (~70),
parse_failed (~37 item-level kernel/etc.), and the ~12 non-ref-param 'type checking failed'.

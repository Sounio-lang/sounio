# E008 over-strictness fix — VERIFIED numbers (corrects commit efa927a9d) — 2026-06-03

Commit efa927a9d (int-literal narrowing on return) made two claims that filename-based
inference got wrong; canonical-bin/souc cross-checks correct them:

## What the int-literal-narrowing fix actually did (verified)
- Census 504: PASS 151 → 209, E008-progs 133 → 9, CRASH 0 → 0.
- **Genuine PASS gain = 54** (NOT 58): of the 58 progs that flipped E008→PASS, **54 are
  also accepted by canonical bin/souc** (genuine). The other **4** (audit_trail_basic,
  closure_linear, seq_basic, str_index_of) pass mc only because mc doesn't enforce
  privacy / linear-consume / etc. that canonical does — pre-existing mc enforcement gaps,
  NOT caused by this narrowing (the fix is int-literal→int-return only).

## CORRECTION: the 9 remaining E008 are NOT genuine
The commit said "remaining 9 are genuine/non-int-literal." **Wrong.** All 9
(connectome_laplacian_eigenvectors, ffi_ctypes, fft_spectral, g2_cohort_comparison,
observe_with_effect, oct_minimal, octonion_cayley_dickson, unit_energy_explicit_conversion,
unobserved_basic) are **STILL OVER-STRICT** — canonical bin/souc compiles all 9. So a
SECOND over-strict E008 sub-class remains: non-int-literal return coercions the modular
checker still rejects (float / unit-dimensional / octonion / ffi return types). Not done.

## Honest status of "the E008"
- fn_sigs-loss SPURIOUS E008: resolved by G1's CRASH-0 *mut work.
- int-literal over-strict E008 (the dominant remaining class): RESOLVED here (124 progs).
- residual over-strict E008 (9 progs, non-int-literal coercions): OPEN — a smaller second
  sub-class, same mechanism (return/assign type check too strict vs canonical), next step
  is to widen the coercion allowance (float-literal, unit-dimensional, etc.) the same way.

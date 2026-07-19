<!-- docs:meta
topic_id: repo.docs.handoff.souc-sqrt-extern-to-sqrtsd-codex-dispatch-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.souc-sqrt-extern-to-sqrtsd-codex-dispatch-2026-07-19
-->

# Dispatch to CODEX-2 — `sqrt` is WRONG for large arguments (P0 correctness) and also slow

> **UPDATED 2026-07-19 — this is now a P0 CORRECTNESS bug, not just a perf issue.** The original
> version of this dispatch claimed `sqrt` was "bit-identical to libm, just an extern call, ~244 ns."
> That was **wrong**: `sqrt(x)` returns a **numerically incorrect result for |x| ≳ 1e6**. Do NOT preserve
> the current behaviour. The perf ask (lower to `sqrtsd`) is retained below because the correct hardware
> instruction fixes *both* the correctness and the speed at once.

**Date:** 2026-07-19
**Owner:** CODEX-2 (compiler back-end / codegen for math builtins; `self-hosted/`)
**Author:** data-science lane (surfaced while validating `bf_corr` on large-magnitude columns)
**Status:** confirmed P0 correctness defect + perf defect, minimal repro included

---

## TL;DR

`sqrt(x: f64) -> f64` is **correct for small `x` (≲ 1e6) but silently returns a wrong value for large
`x`** — e.g. `sqrt(1e12)` gives `1279996.5` instead of `1000000`, `sqrt(1e20)` gives `9.54e13` instead
of `1e10`. The magnitude of the error grows with `x`. This is a **silent-wrong** result (no crash, no
NaN), so it corrupts any statistic whose intermediate exceeds ~1e6 without any signal. It is *also*
slow (~244 ns/call vs a ~1–4 ns hardware instruction). Lowering the f64 `sqrt` builtin to the hardware
`sqrtsd` (x86-64) / `fsqrt` (arm64) instruction fixes **both** — `sqrtsd` is IEEE-754 correctly-rounded
at every magnitude.

## Evidence — correctness (reproducible now, `lean_single` engine)

`sqrt(<v>)` vs CPython `math.sqrt`:

| input `v` | `sqrt(v)` (souc) | correct | status |
|---|---|---|---|
| 4 | 2 | 2 | ok |
| 100 | 10 | 10 | ok |
| 1e6 | 1000 | 1000 | ok (boundary) |
| **1e12** | **1279996.53** | **1000000** | **WRONG** |
| **2.5e19** | **2.384e13** | **5e9** | **WRONG** |
| **1e20** | **9.537e13** | **1e10** | **WRONG** |
| **2.777778e28** | **2.649e22** | **1.667e14** | **WRONG** |
| **1e30** | **9.537e23** | **1e15** | **WRONG** |

Repro (either engine that compiles it; confirmed on `lean_single`):
```
fn main() -> i32 with IO, Mut, Panic, Div {
    print(sqrt(1000000000000.0)); print(" "); print(sqrt(1e20)); print("\n"); return 0
}
// prints "1279996.526738 95367431990150.328125"; correct is "1000000 10000000000"
```
(On the default `madaros` engine this exact single-file probe hit an unrelated `E137` at compile; the
correctness bug is confirmed on `lean_single`, which is the engine the data stack uses. Please also
check `madaros` once it compiles.)

## Evidence — end-to-end impact

`bf_corr` on two **perfectly correlated** large-magnitude columns (`x = 1e9 + i`, `y = 1e9 + 2i`,
n = 100000) returned **6.29e-9** instead of **1.0**. The covariance/co-moment intermediates
(`cxx`, `cyy`, `cxy`) were all correct to full precision (verified against CPython); the *only* wrong
step was `sqrt(cxx*cyy)` with `cxx*cyy ≈ 2.78e28` → souc returned `2.649e22` instead of `1.667e14`.

## Root-cause hypothesis (REVISED)

The earlier "extern libc call, PLT overhead" hypothesis is **suspect** — a real `libm` `sqrt` would be
correct at all magnitudes. A result that is *correct for small `x` and drifts for large `x`* is the
signature of a **hand-rolled approximation** (e.g. a bit-hack seed with too few Newton-Raphson
refinement steps, or a reduced-precision / single-iteration path), not a call into `libm`. CODEX-2 to
confirm what `sqrt` actually lowers to. (An independent inline Newton-Raphson with the same bit-hack
seed but **6** iterations is correct to ≤ 1 ULP at every magnitude — so if the current path is
under-iterated, adding iterations would also fix correctness, but the hardware instruction is the right
answer.)

## The ask

Lower the `f64` `sqrt` builtin to the hardware square-root instruction:
- **x86-64:** `sqrtsd %xmmSrc, %xmmDst` (or `vsqrtsd`).
- **arm64:** `fsqrt d_dst, d_src`.

`sqrtsd`/`fsqrt` are IEEE-754 correctly-rounded, so the result is correct at every magnitude and ~1–4 ns.
This resolves the correctness bug and the perf issue together. If `fabs`, `floor`, `ceil`, `trunc`,
`rint`/`round`, `min`/`max` share the same (approximation) path, audit them for the same correctness
problem and lower them to their single instructions too.

## Acceptance

- `sqrt(v)` is correctly-rounded (matches CPython `math.sqrt` bits, ≤ 0.5 ULP) across a magnitude
  sweep from subnormals to ~1e300, INCLUDING the large values in the table above.
- `bf_corr` on the perfectly-correlated large-magnitude columns returns 1.0.
- The 30M-call `sqrt` loop drops from ~7.3 s to well under ~1 s.

## Scope / impact

- **P0 correctness:** silent-wrong `sqrt` corrupts every call site whose argument exceeds ~1e6. Known
  stdlib call sites (grep `fn sqrt`): `stdlib/optimize/{uncertainty,differential_evolution,nelder_mead,
  levenberg_marquardt}.sio`, `stdlib/fusion/eeg_fmri.sio`, and — most importantly for the measurement
  thesis — the **GUM uncertainty combine** `u_c = sqrt(Σ u_i²)` (`epistemic::gum` / `formal/GUM.lean`):
  any of these on large-magnitude inputs is silently wrong today. `stdlib/data/bigframe_ops.sio`
  (`bf_expanding_std`, and `bf_corr`) has been given a correct in-stdlib Newton `bf_sqrt` as a stopgap;
  the compiler fix lets those revert to the builtin.
- **Low blast radius to fix:** codegen for one (or a few) math builtins; no front-end or type changes.
- Complements the `mem_copy` builtin (`docs/handoff/mem_copy_builtin_codex_dispatch_2026-07-19.md`,
  `bf_shift`) and C3 SIMD auto-vectorisation (`bf_diff`) dispatches.

## Pointers

- Repro + magnitude table: this file.
- Correct stopgap: `stdlib/data/bigframe_ops.sio::bf_sqrt` (bit-hack seed + 6 Newton iterations, ≤ 1 ULP).
- Impacted verbs: `bf_expanding_std`, `bf_corr` (both now route through `bf_sqrt`).

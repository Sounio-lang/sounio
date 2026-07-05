<!-- docs:meta
topic_id: repo.docs.exact-core
authority: repo_only
audience: users
-->

# EXACT_CORE — the honest contract of the exact algebraic layer

> The combinatorial/primitive layer is proved exactly in Lean and now **executed exactly in the
> Sounio runtime**. The f64 path remains the analytic layer. **Algebraic exactness precedes any
> move to float256.**

## Why this exists

The sedenion algebra 𝕊 (dimension 16, the Cayley–Dickson double of the octonions) contains
**zero divisors**: non-zero `a, b` with `a·b = 0`. The existing runtime detects them with a
**tolerance gate** — `||a·b||² < eps` under `f64` (`stdlib/math/sedenion.sio`,
`tests/run-pass/sedenion_zero_divisor.sio` at `prod_n2 < 1e-7`;
`examples/sedenion_zero_div_hunt.sio` at `pn < 1e-9`). A float that happens to be `0.0` is **not
a proof**. This layer adds, *alongside* the f64 path (which is unchanged), an **exact** layer where
annihilation is decidable integer equality: `a·b == 0` in an exact ℤ-vector.

> **Invariant: exactness is a property of the CONTRACT, not of the number.**

## The two layers

| Layer | Coefficients | Zero-divisor test | Verdict |
|---|---|---|---|
| Analytic (existing, unchanged) | `f64` | `\|\|a·b\|\|² < eps` | `MeasuredF64(eps)` |
| **Exact (this work)** | **ℤ (i64)** | **`a·b == 0` componentwise, decidable** | **`Proved`** |

The exact engine `stdlib/algebra/cayley_dickson_exact_i64.sio` mirrors the f64
`cayley_dickson.sio` (same sign kernel `cd_sigma`, same product structure) but accumulates in `i64`
and selects add/subtract from the ±1 sign — no floating point anywhere in the annihilation path.

## Minimal field: ℤ suffices

Zero-divisor **detection is homogeneous over ℤ** (verified against
`formal/lean4/SounioZeroDivisorBridge.lean`): the whole path is ±1 sign algebra, integer sums
compared to zero — no norms, no division, no √. So **`F = i64` fully proves the n=4 census**. `ℚ`
(`stdlib/math/rational.sio`) is only for norm-bearing observables and is a *secondary* target; `ℚ(√2)`
is reserved for norms, never annihilation.

## The typed epistemic boundary (`Verdict`)

`stdlib/algebra/sedenion_verdict.sio` makes provenance a first-class value:

```
enum Verdict { Proved, MeasuredF64 { eps }, MeasuredF256 { eps256 } }
```

`Proved` (exact ℤ equality), `MeasuredF64` (f64 tolerance), `MeasuredF256` (future high-precision
witness shape — **no f256 arithmetic is implemented**). The load-bearing rule is enforced by
`requires_proof(v)`, which accepts **only** `Proved` — **a measurement can never be laundered into
a proof.** There is no function that converts one variant to another. (Operator's `Knowledge<Verdict>`
wrapping lands once the generic-struct-return compiler gap is fixed; until then gates return the
plain `Verdict` enum, which compiles and runs today.)

## What is proved, executed, and verified (souc v0.80.0)

- **Exact product runs** — `tests/run-pass/sedenion_zd_exact_smoke.sio`: the canonical pair
  `(e₃+e₁₀)·(e₆−e₁₅)` **annihilates exactly** by decidable i64 equality (`ZD PROVED`); `e₁·e₁ = −1`
  (`SQ PASS`); `e₁·e₂ ≠ 0` (`NONZERO PASS`).
- **168-census executed AND cross-toolchain-verified** — `tests/run-pass/sedenion_zd_census_168.sio`
  reproduces the Lean bridge census **from the exact product**:
  `validPrims = 84 → orderedZDPairs = 336 → unorderedZDPairs = 168` (168 = |PSL(2,7)|). It emits the
  168 **specific** canonical pairs as data, and `scripts/ci/sedenion_zd168_crosscheck_gate.sh` asserts
  that set is **element-wise identical** to an INDEPENDENT Python oracle
  (`scripts/research/verify_zd168_oracle.py`, transcribed directly from `formal/lean4/*.lean`, run on
  a different toolchain) → `CROSS-VERIFIED: 168/168 identical pairs`.

  **Why this matters (contract, not number):** souc v0.80.0 has a documented false-green mode, so a
  bare `PASS` is not itself proof of execution — isomorphic to `||ab||<eps` not being `ab==0`. A stub
  can forge a *count*; it cannot forge 168 *specific* pairs that match an independent computation.
  Honest scope: Lean is not runnable in this environment, so the element-wise diff is souc-vs-Python
  (two independent toolchains); Lean's leg is its `native_decide`-proven **counts** (`prim_count_84`,
  `zd_pair_count_336`, `zd_projective_count_168`). A Lean-runtime third leg can be added by installing
  `elan`/`lean` and `#eval`-emitting `unorderedZDPairs`.
- **Boundary enforced** — `tests/run-pass/sedenion_verdict_boundary.sio`: `Proved` accepted,
  `MeasuredF64`/`MeasuredF256` rejected.

This is the headline: **Sounio executes the Cayley–Dickson product and proves annihilation exactly at
n=4** — the same 168 the Lean `SounioZeroDivisorBridge.lean` proves by `native_decide`, now computed
by the running language.

## Honest caveats (souc v0.80.0 environment, not defects of this work)

- The **f64 layer does not type-check under this build** (`cayley_dickson.sio` → `error[E004]` on
  `<<`; `sedenion_zero_divisor.sio` → multimodule link failure). So the exact engine **inlines**
  `cd_sigma` (identical recursion + `as i32` casts) rather than importing it, and Phase-4
  "run both f64 and exact assertions side by side" cannot execute here — the exact assertions stand
  alone. When the f64 layer is repaired, the exact assertions should be added *alongside* the existing
  tolerance gates (never replacing them, never relabeling an eps-gated pass as "exact").
- **Multi-module compact-stub false-green**: a test importing 2+ modules compiles to a do-nothing
  stub. All exact tests here import **one** module (the engine) to genuinely execute; the reusable
  `sedenion_verdict.sio` module is import-ready once multi-module is fixed.
- A data-carrying-enum `match` codegen flake was observed (a redundant `is_measurement` returned a
  wrong value while `requires_proof` was fully correct); the flaky redundant helper was dropped.

## The `<F>` migration target

The generic `stdlib/algebra/cayley_dickson_exact.sio` (`CDElementExact<F: ExactRing>`) is a
**design skeleton** — it needs four compiler features (generic-struct-return **#1**, bodyless
trait-method-sig parsing **2a**, `impl Trait for Type` **2b**, trait-bounded dispatch **#3**), all
commissioned in `docs/handoff/*`. The concrete-i64 engine is exactly what it monomorphizes to, so
nothing is wasted; the science ships now on `i64`. See `docs/handoff/exact_engine_prereqs.md`.

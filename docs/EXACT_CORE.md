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

  **Why this matters (contract, not number):** souc v0.80.0 has **confirmed codegen defects that
  return wrong results** — e.g. a `match` over a data-carrying enum returns the wrong arm's value
  (issue #639), and a cross-module aggregate call SIGSEGVs (issue #637). Under such a toolchain a bare
  `PASS` is not itself proof of correct execution — isomorphic to `||ab||<eps` not being `ab==0`. A
  miscompile or stub can forge a *count*; it cannot forge 168 *specific* pairs that match an
  independent computation.
  Honest scope: Lean is not runnable in this environment, so the element-wise diff is souc-vs-Python
  (two independent toolchains); Lean's leg is its `native_decide`-proven **counts** (`prim_count_84`,
  `zd_pair_count_336`, `zd_projective_count_168`). A Lean-runtime third leg can be added by installing
  `elan`/`lean` and `#eval`-emitting `unorderedZDPairs`.
- **The OTHER 168 executed AND cross-verified** — `tests/run-pass/octonion_nonfano_census_168.sio`
  computes the **non-Fano (non-associative) octonion census** exactly (associator signs `α≠β` by
  decidable integer inequality): `TRIPLES 343 / NONFANO 168 / FANO 175`, plus the **Binary Norm
  Theorem** (`BINARY-NORM {-2,0,+2}` — the associator wave `α−β` is always −2, 0, or +2) and the
  **arrow symmetry** (`FORWARD 84 / BACKWARD 84 / ARROW-SYMMETRY` — exactly 84 triples at wave +2 and
  84 at −2). This reproduces `SounioCayleyDickson.lean`'s `non_fano_count_168` / `fano_count_175` /
  `partition_343` / `arrow_forward_84` / `arrow_backward_84` / `arrow_symmetry`. The gate
  also checks its 168 **specific triples** are element-wise identical to the oracle
  (`non-Fano triples: 168/168 identical`). The Lean `nonfano_zd_bridge` theorem proves these two
  168s are **equal** — so both faces of the 168-theorem (zero-divisors and non-associativity) are now
  executed exactly by the running language and cross-toolchain-verified, both = |PSL(2,7)|.
- **The 84↔84 bijection executed as an explicit map** — `tests/run-pass/octonion_dagger_bijection_84.sio`
  runs the **dagger involution** `(i,j,k) ↦ (k,j,i)`, which negates the associator wave
  (`DAGGER-REVERSAL 343`) and is **free** on the 168 non-Fano triples (`FREE-INVOLUTION (0 self-dual)`)
  — hence a concrete bijection carrying the 84 forward-arrows onto the 84 backward-arrows
  (`FWD->BWD 84 / BWD->FWD 84 / BIJECTION 84<->84`). This is the structural *why* behind 168 = 84+84
  (`SounioCayleyDickson.lean` `dagger_reversal` / `no_nonfano_self_dual`). The gate diffs the 84
  emitted forward→backward **arrows** element-wise against the oracle (`84/84 identical arrows`) — the
  *map itself*, not just its cardinality, is cross-toolchain-verified.
- **Boundary enforced** — `tests/run-pass/sedenion_verdict_boundary.sio`: `Proved` accepted,
  `MeasuredF64`/`MeasuredF256` rejected.

This is the headline: **Sounio executes the Cayley–Dickson product and proves — on BOTH faces of the
168-theorem — annihilation (n=4 zero divisors) and non-associativity (n=3 non-Fano triples) exactly**,
the same two 168s the Lean layer proves by `native_decide` and proves equal, now computed by the
running language and element-wise cross-verified against an independent toolchain.

## Honest caveats (souc v0.80.0 environment, not defects of this work)

- The **f64 layer does not type-check under this build** (`cayley_dickson.sio` → `error[E004]` on
  `<<`; `sedenion_zero_divisor.sio` → multimodule link failure). So the exact engine **inlines**
  `cd_sigma` (identical recursion + `as i32` casts) rather than importing it, and Phase-4
  "run both f64 and exact assertions side by side" cannot execute here — the exact assertions stand
  alone. When the f64 layer is repaired, the exact assertions should be added *alongside* the existing
  tolerance gates (never replacing them, never relabeling an eps-gated pass as "exact").
- **Cross-module aggregate-lowering SIGSEGV (issue #637)**: importing a module with the `[i64;2048]`
  `CDElementExactI64` engine and delegating across an aggregate-param-arity mismatch crashes the
  compiler. This — not a generic "multi-module false-green" — is why the 168-census is self-contained
  (inlines the scalar sign function, no engine import). CORRECTION: an earlier note attributed the
  self-contained design to a "multi-module compact-stub false-green"; that mode was **not
  reproducible** in this build (a clean 2-module case compiles and runs correctly), so the real
  driver is #637. See `docs/handoff/souc_v0800_defects.md` (D1 = honest negative).
- **Data-enum `match` returns wrong value (issue #639)**: a redundant `is_measurement` returned the
  wrong arm value while `requires_proof` was fully correct; the flaky redundant helper was dropped and
  the bug filed. `requires_proof` (the load-bearing boundary) is verified correct across all variants.
- All four souc v0.80.0 findings (incl. the not-reproduced multi-module one) are documented with
  minimal repros in `docs/handoff/souc_v0800_defects.md`; issues #637 (segfault), #638 (`<<`→i64),
  #639 (`match`).

## The `<F>` migration target

The generic `stdlib/algebra/cayley_dickson_exact.sio` (`CDElementExact<F: ExactRing>`) is a
**design skeleton** — it needs four compiler features (generic-struct-return **#1**, bodyless
trait-method-sig parsing **2a**, `impl Trait for Type` **2b**, trait-bounded dispatch **#3**), all
commissioned in `docs/handoff/*`. The concrete-i64 engine is exactly what it monomorphizes to, so
nothing is wasted; the science ships now on `i64`. See `docs/handoff/exact_engine_prereqs.md`.

<!-- docs:meta
topic_id: repo.docs.exact-core
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.exact-core
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
witness shape — **no f256 arithmetic is implemented under Madaros, this project's default engine**).
The load-bearing rule is enforced by
`requires_proof(v)`, which accepts **only** `Proved` — **a measurement can never be laundered into
a proof.** There is no function that converts one variant to another. (Operator's `Knowledge<Verdict>`
wrapping lands once the generic-struct-return compiler gap is fixed; until then gates return the
plain `Verdict` enum, which compiles and runs today.)

**Engine split (verified 2026-08-17).** Sounio ships two compiler engines — default Madaros
(`bin/souc`) and the bootstrap seed (`SOUNIO_SOUC_ENGINE=lean_single` / `bin/souc-lean-single-x86_64`).
The "no f256 arithmetic is implemented" claim above holds only for Madaros:

| Engine | `fn add(a: f256, b: f256) -> f256 { a + b }` |
|---|---|
| **Madaros** (default `bin/souc`) | Rejects with **`error[E218]`**: *"f128/f256 is reserved for compiler-owned format identity; source values are unavailable in V0-A"*. |
| **lean_single** (bootstrap seed) | **Compiles and executes.** No E218, no diagnostic; the emitted ELF runs to completion (`rc=0`). |

So `MeasuredF256` is unreachable under the engine this document otherwise treats as authoritative
(Madaros), but f256 arithmetic is not, in fact, unimplemented in this codebase — it exists,
unverified and undocumented as a witness shape, under lean_single. Same class of gap as V0-A in
`docs/architecture/F128_F256_LADDER.md`, which carries the equivalent table for f128. Do not read
lean_single's acceptance as license to treat f256 as available: it has no `MeasuredF256` witness
construction, no epistemic surface, and no gate — it simply fails to refuse.

The dual-engine split is **not** limited to the tilde / f128–f256 parser boundary. Two further
measured cases (2026-08-17), recorded so this document does not leave the reader thinking
“engine divergence = only E218”:

| Case | Madaros (default) | lean_single | Status |
|---|---|---|---|
| Forward ontology `inverse_of` (#1798) | **Accepted** a role whose inverse target was declared later | **E158** reject | **CLOSED** — Madaros aligned to declaration-order; gate `scripts/ci/madaros_ontology_enforcement_gate.sh` |
| GUM variance on dissertation surfaces (#1792) | Prints `var(...)=0.000000` (and related ep28 confidence bit-pattern fabrication) | Non-zero variance ~1e-5 / ~1e-9 on the same adaptive witness | **OPEN** — fail-closed detect gate `scripts/ci/epistemic_fabrication_detect_gate.sh`; not a full ABI fix |

#1792 is thesis-critical: silent zero variance under the default engine is fabricated science, not a
docs nit. See also `CLAUDE.md` §13 and `docs/audit/EPISTEMIC_FABRICATION_DETECT_2026-08-17.md`.

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

## Scope — what this certifies, and what it does NOT (read before citing)

**CERTIFIED** (exact over ℤ, decidable integer equality, element-wise cross-verified against a
non-`souc` oracle): the **combinatorial-structural** object — the ZD census 84/336/168, the non-Fano
partition 168/175/343, the Binary Norm Theorem (wave ∈ {−2,0,+2}), the 84↔84 dagger bijection, and
the two-face `nonfano_zd_bridge` collapse onto |PSL(2,7)|. This certifies the *annihilation locus*:
which pairs annihilate, how many, under which group, with which projective geometry.

**MEASURE LAYER — a first exact instance executed** (over ℚ; Frente A):
`tests/run-pass/sedenion_measure_annihilation_exact.sio` executes the measure claim for a concrete
exact empirical measure on the canonical channel `a=αe₃+βe₁₀, b=γe₆+δe₁₅` (functional `F = r5 =
αγ+βδ`). Support **on** the locus → `E[F]=0/1` and `Var[F]=0/1` (exact zero); support perturbed
**off** by exact `ε=1/10` → `E[F]=0/1` but `Var[F]=1/150` (exact positive rational, = GUM `2ε²/3`).
By **decidable rational equality**, cross-verified against Python `fractions` (unbounded exact). The
`Var` flips `0/1 → 1/150` as support leaves the locus — the confidence collapse, now exact:
`Var>0` was the number, `Var=0/1` is the contract.

> **Formalization note:** the measure-theoretic statement is **not** formalized in Lean
> (`SounioSedenionMeasurement.lean` defers it — "requires Mathlib/Hilbert"). This artifact *defines*
> the exact ℚ statement, grounded in the float witness `sedenion_zero_divisor.sio`. It is a first
> exact instance, not the general theorem.

**GENERALIZED — the i64 exactness boundary located and cross-verified**
(`tests/run-pass/sedenion_measure_annihilation_general.sio`): sweeping the off-locus measure across
scales `ε=1/10^k`, `Var = 2/(3·10^(2k))` is computed by **overflow-checked** rationals that return an
in-band **INVALID (censored)** the instant an i64 product would overflow — never a silently-wrapped
false "exact" value. souc is exact for **k=1..9** and censors `OVERFLOW` for **k≥10**, and the gate
confirms this is **exactly** where the unbounded Python `fractions` oracle needs BIGINT
(`i64 exactness boundary located at k=9`). This is the daring generalization's honest yield: it
*locates* the substrate ceiling precisely (exact ℚ on i64 holds to ε=10⁻⁹) and proves souc censors
correctly rather than corrupting — the Firewall applied to arithmetic itself.

**UNBOUNDED — the i64 wall REMOVED via a from-scratch bigint**
(`tests/run-pass/sedenion_measure_annihilation_bigint.sio`): a minimal arbitrary-precision `BigNat`
(base-10⁹ limbs; `mul_small`/`div_small`/`pow10`/decimal-print, all built in Sounio) computes the exact
`Var = 2/(3·10^(2k))` for **k=1..20** — up to denominator `1.5×10⁴⁰`, far past the i64 wall (k=9). The
6th gate face diffs all 20 values element-wise against the unbounded Python `fractions` oracle → exact
match. So exact ℚ in Sounio is **no longer bounded by i64**; the boundary at k=9 is *located* (i64
engine) and *removed* (bigint engine). The honest reduction uses only `mul_small`+`div_small` (gcd(2,
3·10^(2k))=2 exactly), no full bigint division.

**STILL NOT executed (the fully general theorem)**: **arbitrary** probability measures and **general
locus parameterizations** (this executes the canonical channel's off-locus family across scales, not
every measure on every locus). A fully general engine also wants a complete bigint (mul/div/gcd) and
must route around the souc codegen defects surfaced here (`docs/handoff/souc_v0800_defects.md`,
incl. D5 #641). The general statement remains proven at the statement level; the *precision* barrier
is gone, the *generality* barrier (arbitrary μ, arbitrary locus) remains.

Reconciliation (why this is the same contract-vs-number theme, one layer up): the float artifact
`sedenion_zero_divisor.sio` shows `E[a·b]=0` but `Var>0` — the "confidence collapse". That is what a
measure supported *near* the locus (perturbed float coefficients) looks like. The exact claim
`Var[F_N]=0` is what a measure supported *exactly on* the locus (exact ℚ coefficients) looks like.
`Var>0` is the number; `Var=0` is the contract. The exact ℚ execution is the measure-layer analogue
of what this ℤ census did for the structure.

## General 16-component CD product over ℚ (the aggregate wall, circumvented)

`tests/run-pass/sedenion_cd_full16_q.sio` computes the **full 16-component** Cayley-Dickson product for
an **arbitrary rational pair** (not the ±1 canonical channel) exactly over ℚ. Key move: the
**common-denominator representation** — a rational sedenion is 16 integer numerators over one common
denominator, so the rational product reduces to an *integer* 16-component product of numerators
(denominator `da·db`). There is no `[Rational;16]` by-value struct, so the #637 aggregate wall is
**circumvented, not hit**. Canonical pair → all 16 components exactly 0 (the whole product annihilates,
generalizing beyond the hand-derived r5/r12); a general rational pair → exact reduced components. 9th
gate face: 34/34 DEN+COMP lines identical souc == independent oracle (`scripts/research/cd16_oracle.py`).

**Scope / residual:** the executed version uses i64 numerators (rational coefficients with num/den up
to ~10⁸; output fits i64). Extending to **bigint** coefficients (unbounded) is capability-proven
*separately* (the bigint sweep to 10⁴⁰; the ratbig channel with `t=123456789/7`) but does not *compose*
into the full-16 under souc v0.80.0: the module-import path SIGSEGVs on `[BigInt;16]` (aggregate #637),
and an inline signed-multi-limb + 16-accumulator engine would exceed the ~24-function whole-program
codegen capacity wall. So: the general 16-component product over ℚ is executed and cross-verified for
i64-range coefficients; the unbounded-width integration is a compiler-capacity residual, not a math gap.

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

## The `<F>` migration target — UNBLOCKED for scalar `F`, struct-`F` still walled

The four compiler features the generic engine needed (generic-struct-return **#1**, bodyless
trait-method-sig parsing **2a**, `impl Trait for Type` **2b**, trait-bounded dispatch **#3**) **all
landed 2026-07-06** on the fable5 compiler-generic-F lane (PR #650, merge commit `2adb8f061`,
against the prompt `docs/handoff/compiler_generic_F_engine_unblock_prompt.md`). The generic engine
`stdlib/algebra/cayley_dickson_exact.sio` (`CDElementExact<F: ExactRing>`) now **compiles and runs**:

- **`F = i64` — adopted and proven equivalent.** `tests/run-pass/cd_exact_generic_i64.sio` proves the
  canonical ZD pair `(e₃+e₁₀)(e₆−e₁₅)=0` (16× `COMP 0`), and `cd_exact_generic_vs_concrete.sio`
  shows the generic engine reproduces `cayley_dickson_exact_i64.sio` **byte-for-byte**
  (`BYTECOMPARE PASS`). The concrete-i64 engine and the generic-at-i64 engine are now
  cross-certified against each other; the concrete one is retained as the load-bearing i64 path.
- **`F = Rational` / `F = BigInt` (struct coefficients) — blocked, but NOT by generics (issue #651).**
  The engine over a *struct* coefficient type compiles and runs but is **wrong**: the `[Rational;N]`
  multiply-accumulate loop miscompiles — a **deterministic garbage** value at N=16
  (`c[0]=4206741/1`; correct `1/1`) and a **SIGSEGV** at N=2048. Crucially this reproduces with **no
  generics at all** (a plain concrete `[Rational;16]` CD-multiply loop is equally wrong), and every
  sub-operation in isolation is correct (single constant-index RMW, variable-index read, variable-indexed
  by-value arg, one loop iteration) — only the full nested accumulation loop over array-of-struct
  temporaries corrupts. So it is a `[struct;N]` **aggregate/struct-temporary codegen bug** (same family
  as #643/D6 struct value-copy corruption; **distinct** from #637, which is a cross-module
  arity-mismatch *compiler* SIGSEGV). Scalar arrays (`[i64;N]`) are unaffected. Repro + full isolation:
  `docs/handoff/repros/d8_generic_struct_F_mul_segv.sio`.

**Net:** the old `<F>` residual (no generics at all) is *closed* — generics work end-to-end for scalar
`F`. The exact CD product over **unbounded ℚ for all 16 components** — which needs `F = BigInt`/`BigRational`,
i.e. array-of-struct coefficients — remains open, blocked by the filed non-generic `[struct;N]`
aggregate-loop codegen bug (#651), not by the math, the parser, or the generics. Until #651 lands,
unbounded-ℚ work continues via the common-denominator **integer** representation
(`sedenion_cd_full16_q.sio`, `[i64;N]`, no array-of-struct — unaffected). See
`docs/handoff/exact_engine_prereqs.md`.

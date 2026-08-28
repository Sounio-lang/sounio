<!-- docs:meta
topic_id: repo.docs.spec.s09-uncertainty-propagation
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s09-uncertainty-propagation
-->

# §9 — Uncertainty propagation

Spec-Section: `SOUNIO-SPEC-09`
Frame: `docs/spec/E2E_SPECIFICATION_FRAME.md`

Status: **undefined.** No normative statement has been ruled.

The form follows `S07_EFFECT_HANDLERS.md`. `S12_NUMERIC_TOWER.md` does not exist
on `origin/main` at the commit this section was written against
(`67aa2aec12`).

Founder decision, 2026-08-20: an epistemic value is to carry a Beta posterior
`(α, β)`, not a scalar confidence. That decision is **not** a composition rule.
This section writes the proposed rule, measures it against an independent
oracle, and stops. It does not migrate `knowledge.sio`, does not touch
`stdlib/darwin_pbpk`, and does not rewrite the 26 scalar-confidence sites.

## 9.1 Normative

*(empty until a composition ruling exists.)*

## 9.2 What is measured today

### 9.2.1 Two layers, neither of which is this rule

The frame marks §9 `contested`. What exists on `origin/main` is not one
propagation algebra. It is two disconnected implementations, plus a scalar
confidence that is not a posterior:

| Layer | Where | What it does with uncertainty | What it does with confidence |
|---|---|---|---|
| Compiler IR | `self-hosted/ir/lower.sio` (`emit_variance_add`, `emit_variance_independent_product`, `emit_variance_independent_div`, `emit_variance_scale`) | Machine-level GUM emitters. Independence is the **name of a function**; nothing checks that the operands are independent. | Absent. |
| Stdlib value | `stdlib/epistemic/knowledge.sio` `Epistemic { val, variance, confidence }` | GUM on **variance**: `ep_add` adds variances; `ep_mul` is the first-order product `b²·Var(a) + a²·Var(b)`; `ep_scale` multiplies variance by `c²`. | Integer `0..1000`. `ep_add` takes `min` then `* 99/100`. `ep_merge` **averages** the two integers. `ep_scale` **preserves** the integer. |

`docs/internal/concepts/verified-lowering.md` already records the IR fact:
uncertainty propagation is a sequence of instructions bound to a slot, not a
property of a value. Forgetting to bind the slot is the natural failure mode
(the FO-matrix defect). This section does not repair that.

The stdlib fact is the one the Beta rule has to answer. A scalar `500` from two
observations and a scalar `500` from two thousand observations are the same
integer. `ep_merge` of two `500`s is `500`. Nothing in `knowledge.sio`
distinguishes the two.

### 9.2.2 Companion type on this branch, not a replacement

This branch adds `stdlib/epistemic/beta_confidence.sio` with

```
struct EpistemicBeta {
    value: f64,
    uncertainty: f64,   // GUM standard uncertainty u, not variance
    conf_alpha: f64,
    conf_beta: f64,
}
```

It is a **companion**. `Epistemic` is untouched. Callers of `ep_add` still
decay an integer. The 26 scalar sites still store an integer. No file in
`examples/` or `stdlib/darwin_pbpk` is edited.

`uncertainty` here is JCGM 100:2008 standard uncertainty *u* (one standard
deviation). `knowledge.sio` stores **variance**. Adding two `uncertainty`
fields as if they were variances is a different, wrong, rule; the tests pin
the RSS combination.

### 9.2.3 Oracle

`scripts/dev/beta_confidence_oracle.py` computes the same compositions in
`fractions.Fraction` (moments of `Beta(α, β)` with rational parameters are
exact). Square roots use `decimal` at 80 digits, then round to IEEE f64 for
the Sounio witness. `mpmath` is not required and was not used: the pod's
Python is PEP 668-externally-managed and `mpmath` is not installed.

Re-run:

```
python3 scripts/dev/beta_confidence_oracle.py
```

The Sounio goldens in `beta_confidence.sio` / `tests/run-pass/beta_confidence_rule.sio`
are those printed values, not hand-tuned tolerances.

## 9.3 Proposed composition rule (not a ruling)

Two channels, two algebras. Mixing them is the defect this section exists to
name.

| Slot | Algebra | Closed? |
|---|---|---|
| `(value, uncertainty)` | GUM / JCGM 100:2008, first-order | Linear combinations: yes. Products: first-order, with a named dropped term. |
| `(conf_alpha, conf_beta)` | Evidence. Uniform `Beta(1,1)` is zero evidence, prior weight `W = 1`. | Independent fusion of the *same* Bernoulli: yes (`α' = αa+αb−W`). Product of two Betas: **not** Beta. |

**Combining evidence is not the same operation as propagating error.** Fusion
answers "these are two observations of the same coin." Arithmetic answers
"these are two quantities, possibly different, combined by `+` or `×`." Using
fusion for `a + b` of different measurements would add evidence that nobody
collected.

### 9.3.1 Question 1 — sum of two measurements

Operands `A`, `B` are independent measurements of **different** quantities.

- **GUM.** `value' = value_A + value_B`.
  `u' = √(u_A² + u_B²)`. Exact for the linear combination; no Jacobian
  remainder. Oracle golden: `1 ± 0.1` plus `2 ± 0.2` gives `u = 0.22360679774997896`
  (`√0.05`).
- **Evidence.** Not fusion. The sum is not more-confirmed than both operands.
  The proposed rule is the **derived AND**: treat the two confidences as
  independent Bernoulli reliabilities, take the product random variable
  `C' = C_A · C_B`, and moment-match the first two moments of that product
  to a Beta.

  Let `μ_i = α_i / n_i`, `n_i = α_i + β_i`, and `m_k(α,β) = (α)^{(k)} / (α+β)^{(k)}`
  (rising Pochhammer). Then

  ```
  μ  = m1(αA,βA) · m1(αB,βB)
  m2 = m2(αA,βA) · m2(αB,βB)
  var = m2 − μ²
  n' = μ(1−μ)/var − 1
  α' = μ · n'
  β' = (1−μ) · n'
  ```

  For `Beta(2,2) × Beta(3,1)` the oracle gives the exact rationals
  `α' = 13/7`, `β' = 65/21` (IEEE `1.8571428571428572`, `3.0952380952380953`).

  Independent fusion of those same parameters would have produced
  `α' = 2+3−1 = 4`, `β' = 2+1−1 = 2`. That is a different number, and it is
  the number the saboteur `eb_fuse_naive` / fusion-in-place-of-AND is there
  to be distinguished from.

### 9.3.2 Question 2 — product and non-linear functions

A Beta posterior is **not closed** under a non-linear transform of the
*value*. The evidence channel therefore cannot follow the value through the
Jacobian. The proposed rule keeps the two channels separate:

- **GUM, first-order, independent product.**
  `value' = value_A · value_B`.
  `u'² = value_B² · u_A² + value_A² · u_B²`.
  The exact independent variance of a product of independent random variables
  also contains `u_A² · u_B²`. That term is **dropped**. On the test point
  `(2 ± 0.1) × (3 ± 0.2)`:

  | quantity | oracle |
  |---|---|
  | GUM variance | `0.25000000000000006` |
  | exact independent variance | `0.25040000000000007` |
  | dropped term | `0.00040000000000000013` |
  | dropped / exact | `0.0015974440894568692` |

  Relative error of the GUM product variance on that point is **0.16 %**.
  `knowledge.sio` `ep_mul` drops the same term (it stores variance, so the
  formula is identical). This is JCGM 100:2008 equation (13) / first-order
  Taylor, not a Sounio invention.

- **Evidence.** The same derived AND as 9.3.1. The product of two Beta
  random variables is not Beta. Moment-matching the first two moments is
  exact for those two moments (in exact arithmetic). The error is the
  mismatch of higher moments; see 9.4.

- **General non-linear `f`.** Not specified. A first-order GUM Jacobian on
  `(value, u)` plus identity or derived-AND on `(α, β)` would be the
  mechanical extension. The error of that extension is **not quantified
  here**. That is a ruling owed, not a silent default.

### 9.3.3 Question 3 — shared origin

`x + x` is not the independent sum. GUM with correlation `ρ = 1` gives
`u' = 2 u` (equivalently `Var(2x) = 4 Var(x)`). Evidence is **identity**:
the same posterior, not a fused one.

Summing `Beta(1,1)` with itself is **not** `Beta(2,2)`. `Beta(2,2)` is the
naive add `αa+αb` (the saboteur). Independent fusion of two genuine
`Beta(1000,1000)` observations of the *same* Bernoulli *is* `Beta(1999,1999)`
(`αa+αb−W`); that operation exists in the file as `eb_fuse_independent` and
is **not** what `x + x` does.

| Operation on `Beta(1,1)` | `(α, β)` |
|---|---|
| `eb_add_same_origin` (the proposed `x+x`) | `(1, 1)` |
| `eb_fuse_naive` (saboteur) | `(2, 2)` |
| `eb_fuse_independent` (two genuine i.i.d. observations) | `(1, 1)` — zero evidence fused with zero evidence stays zero evidence |

| Operation on `Beta(1000,1000)` | `(α, β)` |
|---|---|
| `eb_add_same_origin` | `(1000, 1000)` |
| `eb_fuse_independent` | `(1999, 1999)` |

Test 16 asserts `same_origin.α ≠ naive.α`. Test 21 asserts
`same_origin.α ≠ fused.α`. If the rule is sabotaged to naive add or to
fusion, those tests fail. A witness that cannot fail has not measured.

Measured saboteur, Madaros v0.80.0, 2026-08-20: `eb_add_same_origin` rewritten
to `α' = 2α` (naive add of a record with itself). Honest run `45/45 PASS`.
Sabotaged run `39/45`, `run_rc=1`, failures `[12] [13] [16] [17] [18] [53]`.
Reverted; second honest run `45/45 PASS`.

### 9.3.4 Question 4 — the scalar as a limit

A scalar confidence `c` on `0..1000` together with a **declared** strength
`n = α+β` maps to

```
μ = c / 1000
α = μ · n
β = (1 − μ) · n
```

This is **not reversible**. `c` does not determine `n`:

| `c` | `n` | `(α, β)` |
|---|---|---|
| 500 | 2 | `Beta(1, 1)` — zero evidence |
| 500 | 2000 | `Beta(1000, 1000)` |

The same integer lifts to opposite epistemic claims. Therefore the migration
of the 26 scalar sites **cannot be mechanical** without a default-`n` ruling
(9.7). `eb_to_scalar` returns `1000 · α/(α+β)` and discards `n`; the round
trip is lossy by construction.

The axis is refused: `c = 0` or `c = 1000` would produce `α = 0` or `β = 0`,
which is not a Beta density. `eb_from_scalar` returns an invalid record
(`α = 0`) in those cases, and `eb_valid` is 0. `n ≤ 0` is likewise refused.

## 9.4 Quantified error of the approximations

### 9.4.1 GUM product (dropped `u_A² u_B²`)

On `(2 ± 0.1) × (3 ± 0.2)`: relative error of the first-order variance is
`1.597… × 10⁻³` (oracle). The Sounio test requires `dropped/exact < 0.002`.
GUM addition of independent terms has **no** dropped term.

### 9.4.2 Derived AND (product of Betas, moment-matched)

First two moments match exactly in `fractions.Fraction`. Third- and
fourth-moment relative errors, oracle, IEEE print of the exact rationals:

| factors | `α'` (exact) | `β'` (exact) | rel₃ | rel₄ |
|---|---|---|---|---|
| `Beta(2,2) × Beta(3,1)` | `13/7` | `65/21` | `1.370 × 10⁻³` | `3.762 × 10⁻³` |
| `Beta(2,2) × Beta(2,2)` | `16/11` | `48/11` | `5.814 × 10⁻³` | `1.565 × 10⁻²` |
| `Beta(1,1) × Beta(1,1)` | `5/7` | `15/7` | `6.536 × 10⁻³` | `1.562 × 10⁻²` |
| `Beta(5,1) × Beta(1,5)` | `185/187` | `1147/187` | `1.396 × 10⁻³` | `3.868 × 10⁻³` |
| `Beta(2,8) × Beta(8,2)` | `376/191` | `1974/191` | `1.171 × 10⁻³` | `3.557 × 10⁻³` |
| `Beta(1000,1000) × Beta(1000,1000)` | `3002000/4003` | `9006000/4003` | `1.658 × 10⁻⁷` | `6.620 × 10⁻⁷` |

Small-`n` third-moment relative error is **0.14 % – 0.65 %**. At `n = 2000`
it is `~1.7 × 10⁻⁷`. The Sounio tests require `rel₃ < 0.002` on the Q1 point
and `rel₃ < 0.01` on the Q2 point; both goldens sit inside those bounds.

This bound is for **independent** Beta factors and for **moment-matching
to a Beta**. It is not a bound on a Jacobian through an arbitrary `f`, and
it is not a bound on correlated factors.

## 9.5 Named edges

| Edge | Rule | Witness |
|---|---|---|
| Zero evidence | `Beta(1,1)`. Fusion of two zero-evidence records stays `Beta(1,1)` (`1+1−W`). | tests 10–13, 54–55 |
| Huge evidence | `α = β = 10⁶` remains valid; `x+x` does not inflate it. | tests 52–53 |
| `α = 0` or `β = 0` | Not a Beta. `eb_valid` is 0. Construction via `eb_from_scalar` at `c ∈ {0,1000}` refuses the same way. | tests 48–51 |
| Shared origin | `x+x` identity on `(α,β)`; GUM `ρ = 1` on `u`. Distinct from naive add and from fusion. | tests 10–21 |

## 9.6 What this does not claim

- It does not claim that `EpistemicBeta` is now `Knowledge<T>`, nor that the
  compiler IR emitters implement this rule.
- It does not claim that Madaros and lean_single agree on these numbers. The
  witness is Madaros (`bin/souc`, default engine). lean_single was not run.
- It does not claim that the 26 scalar sites can be rewritten by a script.
  Question 4 says they cannot, until 9.7 (default `n`) is ruled.
- It does not claim that derived AND is the unique correct evidence rule for
  `a + b`. It is the proposed rule; 9.1 is empty.
- It does not claim that a Beta remains a Beta under a non-linear transform
  of the value. It claims the opposite, and then approximates the *evidence*
  channel by derived AND, with the table in 9.4 as the error.
- It does not claim Dempster–Shafer (`stdlib/epistemic/fusion.sio`) or
  inverse-variance `ep_merge`. Those are different operations on different
  objects.
- It does not migrate any caller.

## 9.7 Rulings owed

- **Default `n` for a scalar lift.** Without it the 26-site migration is not
  a function. Candidate answers (none adopted): `n = 2` (every existing `c`
  becomes near-zero evidence); `n = 1000` (a `c = 500` becomes
  `Beta(500,500)`); refuse to lift and require every site to declare `n`.
- **Is derived AND the evidence rule for arithmetic of different
  quantities?** The alternative "leave `(α,β)` as `min` of the two, in some
  order-statistic on `n`" is coherent and cheaper. It has no oracle in this
  delivery.
- **Partial correlation.** `ρ = 0` and `ρ = 1` are written. `0 < ρ < 1` is
  not. GUM has a formula; the evidence channel does not.
- **Does `emit_variance_independent_product` have to match 9.3.2?** The IR
  already drops the same second-order term. Whether that is a specification
  of the language or an implementation accident is owed. This section does
  not decide for the compiler.
- **Non-linear `f` beyond `+` and `×`.** Error of "GUM Jacobian on `(value,u)`,
  derived AND or identity on `(α,β)`" is unquantified.
- **Which engine is the oracle for §9?** The frame marks the cell `contested`.
  This delivery ran Madaros only.

## Claims Forbidden

- Do not read 9.3 as a description of `knowledge.sio`. That file still decays
  an integer.
- Do not read 9.1 as filled. A proposed rule with tests is not a ruling.
- Do not report a scalar `c` as equivalent to a unique `(α, β)`.
- Do not describe fusion of evidence and GUM propagation of error as the
  same operation.
- Do not claim the 26-site migration is mechanical.
- Do not cite this section as closing the frame's `contested` cell. The cell
  is about two engines and an IR emitter; this section did not run the seed
  and did not edit `ir/lower.sio`.

<!-- docs:meta
topic_id: repo.docs.research.paper-a-section2-draft-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-section2-draft-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — §2 *The defect class, by example* (full draft, 2026-08-25)

> Draft prose for §2 of the anti-garbling paper. All code is verbatim from
> `stdlib/epistemic/knowledge.sio` (line numbers current as of 2026-08-25).
> Notation: an *epistemic value* is a triple `(val, variance, confidence)`; we write
> `x = (m, v, ·)` for a value with mean `m` and variance `v = Var(X)`, and use `Var`
> for the true variance of the underlying random variable.

---

## 2. The defect class, by example

The uncertainty library we study is not careless. Every arithmetic operation in
`knowledge.sio` is annotated with the exact GUM propagation formula it implements, and
each formula is *correct* — under one assumption stated nowhere in the type and checked
nowhere in the code: that the operands are **statistically independent**. The library
ships the independence special case of each GUM formula as if it were the general law.
Where the assumption fails, the propagated variance is not merely imprecise; it is
biased in a specific, dangerous direction — *downward*. The program reports more
precision than its inputs justify.

We make this concrete with four operations from a single file, then show why the bias
survives testing, and close with the realistic case that motivates the whole paper.

### 2.1 The same operation, two variances

Consider squaring an epistemic value. The library offers two ways to compute `x · x`:

```sio
// GUM delta method: Var(XY) ≈ Y²·Var(X) + X²·Var(Y)          knowledge.sio:110
pub fn ep_mul(a: &Epistemic, b: &Epistemic) -> Epistemic {
    let new_var = b.val * b.val * a.variance + a.val * a.val * b.variance   // :112
    ...
}

// Var(X²) ≈ 4X²·Var(X)                                        knowledge.sio:150
pub fn ep_square(a: &Epistemic) -> Epistemic {
    Epistemic {
        val: a.val * a.val,
        variance: 4.0 * a.val * a.val * a.variance,             // :154
        ...
    }
}
```

Evaluate both on the same `x = (m, v, ·)`:

| Expression | Formula applied | Variance |
|---|---|---|
| `ep_mul(&x, &x)` | `b²·Var(a) + a²·Var(b)` with `a = b = x` | `m²v + m²v = 2m²v` |
| `ep_square(&x)` | `4·a²·Var(a)` | `4m²v` |

`ep_square` is right: for `Y = X²`, the delta method gives `Var(Y) ≈ (dY/dX)²·Var(X) =
(2X)²·Var(X) = 4m²v`. `ep_mul(&x, &x)` returns **half** of that. The discrepancy is not
rounding — it is the entire covariance term. The general delta-method formula for a
product is

    Var(XY) ≈ Y²·Var(X) + X²·Var(Y) + 2XY·Cov(X, Y).

`ep_mul` drops the last term. When `X` and `Y` are independent, `Cov = 0` and the drop
is exact. When `Y` *is* `X`, `Cov(X, X) = Var(X) = v`, the missing term is
`2·m·m·v = 2m²v`, and the two formulas differ by exactly that amount:
`4m²v − 2m²v = 2m²v`. The library computes both, exposes both, and **nothing routes
`x · x` to the sound one**: `ep_mul(&x, &x)` is legal, well-typed, and understates the
variance of a squared quantity by a factor of two.

### 2.2 The asymmetry the frame predicts

The same omission runs through addition and subtraction, and it produces a *directional*
signature that is itself evidence the diagnosis is right.

```sio
// GUM: Var(X+Y) = Var(X) + Var(Y)                            knowledge.sio (ep_add)
pub fn ep_add(a: &Epistemic, b: &Epistemic) -> Epistemic {
    ... variance: a.variance + b.variance, ...                 // :96
}

// GUM: Var(X-Y) = Var(X) + Var(Y)                            knowledge.sio:101
pub fn ep_sub(a: &Epistemic, b: &Epistemic) -> Epistemic {
    ... variance: a.variance + b.variance, ...                 // :105
}
```

Both add the operand variances. The general laws are `Var(X±Y) = Var(X) + Var(Y) ±
2·Cov(X, Y)`: addition *adds* twice the covariance, subtraction *subtracts* it. Evaluate
each on maximally-correlated operands (`Y = X`, `Cov = v`):

| Expression | Library | True (`ρ = 1`) | Error | Direction |
|---|---|---|---|---|
| `ep_add(&x, &x)` = `2x` | `2v` | `Var(2X) = 4v` | `−2v` | **understates** — unsound |
| `ep_sub(&x, &x)` = `0` | `2v` | `Var(0) = 0` | `+2v` | overstates — merely conservative |
| `ep_mul(&x, &x)` = `x²` | `2m²v` | `4m²v` | `−2m²v` | **understates** — unsound |
| `ep_square(&x)` = `x²` | `4m²v` | `4m²v` | `0` | sound |

The asymmetry is the tell. Correlated addition and multiplication **understate** — they
manufacture precision, the failure that matters. Correlated subtraction **overstates** —
it is wrong, but wrong in the safe direction, reporting *more* uncertainty than the truth.
A library that were simply buggy would err in both directions at random. This one errs in
exactly the direction predicted by the sign of the dropped covariance term: `+2Cov` for
add, `−2Cov` for sub. §4 names this the anti-garbling signature — an operation may only
*lose* information (overstate uncertainty), never *create* it (understate) — and the
add/sub split is that criterion read off the source.

> A note on the `* 99 / 100`, `* 98 / 100` confidence multipliers (`:97`, `:116`, …):
> these decay a separate integer `confidence` field and are orthogonal to the variance
> defect. They are also heuristic rather than derived (a limitation we return to in §10),
> but they neither cause nor mask the understatement analyzed here.

### 2.3 Why tests do not catch it

An understated variance is invisible to ordinary testing for a structural reason: **the
wrong answer is the more attractive one.** A tighter error bar reads as a better result.
Nothing about `ep_add(&x, &x)` returning `2v` instead of `4v` looks like a failure — the
value `2x` is exact, the confidence field is populated, the number is simply *more
precise*. A regression test asserting "the answer is within tolerance" passes; a test
asserting "the uncertainty is not too large" passes; only a test that already knows the
*correct* larger variance — i.e. a test written by someone who has already spotted the
bug — fails. The library's own test suite (`knowledge.sio:290–310`) checks `ep_add`,
`ep_sub`, and `ep_mul` on **independent** operands, where every formula is exact, and so
certifies nothing about the correlated case.

The failure is therefore not a coding slip that better testing would have caught. It is a
**missing precondition**: the operations are sound on a domain (independent operands) that
the type system neither records nor enforces, and are silently applied outside it. This is
precisely the shape a type discipline can address and testing cannot — the defect is a
property of *which values may flow into which operation*, not of any single execution.

### 2.4 The instance that matters: shared provenance in a PBPK model

The correlated case is not exotic. It is the *normal* case whenever two quantities are
computed from a common measurement — which, in a physiologically-based pharmacokinetic
(PBPK) model, is every pair of compartment states, because they all descend from the same
measured clearance and volume parameters.

Consider a two-compartment model reporting total drug exposure as the sum of central and
peripheral AUC:

```
auc_total = ep_add(&auc_central, &auc_peripheral)
```

Both `auc_central` and `auc_peripheral` are derived, through the model's ODE solution,
from the *same* measured elimination rate constant `k`. Their errors are strongly
positively correlated: an overestimate of `k` pushes both AUCs the same way. The true
variance of the sum includes `+2·Cov(auc_central, auc_peripheral) > 0`; `ep_add`
discards it. Every inter-compartmental sum in the model therefore reports a variance
smaller than the truth, and the understatement compounds with each additional
compartment.

The clinical consequence is the opposite of academic. The dissertation's therapeutic-drug
-monitoring result (vancomycin, rapamycin) issues a `WARN` when the *upper* credible bound
on exposure crosses into toxicity even though the point estimate reads safe. A variance
that is understated at every summation **shrinks that credible bound**, and the `WARN`
that should fire — the entire safety value of propagating uncertainty — does not. The
defect does not just weaken a number in a table; it silences the alarm the system exists
to raise. §8 quantifies this on the real model; here it fixes the stakes: manufacturing
precision in an uncertainty library is not a cosmetic inaccuracy but a safety failure, and
the operation that commits it type-checks today.

---

### Verbatim-source appendix (for the artifact / referee)

| Item | Location | Content |
|---|---|---|
| `ep_add` variance | `stdlib/epistemic/knowledge.sio:96` | `a.variance + b.variance` |
| `ep_sub` variance | `:105` | `a.variance + b.variance` |
| `ep_mul` variance | `:112` | `b.val*b.val*a.variance + a.val*a.val*b.variance` |
| `ep_square` variance | `:154` | `4.0 * a.val * a.val * a.variance` |
| GUM comments (intent) | `:101, :110, :150` | correct formulas, independence implicit |
| tests use independent operands only | `:290–310` | correlated case untested |

**Numeric verification** (for `x = (m, v, ·)`): `ep_add(&x,&x).variance = v+v = 2v` vs
`Var(2X)=4v`; `ep_mul(&x,&x).variance = m²v+m²v = 2m²v` vs `ep_square(&x).variance =
4m²v`; understatement gaps `2v` and `2m²v` equal `2·Cov(X,X)` and `2·m²·Cov(X,X)`
respectively — the exact dropped covariance terms.

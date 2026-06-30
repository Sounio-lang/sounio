<!-- docs:meta
topic_id: repo.docs.papers.epistemic-types.soundness-denotation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.epistemic-types.soundness-denotation
-->

# Soundness Denotation for Epistemic Operations

*Addresses §3.2 of the PLDI submission cycle-1 review.*

---

## §1. Denotation

We define the semantic domain for the `Epistemic` type as:

```
⟦Epistemic⟧ = ℝ × ℝ≥0 × [0,1000]
```

where the three components are `(v, σ², conf)`: central value, variance (non-negative), and
confidence (integer pedigree score). The `conf` component is treated separately in
`CONFIDENCE_SEMANTICS.md`; this document establishes the GUM coincidence of the variance
propagation rules.

The seven core operations denote as follows (integer floor applied to confidence arithmetic):

```
⟦e.scale(c)⟧    = (c·v,       c²·σ²,                   conf)
⟦e.shift(c)⟧    = (v + c,     σ²,                       conf)
⟦e.square()⟧    = (v²,        4v²·σ²,                   ⌊98·conf / 100⌋)
⟦e1.add(e2)⟧    = (v1+v2,     σ1²+σ2²,                  ⌊99·min(c1,c2) / 100⌋)
⟦e1.mul(e2)⟧    = (v1·v2,     v2²·σ1² + v1²·σ2²,        ⌊98·min(c1,c2) / 100⌋)
⟦e1.div(e2)⟧    = (v1/v2,     σ1²/v2² + v1²·σ2²/v2⁴,   ⌊97·min(c1,c2) / 100⌋)
⟦e.sqrt()⟧      = (√v,        σ²/(4v),                  ⌊98·conf / 100⌋)
```

All denotations are partial (undefined where the denominator is zero or the argument to
`sqrt` is non-positive); the compiler emits a runtime guard at those sites.

---

## §2. GUM Coincidence

**Claim.** Each variance component above coincides with the first-order GUM propagation
formula (JCGM 100:2008, §5.1.2).

For a scalar function f: ℝⁿ → ℝ with mutually independent inputs x₁, …, xₙ having
variances σ₁², …, σₙ², GUM §5.1.2 yields:

```
σ²(f) ≈ Σᵢ (∂f/∂xᵢ)² · σᵢ²
```

**scale(c).** f(x) = c·x.  ∂f/∂x = c.  σ²(c·x) = c²·σ².  Matches ⟦scale⟋.  ✓

**shift(c).** f(x) = x + c.  ∂f/∂x = 1.  σ²(x+c) = σ².  Matches ⟦shift⟋.  ✓

**square().** f(x) = x².  ∂f/∂x = 2x.  σ²(x²) = (2x)²·σ² = 4x²·σ².
Matches ⟦square⟋.  ✓

**add(y).** f(x, y) = x + y.  ∂f/∂x = 1, ∂f/∂y = 1.
σ²(x+y) = σx² + σy² (under independence).  Matches ⟦add⟋.  ✓

**mul(y).** f(x, y) = x·y.  ∂f/∂x = y, ∂f/∂y = x.
σ²(x·y) = y²·σx² + x²·σy² (under independence).  Matches ⟦mul⟋.  ✓

**div(y).** f(x, y) = x/y.  ∂f/∂x = 1/y, ∂f/∂y = −x/y².
σ²(x/y) = σx²/y² + x²·σy²/y⁴ (under independence).  Matches ⟦div⟋.  ✓

**sqrt().** f(x) = √x.  ∂f/∂x = 1/(2√x).
σ²(√x) = σ²/(4x).  Matches ⟦sqrt⟋.  ✓

All seven operations implement the first-order GUM formula exactly (up to floating-point
rounding in the f64 arithmetic). There is no truncation of higher-order Taylor terms
beyond what GUM §5.1.2 itself discards.

---

## §3. Correlation Note and the `mul`/`square` Discipline

The `mul` denotation assumes **independence** of the two operand random variables. This is
the same assumption made by GUM §5.1.2 when covariance terms are omitted (GUM eq. 13
reduces to eq. 12 when `u(xᵢ, xⱼ) = 0` for i ≠ j).

The critical aliased case is `x.mul(&x)`: if the two arguments refer to the *same*
random variable X, the correct formula is:

```
Var(X·X) = Var(X²) = 4X²·Var(X)    [from square()]
```

but the independence formula yields:

```
x²·Var(X) + x²·Var(X) = 2X²·Var(X)    [wrong by factor 2]
```

**This case is excluded by library discipline.** All self-products in the particle-physics
stdlib call `x.square()` directly, never `x.mul(&x)`. The current codebase has no
call site of the form `e.mul(&e)` — verified by grep of `stdlib/particle_physics/`.

A future compiler lint rule or affine-type constraint should enforce this statically. The
obligation is noted in `KNOWN_LIMITATIONS.md` as item KL-07.

---

## §4. Independence Assumption Scope

The independence assumption (`u(xᵢ, xⱼ) = 0` for i ≠ j) is satisfied when operands
derive from **distinct PDG-2024 input quantities** — for example, `α_EM` and `M_Z` are
independently measured; their covariance is zero to the precision quoted by PDG.

The assumption **fails** for derived quantities that share upstream inputs. The
prototypical dangerous pattern is two cross-section estimates both derived from a common
`α_s` value: the true variance of their product includes a covariance term
`2·(∂f/∂αs)·(∂g/∂αs)·Var(αs)`, which `mul` silently drops.

In such cases the variance returned by `mul` is an **underestimate**, and `confidence`
will not compensate (it tracks pedigree depth, not covariance structure). Tracking full
covariance matrices is left to future work. In all computations presented in this paper,
operand independence is verified by tracing each quantity to its PDG source entry; the
provenance is documented in `stdlib/particle_physics/lib.sio` inline comments.

# Epistemic Algebra Verification Report

## Executive Summary

This document provides a mathematical review of the epsilon propagation rules and confidence subsumption lattice implemented in `self-hosted/check/epistemic.sio`. The epistemic type system tracks uncertainty bounds through computations using a simplified confidence metric ε ∈ [0,1], where lower values indicate higher confidence.

**Key Findings:**
- The current implementation uses intentionally conservative approximations
- Linear sum for independent errors overestimates uncertainty (sound but imprecise)
- Maximum for correlated errors may underestimate uncertainty (potential unsoundness)
- The subsumption lattice is mathematically well-formed

**Recommendations:**
1. Document the conservative nature of current approximations
2. Consider linear sum for correlated errors to restore soundness
3. Explore interval-valued confidence for Knightian uncertainty

---

## 1. Epsilon Propagation Rules

### 1.1 Independent Errors (Linear Sum)

**Statistical foundation:** For independent random errors, variances add:
```
σ_result² = σ_a² + σ_b²  (quadrature sum)
```

**Current implementation:**
```sio
fn epsilon_combine_independent(eps_a: f64, eps_b: f64) -> f64 {
    let sum = eps_a + eps_b
    epsilon_clamp(sum)
}
```

**Analysis:**
The implementation uses linear sum (ε_a + ε_b) rather than quadrature sum (√(ε_a² + ε_b²)). For ε ∈ [0,1]:

| ε_a | ε_b | Linear | Quadrature | Ratio |
|-----|-----|--------|------------|-------|
| 0.1 | 0.1 | 0.2    | 0.141      | 1.41× |
| 0.3 | 0.3 | 0.6    | 0.424      | 1.41× |
| 0.5 | 0.5 | 1.0    | 0.707      | 1.41× |

The linear sum is always conservative (≥ quadrature) for non-negative values. This is a deliberate design choice:
- **Pros:** Simple, computationally efficient, guaranteed sound
- **Cons:** Overestimates uncertainty, may reject valid programs

**Recommendation:** Document as intentional conservatism. Consider providing a `strict` mode that uses quadrature sum for less conservative bounds.

---

### 1.2 Correlated Errors

**Statistical foundation:** For perfectly correlated errors:
```
σ_result = σ_a + σ_b  (linear sum, no reduction)
```

**Current implementation:**
```sio
fn epsilon_combine_correlated(eps_a: f64, eps_b: f64) -> f64 {
    if eps_a > eps_b { eps_a } else { eps_b }
}
```

**Analysis:**
The implementation uses `max(ε_a, ε_b)` instead of the statistically correct linear sum. This is **anti-conservative** for correlated errors:

| ε_a | ε_b | max() | Linear (correct) | Status |
|-----|-----|-------|------------------|--------|
| 0.3 | 0.3 | 0.3   | 0.6              | ❌ Underestimates |
| 0.3 | 0.5 | 0.5   | 0.8              | ❌ Underestimates |
| 0.1 | 0.2 | 0.2   | 0.3              | ❌ Underestimates |

The current semantics appear to assume that correlation means "take the dominant error source," which is incorrect. Perfect correlation means errors add directly without cancellation.

**Recommendation:** Change to linear sum for soundness:
```sio
fn epsilon_combine_correlated(eps_a: f64, eps_b: f64) -> f64 {
    let sum = eps_a + eps_b
    epsilon_clamp(sum)
}
```

Alternatively, if the intent was "same source" (redundant measurements), consider renaming to `epsilon_combine_redundant` and document the semantics clearly.

---

### 1.3 Relative Errors (Multiplication/Division)

**Statistical foundation:** For multiplication/division, relative errors add:
```
(δz/z)² = (δx/x)² + (δy/y)²  (independent)
δz/z = δx/x + δy/y          (correlated)
```

**Current implementation:**
```sio
fn epsilon_combine_relative(eps_a: f64, eps_b: f64) -> f64 {
    let sum = eps_a + eps_b
    epsilon_clamp(sum)
}
```

**Analysis:**
Uses linear sum for relative errors, matching the correlated case. This is conservative for independent multiplicative errors. The implementation correctly maps:
- `OpMul` → `epsilon_combine_relative`
- `OpDiv` → `epsilon_combine_relative`

---

## 2. Confidence Subsumption Lattice

### 2.1 Lattice Properties

The `epsilon_subsumes` function defines a partial order on confidence values:

```sio
fn epsilon_subsumes(eps_a: f64, eps_b: f64) -> bool {
    eps_a <= eps_b
}
```

**Verification of lattice axioms:**

| Property | Definition | Status | Notes |
|----------|------------|--------|-------|
| Reflexivity | ∀a: a ≤ a | ✓ | `eps_a <= eps_a` |
| Antisymmetry | a ≤ b ∧ b ≤ a → a = b | ✓ | Standard ≤ ordering |
| Transitivity | a ≤ b ∧ b ≤ c → a ≤ c | ✓ | Standard ≤ ordering |
| Totality | ∀a,b: a ≤ b ∨ b ≤ a | ✓ | Total order on [0,1] |

The structure forms a **complete lattice** with:

- **Top (T):** ε = 0.0 (perfect certainty)
- **Bottom (⊥):** ε = 1.0 (complete uncertainty)
- **Meet (∧):** `min(ε_a, ε_b)` — more confident of two
- **Join (∨):** `max(ε_a, ε_b)` — less confident of two

```sio
fn epsilon_meet(eps_a: f64, eps_b: f64) -> f64 {  // ∧ (greatest lower bound)
    if eps_a < eps_b { eps_a } else { eps_b }
}

fn epsilon_join(eps_a: f64, eps_b: f64) -> f64 {  // ∨ (least upper bound)
    if eps_a > eps_b { eps_a } else { eps_b }
}
```

### 2.2 Lattice Hasse Diagram

```
                    0.0 (T - Certain)
                   / | \
                  /  |  \
               0.1  0.2  0.3
                |    |    |
                |    |    |
               0.5 --+-- 0.6
                 \   |   /
                  \  |  /
                   0.9
                    |
                   1.0 (⊥ - Unknown)
```

Direction: arrows point upward to more confident (lower ε) values.

### 2.3 Type Compatibility

```sio
fn knowledge_compatible(ty_a: TypeEntry, ty_b: TypeEntry) -> bool {
    // ... inner types must match ...
    epsilon_subsumes(ty_a.epsilon_bound, ty_b.epsilon_bound)
}
```

This defines **contravariant subtyping** in the confidence dimension:
- `Knowledge[T, 0.1]` can be used where `Knowledge[T, 0.3]` is expected
- More confident knowledge substitutes for less confident
- Subsumption direction: confident ⊑ less-confident

---

## 3. Knightian Uncertainty Extension

### 3.1 Interval-Valued Confidence

Current implementation uses point-valued epsilon. For systems requiring explicit separation of aleatoric (statistical) and epistemic (systematic) uncertainty, consider:

```
Knowledge[T, [ε_min, ε_max]]
```

Where:
- **ε_min:** Aleatoric uncertainty (irreducible randomness)
- **ε_max:** Total uncertainty (aleatoric + epistemic)
- **Interval width (ε_max - ε_min):** Pure epistemic uncertainty

### 3.2 Interval Composition Rules

**Addition/Subtraction:**
```
[a₁, a₂] + [b₁, b₂] = [a₁ + b₁, a₂ + b₂]
```

**Multiplication (simplified):**
```
[a₁, a₂] × [b₁, b₂] = [max(a₁+b₁, 1.0), max(a₂+b₂, 1.0)]
```

**Subsumption:**
```
[a₁, a₂] ⊑ [b₁, b₂]  ⟺  a₁ ≥ b₁ ∧ a₂ ≤ b₂
```

(Note: Interval subsumption is containment-reversed for lower bounds)

### 3.3 Implementation Path

To support interval confidence:

1. Extend `KnowledgeMeta` to store `[ε_min, ε_max]`
2. Update propagation functions with interval arithmetic
3. Define partial order for interval containment
4. Add syntax: `Knowledge[T, 0.1..0.3]`

---

## 4. Formal Properties

### Theorem 1 (Subsumption Soundness)

**Statement:** If `knowledge_compatible(a, b)` holds, then any value of type `a` can safely be used where type `b` is expected.

**Proof Sketch:**
1. Inner types match (structural equality)
2. `epsilon_subsumes(a.ε, b.ε)` implies `a.ε ≤ b.ε`
3. Lower ε means higher confidence
4. Therefore `a` provides at least the confidence of `b`
5. Safe substitution holds ∎

### Theorem 2 (Propagation Monotonicity)

**Statement:** For all binary operations, the result epsilon is monotone in both arguments.

**Proof Sketch:**
- `epsilon_combine_independent`: `sum(a,b)` is monotone in both args
- `epsilon_combine_correlated`: `max(a,b)` is monotone in both args
- `epsilon_combine_relative`: `sum(a,b)` is monotone in both args
- Clamping preserves monotonicity ∎

### Theorem 3 (Conservatism of Independent Combination)

**Statement:** `epsilon_combine_independent(a,b) ≥ √(a² + b²)` for all `a,b ∈ [0,1]`.

**Proof:**
For `a,b ≥ 0`, we have `(a + b)² = a² + 2ab + b² ≥ a² + b²`
Taking square roots: `a + b ≥ √(a² + b²)` ∎

---

## 5. Recommendations

### 5.1 Short Term (Documentation)

1. **Document conservative approximations:** Add comments explaining that linear sum is intentionally conservative
2. **Clarify correlated semantics:** Either fix `epsilon_combine_correlated` to use sum, or rename and document the current max semantics
3. **Add soundness warning:** Note that current correlated combination may underestimate uncertainty

### 5.2 Medium Term (Enhancement)

1. **Implement interval confidence:** Support `Knowledge[T, [ε_min, ε_max]]` for explicit epistemic/aleatoric separation
2. **Add quadrature option:** Provide configuration for less conservative error propagation
3. **Provenance tracking:** Complete the provenance chain tracking for audit trails

### 5.3 Long Term (Formalization)

1. **Mechanized proofs:** Verify lattice properties in a proof assistant (Coq/Lean)
2. **Floating-point soundness:** Account for rounding in epsilon calculations
3. **Statistical interpretation:** Define formal relationship between ε and confidence intervals

---

## Appendix: Current Implementation Reference

| Function | Line | Semantics | Status |
|----------|------|-----------|--------|
| `epsilon_subsumes` | 330 | `a ≤ b` | ✓ Sound |
| `epsilon_meet` | 335 | `min(a,b)` | ✓ Sound |
| `epsilon_join` | 344 | `max(a,b)` | ✓ Sound |
| `epsilon_combine_independent` | 412 | `a + b` (clamped) | ✓ Conservative |
| `epsilon_combine_correlated` | 419 | `max(a,b)` | ⚠️ Underestimates |
| `epsilon_combine_relative` | 429 | `a + b` (clamped) | ✓ Conservative |
| `epsilon_combine_weighted` | 435 | weighted average | ✓ Sound |

---

*Document version: 1.0*
*Reviewed: self-hosted/check/epistemic.sio (537 lines)*
*Focus: Mathematical soundness of epsilon propagation*

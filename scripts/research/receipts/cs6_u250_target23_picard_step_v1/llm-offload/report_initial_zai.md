# Review: CS6 U250 target-23 Picard certificate

```
[WRONG] Four-dimensional field definition: x'=2y²−xy, y'=xy−y(w+zs)/2, w'=xy−w−zs, ℓ'=x−y−(w+zs)/2−1
  `zs` appears in three equations but is never defined. If constant, its value is missing; if state-dependent, the Jacobian (and hence the contraction bound) is computed on an unspecified field.
  Correction: state the value of zs, or replace it with the intended symbol/expression.
```

```
[OK] Picard operator hull(B, X₀+[0,h]·F(B)) iterated monotonically to a fixed point
  Standard interval Picard-Lindelöf set operator; hull preserves inclusion-monotonicity, fixed point yields a pre-image (self-map) enclosure.
```

```
[OK] Certificate requires strict self-map (T(B)⊊B all 4 axes) AND h·‖J(B)‖_∞ < 1
  These are the correct two Banach fixed-point conditions for one-step existence/uniqueness on box B. Row-sum bound is a valid (not tightest) bound on the induced ∞-norm.
```

```
[OK] Row-sum ‖J‖_∞ as outward-rounded interval extension of |∂fᵢ/∂xⱼ| summed per row
  Correct for polynomial field; interval evaluation of each partial on B, absolute value, sum, max over rows.
```

```
[OK] Contraction raw F96 value 12543560845867825682829769920 → real ≈ 0.15832199621160117
  12543560845867825682829769920 / 2⁹⁶ ≈ 1.2544×10²⁸ / 7.9228×10²⁸ ≈ 0.15832; consistent to displayed digits. Value < 1 as required.
```

```
[OK] Self-map margin 13362935843645108892 raw F96 units is strictly positive
  ≈1.686×10⁻¹⁰ in real terms; positive ⇒ strict containment on all axes, consistent with the claim. Magnitude (~10⁻¹⁰) is plausible given the 2⁶⁴-ULP inflation (2⁻³² ≈ 2.33×10⁻¹⁰ in real terms) applied post-fixed-point.
```

```
[OK] h = 2⁻⁸; contraction implies ‖J‖_∞ ≈ 0.1583/h ≈ 40.5
  Internally consistent with a polynomial vector field on a moderate box; no issue.
```

```
[OK] S1.I31.F96 = 1 sign + 31 integer + 96 fractional bits = 128-bit signed fixed-point
  Correct format description; LSB = 2⁻⁹⁶.
```

```
[OK] ℓ is dynamically decoupled (does not appear in x′, y′, w′)
  Jacobian column 4 is zero for rows 1–3 (assuming zs is a constant). This is a structural property, not an error, but it means the contraction on the ℓ-axis is trivial and the effective coupling is 3D→ℓ.
```

```
[OVERREACH] "Four cases agree on all 88 output words" is stated without decomposing the 88-word allocation
  88 words / 4 cases = 22 words/case = 5.5 × 128-bit values, which does not cleanly map to {4 × 2 endpoints + margin + bound = 10 words} per case. The mapping should be specified to make the agreement claim falsifiable.
```

```
[OK] Claim boundary is honest: one-step Picard self-map + contraction only, no Taylor remainder, no orbit, no Poincaré, no global result
  Appropriately scoped; no mathematical overclaim about what the certificate proves.
```

---

**Summary.** The interval-Picard framework, contraction test, and F96 arithmetic are mathematically sound. The **single critical defect** is the undefined `zs` in three of four field equations — without it the field, Jacobian, and contraction bound cannot be independently reproduced. The output-word accounting is under-specified but does not affect correctness of the math. All other claims are consistent.

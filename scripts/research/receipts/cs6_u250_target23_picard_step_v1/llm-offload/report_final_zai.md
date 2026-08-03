# Math Review

## Claims checked

```
[OK] Vector field F: ℝ⁴→ℴ⁴ defined by x'=2y²−xy, y'=xy−y(w+zs)/2, w'=xy−w−zs, ℓ'=x−y−(w+zs)/2−1
  Polynomial; locally Lipschitz on bounded boxes. No structural issue.

[OK] Jacobian infinity-norm / row-sum approach for contraction certificate
  J rows: (−y, 4y−x, 0, 0); (y, x−(w+zs)/2, −y/2, 0); (y, x, −1, 0); (1, −1, −1/2, 0).
  Row-sum of absolute values is the correct sup-norm Lipschitz constant L on a box B.

[OK] Sufficient conditions: strict self-map (Picard image ⊂ B) + h·‖J(B)‖∞ < 1
  Standard Picard–Lindelöf / interval-ODE pair: self-map gives existence (Brouwer), contraction gives uniqueness. Correctly stated and correctly separated.

[OK] h = 2⁻⁸ = 1/256, contraction factor ≈ 0.15832199621160117 < 1
  Implied ‖J(B)‖∞ ≈ 40.5. Consistent with quadratic field on a compact box. Pass condition satisfied.

[OK] F96 raw-to-real for contraction bound
  12543560845867825682829769920 / 2⁹⁶ = 12543560845867825682829769920 / 79228162514264337593543950336 ≈ 0.15832200. Matches stated real value to 8 sig figs.

[OK] Strict self-map margin 13362935843645108892 raw F96 ≈ 1.69×10⁻¹⁰ real
  Positive on all four axes ⟹ strict containment. Small but valid.

[OK] Output word budget: 8 VF-endpoint + 8 Picard-endpoint + 4 row-sum + 1 contraction + 1 status = 22 per case; 4×22 = 88. Sums correctly.

[OK] Fixed-point iteration "hull(B, X₀+[0,h]F(B))" then inflate-by-2⁶⁴, then re-verify strict containment
  Standard inflation / epsilon-inflation (Miranker) technique. Growing phase until T(B)=B, then inflate for strict margin, then independent re-check. Mathematically sound.

[OK] Outward-rounded row-sum bound for ‖J(B)‖∞
  Correct directed-rounding practice for a rigorous certificate.

[OK] Claim boundary honesty
  Explicitly disclaims Taylor remainder, state advance, orbit/leaf coverage, Poincaré isolation, novelty. No overreach detected.
```

## Minor notes (not errors)

```
[TIGHTENABLE] Notation "hull(B, X₀+[0,h]F(B))" is ambiguous about whether B is the iteration variable or a fixed outer container.
  Recommend writing "iterate B_{n+1} = hull(B_n, X₀+[0,h]·F(B_n)) until B_{n+1}=B_n, then set B := inflate(B*, 2⁶⁴)" to disambiguate grow-phase from verify-phase.

[TIGHTENABLE] Jacobian row-sum upper bounds (4 words) are not given numerically, so the max-of-four and the product h·max cannot be independently cross-checked from this artifact alone.
  Not an error — just unverifiable from the text. The transcript receipt presumably contains them.
```

## Verdict

No mathematical errors found. The certificate structure (self-map + contraction under outward-rounded interval arithmetic, F96 scaling, word-budget arithmetic) is internally consistent and uses standard validated-ODE methodology correctly. The disclaimers are accurate and appropriately scoped.

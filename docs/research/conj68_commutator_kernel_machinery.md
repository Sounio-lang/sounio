# Conjecture 6.8: the commutator-kernel machinery (ker ad_u) — a new tool, and what it shows

**2026-09-01.** Built the exact commutator operator ad_u = L_u − R_u for 𝕊 — the
commutativity analog of the repo's ker(L_u) orthogonality theory (routon 2-cycle
criterion, nullity-histogram law). Prior repo work computed only ker(L_u)
(annihilation/orthogonality); this is the first ker(ad_u) (commutation) treatment.

## Established (exact)

1. **Canonical commutator nullity histogram.** For u = e_i + e_j over all 105
   pairs: nullity(ad_u) ∈ {2, 6}, with **{2: 63, 6: 42}**. The 42 nullity-6 pairs
   are exactly the 42 zero-divisor support-quartets (SEDENION_ZERO_DIVISOR_GEOMETRY).
   So the commutator obeys the same ZD law as ker(L_u): elevated kernel ⟺ ZD.
   (χ_char underlies it: distinct nonzero units always anticommute.)

2. **C_S(x) = ker(ad_x) has dim 6** for a sedenion ZD x (e.g. x=(e₁,e₂)): matches
   GZ Lemma 6.2(2), C(x) = ℝ ⊕ ℝx ⊕ O(x), so Im C(x) is 5-dim. Confirmed exactly.

3. **Every witness is a nullity-6 ZD.** For real witnesses (u,w) found across ~8
   generic configs: nullity(ad_u) = nullity(ad_w) = 6 in every case. But this does
   NOT distinguish witnesses: every nonzero u ∈ Im C(x) is already a nullity-6 ZD
   (Im C(x) ⊂ ZD variety). So the signature is automatic, not selective.

## Exact reframe of the witness

[u,w]=0 ⟺ w ∈ ker(ad_u) = Im C(u). So a witness is
  u ∈ Im C(x)  with  Im C(u) ∩ Im C(x′) ≠ {0},
i.e. a length-3 commuting chain x — u — w — x′. This is exactly Conjecture 6.8
(x,x′ at distance ≥3 means Im C(x) ∩ Im C(x′) = {0}, verified — no common
neighbour). The commutator machinery makes the object exact but does not shortcut
the existence question.

## Hypotheses tested and FALSIFIED (exact)

- **Γ_O-invariant coupling.** Do witnesses satisfy μ(u)=u₁u₂ ∝ μ(w)=w₁w₂ (same
  Fano line)? NO: cos∠(μ(u),μ(w)) ∈ {0.00, 0.66, 0.73, 0.76, 0.94, 0.97, 0.997}
  across configs — no fixed relation. (And distinct witnesses on the same curve
  give different μ, so it cannot be a fixed law.)

This joins the falsified-reframes scoreboard (proof_strategy §11): deg=7, deg=dim
Im A, reduced-degree-odd, scheme-mult-odd, and now Γ_O-coupling — all killed by
exact computation. The robust survivor remains "D has an odd-degree ℝ-component /
a real witness exists," with no clean combinatorial or degree mechanism.

## Status

The commutator machinery (ker ad_u) is a genuine, reusable new tool in the repo's
exact idiom, and it correctly recovers the ZD nullity law and the GZ centralizer
dimension. But the witness-existence proof is not reduced to a combinatorial law:
the same wall as the algebraic-geometry route. Conjecture 6.8 remains open; every
clean structural law tested (repo-side, bus-side, and here) is falsified by exact
computation, leaving only the bare existence statement — the signature of a
genuine open problem.

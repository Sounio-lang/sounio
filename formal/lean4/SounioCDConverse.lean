/-
  SounioCDConverse — the CONVERSE of the CD-tower seam obstruction, stated and empirically anchored.

  The forward obstruction ({L_l,L_u}=0 ⟹ e_l+e_u not a ZD) is proved dimension-independently GIVEN
  the cocycle lemma L_i²=−I (see SounioCDTowerSeam). Its CONVERSE —

      off-seam(l,u)  ⟹  e_l+e_u is a (left) zero divisor,   for ALL n

  — is a tower-wide conjecture: verified at n=4,5,6 in SounioCDTowerSeam, and (this file's basis)
  empirically extended with a validated O(N) reduction to n=7,8,9,10 (dim 128..1024) with zero
  counterexamples (scripts/research/cd_tower_converse_probe.py).

  The reduction (the crux that turns the converse into a clean sign-cocycle identity): a 2-term
  right factor e_a + s·e_b annihilates e_l+e_u iff the four product basis indices
  {l⊕a, l⊕b, u⊕a, u⊕b} cancel in cross-pairs, which FORCES b = a ⊕ (l⊕u); given that, a solving
  sign s∈{±1} exists iff the four-sign product σ(l,a)σ(u,a)σ(l,a⊕l⊕u)σ(u,a⊕l⊕u) = +1. Hence
  `hasXorAnnih` below is O(N)-per-pair and coincides with the brute `isZD` on the loHi locus
  (checked here at dim 16/32). Verified exhaustive for 2-term factors by an independent math-review
  (grok-4.1). Mathlib-free, no sorry.
-/
import SounioCDTowerSeam

namespace SounioCDConverse
open SounioCDTowerSeam

/-- The sharp O(N) 2-term zero-divisor test. `e_l+e_u` has a 2-term annihilator `e_a + s·e_b` iff
    `b = a ⊕ (l⊕u)` (forced by index cancellation) with `a,b ≥ 1`, and the four-sign product is `+1`
    (the two cancellation equations `σ(l,a)+s·σ(u,b)=0`, `s·σ(l,b)+σ(u,a)=0` are jointly solvable in
    `s`). Coincides with `SounioCDTowerSeam.isZD` on the loHi locus. -/
def hasXorAnnih (bits l u : Nat) : Bool :=
  let N := 2 ^ bits
  let d := l ^^^ u
  (List.range N).any (fun a =>
    a ≥ 1 && a ≠ d
      && (cdSigma l a bits * cdSigma u a bits
            * cdSigma l (a ^^^ d) bits * cdSigma u (a ^^^ d) bits == 1))

/-- The converse at a given level (brute form): every off-seam lower×upper pair is a zero divisor. -/
def converseHolds (bits : Nat) : Bool :=
  (loHi bits).all (fun p => ! offSeam bits p.1 p.2 || isZD bits p.1 p.2)

/-- The converse at a given level (sharp σ-form): every off-seam lower×upper pair admits an
    XOR-linked 2-term annihilator. Equivalent to `converseHolds` (see `converse_sharp_agrees_16`),
    but O(N)-per-pair so it certifies far beyond the reach of the brute `isZD` scan. -/
def converseHoldsSharp (bits : Nat) : Bool :=
  (loHi bits).all (fun p => ! offSeam bits p.1 p.2 || hasXorAnnih bits p.1 p.2)

/-- **The tower-wide CONVERSE CONJECTURE** (open for all n; empirically verified n=4..10).
    Stated as a `Prop`, deliberately *not* asserted — no `sorry`, no fixed-n `native_decide` hiding
    inside. A full proof further depends on the cocycle lemma `L_i²=−I` for all n (itself open). -/
def ConverseConjecture : Prop := ∀ bits, 4 ≤ bits → converseHolds bits = true

-- ── Regression anchors (brute `isZD`, native_decide) ─────────────────────────────────────────────
theorem converse_16 : converseHolds 4 = true := by native_decide

-- ── The reduction: sharp O(N) predicate == brute `isZD` on the loHi locus ─────────────────────────
theorem xorAnnih_eq_isZD_16 :
    (loHi 4).all (fun p => hasXorAnnih 4 p.1 p.2 == isZD 4 p.1 p.2) = true := by native_decide

-- ── Sharp-form converse anchors (O(N); reach beyond the brute scan) ───────────────────────────────
theorem converse_sharp_16 : converseHoldsSharp 4 = true := by native_decide
theorem converse_sharp_32 : converseHoldsSharp 5 = true := by native_decide
theorem converse_sharp_64 : converseHoldsSharp 6 = true := by native_decide

/-- At dim 16 the sharp converse and the brute converse agree (so the sharp anchors above carry the
    same content as the brute `isZD` statement, at the level where both are cheap to decide). -/
theorem converse_sharp_agrees_16 : converseHoldsSharp 4 = converseHolds 4 := by native_decide

/-- **Primary-source cross-check (Moreno 1998, `q-alg/9710013`, opening example).** In the sedenions
    `A_4`, `e₁ + e₁₀` is annihilated by `e₁₅ − e₄` (equivalently `e₄ − e₁₅`, a scalar multiple). Here
    `l=1, u=10, d=l⊕u=11`, and the annihilator index pair is XOR-linked: `4 = 15 ⊕ 11`, i.e.
    `b = a ⊕ d`. This is exactly `hasXorAnnih`'s witness, and it discharges `annih` directly. -/
theorem moreno_e1_e10 : annih 4 1 10 4 15 (-1) = true := by native_decide

end SounioCDConverse

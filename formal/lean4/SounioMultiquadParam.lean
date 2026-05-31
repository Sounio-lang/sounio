import SounioMultiquadIndep

set_option maxHeartbeats 1000000

/-!
# SounioMultiquadParam — parametric multiquadratic framework over `primes : List Nat`

Mathlib-free (core Lean 4 only). Generalises the hand-built `{3,5,11}` ladder of
`SounioMultiquadIndep` to an arbitrary list of primes `S`, working in the from-scratch
Cauchy-quotient reals `Real` of `SounioRootedFieldReal`.

This file is the **bottom-up parametric core**:

* `sqrtR p` — the Newton square-root of any `p : Nat` (default `0` below `1`), with `sqrtR_sq`.
* `radS S m` — the radical monomial `√(∏_{bit i of m set} S.get i)`.
* `evalS S c` — `Σ_{m < 2^|S|} c m · radS S m`, a general element of `ℚ(√S)`.
* `IndepMultiquad S` — the ℚ-linear independence obligation of the `2^|S|` radicals.
* `HasRadicals S` — the well-formedness predicate (distinct primes ≥ 2).
* `MultiquadField S` — a lightweight package of the support well-formedness plus independence.
* `EvalSConsSplitObligation`, `SqrtNewNotInTower`, `TowerHasInverses`, `IndepStepObligation`
  — named contracts for the generic inductive step, kept explicit until the recursive proof lands.
* base case `indep_base : IndepMultiquad []` (via `qR_inj`).

The inductive engine (`sqrt_new_not_in`, the generic inverse, and `indep_multiquad`) is
developed in the later sections / proof obligations.
-/

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)

namespace Multiquad

/-! ## 1. Generic Newton square root `sqrtR : Nat → Real`. -/

/-- The Newton square-root class of a natural `p`. For `p ≥ 1` it is the class of the Newton
    iteration `⟨newton p, newton_cauchy p _⟩`; below `1` (only `p = 0`) it defaults to `0`,
    so `sqrtR` is total. All theorems below carry the `1 ≤ p` guard. -/
noncomputable def sqrtR (p : Nat) : Real :=
  if h : (1 : Rat) ≤ (p : Rat) then mkR ⟨newton (p : Rat), newton_cauchy (p : Rat) h⟩ else zeroR'

/-- `(√p)² = p` for any `p ≥ 1`, via `newton_sq_tendsto`. The generic analogue of `R3_sq`/`R5_sq`. -/
theorem sqrtR_sq (p : Nat) (hp : (1 : Rat) ≤ (p : Rat)) :
    mulR (sqrtR p) (sqrtR p) = qR (p : Rat) := by
  unfold sqrtR
  rw [dif_pos hp]
  apply Quotient.sound
  exact newton_sq_tendsto (p : Rat) hp

/-! ## 2. Radical monomials and the evaluation map. -/

/-- `radS S m` = product of `√(S.get i)` over the set bits `i` of `m` (low bit ↔ head of `S`). -/
noncomputable def radS : List Nat → Nat → Real
  | [], _ => oneR
  | p :: S', m => if m % 2 = 1 then mulR (sqrtR p) (radS S' (m / 2)) else radS S' (m / 2)

/-- The general element `Σ_{m < 2^|S|} (c m) · radS S m` of `ℚ(√S)`. -/
noncomputable def evalS (S : List Nat) (c : Nat → Rat) : Real :=
  (List.range (2 ^ S.length)).foldr (fun m acc => addR (mulR (qR (c m)) (radS S m)) acc) zeroR'

/-! ## 3. Well-formedness and the independence obligation. -/

/-- A valid radical support: the primes are pairwise distinct and each is `≥ 2`. (Distinctness +
    primality is what makes every nonempty product `∏T` squarefree and non-square.) -/
def HasRadicals (S : List Nat) : Prop :=
  S.Nodup ∧ ∀ p ∈ S, 2 ≤ p

/-- **ℚ-linear independence of the `2^|S|` radicals.** If `Σ c m · radS S m = 0` then every
    coefficient `c m` (for `m < 2^|S|`) vanishes. This is the parametric generalisation of
    `indep_1_R3`, `indep4`, `indep8`. -/
def IndepMultiquad (S : List Nat) : Prop :=
  ∀ c : Nat → Rat, evalS S c = zeroR' → ∀ m, m < 2 ^ S.length → c m = 0

/-! ## 3a. Generic step interface: coefficient split, tower freshness, tower inverses. -/

/-- Low-parity coefficients when splitting `evalS (p :: S') c` as `A + √p·B`. -/
def lowCoeff (c : Nat → Rat) : Nat → Rat := fun m => c (2 * m)

/-- High-parity coefficients when splitting `evalS (p :: S') c` as `A + √p·B`. -/
def highCoeff (c : Nat → Rat) : Nat → Rat := fun m => c (2 * m + 1)

/-- Bookkeeping obligation for the mask split
    `evalS (p :: S') c = evalS S' c_low + √p * evalS S' c_high`. -/
def EvalSConsSplitObligation (p : Nat) (S' : List Nat) : Prop :=
  ∀ c : Nat → Rat,
    evalS (p :: S') c =
      addR (evalS S' (lowCoeff c)) (mulR (sqrtR p) (evalS S' (highCoeff c)))

/-- Tower-freshness obligation: the new generator `√p` is not already in `K_S'`. -/
def SqrtNewNotInTower (p : Nat) (S' : List Nat) : Prop :=
  ¬ ∃ c : Nat → Rat, evalS S' c = sqrtR p

/-- Inversion obligation for the finite tower represented by `evalS S`: every nonzero coefficient
    vector has a right inverse represented in the same tower. With `IndepMultiquad S`, the
    coefficient-side nonzero witness is the tower nonzero witness. -/
def TowerHasInverses (S : List Nat) : Prop :=
  ∀ c : Nat → Rat,
    (∃ m, m < 2 ^ S.length ∧ c m ≠ 0) →
      ∃ d : Nat → Rat, mulR (evalS S c) (evalS S d) = oneR

/-- The generic induction-step contract. It is named as an obligation rather than assumed as a
    theorem until the recursive norm/conjugate proof is mechanised. -/
def IndepStepObligation (p : Nat) (S' : List Nat) : Prop :=
  EvalSConsSplitObligation p S' →
  SqrtNewNotInTower p S' →
  TowerHasInverses S' →
  IndepMultiquad S' →
  IndepMultiquad (p :: S')

/-- Lightweight package for the multiquadratic tower on a fixed support in this `Real` model.
    The actual operations are the global `radS`/`evalS`; the package records the obligations
    needed to treat that support as a faithful multiquadratic field. -/
structure MultiquadField (S : List Nat) where
  hasRadicals : HasRadicals S
  independent : IndepMultiquad S

/-! ## 4. Base case `|S| = 0`: `ℚ` embeds injectively. -/

/-- `evalS [] c = qR (c 0)` — the empty support evaluates to the rational coordinate. -/
theorem evalS_nil (c : Nat → Rat) : evalS [] c = qR (c 0) := by
  show (List.range (2 ^ (0 : Nat))).foldr
        (fun m acc => addR (mulR (qR (c m)) (radS [] m)) acc) zeroR' = qR (c 0)
  rw [show (2 ^ (0 : Nat)) = 1 from rfl, show List.range 1 = [0] from rfl]
  show addR (mulR (qR (c 0)) (radS [] 0)) zeroR' = qR (c 0)
  show addR (mulR (qR (c 0)) oneR) zeroR' = qR (c 0)
  rw [mulR_one, addR_zero]

/-- **Base case.** `IndepMultiquad []`: the single radical `1` is independent (via `qR_inj`). -/
theorem indep_base : IndepMultiquad [] := by
  intro c hc m hm
  have hm0 : m = 0 := by
    have : m < 1 := by simpa using hm
    omega
  subst hm0
  rw [evalS_nil] at hc
  have : qR (c 0) = qR 0 := by rw [hc, qR_zero]
  exact qR_inj this

/-! ## 5. Identification of the generic `sqrtR` with the hardcoded `rootR`.

The reference ladder of `SounioMultiquadIndep` works over `rootR : Fin 4 → Real`
(`rootR 0 = √3`, `rootR 1 = √5`, `rootR 3 = √11`). These are the *same reals* as the
generic `sqrtR`, because both are the class of the Newton iteration on the same rational.
The bridge lets us discharge concrete `IndepMultiquad` instances through the proven `indep8`. -/

theorem sqrtR_3 : sqrtR 3 = rootR 0 := by
  have hp : (1 : Rat) ≤ ((3 : Nat) : Rat) := by native_decide
  unfold sqrtR
  rw [dif_pos hp]
  apply mk_eq_of_seq_eq
  intro n
  show newton ((3 : Nat) : Rat) n = (newtonC 0).seq n
  have harg : ((3 : Nat) : Rat) = rfOfNat Rat.add 0 1 (primeNatLit 0) := by native_decide
  rw [harg]; rfl

theorem sqrtR_5 : sqrtR 5 = rootR 1 := by
  have hp : (1 : Rat) ≤ ((5 : Nat) : Rat) := by native_decide
  unfold sqrtR
  rw [dif_pos hp]
  apply mk_eq_of_seq_eq
  intro n
  show newton ((5 : Nat) : Rat) n = (newtonC 1).seq n
  have harg : ((5 : Nat) : Rat) = rfOfNat Rat.add 0 1 (primeNatLit 1) := by native_decide
  rw [harg]; rfl

theorem sqrtR_11 : sqrtR 11 = rootR 3 := by
  have hp : (1 : Rat) ≤ ((11 : Nat) : Rat) := by native_decide
  unfold sqrtR
  rw [dif_pos hp]
  apply mk_eq_of_seq_eq
  intro n
  show newton ((11 : Nat) : Rat) n = (newtonC 3).seq n
  have harg : ((11 : Nat) : Rat) = rfOfNat Rat.add 0 1 (primeNatLit 3) := by native_decide
  rw [harg]; rfl

/-! ## 6. Concrete independence for `{3,11}` (|S|=2, degree 4) via the proven `indep8`.

`IndepMultiquad [3,11]` is the |S|=2 instance feeding the Moser spindle. Its four radicals
`{1, √3, √11, √33}` are exactly the masks `{0,1,8,9}` of the 16-dim de Grey kernel, hence a
sub-relation of `indep8` (the 8-radical relation over `{√3,√5,√11}`). We embed with the `√5`
coordinates set to `0`. -/

/-- **|S|=2 independence.** `{1, √3, √11, √33}` is ℚ-linearly independent. -/
theorem indep_3_11 : IndepMultiquad [3, 11] := by
  intro c hc m hm
  have hL : evalS [3, 11] c
      = addR (qR (c 0))
          (addR (mulR (qR (c 1)) (rootR 0))
          (addR (mulR (qR (c 2)) (rootR 3))
                (mulR (qR (c 3)) (mulR (rootR 0) (rootR 3))))) := by
    show addR (mulR (qR (c 0)) (radS [3,11] 0))
          (addR (mulR (qR (c 1)) (radS [3,11] 1))
          (addR (mulR (qR (c 2)) (radS [3,11] 2))
          (addR (mulR (qR (c 3)) (radS [3,11] 3)) zeroR'))) = _
    simp only [show radS [3,11] 0 = oneR from rfl,
               show radS [3,11] 1 = mulR (sqrtR 3) oneR from rfl,
               show radS [3,11] 2 = mulR (sqrtR 11) oneR from rfl,
               show radS [3,11] 3 = mulR (sqrtR 3) (mulR (sqrtR 11) oneR) from rfl,
               mulR_one, addR_zero, sqrtR_3, sqrtR_11]
  have hR : addR (alpha4 (c 0) (c 1) 0 0) (mulR (alpha4 (c 2) (c 3) 0 0) (rootR 3))
      = addR (addR (qR (c 0)) (mulR (qR (c 1)) (rootR 0)))
          (addR (mulR (qR (c 2)) (rootR 3))
                (mulR (qR (c 3)) (mulR (rootR 0) (rootR 3)))) := by
    simp only [alpha4, qR_zero, zeroR'_mul, addR_zero]
    rw [rightDistribR, mulR_assoc]
  have hbridge : addR (alpha4 (c 0) (c 1) 0 0) (mulR (alpha4 (c 2) (c 3) 0 0) (rootR 3)) = qR 0 := by
    rw [hR, addR_assoc, ← hL, hc, qR_zero]
  obtain ⟨h0, h1, _, _, h2, h3, _, _⟩ :=
    indep8 (c 0) (c 1) 0 0 (c 2) (c 3) 0 0 0 hbridge
  have hm4 : m < 4 := hm
  match m, hm4 with
  | 0, _ => exact h0
  | 1, _ => exact h1
  | 2, _ => exact h2
  | 3, _ => exact h3
  | (n + 4), h => exact absurd h (by omega)

/-! ## 7. `sqrt_new_not_in_tower` — the squarefree / irrationality core.

The inductive step `IndepMultiquad S' → IndepMultiquad (p :: S')` needs `√p ∉ K_{S'}`
(`p` genuinely new). The standard reduction peels the top generator and lands on
"no element of `K_{S'}` squares to a radicand `p · ∏T` (`T ⊆ S'`)", which in turn — because
`p · ∏T` is squarefree for distinct primes — reduces to `no_rat_sqrt (p · ∏T)`.

* `no_rat_sqrt` (already generic in `SounioMultiquadIndep`) supplies the ℚ-level fact from any
  non-square hypothesis.
* `nonSquare_radicand` discharges the non-square hypothesis for every radicand of a subset of
  `{3,5,11}` (`{3,5,11,15,33,55,165}`) by the enumerated `not_sq_radicand` — i.e. for every
  support actually instantiated here.

The **tower-level** statement `√p ∉ K_{S'}` for *arbitrary* `S'` (the field-expansion peel) is
discharged concretely for the reference tower by `sqrt5_not_in_Q_sqrt3` and
`sqrt11_not_in_Q_sqrt3_sqrt5` in `SounioMultiquadIndep`; the unrestricted `∀S'` version, together
with a Mathlib-free squarefree-by-prime-factorisation lemma, is recorded as the open obligation
`MultiquadIndepTheorem` in §8. -/

/-- Radicands of nonempty subsets of `{3,5,11}` are never perfect squares (enumerated core). -/
theorem nonSquare_radicand (m : Nat)
    (hm : m = 3 ∨ m = 5 ∨ m = 11 ∨ m = 15 ∨ m = 33 ∨ m = 55 ∨ m = 165) :
    ∀ k : Nat, k * k ≠ m :=
  not_sq_radicand m hm

/-- **ℚ-level `sqrt_new`.** For any non-square radicand `m`, no rational squares to `m` in `Real`:
    `√m ∉ ℚ`. Fully generic in `m` (only the non-square hypothesis). This is the base of every
    tower-level `√p ∉ K_{S'}` argument. -/
theorem sqrt_new_not_in_Q (m : Nat) (hns : ∀ k : Nat, k * k ≠ m) :
    ¬ ∃ q : Rat, mulR (qR q) (qR q) = qR (m : Rat) := by
  rintro ⟨q, hq⟩
  rw [qR_mul] at hq
  exact no_rat_sqrt m hns ⟨q, qR_inj hq⟩

/-- **Generic norm-nonzero / domain core for `ℚ(√p)` (|S|=1).** For any non-square `p`, the norm
    `N(a + b√p) = a² - p·b²` is nonzero whenever `(a,b) ≠ (0,0)`. This generalises `N5_ne_zero`
    to an *arbitrary* non-square radicand and is the kernel of the conjugate/norm inverse
    `(a + b√p)⁻¹ = (a - b√p) / (a² - p·b²)`: it shows `ℚ(√p)` is a domain and `a + b√p` is
    invertible. The recursive multi-level lift (generalising `Q35_inv` to the full `K_{S'}` tower)
    builds on this and is part of the §8 obligation. -/
theorem Np_ne_zero (p : Nat) (hns : ∀ k : Nat, k * k ≠ p) (a b : Rat)
    (hab : ¬ (a = 0 ∧ b = 0)) : a * a - (p : Rat) * (b * b) ≠ 0 := by
  intro h
  by_cases hb : b = 0
  · by_cases ha : a = 0
    · exact hab ⟨ha, hb⟩
    · rw [hb, Rat.mul_zero, Rat.mul_zero, Rat.sub_eq_add_neg, Rat.neg_zero, Rat.add_zero] at h
      apply ha
      have hinv : a⁻¹ * a = 1 := by rw [Rat.mul_comm]; exact Rat.mul_inv_cancel a ha
      calc a = (a⁻¹ * a) * a := by rw [hinv, Rat.one_mul]
        _ = a⁻¹ * (a * a) := by rw [Rat.mul_assoc]
        _ = a⁻¹ * 0 := by rw [h]
        _ = 0 := Rat.mul_zero a⁻¹
  · apply no_rat_sqrt p hns
    refine ⟨a * b⁻¹, ?_⟩
    have hbinv : b * b⁻¹ = 1 := Rat.mul_inv_cancel b hb
    have heq : a * a = (p : Rat) * (b * b) := by
      have h2 := eq_add_of_sub_eq h
      rwa [Rat.add_zero] at h2
    calc (a * b⁻¹) * (a * b⁻¹)
        = (a * a) * (b⁻¹ * b⁻¹) := sq_prod a b⁻¹
      _ = ((p : Rat) * (b * b)) * (b⁻¹ * b⁻¹) := by rw [heq]
      _ = (p : Rat) * ((b * b) * (b⁻¹ * b⁻¹)) := by rw [Rat.mul_assoc]
      _ = (p : Rat) * ((b * b⁻¹) * (b * b⁻¹)) := by rw [← sq_prod b b⁻¹]
      _ = (p : Rat) := by rw [hbinv, Rat.mul_one, Rat.mul_one]

/-! ## 8. The general theorem as an explicit obligation.

`IndepMultiquad` is proven here for `|S| = 0` (`indep_base`) and discharged concretely for the
instantiated supports `[3,11]` (§6) and `[3,5,11]` (`SounioMoserSpindleQ311`, via `indep8`). The
fully generic `∀S` statement is named below; its proof requires the recursive norm/conjugate
inverse tower (generalising `Q5_inv`/`Q35_inv`) plus a Mathlib-free squarefree-by-primality
lemma, and remains a research obligation — *not* an axiom, *not* a `sorry`. -/

/-- The general multiquadratic independence theorem (degree `2^|S|` for distinct primes). -/
def MultiquadIndepTheorem : Prop := ∀ S : List Nat, HasRadicals S → IndepMultiquad S

/-- `HasRadicals []` holds, so `indep_base` discharges `MultiquadIndepTheorem` on the empty
    support — the genuine base case of the general statement. -/
theorem hasRadicals_nil : HasRadicals [] := ⟨List.nodup_nil, by intro p hp; cases hp⟩

/-! ## 9. Migration of the `{3,5,11}` line (degree 8) into the framework.

`IndepMultiquad [3,5,11]` is the |S|=3 instance — the eight radicals
`{1,√3,√5,√15,√11,√33,√55,√165}`. Proving it in the framework and showing it recovers `indep8`
re-grounds the existing `χ ≥ 5` result of `SounioDeGreyChi5` as a *case of the parametric
framework*: the de Grey G₅₂₉ realisation over `ℚ(√3,√5,√11)` is exactly the `S = [3,5,11]`
support, so `χ(ℚ(√3,√5,√11)²) ≥ 5` is the framework instance at degree `2³ = 8`. -/

/-- **|S|=3 independence (degree 8).** `{1,√3,√5,√15,√11,√33,√55,√165}` is ℚ-linearly
    independent — the framework instance recovering `indep8`. -/
theorem indep_3_5_11 : IndepMultiquad [3, 5, 11] := by
  intro c hc m hm
  have key :
      addR (alpha4 (c 0) (c 1) (c 2) (c 3))
           (mulR (alpha4 (c 4) (c 5) (c 6) (c 7)) (rootR 3))
        = evalS [3, 5, 11] c := by
    show addR (alpha4 (c 0) (c 1) (c 2) (c 3))
           (mulR (alpha4 (c 4) (c 5) (c 6) (c 7)) (rootR 3))
        = addR (mulR (qR (c 0)) (radS [3,5,11] 0))
            (addR (mulR (qR (c 1)) (radS [3,5,11] 1))
            (addR (mulR (qR (c 2)) (radS [3,5,11] 2))
            (addR (mulR (qR (c 3)) (radS [3,5,11] 3))
            (addR (mulR (qR (c 4)) (radS [3,5,11] 4))
            (addR (mulR (qR (c 5)) (radS [3,5,11] 5))
            (addR (mulR (qR (c 6)) (radS [3,5,11] 6))
            (addR (mulR (qR (c 7)) (radS [3,5,11] 7)) zeroR')))))))
    simp only [show radS [3,5,11] 0 = oneR from rfl,
               show radS [3,5,11] 1 = mulR (sqrtR 3) oneR from rfl,
               show radS [3,5,11] 2 = mulR (sqrtR 5) oneR from rfl,
               show radS [3,5,11] 3 = mulR (sqrtR 3) (mulR (sqrtR 5) oneR) from rfl,
               show radS [3,5,11] 4 = mulR (sqrtR 11) oneR from rfl,
               show radS [3,5,11] 5 = mulR (sqrtR 3) (mulR (sqrtR 11) oneR) from rfl,
               show radS [3,5,11] 6 = mulR (sqrtR 5) (mulR (sqrtR 11) oneR) from rfl,
               show radS [3,5,11] 7 = mulR (sqrtR 3) (mulR (sqrtR 5) (mulR (sqrtR 11) oneR)) from rfl,
               alpha4, R15_def, mulR_one, addR_zero, rightDistribR, mulR_assoc, addR_assoc,
               sqrtR_3, sqrtR_5, sqrtR_11]
  have hbridge :
      addR (alpha4 (c 0) (c 1) (c 2) (c 3))
           (mulR (alpha4 (c 4) (c 5) (c 6) (c 7)) (rootR 3)) = qR 0 := by
    rw [key, hc, qR_zero]
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩ :=
    indep8 (c 0) (c 1) (c 2) (c 3) (c 4) (c 5) (c 6) (c 7) 0 hbridge
  have hm8 : m < 8 := hm
  match m, hm8 with
  | 0, _ => exact h0
  | 1, _ => exact h1
  | 2, _ => exact h2
  | 3, _ => exact h3
  | 4, _ => exact h4
  | 5, _ => exact h5
  | 6, _ => exact h6
  | 7, _ => exact h7
  | (n + 8), h => exact absurd h (by omega)

#print axioms sqrtR_sq
#print axioms evalS_nil
#print axioms indep_base
#print axioms indep_3_11
#print axioms indep_3_5_11
#print axioms sqrt_new_not_in_Q
#print axioms Np_ne_zero

end Multiquad

end SounioSqrt.RealCauchyField

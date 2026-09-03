import SounioDeGreyChi5TransferWf
import SounioSqrtFieldReal
import SounioRealInverseImpl
import SounioRealOrderAxiomsImpl
import SounioNewtonSqrtImpl

set_option maxHeartbeats 0

/-!
# Sounio — `RootedField ℝ` and χ(ℝ²) ≥ 5 (Mathlib-free)

Assembles the Cauchy-quotient real numbers `Real := Quotient realSetoid` into a
`SounioSqrt.RootedField` (ordered field + the four prime square-roots √3,√5,√7,√11, NO total `sqrt`,
NO completeness), then fires `DeGrey529.TransferWf.rootedField_chi_ge_5` to obtain χ(ℝ²) ≥ 5.

The analytic content was proved in:
* `SounioSqrtFieldReal` — `RealEq` equivalence, `realSetoid`, additive/multiplicative descent
  (`realOpsCong_add`, `mul_cong`), `bounded_away`.
* `SounioRealInverseImpl` — `invSeq`, `inv_cauchy`, `mul_inv_tendsto`, `inv_cong`.
* `SounioRealOrderAxiomsImpl` — canonical ε-order `leR` and its axioms.
* `SounioNewtonSqrtImpl` — Newton √p sequences (`newton_cauchy`, `newton_sq_tendsto`, `newton_ge_one`).
-/

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)
open Sounio.RealCauchy.SounioRealCauchy (zero one ofRat)

/-- The Mathlib-free real numbers: Cauchy `Rat` sequences modulo `RealEq`. -/
abbrev Real : Type := Quotient realSetoid

/-- Class of a representative. -/
def mkR (a : SounioRealCauchy) : Real := Quotient.mk realSetoid a

/-- Pointwise-equal representatives have the same class. -/
theorem realEq_of_seq_eq {a b : SounioRealCauchy} (h : ∀ n, a.seq n = b.seq n) : RealEq a b := by
  intro ε hε
  refine ⟨0, fun n _ => ?_⟩
  show a.seq n - b.seq n ≤ ε ∧ -(a.seq n - b.seq n) ≤ ε
  have h0 : a.seq n - b.seq n = 0 := by rw [h n]; exact Rat.sub_self
  rw [h0]
  exact ⟨Rat.le_of_lt hε, by rw [Rat.neg_zero]; exact Rat.le_of_lt hε⟩

theorem mk_eq_of_seq_eq {a b : SounioRealCauchy} (h : ∀ n, a.seq n = b.seq n) : mkR a = mkR b :=
  Quotient.sound (realEq_of_seq_eq h)

/-! ## Representative-level Cauchy operations. -/

theorem neg_cauchy {f : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f) :
    Sounio.RealCauchy.IsCauchy (fun n => -(f n)) := by
  intro ε hε
  obtain ⟨N, hN⟩ := hf ε hε
  refine ⟨N, fun m n hm hn => ?_⟩
  obtain ⟨h1, h2⟩ := hN m n hm hn
  refine ⟨?_, ?_⟩
  · show -(f m) - -(f n) ≤ ε
    rw [show -(f m) - -(f n) = f n - f m from by
      rw [Rat.sub_eq_add_neg, Rat.neg_neg, Rat.add_comm, ← Rat.sub_eq_add_neg]]
    exact h2
  · show -(f n) - -(f m) ≤ ε
    rw [show -(f n) - -(f m) = f m - f n from by
      rw [Rat.sub_eq_add_neg, Rat.neg_neg, Rat.add_comm, ← Rat.sub_eq_add_neg]]
    exact h1

def addC (a b : SounioRealCauchy) : SounioRealCauchy :=
  ⟨fun n => a.seq n + b.seq n, add_cauchy a.cauchy b.cauchy⟩

def mulC (a b : SounioRealCauchy) : SounioRealCauchy :=
  ⟨fun n => a.seq n * b.seq n, mul_cauchy a.cauchy b.cauchy⟩

def negC (a : SounioRealCauchy) : SounioRealCauchy :=
  ⟨fun n => -(a.seq n), neg_cauchy a.cauchy⟩

theorem neg_cong {a a' : SounioRealCauchy} (h : RealEq a a') : RealEq (negC a) (negC a') := by
  intro ε hε
  obtain ⟨N, hN⟩ := h ε hε
  refine ⟨N, fun n hn => ?_⟩
  obtain ⟨h1, h2⟩ := hN n hn
  show (-(a.seq n)) - (-(a'.seq n)) ≤ ε ∧ -((-(a.seq n)) - (-(a'.seq n))) ≤ ε
  have key : (-(a.seq n)) - (-(a'.seq n)) = -(a.seq n - a'.seq n) := by
    rw [Rat.neg_sub, Rat.sub_eq_add_neg, Rat.neg_neg, Rat.add_comm, ← Rat.sub_eq_add_neg]
  rw [key]
  exact ⟨h2, by rw [Rat.neg_neg]; exact h1⟩

/-! ## Quotient operations. -/

def addR : Real → Real → Real :=
  Quotient.lift₂ (fun a b => mkR (addC a b))
    (fun a b a' b' haa hbb => Quotient.sound (realOpsCong_add a b a' b' haa hbb))

def mulR : Real → Real → Real :=
  Quotient.lift₂ (fun a b => mkR (mulC a b))
    (fun a b a' b' haa hbb => Quotient.sound (mul_cong a b a' b' haa hbb))

def negR : Real → Real :=
  Quotient.lift (fun a => mkR (negC a)) (fun a a' h => Quotient.sound (neg_cong h))

def zeroR' : Real := mkR zero
def oneR : Real := mkR one

/-! ## Ring axioms (each reduces to a pointwise `Rat` identity). -/

theorem addR_assoc (x y z : Real) : addR (addR x y) z = addR x (addR y z) := by
  refine Quotient.inductionOn₃ x y z (fun a b c => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.add_assoc _ _ _)

theorem addR_comm (x y : Real) : addR x y = addR y x := by
  refine Quotient.inductionOn₂ x y (fun a b => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.add_comm _ _)

theorem mulR_assoc (x y z : Real) : mulR (mulR x y) z = mulR x (mulR y z) := by
  refine Quotient.inductionOn₃ x y z (fun a b c => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.mul_assoc _ _ _)

theorem mulR_comm (x y : Real) : mulR x y = mulR y x := by
  refine Quotient.inductionOn₂ x y (fun a b => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.mul_comm _ _)

theorem leftDistribR (x y z : Real) : mulR x (addR y z) = addR (mulR x y) (mulR x z) := by
  refine Quotient.inductionOn₃ x y z (fun a b c => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.mul_add _ _ _)

theorem rightDistribR (x y z : Real) : mulR (addR x y) z = addR (mulR x z) (mulR y z) := by
  refine Quotient.inductionOn₃ x y z (fun a b c => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.add_mul _ _ _)

theorem addR_zero (x : Real) : addR x zeroR' = x := by
  refine Quotient.inductionOn x (fun a => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.add_zero _)

theorem mulR_one (x : Real) : mulR x oneR = x := by
  refine Quotient.inductionOn x (fun a => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.mul_one _)

theorem addR_neg (x : Real) : addR x (negR x) = zeroR' := by
  refine Quotient.inductionOn x (fun a => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.add_neg_cancel _)

/-! ## Multiplicative inverse (Phase 2b): well-defined on the quotient via `bounded_away`. -/

theorem invSeq_cauchy_of_ne (a : SounioRealCauchy) (h : ¬ RealEq a zeroR) :
    Sounio.RealCauchy.IsCauchy (invSeq a.seq) := by
  obtain ⟨δ, N, hδ, hba⟩ := bounded_away h
  exact inv_cauchy a.cauchy δ N hδ hba

open Classical in
/-- The inverse of a representative: `1/aₙ` patched to `0` for the null class (`a ≈ 0`). -/
noncomputable def invReal (a : SounioRealCauchy) : SounioRealCauchy :=
  if h : RealEq a zeroR then zero else ⟨invSeq a.seq, invSeq_cauchy_of_ne a h⟩

/-- Rational min, for selecting a common bounded-away constant in `invReal_resp`. -/
def ratMin (a b : Rat) : Rat := if a ≤ b then a else b

theorem ratMin_le_left (a b : Rat) : ratMin a b ≤ a := by
  unfold ratMin; split
  · exact Rat.le_refl
  · next h => exact Rat.le_of_lt (Rat.not_le.mp h)

theorem ratMin_le_right (a b : Rat) : ratMin a b ≤ b := by
  unfold ratMin; split
  · next h => exact h
  · exact Rat.le_refl

theorem ratMin_pos {a b : Rat} (ha : 0 < a) (hb : 0 < b) : 0 < ratMin a b := by
  unfold ratMin; split
  · exact ha
  · exact hb

theorem invReal_resp (a b : SounioRealCauchy) (hab : RealEq a b) :
    mkR (invReal a) = mkR (invReal b) := by
  rcases Classical.em (RealEq a zeroR) with ha0 | ha0
  · have hb0 : RealEq b zeroR := realEq_trans (realEq_symm hab) ha0
    have ea : invReal a = zero := by unfold invReal; rw [dif_pos ha0]
    have eb : invReal b = zero := by unfold invReal; rw [dif_pos hb0]
    rw [ea, eb]
  · have hb0 : ¬ RealEq b zeroR := fun hbz => ha0 (realEq_trans hab hbz)
    have ea : invReal a = ⟨invSeq a.seq, invSeq_cauchy_of_ne a ha0⟩ := by
      unfold invReal; rw [dif_neg ha0]
    have eb : invReal b = ⟨invSeq b.seq, invSeq_cauchy_of_ne b hb0⟩ := by
      unfold invReal; rw [dif_neg hb0]
    rw [ea, eb]
    apply Quotient.sound
    obtain ⟨δa, Na, hδa, hbaa⟩ := bounded_away ha0
    obtain ⟨δb, Nb, hδb, hbab⟩ := bounded_away hb0
    exact inv_cong a.cauchy b.cauchy (ratMin δa δb) Na Nb (ratMin_pos hδa hδb)
      (fun n hn => Rat.le_trans (ratMin_le_left δa δb) (hbaa n hn))
      (fun n hn => Rat.le_trans (ratMin_le_right δa δb) (hbab n hn))
      hab

noncomputable def invR : Real → Real :=
  Quotient.lift (fun a => mkR (invReal a)) invReal_resp

theorem mulR_inv (x : Real) (hx : x ≠ zeroR') : mulR x (invR x) = oneR := by
  revert hx
  refine Quotient.inductionOn x (fun a => ?_)
  intro hx
  have ha0 : ¬ RealEq a zeroR := fun h => hx (Quotient.sound h)
  have eqinv : invR (mkR a) = mkR ⟨invSeq a.seq, invSeq_cauchy_of_ne a ha0⟩ := by
    show mkR (invReal a) = mkR ⟨invSeq a.seq, invSeq_cauchy_of_ne a ha0⟩
    unfold invReal; rw [dif_neg ha0]
  show mulR (mkR a) (invR (mkR a)) = oneR
  rw [eqinv]
  apply Quotient.sound
  obtain ⟨δ, N, hδ, hba⟩ := bounded_away ha0
  exact mul_inv_tendsto a.cauchy δ N hδ hba

theorem zeroNeOneR : zeroR' ≠ oneR := fun h => zero_ne_oneR (Quotient.exact h)

/-! ## Order (Phase 2c): the ε-eventual order `leR` descends to `leReal` on the quotient. -/

theorem rat_third_pos {ε : Rat} (hε : 0 < ε) : 0 < ε * (1/3) :=
  Rat.mul_pos hε (by native_decide)

theorem rat_add_thirds (ε : Rat) : ε * (1/3) + ε * (1/3) + ε * (1/3) = ε := by
  rw [← Rat.mul_add, ← Rat.mul_add,
      show ((1:Rat)/3 + 1/3 + 1/3) = 1 from by native_decide, Rat.mul_one]

theorem rat_collect_thirds (y ε : Rat) :
    y + ε * (1/3) + ε * (1/3) + ε * (1/3) = y + ε := by
  rw [Rat.add_assoc, Rat.add_assoc,
      show ε * (1/3) + (ε * (1/3) + ε * (1/3)) = ε from by
        rw [← Rat.add_assoc]; exact rat_add_thirds ε]

theorem leR_congr {a b a' b' : SounioRealCauchy} (haa : RealEq a a') (hbb : RealEq b b')
    (hab : leR a b) : leR a' b' := by
  intro ε hε
  obtain ⟨Na, hNa⟩ := haa (ε * (1/3)) (rat_third_pos hε)
  obtain ⟨Nb, hNb⟩ := hbb (ε * (1/3)) (rat_third_pos hε)
  obtain ⟨Nc, hNc⟩ := hab (ε * (1/3)) (rat_third_pos hε)
  refine ⟨max (max Na Nb) Nc, fun n hn => ?_⟩
  have hn_a : Na ≤ n :=
    Nat.le_trans (Nat.le_trans (Nat.le_max_left Na Nb) (Nat.le_max_left _ Nc)) hn
  have hn_b : Nb ≤ n :=
    Nat.le_trans (Nat.le_trans (Nat.le_max_right Na Nb) (Nat.le_max_left _ Nc)) hn
  have hn_c : Nc ≤ n := Nat.le_trans (Nat.le_max_right _ Nc) hn
  obtain ⟨_, ha2⟩ := hNa n hn_a
  have hc1 := hNc n hn_c
  obtain ⟨hb1, _⟩ := hNb n hn_b
  have e1 : a'.seq n ≤ a.seq n + ε * (1/3) := by
    have hsub : a'.seq n - a.seq n ≤ ε * (1/3) := by
      rw [show a'.seq n - a.seq n = -(a.seq n - a'.seq n) from by rw [Rat.neg_sub]]
      exact ha2
    have hadd : (a'.seq n - a.seq n) + a.seq n ≤ ε * (1/3) + a.seq n :=
      (Rat.add_le_add_right).mpr hsub
    rw [Rat.sub_add_cancel, Rat.add_comm (ε * (1/3)) (a.seq n)] at hadd
    exact hadd
  have hb1' : b.seq n - b'.seq n ≤ ε * (1/3) := hb1
  have e3 : b.seq n ≤ b'.seq n + ε * (1/3) := by
    have hadd : (b.seq n - b'.seq n) + b'.seq n ≤ ε * (1/3) + b'.seq n :=
      (Rat.add_le_add_right).mpr hb1'
    rw [Rat.sub_add_cancel, Rat.add_comm (ε * (1/3)) (b'.seq n)] at hadd
    exact hadd
  have step : a'.seq n ≤ b'.seq n + ε * (1/3) + ε * (1/3) + ε * (1/3) := by
    have t2 : a.seq n + ε * (1/3) ≤ (b.seq n + ε * (1/3)) + ε * (1/3) :=
      (Rat.add_le_add_right).mpr hc1
    have t3 : (b.seq n + ε * (1/3)) + ε * (1/3)
        ≤ ((b'.seq n + ε * (1/3)) + ε * (1/3)) + ε * (1/3) :=
      (Rat.add_le_add_right).mpr ((Rat.add_le_add_right).mpr e3)
    exact Rat.le_trans e1 (Rat.le_trans t2 t3)
  rw [rat_collect_thirds (b'.seq n) ε] at step
  exact step

def leReal : Real → Real → Prop :=
  Quotient.lift₂ leR
    (fun a b a' b' haa hbb =>
      propext ⟨leR_congr haa hbb, leR_congr (realEq_symm haa) (realEq_symm hbb)⟩)

theorem leReal_refl (x : Real) : leReal x x := by
  refine Quotient.inductionOn x (fun a => ?_); exact leR_refl a

theorem leReal_trans {x y z : Real} (h1 : leReal x y) (h2 : leReal y z) : leReal x z := by
  revert h1 h2
  refine Quotient.inductionOn₃ x y z (fun a b c => ?_)
  intro h1 h2; exact leR_trans h1 h2

theorem leReal_antisymm {x y : Real} (h1 : leReal x y) (h2 : leReal y x) : x = y := by
  revert h1 h2
  refine Quotient.inductionOn₂ x y (fun a b => ?_)
  intro h1 h2; exact Quotient.sound (leR_antisymm h1 h2)

theorem leReal_total (x y : Real) : leReal x y ∨ leReal y x := by
  refine Quotient.inductionOn₂ x y (fun a b => ?_); exact leR_total a b

theorem leReal_add_le_add_right {x y z : Real} (h : leReal x y) :
    leReal (addR x z) (addR y z) := by
  revert h
  refine Quotient.inductionOn₃ x y z (fun a b c => ?_)
  intro h; exact leR_add_le_add_right_seq h

theorem leReal_mul_nonneg {x y : Real} (hx : leReal zeroR' x) (hy : leReal zeroR' y) :
    leReal zeroR' (mulR x y) := by
  revert hx hy
  refine Quotient.inductionOn₂ x y (fun a b => ?_)
  intro hx hy; exact leR_mul_nonneg hx hy

/-! ## Newton roots (Phase 3 + 2d): √3, √5, √7, √11 as classes of Newton sequences. -/

theorem one_le_primeRat (j : Fin 4) :
    (1 : Rat) ≤ rfOfNat Rat.add 0 1 (primeNatLit j) := by
  have h : ∀ k : Fin 4, (1 : Rat) ≤ rfOfNat Rat.add 0 1 (primeNatLit k) := by native_decide
  exact h j

noncomputable def newtonC (j : Fin 4) : SounioRealCauchy :=
  ⟨newton (rfOfNat Rat.add 0 1 (primeNatLit j)), newton_cauchy _ (one_le_primeRat j)⟩

noncomputable def rootR (j : Fin 4) : Real := mkR (newtonC j)

/-- The quotient `ℕ`-cast equals the class of the constant rational `ℕ`-cast. -/
theorem rfOfNat_real (k : Nat) :
    rfOfNat addR zeroR' oneR k = mkR (ofRat (rfOfNat Rat.add 0 1 k)) := by
  induction k with
  | zero => exact mk_eq_of_seq_eq (fun n => rfl)
  | succ k ih =>
    show addR (rfOfNat addR zeroR' oneR k) oneR = mkR (ofRat (Rat.add (rfOfNat Rat.add 0 1 k) 1))
    rw [ih]
    exact mk_eq_of_seq_eq (fun n => rfl)

theorem rootR_nonneg (j : Fin 4) : leReal zeroR' (rootR j) := by
  intro ε hε
  refine ⟨0, fun n _ => ?_⟩
  show (0 : Rat) ≤ newton (rfOfNat Rat.add 0 1 (primeNatLit j)) n + ε
  have h1 : (1 : Rat) ≤ newton (rfOfNat Rat.add 0 1 (primeNatLit j)) n :=
    newton_ge_one _ (one_le_primeRat j) n
  have h0 : (0 : Rat) ≤ newton (rfOfNat Rat.add 0 1 (primeNatLit j)) n :=
    Rat.le_trans (by native_decide) h1
  have hstep : newton (rfOfNat Rat.add 0 1 (primeNatLit j)) n + 0
      ≤ newton (rfOfNat Rat.add 0 1 (primeNatLit j)) n + ε :=
    (Rat.add_le_add_left).mpr (Rat.le_of_lt hε)
  rw [Rat.add_zero] at hstep
  exact Rat.le_trans h0 hstep

theorem rootR_sq (j : Fin 4) :
    mulR (rootR j) (rootR j) = rfOfNat addR zeroR' oneR (primeNatLit j) := by
  rw [rfOfNat_real (primeNatLit j)]
  apply Quotient.sound
  exact newton_sq_tendsto _ (one_le_primeRat j)

/-! ## Assembly + transfer (Phase 4): `RootedField ℝ` and χ(ℝ²) ≥ 5. -/

noncomputable def rootedFieldReal : RootedField where
  F := Real
  add := addR
  mul := mulR
  neg := negR
  zero := zeroR'
  one := oneR
  inv := invR
  le := leReal
  add_assoc := addR_assoc
  add_comm := addR_comm
  mul_assoc := mulR_assoc
  mul_comm := mulR_comm
  left_distrib := leftDistribR
  right_distrib := rightDistribR
  add_zero := addR_zero
  mul_one := mulR_one
  add_neg := addR_neg
  mul_inv := mulR_inv
  zero_ne_one := zeroNeOneR
  le_refl := leReal_refl
  le_trans := leReal_trans
  le_antisymm := leReal_antisymm
  le_total := leReal_total
  add_le_add_right := leReal_add_le_add_right
  mul_nonneg := leReal_mul_nonneg
  root := rootR
  root_nonneg := rootR_nonneg
  root_sq := rootR_sq

/-- **χ(ℝ²) ≥ 5** (conditional on the SAT leg `¬ VColourable`, discharged elsewhere by the
    cake_lpr-verified G₅₂₉ certificate). The real numbers, built Mathlib-free as the Cauchy-sequence
    quotient, form a `RootedField`; firing the abstract de Grey transfer gives the unit-distance
    plane on `ℝ²` chromatic number ≥ 5. -/
theorem chi_R2_ge_5 (h_sat : ¬ DeGrey529.Concrete.VColourable) :
    ¬ Nonempty (UnitDistanceChromatic.PlaneColouring (Real × Real)
      (DeGrey529.TransferWf.rootedTransfer rootedFieldReal).unit 4) :=
  DeGrey529.TransferWf.rootedField_chi_ge_5 rootedFieldReal h_sat

#print axioms rootedFieldReal
#print axioms chi_R2_ge_5

#eval IO.println "SounioRootedFieldReal: ℝ := Quotient realSetoid assembled into a Mathlib-free RootedField (ordered field + √3,√5,√7,√11, NO total sqrt, NO completeness); DeGrey529.TransferWf.rootedField_chi_ge_5 fired ⇒ χ(ℝ²) ≥ 5 (conditional on the SAT leg ¬VColourable)."

end SounioSqrt.RealCauchyField

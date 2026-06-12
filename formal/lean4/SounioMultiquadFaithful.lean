import SounioMultiquadIndep
import SounioMultiquadHom

set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

namespace SounioSqrt.RealCauchyField

open SounioSqrt SounioSqrt.RootedField SounioMultiquadHom
open MultiquadRing

/-!
# SounioMultiquadFaithful — bridging the native-basis 8-radical independence to `evalNum`

Mathlib-free (core Lean 4 only). This file connects the from-scratch ℝ (Cauchy-quotient)
8-radical ℚ-linear independence `indep8` (over the native `rootR`/`alpha4` basis,
`SounioMultiquadIndep`) to the abstract QF→ℝ numerator homomorphism `evalNum` of
`SounioMultiquadHom` (16 integer coefficients over `r i`).

Working ring: `R := rootedFieldReal`, so `R.F = Real`, `R.mul = mulR`, `R.add = addR`,
`R.one = oneR`, `R.zero = zeroR'`, `R.root j = rootR j`.

Main result: `evalNum_faithful_on_support` — on the √7-free support `{0,1,2,3,8,9,10,11}`
(masks `{1,√3,√5,√15,√11,√33,√55,√165}`), if `evalNum l = 0` and the eight √7-touching
coefficients vanish, then *all eight* used coefficients vanish. This is the de Grey
faithfulness claim at the abstract-homomorphism level.
-/

local notation "rR" => (@r rootedFieldReal)
local notation "oiR" => (@ofInt rootedFieldReal)
local notation "onR" => (@ofNat rootedFieldReal)
local notation "enR" => (@evalNum rootedFieldReal)

/-! ## Small ring helpers in `rootedFieldReal` -/

/-- Left unit for `mulR` (the in-file `mulR_one` is the right unit). -/
theorem oneR_mulR (x : Real) : mulR oneR x = x := by rw [mulR_comm, mulR_one]

/-- `addR` left-zero (the in-file `addR_zero` is the right-zero). -/
theorem zeroR'_addR (x : Real) : addR zeroR' x = x := by rw [addR_comm, addR_zero]

/-- `ofInt 0` is the additive zero in `rootedFieldReal`. -/
theorem ofIntR_zero : oiR 0 = zeroR' := rfl

/-! ## 1. The eight generator identities `r i` on the used (√7-free) support.

    Index→bits: bit0 = √3 = `rootR 0`, bit1 = √5 = `rootR 1`, bit3 = √11 = `rootR 3`
    (bit2 = √7 is never set on this support). Each `r m` is the nested 4-fold `R.mul`
    over `radicalBit m`; the unset bits contribute `oneR`, cleared by `mulR_one`/`oneR_mulR`. -/

theorem r0_R : rR 0 = oneR := r_zero

theorem r1_R : rR 1 = rootR 0 := by
  show mulR (rootR 0) (mulR oneR (mulR oneR oneR)) = rootR 0
  simp only [mulR_one, oneR_mulR]

theorem r2_R : rR 2 = rootR 1 := by
  show mulR oneR (mulR (rootR 1) (mulR oneR oneR)) = rootR 1
  simp only [mulR_one, oneR_mulR]

theorem r3_R : rR 3 = mulR (rootR 0) (rootR 1) := by
  show mulR (rootR 0) (mulR (rootR 1) (mulR oneR oneR)) = mulR (rootR 0) (rootR 1)
  simp only [mulR_one, oneR_mulR]

theorem r8_R : rR 8 = rootR 3 := by
  show mulR oneR (mulR oneR (mulR oneR (rootR 3))) = rootR 3
  simp only [mulR_one, oneR_mulR]

theorem r9_R : rR 9 = mulR (rootR 0) (rootR 3) := by
  show mulR (rootR 0) (mulR oneR (mulR oneR (rootR 3))) = mulR (rootR 0) (rootR 3)
  simp only [mulR_one, oneR_mulR]

theorem r10_R : rR 10 = mulR (rootR 1) (rootR 3) := by
  show mulR oneR (mulR (rootR 1) (mulR oneR (rootR 3))) = mulR (rootR 1) (rootR 3)
  simp only [mulR_one, oneR_mulR]

theorem r11_R : rR 11 = mulR (mulR (rootR 0) (rootR 1)) (rootR 3) := by
  show mulR (rootR 0) (mulR (rootR 1) (mulR oneR (rootR 3)))
      = mulR (mulR (rootR 0) (rootR 1)) (rootR 3)
  simp only [mulR_one, oneR_mulR]
  rw [mulR_assoc]

/-! ## 2. The `ofInt → qR` bridge: `ofInt n = qR (↑n)`.

    `ofInt`/`ofNat` are the ℤ/ℕ ring-hom casts into `R.F`; `qR` is the rational embedding into
    the Cauchy-quotient ℝ. They agree on the rational image of the integer. -/

/-- `ofNat n = qR (↑n)` (ℕ form), by induction mirroring `ofNat`'s definition. -/
theorem ofNatR_qR (n : Nat) : onR n = qR ((n : Rat)) := by
  induction n with
  | zero =>
    rw [show onR 0 = zeroR' from rfl,
        show ((0 : Nat) : Rat) = (0 : Rat) from by norm_cast, qR_zero]
  | succ n ih =>
    rw [show onR (n + 1) = addR (onR n) oneR from rfl,
        show ((n + 1 : Nat) : Rat) = ((n : Nat) : Rat) + 1 from by exact_mod_cast rfl,
        ← qR_one, ← qR_add, ih]

/-- **`ofInt → qR` bridge.** `ofInt n = qR (↑n)` for every integer `n`, via cases on
    `Int.ofNat`/`Int.negSucc` and the `ofNatR_qR` ℕ bridge (negatives through `qR_neg`). -/
theorem ofIntR_qR (n : Int) : oiR n = qR ((n : Rat)) := by
  match n with
  | Int.ofNat m =>
    rw [show oiR (Int.ofNat m) = onR m from rfl, ofNatR_qR m,
        show ((Int.ofNat m : Int) : Rat) = ((m : Nat) : Rat) from by exact_mod_cast rfl]
  | Int.negSucc m =>
    rw [show oiR (Int.negSucc m) = negR (onR (m + 1)) from rfl, ofNatR_qR (m + 1),
        show ((Int.negSucc m : Int) : Rat) = -(((m + 1 : Nat) : Rat)) from by exact_mod_cast rfl,
        ← qR_neg]

/-! ## 3. MAIN: faithfulness of `evalNum` on the √7-free support. -/

/-- **Faithfulness on the used support.** The eight used radical generators
    `{1,√3,√5,√15,√11,√33,√55,√165}` are ℤ-linearly independent in the from-scratch ℝ: if the
    QF→ℝ numerator map `evalNum l` is `0` and the eight √7-touching coefficients vanish, then all
    eight √7-free coefficients vanish too. Equivalently, `evalNum` is faithful on the √7-free
    support — the de Grey faithfulness claim at the abstract-homomorphism level. -/
theorem evalNum_faithful_on_support (l : List Int)
    (h7 : gi l 4 = 0 ∧ gi l 5 = 0 ∧ gi l 6 = 0 ∧ gi l 7 = 0 ∧
          gi l 12 = 0 ∧ gi l 13 = 0 ∧ gi l 14 = 0 ∧ gi l 15 = 0)
    (hz : (@evalNum rootedFieldReal l) = zeroR') :
    gi l 0 = 0 ∧ gi l 1 = 0 ∧ gi l 2 = 0 ∧ gi l 3 = 0 ∧
    gi l 8 = 0 ∧ gi l 9 = 0 ∧ gi l 10 = 0 ∧ gi l 11 = 0 := by
  obtain ⟨h4, h5, h6, h7', h12, h13, h14, h15⟩ := h7
  -- (a) collapse the 16-term fsum into an explicit right-nested `addR` tree (defeq).
  rw [show enR l
        = addR (mulR (oiR (gi l 0)) (rR 0))
          (addR (mulR (oiR (gi l 1)) (rR 1))
          (addR (mulR (oiR (gi l 2)) (rR 2))
          (addR (mulR (oiR (gi l 3)) (rR 3))
          (addR (mulR (oiR (gi l 4)) (rR 4))
          (addR (mulR (oiR (gi l 5)) (rR 5))
          (addR (mulR (oiR (gi l 6)) (rR 6))
          (addR (mulR (oiR (gi l 7)) (rR 7))
          (addR (mulR (oiR (gi l 8)) (rR 8))
          (addR (mulR (oiR (gi l 9)) (rR 9))
          (addR (mulR (oiR (gi l 10)) (rR 10))
          (addR (mulR (oiR (gi l 11)) (rR 11))
          (addR (mulR (oiR (gi l 12)) (rR 12))
          (addR (mulR (oiR (gi l 13)) (rR 13))
          (addR (mulR (oiR (gi l 14)) (rR 14))
          (addR (mulR (oiR (gi l 15)) (rR 15)) zeroR')
          )))))))))))))) from rfl] at hz
  -- (b) kill the eight √7-touching terms (coefficient 0 ⇒ `ofInt 0 = 0`, `0·x = 0`, drop).
  rw [h4, h5, h6, h7', h12, h13, h14, h15] at hz
  simp only [ofIntR_zero, zeroR'_mul, zeroR'_addR, addR_zero] at hz
  -- (c) rewrite the surviving generators into the native basis and `ofInt → qR`.
  rw [r0_R, r1_R, r2_R, r3_R, r8_R, r9_R, r10_R, r11_R, mulR_one,
      ofIntR_qR (gi l 0), ofIntR_qR (gi l 1), ofIntR_qR (gi l 2), ofIntR_qR (gi l 3),
      ofIntR_qR (gi l 8), ofIntR_qR (gi l 9), ofIntR_qR (gi l 10), ofIntR_qR (gi l 11)] at hz
  -- (d) reassemble the 8-term sum into the `indep8` shape `A + B·√11`.
  have hform :
      addR (alpha4 ((gi l 0 : Int) : Rat) ((gi l 1 : Int) : Rat)
                   ((gi l 2 : Int) : Rat) ((gi l 3 : Int) : Rat))
           (mulR (alpha4 ((gi l 8 : Int) : Rat) ((gi l 9 : Int) : Rat)
                         ((gi l 10 : Int) : Rat) ((gi l 11 : Int) : Rat)) (rootR 3))
        = addR (qR ((gi l 0 : Rat)))
          (addR (mulR (qR ((gi l 1 : Rat))) (rootR 0))
          (addR (mulR (qR ((gi l 2 : Rat))) (rootR 1))
          (addR (mulR (qR ((gi l 3 : Rat))) (mulR (rootR 0) (rootR 1)))
          (addR (mulR (qR ((gi l 8 : Rat))) (rootR 3))
          (addR (mulR (qR ((gi l 9 : Rat))) (mulR (rootR 0) (rootR 3)))
          (addR (mulR (qR ((gi l 10 : Rat))) (mulR (rootR 1) (rootR 3)))
                (mulR (qR ((gi l 11 : Rat))) (mulR (mulR (rootR 0) (rootR 1)) (rootR 3))))))))) := by
    unfold alpha4
    simp only [R15_def]
    rw [rightDistribR, rightDistribR, rightDistribR,
        mulR_assoc (qR ((gi l 9 : Rat))) (rootR 0) (rootR 3),
        mulR_assoc (qR ((gi l 10 : Rat))) (rootR 1) (rootR 3),
        mulR_assoc (qR ((gi l 11 : Rat))) (mulR (rootR 0) (rootR 1)) (rootR 3)]
    simp only [addR_assoc]
  rw [← hform, ← qR_zero] at hz
  -- (e) apply the 8-radical independence; cast injectivity ℚ→ℤ closes each coordinate.
  obtain ⟨e0, e1, e2, e3, e8, e9, e10, e11⟩ :=
    indep8 ((gi l 0 : Int) : Rat) ((gi l 1 : Int) : Rat) ((gi l 2 : Int) : Rat)
           ((gi l 3 : Int) : Rat) ((gi l 8 : Int) : Rat) ((gi l 9 : Int) : Rat)
           ((gi l 10 : Int) : Rat) ((gi l 11 : Int) : Rat) 0 hz
  exact ⟨by exact_mod_cast e0, by exact_mod_cast e1, by exact_mod_cast e2,
         by exact_mod_cast e3, by exact_mod_cast e8, by exact_mod_cast e9,
         by exact_mod_cast e10, by exact_mod_cast e11⟩

#print axioms r0_R
#print axioms r1_R
#print axioms r2_R
#print axioms r3_R
#print axioms r8_R
#print axioms r9_R
#print axioms r10_R
#print axioms r11_R
#print axioms ofNatR_qR
#print axioms ofIntR_qR
#print axioms evalNum_faithful_on_support

end SounioSqrt.RealCauchyField

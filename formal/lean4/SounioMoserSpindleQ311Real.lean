import Init.Data.Rat.Lemmas
import SounioMoserSpindleQ311
import SounioMultiquadParam

set_option maxHeartbeats 1000000

/-!
# SounioMoserSpindleQ311Real — χ(ℝ²) ≥ 4 via the Madore spindle in `Real`

Mathlib-free. The spindle embeds into `Real × Real` through the `ℚ(√3,√11)` basis
`alpha311 = alpha4(·,·,0,0) + alpha4(·,·,0,0)·√11`.  `E_mul311` certifies the
ring homomorphism on `Qf`; unit edges carry `dist² = 144` at the native `×12` integer
scale (Euclidean unit after global scale `1/12`).
-/

namespace MoserSpindleQ311.RealPlane

open MoserSpindleQ311
open UnitDistanceChromatic
open SounioSqrt.RealCauchyField
open SounioSqrt.RealCauchyField.Multiquad

def qneg (x : Qf) : Qf := ⟨-x.a, -x.b, -x.c, -x.d⟩

theorem qadd_sub (x y : Qf) : qadd x (qneg y) = qsub x y := rfl

/-- `a + b√3 + c√11 + d√33` over the de Grey `{1,√3,√5,√15}`/`√11` bridge. -/
noncomputable def alpha311 (a b c d : Rat) : Real :=
  addR (alpha4 a b 0 0) (mulR (alpha4 c d 0 0) (rootR 3))

noncomputable def phi311 (q : Qf) : Real :=
  alpha311 (q.a : Rat) (q.b : Rat) (q.c : Rat) (q.d : Rat)

/-- `alpha4 a b 0 0 = a + b√3` after the unused `{√5,√15}` slots vanish. -/
theorem alpha4_00 (a b : Rat) : alpha4 a b 0 0 = addR (qR a) (mulR (qR b) (rootR 0)) := by
  unfold alpha4
  simp [qR_zero, zeroR'_mul, addR_zero]

theorem alpha4_sub00 (a b : Rat) : alpha4 a b (0 - 0) (0 - 0) = alpha4 a b 0 0 := by
  congr 1 <;> native_decide

theorem negR_alpha4_00 (a b : Rat) :
    negR (alpha4 a b 0 0) = alpha4 (0 - a) (0 - b) 0 0 := by
  rw [negR_alpha4]
  exact alpha4_sub00 (0 - a) (0 - b)

theorem alpha4_add_00 (a b e f : Rat) :
    alpha4 (a + e) (b + f) 0 0 = addR (alpha4 a b 0 0) (alpha4 e f 0 0) := by
  rw [alpha4_00, alpha4_00, alpha4_00, ← qR_add, ← qR_add, rightDistribR (qR b) (qR f) (rootR 0),
      addR_swap4 (qR a) (qR e) (mulR (qR b) (rootR 0)) (mulR (qR f) (rootR 0))]

theorem alpha311_add (a b c d e f g h : Rat) :
    alpha311 (a + e) (b + f) (c + g) (d + h)
      = addR (alpha311 a b c d) (alpha311 e f g h) := by
  unfold alpha311
  rw [alpha4_add_00, alpha4_add_00, addR_swap4, addR_assoc, addR_assoc, rightDistribR]

theorem alpha311_neg (a b c d : Rat) :
    alpha311 (0 - a) (0 - b) (0 - c) (0 - d) = negR (alpha311 a b c d) := by
  unfold alpha311
  rw [negR_addR, negR_alpha4_00, negR_mulR_left, negR_alpha4_00]

theorem gen_mul11 (u v u' v' : Real) :
    mulR (addR u (mulR v (rootR 3))) (addR u' (mulR v' (rootR 3)))
      = addR (addR (mulR u u') (mulR (qR 11) (mulR v v')))
             (mulR (addR (mulR u v') (mulR u' v)) (rootR 3)) := by
  have e2 : mulR u (mulR v' (rootR 3)) = mulR (mulR u v') (rootR 3) := by rw [← mulR_assoc]
  have e3 : mulR (mulR v (rootR 3)) u' = mulR (mulR u' v) (rootR 3) := by
    rw [mulR_comm (mulR v (rootR 3)) u', ← mulR_assoc]
  have e4 : mulR (mulR v (rootR 3)) (mulR v' (rootR 3)) = mulR (qR 11) (mulR v v') := by
    rw [mulR_assoc v (rootR 3) (mulR v' (rootR 3)), ← mulR_assoc (rootR 3) v' (rootR 3),
      mulR_comm (rootR 3) v', mulR_assoc v' (rootR 3) (rootR 3), R11_sq,
      mulR_comm v' (qR 11), ← mulR_assoc v (qR 11) v', mulR_comm v (qR 11), mulR_assoc (qR 11) v v']
  rw [rightDistribR, leftDistribR, leftDistribR, e2, e3, e4, addR_regroup_ad,
      ← rightDistribR (mulR u v') (mulR u' v) (rootR 3)]

theorem alpha4_mul_sqrt3 (a b c d : Rat) :
    mulR (alpha4 a b 0 0) (alpha4 c d 0 0)
      = alpha4 (a * c + 3 * (b * d)) (a * d + b * c) 0 0 := by
  rw [E_mul35 a b 0 0 c d 0 0]
  unfold alpha4
  grind

theorem E_mul311 (a b c d e f g h : Rat) :
    mulR (alpha311 a b c d) (alpha311 e f g h)
      = alpha311 (a*e + 3*(b*f) + 11*(c*g) + 33*(d*h))
                 (a*f + b*e + 11*(c*h + d*g))
                 (a*g + c*e + 3*(b*h + d*f))
                 (a*h + b*g + c*f + d*e) := by
  unfold alpha311
  rw [gen_mul11, alpha4_mul_sqrt3 a b e f, alpha4_mul_sqrt3 c d g h,
    alpha4_mul_sqrt3 a b g h, alpha4_mul_sqrt3 e f c d]
  rw [alpha4_00, alpha4_00, alpha4_00, alpha4_00]
  have hA : mulR (qR 11) (addR (qR (c*g + 3*(d*h))) (mulR (qR (c*h + d*g)) (rootR 0)))
      = addR (qR (11*(c*g + 3*(d*h)))) (mulR (qR (11*(c*h + d*g))) (rootR 0)) := by
    rw [leftDistribR, qR_mul, ← mulR_assoc, qR_mul]
  have hC1 : (a*e + 3*(b*f)) + 11*(c*g + 3*(d*h)) = a*e + 3*(b*f) + 11*(c*g) + 33*(d*h) := by
    rw [Rat.mul_add, ← Rat.mul_assoc 11 3 (d*h),
      show (11:Rat)*3 = 33 from by native_decide, ← Rat.add_assoc]
  have hC3 : (a*f + b*e) + 11*(c*h + d*g) = a*f + b*e + 11*(c*h) + 11*(d*g) := by
    rw [Rat.mul_add, ← Rat.add_assoc]
  have hC11 : (a*g + 3*(b*h)) + (e*c + 3*(f*d)) = a*g + c*e + 3*(b*h) + 3*(d*f) := by
    rw [Rat.mul_comm e c, Rat.mul_comm f d, add_swap4 (a*g) (3*(b*h)) (c*e) (3*(d*f)),
      ← Rat.add_assoc (a*g + c*e) (3*(b*h)) (3*(d*f))]
  have hC33 : (a*h + b*g) + (e*d + f*c) = a*h + b*g + c*f + d*e := by
    rw [Rat.mul_comm e d, Rat.mul_comm f c, Rat.add_comm (d*e) (c*f),
      ← Rat.add_assoc (a*h + b*g) (c*f) (d*e)]
  have hPartA :
      addR (addR (qR (a*e+3*(b*f))) (mulR (qR (a*f+b*e)) (rootR 0)))
           (mulR (qR 11) (addR (qR (c*g+3*(d*h))) (mulR (qR (c*h+d*g)) (rootR 0))))
        = addR (qR (a*e + 3*(b*f) + 11*(c*g) + 33*(d*h)))
               (mulR (qR (a*f + b*e + 11*(c*h) + 11*(d*g))) (rootR 0)) := by
    rw [hA, addR_swap4, qR_add, ← rightDistribR (qR (a*f+b*e)) (qR (11*(c*h+d*g))) (rootR 0),
      qR_add, hC1, hC3]
  have hPartB :
      mulR (addR (addR (qR (a*g+3*(b*h))) (mulR (qR (a*h+b*g)) (rootR 0)))
                 (addR (qR (e*c+3*(f*d))) (mulR (qR (e*d+f*c)) (rootR 0))))
            (rootR 3)
        = addR (mulR (qR (a*g + c*e + 3*(b*h) + 3*(d*f))) (rootR 3))
               (mulR (qR (a*h + b*g + c*f + d*e)) (mulR (rootR 0) (rootR 3))) := by
    rw [addR_swap4, qR_add, ← rightDistribR (qR (a*h+b*g)) (qR (e*d+f*c)) (rootR 0), qR_add,
      hC11, hC33, rightDistribR, mulR_assoc (qR (a*h+b*g+c*f+d*e)) (rootR 0) (rootR 3)]
  rw [hPartA, hPartB, ← alpha4_00]
  rw [← mulR_assoc (qR (a*h + b*g + c*f + d*e)) (rootR 0) (rootR 3),
      ← rightDistribR (qR (a*g + c*e + 3*(b*h) + 3*(d*f)))
        (mulR (qR (a*h + b*g + c*f + d*e)) (rootR 0)) (rootR 3),
      ← alpha4_00]
  have hclose :
      addR (alpha4 (a*e + 3*(b*f) + 11*(c*g) + 33*(d*h)) (a*f + b*e + 11*(c*h) + 11*(d*g)) 0 0)
           (mulR (alpha4 (a*g + c*e + 3*(b*h) + 3*(d*f)) (a*h + b*g + c*f + d*e) 0 0) (rootR 3))
        = alpha311 (a*e + 3*(b*f) + 11*(c*g) + 33*(d*h))
                   (a*f + b*e + 11*(c*h + d*g))
                   (a*g + c*e + 3*(b*h + d*f))
                   (a*h + b*g + c*f + d*e) := by
    unfold alpha311
    congr 1
    · congr 1; exact hC3.symm
    · grind
  exact hclose

theorem phi311_add (x y : Qf) : phi311 (qadd x y) = addR (phi311 x) (phi311 y) := by
  rcases x with ⟨xa, xb, xc, xd⟩
  rcases y with ⟨ya, yb, yc, yd⟩
  simp only [phi311, alpha311, qadd]
  rw [show ((xa + ya : Int) : Rat) = (xa : Rat) + (ya : Rat) from by grind,
    show ((xb + yb : Int) : Rat) = (xb : Rat) + (yb : Rat) from by grind,
    show ((xc + yc : Int) : Rat) = (xc : Rat) + (yc : Rat) from by grind,
    show ((xd + yd : Int) : Rat) = (xd : Rat) + (yd : Rat) from by grind]
  exact alpha311_add _ _ _ _ _ _ _ _

theorem phi311_neg (x : Qf) : phi311 (qneg x) = negR (phi311 x) := by
  rcases x with ⟨xa, xb, xc, xd⟩
  simp only [phi311, alpha311, qneg]
  rw [show ((-xa : Int) : Rat) = (0 : Rat) - (xa : Rat) from by grind,
    show ((-xb : Int) : Rat) = (0 : Rat) - (xb : Rat) from by grind,
    show ((-xc : Int) : Rat) = (0 : Rat) - (xc : Rat) from by grind,
    show ((-xd : Int) : Rat) = (0 : Rat) - (xd : Rat) from by grind]
  exact alpha311_neg _ _ _ _

theorem phi311_sub (x y : Qf) : phi311 (qsub x y) = addR (phi311 x) (negR (phi311 y)) := by
  rw [← qadd_sub, phi311_add, phi311_neg]

theorem phi311_mul (x y : Qf) : phi311 (qmul x y) = mulR (phi311 x) (phi311 y) := by
  rcases x with ⟨xa, xb, xc, xd⟩
  rcases y with ⟨ya, yb, yc, yd⟩
  simp only [phi311, qmul]
  have h0 := show ((xa * ya + 3 * xb * yb + 11 * xc * yc + 33 * xd * yd : Int) : Rat) =
      (xa : Rat) * (ya : Rat) + 3 * (xb : Rat) * (yb : Rat) + 11 * (xc : Rat) * (yc : Rat) + 33 * (xd : Rat) * (yd : Rat) from by grind
  have h1 := show ((xa * yb + ya * xb + 11 * (xc * yd + yc * xd) : Int) : Rat) =
      (xa : Rat) * (yb : Rat) + (xb : Rat) * (ya : Rat) + 11 * ((xc : Rat) * (yd : Rat) + (xd : Rat) * (yc : Rat)) from by grind
  have h2 := show ((xa * yc + ya * xc + 3 * (xb * yd + yb * xd) : Int) : Rat) =
      (xa : Rat) * (yc : Rat) + (xc : Rat) * (ya : Rat) + 3 * ((xb : Rat) * (yd : Rat) + (xd : Rat) * (yb : Rat)) from by grind
  have h3 := show ((xa * yd + ya * xd + xb * yc + yb * xc : Int) : Rat) =
      (xa : Rat) * (yd : Rat) + (xb : Rat) * (yc : Rat) + (xc : Rat) * (yb : Rat) + (xd : Rat) * (ya : Rat) from by grind
  rw [h0, h1, h2, h3]
  have hform :
      alpha311 (↑xa * ↑ya + 3 * ↑xb * ↑yb + 11 * ↑xc * ↑yc + 33 * ↑xd * ↑yd)
               (↑xa * ↑yb + ↑xb * ↑ya + 11 * (↑xc * ↑yd + ↑xd * ↑yc))
               (↑xa * ↑yc + ↑xc * ↑ya + 3 * (↑xb * ↑yd + ↑xd * ↑yb))
               (↑xa * ↑yd + ↑xb * ↑yc + ↑xc * ↑yb + ↑xd * ↑ya)
        = alpha311 (↑xa * ↑ya + 3 * (↑xb * ↑yb) + 11 * (↑xc * ↑yc) + 33 * (↑xd * ↑yd))
                   (↑xa * ↑yb + ↑xb * ↑ya + 11 * (↑xc * ↑yd + ↑xd * ↑yc))
                   (↑xa * ↑yc + ↑xc * ↑ya + 3 * (↑xb * ↑yd + ↑xd * ↑yb))
                   (↑xa * ↑yd + ↑xb * ↑yc + ↑xc * ↑yb + ↑xd * ↑ya) := by
    congr 1 <;> grind
  exact hform.trans (E_mul311 (xa : Rat) (xb : Rat) (xc : Rat) (xd : Rat) (ya : Rat) (yb : Rat) (yc : Rat) (yd : Rat)).symm

theorem phi311_sq (x : Qf) : phi311 (qsq x) = mulR (phi311 x) (phi311 x) := by
  unfold qsq; exact phi311_mul x x

def subR (x y : Real) : Real := addR x (negR y)
def sqR (x : Real) : Real := mulR x x

def dist2Real (p q : Real × Real) : Real :=
  addR (sqR (subR p.1 q.1)) (sqR (subR p.2 q.2))

def unitReal144 (p q : Real × Real) : Prop := dist2Real p q = qR 144

noncomputable def embReal (i : Fin 7) : Real × Real := (phi311 (vx i), phi311 (vy i))

theorem dist2Real_emb (i j : Fin 7) :
    dist2Real (embReal i) (embReal j) = phi311 (dist2 i j) := by
  unfold dist2Real embReal dist2 dist2pt sqR subR
  rw [show qsub (vx i, vy i).fst (vx j, vy j).fst = qsub (vx i) (vx j) from rfl,
      show qsub (vx i, vy i).snd (vx j, vy j).snd = qsub (vy i) (vy j) from rfl]
  rw [← phi311_sub, ← phi311_sub, ← phi311_sq, ← phi311_sq, phi311_add]

theorem embReal_unit144 (e : Fin 7 × Fin 7) (he : e ∈ spindleG.edges) :
    unitReal144 (embReal e.1) (embReal e.2) := by
  unfold unitReal144
  rw [dist2Real_emb, edges_unit e he]
  simp only [phi311, alpha311]
  rw [alpha4_00, alpha4_00]
  simp [qR_zero, zeroR'_mul, addR_zero]

theorem chi_R2_ge_4 :
    ¬ Nonempty (PlaneColouring (Real × Real) unitReal144 3) :=
  not_colourable_implies_plane_chromatic_gt spindleG unitReal144 embReal embReal_unit144
    spindle_not_3_colourable

#print axioms chi_R2_ge_4

#eval IO.println "SounioMoserSpindleQ311Real: χ(ℝ²) ≥ 4 (Madore spindle embedded in Mathlib-free Real)."

end MoserSpindleQ311.RealPlane

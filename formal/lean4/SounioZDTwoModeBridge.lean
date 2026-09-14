import SounioZDFiberAntisym
import SounioZDTwoMode

/-! Concrete bridge to the existing Cayley-Dickson recursions.
Only s3 and cp2 trajectories are classified; no full-spectrum or label-equivalence claim.
-/
namespace Sounio.ZDTwoModeBridge
open SounioZDFiberAntisym

def s (m W : Nat) : Int := tri3 (2^(m+1)) (fun x y => P3 x y W m)
def c (m W : Nat) : Int :=
  sumLtI (2^(m+1)) (fun a => sumLtI (2^(m+1)) (fun b =>
    P3 a b W m * P3 b (a ^^^ W) W m))

theorem step (m W V : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hV : V < 2^(m+1)) (hV0 : V ≠ 0) :
    s (m+1) W - s (m+1) V = 8*(s m W-s m V)+24*(c m W-c m V) ∧
    c (m+1) W - c (m+1) V = 4*(c m W-c m V) := by
  have sw := s3_level_recursion m W hW hW0
  have sv := s3_level_recursion m V hV hV0
  have cw := cp2_level_recursion m W hW hW0
  have cv := cp2_level_recursion m V hV hV0
  have hp : (2:Nat)^(m+1+1) = 2^(m+1)+2^(m+1) := by
    rw [Nat.pow_succ]; omega
  unfold s c
  rw [hp]
  constructor <;> omega

theorem transfer (j W V : Nat) (hW : W < 2^(j+1)) (hW0 : W ≠ 0)
    (hV : V < 2^(j+1)) (hV0 : V ≠ 0) :
    ZDTwoMode.Transfer (fun i => s (j+i) W-s (j+i) V)
      (fun i => c (j+i) W-c (j+i) V) := by
  intro i
  have hp : (2:Nat)^(j+1) ≤ 2^(j+i+1) :=
    Nat.pow_le_pow_right (by omega) (by omega)
  have hh := step (j+i) W V (by omega) hW0 (by omega) hV0
  simpa only [Nat.add_assoc] using hh

theorem closed (j i W V : Nat) (hW : W < 2^(j+1)) (hW0 : W ≠ 0)
    (hV : V < 2^(j+1)) (hV0 : V ≠ 0) :
    s (j+i) W-s (j+i) V =
      (8:Int)^i * ((s j W-s j V)+6*(c j W-c j V))
        -6*(4:Int)^i*(c j W-c j V) := by
  simpa using ZDTwoMode.closed _ _ (transfer j W V hW hW0 hV hV0) i

theorem homogeneous (j i W V : Nat) (hW : W < 2^(j+1)) (hW0 : W ≠ 0)
    (hV : V < 2^(j+1)) (hV0 : V ≠ 0) (hc : c j W = c j V) :
    c (j+i) W = c (j+i) V ∧
    s (j+i) W-s (j+i) V = (8:Int)^i*(s j W-s j V) := by
  have hh := ZDTwoMode.homogeneous_of_initial_eq _ _
    (transfer j W V hW hW0 hV hV0) (by simp [hc]) i
  simp only [Nat.add_zero] at hh
  constructor
  · omega
  · exact hh.2

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 0 in
/-- Exact CD seed: s3 alone cannot distinguish these labels at level 3. -/
theorem seed_W12 : s 3 12 - s 3 1 = 0 ∧ c 3 12 - c 3 1 = 192 := by
  decide

/-- The general transfer explains the previous W12 negative control at every later level. -/
theorem W12_all_levels (i : Nat) :
    s (3+i) 12 - s (3+i) 1 = 1152 * ((8:Int)^i - (4:Int)^i) := by
  have h := closed 3 i 12 1 (by decide) (by decide) (by decide) (by decide)
  rw [seed_W12.1, seed_W12.2] at h
  grind

/-- Equal s3 at one level is insufficient to predict equal s3 at the next. -/
theorem s3_alone_insufficient :
    s 3 12 = s 3 1 ∧ s 4 12 - s 4 1 = 4608 := by
  have hs := seed_W12.1
  have ht := W12_all_levels 1
  have hn : 1152 * ((8:Int)^1 - (4:Int)^1) = 4608 := by decide
  rw [hn] at ht
  constructor
  · omega
  · exact ht

#print axioms seed_W12
#print axioms W12_all_levels
#print axioms closed
#print axioms homogeneous
end Sounio.ZDTwoModeBridge

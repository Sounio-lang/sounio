import SounioZDScalarNativeBoundary

/-! A persistent density certificate for exact search over arbitrary C/T words.
The certificate is closed under both extensions and excludes every later
small-cone reset/T collision. The external finite enumeration is not a Lean
proof of unbounded ResidualSeparation. -/
namespace Sounio.ZDScalarResidualSearch
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility
open Sounio.ZDScalarNativeBoundary
set_option maxRecDepth 16384
set_option maxHeartbeats 16000000

def PruneCertificate {d : Nat} (x : Code d) : Prop :=
  3*gap x ≤ 3*edges x+4*positives x ∧ 9*order d ≤ 16*positives x

/-- Words are applied from the seed outwards, exactly as in the search. -/
inductive Extension {d : Nat} (x : Code d) : {e : Nat} → Code e → Prop
  | refl : Extension x x
  | t {e : Nat} {y : Code e} : Extension x y → Extension x (.t y)
  | c {e : Nat} {y : Code e} : Extension x y → Extension x (.c y)

theorem certificate_t {d : Nat} (x : Code d) (hx : PruneCertificate x) :
    PruneCertificate (.t x) := by
  rcases hx with ⟨hd,hp⟩
  have hq := order_ge d
  constructor
  · simp only [gap_t,edges,positives]
    omega
  · simp only [order,positives]
    omega

theorem certificate_c {d : Nat} (x : Code d) (hx : PruneCertificate x) :
    PruneCertificate (.c x) := by
  rcases hx with ⟨hd,hp⟩
  have hq := order_ge d
  constructor
  · simp only [gap_c,edges,positives]
    omega
  · simp only [order,positives]
    omega

theorem certificate_extension {d e : Nat} (x : Code d) (y : Code e)
    (hx : PruneCertificate x) (h : Extension x y) : PruneCertificate y := by
  induction h with
  | refl => exact hx
  | t h ih => exact certificate_t _ ih
  | c h ih => exact certificate_c _ ih

theorem zero_origin_extension {d : Nat} (x : Code d) (hx : coreOrigin x=false) :
    Extension Code.z x := by
  induction x with
  | k => simp [coreOrigin] at hx
  | z => exact .refl
  | reset => simp [coreOrigin] at hx
  | t x ih => exact .t (ih hx)
  | c x ih => exact .c (ih hx)

theorem certificate_numerator_positive {d : Nat} (x : Code d)
    (hx : PruneCertificate x) : 0<3*edges x+4*positives x := by
  have hq := order_ge d
  have hp := hx.2
  omega

theorem certificate_excludes_low {d e : Nat} (x : Code d) (y : Code e)
    (hx : PruneCertificate x) (h : Extension x y) :
    ¬(3*edges y+4*positives y<3*gap y) := by
  have hd := (certificate_extension x y hx h).1
  omega

private theorem weighted_impossible (g N a b : Nat)
    (hN : 0<N) (hd : 3*g≤N) (hab : a<3*b) : a*g≠b*N := by
  intro he
  by_cases hg : g=0
  · have hb : 0<b := by omega
    have hh := Nat.mul_pos hb hN
    simp only [hg,Nat.mul_zero] at he
    omega
  · have hgp : 0<g := Nat.pos_of_ne_zero hg
    have hl := Nat.mul_lt_mul_of_pos_right hab hgp
    have hu := Nat.mul_le_mul_left b hd
    have hc : (3*b)*g=b*(3*g) := by grind only
    rw [hc] at hl
    omega

theorem certificate_weighted_ne {d : Nat} (x : Code d) (a b : Nat)
    (hx : PruneCertificate x) (hab : a<3*b) :
    a*gap x ≠ b*(3*edges x+4*positives x) :=
  weighted_impossible _ _ _ _ (certificate_numerator_positive x hx) hx.1 hab

/-- Native graph statement: no parity restriction is required in a certified subtree. -/
theorem certificate_native_ne {d e : Nat} (x : Code d) (y : Code e) (a b : Nat)
    (hx : PruneCertificate x) (h : Extension x y) (hab : a<3*b) :
    weightedCount (Code.reset (d:=e)) a b ≠ weightedCount (.t y) a b := by
  intro he
  have hd := certificate_extension x y hx h
  have hh := (weighted_reset_collision_iff y a b).mp he
  exact certificate_weighted_ne y a b hd hab hh

theorem order_le_capacity (d : Nat) : order d≤capacity d := by
  cases d with
  | zero => decide
  | succ d =>
    have hq := order_ge d
    simp only [order,capacity]
    omega

/-- A node that is actually expanded has small p, controlling integer width. -/
theorem unpruned_positive_bound {d : Nat} (x : Code d)
    (hx : ¬PruneCertificate x) : positives x≤capacity d := by
  by_cases hp : positives x≤capacity d
  · exact hp
  · have hg := gap_add_edges x
    have hq := order_le_capacity d
    have hc : PruneCertificate x := by constructor <;> omega
    exact False.elim (hx hc)

theorem expanded_child_positive_bound {d : Nat} (x : Code d)
    (hx : ¬PruneCertificate x) :
    positives (.t x)≤14*capacity d ∧ positives (.c x)≤14*capacity d := by
  have hp := unpruned_positive_bound x hx
  have hm := edges_le_capacity x
  simp only [positives]
  omega

/-- A strict dense node is prunable once the second inequality has been checked. -/
theorem pruning_check_sound {d : Nat} (x : Code d)
    (hN : 3*edges x+4*positives x≥3*gap x)
    (hp : 16*positives x≥9*order d) : PruneCertificate x := ⟨hN,hp⟩


def controlTriples : List (Nat × Nat × Nat) :=
  [(0,1,9),(1,1,3),(3,4,747),(6,7,48987),(9,10,3144411),
   (12,13,201316059),(15,16,12884817627),(18,19,824633046747),
   (21,22,52776552740571)]

def controlHolds (r k a : Nat) : Prop :=
  3*edges (towerT k (zeroAt r))+4*positives (towerT k (zeroAt r))=a*capacity r ∧
  a%2=1 ∧ a%3=0 ∧ 0<a ∧ a<3*4^k
deriving Decidable

/-- Positive controls, including every collision reported by the depth48 search.
The kernel checks these finite arithmetic equalities, not the search completeness. -/
theorem known_collision_controls :
    controlTriples.all (fun row => decide (controlHolds row.1 row.2.1 row.2.2))=true := by
  decide +kernel

theorem zero_tower_native_collision (r k a : Nat)
    (h : controlHolds r k a) :
    weightedCount (Code.reset (d:=r+k)) a (4^k) =
      weightedCount (.t (towerT k (zeroAt r))) a (4^k) := by
  apply (weighted_reset_collision_iff (towerT k (zeroAt r)) a (4^k)).mpr
  have hg := tower_gap (zeroAt r) k
  have hz : gap (zeroAt r)=capacity r := by simp [gap,(zeroAt_counts r).1]
  rw [hz] at hg
  rw [hg,h.1]
  grind only

#print axioms certificate_t
#print axioms certificate_c
#print axioms certificate_extension
#print axioms zero_origin_extension
#print axioms certificate_numerator_positive
#print axioms certificate_excludes_low
#print axioms weighted_impossible
#print axioms certificate_weighted_ne
#print axioms certificate_native_ne
#print axioms order_le_capacity
#print axioms unpruned_positive_bound
#print axioms expanded_child_positive_bound
#print axioms pruning_check_sound
#print axioms known_collision_controls
#print axioms zero_tower_native_collision
end Sounio.ZDScalarResidualSearch

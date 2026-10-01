import SounioZDScalarCrossOrigin

/-! Growth and finite-tail bounds for the still-open scalar divisibility problem. -/
namespace Sounio.ZDScalarDivisibility
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def towerT {d : Nat} : (k : Nat) → Code d → Code (d+k)
  | 0, x => x
  | k+1, x => .t (towerT k x)

def zeroAt : (d : Nat) → Code d
  | 0 => .z
  | d+1 => .c (zeroAt d)

def energy {d : Nat} (x : Code d) : Nat :=
  42*edges x+28*positives x+21*order d+3

theorem zeroAt_counts (d : Nat) :
    edges (zeroAt d)=0 ∧ positives (zeroAt d)=0 := by
  induction d with
  | zero => decide
  | succ d ih => simp [zeroAt,edges,positives,ih]

theorem energy_t {d : Nat} (x : Code d) :
    energy (.t x)=8*energy x := by
  simp only [energy,edges,positives,order]
  grind

theorem tower_energy {d : Nat} (x : Code d) (k : Nat) :
    energy (towerT k x)=8^k*energy x := by
  induction k with
  | zero => simp [towerT]
  | succ k ih =>
    simp only [towerT,energy_t,ih,Nat.pow_succ]
    grind

theorem tower_gap {d : Nat} (x : Code d) (k : Nat) :
    gap (towerT k x)=4^k*gap x := by
  induction k with
  | zero => simp [towerT]
  | succ k ih =>
    simp only [towerT,gap_t,ih,Nat.pow_succ]
    grind

theorem tower_positive_lower {d : Nat} (x : Code d) (k : Nat) :
    8^k*(252*order d+18) ≤ positives (towerT (k+3) x) := by
  induction k with
  | zero =>
    simp only [Nat.pow_zero,Nat.one_mul,towerT,positives,edges,order]
    have horder : order (d.add 0)=order d := rfl
    omega
  | succ k ih =>
    change 8^(k+1)*(252*order d+18) ≤
      6*edges (towerT (k+3) x)+8*positives (towerT (k+3) x)
    have hm := Nat.mul_le_mul_left 8 ih
    simp only [Nat.pow_succ]
    grind

theorem dense_t {d : Nat} (x : Code d)
    (hx : 3*capacity d < 3*edges x+4*positives x) :
    3*capacity (d+1) < 3*edges (.t x)+4*positives (.t x) := by
  simp only [capacity,edges,positives]
  omega

/-- After r+2 exterior T steps over any depth-r tail the density exceeds
the reset comparison threshold. -/
theorem tower_density_cutoff {r : Nat} (x : Code r) :
    3*capacity (r+(r+2)) <
      3*edges (towerT (r+2) x)+4*positives (towerT (r+2) x) := by
  cases r with
  | zero => cases x <;> decide
  | succ s =>
    have hq := order_scale (s+1)
    have htwo := Nat.two_pow_pos s
    simp only [Nat.pow_succ] at hq
    have hqlo : 7*2^s ≤ order (s+1) := by omega
    have hlo := tower_positive_lower x s
    have hfactor : 1764*2^s ≤ 252*order (s+1)+18 := by omega
    have hm := Nat.mul_le_mul_left (8^s) hfactor
    have hp16 : 8^s*2^s=16^s := by rw [← Nat.mul_pow]
    have hid : 8^s*(1764*2^s)=1764*16^s := by rw [← hp16]; grind
    rw [hid] at hm
    have hp : 1764*16^s ≤ positives (towerT (s+3) x) := Nat.le_trans hm hlo
    have hc := (capacity_bounds ((s+1)+(s+3))).2
    have hpower : 4^((s+1)+(s+3))=256*16^s := by
      have he : (s+1)+(s+3)=2*s+4 := by omega
      rw [he]
      rw [Nat.pow_add, Nat.pow_mul]
      simp only [show (4:Nat)^2=16 from rfl, show (4:Nat)^4=256 from rfl]
      exact Nat.mul_comm _ _
    rw [hpower] at hc
    have hpos : 0<16^s := Nat.pow_pos (by decide)
    change 3*capacity ((s+1)+(s+3)) <
      3*edges (towerT (s+3) x)+4*positives (towerT (s+3) x)
    omega

theorem tower_density_more {r : Nat} (x : Code r) (s : Nat) :
    3*capacity (r+(r+2+s)) <
      3*edges (towerT (r+2+s) x)+4*positives (towerT (r+2+s) x) := by
  induction s with
  | zero => exact tower_density_cutoff x
  | succ s ih => exact dense_t (towerT (r+2+s) x) ih

theorem tower_density_after {r : Nat} (x : Code r) (k : Nat)
    (hk : r+2≤k) :
    3*capacity (r+k) <
      3*edges (towerT k x)+4*positives (towerT k x) := by
  have h : k=r+2+(k-(r+2)) := by omega
  rw [h]
  exact tower_density_more x _

theorem reset_ne_long_tower {r : Nat} (x : Code r) (k a b : Nat)
    (hk : r+2≤k) (hb : 0<b) (hab : 2*a≤3*b) :
    value a b (Code.reset (d:=r+k)) ≠ value a b (.t (towerT k x)) :=
  Nat.ne_of_lt (reset_lt_t_of_density (towerT k x) hb hab
    (tower_density_after x k hk))

theorem reset_collision_tower_bound {r : Nat} (x : Code r) (k a b : Nat)
    (hb : 0<b) (hab : 2*a≤3*b)
    (he : value a b (Code.reset (d:=r+k)) =
      value a b (.t (towerT k x))) : k≤r+1 := by
  by_cases hk : k≤r+1
  · exact hk
  · exact False.elim (reset_ne_long_tower x k a b (by omega) hb hab he)

theorem power_four_le_eight (r : Nat) : 4^r≤8^r := by
  induction r with
  | zero => decide
  | succ r ih => simp only [Nat.pow_succ]; omega

/-- The cutoff is sharp as a uniform density bound: r+1 exterior T steps
over the all-C zero tail still lie below even the stronger gap threshold. -/
theorem tower_boundary_low (r : Nat) :
    3*edges (towerT (r+1) (zeroAt r))+
      4*positives (towerT (r+1) (zeroAt r)) <
      3*gap (towerT (r+1) (zeroAt r)) := by
  let x := towerT (r+1) (zeroAt r)
  have hbase := zeroAt_counts r
  have hq := order_scale r
  have he := tower_energy (zeroAt r) (r+1)
  have h816 : 8^r*2^r=16^r := by rw [← Nat.mul_pow]
  have hi : energy x+144*8^r=672*16^r := by
    simp only [energy,hbase.1,hbase.2,Nat.mul_zero,Nat.zero_add] at he
    simp only [Nat.pow_succ] at he
    change (42*edges x+28*positives x+21*order (r+(r+1))+3)+
      144*8^r=672*16^r
    dsimp only [x]
    grind
  have hp2 : 2^(r+(r+1))=2*4^r := by
    have he : r+(r+1)=2*r+1 := by omega
    rw [he]
    rw [Nat.pow_add, Nat.pow_mul]
    simp only [show (2:Nat)^2=4 from rfl, Nat.pow_one]
    exact Nat.mul_comm _ _
  have hp4 : 4^(r+(r+1))=4*16^r := by
    have he : r+(r+1)=2*r+1 := by omega
    rw [he]
    rw [Nat.pow_add, Nat.pow_mul]
    simp only [show (4:Nat)^2=16 from rfl, Nat.pow_one]
    exact Nat.mul_comm _ _
  have hc := capacity_scale (r+(r+1))
  have hqx := order_scale (r+(r+1))
  rw [hp2,hp4] at hc
  rw [hp2] at hqx
  have hm := power_four_le_eight r
  have hpos : 0<8^r := Nat.pow_pos (by decide)
  have hg := gap_add_edges x
  simp only [energy] at hi
  change 3*edges x+4*positives x<3*gap x
  omega


theorem headT_tower {r : Nat} (x : Code r) (k : Nat) :
    headT (towerT k x)=headT x+k := by
  induction k with
  | zero => simp [towerT]
  | succ k ih => simp [towerT,headT,ih,Nat.add_assoc]

/-- Every candidate coefficient exponent is bounded by the parent depth,
when the trailing code has no initial T. -/
theorem reset_collision_coefficient_bound {r : Nat} (x : Code r)
    (k a j : Nat) (hx : headT x=0) (ha : a%2=1)
    (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=r+k)) =
      value a (4*2^j) (.t (towerT k x))) : j≤r+k := by
  have hj := reset_collision_exponent (towerT k x) a j ha he
  rw [headT_tower,hx,Nat.zero_add] at hj
  have hp : 0<2^j := Nat.two_pow_pos j
  have hk := reset_collision_tower_bound x k a (4*2^j) (by omega) (by omega) he
  omega

/-- A counterexample to omitting the small-ratio condition, not to the
admissible scalar separation conjecture. -/
def outsideTail : Code 24 := towerT 20 (zeroAt 4)

theorem outside_counts :
    edges outsideTail=104453503975425 ∧
    positives outsideTail=54598911716432244150 ∧
    gap outsideTail=2147346209046528 := by decide

theorem outside_dyadic_control :
    3*edges outsideTail+4*positives outsideTail =
      3*651*111825888492698875 ∧
    gap outsideTail=3*651*4^20 ∧
    111825888492698875%2=1 ∧
    111825888492698875%3=1 ∧
    6*2^39<111825888492698875 := by decide

theorem outside_reset_collision :
    value 111825888492698875 (4*2^39) (Code.reset (d:=24)) =
    value 111825888492698875 (4*2^39) (.t outsideTail) := by decide

#print axioms headT_tower
#print axioms reset_collision_coefficient_bound
#print axioms outside_counts
#print axioms outside_dyadic_control
#print axioms outside_reset_collision

#print axioms zeroAt_counts
#print axioms energy_t
#print axioms tower_energy
#print axioms tower_gap
#print axioms tower_positive_lower
#print axioms dense_t
#print axioms tower_density_cutoff
#print axioms tower_density_more
#print axioms tower_density_after
#print axioms reset_ne_long_tower
#print axioms reset_collision_tower_bound
#print axioms power_four_le_eight
#print axioms tower_boundary_low
end Sounio.ZDScalarDivisibility

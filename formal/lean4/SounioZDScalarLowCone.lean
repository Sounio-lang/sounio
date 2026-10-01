import SounioZDScalarDivisibility

/-! Excluding the boundary strip in the admissible scalar cone. -/
namespace Sounio.ZDScalarLowCone
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

theorem positive_edges_lower {r : Nat} (x : Code r) (hx : 0<edges x) :
    3*order r ≤ 2*edges x+3 := by
  induction x with
  | k => decide
  | z => simp [edges] at hx
  | @reset d =>
    have hr := reset_gt d
    simp only [edges,order]
    omega
  | @t d x ih =>
    simp only [edges,order]
    omega
  | @c d x ih =>
    have hm : 0<edges x := by simpa [edges] using hx
    have h := ih hm
    have hq := order_ge d
    simp only [edges,order]
    omega

theorem edges_zero_code {r : Nat} (x : Code r) (hx : edges x=0) :
    x=zeroAt r := by
  induction x with
  | k => simp [edges] at hx
  | z => rfl
  | @reset d =>
    have hr := reset_gt d
    simp only [edges] at hx
    omega
  | @t d x ih =>
    have hq := order_ge d
    simp only [edges] at hx
    omega
  | @c d x ih =>
    have hm : edges x=0 := by simp only [edges] at hx; omega
    simp only [zeroAt,ih hm]

theorem power_eight_le_sixteen (r : Nat) : 8^r≤16^r := by
  induction r with
  | zero => decide
  | succ r ih => simp only [Nat.pow_succ]; omega

theorem positive_tail_energy {r : Nat} (x : Code r) (hx : 0<edges x) :
    336*2^r ≤ energy x+144 := by
  have hm := positive_edges_lower x hx
  have hq := order_scale r
  simp only [energy]
  omega

/-- The exceptional all-C zero tail is the only possible non-dense boundary tail. -/
theorem positive_tail_boundary_dense {r : Nat} (x : Code r)
    (hx : 0<edges x) :
    3*capacity (r+(r+1)) <
      3*edges (towerT (r+1) x)+4*positives (towerT (r+1) x) := by
  let y := towerT (r+1) x
  have he := tower_energy x (r+1)
  have hi := positive_tail_energy x hx
  have hm := Nat.mul_le_mul_left (8^(r+1)) hi
  have h816 : 8^r*2^r=16^r := by rw [← Nat.mul_pow]
  have hl : 2688*16^r ≤ energy y+1152*8^r := by
    simp only [Nat.pow_succ] at he hm
    dsimp only [y]
    grind
  have h8 := power_eight_le_sixteen r
  have hlo : 1536*16^r ≤ energy y := by omega
  have hp2 : 2^(r+(r+1))=2*4^r := by
    have he : r+(r+1)=2*r+1 := by omega
    rw [he,Nat.pow_add,Nat.pow_mul]
    simp only [show (2:Nat)^2=4 from rfl,Nat.pow_one]
    exact Nat.mul_comm _ _
  have hp4 : 4^(r+(r+1))=4*16^r := by
    have he : r+(r+1)=2*r+1 := by omega
    rw [he,Nat.pow_add,Nat.pow_mul]
    simp only [show (4:Nat)^2=16 from rfl,Nat.pow_one]
    exact Nat.mul_comm _ _
  have hc := capacity_scale (r+(r+1))
  have hq := order_scale (r+(r+1))
  rw [hp2,hp4] at hc
  rw [hp2] at hq
  have hcap := edges_le_capacity y
  have hpos : 0<16^r := Nat.pow_pos (by decide)
  simp only [energy] at hlo
  change 3*capacity (r+(r+1)) < 3*edges y+4*positives y
  omega

theorem zeroAt_headT (r : Nat) : headT (zeroAt r)=0 := by
  cases r <;> rfl

/-- The exact coefficient equation on the all-C boundary forces a multiple of 3. -/
theorem zero_boundary_coefficient (r a j : Nat) (ha : a%2=1)
    (he : value a (4*2^j) (Code.reset (d:=r+(r+1))) =
      value a (4*2^j) (.t (towerT (r+1) (zeroAt r)))) :
    7*a+9*2^(r+1)+3=21*(2^(r+1))^2 := by
  let t := 2^(r+1)
  let y := towerT (r+1) (zeroAt r)
  have hj := reset_collision_exponent y a j ha he
  change j+1=2*headT (towerT (r+1) (zeroAt r)) at hj
  rw [headT_tower,zeroAt_headT,Nat.zero_add] at hj
  have ht2 : 2^(j+1)=t*t := by
    rw [hj,show 2*(r+1)=(r+1)*2 by omega,Nat.pow_mul]
    simp only [t,Nat.pow_succ,Nat.pow_zero,Nat.one_mul]
  have ht4 : 4^(r+1)=t*t := by
    rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
  have ht8 : 8^(r+1)=t*t*t := by
    rw [show (8:Nat)=2*2*2 from rfl,Nat.mul_pow,Nat.mul_pow]
  have hb := zeroAt_counts r
  have hgap := tower_gap (zeroAt r) (r+1)
  have hgb : gap (zeroAt r)=capacity r := by simp [gap,hb.1]
  rw [hgb,ht4] at hgap
  have hcoll := reset_eq_gap y a j he
  rw [ht2] at hcoll
  have hq0 := order_scale r
  have hq1 := order_scale (r+(r+1))
  have hpow : 2^(r+(r+1))=2^r*2^(r+1) := Nat.pow_add _ _ _
  rw [hpow] at hq1
  have htt : t=2*2^r := by dsimp [t]; rw [Nat.pow_succ,Nat.mul_comm]
  have hc0 := capacity_identity r
  have hc1 := capacity_identity (r+(r+1))
  have heg := gap_add_edges y
  have hei := tower_energy (zeroAt r) (r+1)
  rw [ht8] at hei
  have ht : 0<t := Nat.two_pow_pos (r+1)
  have hproduct : (a*capacity r)*(t*t)=
      (3*edges y+4*positives y)*(t*t) := by
    change a*gap (towerT (r+1) (zeroAt r))=
      t*t*(3*edges y+4*positives y) at hcoll
    rw [hgap] at hcoll
    grind only
  have hcancel := Nat.eq_of_mul_eq_mul_right (Nat.mul_pos ht ht) hproduct
  have hq0t : order r+1=2*t := by omega
  have hq1t : order (r+(r+1))+1=2*t*t := by
    change order (r+(r+1))+1=4*(2^r*t) at hq1
    grind only
  have hh : (7*a+9*t+3)*(capacity r)=
      (21*t^2)*(capacity r) := by
    simp only [energy,hb.1,hb.2,Nat.mul_zero,Nat.zero_add] at hei
    have hge : gap y=t*t*capacity r := hgap
    grind only
  have hcap : 0<capacity r := by
    have h := (capacity_bounds r).1
    have hp : 0<4^r := Nat.pow_pos (by decide)
    omega
  have h := Nat.eq_of_mul_eq_mul_right hcap hh
  exact h

theorem zero_boundary_three (r a j : Nat) (ha : a%2=1)
    (he : value a (4*2^j) (Code.reset (d:=r+(r+1))) =
      value a (4*2^j) (.t (towerT (r+1) (zeroAt r)))) :
    a%3=0 := by
  have h := zero_boundary_coefficient r a j ha he
  omega


theorem reset_ne_boundary {r : Nat} (x : Code r) (a j : Nat)
    (ha : a%2=1) (h3 : a%3≠0) (hab : a<6*2^j) :
    value a (4*2^j) (Code.reset (d:=r+(r+1))) ≠
      value a (4*2^j) (.t (towerT (r+1) x)) := by
  intro he
  by_cases hx : edges x=0
  · have hc := edges_zero_code x hx
    rw [hc] at he
    exact h3 (zero_boundary_three r a j ha he)
  · have hp : 0<2^j := Nat.two_pow_pos j
    have h := reset_lt_t_of_density (towerT (r+1) x)
      (show 0<4*2^j by omega) (show 2*a≤3*(4*2^j) by omega)
      (positive_tail_boundary_dense x (by omega))
    omega

theorem reset_collision_strict_tower_bound {r : Nat} (x : Code r)
    (k a j : Nat) (ha : a%2=1) (h3 : a%3≠0) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=r+k)) =
      value a (4*2^j) (.t (towerT k x))) : k≤r := by
  have hp : 0<2^j := Nat.two_pow_pos j
  have hk := reset_collision_tower_bound x k a (4*2^j)
    (by omega) (by omega) he
  by_cases hr : k≤r
  · exact hr
  · have heq : k=r+1 := by omega
    subst k
    exact False.elim (reset_ne_boundary x a j ha h3 hab he)

theorem reset_collision_strict_coefficient_bound {r : Nat} (x : Code r)
    (k a j : Nat) (hx : headT x=0) (ha : a%2=1)
    (h3 : a%3≠0) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=r+k)) =
      value a (4*2^j) (.t (towerT k x))) : j<r+k := by
  have hj := reset_collision_exponent (towerT k x) a j ha he
  rw [headT_tower,hx,Nat.zero_add] at hj
  have hk := reset_collision_strict_tower_bound x k a j ha h3 hab he
  omega



private theorem polynomial_elimination (u t q0 q1 f0 f1 m p : Nat)
    (hq0 : q0+1=2*u) (hq1 : q1+1=2*u*t)
    (hc0 : 2*f0+q0=q0*q0) (hc1 : 2*f1+q1=q1*q1)
    (hm : t*t*f0+m=f1)
    (hi : 42*m+28*p+21*q1+3=t*t*t*(21*q0+3)) :
    7*(3*m+4*p)+18*t^3+63*u*t^2+3 =
      42*u*t^3+21*t^2+21*u*t := by
  have hq0s := congrArg (fun z : Nat => z*z) hq0
  have hq1s := congrArg (fun z : Nat => z*z) hq1
  have hq0t := congrArg (fun z : Nat => z*t*t*t) hq0
  have hf0 : f0+3*u=2*u*u+1 := by grind only
  have hf1 : f1+3*u*t=2*u*u*t*t+1 := by grind only
  have hf0t := congrArg (fun z : Nat => z*t*t) hf0
  have hmc : m+3*u*t+t*t=3*u*t*t+1 := by grind only
  simp only [Nat.pow_succ,Nat.pow_zero,Nat.one_mul]
  grind only

private theorem polynomial_factoring (u t n : Nat) (hu : 0<u) (ht : 0<t)
    (h : 7*n+18*t^3+63*u*t^2+3=42*u*t^3+21*t^2+21*u*t) :
    7*n=3*(t-1)*(2*t-1)*((7*u-3)*t-1) := by
  cases u with
  | zero => omega
  | succ u =>
    cases t with
    | zero => omega
    | succ t =>
      have h7 : 7*(u+1)-3=7*u+4 := by omega
      have h2 : 2*(t+1)-1=2*t+1 := by omega
      have hlast : (7*(u+1)-3)*(t+1)-1=(7*u+4)*t+7*u+3 := by
        rw [h7,Nat.mul_add]
        omega
      simp only [Nat.add_sub_cancel,h2,hlast]
      simp only [Nat.pow_succ,Nat.pow_zero,Nat.one_mul] at h
      grind only

/-- Exact count identity underlying the separate classical number-theory argument. -/
theorem zero_tower_polynomial (r k : Nat) :
    7*(3*edges (towerT k (zeroAt r))+4*positives (towerT k (zeroAt r)))+
      18*(2^k)^3+63*2^(r+1)*(2^k)^2+3 =
      42*2^(r+1)*(2^k)^3+21*(2^k)^2+21*2^(r+1)*2^k := by
  let t := 2^k
  let u := 2^(r+1)
  let y := towerT k (zeroAt r)
  have hb := zeroAt_counts r
  have ht4 : 4^k=t*t := by rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
  have ht8 : 8^k=t*t*t := by
    rw [show (8:Nat)=2*2*2 from rfl,Nat.mul_pow,Nat.mul_pow]
  have hq0 := order_scale r
  have hq1 := order_scale (r+k)
  rw [Nat.pow_add] at hq1
  have hu : u=2*2^r := by dsimp [u]; rw [Nat.pow_succ,Nat.mul_comm]
  have hq0u : order r+1=2*u := by omega
  have hq1u : order (r+k)+1=2*u*t := by grind only
  have hc0 := capacity_identity r
  have hc1 := capacity_identity (r+k)
  have hgap := tower_gap (zeroAt r) k
  have hgb : gap (zeroAt r)=capacity r := by simp [gap,hb.1]
  rw [hgb,ht4] at hgap
  have heg := gap_add_edges y
  have hei := tower_energy (zeroAt r) k
  rw [ht8] at hei
  simp only [energy,hb.1,hb.2,Nat.mul_zero,Nat.zero_add] at hei
  have hge : t*t*capacity r+edges y=capacity (r+k) := by
    change gap y+edges y=capacity (r+k) at heg
    have hh : gap y=t*t*capacity r := hgap
    rw [hh] at heg
    exact heg
  exact polynomial_elimination u t (order r) (order (r+k))
    (capacity r) (capacity (r+k)) (edges y) (positives y)
    hq0u hq1u hc0 hc1 hge hei

theorem zero_tower_factorization (r k : Nat) :
    7*(3*edges (towerT k (zeroAt r))+4*positives (towerT k (zeroAt r))) =
      3*(2^k-1)*(2*2^k-1)*((7*2^(r+1)-3)*2^k-1) := by
  have h := zero_tower_polynomial r k
  have ht : 0<2^k := Nat.two_pow_pos k
  have hu : 0<2^(r+1) := Nat.two_pow_pos (r+1)
  exact polynomial_factoring (2^(r+1)) (2^k) _ hu ht h

/-- The primitive-divisor exception n=6 (r=4) has no dyadic candidate k=1..5. -/
theorem zsigmondy_exception_control :
    ∀ k : Fin 5,
      (3*edges (towerT (k.val+1) (zeroAt 4))+
        4*positives (towerT (k.val+1) (zeroAt 4))) % 1953 ≠ 0 := by decide

#print axioms polynomial_elimination
#print axioms polynomial_factoring
#print axioms zero_tower_polynomial
#print axioms zero_tower_factorization
#print axioms zsigmondy_exception_control

#print axioms reset_ne_boundary
#print axioms reset_collision_strict_tower_bound
#print axioms reset_collision_strict_coefficient_bound

#print axioms positive_edges_lower
#print axioms edges_zero_code
#print axioms power_eight_le_sixteen
#print axioms positive_tail_energy
#print axioms positive_tail_boundary_dense
#print axioms zeroAt_headT
#print axioms zero_boundary_coefficient
#print axioms zero_boundary_three
end Sounio.ZDScalarLowCone

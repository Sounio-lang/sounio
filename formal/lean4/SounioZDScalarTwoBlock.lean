import SounioZDScalarLowCone
namespace Sounio.ZDScalarTwoBlock
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility Sounio.ZDScalarLowCone
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def towerC {d : Nat} : (c : Nat) → Code d → Code (d+c)
  | 0, x => x
  | c+1, x => .c (towerC c x)

def twoBlock (r h c k : Nat) : Code (((r+h)+c)+k) :=
  towerT k (towerC c (towerT h (zeroAt r)))

theorem towerC_counts {d : Nat} (x : Code d) (c : Nat) :
    edges (towerC c x)=4^c*edges x ∧
    positives (towerC c x)=8^c*positives x := by
  induction c with
  | zero => simp [towerC]
  | succ c ih =>
    simp only [towerC,edges,positives,ih,Nat.pow_succ]
    constructor <;> grind only

theorem towerT_positive {d : Nat} (x : Code d) (k : Nat) :
    8^k*positives x ≤ positives (towerT k x) := by
  induction k with
  | zero => simp [towerT]
  | succ k ih =>
    simp only [towerT,positives,Nat.pow_succ]
    have hm := Nat.mul_le_mul_left 8 ih
    grind only

theorem order_lower (r : Nat) : 3*2^r ≤ order r := by
  have h := order_scale r
  have hp := Nat.two_pow_pos r
  omega

theorem tower_two_positive {d : Nat} (x : Code d) (h : Nat) :
    18*order d*8^h ≤ positives (towerT (h+2) x) := by
  induction h with
  | zero =>
    simp only [Nat.pow_zero,Nat.mul_one,towerT,positives,edges]
    have hq : order (d.add 0)=order d := rfl
    omega
  | succ h ih =>
    change 18*order d*8^(h+1) ≤
      6*edges (towerT (h+2) x)+8*positives (towerT (h+2) x)
    have hm := Nat.mul_le_mul_left 8 ih
    simp only [Nat.pow_succ]
    grind only

theorem twoBlock_positive (r h c k : Nat) :
    18*order r*8^(h+c+k) ≤ positives (twoBlock r (h+2) c k) := by
  have hi := tower_two_positive (zeroAt r) h
  have hc := (towerC_counts (towerT (h+2) (zeroAt r)) c).2
  have ht := towerT_positive (towerC c (towerT (h+2) (zeroAt r))) k
  have hm := Nat.mul_le_mul_left (8^k*8^c) hi
  simp only [twoBlock]
  rw [hc] at ht
  simp only [Nat.pow_add]
  grind only

private theorem triangle_density (r e p : Nat)
    (he : r+1≤e) (hp : 18*order r*8^e≤p) :
    3*capacity (r+e+2)<4*p := by
  let j := e-(r+1)
  have hj : e=r+1+j := by dsimp [j]; omega
  have hq := order_lower r
  have hm := Nat.mul_le_mul_left (18*8^e) hq
  have hp' : 54*2^r*8^e≤p := by grind only
  have he8 : 2^r*8^e=8*16^r*8^j := by
    rw [hj,Nat.pow_add,Nat.pow_succ]
    have hid : 2^r*8^r=16^r := by rw [←Nat.mul_pow]
    grind only
  have hlo : 432*16^r*8^j≤p := by
    have hh : 54*(2^r*8^e)=54*(8*16^r*8^j) := congrArg (fun z=>54*z) he8
    grind only
  have hc := (capacity_bounds (r+e+2)).2
  have h4 : 4^(r+e+2)=64*16^r*4^j := by
    have hd : r+e+2=2*r+j+3 := by omega
    rw [hd,Nat.pow_add,Nat.pow_add,Nat.pow_mul]
    simp only [show (4:Nat)^2=16 from rfl,show (4:Nat)^3=64 from rfl]
    grind only
  rw [h4] at hc
  have h48 := power_four_le_eight j
  have hmul := Nat.mul_le_mul_left (16^r) h48
  have hpos : 0<16^r*8^j := Nat.mul_pos (Nat.pow_pos (by decide)) (Nat.pow_pos (by decide))
  have hprod : 16^r*4^j≤16^r*8^j := hmul
  have hc1 : capacity (r+e+2)<512*(16^r*4^j) := by grind only
  have hp1 : 432*(16^r*8^j)≤p := by grind only
  omega

theorem twoBlock_span_dense (r h c k : Nat)
    (hh : 2≤h) (hl : r+3≤h+c+k) :
    3*capacity (((r+h)+c)+k)<4*positives (twoBlock r h c k) := by
  obtain ⟨h',rfl⟩ : ∃h',h=h'+2 := ⟨h-2,by omega⟩
  have hp := twoBlock_positive r h' c k
  have hd : r+(h'+c+k)+2=(((r+(h'+2))+c)+k) := by omega
  have hh := triangle_density r (h'+c+k) _ (by omega) hp
  rw [hd] at hh
  exact hh

theorem twoBlock_low_span (r h c k : Nat) (hh : 2≤h)
    (hlow : 3*edges (twoBlock r h c k)+4*positives (twoBlock r h c k)
      <3*gap (twoBlock r h c k)) : h+c+k≤r+2 := by
  by_cases hl : h+c+k≤r+2
  · exact hl
  · have hd := twoBlock_span_dense r h c k hh (by omega)
    have hg := gap_add_edges (twoBlock r h c k)
    omega

/-- A concrete low-ratio point attaining the span bound; not a dyadic collision. -/
theorem span_sharp_control :
    2+3+1=4+2 ∧
    3*edges (twoBlock 4 2 3 1)+4*positives (twoBlock 4 2 3 1)
      <3*gap (twoBlock 4 2 3 1) := by decide

private theorem tower_polynomial_algebra (z t q0 q1 f0 f1 m0 p0 m p g : Nat)
    (hq0 : q0+1=4*z) (hq1 : q1+1=4*z*t)
    (hc0 : 2*f0+q0=q0*q0) (hc1 : 2*f1+q1=q1*q1)
    (hg0 : g+m0=f0) (hg1 : t*t*g+m=f1)
    (hi : 42*m+28*p+21*q1+3=t*t*t*(42*m0+28*p0+21*q0+3)) :
    7*(3*m+4*p)+18*t^3+21*m0*t^2+126*z*t^2+3 =
      42*m0*t^3+28*p0*t^3+84*z*t^3+21*t^2+42*z*t := by
  have hq0s := congrArg (fun x : Nat=>x*x) hq0
  have hq1s := congrArg (fun x : Nat=>x*x) hq1
  have hq0t := congrArg (fun x : Nat=>x*t*t*t) hq0
  have hf0 : f0+6*z=8*z*z+1 := by grind only
  have hf1 : f1+6*z*t=8*z*z*t*t+1 := by grind only
  have hf0t := congrArg (fun x : Nat=>x*t*t) hf0
  have hgt := congrArg (fun x : Nat=>x*t*t) hg0
  simp only [Nat.pow_succ,Nat.pow_zero,Nat.one_mul]
  grind only

theorem tower_polynomial {d : Nat} (x : Code d) (k : Nat) :
    7*(3*edges (towerT k x)+4*positives (towerT k x))+
      18*(2^k)^3+21*edges x*(2^k)^2+126*2^d*(2^k)^2+3 =
    42*edges x*(2^k)^3+28*positives x*(2^k)^3+
      84*2^d*(2^k)^3+21*(2^k)^2+42*2^d*2^k := by
  let t := 2^k
  have hq0 := order_scale d
  have hq1 := order_scale (d+k)
  rw [Nat.pow_add] at hq1
  have hq1' : order (d+k)+1=4*2^d*t := by grind only
  have he := tower_energy x k
  have hg := tower_gap x k
  have ht4 : 4^k=t*t := by rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
  have ht8 : 8^k=t*t*t := by
    rw [show (8:Nat)=2*2*2 from rfl,Nat.mul_pow,Nat.mul_pow]
  rw [ht4] at hg
  rw [ht8] at he
  have hge := gap_add_edges (towerT k x)
  rw [hg] at hge
  exact tower_polynomial_algebra (2^d) t (order d) (order (d+k))
    (capacity d) (capacity (d+k)) (edges x) (positives x)
    (edges (towerT k x)) (positives (towerT k x)) (gap x)
    hq0 hq1' (capacity_identity d) (capacity_identity (d+k)) (gap_add_edges x) hge he


theorem oneInner_positive (r c k : Nat) :
    18*order r*4^c*8^k ≤ positives (twoBlock r 1 c (k+1)) := by
  induction k with
  | zero =>
    have hc := towerC_counts (towerT 1 (zeroAt r)) c
    have hz := zeroAt_counts r
    simp only [twoBlock,towerT,positives,edges,Nat.pow_zero,Nat.mul_one] at *
    have hq : order (r.add 0)=order r := rfl
    grind only
  | succ k ih =>
    change 18*order r*4^c*8^(k+1) ≤
      6*edges (twoBlock r 1 c (k+1))+8*positives (twoBlock r 1 c (k+1))
    have hm := Nat.mul_le_mul_left 8 ih
    simp only [Nat.pow_succ]
    grind only

private theorem triangle_density_scaled (r e c p : Nat)
    (he : r+1≤e) (hp : 18*order r*4^c*8^e≤p) :
    3*capacity (r+e+c+2)<4*p := by
  let j := e-(r+1)
  have hj : e=r+1+j := by dsimp [j]; omega
  have hq := order_lower r
  have hm := Nat.mul_le_mul_left (18*4^c*8^e) hq
  have hp' : 54*2^r*4^c*8^e≤p := by grind only
  have he8 : 2^r*8^e=8*16^r*8^j := by
    rw [hj,Nat.pow_add,Nat.pow_succ]
    have hid : 2^r*8^r=16^r := by rw [←Nat.mul_pow]
    grind only
  have hm8 := congrArg (fun z : Nat=>54*4^c*z) he8
  have hlo : 432*(16^r*4^c*8^j)≤p := by grind only
  have hc := (capacity_bounds (r+e+c+2)).2
  have h4 : 4^(r+e+c+2)=64*16^r*4^c*4^j := by
    have hd : r+e+c+2=2*r+c+j+3 := by omega
    rw [hd,Nat.pow_add,Nat.pow_add,Nat.pow_add,Nat.pow_mul]
    simp only [show (4:Nat)^2=16 from rfl,show (4:Nat)^3=64 from rfl]
    grind only
  rw [h4] at hc
  have hc1 : capacity (r+e+c+2)<512*(16^r*4^c*4^j) := by grind only
  have h48 := power_four_le_eight j
  have hmul := Nat.mul_le_mul_left (16^r*4^c) h48
  have hpos : 0<16^r*4^c*8^j :=
    Nat.mul_pos (Nat.mul_pos (Nat.pow_pos (by decide)) (Nat.pow_pos (by decide))) (Nat.pow_pos (by decide))
  omega

theorem oneInner_dense (r c k : Nat) (hk : r+2≤k) :
    3*capacity (((r+1)+c)+k)<4*positives (twoBlock r 1 c k) := by
  obtain ⟨e,rfl⟩ : ∃e,k=e+1 := ⟨k-1,by omega⟩
  have hp := oneInner_positive r c e
  have h := triangle_density_scaled r e c _ (by omega) hp
  have hd : r+e+c+2=(((r+1)+c)+(e+1)) := by omega
  rw [hd] at h
  exact h

theorem oneInner_low_outer (r c k : Nat)
    (hlow : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)
      <3*gap (twoBlock r 1 c k)) : k≤r+1 := by
  by_cases hk : k≤r+1
  · exact hk
  · have hd := oneInner_dense r c k (by omega)
    have hg := gap_add_edges (twoBlock r 1 c k)
    omega

private theorem mod_cancel (a b s : Nat) (hs : 0<s)
    (h : (a+b)%s=b%s) : a%s=0 := by
  have ha := Nat.mod_lt a hs
  have hb := Nat.mod_lt b hs
  rw [Nat.add_mod_eq_ite] at h
  split at h <;> omega

theorem towerC_mod {d : Nat} (x : Code d) (c : Nat) :
    edges (towerC c x)%2^c=0 ∧ positives (towerC c x)%2^c=0 ∧
      (2^(d+c))%2^c=0 ∧ gap (towerC c x)%2^c=1%2^c := by
  let s := 2^c
  have h4 : 4^c=s*s := by rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
  have h8 : 8^c=s*s*s := by
    rw [show (8:Nat)=2*2*2 from rfl,Nat.mul_pow,Nat.mul_pow]
  have hm : edges (towerC c x)%s=0 := by
    rw [(towerC_counts x c).1,h4]; simp [Nat.mul_mod]
  have hp : positives (towerC c x)%s=0 := by
    rw [(towerC_counts x c).2,h8]; simp [Nat.mul_mod]
  have hz : 2^(d+c)%s=0 := by rw [Nat.pow_add]; simp [s]
  have hw : 4^(d+c)%s=0 := by
    rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
    simp only [Nat.mul_mod,hz,Nat.zero_mul,Nat.zero_mod]
  have hc := congrArg (fun v : Nat=>v%s) (capacity_scale (d+c))
  simp only [Nat.add_mod,Nat.mul_mod,hz,hw,Nat.mul_zero,Nat.zero_mod,Nat.add_zero,Nat.zero_add,Nat.mod_mod] at hc
  have hg := congrArg (fun v : Nat=>v%s) (gap_add_edges (towerC c x))
  simp only [Nat.add_mod,hm,Nat.add_zero,Nat.mod_mod] at hg
  exact ⟨hm,hp,hz,hg.trans hc⟩

theorem separator_polynomial_mod {d : Nat} (x : Code d) (c k a : Nat)
    (he : 3*edges (towerT k (towerC c x))+4*positives (towerT k (towerC c x))
      =a*gap (towerC c x)) :
    (7*a+18*(2^k)^3+3)%2^c=(21*(2^k)^2)%2^c := by
  cases c with
  | zero => simp only [Nat.pow_zero]; omega
  | succ c =>
    have h := tower_polynomial (towerC (c+1) x) k
    have hm := (towerC_mod x (c+1)).1
    have hp := (towerC_mod x (c+1)).2.1
    have hz := (towerC_mod x (c+1)).2.2.1
    have hg := (towerC_mod x (c+1)).2.2.2
    have hs : 2≤2^(c+1) := Nat.pow_le_pow_right (n:=2) (by decide) (by omega : 1≤c+1)
    have h1 : 1%2^(c+1)=1 := Nat.mod_eq_of_lt (by omega)
    rw [he] at h
    have hh := congrArg (fun v : Nat=>v%2^(c+1)) h
    simp only [Nat.add_mod,Nat.mul_mod,hm,hp,hz,hg,h1,Nat.mul_zero,Nat.zero_mul,
      Nat.zero_mod,Nat.add_zero,Nat.zero_add,Nat.mod_mod,Nat.mul_one] at hh
    simpa only [Nat.add_mod,Nat.mul_mod,Nat.mod_mod] using hh

theorem checksum_identity (t : Nat) (ht : 0<t) :
    3*(t-1)*(2*t-1)*(3*t+1)+21*t^2=18*t^3+3 := by
  cases t with
  | zero => omega
  | succ t =>
    have hh : 2*(t+1)-1=2*t+1 := by omega
    simp only [Nat.add_sub_cancel,hh,Nat.pow_succ,Nat.pow_zero,Nat.one_mul]
    grind only

theorem separator_checksum {d : Nat} (x : Code d) (c k a : Nat)
    (he : 3*edges (towerT k (towerC c x))+4*positives (towerT k (towerC c x))
      =a*gap (towerC c x)) :
    2^c ∣ 7*a+3*(2^k-1)*(2*2^k-1)*(3*2^k+1) := by
  have h := separator_polynomial_mod x c k a he
  have hq := checksum_identity (2^k) (Nat.two_pow_pos k)
  have heq : 7*a+18*(2^k)^3+3 =
      (7*a+3*(2^k-1)*(2*2^k-1)*(3*2^k+1))+21*(2^k)^2 := by omega
  rw [heq] at h
  exact Nat.dvd_of_mod_eq_zero (mod_cancel _ _ _ (Nat.two_pow_pos c) h)

theorem separator_bound {d : Nat} (x : Code d) (c k a : Nat)
    (hk : 1≤k) (ha : a<3*(2^k)^2)
    (he : 3*edges (towerT k (towerC c x))+4*positives (towerT k (towerC c x))
      =a*gap (towerC c x)) : c≤3*k+4 := by
  let t := 2^k
  let S := 7*a+3*(t-1)*(2*t-1)*(3*t+1)
  have ht : 2≤t := by
    have h := Nat.pow_le_pow_right (n:=2) (by decide) hk
    exact h
  have hid := checksum_identity t (by omega)
  have hs0 : 0<S := by
    have h1 : 0<t-1 := by omega
    have h2 : 0<2*t-1 := by omega
    have h3 : 0<3*t+1 := by omega
    have h := Nat.mul_pos (Nat.mul_pos (Nat.mul_pos (by decide : 0<3) h1) h2) h3
    dsimp [S]; omega
  have hS : S<18*t^3+3 := by
    change a<3*t^2 at ha
    dsimp [S]
    omega
  have ht3 : 0<t^3 := Nat.pow_pos (by omega)
  have hS' : S<32*t^3 := by omega
  have hpow : 32*t^3=2^(3*k+5) := by
    rw [Nat.pow_add,Nat.pow_mul]
    have he : 2^(3*k)=t^3 := by
      rw [Nat.mul_comm,Nat.pow_mul]
    change 32*t^3=(2^3)^k*32
    have hh : (2^3)^k=t^3 := by
      rw [←Nat.pow_mul,Nat.mul_comm,Nat.pow_mul]
    rw [hh,Nat.mul_comm]
  have hd := separator_checksum x c k a he
  have hsle : 2^c≤S := Nat.le_of_dvd hs0 hd
  have hh : 2^c<2^(3*k+5) := by rw [←hpow]; omega
  have hexp := (Nat.pow_lt_pow_iff_right (by decide : 1<2)).1 hh
  omega



theorem towerC_headT {d : Nat} (x : Code d) (c : Nat) (hc : 1≤c) :
    headT (towerC c x)=0 := by
  cases c with
  | zero => omega
  | succ c => rfl

theorem collision_reduced {d : Nat} (x : Code d) (c k a j : Nat)
    (hc : 1≤c) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=(d+c)+k)) =
      value a (4*2^j) (.t (towerT k (towerC c x)))) :
    j+1=2*k ∧
    3*edges (towerT k (towerC c x))+4*positives (towerT k (towerC c x))=a*gap (towerC c x) ∧
    a<3*(2^k)^2 ∧
    3*edges (towerT k (towerC c x))+4*positives (towerT k (towerC c x))
      <3*gap (towerT k (towerC c x)) := by
  let y := towerC c x
  let t := 2^k
  have hj := reset_collision_exponent (towerT k y) a j ha he
  rw [headT_tower,towerC_headT x c hc,Nat.zero_add] at hj
  have ht4 : 4^k=t*t := by rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
  have hp2 : 2^(j+1)=t*t := by
    rw [hj,Nat.mul_comm,Nat.pow_mul]
    simp only [Nat.pow_succ,Nat.pow_zero,Nat.one_mul]
    rfl
  have hpowj : 2*2^j=t*t := by rw [Nat.pow_succ] at hp2; omega
  have hag : a<3*t^2 := by
    simp only [Nat.pow_succ,Nat.pow_zero,Nat.one_mul]
    omega
  have heq := reset_eq_gap (towerT k y) a j he
  have hgap := tower_gap y k
  rw [ht4] at hgap
  rw [hp2,hgap] at heq
  have hprod : (3*edges (towerT k y)+4*positives (towerT k y))*(t*t) =
      (a*gap y)*(t*t) := by grind only
  have ht : 0<t := Nat.two_pow_pos k
  have hn := Nat.eq_of_mul_eq_mul_right (Nat.mul_pos ht ht) hprod
  have hgmod := (towerC_mod x c).2.2.2
  have hs : 2≤2^c := Nat.pow_le_pow_right (n:=2) (by decide) hc
  have h1 : 1%2^c=1 := Nat.mod_eq_of_lt (by omega)
  rw [h1] at hgmod
  have hg : 0<gap y := by
    change gap y%2^c=1 at hgmod
    have hle := Nat.mod_le (gap y) (2^c)
    omega
  have hmul := Nat.mul_lt_mul_of_pos_right hag hg
  have hlow : 3*edges (towerT k y)+4*positives (towerT k y)<3*gap (towerT k y) := by
    simp only [Nat.pow_succ,Nat.pow_zero,Nat.one_mul] at hmul
    grind only
  exact ⟨hj,hn,hag,hlow⟩

/-- Every two-T-block collision in the odd small scalar cone lies in this finite
region for each fixed r. The finite r<=64 enumeration is a separate Python gate. -/
theorem twoBlock_collision_region (r h c k a j : Nat)
    (hh : 1≤h) (hc : 1≤c) (hk : 1≤k)
    (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+h)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r h c k))) :
    (h=1 ∧ k≤r+1 ∧ c≤3*k+4) ∨ (2≤h ∧ h+c+k≤r+2) := by
  have hr := collision_reduced (towerT h (zeroAt r)) c k a j hc ha hab he
  have hlow : 3*edges (twoBlock r h c k)+4*positives (twoBlock r h c k)
      <3*gap (twoBlock r h c k) := hr.2.2.2
  by_cases h1 : h=1
  · subst h
    exact Or.inl ⟨rfl,oneInner_low_outer r c k hlow,
      separator_bound (towerT 1 (zeroAt r)) c k a hk hr.2.2.1 hr.2.1⟩
  · exact Or.inr ⟨by omega,twoBlock_low_span r h c k (by omega) hlow⟩


#print axioms towerC_headT
#print axioms collision_reduced
#print axioms twoBlock_collision_region
#print axioms oneInner_positive
#print axioms triangle_density_scaled
#print axioms oneInner_dense
#print axioms oneInner_low_outer
#print axioms mod_cancel
#print axioms towerC_mod
#print axioms separator_polynomial_mod
#print axioms checksum_identity
#print axioms separator_checksum
#print axioms separator_bound
#print axioms towerC_counts
#print axioms towerT_positive
#print axioms order_lower
#print axioms tower_two_positive
#print axioms twoBlock_positive
#print axioms triangle_density
#print axioms twoBlock_span_dense
#print axioms twoBlock_low_span
#print axioms span_sharp_control
#print axioms tower_polynomial_algebra
#print axioms tower_polynomial
end Sounio.ZDScalarTwoBlock

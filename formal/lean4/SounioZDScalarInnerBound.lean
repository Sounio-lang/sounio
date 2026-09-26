import SounioZDScalarTwoBlock
namespace Sounio.ZDScalarInnerBound
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility Sounio.ZDScalarLowCone
open Sounio.ZDScalarTwoBlock
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def innerM (u v : Int) : Int := (v-1)*(3*u*v-v-1)
def innerP28 (u v : Int) : Int := 6*(v-2)*(v-1)*(7*u*v-3*v-2)
def gapPoly (u v s : Int) : Int :=
  s*s*v*v*(2*u-1)*(u-1)+3*u*v*s*(s-1)-s*s+1
def numPoly (u v s t : Int) : Int :=
  42*(s*s*innerM u v)*t^3+(s^3*innerP28 u v)*t^3+
  42*u*v*s*t^3-21*(s*s*innerM u v)*t^2-63*u*v*s*t^2+
  21*t^2+21*u*v*s*t-18*t^3-3
def collisionPoly (u v s t a : Int) : Int :=
  7*a*gapPoly u v s-numPoly u v s t
def innerChecksum (s t a : Int) : Int :=
  7*a*(1-s*s)+24*s^3*t^3-42*s*s*t^3+21*s*s*t*t+
    18*t^3-21*t*t+3

private theorem m_from_gap (u v m : Int)
    (h : v*v*(2*u*u+1-3*u)+m=2*u*u*v*v+1-3*u*v) :
    m=innerM u v := by
  simp only [innerM]
  grind only

private theorem inner_count_algebra (u v q0 q1 f0 f1 m p : Int)
    (hq0 : q0+1=2*u) (hq1 : q1+1=2*u*v)
    (hc0 : 2*f0+q0=q0*q0) (hc1 : 2*f1+q1=q1*q1)
    (hm : v*v*f0+m=f1)
    (hi : 42*m+28*p+21*q1+3=v*v*v*(21*q0+3)) :
    m=innerM u v ∧ 28*p=innerP28 u v := by
  have hq0s := congrArg (fun z : Int=>z*z) hq0
  have hq1s := congrArg (fun z : Int=>z*z) hq1
  have hq0v := congrArg (fun z : Int=>z*v*v*v) hq0
  have hf0 : f0+3*u=2*u*u+1 := by grind only
  have hf1 : f1+3*u*v=2*u*u*v*v+1 := by grind only
  have hf0v := congrArg (fun z : Int=>z*v*v) hf0
  have hf0' : f0=2*u*u+1-3*u := by omega
  have hf1' : f1=2*u*u*v*v+1-3*u*v := by omega
  rw [hf0',hf1'] at hm
  have hmc := m_from_gap u v m hm
  have hmv := congrArg (fun z : Int=>42*z) hmc
  constructor
  · exact hmc
  · simp only [innerP28,innerM] at *
    grind only

theorem inner_counts (r h : Nat) :
    (edges (towerT h (zeroAt r)) : Int)=innerM (2^(r+1)) (2^h) ∧
    28*(positives (towerT h (zeroAt r)) : Int)=innerP28 (2^(r+1)) (2^h) := by
  let u : Nat := 2^(r+1)
  let v : Nat := 2^h
  let x := towerT h (zeroAt r)
  have hb := zeroAt_counts r
  have hq0 := order_scale r
  have hq1 := order_scale (r+h)
  have hu : u=2*2^r := by dsimp [u]; rw [Nat.pow_succ,Nat.mul_comm]
  rw [Nat.pow_add] at hq1
  have hq0u : order r+1=2*u := by omega
  have hq1u : order (r+h)+1=2*u*v := by grind only
  have ht4 : 4^h=v*v := by rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
  have ht8 : 8^h=v*v*v := by
    rw [show (8:Nat)=2*2*2 from rfl,Nat.mul_pow,Nat.mul_pow]
  have hg := tower_gap (zeroAt r) h
  have hgb : gap (zeroAt r)=capacity r := by simp [gap,hb.1]
  rw [hgb,ht4] at hg
  have hm := gap_add_edges x
  change gap (towerT h (zeroAt r))+edges x=capacity (r+h) at hm
  rw [hg] at hm
  have hi := tower_energy (zeroAt r) h
  rw [ht8] at hi
  simp only [energy,hb.1,hb.2,Nat.mul_zero,Nat.zero_add] at hi
  have H := inner_count_algebra (u:Int) (v:Int) (order r) (order (r+h))
    (capacity r) (capacity (r+h)) (edges x) (positives x)
    (congrArg (fun z : Nat=>(z:Int)) hq0u)
    (congrArg (fun z : Nat=>(z:Int)) hq1u)
    (congrArg (fun z : Nat=>(z:Int)) (capacity_identity r))
    (congrArg (fun z : Nat=>(z:Int)) (capacity_identity (r+h)))
    (congrArg (fun z : Nat=>(z:Int)) hm)
    (congrArg (fun z : Nat=>(z:Int)) hi)
  simpa only [u,v,x,Int.natCast_pow,show ((2:Nat):Int)=2 from rfl] using H

theorem twoBlock_gap (r h c : Nat) :
    (gap (towerC c (towerT h (zeroAt r))) : Int)=
      gapPoly (2^(r+1)) (2^h) (2^c) := by
  let u : Int := 2^(r+1)
  let v : Int := 2^h
  let s : Int := 2^c
  let y := towerC c (towerT h (zeroAt r))
  have hmc := (towerC_counts (towerT h (zeroAt r)) c).1
  have hm := (inner_counts r h).1
  have hm' : (edges y:Int)=s*s*innerM u v := by
    have h4 : (4^c:Nat)=(2^c)*(2^c) := by
      rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
    rw [h4] at hmc
    have H := congrArg (fun z : Nat=>(z:Int)) hmc
    change (edges y:Int)=s*s*(edges (towerT h (zeroAt r)):Int) at H
    rw [hm] at H
    exact H
  have hq := order_scale ((r+h)+c)
  have hpow : 4*2^((r+h)+c)=2*2^(r+1)*2^h*2^c := by
    rw [Nat.pow_add,Nat.pow_add,Nat.pow_succ]
    grind only
  rw [hpow] at hq
  have hqi : (order ((r+h)+c):Int)+1=2*u*v*s :=
    congrArg (fun z : Nat=>(z:Int)) hq
  have hqi2 := congrArg (fun z : Int=>z*z) hqi
  have hf := congrArg (fun z : Nat=>(z:Int)) (capacity_identity ((r+h)+c))
  have hg := congrArg (fun z : Nat=>(z:Int)) (gap_add_edges y)
  change (gap y:Int)+(edges y:Int)=(capacity ((r+h)+c):Int) at hg
  rw [hm'] at hg
  change 2*(capacity ((r+h)+c):Int)+(order ((r+h)+c):Int)=
    (order ((r+h)+c):Int)*(order ((r+h)+c):Int) at hf
  change (gap y:Int)=gapPoly u v s
  simp only [gapPoly,innerM] at *
  grind only

private theorem outer_count_algebra (u v s t z m p N : Int)
    (hm : m=s*s*innerM u v) (hp : 28*p=s^3*innerP28 u v)
    (hz : 2*z=u*v*s)
    (H : 7*N+18*t^3+21*m*t^2+126*z*t^2+3=
      42*m*t^3+28*p*t^3+84*z*t^3+21*t^2+42*z*t) :
    7*N=numPoly u v s t := by
  have hpt := congrArg (fun x : Int=>x*t^3) hp
  have hzt := congrArg (fun x : Int=>x*t) hz
  have hzt2 := congrArg (fun x : Int=>x*t^2) hz
  have hzt3 := congrArg (fun x : Int=>x*t^3) hz
  rw [hm] at H
  simp only [numPoly,Int.pow_succ,Int.pow_zero] at *
  grind only

theorem twoBlock_numerator (r h c k : Nat) :
    7*(3*(edges (twoBlock r h c k):Int)+4*(positives (twoBlock r h c k):Int)) =
      numPoly (2^(r+1)) (2^h) (2^c) (2^k) := by
  let u : Int := 2^(r+1)
  let v : Int := 2^h
  let s : Int := 2^c
  let t : Int := 2^k
  let y := towerC c (towerT h (zeroAt r))
  have hcs := towerC_counts (towerT h (zeroAt r)) c
  have his := inner_counts r h
  have h4 : (4^c:Nat)=2^c*2^c := by rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
  have h8 : (8^c:Nat)=2^c*2^c*2^c := by
    rw [show (8:Nat)=2*2*2 from rfl,Nat.mul_pow,Nat.mul_pow]
  have hm : (edges y:Int)=s*s*innerM u v := by
    have H := congrArg (fun z : Nat=>(z:Int)) hcs.1
    rw [h4] at H
    change (edges y:Int)=s*s*(edges (towerT h (zeroAt r)):Int) at H
    rw [his.1] at H
    exact H
  have hp : 28*(positives y:Int)=s^3*innerP28 u v := by
    have H := congrArg (fun z : Nat=>(z:Int)) hcs.2
    rw [h8] at H
    change (positives y:Int)=s*s*s*(positives (towerT h (zeroAt r)):Int) at H
    have hi := congrArg (fun z : Int=>s*s*s*z) his.2
    simp only [Int.pow_succ,Int.pow_zero]
    grind only
  have hz : 2*(2^((r+h)+c):Int)=u*v*s := by
    have H : 2*2^((r+h)+c)=2^(r+1)*2^h*2^c := by
      rw [Nat.pow_add,Nat.pow_add,Nat.pow_succ]
      grind only
    exact congrArg (fun z : Nat=>(z:Int)) H
  have ht := congrArg (fun z : Nat=>(z:Int)) (tower_polynomial y k)
  change 7*(3*(edges (towerT k y):Int)+4*(positives (towerT k y):Int))+
    18*t^3+21*(edges y:Int)*t^2+126*(2^((r+h)+c):Int)*t^2+3 =
    42*(edges y:Int)*t^3+28*(positives y:Int)*t^3+
    84*(2^((r+h)+c):Int)*t^3+21*t^2+42*(2^((r+h)+c):Int)*t at ht
  exact outer_count_algebra u v s t (2^((r+h)+c)) (edges y) (positives y)
    (3*(edges (towerT k y):Int)+4*(positives (towerT k y):Int)) hm hp hz ht

theorem collision_polynomial (r h c k a : Nat)
    (he : 3*edges (twoBlock r h c k)+4*positives (twoBlock r h c k)=
      a*gap (towerC c (towerT h (zeroAt r)))) :
    collisionPoly (2^(r+1)) (2^h) (2^c) (2^k) a=0 := by
  have hg := twoBlock_gap r h c
  have hn := twoBlock_numerator r h c k
  have H := congrArg (fun z : Nat=>(z:Int)) he
  change 3*(edges (twoBlock r h c k):Int)+4*(positives (twoBlock r h c k):Int)=
    (a:Int)*(gap (towerC c (towerT h (zeroAt r))):Int) at H
  rw [hg] at H
  have H7 := congrArg (fun z : Int=>7*z) H
  simp only [collisionPoly]
  grind only


def quotientPoly (w v s t a : Int) : Int :=
  7*a*(s*s*(2*v*v*w*w-3*v*w+1)+3*w*s*(s-1))-
  (42*s*s*(3*v*w-3*w-1)*t^3+
   6*s^3*(7*v*v*w-21*v*w+14*w-3*v+7)*t^3+
   42*w*s*t^3-21*s*s*(3*v*w-3*w-1)*t^2-63*w*s*t^2+21*w*s*t)

theorem inner_decomposition (w v s t a : Int) :
    collisionPoly (v*w) v s t a=innerChecksum s t a+v*v*quotientPoly w v s t a := by
  simp only [collisionPoly,gapPoly,numPoly,innerM,innerP28,innerChecksum,quotientPoly,
    Int.pow_succ,Int.pow_zero]
  grind only

theorem low_inner_depth (r h c k : Nat) (hc : 1≤c) (hk : 1≤k)
    (hlow : 3*edges (twoBlock r h c k)+4*positives (twoBlock r h c k)
      <3*gap (twoBlock r h c k)) : h≤r+1 := by
  by_cases hh : 2≤h
  · have hs := twoBlock_low_span r h c k hh hlow
    omega
  · omega

theorem inner_checksum_divides (r h c k a : Nat) (hr : h≤r+1)
    (he : 3*edges (twoBlock r h c k)+4*positives (twoBlock r h c k)=
      a*gap (towerC c (towerT h (zeroAt r)))) :
    (4:Int)^h ∣ innerChecksum (2^c) (2^k) a := by
  obtain ⟨w,hw⟩ := Nat.pow_dvd_pow 2 hr
  have hu : (2:Int)^(r+1)=(2:Int)^h*(w:Int) := by
    have H := congrArg (fun z : Nat=>(z:Int)) hw
    simpa only [Int.natCast_mul,Int.natCast_pow,show ((2:Nat):Int)=2 from rfl] using H
  have H := collision_polynomial r h c k a he
  rw [hu] at H
  have hd := inner_decomposition (w:Int) ((2:Int)^h) ((2:Int)^c) ((2:Int)^k) a
  have h4 : (4:Int)^h=(2:Int)^h*(2:Int)^h := by
    rw [show (4:Int)=2*2 from rfl,Int.mul_pow]
  refine ⟨-quotientPoly w (2^h) (2^c) (2^k) a,?_⟩
  rw [h4]
  grind only

theorem checksum_factor (s t a : Int) :
    innerChecksum s t a =
      7*a+3*(t-1)*(2*t-1)*(3*t+1)+
      s*s*(6*t^3*(4*s-7)+7*(3*t*t-a)) := by
  simp only [innerChecksum,Int.pow_succ,Int.pow_zero]
  grind only

private theorem checksum_upper_identity (s t a : Int) :
    innerChecksum s t a+7*a*(s*s-1)+42*s*s*t^3+21*t*t =
      24*(s^3*t^3)+21*(s*s*t*t)+18*t^3+3 := by
  simp only [innerChecksum,Int.pow_succ,Int.pow_zero]
  grind only

theorem checksum_bounds (s t a : Int) (hs : 2≤s) (ht : 2≤t)
    (ha : 0<a) (hab : a<3*t*t) :
    0<innerChecksum s t a ∧ innerChecksum s t a<32*(s^3*t^3) := by
  have hs0 : 0<s := by omega
  have ht0 : 0<t := by omega
  have hs2 : 4≤s*s := Int.mul_le_mul hs hs (by decide) (by omega)
  have ht2 : 4≤t*t := Int.mul_le_mul ht ht (by decide) (by omega)
  have hs3 : 8≤s^3 := by
    have H := Int.mul_le_mul hs2 hs (by decide) (by omega)
    simp only [Int.pow_succ,Int.pow_zero,Int.one_mul]
    exact H
  have ht3 : 8≤t^3 := by
    have H := Int.mul_le_mul ht2 ht (by decide) (by omega)
    simp only [Int.pow_succ,Int.pow_zero,Int.one_mul]
    exact H
  have hst : 4≤s*t := Int.mul_le_mul hs ht (by decide) (by omega)
  have hP : 64≤s^3*t^3 := Int.mul_le_mul hs3 ht3 (by decide) (by omega)
  have hQ0 : 0≤s*s*t*t :=
    Int.le_of_lt (Int.mul_pos (Int.mul_pos (Int.mul_pos hs0 hs0) ht0) ht0)
  have hQ' := Int.mul_le_mul_of_nonneg_right hst hQ0
  have hQ : 4*(s*s*t*t)≤s^3*t^3 := by
    simp only [Int.pow_succ,Int.pow_zero] at *
    grind only
  have hR : 8*t^3≤s^3*t^3 :=
    Int.mul_le_mul_of_nonneg_right hs3 (by omega)
  have hbad1 : 0≤7*a*(s*s-1) :=
    Int.mul_nonneg (by omega) (by omega)
  have hbad2 : 0≤42*s*s*t^3 := by
    have H := Int.mul_nonneg (show 0≤s*s by omega) (show 0≤t^3 by omega)
    grind only
  have hbad3 : 0≤21*t*t := by
    have H := Int.mul_nonneg (by decide : (0:Int)≤21) (show 0≤t*t by omega)
    grind only
  have hident := checksum_upper_identity s t a
  have hupper : innerChecksum s t a<32*(s^3*t^3) := by omega
  have hf := checksum_factor s t a
  have hfirst : 0<7*a := by omega
  have hproduct : 0<3*(t-1)*(2*t-1)*(3*t+1) :=
    Int.mul_pos (Int.mul_pos (Int.mul_pos (by decide) (by omega)) (by omega)) (by omega)
  have hB1 : 0<6*t^3*(4*s-7) :=
    Int.mul_pos (by omega) (by omega)
  have hB2 : 0<7*(3*t*t-a) := by omega
  have hlast : 0<s*s*(6*t^3*(4*s-7)+7*(3*t*t-a)) :=
    Int.mul_pos (by omega) (by omega)
  exact ⟨by omega,hupper⟩



theorem inner_run_bound (r h c k a : Nat) (hc : 1≤c) (hk : 1≤k)
    (ha : 0<a) (hab : a<3*(2^k)^2) (hr : h≤r+1)
    (he : 3*edges (twoBlock r h c k)+4*positives (twoBlock r h c k)=
      a*gap (towerC c (towerT h (zeroAt r)))) :
    2*h≤3*c+3*k+4 := by
  have hsNat : 2≤2^c := Nat.pow_le_pow_right (n:=2) (by decide) hc
  have htNat : 2≤2^k := Nat.pow_le_pow_right (n:=2) (by decide) hk
  have hs : (2:Int)≤2^c := by
    have H : (2:Int)≤((2^c:Nat):Int) := by omega
    simpa only [Int.natCast_pow,show ((2:Nat):Int)=2 from rfl] using H
  have ht : (2:Int)≤2^k := by
    have H : (2:Int)≤((2^k:Nat):Int) := by omega
    simpa only [Int.natCast_pow,show ((2:Nat):Int)=2 from rfl] using H
  have hai : (0:Int)<a := by omega
  have habi : (a:Int)<3*(2^k:Int)*(2^k:Int) := by
    have H : (a:Int)<((3*(2^k)^2:Nat):Int) := by omega
    simpa only [Int.natCast_mul,Int.natCast_pow,
      show ((2:Nat):Int)=2 from rfl,show ((3:Nat):Int)=3 from rfl,
      Int.pow_succ,Int.pow_zero,Int.one_mul,Int.mul_assoc] using H
  have hb := checksum_bounds ((2:Int)^c) ((2:Int)^k) a hs ht hai habi
  have hd := inner_checksum_divides r h c k a hr he
  have hh := Int.lt_of_le_of_lt (Int.le_of_dvd hb.1 hd) hb.2
  have H : (4^h:Nat)<32*((2^c:Nat)^3*(2^k)^3) := by
    have hcast : ((4^h:Nat):Int)<((32*((2^c:Nat)^3*(2^k)^3):Nat):Int) := by
      simpa only [Int.natCast_mul,Int.natCast_pow,
        show ((2:Nat):Int)=2 from rfl,show ((4:Nat):Int)=4 from rfl,
        show ((32:Nat):Int)=32 from rfl] using hh
    omega
  have h4 : (4:Nat)^h=2^(2*h) := by rw [Nat.pow_mul]
  have hmax : 32*((2^c:Nat)^3*(2^k)^3)=2^(3*c+3*k+5) := by
    rw [Nat.pow_add,Nat.pow_add,Nat.pow_mul,Nat.pow_mul]
    have hc3 : (2^3:Nat)^c=(2^c)^3 := by rw [←Nat.pow_mul,Nat.mul_comm,Nat.pow_mul]
    have hk3 : (2^3:Nat)^k=(2^k)^3 := by rw [←Nat.pow_mul,Nat.mul_comm,Nat.pow_mul]
    rw [hc3,hk3]
    grind only
  rw [h4,hmax] at H
  have hp := (Nat.pow_lt_pow_iff_right (by decide : 1<2)).1 H
  omega

def numSlope (v s t : Int) : Int :=
  126*s*s*v*(v-1)*t^3+42*s^3*v*(v-2)*(v-1)*t^3+42*v*s*t^3-
  63*s*s*v*(v-1)*t^2-63*v*s*t^2+21*v*s*t
def numConstant (v s t : Int) : Int :=
  -42*s*s*(v*v-1)*t^3-6*s^3*(v-2)*(v-1)*(3*v+2)*t^3+
    21*s*s*(v*v-1)*t^2+21*t^2-18*t^3-3
def coeffA (v s a : Int) : Int := 14*a*s*s*v*v
def coeffB (v s t a : Int) : Int :=
  -21*a*s*v*(s*v-s+1)-numSlope v s t
def coeffC (v s t a : Int) : Int :=
  7*a*(s*s*v*v-s*s+1)-numConstant v s t

theorem polynomial_quadratic (u v s t a : Int) :
    collisionPoly u v s t a=
      coeffA v s a*u*u+coeffB v s t a*u+coeffC v s t a := by
  simp only [collisionPoly,gapPoly,numPoly,innerM,innerP28,
    coeffA,coeffB,coeffC,numSlope,numConstant,Int.pow_succ,Int.pow_zero]
  grind only

theorem leading_positive (v s a : Int) (hv : 0<v) (hs : 0<s) (ha : 0<a) :
    0<coeffA v s a := by
  exact Int.mul_pos (Int.mul_pos (Int.mul_pos (Int.mul_pos (Int.mul_pos
    (by decide : (0:Int)<14) ha) hs) hs) hv) hv

theorem discriminant_identity (A B C u : Int) (h : A*u*u+B*u+C=0) :
    B*B-4*A*C=(2*A*u+B)*(2*A*u+B) := by
  have H := congrArg (fun z : Int=>4*A*z) h
  grind only

theorem collision_discriminant (r h c k a : Nat)
    (he : 3*edges (twoBlock r h c k)+4*positives (twoBlock r h c k)=
      a*gap (towerC c (towerT h (zeroAt r)))) :
    let u : Int := 2^(r+1)
    let v : Int := 2^h
    let s : Int := 2^c
    let t : Int := 2^k
    let A := coeffA v s a
    let B := coeffB v s t a
    let C := coeffC v s t a
    B*B-4*A*C=(2*A*u+B)*(2*A*u+B) := by
  have H := collision_polynomial r h c k a he
  rw [polynomial_quadratic] at H
  exact discriminant_identity _ _ _ _ H

theorem collision_length_bounds (r h c k a j : Nat) (hc : 1≤c) (hk : 1≤k)
    (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+h)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r h c k))) :
    c≤3*k+4 ∧ 2*h≤3*c+3*k+4 ∧ h≤r+1 := by
  have H := collision_reduced (towerT h (zeroAt r)) c k a j hc ha hab he
  have hr := low_inner_depth r h c k hc hk H.2.2.2
  exact ⟨separator_bound (towerT h (zeroAt r)) c k a hk H.2.2.1 H.2.1,
    inner_run_bound r h c k a hc hk (by omega) H.2.2.1 hr H.2.1,hr⟩



/-- A genuine two-T-block dyadic collision outside the small cone.
It preserves oddness and 3-nondivisibility; it does not refute ResidualSeparation. -/
def outsideMixedA : Nat := 579663883630356344501457359869743367959045525525856898331511710651301541654612402055217130079266318505255748160833689621276392191880315593336659554572911679110416530088923

theorem outside_mixed_control :
    3*edges (twoBlock 1 1 5 189)+4*positives (twoBlock 1 1 5 189)=
      outsideMixedA*gap (towerC 5 (towerT 1 (zeroAt 1))) ∧
    outsideMixedA%2=1 ∧ outsideMixedA%3=1 ∧ 3*4^189<outsideMixedA := by decide


#print axioms outside_mixed_control
#print axioms inner_run_bound
#print axioms polynomial_quadratic
#print axioms leading_positive
#print axioms discriminant_identity
#print axioms collision_discriminant
#print axioms collision_length_bounds
#print axioms inner_decomposition
#print axioms low_inner_depth
#print axioms inner_checksum_divides
#print axioms checksum_factor
#print axioms checksum_upper_identity
#print axioms checksum_bounds
#print axioms m_from_gap
#print axioms inner_count_algebra
#print axioms inner_counts
#print axioms twoBlock_gap
#print axioms outer_count_algebra
#print axioms twoBlock_numerator
#print axioms collision_polynomial
end Sounio.ZDScalarInnerBound

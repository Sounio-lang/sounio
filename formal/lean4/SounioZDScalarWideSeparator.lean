import SounioZDScalarDiscriminant

/-! Uniform exclusion for a one-step inner T block with separator c >= k.
Finite residue certificates are proved by kernel reduction. -/
namespace Sounio.ZDScalarWideSeparator
open Sounio.ZDScalarInnerBound
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility Sounio.ZDScalarLowCone
open Sounio.ZDScalarTwoBlock
set_option maxRecDepth 32768
set_option maxHeartbeats 32000000

def scaledGapPart (w z t : Int) : Int :=
  8*w*w*z*z*t*t-6*w*w*z*t-6*w*z+3*w*w

def scaledNumPart (w z t : Int) : Int :=
  252*w*w*z*t^4+(-126*w*w-126*w*w*z+84*w*z)*t^3+
  (63*w*w-126*w*z)*t*t+(42*w*z-18)*t+21

def scaledResidual (w z t j : Int) : Int :=
  ((168-8*j)*w*w*z*z-252*w*w*z)*t^4+
  (6*j*w*w*z+126*w*w-84*w*z)*t^3+
  (6*j*w*z-3*j*w*w-24*w*w*z*z)*t*t+
  (18*w*w*z-42*w*z+18)*t+18*w*z-9*w*w-j

theorem rescale (w z t a : Int) :
    collisionPoly (t*z) 2 (t*w) t a =
      7*a+3+t*t*(7*a*scaledGapPart w z t-scaledNumPart w z t) := by
  simp only [collisionPoly,gapPoly,numPoly,innerM,innerP28,
    scaledGapPart,scaledNumPart,Int.pow_succ,Int.pow_zero]
  grind only

theorem deficit_rescale (w z t a j : Int) (h : 7*a=(21-j)*t*t-3) :
    collisionPoly (t*z) 2 (t*w) t a=t*t*scaledResidual w z t j := by
  rw [rescale]
  unfold scaledGapPart scaledNumPart scaledResidual
  simp only [Int.pow_succ,Int.pow_zero]
  grind only

private theorem quotient_bounds (n x q : Int) (hn : 0<n)
    (hx : 0<x) (hu : x<21*n) (he : x=n*q) : 0<q ∧ q<21 := by
  constructor
  · by_cases hq : 0<q
    · exact hq
    have hh := Int.mul_le_mul_of_nonneg_left (show q≤0 by omega) (show 0≤n by omega)
    omega
  · by_cases hq : q<21
    · exact hq
    have hh := Int.mul_le_mul_of_nonneg_left (show 21≤q by omega) (show 0≤n by omega)
    have hm : n*21=21*n := Int.mul_comm _ _
    omega

theorem collision_deficit (w z t a : Int) (ht : 0<t) (ha : 0<a)
    (hab : a<3*t*t) (hf : collisionPoly (t*z) 2 (t*w) t a=0) :
    ∃ j : Nat, 1≤j ∧ j≤20 ∧ 7*a=(21-(j:Int))*t*t-3 ∧
      scaledResidual w z t j=0 := by
  have hres := rescale w z t a
  have hn : 0<t*t := Int.mul_pos ht ht
  let q := scaledNumPart w z t-7*a*scaledGapPart w z t
  have he : 7*a+3=(t*t)*q := by dsimp [q]; grind only
  have hu : 7*a+3<21*(t*t) := by
    have hm : 3*t*t=3*(t*t) := Int.mul_assoc _ _ _
    omega
  have hq := quotient_bounds (t*t) (7*a+3) q hn (by omega) hu he
  let j := (21-q).toNat
  have hj : (j:Int)=21-q := Int.toNat_of_nonneg (by omega)
  have hjlo : 1≤j := by omega
  have hjhi : j≤20 := by omega
  have haeq : 7*a=(21-(j:Int))*t*t-3 := by grind only
  have hp := deficit_rescale w z t a j haeq
  have hzero : t*t*scaledResidual w z t j=0 := by omega
  have hP := (Int.mul_eq_zero.mp hzero).resolve_left (by omega)
  exact ⟨j,hjlo,hjhi,haeq,hP⟩

theorem power_mod_cap (n M : Nat) :
    (2:Int)^n % (2:Int)^M = (2:Int)^(min n M) % (2:Int)^M := by
  by_cases h : n<M
  · rw [Nat.min_eq_left (by omega)]
  · have hm : M≤n := by omega
    rw [Nat.min_eq_right hm]
    obtain ⟨q,hq⟩ := Nat.exists_eq_add_of_le hm
    rw [hq,Int.pow_add]
    simp

theorem residual_mod_congr (w z t j w' z' t' m : Int)
    (hw : w%m=w'%m) (hz : z%m=z'%m) (ht : t%m=t'%m) :
    scaledResidual w z t j %m=scaledResidual w' z' t' j %m := by
  simp [scaledResidual,Int.pow_succ,
    Int.add_emod,Int.sub_emod,Int.mul_emod,hw,hz,ht]

theorem small_mod_certificate :
    ∀ (k : Fin 6) (j : Fin 21) (d e : Fin 9),
    1≤k.val → 1≤j.val →
    (((21:Int)-j.val)*(2:Int)^k.val*(2:Int)^k.val-3)%7=0 →
    scaledResidual ((2:Int)^d.val) ((2:Int)^e.val)
      ((2:Int)^k.val) j.val %256≠0 := by decide

theorem large_mod_certificate :
    ∀ (d e : Fin 7) (j : Fin 21),
    1≤j.val →
    (18*(2:Int)^d.val*(2:Int)^e.val-9*(2:Int)^d.val*(2:Int)^d.val-j.val)%64=0 →
    (d.val=0 ∧ e.val=0 ∧ j.val=9) ∨
    (d.val=0 ∧ e.val=3 ∧ j.val=7) ∨
    (d.val=2 ∧ e.val=2 ∧ j.val=16) ∨
    (d.val=3 ∧ e.val=0 ∧ j.val=16) := by decide

theorem small_exclusion (k d e j : Nat) (hk : 1≤k) (hk5 : k≤5)
    (hj : 1≤j) (hj20 : j≤20)
    (ha : (((21:Int)-j)*(2:Int)^k*(2:Int)^k-3)%7=0) :
    scaledResidual ((2:Int)^d) ((2:Int)^e) ((2:Int)^k) j ≠0 := by
  intro hp
  have hd := power_mod_cap d 8
  have he := power_mod_cap e 8
  have hcon := residual_mod_congr ((2:Int)^d) ((2:Int)^e) ((2:Int)^k) j
    ((2:Int)^(min d 8)) ((2:Int)^(min e 8)) ((2:Int)^k) 256 hd he rfl
  have cert := small_mod_certificate ⟨k,by omega⟩ ⟨j,by omega⟩
    ⟨min d 8,by omega⟩ ⟨min e 8,by omega⟩ hk hj ha
  rw [hp] at hcon
  exact cert hcon.symm

private theorem constant_mod_congr (w z w' z' j : Int)
    (hw : w%64=w'%64) (hz : z%64=z'%64) :
    (18*w*z-9*w*w-j)%64=(18*w'*z'-9*w'*w'-j)%64 := by
  simp [Int.sub_emod,Int.mul_emod,hw,hz]

theorem large_patterns (d e j : Nat) (hj : 1≤j) (hj20 : j≤20)
    (hp : (18*(2:Int)^d*(2:Int)^e-9*(2:Int)^d*(2:Int)^d-j)%64=0) :
    (d=0 ∧ e=0 ∧ j=9) ∨ (d=0 ∧ e=3 ∧ j=7) ∨
    (d=2 ∧ e=2 ∧ j=16) ∨ (d=3 ∧ e=0 ∧ j=16) := by
  have hd := power_mod_cap d 6
  have he := power_mod_cap e 6
  have hc := constant_mod_congr _ _ _ _ j hd he
  rw [hp] at hc
  have H := large_mod_certificate ⟨min d 6,by omega⟩
    ⟨min e 6,by omega⟩ ⟨j,by omega⟩ hj hc.symm
  rcases H with H | H | H | H <;> simp only at H <;> omega

private theorem positive_horner (a b x : Int)
    (ha : 0<a) (hb : 0<b) (hx : 0≤x) : 0<a*x+b := by
  have H := Int.mul_nonneg (show 0≤a by omega) hx
  omega

theorem residual_009_negative (t : Int) (ht : 1≤t) :
    scaledResidual 1 1 t 9<0 := by
  have h1 := positive_horner 26 49 (t-1) (by decide) (by decide) (by omega)
  have h2 := positive_horner (26*(t-1)+49) 21 (t-1) h1 (by decide) (by omega)
  have hq : 0<26*t*t-3*t-2 := by grind only
  have hprod : 0<3*t*(2*t-1)*(26*t*t-3*t-2) :=
    Int.mul_pos (Int.mul_pos (Int.mul_pos (by decide) (by omega)) (by omega)) hq
  have hi : scaledResidual 1 1 t 9= -(3*t*(2*t-1)*(26*t*t-3*t-2)) := by
    unfold scaledResidual
    simp only [Int.pow_succ,Int.pow_zero]
    grind only
  omega

theorem residual_2216_negative (t : Int) (ht : 1≤t) :
    scaledResidual 4 4 t 16<0 := by
  have h1 := positive_horner 736 1540 (t-1) (by decide) (by decide) (by omega)
  have h2 := positive_horner (736*(t-1)+1540) 1377 (t-1) h1 (by decide) (by omega)
  have h3 := positive_horner ((736*(t-1)+1540)*(t-1)+1377) 637 (t-1) h2 (by decide) (by omega)
  have hq : 0<736*t*t*t-668*t*t+505*t+64 := by grind only
  have hprod : 0<2*(4*t-1)*(736*t*t*t-668*t*t+505*t+64) :=
    Int.mul_pos (Int.mul_pos (by decide) (by omega)) hq
  have hi : scaledResidual 4 4 t 16= -(2*(4*t-1)*(736*t*t*t-668*t*t+505*t+64)) := by
    unfold scaledResidual
    simp only [Int.pow_succ,Int.pow_zero]
    grind only
  omega

theorem residual_3016_negative (t : Int) (ht : 1≤t) :
    scaledResidual 8 1 t 16<0 := by
  have h1 := positive_horner 6784 20368 (t-1) (by decide) (by decide) (by omega)
  have h2 := positive_horner (6784*(t-1)+20368) 22320 (t-1) h1 (by decide) (by omega)
  have h3 := positive_horner ((6784*(t-1)+20368)*(t-1)+22320) 10255 (t-1) h2 (by decide) (by omega)
  have h4 := positive_horner (((6784*(t-1)+20368)*(t-1)+22320)*(t-1)+10255)
    1743 (t-1) h3 (by decide) (by omega)
  have hq : 0<6784*t*t*t*t-6768*t*t*t+1920*t*t-417*t+224 := by grind only
  have hi : scaledResidual 8 1 t 16= -2*(6784*t*t*t*t-6768*t*t*t+1920*t*t-417*t+224) := by
    unfold scaledResidual
    simp only [Int.pow_succ,Int.pow_zero]
    grind only
  omega

theorem large_exclusion (k d e j : Nat) (hk : 6≤k)
    (hj : 1≤j) (hj20 : j≤20)
    (ha : (((21:Int)-j)*(2:Int)^k*(2:Int)^k-3)%7=0) :
    scaledResidual ((2:Int)^d) ((2:Int)^e) ((2:Int)^k) j ≠0 := by
  intro hp
  have ht0 : (2:Int)^k%64=0 := by
    have h := power_mod_cap k 6
    rw [Nat.min_eq_right hk] at h
    exact h
  have hc := congrArg (fun n : Int => n%64) hp
  have hc' : (18*(2:Int)^d*(2:Int)^e-9*(2:Int)^d*(2:Int)^d-j)%64=0 := by
    simpa [scaledResidual,Int.pow_succ,Int.add_emod,Int.sub_emod,Int.mul_emod,ht0] using hc
  have ht : (1:Int)≤(2:Int)^k := by
    have hh : (0:Int)<(2:Int)^k := Int.pow_pos (by decide)
    omega
  rcases large_patterns d e j hj hj20 hc' with H | H | H | H
  · obtain ⟨rfl,rfl,rfl⟩ := H
    have hn := residual_009_negative ((2:Int)^k) ht
    exact (Int.ne_of_lt hn) hp
  · obtain ⟨rfl,rfl,rfl⟩ := H
    have hx : (((21:Int)-7)*(2:Int)^k*(2:Int)^k-3)%7=4 := by
      simp [Int.sub_emod,Int.mul_emod]
    have hz : (4:Int)=0 := hx.symm.trans ha
    contradiction
  · obtain ⟨rfl,rfl,rfl⟩ := H
    have hn := residual_2216_negative ((2:Int)^k) ht
    exact (Int.ne_of_lt hn) hp
  · obtain ⟨rfl,rfl,rfl⟩ := H
    have hn := residual_3016_negative ((2:Int)^k) ht
    exact (Int.ne_of_lt hn) hp

theorem scaled_no_collision (k d e : Nat) (a : Int)
    (hk : 1≤k) (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^k*(2:Int)^e) 2
      ((2:Int)^k*(2:Int)^d) ((2:Int)^k) a≠0 := by
  intro hf
  obtain ⟨j,hj,hj20,haeq,hp⟩ := collision_deficit ((2:Int)^d) ((2:Int)^e)
    ((2:Int)^k) a (Int.pow_pos (by decide)) ha hab hf
  have hamod : (((21:Int)-j)*(2:Int)^k*(2:Int)^k-3)%7=0 := by
    rw [←haeq]
    simp
  by_cases hk5 : k≤5
  · exact small_exclusion k d e j hk hk5 hj hj20 hamod hp
  · exact large_exclusion k d e j (by omega) hj hj20 hamod hp

theorem wide_polynomial_no_collision (r c k : Nat) (a : Int)
    (hk : 1≤k) (hc : k≤c) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le hc
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  rw [hd,he,Int.pow_add,Int.pow_add]
  exact scaled_no_collision k d e a hk ha hab

/-- The arithmetic exclusion requires no parity or 3-nondivisibility condition on a. -/
theorem wide_numerator_no_collision (r c k a : Nat)
    (hk : 1≤k) (hc : k≤c) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2^k)^2) :
    3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k) ≠
      a*gap (towerC c (towerT 1 (zeroAt r))) := by
  intro he
  have hp := collision_polynomial r 1 c k a he
  have habi : (a:Int)<3*(2:Int)^k*(2:Int)^k := by
    have h : (a:Int)<((3*(2^k)^2:Nat):Int) := by omega
    simpa only [Int.natCast_mul,Int.natCast_pow,
      show ((2:Nat):Int)=2 from rfl,show ((3:Nat):Int)=3 from rfl,
      Int.pow_succ,Int.pow_zero,Int.one_mul,Int.mul_assoc] using h
  exact wide_polynomial_no_collision r c k a hk hc hr (by omega) habi hp


/-- Uniform count-level exclusion with no tail-depth, parity, or mod-3 hypothesis. -/
theorem wide_small_numerator_no_collision (r c k a : Nat)
    (hk : 1≤k) (hc : k≤c) (ha : 0<a) (hab : a<3*(2^k)^2) :
    3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k) ≠
      a*gap (towerC c (towerT 1 (zeroAt r))) := by
  intro he
  let y := towerC c (towerT 1 (zeroAt r))
  have hgmod := (towerC_mod (towerT 1 (zeroAt r)) c).2.2.2
  have hs : 2≤2^c := Nat.pow_le_pow_right (n:=2) (by decide) (show 1≤c by omega)
  have h1 : 1%2^c=1 := Nat.mod_eq_of_lt (by omega)
  rw [h1] at hgmod
  have hg : 0<gap y := by
    change gap y%2^c=1 at hgmod
    have hle := Nat.mod_le (gap y) (2^c)
    omega
  have ht4 : (4:Nat)^k=(2^k)^2 := by
    rw [show (4:Nat)=2*2 from rfl,Nat.mul_pow]
    simp [Nat.pow_succ]
  have hgap := tower_gap y k
  rw [ht4] at hgap
  have hmul := Nat.mul_lt_mul_of_pos_right hab hg
  have hlow : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)
      <3*gap (twoBlock r 1 c k) := by
    change gap (twoBlock r 1 c k)=(2^k)^2*gap y at hgap
    change 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=a*gap y at he
    rw [he,hgap]
    simpa only [Nat.mul_assoc] using hmul
  exact wide_numerator_no_collision r c k a hk hc
    (oneInner_low_outer r c k hlow) ha hab he

/-- Any small odd value collision with h=1 would require c<k, for all r and k. -/
theorem one_inner_collision_requires_narrow_separator (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) : c<k := by
  by_cases hck : c<k
  · exact hck
  exfalso
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact wide_small_numerator_no_collision r c k a hk (by omega) (by omega) H.2.2.1 H.2.1


#print axioms rescale
#print axioms deficit_rescale
#print axioms quotient_bounds
#print axioms collision_deficit
#print axioms power_mod_cap
#print axioms residual_mod_congr
#print axioms small_mod_certificate
#print axioms large_mod_certificate
#print axioms small_exclusion
#print axioms constant_mod_congr
#print axioms large_patterns
#print axioms positive_horner
#print axioms residual_009_negative
#print axioms residual_2216_negative
#print axioms residual_3016_negative
#print axioms large_exclusion
#print axioms scaled_no_collision
#print axioms wide_polynomial_no_collision
#print axioms wide_numerator_no_collision
#print axioms wide_small_numerator_no_collision
#print axioms one_inner_collision_requires_narrow_separator
end Sounio.ZDScalarWideSeparator

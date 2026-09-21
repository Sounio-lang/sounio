import SounioZDScalarWideSeparator

/-! A uniform exclusion inside the remaining narrow-separator region.
The result does not prove global ResidualSeparation. -/
namespace Sounio.ZDScalarNarrowSeparator
open Sounio.ZDScalarInnerBound Sounio.ZDScalarWideSeparator Sounio.ZDScalarDiscriminant
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility Sounio.ZDScalarLowCone
open Sounio.ZDScalarTwoBlock
set_option maxRecDepth 32768
set_option maxHeartbeats 32000000

def narrowGapPart (s w z : Int) : Int := 8*s*s*w*w*z*z-6*s*w*z-6*w*z+3
def narrowNumPart (s w z : Int) : Int :=
  252*s^4*w^4*z+84*s^3*w^4*z-126*s^3*w^3*z-126*s^3*w^3+
  63*s*s*w*w-126*s*s*w^3*z-18*s*w^3+42*s*w*w*z+21*w*w
def narrowA (w z j : Int) : Int :=
  -8*j*w*w*z*z+168*w^4*z*z-252*w^4*z
def narrowB (w z j : Int) : Int := 6*j*w*z-84*w^4*z+126*w^3
def narrowC (w z j : Int) : Int := 6*j*w*z-3*j-24*w*w*z*z
def narrowD (w z : Int) : Int := 18*w^3-42*w*w*z+18*w*z
def narrowResidual (s w z j : Int) : Int :=
  narrowA w z j*s^4+narrowB w z j*s^3+narrowC w z j*s*s+
  narrowD w z*s+18*w*z-9-j

theorem narrow_rescale (s w z a : Int) :
    collisionPoly (s*w*z) 2 s (s*w) a =
      7*a+3+s*s*(7*a*narrowGapPart s w z-narrowNumPart s w z) := by
  simp only [collisionPoly,gapPoly,numPoly,innerM,innerP28,narrowGapPart,narrowNumPart,
    Int.pow_succ,Int.pow_zero]
  grind only

theorem narrow_deficit_rescale (s w z a j : Int)
    (ha : 7*a=(21*w*w-j)*s*s-3) :
    collisionPoly (s*w*z) 2 s (s*w) a=s*s*narrowResidual s w z j := by
  rw [narrow_rescale]
  simp only [narrowGapPart,narrowNumPart,narrowResidual,narrowA,narrowB,narrowC,narrowD,
    Int.pow_succ,Int.pow_zero]
  grind only

private theorem quotient_range (n x q m : Int) (hn : 0<n)
    (hx : 0<x) (hu : x<m*n) (he : x=n*q) : 0<q ∧ q<m := by
  constructor
  · by_cases hq : 0<q
    · exact hq
    have hh := Int.mul_le_mul_of_nonneg_left (show q≤0 by omega) (show 0≤n by omega)
    omega
  · by_cases hq : q<m
    · exact hq
    have hh := Int.mul_le_mul_of_nonneg_left (show m≤q by omega) (show 0≤n by omega)
    have hm : n*m=m*n := Int.mul_comm _ _
    omega

theorem narrow_collision_deficit (s w z a : Int)
    (hs : 0<s) (ha : 0<a) (hab : a<3*(s*w)*(s*w))
    (hf : collisionPoly (s*w*z) 2 s (s*w) a=0) :
    ∃ j : Int, 0<j ∧ j<21*w*w ∧ 7*a=(21*w*w-j)*s*s-3 ∧
      narrowResidual s w z j=0 := by
  have hres := narrow_rescale s w z a
  let q := narrowNumPart s w z-7*a*narrowGapPart s w z
  have he : 7*a+3=s*s*q := by dsimp [q]; grind only
  have hn : 0<s*s := Int.mul_pos hs hs
  have hu : 7*a+3<(21*w*w)*(s*s) := by
    have hm : (21*w*w)*(s*s)=7*(3*(s*w)*(s*w)) := by grind only
    omega
  have hq := quotient_range (s*s) (7*a+3) q (21*w*w) hn (by omega) hu he
  let j := 21*w*w-q
  have hj : 0<j ∧ j<21*w*w := by dsimp [j]; omega
  have haeq : 7*a=(21*w*w-j)*s*s-3 := by dsimp [j]; grind only
  have hp := narrow_deficit_rescale s w z a j haeq
  have hz : s*s*narrowResidual s w z j=0 := by omega
  exact ⟨j,hj.1,hj.2,haeq,(Int.mul_eq_zero.mp hz).resolve_left (by omega)⟩

private theorem pow_mono (a b : Nat) (h : a≤b) : (2:Int)^a≤(2:Int)^b := by
  have H := Nat.pow_le_pow_right (n:=2) (by decide) h
  have hh : ((2^a:Nat):Int)≤((2^b:Nat):Int) := by omega
  simpa only [Int.natCast_pow,show ((2:Nat):Int)=2 from rfl] using hh

theorem narrow_modulus_factor (w z v j : Int) :
    let s := 32*w*v
    narrowResidual s w z j =
      18*w*z-9-j+64*w*w*
        (16*v*v*(narrowA w z j*s*s+narrowB w z j*s+narrowC w z j)+
          3*v*(3*w*w-7*w*z+3*z)) := by
  dsimp
  simp only [narrowResidual,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem narrow_constant_congruence (c d e : Nat) (hcd : d+5≤c)
    (j : Int) (hp : narrowResidual ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e)%(64*(2:Int)^d*(2:Int)^d)=0 := by
  obtain ⟨n,hn⟩ := Nat.exists_eq_add_of_le hcd
  have hs : (2:Int)^c=32*(2:Int)^d*(2:Int)^n := by
    rw [hn,Int.pow_add,Int.pow_add]
    grind only
  rw [hs] at hp
  have h := narrow_modulus_factor ((2:Int)^d) ((2:Int)^e) ((2:Int)^n) j
  dsimp only at h
  have hdvd : 64*(2:Int)^d*(2:Int)^d ∣ j+9-18*(2:Int)^d*(2:Int)^e := by
    refine ⟨16*(2:Int)^n*(2:Int)^n*
      (narrowA ((2:Int)^d) ((2:Int)^e) j*(32*(2:Int)^d*(2:Int)^n)*(32*(2:Int)^d*(2:Int)^n)+
      narrowB ((2:Int)^d) ((2:Int)^e) j*(32*(2:Int)^d*(2:Int)^n)+
      narrowC ((2:Int)^d) ((2:Int)^e) j)+
      3*(2:Int)^n*(3*(2:Int)^d*(2:Int)^d-7*(2:Int)^d*(2:Int)^e+3*(2:Int)^e),?_⟩
    grind only
  exact Int.emod_eq_zero_of_dvd hdvd

theorem quotient_residue_certificate :
    ∀ (f : Fin 7) (n : Fin 24), 1≤f.val → 1≤n.val →
      (n.val:Int)%64=(18*(2:Int)^f.val)%64 →
      (f.val=2 ∧ n.val=8) ∨ (f.val=3 ∧ n.val=16) := by decide

theorem narrow_residue_patterns (d e : Nat) (_hd : 1≤d) (J : Int)
    (hJ : 0<J) (hJu : J<24*(2:Int)^d*(2:Int)^d)
    (hmod : (J-18*(2:Int)^d*(2:Int)^e)%(64*(2:Int)^d*(2:Int)^d)=0) :
    (e≤d ∧ J=18*(2:Int)^d*(2:Int)^e) ∨
    (e=d+2 ∧ J=8*(2:Int)^d*(2:Int)^d) ∨
    (e=d+3 ∧ J=16*(2:Int)^d*(2:Int)^d) := by
  let w : Int := 2^d
  let z : Int := 2^e
  have hw : 0<w := Int.pow_pos (by decide)
  have hz : 0<z := Int.pow_pos (by decide)
  have hw2 : 0<w*w := Int.mul_pos hw hw
  change 0<J at hJ
  change J<24*w*w at hJu
  change (J-18*w*z)%(64*w*w)=0 at hmod
  by_cases hed : e≤d
  · have hzw : z≤w := pow_mono e d hed
    have hmul := Int.mul_le_mul_of_nonneg_left hzw (show 0≤18*w by omega)
    have h18 : 0≤18*w*z := Int.le_of_lt (Int.mul_pos (by omega) hz)
    have hnorm1 : 18*w*w=18*(w*w) := Int.mul_assoc _ _ _
    have hnorm2 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
    have hnorm3 : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
    have hEq := Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr hmod
    rw [Int.emod_eq_of_lt (by omega) (by omega),
      Int.emod_eq_of_lt h18 (by omega)] at hEq
    exact Or.inl ⟨hed,hEq⟩
  · obtain ⟨f,hf⟩ := Nat.exists_eq_add_of_le (show d≤e by omega)
    have hfpos : 1≤f := by omega
    have hzw : z=w*(2:Int)^f := by dsimp [w,z]; rw [hf,Int.pow_add]
    obtain ⟨v,hv⟩ := Int.dvd_of_emod_eq_zero hmod
    let n : Int := 18*(2:Int)^f+64*v
    have heq : J=(w*w)*n := by rw [hzw] at hv; dsimp [n]; grind only
    have hupper : J<24*(w*w) := by simpa only [Int.mul_assoc] using hJu
    have hn := quotient_range (w*w) J n 24 hw2 hJ hupper heq
    let nn := n.toNat
    have hnn : (nn:Int)=n := Int.toNat_of_nonneg (by omega)
    have hm : n%64=(18*(2:Int)^f)%64 := by dsimp [n]; simp [Int.add_emod]
    have hcap := power_mod_cap f 6
    change (2:Int)^f%64=(2:Int)^(min f 6)%64 at hcap
    have hm' : (nn:Int)%64=(18*(2:Int)^(min f 6))%64 := by
      rw [hnn,hm]
      simp [Int.mul_emod,hcap]
    have hfm : 1≤min f 6 := by omega
    have hnnlo : 1≤nn := by omega
    have hnnhi : nn<24 := by omega
    have cert := quotient_residue_certificate ⟨min f 6,by omega⟩ ⟨nn,hnnhi⟩ hfm hnnlo hm'
    rcases cert with hc | hc
    · have hf2 : f=2 := by simp only at hc; omega
      have hn8 : n=8 := by simp only at hc; omega
      right; left
      constructor
      · omega
      · rw [hn8] at heq; change J=8*w*w; grind only
    · have hf3 : f=3 := by simp only at hc; omega
      have hn16 : n=16 := by simp only at hc; omega
      right; right
      constructor
      · omega
      · rw [hn16] at heq; change J=16*w*w; grind only


theorem wrap_four_factor (q v : Int) :
    let w := 2*q
    let s := 32*w*v
    let j := 8*w*w-9
    narrowResidual s w (4*w) j =
      64*w*w*(1+4*(4*v*v*(narrowA w (4*w) j*s*s+
        narrowB w (4*w) j*s+narrowC w (4*w) j)+3*q*v*(6-25*q))) := by
  dsimp
  simp only [narrowResidual,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem wrap_eight_factor (q v : Int) :
    let w := 2*q
    let s := 32*w*v
    let j := 16*w*w-9
    narrowResidual s w (8*w) j =
      128*w*w*(1+2*(4*v*v*(narrowA w (8*w) j*s*s+
        narrowB w (8*w) j*s+narrowC w (8*w) j)+3*q*v*(12-53*q))) := by
  dsimp
  simp only [narrowResidual,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem wrap_four_nonzero (c d : Nat) (hd : 1≤d) (hcd : d+5≤c) :
    narrowResidual ((2:Int)^c) ((2:Int)^d) (4*(2:Int)^d)
      (8*(2:Int)^d*(2:Int)^d-9)≠0 := by
  obtain ⟨n,hn⟩ := Nat.exists_eq_add_of_le hd
  obtain ⟨m,hm⟩ := Nat.exists_eq_add_of_le hcd
  have hw : (2:Int)^d=2*(2:Int)^n := by rw [hn,Int.pow_add]; rfl
  have hs : (2:Int)^c=32*(2:Int)^d*(2:Int)^m := by
    rw [hm,Int.pow_add,Int.pow_add]; grind only
  intro hp
  rw [hs,hw] at hp
  have H := wrap_four_factor ((2:Int)^n) ((2:Int)^m)
  dsimp only at H
  have hwpos : (0:Int)<2*(2:Int)^n := Int.mul_pos (by decide) (Int.pow_pos (by decide))
  have hpos : (0:Int)<64*(2*(2:Int)^n)*(2*(2:Int)^n) :=
    Int.mul_pos (Int.mul_pos (by decide) hwpos) hwpos
  rw [hp] at H
  have hz := (Int.mul_eq_zero.mp H.symm).resolve_left (by omega)
  omega

theorem wrap_eight_nonzero (c d : Nat) (hd : 1≤d) (hcd : d+5≤c) :
    narrowResidual ((2:Int)^c) ((2:Int)^d) (8*(2:Int)^d)
      (16*(2:Int)^d*(2:Int)^d-9)≠0 := by
  obtain ⟨n,hn⟩ := Nat.exists_eq_add_of_le hd
  obtain ⟨m,hm⟩ := Nat.exists_eq_add_of_le hcd
  have hw : (2:Int)^d=2*(2:Int)^n := by rw [hn,Int.pow_add]; rfl
  have hs : (2:Int)^c=32*(2:Int)^d*(2:Int)^m := by
    rw [hm,Int.pow_add,Int.pow_add]; grind only
  intro hp
  rw [hs,hw] at hp
  have H := wrap_eight_factor ((2:Int)^n) ((2:Int)^m)
  dsimp only at H
  have hwpos : (0:Int)<2*(2:Int)^n := Int.mul_pos (by decide) (Int.pow_pos (by decide))
  have hpos : (0:Int)<128*(2*(2:Int)^n)*(2*(2:Int)^n) :=
    Int.mul_pos (Int.mul_pos (by decide) hwpos) hwpos
  rw [hp] at H
  have hz := (Int.mul_eq_zero.mp H.symm).resolve_left (by omega)
  omega

theorem narrowD_valuation (d e : Nat) (hd : 1≤d) (he : e≤d) :
    ExactTwoVal (narrowD ((2:Int)^d) ((2:Int)^e)) (d+e+1) := by
  obtain ⟨n,hn⟩ := Nat.exists_eq_add_of_le he
  let w : Int := 2^d
  let z : Int := 2^e
  let B : Int := 3*w*(2:Int)^n-7*w+3
  have hw : w=z*(2:Int)^n := by dsimp [w,z]; rw [hn,Int.pow_add]
  have hwmod : w%2=0 := pow_two_even d hd
  have hodd : (3*B)%2=1 := by simp [B,Int.add_emod,Int.sub_emod,Int.mul_emod,hwmod]
  have hpow : (2:Int)^(d+e+1)=2*w*z := by
    rw [Int.pow_add,Int.pow_add]
    dsimp [w,z]
    grind only
  refine ⟨3*B,hodd,?_⟩
  rw [hpow]
  change narrowD w z=2*w*z*(3*B)
  simp only [narrowD,B,Int.pow_succ,Int.pow_zero]
  grind only

theorem main_pattern_factor (s w z : Int) :
    narrowResidual s w z (18*w*z-9) =
      s*(narrowD w z+s*(narrowC w z (18*w*z-9)+
        s*narrowB w z (18*w*z-9)+s*s*narrowA w z (18*w*z-9))) := by
  simp only [narrowResidual,Int.pow_succ,Int.pow_zero]
  grind only

theorem main_pattern_exponent (c d e : Nat) (hc : 1≤c) (hd : 1≤d) (he : e≤d)
    (hp : narrowResidual ((2:Int)^c) ((2:Int)^d) ((2:Int)^e)
      (18*(2:Int)^d*(2:Int)^e-9)=0) : c=d+e+1 := by
  let s : Int := 2^c
  let w : Int := 2^d
  let z : Int := 2^e
  let j : Int := 18*w*z-9
  let Q := narrowC w z j+s*narrowB w z j+s*s*narrowA w z j
  have hs : 0<s := Int.pow_pos (by decide)
  have hsmod : s%2=0 := pow_two_even c hc
  have hq : Q%2=1 := by
    simp [Q,j,narrowC,Int.add_emod,Int.sub_emod,Int.mul_emod,hsmod]
  have hfactor := main_pattern_factor s w z
  change narrowResidual s w z j=0 at hp
  change narrowResidual s w z j=s*(narrowD w z+s*Q) at hfactor
  rw [hp] at hfactor
  have hzero := (Int.mul_eq_zero.mp hfactor.symm).resolve_left (by omega)
  have hv : ExactTwoVal (narrowD w z) c := by
    refine ⟨-Q,by omega,?_⟩
    change narrowD w z=s*(-Q)
    grind only
  exact exact_two_val_unique (narrowD w z) c (d+e+1) hv (narrowD_valuation d e hd he)

def criticalH (w z : Int) : Int :=
  224*w^6*z^5-336*w^6*z^4-192*w^5*z^6-56*w^5*z^3+
  96*w^4*z^5+84*w^4*z*z+72*w^3*z^4-8*w*w*z^3+3*w*w-
  36*w*z*z-7*w*z+12*z

theorem critical_factor (w z : Int) :
    narrowResidual (2*w*z) w z (18*w*z-9)=12*w*w*z*criticalH w z := by
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,criticalH,
    Int.pow_succ,Int.pow_zero]
  grind only


def criticalNat (x y : Nat) : Nat :=
  (((((((((((224*y+17584)*y+551936)*y+8658944)*y+67895296)*y+212860928)*x+((((((1152*y+108576)*y+4262400)*y+89210824)*y+1049884032)*y+6587111424)*y+17213194240))*x+(((((((2400*y+263760)*y+12418656)*y+324717800)*y+5092375040)*y+47897210964)*y+250175163008)*y+559770260480))*x+((((((((2560*y+320960)*y+17597824)*y+551112144)*y+10782101832)*y+134939816784)*y+1054970474240)*y+4710591754240)*y+9197091356672))*x+(((((((((1440*y+202320)*y+12626496)*y+459389392)*y+10737839832)*y+167212971384)*y+1734697872888)*y+11560236011136)*y+44903660120064)*y+77454894399491))*x+((((((((((384*y+59424)*y+4133760)*y+170213096)*y+4593911384)*y+84908005968)*y+1088281940208)*y+9550309825536)*y+54909512294364)*y+186748123216767)*y+285252080622576))*x+(((((((((((32*y+5296)*y+396896)*y+17769928)*y+527787080)*y+10910564308)*y+160029607800)*y+1663315372416)*y+11986284883932)*y+56909447493948)*y+159730225353612)*y+199891326188736))

theorem critical_shift (x y : Nat) :
    criticalH ((x:Int)+(y:Int)+16) ((y:Int)+16) = (criticalNat x y:Int) := by
  simp only [criticalH,criticalNat,Int.natCast_add,Int.natCast_mul,
    Int.pow_succ,Int.pow_zero]
  grind only

theorem critical_positive (w z : Int) (hw : z≤w) (hz : 16≤z) :
    0<criticalH w z := by
  let x := (w-z).toNat
  let y := (z-16).toNat
  have hx : (x:Int)=w-z := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=z-16 := Int.toNat_of_nonneg (by omega)
  have hn : 0<criticalNat x y := by unfold criticalNat; omega
  have hni : (0:Int)<(criticalNat x y:Int) := by omega
  have hi := critical_shift x y
  have hw' : (x:Int)+(y:Int)+16=w := by omega
  have hz' : (y:Int)+16=z := by omega
  rw [hw',hz'] at hi
  omega

theorem main_pattern_nonzero (c d e : Nat) (hd : 1≤d) (hcd : d+5≤c) (he : e≤d) :
    narrowResidual ((2:Int)^c) ((2:Int)^d) ((2:Int)^e)
      (18*(2:Int)^d*(2:Int)^e-9)≠0 := by
  intro hp
  have hc := main_pattern_exponent c d e (by omega) hd he hp
  have hs : (2:Int)^c=2*(2:Int)^d*(2:Int)^e := by
    rw [hc,Int.pow_add,Int.pow_add]; grind only
  have he4 : 4≤e := by omega
  have hz16 : (16:Int)≤(2:Int)^e := pow_mono 4 e he4
  have hwz : (2:Int)^e≤(2:Int)^d := pow_mono e d he
  have hH := critical_positive ((2:Int)^d) ((2:Int)^e) hwz hz16
  have hw : (0:Int)<(2:Int)^d := Int.pow_pos (by decide)
  have hz : (0:Int)<(2:Int)^e := Int.pow_pos (by decide)
  have hpos : (0:Int)<12*(2:Int)^d*(2:Int)^d*(2:Int)^e*criticalH ((2:Int)^d) ((2:Int)^e) :=
    Int.mul_pos (Int.mul_pos (Int.mul_pos (Int.mul_pos (by decide) hw) hw) hz) hH
  rw [hs,critical_factor] at hp
  omega

theorem narrow_scaled_no_collision (c d e : Nat) (a : Int)
    (hd : 1≤d) (hcd : d+5≤c) (ha : 0<a)
    (hab : a<3*((2:Int)^c*(2:Int)^d)*((2:Int)^c*(2:Int)^d)) :
    collisionPoly ((2:Int)^c*(2:Int)^d*(2:Int)^e) 2
      ((2:Int)^c) ((2:Int)^c*(2:Int)^d) a≠0 := by
  intro hf
  obtain ⟨j,hj,hju,haeq,hp⟩ := narrow_collision_deficit
    ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) a (Int.pow_pos (by decide)) ha hab hf
  have hcon := narrow_constant_congruence c d e hcd j hp
  have hw : (2:Int)≤(2:Int)^d := pow_mono 1 d hd
  have hw2 : (4:Int)≤(2:Int)^d*(2:Int)^d :=
    Int.mul_le_mul hw hw (by decide) (by omega)
  have hJu : j+9<24*(2:Int)^d*(2:Int)^d := by
    have h1 : 21*(2:Int)^d*(2:Int)^d=21*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
    have h2 : 24*(2:Int)^d*(2:Int)^d=24*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
    omega
  rcases narrow_residue_patterns d e hd (j+9) (by omega) hJu hcon with H | H | H
  · obtain ⟨he,hJ⟩ := H
    have hj' : j=18*(2:Int)^d*(2:Int)^e-9 := by omega
    rw [hj'] at hp
    exact main_pattern_nonzero c d e hd hcd he hp
  · obtain ⟨he,hJ⟩ := H
    have hj' : j=8*(2:Int)^d*(2:Int)^d-9 := by omega
    have hz : (2:Int)^e=4*(2:Int)^d := by rw [he,Int.pow_add]; grind only
    rw [hj',hz] at hp
    exact wrap_four_nonzero c d hd hcd hp
  · obtain ⟨he,hJ⟩ := H
    have hj' : j=16*(2:Int)^d*(2:Int)^d-9 := by omega
    have hz : (2:Int)^e=8*(2:Int)^d := by rw [he,Int.pow_add]; grind only
    rw [hj',hz] at hp
    exact wrap_eight_nonzero c d hd hcd hp

theorem narrow_polynomial_no_collision (r c k : Nat) (a : Int)
    (hck : c<k) (hbal : k+5≤2*c) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have ht : (2:Int)^k=(2:Int)^c*(2:Int)^d := by rw [hd,Int.pow_add]
  have hu : (2:Int)^(r+1)=(2:Int)^c*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [ht,hu]
  exact narrow_scaled_no_collision c d e a (by omega) (by omega) ha hab

theorem small_numerator_depth (r c k a : Nat) (hc : 1≤c)
    (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) : k≤r+1 := by
  let y := towerC c (towerT 1 (zeroAt r))
  have hgmod := (towerC_mod (towerT 1 (zeroAt r)) c).2.2.2
  have hs : 2≤2^c := Nat.pow_le_pow_right (n:=2) (by decide) hc
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
  exact oneInner_low_outer r c k hlow

/-- All r are covered; no parity or mod-3 condition on a is needed. -/
theorem narrow_numerator_no_collision (r c k a : Nat)
    (hc : 1≤c) (hck : c<k) (hbal : k+5≤2*c)
    (ha : 0<a) (hab : a<3*(2^k)^2) :
    3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k) ≠
      a*gap (towerC c (towerT 1 (zeroAt r))) := by
  intro he
  have hr := small_numerator_depth r c k a hc hab he
  have hp := collision_polynomial r 1 c k a he
  have habi : (a:Int)<3*(2:Int)^k*(2:Int)^k := by
    have h : (a:Int)<((3*(2^k)^2:Nat):Int) := by omega
    simpa only [Int.natCast_mul,Int.natCast_pow,
      show ((2:Nat):Int)=2 from rfl,show ((3:Nat):Int)=3 from rfl,
      Int.pow_succ,Int.pow_zero,Int.one_mul,Int.mul_assoc] using h
  exact narrow_polynomial_no_collision r c k a hck hbal hr (by omega) habi hp

/-- Any remaining small count collision lies below the new separator bound. -/
theorem one_inner_small_collision_bounds (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) : c<k ∧ 2*c≤k+4 := by
  have hck : c<k := by
    by_cases H : c<k
    · exact H
    exact False.elim (wide_small_numerator_no_collision r c k a hk (by omega) ha hab he)
  refine ⟨hck,?_⟩
  by_cases H : 2*c≤k+4
  · exact H
  exact False.elim (narrow_numerator_no_collision r c k a hc hck (by omega) ha hab he)

/-- The original odd scalar value collision inherits both separator bounds. -/
theorem one_inner_value_collision_bounds (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) : c<k ∧ 2*c≤k+4 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_small_collision_bounds r c k a hc hk (by omega) H.2.2.1 H.2.1



def upperNat (x y z : Nat) : Nat :=
  ((((56*y+224)*y+224)*x+(((((644*z+3990)*z+8232)*z+5614)*y+(((2492*z+15582)*z+32382)*z+22162))*y+(((2408*z+15204)*z+31836)*z+21868)))*x+((((((((1568*z+19824)*z+104160)*z+290990)*z+455637)*z+378924)*z+130697)*y+((((((5600*z+72240)*z+386064)*z+1093736)*z+1731828)*z+1452528)*z+503972))*y+((((((4928*z+65184)*z+355488)*z+1023530)*z+1641195)*z+1389492)*z+485226)))

theorem upper_shift (x y z : Nat) :
    7*gapPoly (8*((z:Int)+2)^3+(x:Int)) 2 ((y:Int)+2) -
      numPoly (8*((z:Int)+2)^3+(x:Int)) 2 ((y:Int)+2) ((z:Int)+2) =
      (upperNat x y z:Int) := by
  simp only [upperNat,gapPoly,numPoly,innerM,innerP28,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem numerator_below_gap (s t u : Int) (hs : 2≤s) (ht : 2≤t)
    (hu : 8*t^3≤u) : numPoly u 2 s t < 7*gapPoly u 2 s := by
  let x := (u-8*t^3).toNat
  let y := (s-2).toNat
  let z := (t-2).toNat
  have hx : (x:Int)=u-8*t^3 := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=s-2 := Int.toNat_of_nonneg (by omega)
  have hz : (z:Int)=t-2 := Int.toNat_of_nonneg (by omega)
  have hp : 0<upperNat x y z := by unfold upperNat; omega
  have hpi : (0:Int)<(upperNat x y z:Int) := by omega
  have H := upper_shift x y z
  have hys : (y:Int)+2=s := by omega
  have hzt : (z:Int)+2=t := by omega
  rw [hys,hzt] at H
  have hxu : 8*t^3+(x:Int)=u := by omega
  rw [hxu] at H
  omega

/-- A positive small coefficient leaves only finitely many r for each k. -/
theorem one_inner_tail_upper (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) : r≤3*k+1 := by
  by_cases H : r≤3*k+1
  · exact H
  exfalso
  have hpow : (2:Int)^(3*k+3)=8*((2:Int)^k)^3 := by
    rw [Int.pow_add,show 3*k=k*3 by omega,Int.pow_mul]
    grind only
  have hu : 8*((2:Int)^k)^3≤(2:Int)^(r+1) := by
    rw [←hpow]
    exact pow_mono _ _ (by omega)
  have hng := numerator_below_gap ((2:Int)^c) ((2:Int)^k) ((2:Int)^(r+1))
    (pow_mono 1 c hc) (pow_mono 1 k hk) hu
  have hnum := twoBlock_numerator r 1 c k
  have hgap := twoBlock_gap r 1 c
  let G := gap (towerC c (towerT 1 (zeroAt r)))
  let N := 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)
  have hmul := Nat.mul_le_mul_right G (show 1≤a by omega)
  have hGN : G≤N := by change N=a*G at he; omega
  have hi : (G:Int)≤(N:Int) := by omega
  have hnum' : 7*(N:Int)=numPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) := by
    simpa [N] using hnum
  have hgap' : (G:Int)=gapPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) := hgap
  rw [←hnum',←hgap'] at hng
  omega

/-- Uniform reduction to a finite integer domain for each fixed outer length. -/
theorem one_inner_small_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c≤k+4 ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := one_inner_small_collision_bounds r c k a hc hk ha hab he
  exact ⟨H.1,H.2,small_numerator_depth r c k a hc hab he,
    one_inner_tail_upper r c k a hc hk ha he⟩

theorem one_inner_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c≤k+4 ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_small_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1


/-- These are the exact integer formulas transcribed in narrow_scan.py. -/
def scanGap (u s : Int) : Int := 8*s*s*u*u-6*s*(s+1)*u+3*s*s+1
def scanSlope (s t : Int) : Int :=
  252*s*s*t^3-126*s*s*t*t+84*s*t^3-126*s*t*t+42*s*t
def scanConstant (s t : Int) : Int :=
  -126*s*s*t^3+63*s*s*t*t+21*t*t-18*t^3-3

theorem scan_gap_identity (u s : Int) : gapPoly u 2 s=scanGap u s := by
  unfold gapPoly scanGap
  grind only

theorem scan_numerator_identity (u s t : Int) :
    numPoly u 2 s t=scanSlope s t*u+scanConstant s t := by
  simp only [numPoly,innerM,innerP28,scanSlope,scanConstant,Int.pow_succ,Int.pow_zero]
  grind only

theorem scan_factored_identity (u s t : Int) :
    scanSlope s t*u+scanConstant s t =
      3*(2*t-1)*(42*s*s*t*t*u-21*s*s*t*t+14*s*t*t*u-
        14*s*t*u-3*t*t+2*t+1) := by
  simp only [scanSlope,scanConstant,Int.pow_succ,Int.pow_zero]
  grind only

/-- Uniform source-level bridge for both formulas used by the external census.
This does not verify the Python interpreter or certify the finite census. -/
theorem scan_count_formulas (r c k : Nat) :
    (gap (towerC c (towerT 1 (zeroAt r))):Int)=
      scanGap ((2:Int)^(r+1)) ((2:Int)^c) ∧
    7*(3*(edges (twoBlock r 1 c k):Int)+4*(positives (twoBlock r 1 c k):Int)) =
      scanSlope ((2:Int)^c) ((2:Int)^k)*(2:Int)^(r+1)+
        scanConstant ((2:Int)^c) ((2:Int)^k) := by
  constructor
  · exact (twoBlock_gap r 1 c).trans (scan_gap_identity _ _)
  · exact (twoBlock_numerator r 1 c k).trans (scan_numerator_identity _ _ _)


#print axioms narrow_rescale
#print axioms narrow_deficit_rescale
#print axioms quotient_range
#print axioms narrow_collision_deficit
#print axioms pow_mono
#print axioms narrow_modulus_factor
#print axioms narrow_constant_congruence
#print axioms quotient_residue_certificate
#print axioms narrow_residue_patterns
#print axioms wrap_four_factor
#print axioms wrap_eight_factor
#print axioms wrap_four_nonzero
#print axioms wrap_eight_nonzero
#print axioms narrowD_valuation
#print axioms main_pattern_factor
#print axioms main_pattern_exponent
#print axioms critical_factor
#print axioms critical_shift
#print axioms critical_positive
#print axioms main_pattern_nonzero
#print axioms narrow_scaled_no_collision
#print axioms narrow_polynomial_no_collision
#print axioms small_numerator_depth
#print axioms narrow_numerator_no_collision
#print axioms one_inner_small_collision_bounds
#print axioms one_inner_value_collision_bounds
#print axioms upper_shift
#print axioms numerator_below_gap
#print axioms one_inner_tail_upper
#print axioms one_inner_small_collision_region
#print axioms one_inner_value_collision_region
#print axioms scan_gap_identity
#print axioms scan_numerator_identity
#print axioms scan_factored_identity
#print axioms scan_count_formulas
end Sounio.ZDScalarNarrowSeparator

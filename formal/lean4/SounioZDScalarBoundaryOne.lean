import SounioZDScalarBoundaryTwo

/-! Uniform exclusion of 2c=k+1 in the explicit h=1 small-count model.
This does not assert global ResidualSeparation or bibliographic priority. -/
namespace Sounio.ZDScalarBoundaryOne
open Sounio.ZDScalarBoundaryTwo
open Sounio.ZDScalarHalfSeparator
open Sounio.ZDScalarNarrowSeparator Sounio.ZDScalarInnerBound
open Sounio.ZDScalarWideSeparator Sounio.ZDScalarDiscriminant
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility Sounio.ZDScalarLowCone
open Sounio.ZDScalarTwoBlock
set_option maxRecDepth 32768
set_option maxHeartbeats 32000000

private theorem pow_mono (a b : Nat) (h : a≤b) : (2:Int)^a≤(2:Int)^b := by
  have H := Nat.pow_le_pow_right (n:=2) (by decide) h
  have hh : ((2^a:Nat):Int)≤((2^b:Nat):Int) := by omega
  simpa only [Int.natCast_pow,show ((2:Nat):Int)=2 from rfl] using hh


private theorem small_mod_unique (R J M : Int)
    (hR : 0≤R) (hRM : R<M) (hJ : 0≤J) (hJM : J<M) (hmod : (J-R)%M=0) : J=R := by
  have H := Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr hmod
  rw [Int.emod_eq_of_lt hJ hJM,Int.emod_eq_of_lt hR hRM] at H
  exact H

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



def lowZeroNat (x : Nat) : Nat := ((((((((3392*x+57248)*x+418640)*x+1733216)*x+4444412)*x+7229156)*x+7284792)*x+4158384)*x+1029600)
theorem lowZero_shift (x : Nat) :
    let w := (x:Int)+2
    (-(narrowResidual (2*w) w 1 (16*w*w+18*w-9)))=(lowZeroNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,lowZeroNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem lowZero_sign (w : Int) (hw : 2≤w) :
    narrowResidual (2*w) w 1 (16*w*w+18*w-9)<0 := by
  let x := (w-2).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have H := lowZero_shift x
  dsimp only at H
  have hw' : (x:Int)+2=w := by omega
  rw [hw'] at H
  have hn : 0<lowZeroNat x := by unfold lowZeroNat; omega
  have hi : (0:Int)<(lowZeroNat x:Int) := by omega
  omega

def lowThreeNat (x : Nat) : Nat := ((((((((41472*x+9431808)*x+923719152)*x+50633987328)*x+1686205703700)*x+34489907214816)*x+413094191551872)*x+2509146123853824)*x+4968978010472448)
theorem lowThree_shift (x : Nat) :
    let w := (x:Int)+32
    narrowResidual (2*w) w 8 (12*w*w+144*w-9)=(lowThreeNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,lowThreeNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem lowThree_sign (w : Int) (hw : 32≤w) :
    0<narrowResidual (2*w) w 8 (12*w*w+144*w-9) := by
  let x := (w-32).toNat
  have hx : (x:Int)=w-32 := Int.toNat_of_nonneg (by omega)
  have H := lowThree_shift x
  dsimp only at H
  have hw' : (x:Int)+32=w := by omega
  rw [hw'] at H
  have hn : 0<lowThreeNat x := by unfold lowThreeNat; omega
  have hi : (0:Int)<(lowThreeNat x:Int) := by omega
  omega

def upperOneNat (x : Nat) : Nat := ((((((((((2560*x+94336)*x+1556160)*x+15121920)*x+95780208)*x+412737312)*x+1223807628)*x+2461197576)*x+3205498976)*x+2433662336)*x+814130688)
theorem upperOne_shift (x : Nat) :
    let w := (x:Int)+4
    narrowResidual (2*w) w (2*w) (16*w*w-9)=(upperOneNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,upperOneNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem upperOne_sign (w : Int) (hw : 4≤w) :
    0<narrowResidual (2*w) w (2*w) (16*w*w-9) := by
  let x := (w-4).toNat
  have hx : (x:Int)=w-4 := Int.toNat_of_nonneg (by omega)
  have H := upperOne_shift x
  dsimp only at H
  have hw' : (x:Int)+4=w := by omega
  rw [hw'] at H
  have hn : 0<upperOneNat x := by unfold upperOneNat; omega
  have hi : (0:Int)<(upperOneNat x:Int) := by omega
  omega

def upperFourNat (x : Nat) : Nat := ((((((((((294912*x+5833728)*x+52207104)*x+278381568)*x+979488240)*x+2375957568)*x+4023132948)*x+4694058720)*x+3610305504)*x+1652080512)*x+341377344)
theorem upperFour_shift (x : Nat) :
    let w := (x:Int)+2
    narrowResidual (2*w) w (16*w) (12*w*w-9)=(upperFourNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,upperFourNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem upperFour_sign (w : Int) (hw : 2≤w) :
    0<narrowResidual (2*w) w (16*w) (12*w*w-9) := by
  let x := (w-2).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have H := upperFour_shift x
  dsimp only at H
  have hw' : (x:Int)+2=w := by omega
  rw [hw'] at H
  have hn : 0<upperFourNat x := by unfold upperFourNat; omega
  have hi : (0:Int)<(upperFourNat x:Int) := by omega
  omega

def smallInverse (d e : Nat) : Int :=
  match d with
  | 1 => (#[7023,12207,14383,2351,11055,12079,14127,1839,10031,10031,10031,10031,10031,10031,10031] : Array Int).getD e 0
  | 2 => (#[15039,1215,6335,191,4287,12479,12479,12479,12479,12479,12479,12479,12479,12479,12479] : Array Int).getD e 0
  | 3 => (#[4863,8959,767,767,767,767,767,767,767,767,767,767,767,767,767] : Array Int).getD e 0
  | _ => 0

def smallRoot (d e : Nat) : Int :=
  match d with
  | 1 => (#[15483,12303,5431,6023,15399,1383,6119,15591,1767,6887,743,4839,13031,13031,13031] : Array Int).getD e 0
  | 2 => (#[3135,7623,8407,9975,13111,2999,15543,7863,8887,10935,15031,6839,6839,6839,6839] : Array Int).getD e 0
  | 3 => (#[7303,7703,8503,10103,13303,3319,16119,8951,10999,15095,6903,6903,6903,6903,6903] : Array Int).getD e 0
  | _ => 0

def boundarySlope (w z : Int) : Int :=
  narrowResidual (2*w) w z 1-narrowResidual (2*w) w z 0

theorem boundary_affine (w z j : Int) :
    narrowResidual (2*w) w z j =
      narrowResidual (2*w) w z 0+boundarySlope w z*j := by
  simp only [boundarySlope,narrowResidual,narrowA,narrowB,narrowC,narrowD,
    Int.pow_succ,Int.pow_zero]
  grind only

set_option maxRecDepth 100000 in
set_option maxHeartbeats 8000000 in
theorem small_root_certificate :
    ∀ (d : Fin 4) (e : Fin 15), 1≤d.val →
    let w := (2:Int)^d.val
    let z := (2:Int)^e.val
    let v := smallInverse d.val e.val
    let r := smallRoot d.val e.val
    (v*boundarySlope w z)%16384=1 ∧
    narrowResidual (2*w) w z r%16384=0 ∧
    0≤r ∧ r<16384 ∧ (r=0 ∨ 21*w*w≤r) := by decide

theorem boundary_mod_congr (w z z' j m : Int) (hz : z%m=z'%m) :
    narrowResidual (2*w) w z j%m=narrowResidual (2*w) w z' j%m := by
  simp [narrowResidual,narrowA,narrowB,narrowC,narrowD,Int.pow_succ,
    Int.add_emod,Int.sub_emod,Int.mul_emod,hz]

private theorem affine_unit_residue (F B j v r M : Int)
    (hp : (F+B*j)%M=0) (hr : (F+B*r)%M=0)
    (hu : (v*B)%M=1) : (j-r)%M=0 := by
  have hd : (B*(j-r))%M=0 := by
    have hid : B*(j-r)=(F+B*j)-(F+B*r) := by grind only
    rw [hid]
    simp [Int.sub_emod,hp,hr]
  have hm : (v*B*(j-r))%M=0 := by
    rw [Int.mul_assoc,Int.mul_emod,hd]
    simp
  have he : (v*B*(j-r))%M=(j-r)%M := by rw [Int.mul_emod,hu]; simp
  omega

theorem small_boundary_nonzero (d e : Nat) (hd : 1≤d) (hd3 : d≤3)
    (j : Int) (hj : 0<j) (hju : j<21*(2:Int)^d*(2:Int)^d) :
    narrowResidual (2*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  let w : Int := 2^d
  let z : Int := 2^(min e 14)
  have hw : w≤8 := pow_mono d 3 hd3
  have hwpos : 0<w := Int.pow_pos (by decide)
  have hw2 : w*w≤64 := Int.mul_le_mul hw hw (by omega) (by decide)
  have hnorm : 21*w*w=21*(w*w) := Int.mul_assoc _ _ _
  change j<21*w*w at hju
  have hz := power_mod_cap e 14
  have H := boundary_mod_congr w ((2:Int)^e) z j 16384 hz
  rw [hp] at H
  have cert := small_root_certificate ⟨d,by omega⟩ ⟨min e 14,by omega⟩ hd
  dsimp only at cert
  change
    (smallInverse d (min e 14)*boundarySlope w z)%16384=1 ∧
    narrowResidual (2*w) w z (smallRoot d (min e 14))%16384=0 ∧
    0≤smallRoot d (min e 14) ∧ smallRoot d (min e 14)<16384 ∧
    (smallRoot d (min e 14)=0 ∨ 21*w*w≤smallRoot d (min e 14)) at cert
  have hr := cert.2.1
  rw [boundary_affine] at H hr
  have hm := affine_unit_residue _ _ j (smallInverse d (min e 14))
    (smallRoot d (min e 14)) 16384 H.symm hr cert.1
  have heq := small_mod_unique (smallRoot d (min e 14)) j 16384
    cert.2.2.1 cert.2.2.2.1 (by omega) (by omega) hm
  rcases cert.2.2.2.2 with h | h <;> omega

theorem boundary_factor (q z n : Int) :
    let w := 16*q
    let j := 16*n-9
    narrowResidual (2*w) w z j =
      18*w*z-(j+9)+(44+36*z)*w*w+64*w*w*
      (64*q*q*narrowA w z j+2*q*narrowB w z j+
        6*j*q*z-3*n+1-384*q*q*z*z+144*q*q-21*q*z) := by
  dsimp
  simp only [narrowResidual,narrowC,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem boundary_congruence (d e : Nat) (hd : 4≤d) (j : Int)
    (hp : narrowResidual (2*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e-(44+36*(2:Int)^e)*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0 := by
  obtain ⟨q,hq⟩ := Nat.exists_eq_add_of_le hd
  have hw : (2:Int)^d=16*(2:Int)^q := by rw [hq,Int.pow_add]; rfl
  have hp' := hp
  rw [hw] at hp'
  have hm := congrArg (fun x:Int => x%16) hp'
  have hjm : (j+9)%16=0 := by
    simp [narrowResidual,narrowA,narrowB,narrowC,narrowD,
      Int.pow_succ,Int.add_emod,Int.sub_emod,Int.mul_emod] at hm ⊢
    omega
  obtain ⟨n,hn⟩ := Int.dvd_of_emod_eq_zero hjm
  have hj' : j=16*n-9 := by omega
  rw [hj'] at hp' ⊢
  have H := boundary_factor ((2:Int)^q) ((2:Int)^e) n
  dsimp only at H
  rw [hp'] at H
  rw [hw]
  apply Int.emod_eq_zero_of_dvd
  refine ⟨64*(2:Int)^q*(2:Int)^q*narrowA (16*(2:Int)^q) ((2:Int)^e) (16*n-9)+
    2*(2:Int)^q*narrowB (16*(2:Int)^q) ((2:Int)^e) (16*n-9)+
    6*(16*n-9)*(2:Int)^q*(2:Int)^e-3*n+1-
    384*(2:Int)^q*(2:Int)^q*(2:Int)^e*(2:Int)^e+
    144*(2:Int)^q*(2:Int)^q-21*(2:Int)^q*(2:Int)^e,?_⟩
  omega

theorem boundary_quotient_certificate :
    ∀ (f : Fin 7) (n : Fin 24), 1≤f.val → 1≤n.val →
      (n.val:Int)%64=(18*(2:Int)^f.val+44)%64 →
      (f.val=1 ∧ n.val=16) ∨ (f.val=4 ∧ n.val=12) := by decide

theorem high_boundary_patterns (d e : Nat) (J : Int)
    (hJ : 0<J) (hJu : J<24*(2:Int)^d*(2:Int)^d)
    (hmod : (J-18*(2:Int)^d*(2:Int)^e-44*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0) :
    (e=d+1 ∧ J=16*(2:Int)^d*(2:Int)^d) ∨
    (e=d+4 ∧ J=12*(2:Int)^d*(2:Int)^d) := by
  let w : Int := 2^d
  let z : Int := 2^e
  have hw : 0<w := Int.pow_pos (by decide)
  have hz : 0<z := Int.pow_pos (by decide)
  have hw2 : 0<w*w := Int.mul_pos hw hw
  change J<24*w*w at hJu
  change (J-18*w*z-44*w*w)%(64*w*w)=0 at hmod
  have hn18 : 18*w*w=18*(w*w) := Int.mul_assoc _ _ _
  have hn24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
  have hn44 : 44*w*w=44*(w*w) := Int.mul_assoc _ _ _
  have hn64 : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
  by_cases hed : e≤d
  · have hzw : z≤w := pow_mono e d hed
    have hmul := Int.mul_le_mul_of_nonneg_left hzw (show 0≤18*w by omega)
    have hmpos : 0<18*w*z := Int.mul_pos (by omega) hz
    have hm : (J-(18*w*z+44*w*w))%(64*w*w)=0 := by
      have h : J-(18*w*z+44*w*w)=J-18*w*z-44*w*w := by omega
      rw [h]; exact hmod
    have H := small_mod_unique (18*w*z+44*w*w) J (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · obtain ⟨f,hf⟩ := Nat.exists_eq_add_of_le (show d≤e by omega)
    have hfpos : 1≤f := by omega
    have hzw : z=w*(2:Int)^f := by dsimp [w,z]; rw [hf,Int.pow_add]
    obtain ⟨v,hv⟩ := Int.dvd_of_emod_eq_zero hmod
    let n : Int := 18*(2:Int)^f+44+64*v
    have he : J=(w*w)*n := by rw [hzw] at hv; dsimp [n]; grind only
    have hn := quotient_range (w*w) J n 24 hw2 hJ (by omega) he
    let nn := n.toNat
    have hnn : (nn:Int)=n := Int.toNat_of_nonneg (by omega)
    have hm : n%64=(18*(2:Int)^f+44)%64 := by dsimp [n]; simp [Int.add_emod]
    have hcap := power_mod_cap f 6
    change (2:Int)^f%64=(2:Int)^(min f 6)%64 at hcap
    have hm' : (nn:Int)%64=(18*(2:Int)^(min f 6)+44)%64 := by
      rw [hnn,hm]
      simp [Int.add_emod,Int.mul_emod,hcap]
    have hfm : 1≤min f 6 := by omega
    have hnnlo : 1≤nn := by omega
    have hnnhi : nn<24 := by omega
    have cert := boundary_quotient_certificate ⟨min f 6,by omega⟩ ⟨nn,hnnhi⟩ hfm hnnlo hm'
    rcases cert with H | H
    · have hf1 : f=1 := by simp only at H; omega
      have hn16 : n=16 := by simp only at H; omega
      left
      constructor
      · omega
      · rw [hn16] at he; change J=16*w*w; grind only
    · have hf4 : f=4 := by simp only at H; omega
      have hn12 : n=12 := by simp only at H; omega
      right
      constructor
      · omega
      · rw [hn12] at he; change J=12*w*w; grind only

theorem high_boundary_congruence (d e : Nat) (hd : 4≤d) (he : 4≤e) (j : Int)
    (hp : narrowResidual (2*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e-44*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0 := by
  have H := boundary_congruence d e hd j hp
  obtain ⟨t,ht⟩ := Nat.exists_eq_add_of_le he
  have hz : (2:Int)^e=16*(2:Int)^t := by rw [ht,Int.pow_add]; rfl
  have hid : j+9-18*(2:Int)^d*(2:Int)^e-(44+36*(2:Int)^e)*(2:Int)^d*(2:Int)^d =
      (j+9-18*(2:Int)^d*(2:Int)^e-44*(2:Int)^d*(2:Int)^d)-
      (64*(2:Int)^d*(2:Int)^d)*(9*(2:Int)^t) := by rw [hz]; grind only
  rw [hid] at H
  simpa [Int.sub_emod] using H

theorem high_boundary_nonzero (d e : Nat) (hd : 4≤d) (he : 4≤e)
    (j : Int) (hj : 0<j+9) (hju : j+9<24*(2:Int)^d*(2:Int)^d) :
    narrowResidual (2*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  have hm := high_boundary_congruence d e hd he j hp
  have hw16 : (16:Int)≤(2:Int)^d := pow_mono 4 d hd
  rcases high_boundary_patterns d e (j+9) hj hju hm with H | H
  · obtain ⟨heq,hj'⟩ := H
    have jeq : j=16*(2:Int)^d*(2:Int)^d-9 := by omega
    have hz : (2:Int)^e=2*(2:Int)^d := by rw [heq,Int.pow_add]; grind only
    rw [hz,jeq] at hp
    have hs := upperOne_sign ((2:Int)^d) (by omega)
    omega
  · obtain ⟨heq,hj'⟩ := H
    have jeq : j=12*(2:Int)^d*(2:Int)^d-9 := by omega
    have hz : (2:Int)^e=16*(2:Int)^d := by rw [heq,Int.pow_add]; grind only
    rw [hz,jeq] at hp
    have hs := upperFour_sign ((2:Int)^d) (by omega)
    omega

theorem lowTwo_sixteen_value :
    narrowResidual 32 16 4 119=110656713129984 := by decide

theorem lowThree_sixteen_value :
    narrowResidual 32 16 8 5367= -138588845801472 := by decide

theorem low_boundary_nonzero (d e : Nat) (hd : 4≤d) (he : e≤3)
    (j : Int) (hj : 0<j+9) (hju : j+9<24*(2:Int)^d*(2:Int)^d) :
    narrowResidual (2*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  have H := boundary_congruence d e hd j hp
  let w : Int := 2^d
  have hw : (16:Int)≤w := pow_mono 4 d hd
  have hw2 : (256:Int)≤w*w := Int.mul_le_mul hw hw (by decide) (by omega)
  have h144 := Int.mul_le_mul_of_nonneg_left hw (show 0≤9*w by omega)
  have hn144 : 9*w*16=144*w := by grind only
  have hn9 : 9*w*w=9*(w*w) := Int.mul_assoc _ _ _
  have hn12 : 12*w*w=12*(w*w) := Int.mul_assoc _ _ _
  have hn16 : 16*w*w=16*(w*w) := Int.mul_assoc _ _ _
  have hn24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
  have hn52 : 52*w*w=52*(w*w) := Int.mul_assoc _ _ _
  have hn60 : 60*w*w=60*(w*w) := Int.mul_assoc _ _ _
  have hn64 : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
  change j+9<24*w*w at hju
  have cases : e=0 ∨ e=1 ∨ e=2 ∨ e=3 := by omega
  rcases cases with he0 | he1 | he2 | he3
  · subst e
    change (j+9-18*w*1-(44+36*1)*w*w)%(64*w*w)=0 at H
    have hid : j+9-18*w*1-(44+36*1)*w*w=(j+9-(18*w+16*w*w))-(64*w*w) := by grind only
    rw [hid] at H
    have hm : (j+9-(18*w+16*w*w))%(64*w*w)=0 := by simpa [Int.sub_emod] using H
    have heq := small_mod_unique (18*w+16*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    have jeq : j=16*w*w+18*w-9 := by omega
    change narrowResidual (2*w) w 1 j=0 at hp
    rw [jeq] at hp
    have hn := lowZero_sign w (by omega)
    omega
  · subst e
    change (j+9-18*w*2-(44+36*2)*w*w)%(64*w*w)=0 at H
    have hid : j+9-18*w*2-(44+36*2)*w*w=(j+9-(36*w+52*w*w))-(64*w*w) := by grind only
    rw [hid] at H
    have hm : (j+9-(36*w+52*w*w))%(64*w*w)=0 := by simpa [Int.sub_emod] using H
    have heq := small_mod_unique (36*w+52*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · subst e
    change (j+9-18*w*4-(44+36*4)*w*w)%(64*w*w)=0 at H
    by_cases hd5 : 5≤d
    · have hw32 : (32:Int)≤w := pow_mono 5 d hd5
      have h96 := Int.mul_le_mul_of_nonneg_left hw32 (show 0≤3*w by omega)
      have hn96 : 3*w*32=96*w := by grind only
      have hn3 : 3*w*w=3*(w*w) := Int.mul_assoc _ _ _
      have hid : j+9-18*w*4-(44+36*4)*w*w=(j+9-(72*w+60*w*w))-(64*w*w)*2 := by grind only
      rw [hid] at H
      have hm : (j+9-(72*w+60*w*w))%(64*w*w)=0 := by simpa [Int.sub_emod] using H
      have heq := small_mod_unique (72*w+60*w*w) (j+9) (64*w*w)
        (by omega) (by omega) (by omega) (by omega) hm
      omega
    · have hdeq : d=4 := by omega
      have hweq : w=16 := by dsimp [w]; rw [hdeq]; rfl
      rw [hweq] at H hju
      have jeq : j=119 := by omega
      rw [hdeq,jeq] at hp
      change narrowResidual 32 16 4 119=0 at hp
      have hn := lowTwo_sixteen_value
      omega
  · subst e
    change (j+9-18*w*8-(44+36*8)*w*w)%(64*w*w)=0 at H
    have hid : j+9-18*w*8-(44+36*8)*w*w=(j+9-(144*w+12*w*w))-(64*w*w)*5 := by grind only
    rw [hid] at H
    have hm : (j+9-(144*w+12*w*w))%(64*w*w)=0 := by simpa [Int.sub_emod] using H
    have heq := small_mod_unique (144*w+12*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    by_cases hd5 : 5≤d
    · have hw32 : (32:Int)≤w := pow_mono 5 d hd5
      have jeq : j=12*w*w+144*w-9 := by omega
      change narrowResidual (2*w) w 8 j=0 at hp
      rw [jeq] at hp
      have hn := lowThree_sign w hw32
      omega
    · have hdeq : d=4 := by omega
      have hweq : w=16 := by dsimp [w]; rw [hdeq]; rfl
      rw [hweq] at heq
      have jeq : j=5367 := by omega
      rw [hdeq,jeq] at hp
      change narrowResidual 32 16 8 5367=0 at hp
      have hn := lowThree_sixteen_value
      omega

theorem boundary_one_residual_nonzero (d e : Nat) (hd : 1≤d)
    (j : Int) (hj : 0<j) (hju : j<21*(2:Int)^d*(2:Int)^d) :
    narrowResidual (2*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  by_cases hd3 : d≤3
  · exact small_boundary_nonzero d e hd hd3 j hj hju
  have hw : (2:Int)≤(2:Int)^d := pow_mono 1 d hd
  have hw2 : (4:Int)≤(2:Int)^d*(2:Int)^d := Int.mul_le_mul hw hw (by decide) (by omega)
  have hnorm21 : 21*(2:Int)^d*(2:Int)^d=21*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
  have hnorm24 : 24*(2:Int)^d*(2:Int)^d=24*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
  have hJu : j+9<24*(2:Int)^d*(2:Int)^d := by omega
  by_cases he3 : e≤3
  · exact low_boundary_nonzero d e (by omega) he3 j (by omega) hJu
  · exact high_boundary_nonzero d e (by omega) (by omega) j (by omega) hJu

theorem boundary_one_scaled_no_collision (d e : Nat) (a : Int)
    (hd : 1≤d) (ha : 0<a)
    (hab : a<3*(2*(2:Int)^d*(2:Int)^d)*(2*(2:Int)^d*(2:Int)^d)) :
    collisionPoly (2*(2:Int)^d*(2:Int)^d*(2:Int)^e) 2
      (2*(2:Int)^d) (2*(2:Int)^d*(2:Int)^d) a≠0 := by
  intro hf
  obtain ⟨j,hj,hju,_,hp⟩ := narrow_collision_deficit
    (2*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) a
    (Int.mul_pos (by decide) (Int.pow_pos (by decide))) ha hab hf
  exact boundary_one_residual_nonzero d e hd j hj hju hp

theorem boundary_one_polynomial_no_collision (r c k : Nat) (a : Int)
    (hck : c<k) (hbal : 2*c=k+1) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have hc : c=d+1 := by omega
  have hs : (2:Int)^c=2*(2:Int)^d := by rw [hc,Int.pow_add]; grind only
  have ht : (2:Int)^k=2*(2:Int)^d*(2:Int)^d := by rw [hd,Int.pow_add,hs]
  have hu : (2:Int)^(r+1)=2*(2:Int)^d*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [hs,ht,hu]
  exact boundary_one_scaled_no_collision d e a (by omega) ha hab

theorem boundary_one_numerator_no_collision (r c k a : Nat)
    (hc : 1≤c) (hck : c<k) (hbal : 2*c=k+1)
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
  exact boundary_one_polynomial_no_collision r c k a hck hbal hr (by omega) habi hp

theorem one_inner_boundary_one_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c≤k ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := one_inner_boundary_two_collision_region r c k a hc hk ha hab he
  have hb : 2*c≤k := by
    by_cases hb : 2*c≤k
    · exact hb
    exact False.elim (boundary_one_numerator_no_collision r c k a hc H.1 (by omega) ha hab he)
  exact ⟨H.1,hb,H.2.2⟩

theorem one_inner_boundary_one_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c≤k ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_boundary_one_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1


#print axioms pow_mono
#print axioms small_mod_unique
#print axioms quotient_range
#print axioms lowZero_shift
#print axioms lowZero_sign
#print axioms lowThree_shift
#print axioms lowThree_sign
#print axioms upperOne_shift
#print axioms upperOne_sign
#print axioms upperFour_shift
#print axioms upperFour_sign
#print axioms boundary_affine
#print axioms small_root_certificate
#print axioms boundary_mod_congr
#print axioms affine_unit_residue
#print axioms small_boundary_nonzero
#print axioms boundary_factor
#print axioms boundary_congruence
#print axioms boundary_quotient_certificate
#print axioms high_boundary_patterns
#print axioms high_boundary_congruence
#print axioms high_boundary_nonzero
#print axioms lowTwo_sixteen_value
#print axioms lowThree_sixteen_value
#print axioms low_boundary_nonzero
#print axioms boundary_one_residual_nonzero
#print axioms boundary_one_scaled_no_collision
#print axioms boundary_one_polynomial_no_collision
#print axioms boundary_one_numerator_no_collision
#print axioms one_inner_boundary_one_collision_region
#print axioms one_inner_boundary_one_value_collision_region
end Sounio.ZDScalarBoundaryOne

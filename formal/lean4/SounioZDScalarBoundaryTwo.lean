import SounioZDScalarHalfSeparator

/-! Exclusion of the boundary 2c=k+2 for the explicit h=1 small-count model.
No assertion of global ResidualSeparation or bibliographic priority. -/
namespace Sounio.ZDScalarBoundaryTwo
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


def equalNat (x : Nat) : Nat := ((((((((((38912*x+713728)*x+5856000)*x+28275456)*x+88873152)*x+189734784)*x+278139360)*x+275811144)*x+176495728)*x+65511008)*x+10635072)

theorem equal_shift (x : Nat) :
    let w := (x:Int)+2
    narrowResidual (4*w) w w (2*w*w-9)=(equalNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,equalNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem equal_sign (w : Int) (hw : 2≤w) :
    0<narrowResidual (4*w) w w (2*w*w-9) := by
  let x := (w-2).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have H := equal_shift x
  dsimp only at H
  have hw' : (x:Int)+2=w := by omega
  rw [hw'] at H
  have hn : 0<equalNat x := by unfold equalNat; omega
  have hi : (0:Int)<(equalNat x:Int) := by omega
  omega

def upperOneNat (x : Nat) : Nat := ((((((((((8192*x+1181696)*x+75855360)*x+2845522944)*x+68803520640)*x+1113771939072)*x+12106599957624)*x+85785085681296)*x+366382426025152)*x+779221523752960)*x+415507023052800)

theorem upperOne_shift (x : Nat) :
    let w := (x:Int)+16
    narrowResidual (4*w) w (2*w) (20*w*w-9)=(upperOneNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,upperOneNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem upperOne_sign (w : Int) (hw : 16≤w) :
    0<narrowResidual (4*w) w (2*w) (20*w*w-9) := by
  let x := (w-16).toNat
  have hx : (x:Int)=w-16 := Int.toNat_of_nonneg (by omega)
  have H := upperOne_shift x
  dsimp only at H
  have hw' : (x:Int)+16=w := by omega
  rw [hw'] at H
  have hn : 0<upperOneNat x := by unfold upperOneNat; omega
  have hi : (0:Int)<(upperOneNat x:Int) := by omega
  omega

def upperFourNat (x : Nat) : Nat := ((((((((((2621440*x+51396608)*x+457912320)*x+2442166272)*x+8634564480)*x+21141201408)*x+36281855688)*x+43060026048)*x+33789696128)*x+15813835264)*x+3348159360)

theorem upperFour_shift (x : Nat) :
    let w := (x:Int)+2
    narrowResidual (4*w) w (16*w) (16*w*w-9)=(upperFourNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,upperFourNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem upperFour_sign (w : Int) (hw : 2≤w) :
    0<narrowResidual (4*w) w (16*w) (16*w*w-9) := by
  let x := (w-2).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have H := upperFour_shift x
  dsimp only at H
  have hw' : (x:Int)+2=w := by omega
  rw [hw'] at H
  have hn : 0<upperFourNat x := by unfold upperFourNat; omega
  have hi : (0:Int)<(upperFourNat x:Int) := by omega
  omega

def lowFourNat (x : Nat) : Nat := ((((((((94208*x+14439424)*x+941604992)*x+34376673280)*x+772225426616)*x+10965769341856)*x+96353998986048)*x+479813370808320)*x+1038093168623616)

theorem lowFour_shift (x : Nat) :
    let w := (x:Int)+16
    (-(narrowResidual (4*w) w 4 (16*w*w+72*w-9)))=(lowFourNat x:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,lowFourNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem lowFour_sign (w : Int) (hw : 16≤w) :
    narrowResidual (4*w) w 4 (16*w*w+72*w-9)<0 := by
  let x := (w-16).toNat
  have hx : (x:Int)=w-16 := Int.toNat_of_nonneg (by omega)
  have H := lowFour_shift x
  dsimp only at H
  have hw' : (x:Int)+16=w := by omega
  rw [hw'] at H
  have hn : 0<lowFourNat x := by unfold lowFourNat; omega
  have hi : (0:Int)<(lowFourNat x:Int) := by omega
  omega

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 8000000 in
theorem small_boundary_certificate :
    ∀ (d : Fin 3) (e : Fin 13) (j : Fin 336),
      1≤d.val → 1≤j.val → j.val<21*2^d.val*2^d.val →
      narrowResidual (4*(2:Int)^d.val) ((2:Int)^d.val)
        ((2:Int)^e.val) j.val %4096≠0 := by decide

theorem boundary_mod_congr (w z z' j m : Int) (hz : z%m=z'%m) :
    narrowResidual (4*w) w z j%m=narrowResidual (4*w) w z' j%m := by
  simp [narrowResidual,narrowA,narrowB,narrowC,narrowD,Int.pow_succ,
    Int.add_emod,Int.sub_emod,Int.mul_emod,hz]

theorem small_boundary_nonzero (d e : Nat) (hd : 1≤d) (hd2 : d≤2)
    (j : Int) (hj : 0<j) (hju : j<21*(2:Int)^d*(2:Int)^d) :
    narrowResidual (4*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  have hw : (2:Int)^d≤4 := pow_mono d 2 hd2
  have hwpos : (0:Int)<(2:Int)^d := Int.pow_pos (by decide)
  have hww : (2:Int)^d*(2:Int)^d≤16 :=
    Int.mul_le_mul hw hw (by omega) (by decide)
  have hnorm : 21*(2:Int)^d*(2:Int)^d=21*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
  let jj := j.toNat
  have hjj : (jj:Int)=j := Int.toNat_of_nonneg (by omega)
  have hjupper : jj<336 := by omega
  have hjbound : jj<21*2^d*2^d := by
    have hcast : ((21*2^d*2^d:Nat):Int)=21*(2:Int)^d*(2:Int)^d := by
      simp only [Int.natCast_mul,Int.natCast_pow]; rfl
    omega
  have hz := power_mod_cap e 12
  have hm := boundary_mod_congr ((2:Int)^d) ((2:Int)^e) ((2:Int)^(min e 12)) j 4096 hz
  have hjlower : 1≤jj := by omega
  have cert := small_boundary_certificate ⟨d,by omega⟩ ⟨min e 12,by omega⟩
    ⟨jj,hjupper⟩ hd hjlower hjbound
  simp only at cert
  rw [hjj] at cert
  rw [hp] at hm
  exact cert hm.symm

theorem boundary_factor (q z n : Int) :
    let w := 8*q
    let j := 8*n-9
    narrowResidual (4*w) w z j =
      18*w*z-(j+9)+(48+8*z)*w*w+64*w*w*
      (4*w*w*narrowA w z j+w*narrowB w z j+
        12*j*q*z-6*n+6-6*w*w*z*z+72*q*q-21*q*z+z) := by
  dsimp
  simp only [narrowResidual,narrowC,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem boundary_congruence (d e : Nat) (hd : 3≤d) (j : Int)
    (hp : narrowResidual (4*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e-(48+8*(2:Int)^e)*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0 := by
  obtain ⟨q,hq⟩ := Nat.exists_eq_add_of_le hd
  have hw : (2:Int)^d=8*(2:Int)^q := by rw [hq,Int.pow_add]; rfl
  have hp' := hp
  rw [hw] at hp'
  have hm := congrArg (fun x:Int => x%8) hp'
  have hjm : (j+9)%8=0 := by
    simp [narrowResidual,narrowA,narrowB,narrowC,narrowD,
      Int.pow_succ,Int.add_emod,Int.sub_emod,Int.mul_emod] at hm ⊢
    omega
  obtain ⟨n,hn⟩ := Int.dvd_of_emod_eq_zero hjm
  have hj' : j=8*n-9 := by omega
  rw [hj'] at hp' ⊢
  have H := boundary_factor ((2:Int)^q) ((2:Int)^e) n
  dsimp only at H
  rw [hp'] at H
  rw [hw]
  apply Int.emod_eq_zero_of_dvd
  refine ⟨4*(8*(2:Int)^q)*(8*(2:Int)^q)*narrowA (8*(2:Int)^q) ((2:Int)^e) (8*n-9)+
    (8*(2:Int)^q)*narrowB (8*(2:Int)^q) ((2:Int)^e) (8*n-9)+
    12*(8*n-9)*(2:Int)^q*(2:Int)^e-6*n+6-
    6*(8*(2:Int)^q)*(8*(2:Int)^q)*(2:Int)^e*(2:Int)^e+
    72*(2:Int)^q*(2:Int)^q-21*(2:Int)^q*(2:Int)^e+(2:Int)^e,?_⟩
  omega

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


theorem boundary_quotient_certificate :
    ∀ (f : Fin 7) (n : Fin 24), 1≤f.val → 1≤n.val →
      (n.val:Int)%64=(18*(2:Int)^f.val+48)%64 →
      (f.val=1 ∧ n.val=20) ∨ (f.val=4 ∧ n.val=16) := by decide

theorem high_boundary_patterns (d e : Nat) (J : Int)
    (hJ : 0<J) (hJu : J<24*(2:Int)^d*(2:Int)^d)
    (hmod : (J-18*(2:Int)^d*(2:Int)^e-48*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0) :
    (e=d ∧ J=2*(2:Int)^d*(2:Int)^d) ∨
    (e=d+1 ∧ J=20*(2:Int)^d*(2:Int)^d) ∨
    (e=d+4 ∧ J=16*(2:Int)^d*(2:Int)^d) := by
  let w : Int := 2^d
  let z : Int := 2^e
  have hw : 0<w := Int.pow_pos (by decide)
  have hz : 0<z := Int.pow_pos (by decide)
  have hw2 : 0<w*w := Int.mul_pos hw hw
  change J<24*w*w at hJu
  change (J-18*w*z-48*w*w)%(64*w*w)=0 at hmod
  have hn2 : 2*w*w=2*(w*w) := Int.mul_assoc _ _ _
  have hn24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
  have hn48 : 48*w*w=48*(w*w) := Int.mul_assoc _ _ _
  have hn64 : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
  by_cases heq : e=d
  · have hz' : z=w := by dsimp [w,z]; rw [heq]
    rw [hz'] at hmod
    have hid : J-18*w*w-48*w*w=(J-2*w*w)-(64*w*w) := by grind only
    rw [hid] at hmod
    have hm : (J-2*w*w)%(64*w*w)=0 := by simpa [Int.sub_emod] using hmod
    have H := small_mod_unique (2*w*w) J (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    exact Or.inl ⟨heq,H⟩
  by_cases hed : e<d
  · have hpow : (2:Int)^(e+1)≤(2:Int)^d := pow_mono (e+1) d (by omega)
    have hz2 : 2*z≤w := by
      dsimp [w,z]
      rw [Int.pow_succ] at hpow
      omega
    have hmul := Int.mul_le_mul_of_nonneg_left hz2 (show 0≤9*w by omega)
    have hnorm : 9*w*(2*z)=18*w*z := by grind only
    have hnorm' : 9*w*w=9*(w*w) := Int.mul_assoc _ _ _
    have hmpos : 0<18*w*z := Int.mul_pos (by omega) hz
    have hm : (J-(18*w*z+48*w*w))%(64*w*w)=0 := by
      have h : J-(18*w*z+48*w*w)=J-18*w*z-48*w*w := by omega
      rw [h]; exact hmod
    have H := small_mod_unique (18*w*z+48*w*w) J (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · obtain ⟨f,hf⟩ := Nat.exists_eq_add_of_le (show d≤e by omega)
    have hfpos : 1≤f := by omega
    have hzw : z=w*(2:Int)^f := by dsimp [w,z]; rw [hf,Int.pow_add]
    obtain ⟨v,hv⟩ := Int.dvd_of_emod_eq_zero hmod
    let n : Int := 18*(2:Int)^f+48+64*v
    have he : J=(w*w)*n := by rw [hzw] at hv; dsimp [n]; grind only
    have hn := quotient_range (w*w) J n 24 hw2 hJ (by omega) he
    let nn := n.toNat
    have hnn : (nn:Int)=n := Int.toNat_of_nonneg (by omega)
    have hm : n%64=(18*(2:Int)^f+48)%64 := by dsimp [n]; simp [Int.add_emod]
    have hcap := power_mod_cap f 6
    change (2:Int)^f%64=(2:Int)^(min f 6)%64 at hcap
    have hm' : (nn:Int)%64=(18*(2:Int)^(min f 6)+48)%64 := by
      rw [hnn,hm]
      simp [Int.add_emod,Int.mul_emod,hcap]
    have hfm : 1≤min f 6 := by omega
    have hnnlo : 1≤nn := by omega
    have hnnhi : nn<24 := by omega
    have cert := boundary_quotient_certificate ⟨min f 6,by omega⟩ ⟨nn,hnnhi⟩ hfm hnnlo hm'
    rcases cert with H | H
    · have hf1 : f=1 := by simp only at H; omega
      have hn20 : n=20 := by simp only at H; omega
      right; left
      constructor
      · omega
      · rw [hn20] at he; change J=20*w*w; grind only
    · have hf4 : f=4 := by simp only at H; omega
      have hn16 : n=16 := by simp only at H; omega
      right; right
      constructor
      · omega
      · rw [hn16] at he; change J=16*w*w; grind only

theorem high_boundary_congruence (d e : Nat) (hd : 3≤d) (he : 3≤e) (j : Int)
    (hp : narrowResidual (4*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e-48*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0 := by
  have H := boundary_congruence d e hd j hp
  obtain ⟨t,ht⟩ := Nat.exists_eq_add_of_le he
  have hz : (2:Int)^e=8*(2:Int)^t := by rw [ht,Int.pow_add]; rfl
  have hid : j+9-18*(2:Int)^d*(2:Int)^e-(48+8*(2:Int)^e)*(2:Int)^d*(2:Int)^d =
      (j+9-18*(2:Int)^d*(2:Int)^e-48*(2:Int)^d*(2:Int)^d)-
      (64*(2:Int)^d*(2:Int)^d)*(2:Int)^t := by rw [hz]; grind only
  rw [hid] at H
  simpa [Int.sub_emod] using H

theorem high_boundary_nonzero (d e : Nat) (hd : 3≤d) (he : 3≤e)
    (j : Int) (hj : 0<j+9) (hju : j+9<24*(2:Int)^d*(2:Int)^d) :
    narrowResidual (4*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  have hm := high_boundary_congruence d e hd he j hp
  have hw8 : (8:Int)≤(2:Int)^d := pow_mono 3 d hd
  rcases high_boundary_patterns d e (j+9) hj hju hm with H | H | H
  · obtain ⟨heq,hj'⟩ := H
    have jeq : j=2*(2:Int)^d*(2:Int)^d-9 := by omega
    rw [heq,jeq] at hp
    have hs := equal_sign ((2:Int)^d) (by omega)
    omega
  · obtain ⟨heq,hj'⟩ := H
    have jeq : j=20*(2:Int)^d*(2:Int)^d-9 := by omega
    have hz : (2:Int)^e=2*(2:Int)^d := by rw [heq,Int.pow_add]; grind only
    rw [hz,jeq] at hp
    by_cases hd4 : 4≤d
    · have hw16 : (16:Int)≤(2:Int)^d := pow_mono 4 d hd4
      have hs := upperOne_sign ((2:Int)^d) hw16
      omega
    · have hdeq : d=3 := by omega
      subst d
      have hs : narrowResidual (4*(2:Int)^3) ((2:Int)^3) (2*(2:Int)^3)
        (20*(2:Int)^3*(2:Int)^3-9)≠0 := by decide
      exact hs hp
  · obtain ⟨heq,hj'⟩ := H
    have jeq : j=16*(2:Int)^d*(2:Int)^d-9 := by omega
    have hz : (2:Int)^e=16*(2:Int)^d := by rw [heq,Int.pow_add]; grind only
    rw [hz,jeq] at hp
    have hs := upperFour_sign ((2:Int)^d) (by omega)
    omega

theorem low_boundary_nonzero (d e : Nat) (hd : 3≤d) (he : e≤2)
    (j : Int) (hj : 0<j+9) (hju : j+9<24*(2:Int)^d*(2:Int)^d) :
    narrowResidual (4*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  have H := boundary_congruence d e hd j hp
  let w : Int := 2^d
  have hw : (8:Int)≤w := pow_mono 3 d hd
  have hw2 : (64:Int)≤w*w := Int.mul_le_mul hw hw (by decide) (by omega)
  have h72 := Int.mul_le_mul_of_nonneg_left hw (show 0≤9*w by omega)
  have hn72 : 9*w*8=72*w := by grind only
  have hn9 : 9*w*w=9*(w*w) := Int.mul_assoc _ _ _
  have hn16 : 16*w*w=16*(w*w) := Int.mul_assoc _ _ _
  have hn24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
  have hn56 : 56*w*w=56*(w*w) := Int.mul_assoc _ _ _
  have hn64 : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
  change j+9<24*w*w at hju
  have cases : e=0 ∨ e=1 ∨ e=2 := by omega
  rcases cases with he0 | he1 | he2
  · subst e
    change (j+9-18*w*1-(48+8*1)*w*w)%(64*w*w)=0 at H
    have hid : j+9-18*w*1-(48+8*1)*w*w=j+9-(18*w+56*w*w) := by grind only
    rw [hid] at H
    have heq := small_mod_unique (18*w+56*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) H
    omega
  · subst e
    change (j+9-18*w*2-(48+8*2)*w*w)%(64*w*w)=0 at H
    have hid : j+9-18*w*2-(48+8*2)*w*w=(j+9-36*w)-(64*w*w) := by grind only
    rw [hid] at H
    have hm : (j+9-36*w)%(64*w*w)=0 := by simpa [Int.sub_emod] using H
    have heq := small_mod_unique (36*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    have jeq : j=18*(2:Int)^d*(2:Int)^1-9 := by change j=18*w*2-9; grind only
    rw [jeq] at hp
    have hs : (2:Int)^(d+2)=4*(2:Int)^d := by rw [Int.pow_add]; grind only
    have hn := main_dyadic_nonzero (d+2) d 1 (by omega) (by omega) (by omega)
    rw [hs] at hn
    exact hn hp
  · subst e
    change (j+9-18*w*4-(48+8*4)*w*w)%(64*w*w)=0 at H
    have hid : j+9-18*w*4-(48+8*4)*w*w=(j+9-(72*w+16*w*w))-(64*w*w) := by grind only
    rw [hid] at H
    have hm : (j+9-(72*w+16*w*w))%(64*w*w)=0 := by simpa [Int.sub_emod] using H
    have heq := small_mod_unique (72*w+16*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    by_cases hd4 : 4≤d
    · have hw16 : (16:Int)≤w := pow_mono 4 d hd4
      have jeq : j=16*w*w+72*w-9 := by omega
      change narrowResidual (4*w) w 4 j=0 at hp
      rw [jeq] at hp
      have hn := lowFour_sign w hw16
      omega
    · have hdeq : d=3 := by omega
      have hweq : w=8 := by dsimp [w]; rw [hdeq]; rfl
      rw [hweq] at hju heq
      omega

theorem boundary_two_residual_nonzero (d e : Nat) (hd : 1≤d)
    (j : Int) (hj : 0<j) (hju : j<21*(2:Int)^d*(2:Int)^d) :
    narrowResidual (4*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  by_cases hd2 : d≤2
  · exact small_boundary_nonzero d e hd hd2 j hj hju
  have hw : (2:Int)≤(2:Int)^d := pow_mono 1 d hd
  have hw2 : (4:Int)≤(2:Int)^d*(2:Int)^d := Int.mul_le_mul hw hw (by decide) (by omega)
  have hnorm21 : 21*(2:Int)^d*(2:Int)^d=21*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
  have hnorm24 : 24*(2:Int)^d*(2:Int)^d=24*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
  have hJu : j+9<24*(2:Int)^d*(2:Int)^d := by omega
  by_cases he2 : e≤2
  · exact low_boundary_nonzero d e (by omega) he2 j (by omega) hJu
  · exact high_boundary_nonzero d e (by omega) (by omega) j (by omega) hJu

theorem boundary_two_scaled_no_collision (d e : Nat) (a : Int)
    (hd : 1≤d) (ha : 0<a)
    (hab : a<3*(4*(2:Int)^d*(2:Int)^d)*(4*(2:Int)^d*(2:Int)^d)) :
    collisionPoly (4*(2:Int)^d*(2:Int)^d*(2:Int)^e) 2
      (4*(2:Int)^d) (4*(2:Int)^d*(2:Int)^d) a≠0 := by
  intro hf
  obtain ⟨j,hj,hju,_,hp⟩ := narrow_collision_deficit
    (4*(2:Int)^d) ((2:Int)^d) ((2:Int)^e) a
    (Int.mul_pos (by decide) (Int.pow_pos (by decide))) ha hab hf
  exact boundary_two_residual_nonzero d e hd j hj hju hp

theorem boundary_two_polynomial_no_collision (r c k : Nat) (a : Int)
    (hck : c<k) (hbal : 2*c=k+2) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have hc : c=d+2 := by omega
  have hs : (2:Int)^c=4*(2:Int)^d := by rw [hc,Int.pow_add]; grind only
  have ht : (2:Int)^k=4*(2:Int)^d*(2:Int)^d := by rw [hd,Int.pow_add,hs]
  have hu : (2:Int)^(r+1)=4*(2:Int)^d*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [hs,ht,hu]
  exact boundary_two_scaled_no_collision d e a (by omega) ha hab

theorem boundary_two_numerator_no_collision (r c k a : Nat)
    (hc : 1≤c) (hck : c<k) (hbal : 2*c=k+2)
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
  exact boundary_two_polynomial_no_collision r c k a hck hbal hr (by omega) habi hp

theorem one_inner_boundary_two_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c≤k+1 ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := one_inner_half_collision_region r c k a hc hk ha hab he
  have hb : 2*c≤k+1 := by
    by_cases hb : 2*c≤k+1
    · exact hb
    exact False.elim (boundary_two_numerator_no_collision r c k a hc H.1 (by omega) ha hab he)
  exact ⟨H.1,hb,H.2.2⟩

theorem one_inner_boundary_two_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c≤k+1 ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_boundary_two_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1

#print axioms pow_mono
#print axioms equal_shift
#print axioms equal_sign
#print axioms upperOne_shift
#print axioms upperOne_sign
#print axioms upperFour_shift
#print axioms upperFour_sign
#print axioms lowFour_shift
#print axioms lowFour_sign
#print axioms small_boundary_certificate
#print axioms boundary_mod_congr
#print axioms small_boundary_nonzero
#print axioms boundary_factor
#print axioms boundary_congruence
#print axioms small_mod_unique
#print axioms quotient_range
#print axioms boundary_quotient_certificate
#print axioms high_boundary_patterns
#print axioms high_boundary_congruence
#print axioms high_boundary_nonzero
#print axioms low_boundary_nonzero
#print axioms boundary_two_residual_nonzero
#print axioms boundary_two_scaled_no_collision
#print axioms boundary_two_polynomial_no_collision
#print axioms boundary_two_numerator_no_collision
#print axioms one_inner_boundary_two_collision_region
#print axioms one_inner_boundary_two_value_collision_region
end Sounio.ZDScalarBoundaryTwo

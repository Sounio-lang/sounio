import SounioZDScalarNarrowSeparator

/-! Uniform boundary exclusion in the h=1 small-count collision problem.
This module does not assert global ResidualSeparation or bibliographic priority. -/
namespace Sounio.ZDScalarHalfSeparator
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

def critical1Nat (x : Nat) : Nat := x*(x*(x*(x*(x*(112*x + 1592) + 9020) + 26328) + 41973) + 34783) + 11742

theorem critical1_shift (x : Nat) :
    -criticalH ((x:Int)+2) 1=(critical1Nat x:Int) := by
  simp only [criticalH,critical1Nat,Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem critical1_sign (w : Int) (hw : 2≤w) : 0< -criticalH w 1 := by
  let x := (w-2).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have h := critical1_shift x
  have he : (x:Int)+2=w := by omega
  rw [he] at h
  have hn : 0<critical1Nat x := by unfold critical1Nat; omega
  have hi : (0:Int)<(critical1Nat x:Int) := by omega
  omega

def critical2Nat (x : Nat) : Nat := x*(x*(x*(x*(x*(1792*x + 73280) + 1214288) + 10309248) + 46228419) + 98687890) + 66972648

theorem critical2_shift (x : Nat) :
    criticalH ((x:Int)+8) 2=(critical2Nat x:Int) := by
  simp only [criticalH,critical2Nat,Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem critical2_sign (w : Int) (hw : 8≤w) : 0<criticalH w 2 := by
  let x := (w-8).toNat
  have hx : (x:Int)=w-8 := Int.toNat_of_nonneg (by omega)
  have h := critical2_shift x
  have he : (x:Int)+8=w := by omega
  rw [he] at h
  have hn : 0<critical2Nat x := by unfold critical2Nat; omega
  have hi : (0:Int)<(critical2Nat x:Int) := by omega
  omega

def critical4Nat (x : Nat) : Nat := x*(x*(x*(x*(x*(143360*x + 6091264) + 106124608) + 965603328) + 4801863171) + 12213804500) + 12111277584

theorem critical4_shift (x : Nat) :
    criticalH ((x:Int)+8) 4=(critical4Nat x:Int) := by
  simp only [criticalH,critical4Nat,Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem critical4_sign (w : Int) (hw : 8≤w) : 0<criticalH w 4 := by
  let x := (w-8).toNat
  have hx : (x:Int)=w-8 := Int.toNat_of_nonneg (by omega)
  have h := critical4_shift x
  have he : (x:Int)+8=w := by omega
  rw [he] at h
  have hn : 0<critical4Nat x := by unfold critical4Nat; omega
  have hi : (0:Int)<(critical4Nat x:Int) := by omega
  omega

def critical8Nat (x : Nat) : Nat := x*(x*(x*(x*(x*(5963776*x + 522162176) + 18875225344) + 359832076288) + 3804725899267) + 21070618687272) + 47456652849120

theorem critical8_shift (x : Nat) :
    criticalH ((x:Int)+16) 8=(critical8Nat x:Int) := by
  simp only [criticalH,critical8Nat,Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem critical8_sign (w : Int) (hw : 16≤w) : 0<criticalH w 8 := by
  let x := (w-16).toNat
  have hx : (x:Int)=w-16 := Int.toNat_of_nonneg (by omega)
  have h := critical8_shift x
  have he : (x:Int)+16=w := by omega
  rw [he] at h
  have hn : 0<critical8Nat x := by unfold critical8Nat; omega
  have hi : (0:Int)<(critical8Nat x:Int) := by omega
  omega

/-- Nonvanishing on the full dyadic main-case domain, including negative values. -/
theorem critical_dyadic_nonzero (d e : Nat) (hd : 1≤d) (he : e≤d) :
    criticalH ((2:Int)^d) ((2:Int)^e)≠0 := by
  by_cases he4 : 4≤e
  · have H := critical_positive ((2:Int)^d) ((2:Int)^e) (pow_mono e d he) (pow_mono 4 e he4)
    omega
  have cases : e=0 ∨ e=1 ∨ e=2 ∨ e=3 := by omega
  rcases cases with h | h | h | h
  · subst e
    have H := critical1_sign ((2:Int)^d) (pow_mono 1 d hd)
    simpa only [Int.pow_zero] using (show criticalH ((2:Int)^d) 1≠0 by omega)
  · subst e
    by_cases h3 : 3≤d
    · have H := critical2_sign ((2:Int)^d) (pow_mono 3 d h3)
      change criticalH ((2:Int)^d) 2≠0
      omega
    · have cases : d=1 ∨ d=2 := by omega
      rcases cases with h | h <;> subst d <;> decide
  · subst e
    by_cases h3 : 3≤d
    · have H := critical4_sign ((2:Int)^d) (pow_mono 3 d h3)
      change criticalH ((2:Int)^d) 4≠0
      omega
    · have h : d=2 := by omega
      subst d
      decide
  · subst e
    by_cases h4 : 4≤d
    · have H := critical8_sign ((2:Int)^d) (pow_mono 4 d h4)
      change criticalH ((2:Int)^d) 8≠0
      omega
    · have h : d=3 := by omega
      subst d
      decide

theorem main_dyadic_nonzero (c d e : Nat) (hc : 1≤c) (hd : 1≤d) (he : e≤d) :
    narrowResidual ((2:Int)^c) ((2:Int)^d) ((2:Int)^e)
      (18*(2:Int)^d*(2:Int)^e-9)≠0 := by
  intro hp
  have hc' := main_pattern_exponent c d e hc hd he hp
  have hs : (2:Int)^c=2*(2:Int)^d*(2:Int)^e := by
    rw [hc',Int.pow_add,Int.pow_add]; grind only
  rw [hs,critical_factor] at hp
  have hw : (0:Int)<(2:Int)^d := Int.pow_pos (by decide)
  have hz : (0:Int)<(2:Int)^e := Int.pow_pos (by decide)
  have hpref : (0:Int)<12*(2:Int)^d*(2:Int)^d*(2:Int)^e :=
    Int.mul_pos (Int.mul_pos (Int.mul_pos (by decide) hw) hw) hz
  exact critical_dyadic_nonzero d e hd he ((Int.mul_eq_zero.mp hp).resolve_left (by omega))

def wrap2Nat (x y : Nat) : Nat := x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(y*(y*(y*(6815744*y + 27262976) + 40894464) + 27262976) + 6815744) + y*(y*(y*(132186112*y + 528744448) + 793116672) + 528744448) + 132186112) + y*(y*(y*(1157234688*y + 4628766720) + 6942892032) + 4628422656) + 1157062656) + y*(y*(y*(6024069120*y + 24093622272) + 36136452096) + 24088313856) + 6021414912) + y*(y*(y*(20654850048*y + 82601573376) + 123875607552) + 82565895168) + 20637010944) + y*(y*(y*(48752492544*y + 194941820928) + 292310360064) + 194805227520) + 48684195840) + y*(y*(y*(80241229824*y + 320802533376) + 480959468544) + 320476255056) + 80078090064) + y*(y*(y*(90949287936*y + 363549818880) + 544951640064) + 363050966208) + 90699857088) + y*(y*(y*(67947724800*y + 271555313664) + 406976276160) + 271077485184) + 67708797952) + y*(y*(y*(30215766016*y + 120734482432) + 180906007296) + 120471599872) + 30084309248) + y*(y*(y*(6073352192*y + 24262541312) + 36346485504) + 24198741248) + 6041445120

theorem wrap2_shift (x y : Nat) :
    let w := (x:Int)+2
    let v := (y:Int)+1
    narrowResidual (8*w*v) w (4*w) (8*w*w-9)=(wrap2Nat x y:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,wrap2Nat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem wrap2_positive (w v : Int) (hw : 2≤w) (hv : 1≤v) :
    0<narrowResidual (8*w*v) w (4*w) (8*w*w-9) := by
  let x := (w-2).toNat
  let y := (v-1).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=v-1 := Int.toNat_of_nonneg (by omega)
  have h := wrap2_shift x y
  dsimp only at h
  have hw' : (x:Int)+2=w := by omega
  have hv' : (y:Int)+1=v := by omega
  rw [hw',hv'] at h
  have hn : 0<wrap2Nat x y := by unfold wrap2Nat; omega
  have hi : (0:Int)<(wrap2Nat x y:Int) := by omega
  omega

theorem wrap2_nonzero (c d : Nat) (hd : 1≤d) (hcd : d+3≤c) :
    narrowResidual ((2:Int)^c) ((2:Int)^d) (4*(2:Int)^d)
      (8*(2:Int)^d*(2:Int)^d-9)≠0 := by
  obtain ⟨v,hv⟩ := Nat.exists_eq_add_of_le hcd
  have hs : (2:Int)^c=8*(2:Int)^d*(2:Int)^v := by
    rw [hv,Int.pow_add,Int.pow_add]; grind only
  rw [hs]
  have H := wrap2_positive ((2:Int)^d) ((2:Int)^v) (pow_mono 1 d hd) (pow_mono 0 v (by omega))
  omega

def wrap3Nat (x y : Nat) : Nat := x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(y*(y*(y*(10485760*y + 41943040) + 62914560) + 41943040) + 10485760) + y*(y*(y*(201457664*y + 805830656) + 1208745984) + 805830656) + 201457664) + y*(y*(y*(1757675520*y + 7030358016) + 10545020928) + 7029669888) + 1757331456) + y*(y*(y*(9179234304*y + 36711825408) + 55060070400) + 36701601792) + 9174122496) + y*(y*(y*(31797018624*y + 127155108864) + 190683165696) + 127089079296) + 31764003840) + y*(y*(y*(76365692928*y + 305342214144) + 457831895040) + 305099919360) + 76244545536) + y*(y*(y*(128773521408*y + 514820493312) + 771817371648) + 514267346448) + 128496946704) + y*(y*(y*(150491627520*y + 601571622912) + 901756993536) + 600765609216) + 150088611072) + y*(y*(y*(116568096768*y + 465917853696) + 698332448448) + 465183668736) + 116200977344) + y*(y*(y*(53989081088*y + 215774855168) + 323379665664) + 215391022592) + 53797131520) + y*(y*(y*(11341398016*y + 45324894208) + 67922664192) + 45236206336) + 11297038848

theorem wrap3_shift (x y : Nat) :
    let w := (x:Int)+2
    let v := (y:Int)+1
    narrowResidual (8*w*v) w (8*w) (16*w*w-9)=(wrap3Nat x y:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,wrap3Nat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem wrap3_positive (w v : Int) (hw : 2≤w) (hv : 1≤v) :
    0<narrowResidual (8*w*v) w (8*w) (16*w*w-9) := by
  let x := (w-2).toNat
  let y := (v-1).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=v-1 := Int.toNat_of_nonneg (by omega)
  have h := wrap3_shift x y
  dsimp only at h
  have hw' : (x:Int)+2=w := by omega
  have hv' : (y:Int)+1=v := by omega
  rw [hw',hv'] at h
  have hn : 0<wrap3Nat x y := by unfold wrap3Nat; omega
  have hi : (0:Int)<(wrap3Nat x y:Int) := by omega
  omega

theorem wrap3_nonzero (c d : Nat) (hd : 1≤d) (hcd : d+3≤c) :
    narrowResidual ((2:Int)^c) ((2:Int)^d) (8*(2:Int)^d)
      (16*(2:Int)^d*(2:Int)^d-9)≠0 := by
  obtain ⟨v,hv⟩ := Nat.exists_eq_add_of_le hcd
  have hs : (2:Int)^c=8*(2:Int)^d*(2:Int)^v := by
    rw [hv,Int.pow_add,Int.pow_add]; grind only
  rw [hs]
  have H := wrap3_positive ((2:Int)^d) ((2:Int)^v) (pow_mono 1 d hd) (pow_mono 0 v (by omega))
  omega

def lowNat (x y : Nat) : Nat := x*(x*(x*(x*(x*(x*(x*(x*(y*(y*(y*(868352*y + 3473408) + 5210112) + 3473408) + 868352) + y*(y*(y*(14483456*y + 57976832) + 87029760) + 58062848) + 14526464) + y*(y*(y*(105218048*y + 421360640) + 632773632) + 422337536) + 105706496) + y*(y*(y*(435027968*y + 1742305280) + 2616741888) + 1746679808) + 437215232) + y*(y*(y*(1120010240*y + 4484738048) + 6734088960) + 4494004592) + 1124643440) + y*(y*(y*(1839202304*y + 7360716800) + 11046679296) + 7368016592) + 1842851792) + y*(y*(y*(1881669632*y + 7524540416) + 11283096384) + 7519248464) + 1879022880) + y*(y*(y*(1096810496*y + 4381147136) + 6562089216) + 4367977856) + 1090225344) + y*(y*(y*(278921216*y + 1112588288) + 1664052480) + 1106024768) + 275639424

theorem low_shift (x y : Nat) :
    let w := (x:Int)+2
    let v := (y:Int)+1
    (-narrowResidual (8*w*v) w 1 (16*w*w+18*w-9))=(lowNat x y:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,lowNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem low_negative (w v : Int) (hw : 2≤w) (hv : 1≤v) :
    narrowResidual (8*w*v) w 1 (16*w*w+18*w-9)<0 := by
  let x := (w-2).toNat
  let y := (v-1).toNat
  have hx : (x:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=v-1 := Int.toNat_of_nonneg (by omega)
  have h := low_shift x y
  dsimp only at h
  have hw' : (x:Int)+2=w := by omega
  have hv' : (y:Int)+1=v := by omega
  rw [hw',hv'] at h
  have hn : 0<lowNat x y := by unfold lowNat; omega
  have hi : (0:Int)<(lowNat x y:Int) := by omega
  omega

private theorem zero_factor_mod (F R J M Q : Int)
    (h : F=R-J+M*Q) (hf : F=0) : (J-R)%M=0 := by
  apply Int.emod_eq_zero_of_dvd
  exact ⟨Q,by omega⟩

theorem high_factor (q t v j : Int) :
    let w := 2*q
    let z := 4*t
    let s := 8*w*v
    narrowResidual s w z j=18*w*z-(j+9)+64*w*w*
      (v*v*(narrowA w z j*s*s+narrowB w z j*s+narrowC w z j)+
        3*v*(3*q*q-14*q*t+3*t)) := by
  dsimp
  simp only [narrowResidual,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem middle_factor (q v j : Int) :
    let w := 2*q
    let s := 8*w*v
    narrowResidual s w 2 j=36*w-(j+9)+32*w*w*
      (2*v*v*(narrowA w 2 j*s*s+narrowB w 2 j*s+narrowC w 2 j)+
        3*v*(6*q*q-14*q+3)) := by
  dsimp
  simp only [narrowResidual,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem low_factor (w v j : Int) :
    let s := 8*w*v
    narrowResidual s w 1 j=18*w-(j+9)+16*w*w*
      (4*v*v*(narrowA w 1 j*s*s+narrowB w 1 j*s+narrowC w 1 j)+
        3*v*(3*w*w-7*w+3)) := by
  dsimp
  simp only [narrowResidual,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem high_congruence (c d e : Nat) (hd : 1≤d) (he : 2≤e) (hcd : d+3≤c)
    (j : Int) (hp : narrowResidual ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e)%(64*(2:Int)^d*(2:Int)^d)=0 := by
  obtain ⟨q,hq⟩ := Nat.exists_eq_add_of_le hd
  obtain ⟨t,ht⟩ := Nat.exists_eq_add_of_le he
  obtain ⟨v,hv⟩ := Nat.exists_eq_add_of_le hcd
  have hw : (2:Int)^d=2*(2:Int)^q := by rw [hq,Int.pow_add]; rfl
  have hz : (2:Int)^e=4*(2:Int)^t := by rw [ht,Int.pow_add]; rfl
  have hs : (2:Int)^c=8*(2:Int)^d*(2:Int)^v := by
    rw [hv,Int.pow_add,Int.pow_add]; grind only
  rw [hs,hw,hz] at hp
  have H := high_factor ((2:Int)^q) ((2:Int)^t) ((2:Int)^v) j
  dsimp only at H
  have hm := zero_factor_mod _ _ _ _ _ H hp
  simpa only [hw,hz] using hm

private theorem small_mod_unique (R J M : Int)
    (hR : 0≤R) (hRM : R<M) (hJ : 0≤J) (hJM : J<M) (hmod : (J-R)%M=0) : J=R := by
  have H := Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr hmod
  rw [Int.emod_eq_of_lt hJ hJM,Int.emod_eq_of_lt hR hRM] at H
  exact H

private theorem sixteen_choices (w J : Int) (hw : 2≤w)
    (hJ : 0<J) (hJu : J<24*w*w) (hm : (J-18*w)%(16*w*w)=0) :
    J=18*w ∨ J=18*w+16*w*w := by
  have hw2 : 4≤w*w := Int.mul_le_mul hw hw (by decide) (by omega)
  have h18 := Int.mul_le_mul_of_nonneg_left hw (show 0≤9*w by omega)
  have h1 : 9*w*2=18*w := by grind only
  have h2 : 9*w*w=9*(w*w) := Int.mul_assoc _ _ _
  have h16 : 16*w*w=16*(w*w) := Int.mul_assoc _ _ _
  have h24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
  by_cases H : J<16*w*w
  · left
    exact small_mod_unique (18*w) J (16*w*w) (by omega) (by omega) (by omega) H hm
  · have hmod : ((J-16*w*w)-18*w)%(16*w*w)=0 := by
      have heq : (J-16*w*w)-18*w=(J-18*w)-(16*w*w) := by omega
      rw [heq]
      simpa [Int.sub_emod] using hm
    have heq := small_mod_unique (18*w) (J-16*w*w) (16*w*w)
      (by omega) (by omega) (by omega) (by omega) hmod
    right
    omega

theorem small_exponent_nonzero (c d e : Nat) (hd : 1≤d) (hcd : d+3≤c) (he : e≤1)
    (j : Int) (hj : 0<j+9) (hju : j+9<24*(2:Int)^d*(2:Int)^d) :
    narrowResidual ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  obtain ⟨v,hv⟩ := Nat.exists_eq_add_of_le hcd
  have hs : (2:Int)^c=8*(2:Int)^d*(2:Int)^v := by
    rw [hv,Int.pow_add,Int.pow_add]; grind only
  have hw : (2:Int)≤(2:Int)^d := pow_mono 1 d hd
  have hvpos : (1:Int)≤(2:Int)^v := pow_mono 0 v (by omega)
  intro hp
  have cases : e=0 ∨ e=1 := by omega
  rcases cases with he0 | he1
  · subst e
    change narrowResidual ((2:Int)^c) ((2:Int)^d) 1 j=0 at hp
    have H := low_factor ((2:Int)^d) ((2:Int)^v) j
    dsimp only at H
    have hp' := hp
    rw [hs] at hp'
    have hm := zero_factor_mod _ _ _ _ _ H hp'
    rcases sixteen_choices ((2:Int)^d) (j+9) hw hj hju hm with H | H
    · have hj' : j=18*(2:Int)^d*(2:Int)^0-9 := by simp only [Int.pow_zero,Int.mul_one]; omega
      rw [hj'] at hp
      exact main_dyadic_nonzero c d 0 (by omega) hd (by omega) hp
    · have hj' : j=16*(2:Int)^d*(2:Int)^d+18*(2:Int)^d-9 := by omega
      rw [hj'] at hp'
      have hn := low_negative ((2:Int)^d) ((2:Int)^v) hw hvpos
      omega
  · subst e
    change narrowResidual ((2:Int)^c) ((2:Int)^d) 2 j=0 at hp
    obtain ⟨q,hq⟩ := Nat.exists_eq_add_of_le hd
    have hw' : (2:Int)^d=2*(2:Int)^q := by rw [hq,Int.pow_add]; rfl
    have H := middle_factor ((2:Int)^q) ((2:Int)^v) j
    dsimp only at H
    have hp' := hp
    rw [hs,hw'] at hp'
    have hm' := zero_factor_mod _ _ _ _ _ H hp'
    have hm : (j+9-36*(2:Int)^d)%(32*(2:Int)^d*(2:Int)^d)=0 := by simpa only [hw'] using hm'
    let w : Int := 2^d
    change 2≤w at hw
    have hw2 : 4≤w*w := Int.mul_le_mul hw hw (by decide) (by omega)
    have h36 := Int.mul_le_mul_of_nonneg_left hw (show 0≤18*w by omega)
    have h1 : 18*w*2=36*w := by grind only
    have h2 : 18*w*w=18*(w*w) := Int.mul_assoc _ _ _
    have h24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
    have h32 : 32*w*w=32*(w*w) := Int.mul_assoc _ _ _
    change j+9<24*w*w at hju
    have heq := small_mod_unique (36*w) (j+9) (32*w*w)
      (by dsimp [w]; omega) (by omega) (by omega) (by omega) hm
    have hj' : j=18*(2:Int)^d*(2:Int)^1-9 := by change j=18*w*2-9; grind only
    rw [hj'] at hp
    exact main_dyadic_nonzero c d 1 (by omega) hd hd hp

theorem half_scaled_no_collision (c d e : Nat) (a : Int)
    (hd : 1≤d) (hcd : d+3≤c) (ha : 0<a)
    (hab : a<3*((2:Int)^c*(2:Int)^d)*((2:Int)^c*(2:Int)^d)) :
    collisionPoly ((2:Int)^c*(2:Int)^d*(2:Int)^e) 2
      ((2:Int)^c) ((2:Int)^c*(2:Int)^d) a≠0 := by
  intro hf
  obtain ⟨j,hj,hju,haeq,hp⟩ := narrow_collision_deficit
    ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) a (Int.pow_pos (by decide)) ha hab hf
  have hw : (2:Int)≤(2:Int)^d := pow_mono 1 d hd
  have hw2 : (4:Int)≤(2:Int)^d*(2:Int)^d :=
    Int.mul_le_mul hw hw (by decide) (by omega)
  have hJu : j+9<24*(2:Int)^d*(2:Int)^d := by
    have h1 : 21*(2:Int)^d*(2:Int)^d=21*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
    have h2 : 24*(2:Int)^d*(2:Int)^d=24*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
    omega
  by_cases he : 2≤e
  · have hcon := high_congruence c d e hd he hcd j hp
    rcases narrow_residue_patterns d e hd (j+9) (by omega) hJu hcon with H | H | H
    · obtain ⟨he',hJ⟩ := H
      have hj' : j=18*(2:Int)^d*(2:Int)^e-9 := by omega
      rw [hj'] at hp
      exact main_dyadic_nonzero c d e (by omega) hd he' hp
    · obtain ⟨he',hJ⟩ := H
      have hj' : j=8*(2:Int)^d*(2:Int)^d-9 := by omega
      have hz : (2:Int)^e=4*(2:Int)^d := by rw [he',Int.pow_add]; grind only
      rw [hj',hz] at hp
      exact wrap2_nonzero c d hd hcd hp
    · obtain ⟨he',hJ⟩ := H
      have hj' : j=16*(2:Int)^d*(2:Int)^d-9 := by omega
      have hz : (2:Int)^e=8*(2:Int)^d := by rw [he',Int.pow_add]; grind only
      rw [hj',hz] at hp
      exact wrap3_nonzero c d hd hcd hp
  · exact small_exponent_nonzero c d e hd hcd (by omega) j (by omega) hJu hp

theorem half_polynomial_no_collision (r c k : Nat) (a : Int)
    (hck : c<k) (hbal : k+3≤2*c) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have ht : (2:Int)^k=(2:Int)^c*(2:Int)^d := by rw [hd,Int.pow_add]
  have hu : (2:Int)^(r+1)=(2:Int)^c*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [ht,hu]
  exact half_scaled_no_collision c d e a (by omega) (by omega) ha hab

theorem half_numerator_no_collision (r c k a : Nat)
    (hc : 1≤c) (hck : c<k) (hbal : k+3≤2*c)
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
  exact half_polynomial_no_collision r c k a hck hbal hr (by omega) habi hp

theorem one_inner_half_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c≤k+2 ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := one_inner_small_collision_region r c k a hc hk ha hab he
  have hbal : 2*c≤k+2 := by
    by_cases hb : 2*c≤k+2
    · exact hb
    exact False.elim (half_numerator_no_collision r c k a hc H.1 (by omega) ha hab he)
  exact ⟨H.1,hbal,H.2.2⟩

theorem one_inner_half_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c≤k+2 ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_half_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1

#print axioms pow_mono
#print axioms critical1_shift
#print axioms critical1_sign
#print axioms critical2_shift
#print axioms critical2_sign
#print axioms critical4_shift
#print axioms critical4_sign
#print axioms critical8_shift
#print axioms critical8_sign
#print axioms critical_dyadic_nonzero
#print axioms main_dyadic_nonzero
#print axioms wrap2_shift
#print axioms wrap2_positive
#print axioms wrap2_nonzero
#print axioms wrap3_shift
#print axioms wrap3_positive
#print axioms wrap3_nonzero
#print axioms low_shift
#print axioms low_negative
#print axioms zero_factor_mod
#print axioms high_factor
#print axioms middle_factor
#print axioms low_factor
#print axioms high_congruence
#print axioms small_mod_unique
#print axioms sixteen_choices
#print axioms small_exponent_nonzero
#print axioms half_scaled_no_collision
#print axioms half_polynomial_no_collision
#print axioms half_numerator_no_collision
#print axioms one_inner_half_collision_region
#print axioms one_inner_half_value_collision_region
end Sounio.ZDScalarHalfSeparator

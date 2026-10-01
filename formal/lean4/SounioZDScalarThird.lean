import SounioZDScalarGap

/-! Mixed dyadic/mod7 obstruction in the explicit h=1 small-count model.
This does not assert global ResidualSeparation or bibliographic priority. -/
namespace Sounio.ZDScalarThird
open Sounio.ZDScalarGap
open Sounio.ZDScalarBalanced
open Sounio.ZDScalarBoundaryOne
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




def rootQ (s v z : Int) : Int :=
  -144*s^5*v^5*z^2 -672*s^5*v^4*z^4 +864*s^5*v^3*z^3 +648*s^5*v^2*z^2 +336*s^4*v^4*z^3 +864*s^4*v^3*z^3 +168*s^3*v^4*z^2 -144*s^3*v^4*z +360*s^3*v^3*z^3 -864*s^3*v^2*z^2 -486*s^3*v*z +24*s^2*v^4*z +360*s^2*v^3*z^3 -252*s^2*v^3*z^2 -1296*s^2*v^2*z^2 -486*s^2*v*z -252*s*v^3*z^2 +72*s*v^3 -720*s*v^2*z^2 +486*s*v*z +243*s +216*v^2*z^2 +126*v^2*z +486*v*z


def rootR (s v z : Int) : Int :=
  -9+18*v*s*z+(27+18*v*z)*s*s-(108*v+42*v*v)*s^3*z+
    (18*v^3+84*v*v*z*z-108*v*z-81)*s^4

def thirdR (s v z : Int) : Int :=
  27*s*s+18*v*s*z*(s+1)-6*v*(18+7*v)*s^3*z-81*s^4

def rootB (s v z : Int) : Int :=
  3-6*v*(s+1)*s*z+8*v*v*s^4*z*z

def rootV (s v z : Int) : Int :=
  1-s*s*rootB s v z+s^4*(rootB s v z)^2

theorem root_factor (s v z : Int) :
    narrowResidual s (v*s) z (rootR s v z)=s^5*rootQ s v z := by
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,rootR,rootQ,
    Int.pow_succ,Int.pow_zero]
  grind only

theorem root_slope (s v z : Int) :
    gapSlope s (v*s) z=1+s*s*rootB s v z := by
  simp only [gapSlope,rootB,Int.pow_succ,Int.pow_zero]
  grind only

theorem unit_identity (s b : Int) :
    (1-s*s*b+s^4*b^2)*(1+s*s*b)=1+s^6*b^3 := by
  simp only [Int.pow_succ,Int.pow_zero]
  grind only

theorem root_unit_factor (q v z : Int) :
    let s := 16*q
    rootV s v z*gapSlope s (v*s) z =
      1+(16*s^4)*(16*q*q*(rootB s v z)^3) := by
  dsimp only
  rw [root_slope]
  unfold rootV
  rw [unit_identity]
  simp only [Int.pow_succ,Int.pow_zero]
  grind only

theorem root_congruence_full (q v z j : Int)
    (hp : narrowResidual (16*q) (v*(16*q)) z j=0) :
    (j-rootR (16*q) v z)%(16*(16*q)^4)=0 := by
  let s := 16*q
  let M := 16*s^4
  let R := rootR s v z
  let L := gapSlope s (v*s) z
  have hfac := root_factor s v z
  have haff := gap_affine s (v*s) z j R
  change narrowResidual s (v*s) z j=0 at hp
  rw [hp] at haff
  have hid : L*(j-R)=M*(q*rootQ s v z) := by
    dsimp [L,M,R,s] at *
    simp only [Int.pow_succ,Int.pow_zero] at *
    grind only
  have hm : (L*(j-R))%M=0 := by rw [hid]; simp
  have hmul : (rootV s v z*L*(j-R))%M=0 := by
    rw [Int.mul_assoc,Int.mul_emod,hm]; simp
  have hu := root_unit_factor q v z
  change rootV s v z*L=1+M*(16*q*q*(rootB s v z)^3) at hu
  rw [hu] at hmul
  have heq : (1+M*(16*q*q*(rootB s v z)^3))*(j-R)=
      (j-R)+M*((16*q*q*(rootB s v z)^3)*(j-R)) := by grind only
  rw [heq] at hmul
  change (j-R)%M=0
  simpa [Int.add_emod] using hmul

theorem third_congruence (q b t j : Int)
    (hp : narrowResidual (16*q) ((2*b)*(16*q)) (2*t) j=0) :
    (j+9-thirdR (16*q) (2*b) (2*t))%(16*(16*q)^4)=0 := by
  have H := root_congruence_full q (2*b) (2*t) j hp
  have hid : j-rootR (16*q) (2*b) (2*t) =
      (j+9-thirdR (16*q) (2*b) (2*t))-
      (16*(16*q)^4)*(9*b^3+84*b*b*t*t-27*b*t) := by
    simp only [rootR,thirdR,Int.pow_succ,Int.pow_zero]
    grind only
  rw [hid] at H
  simpa [Int.sub_emod] using H

theorem high_f_certificate :
    ∀ h : Fin 8, (108+42*(2:Int)^h.val)%128≠0 := by decide

theorem small_f_certificate :
    ∀ (f : Fin 7) (h : Fin 11), 1≤h.val →
      (81*(2:Int)^f.val+108+42*(2:Int)^h.val)%(16*(2:Int)^f.val)=0 →
      f.val=2 ∧ h.val=3 := by decide

theorem exceptional_dyadic_pair (f h : Nat) (hh : 1≤h)
    (hm : (81*(2:Int)^f+108+42*(2:Int)^h)%(16*(2:Int)^f)=0) :
    f=2 ∧ h=3 := by
  have hf6 : f≤6 := by
    by_cases hf6 : f≤6
    · exact hf6
    obtain ⟨g,hg⟩ := Nat.exists_eq_add_of_le (show 7≤f by omega)
    have hfp : (2:Int)^f=128*(2:Int)^g := by rw [hg,Int.pow_add]; rfl
    have hdiv : 128 ∣ 16*(2:Int)^f := by rw [hfp]; refine ⟨16*(2:Int)^g,?_⟩; grind only
    have hz : (81*(2:Int)^f+108+42*(2:Int)^h)%128=0 :=
      Int.emod_eq_zero_of_dvd (Int.dvd_trans hdiv (Int.dvd_of_emod_eq_zero hm))
    have hcap := power_mod_cap h 7
    change (2:Int)^h%128=(2:Int)^(min h 7)%128 at hcap
    have hz' : (108+42*(2:Int)^(min h 7))%128=0 := by
      rw [hfp] at hz
      simp [Int.add_emod,Int.mul_emod] at hz
      simpa [Int.add_emod,Int.mul_emod,show (2:Int)^7=128 from rfl,hcap] using hz
    exact False.elim (high_f_certificate ⟨min h 7,by omega⟩ hz')
  have hpowle : (2:Int)^f≤64 := pow_mono f 6 hf6
  have hfc : 16*(2:Int)^f ∣ 1024 := by
    obtain ⟨g,hg⟩ := Nat.exists_eq_add_of_le hf6
    refine ⟨(2:Int)^g,?_⟩
    have hp : (2:Int)^6=(2:Int)^f*(2:Int)^g := by rw [hg,Int.pow_add]
    change 1024=16*(2:Int)^f*(2:Int)^g
    have hv : (2:Int)^6=64 := rfl
    grind only
  have hcap := power_mod_cap h 10
  have hdiv : (16*(2:Int)^f) ∣ ((2:Int)^h-(2:Int)^(min h 10)) :=
    Int.dvd_trans hfc (Int.dvd_of_emod_eq_zero (Int.emod_eq_emod_iff_emod_sub_eq_zero.mp hcap))
  have hmcap : (2:Int)^h%(16*(2:Int)^f) =
      (2:Int)^(min h 10)%(16*(2:Int)^f) :=
    Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr (Int.emod_eq_zero_of_dvd hdiv)
  have hm' : (81*(2:Int)^f+108+42*(2:Int)^(min h 10))%(16*(2:Int)^f)=0 := by
    simpa [Int.add_emod,Int.mul_emod,hmcap] using hm
  have hmin : 1≤min h 10 := by omega
  have H := small_f_certificate ⟨f,by omega⟩ ⟨min h 10,by omega⟩ hmin hm'
  simp only at H
  omega

theorem dyadic_ratio_cases (a b : Nat) :
    32*(2:Int)^b≤(2:Int)^a ∨
    (2:Int)^a=16*(2:Int)^b ∨ (2:Int)^a=8*(2:Int)^b ∨
    (2:Int)^a=4*(2:Int)^b ∨ (2:Int)^a=2*(2:Int)^b ∨
    (2:Int)^b=(2:Int)^a ∨ (2:Int)^b=2*(2:Int)^a ∨
    (2:Int)^b=4*(2:Int)^a ∨ 8*(2:Int)^a ∣ (2:Int)^b := by
  by_cases hlo : b+5≤a
  · left
    have H := pow_mono (b+5) a hlo
    have heq : (2:Int)^(b+5)=32*(2:Int)^b := by rw [Int.pow_add]; grind only
    omega
  by_cases hhi : a+3≤b
  · right; right; right; right; right; right; right; right
    obtain ⟨g,hg⟩ := Nat.exists_eq_add_of_le hhi
    refine ⟨(2:Int)^g,?_⟩
    rw [hg,Int.pow_add,Int.pow_add]
    grind only
  have cases : a=b+4 ∨ a=b+3 ∨ a=b+2 ∨ a=b+1 ∨ b=a ∨ b=a+1 ∨ b=a+2 := by omega
  rcases cases with H | H | H | H | H | H | H
  · right; left
    rw [H,Int.pow_add]; grind only
  · right; right; left
    rw [H,Int.pow_add]; grind only
  · right; right; right; left
    rw [H,Int.pow_add]; grind only
  · right; right; right; right; left
    rw [H,Int.pow_add]; grind only
  · right; right; right; right; right; left
    rw [H]
  · right; right; right; right; right; right; left
    rw [H,Int.pow_add]; grind only
  · right; right; right; right; right; right; right; left
    rw [H,Int.pow_add]; grind only

theorem single_power_impossible (a b : Nat) (ha : 10≤a) (n : Int)
    (hn : 0<n) (hnu : 32*n<3*(2:Int)^a)
    (hm : (n-(27+18*(2:Int)^b-81*(2:Int)^a))%(16*(2:Int)^a)=0) : False := by
  let S : Int := 2^a
  let t : Int := 2^b
  have hS : 1024≤S := pow_mono 10 a ha
  have ht : 1≤t := by have hh : 0<t := Int.pow_pos (by decide); omega
  change 32*n<3*S at hnu
  change (n-(27+18*t-81*S))%(16*S)=0 at hm
  have cases := dyadic_ratio_cases a b
  change 32*t≤S ∨ S=16*t ∨ S=8*t ∨ S=4*t ∨ S=2*t ∨
    t=S ∨ t=2*S ∨ t=4*S ∨ 8*S∣t at cases
  rcases cases with H | H | H | H | H | H | H | H | H
  · have hid : n-(27+18*t-81*S)=(n-(15*S+18*t+27))+(16*S)*6 := by omega
    rw [hid] at hm
    have hm' : (n-(15*S+18*t+27))%(16*S)=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (15*S+18*t+27) n (16*S)
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · rw [H] at hm hnu hS
    have hid : n-(27+18*t-81*(16*t))=(n-(2*t+27))+(16*(16*t))*5 := by omega
    rw [hid] at hm
    have hm' : (n-(2*t+27))%(16*(16*t))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (2*t+27) n (16*(16*t))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · rw [H] at hm hnu hS
    have hid : n-(27+18*t-81*(8*t))=(n-(10*t+27))+(16*(8*t))*5 := by omega
    rw [hid] at hm
    have hm' : (n-(10*t+27))%(16*(8*t))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (10*t+27) n (16*(8*t))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · rw [H] at hm hnu hS
    have hid : n-(27+18*t-81*(4*t))=(n-(14*t+27))+(16*(4*t))*5 := by omega
    rw [hid] at hm
    have hm' : (n-(14*t+27))%(16*(4*t))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (14*t+27) n (16*(4*t))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · rw [H] at hm hnu hS
    have hid : n-(27+18*t-81*(2*t))=(n-(16*t+27))+(16*(2*t))*5 := by omega
    rw [hid] at hm
    have hm' : (n-(16*t+27))%(16*(2*t))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (16*t+27) n (16*(2*t))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · rw [H] at hm
    have hid : n-(27+18*(S)-81*S)=(n-(S+27))+(16*S)*4 := by omega
    rw [hid] at hm
    have hm' : (n-(S+27))%(16*S)=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (S+27) n (16*S)
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · rw [H] at hm
    have hid : n-(27+18*(2*S)-81*S)=(n-(3*S+27))+(16*S)*3 := by omega
    rw [hid] at hm
    have hm' : (n-(3*S+27))%(16*S)=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (3*S+27) n (16*S)
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · rw [H] at hm
    have hid : n-(27+18*(4*S)-81*S)=(n-(7*S+27))+(16*S)*1 := by omega
    rw [hid] at hm
    have hm' : (n-(7*S+27))%(16*S)=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (7*S+27) n (16*S)
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · obtain ⟨q,hq⟩ := H
    have hid : n-(27+18*t-81*S)=(n-(15*S+27))+(16*S)*(6-9*q) := by rw [hq]; grind only
    rw [hid] at hm
    have hm' : (n-(15*S+27))%(16*S)=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (15*S+27) n (16*S)
      (by omega) (by omega) (by omega) (by omega) hm'
    omega

theorem two_power_impossible (c g : Nat) (hc : 5≤c) (n : Int)
    (hn : 0<n) (hnu : 32*n<3*((2:Int)^c*(2:Int)^c))
    (hm : (n-(27+18*((2:Int)^c+1)*(2:Int)^g-81*((2:Int)^c*(2:Int)^c)))%
      (16*((2:Int)^c*(2:Int)^c))=0) : False := by
  let s : Int := 2^c
  let t : Int := 2^g
  have hs : 32≤s := pow_mono 5 c hc
  have ht : 1≤t := by have hh : 0<t := Int.pow_pos (by decide); omega
  have hsq : 1024≤s*s := Int.mul_le_mul hs hs (by decide) (by omega)
  have hsprod : 32*s≤s*s := by
    have hh := Int.mul_le_mul_of_nonneg_right hs (show 0≤s by omega)
    exact hh
  have htprod : t≤t*t := by
    have hh := Int.mul_le_mul_of_nonneg_left ht (show 0≤t by omega)
    simpa using hh
  change 32*n<3*(s*s) at hnu
  change (n-(27+18*(s+1)*t-81*(s*s)))%(16*(s*s))=0 at hm
  have cases := dyadic_ratio_cases c g
  change 32*t≤s ∨ s=16*t ∨ s=8*t ∨ s=4*t ∨ s=2*t ∨
    t=s ∨ t=2*s ∨ t=4*s ∨ 8*s∣t at cases
  rcases cases with H | H | H | H | H | H | H | H | H
  · have hmul := Int.mul_le_mul_of_nonneg_left H (show 0≤18*(s+1) by omega)
    have hmul' : 32*(18*(s+1)*t)≤18*(s*s)+18*s := by grind only
    have hid : n-(27+18*(s+1)*t-81*(s*s))=
        (n-(15*(s*s)+18*(s+1)*t+27))+(16*(s*s))*6 := by omega
    rw [hid] at hm
    have hm' : (n-(15*(s*s)+18*(s+1)*t+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have htpos : 0<18*(s+1)*t := Int.mul_pos (by omega) (by omega)
    have he := small_mod_unique (15*(s*s)+18*(s+1)*t+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · have hss : s*s=256*(t*t) := by rw [H]; grind only
    have hid : n-(27+18*(s+1)*t-81*(s*s))=(n-(32*(t*t)+18*t+27))+(16*(s*s))*5 := by rw [H]; grind only
    rw [hid] at hm
    have hm' : (n-(32*(t*t)+18*t+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (32*(t*t)+18*t+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · have hss : s*s=64*(t*t) := by rw [H]; grind only
    have hid : n-(27+18*(s+1)*t-81*(s*s))=(n-(80*(t*t)+18*t+27))+(16*(s*s))*5 := by rw [H]; grind only
    rw [hid] at hm
    have hm' : (n-(80*(t*t)+18*t+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (80*(t*t)+18*t+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · have hss : s*s=16*(t*t) := by rw [H]; grind only
    have hid : n-(27+18*(s+1)*t-81*(s*s))=(n-(56*(t*t)+18*t+27))+(16*(s*s))*5 := by rw [H]; grind only
    rw [hid] at hm
    have hm' : (n-(56*(t*t)+18*t+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (56*(t*t)+18*t+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · have hss : s*s=4*(t*t) := by rw [H]; grind only
    have hid : n-(27+18*(s+1)*t-81*(s*s))=(n-(32*(t*t)+18*t+27))+(16*(s*s))*5 := by rw [H]; grind only
    rw [hid] at hm
    have hm' : (n-(32*(t*t)+18*t+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (32*(t*t)+18*t+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · have hid : n-(27+18*(s+1)*t-81*(s*s))=(n-(1*(s*s)+18*s+27))+(16*(s*s))*4 := by rw [H]; grind only
    rw [hid] at hm
    have hm' : (n-(1*(s*s)+18*s+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (1*(s*s)+18*s+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · have hid : n-(27+18*(s+1)*t-81*(s*s))=(n-(3*(s*s)+36*s+27))+(16*(s*s))*3 := by rw [H]; grind only
    rw [hid] at hm
    have hm' : (n-(3*(s*s)+36*s+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (3*(s*s)+36*s+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · have hid : n-(27+18*(s+1)*t-81*(s*s))=(n-(7*(s*s)+72*s+27))+(16*(s*s))*1 := by rw [H]; grind only
    rw [hid] at hm
    have hm' : (n-(7*(s*s)+72*s+27))%(16*(s*s))=0 := by simpa [Int.add_emod] using hm
    have he := small_mod_unique (7*(s*s)+72*s+27) n (16*(s*s))
      (by omega) (by omega) (by omega) (by omega) hm'
    omega
  · obtain ⟨q,hq⟩ := H
    have hid : n-(27+18*(s+1)*t-81*(s*s)) =
        (n-(27+18*t-81*(s*s)))-(16*(s*s))*(9*q) := by rw [hq]; grind only
    rw [hid] at hm
    have hm' : (n-(27+18*t-81*(s*s)))%(16*(s*s))=0 := by simpa [Int.sub_emod] using hm
    have hp : (2:Int)^(2*c)=s*s := by
      dsimp [s]
      rw [show 2*c=c+c by omega,Int.pow_add]
    apply single_power_impossible (2*c) g (by omega) n hn
    · rw [hp]; exact hnu
    · rw [hp]; exact hm'

theorem low_scaled_intervals (s v z u J : Int)
    (hs : 32≤s) (hv : 2≤v) (hz : 2≤z) (hu : 1≤u)
    (hsv : 16*v≤s) (huv : u*v*z=s)
    (hJ : 0<J) (hJu : J<24*(v*s)*(v*s)) :
    0<u*J ∧ u*J<s^4 ∧
    0<s*s*(27*u+18*s+18) ∧ s*s*(27*u+18*s+18)<s^4 := by
  have hss : 0<s*s := Int.mul_pos (by omega) (by omega)
  have hsss : 0≤s*s*s := Int.mul_nonneg (by omega) (by omega)
  have hvsss : 0≤v*s*s*s := Int.mul_nonneg
    (Int.mul_nonneg (Int.mul_nonneg (by omega) (by omega)) (by omega)) (by omega)
  have hvz : 4≤v*z := Int.mul_le_mul hv hz (by decide) (by omega)
  have hu4 : 4*u≤s := by
    have hh := Int.mul_le_mul_of_nonneg_left hvz (show 0≤u by omega)
    grind only
  have huv2 : 2*u*v≤s := by
    have hpos : 0≤u*v := Int.mul_nonneg (by omega) (by omega)
    have hh := Int.mul_le_mul_of_nonneg_left hz hpos
    grind only
  have hsprod : 32*s≤s*s := Int.mul_le_mul_of_nonneg_right hs (by omega)
  have halpha : 27*u+18*s+18<s*s := by omega
  have hd := Int.mul_lt_mul_of_pos_left halpha hss
  have hp4 : (s*s)*(s*s)=s^4 := by simp only [Int.pow_succ,Int.pow_zero]; grind only
  rw [hp4] at hd
  have h1 := Int.mul_le_mul_of_nonneg_right huv2
    (show 0≤12*v*s*s from Int.mul_nonneg
      (Int.mul_nonneg (Int.mul_nonneg (by decide) (by omega)) (by omega)) (by omega))
  have h2 := Int.mul_le_mul_of_nonneg_right hsv hsss
  have hbound : u*(24*(v*s)*(v*s))≤s^4 := by
    simp only [Int.pow_succ,Int.pow_zero]
    grind only
  have hstrict := Int.mul_lt_mul_of_pos_left hJu (show 0<u by omega)
  exact ⟨Int.mul_pos (by omega) hJ,by omega,
    Int.mul_pos hss (by omega),hd⟩

theorem low_scaled_congruence (s v z u J : Int)
    (hs : 32≤s) (hv : 2≤v) (hz : 2≤z) (hu : 1≤u)
    (hsv : 16*v≤s) (huv : u*v*z=s)
    (hJ : 0<J) (hJu : J<24*(v*s)*(v*s))
    (hm : (J-thirdR s v z)%(16*s^4)=0) :
    u*J=s*s*(27*u+18*s+18) ∧ (81*u+108+42*v)%(16*u)=0 := by
  have bounds := low_scaled_intervals s v z u J hs hv hz hu hsv huv hJ hJu
  obtain ⟨b,hb⟩ := Int.dvd_of_emod_eq_zero hm
  have hid : u*(J-thirdR s v z) =
      u*J-s*s*(27*u+18*s+18)+(81*u+108+42*v)*s^4 := by
    rw [←huv]
    simp only [thirdR,Int.pow_succ,Int.pow_zero]
    grind only
  have hb' := congrArg (fun x:Int => u*x) hb
  have he : u*J-s*s*(27*u+18*s+18) =
      s^4*(16*u*b-(81*u+108+42*v)) := by grind only
  have hmod : (u*J-s*s*(27*u+18*s+18))%(s^4)=0 := by rw [he]; simp
  have heq := small_mod_unique (s*s*(27*u+18*s+18)) (u*J) (s^4)
    (by omega) bounds.2.2.2 (by omega) bounds.2.1 hmod
  constructor
  · exact heq
  · have hfac : s^4*(16*u*b-(81*u+108+42*v))=0 := by omega
    have hs4 : 0<s^4 := Int.pow_pos (by omega)
    have hn : 16*u*b-(81*u+108+42*v)=0 := by
      rcases Int.mul_eq_zero.mp hfac with H | H
      · omega
      · exact H
    have hrew : 81*u+108+42*v=(16*u)*b := by omega
    rw [hrew]; simp

theorem cube_power_mod7 (c : Nat) : (((2:Int)^c)^3)%7=1 := by
  induction c with
  | zero => decide
  | succ c ih =>
    have hid : ((2:Int)^(c+1))^3=8*((2:Int)^c)^3 := by
      simp only [Int.pow_succ,Int.pow_zero]
      grind only
    rw [hid]
    simp [Int.mul_emod,ih]

theorem seven_certificate :
    ∀ (s j : Fin 7), (((s.val:Int)^3)%7=1) →
      (2*((j.val:Int)+9))%7=(9*(s.val:Int)^3+63*(s.val:Int)*(s.val:Int))%7 →
      (-(j.val:Int)*(s.val:Int)*(s.val:Int)-3)%7≠0 := by decide

theorem exceptional_mod7_impossible (c : Nat) (w j a : Int)
    (hJ : 2*(j+9)=9*((2:Int)^c)^3+63*(2:Int)^c*(2:Int)^c)
    (ha : 7*a=(21*w*w-j)*(2:Int)^c*(2:Int)^c-3) : False := by
  let s : Int := 2^c
  let sn := (s%7).toNat
  let jn := (j%7).toNat
  have hs0 := Int.emod_nonneg s (show (7:Int)≠0 by decide)
  have hs7 := Int.emod_lt_of_pos s (show (0:Int)<7 by decide)
  have hj0 := Int.emod_nonneg j (show (7:Int)≠0 by decide)
  have hj7 := Int.emod_lt_of_pos j (show (0:Int)<7 by decide)
  have hsn : (sn:Int)=s%7 := Int.toNat_of_nonneg hs0
  have hjn : (jn:Int)=j%7 := Int.toNat_of_nonneg hj0
  have hcube := cube_power_mod7 c
  change s^3%7=1 at hcube
  have hc : (sn:Int)^3%7=1 := by rw [hsn]; simpa [Int.pow_succ, Int.pow_zero, Int.mul_emod] using hcube
  have hJJ := congrArg (fun x:Int => x%7) hJ
  change (2*(j+9))%7=(9*s^3+63*s*s)%7 at hJJ
  have hJJ' : (2*((jn:Int)+9))%7=(9*(sn:Int)^3+63*(sn:Int)*(sn:Int))%7 := by
    rw [hsn,hjn]
    simpa [Int.add_emod,Int.mul_emod,Int.pow_succ, Int.pow_zero, Int.mul_emod] using hJJ
  have haa := congrArg (fun x:Int => x%7) ha
  change (7*a)%7=((21*w*w-j)*s*s-3)%7 at haa
  have haa' : (-(jn:Int)*(sn:Int)*(sn:Int)-3)%7=0 := by
    rw [hsn,hjn]
    simpa [Int.add_emod,Int.sub_emod,Int.mul_emod,Int.neg_emod] using haa.symm
  exact seven_certificate ⟨sn,by omega⟩ ⟨jn,by omega⟩ hc hJJ' haa'


theorem high_normalization (s v z t J : Int)
    (hs : 32≤s) (hv : 2≤v) (hsv : 16*v≤s) (hvz : v*z=s*t)
    (hJ : 0<J) (hJu : J<24*(v*s)*(v*s))
    (hm : (J-thirdR s v z)%(16*s^4)=0) :
    ∃ n : Int, J=s*s*n ∧ 0<n ∧ 32*n<3*(s*s) ∧
      (n-(27+18*(s+1)*t-6*(18+7*v)*s*s*t-81*s*s))%(16*(s*s))=0 := by
  obtain ⟨b,hb⟩ := Int.dvd_of_emod_eq_zero hm
  let n := 27+18*(s+1)*t-6*(18+7*v)*s*s*t-81*s*s+16*s*s*b
  have hr : thirdR s v z =
      s*s*(27+18*(s+1)*t-6*(18+7*v)*s*s*t-81*s*s) := by
    simp only [thirdR,Int.pow_succ,Int.pow_zero]
    grind only
  have heq : J=s*s*n := by
    rw [hr] at hb
    dsimp only [n]
    clear hr hvz hm hs hv hsv hJ hJu
    simp only [Int.pow_succ,Int.pow_zero] at hb
    grind only
  have hss : 0<s*s := Int.mul_pos (by omega) (by omega)
  have hu : J<(24*v*v)*(s*s) := by grind only
  have hn := quotient_range (s*s) J n (24*v*v) hss hJ hu heq
  have hsq := Int.mul_le_mul hsv hsv
    (show 0≤16*v by omega) (show 0≤s by omega)
  have hbound : 32*n<3*(s*s) := by grind only
  refine ⟨n,heq,hn.1,hbound,?_⟩
  have hid : n-(27+18*(s+1)*t-6*(18+7*v)*s*s*t-81*s*s)=
      (16*(s*s))*b := by dsimp [n]; grind only
  rw [hid]; simp

theorem high_dyadic_impossible (c h g : Nat) (hc : 5≤c)
    (hh : 1≤h) (hg : 1≤g) (n : Int) (hn : 0<n)
    (hnu : 32*n<3*((2:Int)^c*(2:Int)^c))
    (hm : (n-(27+18*((2:Int)^c+1)*(2:Int)^g-
      6*(18+7*(2:Int)^h)*(2:Int)^c*(2:Int)^c*(2:Int)^g-
      81*(2:Int)^c*(2:Int)^c))%(16*((2:Int)^c*(2:Int)^c))=0) : False := by
  let s : Int := 2^c
  have hs : 32≤s := pow_mono 5 c hc
  have hss : 0<s*s := Int.mul_pos (by omega) (by omega)
  change 32*n<3*(s*s) at hnu
  change (n-(27+18*(s+1)*(2:Int)^g-
    6*(18+7*(2:Int)^h)*s*s*(2:Int)^g-81*s*s))%(16*(s*s))=0 at hm
  by_cases hh1 : h=1
  · rw [hh1] at hm
    have hid : n-(27+18*(s+1)*(2:Int)^g-6*(18+7*(2:Int)^1)*s*s*(2:Int)^g-81*s*s)=
        (n-(27+18*(s+1)*(2:Int)^g-81*(s*s)))+(16*(s*s))*(12*(2:Int)^g) := by
      grind only
    rw [hid] at hm
    have hm' : (n-(27+18*(s+1)*(2:Int)^g-81*(s*s)))%(16*(s*s))=0 := by
      simpa [Int.add_emod] using hm
    exact two_power_impossible c g hc n hn hnu hm'
  by_cases hg2 : 2≤g
  · obtain ⟨b,hb⟩ := Nat.exists_eq_add_of_le hh
    obtain ⟨u,hu⟩ := Nat.exists_eq_add_of_le hg2
    have hv : (2:Int)^h=2*(2:Int)^b := by rw [hb,Int.pow_add]; rfl
    have ht : (2:Int)^g=4*(2:Int)^u := by rw [hu,Int.pow_add]; rfl
    have hid : n-(27+18*(s+1)*(2:Int)^g-6*(18+7*(2:Int)^h)*s*s*(2:Int)^g-81*s*s)=
        (n-(27+18*(s+1)*(2:Int)^g-81*(s*s)))+
          (16*(s*s))*(3*(9+7*(2:Int)^b)*(2:Int)^u) := by
      rw [hv,ht]; grind only
    rw [hid] at hm
    have hm' : (n-(27+18*(s+1)*(2:Int)^g-81*(s*s)))%(16*(s*s))=0 := by
      simpa [Int.add_emod] using hm
    exact two_power_impossible c g hc n hn hnu hm'
  · have hg1 : g=1 := by omega
    obtain ⟨b,hb⟩ := Nat.exists_eq_add_of_le (show 2≤h by omega)
    have hv : (2:Int)^h=4*(2:Int)^b := by rw [hb,Int.pow_add]; rfl
    rw [hg1,hv] at hm
    have hid : n-(27+18*(s+1)*(2:Int)^1-6*(18+7*(4*(2:Int)^b))*s*s*(2:Int)^1-81*s*s)=
        (n-(7*s*s+36*s+63))+(16*(s*s))*(19+21*(2:Int)^b) := by grind only
    rw [hid] at hm
    have hm' : (n-(7*s*s+36*s+63))%(16*(s*s))=0 := by
      simpa [Int.add_emod] using hm
    have hsprod := Int.mul_le_mul_of_nonneg_right hs (show 0≤s by omega)
    have heq := small_mod_unique (7*s*s+36*s+63) n (16*(s*s))
      (by grind only) (by grind only) (by omega) (by omega) hm'
    grind only

theorem third_dyadic_congruence (c h e : Nat) (hc : 4≤c)
    (hh : 1≤h) (he : 1≤e) (j : Int)
    (hp : narrowResidual ((2:Int)^c) ((2:Int)^h*(2:Int)^c) ((2:Int)^e) j=0) :
    (j+9-thirdR ((2:Int)^c) ((2:Int)^h) ((2:Int)^e))%(16*((2:Int)^c)^4)=0 := by
  obtain ⟨q,hq⟩ := Nat.exists_eq_add_of_le hc
  obtain ⟨b,hb⟩ := Nat.exists_eq_add_of_le hh
  obtain ⟨t,ht⟩ := Nat.exists_eq_add_of_le he
  have hs : (2:Int)^c=16*(2:Int)^q := by rw [hq,Int.pow_add]; rfl
  have hv : (2:Int)^h=2*(2:Int)^b := by rw [hb,Int.pow_add]; rfl
  have hz : (2:Int)^e=2*(2:Int)^t := by rw [ht,Int.pow_add]; rfl
  rw [hs,hv,hz] at hp ⊢
  exact third_congruence ((2:Int)^q) ((2:Int)^b) ((2:Int)^t) j hp

theorem mixed_residual_impossible (c h e : Nat) (hc : h+4≤c)
    (hh : 1≤h) (he : 1≤e) (j a : Int)
    (hj : 0<j) (hju : j<21*((2:Int)^h*(2:Int)^c)*((2:Int)^h*(2:Int)^c))
    (ha : 7*a=(21*((2:Int)^h*(2:Int)^c)*((2:Int)^h*(2:Int)^c)-j)*
      (2:Int)^c*(2:Int)^c-3)
    (hp : narrowResidual ((2:Int)^c) ((2:Int)^h*(2:Int)^c) ((2:Int)^e) j=0) : False := by
  let s : Int := 2^c
  let v : Int := 2^h
  let z : Int := 2^e
  have hs : 32≤s := pow_mono 5 c (by omega)
  have hv : 2≤v := pow_mono 1 h hh
  have hz : 2≤z := pow_mono 1 e he
  have hsv : 16*v≤s := by
    have H := pow_mono (h+4) c hc
    have hid : (2:Int)^(h+4)=16*v := by rw [Int.pow_add]; dsimp [v]; grind only
    rw [hid] at H
    exact H
  have hm := third_dyadic_congruence c h e (by omega) hh he j hp
  change (j+9-thirdR s v z)%(16*s^4)=0 at hm
  change j<21*(v*s)*(v*s) at hju
  have hw : 2≤v*s := by
    have H := Int.mul_le_mul hv hs (by decide) (by omega)
    omega
  have hw2 := Int.mul_le_mul hw hw (by decide) (show 0≤v*s by omega)
  have hJ : 0<j+9 := by omega
  have hJu : j+9<24*(v*s)*(v*s) := by grind only
  by_cases hlo : h+e≤c
  · obtain ⟨f,hf⟩ := Nat.exists_eq_add_of_le hlo
    let u : Int := 2^f
    have hu : 1≤u := pow_mono 0 f (by omega)
    have huv : u*v*z=s := by
      dsimp [u,v,z,s]
      rw [hf,Int.pow_add,Int.pow_add]
      grind only
    have H := low_scaled_congruence s v z u (j+9) hs hv hz hu hsv huv hJ hJu hm
    have pair := exceptional_dyadic_pair f h hh H.2
    have hu4 : u=4 := by dsimp [u]; rw [pair.1]; rfl
    have heq : 2*(j+9)=9*((2:Int)^c)^3+63*(2:Int)^c*(2:Int)^c := by
      have hx := H.1
      rw [hu4] at hx
      change 4*(j+9)=s*s*(27*4+18*s+18) at hx
      change 2*(j+9)=9*s^3+63*s*s
      simp only [Int.pow_succ,Int.pow_zero]
      grind only
    exact exceptional_mod7_impossible c (v*s) j a heq ha
  · obtain ⟨g,hg⟩ := Nat.exists_eq_add_of_le (show c≤h+e by omega)
    have hvz : v*z=s*(2:Int)^g := by
      dsimp [v,z,s]
      rw [←Int.pow_add,hg,Int.pow_add]
    obtain ⟨n,_,hn,hnu,hmn⟩ := high_normalization s v z ((2:Int)^g) (j+9)
      hs hv hsv hvz hJ hJu hm
    exact high_dyadic_impossible c h g (by omega) hh (by omega) n hn hnu hmn


theorem third_scaled_no_collision (c d e : Nat) (a : Int)
    (hcd : c+1≤d) (hdc : d+4≤2*c) (ha : 0<a)
    (hab : a<3*((2:Int)^c*(2:Int)^d)*((2:Int)^c*(2:Int)^d)) :
    collisionPoly ((2:Int)^c*(2:Int)^d*(2:Int)^e) 2
      ((2:Int)^c) ((2:Int)^c*(2:Int)^d) a≠0 := by
  intro hf
  obtain ⟨j,hj,hju,hseven,hp⟩ := narrow_collision_deficit
    ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) a
    (Int.pow_pos (by decide)) ha hab hf
  have H := gap_residual_window c d e (by omega) (by omega) j hj hju hp
  obtain ⟨h,hh⟩ := Nat.exists_eq_add_of_le (show c≤d by omega)
  have hw : (2:Int)^d=(2:Int)^h*(2:Int)^c := by rw [hh,Int.pow_add]; grind only
  rw [hw] at hju hseven hp
  exact mixed_residual_impossible c h e (by omega) (by omega) H.1 j a hj hju hseven hp

theorem third_polynomial_no_collision (r c k : Nat) (a : Int)
    (hck : 2*c<k) (hthird : k+4≤3*c) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have ht : (2:Int)^k=(2:Int)^c*(2:Int)^d := by rw [hd,Int.pow_add]
  have hu : (2:Int)^(r+1)=(2:Int)^c*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [ht,hu]
  exact third_scaled_no_collision c d e a (by omega) (by omega) ha hab

theorem one_inner_third_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c<k ∧ 3*c≤k+3 ∧ k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have H := one_inner_gap_collision_region r c k a hc hk ha hab he
  have hthird : 3*c≤k+3 := by
    by_cases hb : 3*c≤k+3
    · exact hb
    have hp := collision_polynomial r 1 c k a he
    have habi : (a:Int)<3*(2:Int)^k*(2:Int)^k := by
      have h : (a:Int)<((3*(2^k)^2:Nat):Int) := by omega
      simpa only [Int.natCast_mul,Int.natCast_pow,
        show ((2:Nat):Int)=2 from rfl,show ((3:Nat):Int)=3 from rfl,
        Int.pow_succ,Int.pow_zero,Int.one_mul,Int.mul_assoc] using h
    exact False.elim (third_polynomial_no_collision r c k a H.2.1
      (by omega) (by omega) (by omega) habi hp)
  exact ⟨H.1,H.2.1,hthird,H.2.2⟩

theorem one_inner_third_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c<k ∧ 3*c≤k+3 ∧ k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_third_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1

#print axioms pow_mono
#print axioms small_mod_unique
#print axioms quotient_range
#print axioms root_factor
#print axioms root_slope
#print axioms unit_identity
#print axioms root_unit_factor
#print axioms root_congruence_full
#print axioms third_congruence
#print axioms high_f_certificate
#print axioms small_f_certificate
#print axioms exceptional_dyadic_pair
#print axioms dyadic_ratio_cases
#print axioms single_power_impossible
#print axioms two_power_impossible
#print axioms low_scaled_intervals
#print axioms low_scaled_congruence
#print axioms cube_power_mod7
#print axioms seven_certificate
#print axioms exceptional_mod7_impossible
#print axioms high_normalization
#print axioms high_dyadic_impossible
#print axioms third_dyadic_congruence
#print axioms mixed_residual_impossible
#print axioms third_scaled_no_collision
#print axioms third_polynomial_no_collision
#print axioms one_inner_third_collision_region
#print axioms one_inner_third_value_collision_region
end Sounio.ZDScalarThird

import SounioZDScalarThird

/-! Corrected third-boundary obstruction in the explicit h=1 small-count model.
This does not assert global ResidualSeparation or bibliographic priority. -/
namespace Sounio.ZDScalarThirdBoundary
open Sounio.ZDScalarThird
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




theorem single_power_classify (a b : Nat) (ha : 10≤a) (n : Int)
    (hn : 0<n) (hnu : 8*n<3*(2:Int)^a)
    (hm : (n-(27+18*(2:Int)^b-81*(2:Int)^a))%(16*(2:Int)^a)=0) :
    (2:Int)^a=16*(2:Int)^b ∧ n=2*(2:Int)^b+27 := by
  let S : Int := 2^a
  let t : Int := 2^b
  have hS : 1024≤S := pow_mono 10 a ha
  have ht : 1≤t := by have hh : 0<t := Int.pow_pos (by decide); omega
  change 8*n<3*S at hnu
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
    exact ⟨H,he⟩
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

theorem two_power_classify (c g : Nat) (hc : 5≤c) (n : Int)
    (hn : 0<n) (hnu : 8*n<3*((2:Int)^c*(2:Int)^c))
    (hm : (n-(27+18*((2:Int)^c+1)*(2:Int)^g-81*((2:Int)^c*(2:Int)^c)))%
      (16*((2:Int)^c*(2:Int)^c))=0) :
    (((2:Int)^c=16*(2:Int)^g) ∧ n=32*((2:Int)^g*(2:Int)^g)+18*(2:Int)^g+27) ∨
    (((2:Int)^c*(2:Int)^c=16*(2:Int)^g) ∧ n=2*(2:Int)^g+27) := by
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
  change 8*n<3*(s*s) at hnu
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
    exact Or.inl ⟨H,he⟩
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
    have result := single_power_classify (2*c) g (by omega) n hn
      (by rw [hp]; exact hnu) (by rw [hp]; exact hm')
    rw [hp] at result
    exact Or.inr result


theorem boundary_high_normalization (s v z t J : Int)
    (hs : 32≤s) (hv : 2≤v) (hsv : 8*v≤s) (hvz : v*z=s*t)
    (hJ : 0<J) (hJu : J<24*(v*s)*(v*s))
    (hm : (J-thirdR s v z)%(16*s^4)=0) :
    ∃ n : Int, J=s*s*n ∧ 0<n ∧ 8*n<3*(s*s) ∧
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
    (show 0≤8*v by omega) (show 0≤s by omega)
  have hbound : 8*n<3*(s*s) := by grind only
  refine ⟨n,heq,hn.1,hbound,?_⟩
  have hid : n-(27+18*(s+1)*t-6*(18+7*v)*s*s*t-81*s*s)=
      (16*(s*s))*b := by dsimp [n]; grind only
  rw [hid]; simp

theorem boundary_high_classify (c h g : Nat) (hc : 5≤c)
    (hh : 1≤h) (hg : 1≤g) (n : Int) (hn : 0<n)
    (hnu : 8*n<3*((2:Int)^c*(2:Int)^c))
    (hm : (n-(27+18*((2:Int)^c+1)*(2:Int)^g-
      6*(18+7*(2:Int)^h)*(2:Int)^c*(2:Int)^c*(2:Int)^g-
      81*(2:Int)^c*(2:Int)^c))%(16*((2:Int)^c*(2:Int)^c))=0) :
    (((2:Int)^c=16*(2:Int)^g) ∧ n=32*((2:Int)^g*(2:Int)^g)+18*(2:Int)^g+27) ∨
    (((2:Int)^c*(2:Int)^c=16*(2:Int)^g) ∧ n=2*(2:Int)^g+27) := by
  let s : Int := 2^c
  have hs : 32≤s := pow_mono 5 c hc
  have hss : 0<s*s := Int.mul_pos (by omega) (by omega)
  change 8*n<3*(s*s) at hnu
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
    exact two_power_classify c g hc n hn hnu hm'
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
    exact two_power_classify c g hc n hn hnu hm'
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

def psi (t : Int) : Int :=
  1744830464*t^10-1132462080*t^9-916979712*t^8+1376256*t^6+
  187392*t^5+171264*t^4+10404*t^3-261*t^2-27*t-20

def proposedJ (t : Int) : Int := 8192*t^4+4608*t^3+6912*t^2-9

theorem proposed_identity (t : Int) :
    narrowResidual (16*t) (32*t*t) (8*t) (proposedJ t)=
      262144*t^4*psi t := by
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,proposedJ,psi,
    Int.pow_succ,Int.pow_zero]
  grind only

theorem psi_mod8 (u : Int) : psi (8*u)%8=4 := by
  simp [psi,Int.pow_succ,Int.pow_zero,Int.add_emod,Int.sub_emod,Int.mul_emod]

theorem proposed_family_nonzero (u : Int) (hu : 0<u) :
    narrowResidual (16*(8*u)) (32*(8*u)*(8*u)) (8*(8*u)) (proposedJ (8*u))≠0 := by
  intro hp
  have hid := proposed_identity (8*u)
  rw [hp] at hid
  have hpsi := psi_mod8 u
  have hpp : psi (8*u)≠0 := by intro hz; rw [hz] at hpsi; contradiction
  have hpow : 0<(8*u)^4 := Int.pow_pos (by omega)
  have hcoef : 0<262144*(8*u)^4 := Int.mul_pos (by decide) hpow
  rcases Int.mul_eq_zero.mp hid.symm with H | H
  · omega
  · exact hpp H

theorem second_seven_certificate :
    ∀ (s j : Fin 7), (((s.val:Int)^3)%7=1) →
      (8*((j.val:Int)+9))%7=((s.val:Int)^4+216*(s.val:Int)*(s.val:Int))%7 →
      (-(j.val:Int)*(s.val:Int)*(s.val:Int)-3)%7≠0 := by decide +kernel

theorem second_mod7_impossible (c : Nat) (w j a : Int)
    (hJ : 8*(j+9)=((2:Int)^c)^4+216*(2:Int)^c*(2:Int)^c)
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
  change (8*(j+9))%7=(s^4+216*s*s)%7 at hJJ
  have hJJ' : (8*((jn:Int)+9))%7=((sn:Int)^4+216*(sn:Int)*(sn:Int))%7 := by
    rw [hsn,hjn]
    simpa [Int.add_emod,Int.mul_emod,Int.pow_succ, Int.pow_zero, Int.mul_emod] using hJJ
  have haa := congrArg (fun x:Int => x%7) ha
  change (7*a)%7=((21*w*w-j)*s*s-3)%7 at haa
  have haa' : (-(jn:Int)*(sn:Int)*(sn:Int)-3)%7=0 := by
    rw [hsn,hjn]
    simpa [Int.add_emod,Int.sub_emod,Int.mul_emod,Int.neg_emod] using haa.symm
  exact second_seven_certificate ⟨sn,by omega⟩ ⟨jn,by omega⟩ hc hJJ' haa'



private theorem double_congruence (M J R R2 K : Int)
    (hm : (J-R)%M=0)
    (hid : 2*(J-R)=(2*J-R2)+(2*M)*K) :
    (2*J-R2)%(2*M)=0 := by
  obtain ⟨b,hb⟩ := Int.dvd_of_emod_eq_zero hm
  have he : 2*J-R2=(2*M)*(b-K) := by grind only
  rw [he]; simp

private theorem low_interval (s J A B C : Int)
    (hs : 256≤s) (hJ : 0<J) (hJu : 8*J<3*s^4)
    (hA : 6≤A ∧ A≤18) (hB : 0≤B ∧ B≤36) (hC : 0≤C ∧ C≤90)
    (hm : (2*J-(A*s^4+B*s^3+C*s^2))%(32*s^4)=0) : False := by
  have hs2 : 0<s^2 := Int.pow_pos (by omega)
  have hs3 : 0<s^3 := Int.pow_pos (by omega)
  have hs4 : 0<s^4 := Int.pow_pos (by omega)
  have h32 : 32*s^2≤s^3 := by
    have H := Int.mul_le_mul_of_nonneg_right (show 32≤s by omega) (show 0≤s^2 by omega)
    simp only [Int.pow_succ,Int.pow_zero] at *
    grind only
  have h43 : 32*s^3≤s^4 := by
    have H := Int.mul_le_mul_of_nonneg_right (show 32≤s by omega) (show 0≤s^3 by omega)
    simp only [Int.pow_succ,Int.pow_zero] at *
    grind only
  have a0 := Int.mul_le_mul_of_nonneg_right hA.1 (show 0≤s^4 by omega)
  have a1 := Int.mul_le_mul_of_nonneg_right hA.2 (show 0≤s^4 by omega)
  have b0 := Int.mul_le_mul_of_nonneg_right hB.1 (show 0≤s^3 by omega)
  have b1 := Int.mul_le_mul_of_nonneg_right hB.2 (show 0≤s^3 by omega)
  have c0 := Int.mul_le_mul_of_nonneg_right hC.1 (show 0≤s^2 by omega)
  have c1 := Int.mul_le_mul_of_nonneg_right hC.2 (show 0≤s^2 by omega)
  have H := small_mod_unique (A*s^4+B*s^3+C*s^2) (2*J) (32*s^4)
    (by omega) (by omega) (by omega) (by omega) hm
  omega

theorem boundary_low_impossible (q : Int) (e : Nat) (J : Int)
    (hq : 1≤q) (he : 1≤e) (heu : e≤3)
    (hJ : 0<J) (hJu : 8*J<3*(256*q)^4)
    (hm : (J-thirdR (256*q) (32*q) ((2:Int)^e))%(16*(256*q)^4)=0) :
    False := by
  have cases : e=1 ∨ e=2 ∨ e=3 := by omega
  rcases cases with H | H | H
  · rw [H] at hm
    have hi : 2*(J-thirdR (256*q) (32*q) ((2:Int)^1)) =
        (2*J-(8*(256*q)^4+9*(256*q)^3+63*(256*q)^2))+
          (2*(16*(256*q)^4))*(7+21*q) := by
      simp only [thirdR,Int.pow_succ,Int.pow_zero]; grind only
    have hm2 := double_congruence _ _ _ _ _ hm hi
    have hx : 2*(16*(256*q)^4)=32*(256*q)^4 := by omega
    rw [hx] at hm2
    exact low_interval (256*q) J 8 9 63 (by omega) hJ hJu
      (by decide) (by decide) (by decide) hm2
  · rw [H] at hm
    have hi : 2*(J-thirdR (256*q) (32*q) ((2:Int)^2)) =
        (2*J-(18*(256*q)^4+18*(256*q)^3+72*(256*q)^2))+
          (2*(16*(256*q)^4))*(9+42*q) := by
      simp only [thirdR,Int.pow_succ,Int.pow_zero]; grind only
    have hm2 := double_congruence _ _ _ _ _ hm hi
    have hx : 2*(16*(256*q)^4)=32*(256*q)^4 := by omega
    rw [hx] at hm2
    exact low_interval (256*q) J 18 18 72 (by omega) hJ hJu
      (by decide) (by decide) (by decide) hm2
  · rw [H] at hm
    have hi : 2*(J-thirdR (256*q) (32*q) ((2:Int)^3)) =
        (2*J-(6*(256*q)^4+36*(256*q)^3+90*(256*q)^2))+
          (2*(16*(256*q)^4))*(12+84*q) := by
      simp only [thirdR,Int.pow_succ,Int.pow_zero]; grind only
    have hm2 := double_congruence _ _ _ _ _ hm hi
    have hx : 2*(16*(256*q)^4)=32*(256*q)^4 := by omega
    rw [hx] at hm2
    exact low_interval (256*q) J 6 36 90 (by omega) hJ hJu
      (by decide) (by decide) (by decide) hm2

def smallJ (c e : Nat) : Int :=
  (thirdR ((2:Int)^c) ((2:Int)^(c-3)) ((2:Int)^e)-9)%(16*((2:Int)^c)^4)

theorem small_boundary_certificate :
    ∀ (c : Fin 8) (e : Fin 33), 4≤c.val → 1≤e.val →
      let s : Int := 2^c.val
      let w : Int := 2^(c.val-3)*s
      let j := smallJ c.val e.val
      j=0 ∨ 21*w*w≤j ∨ ((21*w*w-j)*s*s-3)%7≠0 := by decide +kernel


theorem small_bound_certificate :
    ∀ (c : Fin 8), 4≤c.val →
      21*((2:Int)^(c.val-3)*(2:Int)^c.val)*((2:Int)^(c.val-3)*(2:Int)^c.val) <
        16*((2:Int)^c.val)^4 := by decide +kernel


theorem thirdR_mod_congr (s v z z' m : Int) (hz : z%m=z'%m) :
    (thirdR s v z-9)%m=(thirdR s v z'-9)%m := by
  simp [thirdR,Int.add_emod,Int.sub_emod,Int.mul_emod,hz]


theorem boundary_small_impossible (c e : Nat) (hc : 4≤c) (hcu : c≤7)
    (he : 1≤e) (j a : Int) (hj : 0<j)
    (hju : j<21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c))
    (ha : 7*a=(21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c)-j)*
      (2:Int)^c*(2:Int)^c-3)
    (hm : (j+9-thirdR ((2:Int)^c) ((2:Int)^(c-3)) ((2:Int)^e))%
      (16*((2:Int)^c)^4)=0) : False := by
  let M : Int := 16*((2:Int)^c)^4
  let E := min e (4*c+4)
  have hM : M=(2:Int)^(4*c+4) := by
    dsimp [M]
    rw [show 4*c+4=(c+c)+(c+c)+4 by omega,Int.pow_add,Int.pow_add,Int.pow_add]
    simp only [Int.pow_succ,Int.pow_zero]
    grind only
  have hcap := power_mod_cap e (4*c+4)
  rw [←hM] at hcap
  have hcon := thirdR_mod_congr ((2:Int)^c) ((2:Int)^(c-3))
    ((2:Int)^e) ((2:Int)^E) M hcap
  have hji : (j-(thirdR ((2:Int)^c) ((2:Int)^(c-3)) ((2:Int)^e)-9))%M=0 := by
    have hi : j-(thirdR ((2:Int)^c) ((2:Int)^(c-3)) ((2:Int)^e)-9)=
      j+9-thirdR ((2:Int)^c) ((2:Int)^(c-3)) ((2:Int)^e) := by omega
    rw [hi]; exact hm
  have hje := Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr hji
  rw [hcon] at hje
  have hb := small_bound_certificate ⟨c,by omega⟩ hc
  have hjM : j<M := by dsimp [M]; exact Int.lt_trans hju hb
  rw [Int.emod_eq_of_lt (by omega) hjM] at hje
  have hjs : j=smallJ c E := hje
  have hE : 1≤E := by dsimp [E]; omega
  have hEu : E<33 := by dsimp [E]; omega
  have cert := small_boundary_certificate ⟨c,by omega⟩ ⟨E,hEu⟩ hc hE
  change smallJ c E=0 ∨
    21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c)≤smallJ c E ∨
    ((21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c)-smallJ c E)*
      (2:Int)^c*(2:Int)^c-3)%7≠0 at cert
  rw [←hjs] at cert
  rcases cert with H | H | H
  · omega
  · omega
  · apply H
    rw [←ha]; simp

theorem first_family_impossible (s v z t n j u : Int)
    (hu : 0<u) (ht : t=8*u) (hs : s=16*t) (hv : s=8*v)
    (hvz : v*z=s*t) (hJ : j+9=s*s*n)
    (hn : n=32*(t*t)+18*t+27)
    (hp : narrowResidual s (v*s) z j=0) : False := by
  have hv' : v=2*t := by omega
  have hz : z=8*t := by
    have he : v*(z-8*t)=0 := by grind only
    rcases Int.mul_eq_zero.mp he with H | H
    · omega
    · omega
  have hw : v*s=32*t*t := by rw [hv',hs]; grind only
  have hj : j=proposedJ t := by
    rw [hs,hn] at hJ
    simp only [proposedJ,Int.pow_succ,Int.pow_zero]
    clear hp hvz hs hv hz hw hv' hu ht
    grind only
  rw [hw,hs,hz,hj,ht] at hp
  exact proposed_family_nonzero u hu hp

theorem boundary_large_impossible (c e : Nat) (hc : 8≤c)
    (he : 1≤e) (j a : Int) (hj : 0<j)
    (hju : j<21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c))
    (ha : 7*a=(21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c)-j)*
      (2:Int)^c*(2:Int)^c-3)
    (hp : narrowResidual ((2:Int)^c) ((2:Int)^(c-3)*(2:Int)^c) ((2:Int)^e) j=0) :
    False := by
  let s : Int := 2^c
  let v : Int := 2^(c-3)
  let z : Int := 2^e
  obtain ⟨b,hb⟩ := Nat.exists_eq_add_of_le hc
  let q : Int := 2^b
  have hq : 1≤q := pow_mono 0 b (by omega)
  have hs : s=256*q := by dsimp [s,q]; rw [hb,Int.pow_add]; rfl
  have hv : v=32*q := by
    dsimp [v,q]
    rw [show c-3=5+b by omega,Int.pow_add]; rfl
  have hsv : s=8*v := by rw [hs,hv]; omega
  have hm := third_dyadic_congruence c (c-3) e (by omega) (by omega) he j hp
  change (j+9-thirdR s v z)%(16*s^4)=0 at hm
  change j<21*(v*s)*(v*s) at hju
  have hw : 2≤v*s := by
    have H := Int.mul_le_mul (show 2≤v by omega) (show 1≤s by omega)
      (by decide) (show 0≤v by omega)
    omega
  have hw2 := Int.mul_le_mul hw hw (by decide) (show 0≤v*s by omega)
  have hJ : 0<j+9 := by omega
  have hJu : j+9<24*(v*s)*(v*s) := by grind only
  by_cases hlo : e≤3
  · have hbound : 8*(j+9)<3*s^4 := by
      rw [hsv] at hJu ⊢
      simp only [Int.pow_succ,Int.pow_zero]
      clear hp hm ha hs hv
      grind only
    rw [hs,hv] at hm
    rw [hs] at hbound
    exact boundary_low_impossible q e (j+9) hq he hlo hJ hbound hm
  · let g := e-3
    have hg : 1≤g := by dsimp [g]; omega
    have hz : z=8*(2:Int)^g := by
      change (2:Int)^e=8*(2:Int)^g
      calc
        (2:Int)^e=(2:Int)^(3+g) := congrArg (fun x:Nat => (2:Int)^x) (by dsimp [g]; omega)
        _=8*(2:Int)^g := by rw [Int.pow_add]; rfl
    have hvz : v*z=s*(2:Int)^g := by rw [hz,hsv]; grind only
    obtain ⟨n,hJn,hn,hnu,hmn⟩ := boundary_high_normalization s v z ((2:Int)^g)
      (j+9) (by omega) (by omega) (by omega) hvz hJ hJu hm
    have cases := boundary_high_classify c (c-3) g (by omega)
      (by omega) hg n hn hnu hmn
    rcases cases with ⟨hfirst,hnfirst⟩ | ⟨hsecond,hnsecond⟩
    · have ht : (2:Int)^g=8*(2*q) := by change s=16*(2:Int)^g at hfirst; omega
      exact first_family_impossible s v z ((2:Int)^g) n j (2*q)
        (by omega) ht hfirst hsv hvz hJn hnfirst hp
    · have hJJ : 8*(j+9)=((2:Int)^c)^4+216*(2:Int)^c*(2:Int)^c := by
        change s*s=16*(2:Int)^g at hsecond
        change 8*(j+9)=s^4+216*s*s
        rw [hnsecond] at hJn
        simp only [Int.pow_succ,Int.pow_zero]
        clear hp hm hmn hvz ha hs hv hsv hju hJu hnu
        grind only
      exact second_mod7_impossible c (v*s) j a hJJ ha

theorem boundary_residual_impossible (c e : Nat) (hc : 4≤c)
    (he : 1≤e) (j a : Int) (hj : 0<j)
    (hju : j<21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c))
    (ha : 7*a=(21*((2:Int)^(c-3)*(2:Int)^c)*((2:Int)^(c-3)*(2:Int)^c)-j)*
      (2:Int)^c*(2:Int)^c-3)
    (hp : narrowResidual ((2:Int)^c) ((2:Int)^(c-3)*(2:Int)^c) ((2:Int)^e) j=0) :
    False := by
  by_cases hsmall : c≤7
  · exact boundary_small_impossible c e hc hsmall he j a hj hju ha
      (third_dyadic_congruence c (c-3) e hc (by omega) he j hp)
  · exact boundary_large_impossible c e (by omega) he j a hj hju ha hp

theorem boundary_scaled_no_collision (c d e : Nat) (a : Int)
    (hcd : c+1≤d) (hdc : d+3=2*c) (ha : 0<a)
    (hab : a<3*((2:Int)^c*(2:Int)^d)*((2:Int)^c*(2:Int)^d)) :
    collisionPoly ((2:Int)^c*(2:Int)^d*(2:Int)^e) 2
      ((2:Int)^c) ((2:Int)^c*(2:Int)^d) a≠0 := by
  intro hf
  obtain ⟨j,hj,hju,hseven,hp⟩ := narrow_collision_deficit
    ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) a
    (Int.pow_pos (by decide)) ha hab hf
  have H := gap_residual_window c d e (by omega) (by omega) j hj hju hp
  have hw : (2:Int)^d=(2:Int)^(c-3)*(2:Int)^c := by
    rw [show d=(c-3)+c by omega,Int.pow_add]
  rw [hw] at hju hseven hp
  exact boundary_residual_impossible c e (by omega) H.1 j a hj hju hseven hp

theorem boundary_polynomial_no_collision (r c k : Nat) (a : Int)
    (hck : 2*c<k) (hthird : k+3=3*c) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have ht : (2:Int)^k=(2:Int)^c*(2:Int)^d := by rw [hd,Int.pow_add]
  have hu : (2:Int)^(r+1)=(2:Int)^c*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [ht,hu]
  exact boundary_scaled_no_collision c d e a (by omega) (by omega) ha hab

theorem one_inner_third_boundary_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c<k ∧ 3*c≤k+2 ∧ k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have H := one_inner_third_collision_region r c k a hc hk ha hab he
  have hthird : 3*c≤k+2 := by
    by_cases hb : 3*c≤k+2
    · exact hb
    have hp := collision_polynomial r 1 c k a he
    have habi : (a:Int)<3*(2:Int)^k*(2:Int)^k := by
      have h : (a:Int)<((3*(2^k)^2:Nat):Int) := by omega
      simpa only [Int.natCast_mul,Int.natCast_pow,
        show ((2:Nat):Int)=2 from rfl,show ((3:Nat):Int)=3 from rfl,
        Int.pow_succ,Int.pow_zero,Int.one_mul,Int.mul_assoc] using h
    exact False.elim (boundary_polynomial_no_collision r c k a H.2.1
      (by omega) (by omega) (by omega) habi hp)
  exact ⟨H.1,H.2.1,hthird,H.2.2.2⟩

theorem one_inner_third_boundary_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c<k ∧ 3*c≤k+2 ∧ k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_third_boundary_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1

#print axioms pow_mono
#print axioms small_mod_unique
#print axioms quotient_range
#print axioms single_power_classify
#print axioms two_power_classify
#print axioms boundary_high_normalization
#print axioms boundary_high_classify
#print axioms proposed_identity
#print axioms psi_mod8
#print axioms proposed_family_nonzero
#print axioms second_seven_certificate
#print axioms second_mod7_impossible
#print axioms double_congruence
#print axioms low_interval
#print axioms boundary_low_impossible
#print axioms small_boundary_certificate
#print axioms small_bound_certificate
#print axioms thirdR_mod_congr
#print axioms boundary_small_impossible
#print axioms first_family_impossible
#print axioms boundary_large_impossible
#print axioms boundary_residual_impossible
#print axioms boundary_scaled_no_collision
#print axioms boundary_polynomial_no_collision
#print axioms one_inner_third_boundary_collision_region
#print axioms one_inner_third_boundary_value_collision_region
end Sounio.ZDScalarThirdBoundary

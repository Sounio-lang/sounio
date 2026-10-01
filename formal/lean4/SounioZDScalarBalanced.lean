import SounioZDScalarBoundaryOne

/-! Uniform exclusion of 2c=k in the explicit h=1 small-count model.
This does not assert global ResidualSeparation or bibliographic priority. -/
namespace Sounio.ZDScalarBalanced
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



def smallInverse (d e : Nat) : Int :=
  match d with
  | 1 => (#[49195,47387,125179,49339,20539,61243,11579,43323,106811,102715,94523,78139,45371,110907,110907,110907,110907,110907] : Array Int).getD e 0
  | 2 => (#[77743,14383,117039,60207,77615,112431,50991,59183,75567,108335,42799,42799,42799,42799,42799,42799,42799,42799] : Array Int).getD e 0
  | 3 => (#[66751,39103,114879,4287,45247,127167,28863,94399,94399,94399,94399,94399,94399,94399,94399,94399,94399,94399] : Array Int).getD e 0
  | 4 => (#[41727,17151,99071,767,66303,66303,66303,66303,66303,66303,66303,66303,66303,66303,66303,66303,66303,66303] : Array Int).getD e 0
  | 5 => (#[68607,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071,3071] : Array Int).getD e 0
  | _ => 0

def smallRoot (d e : Nat) : Int :=
  match d with
  | 1 => (#[120687,61227,82467,22035,79347,120243,38195,5171,70195,69171,67123,63027,54835,38451,5683,71219,71219,71219] : Array Int).getD e 0
  | 2 => (#[83599,631,78407,29159,28967,28583,27815,26279,23207,17063,4775,111271,62119,94887,29351,29351,29351,29351] : Array Int).getD e 0
  | 3 => (#[94151,84183,31479,57143,108471,80055,23223,40631,75447,14007,22199,38583,71351,5815,5815,5815,5815,5815] : Array Int).getD e 0
  | 4 => (#[118295,33079,124791,46071,19703,98039,123639,43767,15095,88823,105207,6903,72439,72439,72439,72439,72439,72439] : Array Int).getD e 0
  | 5 => (#[112183,65655,103671,48631,69623,111607,64503,101367,44023,60407,93175,27639,27639,27639,27639,27639,27639,27639] : Array Int).getD e 0
  | _ => 0

def boundarySlope (w z : Int) : Int :=
  narrowResidual w w z 1-narrowResidual w w z 0

theorem boundary_affine (w z j : Int) :
    narrowResidual w w z j =
      narrowResidual w w z 0+boundarySlope w z*j := by
  simp only [boundarySlope,narrowResidual,narrowA,narrowB,narrowC,narrowD,
    Int.pow_succ,Int.pow_zero]
  grind only

set_option maxRecDepth 100000 in
set_option maxHeartbeats 8000000 in
theorem small_root_certificate :
    ∀ (d : Fin 6) (e : Fin 18), 1≤d.val →
    let w := (2:Int)^d.val
    let z := (2:Int)^e.val
    let v := smallInverse d.val e.val
    let r := smallRoot d.val e.val
    (v*boundarySlope w z)%131072=1 ∧
    narrowResidual w w z r%131072=0 ∧
    0≤r ∧ r<131072 ∧ (r=0 ∨ 21*w*w≤r) := by decide

theorem boundary_mod_congr (w z z' j m : Int) (hz : z%m=z'%m) :
    narrowResidual w w z j%m=narrowResidual w w z' j%m := by
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

theorem small_boundary_nonzero (d e : Nat) (hd : 1≤d) (hd5 : d≤5)
    (j : Int) (hj : 0<j) (hju : j<21*(2:Int)^d*(2:Int)^d) :
    narrowResidual ((2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  let w : Int := 2^d
  let z : Int := 2^(min e 17)
  have hw : w≤32 := pow_mono d 5 hd5
  have hwpos : 0<w := Int.pow_pos (by decide)
  have hw2 : w*w≤1024 := Int.mul_le_mul hw hw (by omega) (by decide)
  have hnorm : 21*w*w=21*(w*w) := Int.mul_assoc _ _ _
  change j<21*w*w at hju
  have hz := power_mod_cap e 17
  have H := boundary_mod_congr w ((2:Int)^e) z j 131072 hz
  rw [hp] at H
  have cert := small_root_certificate ⟨d,by omega⟩ ⟨min e 17,by omega⟩ hd
  dsimp only at cert
  change
    (smallInverse d (min e 17)*boundarySlope w z)%131072=1 ∧
    narrowResidual w w z (smallRoot d (min e 17))%131072=0 ∧
    0≤smallRoot d (min e 17) ∧ smallRoot d (min e 17)<131072 ∧
    (smallRoot d (min e 17)=0 ∨ 21*w*w≤smallRoot d (min e 17)) at cert
  have hr := cert.2.1
  rw [boundary_affine] at H hr
  have hm := affine_unit_residue _ _ j (smallInverse d (min e 17))
    (smallRoot d (min e 17)) 131072 H.symm hr cert.1
  have heq := small_mod_unique (smallRoot d (min e 17)) j 131072
    cert.2.2.1 cert.2.2.2.1 (by omega) (by omega) hm
  rcases cert.2.2.2.2 with h | h <;> omega


theorem boundary_factor (q z n : Int) :
    let w := 64*q
    let j := 64*n-9
    narrowResidual w w z j =
      18*w*z-(j+9)+(27+18*z)*w*w+64*w*w*
      (64*q*q*narrowA w z j+q*narrowB w z j+
        6*j*q*z-3*n-1536*q*q*z*z+1152*q*q-42*q*z) := by
  dsimp
  simp only [narrowResidual,narrowC,narrowD,Int.pow_succ,Int.pow_zero]
  grind only

theorem boundary_congruence (d e : Nat) (hd : 6≤d) (j : Int)
    (hp : narrowResidual ((2:Int)^d) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e-(27+18*(2:Int)^e)*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0 := by
  obtain ⟨q,hq⟩ := Nat.exists_eq_add_of_le hd
  have hw : (2:Int)^d=64*(2:Int)^q := by rw [hq,Int.pow_add]; rfl
  have hp' := hp
  rw [hw] at hp'
  have hm := congrArg (fun x:Int => x%64) hp'
  have hjm : (j+9)%64=0 := by
    simp [narrowResidual,narrowA,narrowB,narrowC,narrowD,
      Int.pow_succ,Int.add_emod,Int.sub_emod,Int.mul_emod] at hm ⊢
    omega
  obtain ⟨n,hn⟩ := Int.dvd_of_emod_eq_zero hjm
  have hj' : j=64*n-9 := by omega
  rw [hj'] at hp' ⊢
  have H := boundary_factor ((2:Int)^q) ((2:Int)^e) n
  dsimp only at H
  rw [hp'] at H
  rw [hw]
  apply Int.emod_eq_zero_of_dvd
  refine ⟨64*(2:Int)^q*(2:Int)^q*narrowA (64*(2:Int)^q) ((2:Int)^e) (64*n-9)+
    (2:Int)^q*narrowB (64*(2:Int)^q) ((2:Int)^e) (64*n-9)+
    6*(64*n-9)*(2:Int)^q*(2:Int)^e-3*n-
    1536*(2:Int)^q*(2:Int)^q*(2:Int)^e*(2:Int)^e+
    1152*(2:Int)^q*(2:Int)^q-42*(2:Int)^q*(2:Int)^e,?_⟩
  omega

theorem boundary_quotient_certificate :
    ∀ (f : Fin 7) (n : Fin 24), 1≤f.val → 1≤n.val →
      (n.val:Int)%64≠(18*(2:Int)^f.val+27)%64 := by decide

theorem high_boundary_impossible (d e : Nat) (J : Int)
    (hJ : 0<J) (hJu : J<24*(2:Int)^d*(2:Int)^d)
    (hmod : (J-18*(2:Int)^d*(2:Int)^e-27*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0) : False := by
  let w : Int := 2^d
  let z : Int := 2^e
  have hw : 0<w := Int.pow_pos (by decide)
  have hz : 0<z := Int.pow_pos (by decide)
  have hw2 : 0<w*w := Int.mul_pos hw hw
  change J<24*w*w at hJu
  change (J-18*w*z-27*w*w)%(64*w*w)=0 at hmod
  have hn18 : 18*w*w=18*(w*w) := Int.mul_assoc _ _ _
  have hn24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
  have hn27 : 27*w*w=27*(w*w) := Int.mul_assoc _ _ _
  have hn64 : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
  by_cases hed : e≤d
  · have hzw : z≤w := pow_mono e d hed
    have hmul := Int.mul_le_mul_of_nonneg_left hzw (show 0≤18*w by omega)
    have hmpos : 0<18*w*z := Int.mul_pos (by omega) hz
    have hm : (J-(18*w*z+27*w*w))%(64*w*w)=0 := by
      have h : J-(18*w*z+27*w*w)=J-18*w*z-27*w*w := by omega
      rw [h]; exact hmod
    have H := small_mod_unique (18*w*z+27*w*w) J (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · obtain ⟨f,hf⟩ := Nat.exists_eq_add_of_le (show d≤e by omega)
    have hfpos : 1≤f := by omega
    have hzw : z=w*(2:Int)^f := by dsimp [w,z]; rw [hf,Int.pow_add]
    obtain ⟨v,hv⟩ := Int.dvd_of_emod_eq_zero hmod
    let n : Int := 18*(2:Int)^f+27+64*v
    have he : J=(w*w)*n := by rw [hzw] at hv; dsimp [n]; grind only
    have hn := quotient_range (w*w) J n 24 hw2 hJ (by omega) he
    let nn := n.toNat
    have hnn : (nn:Int)=n := Int.toNat_of_nonneg (by omega)
    have hm : n%64=(18*(2:Int)^f+27)%64 := by dsimp [n]; simp [Int.add_emod]
    have hcap := power_mod_cap f 6
    change (2:Int)^f%64=(2:Int)^(min f 6)%64 at hcap
    have hm' : (nn:Int)%64=(18*(2:Int)^(min f 6)+27)%64 := by
      rw [hnn,hm]
      simp [Int.add_emod,Int.mul_emod,hcap]
    have hfm : 1≤min f 6 := by omega
    have hnnlo : 1≤nn := by omega
    have hnnhi : nn<24 := by omega
    exact boundary_quotient_certificate ⟨min f 6,by omega⟩ ⟨nn,hnnhi⟩ hfm hnnlo hm'

theorem high_boundary_congruence (d e : Nat) (hd : 6≤d) (he : 5≤e) (j : Int)
    (hp : narrowResidual ((2:Int)^d) ((2:Int)^d) ((2:Int)^e) j=0) :
    (j+9-18*(2:Int)^d*(2:Int)^e-27*(2:Int)^d*(2:Int)^d)%
      (64*(2:Int)^d*(2:Int)^d)=0 := by
  have H := boundary_congruence d e hd j hp
  obtain ⟨t,ht⟩ := Nat.exists_eq_add_of_le he
  have hz : (2:Int)^e=32*(2:Int)^t := by rw [ht,Int.pow_add]; rfl
  have hid : j+9-18*(2:Int)^d*(2:Int)^e-(27+18*(2:Int)^e)*(2:Int)^d*(2:Int)^d =
      (j+9-18*(2:Int)^d*(2:Int)^e-27*(2:Int)^d*(2:Int)^d)-
      (64*(2:Int)^d*(2:Int)^d)*(9*(2:Int)^t) := by rw [hz]; grind only
  rw [hid] at H
  simpa [Int.sub_emod] using H

theorem high_boundary_nonzero (d e : Nat) (hd : 6≤d) (he : 5≤e)
    (j : Int) (hj : 0<j+9) (hju : j+9<24*(2:Int)^d*(2:Int)^d) :
    narrowResidual ((2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  exact high_boundary_impossible d e (j+9) hj hju (high_boundary_congruence d e hd he j hp)

theorem low_boundary_nonzero (d e : Nat) (hd : 6≤d) (he : e≤4)
    (j : Int) (hj : 0<j+9) (hju : j+9<24*(2:Int)^d*(2:Int)^d) :
    narrowResidual ((2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  intro hp
  have H := boundary_congruence d e hd j hp
  let w : Int := 2^d
  have hw : (64:Int)≤w := pow_mono 6 d hd
  have hw2 : (4096:Int)≤w*w := Int.mul_le_mul hw hw (by decide) (by omega)
  have hww := Int.mul_le_mul_of_nonneg_left hw (show 0≤w by omega)
  have hcomm : w*64=64*w := Int.mul_comm _ _
  change j+9<24*w*w at hju
  have hn24 : 24*w*w=24*(w*w) := Int.mul_assoc _ _ _
  have hn64 : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
  have cases : e=0 ∨ e=1 ∨ e=2 ∨ e=3 ∨ e=4 := by omega
  rcases cases with he0 | he1 | he2 | he3 | he4
  · subst e
    change (j+9-18*w*1-(27+18*1)*w*w)%(64*w*w)=0 at H
    have hn : 45*w*w=45*(w*w) := Int.mul_assoc _ _ _
    have hid : j+9-18*w*1-(27+18*1)*w*w=
        (j+9-(18*w+45*w*w))-(64*w*w)*0 := by grind only
    rw [hid] at H
    have hm : (j+9-(18*w+45*w*w))%(64*w*w)=0 := by
      simpa [Int.sub_emod] using H
    have heq := small_mod_unique (18*w+45*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · subst e
    change (j+9-18*w*2-(27+18*2)*w*w)%(64*w*w)=0 at H
    have hn : 63*w*w=63*(w*w) := Int.mul_assoc _ _ _
    have hid : j+9-18*w*2-(27+18*2)*w*w=
        (j+9-(36*w+63*w*w))-(64*w*w)*0 := by grind only
    rw [hid] at H
    have hm : (j+9-(36*w+63*w*w))%(64*w*w)=0 := by
      simpa [Int.sub_emod] using H
    have heq := small_mod_unique (36*w+63*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · subst e
    change (j+9-18*w*4-(27+18*4)*w*w)%(64*w*w)=0 at H
    have hn : 35*w*w=35*(w*w) := Int.mul_assoc _ _ _
    have hid : j+9-18*w*4-(27+18*4)*w*w=
        (j+9-(72*w+35*w*w))-(64*w*w)*1 := by grind only
    rw [hid] at H
    have hm : (j+9-(72*w+35*w*w))%(64*w*w)=0 := by
      simpa [Int.sub_emod] using H
    have heq := small_mod_unique (72*w+35*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · subst e
    change (j+9-18*w*8-(27+18*8)*w*w)%(64*w*w)=0 at H
    have hn : 43*w*w=43*(w*w) := Int.mul_assoc _ _ _
    have hid : j+9-18*w*8-(27+18*8)*w*w=
        (j+9-(144*w+43*w*w))-(64*w*w)*2 := by grind only
    rw [hid] at H
    have hm : (j+9-(144*w+43*w*w))%(64*w*w)=0 := by
      simpa [Int.sub_emod] using H
    have heq := small_mod_unique (144*w+43*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
  · subst e
    change (j+9-18*w*16-(27+18*16)*w*w)%(64*w*w)=0 at H
    have hn : 59*w*w=59*(w*w) := Int.mul_assoc _ _ _
    have hid : j+9-18*w*16-(27+18*16)*w*w=
        (j+9-(288*w+59*w*w))-(64*w*w)*4 := by grind only
    rw [hid] at H
    have hm : (j+9-(288*w+59*w*w))%(64*w*w)=0 := by
      simpa [Int.sub_emod] using H
    have heq := small_mod_unique (288*w+59*w*w) (j+9) (64*w*w)
      (by omega) (by omega) (by omega) (by omega) hm
    omega
theorem balanced_residual_nonzero (d e : Nat) (hd : 1≤d)
    (j : Int) (hj : 0<j) (hju : j<21*(2:Int)^d*(2:Int)^d) :
    narrowResidual ((2:Int)^d) ((2:Int)^d) ((2:Int)^e) j≠0 := by
  by_cases hd5 : d≤5
  · exact small_boundary_nonzero d e hd hd5 j hj hju
  have hw : (2:Int)≤(2:Int)^d := pow_mono 1 d hd
  have hw2 : (4:Int)≤(2:Int)^d*(2:Int)^d := Int.mul_le_mul hw hw (by decide) (by omega)
  have hnorm21 : 21*(2:Int)^d*(2:Int)^d=21*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
  have hnorm24 : 24*(2:Int)^d*(2:Int)^d=24*((2:Int)^d*(2:Int)^d) := Int.mul_assoc _ _ _
  have hJu : j+9<24*(2:Int)^d*(2:Int)^d := by omega
  by_cases he4 : e≤4
  · exact low_boundary_nonzero d e (by omega) he4 j (by omega) hJu
  · exact high_boundary_nonzero d e (by omega) (by omega) j (by omega) hJu

theorem balanced_scaled_no_collision (d e : Nat) (a : Int)
    (hd : 1≤d) (ha : 0<a)
    (hab : a<3*((2:Int)^d*(2:Int)^d)*((2:Int)^d*(2:Int)^d)) :
    collisionPoly ((2:Int)^d*(2:Int)^d*(2:Int)^e) 2
      ((2:Int)^d) ((2:Int)^d*(2:Int)^d) a≠0 := by
  intro hf
  obtain ⟨j,hj,hju,_,hp⟩ := narrow_collision_deficit
    ((2:Int)^d) ((2:Int)^d) ((2:Int)^e) a
    (Int.pow_pos (by decide)) ha hab hf
  exact balanced_residual_nonzero d e hd j hj hju hp

theorem balanced_polynomial_no_collision (r c k : Nat) (a : Int)
    (hck : c<k) (hbal : 2*c=k) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k) :
    collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a≠0 := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have hc : c=d := by omega
  have hs : (2:Int)^c=(2:Int)^d := by rw [hc]
  have ht : (2:Int)^k=(2:Int)^d*(2:Int)^d := by rw [hd,Int.pow_add,hs]
  have hu : (2:Int)^(r+1)=(2:Int)^d*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [hs,ht,hu]
  exact balanced_scaled_no_collision d e a (by omega) ha hab

theorem balanced_numerator_no_collision (r c k a : Nat)
    (hc : 1≤c) (hck : c<k) (hbal : 2*c=k)
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
  exact balanced_polynomial_no_collision r c k a hck hbal hr (by omega) habi hp


theorem one_inner_balanced_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c<k ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := one_inner_boundary_one_collision_region r c k a hc hk ha hab he
  have hb : 2*c<k := by
    by_cases hb : 2*c<k
    · exact hb
    exact False.elim (balanced_numerator_no_collision r c k a hc H.1 (by omega) ha hab he)
  exact ⟨H.1,hb,H.2.2⟩

theorem one_inner_balanced_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c<k ∧ k≤r+1 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_balanced_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1

#print axioms pow_mono
#print axioms small_mod_unique
#print axioms quotient_range
#print axioms boundary_affine
#print axioms small_root_certificate
#print axioms boundary_mod_congr
#print axioms affine_unit_residue
#print axioms small_boundary_nonzero
#print axioms boundary_factor
#print axioms boundary_congruence
#print axioms boundary_quotient_certificate
#print axioms high_boundary_impossible
#print axioms high_boundary_congruence
#print axioms high_boundary_nonzero
#print axioms low_boundary_nonzero
#print axioms balanced_residual_nonzero
#print axioms balanced_scaled_no_collision
#print axioms balanced_polynomial_no_collision
#print axioms balanced_numerator_no_collision
#print axioms one_inner_balanced_collision_region
#print axioms one_inner_balanced_value_collision_region
end Sounio.ZDScalarBalanced

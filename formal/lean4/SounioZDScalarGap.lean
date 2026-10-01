import SounioZDScalarBalanced

/-! Uniform depth-window obstruction in the explicit h=1 small-count model.
This does not assert global ResidualSeparation or bibliographic priority. -/
namespace Sounio.ZDScalarGap
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



def gapSlope (s w z : Int) : Int :=
  8*s^4*w*w*z*z-6*s*s*(s+1)*w*z+3*s*s+1

def slopeNat (x y t : Nat) : Nat :=
  ((((((((8*t+16)*t+8)*y+((32*t+64)*t+32))*y+((32*t+64)*t+32))*x+((((64*t+128)*t+64)*y+((256*t+506)*t+250))*y+((256*t+500)*t+244)))*x+((((192*t+384)*t+192)*y+((768*t+1494)*t+726))*y+((768*t+1452)*t+687)))*x+((((256*t+512)*t+256)*y+((1024*t+1952)*t+928))*y+((1024*t+1856)*t+844)))*x+((((128*t+256)*t+128)*y+((512*t+952)*t+440))*y+((512*t+880)*t+381)))

def lowNat (x y : Nat) : Nat :=
  ((((((((84*y+672)*y+2016)*y+2688)*y+1344)*x+((((756*y+5922)*y+17388)*y+22680)*y+11088))*x+((((2520*y+19404)*y+55968)*y+71664)*y+34368))*x+((((3696*y+28038)*y+79662)*y+100446)*y+47412))*x+((((2016*y+15084)*y+42300)*y+52650)*y+24525))

def tailNat (x y t : Nat) : Nat :=
  ((((((((((16640*y+99840)*y+(772*t+249600))*y+(3088*t+332800))*y+((8*t+4632)*t+249600))*y+((16*t+3088)*t+99840))*y+((8*t+772)*t+16640))*x+((((((260864*y+1573248)*y+(12268*t+3953280))*y+(49198*t+5297662))*y+((128*t+73986)*t+3992826))*y+((256*t+49444)*t+1604730))*y+((128*t+12388)*t+268670)))*x+((((((1434624*y+8712576)*y+(70032*t+22043520))*y+(281766*t+29737320))*y+((744*t+425106)*t+22557177))*y+((1488*t+284964)*t+9121338))*y+((744*t+71592)*t+1535916)))*x+((((((3215360*y+19743744)*y+(169024*t+50485632))*y+(683152*t+68798002))*y+((1856*t+1035270)*t+52686750))*y+((3712*t+696862)*t+21495462))*y+((1856*t+175720)*t+3649490)))*x+((((((2342912*y+14702592)*y+(143104*t+38358528))*y+(582496*t+53249608))*y+((1664*t+888696)*t+41478867))*y+((3328*t+601930)*t+17187534))*y+((1664*t+152626)*t+2959467)))


theorem slope_shift (x y t : Nat) :
    gapSlope ((x:Int)+2) ((y:Int)+2) ((t:Int)+1)=(slopeNat x y t:Int) := by
  simp only [gapSlope,slopeNat,Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem gap_slope_pos (s w z : Int) (hs : 2≤s) (hw : 2≤w) (hz : 1≤z) :
    0<gapSlope s w z := by
  let x := (s-2).toNat
  let y := (w-2).toNat
  let t := (z-1).toNat
  have hx : (x:Int)=s-2 := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have ht : (t:Int)=z-1 := Int.toNat_of_nonneg (by omega)
  have H := slope_shift x y t
  have hx' : (x:Int)+2=s := by omega
  have hy' : (y:Int)+2=w := by omega
  have ht' : (t:Int)+1=z := by omega
  rw [hx',hy',ht'] at H
  have hn : 0<slopeNat x y t := by unfold slopeNat; omega
  have hi : (0:Int)<(slopeNat x y t:Int) := by omega
  omega

theorem gap_affine (s w z j J : Int) :
    narrowResidual s w z j=narrowResidual s w z J+(J-j)*gapSlope s w z := by
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,gapSlope,
    Int.pow_succ,Int.pow_zero]
  grind only

theorem low_shift (x y : Nat) :
    (-(narrowResidual ((x:Int)+2) ((y:Int)+2) 1 0))=(lowNat x y:Int) := by
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,lowNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem low_zero_negative (s w : Int) (hs : 2≤s) (hw : 2≤w) :
    narrowResidual s w 1 0<0 := by
  let x := (s-2).toNat
  let y := (w-2).toNat
  have hx : (x:Int)=s-2 := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=w-2 := Int.toNat_of_nonneg (by omega)
  have H := low_shift x y
  have hx' : (x:Int)+2=s := by omega
  have hy' : (y:Int)+2=w := by omega
  rw [hx',hy'] at H
  have hn : 0<lowNat x y := by unfold lowNat; omega
  have hi : (0:Int)<(lowNat x y:Int) := by omega
  omega

theorem gap_low_nonzero (s w j : Int) (hs : 2≤s) (hw : 2≤w) (hj : 0≤j) :
    narrowResidual s w 1 j≠0 := by
  have hL := gap_slope_pos s w 1 hs hw (by decide)
  have hlow := low_zero_negative s w hs hw
  have hid := gap_affine s w 1 j 0
  have hm := Int.mul_le_mul_of_nonneg_right (show 0-j≤0 by omega) (show 0≤gapSlope s w 1 by omega)
  omega

theorem tail_shift (x y t : Nat) :
    let s := (x:Int)+4
    let w := (y:Int)+1
    narrowResidual s w (64*w*w+(t:Int)) (21*w*w-1)=(tailNat x y t:Int) := by
  dsimp
  simp only [narrowResidual,narrowA,narrowB,narrowC,narrowD,tailNat,
    Int.natCast_add,Int.natCast_mul,Int.pow_succ,Int.pow_zero]
  grind only

theorem tail_max_positive (s w z : Int) (hs : 4≤s) (hw : 1≤w)
    (hz : 64*w*w≤z) : 0<narrowResidual s w z (21*w*w-1) := by
  let x := (s-4).toNat
  let y := (w-1).toNat
  let t := (z-64*w*w).toNat
  have hx : (x:Int)=s-4 := Int.toNat_of_nonneg (by omega)
  have hy : (y:Int)=w-1 := Int.toNat_of_nonneg (by omega)
  have ht : (t:Int)=z-64*w*w := Int.toNat_of_nonneg (by omega)
  have H := tail_shift x y t
  dsimp only at H
  have hx' : (x:Int)+4=s := by omega
  have hy' : (y:Int)+1=w := by omega
  rw [hx',hy'] at H
  have ht' : 64*w*w+(t:Int)=z := by omega
  rw [ht'] at H
  have hn : 0<tailNat x y t := by unfold tailNat; omega
  have hi : (0:Int)<(tailNat x y t:Int) := by omega
  omega

theorem gap_tail_nonzero (s w z j : Int) (hs : 4≤s) (hw : 2≤w)
    (hz : 64*w*w≤z) (hj : j<21*w*w) : narrowResidual s w z j≠0 := by
  have hw2 : 4≤w*w := Int.mul_le_mul hw hw (by decide) (by omega)
  have hnorm : 64*w*w=64*(w*w) := Int.mul_assoc _ _ _
  have hL := gap_slope_pos s w z (by omega) hw (by omega)
  have htail := tail_max_positive s w z hs (by omega) hz
  have hid := gap_affine s w z j (21*w*w-1)
  have hm := Int.mul_nonneg (show 0≤21*w*w-1-j by omega) (show 0≤gapSlope s w z by omega)
  omega

theorem gap_residual_window (c d e : Nat) (hc : 1≤c) (hd : 1≤d)
    (j : Int) (hj : 0<j) (hju : j<21*(2:Int)^d*(2:Int)^d)
    (hp : narrowResidual ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) j=0) :
    1≤e ∧ (2≤c → e≤2*d+5) := by
  have hs : (2:Int)≤(2:Int)^c := pow_mono 1 c hc
  have hw : (2:Int)≤(2:Int)^d := pow_mono 1 d hd
  constructor
  · by_cases he : 1≤e
    · exact he
    have he0 : e=0 := by omega
    rw [he0] at hp
    exact False.elim (gap_low_nonzero ((2:Int)^c) ((2:Int)^d) j hs hw (by omega) hp)
  · intro hc2
    by_cases he : e≤2*d+5
    · exact he
    have hs4 : (4:Int)≤(2:Int)^c := pow_mono 2 c hc2
    have hz := pow_mono (2*d+6) e (by omega)
    have hpw : (2:Int)^(2*d+6)=64*(2:Int)^d*(2:Int)^d := by
      rw [show 2*d+6=d+d+6 by omega,Int.pow_add,Int.pow_add]
      grind only
    rw [hpw] at hz
    exact False.elim (gap_tail_nonzero ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) j hs4 hw hz hju hp)

theorem gap_scaled_collision_window (c d e : Nat) (a : Int)
    (hc : 1≤c) (hd : 1≤d) (ha : 0<a)
    (hab : a<3*((2:Int)^c*(2:Int)^d)*((2:Int)^c*(2:Int)^d))
    (hf : collisionPoly ((2:Int)^c*(2:Int)^d*(2:Int)^e) 2
      ((2:Int)^c) ((2:Int)^c*(2:Int)^d) a=0) :
    1≤e ∧ (2≤c → e≤2*d+5) := by
  obtain ⟨j,hj,hju,_,hp⟩ := narrow_collision_deficit
    ((2:Int)^c) ((2:Int)^d) ((2:Int)^e) a (Int.pow_pos (by decide)) ha hab hf
  exact gap_residual_window c d e hc hd j hj hju hp

theorem gap_polynomial_collision_window (r c k : Nat) (a : Int)
    (hc : 1≤c) (hck : c<k) (hr : k≤r+1)
    (ha : 0<a) (hab : a<3*(2:Int)^k*(2:Int)^k)
    (hf : collisionPoly ((2:Int)^(r+1)) 2 ((2:Int)^c) ((2:Int)^k) a=0) :
    k≤r ∧ (2≤c → r+2*c≤3*k+4) := by
  obtain ⟨d,hd⟩ := Nat.exists_eq_add_of_le (show c≤k by omega)
  obtain ⟨e,he⟩ := Nat.exists_eq_add_of_le hr
  have ht : (2:Int)^k=(2:Int)^c*(2:Int)^d := by rw [hd,Int.pow_add]
  have hu : (2:Int)^(r+1)=(2:Int)^c*(2:Int)^d*(2:Int)^e := by rw [he,Int.pow_add,ht]
  rw [ht] at hab
  rw [ht,hu] at hf
  have H := gap_scaled_collision_window c d e a hc (by omega) ha hab hf
  constructor
  · omega
  · intro hc2
    have hh := H.2 hc2
    omega

theorem one_inner_gap_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*(2^k)^2)
    (he : 3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r)))) :
    c<k ∧ 2*c<k ∧ k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have H := one_inner_balanced_collision_region r c k a hc hk ha hab he
  have hp := collision_polynomial r 1 c k a he
  have habi : (a:Int)<3*(2:Int)^k*(2:Int)^k := by
    have h : (a:Int)<((3*(2^k)^2:Nat):Int) := by omega
    simpa only [Int.natCast_mul,Int.natCast_pow,
      show ((2:Nat):Int)=2 from rfl,show ((3:Nat):Int)=3 from rfl,
      Int.pow_succ,Int.pow_zero,Int.one_mul,Int.mul_assoc] using h
  have hg := gap_polynomial_collision_window r c k a hc H.1 H.2.2.1 (by omega) habi hp
  have hu : r+2*c≤3*k+4 := by
    by_cases hc2 : 2≤c
    · exact hg.2 hc2
    have hc1 : c=1 := by omega
    omega
  exact ⟨H.1,H.2.1,hg.1,hu,H.2.2.2⟩

theorem one_inner_gap_value_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : value a (4*2^j) (Code.reset (d:=((r+1)+c)+k)) =
      value a (4*2^j) (.t (twoBlock r 1 c k))) :
    c<k ∧ 2*c<k ∧ k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have H := collision_reduced (towerT 1 (zeroAt r)) c k a j hc ha hab he
  exact one_inner_gap_collision_region r c k a hc hk (by omega) H.2.2.1 H.2.1

#print axioms pow_mono
#print axioms slope_shift
#print axioms gap_slope_pos
#print axioms gap_affine
#print axioms low_shift
#print axioms low_zero_negative
#print axioms gap_low_nonzero
#print axioms tail_shift
#print axioms tail_max_positive
#print axioms gap_tail_nonzero
#print axioms gap_residual_window
#print axioms gap_scaled_collision_window
#print axioms gap_polynomial_collision_window
#print axioms one_inner_gap_collision_region
#print axioms one_inner_gap_value_collision_region
end Sounio.ZDScalarGap

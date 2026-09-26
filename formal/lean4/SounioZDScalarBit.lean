import SounioZDScalarCore

/-!
One-bit completion of scalar invariants on the full principal catalogue.
No reset-separation conjecture is assumed.
-/
namespace Sounio.ZDScalarBit
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDAlgebraBridge Sounio.ZDTwinCover
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

theorem zero_value_injective (d a b : Nat) (hg : CoreGood a b)
    (x y : Code d) (hxc : coreOrigin x=false) (hyc : coreOrigin y=false)
    (he : value a b x=value a b y) : x=y := by
  induction d generalizing a b with
  | zero => cases x <;> cases y <;> simp_all [coreOrigin]
  | succ d ih =>
    have hx := core_value_parity x hg
    have hy := core_value_parity y hg
    cases x with
    | reset => simp [coreOrigin] at hxc
    | t x =>
      cases y with
      | reset => simp [coreOrigin] at hyc
      | t y =>
        obtain ⟨s,A,B,hs,hg',hA,hB⟩ := core_pullback_t hg
        have ht (v : Code d) :
            value a b (.t v)=3*a*order d+s*value A B v := by
          simp only [value,edges,positives]
          grind
        rw [ht x,ht y] at he
        have hv : value A B x=value A B y := by
          have hm : s*value A B x=s*value A B y := by omega
          exact Nat.eq_of_mul_eq_mul_left hs hm
        exact congrArg Code.t (ih A B hg' x y hxc hyc hv)
      | c y => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
    | c x =>
      cases y with
      | reset => simp [coreOrigin] at hyc
      | t y => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
      | c y =>
        obtain ⟨s,A,B,hs,hg',hA,hB⟩ := core_pullback_c hg
        have ht (v : Code d) : value a b (.c v)=s*value A B v := by
          simp only [value,edges,positives]
          grind
        rw [ht x,ht y] at he
        exact congrArg Code.c (ih A B hg' x y hxc hyc (Nat.eq_of_mul_eq_mul_left hs he))

theorem origin_value_injective (d a b : Nat) (hg : CoreGood a b)
    (x y : Code d) (hb : coreOrigin x=coreOrigin y)
    (he : value a b x=value a b y) : x=y := by
  cases hx : coreOrigin x with
  | false =>
    have hy : coreOrigin y=false := by rw [← hb]; exact hx
    exact zero_value_injective d a b hg x y hx hy he
  | true =>
    have hy : coreOrigin y=true := by rw [← hb]; exact hx
    exact core_value_injective d a b hg x y hx hy he

def originBit (d W : Nat) : Bool := coreOrigin (labelCodes d W).1

def nativeScalar (d W a b : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) : Int :=
  2*(a:Int)*nativeEdges d W hW hW0+(b:Int)*nativeTriangles d W hW hW0

theorem native_scalar_counts (d W a b : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) :
    nativeScalar d W a b hW hW0=16*(value a b (labelCodes d W).1 : Int) := by
  have hw := actual_native_counts d W hW hW0
  simp only [nativeScalar]
  rw [hw.1,hw.2]
  simp only [value,Int.natCast_add,Int.natCast_mul]
  grind

theorem native_iso_iff_scalar_bit (d W V a b : Nat)
    (hW : W<2^(d+3)) (hW0 : W≠0) (hV : V<2^(d+3)) (hV0 : V≠0)
    (hg : CoreGood a b) :
    Nonempty (GraphIso (NativeVertex (d+3) W) (NativeVertex (d+3) V)
      (NativeAdj (d+3) W) (NativeAdj (d+3) V)) ↔
    nativeScalar d W a b hW hW0=nativeScalar d V a b hV hV0 ∧
      originBit d W=originBit d V := by
  rw [native_iso_iff_counts d W V hW hW0 hV hV0]
  constructor
  · rintro ⟨he,hp⟩
    have hw := actual_native_counts d W hW hW0
    have hv := actual_native_counts d V hV hV0
    have ec : edges (labelCodes d W).1=edges (labelCodes d V).1 := by omega
    have pc : positives (labelCodes d W).1=positives (labelCodes d V).1 := by omega
    have hc := invariant_injective _ _ ec pc
    constructor
    · simp only [nativeScalar]
      rw [he,hp]
    · simp only [originBit]
      rw [hc]
  · rintro ⟨hs,hb⟩
    have hw := actual_native_counts d W hW hW0
    have hv := actual_native_counts d V hV hV0
    rw [native_scalar_counts,native_scalar_counts] at hs
    have he : value a b (labelCodes d W).1=value a b (labelCodes d V).1 := by omega
    have hc := origin_value_injective d a b hg _ _ hb he
    rw [hc] at hw
    exact ⟨hw.1.trans hv.1.symm,hw.2.trans hv.2.symm⟩

theorem bool_third_eq (x y z : Bool) (hxy : x≠y) (hxz : x≠z) : y=z := by
  cases x <;> cases y <;> cases z <;> simp_all

theorem native_scalar_at_most_two (d W V U a b : Nat)
    (hW : W<2^(d+3)) (hW0 : W≠0) (hV : V<2^(d+3)) (hV0 : V≠0)
    (hU : U<2^(d+3)) (hU0 : U≠0) (hg : CoreGood a b)
    (hWV : nativeScalar d W a b hW hW0=nativeScalar d V a b hV hV0)
    (hWU : nativeScalar d W a b hW hW0=nativeScalar d U a b hU hU0) :
    Nonempty (GraphIso (NativeVertex (d+3) W) (NativeVertex (d+3) V)
      (NativeAdj (d+3) W) (NativeAdj (d+3) V)) ∨
    Nonempty (GraphIso (NativeVertex (d+3) W) (NativeVertex (d+3) U)
      (NativeAdj (d+3) W) (NativeAdj (d+3) U)) ∨
    Nonempty (GraphIso (NativeVertex (d+3) V) (NativeVertex (d+3) U)
      (NativeAdj (d+3) V) (NativeAdj (d+3) U)) := by
  by_cases hbWV : originBit d W=originBit d V
  · exact Or.inl ((native_iso_iff_scalar_bit d W V a b hW hW0 hV hV0 hg).mpr ⟨hWV,hbWV⟩)
  · by_cases hbWU : originBit d W=originBit d U
    · exact Or.inr (Or.inl ((native_iso_iff_scalar_bit d W U a b hW hW0 hU hU0 hg).mpr ⟨hWU,hbWU⟩))
    · have hbVU := bool_third_eq _ _ _ hbWV hbWU
      exact Or.inr (Or.inr ((native_iso_iff_scalar_bit d V U a b hV hV0 hU hU0 hg).mpr
        ⟨hWV.symm.trans hWU,hbVU⟩))

/-- The at-most-two bound is attained by 3E+tau (here scaled by 2). -/
theorem native_scalar_two_sharp :
    originBit 5 208=true ∧ originBit 5 201=false ∧
    nativeScalar 5 208 3 2 (by decide) (by decide)=
      nativeScalar 5 201 3 2 (by decide) (by decide) ∧
    ¬Nonempty (GraphIso (NativeVertex 8 208) (NativeVertex 8 201)
      (NativeAdj 8 208) (NativeAdj 8 201)) := by
  obtain ⟨hWc,hVc,mw,pw,mv,pv⟩ := core_scope_controls
  have hw := actual_native_counts 5 208 (by decide) (by decide)
  have hv := actual_native_counts 5 201 (by decide) (by decide)
  refine ⟨hWc,hVc,?_,?_⟩
  · simp only [nativeScalar]
    omega
  · intro hi
    have he := (native_iso_iff_counts 5 208 201 (by decide) (by decide) (by decide) (by decide)).mp hi
    omega

#print axioms zero_value_injective
#print axioms origin_value_injective
#print axioms native_scalar_counts
#print axioms native_iso_iff_scalar_bit
#print axioms bool_third_eq
#print axioms native_scalar_at_most_two
#print axioms native_scalar_two_sharp
end Sounio.ZDScalarBit

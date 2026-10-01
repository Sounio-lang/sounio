import SounioZDScalarInvariant

/-!
Unconditional scalar completeness on the subcatalogue whose terminal state
is K or reset. No result for arbitrary zero-origin words is assumed.
-/
namespace Sounio.ZDScalarCore
open Sounio.ZDSignedState Sounio.ZDScalarInvariant
open Sounio.ZDAlgebraBridge Sounio.ZDTwinCover
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def coreOrigin : {d : Nat} → Code d → Bool
  | _, .k => true
  | _, .z => false
  | _, .reset => true
  | _, .t x => coreOrigin x
  | _, .c x => coreOrigin x

def capacity : Nat → Nat
  | 0 => 3
  | d+1 => 3*order d+4*capacity d

theorem capacity_identity (d : Nat) :
    2*capacity d+order d=order d*order d := by
  induction d with
  | zero => decide
  | succ d ih => simp only [capacity,order]; grind

theorem capacity_reset (d : Nat) :
    edges (Code.reset (d:=d))=capacity (d+1) := by
  have h := capacity_identity d
  simp only [edges,capacity]
  grind

theorem order_scale (d : Nat) : order d+1=4*2^d := by
  induction d with
  | zero => decide
  | succ d ih => simp only [order,Nat.pow_succ]; omega

theorem capacity_scale (d : Nat) :
    capacity d+6*2^d=8*4^d+1 := by
  induction d with
  | zero => decide
  | succ d ih =>
    have h := order_scale d
    simp only [capacity,Nat.pow_succ]
    omega

theorem capacity_bounds (d : Nat) :
    3*4^d≤capacity d ∧ capacity d<8*4^d := by
  have lo : 3*4^d≤capacity d := by
    induction d with
    | zero => decide
    | succ d ih => simp only [capacity,Nat.pow_succ]; omega
  have h := capacity_scale d
  have hp := Nat.two_pow_pos d
  exact ⟨lo,by omega⟩

theorem edges_le_capacity {d : Nat} (x : Code d) : edges x≤capacity d := by
  induction x with
  | k => decide
  | z => decide
  | reset => rw [capacity_reset]; exact Nat.le_refl _
  | t x ih => simp only [edges,capacity]; omega
  | c x ih => simp only [edges,capacity]; omega

theorem core_edges_lower {d : Nat} (x : Code d) (hx : coreOrigin x=true) :
    3*4^d≤edges x := by
  induction x with
  | k => decide
  | z => simp [coreOrigin] at hx
  | @reset d => rw [capacity_reset]; exact (capacity_bounds (d+1)).1
  | t x ih =>
    have h := ih hx
    simp only [edges,Nat.pow_succ]
    omega
  | c x ih =>
    have h := ih hx
    simp only [edges,Nat.pow_succ]
    omega

theorem core_t_density {d : Nat} (x : Code d) (hx : coreOrigin x=true) :
    3*capacity (d+1)<3*edges (.t x)+4*positives (.t x) := by
  have lo := core_edges_lower x hx
  have hi := (capacity_bounds d).2
  simp only [capacity,edges,positives]
  omega


/-- Positive dyadic second coefficient and an odd first coefficient
inside the closed slope range 2*a <= 3*b. -/
def CoreGood (a b : Nat) : Prop :=
  ∃ j, b=2^j ∧ a%2=1 ∧ 2*a≤3*b

theorem core_good_properties {a b : Nat} (h : CoreGood a b) :
    a%2=1 ∧ 0<b ∧ 2*a≤3*b := by
  obtain ⟨j,rfl,ha,hb⟩ := h
  exact ⟨ha,Nat.two_pow_pos j,hb⟩

theorem core_value_parity {d a b : Nat} (x : Code d) (h : CoreGood a b) :
    value a b x%2 = if isX x then 1 else 0 := by
  have ha := (core_good_properties h).1
  cases hx : isX x <;> simp [value,Nat.add_mod,Nat.mul_mod,ha,positives_even,edges_parity,hx]

theorem core_pullback_t {a b : Nat} (h : CoreGood a b) :
    ∃ s A B, 0<s ∧ CoreGood A B ∧ 4*a+6*b=s*A ∧ 8*b=s*B := by
  obtain ⟨j,rfl,ha,hb⟩ := h
  cases j with
  | zero =>
    simp only [Nat.pow_zero] at hb ⊢
    have he : a=1 := by omega
    subst a
    exact ⟨2,5,4,by decide,⟨2,by decide,by decide,by decide⟩,by decide,by decide⟩
  | succ j =>
    cases j with
    | zero =>
      have he : a=1 ∨ a=3 := by
        have hb' : 2*a≤6 := by simpa using hb
        omega
      rcases he with rfl | rfl
      · exact ⟨16,1,1,by decide,⟨0,by decide,by decide,by decide⟩,by decide,by decide⟩
      · exact ⟨8,3,2,by decide,⟨1,by decide,by decide,by decide⟩,by decide,by decide⟩
    | succ j =>
      have hp : 2^(j+1+1)=4*2^j := by simp only [Nat.pow_succ]; omega
      rw [hp] at hb ⊢
      refine ⟨4,a+6*2^j,8*2^j,by decide,?_,?_,?_⟩
      · refine ⟨j+3,?_,?_,?_⟩
        · simp only [Nat.pow_succ]; omega
        · simpa [Nat.add_mod,Nat.mul_mod] using ha
        · omega
      · omega
      · omega

theorem core_pullback_c {a b : Nat} (h : CoreGood a b) :
    ∃ s A B, 0<s ∧ CoreGood A B ∧ 4*a=s*A ∧ 8*b=s*B := by
  obtain ⟨j,rfl,ha,hb⟩ := h
  refine ⟨4,a,2*2^j,by decide,?_,rfl,?_⟩
  · exact ⟨j+1,by simp [Nat.pow_succ,Nat.mul_comm],ha,by omega⟩
  · omega

theorem dense_arithmetic (a b q F m p : Nat)
    (hb : 0<b) (hc : 2*a≤3*b) (hd : 3*F<3*m+4*p) :
    a*(3*q+4*F)<a*(3*q+4*m)+b*(6*m+8*p) := by
  have h1 := Nat.mul_le_mul_right F hc
  have h2 := Nat.mul_lt_mul_of_pos_left hd hb
  grind

theorem reset_lt_t_of_density {d a b : Nat} (x : Code d)
    (hb : 0<b) (hc : 2*a≤3*b) (hd : 3*capacity d<3*edges x+4*positives x) :
    value a b (Code.reset (d:=d))<value a b (.t x) := by
  simp only [value,positives,Nat.mul_zero,Nat.add_zero]
  rw [capacity_reset]
  simp only [capacity,edges]
  exact dense_arithmetic a b (order d) (capacity d) (edges x) (positives x)
    hb hc hd

theorem reset_lt_t_of_full {d a b : Nat} (x : Code d)
    (hb : 0<b) (hf : edges x=capacity d) :
    value a b (Code.reset (d:=d))<value a b (.t x) := by
  have hlo := (capacity_bounds d).1
  have hp : 0<positives (.t x) := by
    have hpow : 0<4^d := Nat.pow_pos (by decide)
    simp only [positives]
    omega
  have hm := Nat.mul_pos hb hp
  simp only [value,positives,Nat.mul_zero,Nat.add_zero]
  rw [capacity_reset]
  simp only [capacity,edges,hf]
  simp only [positives,hf] at hm
  omega

theorem reset_ne_t_c {d a b : Nat} (x : Code d) (ha : a%2=1) :
    value a b (Code.reset (d:=d+1))≠value a b (.t (.c x)) := by
  intro he
  have hq := order_mod_four (d+1)
  have hq8 : order (d+1)%8=3 ∨ order (d+1)%8=7 := by omega
  have ha8 : a%8=1 ∨ a%8=3 ∨ a%8=5 ∨ a%8=7 := by omega
  have hm := congrArg (fun n : Nat => n%8) he
  have ht : value a b (.t (.c x)) =
      3*a*order (d+1)+8*(2*a*edges x+3*b*edges x+8*b*positives x) := by
    simp only [value,edges,positives]
    grind
  rw [ht] at hm
  rcases hq8 with hq8 | hq8 <;>
    rcases ha8 with ha8 | ha8 | ha8 | ha8 <;>
    simp [value,edges,positives,Nat.add_mod,Nat.mul_mod,hq8,ha8] at hm

theorem reset_ne_t_core {d a b : Nat} (x : Code d)
    (hx : coreOrigin x=true) (hg : CoreGood a b) :
    value a b (Code.reset (d:=d))≠value a b (.t x) := by
  have h := core_good_properties hg
  cases x with
  | k => exact Nat.ne_of_lt (reset_lt_t_of_full .k h.2.1 rfl)
  | z => simp [coreOrigin] at hx
  | reset => exact Nat.ne_of_lt (reset_lt_t_of_full .reset h.2.1 (capacity_reset _))
  | t x =>
    exact Nat.ne_of_lt (reset_lt_t_of_density (.t x) h.2.1 h.2.2 (core_t_density x hx))
  | c x => exact reset_ne_t_c x h.1

theorem core_value_injective (d a b : Nat) (hg : CoreGood a b)
    (x y : Code d) (hxc : coreOrigin x=true) (hyc : coreOrigin y=true)
    (he : value a b x=value a b y) : x=y := by
  induction d generalizing a b with
  | zero => cases x <;> cases y <;> simp_all [coreOrigin]
  | succ d ih =>
    have hx := core_value_parity x hg
    have hy := core_value_parity y hg
    cases x with
    | reset =>
      cases y with
      | reset => rfl
      | t y => exact False.elim (reset_ne_t_core y hyc hg he)
      | c y => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
    | t x =>
      cases y with
      | reset => exact False.elim (reset_ne_t_core x hxc hg he.symm)
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
      | reset => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
      | t y => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
      | c y =>
        obtain ⟨s,A,B,hs,hg',hA,hB⟩ := core_pullback_c hg
        have ht (v : Code d) : value a b (.c v)=s*value A B v := by
          simp only [value,edges,positives]
          grind
        rw [ht x,ht y] at he
        exact congrArg Code.c (ih A B hg' x y hxc hyc (Nat.eq_of_mul_eq_mul_left hs he))

theorem core_sum_injective {d : Nat} (x y : Code d)
    (hx : coreOrigin x=true) (hy : coreOrigin y=true)
    (he : edges x+2*positives x=edges y+2*positives y) : x=y := by
  apply core_value_injective d 1 2 ⟨1,by decide,by decide,by decide⟩ x y hx hy
  simpa [value] using he

def coreCodes (d : Nat) : List (Code d) := (codes d).filter coreOrigin

theorem coreCodes_succ (d : Nat) :
    coreCodes (d+1)=.reset :: ((coreCodes d).map Code.t ++ (coreCodes d).map Code.c) := by
  change List.filter coreOrigin (.reset :: ((codes d).map Code.t ++ (codes d).map Code.c)) = _
  rw [List.filter_cons]
  simp [coreOrigin,List.filter_append,List.filter_map,Function.comp_def,coreCodes]

theorem coreCodes_size (d : Nat) : (coreCodes d).length+1=2^(d+1) := by
  induction d with
  | zero => decide
  | succ d ih =>
    rw [coreCodes_succ]
    simp only [List.length_cons,List.length_append,List.length_map,Nat.pow_succ]
    omega

def coreXCodes (d : Nat) : List (Code d) := (xCodes d).filter coreOrigin

theorem coreXCodes_size (d : Nat) : (coreXCodes d).length=2^d := by
  cases d with
  | zero => decide
  | succ d =>
    have h := coreCodes_size d
    change (List.filter coreOrigin (xCodes (d+1))).length=_
    rw [xCodes_succ,List.filter_cons]
    simpa [coreOrigin,List.filter_map,Function.comp_def,coreCodes,Nat.add_comm] using h

theorem native_core_iso_iff_sum (d W V : Nat)
    (hW : W<2^(d+3)) (hW0 : W≠0) (hV : V<2^(d+3)) (hV0 : V≠0)
    (hWc : coreOrigin (labelCodes d W).1=true)
    (hVc : coreOrigin (labelCodes d V).1=true) :
    Nonempty (GraphIso (NativeVertex (d+3) W) (NativeVertex (d+3) V)
      (NativeAdj (d+3) W) (NativeAdj (d+3) V)) ↔
    nativeEdges d W hW hW0+nativeTriangles d W hW hW0 =
      nativeEdges d V hV hV0+nativeTriangles d V hV hV0 := by
  rw [native_iso_iff_counts d W V hW hW0 hV hV0]
  constructor
  · rintro ⟨he,hp⟩
    omega
  · intro he
    have hw := actual_native_counts d W hW hW0
    have hv := actual_native_counts d V hV hV0
    have hs : edges (labelCodes d W).1+2*positives (labelCodes d W).1 =
        edges (labelCodes d V).1+2*positives (labelCodes d V).1 := by omega
    have hc := core_sum_injective _ _ hWc hVc hs
    rw [hc] at hw
    exact ⟨hw.1.trans hv.1.symm,hw.2.trans hv.2.symm⟩


theorem core_three_sum_injective {d : Nat} (x y : Code d)
    (hx : coreOrigin x=true) (hy : coreOrigin y=true)
    (he : 3*edges x+2*positives x=3*edges y+2*positives y) : x=y := by
  apply core_value_injective d 3 2 ⟨1,by decide,by decide,by decide⟩ x y hx hy
  simpa [value] using he

theorem native_core_iso_iff_three_sum (d W V : Nat)
    (hW : W<2^(d+3)) (hW0 : W≠0) (hV : V<2^(d+3)) (hV0 : V≠0)
    (hWc : coreOrigin (labelCodes d W).1=true)
    (hVc : coreOrigin (labelCodes d V).1=true) :
    Nonempty (GraphIso (NativeVertex (d+3) W) (NativeVertex (d+3) V)
      (NativeAdj (d+3) W) (NativeAdj (d+3) V)) ↔
    3*nativeEdges d W hW hW0+nativeTriangles d W hW hW0 =
      3*nativeEdges d V hV hV0+nativeTriangles d V hV hV0 := by
  rw [native_iso_iff_counts d W V hW hW0 hV hV0]
  constructor
  · rintro ⟨he,hp⟩
    omega
  · intro he
    have hw := actual_native_counts d W hW hW0
    have hv := actual_native_counts d V hV hV0
    have hs : 3*edges (labelCodes d W).1+2*positives (labelCodes d W).1 =
        3*edges (labelCodes d V).1+2*positives (labelCodes d V).1 := by omega
    have hc := core_three_sum_injective _ _ hWc hVc hs
    rw [hc] at hw
    exact ⟨hw.1.trans hv.1.symm,hw.2.trans hv.2.symm⟩

/-- At dimension 64, 4E+tau already fails inside the core subfamily. -/
theorem native_core_four_weight_counterexample :
    coreOrigin (labelCodes 2 16).1=true ∧
    coreOrigin (labelCodes 2 25).1=true ∧
    4*nativeEdges 2 16 (by decide) (by decide)+nativeTriangles 2 16 (by decide) (by decide) =
      4*nativeEdges 2 25 (by decide) (by decide)+nativeTriangles 2 25 (by decide) (by decide) ∧
    ¬Nonempty (GraphIso (NativeVertex 5 16) (NativeVertex 5 25)
      (NativeAdj 5 16) (NativeAdj 5 25)) := by
  have h16 := actual_native_counts 2 16 (by decide) (by decide)
  have h25 := actual_native_counts 2 25 (by decide) (by decide)
  have c16 : edges (labelCodes 2 16).1=105 ∧ positives (labelCodes 2 16).1=0 := by decide +kernel
  have c25 : edges (labelCodes 2 25).1=69 ∧ positives (labelCodes 2 25).1=72 := by decide +kernel
  refine ⟨by decide +kernel,by decide +kernel,?_,?_⟩
  · omega
  · intro hi
    have he := (native_iso_iff_counts 2 16 25 (by decide) (by decide) (by decide) (by decide)).mp hi
    omega


theorem native_core_iso_iff_weight (d W V a b : Nat)
    (hW : W<2^(d+3)) (hW0 : W≠0) (hV : V<2^(d+3)) (hV0 : V≠0)
    (hWc : coreOrigin (labelCodes d W).1=true)
    (hVc : coreOrigin (labelCodes d V).1=true) (hg : CoreGood a b) :
    Nonempty (GraphIso (NativeVertex (d+3) W) (NativeVertex (d+3) V)
      (NativeAdj (d+3) W) (NativeAdj (d+3) V)) ↔
    2*(a:Int)*nativeEdges d W hW hW0+(b:Int)*nativeTriangles d W hW hW0 =
      2*(a:Int)*nativeEdges d V hV hV0+(b:Int)*nativeTriangles d V hV hV0 := by
  rw [native_iso_iff_counts d W V hW hW0 hV hV0]
  constructor
  · rintro ⟨he,hp⟩
    rw [he,hp]
  · intro he
    have hw := actual_native_counts d W hW hW0
    have hv := actual_native_counts d V hV hV0
    have ew :
        2*(a:Int)*nativeEdges d W hW hW0+(b:Int)*nativeTriangles d W hW hW0 =
          16*(value a b (labelCodes d W).1 : Int) := by
      rw [hw.1,hw.2]
      simp only [value,Int.natCast_add,Int.natCast_mul]
      grind
    have ev :
        2*(a:Int)*nativeEdges d V hV hV0+(b:Int)*nativeTriangles d V hV hV0 =
          16*(value a b (labelCodes d V).1 : Int) := by
      rw [hv.1,hv.2]
      simp only [value,Int.natCast_add,Int.natCast_mul]
      grind
    rw [ew,ev] at he
    have hval : value a b (labelCodes d W).1=value a b (labelCodes d V).1 := by omega
    have hc := core_value_injective d a b hg _ _ hWc hVc hval
    rw [hc] at hw
    exact ⟨hw.1.trans hv.1.symm,hw.2.trans hv.2.symm⟩

theorem core_scope_controls :
    coreOrigin (labelCodes 5 208).1=true ∧
    coreOrigin (labelCodes 5 201).1=false ∧
    edges (labelCodes 5 208).1=7629 ∧ positives (labelCodes 5 208).1=51480 ∧
    edges (labelCodes 5 201).1=4557 ∧ positives (labelCodes 5 201).1=56088 := by
  decide +kernel

theorem coreXCodes_membership {d : Nat} (x : Code d) :
    x ∈ coreXCodes d ↔ isX x=true ∧ coreOrigin x=true := by
  simp [coreXCodes,xCodes,codes_complete,and_comm]

theorem coreXCodes_nodup (d : Nat) : (coreXCodes d).Nodup := by
  exact ((codes_nodup d).filter isX).filter coreOrigin

#print axioms capacity_identity
#print axioms capacity_reset
#print axioms order_scale
#print axioms capacity_scale
#print axioms capacity_bounds
#print axioms edges_le_capacity
#print axioms core_edges_lower
#print axioms core_t_density
#print axioms core_good_properties
#print axioms core_value_parity
#print axioms core_pullback_t
#print axioms core_pullback_c
#print axioms dense_arithmetic
#print axioms reset_lt_t_of_density
#print axioms reset_lt_t_of_full
#print axioms reset_ne_t_c
#print axioms reset_ne_t_core
#print axioms core_value_injective
#print axioms core_sum_injective
#print axioms coreCodes_succ
#print axioms coreCodes_size
#print axioms coreXCodes_size
#print axioms native_core_iso_iff_sum
#print axioms core_three_sum_injective
#print axioms native_core_iso_iff_three_sum
#print axioms native_core_four_weight_counterexample
#print axioms native_core_iso_iff_weight
#print axioms core_scope_controls
#print axioms coreXCodes_membership
#print axioms coreXCodes_nodup
end Sounio.ZDScalarCore

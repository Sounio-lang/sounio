import SounioZDSignedState

/-!
Reduction of scalar completeness to one explicit reset-versus-T obligation.
The obligation is a hypothesis, not an axiom and not a proved theorem.
-/
namespace Sounio.ZDScalarInvariant
open Sounio.ZDSignedState Sounio.ZDAlgebraBridge Sounio.ZDTwinCover
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def value {d : Nat} (a b : Nat) (x : Code d) : Nat :=
  a*edges x+b*positives x

def Cone (a b : Nat) : Prop :=
  ∃ j, b=4*2^j ∧ a%2=1 ∧ a%3≠0 ∧ a<6*2^j

def Good (a b : Nat) : Prop :=
  (a=1 ∧ b=1) ∨ (a=1 ∧ b=2) ∨ Cone a b

theorem order_mod_four (d : Nat) : order d%4=3 := by
  induction d with
  | zero => decide
  | succ d ih => simp [order,Nat.add_mod,Nat.mul_mod,ih]

theorem reset_mod_eight (d : Nat) :
    edges (Code.reset (d:=d))%8=(3*order d+4)%8 := by
  have h := order_mod_four d
  have he : order d%8=3 ∨ order d%8=7 := by omega
  rcases he with he | he <;> simp [edges,Nat.add_mod,Nat.mul_mod,he]

theorem edges_mod_four {d : Nat} (x : Code d) :
    edges x%2=1 ∨ edges x%4=0 := by
  cases x with
  | k => decide
  | z => decide
  | reset => left; simp [edges,Nat.mul_mod,Nat.add_mod,order_odd]
  | t x => left; simp [edges,Nat.mul_mod,Nat.add_mod,order_odd]
  | c x => right; simp [edges]

theorem positives_even {d : Nat} (x : Code d) : positives x%2=0 := by
  cases x <;> simp [positives,Nat.add_mod,Nat.mul_mod]

/-- The (m+p) branch needs only residues modulo 4 and 8. -/
theorem reset_ne_sum_one {d : Nat} (x : Code d) :
    value 1 1 (Code.reset (d:=d)) ≠ value 1 1 (.t x) := by
  intro h
  have h8 := reset_mod_eight d
  have hm := edges_mod_four x
  simp only [value,edges,positives,Nat.one_mul,Nat.add_zero] at h
  simp only [edges] at h8
  omega

/-- The (m+2p) branch is separated by residues modulo 8. -/
theorem reset_ne_sum_two {d : Nat} (x : Code d) :
    value 1 2 (Code.reset (d:=d)) ≠ value 1 2 (.t x) := by
  intro h
  have h8 := reset_mod_eight d
  simp only [value,edges,positives,Nat.one_mul,Nat.mul_zero,Nat.add_zero] at h
  simp only [edges] at h8
  omega

theorem good_odd {a b : Nat} (h : Good a b) : a%2=1 := by
  rcases h with ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩ | ⟨j,_,ha,_,_⟩
  · decide
  · decide
  · exact ha

theorem value_parity {d a b : Nat} (x : Code d) (h : Good a b) :
    value a b x%2 = if isX x then 1 else 0 := by
  cases hc : isX x <;> simp [value,Nat.add_mod,Nat.mul_mod,good_odd h,
    positives_even,edges_parity,hc]

theorem cone_seeds : Cone 1 4 ∧ Cone 5 4 := by
  constructor
  · exact ⟨0,by decide,by decide,by decide,by decide⟩
  · exact ⟨0,by decide,by decide,by decide,by decide⟩

theorem pullback_t {a b : Nat} (h : Good a b) :
    ∃ s A B, 0<s ∧ Good A B ∧ 4*a+6*b=s*A ∧ 8*b=s*B := by
  rcases h with ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩ | ⟨j,rfl,ha,h3,hlt⟩
  · exact ⟨2,5,4,by decide,Or.inr (Or.inr cone_seeds.2),by decide,by decide⟩
  · exact ⟨16,1,1,by decide,Or.inl ⟨rfl,rfl⟩,by decide,by decide⟩
  · refine ⟨4,a+6*2^j,8*2^j,by decide,Or.inr (Or.inr ?_),?_,?_⟩
    · refine ⟨j+1,?_,?_,?_,?_⟩
      · simp [Nat.pow_succ]; omega
      · simpa [Nat.add_mod,Nat.mul_mod] using ha
      · simpa [Nat.add_mod,Nat.mul_mod] using h3
      · simp only [Nat.pow_succ]; omega
    · omega
    · omega

theorem pullback_c {a b : Nat} (h : Good a b) :
    ∃ s A B, 0<s ∧ Good A B ∧ 4*a=s*A ∧ 8*b=s*B := by
  rcases h with ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩ | ⟨j,rfl,ha,h3,hlt⟩
  · exact ⟨4,1,2,by decide,Or.inr (Or.inl ⟨rfl,rfl⟩),by decide,by decide⟩
  · exact ⟨4,1,4,by decide,Or.inr (Or.inr cone_seeds.1),by decide,by decide⟩
  · refine ⟨4,a,8*2^j,by decide,Or.inr (Or.inr ?_),?_,?_⟩
    · refine ⟨j+1,?_,ha,h3,?_⟩
      · simp [Nat.pow_succ]; omega
      · simp only [Nat.pow_succ]; omega
    · rfl
    · omega

/-- This is the remaining mathematical obligation. It is NOT asserted. -/
def ResetSeparation : Prop :=
  ∀ (d a b : Nat) (x : Code d), Cone a b →
    value a b (Code.reset (d:=d)) ≠ value a b (.t x)

theorem reset_ne_t (hsep : ResetSeparation) {d a b : Nat}
    (x : Code d) (h : Good a b) :
    value a b (Code.reset (d:=d)) ≠ value a b (.t x) := by
  rcases h with ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩ | hc
  · exact reset_ne_sum_one x
  · exact reset_ne_sum_two x
  · exact hsep d a b x hc

theorem injective_of_reset_separation (hsep : ResetSeparation)
    (d a b : Nat) (hg : Good a b) (x y : Code d)
    (he : value a b x=value a b y) : x=y := by
  induction d generalizing a b with
  | zero =>
    have hp := good_odd hg
    cases x <;> cases y <;> simp_all [value,edges,positives]
    all_goals omega
  | succ d ih =>
    have hx := value_parity x hg
    have hy := value_parity y hg
    cases x with
    | reset =>
      cases y with
      | reset => rfl
      | t y => exact False.elim (reset_ne_t hsep y hg he)
      | c y => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
    | t x =>
      cases y with
      | reset => exact False.elim (reset_ne_t hsep x hg he.symm)
      | t y =>
        obtain ⟨s,A,B,hs,hg',hA,hB⟩ := pullback_t hg
        have ht (v : Code d) :
            value a b (.t v)=3*a*order d+s*value A B v := by
          simp only [value,edges,positives]
          grind
        rw [ht x,ht y] at he
        have hv : value A B x=value A B y := by
          have hm : s*value A B x=s*value A B y := by omega
          exact Nat.eq_of_mul_eq_mul_left hs hm
        exact congrArg Code.t (ih A B hg' x y hv)
      | c y => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
    | c x =>
      cases y with
      | reset => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
      | t y => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at hx hy; omega
      | c y =>
        obtain ⟨s,A,B,hs,hg',hA,hB⟩ := pullback_c hg
        have ht (v : Code d) : value a b (.c v)=s*value A B v := by
          simp only [value,edges,positives]
          grind
        rw [ht x,ht y] at he
        have hv := Nat.eq_of_mul_eq_mul_left hs he
        exact congrArg Code.c (ih A B hg' x y hv)

theorem sum_injective_of_reset_separation (hsep : ResetSeparation)
    {d : Nat} (x y : Code d)
    (h : edges x+2*positives x=edges y+2*positives y) : x=y := by
  apply injective_of_reset_separation hsep d 1 2 (Or.inr (Or.inl ⟨rfl,rfl⟩)) x y
  simpa [value] using h


theorem native_iso_iff_sum_of_reset_separation (hsep : ResetSeparation)
    (d W V : Nat) (hW : W<2^(d+3)) (hW0 : W≠0)
    (hV : V<2^(d+3)) (hV0 : V≠0) :
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
    have hc := sum_injective_of_reset_separation hsep _ _ hs
    rw [hc] at hw
    exact ⟨hw.1.trans hv.1.symm,hw.2.trans hv.2.symm⟩

#print axioms order_mod_four
#print axioms reset_mod_eight
#print axioms edges_mod_four
#print axioms positives_even
#print axioms reset_ne_sum_one
#print axioms reset_ne_sum_two
#print axioms good_odd
#print axioms value_parity
#print axioms cone_seeds
#print axioms pullback_t
#print axioms pullback_c
#print axioms reset_ne_t
#print axioms injective_of_reset_separation
#print axioms sum_injective_of_reset_separation
#print axioms native_iso_iff_sum_of_reset_separation
end Sounio.ZDScalarInvariant

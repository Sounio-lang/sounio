import SounioZDSignedState

/-!
Exact obstructions to replacing the complete pair of counts by weaker
statistics. No injectivity theorem for edges + triangles is asserted.
-/
namespace Sounio.ZDMinimalInvariant
open Sounio.ZDSignedState Sounio.ZDAlgebraBridge Sounio.ZDTwinCover
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def weakA : (k : Nat) → Code (k+2)
  | 0 => .t .reset
  | k+1 => .t (weakA k)

def weakB : (k : Nat) → Code (k+2)
  | 0 => .t (.t .k)
  | k+1 => .t (weakB k)

theorem weak_modes (k : Nat) :
    isX (weakA k)=true ∧ isX (weakB k)=true := by
  cases k <;> exact ⟨rfl,rfl⟩

theorem weak_counts (k : Nat) :
    edges (weakA k)=edges (weakB k) ∧
    0<positives (weakA k) ∧
    positives (weakA k)<positives (weakB k) := by
  induction k with
  | zero => decide +kernel
  | succ k ih =>
    simp only [weakA,weakB,edges,positives]
    omega

def weightedA : (k : Nat) → Code (k+5)
  | 0 => .t (.c (.t .reset))
  | k+1 => .t (weightedA k)

def weightedB : (k : Nat) → Code (k+5)
  | 0 => .t (.c (.t (.t (.t .z))))
  | k+1 => .t (weightedB k)

theorem weighted_modes (k : Nat) :
    isX (weightedA k)=true ∧ isX (weightedB k)=true := by
  cases k <;> exact ⟨rfl,rfl⟩

theorem weighted_counts (k : Nat) :
    3*edges (weightedA k)+2*positives (weightedA k)=
      3*edges (weightedB k)+2*positives (weightedB k) ∧
    edges (weightedB k)<edges (weightedA k) := by
  induction k with
  | zero => decide +kernel
  | succ k ih =>
    simp only [weightedA,weightedB,edges,positives]
    omega

theorem native_counts_of_code {d : Nat} (a : Code d) (hx : isX a=true) :
    nativeEdges d (labelFor a) (labelFor_valid a).1 (labelFor_valid a).2 =
      8*(edges a : Int) ∧
    nativeTriangles d (labelFor a) (labelFor_valid a).1 (labelFor_valid a).2 =
      16*(positives a : Int) := by
  have hc := labelFor_realizes a
  simp only [hx,ite_true] at hc
  have hn := actual_native_counts d (labelFor a) (labelFor_valid a).1 (labelFor_valid a).2
  rw [hc] at hn
  exact hn

/-- At every N=k+6, equal edges and a positive triangle count do not determine
the native graph type. Labels and two-sided graph semantics are inherited from
the certified algebra-to-graph bridge. -/
theorem native_triangle_presence_counterexamples (k : Nat) :
    ∃ W V, ∃ hW : W<2^(k+2+3), ∃ hW0 : W≠0,
      ∃ hV : V<2^(k+2+3), ∃ hV0 : V≠0,
      nativeEdges (k+2) W hW hW0=nativeEdges (k+2) V hV hV0 ∧
      0<nativeTriangles (k+2) W hW hW0 ∧
      0<nativeTriangles (k+2) V hV hV0 ∧
      ¬Nonempty (GraphIso (NativeVertex (k+2+3) W) (NativeVertex (k+2+3) V)
        (NativeAdj (k+2+3) W) (NativeAdj (k+2+3) V)) := by
  let a := weakA k
  let b := weakB k
  have hm := weak_modes k
  have h := weak_counts k
  have ha := native_counts_of_code a hm.1
  have hb := native_counts_of_code b hm.2
  refine ⟨labelFor a,labelFor b,(labelFor_valid a).1,(labelFor_valid a).2,
    (labelFor_valid b).1,(labelFor_valid b).2,?_,?_,?_,?_⟩
  · rw [ha.1,hb.1]
    dsimp only [a,b]
    omega
  · rw [ha.2]
    dsimp only [a]
    omega
  · rw [hb.2]
    dsimp only [b]
    omega
  · intro hi
    have he := (native_iso_iff_counts (k+2) (labelFor a) (labelFor b)
      (labelFor_valid a).1 (labelFor_valid a).2
      (labelFor_valid b).1 (labelFor_valid b).2).mp hi
    dsimp [a,b] at ha hb he
    omega

/-- The fixed scalar 3*edges + triangles fails in every N=k+9. -/
theorem native_weighted_sum_counterexamples (k : Nat) :
    ∃ W V, ∃ hW : W<2^(k+5+3), ∃ hW0 : W≠0,
      ∃ hV : V<2^(k+5+3), ∃ hV0 : V≠0,
      3*nativeEdges (k+5) W hW hW0+nativeTriangles (k+5) W hW hW0 =
        3*nativeEdges (k+5) V hV hV0+nativeTriangles (k+5) V hV hV0 ∧
      ¬Nonempty (GraphIso (NativeVertex (k+5+3) W) (NativeVertex (k+5+3) V)
        (NativeAdj (k+5+3) W) (NativeAdj (k+5+3) V)) := by
  let a := weightedA k
  let b := weightedB k
  have hm := weighted_modes k
  have h := weighted_counts k
  have ha := native_counts_of_code a hm.1
  have hb := native_counts_of_code b hm.2
  refine ⟨labelFor a,labelFor b,(labelFor_valid a).1,(labelFor_valid a).2,
    (labelFor_valid b).1,(labelFor_valid b).2,?_,?_⟩
  · rw [ha.1,ha.2,hb.1,hb.2]
    dsimp only [a,b]
    omega
  · intro hi
    have he := (native_iso_iff_counts (k+5) (labelFor a) (labelFor b)
      (labelFor_valid a).1 (labelFor_valid a).2
      (labelFor_valid b).1 (labelFor_valid b).2).mp hi
    dsimp [a,b] at ha hb he
    omega

theorem concrete_labels :
    labelFor (weakA 0)=8 ∧ labelFor (weakB 0)=1 ∧
    labelFor (weightedA 0)=208 ∧ labelFor (weightedB 0)=201 := by
  decide +kernel

#print axioms weak_modes
#print axioms weighted_modes
#print axioms weak_counts
#print axioms weighted_counts
#print axioms native_counts_of_code
#print axioms native_triangle_presence_counterexamples
#print axioms native_weighted_sum_counterexamples
#print axioms concrete_labels
end Sounio.ZDMinimalInvariant

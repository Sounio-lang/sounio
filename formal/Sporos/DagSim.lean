/-
The DAG semantics simulates the list semantics. `α` reads a DAG state as a list
state, each node standing for its leaves; `dstep_sim` says a DAG step is a list
step of the state it stands for.

This file, part 1: what `build` allocates and what it stands for.
-/
import Sporos.DagStep

namespace Sporos

/-- `ar'` extends `ar`. -/
def Ext (ar ar' : Arena) : Prop := ∃ x, ar' = ar ++ x

theorem Ext.refl (ar : Arena) : Ext ar ar := ⟨[], by simp⟩

theorem Ext.trans {a b c : Arena} (h1 : Ext a b) (h2 : Ext b c) : Ext a c := by
  obtain ⟨x, rfl⟩ := h1; obtain ⟨y, rfl⟩ := h2; exact ⟨x ++ y, by simp⟩

theorem Ext.len {a b : Arena} (h : Ext a b) : a.length ≤ b.length := by
  obtain ⟨x, rfl⟩ := h; simp

/-- A good arena: joins point backwards, node 0 is `empty`. -/
def Good (ar : Arena) : Prop := AWF ar ∧ ar[0]? = some .empty

theorem Good.leaves_ext {a b : Arena} (ha : Good a) (h : Ext a b) {i : Nat} (hi : i < a.length) :
    leaves b i = leaves a i := by
  obtain ⟨x, rfl⟩ := h; exact leaves_append ha.1 x hi

/-- Every node a plan names exists. -/
def PE.ok (n : Nat) : PE → Prop
  | .nil => True
  | .node i => i < n
  | .leaf _ => True
  | .app a b => a.ok n ∧ b.ok n

theorem PE.ok_mono {e : PE} {n m : Nat} (h : e.ok n) (hm : n ≤ m) : e.ok m := by
  induction e with
  | nil => trivial
  | node i => exact Nat.lt_of_lt_of_le h hm
  | leaf _ => trivial
  | app a b iha ihb => exact ⟨iha h.1, ihb h.2⟩

theorem PE.den_ext {e : PE} {a b : Arena} (ha : Good a) (h : Ext a b) (he : e.ok a.length) :
    e.den (leaves b) = e.den (leaves a) := by
  induction e with
  | nil => rfl
  | node i => exact ha.leaves_ext h he
  | leaf _ => rfl
  | app x y ihx ihy => simp only [PE.den]; rw [ihx he.1, ihy he.2]

theorem build_spec : ∀ (e : PE) (ar : Arena), Good ar → e.ok ar.length →
    Good (build e ar).2 ∧ Ext ar (build e ar).2 ∧ (build e ar).1 < (build e ar).2.length ∧
      leaves (build e ar).2 (build e ar).1 = e.den (leaves ar)
  | .nil, ar, hg, _ => by
    refine ⟨hg, Ext.refl _, ?_, ?_⟩
    · have := hg.2; simp only [build]
      rcases ar with _ | ⟨_, _⟩ <;> simp_all
    · simp only [build, PE.den]; exact leaves_empty hg.2
  | .node n, ar, hg, he => ⟨hg, Ext.refl _, he, rfl⟩
  | .leaf o, ar, hg, _ => by
    refine ⟨⟨hg.1.push_leaf o, ?_⟩, ⟨_, rfl⟩, by simp [build], leaves_leaf o⟩
    have h0 := hg.2
    have : 0 < ar.length := by rcases ar with _ | _ <;> simp_all
    simp only [build]; rw [List.getElem?_append_left this]; exact h0
  | .app x y, ar, hg, he => by
    obtain ⟨g1, e1, l1, d1⟩ := build_spec x ar hg he.1
    obtain ⟨g2, e2, l2, d2⟩ := build_spec y (build x ar).2 g1 (PE.ok_mono he.2 e1.len)
    have hlt1 : (build x ar).1 < (build y (build x ar).2).2.length := Nat.lt_of_lt_of_le l1 e2.len
    simp only [build]
    refine ⟨⟨g2.1.push_join hlt1 l2, ?_⟩, Ext.trans (Ext.trans e1 e2) ⟨_, rfl⟩, by simp, ?_⟩
    · have : 0 < (build y (build x ar).2).2.length := by
        have := g2.2; rcases h : (build y (build x ar).2).2 with _ | _ <;> simp_all
      rw [List.getElem?_append_left this]; exact g2.2
    · rw [leaves_join g2.1 hlt1 l2, d2, g1.leaves_ext e2 l1, d1]
      simp only [PE.den]
      rw [PE.den_ext hg e1 he.2]

end Sporos

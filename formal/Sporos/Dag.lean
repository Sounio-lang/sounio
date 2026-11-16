/-
Provenance as a union DAG (decision 20). A provenance is a node of an
append-only arena: `empty`, a `leaf` holding one origin, or the `join` of two
earlier nodes. Its set is the list of leaves reached from it, left to right.
A union costs one node, whatever the size of the sets.

`leaves_append`: allocating new nodes never changes what an existing node means.
This is what the simulation of the list semantics (`Sporos.DagSim`) rests on.
-/
import Sporos.Core

namespace Sporos

inductive DNode where
  | empty
  | leaf (o : Origin)
  | join (a b : Nat)
deriving DecidableEq, Repr

abbrev Arena := List DNode

def lvs (ar : Arena) : Nat → Nat → List Origin
  | 0, _ => []
  | f + 1, i =>
    match ar[i]? with
    | some (.leaf o) => [o]
    | some (.join a b) => lvs ar f a ++ lvs ar f b
    | _ => []

/-- The origins a node stands for. -/
def leaves (ar : Arena) (i : Nat) : List Origin := lvs ar (i + 1) i

/-- Every join points at earlier nodes. -/
def AWF (ar : Arena) : Prop := ∀ i a b, ar[i]? = some (DNode.join a b) → a < i ∧ b < i

theorem lvs_fuel {ar : Arena} (hw : AWF ar) :
    ∀ (f g i : Nat), i < f → i < g → lvs ar f i = lvs ar g i := by
  intro f
  induction f with
  | zero => intro g i h; omega
  | succ f ih =>
    intro g i hf hg
    cases g with
    | zero => omega
    | succ g =>
      simp only [lvs]
      cases hi : ar[i]? with
      | none => rfl
      | some nd =>
        cases nd with
        | empty => rfl
        | leaf o => rfl
        | join a b =>
          obtain ⟨ha, hb⟩ := hw i a b hi
          dsimp only
          rw [ih g a (by omega) (by omega), ih g b (by omega) (by omega)]

theorem lvs_append {ar : Arena} (hw : AWF ar) (x : Arena) :
    ∀ (f i : Nat), i < ar.length → lvs (ar ++ x) f i = lvs ar f i := by
  intro f
  induction f with
  | zero => intro i _; rfl
  | succ f ih =>
    intro i hi
    simp only [lvs]
    rw [List.getElem?_append_left hi]
    cases hn : ar[i]? with
    | none => rfl
    | some nd =>
      cases nd with
      | empty => rfl
      | leaf o => rfl
      | join a b =>
        obtain ⟨ha, hb⟩ := hw i a b hn
        dsimp only
        rw [ih a (by omega), ih b (by omega)]

theorem leaves_append {ar : Arena} (hw : AWF ar) (x : Arena) {i : Nat} (hi : i < ar.length) :
    leaves (ar ++ x) i = leaves ar i := lvs_append hw x (i + 1) i hi

theorem AWF.push_leaf {ar : Arena} (hw : AWF ar) (o : Origin) : AWF (ar ++ [.leaf o]) := by
  intro i a b h
  by_cases hi : i < ar.length
  · rw [List.getElem?_append_left hi] at h; exact hw i a b h
  · rw [List.getElem?_append_right (by omega)] at h
    cases e : i - ar.length with
    | zero => rw [e] at h; simp at h
    | succ k => rw [e] at h; simp at h

theorem AWF.push_join {ar : Arena} (hw : AWF ar) {a b : Nat} (ha : a < ar.length) (hb : b < ar.length) :
    AWF (ar ++ [.join a b]) := by
  intro i a' b' h
  by_cases hi : i < ar.length
  · rw [List.getElem?_append_left hi] at h; exact hw i a' b' h
  · rw [List.getElem?_append_right (by omega)] at h
    cases e : i - ar.length with
    | zero =>
      rw [e] at h; simp at h; obtain ⟨rfl, rfl⟩ := h; omega
    | succ k => rw [e] at h; simp at h

theorem leaves_leaf {ar : Arena} (o : Origin) : leaves (ar ++ [.leaf o]) ar.length = [o] := by
  have h1 : (ar ++ [DNode.leaf o])[ar.length]? = some (DNode.leaf o) := by simp
  show lvs _ (ar.length + 1) ar.length = _
  simp only [lvs, h1]

theorem leaves_join {ar : Arena} (hw : AWF ar) {a b : Nat} (ha : a < ar.length) (hb : b < ar.length) :
    leaves (ar ++ [.join a b]) ar.length = leaves ar a ++ leaves ar b := by
  have h1 : (ar ++ [DNode.join a b])[ar.length]? = some (DNode.join a b) := by simp
  show lvs _ (ar.length + 1) ar.length = _
  simp only [lvs, h1]
  rw [lvs_append hw [.join a b] _ a ha, lvs_append hw [.join a b] _ b hb]
  rw [lvs_fuel hw ar.length (a + 1) a (by omega) (by omega),
      lvs_fuel hw ar.length (b + 1) b (by omega) (by omega)]
  rfl

theorem leaves_empty {ar : Arena} {i : Nat} (h : ar[i]? = some .empty) : leaves ar i = [] := by
  simp [leaves, lvs, h]

end Sporos

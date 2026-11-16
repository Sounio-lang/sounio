/-
Hash-consing (decision 36): an allocator that reuses a node already holding the
same origin or the same union, and appends only when there is none. It meets
`Alloc`'s specification, so `dstep_sim` and `drun_init` hold for it unchanged.

The reference search here is linear; the interpreter keeps a hash table. Any
search that returns an index holding the node, or nothing, meets the same
specification, so the table need not find every duplicate.
-/
import Sporos.DagRun

namespace Sporos

/-- The first index at or after `i` holding `nd`. -/
def findN : Arena → DNode → Nat → Option Nat
  | [], _, _ => none
  | x :: r, nd, i => if x = nd then some i else findN r nd (i + 1)

theorem findN_spec : ∀ (l : Arena) (nd : DNode) (i k : Nat), findN l nd i = some k →
    i ≤ k ∧ l[k - i]? = some nd
  | [], _, _, _, h => by simp [findN] at h
  | x :: r, nd, i, k, h => by
    simp only [findN] at h
    split at h
    · rename_i hx
      simp only [Option.some.injEq] at h
      subst h; subst hx
      simp
    · obtain ⟨h1, h2⟩ := findN_spec r nd (i + 1) k h
      refine ⟨by omega, ?_⟩
      have : k - i = (k - (i + 1)) + 1 := by omega
      rw [this]; simpa using h2

theorem findN_some {ar : Arena} {nd : DNode} {k : Nat} (h : findN ar nd 0 = some k) : ar[k]? = some nd := by
  have := (findN_spec ar nd 0 k h).2
  simpa using this

theorem leaves_at_leaf {ar : Arena} {k : Nat} {o : Origin} (h : ar[k]? = some (.leaf o)) :
    leaves ar k = [o] := by
  simp [leaves, lvs, h]

theorem leaves_at_join {ar : Arena} (hw : AWF ar) {k a b : Nat} (h : ar[k]? = some (.join a b)) :
    leaves ar k = leaves ar a ++ leaves ar b := by
  obtain ⟨ha, hb⟩ := hw k a b h
  show lvs ar (k + 1) k = lvs ar (a + 1) a ++ lvs ar (b + 1) b
  rw [lvs.eq_2, h]
  dsimp only
  rw [lvs_fuel hw k (a + 1) a ha (by omega), lvs_fuel hw k (b + 1) b hb (by omega)]

theorem lt_of_getElem? {ar : Arena} {k : Nat} {nd : DNode} (h : ar[k]? = some nd) : k < ar.length := by
  rcases Nat.lt_or_ge k ar.length with hk | hk
  · exact hk
  · rw [List.getElem?_eq_none hk] at h; contradiction

/-- Build a plan, reusing equal nodes. -/
def hbuild : PE → Arena → Nat × Arena
  | .nil, ar => (0, ar)
  | .node n, ar => (n, ar)
  | .leaf o, ar =>
    match findN ar (.leaf o) 0 with
    | some k => (k, ar)
    | none => (ar.length, ar ++ [.leaf o])
  | .app a b, ar =>
    let (x, ar1) := hbuild a ar
    let (y, ar2) := hbuild b ar1
    match findN ar2 (.join x y) 0 with
    | some k => (k, ar2)
    | none => (ar2.length, ar2 ++ [.join x y])

theorem Good.push_leaf {ar : Arena} (hg : Good ar) (o : Origin) : Good (ar ++ [.leaf o]) := by
  refine ⟨hg.1.push_leaf o, ?_⟩
  have : 0 < ar.length := lt_of_getElem? hg.2
  rw [List.getElem?_append_left this]; exact hg.2

theorem Good.push_join {ar : Arena} (hg : Good ar) {a b : Nat} (ha : a < ar.length) (hb : b < ar.length) :
    Good (ar ++ [.join a b]) := by
  refine ⟨hg.1.push_join ha hb, ?_⟩
  have : 0 < ar.length := lt_of_getElem? hg.2
  rw [List.getElem?_append_left this]; exact hg.2

theorem hbuild_spec : ∀ (e : PE) (ar : Arena), Good ar → e.ok ar.length →
    Good (hbuild e ar).2 ∧ Ext ar (hbuild e ar).2 ∧ (hbuild e ar).1 < (hbuild e ar).2.length ∧
      leaves (hbuild e ar).2 (hbuild e ar).1 = e.den (leaves ar)
  | .nil, ar, hg, _ => by
    refine ⟨hg, Ext.refl _, lt_of_getElem? hg.2, ?_⟩
    simp only [hbuild, PE.den]; exact leaves_empty hg.2
  | .node n, ar, hg, he => ⟨hg, Ext.refl _, he, rfl⟩
  | .leaf o, ar, hg, _ => by
    simp only [hbuild]
    rcases hf : findN ar (.leaf o) 0 with _ | k
    · exact ⟨hg.push_leaf o, ⟨_, rfl⟩, by simp, leaves_leaf o⟩
    · have hk := findN_some hf
      exact ⟨hg, Ext.refl _, lt_of_getElem? hk, leaves_at_leaf hk⟩
  | .app x y, ar, hg, he => by
    obtain ⟨g1, e1, l1, d1⟩ := hbuild_spec x ar hg he.1
    obtain ⟨g2, e2, l2, d2⟩ := hbuild_spec y (hbuild x ar).2 g1 (PE.ok_mono he.2 e1.len)
    have hlt1 : (hbuild x ar).1 < (hbuild y (hbuild x ar).2).2.length := Nat.lt_of_lt_of_le l1 e2.len
    have hden : leaves (hbuild y (hbuild x ar).2).2 (hbuild x ar).1 ++
        leaves (hbuild y (hbuild x ar).2).2 (hbuild y (hbuild x ar).2).1 = (PE.app x y).den (leaves ar) := by
      rw [d2, g1.leaves_ext e2 l1, d1]
      simp only [PE.den]
      rw [PE.den_ext hg e1 he.2]
    simp only [hbuild]
    rcases hf : findN (hbuild y (hbuild x ar).2).2 (.join (hbuild x ar).1 (hbuild y (hbuild x ar).2).1) 0 with _ | k
    · refine ⟨g2.push_join hlt1 l2, Ext.trans (Ext.trans e1 e2) ⟨_, rfl⟩, by simp, ?_⟩
      rw [leaves_join g2.1 hlt1 l2, hden]
    · have hk := findN_some hf
      refine ⟨g2, Ext.trans e1 e2, lt_of_getElem? hk, ?_⟩
      rw [leaves_at_join g2.1 hk, hden]

/-- The hash-consing allocator. -/
def Alloc.hashcons : Alloc := ⟨hbuild, hbuild_spec⟩

/-- Building the same union twice allocates once: the second build finds the
nodes of the first and the arena does not grow. -/
theorem hbuild_reuses :
    let e : PE := .app (.leaf ⟨some 0, 0, 0, 0⟩) (.leaf ⟨none, 0, 1, 0⟩)
    let ar1 := (hbuild e [.empty]).2
    ar1.length = 4 ∧ (hbuild e ar1).2.length = 4 ∧ (hbuild e ar1).1 = (hbuild e [.empty]).1 := by
  decide

/-- With hash-consing, the DAG machine still computes exactly what the list
machine computes, for every program, input and number of steps. -/
theorem drun_init_hashcons (pr : Prog) (ph : Phys) (msize : Nat) (inp : List Nat) (n : Nat) :
    (drun Alloc.hashcons pr ph n (dinit pr msize inp)).abs = run pr ph n (init pr msize inp) :=
  drun_init Alloc.hashcons pr ph msize inp n

end Sporos

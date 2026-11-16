/-
The DAG semantics simulates the list semantics. `α` reads a DAG state as a list
state, each node standing for its leaves; `dstep_sim` says a DAG step is a list
step of the state it stands for.

This file, part 1: what `build` allocates and what it stands for.
-/
import Sporos.DagStep

namespace Sporos

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

/-- The allocator that appends one node per union and per origin. -/
def Alloc.plain : Alloc := ⟨Sporos.build, build_spec⟩

end Sporos

/-! ### Reading a DAG state as a list state -/

namespace Sporos

def αv (ar : Arena) (d : DVal) : Val := ⟨d.w, leaves ar d.n⟩

def αf (ar : Arena) (g : DFrame) : Frame := ⟨g.word, g.pc, g.locs.map (αv ar), g.base, g.abase⟩

def α (t : DSt) : St :=
  { ds := t.ds.map (αv t.ar), aux := t.aux.map (αv t.ar), rs := t.rs.map (αf t.ar),
    mem := fun x => αv t.ar (t.mem x), msize := t.msize, hp := t.hp,
    tabs := t.tabs.map (List.map fun kv => (kv.1, αv t.ar kv.2)),
    inp := t.inp, ipos := t.ipos, out := t.out.map (fun bn => (bn.1, leaves t.ar bn.2)),
    pc := leaves t.ar t.pc, step := t.step }

/-- Every node the state names exists. -/
def Valid (t : DSt) : Prop :=
  Good t.ar ∧ (∀ d ∈ t.ds, d.n < t.ar.length) ∧ (∀ d ∈ t.aux, d.n < t.ar.length) ∧
  (∀ g ∈ t.rs, ∀ d ∈ g.locs, d.n < t.ar.length) ∧ (∀ x, (t.mem x).n < t.ar.length) ∧
  (∀ tb ∈ t.tabs, ∀ kv ∈ tb, kv.2.n < t.ar.length) ∧ (∀ bn ∈ t.out, bn.2 < t.ar.length) ∧
  t.pc < t.ar.length

theorem αv_ext {a b : Arena} (ha : Good a) (h : Ext a b) {d : DVal} (hd : d.n < a.length) :
    αv b d = αv a d := by simp only [αv]; rw [ha.leaves_ext h hd]

theorem map_αv_ext {a b : Arena} (ha : Good a) (h : Ext a b) {l : List DVal} (hl : ∀ d ∈ l, d.n < a.length) :
    l.map (αv b) = l.map (αv a) := List.map_congr_left fun d hd => αv_ext ha h (hl d hd)

theorem qPlan_ok {args : List DVal} {pc n : Nat} (ha : ∀ d ∈ args, d.n < n) (hp : pc < n) (o : Origin) :
    (qPlan args pc o).ok n := by
  refine ⟨⟨?_, hp⟩, trivial⟩
  induction args with
  | nil => trivial
  | cons d r ih => exact ⟨ha d (by simp), ih fun x hx => ha x (by simp [hx])⟩

theorem qPlan_den (ar : Arena) (args : List DVal) (pc : Nat) (o : Origin) :
    (qPlan args pc o).den (leaves ar) =
      ((args.map (αv ar)).foldr (fun v acc => v.p ++ acc) []) ++ leaves ar pc ++ [o] := by
  simp only [qPlan, PE.den]
  congr 2
  induction args with
  | nil => rfl
  | cons d r ih => simp only [List.foldr, PE.den, List.map, ih]; rfl

theorem tagPlan_den (ar : Arena) (pc : Nat) (o : Origin) (d : DVal) :
    (tagPlan pc o d).den (leaves ar) = (tag (leaves ar pc) o (αv ar d)).p := rfl

theorem foldPlan_den (ar : Arena) (f : Nat → DVal) (l : List Nat) :
    (l.foldr (fun i acc => PE.app (.node (f i).n) acc) .nil).den (leaves ar) =
      l.foldr (fun i acc => (αv ar (f i)).p ++ acc) [] := by
  induction l with
  | nil => rfl
  | cons i r ih => simp only [List.foldr, PE.den, ih]; rfl

theorem foldPlan2_den (ar : Arena) (f g : Nat → DVal) (l : List Nat) :
    (l.foldr (fun i acc => PE.app (.app (.node (f i).n) (.node (g i).n)) acc) .nil).den (leaves ar) =
      l.foldr (fun i acc => (αv ar (f i)).p ++ (αv ar (g i)).p ++ acc) [] := by
  induction l with
  | nil => rfl
  | cons i r ih => simp only [List.foldr, PE.den, ih]; rfl

theorem α_ld (t : DSt) (a n : Nat) :
    (α t).ld a n = (t.ld a n).map (fun r => (r.1, r.2.den (leaves t.ar))) := by
  simp only [St.ld, DSt.ld, α]
  split <;> simp [foldPlan_den, αv]

theorem α_st (t : DSt) (a n v qn : Nat) :
    (α t).st a n v (leaves t.ar qn) = (t.st a n v qn).map α := by
  have hm : (α t).msize = t.msize := rfl
  unfold St.st DSt.st; rw [hm]
  split
  · simp only [Option.map_some, Option.some.injEq]
    simp only [α]
    congr 1
    funext x
    split <;> rfl
  · rfl

theorem tfind_map (ar : Arena) (l : List (Nat × DVal)) (k : Nat) :
    tfind (l.map fun kv => (kv.1, αv ar kv.2)) k = (tfind' l k).map (αv ar) := by
  induction l with
  | nil => rfl
  | cons kv r ih =>
    obtain ⟨k', v⟩ := kv
    simp only [List.map, tfind, tfind']
    split <;> simp_all

end Sporos

/-
The collector (decision 36): compacting the arena of a state.

Given the nodes a state names (the roots), `gcAr` keeps node 0 and every node
reachable from a root, in their original order, renumbered by `σ` (the number of
kept nodes below), with each join's children renumbered too. `gc_leaves`: every
kept node stands for the same origins after compaction. Joins still point
backwards and node 0 is still `empty` (`gc_good`).

`gc` compacts a whole state, renumbering every node it names; `gc_spec`: from a
valid state whose memory beyond its size carries no provenance (`MemZero`, kept
by every step, `MemZero.step`), the compacted state is valid and stands for the
same list state. `drunC_init`: a run that collects at any points still computes
exactly what the list machine computes.
-/
import Sporos.DagHash

namespace Sporos

/-- The nodes reachable from `i`, `i` first, with fuel `f`. -/
def nodes (ar : Arena) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, i =>
    i :: match ar[i]? with
      | some (.join a b) => nodes ar f a ++ nodes ar f b
      | _ => []

theorem self_mem_nodes (ar : Arena) {f i : Nat} (h : i < f) : i ∈ nodes ar f i := by
  cases f with
  | zero => omega
  | succ f => simp [nodes]

/-- Reachability is closed under taking children, when the fuel exceeds the index. -/
theorem nodes_closed {ar : Arena} (hw : AWF ar) :
    ∀ (f i : Nat), i < f → ∀ j ∈ nodes ar f i, ∀ a b, ar[j]? = some (.join a b) →
      a ∈ nodes ar f i ∧ b ∈ nodes ar f i := by
  intro f
  induction f with
  | zero => intro i h; omega
  | succ f ih =>
    intro i hi j hj a b hjab
    simp only [nodes, List.mem_cons] at hj ⊢
    rcases hj with rfl | hj
    · obtain ⟨ha, hb⟩ := hw j a b hjab
      simp only [hjab, List.mem_append]
      exact ⟨Or.inr (Or.inl (self_mem_nodes ar (by omega))), Or.inr (Or.inr (self_mem_nodes ar (by omega)))⟩
    · cases hn : ar[i]? with
      | none => simp [hn] at hj
      | some nd =>
        cases nd with
        | empty => simp [hn] at hj
        | leaf o => simp [hn] at hj
        | join x y =>
          simp only [hn, List.mem_append] at hj ⊢
          obtain ⟨hx, hy⟩ := hw i x y hn
          rcases hj with hj | hj
          · obtain ⟨h1, h2⟩ := ih x (by omega) j hj a b hjab
            exact ⟨Or.inr (Or.inl h1), Or.inr (Or.inl h2)⟩
          · obtain ⟨h1, h2⟩ := ih y (by omega) j hj a b hjab
            exact ⟨Or.inr (Or.inr h1), Or.inr (Or.inr h2)⟩

/-- The kept nodes: node 0 and everything reachable from a root. -/
def liveL (ar : Arena) (roots : List Nat) : List Nat := 0 :: roots.flatMap (fun r => nodes ar (r + 1) r)

theorem live_root (ar : Arena) {roots : List Nat} {r : Nat} (h : r ∈ roots) : r ∈ liveL ar roots := by
  simp only [liveL, List.mem_cons, List.mem_flatMap]
  exact Or.inr ⟨r, h, self_mem_nodes ar (by omega)⟩

theorem live_closed {ar : Arena} (hg : Good ar) (roots : List Nat) {j a b : Nat}
    (hj : j ∈ liveL ar roots) (h : ar[j]? = some (.join a b)) :
    a ∈ liveL ar roots ∧ b ∈ liveL ar roots := by
  simp only [liveL, List.mem_cons, List.mem_flatMap] at hj ⊢
  rcases hj with rfl | ⟨r, hr, hj⟩
  · rw [hg.2] at h; simp at h
  · obtain ⟨h1, h2⟩ := nodes_closed hg.1 (r + 1) r (by omega) j hj a b h
    exact ⟨Or.inr ⟨r, hr, h1⟩, Or.inr ⟨r, hr, h2⟩⟩

/-- The new index of a kept node: how many kept nodes lie below it. -/
def σ (L : List Nat) (i : Nat) : Nat := ((List.range i).filter (· ∈ L)).length

theorem σ_succ (L : List Nat) (i : Nat) : σ L (i + 1) = σ L i + (if i ∈ L then 1 else 0) := by
  simp only [σ, List.range_succ, List.filter_append, List.length_append]
  by_cases h : i ∈ L <;> simp [h]

theorem σ_mono (L : List Nat) {i j : Nat} (h : i ≤ j) : σ L i ≤ σ L j := by
  induction j with
  | zero => have : i = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ j ih =>
    rcases Nat.lt_or_ge i (j + 1) with hl | hl
    · rw [σ_succ]; exact Nat.le_trans (ih (by omega)) (Nat.le_add_right _ _)
    · have : i = j + 1 := by omega
      subst this; exact Nat.le_refl _

theorem σ_lt (L : List Nat) {i j : Nat} (hij : i < j) (hi : i ∈ L) : σ L i < σ L j := by
  have h1 := σ_mono L (show i + 1 ≤ j by omega)
  rw [σ_succ] at h1
  simp only [hi, if_true] at h1
  omega

def remap (L : List Nat) : DNode → DNode
  | .empty => .empty
  | .leaf o => .leaf o
  | .join a b => .join (σ L a) (σ L b)

/-- The compacted arena, built over the first `k` nodes. -/
def gcAr (ar : Arena) (L : List Nat) : Nat → Arena
  | 0 => []
  | k + 1 =>
    gcAr ar L k ++ (if k ∈ L then match ar[k]? with | some nd => [remap L nd] | none => [] else [])

theorem gcAr_length {ar : Arena} (L : List Nat) : ∀ k, k ≤ ar.length → (gcAr ar L k).length = σ L k := by
  intro k
  induction k with
  | zero => intro _; rfl
  | succ k ih =>
    intro hk
    simp only [gcAr, List.length_append, σ_succ, ih (by omega)]
    by_cases h : k ∈ L
    · simp only [h, if_true]
      rw [List.getElem?_eq_getElem (by omega)]
      simp
    · simp [h]

theorem gcAr_get {ar : Arena} (L : List Nat) :
    ∀ k, k ≤ ar.length → ∀ i < k, i ∈ L → (gcAr ar L k)[σ L i]? = (ar[i]?).map (remap L) := by
  intro k
  induction k with
  | zero => intro _ i hi; omega
  | succ k ih =>
    intro hk i hi hL
    have hlen := gcAr_length (ar := ar) L k (by omega)
    simp only [gcAr]
    rcases Nat.lt_or_ge i k with hik | hik
    · rw [List.getElem?_append_left (by rw [hlen]; exact σ_lt L hik hL)]
      exact ih (by omega) i hik hL
    · have : i = k := by omega
      subst this
      have hki : i < ar.length := by omega
      rw [List.getElem?_eq_getElem hki]
      rw [← hlen, List.getElem?_append_right (Nat.le_refl _), Nat.sub_self]
      simp [hL]

theorem AWF.push_empty {ar : Arena} (hw : AWF ar) : AWF (ar ++ [.empty]) := by
  intro i a b h
  by_cases hi : i < ar.length
  · rw [List.getElem?_append_left hi] at h; exact hw i a b h
  · rw [List.getElem?_append_right (by omega)] at h
    cases e : i - ar.length with
    | zero => rw [e] at h; simp at h
    | succ k => rw [e] at h; simp at h

theorem gcAr_awf {ar : Arena} (hg : Good ar) (roots : List Nat) :
    ∀ k, k ≤ ar.length → AWF (gcAr ar (liveL ar roots) k) := by
  intro k
  induction k with
  | zero => intro _ i a b h; simp [gcAr] at h
  | succ k ih =>
    intro hk
    have hw := ih (by omega)
    have hlen := gcAr_length (ar := ar) (liveL ar roots) k (by omega)
    simp only [gcAr]
    by_cases hL : k ∈ liveL ar roots
    · simp only [hL, if_true]
      rw [List.getElem?_eq_getElem (by omega)]
      dsimp only
      cases hn : ar[k] with
      | empty => exact hw.push_empty
      | leaf o => exact hw.push_leaf o
      | join a b =>
        have hk' : ar[k]? = some (.join a b) := by
          rw [List.getElem?_eq_getElem (by omega)]; exact congrArg some hn
        obtain ⟨ha, hb⟩ := hg.1 k a b hk'
        obtain ⟨la, lb⟩ := live_closed hg roots hL hk'
        exact hw.push_join (by rw [hlen]; exact σ_lt _ ha la) (by rw [hlen]; exact σ_lt _ hb lb)
    · simp only [hL, if_false, List.append_nil]; exact hw

/-- The compacted arena. -/
def gcArena (ar : Arena) (roots : List Nat) : Arena := gcAr ar (liveL ar roots) ar.length

theorem gc_good {ar : Arena} (hg : Good ar) (roots : List Nat) : Good (gcArena ar roots) := by
  refine ⟨gcAr_awf hg roots _ (Nat.le_refl _), ?_⟩
  have h0 : 0 < ar.length := lt_of_getElem? hg.2
  have := gcAr_get (liveL ar roots) ar.length (Nat.le_refl _) 0 h0 (by simp [liveL])
  simp only [σ, List.range_zero, List.filter_nil, List.length_nil] at this
  simp only [gcArena, this, hg.2, Option.map_some, remap]

/-- Compaction keeps what every kept node stands for. -/
theorem gc_leaves {ar : Arena} (hg : Good ar) (roots : List Nat) :
    ∀ i, i < ar.length → i ∈ liveL ar roots →
      leaves (gcArena ar roots) (σ (liveL ar roots) i) = leaves ar i := by
  intro i
  induction i using Nat.strongRecOn with
  | ind i ih =>
    intro hi hL
    have hget : (gcArena ar roots)[σ (liveL ar roots) i]? = (ar[i]?).map (remap (liveL ar roots)) :=
      gcAr_get (liveL ar roots) ar.length (Nat.le_refl _) i hi hL
    have hw' := (gc_good hg roots).1
    rw [List.getElem?_eq_getElem hi] at hget
    have hi' : ar[i]? = some ar[i] := List.getElem?_eq_getElem hi
    cases hn : ar[i] with
    | empty =>
      rw [hn] at hget hi'
      rw [leaves_empty hget, leaves_empty hi']
    | leaf o =>
      rw [hn] at hget hi'
      rw [leaves_at_leaf hget, leaves_at_leaf hi']
    | join a b =>
      rw [hn] at hget hi'
      obtain ⟨ha, hb⟩ := hg.1 i a b hi'
      obtain ⟨la, lb⟩ := live_closed hg roots hL hi'
      simp only [Option.map_some, remap] at hget
      rw [leaves_at_join hw' hget, leaves_at_join hg.1 hi', ih a ha (by omega) la, ih b hb (by omega) lb]

/-- A dead union and its dead leaf are dropped; the kept leaf moves to index 1. -/
theorem gc_drops_dead :
    let ar : Arena := [.empty, .leaf ⟨some 0, 0, 0, 0⟩, .leaf ⟨some 1, 0, 0, 0⟩, .join 1 2]
    gcArena ar [1] = [.empty, .leaf ⟨some 0, 0, 0, 0⟩] ∧ σ (liveL ar [1]) 1 = 1 := by
  decide

/-! ### Memory beyond its size -/


/-- Memory beyond its size never changes. -/
def MemFix (t s : DSt) : Prop := s.msize = t.msize ∧ ∀ x, t.msize ≤ x → s.mem x = t.mem x

theorem prim_mem (ph : Phys) (p : Prim) (args : List DVal) (t : DSt) (o : Origin) (qn : Nat)
    {res : List (Nat × PE)} {s' : DSt} (h : execPrimD ph p args t o qn = .ok res s') : MemFix t s' := by
  cases p <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;>
    simp only [execPrimD] at h <;> (repeat' split at h) <;> (try contradiction) <;>
    (try (simp only [DPR.ok.injEq] at h; obtain ⟨rfl, rfl⟩ := h)) <;>
    first
      | exact ⟨rfl, fun _ _ => rfl⟩
      | (refine ⟨rfl, fun x hx => ?_⟩; dsimp only; rw [if_neg (by omega)])
      | (rename_i hst; unfold DSt.st at hst; split at hst
         · simp only [Option.some.injEq] at hst; subst hst
           refine ⟨rfl, fun x hx => ?_⟩; dsimp only; rw [if_neg (by omega)]
         · contradiction)
theorem DSt.setTop_mem (X : DSt) (g : DFrame) : (X.setTop g).mem = X.mem ∧ (X.setTop g).msize = X.msize := by
  unfold DSt.setTop; split <;> exact ⟨rfl, rfl⟩

theorem dstep_mem (A : Alloc) (pr : Prog) (ph : Phys) (t : DSt) {t' : DSt}
    (h : dstep A pr ph t = .next t') : MemFix t t' := by
  obtain ⟨ds, aux, rs, mem, msize, hp, tabs, inp, ipos, out, pc, stp, ar⟩ := t
  rcases rs with _ | ⟨f, rest⟩
  · simp [dstep] at h
  · simp only [dstep] at h
    rcases hfe : fetch pr f.word f.pc with _ | ins <;> rw [hfe] at h
    · simp at h
    · cases ins <;> dsimp only at h <;> (repeat' split at h) <;> (try simp at h) <;>
        (try (subst h; exact ⟨rfl, fun _ _ => rfl⟩))
      rename_i heq
      obtain ⟨h1, h2⟩ := prim_mem _ _ _ _ _ _ heq
      subst h
      obtain ⟨m1, m2⟩ := DSt.setTop_mem _ _
      exact ⟨m2.trans h1, fun x hx => by rw [m1]; exact h2 x hx⟩

/-! ### Compacting a state -/

/-- Memory beyond its size carries no provenance. -/
def MemZero (t : DSt) : Prop := ∀ x, t.msize ≤ x → (t.mem x).n = 0

theorem MemZero.step {A : Alloc} {pr : Prog} {ph : Phys} {t t' : DSt} (hz : MemZero t)
    (h : dstep A pr ph t = .next t') : MemZero t' := by
  obtain ⟨h1, h2⟩ := dstep_mem A pr ph t h
  intro x hx
  rw [h1] at hx
  rw [h2 x hx]; exact hz x hx

/-- Every node a state names. -/
def roots (t : DSt) : List Nat :=
  t.pc :: (t.ds.map DVal.n ++ t.aux.map DVal.n ++ t.rs.flatMap (fun g => g.locs.map DVal.n) ++
    (List.range t.msize).map (fun x => (t.mem x).n) ++
    t.tabs.flatMap (fun tb => tb.map (fun kv => kv.2.n)) ++ t.out.map Prod.snd)


theorem roots_pc (t : DSt) : t.pc ∈ roots t := by simp [roots]
theorem roots_ds {t : DSt} {d : DVal} (h : d ∈ t.ds) : d.n ∈ roots t := by
  simp only [roots, List.mem_cons, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range]; exact Or.inr (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl ⟨d, h, rfl⟩)))))
theorem roots_aux {t : DSt} {d : DVal} (h : d ∈ t.aux) : d.n ∈ roots t := by
  simp only [roots, List.mem_cons, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range]; exact Or.inr (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr ⟨d, h, rfl⟩)))))
theorem roots_rs {t : DSt} {g : DFrame} (hg : g ∈ t.rs) {d : DVal} (h : d ∈ g.locs) : d.n ∈ roots t := by
  simp only [roots, List.mem_cons, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range]; exact Or.inr (Or.inl (Or.inl (Or.inl (Or.inr ⟨g, hg, d, h, rfl⟩))))
theorem roots_mem {t : DSt} {x : Nat} (h : x < t.msize) : (t.mem x).n ∈ roots t := by
  simp only [roots, List.mem_cons, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range]; exact Or.inr (Or.inl (Or.inl (Or.inr ⟨x, h, rfl⟩)))
theorem roots_tabs {t : DSt} {tb : List (Nat × DVal)} (htb : tb ∈ t.tabs) {kv : Nat × DVal} (h : kv ∈ tb) :
    kv.2.n ∈ roots t := by
  simp only [roots, List.mem_cons, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range]; exact Or.inr (Or.inl (Or.inr ⟨tb, htb, kv, h, rfl⟩))
theorem roots_out {t : DSt} {bn : Nat × Nat} (h : bn ∈ t.out) : bn.2 ∈ roots t := by
  simp only [roots, List.mem_cons, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range]; exact Or.inr (Or.inr ⟨bn, h, rfl⟩)

/-- Compact the arena, keeping what the state names, and renumber. -/
def gc (t : DSt) : DSt :=
  let L := liveL t.ar (roots t)
  let rv := fun d : DVal => (⟨d.w, σ L d.n⟩ : DVal)
  { ds := t.ds.map rv, aux := t.aux.map rv, rs := t.rs.map (fun g => { g with locs := g.locs.map rv }),
    mem := fun x => rv (t.mem x), msize := t.msize, hp := t.hp,
    tabs := t.tabs.map (List.map fun kv => (kv.1, rv kv.2)), inp := t.inp, ipos := t.ipos,
    out := t.out.map (fun bn => (bn.1, σ L bn.2)), pc := σ L t.pc, step := t.step,
    ar := gcArena t.ar (roots t) }

theorem gc_node {t : DSt} (hg : Good t.ar) {n : Nat} (hn : n < t.ar.length) (hL : n ∈ liveL t.ar (roots t)) :
    σ (liveL t.ar (roots t)) n < (gcArena t.ar (roots t)).length ∧
      leaves (gcArena t.ar (roots t)) (σ (liveL t.ar (roots t)) n) = leaves t.ar n := by
  refine ⟨?_, gc_leaves hg (roots t) n hn hL⟩
  rw [gcArena, gcAr_length _ _ (Nat.le_refl _)]
  exact σ_lt _ hn hL

theorem gc_spec {t : DSt} (hv : Valid t) (hz : MemZero t) :
    Valid (gc t) ∧ α (gc t) = α t ∧ MemZero (gc t) := by
  obtain ⟨hg, hds, hax, hrs, hmem, htab, hout, hpc⟩ := hv
  have hr : ∀ n ∈ roots t, σ (liveL t.ar (roots t)) n < (gcArena t.ar (roots t)).length ∧
      leaves (gcArena t.ar (roots t)) (σ (liveL t.ar (roots t)) n) = leaves t.ar n := by
    intro n hn
    refine gc_node hg ?_ (live_root _ hn)
    simp only [roots, List.mem_cons, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range] at hn
    rcases hn with rfl | ((((⟨d, hd, rfl⟩ | ⟨d, hd, rfl⟩) | ⟨g, hg', d, hd, rfl⟩) | ⟨x, _, rfl⟩) |
      ⟨tb, htb, kv, hkv, rfl⟩) | ⟨bn, hbn, rfl⟩
    · exact hpc
    · exact hds d hd
    · exact hax d hd
    · exact hrs g hg' d hd
    · exact hmem x
    · exact htab tb htb kv hkv
    · exact hout bn hbn
  have h0 : σ (liveL t.ar (roots t)) 0 < (gcArena t.ar (roots t)).length ∧
      leaves (gcArena t.ar (roots t)) (σ (liveL t.ar (roots t)) 0) = leaves t.ar 0 :=
    gc_node hg (lt_of_getElem? hg.2) (by simp [liveL])
  have hm : ∀ x, σ (liveL t.ar (roots t)) (t.mem x).n < (gcArena t.ar (roots t)).length ∧
      leaves (gcArena t.ar (roots t)) (σ (liveL t.ar (roots t)) (t.mem x).n) = leaves t.ar (t.mem x).n := by
    intro x
    rcases Nat.lt_or_ge x t.msize with hx | hx
    · exact hr _ (roots_mem hx)
    · rw [hz x hx]; exact h0
  refine ⟨⟨gc_good hg _, ?_, ?_, ?_, fun x => (hm x).1, ?_, ?_, (hr _ (roots_pc t)).1⟩, ?_, ?_⟩
  · intro d hd
    simp only [gc, List.mem_map] at hd
    obtain ⟨d0, h0', rfl⟩ := hd
    exact (hr _ (roots_ds h0')).1
  · intro d hd
    simp only [gc, List.mem_map] at hd
    obtain ⟨d0, h0', rfl⟩ := hd
    exact (hr _ (roots_aux h0')).1
  · intro g hg' d hd
    simp only [gc, List.mem_map] at hg'
    obtain ⟨g0, hg0, rfl⟩ := hg'
    simp only [List.mem_map] at hd
    obtain ⟨d0, hd0, rfl⟩ := hd
    exact (hr _ (roots_rs hg0 hd0)).1
  · intro tb htb kv hkv
    simp only [gc, List.mem_map] at htb
    obtain ⟨tb0, htb0, rfl⟩ := htb
    simp only [List.mem_map] at hkv
    obtain ⟨kv0, hkv0, rfl⟩ := hkv
    exact (hr _ (roots_tabs htb0 hkv0)).1
  · intro bn hbn
    simp only [gc, List.mem_map] at hbn
    obtain ⟨bn0, hbn0, rfl⟩ := hbn
    exact (hr _ (roots_out hbn0)).1
  · simp only [α, gc, List.map_map]
    congr 1
    · apply List.map_congr_left; intro d hd
      simp only [Function.comp_def, αv, (hr _ (roots_ds hd)).2]
    · apply List.map_congr_left; intro d hd
      simp only [Function.comp_def, αv, (hr _ (roots_aux hd)).2]
    · apply List.map_congr_left; intro g hg'
      simp only [Function.comp_def, αf, List.map_map]
      congr 1
      apply List.map_congr_left; intro d hd
      simp only [Function.comp_def, αv, (hr _ (roots_rs hg' hd)).2]
    · funext x; simp only [αv, (hm x).2]
    · apply List.map_congr_left; intro tb htb
      simp only [Function.comp_def, List.map_map]
      apply List.map_congr_left; intro kv hkv
      simp only [Function.comp_def, αv, (hr _ (roots_tabs htb hkv)).2]
    · apply List.map_congr_left; intro bn hbn
      simp only [Function.comp_def, (hr _ (roots_out hbn)).2]
    · exact (hr _ (roots_pc t)).2
  · intro x hx
    simp only [gc] at hx ⊢
    rw [hz x hx]
    simp [σ]

/-! ### Runs that collect -/

variable (A : Alloc) in
/-- `n` DAG steps, compacting after each step where `pol` says so. The
interpreter's policy is "the arena has doubled"; the theorem holds for any. -/
def drunC (pr : Prog) (ph : Phys) (pol : DSt → Bool) : Nat → DSt → DRes
  | 0, s => .next s
  | n + 1, s =>
    match dstep A pr ph s with
    | .next t => drunC pr ph pol n (if pol t then gc t else t)
    | r => r

theorem drunC_sim (A : Alloc) (pr : Prog) (ph : Phys) (pol : DSt → Bool) :
    ∀ (n : Nat) (t : DSt), Valid t → MemZero t → (drunC A pr ph pol n t).abs = run pr ph n (α t) := by
  intro n
  induction n with
  | zero => intro t _ _; rfl
  | succ n ih =>
    intro t hv hz
    obtain ⟨hs, hn⟩ := dstep_sim (A := A) pr ph t hv
    simp only [drunC, run]
    rw [← hs]
    rcases h : dstep A pr ph t with t' | ⟨c, o, a⟩ | e
    · change (drunC A pr ph pol n (if pol t' then gc t' else t')).abs = run pr ph n (α t')
      have hv' := hn t' h
      have hz' := hz.step h
      by_cases hp : pol t' = true
      · obtain ⟨g1, g2, g3⟩ := gc_spec hv' hz'
        rw [if_pos hp, ih _ g1 g3, g2]
      · rw [if_neg hp]
        exact ih t' hv' hz'
    · rfl
    · rfl

theorem dinit_memzero (pr : Prog) (msize : Nat) (inp : List Nat) : MemZero (dinit pr msize inp) :=
  fun _ _ => rfl

/-- Collecting at any points, with any allocator, the DAG machine computes exactly
what the list machine computes, for every program, input and number of steps. -/
theorem drunC_init (A : Alloc) (pr : Prog) (ph : Phys) (pol : DSt → Bool) (msize : Nat) (inp : List Nat)
    (n : Nat) :
    (drunC A pr ph pol n (dinit pr msize inp)).abs = run pr ph n (init pr msize inp) := by
  rw [drunC_sim A pr ph pol n _ (dinit_valid pr msize inp) (dinit_memzero pr msize inp), α_dinit]

end Sporos

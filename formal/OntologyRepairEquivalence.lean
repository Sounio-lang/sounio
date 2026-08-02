/-
# Sounio.OntologyRepairEquivalence — greedy drop-weaker ≡ priority fold

Formal companion closing the documented gap of frontier 1
(`artifacts/ontology-frontiers/epistemic-alignment-repair/`): the remark in
`OntologyAlignmentRepair.lean` claimed that the *pairwise drop-weaker* greedy
implemented in `alignment_repair.sio` computes the same priority independent
set as the fold `repair`, for distinct confidences, without mechanising it.
Mechanising the claim shows it needs a hypothesis the remark did not state.

## What is proved here

1. `mem_repair_iff` — exact unfolding characterisation of the priority fold:
   `m ∈ repair C kept ms` iff `m` was already kept, or `ms` splits as
   `pre ++ m :: post` with `m` conflicting with nothing in
   `repair C kept pre`. Fully general (no nodup/sorting/decidability).
2. `mem_repair_priority` — the **fold-side fixpoint characterisation**: for a
   nodup candidate list sorted by decreasing confidence with distinct
   confidences, `m` survives the fold iff `m` is a candidate and every
   higher-confidence candidate that survives is conflict-free with `m`.
3. `greedyStep`/`greedyDrop` — a model of the `.sio` pairwise greedy: scan a
   list of conflicting pairs; whenever both endpoints are still alive and
   conflict, drop the lower-confidence endpoint.
4. `mem_greedyDrop_of_pair` / `mem_greedyDrop_of_higher_neighbor` — the greedy
   never leaves a conflicting pair both alive; hence a greedy survivor has no
   higher-confidence conflicting survivor (the greedy kept set is *below* the
   priority fixpoint).
5. `greedyDrop_witness` — under a **cluster-graph** hypothesis on the conflict
   relation (`C a b ∧ C b c ∧ a ≠ c → C a c`, i.e. conflicts form a disjoint
   union of cliques, which covers the frontier instance: disjoint edges), every
   greedy-dropped mapping has a surviving higher-confidence conflicting
   witness. The cluster hypothesis is exactly what collapses the drop chain
   `m ≺ k₁ ≺ k₂ ≺ …` into a direct neighbour.
6. `fixpoint_unique` — the priority fixpoint has at most one solution on a
   candidate list (bounded strong induction on confidence).
7. `repair_iff_greedy` — **main equivalence**: distinct confidences + sorted
   nodup candidates + symmetric cluster conflicts + covering pair list ⇒
   `m ∈ repair C [] ms ↔ m ∈ ms ∧ m ∉ greedyDrop C conf [] ps`.
8. `ex_*` — the concrete `Fin 5` instance of `alignment_repair.sio`: the
   general theorem instantiated (hypotheses discharged by `decide`), plus
   `native_decide` sanity checks.
9. `cx_*` — **necessity of the cluster hypothesis**: on a 3-vertex conflict
   path `0—1—2` with confidences `0 < 1 < 2` scanned as `[(0,1),(1,2)]`, the
   greedy keeps `{2}` while the fold keeps `{0,2}`; the bare "distinct
   confidences" claim is false, certified by `native_decide`.

## What remains unmechanised (honest gaps)

- The conflict oracle is assumed, not derived from OWL/EL++ semantics
  (same gap as `OntologyAlignmentRepair.lean`).
- The equivalence is proved for a *single* pass over the pair list (matching
  the `.sio` prototype). Variants (iterated-to-fixpoint greedy, other pair
  orders) are not covered; the counterexample shows pair order matters in
  general.
- Equal confidences are excluded (the `.sio` tie-break `conf[i] >= conf[j]`
  keeps the lower index; ties are outside the theorem).

Self-contained except for `OntologyAlignmentRepair`. No Mathlib. Zero sorry.
-/

import OntologyAlignmentRepair

namespace Sounio.OntologyAlignmentRepair

variable {α : Type}

-- ---------------------------------------------------------------------------
-- §1. Auxiliary list lemmas (not in this core)
-- ---------------------------------------------------------------------------

/-- Split a list at an occurrence of `a`. -/
theorem mem_split {a : α} {l : List α} (h : a ∈ l) : ∃ s t, l = s ++ a :: t := by
  induction l with
  | nil => cases h
  | cons x xs ih =>
      rw [List.mem_cons] at h
      cases h with
      | inl hxa => exact ⟨[], xs, by subst hxa; rfl⟩
      | inr hmem =>
          obtain ⟨s, t, rfl⟩ := ih hmem
          exact ⟨x :: s, t, rfl⟩

/-- From a nodup append `pre ++ m :: post`, `m` does not occur in `pre`. -/
theorem not_mem_pre_of_nodup_append {pre : List α} {m : α} {post : List α}
    (h : (pre ++ m :: post).Nodup) : m ∉ pre := by
  obtain ⟨-, -, hne⟩ := List.nodup_append.mp h
  intro hm
  exact hne m hm m (List.mem_cons.mpr (Or.inl rfl)) rfl

/-- `repair` distributes over list append. -/
theorem repair_append (C : α → α → Bool) (kept : List α) :
    ∀ l₁ l₂ : List α,
      repair C kept (l₁ ++ l₂) = repair C (repair C kept l₁) l₂ := by
  intro l₁
  induction l₁ generalizing kept with
  | nil => intro l₂; rfl
  | cons a as ih => intro l₂; exact ih (repairStep C kept a) l₂

-- ---------------------------------------------------------------------------
-- §2. Exact characterisation of the fold
-- ---------------------------------------------------------------------------

/-- **Unfolding characterisation**: `m` ends up kept iff it was already kept,
    or it occurs in the candidate list and, at the moment it is processed,
    conflicts with nothing kept so far. Fully general: no nodup, sorting, or
    decidability assumptions. -/
theorem mem_repair_iff {C : α → α → Bool} {kept ms : List α} {m : α} :
    m ∈ repair C kept ms ↔
      m ∈ kept ∨ ∃ pre post, ms = pre ++ m :: post ∧
        conflictsAny C m (repair C kept pre) = false := by
  constructor
  · intro h
    induction ms generalizing kept with
    | nil => exact Or.inl h
    | cons x xs ih =>
        have h2 : m ∈ repair C (repairStep C kept x) xs := h
        have h3 := ih h2
        cases h3 with
        | inl h4 =>
            obtain rfl | hmk := mem_repairStep h4
            · by_cases hmk2 : m ∈ kept
              · exact Or.inl hmk2
              · have hc : conflictsAny C m kept = false := by
                  cases hcc : conflictsAny C m kept with
                  | false => rfl
                  | true =>
                      have hstep : repairStep C kept m = kept := by
                        simp [repairStep, hcc]
                      rw [hstep] at h4
                      exact absurd h4 hmk2
                exact Or.inr ⟨[], xs, rfl, hc⟩
            · exact Or.inl hmk
        | inr h5 =>
            obtain ⟨pre, post, hxs, hc⟩ := h5
            exact Or.inr ⟨x :: pre, post, by subst hxs; rfl, hc⟩
  · intro h
    cases h with
    | inl hk => exact subset_repair hk
    | inr h2 =>
        obtain ⟨pre, post, rfl, hc⟩ := h2
        rw [repair_append]
        show m ∈ repair C (repairStep C (repair C kept pre) m) post
        have hstep : repairStep C (repair C kept pre) m = m :: repair C kept pre := by
          simp [repairStep, hc]
        rw [hstep]
        exact subset_repair (List.mem_cons.mpr (Or.inl rfl))

/-- **Fold-side fixpoint characterisation** (the priority independent set):
    for distinct confidences and a candidate list sorted by decreasing
    confidence, `m` survives the fold iff `m` is a candidate and every
    higher-confidence survivor is conflict-free with `m`. -/
theorem mem_repair_priority {C : α → α → Bool} {conf : α → Nat} {ms : List α}
    (hsorted : ms.Pairwise fun a b => conf b ≤ conf a)
    (hnodup : ms.Nodup)
    (hinj : ∀ a ∈ ms, ∀ b ∈ ms, conf a = conf b → a = b)
    {m : α} :
    m ∈ repair C [] ms ↔
      m ∈ ms ∧ ∀ k, k ∈ ms → conf m < conf k → k ∈ repair C [] ms →
        C m k = false := by
  constructor
  · intro h
    have hmem : m ∈ ms := mem_repair_nil h
    obtain ⟨pre, post, hsplit, hconf⟩ :=
      (mem_repair_iff.mp h).resolve_left List.not_mem_nil
    refine ⟨hmem, fun k hkm hlt hkR => ?_⟩
    have hkpre : k ∈ pre := by
      have hk' : k ∈ pre ++ m :: post := hsplit ▸ hkm
      rw [List.mem_append] at hk'
      cases hk' with
      | inl hk => exact hk
      | inr hk =>
          rw [List.mem_cons] at hk
          cases hk with
          | inl hkm2 => subst hkm2; exact absurd hlt (Nat.lt_irrefl _)
          | inr hkpost =>
              have hs : (pre ++ m :: post).Pairwise (fun a b => conf b ≤ conf a) :=
                hsplit ▸ hsorted
              obtain ⟨-, h2, -⟩ := List.pairwise_append.mp hs
              obtain ⟨h3, -⟩ := List.pairwise_cons.mp h2
              have hle := h3 k hkpost
              omega
    have hkRpre : k ∈ repair C [] pre := by
      have h1 : k ∈ repair C [] (pre ++ m :: post) := hsplit ▸ hkR
      rw [repair_append] at h1
      have h2 := mem_repair h1
      cases h2 with
      | inl hk => exact hk
      | inr hk =>
          rw [List.mem_cons] at hk
          cases hk with
          | inl hkm2 => subst hkm2; exact absurd hlt (Nat.lt_irrefl _)
          | inr hkpost =>
              have hs : (pre ++ m :: post).Nodup := hsplit ▸ hnodup
              obtain ⟨-, -, hne⟩ := List.nodup_append.mp hs
              exact absurd rfl (hne k hkpre k (List.mem_cons.mpr (Or.inr hkpost)))
    exact conflictsAny_false.mp hconf k hkRpre
  · rintro ⟨hmem, hno⟩
    obtain ⟨pre, post, hsplit⟩ := mem_split hmem
    apply mem_repair_iff.mpr
    refine Or.inr ⟨pre, post, hsplit, ?_⟩
    have hf : ∀ k', k' ∈ repair C [] pre → C m k' = false := by
      intro k' hk'
      have hkpre : k' ∈ pre := mem_repair_nil hk'
      have hkms : k' ∈ ms := by
        rw [hsplit]
        exact List.mem_append_left _ hkpre
      have hlt : conf m < conf k' := by
        have hs : (pre ++ m :: post).Pairwise (fun a b => conf b ≤ conf a) :=
          hsplit ▸ hsorted
        obtain ⟨-, -, h3⟩ := List.pairwise_append.mp hs
        have hle : conf m ≤ conf k' :=
          h3 k' hkpre m (List.mem_cons.mpr (Or.inl rfl))
        have hne : k' ≠ m := by
          intro e
          subst e
          exact not_mem_pre_of_nodup_append (hsplit ▸ hnodup) hkpre
        have hcne : conf m ≠ conf k' := by
          intro e
          exact hne (hinj k' hkms m hmem e.symm)
        omega
      have hkR : k' ∈ repair C [] ms := by
        rw [hsplit, repair_append]
        exact subset_repair hk'
      exact hno k' hkms hlt hkR
    exact conflictsAny_false.mpr hf

-- ---------------------------------------------------------------------------
-- §3. Uniqueness of the priority fixpoint
-- ---------------------------------------------------------------------------

/-- **Fixpoint uniqueness**: on a candidate list with confidences bounded by
    `B`, two sets satisfying the priority fixpoint ("`m` is in iff `m` is a
    candidate and no higher-confidence conflicting candidate is in") agree
    everywhere. Bounded strong induction on the confidence. -/
theorem fixpoint_unique {ms : List α} {C : α → α → Bool} {conf : α → Nat}
    {S T : α → Prop} {B : Nat}
    (hB : ∀ a ∈ ms, conf a ≤ B)
    (hS : ∀ m, S m ↔ (m ∈ ms ∧ ∀ k, k ∈ ms → C m k = true →
      conf m < conf k → ¬ S k))
    (hT : ∀ m, T m ↔ (m ∈ ms ∧ ∀ k, k ∈ ms → C m k = true →
      conf m < conf k → ¬ T k)) :
    ∀ m, S m ↔ T m := by
  suffices key : ∀ i, ∀ m, B ≤ conf m + i → (S m ↔ T m) by
    intro m
    exact key B m (Nat.le_add_left B (conf m))
  intro i
  induction i with
  | zero =>
      intro m hm
      have hge : ∀ k ∈ ms, conf k ≤ conf m := fun k hk => Nat.le_trans (hB k hk) hm
      rw [hS m, hT m]
      constructor
      · rintro ⟨hmem, -⟩
        exact ⟨hmem, fun k hk _ hlt => (Nat.not_lt_of_ge (hge k hk) hlt).elim⟩
      · rintro ⟨hmem, -⟩
        exact ⟨hmem, fun k hk _ hlt => (Nat.not_lt_of_ge (hge k hk) hlt).elim⟩
  | succ i ih =>
      intro m hm
      rw [hS m, hT m]
      constructor
      · rintro ⟨hmem, hno⟩
        refine ⟨hmem, fun k hk hc hlt => ?_⟩
        have hik : S k ↔ T k := ih k (by omega)
        exact fun hTk => hno k hk hc hlt (hik.mpr hTk)
      · rintro ⟨hmem, hno⟩
        refine ⟨hmem, fun k hk hc hlt => ?_⟩
        have hik : S k ↔ T k := ih k (by omega)
        exact fun hSk => hno k hk hc hlt (hik.mp hSk)

-- ---------------------------------------------------------------------------
-- §4. The pairwise drop-weaker greedy
-- ---------------------------------------------------------------------------

/-- One greedy step over the pair `p`: if both endpoints are still alive and
    conflict, drop the lower-confidence endpoint (ties drop `p.2`, exactly
    matching the `.sio` `conf[i] >= conf[j]` tie-break, though the theorems
    assume distinct confidences on conflicting pairs). -/
def greedyStep (C : α → α → Bool) (conf : α → Nat) [DecidableEq α]
    (d : List α) (p : α × α) : List α :=
  if p.1 ∈ d ∨ p.2 ∈ d then d
  else if C p.1 p.2 then (if conf p.1 < conf p.2 then p.1 :: d else p.2 :: d)
  else d

/-- The pairwise drop-weaker greedy: fold `greedyStep` over the pair list,
    accumulating dropped mappings. `m` survives iff `m ∉ greedyDrop C conf [] ps`. -/
def greedyDrop (C : α → α → Bool) (conf : α → Nat) [DecidableEq α]
    (d : List α) : List (α × α) → List α
  | [] => d
  | p :: ps => greedyDrop C conf (greedyStep C conf d p) ps

/-- The dropped set only grows within a step. -/
theorem greedyStep_subset [DecidableEq α] {C : α → α → Bool} {conf : α → Nat}
    {d : List α} {p : α × α} : d ⊆ greedyStep C conf d p := by
  intro a ha
  unfold greedyStep
  split
  · exact ha
  · split
    · split
      · exact List.mem_cons.mpr (Or.inr ha)
      · exact List.mem_cons.mpr (Or.inr ha)
    · exact ha

/-- The dropped set only grows along the scan. -/
theorem greedyDrop_subset [DecidableEq α] {C : α → α → Bool} {conf : α → Nat}
    {d : List α} {ps : List (α × α)} : d ⊆ greedyDrop C conf d ps := by
  induction ps generalizing d with
  | nil => exact List.Subset.refl d
  | cons p ps ih =>
      exact fun a ha => ih (greedyStep_subset ha)

/-- **Pair coverage**: any conflicting pair present in the scan loses at least
    one endpoint (the weaker one at scan time, or an endpoint already dropped
    earlier). -/
theorem mem_greedyDrop_of_pair [DecidableEq α] {C : α → α → Bool} {conf : α → Nat}
    (hdist : ∀ a b, C a b = true → conf a ≠ conf b)
    {d : List α} {ps : List (α × α)} {a b : α}
    (hmem : (a, b) ∈ ps) (hC : C a b = true) :
    a ∈ greedyDrop C conf d ps ∨ b ∈ greedyDrop C conf d ps := by
  induction ps generalizing d with
  | nil => cases hmem
  | cons p ps ih =>
      rw [List.mem_cons] at hmem
      cases hmem with
      | inr h => exact ih h
      | inl h =>
          subst h
          show a ∈ greedyDrop C conf (greedyStep C conf d (a, b)) ps ∨
            b ∈ greedyDrop C conf (greedyStep C conf d (a, b)) ps
          by_cases hd : a ∈ d ∨ b ∈ d
          · have hstep : greedyStep C conf d (a, b) = d := by
              simp [greedyStep, hd]
            rw [hstep]
            cases hd with
            | inl ha => exact Or.inl (greedyDrop_subset ha)
            | inr hb => exact Or.inr (greedyDrop_subset hb)
          · have hne : conf a ≠ conf b := hdist a b hC
            by_cases hle : conf a < conf b
            · have hstep : greedyStep C conf d (a, b) = a :: d := by
                simp [greedyStep, hd, hC, hle]
              rw [hstep]
              exact Or.inl (greedyDrop_subset (List.mem_cons.mpr (Or.inl rfl)))
            · have hstep : greedyStep C conf d (a, b) = b :: d := by
                simp [greedyStep, hd, hC, hle]
              rw [hstep]
              exact Or.inr (greedyDrop_subset (List.mem_cons.mpr (Or.inl rfl)))

/-- **Greedy soundness direction**: a greedy survivor has no higher-confidence
    conflicting survivor — every higher conflicting neighbour is dropped. -/
theorem mem_greedyDrop_of_higher_neighbor [DecidableEq α]
    {C : α → α → Bool} {conf : α → Nat} {ms : List α} {ps : List (α × α)}
    (hsym : ∀ a b, C a b = C b a)
    (hdist : ∀ a b, C a b = true → conf a ≠ conf b)
    (hcov : ∀ a b, a ∈ ms → b ∈ ms → C a b = true →
      (a, b) ∈ ps ∨ (b, a) ∈ ps)
    {d : List α} {m k : α}
    (hm : m ∈ ms) (hk : k ∈ ms) (hC : C m k = true) (_hlt : conf m < conf k)
    (hmd : m ∉ greedyDrop C conf d ps) :
    k ∈ greedyDrop C conf d ps := by
  cases hcov m k hm hk hC with
  | inl hp =>
      cases mem_greedyDrop_of_pair hdist hp hC with
      | inl h => exact absurd h hmd
      | inr h => exact h
  | inr hp =>
      have hC' : C k m = true := by
        rw [← hsym m k]
        exact hC
      cases mem_greedyDrop_of_pair hdist hp hC' with
      | inl h => exact h
      | inr h => exact absurd h hmd

/-- **Greedy witness**: under the cluster-graph hypothesis, every dropped
    mapping has a surviving higher-confidence conflicting neighbour. The drop
    chain `m ≺ e ≺ k' ≺ …` collapses because each link's endpoints all
    conflict with `m`. -/
theorem greedyDrop_witness [DecidableEq α] {C : α → α → Bool} {conf : α → Nat}
    (hsym : ∀ a b, C a b = C b a)
    (hclus : ∀ a b c, C a b = true → C b c = true → a ≠ c → C a c = true)
    (hdist : ∀ a b, C a b = true → conf a ≠ conf b)
    {ms : List α} :
    ∀ {ps : List (α × α)}, (∀ p ∈ ps, p.1 ∈ ms ∧ p.2 ∈ ms) →
    ∀ {d : List α} {m : α}, m ∈ greedyDrop C conf d ps →
      m ∈ d ∨ ∃ k, k ∈ ms ∧ C m k = true ∧ conf m < conf k ∧
        k ∉ greedyDrop C conf d ps := by
  intro ps
  induction ps with
  | nil => intro _ d m h; exact Or.inl h
  | cons p ps ih =>
      intro hps d m h
      have hps' : ∀ q ∈ ps, q.1 ∈ ms ∧ q.2 ∈ ms :=
        fun q hq => hps q (List.mem_cons.mpr (Or.inr hq))
      have h2 := ih hps' h
      cases h2 with
      | inr hw => exact Or.inr hw
      | inl hstep =>
          obtain ⟨x, y⟩ := p
          show m ∈ d ∨ ∃ k, k ∈ ms ∧ C m k = true ∧ conf m < conf k ∧
            k ∉ greedyDrop C conf (greedyStep C conf d (x, y)) ps
          by_cases hd : x ∈ d ∨ y ∈ d
          · have hs : greedyStep C conf d (x, y) = d := by simp [greedyStep, hd]
            rw [hs] at hstep
            exact Or.inl hstep
          · have hxnd : x ∉ d := fun hx => hd (Or.inl hx)
            have hynd : y ∉ d := fun hy => hd (Or.inr hy)
            by_cases hC : C x y = true
            · have hne : conf x ≠ conf y := hdist x y hC
              have hxms : x ∈ ms := (hps (x, y) (List.mem_cons.mpr (Or.inl rfl))).1
              have hyms : y ∈ ms := (hps (x, y) (List.mem_cons.mpr (Or.inl rfl))).2
              by_cases hlt : conf x < conf y
              · -- `x` is dropped here because of `y`
                have hs : greedyStep C conf d (x, y) = x :: d := by
                  simp [greedyStep, hd, hC, hlt]
                rw [hs] at hstep ⊢
                rw [List.mem_cons] at hstep
                cases hstep with
                | inr hmd => exact Or.inl hmd
                | inl hmx =>
                    subst m
                    -- now `m` is `x`: dropped while `y` was alive
                    by_cases hyF : y ∈ greedyDrop C conf (x :: d) ps
                    · have h3 := ih hps' hyF
                      cases h3 with
                      | inl hyin =>
                          rw [List.mem_cons] at hyin
                          cases hyin with
                          | inl hyx => exact absurd (congrArg conf hyx) hne.symm
                          | inr hyd => exact absurd hyd hynd
                      | inr hw =>
                          obtain ⟨k', hk'ms, hCyk', hltk', hk'not⟩ := hw
                          have hnk' : x ≠ k' := by
                            intro e
                            subst e
                            exact absurd (Nat.lt_trans hlt hltk') (Nat.lt_irrefl _)
                          have hCxk' : C x k' = true := hclus x y k' hC hCyk' hnk'
                          exact Or.inr ⟨k', hk'ms, hCxk', Nat.lt_trans hlt hltk', hk'not⟩
                    · exact Or.inr ⟨y, hyms, hC, hlt, hyF⟩
              · -- `y` is dropped here because of `x`
                have hs : greedyStep C conf d (x, y) = y :: d := by
                  simp [greedyStep, hd, hC, hlt]
                rw [hs] at hstep ⊢
                rw [List.mem_cons] at hstep
                cases hstep with
                | inr hmd => exact Or.inl hmd
                | inl hmy =>
                    subst m
                    -- now `m` is `y`: dropped while `x` was alive
                    have hCyx : C y x = true := by
                      rw [← hsym x y]
                      exact hC
                    have hlt' : conf y < conf x :=
                      Nat.lt_of_le_of_ne (Nat.le_of_not_lt hlt) hne.symm
                    by_cases hxF : x ∈ greedyDrop C conf (y :: d) ps
                    · have h3 := ih hps' hxF
                      cases h3 with
                      | inl hxin =>
                          rw [List.mem_cons] at hxin
                          cases hxin with
                          | inl hxy => exact absurd (congrArg conf hxy) hne
                          | inr hxd => exact absurd hxd hxnd
                      | inr hw =>
                          obtain ⟨k', hk'ms, hCxk', hltk', hk'not⟩ := hw
                          have hnk' : y ≠ k' := by
                            intro e
                            subst e
                            exact absurd (Nat.lt_trans hlt' hltk') (Nat.lt_irrefl _)
                          have hCyk' : C y k' = true := hclus y x k' hCyx hCxk' hnk'
                          exact Or.inr ⟨k', hk'ms, hCyk', Nat.lt_trans hlt' hltk', hk'not⟩
                    · exact Or.inr ⟨x, hxms, hCyx, hlt', hxF⟩
            · have hs : greedyStep C conf d (x, y) = d := by simp [greedyStep, hd, hC]
              rw [hs] at hstep
              exact Or.inl hstep

-- ---------------------------------------------------------------------------
-- §5. Main equivalence
-- ---------------------------------------------------------------------------

/-- The fold satisfies the priority fixpoint (contrapositive form of
    `mem_repair_priority`). -/
theorem repair_fixpoint {C : α → α → Bool} {conf : α → Nat} {ms : List α}
    (hsorted : ms.Pairwise fun a b => conf b ≤ conf a)
    (hnodup : ms.Nodup)
    (hinj : ∀ a ∈ ms, ∀ b ∈ ms, conf a = conf b → a = b) :
    ∀ m, m ∈ repair C [] ms ↔ (m ∈ ms ∧ ∀ k, k ∈ ms → C m k = true →
      conf m < conf k → k ∉ repair C [] ms) := by
  intro m
  rw [mem_repair_priority hsorted hnodup hinj]
  constructor
  · rintro ⟨hmem, hno⟩
    refine ⟨hmem, fun k hk hc hlt hkR => ?_⟩
    have hf := hno k hk hlt hkR
    rw [hf] at hc
    exact Bool.noConfusion hc
  · rintro ⟨hmem, hno⟩
    refine ⟨hmem, fun k hk hlt hkR => ?_⟩
    cases hc : C m k with
    | false => rfl
    | true => exact absurd hkR (hno k hk hc hlt)

/-- The greedy kept set satisfies the priority fixpoint. -/
theorem greedy_fixpoint [DecidableEq α] {C : α → α → Bool} {conf : α → Nat}
    {ms : List α} {ps : List (α × α)}
    (hsym : ∀ a b, C a b = C b a)
    (hclus : ∀ a b c, C a b = true → C b c = true → a ≠ c → C a c = true)
    (hdist : ∀ a b, C a b = true → conf a ≠ conf b)
    (hcov : ∀ a b, a ∈ ms → b ∈ ms → C a b = true →
      (a, b) ∈ ps ∨ (b, a) ∈ ps)
    (hps : ∀ p ∈ ps, p.1 ∈ ms ∧ p.2 ∈ ms) :
    ∀ m, (m ∈ ms ∧ m ∉ greedyDrop C conf [] ps) ↔
      (m ∈ ms ∧ ∀ k, k ∈ ms → C m k = true → conf m < conf k →
        ¬ (k ∈ ms ∧ k ∉ greedyDrop C conf [] ps)) := by
  intro m
  constructor
  · rintro ⟨hmem, hmd⟩
    refine ⟨hmem, fun k hk hc hlt ⟨_, hkd⟩ => ?_⟩
    exact hkd (mem_greedyDrop_of_higher_neighbor hsym hdist hcov hmem hk hc hlt hmd)
  · rintro ⟨hmem, hno⟩
    refine ⟨hmem, fun hmd => ?_⟩
    have hw := greedyDrop_witness hsym hclus hdist hps hmd
    cases hw with
    | inl h0 => exact absurd h0 List.not_mem_nil
    | inr h2 =>
        obtain ⟨k, hkms, hC, hlt, hknot⟩ := h2
        exact hno k hkms hC hlt ⟨hkms, hknot⟩

/-- **Main equivalence**: for distinct confidences, a sorted nodup candidate
    list, a symmetric cluster-graph conflict relation, and a pair list that
    covers all conflicting candidate pairs and ranges over the candidates,
    the pairwise drop-weaker greedy keeps exactly the priority fold's kept set. -/
theorem repair_iff_greedy [DecidableEq α] {C : α → α → Bool} {conf : α → Nat}
    {ms : List α} {ps : List (α × α)}
    (hsorted : ms.Pairwise fun a b => conf b ≤ conf a)
    (hnodup : ms.Nodup)
    (hinj : ∀ a ∈ ms, ∀ b ∈ ms, conf a = conf b → a = b)
    (hsym : ∀ a b, C a b = C b a)
    (hclus : ∀ a b c, C a b = true → C b c = true → a ≠ c → C a c = true)
    (hdist : ∀ a b, C a b = true → conf a ≠ conf b)
    (hcov : ∀ a b, a ∈ ms → b ∈ ms → C a b = true →
      (a, b) ∈ ps ∨ (b, a) ∈ ps)
    (hps : ∀ p ∈ ps, p.1 ∈ ms ∧ p.2 ∈ ms) :
    ∀ m, m ∈ repair C [] ms ↔ (m ∈ ms ∧ m ∉ greedyDrop C conf [] ps) := by
  have hS := repair_fixpoint (C := C) (conf := conf) hsorted hnodup hinj
  have hT := greedy_fixpoint (C := C) (conf := conf) hsym hclus hdist hcov hps
  have hB : ∀ a ∈ ms, conf a ≤ (ms.map conf).foldr max 0 := by
    have key : ∀ {l : List α} {a : α}, a ∈ l →
        conf a ≤ (l.map conf).foldr max 0 := by
      intro l
      induction l with
      | nil => intro a h; cases h
      | cons x xs ihl =>
          intro a h
          rw [List.mem_cons] at h
          rw [List.map_cons, List.foldr_cons]
          cases h with
          | inl hax => rw [hax]; exact Nat.le_max_left _ _
          | inr hmem => exact Nat.le_trans (ihl hmem) (Nat.le_max_right _ _)
    exact fun a ha => key ha
  exact fun m =>
    fixpoint_unique hB (fun m' => hS m') (fun m' => hT m') m

-- ---------------------------------------------------------------------------
-- §6. Concrete instance: the miniature UMLS-style scenario (Fin 5)
-- ---------------------------------------------------------------------------

/-- Matcher confidences of the `.sio` prototype, in hundredths:
    m0 = 0.30, m1 = 0.06, m2 = 0.95, m3 = 0.40, m4 = 0.80. All distinct. -/
def exConf5 : Fin 5 → Nat :=
  fun m => [30, 6, 95, 40, 80].getD m.val 0

/-- The conflicting pairs scanned by the `.sio` greedy (lexicographic order):
    C1 = {0,1}, C2 = {2,3}. -/
def exPairs : List (Fin 5 × Fin 5) := [(0, 1), (2, 3)]

/-- The greedy drops exactly m1 and m3 — the same mappings the fold drops. -/
theorem ex_greedyDrop :
    greedyDrop exConflicts exConf5 [] exPairs = [3, 1] := by
  native_decide

/-- **Instance of the main equivalence**, obtained by instantiating the
    general theorem; every hypothesis is discharged by `decide`. -/
theorem ex_repair_iff_greedy :
    ∀ m : Fin 5, m ∈ repair exConflicts [] exCandidates ↔
      (m ∈ exCandidates ∧ m ∉ greedyDrop exConflicts exConf5 [] exPairs) :=
  repair_iff_greedy
    (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide)
    (fun a b _ _ h => by
      have hd : ∀ a b : Fin 5, exConflicts a b = true →
          (a, b) ∈ exPairs ∨ (b, a) ∈ exPairs := by decide
      exact hd a b h)
    (by decide)

/-- Sanity check of the same fact by exhaustive evaluation. -/
theorem ex_repair_iff_greedy_check :
    ∀ m : Fin 5, m ∈ repair exConflicts [] exCandidates ↔
      (m ∈ exCandidates ∧ m ∉ greedyDrop exConflicts exConf5 [] exPairs) := by
  native_decide

-- ---------------------------------------------------------------------------
-- §7. Necessity of the cluster hypothesis (counterexample)
-- ---------------------------------------------------------------------------

/-- Conflict path of length 2 over `Fin 3`: 0—1 and 1—2, but not 0—2.
    Not a cluster graph. -/
def cxC : Fin 3 → Fin 3 → Bool := fun a b =>
  (a.val == 0 && b.val == 1) || (a.val == 1 && b.val == 0) ||
  (a.val == 1 && b.val == 2) || (a.val == 2 && b.val == 1)

/-- Distinct confidences 0 < 1 < 2. -/
def cxConf : Fin 3 → Nat := fun m => m.val

/-- Candidates in decreasing confidence order. -/
def cxMs : List (Fin 3) := [2, 1, 0]

/-- Pair scan order: (0,1) before (1,2) — as the `.sio` lexicographic scan. -/
def cxPairs : List (Fin 3 × Fin 3) := [(0, 1), (1, 2)]

/-- The fold keeps {0, 2}: 2 survives, 1 conflicts with 2 and drops,
    0 conflicts with nothing kept. -/
theorem cx_repair : repair cxC [] cxMs = [0, 2] := by
  native_decide

/-- The greedy keeps only {2}: scanning (0,1) drops 0, scanning (1,2) drops 1. -/
theorem cx_greedyDrop : greedyDrop cxC cxConf [] cxPairs = [1, 0] := by
  native_decide

/-- **Counterexample**: without the cluster hypothesis the bare "distinct
    confidences" equivalence is false — the greedy can drop a mapping (here 0)
    whose only higher-confidence conflicting neighbour (1) is itself later
    dropped by an even stronger mapping (2) that does not conflict with it. -/
theorem cx_equivalence_fails :
    ¬ (∀ m : Fin 3, m ∈ repair cxC [] cxMs ↔
      (m ∈ cxMs ∧ m ∉ greedyDrop cxC cxConf [] cxPairs)) := by
  native_decide

end Sounio.OntologyAlignmentRepair

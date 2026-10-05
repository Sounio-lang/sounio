/-
# Sounio.OntologyELPlusClosureComplete — the completeness frontier of the EL⁺ closure

Round 8 of the *ontology frontiers* research line. The previous round
(`OntologyELPlusClosureVerified.lean`) built the role-aware saturation engine
`seedS` / `crStep` / `closeSat` and proved it **sound**: every computed pair
of `closeSat` is `Der`-derivable (`subBPlus_sound`), every computed conflict
is a genuine derived conflict (`conflictBPlus_sound`). That file deliberately
left completeness open: the engine iterates a fixed six rounds
(`satRounds := 6`), enough for the worked SNOMED instance but not a fixpoint
in general — one round of `genTrans` only doubles the length of a
transitivity chain, so a TBox with a chain of 65 stated inclusions already
needs seven rounds.

This file closes the completeness gap via the **canonical-model argument**:

1. §1–§2 List and universe utilities: filter-length monotonicity and strict
   decrease (the pigeonhole engine of the fixpoint proof), `dedupN`
   nodupness, and membership facts about `conceptUniv` / `baseConcepts` —
   in particular `base_of_ex_mem_univ`: if `∃r.D` is in the saturation
   universe then `D` is a *base* concept, so every role-variant `∃s.D` is in
   the universe. This is the fact that lets the canonical model close under
   role-hierarchy and composition chaining.
2. §3 The fixpoint: `closeIter t k := crIter t k (seedS t, [])` is monotone
   in `k`, stays inside the finite fact set `conceptUniv² ∪ (roles ×
   conceptUniv × baseConcepts)` (`closeIter_inv`), so a counting argument
   (`missing`, `strict_progress`, `find_stable`) shows the iteration
   stabilises within `satFuel t := |allPairs| + |allTriples|` rounds.
   `closeSatF t := closeIter t (satFuel t)` is a genuine fixpoint of
   `crStep` (`closeSatF_fixpoint_fst` / `_snd`), hence closed under every
   completion rule (`closeSatF_trans`, `closeSatF_conjElim`,
   `closeSatF_conjIntro`, `closeSatF_stoR`, `closeSatF_Rmono`,
   `closeSatF_roleSub`, `closeSatF_comp`, `closeSatF_RtoS`). The new boolean
   queries `subBPlusC` / `conflictBPlusC` over `closeSatF` inherit soundness
   from `crIter_justified` (`subBPlusC_sound`, `conflictBPlusC_sound`).
3. §4–§5 The canonical model: the domain is the universe itself
   (`canonDom`), atoms and roles are read off the closure
   (`canonI` / `canonIR`). Since `Der` never mentions disjointness axioms,
   the right semantic target is satisfaction of the subsumption/role part of
   the TBox (`SatisfiesSR`, with its own soundness lemma `der_sound_sr`):
   the canonical model satisfies it (`canon_satisfies`) — role inclusions by
   the roleSub closure rule, compositions by the composition rule — and the
   **truth lemma** (`truth_lemma`) says truth in the canonical model is
   exactly membership in the closure, by structural induction on concepts.
4. §6 Completeness: `Der t C D` with `C, D ∈ conceptUniv t` implies
   `(C, D) ∈ (closeSatF t).1` — evaluate the derivation in the canonical
   model at `C` (true by reflexivity), apply the truth lemma. The conflict
   oracle is complete likewise. Hence `subBPlusC_iff` / `conflictBPlusC_iff`:
   the computed boolean answers are *exactly* the deductive closure.
   `subBPlus_le_subBPlusC` shows the round-7 engine's answers are a
   (generally strict) subset of the complete engine's.

Self-contained modulo `OntologyELPlusClosureVerified`. No Mathlib. Zero
sorry. No new axioms (the fixpoint search is constructive — `find_stable` is
proved by induction on the missing-fact count, not by classical choice).
-/

import OntologyELPlusClosureVerified

namespace Sounio.OntologyELPlus

open Sounio.OntologyELReasoner

-- ---------------------------------------------------------------------------
-- §1. List utilities: filter lengths and `dedupN` nodupness
-- ---------------------------------------------------------------------------

section ListUtil

variable {α : Type}

/-- Filtering with a pointwise-weaker predicate shortens the result. -/
theorem length_filter_le_of_imp {l : List α} {g₁ g₂ : α → Bool} :
    (∀ x ∈ l, g₁ x = true → g₂ x = true) →
      (l.filter g₁).length ≤ (l.filter g₂).length := by
  induction l with
  | nil => intro _; exact Nat.le_refl 0
  | cons a l ih =>
      intro h
      have h' : ∀ x ∈ l, g₁ x = true → g₂ x = true :=
        fun x hx => h x (List.mem_cons_of_mem a hx)
      by_cases h1 : g₁ a = true
      · have h2 : g₂ a = true := h a List.mem_cons_self h1
        rw [List.filter_cons, if_pos h1, List.filter_cons, if_pos h2,
          List.length_cons, List.length_cons]
        exact Nat.succ_le_succ (ih h')
      · rw [List.filter_cons, if_neg h1]
        by_cases h2 : g₂ a = true
        · rw [List.filter_cons, if_pos h2, List.length_cons]
          exact Nat.le_succ_of_le (ih h')
        · rw [List.filter_cons, if_neg h2]
          exact ih h'

/-- If the missing-set strictly loses a member (present in a nodup list),
    the filter length strictly decreases. -/
theorem length_filter_lt_of_mem {l : List α} {g₁ g₂ : α → Bool} :
    l.Nodup → (∀ y ∈ l, g₂ y = true → g₁ y = true) → ∀ {x : α}, x ∈ l →
      g₁ x = true → g₂ x = false → (l.filter g₂).length < (l.filter g₁).length := by
  induction l with
  | nil => intro _ _ x hx _ _; exact absurd hx List.not_mem_nil
  | cons a l ih =>
      intro hl himp x hx h1 h2
      obtain ⟨-, hl'⟩ := List.nodup_cons.mp hl
      have himp' : ∀ y ∈ l, g₂ y = true → g₁ y = true :=
        fun y hy => himp y (List.mem_cons_of_mem a hy)
      cases List.mem_cons.mp hx with
      | inl hxa =>
          subst hxa
          have h2' : ¬ (g₂ x = true) := by
            intro hh; rw [h2] at hh; exact Bool.noConfusion hh
          rw [List.filter_cons, if_neg h2', List.filter_cons, if_pos h1,
            List.length_cons]
          exact Nat.lt_succ_of_le (length_filter_le_of_imp himp')
      | inr hxl =>
          by_cases ha2 : g₂ a = true
          · have ha1 : g₁ a = true := himp a List.mem_cons_self ha2
            rw [List.filter_cons, if_pos ha2, List.filter_cons, if_pos ha1,
              List.length_cons, List.length_cons]
            exact Nat.succ_lt_succ (ih hl' himp' hxl h1 h2)
          · rw [List.filter_cons, if_neg ha2]
            by_cases ha1 : g₁ a = true
            · rw [List.filter_cons, if_pos ha1, List.length_cons]
              exact Nat.lt_trans (ih hl' himp' hxl h1 h2) (Nat.lt_succ_self _)
            · rw [List.filter_cons, if_neg ha1]
              exact ih hl' himp' hxl h1 h2

/-- `dedupN` output has no duplicates. -/
theorem nodup_dedupN [DecidableEq α] (l : List α) : (dedupN l).Nodup := by
  induction l with
  | nil => exact List.nodup_nil
  | cons a l ih =>
      show (if a ∈ dedupN l then dedupN l else a :: dedupN l).Nodup
      by_cases h : a ∈ dedupN l
      · rw [if_pos h]; exact ih
      · rw [if_neg h]; exact List.nodup_cons.mpr ⟨h, ih⟩

/-- A decidable universal over a list that fails has a counterexample —
    proved constructively (no classical choice). -/
theorem not_ball {l : List α} {p : α → Prop} [DecidablePred p]
    (h : ¬ ∀ x ∈ l, p x) : ∃ x ∈ l, ¬ p x :=
  Decidable.byContradiction fun hn =>
    h fun x hx => Decidable.byContradiction fun hxp => hn ⟨x, hx, hxp⟩

end ListUtil

-- ---------------------------------------------------------------------------
-- §2. Universe membership utilities
-- ---------------------------------------------------------------------------

/-- Every concept is a subconcept of itself. -/
theorem mem_subconcepts_self {α ρ : Type} (c : Concept α ρ) : c ∈ subconcepts c := by
  cases c with
  | atom a => exact List.mem_cons_self
  | top => exact List.mem_cons_self
  | conj c d => exact List.mem_cons_self
  | ex r c => exact List.mem_cons_self

/-- The left conjunct is a subconcept of the conjunction. -/
theorem sub_conj_left {α ρ : Type} {c d : Concept α ρ} : c ∈ subconcepts (.conj c d) :=
  List.mem_cons_of_mem _ (List.mem_append_left _ (mem_subconcepts_self c))

/-- The right conjunct is a subconcept of the conjunction. -/
theorem sub_conj_right {α ρ : Type} {c d : Concept α ρ} : d ∈ subconcepts (.conj c d) :=
  List.mem_cons_of_mem _ (List.mem_append_right _ (mem_subconcepts_self d))

/-- The filler is a subconcept of the existential. -/
theorem sub_ex_filler {α ρ : Type} {r : ρ} {c : Concept α ρ} : c ∈ subconcepts (.ex r c) :=
  List.mem_cons_of_mem _ (mem_subconcepts_self c)

section UniverseUtil

variable {n m : Nat}

/-- Base concepts lie in the universe. -/
theorem mem_univ_of_base {t : List (AxiomP (Fin n) (Fin m))} {C : Concept (Fin n) (Fin m)}
    (h : C ∈ baseConcepts t) : C ∈ conceptUniv t :=
  List.mem_append_left _ h

/-- ⊤ is a base concept. -/
theorem top_mem_base (t : List (AxiomP (Fin n) (Fin m))) : .top ∈ baseConcepts t :=
  List.mem_append_left _ List.mem_cons_self

/-- ⊤ is in the universe. -/
theorem top_mem_univ (t : List (AxiomP (Fin n) (Fin m))) : .top ∈ conceptUniv t :=
  mem_univ_of_base (top_mem_base t)

/-- Every role-variant of a base concept is in the universe. -/
theorem ex_mem_univ_of_base {t : List (AxiomP (Fin n) (Fin m))}
    {D : Concept (Fin n) (Fin m)} {r : Fin m}
    (h : D ∈ baseConcepts t) : .ex r D ∈ conceptUniv t :=
  List.mem_append_right _
    (List.mem_flatMap.mpr ⟨D, h, List.mem_map.mpr ⟨r, List.mem_finRange r, rfl⟩⟩)

/-- **Key universe fact**: if `∃r.D` is in the saturation universe, then `D`
    is a base concept — so all its role-variants are in the universe. This is
    what makes the canonical model closed under role-hierarchy and
    composition chaining. -/
theorem base_of_ex_mem_univ {t : List (AxiomP (Fin n) (Fin m))}
    {D : Concept (Fin n) (Fin m)} {r : Fin m}
    (h : .ex r D ∈ conceptUniv t) : D ∈ baseConcepts t := by
  rw [conceptUniv, List.mem_append] at h
  cases h with
  | inl hb => exact mem_baseConcepts_sub sub_ex_filler hb
  | inr hv =>
      rw [List.mem_flatMap] at hv
      obtain ⟨b, hb, hv⟩ := hv
      rw [List.mem_map] at hv
      obtain ⟨r', -, he⟩ := hv
      injection he with _ hd
      subst hd
      exact hb

end UniverseUtil

-- ---------------------------------------------------------------------------
-- §3. The fixpoint: `closeSatF` reaches a genuine `crStep` fixpoint
-- ---------------------------------------------------------------------------

section Fixpoint

variable {n m : Nat}

/-- The iterate of the saturation engine after `k` rounds. -/
abbrev closeIter (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) : SRel n m × RRel n m :=
  crIter t k (seedS t, [])

/-- Iteration commutes with one extra step on the outside. -/
theorem crIter_succ (t : List (AxiomP (Fin n) (Fin m))) (k : Nat)
    (SR : SRel n m × RRel n m) :
    crIter t (k + 1) SR = crStep t (crIter t k SR).1 (crIter t k SR).2 := by
  induction k generalizing SR with
  | zero => rfl
  | succ k ih => exact ih (crStep t SR.1 SR.2)

/-- `closeIter` at `k + 1` is one `crStep` of `closeIter` at `k`. -/
theorem closeIter_succ (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) :
    closeIter t (k + 1) = crStep t (closeIter t k).1 (closeIter t k).2 :=
  crIter_succ t k (seedS t, [])

theorem mem_closeIter_succ_fst {t : List (AxiomP (Fin n) (Fin m))} {k : Nat} {x} :
    x ∈ (closeIter t (k + 1)).1 ↔
      x ∈ (crStep t (closeIter t k).1 (closeIter t k).2).1 := by
  rw [closeIter_succ]

theorem mem_closeIter_succ_snd {t : List (AxiomP (Fin n) (Fin m))} {k : Nat} {e} :
    e ∈ (closeIter t (k + 1)).2 ↔
      e ∈ (crStep t (closeIter t k).1 (closeIter t k).2).2 := by
  rw [closeIter_succ]

-- §3.1 Monotonicity

/-- `crStep` keeps every old subsumption pair. -/
theorem mem_crStep_fst {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} {x} (hx : x ∈ S) : x ∈ (crStep t S R).1 := by
  rw [crStep_fst, mem_dedupN]
  exact List.mem_append_left _ hx

/-- `crStep` keeps every old role edge. -/
theorem mem_crStep_snd {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} {e} (he : e ∈ R) : e ∈ (crStep t S R).2 := by
  rw [crStep_snd, mem_dedupN]
  exact List.mem_append_left _ he

/-- Pair membership grows with the round count. -/
theorem mem_closeIter_mono_fst {t : List (AxiomP (Fin n) (Fin m))} {j k : Nat}
    (hjk : j ≤ k) {x} (hx : x ∈ (closeIter t j).1) : x ∈ (closeIter t k).1 := by
  induction hjk with
  | refl => exact hx
  | step _ ih => exact mem_closeIter_succ_fst.mpr (mem_crStep_fst ih)

/-- Edge membership grows with the round count. -/
theorem mem_closeIter_mono_snd {t : List (AxiomP (Fin n) (Fin m))} {j k : Nat}
    (hjk : j ≤ k) {e} (he : e ∈ (closeIter t j).2) : e ∈ (closeIter t k).2 := by
  induction hjk with
  | refl => exact he
  | step _ ih => exact mem_closeIter_succ_snd.mpr (mem_crStep_snd ih)

/-- The seed is contained in every iterate. -/
theorem mem_closeIter_seed {t : List (AxiomP (Fin n) (Fin m))} {x} (k : Nat)
    (hx : x ∈ seedS t) : x ∈ (closeIter t k).1 := by
  induction k with
  | zero => exact hx
  | succ k ih => exact mem_closeIter_succ_fst.mpr (mem_crStep_fst ih)

-- §3.2 Membership characterisations of `crStep`

theorem mem_crStep_fst_iff {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} {x} :
    x ∈ (crStep t S R).1 ↔ x ∈ S ∨ x ∈ genTrans S ∨ x ∈ genConjElim S ∨
      x ∈ genConjIntro S (conceptUniv t) ∨ x ∈ genRtoS R (conceptUniv t) := by
  rw [crStep_fst, mem_dedupN, List.mem_append, List.mem_append, List.mem_append,
    List.mem_append]

theorem mem_crStep_snd_iff {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} {x} :
    x ∈ (crStep t S R).2 ↔ x ∈ R ∨ x ∈ genStoR S ∨
      x ∈ genRmono R S (conceptUniv t) ∨ x ∈ genRroleSub t R (conceptUniv t) ∨
        x ∈ genRcomp t R (conceptUniv t) := by
  rw [crStep_snd, mem_dedupN, List.mem_append, List.mem_append, List.mem_append,
    List.mem_append]

-- §3.3 Generated facts lie in the round output

theorem of_genTrans {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m} {R : RRel n m}
    {x} (h : x ∈ genTrans S) : x ∈ (crStep t S R).1 := by
  rw [crStep_fst, mem_dedupN]
  exact List.mem_append_right _ (List.mem_append_left _ h)

theorem of_genConjElim {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} {x} (h : x ∈ genConjElim S) : x ∈ (crStep t S R).1 := by
  rw [crStep_fst, mem_dedupN]
  exact List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ h))

theorem of_genConjIntro {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} {x} (h : x ∈ genConjIntro S (conceptUniv t)) :
    x ∈ (crStep t S R).1 := by
  rw [crStep_fst, mem_dedupN]
  exact List.mem_append_right _
    (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ h)))

theorem of_genRtoS {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m} {R : RRel n m}
    {x} (h : x ∈ genRtoS R (conceptUniv t)) : x ∈ (crStep t S R).1 := by
  rw [crStep_fst, mem_dedupN]
  exact List.mem_append_right _
    (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ h)))

theorem of_genStoR {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m} {R : RRel n m}
    {x} (h : x ∈ genStoR S) : x ∈ (crStep t S R).2 := by
  rw [crStep_snd, mem_dedupN]
  exact List.mem_append_right _ (List.mem_append_left _ h)

theorem of_genRmono {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m} {R : RRel n m}
    {x} (h : x ∈ genRmono R S (conceptUniv t)) : x ∈ (crStep t S R).2 := by
  rw [crStep_snd, mem_dedupN]
  exact List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ h))

theorem of_genRroleSub {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} {x} (h : x ∈ genRroleSub t R (conceptUniv t)) :
    x ∈ (crStep t S R).2 := by
  rw [crStep_snd, mem_dedupN]
  exact List.mem_append_right _
    (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ h)))

theorem of_genRcomp {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m} {R : RRel n m}
    {x} (h : x ∈ genRcomp t R (conceptUniv t)) : x ∈ (crStep t S R).2 := by
  rw [crStep_snd, mem_dedupN]
  exact List.mem_append_right _
    (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ h)))

-- §3.4 Generators only see membership

theorem genTrans_ext {S₁ S₂ : SRel n m} (h : ∀ x, x ∈ S₁ ↔ x ∈ S₂) {x} :
    x ∈ genTrans S₁ ↔ x ∈ genTrans S₂ := by
  rw [mem_genTrans, mem_genTrans]
  constructor
  · intro ⟨p, hp, q, hq, h1, h2⟩; exact ⟨p, (h p).mp hp, q, (h q).mp hq, h1, h2⟩
  · intro ⟨p, hp, q, hq, h1, h2⟩; exact ⟨p, (h p).mpr hp, q, (h q).mpr hq, h1, h2⟩

theorem genConjElim_ext {S₁ S₂ : SRel n m} (h : ∀ x, x ∈ S₁ ↔ x ∈ S₂) {x} :
    x ∈ genConjElim S₁ ↔ x ∈ genConjElim S₂ := by
  rw [mem_genConjElim, mem_genConjElim]
  constructor
  · intro ⟨p, hp, d₁, d₂, h1, h2⟩; exact ⟨p, (h p).mp hp, d₁, d₂, h1, h2⟩
  · intro ⟨p, hp, d₁, d₂, h1, h2⟩; exact ⟨p, (h p).mpr hp, d₁, d₂, h1, h2⟩

theorem genConjIntro_ext {S₁ S₂ : SRel n m} (h : ∀ x, x ∈ S₁ ↔ x ∈ S₂)
    {univ : List (Concept (Fin n) (Fin m))} {x} :
    x ∈ genConjIntro S₁ univ ↔ x ∈ genConjIntro S₂ univ := by
  rw [mem_genConjIntro, mem_genConjIntro]
  constructor
  · intro ⟨p, hp, q, hq, h1, hu, h2⟩
    exact ⟨p, (h p).mp hp, q, (h q).mp hq, h1, hu, h2⟩
  · intro ⟨p, hp, q, hq, h1, hu, h2⟩
    exact ⟨p, (h p).mpr hp, q, (h q).mpr hq, h1, hu, h2⟩

theorem genRtoS_ext {R₁ R₂ : RRel n m} (h : ∀ e, e ∈ R₁ ↔ e ∈ R₂)
    {univ : List (Concept (Fin n) (Fin m))} {x} :
    x ∈ genRtoS R₁ univ ↔ x ∈ genRtoS R₂ univ := by
  rw [mem_genRtoS, mem_genRtoS]
  constructor
  · intro ⟨e, he, hu, hx⟩; exact ⟨e, (h e).mp he, hu, hx⟩
  · intro ⟨e, he, hu, hx⟩; exact ⟨e, (h e).mpr he, hu, hx⟩

theorem genStoR_ext {S₁ S₂ : SRel n m} (h : ∀ x, x ∈ S₁ ↔ x ∈ S₂) {x} :
    x ∈ genStoR S₁ ↔ x ∈ genStoR S₂ := by
  rw [mem_genStoR, mem_genStoR]
  constructor
  · intro ⟨p, hp, r, d, h1, h2⟩; exact ⟨p, (h p).mp hp, r, d, h1, h2⟩
  · intro ⟨p, hp, r, d, h1, h2⟩; exact ⟨p, (h p).mpr hp, r, d, h1, h2⟩

theorem genRmono_ext {R₁ R₂ : RRel n m} {S₁ S₂ : SRel n m}
    (hR : ∀ e, e ∈ R₁ ↔ e ∈ R₂) (hS : ∀ x, x ∈ S₁ ↔ x ∈ S₂)
    {univ : List (Concept (Fin n) (Fin m))} {x} :
    x ∈ genRmono R₁ S₁ univ ↔ x ∈ genRmono R₂ S₂ univ := by
  rw [mem_genRmono, mem_genRmono]
  constructor
  · intro ⟨e, he, p, hp, h1, hu, h2⟩
    exact ⟨e, (hR e).mp he, p, (hS p).mp hp, h1, hu, h2⟩
  · intro ⟨e, he, p, hp, h1, hu, h2⟩
    exact ⟨e, (hR e).mpr he, p, (hS p).mpr hp, h1, hu, h2⟩

theorem genRroleSub_ext {t : List (AxiomP (Fin n) (Fin m))} {R₁ R₂ : RRel n m}
    (h : ∀ e, e ∈ R₁ ↔ e ∈ R₂) {univ : List (Concept (Fin n) (Fin m))} {x} :
    x ∈ genRroleSub t R₁ univ ↔ x ∈ genRroleSub t R₂ univ := by
  rw [mem_genRroleSub, mem_genRroleSub]
  constructor
  · intro ⟨e, he, s, hs, h1, hu, h2⟩; exact ⟨e, (h e).mp he, s, hs, h1, hu, h2⟩
  · intro ⟨e, he, s, hs, h1, hu, h2⟩; exact ⟨e, (h e).mpr he, s, hs, h1, hu, h2⟩

theorem genRcomp_ext {t : List (AxiomP (Fin n) (Fin m))} {R₁ R₂ : RRel n m}
    (h : ∀ e, e ∈ R₁ ↔ e ∈ R₂) {univ : List (Concept (Fin n) (Fin m))} {x} :
    x ∈ genRcomp t R₁ univ ↔ x ∈ genRcomp t R₂ univ := by
  rw [mem_genRcomp, mem_genRcomp]
  constructor
  · intro ⟨e, he, f, hf, hef, r, s, u, hc, h1, h2, hu, h3⟩
    exact ⟨e, (h e).mp he, f, (h f).mp hf, hef, r, s, u, hc, h1, h2, hu, h3⟩
  · intro ⟨e, he, f, hf, hef, r, s, u, hc, h1, h2, hu, h3⟩
    exact ⟨e, (h e).mpr he, f, (h f).mpr hf, hef, r, s, u, hc, h1, h2, hu, h3⟩

-- §3.5 The finite fact space

/-- All possible subsumption pairs over the universe, deduplicated. -/
def allPairs (t : List (AxiomP (Fin n) (Fin m))) : SRel n m :=
  dedupN ((conceptUniv t).flatMap fun c => (conceptUniv t).map fun d => (c, d))

theorem mem_allPairs' {t : List (AxiomP (Fin n) (Fin m))} {x} :
    x ∈ allPairs t ↔ x.1 ∈ conceptUniv t ∧ x.2 ∈ conceptUniv t := by
  rw [allPairs, mem_dedupN, List.mem_flatMap]
  constructor
  · intro ⟨c, hc, hm⟩
    rw [List.mem_map] at hm
    obtain ⟨d, hd, he⟩ := hm
    subst he
    exact ⟨hc, hd⟩
  · intro ⟨h1, h2⟩
    exact ⟨x.1, h1, List.mem_map.mpr ⟨x.2, h2, Prod.eta x⟩⟩

/-- All possible role edges over the universe (fillers are base concepts),
    deduplicated. -/
def allTriples (t : List (AxiomP (Fin n) (Fin m))) : RRel n m :=
  dedupN ((List.finRange m).flatMap fun r => (conceptUniv t).flatMap fun c =>
    (baseConcepts t).map fun d => (r, c, d))

theorem mem_allTriples' {t : List (AxiomP (Fin n) (Fin m))} {e} :
    e ∈ allTriples t ↔ e.2.1 ∈ conceptUniv t ∧ e.2.2 ∈ baseConcepts t := by
  rw [allTriples, mem_dedupN, List.mem_flatMap]
  constructor
  · intro ⟨r, _, hm⟩
    rw [List.mem_flatMap] at hm
    obtain ⟨c, hc, hm⟩ := hm
    rw [List.mem_map] at hm
    obtain ⟨d, hd, he⟩ := hm
    subst he
    exact ⟨hc, hd⟩
  · intro ⟨h1, h2⟩
    refine ⟨e.1, List.mem_finRange _, List.mem_flatMap.mpr ⟨e.2.1, h1,
      List.mem_map.mpr ⟨e.2.2, h2, ?_⟩⟩⟩
    have h2' := Prod.eta e.2
    have h1' := Prod.eta e
    rw [h2', h1']

-- §3.6 The closure stays inside the finite fact space

/-- The seed satisfies the fact-space invariant. -/
theorem seedS_inv {t : List (AxiomP (Fin n) (Fin m))} :
    (∀ p ∈ seedS t, p.1 ∈ conceptUniv t ∧ p.2 ∈ conceptUniv t) ∧
    (∀ e ∈ ([] : RRel n m), e.2.1 ∈ conceptUniv t ∧ e.2.2 ∈ baseConcepts t) := by
  constructor
  · intro x hx
    rw [seedS, List.mem_append] at hx
    cases hx with
    | inl hx =>
        rw [List.mem_flatMap] at hx
        obtain ⟨c, hc, hx⟩ := hx
        rw [List.mem_cons, List.mem_singleton] at hx
        cases hx with
        | inl he => subst he; exact ⟨hc, hc⟩
        | inr he => subst he; exact ⟨hc, top_mem_univ t⟩
    | inr hx =>
        rw [List.mem_filterMap] at hx
        obtain ⟨a, ha, hsome⟩ := hx
        cases a with
        | sub c d =>
            simp only at hsome
            injection hsome with hx
            subst hx
            exact ⟨mem_conceptUniv_of_mem_axiom ha
                     (List.mem_append_left _ (mem_subconcepts_self _)),
                   mem_conceptUniv_of_mem_axiom ha
                     (List.mem_append_right _ (mem_subconcepts_self _))⟩
        | disj c d =>
            simp only at hsome
            nomatch hsome
        | roleSub r s =>
            simp only at hsome
            nomatch hsome
        | roleComp r s u =>
            simp only at hsome
            nomatch hsome
  · intro e he
    exact absurd he List.not_mem_nil

/-- One saturation round preserves the fact-space invariant. -/
theorem crStep_inv {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m} {R : RRel n m}
    (hS : ∀ p ∈ S, p.1 ∈ conceptUniv t ∧ p.2 ∈ conceptUniv t)
    (hR : ∀ e ∈ R, e.2.1 ∈ conceptUniv t ∧ e.2.2 ∈ baseConcepts t) :
    (∀ p ∈ (crStep t S R).1, p.1 ∈ conceptUniv t ∧ p.2 ∈ conceptUniv t) ∧
    (∀ e ∈ (crStep t S R).2, e.2.1 ∈ conceptUniv t ∧ e.2.2 ∈ baseConcepts t) := by
  constructor
  · intro x hx
    rw [mem_crStep_fst_iff] at hx
    cases hx with
    | inl h => exact hS x h
    | inr hx =>
        cases hx with
        | inl h =>
            rw [mem_genTrans] at h
            obtain ⟨p, hp, q, hq, -, hxe⟩ := h
            subst hxe
            exact ⟨(hS p hp).1, (hS q hq).2⟩
        | inr hx =>
            cases hx with
            | inl h =>
                rw [mem_genConjElim] at h
                obtain ⟨p, hp, d₁, d₂, hp2, hxor⟩ := h
                have hpU := hS p hp
                cases hxor with
                | inl hxe =>
                    subst hxe
                    have hd1 : d₁ ∈ subconcepts p.2 := by rw [hp2]; exact sub_conj_left
                    exact ⟨hpU.1, mem_conceptUniv_sub hd1 hpU.2⟩
                | inr hxe =>
                    subst hxe
                    have hd2 : d₂ ∈ subconcepts p.2 := by rw [hp2]; exact sub_conj_right
                    exact ⟨hpU.1, mem_conceptUniv_sub hd2 hpU.2⟩
            | inr hx =>
                cases hx with
                | inl h =>
                    rw [mem_genConjIntro] at h
                    obtain ⟨p, hp, _, _, _, hu, hxe⟩ := h
                    subst hxe
                    exact ⟨(hS p hp).1, hu⟩
                | inr h =>
                    rw [mem_genRtoS] at h
                    obtain ⟨e, he, hu, hxe⟩ := h
                    subst hxe
                    exact ⟨(hR e he).1, hu⟩
  · intro x hx
    rw [mem_crStep_snd_iff] at hx
    cases hx with
    | inl h => exact hR x h
    | inr hx =>
        cases hx with
        | inl h =>
            rw [mem_genStoR] at h
            obtain ⟨p, hp, r, d, hp2, hxe⟩ := h
            subst hxe
            have hpU := hS p hp
            rw [hp2] at hpU
            exact ⟨hpU.1, base_of_ex_mem_univ hpU.2⟩
        | inr hx =>
            cases hx with
            | inl h =>
                rw [mem_genRmono] at h
                obtain ⟨e, he, _, _, _, hu, hxe⟩ := h
                subst hxe
                exact ⟨(hR e he).1, base_of_ex_mem_univ hu⟩
            | inr hx =>
                cases hx with
                | inl h =>
                    rw [mem_genRroleSub] at h
                    obtain ⟨e, he, _, _, _, _, hxe⟩ := h
                    subst hxe
                    exact hR e he
                | inr h =>
                    rw [mem_genRcomp] at h
                    obtain ⟨e, he, f, hf, _, _, _, _, _, _, _, _, hxe⟩ := h
                    subst hxe
                    exact ⟨(hR e he).1, (hR f hf).2⟩

/-- Every iterate stays inside the finite fact space. -/
theorem closeIter_inv (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) :
    (∀ p ∈ (closeIter t k).1, p.1 ∈ conceptUniv t ∧ p.2 ∈ conceptUniv t) ∧
    (∀ e ∈ (closeIter t k).2, e.2.1 ∈ conceptUniv t ∧ e.2.2 ∈ baseConcepts t) := by
  induction k with
  | zero => exact seedS_inv
  | succ k ih =>
      rw [closeIter_succ]
      exact crStep_inv ih.1 ih.2

end Fixpoint

-- ---------------------------------------------------------------------------
-- §3.7 Stability by counting missing facts
-- ---------------------------------------------------------------------------

section Stability

variable {n m : Nat}

/-- Universe pairs not yet computed at round `k`. -/
abbrev missingS (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) : Nat :=
  ((allPairs t).filter fun p => decide (p ∉ (closeIter t k).1)).length

/-- Universe role edges not yet computed at round `k`. -/
abbrev missingR (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) : Nat :=
  ((allTriples t).filter fun e => decide (e ∉ (closeIter t k).2)).length

/-- The decreasing measure: facts of the finite fact space not yet computed. -/
abbrev missing (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) : Nat :=
  missingS t k + missingR t k

/-- Missing pairs can only shrink from one round to the next. -/
theorem missingS_mono (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) :
    missingS t (k + 1) ≤ missingS t k :=
  length_filter_le_of_imp fun _ _ hp =>
    decide_eq_true fun hpk =>
      of_decide_eq_true hp (mem_closeIter_mono_fst (Nat.le_succ k) hpk)

/-- Missing edges can only shrink from one round to the next. -/
theorem missingR_mono (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) :
    missingR t (k + 1) ≤ missingR t k :=
  length_filter_le_of_imp fun _ _ he =>
    decide_eq_true fun hek =>
      of_decide_eq_true he (mem_closeIter_mono_snd (Nat.le_succ k) hek)

/-- Round `k` is stable when round `k + 1` adds no new fact. -/
abbrev StableAt (t : List (AxiomP (Fin n) (Fin m))) (k : Nat) : Prop :=
  (∀ x ∈ (closeIter t (k + 1)).1, x ∈ (closeIter t k).1) ∧
  (∀ e ∈ (closeIter t (k + 1)).2, e ∈ (closeIter t k).2)

/-- If nothing is missing, every fact the invariant permits is already
    computed, so the round is stable. -/
theorem stable_of_missing_eq_zero {t : List (AxiomP (Fin n) (Fin m))} {k : Nat}
    (h : missing t k = 0) : StableAt t k := by
  obtain ⟨hS, hR⟩ := Nat.add_eq_zero_iff.mp h
  have hzS : (allPairs t).filter (fun p => decide (p ∉ (closeIter t k).1)) = [] :=
    List.length_eq_zero_iff.mp hS
  have hzR : (allTriples t).filter (fun e => decide (e ∉ (closeIter t k).2)) = [] :=
    List.length_eq_zero_iff.mp hR
  constructor
  · intro x hx
    have hinv := (closeIter_inv t (k + 1)).1 x hx
    have hxP : x ∈ allPairs t := mem_allPairs'.mpr hinv
    refine Decidable.byContradiction fun hxk => ?_
    have hg : decide (x ∉ (closeIter t k).1) = true := decide_eq_true hxk
    have hxF : x ∈ (allPairs t).filter (fun p => decide (p ∉ (closeIter t k).1)) :=
      List.mem_filter.mpr ⟨hxP, hg⟩
    rw [hzS] at hxF
    exact absurd hxF List.not_mem_nil
  · intro e he
    have hinv := (closeIter_inv t (k + 1)).2 e he
    have heT : e ∈ allTriples t := mem_allTriples'.mpr hinv
    refine Decidable.byContradiction fun hek => ?_
    have hg : decide (e ∉ (closeIter t k).2) = true := decide_eq_true hek
    have heF : e ∈ (allTriples t).filter (fun e => decide (e ∉ (closeIter t k).2)) :=
      List.mem_filter.mpr ⟨heT, hg⟩
    rw [hzR] at heF
    exact absurd heF List.not_mem_nil

/-- **Strict progress**: an unstable round strictly decreases the missing
    count (the new fact lives in the finite fact space by `closeIter_inv`). -/
theorem strict_progress {t : List (AxiomP (Fin n) (Fin m))} {k : Nat}
    (h : ¬ StableAt t k) : missing t (k + 1) < missing t k := by
  by_cases hA : ∀ x ∈ (closeIter t (k + 1)).1, x ∈ (closeIter t k).1
  · have hB : ¬ ∀ e ∈ (closeIter t (k + 1)).2, e ∈ (closeIter t k).2 :=
      fun hB => h ⟨hA, hB⟩
    obtain ⟨e, he, hne⟩ := not_ball hB
    have hinv := (closeIter_inv t (k + 1)).2 e he
    have heT : e ∈ allTriples t := mem_allTriples'.mpr hinv
    have himp : ∀ y ∈ allTriples t,
        decide (y ∉ (closeIter t (k + 1)).2) = true →
          decide (y ∉ (closeIter t k).2) = true :=
      fun y _ hy =>
        decide_eq_true fun hyk =>
          of_decide_eq_true hy (mem_closeIter_mono_snd (Nat.le_succ k) hyk)
    have h1 : decide (e ∉ (closeIter t k).2) = true := decide_eq_true hne
    have h2 : decide (e ∉ (closeIter t (k + 1)).2) = false :=
      Bool.eq_false_iff.mpr (fun hg => of_decide_eq_true hg he)
    have hlt : missingR t (k + 1) < missingR t k :=
      length_filter_lt_of_mem (nodup_dedupN _) himp heT h1 h2
    exact Nat.add_lt_add_of_le_of_lt (missingS_mono t k) hlt
  · obtain ⟨x, hx, hnx⟩ := not_ball hA
    have hinv := (closeIter_inv t (k + 1)).1 x hx
    have hxP : x ∈ allPairs t := mem_allPairs'.mpr hinv
    have himp : ∀ y ∈ allPairs t,
        decide (y ∉ (closeIter t (k + 1)).1) = true →
          decide (y ∉ (closeIter t k).1) = true :=
      fun y _ hy =>
        decide_eq_true fun hyk =>
          of_decide_eq_true hy (mem_closeIter_mono_fst (Nat.le_succ k) hyk)
    have h1 : decide (x ∉ (closeIter t k).1) = true := decide_eq_true hnx
    have h2 : decide (x ∉ (closeIter t (k + 1)).1) = false :=
      Bool.eq_false_iff.mpr (fun hg => of_decide_eq_true hg hx)
    have hlt : missingS t (k + 1) < missingS t k :=
      length_filter_lt_of_mem (nodup_dedupN _) himp hxP h1 h2
    exact Nat.add_lt_add_of_lt_of_le hlt (missingR_mono t k)

/-- **Fixpoint search, constructively**: within `missing t k` further rounds
    a stable round is reached. -/
theorem find_stable (t : List (AxiomP (Fin n) (Fin m))) :
    ∀ (fuel k : Nat), missing t k ≤ fuel →
      ∃ j, k ≤ j ∧ j ≤ k + fuel ∧ StableAt t j := by
  intro fuel
  induction fuel with
  | zero =>
      intro k h
      exact ⟨k, Nat.le_refl k, Nat.le_refl k,
        stable_of_missing_eq_zero (Nat.eq_zero_of_le_zero h)⟩
  | succ fuel ih =>
      intro k h
      by_cases hk : StableAt t k
      · exact ⟨k, Nat.le_refl k, by omega, hk⟩
      · have hlt := strict_progress (t := t) (k := k) hk
        obtain ⟨j, hkj, hjf, hsj⟩ := ih (k + 1) (by omega)
        exact ⟨j, by omega, by omega, hsj⟩

/-- Stability persists: once stable, the membership of both components is
    fixed forever (generators only see membership). -/
theorem stableAt_persists {t : List (AxiomP (Fin n) (Fin m))} {k : Nat}
    (hk : StableAt t k) : StableAt t (k + 1) := by
  obtain ⟨hS, hR⟩ := hk
  have hiffS : ∀ y, y ∈ (closeIter t (k + 1)).1 ↔ y ∈ (closeIter t k).1 :=
    fun y => ⟨hS y, fun h => mem_closeIter_mono_fst (Nat.le_succ k) h⟩
  have hiffR : ∀ e, e ∈ (closeIter t (k + 1)).2 ↔ e ∈ (closeIter t k).2 :=
    fun e => ⟨hR e, fun h => mem_closeIter_mono_snd (Nat.le_succ k) h⟩
  constructor
  · intro x hx
    rw [mem_closeIter_succ_fst, mem_crStep_fst_iff] at hx
    cases hx with
    | inl h => exact mem_closeIter_mono_fst (Nat.le_succ k) (hS x h)
    | inr hx =>
        cases hx with
        | inl h =>
            exact mem_closeIter_succ_fst.mpr (of_genTrans ((genTrans_ext hiffS).mp h))
        | inr hx =>
            cases hx with
            | inl h =>
                exact mem_closeIter_succ_fst.mpr
                  (of_genConjElim ((genConjElim_ext hiffS).mp h))
            | inr hx =>
                cases hx with
                | inl h =>
                    exact mem_closeIter_succ_fst.mpr
                      (of_genConjIntro ((genConjIntro_ext hiffS).mp h))
                | inr h =>
                    exact mem_closeIter_succ_fst.mpr
                      (of_genRtoS ((genRtoS_ext hiffR).mp h))
  · intro x hx
    rw [mem_closeIter_succ_snd, mem_crStep_snd_iff] at hx
    cases hx with
    | inl h => exact mem_closeIter_mono_snd (Nat.le_succ k) (hR x h)
    | inr hx =>
        cases hx with
        | inl h =>
            exact mem_closeIter_succ_snd.mpr (of_genStoR ((genStoR_ext hiffS).mp h))
        | inr hx =>
            cases hx with
            | inl h =>
                exact mem_closeIter_succ_snd.mpr
                  (of_genRmono ((genRmono_ext hiffR hiffS).mp h))
            | inr hx =>
                cases hx with
                | inl h =>
                    exact mem_closeIter_succ_snd.mpr
                      (of_genRroleSub ((genRroleSub_ext hiffR).mp h))
                | inr h =>
                    exact mem_closeIter_succ_snd.mpr
                      (of_genRcomp ((genRcomp_ext hiffR).mp h))

/-- Stability at `j` gives memberwise equality and stability at every later
    round. -/
theorem stableAt_ge {t : List (AxiomP (Fin n) (Fin m))} {j k : Nat}
    (hk : StableAt t j) (hjk : j ≤ k) :
    (∀ y, y ∈ (closeIter t k).1 ↔ y ∈ (closeIter t j).1) ∧
    (∀ e, e ∈ (closeIter t k).2 ↔ e ∈ (closeIter t j).2) ∧
    StableAt t k := by
  induction hjk with
  | refl => exact ⟨fun _ => Iff.rfl, fun _ => Iff.rfl, hk⟩
  | step _ ih =>
      obtain ⟨ih1, ih2, ihS⟩ := ih
      exact ⟨fun y => ⟨fun hy => (ih1 y).mp (ihS.1 y hy),
                       fun hy => mem_closeIter_mono_fst (Nat.le_succ _) ((ih1 y).mpr hy)⟩,
             fun e => ⟨fun he => (ih2 e).mp (ihS.2 e he),
                       fun he => mem_closeIter_mono_snd (Nat.le_succ _) ((ih2 e).mpr he)⟩,
             stableAt_persists ihS⟩

end Stability

-- ---------------------------------------------------------------------------
-- §3.8 The fixpoint closure and its closure rules
-- ---------------------------------------------------------------------------

section CloseSatF

variable {n m : Nat}

/-- Enough fuel to reach the fixpoint: the size of the finite fact space. -/
def satFuel (t : List (AxiomP (Fin n) (Fin m))) : Nat :=
  (allPairs t).length + (allTriples t).length

/-- **The complete role-aware closure**: iterate until the fixpoint is
    provably reached (contrast round 7's `closeSat`, which stops after six
    rounds and is sound but not complete in general). -/
def closeSatF (t : List (AxiomP (Fin n) (Fin m))) : SRel n m × RRel n m :=
  closeIter t (satFuel t)

/-- The initial missing count fits in the fuel bound. -/
theorem missing_zero_le_satFuel (t : List (AxiomP (Fin n) (Fin m))) :
    missing t 0 ≤ satFuel t :=
  Nat.add_le_add (List.length_filter_le _ _) (List.length_filter_le _ _)

/-- The closure round is stable. -/
theorem closeSatF_stable (t : List (AxiomP (Fin n) (Fin m))) : StableAt t (satFuel t) := by
  obtain ⟨j, -, hj, hsj⟩ := find_stable t (missing t 0) 0 (Nat.le_refl _)
  exact (stableAt_ge hsj (by have h2 := missing_zero_le_satFuel t; omega)).2.2

/-- **The closure is a fixpoint of `crStep`** (pairs). -/
theorem closeSatF_fixpoint_fst {t : List (AxiomP (Fin n) (Fin m))} {x} :
    x ∈ (crStep t (closeSatF t).1 (closeSatF t).2).1 ↔ x ∈ (closeSatF t).1 := by
  have h := closeSatF_stable t
  rw [show crStep t (closeSatF t).1 (closeSatF t).2 = closeIter t (satFuel t + 1) from
    (closeIter_succ t (satFuel t)).symm]
  exact ⟨h.1 x, fun hx => mem_closeIter_mono_fst (Nat.le_succ _) hx⟩

/-- **The closure is a fixpoint of `crStep`** (role edges). -/
theorem closeSatF_fixpoint_snd {t : List (AxiomP (Fin n) (Fin m))} {e} :
    e ∈ (crStep t (closeSatF t).1 (closeSatF t).2).2 ↔ e ∈ (closeSatF t).2 := by
  have h := closeSatF_stable t
  rw [show crStep t (closeSatF t).1 (closeSatF t).2 = closeIter t (satFuel t + 1) from
    (closeIter_succ t (satFuel t)).symm]
  exact ⟨h.2 e, fun he => mem_closeIter_mono_snd (Nat.le_succ _) he⟩

-- §3.8.1 The closure is closed under every completion rule

theorem closeSatF_trans {t : List (AxiomP (Fin n) (Fin m))} {C D E : Concept (Fin n) (Fin m)}
    (hCD : (C, D) ∈ (closeSatF t).1) (hDE : (D, E) ∈ (closeSatF t).1) :
    (C, E) ∈ (closeSatF t).1 :=
  closeSatF_fixpoint_fst.mp
    (of_genTrans (mem_genTrans.mpr ⟨(C, D), hCD, (D, E), hDE, rfl, rfl⟩))

theorem closeSatF_conjElim {t : List (AxiomP (Fin n) (Fin m))}
    {C D₁ D₂ : Concept (Fin n) (Fin m)}
    (h : (C, .conj D₁ D₂) ∈ (closeSatF t).1) :
    (C, D₁) ∈ (closeSatF t).1 ∧ (C, D₂) ∈ (closeSatF t).1 :=
  ⟨closeSatF_fixpoint_fst.mp
     (of_genConjElim (mem_genConjElim.mpr ⟨(C, .conj D₁ D₂), h, D₁, D₂, rfl, Or.inl rfl⟩)),
   closeSatF_fixpoint_fst.mp
     (of_genConjElim (mem_genConjElim.mpr ⟨(C, .conj D₁ D₂), h, D₁, D₂, rfl, Or.inr rfl⟩))⟩

theorem closeSatF_conjIntro {t : List (AxiomP (Fin n) (Fin m))}
    {C D₁ D₂ : Concept (Fin n) (Fin m)}
    (h1 : (C, D₁) ∈ (closeSatF t).1) (h2 : (C, D₂) ∈ (closeSatF t).1)
    (hu : .conj D₁ D₂ ∈ conceptUniv t) : (C, .conj D₁ D₂) ∈ (closeSatF t).1 :=
  closeSatF_fixpoint_fst.mp
    (of_genConjIntro (mem_genConjIntro.mpr ⟨(C, D₁), h1, (C, D₂), h2, rfl, hu, rfl⟩))

theorem closeSatF_stoR {t : List (AxiomP (Fin n) (Fin m))} {C D : Concept (Fin n) (Fin m)}
    {r : Fin m} (h : (C, .ex r D) ∈ (closeSatF t).1) : (r, C, D) ∈ (closeSatF t).2 :=
  closeSatF_fixpoint_snd.mp
    (of_genStoR (mem_genStoR.mpr ⟨(C, .ex r D), h, r, D, rfl, rfl⟩))

theorem closeSatF_Rmono {t : List (AxiomP (Fin n) (Fin m))}
    {C D D' : Concept (Fin n) (Fin m)} {r : Fin m}
    (he : (r, C, D) ∈ (closeSatF t).2) (hp : (D, D') ∈ (closeSatF t).1)
    (hu : .ex r D' ∈ conceptUniv t) : (r, C, D') ∈ (closeSatF t).2 :=
  closeSatF_fixpoint_snd.mp
    (of_genRmono (mem_genRmono.mpr ⟨(r, C, D), he, (D, D'), hp, rfl, hu, rfl⟩))

theorem closeSatF_roleSub {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)} {r s : Fin m}
    (he : (r, C, D) ∈ (closeSatF t).2) (hrole : roleSubB t r s = true)
    (hu : .ex s D ∈ conceptUniv t) : (s, C, D) ∈ (closeSatF t).2 :=
  closeSatF_fixpoint_snd.mp
    (of_genRroleSub (mem_genRroleSub.mpr
      ⟨(r, C, D), he, s, List.mem_finRange s, hrole, hu, rfl⟩))

theorem closeSatF_comp {t : List (AxiomP (Fin n) (Fin m))}
    {C D E : Concept (Fin n) (Fin m)} {r s u : Fin m}
    (he : (r, C, D) ∈ (closeSatF t).2) (hf : (s, D, E) ∈ (closeSatF t).2)
    (hcomp : AxiomP.roleComp r s u ∈ t) (hu : .ex u E ∈ conceptUniv t) :
    (u, C, E) ∈ (closeSatF t).2 :=
  closeSatF_fixpoint_snd.mp
    (of_genRcomp (mem_genRcomp.mpr
      ⟨(r, C, D), he, (s, D, E), hf, rfl, r, s, u, hcomp, rfl, rfl, hu, rfl⟩))

theorem closeSatF_RtoS {t : List (AxiomP (Fin n) (Fin m))} {C D : Concept (Fin n) (Fin m)}
    {r : Fin m} (he : (r, C, D) ∈ (closeSatF t).2) (hu : .ex r D ∈ conceptUniv t) :
    (C, .ex r D) ∈ (closeSatF t).1 :=
  closeSatF_fixpoint_fst.mp
    (of_genRtoS (mem_genRtoS.mpr ⟨(r, C, D), he, hu, rfl⟩))

-- §3.8.2 Seed facts are in the closure

theorem seed_refl_mem {t : List (AxiomP (Fin n) (Fin m))} {C : Concept (Fin n) (Fin m)}
    (hC : C ∈ conceptUniv t) : (C, C) ∈ (closeSatF t).1 :=
  mem_closeIter_seed _
    (List.mem_append_left _ (List.mem_flatMap.mpr ⟨C, hC, List.mem_cons_self⟩))

theorem seed_top_mem {t : List (AxiomP (Fin n) (Fin m))} {C : Concept (Fin n) (Fin m)}
    (hC : C ∈ conceptUniv t) : (C, .top) ∈ (closeSatF t).1 :=
  mem_closeIter_seed _
    (List.mem_append_left _ (List.mem_flatMap.mpr
      ⟨C, hC, List.mem_cons_of_mem _ (List.mem_singleton.mpr rfl)⟩))

theorem seed_axiom_mem {t : List (AxiomP (Fin n) (Fin m))} {c d : Concept (Fin n) (Fin m)}
    (h : AxiomP.sub c d ∈ t) : (c, d) ∈ (closeSatF t).1 :=
  mem_closeIter_seed _
    (List.mem_append_right _ (List.mem_filterMap.mpr ⟨.sub c d, h, rfl⟩))

-- §3.8.3 Soundness of the complete closure (inherited from round 7)

/-- The complete closure is justified: soundness holds for any round count. -/
theorem closeSatF_justified {t : List (AxiomP (Fin n) (Fin m))} :
    SatJustified t (closeSatF t).1 (closeSatF t).2 :=
  crIter_justified _ seedS_justified

/-- Boolean subsumption over the complete closure. -/
def subBPlusC (t : List (AxiomP (Fin n) (Fin m))) (C D : Concept (Fin n) (Fin m)) :
    Bool :=
  decide ((C, D) ∈ (closeSatF t).1)

/-- **Complete-closure soundness**: a computed subsumption is derivable. -/
theorem subBPlusC_sound {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)}
    (h : subBPlusC t C D = true) : Der t C D := by
  obtain ⟨hS, -⟩ := closeSatF_justified (t := t)
  exact hS (C, D) (of_decide_eq_true h)

/-- Boolean conflict oracle over the complete closure. -/
def conflictBPlusC (t : List (AxiomP (Fin n) (Fin m))) (C C' : Concept (Fin n) (Fin m)) :
    Bool :=
  let S := (closeSatF t).1
  S.any fun p => S.any fun q =>
    decide (p.1 = C ∧ q.1 = C' ∧
      (AxiomP.disj p.2 q.2 ∈ t ∨ AxiomP.disj q.2 p.2 ∈ t))

/-- **Oracle soundness over the complete closure**. -/
theorem conflictBPlusC_sound {t : List (AxiomP (Fin n) (Fin m))}
    {C C' : Concept (Fin n) (Fin m)}
    (h : conflictBPlusC t C C' = true) : DerivedConflictP t C C' := by
  have h' : ((closeSatF t).1.any fun p => (closeSatF t).1.any fun q =>
      decide (p.1 = C ∧ q.1 = C' ∧
        (AxiomP.disj p.2 q.2 ∈ t ∨ AxiomP.disj q.2 p.2 ∈ t))) = true := h
  rw [List.any_eq_true] at h'
  obtain ⟨p, hpS, h'⟩ := h'
  rw [List.any_eq_true] at h'
  obtain ⟨q, hqS, h'⟩ := h'
  obtain ⟨hpC, hqC', hD⟩ := of_decide_eq_true h'
  obtain ⟨hS, -⟩ := closeSatF_justified (t := t)
  have h1 : Der t C p.2 := by
    have hp := hS p hpS
    rw [hpC] at hp
    exact hp
  have h2 : Der t C' q.2 := by
    have hq := hS q hqS
    rw [hqC'] at hq
    exact hq
  exact ⟨p.2, q.2, h1, h2, hD⟩

end CloseSatF

-- ---------------------------------------------------------------------------
-- §4. Subsumption/role-only satisfaction and its soundness lemma
-- ---------------------------------------------------------------------------

section SRSemantics

variable {α ρ ι : Type}

/-- Satisfaction of the subsumption and role parts of a TBox. Disjointness
    axioms are omitted: `Der` never uses them (its `ofAxiom` rule only fires
    on `sub`), and the canonical model of an incoherent TBox cannot satisfy
    them — this is the right semantic target for completeness of
    subsumption. -/
def SatisfiesSR (I : α → ι → Prop) (IR : ρ → ι → ι → Prop)
    (t : List (AxiomP α ρ)) : Prop :=
  (∀ c d, AxiomP.sub c d ∈ t → ∀ x, meaning I IR c x → meaning I IR d x) ∧
  (∀ r s, AxiomP.roleSub r s ∈ t → ∀ x y, IR r x y → IR s x y) ∧
  (∀ r s u, AxiomP.roleComp r s u ∈ t → ∀ x y z, IR r x y → IR s y z → IR u x z)

/-- Full satisfaction implies subsumption/role satisfaction. -/
theorem satisfiesSR_of_satisfiesP {I : α → ι → Prop} {IR : ρ → ι → ι → Prop}
    {t : List (AxiomP α ρ)} (hI : SatisfiesP I IR t) : SatisfiesSR I IR t := by
  refine ⟨?_, ?_, ?_⟩
  · intro c d hmem; exact hI _ hmem
  · intro r s hmem; exact hI _ hmem
  · intro r s u hmem; exact hI _ hmem

/-- **Soundness against `SatisfiesSR`**: every derivable subsumption holds in
    every model of the subsumption/role part of the TBox. Mirrors
    `der_sound` case for case. -/
theorem der_sound_sr {I : α → ι → Prop} {IR : ρ → ι → ι → Prop}
    {t : List (AxiomP α ρ)} {c d : Concept α ρ}
    (h : Der t c d) (hI : SatisfiesSR I IR t) :
    ∀ x, meaning I IR c x → meaning I IR d x := by
  induction h with
  | ofAxiom hmem =>
      intro x hx
      exact hI.1 _ _ hmem x hx
  | refl =>
      intro x hx
      exact hx
  | trans _ _ ih1 ih2 =>
      intro x hx
      exact ih2 x (ih1 x hx)
  | conjIntro _ _ ih1 ih2 =>
      intro x hx
      exact ⟨ih1 x hx, ih2 x hx⟩
  | conjElimLeft _ ih =>
      intro x hx
      exact (ih x hx).1
  | conjElimRight _ ih =>
      intro x hx
      exact (ih x hx).2
  | exMono _ ih =>
      intro x hx
      obtain ⟨y, hr, hy⟩ := hx
      exact ⟨y, hr, ih y hy⟩
  | exRoleSub _ hmem ih =>
      intro x hx
      obtain ⟨y, hr, hy⟩ := ih x hx
      exact ⟨y, hI.2.1 _ _ hmem x y hr, hy⟩
  | exComp _ hmem ih =>
      intro x hx
      obtain ⟨y, hr, hy⟩ := ih x hx
      obtain ⟨z, hs, hz⟩ := hy
      exact ⟨z, hI.2.2 _ _ _ hmem x y z hr hs, hz⟩
  | topRule =>
      intro x _
      exact True.intro

end SRSemantics

-- ---------------------------------------------------------------------------
-- §5. The canonical model over the concept universe
-- ---------------------------------------------------------------------------

section CanonicalModel

variable {n m : Nat}

/-- The canonical domain: the concepts of the saturation universe. -/
abbrev canonDom (t : List (AxiomP (Fin n) (Fin m))) :=
  {C : Concept (Fin n) (Fin m) // C ∈ conceptUniv t}

/-- Canonical class interpretation: `a` holds of `X` when `X ⊑ a` is in the
    closure. -/
def canonI (t : List (AxiomP (Fin n) (Fin m))) : Fin n → canonDom t → Prop :=
  fun a X => (X.1, .atom a) ∈ (closeSatF t).1

/-- Canonical role interpretation: `r` relates `X` to `Y` when the edge
    `(r, X, Y)` is in the closure. -/
def canonIR (t : List (AxiomP (Fin n) (Fin m))) : Fin m → canonDom t → canonDom t → Prop :=
  fun r X Y => (r, X.1, Y.1) ∈ (closeSatF t).2

/-- **The truth lemma**: for every universe concept, truth in the canonical
    model is exactly membership in the closure. The existential case is the
    crux: `stoR` reads a computed `∃r.C` off into an edge whose filler is a
    domain element (true of itself by reflexivity), and `Rmono` + `RtoS`
    fold a witness edge back into a computed pair. -/
theorem truth_lemma {t : List (AxiomP (Fin n) (Fin m))} :
    ∀ (C : Concept (Fin n) (Fin m)), C ∈ conceptUniv t →
      ∀ X : canonDom t,
        meaning (canonI t) (canonIR t) C X ↔ (X.1, C) ∈ (closeSatF t).1 := by
  intro C
  induction C with
  | atom a =>
      intro _ X
      exact Iff.rfl
  | top =>
      intro _ X
      exact ⟨fun _ => seed_top_mem X.2, fun _ => True.intro⟩
  | conj c d ihc ihd =>
      intro hC X
      have hc : c ∈ conceptUniv t := mem_conceptUniv_sub sub_conj_left hC
      have hd : d ∈ conceptUniv t := mem_conceptUniv_sub sub_conj_right hC
      constructor
      · intro h
        exact closeSatF_conjIntro ((ihc hc X).mp h.1) ((ihd hd X).mp h.2) hC
      · intro h
        obtain ⟨h1, h2⟩ := closeSatF_conjElim h
        exact ⟨(ihc hc X).mpr h1, (ihd hd X).mpr h2⟩
  | ex r c ih =>
      intro hC X
      have hcb : c ∈ baseConcepts t := base_of_ex_mem_univ hC
      have hc : c ∈ conceptUniv t := mem_univ_of_base hcb
      constructor
      · intro h
        obtain ⟨Y, hr, hY⟩ := h
        have h1 : (Y.1, c) ∈ (closeSatF t).1 := (ih hc Y).mp hY
        have h2 : (r, X.1, c) ∈ (closeSatF t).2 := closeSatF_Rmono hr h1 hC
        exact closeSatF_RtoS h2 hC
      · intro h
        have hr : (r, X.1, c) ∈ (closeSatF t).2 := closeSatF_stoR h
        exact ⟨⟨c, hc⟩, hr, (ih hc ⟨c, hc⟩).mpr (seed_refl_mem hc)⟩

/-- **The canonical model satisfies the subsumption/role part of the TBox**:
    stated inclusions by seed + transitivity through the truth lemma, role
    inclusions by the roleSub closure rule (the edge filler is always a base
    concept, so the role-variant is in the universe), compositions by the
    composition closure rule. -/
theorem canon_satisfies {t : List (AxiomP (Fin n) (Fin m))} :
    SatisfiesSR (canonI t) (canonIR t) t := by
  refine ⟨?_, ?_, ?_⟩
  · intro c d hmem X hX
    have hc : c ∈ conceptUniv t :=
      mem_conceptUniv_of_mem_axiom hmem (List.mem_append_left _ (mem_subconcepts_self _))
    have hd : d ∈ conceptUniv t :=
      mem_conceptUniv_of_mem_axiom hmem (List.mem_append_right _ (mem_subconcepts_self _))
    exact (truth_lemma d hd X).mpr
      (closeSatF_trans ((truth_lemma c hc X).mp hX) (seed_axiom_mem hmem))
  · intro r s hmem X Y hXY
    have hyb : Y.1 ∈ baseConcepts t :=
      ((closeIter_inv t (satFuel t)).2 (r, X.1, Y.1) hXY).2
    exact closeSatF_roleSub hXY (roleSubB_complete (RoleDer.ofAxiom hmem))
      (ex_mem_univ_of_base hyb)
  · intro r s u hmem X Y Z hXY hYZ
    have hzb : Z.1 ∈ baseConcepts t :=
      ((closeIter_inv t (satFuel t)).2 (s, Y.1, Z.1) hYZ).2
    exact closeSatF_comp hXY hYZ hmem (ex_mem_univ_of_base hzb)

end CanonicalModel

-- ---------------------------------------------------------------------------
-- §6. Completeness: the closure computes exactly the deductive closure
-- ---------------------------------------------------------------------------

section Completeness

variable {n m : Nat}

/-- **Completeness, membership form**: every derivable subsumption between
    universe concepts is computed. Evaluate the derivation in the canonical
    model at `C` (true by reflexivity), then read membership off through the
    truth lemma. -/
theorem mem_closeSatF_of_der {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)}
    (h : Der t C D) (hC : C ∈ conceptUniv t) (hD : D ∈ conceptUniv t) :
    (C, D) ∈ (closeSatF t).1 := by
  have hX : meaning (canonI t) (canonIR t) C ⟨C, hC⟩ :=
    (truth_lemma C hC ⟨C, hC⟩).mpr (seed_refl_mem hC)
  have hY := der_sound_sr h canon_satisfies ⟨C, hC⟩ hX
  exact (truth_lemma D hD ⟨C, hC⟩).mp hY

/-- **Completeness of the boolean closure**: `subBPlusC` computes every
    derivable subsumption between universe concepts. -/
theorem subBPlusC_complete {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)}
    (h : Der t C D) (hC : C ∈ conceptUniv t) (hD : D ∈ conceptUniv t) :
    subBPlusC t C D = true :=
  decide_eq_true (mem_closeSatF_of_der h hC hD)

/-- **Completeness of the conflict oracle**: `conflictBPlusC` computes every
    derived conflict between universe concepts. -/
theorem conflictBPlusC_complete {t : List (AxiomP (Fin n) (Fin m))}
    {C C' : Concept (Fin n) (Fin m)}
    (h : DerivedConflictP t C C') (hC : C ∈ conceptUniv t)
    (hC' : C' ∈ conceptUniv t) : conflictBPlusC t C C' = true := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  have hd1 : d₁ ∈ conceptUniv t := by
    cases hd with
    | inl hm =>
        exact mem_conceptUniv_of_mem_axiom hm
          (List.mem_append_left _ (mem_subconcepts_self _))
    | inr hm =>
        exact mem_conceptUniv_of_mem_axiom hm
          (List.mem_append_right _ (mem_subconcepts_self _))
  have hd2 : d₂ ∈ conceptUniv t := by
    cases hd with
    | inl hm =>
        exact mem_conceptUniv_of_mem_axiom hm
          (List.mem_append_right _ (mem_subconcepts_self _))
    | inr hm =>
        exact mem_conceptUniv_of_mem_axiom hm
          (List.mem_append_left _ (mem_subconcepts_self _))
  have hp : (C, d₁) ∈ (closeSatF t).1 := mem_closeSatF_of_der h1 hC hd1
  have hq : (C', d₂) ∈ (closeSatF t).1 := mem_closeSatF_of_der h2 hC' hd2
  have h' : ((closeSatF t).1.any fun p => (closeSatF t).1.any fun q =>
      decide (p.1 = C ∧ q.1 = C' ∧
        (AxiomP.disj p.2 q.2 ∈ t ∨ AxiomP.disj q.2 p.2 ∈ t))) = true := by
    rw [List.any_eq_true]
    exact ⟨(C, d₁), hp, by
      rw [List.any_eq_true]
      exact ⟨(C', d₂), hq, decide_eq_true ⟨rfl, rfl, hd⟩⟩⟩
  exact h'

/-- **The boolean closure is exactly the deductive closure** (over the
    universe): soundness from round 7's invariant, completeness from the
    canonical model. -/
theorem subBPlusC_iff {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)}
    (hC : C ∈ conceptUniv t) (hD : D ∈ conceptUniv t) :
    subBPlusC t C D = true ↔ Der t C D :=
  ⟨subBPlusC_sound, fun h => subBPlusC_complete h hC hD⟩

/-- **The boolean oracle is exactly the derived-conflict relation** (over
    the universe). -/
theorem conflictBPlusC_iff {t : List (AxiomP (Fin n) (Fin m))}
    {C C' : Concept (Fin n) (Fin m)}
    (hC : C ∈ conceptUniv t) (hC' : C' ∈ conceptUniv t) :
    conflictBPlusC t C C' = true ↔ DerivedConflictP t C C' :=
  ⟨conflictBPlusC_sound, fun h => conflictBPlusC_complete h hC hC'⟩

/-- Round 7's six-round answers embed into the complete engine: everything
    the old engine computes, the complete closure computes (and, for chains
    needing more than six rounds, strictly more). -/
theorem subBPlus_le_subBPlusC {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)}
    (h : subBPlus t C D = true) : subBPlusC t C D = true :=
  subBPlusC_complete (subBPlus_sound h)
    ((closeIter_inv t satRounds).1 (C, D) (of_decide_eq_true h)).1
    ((closeIter_inv t satRounds).1 (C, D) (of_decide_eq_true h)).2

/-- Round 7's conflict answers embed into the complete engine. -/
theorem conflictBPlus_le_conflictBPlusC {t : List (AxiomP (Fin n) (Fin m))}
    {C C' : Concept (Fin n) (Fin m)}
    (h : conflictBPlus t C C' = true) : conflictBPlusC t C C' = true := by
  have h' : ((closeSat t).1.any fun p => (closeSat t).1.any fun q =>
      decide (p.1 = C ∧ q.1 = C' ∧
        (AxiomP.disj p.2 q.2 ∈ t ∨ AxiomP.disj q.2 p.2 ∈ t))) = true := h
  rw [List.any_eq_true] at h'
  obtain ⟨p, hpS, h'⟩ := h'
  rw [List.any_eq_true] at h'
  obtain ⟨q, hqS, hdec⟩ := h'
  obtain ⟨hpC, hqC', -⟩ := of_decide_eq_true hdec
  have hCU : C ∈ conceptUniv t := by
    rw [← hpC]
    exact ((closeIter_inv t satRounds).1 p hpS).1
  have hC'U : C' ∈ conceptUniv t := by
    rw [← hqC']
    exact ((closeIter_inv t satRounds).1 q hqS).1
  exact conflictBPlusC_complete (conflictBPlus_sound h) hCU hC'U

end Completeness

-- ---------------------------------------------------------------------------
-- §7. Instance checks
-- ---------------------------------------------------------------------------

section InstanceChecks

/-- A minimal two-class TBox for computational smoke tests (the complete
    engine's fuel bound makes `native_decide` feasible only on tiny
    instances; the SNOMED instance is checked through the theorems). -/
def microTBox : List (AxiomP (Fin 2) (Fin 1)) :=
  [ .sub (.atom 0) (.atom 1) ]

/-- The complete engine computes the stated inclusion. -/
theorem check_micro_sub_0_1 : subBPlusC microTBox (.atom 0) (.atom 1) = true := by
  native_decide

/-- The complete engine does not invent the reverse inclusion. -/
theorem check_micro_sub_1_0 : subBPlusC microTBox (.atom 1) (.atom 0) = false := by
  native_decide

/-- **The SNOMED flagship fact, through completeness** (not computation):
    `Pneumonia ⊑* ∃RoleGroup.Organ` is computed by the complete engine —
    the same fact the constructor derivation and the six-round engine
    reached, now certified by the completeness theorem. -/
theorem check_subBPlusC_snomed_rg_organ :
    subBPlusC snomedTBox (.atom 4) (.ex 0 (.atom 3)) = true :=
  subBPlusC_complete der_pneumonia_rg_organ_via_comp (by decide) (by decide)

/-- Round-trip: the completely-computed subsumption re-derives into `Der`
    through soundness. -/
theorem cc_der_rg_organ : Der snomedTBox (.atom 4) (.ex 0 (.atom 3)) :=
  subBPlusC_sound check_subBPlusC_snomed_rg_organ

/-- The pneumonia–drug conflict, through completeness. -/
theorem check_conflictBPlusC_snomed :
    conflictBPlusC snomedTBox (.atom 4) (.atom 5) = true :=
  conflictBPlusC_complete conflictP_pneumonia_drug (by decide) (by decide)

/-- The drug-induced disorder self-conflict, through completeness. -/
theorem check_conflictBPlusC_did :
    conflictBPlusC snomedTBox (.atom 7) (.atom 7) = true :=
  conflictBPlusC_complete incoherentP_did (by decide) (by decide)

/-- A round-7 computed answer embeds into the complete engine. -/
theorem check_subBPlus_le_snomed : subBPlusC snomedTBox (.atom 4) (.atom 1) = true :=
  subBPlus_le_subBPlusC (by native_decide)

end InstanceChecks

end Sounio.OntologyELPlus

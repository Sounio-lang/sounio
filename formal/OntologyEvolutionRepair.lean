import OntologyEvolution

/-!
# Sounio.OntologyEvolutionRepair — guarded evolution with axiom removal

Formal companion to
`artifacts/ontology-frontiers/consistent-ontology-evolution/version_chain_removal.sio`
(frontier: `consistent-ontology-evolution`; see the `FRONTIER.md` there).
This file closes the lacuna documented in `OntologyEvolution.lean`: *only
axiom addition is modelled*. Here the edit language gains **removal** —
surgery on the active axiom set during evolution — which connects the
evolution frontier with the repair frontier (`epistemic-alignment-repair`).

## What is proved here

1. `consistent_sublist` — **general lemma**: any sublist of a consistent
   version is consistent (induction on the sublist derivation).
2. `consistent_removeAxiom` — (a) removing an axiom from a consistent
   version keeps it consistent (removal deletes ALL occurrences).
3. `mem_versions2_consistent` / `consistent_evolve2` — (b) the a priori
   chain invariant generalises to edit scripts mixing `add` and `remove`:
   every recorded version is consistent.
4. `repair_retry` — (c) repair-then-retry: if `add a` is rejected against
   `v` with conflict witness `k`, and `k` is the ONLY conflicting partner
   of `a` in `v`, then after `remove k` the edit `add a` is accepted and
   the resulting version is consistent.
5. A concrete `Fin 6` instance (§5) extends the example of
   `OntologyEvolution.lean`: `add 4` is rejected against {3,2,1} (axiom 4
   contradicts axiom 2), `remove 2` unblocks it, re-adding 4 is accepted,
   and the final version {4,3,1} is consistent — checked by `native_decide`.

Removal is filter-based over a Boolean equality oracle `beq` that is
required to decide true equality (`hbeq`); the instance uses the `BEq` of
`Fin 6`. Depends on `OntologyEvolution.lean` (and transitively
`OntologyAlignmentRepair.lean`). No Mathlib. Zero sorry. No new axioms.
-/

namespace Sounio.OntologyEvolutionRepair

open Sounio.OntologyEvolution (Consistent applyEdit consistent_applyEdit)
open Sounio.OntologyAlignmentRepair (conflictsAny conflictsAny_false)

variable {α : Type}

-- ---------------------------------------------------------------------------
-- §1. Edits: addition and removal
-- ---------------------------------------------------------------------------

/-- An edit either adds an axiom (guarded) or removes one (always
    succeeds). -/
inductive Edit (α : Type) where
  | add : α → Edit α
  | remove : α → Edit α

/-- Remove ALL occurrences of `a` from the version (filter-based over the
    Boolean equality oracle `beq`). Removal always succeeds. -/
def removeAxiom (beq : α → α → Bool) (a : α) (v : List α) : List α :=
  v.filter (fun x => !beq x a)

/-- Guarded edit with removal: `add` keeps the a priori guard of
    `OntologyEvolution.applyEdit`; `remove` is unconditional. -/
def applyEdit2 (C : α → α → Bool) (beq : α → α → Bool) (v : List α) :
    Edit α → List α
  | .add a => applyEdit C v a
  | .remove a => removeAxiom beq a v

-- ---------------------------------------------------------------------------
-- §2. Removal preserves consistency
-- ---------------------------------------------------------------------------

/-- **General lemma**: any sublist of a consistent version is consistent.
    By induction on the sublist derivation. -/
theorem consistent_sublist {C : α → α → Bool} {w v : List α}
    (hsub : w.Sublist v) (h : Consistent C v) : Consistent C w := by
  induction hsub with
  | slnil => exact h
  | cons _ _ ih =>
      -- skip the head of v: Consistent is preserved on the tail
      exact ih h.2
  | cons_cons _ s ih =>
      -- keep the head: non-conflict against every kept element of the tail
      exact ⟨fun x hx => h.1 x (List.Sublist.mem hx s), ih h.2⟩

/-- (a) Removing an axiom from a consistent version keeps it consistent. -/
theorem consistent_removeAxiom {C : α → α → Bool} {beq : α → α → Bool}
    {v : List α} {a : α} (h : Consistent C v) :
    Consistent C (removeAxiom beq a v) :=
  consistent_sublist List.filter_sublist h

/-- One guarded step — add or remove — preserves consistency. -/
theorem consistent_applyEdit2 {C : α → α → Bool} {beq : α → α → Bool}
    {v : List α} (h : Consistent C v) (e : Edit α) :
    Consistent C (applyEdit2 C beq v e) := by
  cases e with
  | add a => exact consistent_applyEdit h
  | remove a => exact consistent_removeAxiom h

-- ---------------------------------------------------------------------------
-- §3. Version chains with removal
-- ---------------------------------------------------------------------------

/-- The version chain for mixed edit scripts: every intermediate version
    is recorded. -/
def versions2 (C : α → α → Bool) (beq : α → α → Bool) (v : List α) :
    List (Edit α) → List (List α)
  | [] => [v]
  | e :: edits => v :: versions2 C beq (applyEdit2 C beq v e) edits

/-- The final version after a mixed edit script. -/
def evolve2 (C : α → α → Bool) (beq : α → α → Bool) (v : List α) :
    List (Edit α) → List α
  | [] => v
  | e :: edits => evolve2 C beq (applyEdit2 C beq v e) edits

/-- (b) **A priori invariant, generalised**: every version in the chain is
    consistent, provided the initial version is — for scripts mixing `add`
    and `remove`. -/
theorem mem_versions2_consistent {C : α → α → Bool} {beq : α → α → Bool}
    {v : List α} (h : Consistent C v) {edits : List (Edit α)} {w : List α}
    (hw : w ∈ versions2 C beq v edits) : Consistent C w := by
  induction edits generalizing v with
  | nil =>
      have hw' : w ∈ [v] := hw
      rw [List.mem_singleton] at hw'
      rw [hw']
      exact h
  | cons e edits ih =>
      have hw' : w ∈ v :: versions2 C beq (applyEdit2 C beq v e) edits := hw
      rw [List.mem_cons] at hw'
      cases hw' with
      | inl hwv =>
          rw [hwv]
          exact h
      | inr hmem =>
          exact ih (consistent_applyEdit2 h e) hmem

/-- The final version of a mixed edit script is consistent. -/
theorem consistent_evolve2 {C : α → α → Bool} {beq : α → α → Bool}
    {v : List α} (h : Consistent C v) (edits : List (Edit α)) :
    Consistent C (evolve2 C beq v edits) := by
  induction edits generalizing v with
  | nil => exact h
  | cons e edits ih => exact ih (consistent_applyEdit2 h e)

-- ---------------------------------------------------------------------------
-- §4. Repair-then-retry
-- ---------------------------------------------------------------------------

/-- (c) **Repair-then-retry.** If `add a` is rejected against version `v`
    — equivalently (by `conflictsAny_true`) there is a conflict witness
    `k ∈ v` with `C a k = true` — and `k` is the ONLY conflicting partner
    of `a` in `v`, then removing `k` unblocks the edit: afterwards
    `add a` is accepted and the resulting version is consistent.

    The uniqueness hypothesis `honly` is necessary in general: if a second
    conflicting partner `k' ≠ k` remained in `v`, the retry would be
    rejected again. The equality oracle must decide true equality. -/
theorem repair_retry {C : α → α → Bool} {beq : α → α → Bool}
    (hbeq : ∀ x y, beq x y = true ↔ x = y)
    {v : List α} {a k : α}
    (hk : k ∈ v) (hck : C a k = true)
    (honly : ∀ x ∈ v, C a x = true → x = k)
    (hv : Consistent C v) :
    conflictsAny C a v = true ∧
      applyEdit2 C beq (removeAxiom beq k v) (.add a) =
        a :: removeAxiom beq k v ∧
      Consistent C (applyEdit2 C beq (removeAxiom beq k v) (.add a)) := by
  -- The original edit really was rejected: the witness makes it conflict.
  have hrej : conflictsAny C a v = true :=
    Sounio.OntologyAlignmentRepair.conflictsAny_true.mpr ⟨k, hk, hck⟩
  -- After removing k, no conflicting partner remains: any remaining
  -- conflict x would satisfy x = k (uniqueness), but k was removed.
  have hcf : conflictsAny C a (removeAxiom beq k v) = false := by
    rw [conflictsAny_false]
    intro x hx
    obtain ⟨hxv, hxk⟩ := List.mem_filter.mp hx
    cases hcx : C a x with
    | false => rfl
    | true =>
        have hxeq : x = k := honly x hxv hcx
        have hb : beq x k = true := (hbeq x k).mpr hxeq
        simp [hb] at hxk
  -- Hence the retry is accepted and extends the repaired version.
  have hacc : applyEdit2 C beq (removeAxiom beq k v) (.add a) =
      a :: removeAxiom beq k v := by
    simp [applyEdit2, applyEdit, hcf]
  refine ⟨hrej, hacc, ?_⟩
  rw [hacc]
  exact ⟨conflictsAny_false.mp hcf, consistent_removeAxiom hv⟩

-- ---------------------------------------------------------------------------
-- §5. Concrete instance: reject 4 against {3,2,1}, remove 2, re-add 4
-- ---------------------------------------------------------------------------

open Sounio.OntologyEvolution (exC)

/-- Boolean equality on `Fin 6` (the `BEq` instance). -/
def exBeq : Fin 6 → Fin 6 → Bool := fun a b => a == b

/-- `exBeq` decides true equality. -/
theorem exBeq_iff : ∀ x y : Fin 6, exBeq x y = true ↔ x = y :=
  fun _ _ => beq_iff_eq

/-- The edit `add 4` is rejected against the version {3,2,1} (axiom 4
    contradicts the established axiom 2). -/
theorem ex2_add4_rejected :
    applyEdit2 exC exBeq [3, 2, 1] (.add 4) = [3, 2, 1] := by
  native_decide

/-- Surgical removal: axiom 2 (the conflicting partner) is excised; all
    occurrences go. -/
theorem ex2_remove2 : removeAxiom exBeq 2 [3, 2, 1] = [3, 1] := by
  native_decide

/-- After the repair, re-adding axiom 4 is accepted. -/
theorem ex2_readd4_accepted :
    applyEdit2 exC exBeq (removeAxiom exBeq 2 [3, 2, 1]) (.add 4) = [4, 3, 1] := by
  native_decide

/-- Mixed edit script: add 1, 2, 3; add 4 (rejected); remove 2;
    add 4 (accepted). -/
def exEdits2 : List (Edit (Fin 6)) :=
  [.add 1, .add 2, .add 3, .add 4, .remove 2, .add 4]

/-- The chain ends in exactly {4,3,1}: axiom 2 was surgically removed to
    admit axiom 4. -/
theorem ex2_evolve : evolve2 exC exBeq [] exEdits2 = [4, 3, 1] := by
  native_decide

/-- The version {3,2,1} reached after adding 1, 2, 3 is consistent (via
    the addition-only invariant of `OntologyEvolution`). -/
theorem ex2_v321_consistent : Consistent exC [3, 2, 1] :=
  Sounio.OntologyEvolution.mem_versions_consistent (v := []) (edits := [1, 2, 3])
    (by trivial) (by native_decide)

/-- The repaired final version {4,3,1} is consistent (via the generalised
    invariant applied to the mixed script, not by enumeration). -/
theorem ex2_final_consistent : Consistent exC [4, 3, 1] := by
  rw [← ex2_evolve]
  exact consistent_evolve2 (by trivial) exEdits2

/-- The retry on the instance, via the general theorem: 2 is the unique
    conflicting partner of 4 in {3,2,1}. -/
theorem ex2_retry :
    conflictsAny exC 4 [3, 2, 1] = true ∧
      applyEdit2 exC exBeq (removeAxiom exBeq 2 [3, 2, 1]) (.add 4) =
        4 :: removeAxiom exBeq 2 [3, 2, 1] ∧
      Consistent exC (applyEdit2 exC exBeq (removeAxiom exBeq 2 [3, 2, 1]) (.add 4)) :=
  repair_retry exBeq_iff (k := 2)
    (by decide) (by native_decide) (by native_decide) ex2_v321_consistent

/-- Every recorded version of the mixed chain is consistent (via the
    general invariant, not by enumeration). -/
theorem ex2_versions_consistent :
    ∀ w ∈ versions2 exC exBeq [] exEdits2, Consistent exC w := by
  intro w hw
  exact mem_versions2_consistent (by trivial) hw

end Sounio.OntologyEvolutionRepair

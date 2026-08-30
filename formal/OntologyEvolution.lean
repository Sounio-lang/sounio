import OntologyAlignmentRepair

/-!
# Sounio.OntologyEvolution — a priori consistent ontology version chains

Formal companion to
`artifacts/ontology-frontiers/consistent-ontology-evolution/version_chain.sio`
(frontier: `consistent-ontology-evolution`; see the `FRONTIER.md` there for
the literature evidence).

## Setting

Biomedical ontologies evolve continuously (SNOMED CT releases, UMLS source
integration). The standard practice detects inconsistency *a posteriori* —
after a version is published (Bayoudhi, Sassi, Jaziri 2018,
doi:10.1111/exsy.12355). Here the transition is **guarded**: an edit adding
an axiom is applied only when the resulting version stays consistent, and
the invariant is a theorem, not a convention.

## What is proved here

1. `consistent_applyEdit` — the guarded transition preserves consistency.
2. `applyEdit_reject` / `applyEdit_accept` — rejection preserves the
   previous version exactly; acceptance extends it.
3. `mem_versions_consistent` — **a priori invariant**: every version
   recorded in the chain (not just the last) is consistent, provided the
   initial version is.
4. `consistent_evolve` — the final version is consistent.

A concrete `Fin 6` instance reproduces the `.sio` prototype (edit script
`[1,2,3,4,5]`, conflict pair `{2,4}`, axiom 4 rejected) and is checked by
`native_decide`.

Depends on `OntologyAlignmentRepair.lean` for the conflict oracle
(`conflictsAny` and its characterisation lemmas). No Mathlib. Zero sorry.
No new axioms.
-/

namespace Sounio.OntologyEvolution

open Sounio.OntologyAlignmentRepair (conflictsAny conflictsAny_false)

variable {α : Type}

-- ---------------------------------------------------------------------------
-- §1. Versions and consistency
-- ---------------------------------------------------------------------------

/-- A version is a list of active axioms; it is consistent when no element
    conflicts with any element coming after it (each pair checked once). -/
def Consistent (C : α → α → Bool) : List α → Prop
  | [] => True
  | a :: as => (∀ x ∈ as, C a x = false) ∧ Consistent C as

/-- Guarded edit: append axiom `a` to version `v` only when the result
    stays consistent; otherwise reject the edit. -/
def applyEdit (C : α → α → Bool) (v : List α) (a : α) : List α :=
  if conflictsAny C a v then v else a :: v

/-- The guarded transition preserves consistency. -/
theorem consistent_applyEdit {C : α → α → Bool} {v : List α} {a : α}
    (h : Consistent C v) : Consistent C (applyEdit C v a) := by
  cases hc : conflictsAny C a v with
  | true =>
      have hstep : applyEdit C v a = v := by simp [applyEdit, hc]
      rw [hstep]
      exact h
  | false =>
      have hstep : applyEdit C v a = a :: v := by simp [applyEdit, hc]
      rw [hstep]
      exact ⟨conflictsAny_false.mp hc, h⟩

/-- Rejecting an incoherent edit preserves the previous version exactly. -/
theorem applyEdit_reject {C : α → α → Bool} {v : List α} {a : α}
    (hc : conflictsAny C a v = true) : applyEdit C v a = v := by
  simp [applyEdit, hc]

/-- Accepting a coherent edit extends the version. -/
theorem applyEdit_accept {C : α → α → Bool} {v : List α} {a : α}
    (hc : conflictsAny C a v = false) : applyEdit C v a = a :: v := by
  simp [applyEdit, hc]

-- ---------------------------------------------------------------------------
-- §2. Version chains
-- ---------------------------------------------------------------------------

/-- The version chain: every intermediate version is recorded. -/
def versions (C : α → α → Bool) (v : List α) : List α → List (List α)
  | [] => [v]
  | a :: edits => v :: versions C (applyEdit C v a) edits

/-- The final version after all edits. -/
def evolve (C : α → α → Bool) (v : List α) : List α → List α
  | [] => v
  | a :: edits => evolve C (applyEdit C v a) edits

/-- **A priori invariant**: every version in the chain is consistent,
    provided the initial version is. -/
theorem mem_versions_consistent {C : α → α → Bool} {v : List α}
    (h : Consistent C v) {edits : List α} {w : List α}
    (hw : w ∈ versions C v edits) : Consistent C w := by
  induction edits generalizing v with
  | nil =>
      have hw' : w ∈ [v] := hw
      rw [List.mem_singleton] at hw'
      rw [hw']
      exact h
  | cons a edits ih =>
      have hw' : w ∈ v :: versions C (applyEdit C v a) edits := hw
      rw [List.mem_cons] at hw'
      cases hw' with
      | inl hwv =>
          rw [hwv]
          exact h
      | inr hmem =>
          exact ih (consistent_applyEdit h) hmem

/-- The final version is consistent. -/
theorem consistent_evolve {C : α → α → Bool} {v : List α}
    (h : Consistent C v) (edits : List α) : Consistent C (evolve C v edits) := by
  induction edits generalizing v with
  | nil => exact h
  | cons a edits ih => exact ih (consistent_applyEdit h)

-- ---------------------------------------------------------------------------
-- §3. Concrete instance: the `.sio` prototype's scenario
-- ---------------------------------------------------------------------------

/-- Axioms over `Fin 6`; the incoherent pair is {2, 4} (new axiom 4
    contradicts the established axiom 2). -/
def exC : Fin 6 → Fin 6 → Bool := fun a b =>
  (a.val == 2 && b.val == 4) || (a.val == 4 && b.val == 2)

/-- Edit script: add axioms 1, 2, 3, 4, 5 in order. -/
def exEdits : List (Fin 6) := [1, 2, 3, 4, 5]

/-- The incoherent edit (axiom 4) is rejected against the version {3,2,1}. -/
theorem ex_edit4_rejected : applyEdit exC [3, 2, 1] 4 = [3, 2, 1] := by
  native_decide

/-- The subsequent edit (axiom 5) is accepted. -/
theorem ex_edit5_accepted : applyEdit exC [3, 2, 1] 5 = [5, 3, 2, 1] := by
  native_decide

/-- The final version contains exactly {1, 2, 3, 5} — axiom 4 was rejected
    along the way. -/
theorem ex_evolve : evolve exC [] exEdits = [5, 3, 2, 1] := by
  native_decide

/-- Every recorded version of the example chain is consistent (via the
    general invariant, not by enumeration). -/
theorem ex_versions_consistent :
    ∀ w ∈ versions exC [] exEdits, Consistent exC w := by
  intro w hw
  exact mem_versions_consistent (by trivial) hw

end Sounio.OntologyEvolution

/-!
# Sounio.OntologyAlignmentRepair — formal model of epistemic alignment repair

Formal companion to
`artifacts/ontology-frontiers/epistemic-alignment-repair/alignment_repair.sio`
(frontier: `epistemic-alignment-repair`; see the `FRONTIER.md` there for the
literature evidence).

## Setting

Biomedical ontology alignments (UMLS, OAEI tracks) ship candidate mappings
with heuristic matcher confidences. When the integrated alignment is
logically incoherent, repair removes mappings — classically, the
low-confidence side of each conflict (Jiménez-Ruiz, Cuenca Grau, Horrocks
2011, doi:10.1186/2041-1480-2-s1-s2; Solimando, Jiménez-Ruiz, Guerrini
2016, doi:10.1007/s10115-016-0983-3).

## What is proved here

Candidates are processed in **decreasing confidence order** (the list order
*is* the priority). The greedy repair keeps a candidate exactly when it does
not conflict with any already-kept mapping. For this repair operator we prove:

1. `mem_repair_nil` — **soundness**: every retained mapping was a candidate.
2. `conflictFree_repair_nil` / `pairwise_repair_nil` — **correctness**: the
   retained set is conflict-free (for a symmetric conflict oracle).
3. `repair_witness_nil` — **maximality witness**: every dropped candidate
   conflicts with some retained mapping (necessarily of higher or equal
   priority, since it was processed earlier).

A concrete `Fin 5` instance reproduces the miniature UMLS-style scenario of
the `.sio` prototype and is checked by `native_decide`/`decide`.

## Remark (documented gap, not a theorem)

The `.sio` prototype implements the *pairwise drop-weaker* greedy: scan all
conflicting pairs, drop the lower-confidence endpoint. The equivalence with
the fold formalised here is **mechanised** in
`OntologyRepairEquivalence.lean` (`repair_iff_greedy`): for distinct
confidences it holds when the conflict relation is a disjoint union of
cliques (cluster graph — which covers the frontier instance of disjoint
edges). Without that hypothesis the equivalence is FALSE: on the 3-vertex
conflict path 0—1—2 with confidences 0<1<2 scanned as `[(0,1),(1,2)]`, the
greedy keeps `{2}` but the fold keeps `{0,2}` (mechanised counterexample
`cx_equivalence_fails` in that file).

Self-contained. No Mathlib. Zero sorry. No new axioms.
-/

namespace Sounio.OntologyAlignmentRepair

variable {α : Type}

-- ---------------------------------------------------------------------------
-- §1. Conflict oracle over a kept set
-- ---------------------------------------------------------------------------

/-- Does candidate mapping `m` logically conflict with any kept mapping? -/
def conflictsAny (C : α → α → Bool) (m : α) : List α → Bool
  | [] => false
  | k :: ks => (C m k || conflictsAny C m ks)

theorem conflictsAny_true {C : α → α → Bool} {m : α} {ks : List α} :
    conflictsAny C m ks = true ↔ ∃ k, k ∈ ks ∧ C m k = true := by
  induction ks with
  | nil =>
      constructor
      · intro h
        simp [conflictsAny] at h
      · rintro ⟨k, hmem, -⟩
        simp at hmem
  | cons k ks ih =>
      constructor
      · intro h
        have h' : (C m k || conflictsAny C m ks) = true := h
        cases hk : C m k with
        | true =>
            exact ⟨k, List.mem_cons.mpr (Or.inl rfl), hk⟩
        | false =>
            rw [hk, Bool.false_or] at h'
            obtain ⟨k', hmem, hc⟩ := ih.mp h'
            exact ⟨k', List.mem_cons.mpr (Or.inr hmem), hc⟩
      · rintro ⟨k', hmem, hc⟩
        rw [List.mem_cons] at hmem
        show (C m k || conflictsAny C m ks) = true
        cases hmem with
        | inl heq =>
            rw [heq] at hc
            rw [hc]
            exact Bool.true_or _
        | inr hmem' =>
            have h2 := ih.mpr ⟨k', hmem', hc⟩
            cases hk : C m k with
            | true => exact Bool.true_or _
            | false => exact h2

theorem conflictsAny_false {C : α → α → Bool} {m : α} {ks : List α} :
    conflictsAny C m ks = false ↔ ∀ k, k ∈ ks → C m k = false := by
  constructor
  · intro h k hk
    cases hc : C m k with
    | false => rfl
    | true =>
        have h1 := conflictsAny_true.mpr ⟨k, hk, hc⟩
        rw [h] at h1
        exact Bool.noConfusion h1
  · intro h
    cases hc : conflictsAny C m ks with
    | false => rfl
    | true =>
        obtain ⟨k, hk, hck⟩ := conflictsAny_true.mp hc
        rw [h k hk] at hck
        exact Bool.noConfusion hck

-- ---------------------------------------------------------------------------
-- §2. The greedy repair operator
-- ---------------------------------------------------------------------------

/-- One greedy step: keep candidate `m` exactly when it does not conflict
    with any mapping already kept. -/
def repairStep (C : α → α → Bool) (kept : List α) (m : α) : List α :=
  if conflictsAny C m kept then kept else m :: kept

/-- Priority repair: fold the candidate list (given in decreasing confidence
    order) through `repairStep`. -/
def repair (C : α → α → Bool) (kept : List α) : List α → List α
  | [] => kept
  | m :: ms => repair C (repairStep C kept m) ms

-- ---------------------------------------------------------------------------
-- §3. Membership lemmas (soundness direction)
-- ---------------------------------------------------------------------------

theorem mem_repairStep {C : α → α → Bool} {kept : List α} {m x : α}
    (h : x ∈ repairStep C kept m) : x = m ∨ x ∈ kept := by
  unfold repairStep at h
  split at h
  · exact Or.inr h
  · exact List.mem_cons.mp h

theorem subset_repairStep {C : α → α → Bool} {kept : List α} {m x : α}
    (h : x ∈ kept) : x ∈ repairStep C kept m := by
  unfold repairStep
  split
  · exact h
  · exact List.mem_cons.mpr (Or.inr h)

theorem mem_repair {C : α → α → Bool} {kept : List α} {ms : List α} {x : α}
    (h : x ∈ repair C kept ms) : x ∈ kept ∨ x ∈ ms := by
  induction ms generalizing kept with
  | nil =>
      have h2 : x ∈ kept := h
      exact Or.inl h2
  | cons m ms ih =>
      have h2 : x ∈ repair C (repairStep C kept m) ms := h
      have h' := ih h2
      cases h' with
      | inl hk =>
          obtain rfl | hk2 := mem_repairStep hk
          · exact Or.inr (List.mem_cons.mpr (Or.inl rfl))
          · exact Or.inl hk2
      | inr hm => exact Or.inr (List.mem_cons.mpr (Or.inr hm))

theorem subset_repair {C : α → α → Bool} {kept ms : List α} {x : α}
    (h : x ∈ kept) : x ∈ repair C kept ms := by
  induction ms generalizing kept with
  | nil => exact h
  | cons m ms ih =>
      have h2 : x ∈ repairStep C kept m := subset_repairStep h
      exact ih h2

/-- **Soundness**: every retained mapping was a candidate. -/
theorem mem_repair_nil {C : α → α → Bool} {ms : List α} {x : α}
    (h : x ∈ repair C [] ms) : x ∈ ms := by
  have h' := mem_repair h
  cases h' with
  | inl hn => cases hn
  | inr hm => exact hm

-- ---------------------------------------------------------------------------
-- §4. Conflict-freeness (correctness direction)
-- ---------------------------------------------------------------------------

/-- A kept set is conflict-free when no element conflicts with any element
    coming after it. -/
def ConflictFree (C : α → α → Bool) : List α → Prop
  | [] => True
  | k :: ks => (∀ x, x ∈ ks → C k x = false) ∧ ConflictFree C ks

theorem conflictFree_repairStep {C : α → α → Bool} {kept : List α} {m : α}
    (h : ConflictFree C kept) : ConflictFree C (repairStep C kept m) := by
  cases hc : conflictsAny C m kept with
  | true =>
      have hstep : repairStep C kept m = kept := by simp [repairStep, hc]
      rw [hstep]
      exact h
  | false =>
      have hstep : repairStep C kept m = m :: kept := by simp [repairStep, hc]
      rw [hstep]
      have hf := conflictsAny_false.mp hc
      exact ⟨hf, h⟩

theorem conflictFree_repair {C : α → α → Bool} {kept ms : List α}
    (h : ConflictFree C kept) : ConflictFree C (repair C kept ms) := by
  induction ms generalizing kept with
  | nil => exact h
  | cons m ms ih =>
      have h2 : ConflictFree C (repairStep C kept m) := conflictFree_repairStep h
      exact ih h2

/-- **Correctness**: the retained set is conflict-free. -/
theorem conflictFree_repair_nil (C : α → α → Bool) (ms : List α) :
    ConflictFree C (repair C [] ms) := by
  apply conflictFree_repair
  exact trivial

/-- Pairwise reading of conflict-freeness for a symmetric oracle. -/
theorem conflictFree_pairwise {C : α → α → Bool} (hsym : ∀ a b, C a b = C b a)
    {r : List α} (h : ConflictFree C r) :
    ∀ a b, a ∈ r → b ∈ r → a ≠ b → C a b = false := by
  induction r with
  | nil =>
      intro a b ha _ _
      cases ha
  | cons k ks ih =>
      intro a b ha hb hne
      rw [List.mem_cons] at ha hb
      have h' : (∀ x, x ∈ ks → C k x = false) ∧ ConflictFree C ks := h
      obtain ⟨hk1, hcf⟩ := h'
      cases ha with
      | inl ha_eq =>
          cases hb with
          | inl hb_eq => exact absurd (ha_eq.trans hb_eq.symm) hne
          | inr hb_mem =>
              have h2 := hk1 b hb_mem
              rw [← ha_eq] at h2
              exact h2
      | inr ha_mem =>
          cases hb with
          | inl hb_eq =>
              have h2 := hk1 a ha_mem
              rw [← hb_eq] at h2
              rw [hsym a b]
              exact h2
          | inr hb_mem => exact ih hcf a b ha_mem hb_mem hne

/-- **Correctness, pairwise form**: no two distinct retained mappings
    conflict. -/
theorem pairwise_repair_nil {C : α → α → Bool} (hsym : ∀ a b, C a b = C b a)
    (ms : List α) {a b : α} (ha : a ∈ repair C [] ms) (hb : b ∈ repair C [] ms)
    (hne : a ≠ b) : C a b = false :=
  conflictFree_pairwise hsym (conflictFree_repair_nil C ms) a b ha hb hne

-- ---------------------------------------------------------------------------
-- §5. Maximality witness
-- ---------------------------------------------------------------------------

/-- **Maximality witness**: every dropped candidate conflicts with some
    retained mapping (one processed earlier, hence of higher or equal
    priority). -/
theorem repair_witness {C : α → α → Bool} {kept ms : List α} {m : α}
    (hm : m ∈ ms) (hnot : m ∉ repair C kept ms) :
    ∃ k, k ∈ repair C kept ms ∧ C m k = true := by
  induction ms generalizing kept with
  | nil => cases hm
  | cons x xs ih =>
      rw [List.mem_cons] at hm
      have hunfold : repair C kept (x :: xs) = repair C (repairStep C kept x) xs := rfl
      cases hm with
      | inl hmx =>
          subst hmx
          cases hc : conflictsAny C m kept with
          | true =>
              have hstep : repairStep C kept m = kept := by simp [repairStep, hc]
              obtain ⟨k, hk, hck⟩ := conflictsAny_true.mp hc
              rw [hunfold, hstep] at hnot ⊢
              exact ⟨k, subset_repair hk, hck⟩
          | false =>
              have hstep : repairStep C kept m = m :: kept := by
                simp [repairStep, hc]
              have hmem : m ∈ repair C (m :: kept) xs :=
                subset_repair (List.mem_cons.mpr (Or.inl rfl))
              rw [hunfold, hstep] at hnot
              exact absurd hmem hnot
      | inr hmem =>
          rw [hunfold] at hnot
          exact ih hmem hnot

/-- **Maximality witness, from empty start.** -/
theorem repair_witness_nil {C : α → α → Bool} {ms : List α} {m : α}
    (hm : m ∈ ms) (hnot : m ∉ repair C [] ms) :
    ∃ k, k ∈ repair C [] ms ∧ C m k = true :=
  repair_witness hm hnot

-- ---------------------------------------------------------------------------
-- §6. Concrete instance: the miniature UMLS-style scenario
-- ---------------------------------------------------------------------------

/-- Conflict oracle for the prototype scenario over mapping ids `Fin 5`:
    C1 = {0,1} (lymphokine is-a / not-is-a protein),
    C2 = {2,3} (heart is-a organ / muscle-only). -/
def exConflicts : Fin 5 → Fin 5 → Bool := fun a b =>
  (a.val == 0 && b.val == 1) || (a.val == 1 && b.val == 0) ||
  (a.val == 2 && b.val == 3) || (a.val == 3 && b.val == 2)

/-- Candidates in decreasing matcher-confidence order:
    2 (0.95), 4 (0.80), 3 (0.40), 0 (0.30), 1 (0.06). -/
def exCandidates : List (Fin 5) := [2, 4, 3, 0, 1]

/-- The formal repair computes exactly the survivors of the `.sio`
    prototype: m2 (id 2), m4 (id 4), m0 (id 0). -/
theorem ex_repair :
    repair exConflicts [] exCandidates = [0, 4, 2] := by
  native_decide

/-- The conflict oracle is symmetric. -/
theorem exConflicts_sym : ∀ a b : Fin 5, exConflicts a b = exConflicts b a := by
  decide

/-- Instance of soundness: retained ⊆ candidates. -/
theorem ex_sound : ∀ x : Fin 5, x ∈ repair exConflicts [] exCandidates →
    x ∈ exCandidates :=
  fun _ h => mem_repair_nil h

/-- Instance of correctness: the retained set {[0,4,2]} is pairwise
    conflict-free. -/
theorem ex_pairwise :
    ∀ a b : Fin 5, a ∈ repair exConflicts [] exCandidates →
      b ∈ repair exConflicts [] exCandidates → a ≠ b →
      exConflicts a b = false := by
  rw [ex_repair]
  decide

/-- Instance of the witness theorem: the dropped mapping 1 conflicts with
    the retained mapping 0. -/
theorem ex_witness_1 :
    ∃ k : Fin 5, k ∈ repair exConflicts [] exCandidates ∧
      exConflicts 1 k = true := by
  rw [ex_repair]
  exact ⟨0, by decide, by native_decide⟩

/-- Instance of the witness theorem: the dropped mapping 3 conflicts with
    the retained mapping 2. -/
theorem ex_witness_3 :
    ∃ k : Fin 5, k ∈ repair exConflicts [] exCandidates ∧
      exConflicts 3 k = true := by
  rw [ex_repair]
  exact ⟨2, by decide, by native_decide⟩

end Sounio.OntologyAlignmentRepair

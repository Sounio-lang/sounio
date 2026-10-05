/-!
# Sounio.OntologyELReasoner — a mini EL/disjointness reasoner grounding the conflict oracle

Formal companion to the `epistemic-alignment-repair` and
`consistent-ontology-evolution` frontiers
(`artifacts/ontology-frontiers/epistemic-alignment-repair/`,
`artifacts/ontology-frontiers/consistent-ontology-evolution/`), and to the
repair formalisations `OntologyAlignmentRepair.lean`,
`OntologyEvolution.lean`, `OntologyEvolutionRepair.lean`,
`OntologyMinimalRepair.lean`.

## Why this file exists

Those repair proofs are *syntactic*: they take a conflict oracle `C` (the
hardcoded `conflicts` function over mapping ids in the `.sio` prototype;
`exConflicts` in `OntologyAlignmentRepair.lean`) and prove soundness,
conflict-freeness, and maximality **relative to that oracle**. This file
supplies the missing *semantic* ground: we model a mini description logic in
the EL/disjointness style (subsumption axioms `c ⊑ d` and disjointness axioms
`c ⊥ d`), give it Tarski semantics over arbitrary domains, define the
reflexive-transitive subsumption closure as an inductive derivation system
`SubDer`, and prove:

1. `subDer_sound` — every derivable subsumption holds in every model.
2. `incoherent_empty` — an incoherent class (one subsuming both sides of a
   disjointness) is empty in every model.
3. `oracle_sound` — **the oracle connection**: if the closure-derived oracle
   flags a conflict between asserting class `c` and asserting class `c'` over
   the same entity, then *no* model of the TBox can satisfy both assertions.

Point 3 is exactly the logical justification for the repair oracles: when the
hardcoded `exConflicts` relation coincides with `DerivedConflict t` for the
integrated TBox `t`, dropping one side of each flagged pair removes a genuine
unsatisfiability, not a heuristic guess. Conversely, `DerivedConflict` is
symmetric (`derivedConflict_symm`), matching the symmetry hypothesis
`exConflicts_sym` required by `conflictFree_pairwise`.

A concrete `Fin 8` biomedical-flavoured instance (lymphokine ⊑ protein ⊑
molecule, protein ⊥ drug, heart ⊑ organ, organ ⊥ muscle-only, and an
incoherent therapeutic fusion protein that is both protein and drug) provides
constructor-built `SubDer` derivations, one `Incoherent` witness, and
`native_decide`-checked boolean closure computations.

Self-contained. No Mathlib. Zero sorry. No new axioms.
-/

namespace Sounio.OntologyELReasoner

variable {α : Type} {ι : Type}

-- ---------------------------------------------------------------------------
-- §1. Syntax: a mini EL/disjointness TBox
-- ---------------------------------------------------------------------------

/-- TBox axioms: concept inclusion `sub c d` (read `c ⊑ d`) and disjointness
    `disj c d` (read `c ⊥ d`). -/
inductive Axiom (α : Type) : Type
  | sub (c d : α)
  | disj (c d : α)
  deriving DecidableEq, Repr

-- ---------------------------------------------------------------------------
-- §2. Tarski semantics
-- ---------------------------------------------------------------------------

/-- An interpretation of class names over a domain `ι`. -/
abbrev Interp (α ι : Type) := α → ι → Prop

/-- Satisfaction of a single axiom. -/
def SatisfiesAxiom (I : Interp α ι) : Axiom α → Prop
  | .sub c d => ∀ x, I c x → I d x
  | .disj c d => ∀ x, ¬ (I c x ∧ I d x)

/-- Satisfaction of a whole TBox: every axiom holds. -/
def Satisfies (I : Interp α ι) (t : List (Axiom α)) : Prop :=
  ∀ a, a ∈ t → SatisfiesAxiom I a

-- ---------------------------------------------------------------------------
-- §3. Subsumption closure as a derivation system
-- ---------------------------------------------------------------------------

/-- Reflexive-transitive closure of subsumption over the TBox:
    `SubDer t c d` reads "`t` derives `c ⊑* d`". -/
inductive SubDer (t : List (Axiom α)) : α → α → Prop
  | ofAxiom {c d : α} : Axiom.sub c d ∈ t → SubDer t c d
  | refl {c : α} : SubDer t c c
  | trans {c d e : α} : SubDer t c d → SubDer t d e → SubDer t c e

/-- **Semantic soundness of the closure**: every derivable subsumption holds
    in every model of the TBox. -/
theorem subDer_sound {I : Interp α ι} {t : List (Axiom α)} {c d : α}
    (h : SubDer t c d) (hI : Satisfies I t) : ∀ x, I c x → I d x := by
  induction h with
  | ofAxiom hmem =>
      intro x hx
      exact hI _ hmem x hx
  | refl =>
      intro x hx
      exact hx
  | trans _ _ ih1 ih2 =>
      intro x hx
      exact ih2 x (ih1 x hx)

-- ---------------------------------------------------------------------------
-- §4. Incoherence
-- ---------------------------------------------------------------------------

/-- A class `c` is incoherent in `t` when the closure derives that `c` is
    subsumed by both sides of a disjointness axiom. -/
def Incoherent (t : List (Axiom α)) (c : α) : Prop :=
  ∃ d₁ d₂, SubDer t c d₁ ∧ SubDer t c d₂ ∧
    (Axiom.disj d₁ d₂ ∈ t ∨ Axiom.disj d₂ d₁ ∈ t)

/-- **Central soundness theorem**: an incoherent class is empty in every
    model of the TBox. -/
theorem incoherent_empty {I : Interp α ι} {t : List (Axiom α)} {c : α}
    (h : Incoherent t c) (hI : Satisfies I t) : ∀ x, ¬ I c x := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  have s1 := subDer_sound h1 hI
  have s2 := subDer_sound h2 hI
  cases hd with
  | inl hmem =>
      intro x hc
      exact hI _ hmem x ⟨s1 x hc, s2 x hc⟩
  | inr hmem =>
      intro x hc
      exact hI _ hmem x ⟨s2 x hc, s1 x hc⟩

-- ---------------------------------------------------------------------------
-- §5. The closure-derived conflict oracle
-- ---------------------------------------------------------------------------

/-- The derived conflict relation between two class assertions: asserting
    `c` and `c'` over the same entity conflicts when `t` derives that `c` and
    `c'` are subsumed by a disjoint pair. This is the logical ground of the
    hardcoded conflict oracles of `OntologyAlignmentRepair.lean`
    (`exConflicts`) and of the `.sio` prototype (`conflicts`). -/
def DerivedConflict (t : List (Axiom α)) (c c' : α) : Prop :=
  ∃ d₁ d₂, SubDer t c d₁ ∧ SubDer t c' d₂ ∧
    (Axiom.disj d₁ d₂ ∈ t ∨ Axiom.disj d₂ d₁ ∈ t)

/-- Incoherence of `c` is exactly a self-conflict of `c`. -/
theorem incoherent_iff_self_conflict {t : List (Axiom α)} {c : α} :
    Incoherent t c ↔ DerivedConflict t c c := Iff.rfl

/-- The derived conflict relation is symmetric — the property
    (`exConflicts_sym`) that the pairwise reading of conflict-freeness in
    `OntologyAlignmentRepair.lean` requires of its oracle. -/
theorem derivedConflict_symm {t : List (Axiom α)} {c c' : α}
    (h : DerivedConflict t c c') : DerivedConflict t c' c := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  exact ⟨d₂, d₁, h2, h1, hd.symm⟩

/-- **Oracle soundness**: if the closure-derived oracle flags a conflict
    between asserting `c` and asserting `c'` over the same entity, then no
    model of the TBox can satisfy both assertions. Repair operators may
    therefore safely drop either side of a flagged pair. -/
theorem oracle_sound {I : Interp α ι} {t : List (Axiom α)} {c c' : α}
    (h : DerivedConflict t c c') (hI : Satisfies I t) :
    ∀ x, ¬ (I c x ∧ I c' x) := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  have s1 := subDer_sound h1 hI
  have s2 := subDer_sound h2 hI
  cases hd with
  | inl hmem =>
      intro x ⟨hc, hc'⟩
      exact hI _ hmem x ⟨s1 x hc, s2 x hc'⟩
  | inr hmem =>
      intro x ⟨hc, hc'⟩
      exact hI _ hmem x ⟨s2 x hc', s1 x hc⟩

-- ---------------------------------------------------------------------------
-- §6. A computable boolean closure (for concrete instances over `Fin n`)
-- ---------------------------------------------------------------------------

section BoolClosure

variable {n : Nat}

/-- Subsumption edges of the TBox. -/
def subPairs (t : List (Axiom (Fin n))) : List (Fin n × Fin n) :=
  t.filterMap fun a => match a with
    | .sub c d => some (c, d)
    | .disj _ _ => none

/-- Disjointness edges of the TBox, in both directions. -/
def disjPairs (t : List (Axiom (Fin n))) : List (Fin n × Fin n) :=
  t.foldr (fun a acc => match a with
    | .sub _ _ => acc
    | .disj c d => (c, d) :: (d, c) :: acc) []

/-- Deduplicate a list of pairs (keeping first occurrences). -/
def dedupPairs (l : List (Fin n × Fin n)) : List (Fin n × Fin n) :=
  l.foldr (fun x acc => if acc.elem x then acc else x :: acc) []

/-- One transitivity-propagation step over a relation `R`, deduplicated. -/
def transStep (R : List (Fin n × Fin n)) : List (Fin n × Fin n) :=
  dedupPairs (R ++ R.foldr (fun p acc =>
    R.foldr (fun q acc' => if p.2 == q.1 then (p.1, q.2) :: acc' else acc')
      acc) [])

/-- Iterate `transStep` `k` times. -/
def closeSubIter : Nat → List (Fin n × Fin n) → List (Fin n × Fin n)
  | 0, R => R
  | k + 1, R => closeSubIter k (transStep R)

/-- Boolean reflexive-transitive closure of the subsumption edges: seed with
    the TBox edges plus the diagonal over all `n` class ids, then propagate
    transitivity `n + 1` times (longer than any simple chain). -/
def closeSub (t : List (Axiom (Fin n))) : List (Fin n × Fin n) :=
  let R0 := subPairs t ++ (List.finRange n).map fun c => (c, c)
  closeSubIter (n + 1) R0

/-- Boolean subsumption check against the closure. -/
def subB (t : List (Axiom (Fin n))) (c d : Fin n) : Bool :=
  (closeSub t).elem (c, d)

/-- Boolean version of `DerivedConflict`: some closure pair above `c` and
    some closure pair above `c'` form a disjointness edge. -/
def conflictB (t : List (Axiom (Fin n))) (c c' : Fin n) : Bool :=
  let R := closeSub t
  let D := disjPairs t
  R.any fun p => R.any fun q =>
    (p.1 == c) && (q.1 == c') && D.elem (p.2, q.2)

end BoolClosure

-- ---------------------------------------------------------------------------
-- §7. Concrete instance: a small biomedical-flavoured TBox over `Fin 8`
-- ---------------------------------------------------------------------------

section BiomedInstance

/-!
Class ids (`Fin 8`):

- `0` Lymphokine
- `1` Protein
- `2` Molecule
- `3` Drug
- `4` Heart
- `5` Organ
- `6` SkeletalMuscleOnly
- `7` TherapeuticFusionProtein
-/

/-- The instance TBox: lymphokine ⊑ protein ⊑ molecule; protein ⊥ drug;
    heart ⊑ organ; organ ⊥ muscle-only; a therapeutic fusion protein that is
    asserted to be both a protein and a drug (and is therefore incoherent). -/
def biomedTBox : List (Axiom (Fin 8)) :=
  [ .sub 0 1   -- Lymphokine ⊑ Protein
  , .sub 1 2   -- Protein ⊑ Molecule
  , .disj 1 3  -- Protein ⊥ Drug
  , .sub 4 5   -- Heart ⊑ Organ
  , .disj 5 6  -- Organ ⊥ SkeletalMuscleOnly
  , .sub 7 1   -- TherapeuticFusionProtein ⊑ Protein
  , .sub 7 3   -- TherapeuticFusionProtein ⊑ Drug
  ]

-- §7.1 Constructor-built derivations

/-- Lymphokine ⊑* Molecule (via Protein). -/
theorem sub_lymphokine_molecule : SubDer biomedTBox 0 2 :=
  .trans (.ofAxiom (c := 0) (d := 1) (by decide))
         (.ofAxiom (c := 1) (d := 2) (by decide))

/-- TherapeuticFusionProtein ⊑* Molecule (via Protein). -/
theorem sub_fusion_molecule : SubDer biomedTBox 7 2 :=
  .trans (.ofAxiom (c := 7) (d := 1) (by decide))
         (.ofAxiom (c := 1) (d := 2) (by decide))

/-- Reflexivity: every class subsumes itself, e.g. Organ ⊑* Organ. -/
theorem sub_organ_refl : SubDer biomedTBox 5 5 := .refl

/-- The fusion protein is incoherent: it is subsumed by both Protein and
    Drug, which are disjoint. -/
theorem incoherent_fusion : Incoherent biomedTBox 7 :=
  ⟨1, 3, .ofAxiom (by decide), .ofAxiom (by decide), Or.inl (by decide)⟩

/-- The oracle flags the pair (TherapeuticFusionProtein, Lymphokine):
    the fusion protein is subsumed by Drug, the lymphokine by Protein, and
    Protein ⊥ Drug — even though neither class is incoherent on its own. -/
theorem conflict_fusion_lymphokine : DerivedConflict biomedTBox 7 0 :=
  ⟨3, 1, .ofAxiom (by decide), .ofAxiom (by decide), Or.inr (by decide)⟩

-- §7.2 Semantic consequences in arbitrary models

/-- No model of the TBox has any therapeutic fusion protein. -/
theorem no_fusion_in_models {I : Interp (Fin 8) ι}
    (hI : Satisfies I biomedTBox) : ∀ x, ¬ I 7 x :=
  incoherent_empty incoherent_fusion hI

/-- Oracle soundness, instantiated: no model of the TBox can assert of one
    and the same entity that it is a therapeutic fusion protein and that it
    is a lymphokine. This is the licence for a repair operator to drop one of
    the two corresponding mappings. -/
theorem oracle_sound_instance {I : Interp (Fin 8) ι}
    (hI : Satisfies I biomedTBox) :
    ∀ x, ¬ (I 7 x ∧ I 0 x) :=
  oracle_sound conflict_fusion_lymphokine hI

-- §7.3 Decidable checks (`decide` / `native_decide`)

theorem check_sub_mem :
    Axiom.sub (0 : Fin 8) 1 ∈ biomedTBox := by decide

theorem check_disj_mem :
    Axiom.disj (1 : Fin 8) 3 ∈ biomedTBox := by decide

/-- Boolean closure: Lymphokine ⊑* Molecule is computed. -/
theorem check_subB_0_2 : subB biomedTBox 0 2 = true := by native_decide

/-- Boolean closure: TherapeuticFusionProtein ⊑* Molecule (two hops). -/
theorem check_subB_7_2 : subB biomedTBox 7 2 = true := by native_decide

/-- Boolean closure: Heart ⊑* Organ. -/
theorem check_subB_4_5 : subB biomedTBox 4 5 = true := by native_decide

/-- Boolean closure: reflexive pair present. -/
theorem check_subB_5_5 : subB biomedTBox 5 5 = true := by native_decide

/-- Boolean closure: Organ is NOT subsumed by SkeletalMuscleOnly
    (disjointness is not subsumption). -/
theorem check_subB_5_6 : subB biomedTBox 5 6 = false := by native_decide

/-- Boolean oracle: the incoherent fusion protein self-conflicts. -/
theorem check_conflictB_7_7 : conflictB biomedTBox 7 7 = true := by native_decide

/-- Boolean oracle: fusion protein vs lymphokine conflict is computed. -/
theorem check_conflictB_7_0 : conflictB biomedTBox 7 0 = true := by native_decide

/-- Boolean oracle: symmetry spot-check. -/
theorem check_conflictB_0_7 : conflictB biomedTBox 0 7 = true := by native_decide

/-- Boolean oracle: Lymphokine and Heart do not conflict — both can
    consistently be asserted of the same entity (nothing forbids a molecule
    that is also an organ part in this toy TBox). -/
theorem check_conflictB_0_4 : conflictB biomedTBox 0 4 = false := by native_decide

end BiomedInstance

end Sounio.OntologyELReasoner

import OntologyELReasoner
import OntologyELClosureVerified

/-!
# Sounio.OntologyELPlus — the mini EL reasoner extended toward EL+ (the SNOMED CT fragment)

Round 7 ("strong and bold") of the *ontology frontiers* research line
(`artifacts/ontology-frontiers/README.md`). Rounds 3–4 built a mini EL
reasoner over *atomic* classes only (`OntologyELReasoner.lean`:
`Axiom = sub | disj` over class ids, Tarski semantics, `SubDer`,
`subDer_sound`, `incoherent_empty`, `oracle_sound`, plus the verified boolean
closure of `OntologyELClosureVerified.lean`). This file extends that reasoner
to the **EL⁺ description logic — the fragment that SNOMED CT is actually
written in**:

* concepts are built from class names, ⊤, conjunction `C ⊓ D`, and
  existential restrictions `∃r.C`;
* axioms are concept inclusions `C ⊑ D`, disjointness `C ⊥ D`, and
  **role inclusions** `r ⊑ s` (role hierarchy — the feature that separates
  EL⁺ from plain EL).

**Now in scope**: general role *composition* `r ∘ s ⊑ t` (right-hand role
chains), the feature that completes the EL⁺ profile of OWL 2 beyond the
SNOMED-fragment treated in round 7. The file now proves soundness for the
full EL⁺ constructor system including role-composition chaining.

What is proved here is complete for the fragment below:

1. `der_sound` — every subsumption derivable in the constructor system `Der`
   (axiom membership, reflexivity, transitivity, ⊓-intro, ⊓-elim-left/right,
   ∃-monotonicity, ∃-roleSub chaining, ∃-roleComp chaining, ⊤-rule) holds in
   every model of the TBox. The `exRoleSub`/`exComp` cases chain existential
   witnesses through role inclusion and role-composition satisfaction — the
   genuinely new semantic content versus round 3.
2. `IncoherentP` / `incoherentP_empty` — an incoherent concept (one derivably
   subsumed by both sides of a disjointness) is empty in every model.
3. `DerivedConflictP` / `oracle_sound_P` — the oracle bridge of round 4,
   lifted to full concepts: a closure-derived conflict between two concept
   assertions over the same entity is a genuine unsatisfiability, licensing
   repair operators to drop either side.
4. An **atom-only projection** (`projectAxiom` / `project`) that forgets
   complex concepts and role inclusions, reusing round 3/4's verified boolean
   closure (`closeSub`, `subB`, `conflictB`) unchanged; `subBP_sound` /
   `conflictBP_sound` lift the computed answers back into `Der` /
   `DerivedConflictP`, so the cheap boolean engine is proved sound for the
   atomic shadow of an EL⁺ TBox.
5. A concrete SNOMED-flavoured `Fin 8` × `Fin 3` instance: Pneumonia ⊑
   Inflammation ⊑ Disorder, Lung ⊑ Organ, Pneumonia ⊑ ∃DirectSite.(Lung ⊓
   Inflammation), DirectSite ⊑ RoleGroup, Disorder ⊥ Drug, and an incoherent
   DrugInducedDisorder — with constructor-built `Der` derivations (including
   the chain Pneumonia ⊑ ∃RoleGroup.Lung ⊑ ∃RoleGroup.Organ, which exercises
   `exRoleSub`, `conjElimLeft`, `exMono`, and `trans` together) and
   `native_decide` boolean checks.

Self-contained modulo the two ontology-frontiers imports below. No Mathlib.
Zero sorry. No new axioms. No rule was dropped: every rule listed above is
proved sound. Role composition is the new frontier closed in this round.
-/


namespace Sounio.OntologyELPlus

open Sounio.OntologyELReasoner

variable {α ρ ι : Type}

-- ---------------------------------------------------------------------------
-- §1. Syntax: EL⁺ concepts and TBox axioms
-- ---------------------------------------------------------------------------

/-- EL⁺ concepts over class ids `α` and role ids `ρ`:
    class names, ⊤, conjunction `C ⊓ D`, existential restriction `∃r.C`. -/
inductive Concept (α ρ : Type) : Type
  | atom (a : α)
  | top
  | conj (c d : Concept α ρ)
  | ex (r : ρ) (c : Concept α ρ)
  deriving DecidableEq, Repr

/-- EL⁺ TBox axioms: concept inclusion `sub c d` (`c ⊑ d`), disjointness
    `disj c d` (`c ⊥ d`), role inclusion `roleSub r s` (`r ⊑ s`), and role
    composition `roleComp r s t` (`r ∘ s ⊑ t`). -/
inductive AxiomP (α ρ : Type) : Type
  | sub (c d : Concept α ρ)
  | disj (c d : Concept α ρ)
  | roleSub (r s : ρ)
  | roleComp (r s t : ρ)
  deriving DecidableEq, Repr

-- ---------------------------------------------------------------------------
-- §2. Tarski semantics (class names + binary role relations)
-- ---------------------------------------------------------------------------

/-- The meaning of a concept under a class interpretation `I` and a role
    interpretation `IR`: names by `I`, ⊤ by the whole domain, ⊓ by
    intersection, `∃r.C` by an `r`-successor in `⟦C⟧`. -/
def meaning (I : α → ι → Prop) (IR : ρ → ι → ι → Prop) : Concept α ρ → ι → Prop
  | .atom a => I a
  | .top => fun _ => True
  | .conj c d => fun x => meaning I IR c x ∧ meaning I IR d x
  | .ex r c => fun x => ∃ y, IR r x y ∧ meaning I IR c y

/-- Satisfaction of a single axiom. A role inclusion `r ⊑ s` holds when the
    `r`-relation is contained in the `s`-relation; a role composition
    `r ∘ s ⊑ t` holds when the relational composition of `IR r` and `IR s` is
    contained in `IR t`. -/
def SatisfiesAxiomP (I : α → ι → Prop) (IR : ρ → ι → ι → Prop) : AxiomP α ρ → Prop
  | .sub c d => ∀ x, meaning I IR c x → meaning I IR d x
  | .disj c d => ∀ x, ¬ (meaning I IR c x ∧ meaning I IR d x)
  | .roleSub r s => ∀ x y, IR r x y → IR s x y
  | .roleComp r s t => ∀ x y z, IR r x y → IR s y z → IR t x z

/-- Satisfaction of a whole TBox: every axiom holds. -/
def SatisfiesP (I : α → ι → Prop) (IR : ρ → ι → ι → Prop) (t : List (AxiomP α ρ)) : Prop :=
  ∀ a, a ∈ t → SatisfiesAxiomP I IR a

-- ---------------------------------------------------------------------------
-- §3. The EL⁺ derivation system
-- ---------------------------------------------------------------------------

/-- The deductive system: `Der t c d` reads "`t` derives `c ⊑* d`".
    Beyond round 3's `ofAxiom`/`refl`/`trans`, this adds the EL⁺ constructors:
    conjunction introduction and elimination, existential monotonicity,
    existential role chaining through a role inclusion, existential
    role-composition chaining (exComp), and the ⊤-rule. -/
inductive Der (t : List (AxiomP α ρ)) : Concept α ρ → Concept α ρ → Prop
  | ofAxiom {c d : Concept α ρ} : AxiomP.sub c d ∈ t → Der t c d
  | refl {c : Concept α ρ} : Der t c c
  | trans {c d e : Concept α ρ} : Der t c d → Der t d e → Der t c e
  | conjIntro {c d₁ d₂ : Concept α ρ} : Der t c d₁ → Der t c d₂ →
      Der t c (.conj d₁ d₂)
  | conjElimLeft {c d₁ d₂ : Concept α ρ} : Der t c (.conj d₁ d₂) → Der t c d₁
  | conjElimRight {c d₁ d₂ : Concept α ρ} : Der t c (.conj d₁ d₂) → Der t c d₂
  | exMono {c d : Concept α ρ} {r : ρ} : Der t c d → Der t (.ex r c) (.ex r d)
  | exRoleSub {c d : Concept α ρ} {r s : ρ} : Der t c (.ex r d) →
      AxiomP.roleSub r s ∈ t → Der t c (.ex s d)
  | exComp {c d : Concept α ρ} {r s u : ρ} :
      Der t c (.ex r (.ex s d)) → AxiomP.roleComp r s u ∈ t →
      Der t c (.ex u d)
  | topRule {c : Concept α ρ} : Der t c .top

/-- **(1) Semantic soundness of the EL⁺ closure**: every derivable
    subsumption holds in every model of the TBox. The `exRoleSub` and `exComp`
    cases are the new semantic content: an `r`-witness yielded by the premise
    is promoted to an `s`-witness by role-inclusion satisfaction, or composed
    through a role-composition axiom into a `t`-witness. -/
theorem der_sound {I : α → ι → Prop} {IR : ρ → ι → ι → Prop}
    {t : List (AxiomP α ρ)} {c d : Concept α ρ}
    (h : Der t c d) (hI : SatisfiesP I IR t) :
    ∀ x, meaning I IR c x → meaning I IR d x := by
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
      exact ⟨y, hI _ hmem x y hr, hy⟩
  | exComp _ hmem ih =>
      intro x hx
      obtain ⟨y, hr, hy⟩ := ih x hx
      obtain ⟨z, hs, hz⟩ := hy
      exact ⟨z, hI _ hmem x y z hr hs, hz⟩
  | topRule =>
      intro x _
      exact True.intro

-- ---------------------------------------------------------------------------
-- §4. Incoherence over full concepts
-- ---------------------------------------------------------------------------

/-- A concept `c` is incoherent in `t` when the closure derives that `c` is
    subsumed by both sides of a disjointness axiom. Unlike round 3, `c` and
    the disjoint pair may now be arbitrary EL⁺ concepts. -/
def IncoherentP (t : List (AxiomP α ρ)) (c : Concept α ρ) : Prop :=
  ∃ d₁ d₂, Der t c d₁ ∧ Der t c d₂ ∧
    (AxiomP.disj d₁ d₂ ∈ t ∨ AxiomP.disj d₂ d₁ ∈ t)

/-- **(2) Incoherence extension**: an incoherent concept is empty in every
    model of the TBox. -/
theorem incoherentP_empty {I : α → ι → Prop} {IR : ρ → ι → ι → Prop}
    {t : List (AxiomP α ρ)} {c : Concept α ρ}
    (h : IncoherentP t c) (hI : SatisfiesP I IR t) : ∀ x, ¬ meaning I IR c x := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  have s1 := der_sound h1 hI
  have s2 := der_sound h2 hI
  cases hd with
  | inl hmem =>
      intro x hc
      exact hI _ hmem x ⟨s1 x hc, s2 x hc⟩
  | inr hmem =>
      intro x hc
      exact hI _ hmem x ⟨s2 x hc, s1 x hc⟩

-- ---------------------------------------------------------------------------
-- §5. The closure-derived conflict oracle, lifted to EL⁺
-- ---------------------------------------------------------------------------

/-- The derived conflict relation between two concept assertions: asserting
    `c` and `c'` over the same entity conflicts when `t` derives that they
    are subsumed by a disjoint pair. This is the EL⁺ lift of round 3's
    `DerivedConflict` — the logical ground of the repair oracles, now over
    the fragment SNOMED CT actually uses. -/
def DerivedConflictP (t : List (AxiomP α ρ)) (c c' : Concept α ρ) : Prop :=
  ∃ d₁ d₂, Der t c d₁ ∧ Der t c' d₂ ∧
    (AxiomP.disj d₁ d₂ ∈ t ∨ AxiomP.disj d₂ d₁ ∈ t)

/-- Incoherence of `c` is exactly a self-conflict of `c`. -/
theorem incoherentP_iff_self_conflict {t : List (AxiomP α ρ)} {c : Concept α ρ} :
    IncoherentP t c ↔ DerivedConflictP t c c := Iff.rfl

/-- The derived conflict relation is symmetric. -/
theorem derivedConflictP_symm {t : List (AxiomP α ρ)} {c c' : Concept α ρ}
    (h : DerivedConflictP t c c') : DerivedConflictP t c' c := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  exact ⟨d₂, d₁, h2, h1, hd.symm⟩

/-- **(3) Oracle soundness, EL⁺**: if the closure-derived oracle flags a
    conflict between asserting `c` and asserting `c'` over the same entity,
    then no model of the TBox can satisfy both assertions. Repair operators
    may therefore safely drop either side of a flagged pair — the round-4
    licence, now valid for SNOMED-fragment concept assertions. -/
theorem oracle_sound_P {I : α → ι → Prop} {IR : ρ → ι → ι → Prop}
    {t : List (AxiomP α ρ)} {c c' : Concept α ρ}
    (h : DerivedConflictP t c c') (hI : SatisfiesP I IR t) :
    ∀ x, ¬ (meaning I IR c x ∧ meaning I IR c' x) := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  have s1 := der_sound h1 hI
  have s2 := der_sound h2 hI
  cases hd with
  | inl hmem =>
      intro x ⟨hc, hc'⟩
      exact hI _ hmem x ⟨s1 x hc, s2 x hc'⟩
  | inr hmem =>
      intro x ⟨hc, hc'⟩
      exact hI _ hmem x ⟨s2 x hc', s1 x hc⟩

-- ---------------------------------------------------------------------------
-- §6. The atom-only projection and its verified boolean closure
-- ---------------------------------------------------------------------------

section AtomProjection

variable {n : Nat}

/-- Project an EL⁺ axiom onto round 3's atomic TBox language: atomic
    inclusions and disjointness survive; axioms mentioning ⊤, ⊓, ∃, or role
    inclusions are forgotten. -/
def projectAxiom : AxiomP (Fin n) ρ → Option (Axiom (Fin n))
  | .sub (.atom a) (.atom b) => some (.sub a b)
  | .sub _ _ => none
  | .disj (.atom a) (.atom b) => some (.disj a b)
  | .disj _ _ => none
  | .roleSub _ _ => none
  | .roleComp _ _ _ => none

/-- The atomic shadow of an EL⁺ TBox. -/
def project (t : List (AxiomP (Fin n) ρ)) : List (Axiom (Fin n)) :=
  t.filterMap projectAxiom

/-- **(4) Boolean closure over the atomic shadow**, reusing round 3's
    `closeSub` / `subB` unchanged. -/
def subBP (t : List (AxiomP (Fin n) ρ)) (c d : Fin n) : Bool :=
  subB (project t) c d

/-- Boolean conflict oracle over the atomic shadow, reusing round 3's
    `conflictB` unchanged. -/
def conflictBP (t : List (AxiomP (Fin n) ρ)) (c c' : Fin n) : Bool :=
  conflictB (project t) c c'

-- §6.1 Projected axioms lift back to the original TBox

/-- If the projection yields an atomic `sub` axiom, the preimage is exactly
    that axiom over `atom`s. -/
theorem projectAxiom_some_sub {a : AxiomP (Fin n) ρ} {c d : Fin n}
    (h : projectAxiom a = some (Axiom.sub c d)) :
    a = AxiomP.sub (.atom c) (.atom d) := by
  cases a with
  | sub c' d' =>
      cases c' with
      | atom a' =>
          cases d' with
          | atom b' =>
              injection h with hh
              injection hh with h1 h2
              subst h1; subst h2
              rfl
          | top => simp [projectAxiom] at h
          | conj c₁ c₂ => simp [projectAxiom] at h
          | ex r c₁ => simp [projectAxiom] at h
      | top => simp [projectAxiom] at h
      | conj c₁ c₂ => simp [projectAxiom] at h
      | ex r c₁ => simp [projectAxiom] at h
  | disj c' d' =>
      cases c' with
      | atom a' =>
          cases d' with
          | atom b' => simp [projectAxiom] at h
          | top => simp [projectAxiom] at h
          | conj c₁ c₂ => simp [projectAxiom] at h
          | ex r c₁ => simp [projectAxiom] at h
      | top => simp [projectAxiom] at h
      | conj c₁ c₂ => simp [projectAxiom] at h
      | ex r c₁ => simp [projectAxiom] at h
  | roleSub r s => simp [projectAxiom] at h
  | roleComp r s t => simp [projectAxiom] at h

/-- If the projection yields an atomic `disj` axiom, the preimage is exactly
    that axiom over `atom`s. -/
theorem projectAxiom_some_disj {a : AxiomP (Fin n) ρ} {c d : Fin n}
    (h : projectAxiom a = some (Axiom.disj c d)) :
    a = AxiomP.disj (.atom c) (.atom d) := by
  cases a with
  | sub c' d' =>
      cases c' with
      | atom a' =>
          cases d' with
          | atom b' => simp [projectAxiom] at h
          | top => simp [projectAxiom] at h
          | conj c₁ c₂ => simp [projectAxiom] at h
          | ex r c₁ => simp [projectAxiom] at h
      | top => simp [projectAxiom] at h
      | conj c₁ c₂ => simp [projectAxiom] at h
      | ex r c₁ => simp [projectAxiom] at h
  | disj c' d' =>
      cases c' with
      | atom a' =>
          cases d' with
          | atom b' =>
              injection h with hh
              injection hh with h1 h2
              subst h1; subst h2
              rfl
          | top => simp [projectAxiom] at h
          | conj c₁ c₂ => simp [projectAxiom] at h
          | ex r c₁ => simp [projectAxiom] at h
      | top => simp [projectAxiom] at h
      | conj c₁ c₂ => simp [projectAxiom] at h
      | ex r c₁ => simp [projectAxiom] at h
  | roleSub r s => simp [projectAxiom] at h
  | roleComp r s t => simp [projectAxiom] at h

/-- A projected `sub` edge comes from a genuine atomic inclusion of the
    original TBox. -/
theorem mem_project_sub {t : List (AxiomP (Fin n) ρ)} {c d : Fin n}
    (h : Axiom.sub c d ∈ project t) : AxiomP.sub (.atom c) (.atom d) ∈ t := by
  rw [project, List.mem_filterMap] at h
  obtain ⟨a, ha, hsome⟩ := h
  have haq := projectAxiom_some_sub hsome
  subst haq
  exact ha

/-- A projected `disj` edge comes from a genuine atomic disjointness of the
    original TBox. -/
theorem mem_project_disj {t : List (AxiomP (Fin n) ρ)} {c d : Fin n}
    (h : Axiom.disj c d ∈ project t) : AxiomP.disj (.atom c) (.atom d) ∈ t := by
  rw [project, List.mem_filterMap] at h
  obtain ⟨a, ha, hsome⟩ := h
  have haq := projectAxiom_some_disj hsome
  subst haq
  exact ha

/-- Every atomic-shadow derivation lifts to an EL⁺ derivation over `atom`s. -/
theorem subDer_lift {t : List (AxiomP (Fin n) ρ)} {c d : Fin n}
    (h : SubDer (project t) c d) : Der t (.atom c) (.atom d) := by
  induction h with
  | ofAxiom hmem => exact Der.ofAxiom (mem_project_sub hmem)
  | refl => exact Der.refl
  | trans _ _ ih1 ih2 => exact Der.trans ih1 ih2

/-- **Projection soundness for subsumption**: a boolean-computed atomic
    subsumption is a genuine EL⁺ derivation (via round 4's `subB_sound`). -/
theorem subBP_sound {t : List (AxiomP (Fin n) ρ)} {c d : Fin n}
    (h : subBP t c d = true) : Der t (.atom c) (.atom d) :=
  subDer_lift (subB_sound h)

/-- **Projection soundness for conflicts**: a boolean-computed atomic
    conflict is a genuine EL⁺ semantic conflict (via round 4's
    `conflictB_sound`). -/
theorem conflictBP_sound {t : List (AxiomP (Fin n) ρ)} {c c' : Fin n}
    (h : conflictBP t c c' = true) : DerivedConflictP t (.atom c) (.atom c') := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := conflictB_sound h
  refine ⟨.atom d₁, .atom d₂, subDer_lift h1, subDer_lift h2, ?_⟩
  cases hd with
  | inl hmem => exact Or.inl (mem_project_disj hmem)
  | inr hmem => exact Or.inr (mem_project_disj hmem)

end AtomProjection

-- ---------------------------------------------------------------------------
-- §7. Concrete instance: a SNOMED-flavoured TBox over `Fin 8` × `Fin 3`
-- ---------------------------------------------------------------------------

section SnomedInstance

/-!
Class ids (`Fin 8`):

- `0` Inflammation
- `1` Disorder
- `2` Lung
- `3` Organ
- `4` Pneumonia
- `5` Drug
- `6` InflammatoryLesion
- `7` DrugInducedDisorder

Role ids (`Fin 3`):

- `0` RoleGroup
- `1` DirectSite
- `2` PartOf
-/

/-- The instance TBox. Pneumonia is an inflammation located (via a direct
    site, which is a sub-role of the role group) in the lung; a drug-induced
    disorder is asserted to be both a disorder and a drug, which are
    disjoint — the incoherence witness. -/
def snomedTBox : List (AxiomP (Fin 8) (Fin 3)) :=
  [ .sub (.atom 4) (.atom 0)                           -- Pneumonia ⊑ Inflammation
  , .sub (.atom 0) (.atom 1)                           -- Inflammation ⊑ Disorder
  , .sub (.atom 2) (.atom 3)                           -- Lung ⊑ Organ
  , .sub (.atom 4) (.ex 1 (.conj (.atom 2) (.atom 0))) -- Pneumonia ⊑ ∃DirectSite.(Lung ⊓ Inflammation)
  , .roleSub 1 0                                       -- DirectSite ⊑ RoleGroup
  , .roleComp 1 2 0                                    -- DirectSite ∘ PartOf ⊑ RoleGroup
  , .sub (.atom 2) (.ex 2 (.atom 3))                    -- Lung ⊑ ∃PartOf.Organ
  , .disj (.atom 1) (.atom 5)                          -- Disorder ⊥ Drug
  , .sub (.atom 7) (.atom 1)                           -- DrugInducedDisorder ⊑ Disorder
  , .sub (.atom 7) (.atom 5)                           -- DrugInducedDisorder ⊑ Drug
  , .sub (.atom 6) (.atom 0)                           -- InflammatoryLesion ⊑ Inflammation
  ]

-- §7.1 Constructor-built derivations

/-- Pneumonia ⊑* Disorder (via Inflammation). -/
theorem der_pneumonia_disorder : Der snomedTBox (.atom 4) (.atom 1) :=
  .trans (.ofAxiom (c := .atom 4) (d := .atom 0) (by decide))
         (.ofAxiom (c := .atom 0) (d := .atom 1) (by decide))

/-- Pneumonia ⊑* ∃RoleGroup.(Lung ⊓ Inflammation): the stated
    ∃DirectSite restriction is widened along the role hierarchy
    (`exRoleSub` — the EL⁺ role-chaining rule). This is the
    "infectionSite ⊑ ∃roleGroup.(findingSite ⊓ inflammation)" pattern of
    SNOMED CT post-coordination. -/
theorem der_pneumonia_rolegroup :
    Der snomedTBox (.atom 4) (.ex 0 (.conj (.atom 2) (.atom 0))) :=
  .exRoleSub (.ofAxiom (c := .atom 4) (d := .ex 1 (.conj (.atom 2) (.atom 0)))
               (by decide))
             (r := 1) (s := 0) (by decide)

/-- Lung ⊓ Inflammation ⊑* Lung (⊓-elimination over the reflexive
    derivation). -/
theorem der_conj_elim_lung :
    Der snomedTBox (.conj (.atom 2) (.atom 0)) (.atom 2) :=
  .conjElimLeft .refl

/-- Pneumonia ⊑* ∃RoleGroup.Lung: `exRoleSub`, then `exMono` over
    ⊓-elimination, chained by `trans`. -/
theorem der_pneumonia_rg_lung : Der snomedTBox (.atom 4) (.ex 0 (.atom 2)) :=
  .trans der_pneumonia_rolegroup (.exMono der_conj_elim_lung)

/-- Pneumonia ⊑* ∃RoleGroup.Organ: one more `exMono` hop over Lung ⊑ Organ. -/
theorem der_pneumonia_rg_organ : Der snomedTBox (.atom 4) (.ex 0 (.atom 3)) :=
  .trans der_pneumonia_rg_lung
         (.exMono (.ofAxiom (c := .atom 2) (d := .atom 3) (by decide)))

/-- Lung ⊑* ∃PartOf.Organ (stated axiom). -/
theorem der_lung_partof_organ :
    Der snomedTBox (.atom 2) (.ex 2 (.atom 3)) :=
  .ofAxiom (c := .atom 2) (d := .ex 2 (.atom 3)) (by decide)

/-- Pneumonia ⊑* ∃DirectSite.Lung: from the stated axiom by ⊓-elimination
    and existential monotonicity. -/
theorem der_pneumonia_directsite_lung :
    Der snomedTBox (.atom 4) (.ex 1 (.atom 2)) :=
  .trans (.ofAxiom (c := .atom 4) (d := .ex 1 (.conj (.atom 2) (.atom 0)))
           (by decide))
         (.exMono (.conjElimLeft .refl))

/-- ∃DirectSite.Lung ⊑* ∃DirectSite.(∃PartOf.Organ) by existential
    monotonicity over `Lung ⊑ ∃PartOf.Organ`. -/
theorem der_ex_directsite_partof_organ :
    Der snomedTBox (.ex 1 (.atom 2)) (.ex 1 (.ex 2 (.atom 3))) :=
  .exMono der_lung_partof_organ

/-- Pneumonia ⊑* ∃DirectSite.(∃PartOf.Organ): chain the direct-site
    restriction through the part-of restriction. -/
theorem der_pneumonia_directsite_partof_organ :
    Der snomedTBox (.atom 4) (.ex 1 (.ex 2 (.atom 3))) :=
  .trans der_pneumonia_directsite_lung der_ex_directsite_partof_organ

/-- Pneumonia ⊑* ∃RoleGroup.Organ via role composition
    (`DirectSite ∘ PartOf ⊑ RoleGroup`). This is the EL⁺ composition
    chain SNOMED CT uses for multi-step location relations. -/
theorem der_pneumonia_rg_organ_via_comp :
    Der snomedTBox (.atom 4) (.ex 0 (.atom 3)) :=
  .exComp der_pneumonia_directsite_partof_organ (by decide)

/-- ⊓-introduction: Pneumonia ⊑* Disorder ⊓ Inflammation. -/
theorem der_pneumonia_conj :
    Der snomedTBox (.atom 4) (.conj (.atom 1) (.atom 0)) :=
  .conjIntro der_pneumonia_disorder
             (.ofAxiom (c := .atom 4) (d := .atom 0) (by decide))

/-- ⊤-rule: everything, even an existential, is subsumed by ⊤. -/
theorem der_ex_top : Der snomedTBox (.ex 0 (.atom 2)) .top := .topRule

/-- The drug-induced disorder is incoherent: it is subsumed by both Disorder
    and Drug, which are disjoint. -/
theorem incoherentP_did : IncoherentP snomedTBox (.atom 7) :=
  ⟨.atom 1, .atom 5,
   .ofAxiom (by decide), .ofAxiom (by decide), Or.inl (by decide)⟩

/-- The oracle flags (Pneumonia, Drug): pneumonia is subsumed by Disorder,
    Drug by itself, and Disorder ⊥ Drug — even though neither concept is
    incoherent on its own. -/
theorem conflictP_pneumonia_drug : DerivedConflictP snomedTBox (.atom 4) (.atom 5) :=
  ⟨.atom 1, .atom 5,
   der_pneumonia_disorder, .refl, Or.inl (by decide)⟩

-- §7.2 Semantic consequences in arbitrary models

/-- No model of the TBox has any drug-induced disorder. -/
theorem no_did_in_models {I : Fin 8 → ι → Prop} {IR : Fin 3 → ι → ι → Prop}
    (hI : SatisfiesP I IR snomedTBox) : ∀ x, ¬ meaning I IR (.atom 7) x :=
  incoherentP_empty incoherentP_did hI

/-- Oracle soundness, instantiated: no model can assert of one and the same
    entity that it is a pneumonia and that it is a drug — the licence for a
    repair operator to drop one of the two corresponding mappings. -/
theorem oracle_sound_P_instance {I : Fin 8 → ι → Prop} {IR : Fin 3 → ι → ι → Prop}
    (hI : SatisfiesP I IR snomedTBox) :
    ∀ x, ¬ (meaning I IR (.atom 4) x ∧ meaning I IR (.atom 5) x) :=
  oracle_sound_P conflictP_pneumonia_drug hI

-- §7.3 Decidable checks over the atom-only projection

/-- The projection keeps exactly the seven atomic axioms. -/
theorem check_project : project snomedTBox =
    [ .sub 4 0, .sub 0 1, .sub 2 3, .disj 1 5, .sub 7 1, .sub 7 5, .sub 6 0 ] := rfl

/-- Boolean closure: Pneumonia ⊑* Disorder is computed. -/
theorem check_subBP_4_1 : subBP snomedTBox 4 1 = true := by native_decide

/-- Boolean closure: DrugInducedDisorder ⊑* Drug is computed. -/
theorem check_subBP_7_5 : subBP snomedTBox 7 5 = true := by native_decide

/-- Boolean closure: InflammatoryLesion ⊑* Disorder (two hops). -/
theorem check_subBP_6_1 : subBP snomedTBox 6 1 = true := by native_decide

/-- Boolean closure: Organ is NOT subsumed by Disorder — the ∃DirectSite
    axiom and the role inclusion contribute nothing to the atomic shadow. -/
theorem check_subBP_3_1 : subBP snomedTBox 3 1 = false := by native_decide

/-- Boolean oracle: the incoherent drug-induced disorder self-conflicts. -/
theorem check_conflictBP_7_7 : conflictBP snomedTBox 7 7 = true := by native_decide

/-- Boolean oracle: pneumonia vs drug conflict is computed. -/
theorem check_conflictBP_4_5 : conflictBP snomedTBox 4 5 = true := by native_decide

/-- Boolean oracle: symmetry spot-check. -/
theorem check_conflictBP_5_4 : conflictBP snomedTBox 5 4 = true := by native_decide

/-- Boolean oracle: Lung and Drug do not conflict. -/
theorem check_conflictBP_2_5 : conflictBP snomedTBox 2 5 = false := by native_decide

/-- The computed Pneumonia ⊑* Disorder, re-derived through `subBP_sound`:
    boolean engine and constructor derivation agree. -/
theorem cv_der_4_1 : Der snomedTBox (.atom 4) (.atom 1) :=
  subBP_sound check_subBP_4_1

/-- The computed pneumonia–drug conflict, re-derived through
    `conflictBP_sound`. -/
theorem cv_conflict_4_5 : DerivedConflictP snomedTBox (.atom 4) (.atom 5) :=
  conflictBP_sound check_conflictBP_4_5

/-- The computed self-conflict of the drug-induced disorder, re-derived. -/
theorem cv_conflict_7_7 : DerivedConflictP snomedTBox (.atom 7) (.atom 7) :=
  conflictBP_sound check_conflictBP_7_7

end SnomedInstance

end Sounio.OntologyELPlus

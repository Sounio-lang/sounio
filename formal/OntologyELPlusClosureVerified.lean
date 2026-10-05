/-
# Sounio.OntologyELPlusClosureVerified — the role-aware boolean closure of the EL⁺ reasoner

Follow-up closure of round 7 of the *ontology frontiers* research line:
`OntologyELPlus.lean` lifts the mini reasoner to the EL⁺ fragment (concepts
with ⊤, ⊓, ∃, role inclusions, role compositions) and proves the derivation
system `Der` sound (`der_sound`), but its boolean side (`subBP` /
`conflictBP`) still routes through the **atom-only projection** — the
∃-restrictions and the role hierarchy contribute nothing to the computed
answers. This file closes that gap with a **role-aware saturation engine**
over the full concept language, proved sound against `Der`:

1. §1 Role-hierarchy closure: `RoleDer` (the deductive role hierarchy) is
   exactly `SubDer` over the role inclusions read as round-3 `sub` edges
   (`roleDer_iff_subDer`), so round 4's verified boolean closure lifts
   point-for-point: `roleSubB_sound`, `roleSubB_complete`, `roleSubB_iff`.
   `roleDer_der_ex` promotes a role derivation to an ∃-chaining step of
   `Der` — the iterated form of `Der.exRoleSub`.
2. §2 Concept universe: `subconcepts`, `baseConcepts`, and `conceptUniv` —
   the finite universe the engine works over: ⊤, every class id, the
   TBox-mentioned concepts closed under subconcepts, plus every existential
   role-variant `∃s.C` of a base concept (making the role hierarchy and one
   level of composition chaining available while keeping the universe
   finite). Membership facts: `mem_conceptUniv_of_mem_axiom`,
   `mem_conceptUniv_sub`.
3. §3 Saturation engine: `seedS` / `crStep` / `closeSat` compute a set `S`
   of subsumption pairs and a set `R` of role edges by the EL⁺-style
   completion rules (transitivity, ⊓-introduction/elimination,
   ∃-monotonicity, ∃-roleSub chaining, ∃-composition chaining).
   `SatJustified` is the invariant — every pair of `S` is `Der`-derivable,
   every edge `(r, C, D)` of `R` justifies `C ⊑* ∃r.D` — proved for the
   seed (`seedS_justified`), preserved by each round (`crStep_justified`),
   hence held by the closure (`closeSat_justified`). `subBPlus_sound` and
   `conflictBPlus_sound` lift the computed answers into `Der` and
   `DerivedConflictP`.
4. §4 The `Fin 8 × Fin 3` SNOMED instance is re-checked through the engine:
   `subBPlus` computes `Pneumonia ⊑* ∃RoleGroup.Organ` (a fact the
   atom-only projection cannot see — it needs the role hierarchy or the
   composition axiom) and `conflictBPlus` computes the pneumonia–drug
   conflict; both are re-derived into the deductive system through the
   soundness theorems.

Only soundness is claimed: the engine iterates a fixed number of rounds
(`satRounds`), enough for the worked instance; no completeness theorem is
stated. Self-contained modulo `OntologyELPlus`. No Mathlib. Zero sorry.
No new axioms.
-/

import OntologyELPlus

namespace Sounio.OntologyELPlus

open Sounio.OntologyELReasoner

variable {α ρ : Type}

-- ---------------------------------------------------------------------------
-- §1. The role-hierarchy closure, reusing round 4's verified boolean engine
-- ---------------------------------------------------------------------------

/-- The deductive role hierarchy: `RoleDer t r s` reads "`t` derives
    `r ⊑* s`" — the reflexive-transitive closure of the stated role
    inclusions. -/
inductive RoleDer (t : List (AxiomP α ρ)) : ρ → ρ → Prop
  | ofAxiom {r s : ρ} : AxiomP.roleSub r s ∈ t → RoleDer t r s
  | refl {r : ρ} : RoleDer t r r
  | trans {r s u : ρ} : RoleDer t r s → RoleDer t s u → RoleDer t r u

/-- A role derivation chains existential restrictions along the hierarchy:
    the iterated form of `Der.exRoleSub`. -/
theorem roleDer_der_ex {t : List (AxiomP α ρ)} {r s : ρ} (hr : RoleDer t r s) :
    ∀ {X D : Concept α ρ}, Der t X (.ex r D) → Der t X (.ex s D) := by
  induction hr with
  | ofAxiom hmem => intro X D h; exact Der.exRoleSub h hmem
  | refl => intro X D h; exact h
  | trans _ _ ih1 ih2 => intro X D h; exact ih2 (ih1 h)

section RoleClosure

variable {n m : Nat}

/-- The role inclusions of the TBox, read as round-3 atomic `sub` edges
    over the role ids. -/
def roleAxioms (t : List (AxiomP (Fin n) (Fin m))) : List (Axiom (Fin m)) :=
  t.filterMap fun a => match a with
    | .roleSub r s => some (.sub r s)
    | _ => none

/-- A stated role inclusion appears as a `sub` edge of `roleAxioms`. -/
theorem roleSub_mem_roleAxioms {t : List (AxiomP (Fin n) (Fin m))} {r s : Fin m}
    (h : AxiomP.roleSub r s ∈ t) : Axiom.sub r s ∈ roleAxioms t := by
  rw [roleAxioms, List.mem_filterMap]
  exact ⟨.roleSub r s, h, rfl⟩

/-- Every `sub` edge of `roleAxioms` comes from a stated role inclusion. -/
theorem mem_roleAxioms_roleSub {t : List (AxiomP (Fin n) (Fin m))} {r s : Fin m}
    (h : Axiom.sub r s ∈ roleAxioms t) : AxiomP.roleSub r s ∈ t := by
  rw [roleAxioms, List.mem_filterMap] at h
  obtain ⟨a, ha, hsome⟩ := h
  cases a with
  | sub c d =>
      simp only at hsome
      nomatch hsome
  | disj c d =>
      simp only at hsome
      nomatch hsome
  | roleSub r' s' =>
      simp only at hsome
      injection hsome with hsub
      injection hsub with hr hs
      subst hr; subst hs
      exact ha
  | roleComp r' s' u' =>
      simp only at hsome
      nomatch hsome

/-- **The role hierarchy is exactly round 3's closure over the role
    edges** — so round 4's verified boolean engine applies to it. -/
theorem roleDer_iff_subDer {t : List (AxiomP (Fin n) (Fin m))} {r s : Fin m} :
    RoleDer t r s ↔ SubDer (roleAxioms t) r s := by
  constructor
  · intro h
    induction h with
    | ofAxiom hmem => exact SubDer.ofAxiom (roleSub_mem_roleAxioms hmem)
    | refl => exact SubDer.refl
    | trans _ _ ih1 ih2 => exact SubDer.trans ih1 ih2
  · intro h
    induction h with
    | ofAxiom hmem => exact RoleDer.ofAxiom (mem_roleAxioms_roleSub hmem)
    | refl => exact RoleDer.refl
    | trans _ _ ih1 ih2 => exact RoleDer.trans ih1 ih2

/-- Boolean role-subsumption check, reusing round 4's verified closure. -/
def roleSubB (t : List (AxiomP (Fin n) (Fin m))) (r s : Fin m) : Bool :=
  subB (roleAxioms t) r s

/-- **Role-closure soundness**, via round 4's `subB_sound`. -/
theorem roleSubB_sound {t : List (AxiomP (Fin n) (Fin m))} {r s : Fin m}
    (h : roleSubB t r s = true) : RoleDer t r s :=
  roleDer_iff_subDer.mpr (subB_sound h)

/-- **Role-closure completeness**, via round 4's `subB_complete`. -/
theorem roleSubB_complete {t : List (AxiomP (Fin n) (Fin m))} {r s : Fin m}
    (h : RoleDer t r s) : roleSubB t r s = true :=
  subB_complete (roleDer_iff_subDer.mp h)

/-- The boolean role closure is exactly the deductive role hierarchy. -/
theorem roleSubB_iff {t : List (AxiomP (Fin n) (Fin m))} {r s : Fin m} :
    roleSubB t r s = true ↔ RoleDer t r s :=
  ⟨roleSubB_sound, roleSubB_complete⟩

end RoleClosure

-- ---------------------------------------------------------------------------
-- §2. The concept universe
-- ---------------------------------------------------------------------------

/-- All subconcepts of a concept, including itself. -/
def subconcepts : Concept α ρ → List (Concept α ρ)
  | .atom a => [.atom a]
  | .top => [.top]
  | .conj c d => .conj c d :: (subconcepts c ++ subconcepts d)
  | .ex r c => .ex r c :: subconcepts c

/-- Membership in `subconcepts` is transitive. -/
theorem mem_subconcepts_trans {a b c : Concept α ρ} :
    a ∈ subconcepts b → b ∈ subconcepts c → a ∈ subconcepts c := by
  induction c with
  | atom x =>
      intro hab hbc
      rw [show subconcepts (@Concept.atom α ρ x) = [.atom x] from rfl] at hbc
      have hb : b = .atom x := List.mem_singleton.mp hbc
      subst hb
      exact hab
  | top =>
      intro hab hbc
      rw [show subconcepts (@Concept.top α ρ) = [.top] from rfl] at hbc
      have hb : b = .top := List.mem_singleton.mp hbc
      subst hb
      exact hab
  | conj c₁ c₂ ih1 ih2 =>
      intro hab hbc
      rw [show subconcepts (.conj c₁ c₂) =
        .conj c₁ c₂ :: (subconcepts c₁ ++ subconcepts c₂) from rfl] at hbc
      rw [List.mem_cons, List.mem_append] at hbc
      cases hbc with
      | inl hb => subst hb; exact hab
      | inr hb =>
          rw [show subconcepts (.conj c₁ c₂) =
            .conj c₁ c₂ :: (subconcepts c₁ ++ subconcepts c₂) from rfl]
          cases hb with
          | inl h1 =>
              exact List.mem_cons_of_mem _ (List.mem_append_left _ (ih1 hab h1))
          | inr h2 =>
              exact List.mem_cons_of_mem _ (List.mem_append_right _ (ih2 hab h2))
  | ex r c' ih =>
      intro hab hbc
      rw [show subconcepts (.ex r c') = .ex r c' :: subconcepts c' from rfl] at hbc
      rw [List.mem_cons] at hbc
      cases hbc with
      | inl hb => subst hb; exact hab
      | inr h =>
          rw [show subconcepts (.ex r c') = .ex r c' :: subconcepts c' from rfl]
          exact List.mem_cons_of_mem _ (ih hab h)

/-- The concepts mentioned by an axiom, closed under subconcepts. -/
def axiomConcepts : AxiomP α ρ → List (Concept α ρ)
  | .sub c d => subconcepts c ++ subconcepts d
  | .disj c d => subconcepts c ++ subconcepts d
  | .roleSub _ _ => []
  | .roleComp _ _ _ => []

/-- Subconcept membership descends into `axiomConcepts`. -/
theorem mem_axiomConcepts_sub {c d : Concept α ρ} {a : AxiomP α ρ}
    (hcd : c ∈ subconcepts d) (hd : d ∈ axiomConcepts a) : c ∈ axiomConcepts a := by
  cases a with
  | sub c₁ c₂ =>
      have hd' : d ∈ subconcepts c₁ ++ subconcepts c₂ := hd
      cases List.mem_append.mp hd' with
      | inl h => exact List.mem_append_left _ (mem_subconcepts_trans hcd h)
      | inr h => exact List.mem_append_right _ (mem_subconcepts_trans hcd h)
  | disj c₁ c₂ =>
      have hd' : d ∈ subconcepts c₁ ++ subconcepts c₂ := hd
      cases List.mem_append.mp hd' with
      | inl h => exact List.mem_append_left _ (mem_subconcepts_trans hcd h)
      | inr h => exact List.mem_append_right _ (mem_subconcepts_trans hcd h)
  | roleSub r s => exact absurd hd List.not_mem_nil
  | roleComp r s u => exact absurd hd List.not_mem_nil

section ConceptUniverse

variable {n m : Nat}

/-- The base universe: ⊤, every class id, and every concept the TBox
    mentions (closed under subconcepts). -/
def baseConcepts (t : List (AxiomP (Fin n) (Fin m))) : List (Concept (Fin n) (Fin m)) :=
  (.top :: (List.finRange n).map Concept.atom) ++ t.flatMap axiomConcepts

/-- The saturation universe: the base concepts plus every existential
    role-variant `∃s.C` of a base concept. The variants make the role
    hierarchy (and one level of composition chaining) available to the
    engine while keeping the universe finite. -/
def conceptUniv (t : List (AxiomP (Fin n) (Fin m))) : List (Concept (Fin n) (Fin m)) :=
  baseConcepts t ++ (baseConcepts t).flatMap fun c => (List.finRange m).map fun r => .ex r c

/-- The base universe is closed under subconcepts. -/
theorem mem_baseConcepts_sub {t : List (AxiomP (Fin n) (Fin m))}
    {c d : Concept (Fin n) (Fin m)}
    (hcd : c ∈ subconcepts d) (hd : d ∈ baseConcepts t) : c ∈ baseConcepts t := by
  rw [baseConcepts, List.mem_append] at hd
  cases hd with
  | inl hd =>
      rw [List.mem_cons] at hd
      cases hd with
      | inl hdt =>
          subst hdt
          rw [show subconcepts (@Concept.top (Fin n) (Fin m)) = [.top] from rfl] at hcd
          have hc : c = .top := List.mem_singleton.mp hcd
          subst hc
          rw [baseConcepts]
          exact List.mem_append_left _ List.mem_cons_self
      | inr hdat =>
          rw [List.mem_map] at hdat
          obtain ⟨a, -, hda⟩ := hdat
          subst hda
          rw [show subconcepts (Concept.atom a) = [.atom a] from rfl] at hcd
          have hc : c = .atom a := List.mem_singleton.mp hcd
          subst hc
          rw [baseConcepts]
          exact List.mem_append_left _
            (List.mem_cons_of_mem _ (List.mem_map.mpr ⟨a, List.mem_finRange a, rfl⟩))
  | inr hd =>
      rw [List.mem_flatMap] at hd
      obtain ⟨a, ha, hda⟩ := hd
      rw [baseConcepts]
      apply List.mem_append_right
      rw [List.mem_flatMap]
      exact ⟨a, ha, mem_axiomConcepts_sub hcd hda⟩

/-- Every concept mentioned by a TBox axiom lies in the universe. -/
theorem mem_conceptUniv_of_mem_axiom {t : List (AxiomP (Fin n) (Fin m))}
    {a : AxiomP (Fin n) (Fin m)} {c : Concept (Fin n) (Fin m)}
    (ha : a ∈ t) (hc : c ∈ axiomConcepts a) : c ∈ conceptUniv t := by
  rw [conceptUniv]
  apply List.mem_append_left
  rw [baseConcepts]
  apply List.mem_append_right
  rw [List.mem_flatMap]
  exact ⟨a, ha, hc⟩

/-- The universe is closed under subconcepts. -/
theorem mem_conceptUniv_sub {t : List (AxiomP (Fin n) (Fin m))}
    {c d : Concept (Fin n) (Fin m)}
    (hcd : c ∈ subconcepts d) (hd : d ∈ conceptUniv t) : c ∈ conceptUniv t := by
  rw [conceptUniv, List.mem_append] at hd
  cases hd with
  | inl hdb =>
      rw [conceptUniv]
      exact List.mem_append_left _ (mem_baseConcepts_sub hcd hdb)
  | inr hdv =>
      rw [List.mem_flatMap] at hdv
      obtain ⟨b, hbb, hdv⟩ := hdv
      rw [List.mem_map] at hdv
      obtain ⟨r, -, hdrb⟩ := hdv
      subst hdrb
      rw [show subconcepts (.ex r b) = .ex r b :: subconcepts b from rfl,
        List.mem_cons] at hcd
      cases hcd with
      | inl hcd =>
          subst hcd
          rw [conceptUniv]
          exact List.mem_append_right _
            (List.mem_flatMap.mpr ⟨b, hbb, List.mem_map.mpr ⟨r, List.mem_finRange r, rfl⟩⟩)
      | inr hcb =>
          rw [conceptUniv]
          exact List.mem_append_left _ (mem_baseConcepts_sub hcb hbb)

end ConceptUniverse

-- ---------------------------------------------------------------------------
-- §3. The role-aware saturation engine and its soundness
-- ---------------------------------------------------------------------------

section Saturation

variable {n m : Nat}

/-- Subsumption-pair relations over the concept universe. -/
abbrev SRel (n m : Nat) := List (Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m))

/-- Role-edge relations: `(r, C, D)` reads "`C ⊑* ∃r.D`". -/
abbrev RRel (n m : Nat) :=
  List (Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m))

-- §3.1 List utilities

/-- Deduplicate (keeping first occurrences), using decidable equality. -/
def dedupN [DecidableEq α] (l : List α) : List α :=
  l.foldr (fun x acc => if x ∈ acc then acc else x :: acc) []

/-- Membership in `dedupN` is membership in the original list. -/
theorem mem_dedupN [DecidableEq α] {x : α} {l : List α} :
    x ∈ dedupN l ↔ x ∈ l := by
  induction l with
  | nil => exact ⟨id, id⟩
  | cons a l ih =>
      show x ∈ (if a ∈ dedupN l then dedupN l else a :: dedupN l) ↔ x ∈ a :: l
      by_cases h : a ∈ dedupN l
      · rw [if_pos h]
        constructor
        · intro hx
          exact List.mem_cons_of_mem _ (ih.mp hx)
        · intro hx
          cases List.mem_cons.mp hx with
          | inl he => subst he; exact h
          | inr hl => exact ih.mpr hl
      · rw [if_neg h]
        constructor
        · intro hx
          cases List.mem_cons.mp hx with
          | inl he => subst he; exact List.mem_cons_self
          | inr hxl => exact List.mem_cons_of_mem _ (ih.mp hxl)
        · intro hx
          cases List.mem_cons.mp hx with
          | inl he => subst he; exact List.mem_cons_self
          | inr hl => exact List.mem_cons_of_mem _ (ih.mpr hl)

/-- Membership in an `if`-guarded list. -/
theorem mem_ite_nil {P : Prop} [Decidable P] {x : α} {L : List α} :
    x ∈ (if P then L else []) ↔ P ∧ x ∈ L := by
  by_cases h : P
  · rw [if_pos h]
    exact ⟨fun hx => ⟨h, hx⟩, fun ⟨_, hx⟩ => hx⟩
  · rw [if_neg h]
    exact ⟨fun hx => absurd hx List.not_mem_nil, fun hw => absurd hw.1 h⟩

/-- Membership in an `if`-guarded singleton. -/
theorem mem_ite_singleton {P : Prop} [Decidable P] {x y : α} :
    x ∈ (if P then [y] else []) ↔ P ∧ x = y := by
  rw [mem_ite_nil]
  constructor
  · intro ⟨hP, hx⟩
    exact ⟨hP, List.mem_singleton.mp hx⟩
  · intro ⟨hP, hx⟩
    exact ⟨hP, List.mem_singleton.mpr hx⟩

-- §3.2 The completion rules, as candidate generators

/-- Transitivity candidates over `S`. -/
def genTrans (S : SRel n m) : SRel n m :=
  S.flatMap fun p => S.flatMap fun q =>
    if p.2 = q.1 then [(p.1, q.2)] else []

theorem mem_genTrans {x : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {S : SRel n m} :
    x ∈ genTrans S ↔ ∃ p ∈ S, ∃ q ∈ S, p.2 = q.1 ∧ x = (p.1, q.2) := by
  rw [genTrans, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨p, hp, hx⟩ := h
    rw [List.mem_flatMap] at hx
    obtain ⟨q, hq, hx⟩ := hx
    rw [mem_ite_singleton] at hx
    exact ⟨p, hp, q, hq, hx⟩
  · intro h
    obtain ⟨p, hp, q, hq, hpq, hxe⟩ := h
    exact ⟨p, hp, List.mem_flatMap.mpr ⟨q, hq, mem_ite_singleton.mpr ⟨hpq, hxe⟩⟩⟩

/-- ⊓-elimination candidates from a single pair. -/
def conjElimCand (p : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)) : SRel n m :=
  match p.2 with
  | .conj d₁ d₂ => [(p.1, d₁), (p.1, d₂)]
  | _ => []

theorem mem_conjElimCand {x p : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)} :
    x ∈ conjElimCand p ↔
      ∃ d₁ d₂, p.2 = .conj d₁ d₂ ∧ (x = (p.1, d₁) ∨ x = (p.1, d₂)) := by
  obtain ⟨c, d⟩ := p
  cases d with
  | atom a =>
      show x ∈ [] ↔ _
      constructor
      · intro hx; exact absurd hx List.not_mem_nil
      · intro h; obtain ⟨d₁, d₂, hp2, -⟩ := h; nomatch hp2
  | top =>
      show x ∈ [] ↔ _
      constructor
      · intro hx; exact absurd hx List.not_mem_nil
      · intro h; obtain ⟨d₁, d₂, hp2, -⟩ := h; nomatch hp2
  | conj d₁ d₂ =>
      show x ∈ [(c, d₁), (c, d₂)] ↔ _
      rw [List.mem_cons, List.mem_singleton]
      constructor
      · intro hx; exact ⟨d₁, d₂, rfl, hx⟩
      · intro h
        obtain ⟨d₁', d₂', hp2, hx⟩ := h
        injection hp2 with h1 h2
        subst h1; subst h2
        exact hx
  | ex r d' =>
      show x ∈ [] ↔ _
      constructor
      · intro hx; exact absurd hx List.not_mem_nil
      · intro h; obtain ⟨d₁, d₂, hp2, -⟩ := h; nomatch hp2

/-- ⊓-elimination candidates over `S`. -/
def genConjElim (S : SRel n m) : SRel n m := S.flatMap conjElimCand

theorem mem_genConjElim {x : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {S : SRel n m} :
    x ∈ genConjElim S ↔ ∃ p ∈ S, ∃ d₁ d₂, p.2 = .conj d₁ d₂ ∧
      (x = (p.1, d₁) ∨ x = (p.1, d₂)) := by
  rw [genConjElim, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨p, hp, hx⟩ := h
    rw [mem_conjElimCand] at hx
    exact ⟨p, hp, hx⟩
  · intro h
    obtain ⟨p, hp, hx⟩ := h
    exact ⟨p, hp, mem_conjElimCand.mpr hx⟩

/-- ⊓-introduction candidates over `S` (restricted to the universe). -/
def genConjIntro (S : SRel n m) (univ : List (Concept (Fin n) (Fin m))) : SRel n m :=
  S.flatMap fun p => S.flatMap fun q =>
    if p.1 = q.1 ∧ (.conj p.2 q.2) ∈ univ then [(p.1, .conj p.2 q.2)] else []

theorem mem_genConjIntro {x : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {S : SRel n m} {univ : List (Concept (Fin n) (Fin m))} :
    x ∈ genConjIntro S univ ↔ ∃ p ∈ S, ∃ q ∈ S, p.1 = q.1 ∧
      (.conj p.2 q.2) ∈ univ ∧ x = (p.1, .conj p.2 q.2) := by
  rw [genConjIntro, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨p, hp, hx⟩ := h
    rw [List.mem_flatMap] at hx
    obtain ⟨q, hq, hx⟩ := hx
    rw [mem_ite_singleton] at hx
    obtain ⟨⟨hpq, hu⟩, hxe⟩ := hx
    exact ⟨p, hp, q, hq, hpq, hu, hxe⟩
  · intro h
    obtain ⟨p, hp, q, hq, hpq, hu, hxe⟩ := h
    exact ⟨p, hp, List.mem_flatMap.mpr ⟨q, hq, mem_ite_singleton.mpr ⟨⟨hpq, hu⟩, hxe⟩⟩⟩

/-- Role-edge candidates from a single pair: `C ⊑* ∃r.D` becomes
    `(r, C, D)`. -/
def stoRCand (p : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)) : RRel n m :=
  match p.2 with
  | .ex r d => [(r, p.1, d)]
  | _ => []

theorem mem_stoRCand {x : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {p : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)} :
    x ∈ stoRCand p ↔ ∃ r d, p.2 = .ex r d ∧ x = (r, p.1, d) := by
  obtain ⟨c, d⟩ := p
  cases d with
  | atom a =>
      show x ∈ [] ↔ _
      constructor
      · intro hx; exact absurd hx List.not_mem_nil
      · intro h; obtain ⟨r, d', hp2, -⟩ := h; nomatch hp2
  | top =>
      show x ∈ [] ↔ _
      constructor
      · intro hx; exact absurd hx List.not_mem_nil
      · intro h; obtain ⟨r, d', hp2, -⟩ := h; nomatch hp2
  | conj d₁ d₂ =>
      show x ∈ [] ↔ _
      constructor
      · intro hx; exact absurd hx List.not_mem_nil
      · intro h; obtain ⟨r, d', hp2, -⟩ := h; nomatch hp2
  | ex r d' =>
      show x ∈ [(r, c, d')] ↔ _
      rw [List.mem_singleton]
      constructor
      · intro hx; exact ⟨r, d', rfl, hx⟩
      · intro h
        obtain ⟨r', d'', hp2, hx⟩ := h
        injection hp2 with h1 h2
        subst h1; subst h2
        exact hx

/-- Role-edge candidates over `S`. -/
def genStoR (S : SRel n m) : RRel n m := S.flatMap stoRCand

theorem mem_genStoR {x : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {S : SRel n m} :
    x ∈ genStoR S ↔ ∃ p ∈ S, ∃ r d, p.2 = .ex r d ∧ x = (r, p.1, d) := by
  rw [genStoR, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨p, hp, hx⟩ := h
    rw [mem_stoRCand] at hx
    exact ⟨p, hp, hx⟩
  · intro h
    obtain ⟨p, hp, hx⟩ := h
    exact ⟨p, hp, mem_stoRCand.mpr hx⟩

/-- ∃-monotonicity candidates: `(r, C, D)` and `D ⊑* D'` give `(r, C, D')`
    (restricted to the universe). -/
def genRmono (R : RRel n m) (S : SRel n m) (univ : List (Concept (Fin n) (Fin m))) :
    RRel n m :=
  R.flatMap fun e => S.flatMap fun p =>
    if e.2.2 = p.1 ∧ (.ex e.1 p.2) ∈ univ then [(e.1, e.2.1, p.2)] else []

theorem mem_genRmono {x : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {R : RRel n m} {S : SRel n m} {univ : List (Concept (Fin n) (Fin m))} :
    x ∈ genRmono R S univ ↔ ∃ e ∈ R, ∃ p ∈ S, e.2.2 = p.1 ∧
      (.ex e.1 p.2) ∈ univ ∧ x = (e.1, e.2.1, p.2) := by
  rw [genRmono, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨e, he, hx⟩ := h
    rw [List.mem_flatMap] at hx
    obtain ⟨p, hp, hx⟩ := hx
    rw [mem_ite_singleton] at hx
    obtain ⟨⟨hep, hu⟩, hxe⟩ := hx
    exact ⟨e, he, p, hp, hep, hu, hxe⟩
  · intro h
    obtain ⟨e, he, p, hp, hep, hu, hxe⟩ := h
    exact ⟨e, he, List.mem_flatMap.mpr ⟨p, hp, mem_ite_singleton.mpr ⟨⟨hep, hu⟩, hxe⟩⟩⟩

/-- ∃-roleSub candidates: `(r, C, D)` and a derived `r ⊑* s` give
    `(s, C, D)` (restricted to the universe). -/
def genRroleSub (t : List (AxiomP (Fin n) (Fin m))) (R : RRel n m)
    (univ : List (Concept (Fin n) (Fin m))) : RRel n m :=
  R.flatMap fun e => (List.finRange m).flatMap fun s =>
    if roleSubB t e.1 s = true ∧ (.ex s e.2.2) ∈ univ then [(s, e.2.1, e.2.2)] else []

theorem mem_genRroleSub {x : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {t : List (AxiomP (Fin n) (Fin m))} {R : RRel n m}
    {univ : List (Concept (Fin n) (Fin m))} :
    x ∈ genRroleSub t R univ ↔ ∃ e ∈ R, ∃ s ∈ List.finRange m,
      roleSubB t e.1 s = true ∧ (.ex s e.2.2) ∈ univ ∧ x = (s, e.2.1, e.2.2) := by
  rw [genRroleSub, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨e, he, hx⟩ := h
    rw [List.mem_flatMap] at hx
    obtain ⟨s, hs, hx⟩ := hx
    rw [mem_ite_singleton] at hx
    obtain ⟨⟨hrole, hu⟩, hxe⟩ := hx
    exact ⟨e, he, s, hs, hrole, hu, hxe⟩
  · intro h
    obtain ⟨e, he, s, hs, hrole, hu, hxe⟩ := h
    exact ⟨e, he, List.mem_flatMap.mpr ⟨s, hs, mem_ite_singleton.mpr ⟨⟨hrole, hu⟩, hxe⟩⟩⟩

/-- ∃-composition candidates from a single axiom: if `a` is
    `roleComp r s u` matching the chained edges `(r, C, D)` and
    `(s, D, E)`, yield `(u, C, E)` (restricted to the universe). -/
def compCand (e f : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m))
    (univ : List (Concept (Fin n) (Fin m))) (a : AxiomP (Fin n) (Fin m)) :
    Option (Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)) :=
  match a with
  | .roleComp r s u =>
      if r = e.1 ∧ s = f.1 ∧ (.ex u f.2.2) ∈ univ then some (u, e.2.1, f.2.2) else none
  | _ => none

theorem compCand_some {e f : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {univ : List (Concept (Fin n) (Fin m))} {a : AxiomP (Fin n) (Fin m)}
    {x : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    (h : compCand e f univ a = some x) :
    ∃ r s u, a = .roleComp r s u ∧ r = e.1 ∧ s = f.1 ∧
      (.ex u f.2.2) ∈ univ ∧ x = (u, e.2.1, f.2.2) := by
  cases a with
  | sub c d => nomatch h
  | disj c d => nomatch h
  | roleSub r s => nomatch h
  | roleComp r s u =>
      have h' : (if r = e.1 ∧ s = f.1 ∧ (.ex u f.2.2) ∈ univ then
          some (u, e.2.1, f.2.2) else none) = some x := h
      by_cases hc : r = e.1 ∧ s = f.1 ∧ (.ex u f.2.2) ∈ univ
      · rw [if_pos hc] at h'
        injection h' with hx
        exact ⟨r, s, u, rfl, hc.1, hc.2.1, hc.2.2, hx.symm⟩
      · rw [if_neg hc] at h'
        nomatch h'

/-- ∃-composition candidates over chained edge pairs. -/
def genRcomp (t : List (AxiomP (Fin n) (Fin m))) (R : RRel n m)
    (univ : List (Concept (Fin n) (Fin m))) : RRel n m :=
  R.flatMap fun e => R.flatMap fun f =>
    if e.2.2 = f.2.1 then t.filterMap (compCand e f univ) else []

theorem mem_genRcomp {x : Fin m × Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {t : List (AxiomP (Fin n) (Fin m))} {R : RRel n m}
    {univ : List (Concept (Fin n) (Fin m))} :
    x ∈ genRcomp t R univ ↔ ∃ e ∈ R, ∃ f ∈ R, e.2.2 = f.2.1 ∧
      ∃ r s u, AxiomP.roleComp r s u ∈ t ∧ r = e.1 ∧ s = f.1 ∧
        (.ex u f.2.2) ∈ univ ∧ x = (u, e.2.1, f.2.2) := by
  rw [genRcomp, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨e, he, hx⟩ := h
    rw [List.mem_flatMap] at hx
    obtain ⟨f, hf, hx⟩ := hx
    rw [mem_ite_nil] at hx
    obtain ⟨hef, hx⟩ := hx
    rw [List.mem_filterMap] at hx
    obtain ⟨a, ha, hsome⟩ := hx
    obtain ⟨r, s, u, haq, hr, hs, hu, hxe⟩ := compCand_some hsome
    subst haq
    exact ⟨e, he, f, hf, hef, r, s, u, ha, hr, hs, hu, hxe⟩
  · intro h
    obtain ⟨e, he, f, hf, hef, r, s, u, hcomp, hr, hs, hu, hxe⟩ := h
    refine ⟨e, he, List.mem_flatMap.mpr ⟨f, hf, mem_ite_nil.mpr ⟨hef, ?_⟩⟩⟩
    rw [List.mem_filterMap]
    refine ⟨.roleComp r s u, hcomp, ?_⟩
    show (if r = e.1 ∧ s = f.1 ∧ (.ex u f.2.2) ∈ univ then
        some (u, e.2.1, f.2.2) else none) = some x
    rw [if_pos ⟨hr, hs, hu⟩]
    rw [hxe]

/-- Role-edge-to-subsumption candidates: `(r, C, D)` gives the pair
    `C ⊑ ∃r.D` (restricted to the universe). -/
def genRtoS (R : RRel n m) (univ : List (Concept (Fin n) (Fin m))) : SRel n m :=
  R.flatMap fun e =>
    if (.ex e.1 e.2.2) ∈ univ then [(e.2.1, .ex e.1 e.2.2)] else []

theorem mem_genRtoS {x : Concept (Fin n) (Fin m) × Concept (Fin n) (Fin m)}
    {R : RRel n m} {univ : List (Concept (Fin n) (Fin m))} :
    x ∈ genRtoS R univ ↔ ∃ e ∈ R, (.ex e.1 e.2.2) ∈ univ ∧
      x = (e.2.1, .ex e.1 e.2.2) := by
  rw [genRtoS, List.mem_flatMap]
  constructor
  · intro h
    obtain ⟨e, he, hx⟩ := h
    rw [mem_ite_singleton] at hx
    exact ⟨e, he, hx⟩
  · intro h
    obtain ⟨e, he, hu, hxe⟩ := h
    exact ⟨e, he, mem_ite_singleton.mpr ⟨hu, hxe⟩⟩

-- §3.3 The seed and its justification

/-- The seed relation: reflexivity and ⊤-pairs for every universe concept,
    plus the stated inclusions. -/
def seedS (t : List (AxiomP (Fin n) (Fin m))) : SRel n m :=
  (conceptUniv t).flatMap (fun c => [(c, c), (c, .top)]) ++
    t.filterMap (fun a => match a with
      | .sub c d => some (c, d)
      | _ => none)

/-- The saturation invariant: every pair of `S` is derivable, and every
    edge `(r, C, D)` of `R` justifies `C ⊑* ∃r.D`. -/
def SatJustified (t : List (AxiomP (Fin n) (Fin m))) (S : SRel n m) (R : RRel n m) :
    Prop :=
  (∀ p ∈ S, Der t p.1 p.2) ∧ (∀ e ∈ R, Der t e.2.1 (.ex e.1 e.2.2))

/-- The seed relation is justified: reflexivity pairs by `Der.refl`,
    ⊤-pairs by `Der.topRule`, stated inclusions by `Der.ofAxiom`. -/
theorem seedS_justified {t : List (AxiomP (Fin n) (Fin m))} :
    SatJustified t (seedS t) [] := by
  constructor
  · intro x hx
    rw [seedS, List.mem_append] at hx
    cases hx with
    | inl hx =>
        rw [List.mem_flatMap] at hx
        obtain ⟨c, -, hx⟩ := hx
        rw [List.mem_cons, List.mem_singleton] at hx
        cases hx with
        | inl he => subst he; exact Der.refl
        | inr he => subst he; exact Der.topRule
    | inr hx =>
        rw [List.mem_filterMap] at hx
        obtain ⟨a, ha, hsome⟩ := hx
        cases a with
        | sub c d =>
            simp only at hsome
            injection hsome with hx
            subst hx
            exact Der.ofAxiom ha
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

-- §3.4 One saturation round and its justification

/-- One saturation round: close `S` and `R` under the completion rules
    (transitivity, ⊓-introduction/elimination, ∃-monotonicity,
    ∃-roleSub chaining, ∃-composition chaining), deduplicated. -/
def crStep (t : List (AxiomP (Fin n) (Fin m))) (S : SRel n m) (R : RRel n m) :
    SRel n m × RRel n m :=
  (dedupN (S ++ (genTrans S ++ (genConjElim S ++ (genConjIntro S (conceptUniv t) ++
      genRtoS R (conceptUniv t))))),
   dedupN (R ++ (genStoR S ++ (genRmono R S (conceptUniv t) ++
      (genRroleSub t R (conceptUniv t) ++ genRcomp t R (conceptUniv t))))))

/-- `crStep`'s first component, named so proofs can rewrite with it. -/
theorem crStep_fst (t : List (AxiomP (Fin n) (Fin m))) (S : SRel n m) (R : RRel n m) :
    (crStep t S R).1 =
      dedupN (S ++ (genTrans S ++ (genConjElim S ++ (genConjIntro S (conceptUniv t) ++
        genRtoS R (conceptUniv t))))) := rfl

/-- `crStep`'s second component, named so proofs can rewrite with it. -/
theorem crStep_snd (t : List (AxiomP (Fin n) (Fin m))) (S : SRel n m) (R : RRel n m) :
    (crStep t S R).2 =
      dedupN (R ++ (genStoR S ++ (genRmono R S (conceptUniv t) ++
        (genRroleSub t R (conceptUniv t) ++ genRcomp t R (conceptUniv t))))) := rfl

/-- **One-round soundness**: if `(S, R)` is justified, so is one saturation
    round — each completion rule is a rule of `Der` (or, for the role
    rules, a derived rule: `roleDer_der_ex` for the hierarchy, and
    `exMono`+`trans`+`exComp` for composition). -/
theorem crStep_justified {t : List (AxiomP (Fin n) (Fin m))} {S : SRel n m}
    {R : RRel n m} (h : SatJustified t S R) :
    SatJustified t (crStep t S R).1 (crStep t S R).2 := by
  obtain ⟨hS, hR⟩ := h
  constructor
  · intro x hx
    rw [crStep_fst, mem_dedupN, List.mem_append] at hx
    cases hx with
    | inl hx => exact hS x hx
    | inr hx =>
        rw [List.mem_append] at hx
        cases hx with
        | inl hx =>
            rw [mem_genTrans] at hx
            obtain ⟨p, hp, q, hq, hpq, hxe⟩ := hx
            subst hxe
            have h1 := hS p hp
            have h2 := hS q hq
            rw [← hpq] at h2
            exact Der.trans h1 h2
        | inr hx =>
            rw [List.mem_append] at hx
            cases hx with
            | inl hx =>
                rw [mem_genConjElim] at hx
                obtain ⟨p, hp, d₁, d₂, hp2, hx⟩ := hx
                have hpd := hS p hp
                rw [hp2] at hpd
                cases hx with
                | inl hxe => subst hxe; exact Der.conjElimLeft hpd
                | inr hxe => subst hxe; exact Der.conjElimRight hpd
            | inr hx =>
                rw [List.mem_append] at hx
                cases hx with
                | inl hx =>
                    rw [mem_genConjIntro] at hx
                    obtain ⟨p, hp, q, hq, hpq, -, hxe⟩ := hx
                    subst hxe
                    have h1 := hS p hp
                    have h2 := hS q hq
                    rw [← hpq] at h2
                    exact Der.conjIntro h1 h2
                | inr hx =>
                    rw [mem_genRtoS] at hx
                    obtain ⟨e, he, -, hxe⟩ := hx
                    subst hxe
                    exact hR e he
  · intro x hx
    rw [crStep_snd, mem_dedupN, List.mem_append] at hx
    cases hx with
    | inl hx => exact hR x hx
    | inr hx =>
        rw [List.mem_append] at hx
        cases hx with
        | inl hx =>
            rw [mem_genStoR] at hx
            obtain ⟨p, hp, r, d, hp2, hxe⟩ := hx
            subst hxe
            have hpd := hS p hp
            rw [hp2] at hpd
            exact hpd
        | inr hx =>
            rw [List.mem_append] at hx
            cases hx with
            | inl hx =>
                rw [mem_genRmono] at hx
                obtain ⟨e, he, p, hp, hep, -, hxe⟩ := hx
                subst hxe
                have hE := hR e he
                have hP := hS p hp
                rw [← hep] at hP
                exact Der.trans hE (Der.exMono hP)
            | inr hx =>
                rw [List.mem_append] at hx
                cases hx with
                | inl hx =>
                    rw [mem_genRroleSub] at hx
                    obtain ⟨e, he, s, -, hrole, -, hxe⟩ := hx
                    subst hxe
                    exact roleDer_der_ex (roleSubB_sound hrole) (hR e he)
                | inr hx =>
                    rw [mem_genRcomp] at hx
                    obtain ⟨e, he, f, hf, hef, r, s, u, hcomp, hr, hs, -, hxe⟩ := hx
                    subst hxe
                    have hE := hR e he
                    have hF := hR f hf
                    rw [← hef] at hF
                    have hM : Der t (.ex e.1 e.2.2) (.ex e.1 (.ex f.1 f.2.2)) :=
                      Der.exMono hF
                    have hT : Der t e.2.1 (.ex e.1 (.ex f.1 f.2.2)) := Der.trans hE hM
                    rw [hr, hs] at hcomp
                    exact Der.exComp hT hcomp

-- §3.5 The closure and its justification

/-- Iterate `crStep`. -/
def crIter (t : List (AxiomP (Fin n) (Fin m))) :
    Nat → SRel n m × RRel n m → SRel n m × RRel n m
  | 0, SR => SR
  | k + 1, SR => crIter t k (crStep t SR.1 SR.2)

/-- The number of saturation rounds the engine performs. Soundness holds
    for any count; six rounds suffice for every derived fact used in this
    development (the SNOMED instance reaches its targets in four). -/
def satRounds : Nat := 6

/-- The role-aware closure: `satRounds` saturation rounds over the seed. -/
def closeSat (t : List (AxiomP (Fin n) (Fin m))) : SRel n m × RRel n m :=
  crIter t satRounds (seedS t, [])

/-- The justification invariant is preserved by any number of rounds. -/
theorem crIter_justified {t : List (AxiomP (Fin n) (Fin m))} (k : Nat)
    {S : SRel n m} {R : RRel n m} (h : SatJustified t S R) :
    SatJustified t (crIter t k (S, R)).1 (crIter t k (S, R)).2 := by
  induction k generalizing S R with
  | zero => exact h
  | succ k ih => exact ih (crStep_justified h)

/-- The full closure is justified. -/
theorem closeSat_justified {t : List (AxiomP (Fin n) (Fin m))} :
    SatJustified t (closeSat t).1 (closeSat t).2 :=
  crIter_justified _ seedS_justified

-- §3.6 The boolean queries and their soundness

/-- Boolean subsumption over the role-aware closure. -/
def subBPlus (t : List (AxiomP (Fin n) (Fin m))) (C D : Concept (Fin n) (Fin m)) :
    Bool :=
  decide ((C, D) ∈ (closeSat t).1)

/-- **Closure soundness**: a computed subsumption is `Der`-derivable. -/
theorem subBPlus_sound {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)}
    (h : subBPlus t C D = true) : Der t C D := by
  obtain ⟨hS, -⟩ := closeSat_justified (t := t)
  exact hS (C, D) (of_decide_eq_true h)

/-- Boolean conflict oracle over the role-aware closure: some computed
    subsumer of `C` and some computed subsumer of `C'` form a stated
    disjointness (in either orientation). -/
def conflictBPlus (t : List (AxiomP (Fin n) (Fin m))) (C C' : Concept (Fin n) (Fin m)) :
    Bool :=
  let S := (closeSat t).1
  S.any fun p => S.any fun q =>
    decide (p.1 = C ∧ q.1 = C' ∧
      (AxiomP.disj p.2 q.2 ∈ t ∨ AxiomP.disj q.2 p.2 ∈ t))

/-- **Oracle soundness, role-aware**: a computed conflict is a genuine
    derived conflict — the licence for repair operators to drop either
    side, now computed over the full EL⁺ closure rather than the atom-only
    projection. -/
theorem conflictBPlus_sound {t : List (AxiomP (Fin n) (Fin m))}
    {C C' : Concept (Fin n) (Fin m)}
    (h : conflictBPlus t C C' = true) : DerivedConflictP t C C' := by
  have h' : ((closeSat t).1.any fun p => (closeSat t).1.any fun q =>
      decide (p.1 = C ∧ q.1 = C' ∧
        (AxiomP.disj p.2 q.2 ∈ t ∨ AxiomP.disj q.2 p.2 ∈ t))) = true := h
  rw [List.any_eq_true] at h'
  obtain ⟨p, hpS, h'⟩ := h'
  rw [List.any_eq_true] at h'
  obtain ⟨q, hqS, h'⟩ := h'
  obtain ⟨hpC, hqC', hD⟩ := of_decide_eq_true h'
  obtain ⟨hS, -⟩ := closeSat_justified (t := t)
  have h1 : Der t C p.2 := by
    have hp := hS p hpS
    rw [hpC] at hp
    exact hp
  have h2 : Der t C' q.2 := by
    have hq := hS q hqS
    rw [hqC'] at hq
    exact hq
  exact ⟨p.2, q.2, h1, h2, hD⟩

end Saturation

-- ---------------------------------------------------------------------------
-- §4. The SNOMED instance, re-checked through the role-aware closure
-- ---------------------------------------------------------------------------

section SnomedVerified

/-- The role hierarchy is computed: DirectSite ⊑* RoleGroup. -/
theorem check_roleSubB_1_0 : roleSubB snomedTBox 1 0 = true := by native_decide

/-- The computed role derivation, re-derived through `roleSubB_sound`. -/
theorem cv_roleder_1_0 : RoleDer snomedTBox 1 0 :=
  roleSubB_sound check_roleSubB_1_0

/-- Boolean role-aware closure: `Pneumonia ⊑* ∃RoleGroup.Organ` is computed
    — the fact the atom-only projection of `OntologyELPlus.lean` could not
    see, since it needs the role hierarchy (or the composition axiom). -/
theorem check_subBPlus_4_rg3 :
    subBPlus snomedTBox (.atom 4) (.ex 0 (.atom 3)) = true := by native_decide

/-- The computed `Pneumonia ⊑* ∃RoleGroup.Organ`, re-derived through
    `subBPlus_sound`: the saturation engine agrees with the
    constructor-built `der_pneumonia_rg_organ_via_comp`. -/
theorem cv_der_rg_organ_via_comp :
    Der snomedTBox (.atom 4) (.ex 0 (.atom 3)) :=
  subBPlus_sound check_subBPlus_4_rg3

/-- Boolean oracle over the full closure: the pneumonia–drug conflict is
    computed. -/
theorem check_conflictBPlus_4_5 :
    conflictBPlus snomedTBox (.atom 4) (.atom 5) = true := by native_decide

/-- The computed pneumonia–drug conflict, re-derived through
    `conflictBPlus_sound`. -/
theorem cv_conflictP_4_5 : DerivedConflictP snomedTBox (.atom 4) (.atom 5) :=
  conflictBPlus_sound check_conflictBPlus_4_5

/-- The incoherent drug-induced disorder self-conflicts, computed over the
    full closure. -/
theorem check_conflictBPlus_7_7 :
    conflictBPlus snomedTBox (.atom 7) (.atom 7) = true := by native_decide

/-- The computed self-conflict, re-derived. -/
theorem cv_conflictP_7_7 : DerivedConflictP snomedTBox (.atom 7) (.atom 7) :=
  conflictBPlus_sound check_conflictBPlus_7_7

end SnomedVerified

end Sounio.OntologyELPlus

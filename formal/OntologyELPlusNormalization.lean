/-
# Sounio.OntologyELPlusNormalization — all-concepts completeness via fresh-name flattening

Round 9 of the *ontology frontiers* research line. Round 8
(`OntologyELPlusClosureComplete.lean`) proved the role-aware closure complete
**over the saturation universe**: `subBPlusC t C D = true ↔ Der t C D` for
`C, D ∈ conceptUniv t`. The open question documented in
`docs/research/ontology_elplus_complete_universe_open_question_2026-08-03.md`
is how to lift this to *arbitrary* concepts. The classical route (in the
spirit of Baader–Brandt–Lutz, "Pushing the EL Envelope") is **query-side
normalization**: replace the query concepts by fresh names, add
structure-sharing definitions to the TBox, and saturate. (This is a
definitional extension of the query, not the full BBL TBox normal-form
decomposition.) This file formalizes exactly that reduction:

1. §1 Renaming: `renameConcept` / `renameAxiom` embed a TBox into a larger
   class-id space; `der_rename` and `der_mono` show `Der` is preserved.
2. §2 The flat list: `flatList C` collects the *named* subconcepts of `C`
   (⊤, ⊓-compounds, ∃-restrictions — everything but atoms) in post-order;
   `mem_flatList_iff` characterizes membership exactly.
3. §3 Fresh names and definitional axioms: every named subconcept `E` of `C`
   gets the fresh class id `freshFin C E` (its position in `flatList C`,
   offset above the old class ids), and `axiomsOf` emits the EL⁺
   definition of `E` — `A_E ⊑ A_{E₁} ⊓ A_{E₂}`, `A_{E₁} ⊓ A_{E₂} ⊑ A_E`,
   `A_E ⊑ ∃r.A_{E'}`, `∃r.A_{E'} ⊑ A_E`, and `⊤ ⊑ A_⊤` — using
   `repConcept` to represent children (atoms keep their embedded name).
   `normConcept C = (normAxioms C, normName C)` is the fresh-name flattening.
   *Design note:* the open-question document sketches the signature
   `normConcept : Concept α ρ → List (AxiomP α ρ) × α`; since genuinely fresh
   names cannot live in the same `α`, the name component lands in the
   extended class-id type `Fin (ν + (flatList C).length)`.
4. §4 The collapse: `collapse ν tbl` maps every concept of the extension
   back to the original signature, sending each fresh name to the concept it
   defines. It is a retraction of the embedding (`collapse_rename`) and of
   every representative (`collapse_repConcept`), and every normalization
   axiom collapses to a reflexive pair (`normAxioms_collapse_eq`). The
   conservativity engine `der_collapse` then drops a whole derivation over
   the extension to a derivation over the original TBox — the syntactic
   form of *conservativity of definitional extensions*.
5. §5 The representatives are provably equivalent to the concepts they name
   (`repConcept_der_rename`), hence so are the fresh names
   (`normName_der_rename`).
6. §6 Composition: for the query `C ⊑? D` over `t`, the two-stage extension
   `normTBox t C D` (normalize `C`, then the lifted `D`) satisfies
   `der_normTBox_iff` —
   `Der (normTBox t C D) (atom A_C) (atom A_D) ↔ Der t C D` — and, since
   atoms always lie in the universe, composing with round 8's
   `subBPlusC_iff` gives **all-concepts completeness**:
   `subBPlusC (normTBox t C D) (atom A_C) (atom A_D) = true ↔ Der t C D`
   for ALL `C, D`, no universe membership side condition.

Self-contained modulo `OntologyELPlusClosureComplete`. No Mathlib. Zero
sorry. No new axioms.
-/

import OntologyELPlusClosureComplete

namespace Sounio.OntologyELPlus

open Sounio.OntologyELReasoner

-- ---------------------------------------------------------------------------
-- §1. Renaming class ids, and its interaction with `Der`
-- ---------------------------------------------------------------------------

section Rename

variable {α β ρ : Type}

/-- Rename the class ids of a concept along `e`. -/
def renameConcept (e : α → β) : Concept α ρ → Concept β ρ
  | .atom a => .atom (e a)
  | .top => .top
  | .conj c d => .conj (renameConcept e c) (renameConcept e d)
  | .ex r c => .ex r (renameConcept e c)

/-- Rename the class ids of an axiom along `e`; role axioms are untouched. -/
def renameAxiom (e : α → β) : AxiomP α ρ → AxiomP β ρ
  | .sub c d => .sub (renameConcept e c) (renameConcept e d)
  | .disj c d => .disj (renameConcept e c) (renameConcept e d)
  | .roleSub r s => .roleSub r s
  | .roleComp r s u => .roleComp r s u

/-- `Der` is preserved by renaming. -/
theorem der_rename {t : List (AxiomP α ρ)} {c d : Concept α ρ} (e : α → β)
    (h : Der t c d) :
    Der (t.map (renameAxiom e)) (renameConcept e c) (renameConcept e d) := by
  induction h with
  | ofAxiom hmem => exact Der.ofAxiom (List.mem_map.mpr ⟨.sub _ _, hmem, rfl⟩)
  | refl => exact Der.refl
  | trans _ _ ih1 ih2 => exact Der.trans ih1 ih2
  | conjIntro _ _ ih1 ih2 => exact Der.conjIntro ih1 ih2
  | conjElimLeft _ ih => exact Der.conjElimLeft ih
  | conjElimRight _ ih => exact Der.conjElimRight ih
  | exMono _ ih => exact Der.exMono ih
  | exRoleSub _ hmem ih =>
      exact Der.exRoleSub ih (List.mem_map.mpr ⟨.roleSub _ _, hmem, rfl⟩)
  | exComp _ hmem ih =>
      exact Der.exComp ih (List.mem_map.mpr ⟨.roleComp _ _ _, hmem, rfl⟩)
  | topRule => exact Der.topRule

/-- `Der` is monotone in the TBox. -/
theorem der_mono {t t' : List (AxiomP α ρ)} {c d : Concept α ρ}
    (hsub : ∀ a ∈ t, a ∈ t') (h : Der t c d) : Der t' c d := by
  induction h with
  | ofAxiom hmem => exact Der.ofAxiom (hsub _ hmem)
  | refl => exact Der.refl
  | trans _ _ ih1 ih2 => exact Der.trans ih1 ih2
  | conjIntro _ _ ih1 ih2 => exact Der.conjIntro ih1 ih2
  | conjElimLeft _ ih => exact Der.conjElimLeft ih
  | conjElimRight _ ih => exact Der.conjElimRight ih
  | exMono _ ih => exact Der.exMono ih
  | exRoleSub _ hmem ih => exact Der.exRoleSub ih (hsub _ hmem)
  | exComp _ hmem ih => exact Der.exComp ih (hsub _ hmem)
  | topRule => exact Der.topRule

end Rename

-- ---------------------------------------------------------------------------
-- §2. The flat list of named subconcepts
-- ---------------------------------------------------------------------------

section FlatList

variable {ν m : Nat}

/-- A concept needs a fresh name iff it is not an atom. -/
def isNamed : Concept (Fin ν) (Fin m) → Bool
  | .atom _ => false
  | .top => true
  | .conj _ _ => true
  | .ex _ _ => true

/-- The named subconcepts of a concept, in post-order (children before
    parents, no deduplication — duplicate names are harmless). -/
def flatList : Concept (Fin ν) (Fin m) → List (Concept (Fin ν) (Fin m))
  | .atom _ => []
  | .top => [.top]
  | .conj c d => flatList c ++ flatList d ++ [.conj c d]
  | .ex r c => flatList c ++ [.ex r c]

/-- **Membership characterization**: the flat list contains exactly the named
    subconcepts. -/
theorem mem_flatList_iff {C E : Concept (Fin ν) (Fin m)} :
    E ∈ flatList C ↔ isNamed E = true ∧ E ∈ subconcepts C := by
  induction C with
  | atom a =>
      constructor
      · intro h
        exact absurd h List.not_mem_nil
      · intro ⟨hn, hsub⟩
        rw [show subconcepts (@Concept.atom (Fin ν) (Fin m) a) = [.atom a] from rfl] at hsub
        have he : E = .atom a := List.mem_singleton.mp hsub
        subst he
        simp [isNamed] at hn
  | top =>
      constructor
      · intro h
        rw [show flatList (@Concept.top (Fin ν) (Fin m)) = [.top] from rfl] at h
        have he : E = .top := List.mem_singleton.mp h
        subst he
        exact ⟨rfl, mem_subconcepts_self _⟩
      · intro ⟨_, hsub⟩
        rw [show subconcepts (@Concept.top (Fin ν) (Fin m)) = [.top] from rfl] at hsub
        have he : E = .top := List.mem_singleton.mp hsub
        subst he
        rw [show flatList (@Concept.top (Fin ν) (Fin m)) = [.top] from rfl]
        exact List.mem_singleton_self _
  | conj c d ihc ihd =>
      rw [show flatList (.conj c d) = flatList c ++ flatList d ++ [.conj c d] from rfl,
        show subconcepts (.conj c d) = .conj c d :: (subconcepts c ++ subconcepts d) from rfl]
      constructor
      · intro h
        rw [List.mem_append, List.mem_append] at h
        cases h with
        | inl h =>
            cases h with
            | inl hc =>
                obtain ⟨hn, hsub⟩ := ihc.mp hc
                exact ⟨hn, List.mem_cons_of_mem _ (List.mem_append_left _ hsub)⟩
            | inr hd =>
                obtain ⟨hn, hsub⟩ := ihd.mp hd
                exact ⟨hn, List.mem_cons_of_mem _ (List.mem_append_right _ hsub)⟩
        | inr h =>
            have he : E = .conj c d := List.mem_singleton.mp h
            subst he
            exact ⟨rfl, List.mem_cons_self⟩
      · intro ⟨hn, hsub⟩
        rw [List.mem_cons] at hsub
        cases hsub with
        | inl he =>
            subst he
            exact List.mem_append_right _ (List.mem_singleton_self _)
        | inr hsub =>
            rw [List.mem_append] at hsub
            cases hsub with
            | inl hc =>
                exact List.mem_append_left _ (List.mem_append_left _ (ihc.mpr ⟨hn, hc⟩))
            | inr hd =>
                exact List.mem_append_left _ (List.mem_append_right _ (ihd.mpr ⟨hn, hd⟩))
  | ex r c ih =>
      rw [show flatList (.ex r c) = flatList c ++ [.ex r c] from rfl,
        show subconcepts (.ex r c) = .ex r c :: subconcepts c from rfl]
      constructor
      · intro h
        rw [List.mem_append] at h
        cases h with
        | inl hc =>
            obtain ⟨hn, hsub⟩ := ih.mp hc
            exact ⟨hn, List.mem_cons_of_mem _ hsub⟩
        | inr h =>
            have he : E = .ex r c := List.mem_singleton.mp h
            subst he
            exact ⟨rfl, List.mem_cons_self⟩
      · intro ⟨hn, hsub⟩
        rw [List.mem_cons] at hsub
        cases hsub with
        | inl he =>
            subst he
            exact List.mem_append_right _ (List.mem_singleton_self _)
        | inr hsub =>
            exact List.mem_append_left _ (ih.mpr ⟨hn, hsub⟩)

/-- A named concept is a member of its own flat list. -/
theorem self_mem_flatList {C : Concept (Fin ν) (Fin m)} (hC : isNamed C = true) :
    C ∈ flatList C :=
  mem_flatList_iff.mpr ⟨hC, mem_subconcepts_self C⟩

end FlatList

-- ---------------------------------------------------------------------------
-- §3. Fresh names, representatives, and the definitional axioms
-- ---------------------------------------------------------------------------

section NormAxioms

variable {ν m : Nat}

/-- The fresh class id of a named subconcept: its position in the flat list,
    offset above the `ν` original class ids. -/
def freshFin (C E : Concept (Fin ν) (Fin m)) (h : E ∈ flatList C) :
    Fin (ν + (flatList C).length) :=
  ⟨ν + (flatList C).idxOf E, by
    have h' := List.idxOf_lt_length_of_mem h
    omega⟩

/-- The representative of a subconcept `E` of `C` in the definitional
    extension: atoms keep their (embedded) name; every named subconcept gets
    its fresh name. The `else` branch is dead code — it is only reached when
    `E` is named but not a subconcept of `C`, which never happens at the use
    sites. -/
def repConcept (C : Concept (Fin ν) (Fin m)) (E : Concept (Fin ν) (Fin m)) :
    Concept (Fin (ν + (flatList C).length)) (Fin m) :=
  match E with
  | .atom a => .atom (Fin.castAdd _ a)
  | .top => if h : .top ∈ flatList C then .atom (freshFin C .top h) else .top
  | .conj c d =>
      if h : .conj c d ∈ flatList C then .atom (freshFin C (.conj c d) h) else .top
  | .ex r c =>
      if h : .ex r c ∈ flatList C then .atom (freshFin C (.ex r c) h) else .top

/-- Reduction of `repConcept` on named subconcepts. -/
theorem repConcept_of_named {C E : Concept (Fin ν) (Fin m)} (hn : isNamed E = true)
    (hfl : E ∈ flatList C) : repConcept C E = .atom (freshFin C E hfl) := by
  cases E with
  | atom a => simp [isNamed] at hn
  | top =>
      show (if h : Concept.top ∈ flatList C then Concept.atom (freshFin C Concept.top h)
          else Concept.top) = _
      rw [dif_pos hfl]
  | conj c d =>
      show (if h : .conj c d ∈ flatList C then Concept.atom (freshFin C (.conj c d) h)
          else Concept.top) = _
      rw [dif_pos hfl]
  | ex r c =>
      show (if h : .ex r c ∈ flatList C then Concept.atom (freshFin C (.ex r c) h)
          else Concept.top) = _
      rw [dif_pos hfl]

/-- The definitional axioms of one named subconcept: the EL⁺ definition of
    its fresh name in terms of the representatives of its children. ⊤ gets a
    one-sided definition (`A_⊤ ⊑ ⊤` is free by `Der.topRule`). -/
def axiomsOf (C : Concept (Fin ν) (Fin m)) :
    (E : Concept (Fin ν) (Fin m)) → E ∈ flatList C →
      List (AxiomP (Fin (ν + (flatList C).length)) (Fin m))
  | .atom _, _ => []
  | .top, h => [.sub .top (.atom (freshFin C .top h))]
  | .conj c d, h =>
      [ .sub (.atom (freshFin C (.conj c d) h)) (.conj (repConcept C c) (repConcept C d))
      , .sub (.conj (repConcept C c) (repConcept C d)) (.atom (freshFin C (.conj c d) h)) ]
  | .ex r c, h =>
      [ .sub (.atom (freshFin C (.ex r c) h)) (.ex r (repConcept C c))
      , .sub (.ex r (repConcept C c)) (.atom (freshFin C (.ex r c) h)) ]

/-- The definitional extension axioms of `C`: one definition per named
    subconcept. -/
def normAxioms (C : Concept (Fin ν) (Fin m)) :
    List (AxiomP (Fin (ν + (flatList C).length)) (Fin m)) :=
  (flatList C).attach.flatMap fun ⟨E, hE⟩ => axiomsOf C E hE

/-- The fresh class id of `C` itself (atoms keep their embedded name). -/
def normName : (C : Concept (Fin ν) (Fin m)) → Fin (ν + (flatList C).length)
  | .atom a => Fin.castAdd _ a
  | .top => freshFin .top .top (self_mem_flatList rfl)
  | .conj c d => freshFin (.conj c d) (.conj c d) (self_mem_flatList rfl)
  | .ex r c => freshFin (.ex r c) (.ex r c) (self_mem_flatList rfl)

/-- **Fresh-name flattening**: the definitional axioms of `C` together with
    the fresh name standing for `C`. -/
def normConcept (C : Concept (Fin ν) (Fin m)) :
    List (AxiomP (Fin (ν + (flatList C).length)) (Fin m)) ×
      Fin (ν + (flatList C).length) :=
  (normAxioms C, normName C)

/-- The representative of `C` itself is its fresh name. -/
theorem repConcept_self {C : Concept (Fin ν) (Fin m)} :
    repConcept C C = .atom (normName C) := by
  cases C with
  | atom a => rfl
  | top => exact repConcept_of_named rfl (self_mem_flatList rfl)
  | conj c d => exact repConcept_of_named rfl (self_mem_flatList rfl)
  | ex r c => exact repConcept_of_named rfl (self_mem_flatList rfl)

-- §3.1 Membership facts for the generated axioms

/-- Every axiom generated for a flat-list member lands in `normAxioms`. -/
theorem mem_normAxioms {C E : Concept (Fin ν) (Fin m)} (hE : E ∈ flatList C)
    {x : AxiomP (Fin (ν + (flatList C).length)) (Fin m)}
    (hx : x ∈ axiomsOf C E hE) : x ∈ normAxioms C :=
  List.mem_flatMap.mpr ⟨⟨E, hE⟩, List.mem_attach _ _, hx⟩

theorem topAx_mem {C : Concept (Fin ν) (Fin m)} (h : Concept.top ∈ flatList C) :
    AxiomP.sub .top (.atom (freshFin C .top h)) ∈ normAxioms C :=
  mem_normAxioms h (List.mem_singleton_self _)

theorem conjAx1_mem {C c d : Concept (Fin ν) (Fin m)}
    (h : Concept.conj c d ∈ flatList C) :
    AxiomP.sub (.atom (freshFin C (.conj c d) h))
      (.conj (repConcept C c) (repConcept C d)) ∈ normAxioms C :=
  mem_normAxioms h List.mem_cons_self

theorem conjAx2_mem {C c d : Concept (Fin ν) (Fin m)}
    (h : Concept.conj c d ∈ flatList C) :
    AxiomP.sub (.conj (repConcept C c) (repConcept C d))
      (.atom (freshFin C (.conj c d) h)) ∈ normAxioms C :=
  mem_normAxioms h (List.mem_cons_of_mem _ (List.mem_singleton_self _))

theorem exAx1_mem {C c : Concept (Fin ν) (Fin m)} {r : Fin m}
    (h : Concept.ex r c ∈ flatList C) :
    AxiomP.sub (.atom (freshFin C (.ex r c) h)) (.ex r (repConcept C c)) ∈ normAxioms C :=
  mem_normAxioms h List.mem_cons_self

theorem exAx2_mem {C c : Concept (Fin ν) (Fin m)} {r : Fin m}
    (h : Concept.ex r c ∈ flatList C) :
    AxiomP.sub (.ex r (repConcept C c)) (.atom (freshFin C (.ex r c) h)) ∈ normAxioms C :=
  mem_normAxioms h (List.mem_cons_of_mem _ (List.mem_singleton_self _))

-- §3.2 The generated axioms are concept inclusions only

/-- Normalization generates no role inclusions. -/
theorem normAxioms_no_roleSub {C : Concept (Fin ν) (Fin m)} {r s : Fin m} :
    AxiomP.roleSub r s ∉ normAxioms C := by
  intro h
  obtain ⟨⟨E, hE⟩, -, hx⟩ := List.mem_flatMap.mp h
  cases E with
  | atom a => exact absurd hx List.not_mem_nil
  | top =>
      have hx' : AxiomP.roleSub r s = .sub .top (.atom (freshFin C .top hE)) :=
        List.mem_singleton.mp hx
      nomatch hx'
  | conj c d =>
      have hx' : AxiomP.roleSub r s ∈
          [ .sub (.atom (freshFin C (.conj c d) hE)) (.conj (repConcept C c) (repConcept C d))
          , .sub (.conj (repConcept C c) (repConcept C d))
              (.atom (freshFin C (.conj c d) hE)) ] := hx
      rw [List.mem_cons, List.mem_singleton] at hx'
      cases hx' with
      | inl h'' => nomatch h''
      | inr h'' => nomatch h''
  | ex r' c =>
      have hx' : AxiomP.roleSub r s ∈
          [ .sub (.atom (freshFin C (.ex r' c) hE)) (.ex r' (repConcept C c))
          , .sub (.ex r' (repConcept C c)) (.atom (freshFin C (.ex r' c) hE)) ] := hx
      rw [List.mem_cons, List.mem_singleton] at hx'
      cases hx' with
      | inl h'' => nomatch h''
      | inr h'' => nomatch h''

/-- Normalization generates no role compositions. -/
theorem normAxioms_no_roleComp {C : Concept (Fin ν) (Fin m)} {r s u : Fin m} :
    AxiomP.roleComp r s u ∉ normAxioms C := by
  intro h
  obtain ⟨⟨E, hE⟩, -, hx⟩ := List.mem_flatMap.mp h
  cases E with
  | atom a => exact absurd hx List.not_mem_nil
  | top =>
      have hx' : AxiomP.roleComp r s u = .sub .top (.atom (freshFin C .top hE)) :=
        List.mem_singleton.mp hx
      nomatch hx'
  | conj c d =>
      have hx' : AxiomP.roleComp r s u ∈
          [ .sub (.atom (freshFin C (.conj c d) hE)) (.conj (repConcept C c) (repConcept C d))
          , .sub (.conj (repConcept C c) (repConcept C d))
              (.atom (freshFin C (.conj c d) hE)) ] := hx
      rw [List.mem_cons, List.mem_singleton] at hx'
      cases hx' with
      | inl h'' => nomatch h''
      | inr h'' => nomatch h''
  | ex r' c =>
      have hx' : AxiomP.roleComp r s u ∈
          [ .sub (.atom (freshFin C (.ex r' c) hE)) (.ex r' (repConcept C c))
          , .sub (.ex r' (repConcept C c)) (.atom (freshFin C (.ex r' c) hE)) ] := hx
      rw [List.mem_cons, List.mem_singleton] at hx'
      cases hx' with
      | inl h'' => nomatch h''
      | inr h'' => nomatch h''

end NormAxioms

-- ---------------------------------------------------------------------------
-- §4. The collapse: back to the original signature, and conservativity
-- ---------------------------------------------------------------------------

section Collapse

variable {ν m : Nat}

/-- List lookup is congruent: indices with equal values give equal results.
    (Used to reroute `Fin` indices whose embedded proofs block `rw`.) -/
theorem get_congr' {α : Type} {l : List α} {i : Fin l.length} {j : Nat}
    (hj : j < l.length) (h : i.val = j) : l.get i = l.get ⟨j, hj⟩ :=
  congrArg l.get (Fin.ext h)

/-- Lookup: the element at its own `idxOf` position is itself. -/
theorem get_idxOf {α : Type} [DecidableEq α] {l : List α} {x : α} (h : x ∈ l) :
    l.get ⟨l.idxOf x, List.idxOf_lt_length_of_mem h⟩ = x := by
  induction l with
  | nil => exact absurd h List.not_mem_nil
  | cons y ys ih =>
      cases List.mem_cons.mp h with
      | inl hxy =>
          subst hxy
          exact Eq.trans
            (get_congr' (Nat.zero_lt_succ _) List.idxOf_cons_self) rfl
      | inr hxs =>
          by_cases hyx : y = x
          · subst hyx
            exact Eq.trans
              (get_congr' (Nat.zero_lt_succ _) List.idxOf_cons_self) rfl
          · have hfalse : (y == x) = false := by
              rw [Bool.eq_false_iff]
              intro h'
              exact hyx (beq_iff_eq.mp h')
            have hidx : (y :: ys).idxOf x = ys.idxOf x + 1 := by
              rw [List.idxOf_cons, hfalse, cond_false]
            exact Eq.trans
              (get_congr' (Nat.succ_lt_succ (List.idxOf_lt_length_of_mem hxs)) hidx)
              (ih hxs)

/-- The collapse: map a concept of the extension back to the original
    signature. Original class ids keep their name; the fresh name `ν + i`
    collapses to the concept it defines, `tbl[i]`. -/
def collapse (ν : Nat) (tbl : List (Concept (Fin ν) (Fin m))) :
    Concept (Fin (ν + tbl.length)) (Fin m) → Concept (Fin ν) (Fin m)
  | .atom a =>
      if h : a.val < ν then .atom ⟨a.val, h⟩
      else tbl.get ⟨a.val - ν, by have := a.isLt; omega⟩
  | .top => .top
  | .conj c d => .conj (collapse ν tbl c) (collapse ν tbl d)
  | .ex r c => .ex r (collapse ν tbl c)

/-- Collapse of an original class id. -/
theorem collapse_atom_of_lt {tbl : List (Concept (Fin ν) (Fin m))}
    {a : Fin (ν + tbl.length)} (h : a.val < ν) :
    collapse ν tbl (.atom a) = .atom ⟨a.val, h⟩ := by
  show (if h' : a.val < ν then Concept.atom ⟨a.val, h'⟩ else
      tbl.get ⟨a.val - ν, by have := a.isLt; omega⟩) = .atom ⟨a.val, h⟩
  rw [dif_pos h]

/-- Collapse of a fresh class id. -/
theorem collapse_atom_of_ge {tbl : List (Concept (Fin ν) (Fin m))}
    {a : Fin (ν + tbl.length)} (h : ¬ a.val < ν) :
    collapse ν tbl (.atom a) = tbl.get ⟨a.val - ν, by have := a.isLt; omega⟩ := by
  show (if h' : a.val < ν then Concept.atom ⟨a.val, h'⟩ else
      tbl.get ⟨a.val - ν, by have := a.isLt; omega⟩) =
    tbl.get ⟨a.val - ν, by have := a.isLt; omega⟩
  rw [dif_neg h]

/-- Collapse of an embedded original name. -/
theorem collapse_atom_castAdd {tbl : List (Concept (Fin ν) (Fin m))} (a : Fin ν) :
    collapse ν tbl (.atom (Fin.castAdd tbl.length a)) = .atom a := by
  rw [collapse_atom_of_lt a.isLt]
  exact congrArg Concept.atom (Fin.ext rfl)

/-- **The collapse is a retraction of the embedding.** -/
theorem collapse_rename {tbl : List (Concept (Fin ν) (Fin m))}
    (C : Concept (Fin ν) (Fin m)) :
    collapse ν tbl (renameConcept (Fin.castAdd tbl.length) C) = C := by
  induction C with
  | atom a => exact collapse_atom_castAdd a
  | top => rfl
  | conj c d ihc ihd =>
      show Concept.conj (collapse ν tbl (renameConcept (Fin.castAdd tbl.length) c))
        (collapse ν tbl (renameConcept (Fin.castAdd tbl.length) d)) = .conj c d
      rw [ihc, ihd]
  | ex r c ih =>
      show Concept.ex r (collapse ν tbl (renameConcept (Fin.castAdd tbl.length) c)) = .ex r c
      rw [ih]

/-- The collapse of a fresh name is the concept it defines. -/
theorem collapse_freshFin {C E : Concept (Fin ν) (Fin m)} (h : E ∈ flatList C) :
    collapse ν (flatList C) (.atom (freshFin C E h)) = E := by
  have hlt : ¬ (freshFin C E h).val < ν := by
    have hidx := List.idxOf_lt_length_of_mem h
    show ¬ (ν + (flatList C).idxOf E < ν)
    omega
  rw [collapse_atom_of_ge hlt]
  exact Eq.trans
    (get_congr' (List.idxOf_lt_length_of_mem h) (by
      have hval : (freshFin C E h).val = ν + (flatList C).idxOf E := rfl
      show (freshFin C E h).val - ν = (flatList C).idxOf E
      omega))
    (get_idxOf h)

/-- The collapse of a representative is the represented concept. -/
theorem collapse_repConcept {C E : Concept (Fin ν) (Fin m)} (hE : E ∈ subconcepts C) :
    collapse ν (flatList C) (repConcept C E) = E := by
  cases E with
  | atom a =>
      show collapse ν (flatList C) (.atom (Fin.castAdd _ a)) = .atom a
      exact collapse_atom_castAdd a
  | top =>
      have hfl : Concept.top ∈ flatList C := mem_flatList_iff.mpr ⟨rfl, hE⟩
      rw [repConcept_of_named rfl hfl]
      exact collapse_freshFin hfl
  | conj c d =>
      have hfl : .conj c d ∈ flatList C := mem_flatList_iff.mpr ⟨rfl, hE⟩
      rw [repConcept_of_named rfl hfl]
      exact collapse_freshFin hfl
  | ex r c =>
      have hfl : .ex r c ∈ flatList C := mem_flatList_iff.mpr ⟨rfl, hE⟩
      rw [repConcept_of_named rfl hfl]
      exact collapse_freshFin hfl

/-- The collapse of the fresh name of `C` is `C`. -/
theorem collapse_normName {C : Concept (Fin ν) (Fin m)} :
    collapse ν (flatList C) (.atom (normName C)) = C := by
  cases C with
  | atom a => exact collapse_atom_castAdd a
  | top => exact collapse_freshFin (self_mem_flatList rfl)
  | conj c d => exact collapse_freshFin (self_mem_flatList rfl)
  | ex r c => exact collapse_freshFin (self_mem_flatList rfl)

/-- **Every normalization axiom collapses to a reflexive pair**: the
    definition of `A_E` collapses to `E ⊑ E`. -/
theorem normAxioms_collapse_eq {C : Concept (Fin ν) (Fin m)}
    {X Y : Concept (Fin (ν + (flatList C).length)) (Fin m)}
    (h : AxiomP.sub X Y ∈ normAxioms C) :
    collapse ν (flatList C) X = collapse ν (flatList C) Y := by
  obtain ⟨⟨E, hE⟩, -, hx⟩ := List.mem_flatMap.mp h
  cases E with
  | atom a => exact absurd hx List.not_mem_nil
  | top =>
      have hx' : AxiomP.sub X Y = .sub .top (.atom (freshFin C .top hE)) :=
        List.mem_singleton.mp hx
      injection hx' with hX hY
      subst hX; subst hY
      exact (collapse_freshFin hE).symm
  | conj c d =>
      have hsub : .conj c d ∈ subconcepts C := (mem_flatList_iff.mp hE).2
      have hc : c ∈ subconcepts C := mem_subconcepts_trans sub_conj_left hsub
      have hd : d ∈ subconcepts C := mem_subconcepts_trans sub_conj_right hsub
      have hx' : AxiomP.sub X Y ∈
          [ .sub (.atom (freshFin C (.conj c d) hE)) (.conj (repConcept C c) (repConcept C d))
          , .sub (.conj (repConcept C c) (repConcept C d))
              (.atom (freshFin C (.conj c d) hE)) ] := hx
      rw [List.mem_cons, List.mem_singleton] at hx'
      cases hx' with
      | inl h'' =>
          injection h'' with hX hY
          subst hX; subst hY
          rw [collapse_freshFin hE]
          show Concept.conj c d = .conj (collapse ν (flatList C) (repConcept C c))
            (collapse ν (flatList C) (repConcept C d))
          rw [collapse_repConcept hc, collapse_repConcept hd]
      | inr h'' =>
          injection h'' with hX hY
          subst hX; subst hY
          show Concept.conj (collapse ν (flatList C) (repConcept C c))
            (collapse ν (flatList C) (repConcept C d)) = _
          rw [collapse_repConcept hc, collapse_repConcept hd, collapse_freshFin hE]
  | ex r c =>
      have hsub : .ex r c ∈ subconcepts C := (mem_flatList_iff.mp hE).2
      have hc : c ∈ subconcepts C := mem_subconcepts_trans sub_ex_filler hsub
      have hx' : AxiomP.sub X Y ∈
          [ .sub (.atom (freshFin C (.ex r c) hE)) (.ex r (repConcept C c))
          , .sub (.ex r (repConcept C c)) (.atom (freshFin C (.ex r c) hE)) ] := hx
      rw [List.mem_cons, List.mem_singleton] at hx'
      cases hx' with
      | inl h'' =>
          injection h'' with hX hY
          subst hX; subst hY
          rw [collapse_freshFin hE]
          show Concept.ex r c = .ex r (collapse ν (flatList C) (repConcept C c))
          rw [collapse_repConcept hc]
      | inr h'' =>
          injection h'' with hX hY
          subst hX; subst hY
          show Concept.ex r (collapse ν (flatList C) (repConcept C c)) = _
          rw [collapse_repConcept hc, collapse_freshFin hE]

/-- The collapsed normalization axioms are derivable (by reflexivity) over
    any TBox of the original signature. -/
theorem normAxioms_der {s : List (AxiomP (Fin ν) (Fin m))}
    {C : Concept (Fin ν) (Fin m)}
    {X Y : Concept (Fin (ν + (flatList C).length)) (Fin m)}
    (h : AxiomP.sub X Y ∈ normAxioms C) :
    Der s (collapse ν (flatList C) X) (collapse ν (flatList C) Y) := by
  rw [normAxioms_collapse_eq h]
  exact Der.refl

/-- **Conservativity of definitional extensions (syntactic form)**: a
    derivation over the renamed TBox plus fresh definitions collapses to a
    derivation over the original TBox, provided the new axioms are
    concept inclusions whose collapses are derivable. -/
theorem der_collapse {t : List (AxiomP (Fin ν) (Fin m))}
    {tbl : List (Concept (Fin ν) (Fin m))}
    {newAx : List (AxiomP (Fin (ν + tbl.length)) (Fin m))}
    (hsub : ∀ {X Y}, AxiomP.sub X Y ∈ newAx →
      Der t (collapse ν tbl X) (collapse ν tbl Y))
    (hrole : ∀ {r s}, AxiomP.roleSub r s ∉ newAx)
    (hcomp : ∀ {r s u}, AxiomP.roleComp r s u ∉ newAx)
    {X Y : Concept (Fin (ν + tbl.length)) (Fin m)}
    (h : Der (t.map (renameAxiom (Fin.castAdd tbl.length)) ++ newAx) X Y) :
    Der t (collapse ν tbl X) (collapse ν tbl Y) := by
  induction h with
  | ofAxiom hmem =>
      cases List.mem_append.mp hmem with
      | inl hm =>
          obtain ⟨a, ha, haeq⟩ := List.mem_map.mp hm
          cases a with
          | sub c₀ d₀ =>
              injection haeq with h1 h2
              subst h1; subst h2
              rw [collapse_rename, collapse_rename]
              exact Der.ofAxiom ha
          | disj c₀ d₀ => nomatch haeq
          | roleSub r s => nomatch haeq
          | roleComp r s u => nomatch haeq
      | inr hm => exact hsub hm
  | refl => exact Der.refl
  | trans _ _ ih1 ih2 => exact Der.trans ih1 ih2
  | conjIntro _ _ ih1 ih2 => exact Der.conjIntro ih1 ih2
  | conjElimLeft _ ih => exact Der.conjElimLeft ih
  | conjElimRight _ ih => exact Der.conjElimRight ih
  | exMono _ ih => exact Der.exMono ih
  | exRoleSub _ hmem ih =>
      cases List.mem_append.mp hmem with
      | inl hm =>
          obtain ⟨a, ha, haeq⟩ := List.mem_map.mp hm
          cases a with
          | sub c₀ d₀ => nomatch haeq
          | disj c₀ d₀ => nomatch haeq
          | roleSub r s =>
              injection haeq with h1 h2
              subst h1; subst h2
              exact Der.exRoleSub ih ha
          | roleComp r s u => nomatch haeq
      | inr hm => exact absurd hm hrole
  | exComp _ hmem ih =>
      cases List.mem_append.mp hmem with
      | inl hm =>
          obtain ⟨a, ha, haeq⟩ := List.mem_map.mp hm
          cases a with
          | sub c₀ d₀ => nomatch haeq
          | disj c₀ d₀ => nomatch haeq
          | roleSub r s => nomatch haeq
          | roleComp r₀ s₀ u₀ =>
              injection haeq with h1 h2 h3
              subst h1; subst h2; subst h3
              exact Der.exComp ih ha
      | inr hm => exact absurd hm hcomp
  | topRule => exact Der.topRule

end Collapse

-- ---------------------------------------------------------------------------
-- §5. Representatives are provably equivalent to the concepts they name
-- ---------------------------------------------------------------------------

section NameSpec

variable {ν m : Nat}

/-- The definitional extension of a TBox `s` by the normalization of `C`. -/
abbrev normExt (s : List (AxiomP (Fin ν) (Fin m))) (C : Concept (Fin ν) (Fin m)) :
    List (AxiomP (Fin (ν + (flatList C).length)) (Fin m)) :=
  s.map (renameAxiom (Fin.castAdd (flatList C).length)) ++ normAxioms C

/-- **Name specification**: over the definitional extension, the
    representative of any subconcept `E` of `C` is provably equivalent
    (in both directions) to the renamed `E`. Structural induction on `E`;
    the compound cases unfold one level of the definition, transport the
    children through the induction hypotheses, and fold back. -/
theorem repConcept_der_rename (s : List (AxiomP (Fin ν) (Fin m)))
    (C : Concept (Fin ν) (Fin m)) :
    ∀ {E : Concept (Fin ν) (Fin m)}, E ∈ subconcepts C →
      Der (normExt s C) (repConcept C E) (renameConcept (Fin.castAdd _) E) ∧
      Der (normExt s C) (renameConcept (Fin.castAdd _) E) (repConcept C E) := by
  intro E
  induction E with
  | atom a => intro _; exact ⟨Der.refl, Der.refl⟩
  | top =>
      intro hE
      have hfl : Concept.top ∈ flatList C := mem_flatList_iff.mpr ⟨rfl, hE⟩
      rw [repConcept_of_named rfl hfl]
      exact ⟨Der.topRule,
        Der.ofAxiom (List.mem_append_right _ (topAx_mem hfl))⟩
  | conj c d ihc ihd =>
      intro hE
      have hc : c ∈ subconcepts C := mem_subconcepts_trans sub_conj_left hE
      have hd : d ∈ subconcepts C := mem_subconcepts_trans sub_conj_right hE
      obtain ⟨ih1, ih2⟩ := ihc hc
      obtain ⟨ih3, ih4⟩ := ihd hd
      have hfl : .conj c d ∈ flatList C := mem_flatList_iff.mpr ⟨rfl, hE⟩
      rw [repConcept_of_named rfl hfl]
      have ax1 : Der (normExt s C) (.atom (freshFin C (.conj c d) hfl))
          (.conj (repConcept C c) (repConcept C d)) :=
        Der.ofAxiom (List.mem_append_right _ (conjAx1_mem hfl))
      have ax2 : Der (normExt s C) (.conj (repConcept C c) (repConcept C d))
          (.atom (freshFin C (.conj c d) hfl)) :=
        Der.ofAxiom (List.mem_append_right _ (conjAx2_mem hfl))
      exact ⟨Der.conjIntro
               (Der.trans ax1 (Der.trans (Der.conjElimLeft Der.refl) ih1))
               (Der.trans ax1 (Der.trans (Der.conjElimRight Der.refl) ih3)),
             Der.trans
               (Der.conjIntro
                 (Der.trans (Der.conjElimLeft Der.refl) ih2)
                 (Der.trans (Der.conjElimRight Der.refl) ih4))
               ax2⟩
  | ex r c ih =>
      intro hE
      have hc : c ∈ subconcepts C := mem_subconcepts_trans sub_ex_filler hE
      obtain ⟨ih1, ih2⟩ := ih hc
      have hfl : .ex r c ∈ flatList C := mem_flatList_iff.mpr ⟨rfl, hE⟩
      rw [repConcept_of_named rfl hfl]
      have ax1 : Der (normExt s C) (.atom (freshFin C (.ex r c) hfl))
          (.ex r (repConcept C c)) :=
        Der.ofAxiom (List.mem_append_right _ (exAx1_mem hfl))
      have ax2 : Der (normExt s C) (.ex r (repConcept C c))
          (.atom (freshFin C (.ex r c) hfl)) :=
        Der.ofAxiom (List.mem_append_right _ (exAx2_mem hfl))
      exact ⟨Der.trans ax1 (Der.exMono ih1), Der.trans (Der.exMono ih2) ax2⟩

/-- **The fresh name of `C` is provably equivalent to the renamed `C`** over
    the definitional extension. -/
theorem normName_der_rename (s : List (AxiomP (Fin ν) (Fin m)))
    (C : Concept (Fin ν) (Fin m)) :
    Der (normExt s C) (.atom (normName C)) (renameConcept (Fin.castAdd _) C) ∧
    Der (normExt s C) (renameConcept (Fin.castAdd _) C) (.atom (normName C)) := by
  rw [← repConcept_self (C := C)]
  exact repConcept_der_rename s C (mem_subconcepts_self C)

end NameSpec

-- ---------------------------------------------------------------------------
-- §6. The two-stage extension and all-concepts completeness
-- ---------------------------------------------------------------------------

section AllConcepts

variable {n m : Nat}

/-- The query concept `D`, lifted into the first extension layer (the one
    that names the subconcepts of `C`). -/
abbrev liftD (C D : Concept (Fin n) (Fin m)) :
    Concept (Fin (n + (flatList C).length)) (Fin m) :=
  renameConcept (Fin.castAdd (flatList C).length) D

/-- **The fully normalized TBox for the query `C ⊑? D`**: the original TBox
    plus the definitional axioms of `C` and of the lifted `D`. -/
def normTBox (t : List (AxiomP (Fin n) (Fin m))) (C D : Concept (Fin n) (Fin m)) :
    List (AxiomP (Fin (n + (flatList C).length + (flatList (liftD C D)).length)) (Fin m)) :=
  normExt (normExt t C) (liftD C D)

/-- The fresh name of `C`, lifted into the second extension layer. -/
abbrev nameC (C D : Concept (Fin n) (Fin m)) :
    Fin (n + (flatList C).length + (flatList (liftD C D)).length) :=
  Fin.castAdd _ (normName C)

/-- The fresh name of (the lifted) `D`. -/
abbrev nameD (C D : Concept (Fin n) (Fin m)) :
    Fin (n + (flatList C).length + (flatList (liftD C D)).length) :=
  normName (liftD C D)

/-- Collapsing the lifted `D` through the first stage recovers `D`. -/
theorem collapse_liftD {C D : Concept (Fin n) (Fin m)} :
    collapse n (flatList C) (liftD C D) = D :=
  collapse_rename D

/-- Every class id of a signature lies in the saturation universe of every
    TBox over it. -/
theorem atom_mem_conceptUniv {ν k : Nat} {t : List (AxiomP (Fin ν) (Fin k))}
    (a : Fin ν) : Concept.atom a ∈ conceptUniv t :=
  List.mem_append_left _ (List.mem_append_left _
    (List.mem_cons_of_mem _ (List.mem_map.mpr ⟨a, List.mem_finRange a, rfl⟩)))

/-- **Conservativity of the two-stage normalization**: subsumption between
    the fresh names over the normalized TBox is exactly subsumption between
    the original concepts over the original TBox. The `→` direction is the
    syntactic conservativity of definitional extensions (two `der_collapse`
    steps); the `←` direction renames the derivation and bridges both ends
    through the name specifications. -/
theorem der_normTBox_iff {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)} :
    Der (normTBox t C D) (.atom (nameC C D)) (.atom (nameD C D)) ↔ Der t C D := by
  constructor
  · intro h
    have h2 := der_collapse (t := normExt t C) (tbl := flatList (liftD C D))
      (newAx := normAxioms (liftD C D))
      (hsub := normAxioms_der) (hrole := normAxioms_no_roleSub)
      (hcomp := normAxioms_no_roleComp) h
    rw [collapse_atom_castAdd, collapse_normName] at h2
    have h1 := der_collapse (t := t) (tbl := flatList C) (newAx := normAxioms C)
      (hsub := normAxioms_der) (hrole := normAxioms_no_roleSub)
      (hcomp := normAxioms_no_roleComp) h2
    rw [collapse_normName, collapse_liftD] at h1
    exact h1
  · intro h
    have h1 : Der (normExt t C)
        (renameConcept (Fin.castAdd _) C) (renameConcept (Fin.castAdd _) D) :=
      der_mono (fun a ha => List.mem_append_left _ ha) (der_rename _ h)
    have h2 : Der (normTBox t C D)
        (renameConcept (Fin.castAdd _) (renameConcept (Fin.castAdd _) C))
        (renameConcept (Fin.castAdd _) (renameConcept (Fin.castAdd _) D)) :=
      der_mono (fun a ha => List.mem_append_left _ ha) (der_rename _ h1)
    have hC' : Der (normTBox t C D) (.atom (nameC C D))
        (renameConcept (Fin.castAdd _) (renameConcept (Fin.castAdd _) C)) :=
      der_mono (fun a ha => List.mem_append_left _ ha)
        (der_rename _ (normName_der_rename t C).1)
    have hD' : Der (normTBox t C D)
        (renameConcept (Fin.castAdd _) (liftD C D)) (.atom (nameD C D)) :=
      (normName_der_rename (normExt t C) (liftD C D)).2
    exact Der.trans hC' (Der.trans h2 hD')

/-- **All-concepts completeness**: for ARBITRARY EL⁺ concepts `C, D` (no
    universe-membership side condition), the complete role-aware closure of
    the normalized TBox decides `Der t C D` exactly. Soundness is round 7/8's
    invariant; completeness is round 8's canonical model — atoms always lie
    in the universe — composed with the conservativity of normalization. -/
theorem subBPlusC_normTBox_iff {t : List (AxiomP (Fin n) (Fin m))}
    {C D : Concept (Fin n) (Fin m)} :
    subBPlusC (normTBox t C D) (.atom (nameC C D)) (.atom (nameD C D)) = true ↔
      Der t C D :=
  ⟨fun h => der_normTBox_iff.mp (subBPlusC_sound h),
   fun h => subBPlusC_complete (der_normTBox_iff.mpr h)
     (atom_mem_conceptUniv _) (atom_mem_conceptUniv _)⟩

end AllConcepts

-- ---------------------------------------------------------------------------
-- §7. Instance checks
-- ---------------------------------------------------------------------------

section InstanceChecks

/-- ⊓-query through normalization: `A ⊓ B ⊑* B` is computed by the complete
    engine over the normalized TBox — a subsumption whose left endpoint is
    not a universe member of `microTBox`. -/
theorem check_norm_conj :
    subBPlusC (normTBox microTBox (.conj (.atom 0) (.atom 1)) (.atom 1))
      (.atom (nameC (.conj (.atom 0) (.atom 1)) (.atom 1)))
      (.atom (nameD (.conj (.atom 0) (.atom 1)) (.atom 1))) = true :=
  subBPlusC_normTBox_iff.mpr (Der.conjElimRight Der.refl)

/-- ∃-query through normalization: `∃r.A ⊑* ∃r.B` from the stated `A ⊑ B` —
    the engine decides it after flattening the existential endpoints. -/
theorem check_norm_ex :
    subBPlusC (normTBox microTBox (.ex 0 (.atom 0)) (.ex 0 (.atom 1)))
      (.atom (nameC (.ex 0 (.atom 0)) (.ex 0 (.atom 1))))
      (.atom (nameD (.ex 0 (.atom 0)) (.ex 0 (.atom 1)))) = true :=
  subBPlusC_normTBox_iff.mpr (Der.exMono (Der.ofAxiom (by decide)))

/-- Round trip: the computed normalized answer re-derives into `Der` over
    the original TBox through soundness + conservativity. -/
theorem check_norm_ex_der : Der microTBox (.ex 0 (.atom 0)) (.ex 0 (.atom 1)) :=
  der_normTBox_iff.mp (subBPlusC_sound check_norm_ex)

/-- Direct computation: the engine really answers `true` on a concrete
    normalized TBox (the derivation route above is what certifies it; this
    is the same answer obtained by evaluation). -/
theorem check_norm_compute :
    subBPlusC
      (normTBox ([] : List (AxiomP (Fin 1) (Fin 1)))
        (.conj (.atom (0 : Fin 1)) (.atom (0 : Fin 1))) (.atom (0 : Fin 1)))
      (.atom (nameC (.conj (.atom (0 : Fin 1)) (.atom (0 : Fin 1))) (.atom (0 : Fin 1))))
      (.atom (nameD (.conj (.atom (0 : Fin 1)) (.atom (0 : Fin 1))) (.atom (0 : Fin 1)))) =
        true := by
  native_decide

end InstanceChecks

end Sounio.OntologyELPlus

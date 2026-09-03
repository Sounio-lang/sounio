-- formal/lean4/SounioEpistemic.lean
import SounioEffects
import SounioLinear

/-!
# Sounio Epistemic Type Theory -- Lean 4 Formalization

Formalizes the epistemic type system of Sounio: Knowledge types with confidence
tracking, uncertainty propagation, measurement operations, and Bayesian
composition. Connects epistemic types to the algebraic effect system and
linear resource discipline.

This is a NOVEL formalization -- no prior Lean 4 project covers epistemic types
with confidence lattices, provenance tracking, and effect-system integration.

References:
- van Ditmarsch, H. et al. (2007). "Dynamic Epistemic Logic." Springer.
- Halpern, J. (2003). "Reasoning About Uncertainty." MIT Press.
- Shannon, C. (1948). "A Mathematical Theory of Communication." Bell System Tech J.
- Dempster, A. (1967). "Upper and Lower Probabilities Induced by a
  Multivalued Mapping." Annals of Mathematical Statistics.

No sorry. No Mathlib.
-/

open Sounio.Effects Sounio.Linear

namespace Sounio.Epistemic

-- ================================================================
-- S1. Confidence Levels (Abstract Bounded Lattice)
-- ================================================================

/-- Confidence level: 0 = no confidence, higher = more confident.
    We use Nat bounded by a maximum, avoiding Mathlib reals. -/
abbrev Confidence := Nat

/-- The minimum confidence (complete uncertainty). -/
def confBot : Confidence := 0

/-- A confidence level is valid if it does not exceed the bound. -/
def validConf (c : Confidence) (bound : Nat) : Prop := c ≤ bound

/-- Greatest lower bound of two confidence levels. -/
def confMeet (a b : Confidence) : Confidence := if a ≤ b then a else b

/-- Least upper bound of two confidence levels. -/
def confJoin (a b : Confidence) : Confidence := if a ≤ b then b else a

-- S1.1  Lattice Laws

theorem confMeet_comm (a b : Confidence) : confMeet a b = confMeet b a := by
  unfold confMeet Confidence; split <;> split <;> omega

theorem confMeet_assoc (a b c : Confidence) :
    confMeet (confMeet a b) c = confMeet a (confMeet b c) := by
  unfold confMeet Confidence; split <;> (try split) <;> (try split) <;> (try split) <;> omega

theorem confMeet_idempotent (a : Confidence) : confMeet a a = a := by
  unfold confMeet Confidence; split <;> omega

theorem confJoin_comm (a b : Confidence) : confJoin a b = confJoin b a := by
  unfold confJoin Confidence; split <;> split <;> omega

theorem confJoin_assoc (a b c : Confidence) :
    confJoin (confJoin a b) c = confJoin a (confJoin b c) := by
  unfold confJoin Confidence; split <;> (try split) <;> (try split) <;> (try split) <;> omega

theorem confJoin_idempotent (a : Confidence) : confJoin a a = a := by
  unfold confJoin Confidence; split <;> omega

/-- Absorption: meet a (join a b) = a. -/
theorem confMeet_absorption (a b : Confidence) :
    confMeet a (confJoin a b) = a := by
  unfold confMeet confJoin Confidence; split <;> (try split) <;> omega

/-- Absorption: join a (meet a b) = a. -/
theorem confJoin_absorption (a b : Confidence) :
    confJoin a (confMeet a b) = a := by
  unfold confJoin confMeet Confidence; split <;> (try split) <;> omega

theorem confBot_join_left (c : Confidence) : confJoin confBot c = c := by
  unfold confJoin confBot Confidence; split <;> omega

theorem confBot_meet_left (c : Confidence) : confMeet confBot c = confBot := by
  unfold confMeet confBot Confidence; split <;> omega

-- ================================================================
-- S2. Knowledge Type
-- ================================================================

/-- A knowledge value: confidence level and provenance chain length.
    Confidence tracks epistemic certainty; provenance tracks
    derivation steps from an original measurement. -/
structure Knowledge where
  confidence : Confidence
  provenance : Nat
  deriving DecidableEq, Repr

theorem Knowledge.ext_iff (k₁ k₂ : Knowledge) :
    k₁ = k₂ ↔ k₁.confidence = k₂.confidence ∧ k₁.provenance = k₂.provenance := by
  constructor
  · intro h; subst h; exact ⟨rfl, rfl⟩
  · intro ⟨hc, hp⟩; cases k₁; cases k₂; simp at hc hp; subst hc; subst hp; rfl

/-- Direct measurement: assigns confidence, zero provenance. -/
def measure (conf : Confidence) : Knowledge := ⟨conf, 0⟩

/-- Derived knowledge: confidence degrades, provenance grows. -/
def derive (k : Knowledge) (degradation : Nat) : Knowledge :=
  ⟨k.confidence - degradation, k.provenance + 1⟩

/-- Combine two independent measurements: conservative (min confidence,
    max provenance). -/
def combine (k₁ k₂ : Knowledge) : Knowledge :=
  ⟨confMeet k₁.confidence k₂.confidence, if k₁.provenance ≤ k₂.provenance then k₂.provenance else k₁.provenance⟩

/-- Strengthen with additional evidence: optimistic (max confidence,
    min provenance). -/
def strengthen (k₁ k₂ : Knowledge) : Knowledge :=
  ⟨confJoin k₁.confidence k₂.confidence, if k₁.provenance ≤ k₂.provenance then k₁.provenance else k₂.provenance⟩

-- ================================================================
-- S3. Knowledge Ordering (Partial Order)
-- ================================================================

/-- Knowledge k1 is at least as informative as k2:
    higher confidence AND shorter provenance chain. -/
def knowledgeLeq (k₁ k₂ : Knowledge) : Prop :=
  k₁.confidence ≥ k₂.confidence ∧ k₁.provenance ≤ k₂.provenance

theorem knowledgeLeq_refl (k : Knowledge) : knowledgeLeq k k :=
  ⟨Nat.le_refl _, Nat.le_refl _⟩

theorem knowledgeLeq_trans (k₁ k₂ k₃ : Knowledge)
    (h₁ : knowledgeLeq k₁ k₂) (h₂ : knowledgeLeq k₂ k₃) :
    knowledgeLeq k₁ k₃ :=
  ⟨Nat.le_trans h₂.1 h₁.1, Nat.le_trans h₁.2 h₂.2⟩

theorem knowledgeLeq_antisymm (k₁ k₂ : Knowledge)
    (h₁ : knowledgeLeq k₁ k₂) (h₂ : knowledgeLeq k₂ k₁) :
    k₁ = k₂ := by
  have hc := Nat.le_antisymm h₂.1 h₁.1
  have hp := Nat.le_antisymm h₁.2 h₂.2
  exact (Knowledge.ext_iff k₁ k₂).mpr ⟨hc, hp⟩

-- ================================================================
-- S4. Measurement Laws
-- ================================================================

theorem measure_confidence (c : Confidence) : (measure c).confidence = c := rfl
theorem measure_zero_provenance (c : Confidence) : (measure c).provenance = 0 := rfl

theorem derive_decreases_confidence (k : Knowledge) (d : Nat) :
    (derive k d).confidence ≤ k.confidence := by
  simp only [derive, Confidence]; omega

theorem derive_increases_provenance (k : Knowledge) (d : Nat) :
    (derive k d).provenance = k.provenance + 1 := rfl

theorem combine_comm (k₁ k₂ : Knowledge) : combine k₁ k₂ = combine k₂ k₁ := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [combine, confMeet, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> (try split) <;> omega)

theorem combine_assoc (k₁ k₂ k₃ : Knowledge) :
    combine (combine k₁ k₂) k₃ = combine k₁ (combine k₂ k₃) := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [combine, confMeet, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> (try split) <;> omega)

theorem combine_idempotent (k : Knowledge) : combine k k = k := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [combine, confMeet, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> (try split) <;> omega)

theorem strengthen_comm (k₁ k₂ : Knowledge) :
    strengthen k₁ k₂ = strengthen k₂ k₁ := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [strengthen, confJoin, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> (try split) <;> omega)

theorem strengthen_assoc (k₁ k₂ k₃ : Knowledge) :
    strengthen (strengthen k₁ k₂) k₃ = strengthen k₁ (strengthen k₂ k₃) := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [strengthen, confJoin, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> (try split) <;> omega)

theorem strengthen_idempotent (k : Knowledge) : strengthen k k = k := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [strengthen, confJoin, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> (try split) <;> omega)

theorem combine_confidence_le_left (k₁ k₂ : Knowledge) :
    (combine k₁ k₂).confidence ≤ k₁.confidence := by
  simp only [combine, confMeet, Confidence]; split <;> omega

theorem combine_confidence_le_right (k₁ k₂ : Knowledge) :
    (combine k₁ k₂).confidence ≤ k₂.confidence := by
  simp only [combine, confMeet, Confidence]; split <;> omega

theorem strengthen_confidence_ge_left (k₁ k₂ : Knowledge) :
    (strengthen k₁ k₂).confidence ≥ k₁.confidence := by
  simp only [strengthen, confJoin, Confidence]; split <;> omega

theorem strengthen_confidence_ge_right (k₁ k₂ : Knowledge) :
    (strengthen k₁ k₂).confidence ≥ k₂.confidence := by
  simp only [strengthen, confJoin, Confidence]; split <;> omega

-- ================================================================
-- S5. Uncertainty Propagation
-- ================================================================

/-- Uncertainty is the complement of confidence relative to a bound. -/
def uncertainty (k : Knowledge) (bound : Nat) : Nat := bound - k.confidence

theorem uncertainty_zero_is_certain (bound : Nat) :
    uncertainty (measure bound) bound = 0 := by
  simp only [uncertainty, measure, Confidence]; omega

theorem uncertainty_monotone (k₁ k₂ : Knowledge) (bound : Nat)
    (hv₁ : k₁.confidence ≤ bound) (hv₂ : k₂.confidence ≤ bound)
    (h : k₁.confidence ≤ k₂.confidence) :
    uncertainty k₂ bound ≤ uncertainty k₁ bound := by
  simp only [uncertainty, Confidence] at *; omega

theorem combine_uncertainty_ge (k₁ k₂ : Knowledge) (bound : Nat)
    (hv₁ : k₁.confidence ≤ bound) (hv₂ : k₂.confidence ≤ bound) :
    uncertainty (combine k₁ k₂) bound ≥
      max (uncertainty k₁ bound) (uncertainty k₂ bound) := by
  simp only [uncertainty, combine, confMeet, Confidence]; split <;> omega

theorem strengthen_uncertainty_le (k₁ k₂ : Knowledge) (bound : Nat)
    (hv₁ : k₁.confidence ≤ bound) (hv₂ : k₂.confidence ≤ bound) :
    uncertainty (strengthen k₁ k₂) bound ≤
      min (uncertainty k₁ bound) (uncertainty k₂ bound) := by
  simp only [uncertainty, strengthen, confJoin, Confidence]; split <;> omega

theorem derive_uncertainty_ge (k : Knowledge) (d : Nat) (bound : Nat)
    (hv : k.confidence ≤ bound) :
    uncertainty (derive k d) bound ≥ uncertainty k bound := by
  simp only [uncertainty, derive, Confidence]; omega

-- ================================================================
-- S6. Bayesian Composition (Abstract Model)
-- ================================================================

/-- Prior-to-posterior update: confidence accumulates (capped at bound),
    provenance grows to reflect the derivation chain. -/
def bayesianUpdate (prior evidence : Knowledge) (bound : Nat) : Knowledge :=
  ⟨if prior.confidence + evidence.confidence ≤ bound then prior.confidence + evidence.confidence else bound,
   prior.provenance + evidence.provenance + 1⟩

theorem bayesianUpdate_increases_confidence
    (prior evidence : Knowledge) (bound : Nat)
    (hv : prior.confidence ≤ bound) :
    (bayesianUpdate prior evidence bound).confidence ≥ prior.confidence := by
  simp only [bayesianUpdate, Confidence]; split <;> omega

theorem bayesianUpdate_increases_provenance
    (prior evidence : Knowledge) (bound : Nat) :
    (bayesianUpdate prior evidence bound).provenance > prior.provenance := by
  simp only [bayesianUpdate, Confidence]; omega

theorem bayesianUpdate_bounded (prior evidence : Knowledge) (bound : Nat) :
    (bayesianUpdate prior evidence bound).confidence ≤ bound := by
  simp only [bayesianUpdate, Confidence]; split <;> omega

theorem bayesianUpdate_zero_evidence (prior : Knowledge) (bound : Nat)
    (hv : prior.confidence ≤ bound) :
    (bayesianUpdate prior (measure 0) bound).confidence = prior.confidence := by
  -- `split` cannot case on this `ite` under Lean 4.33; it can under the pinned
  -- 4.32.2. That difference is what took `main` red when `leanprover/lean4:stable`
  -- moved (fixed by pinning, 7cd35ba73c). No case analysis is needed either way:
  -- `measure 0` contributes zero confidence, so `Nat.add_zero` reduces the guard to
  -- exactly `hv` and `if_pos` takes the then-branch. Verified to compile under BOTH
  -- 4.32.2 and 4.33.0, so the pin can be moved forward without reopening this.
  simp only [bayesianUpdate, measure, Confidence, Nat.add_zero]
  exact if_pos hv

theorem bayesianUpdate_provenance_sum (k e₁ e₂ : Knowledge) (bound : Nat) :
    (bayesianUpdate (bayesianUpdate k e₁ bound) e₂ bound).provenance =
      k.provenance + e₁.provenance + e₂.provenance + 2 := by
  show (k.provenance + e₁.provenance + 1) + e₂.provenance + 1 =
    k.provenance + e₁.provenance + e₂.provenance + 2
  omega

-- ================================================================
-- S7. Epistemic Effect Interaction
-- ================================================================

/-- The epistemic effect row: computations that access uncertain knowledge. -/
def epistemicRow : EffectRow := singleRow .Epistemic

theorem pure_no_epistemic : ¬(.Epistemic ∈ᵣ pureRow) :=
  memberOf_pure_false .Epistemic

theorem epistemic_handled : mask epistemicRow .Epistemic = pureRow :=
  single_mask_pure .Epistemic

theorem epistemic_in_row : Effect.Epistemic ∈ᵣ epistemicRow :=
  singleRow_member .Epistemic

theorem io_not_in_epistemic : ¬(.IO ∈ᵣ epistemicRow) := by
  simp [memberOf, epistemicRow, singleRow]

theorem gpu_not_in_epistemic : ¬(.GPU ∈ᵣ epistemicRow) := by
  simp [memberOf, epistemicRow, singleRow]

theorem epistemic_subrow_of_member (r : EffectRow)
    (h : Effect.Epistemic ∈ᵣ r) :
    effectSubrow epistemicRow r := by
  intro e he
  simp [epistemicRow, singleRow, memberOf] at he
  rw [he]; exact h

theorem epistemic_union_io :
    (Effect.Epistemic ∈ᵣ rowUnion epistemicRow (singleRow .IO)) ∧
    (Effect.IO ∈ᵣ rowUnion epistemicRow (singleRow .IO)) := by
  constructor
  · exact effectSubrow_union_left epistemicRow (singleRow .IO) .Epistemic
      (singleRow_member .Epistemic)
  · exact effectSubrow_union_right epistemicRow (singleRow .IO) .IO
      (singleRow_member .IO)

theorem epistemic_disjoint_gpu :
    rowDisjoint epistemicRow (singleRow .GPU) := by
  apply rowDisjoint_single_absent
  simp [memberOf, epistemicRow, singleRow]

theorem handle_epistemic_preserves_io :
    mask (rowUnion epistemicRow (singleRow .IO)) .Epistemic =
      singleRow .IO := by
  funext f; simp only [mask, rowUnion, epistemicRow, singleRow]
  by_cases h : f = .Epistemic
  · subst h; simp
  · by_cases hio : f = .IO <;> simp [h, hio]

-- ================================================================
-- S8. Knowledge Degradation Chain
-- ================================================================

/-- Apply n derivation steps, each with degradation d. -/
def deriveChain (k : Knowledge) (d : Nat) : Nat → Knowledge
  | 0 => k
  | n + 1 => derive (deriveChain k d n) d

theorem deriveChain_zero (k : Knowledge) (d : Nat) :
    deriveChain k d 0 = k := rfl

theorem deriveChain_provenance (k : Knowledge) (d : Nat) (n : Nat) :
    (deriveChain k d n).provenance = k.provenance + n := by
  induction n with
  | zero => simp [deriveChain]
  | succ n ih => simp [deriveChain, derive, ih]; omega

theorem deriveChain_confidence_mono (k : Knowledge) (d : Nat) (n : Nat) :
    (deriveChain k d n).confidence ≤ k.confidence := by
  induction n with
  | zero => simp [deriveChain]
  | succ n ih =>
    simp only [deriveChain, derive]
    calc (deriveChain k d n).confidence - d
        ≤ (deriveChain k d n).confidence := Nat.sub_le _ _
      _ ≤ k.confidence := ih

theorem deriveChain_one (k : Knowledge) (d : Nat) :
    deriveChain k d 1 = derive k d := rfl

theorem deriveChain_add_provenance (k : Knowledge) (d : Nat) (n m : Nat) :
    (deriveChain k d (n + m)).provenance =
      (deriveChain (deriveChain k d n) d m).provenance := by
  simp [deriveChain_provenance]; omega

-- ================================================================
-- S9. Information Content
-- ================================================================

/-- Shannon-inspired information content: higher confidence = more information. -/
def infoContent (k : Knowledge) : Nat := k.confidence

theorem infoContent_nonneg (k : Knowledge) : infoContent k ≥ 0 :=
  Nat.zero_le _

theorem combine_info_min (k₁ k₂ : Knowledge) :
    infoContent (combine k₁ k₂) = min (infoContent k₁) (infoContent k₂) := by
  simp only [infoContent, combine, confMeet, Confidence]; split <;> omega

theorem strengthen_info_max (k₁ k₂ : Knowledge) :
    infoContent (strengthen k₁ k₂) = max (infoContent k₁) (infoContent k₂) := by
  simp only [infoContent, strengthen, confJoin, Confidence]; split <;> omega

theorem derive_info_le (k : Knowledge) (d : Nat) :
    infoContent (derive k d) ≤ infoContent k := by
  simp only [infoContent, derive, Confidence]; omega

theorem measure_info (c : Confidence) : infoContent (measure c) = c := rfl

-- ================================================================
-- S10. Combine/Strengthen Form a Lattice on Knowledge
-- ================================================================

theorem combine_leq_left (k₁ k₂ : Knowledge) :
    knowledgeLeq k₁ (combine k₁ k₂) := by
  simp only [knowledgeLeq, combine, confMeet, Confidence]
  constructor <;> (split <;> (try split) <;> omega)

theorem combine_leq_right (k₁ k₂ : Knowledge) :
    knowledgeLeq k₂ (combine k₁ k₂) := by
  simp only [knowledgeLeq, combine, confMeet, Confidence]
  constructor <;> (split <;> (try split) <;> omega)

theorem strengthen_geq_left (k₁ k₂ : Knowledge) :
    knowledgeLeq (strengthen k₁ k₂) k₁ := by
  simp only [knowledgeLeq, strengthen, confJoin, Confidence]
  constructor <;> (split <;> (try split) <;> omega)

theorem strengthen_geq_right (k₁ k₂ : Knowledge) :
    knowledgeLeq (strengthen k₁ k₂) k₂ := by
  simp only [knowledgeLeq, strengthen, confJoin, Confidence]
  constructor <;> (split <;> (try split) <;> omega)

-- ================================================================
-- S11. Epistemic Modality Interaction
-- ================================================================

theorem epistemic_linear_must_use :
    Modality.mustUse .Linear = true := rfl

theorem epistemic_affine_droppable :
    Modality.allowsWeakening .Affine = true := rfl

theorem epistemic_unrestricted_shareable :
    Modality.allowsContraction .Unrestricted = true := rfl

theorem epistemic_single_use_always_ok (m : Modality) :
    wellUsed m .One := wellUsed_one_always m

-- ================================================================
-- S12. Confidence Validity Preservation
-- ================================================================

theorem combine_preserves_valid (k₁ k₂ : Knowledge) (bound : Nat)
    (h₁ : validConf k₁.confidence bound)
    (h₂ : validConf k₂.confidence bound) :
    validConf (combine k₁ k₂).confidence bound := by
  simp only [validConf, combine, confMeet, Confidence] at *; split <;> omega

theorem strengthen_preserves_valid (k₁ k₂ : Knowledge) (bound : Nat)
    (h₁ : validConf k₁.confidence bound)
    (h₂ : validConf k₂.confidence bound) :
    validConf (strengthen k₁ k₂).confidence bound := by
  simp only [validConf, strengthen, confJoin, Confidence] at *; split <;> omega

theorem derive_preserves_valid (k : Knowledge) (d : Nat) (bound : Nat)
    (h : validConf k.confidence bound) :
    validConf (derive k d).confidence bound := by
  simp only [validConf, derive, Confidence] at *; omega

theorem bayesianUpdate_preserves_valid (prior evidence : Knowledge) (bound : Nat) :
    validConf (bayesianUpdate prior evidence bound).confidence bound := by
  simp only [validConf, bayesianUpdate, Confidence]; split <;> omega

-- ================================================================
-- S13. Absorption Laws for Knowledge Operations
-- ================================================================

/-- Absorption: combine k (strengthen k k') = k. -/
theorem knowledge_absorption_combine (k k' : Knowledge) :
    combine k (strengthen k k') = k := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [combine, strengthen, confMeet, confJoin, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> omega)

/-- Absorption: strengthen k (combine k k') = k. -/
theorem knowledge_absorption_strengthen (k k' : Knowledge) :
    strengthen k (combine k k') = k := by
  apply (Knowledge.ext_iff _ _).mpr
  simp only [strengthen, combine, confJoin, confMeet, Confidence]
  constructor <;> (split <;> (try split) <;> (try split) <;> omega)

-- ================================================================
-- S14. Distributivity
-- ================================================================

/-- Combine distributes over strengthen (confidence component). -/
theorem combine_strengthen_distrib_confidence (k₁ k₂ k₃ : Knowledge) :
    (combine k₁ (strengthen k₂ k₃)).confidence =
    (strengthen (combine k₁ k₂) (combine k₁ k₃)).confidence := by
  simp only [combine, strengthen, confMeet, confJoin, Confidence]; split <;> (try split) <;> (try split) <;> (try split) <;> (try split) <;> omega

-- ================================================================
-- S15. Provenance Bounds
-- ================================================================

theorem combine_provenance_ge_left (k₁ k₂ : Knowledge) :
    (combine k₁ k₂).provenance ≥ k₁.provenance := by
  simp only [combine]; split <;> omega

theorem combine_provenance_ge_right (k₁ k₂ : Knowledge) :
    (combine k₁ k₂).provenance ≥ k₂.provenance := by
  simp only [combine]; split <;> omega

theorem strengthen_provenance_le_left (k₁ k₂ : Knowledge) :
    (strengthen k₁ k₂).provenance ≤ k₁.provenance := by
  simp only [strengthen]; split <;> omega

theorem strengthen_provenance_le_right (k₁ k₂ : Knowledge) :
    (strengthen k₁ k₂).provenance ≤ k₂.provenance := by
  simp only [strengthen]; split <;> omega

end Sounio.Epistemic

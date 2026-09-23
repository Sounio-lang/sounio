/-
  Concrete declared-action quotient witness for the frozen Pireus V13 outer
  canonicalizer.  This module proves the local finite-action and declared-orbit
  obligations; it does not close the remaining executable parity boundary.

  A state is the exact big-endian BitVec 256 encoding of a sign table in the
  proved 11-bit direct gauge section.  Each analytic GL(4,F2) x input-swap
  view acts by raw pullback followed by the proved gauge normalization.
  Covariance of the gauge action makes normalization absorptive, yielding
  lawful identity, composition, and inverse operations on the quotient.

  The generic finite-action theorem therefore applies to this concrete
  40320-view ordered state space.  The remaining V13 boundary is the equality
  between this proved quotient minimum and the separately frozen Sounio
  streaming implementation.
-/
import SounioPireusAnalyticActionClosure

namespace SounioPireusConcreteQuotientAction

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusFiniteActionCanonicalization
open SounioPireusGaugeCoboundaryAction
open SounioPireusGaugeSectionCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusBasisFixedGaugeRebase
open SounioPireusGL4AnalyticActionCensus
open SounioPireusSignTableBitVecLex
open SounioPireusAnalyticActionClosure

theorem normalize_raw_action_absorbs_normalize
    (action : LinearSwapAction) (table : SignTable) :
    normalizeGauge (rawAct action (normalizeGauge table)) =
      normalizeGauge (rawAct action table) := by
  change normalizeGauge
      (rawAct action (gaugeAct (directSectionWord table) table)) =
    normalizeGauge (rawAct action table)
  rw [raw_action_transports_basis_fixed_gauge]
  exact normalize_gauge_invariant _ _

def IsNormalizedBits (bits : BitVec 256) : Prop :=
  IsDirectGaugeSection (unpackTable bits)

abbrev NormalizedBits := {bits : BitVec 256 // IsNormalizedBits bits}

instance normalizedBitsMin : Min NormalizedBits where
  min left right := if left.val ≤ right.val then left else right

instance normalizedBitsLawfulOrderMin :
    Std.LawfulOrderMin NormalizedBits := by
  apply Std.LawfulOrderMin.of_min_le
  · intro left right
    by_cases ordered : left.val ≤ right.val
    · change (if left.val ≤ right.val then left else right) ≤ left
      rw [if_pos ordered]
      exact BitVec.le_refl left.val
    · change (if left.val ≤ right.val then left else right) ≤ left
      rw [if_neg ordered]
      exact (BitVec.le_total left.val right.val).resolve_left ordered
  · intro left right
    by_cases ordered : left.val ≤ right.val
    · change (if left.val ≤ right.val then left else right) ≤ right
      rw [if_pos ordered]
      exact ordered
    · change (if left.val ≤ right.val then left else right) ≤ right
      rw [if_neg ordered]
      exact BitVec.le_refl right.val
  · intro left right
    by_cases ordered : left.val ≤ right.val
    · change (if left.val ≤ right.val then left else right) = left ∨
        (if left.val ≤ right.val then left else right) = right
      rw [if_pos ordered]
      exact Or.inl rfl
    · change (if left.val ≤ right.val then left else right) = left ∨
        (if left.val ≤ right.val then left else right) = right
      rw [if_neg ordered]
      exact Or.inr rfl

def normalizedBitsOfTable (table : SignTable) : NormalizedBits :=
  ⟨packTable (normalizeGauge table), by
    unfold IsNormalizedBits
    rw [unpack_pack_table]
    exact normalize_gauge_has_zero_section_bits table⟩

def tableOfNormalizedBits (state : NormalizedBits) : SignTable :=
  unpackTable state.val

theorem table_of_normalized_bits_in_section (state : NormalizedBits) :
    IsDirectGaugeSection (tableOfNormalizedBits state) :=
  state.property

theorem table_of_normalized_bits_of_table (table : SignTable) :
    tableOfNormalizedBits (normalizedBitsOfTable table) =
      normalizeGauge table := by
  simp [tableOfNormalizedBits, normalizedBitsOfTable, unpack_pack_table]

theorem normalized_bits_eq_of_tables_eq
    {left right : NormalizedBits}
    (tablesEqual :
      tableOfNormalizedBits left = tableOfNormalizedBits right) :
    left = right := by
  apply Subtype.ext
  apply unpack_table_injective
  exact tablesEqual

def quotientAct
    (view : AnalyticActionView) (state : NormalizedBits) : NormalizedBits :=
  normalizedBitsOfTable
    (rawAct (actionOfView view) (tableOfNormalizedBits state))

theorem table_of_quotient_act
    (view : AnalyticActionView) (state : NormalizedBits) :
    tableOfNormalizedBits (quotientAct view state) =
      normalizeGauge
        (rawAct (actionOfView view) (tableOfNormalizedBits state)) := by
  simp [tableOfNormalizedBits, quotientAct, normalizedBitsOfTable,
    unpack_pack_table]

theorem normalized_bits_of_table_eq_iff_same_gauge_orbit
    (left right : SignTable) :
    normalizedBitsOfTable left = normalizedBitsOfTable right ↔
      SameGaugeOrbit left right := by
  constructor
  · intro statesEqual
    apply (normalize_gauge_equal_iff_same_orbit left right).mp
    have tablesEqual := congrArg tableOfNormalizedBits statesEqual
    simpa only [table_of_normalized_bits_of_table] using tablesEqual
  · intro sameGaugeOrbit
    apply normalized_bits_eq_of_tables_eq
    rw [table_of_normalized_bits_of_table,
      table_of_normalized_bits_of_table]
    exact normalize_gauge_equal_of_same_orbit sameGaugeOrbit

theorem quotient_act_on_normalized_table
    (view : AnalyticActionView) (table : SignTable) :
    quotientAct view (normalizedBitsOfTable table) =
      normalizedBitsOfTable (rawAct (actionOfView view) table) := by
  apply normalized_bits_eq_of_tables_eq
  rw [table_of_quotient_act, table_of_normalized_bits_of_table,
    table_of_normalized_bits_of_table]
  exact normalize_raw_action_absorbs_normalize _ _

theorem quotient_action_identity (state : NormalizedBits) :
    quotientAct identityView state = state := by
  apply normalized_bits_eq_of_tables_eq
  rw [table_of_quotient_act, raw_act_identity_view]
  exact normalize_gauge_fixed_of_section
    (table_of_normalized_bits_in_section state)

theorem quotient_action_compose
    (outer inner : AnalyticActionView) (state : NormalizedBits) :
    quotientAct (composeView outer inner) state =
      quotientAct outer (quotientAct inner state) := by
  apply normalized_bits_eq_of_tables_eq
  rw [table_of_quotient_act, table_of_quotient_act,
    table_of_quotient_act, raw_act_compose_view,
    normalize_raw_action_absorbs_normalize]

theorem quotient_action_inverse
    (view : AnalyticActionView) (state : NormalizedBits) :
    quotientAct (inverseView view) (quotientAct view state) = state := by
  apply normalized_bits_eq_of_tables_eq
  rw [table_of_quotient_act, table_of_quotient_act,
    normalize_raw_action_absorbs_normalize, raw_act_inverse_view]
  exact normalize_gauge_fixed_of_section
    (table_of_normalized_bits_in_section state)

def concreteQuotientActionSystem :
    FiniteActionSystem AnalyticActionView NormalizedBits :=
  { actions := analyticActionViews
  , identity := identityView
  , compose := composeView
  , inverse := inverseView
  , act := quotientAct
  , identity_mem := identity_view_mem
  , compose_mem := compose_view_mem
  , inverse_mem := inverse_view_mem
  , act_identity := quotient_action_identity
  , act_compose := quotient_action_compose
  , act_inverse := quotient_action_inverse }

theorem concrete_quotient_action_count_is_40320 :
    concreteQuotientActionSystem.actions.length = 40320 := by
  simpa [concreteQuotientActionSystem, analyticConcreteActionList] using
    analytic_concrete_action_list_length_is_40320

theorem concrete_quotient_canonical_eq_iff_same_orbit
    (left right : NormalizedBits) :
    concreteQuotientActionSystem.canonicalOption left =
        concreteQuotientActionSystem.canonicalOption right ↔
      concreteQuotientActionSystem.sameOrbit left right := by
  exact canonicalOption_eq_iff_sameOrbit concreteQuotientActionSystem left right

def SameDeclaredLinearSwapGaugeOrbit
    (left right : SignTable) : Prop :=
  ∃ view : AnalyticActionView,
    view ∈ analyticActionViews ∧
      ∃ word : GaugeWord,
        gaugeAct word (rawAct (actionOfView view) left) = right

theorem normalized_same_orbit_iff_same_declared_linear_swap_gauge_orbit
    (left right : SignTable) :
    concreteQuotientActionSystem.sameOrbit
        (normalizedBitsOfTable left) (normalizedBitsOfTable right) ↔
      SameDeclaredLinearSwapGaugeOrbit left right := by
  constructor
  · rintro ⟨view, viewMem, acts⟩
    refine ⟨view, viewMem, ?_⟩
    apply (normalized_bits_of_table_eq_iff_same_gauge_orbit
      (rawAct (actionOfView view) left) right).mp
    rw [← quotient_act_on_normalized_table]
    exact acts
  · rintro ⟨view, viewMem, word, acts⟩
    refine ⟨view, viewMem, ?_⟩
    change quotientAct view (normalizedBitsOfTable left) =
      normalizedBitsOfTable right
    rw [quotient_act_on_normalized_table]
    apply (normalized_bits_of_table_eq_iff_same_gauge_orbit
      (rawAct (actionOfView view) left) right).mpr
    exact ⟨word, acts⟩

def declaredCanonicalOption (table : SignTable) : Option NormalizedBits :=
  concreteQuotientActionSystem.canonicalOption
    (normalizedBitsOfTable table)

theorem declared_canonical_eq_iff_same_declared_linear_swap_gauge_orbit
    (left right : SignTable) :
    declaredCanonicalOption left = declaredCanonicalOption right ↔
      SameDeclaredLinearSwapGaugeOrbit left right := by
  change concreteQuotientActionSystem.canonicalOption
      (normalizedBitsOfTable left) =
        concreteQuotientActionSystem.canonicalOption
          (normalizedBitsOfTable right) ↔
    SameDeclaredLinearSwapGaugeOrbit left right
  exact (concrete_quotient_canonical_eq_iff_same_orbit
      (normalizedBitsOfTable left) (normalizedBitsOfTable right)).trans
    (normalized_same_orbit_iff_same_declared_linear_swap_gauge_orbit
      left right)

structure ConcreteQuotientBoundary where
  exactLexBitVecRepresentationProved : Bool
  analytic40320ActionClosureProved : Bool
  gaugeNormalizationAbsorptionProved : Bool
  concreteQuotientActionLawsProved : Bool
  concreteQuotientCanonicalIffOrbitProved : Bool
  executedSounioStreamingMinimumEqualityProved : Bool
  concreteCanonicalEqualityIffDeclaredLinearSwapGaugeOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def concreteQuotientBoundary : ConcreteQuotientBoundary :=
  { exactLexBitVecRepresentationProved := true
  , analytic40320ActionClosureProved := true
  , gaugeNormalizationAbsorptionProved := true
  , concreteQuotientActionLawsProved := true
  , concreteQuotientCanonicalIffOrbitProved := true
  , executedSounioStreamingMinimumEqualityProved := false
  , concreteCanonicalEqualityIffDeclaredLinearSwapGaugeOrbitProved := true
  , formalTarget03Closed := true
  , formalParityClosed := false
  , claimReady := false }

theorem concrete_quotient_closes_target03_without_claim_promotion :
    (concreteQuotientBoundary.exactLexBitVecRepresentationProved &&
      concreteQuotientBoundary.analytic40320ActionClosureProved &&
      concreteQuotientBoundary.gaugeNormalizationAbsorptionProved &&
      concreteQuotientBoundary.concreteQuotientActionLawsProved &&
      concreteQuotientBoundary.concreteQuotientCanonicalIffOrbitProved &&
      !concreteQuotientBoundary.executedSounioStreamingMinimumEqualityProved &&
      concreteQuotientBoundary.concreteCanonicalEqualityIffDeclaredLinearSwapGaugeOrbitProved &&
      concreteQuotientBoundary.formalTarget03Closed &&
      !concreteQuotientBoundary.formalParityClosed &&
      !concreteQuotientBoundary.claimReady) = true := by
  decide

end SounioPireusConcreteQuotientAction

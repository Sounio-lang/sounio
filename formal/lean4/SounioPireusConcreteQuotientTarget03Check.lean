import SounioPireusConcreteQuotientAction

namespace SounioPireusConcreteQuotientTarget03Check

set_option maxHeartbeats 0

open SounioPireusConcreteQuotientAction

example (table : SounioPireusGaugeCoboundaryAction.SignTable) :
    SounioPireusSignTableBitVecLex.unpackTable
        (SounioPireusSignTableBitVecLex.packTable table) = table :=
  SounioPireusSignTableBitVecLex.unpack_pack_table table

example (bits : BitVec 256) :
    SounioPireusSignTableBitVecLex.packTable
        (SounioPireusSignTableBitVecLex.unpackTable bits) = bits :=
  SounioPireusSignTableBitVecLex.pack_unpack_table bits

example (table : SounioPireusGaugeCoboundaryAction.SignTable) :
    SounioPireusLinearSwapGaugeDescent.rawAct
        (SounioPireusGL4AnalyticActionCensus.actionOfView
          SounioPireusAnalyticActionClosure.identityView) table = table :=
  SounioPireusAnalyticActionClosure.raw_act_identity_view table

example
    (outer inner : SounioPireusAnalyticActionClosure.AnalyticActionView)
    (table : SounioPireusGaugeCoboundaryAction.SignTable) :
    SounioPireusLinearSwapGaugeDescent.rawAct
        (SounioPireusGL4AnalyticActionCensus.actionOfView
          (SounioPireusAnalyticActionClosure.composeView outer inner)) table =
      SounioPireusLinearSwapGaugeDescent.rawAct
        (SounioPireusGL4AnalyticActionCensus.actionOfView outer)
        (SounioPireusLinearSwapGaugeDescent.rawAct
          (SounioPireusGL4AnalyticActionCensus.actionOfView inner) table) :=
  SounioPireusAnalyticActionClosure.raw_act_compose_view outer inner table

example
    (view : SounioPireusAnalyticActionClosure.AnalyticActionView)
    (table : SounioPireusGaugeCoboundaryAction.SignTable) :
    SounioPireusLinearSwapGaugeDescent.rawAct
        (SounioPireusGL4AnalyticActionCensus.actionOfView
          (SounioPireusAnalyticActionClosure.inverseView view))
        (SounioPireusLinearSwapGaugeDescent.rawAct
          (SounioPireusGL4AnalyticActionCensus.actionOfView view) table) = table :=
  SounioPireusAnalyticActionClosure.raw_act_inverse_view view table

example
    (action : SounioPireusLinearSwapGaugeDescent.LinearSwapAction)
    (table : SounioPireusGaugeCoboundaryAction.SignTable) :
    SounioPireusGaugeSectionCanonicalization.normalizeGauge
        (SounioPireusLinearSwapGaugeDescent.rawAct action
          (SounioPireusGaugeSectionCanonicalization.normalizeGauge table)) =
      SounioPireusGaugeSectionCanonicalization.normalizeGauge
        (SounioPireusLinearSwapGaugeDescent.rawAct action table) :=
  normalize_raw_action_absorbs_normalize action table

example (state : NormalizedBits) :
    quotientAct SounioPireusAnalyticActionClosure.identityView state = state :=
  quotient_action_identity state

example
    (outer inner : SounioPireusAnalyticActionClosure.AnalyticActionView)
    (state : NormalizedBits) :
    quotientAct
        (SounioPireusAnalyticActionClosure.composeView outer inner) state =
      quotientAct outer (quotientAct inner state) :=
  quotient_action_compose outer inner state

example
    (view : SounioPireusAnalyticActionClosure.AnalyticActionView)
    (state : NormalizedBits) :
    quotientAct (SounioPireusAnalyticActionClosure.inverseView view)
        (quotientAct view state) = state :=
  quotient_action_inverse view state

example : concreteQuotientActionSystem.actions.length = 40320 :=
  concrete_quotient_action_count_is_40320

example (left right : SounioPireusGaugeCoboundaryAction.SignTable) :
    declaredCanonicalOption left = declaredCanonicalOption right ↔
      SameDeclaredLinearSwapGaugeOrbit left right :=
  declared_canonical_eq_iff_same_declared_linear_swap_gauge_orbit left right

example :
    (concreteQuotientBoundary.exactLexBitVecRepresentationProved &&
      concreteQuotientBoundary.analytic40320ActionClosureProved &&
      concreteQuotientBoundary.gaugeNormalizationAbsorptionProved &&
      concreteQuotientBoundary.concreteQuotientActionLawsProved &&
      concreteQuotientBoundary.concreteQuotientCanonicalIffOrbitProved &&
      !concreteQuotientBoundary.executedSounioStreamingMinimumEqualityProved &&
      concreteQuotientBoundary.concreteCanonicalEqualityIffDeclaredLinearSwapGaugeOrbitProved &&
      concreteQuotientBoundary.formalTarget03Closed &&
      !concreteQuotientBoundary.formalParityClosed &&
      !concreteQuotientBoundary.claimReady) = true :=
  concrete_quotient_closes_target03_without_claim_promotion

end SounioPireusConcreteQuotientTarget03Check

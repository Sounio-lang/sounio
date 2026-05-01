import Lake
open Lake DSL

package «SounioFormal» where

@[default_target]
lean_lib «SounioLinear» where

@[default_target]
lean_lib «SounioEffects» where

@[default_target]
lean_lib «SounioTyping» where

@[default_target]
lean_lib «SounioUnits» where

@[default_target]
lean_lib «SounioRowPoly» where

@[default_target]
lean_lib «SounioSemantics» where

@[default_target]
lean_lib «SounioEpistemic» where

@[default_target]
lean_lib «SounioProgress» where

@[default_target]
lean_lib «SounioSubstitution» where

@[default_target]
lean_lib «SounioPreservation» where

@[default_target]
lean_lib «SounioCausality» where

@[default_target]
lean_lib «SounioCayleyDickson» where

@[default_target]
lean_lib «SounioSkewCategory» where

@[default_target]
lean_lib «SounioBidirectionalBridge» where

@[default_target]
lean_lib «SounioCompositionAlgebra» where

@[default_target]
lean_lib «SounioFormal» where

-- Gen 17/18: EGC proof obligations — verified by CI (lean-proofs job)
@[default_target]
lean_lib «SounioGradedModal» where

@[default_target]
lean_lib «SounioMeasConf» where

@[default_target]
lean_lib «SounioProofObligation» where

@[default_target]
lean_lib «SounioZeroDivisorBridge» where

@[default_target]
lean_lib «SounioImpossibilityChain» where

@[default_target]
lean_lib «SounioSurgicalInterventions» where

@[default_target]
lean_lib «SounioSurgicalCalculus» where

@[default_target]
lean_lib «SounioInterpBasis» where

@[default_target]
lean_lib «SounioRegulatory» where

@[default_target]
lean_lib «SounioLearningDynamics» where

@[default_target]
lean_lib «SounioPathionBridge» where

-- M1: Vancomycin-Knightian thrust — Approx × Causal × Knowledge composition
@[default_target]
lean_lib «SounioApproxCausalKnowledge» where

-- M2: Vancomycin-Knightian thrust — Ferson p-box operator
@[default_target]
lean_lib «SounioKnightian» where

-- M3: Vancomycin-Knightian thrust — clinical dosing safety obligation
@[default_target]
lean_lib «SounioVancomycinDosingSafety» where

-- M2.5: Vancomycin-Knightian thrust — Fréchet outer enclosure for
-- monotone-in-each-arg functions (joint-dependence resolution)
@[default_target]
lean_lib «SounioFrechet» where

-- M3.5: Vancomycin-Knightian thrust — Walley ε-contamination credal
-- set elicitation surface (collapse-at-zero, vacuous-at-one,
-- gap-monotone-in-ε)
@[default_target]
lean_lib «SounioWalley» where

-- M3.5+: Klibanoff–Marinacci–Mukerji smooth-ambiguity certainty-
-- equivalent operator (boundary theorems: alpha=0 collapse,
-- lambda=0/1 Walley alignment, Fréchet composition)
@[default_target]
lean_lib «SounioKlibanoff» where

-- Track 2 / Stage 1: Rat-shadow lift of SounioFrechet.lean — first
-- step in the Float-Real lift roadmap (Nat → Rat → ℝ → Float)
@[default_target]
lean_lib «SounioFrechetRat» where


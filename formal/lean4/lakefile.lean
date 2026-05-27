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

-- Erdős resolution thrust (#508 + #704)
@[default_target]
lean_lib «SounioErdosUnitDistance» where

-- M1: Vancomycin-Knightian thrust — Approx × Causal × Knowledge composition
@[default_target]
lean_lib «SounioApproxCausalKnowledge» where

-- M2: Vancomycin-Knightian thrust — Ferson p-box operator
@[default_target]
lean_lib «SounioKnightian» where

-- M3: Vancomycin-Knightian thrust — clinical dosing safety obligation
@[default_target]
lean_lib «SounioVancomycinDosingSafety» where

-- Tacrolimus dissertation thrust — oral C24h-trough Knightian gate
-- (F·D/(V·(eᶿ−1)) closed-form; mirrors SounioVancomycinDosingSafety)
@[default_target]
lean_lib «SounioTacrolimusDosingSafety» where

-- Tacrolimus + sirolimus DDI — F-boost monotonicity + Fréchet
-- composition widens the combined F_oral PBox (irreducible
-- epistemic floor argument).
@[default_target]
lean_lib «SounioTacrolimusDDI» where

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

-- Track 2 / Stage 2: typeclass abstraction (Mathlib-free) capturing
-- the minimal algebraic content of the Sounio epistemic theorems
@[default_target]
lean_lib «SounioOrderedCarrier» where

-- Track 2 / Stage 2: generic Fréchet enclosure proven once over
-- any OrderedCarrier; Nat/Rat versions are direct specialisations
@[default_target]
lean_lib «SounioFrechetGeneric» where

-- Track 2 / Stage 3a (Route A): Mathlib-free SounioReal as the
-- rational subset of ℝ, with OrderedCarrier instance inherited
-- from Rat (Cauchy completion deferred)
@[default_target]
lean_lib «SounioRealOrder» where

-- Track 2 / Stage 3b: BoundedOrderedCarrier typeclass for
-- IEEE-754 Float with relaxed laws + bounded-Fréchet theorem
-- (Float instance deferred to external IEEE-754 model)
@[default_target]
lean_lib «SounioFloatBounded» where

-- Track 3: Stage 2 lift of M3.5 Walley elicitation theorems
-- (collapse-at-zero, vacuous-at-one, gap-monotone, Fréchet
-- composition) over OrderedCarrier
@[default_target]
lean_lib «SounioWalleyGeneric» where

-- Track 3: Stage 2 lift of M3.5+ Klibanoff boundary theorems
-- (lambda=0/1 walley CE, Fréchet composition) over OrderedCarrier
@[default_target]
lean_lib «SounioKlibanoffGeneric» where

-- Stage 3a-Cauchy: irrational extension of SounioReal via
-- Cauchy sequences (structure + LE eventual + reflexivity +
-- transitivity + ofRat bridge; full OrderedCarrier instance
-- deferred as obligation prop)
@[default_target]
lean_lib «SounioRealCauchy» where

-- Stage 3a-Cauchy partial: mul_le_mul_of_nonneg_right pointwise
-- + OrderedCarrier obligation modulo MulPreservesCauchy +
-- le_p → le_eps lift. The hard half (MulPreservesCauchy)
-- remains deferred to a future SounioRealCauchyMul.lean
-- milestone. Honest naming chosen by post-impl math-review.
@[default_target]
lean_lib «SounioRealCauchyPartial» where

-- Stage 3b-F Phase 1: canonical IEEE-754 binary64 spec
-- (5 axioms). Higham 2002 §2.1 basic-operation model.
-- The 4 BoundedOrderedCarrier Float typeclass methods are
-- derived as theorems in SounioFloatInstance.lean.
@[default_target]
lean_lib «SounioIEEE754Spec» where

-- Stage 3b Float instance via Route C: 4 axioms for IEEE-754
-- binary64 typeclass methods, with the BoundedOrderedCarrier
-- Float instance built on top. Axiomatic interim until Route A
-- (Mathlib bridge) or Route B (in-tree IEEE-754) lands.
@[default_target]
lean_lib «SounioFloatInstance» where

-- M4: Octonion homology functor — discrete G₂ naturality skeleton
-- (Fano-permutation enumeration + naturality square + concrete
-- decideable closure on the canonical basis embed)
@[default_target]
lean_lib «SounioNaturalityG2» where

-- M4: per-function NaturalityG2 obligations (auto-generated by
-- tools/ocssm/generate_naturality_obligations.py from `with NaturalityG2`
-- declarations in stdlib/, tools/, tests/run-pass/).
@[default_target]
lean_lib «SounioNaturalityObligations» where


-- M4 Option C: compositional decomposition of embed_into. Proves
-- 2/3 of the pipeline is exactly G₂-equivariant; locates the residual
-- at Box-Muller. Pure ℕ/ℤ/ℚ Lean, no Mathlib.
@[default_target]
lean_lib «SounioNaturalityG2Decomp» where

-- M4 Option A: runtime Halton-Box-Muller table dump. 1024 rows of
-- the actual runtime embed table at fixed-point Int scale 10^9,
-- auto-generated by tools/ocssm/dump_runtime_table.sio.
@[default_target]
lean_lib «SounioNaturalityG2Runtime» where

-- Incidence geometry: Fano arcs / hyperovals / blocking sets bridged to the
-- 168 / zero-divisor / surgical-calculus structure (UNLEARN kernel = hyperoval).
-- Sibling of SounioErdosUnitDistance; see docs/research/fano-arcs-blocking-sounio-note.md
@[default_target]
lean_lib «SounioFanoArcsBlocking» where

-- Erdős [20] sunflower conjecture, read on the 168/ZD/Surgical set system:
-- 4-regular ⇒ uniform 4-petal stars; intra-fiber ⇒ bounded cross-sunflowers.
-- See docs/research/sunflower-168-sounio-note.md
@[default_target]
lean_lib «SounioSunflower» where

-- Erdős [90] CLASSICAL planar attack: exact triangular-lattice (Eisenstein ℤ[ω]) lower
-- bound u(n) ≥ ⌊3n−√(12n−3)⌋, witnessed. Baseline for the cluster search.
-- See docs/research/erdos-90-planar-search-plan.md
@[default_target]
lean_lib «SounioErdos90PlanarLowerBound» where

-- Erdős [90] unit-distance count spectrum on the 168/ZD structure (7-vertex probe):
-- associator lever beats linear (max 12 vs 9, 6 vs 2 distinct counts) + interleaved-
-- star count-growth (gap 6<20<26). See docs/research/erdos-90-168-spectrum-note.md
@[default_target]
lean_lib «SounioErdos90UnitSpectrum» where

-- Erdős [90] refinement: the count-growth separation lifted to the pathion level
-- (32-D, Cayley-Dickson 5). Pathion associator star breaks the sedenion ceiling of 40
-- (reaches 44 at 15 verts, 168 at 31), 3696 verified pathion ZD pairs.
@[default_target]
lean_lib «SounioErdos90PathionGrowth» where

-- Slice D of the SOTA push: Lean soundness sketch of the
-- epistemic-effect calculus. Backs the registered SOTA claim that
-- epistemic gradual compilation is the language-level contribution
-- (see project_pl_contribution_sota; docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md §3).
-- Open obligation: effect_preservation/subst_preserves_typing (see SounioSubstitution.lean).
@[default_target]
lean_lib «EpistemicEffects» where

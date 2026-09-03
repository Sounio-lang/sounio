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

-- Thin projective-measurement corollary file over SounioZeroDivisorBridge:
-- re-states the canonical 4-annihilator structure of A = e3+e10 in
-- measurement-channel language (forgettableBasis, erasure_*, channel count).
-- No new sed_mul claims; pure re-export with `native_decide`. Aligned to the
-- runtime after stdlib/algebra/sedenion.sio sign-fix in commit 16b6ad3ea.
@[default_target]
lean_lib «SounioSedenionMeasurement» where

@[default_target]
lean_lib «SounioCDCocycle» where

@[default_target]
lean_lib «SounioCDTowerSeam» where

@[default_target]
lean_lib «SounioCDConverse» where

@[default_target]
lean_lib «SounioCDRecursiveSeam» where

@[default_target]
lean_lib «SounioCDqbig» where

-- CD core-law twin recursion — per-dimension native_decide certificate (dims 16/32/64) of BOTH
-- doubling recursions S=2S'-8·[hi_lo≠0] and S=8-2S', plus Dmax=4(2^(n-3)-1). The ∀n proof of the
-- recursions is in SounioSeamFlip; this file anchors them at fixed dims (regression, like lsq_16/32/64).
@[default_target]
lean_lib «SounioCDCoreLaw» where

-- Seam-flip law — the ∀n KEYSTONE under the whole 168 lane (lift / orbit theorem / annihilation=
-- associator bridge / core-law twin recursion all bottom out on it). Proves, for ALL n, Mathlib-free,
-- no sorry, no native_decide: the one-step cocycle recursion R (four branches), antisymmetry, cdSigma=±1,
-- and the FULL associator seam-flip law — all eight (p,q,r) seam configurations over the whole locus
-- (generic + degenerate), with exact chi-corrections. Axioms [propext, Classical.choice, Quot.sound].
@[default_target]
lean_lib «SounioSeamFlip» where

@[default_target]
lean_lib «SounioSeamBridge» where

@[default_target]
lean_lib «SounioGresnigtG2S3» where

@[default_target]
lean_lib «SounioGresnigtFamilyS3» where

@[default_target]
lean_lib «SounioFureyChargeG2» where

@[default_target]
lean_lib «SounioSedenionGresnigtOctonions» where

@[default_target]
lean_lib «SounioSedenionOctonionCensus» where

-- Frente B vector 4/3: sedenion left-mult algebra = Cℓ(8) (peer-reviewed; Gresnigt).
@[default_target]
lean_lib «SounioSedenionClifford8» where

-- Frente B vector 4/3 Part A: Sounio reproduces Furey's octonion -> one Standard-Model generation.
-- Fermionic ladder algebra {A_i,A_j}=0, {A_i,A_j^dag}=4 delta_ij I over Z[i] (native_decide) + the
-- one-generation charge multiplicities C(3,n)=[1,3,3,1]. Mathlib-free, no sorry.
@[default_target]
lean_lib «SounioFureyOctonion» where

-- Frente B vector 4/1: emergent metric — integral spectra of the ZD-geometry graphs.
@[default_target]
lean_lib «SounioSedenionSpectra» where

-- Frente B vector 4/2: substrate dynamics — spanning-tree complexity (Matrix-Tree, exact Bareiss
-- integer det) + random-walk return counts of the ZD-geometry graphs. Mathlib-free, native_decide.
@[default_target]
lean_lib «SounioSedenionDynamics» where

-- Frente B: the sedenion signed-automorphism group = 168 = |PSL(2,7)|, fixing e8. NOT a
-- default_target: 3 native_decide sweeps over GL(4,2)=65536 take ~1 min; `lake build SounioSedenionAutomorphism`.
lean_lib «SounioSedenionAutomorphism» where

-- Frente B vector 4/3 B: the sedenion extension of the Furey ladder — octonion SM generation persists
-- (B1) and the doubling adds exactly one fermionic mode (greedy rank 3 -> 4). All three native_decide
-- sweeps (16x16 complex greedy over 1..15) build in ~10s, so kept as a default_target.
@[default_target]
lean_lib «SounioSedenionLadderExtension» where

-- Frente B vector 1: the 7 fibers = Fano plane PG(2,2), Aut = PGL(3,2). Corollary of the
-- automorphism sweep; NON-default_target (~1 min): `lake build SounioSedenionFano`.
lean_lib «SounioSedenionFano» where

-- Frente B: quartet<->fiber incidence of the sedenion ZD geometry (42 quartets = 2*K_7 on 7 fibers).
@[default_target]
lean_lib «SounioSedenionIncidence» where

@[default_target]
lean_lib «SounioSedenionQuartets» where

-- Frente B: the associator side of the sedenion tower — 1848 = 11*168 ordered non-associative
-- basis triples (confirms the ZD-geometry report conjecture). Mathlib-free native_decide, no sorry.
@[default_target]
lean_lib «SounioSedenionAssociator1848» where

@[default_target]
lean_lib «SounioSedenionFiberIdentity» where

-- Independent-spec (native_decide) leg for the sedenion ZD e8-boundary + 7-fiber decomposition.
-- Mathlib-free, no sorry; the third checker (Lean kernel) behind the executed-in-Sounio results
-- tests/run-pass/sedenion_e8_boundary.sio and sedenion_zd_fibers.sio.
@[default_target]
lean_lib «SounioSedenionE8Fibers» where

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

-- Associator-shadow lever (non-associative sedenion product `Sounio.AssociatorShadow.smul`
-- over 16 components). Algebraic substrate imported by the Erdős [90] count-growth files
-- (SounioErdos90UnitSpectrum / SounioErdos90PathionGrowth). Mathlib-free, native_decide, no sorry.
@[default_target]
lean_lib «SounioAssociatorShadow» where

-- Erdős resolution thrust (#508 + #704)
@[default_target]
lean_lib «SounioErdosUnitDistance» where

-- chi(R^2) >= 5 GEOMETRY LEG: G_529 (Heule) is a unit-distance graph, exact over
-- Q(sqrt3,sqrt5,sqrt7,sqrt11). Auto-generated by examples/erdos/gen_lean_geometry.sh.
-- NOT a default_target: native_decide over 529x16 literals takes ~3 min; build on
-- demand with `lake build SounioDeGreyUnitDistance` or `lean <file>`.
lean_lib «SounioDeGreyUnitDistance» where

-- chi(R^2) >= 5 REDUCTION LEG: combines the geometry leg (G_529 unit-distance) and the
-- SAT leg (G_529 not 4-colourable, cake_lpr-verified) into the logical reduction
-- "unit-distance graph + not k-colourable => plane chromatic number > k". Pure core Lean
-- (no Mathlib, no native_decide): the two legs enter as explicit, externally-discharged
-- hypotheses. Checks in <1s, so kept as a default_target (CI lean-proofs).
@[default_target]
lean_lib «SounioDeGreyChi5» where

-- chi(R^2) >= 5 GEOMETRY-LEG-DISCHARGED: instantiates the SounioDeGreyChi5 reduction on the
-- concrete G_529 over the exact symbolic field-plane QF×QF. Turns the geometry hypothesis into
-- a PROVED fact (via the native_decide certificate), leaving only the SAT leg hypothetical.
-- NOT a default_target: imports SounioDeGreyUnitDistance (~3 min native_decide) and runs its
-- own native_decide; build on demand with `lake build SounioDeGreyChi5Concrete`.
lean_lib «SounioDeGreyChi5Concrete» where

-- Multiquadratic-field faithfulness groundwork: ring laws for the QF kernel (qadd/qmul),
-- proved WITHOUT Mathlib / `ring`. PROVED: qadd_comm, qadd_zero_{left,right}, qmul_comm
-- (XOR-permutation symmetry via finite native_decide). OPEN (stated as Props, not assumed):
-- qmul assoc/distrib/unit/neg. Standalone (no imports), checks fast → default_target.
-- See docs/research/multiquad-faithfulness-note.md.
@[default_target]
lean_lib «SounioMultiquadRing» where

-- Value-equivalence quotient of the QF multiquadratic kernel (imports SounioMultiquadRing).
-- PROVED: QFeq Setoid, qadd/qmul/qsub congruence, neg, left/right distrib on quotient.
-- OPEN: QmulAssocObligation (staged). Build: `lake build SounioMultiquadQuotient`.
@[default_target]
lean_lib «SounioMultiquadQuotient» where

-- Abstract ordered field with square roots: target interface for ℚ(√3,√5,√7,√11) ↪ ℝ.
-- PROVED: nonneg_sqrt_unique, mul_sqrt, ofNat_nonneg, s_sq, r_zero. STAGED:
-- GeneratorLawObligation (multiquadratic generator law on radical map `r`).
@[default_target]
lean_lib «SounioSqrtField» where

-- QF -> SqrtField numerator ring-homomorphism core (imports SounioSqrtField + SounioMultiquadRing).
-- PROVED: Mathlib-free finite-sum library (fsum), evalNum numerator map, and the multiplicative
-- core evalNum (qmul x y) = mul (evalNum x) (evalNum y) via generator_law + perm_range_xor reindex.
-- Build: `lake build SounioMultiquadHom`.
@[default_target]
lean_lib «SounioMultiquadHom» where

-- chi(R^2) >= 5 SAT-LEG INTERNALISATION (B1): soundness of the graph-colouring SAT
-- encoding. Proves `(colourCNF n k edges).Unsat -> no proper k-colouring (Fin)` in pure
-- core Lean (axioms: [propext, Quot.sound]; no Mathlib, no native_decide). This is the
-- scale-independent bridge that converts any in-Lean Unsat into a chromatic lower bound.
@[default_target]
lean_lib «SounioSatColouringBridge» where

-- χ≥5 SAT-LEG, reflection harness: unverified (soundness-irrelevant) LRAT-text
-- parser used by the file-loaded "souc_check" route — embeds a souc_sat LRAT as a
-- String literal parsed under native_decide, sidestepping the Array-IntAction
-- term-size wall. See examples/erdos/gen_lean_sat_reflect.sh.
@[default_target]
lean_lib «SounioSatReflect» where

-- χ≥5 SAT-LEG, WLOG leg: triangle-precolour symmetry break is satisfiability-
-- preserving for k=4. Lifts souc_sat's SB-augmented Unsat to an UNCONDITIONAL
-- "no proper 4-colouring" (χ≥5). relabel4 bijectivity decided over Fin 4; axioms
-- [propext, Quot.sound] (no Mathlib, no native_decide).
@[default_target]
lean_lib «SounioSatColouringSB» where

-- B1 mechanism spike: a tiny hand-written UNSAT certificate re-checked by Lean core's
-- verified LRAT checker (check/check_sound + native_decide). Standalone; fast.
@[default_target]
lean_lib «SounioSatCheckSpike» where

-- B1 end-to-end on a REAL souc_sat certificate (K_7/6): souc_sat's own CDCL LRAT proof,
-- re-checked by Lean core's verified checker, yields `k76_cnf.Unsat`, then the colouring
-- bridge yields `¬ 6-colourable K_7`. Autogenerated by examples/erdos/gen_lean_sat.sh.
-- NOT a default_target: native_decide over ~1100 LRAT actions takes a few minutes; build
-- on demand with `lake build SounioSatK76`.
lean_lib «SounioSatK76» where

-- χ≥5 FLAGSHIP: G₅₂₉ (de Grey unit-distance fragment, 529 verts) is NOT
-- 4-colourable — souc_sat's 98 616-line CDCL LRAT re-checked inside Lean core via
-- *file-loaded-style reflection* (the LRAT is a String literal parsed at native-
-- eval time, NOT an embedded Array IntAction term — sidesteps the term-size wall;
-- ~11 s). g529_not_colourable lifts it to unconditional χ(G₅₂₉)≥5 via the WLOG
-- triangle-precolour leg. NOT a default_target (30 MB source; build on demand:
-- `lake build SounioSatG529`).
lean_lib «SounioSatG529» where

-- χ≥6 SEARCH SMOKE: generated reflected LRAT certificate for finite K₆ not being
-- 5-colourable. This exercises the exact no-5 SAT certificate plumbing for future
-- witnesses, but is intentionally not a Euclidean/unit-distance χ≥6 theorem.
lean_lib «SounioSatK65Reflect» where

-- χ≥5 COMPOSITION: wires the now-proven SAT leg (SounioSatG529.g529_not_colourable)
-- into the geometry reduction (SounioDeGreyChi5Concrete), discharging the last
-- hypothesis. g529_field_plane_chi_ge_5: the exact field-plane QF×QF unit-distance
-- graph has no proper 4-colouring — χ(QF²)≥5, ZERO hypotheses, no Mathlib, no sorry.
-- NOT a default_target (imports the 30 MB G529 cert; `lake build SounioDeGreyChi5Closed`).
lean_lib «SounioDeGreyChi5Closed» where

-- ABSTRACT TRANSFER leg: χ(F²)≥5 for ANY commutative-ring-like F receiving QF via a
-- homomorphism (QFTransfer). Proved with NO Mathlib — only the hom equations + unit
-- detection; the F-squared-distance collapses through φ onto the QF certificate. The
-- `qfSelf` (id) instance recovers the field-plane result, proving the abstraction faithful.
-- The sole remaining χ(ℝ²)≥5 step is providing the single ℝ instance (Real.sqrt). Imports
-- SounioDeGreyChi5Concrete (~3 min native_decide); `lake build SounioDeGreyChi5Transfer`.
lean_lib «SounioDeGreyChi5Transfer» where

-- Guarded abstract transfer: χ(F²)≥5 for EVERY SqrtField F (Mathlib-free). Packages the
-- transfer with the den≠0 well-formedness guard the proved fraction homomorphism satisfies
-- (SounioMultiquadHom.phi_qmul/qadd/qsub/phi_unit), discharges the guard on the de Grey
-- edge set (all emb denominators nonzero), and exhibits the SqrtField instance. The sole
-- remaining χ(ℝ²)≥5 input is the analytic 'ℝ is a SqrtField'. `lake build SounioDeGreyChi5TransferWf`.
lean_lib «SounioDeGreyChi5TransferWf» where

-- Scoped current-fragment transfer: the reflected G529 LRAT obstruction transfers through any
-- target receiving the checked QF operations used by the current {3,5,11} support. This is an
-- honest boundary layer below the standalone RootedField3511 interface.
-- `lake build SounioDeGreyChi5Transfer3511`.
lean_lib «SounioDeGreyChi5Transfer3511» where

-- Named three-root target interface for the current {3,5,11} G529 transfer. This removes the
-- public transfer theorem's dependence on the older four-root RootedField interface, while keeping
-- compatibility phi-law fields for the existing transfer theorem. The adjacent evaluator target
-- below is the canonical derived phi3511/QF3511TransferWf path.
-- `lake build SounioDeGreyChi5Rooted3511`.
lean_lib «SounioDeGreyChi5Rooted3511» where

-- Derived 8-mask evaluator for the named three-root target. Proves the compressed qmul core from
-- RootedField3511 algebra, bridges de Grey's 16-mask qmul through {3,5,11} support, then proves
-- phi3511 qmul/qadd/qsub/unit under explicit IntCastNonzero and packages that evaluator as a
-- QF3511TransferWf target.
-- `lake build SounioDeGreyChi5Eval3511`.
lean_lib «SounioDeGreyChi5Eval3511» where

-- Start of the analytic SqrtField ℝ construction (Mathlib-free): RealEq (Cauchy null-difference)
-- proved reflexive + symmetric, plus the obligation ledger enumerating the deferred analytic core
-- (ε-N transitivity, op-congruence, field/order axioms, completeness, constructive sqrt).
-- Discharging the ledger + sqrtField_chi_ge_5 yields χ(ℝ²)≥5. `lake build SounioSqrtFieldReal`.
lean_lib «SounioSqrtFieldReal» where

-- Phase 2b: multiplicative inverse on Cauchy-sequence reals (sequence level):
-- invSeq + inv_cauchy + mul_inv_tendsto + inv_cong. `lake build SounioRealInverseImpl`.
lean_lib «SounioRealInverseImpl» where

-- Phase 2c: canonical ε-eventual order leR + order axioms (le_refl/trans/antisymm/total,
-- add_le_add_right, mul_nonneg, zero_ne_one) at representative level. `lake build SounioRealOrderAxiomsImpl`.
lean_lib «SounioRealOrderAxiomsImpl» where

-- Phase 3: rational Newton √p sequences for p∈{3,5,7,11}: newton_ge_one + newton_sq_tendsto +
-- newton_cauchy (CRUX 2, full convergence). `lake build SounioNewtonSqrtImpl`.
lean_lib «SounioNewtonSqrtImpl» where

-- Phase 2d + 4: assemble RootedField ℝ from the above and fire the de Grey transfer → χ(ℝ²)≥5.
lean_lib «SounioRootedFieldReal» where

-- Multiquadratic linear-independence programme (faithfulness of QF↪ℝ): irrationality core.
-- no_rat_sqrt (squarefree m has no rational √), not_sq_radicand, ofRat_inj, sqrt_radicand_irrational
-- for the minimal support S={3,5,11} radicands {3,5,11,15,33,55,165}. `lake build SounioMultiquadIndep`.
-- Faithfulness bridge: connect indep8 (native rootR/alpha4 basis) to the abstract QF
-- homomorphism phi/evalNum (16 integer coeffs over r i). `lake build SounioMultiquadFaithful`.
lean_lib «SounioMultiquadFaithful» where

-- Single-entry showcase tying together the four pillars (any-RootedField χ≥5 sharp theorem +
-- chi_R2_ge_5_unconditional + indep8 + evalNum_faithful_on_support). `lake build DeGreyChi5Vitrine`.
lean_lib «DeGreyChi5Vitrine» where

-- Erdos-lane vitrine: packages the closed Madore/Q311 χ(ℝ²)≥4 base case, the scoped
-- current {3,5,11} G529 transfer/minimal-support surface, and a separate chi>=6 smoke boundary.
-- Packaging only; no new theorem beyond the imported vitrines. `lake build ErdosVitrine`.
lean_lib «ErdosVitrine» where

-- Reusable finite unit-distance obstruction interface: exact geometry + finite no-k-colouring
-- certificate ⇒ no ambient k-colouring. The chi>=6 plug-in shape is the same with k=5.
lean_lib «SounioFiniteUnitDistanceWitness» where

-- Tiny finite K6/no-5-colouring smoke for the chi>=6 witness interface.
-- This is intentionally not a Euclidean unit-distance theorem.
lean_lib «SounioFiniteUnitDistanceWitnessSmoke» where

-- Tiny exact Euclidean geometry smoke for the chi>=6 promotion contract.
-- This inhabits EuclideanNatEdgeExactGeometry over Rat^2, but attaches no no-5 certificate.
lean_lib «SounioFiniteUnitDistanceEuclideanSmoke» where

lean_lib «SounioMultiquadIndep» where

-- Canonical Mathlib-free `Real × Real` squared-distance formula used by generated
-- public plane promotion gates. `lake build SounioRealPlaneGeometry`.
lean_lib «SounioRealPlaneGeometry» where

-- Parametric multiquadratic framework over `primes : List Nat` (generalises the {3,5,11}
-- ladder of SounioMultiquadIndep). Generic Newton sqrtR, radS/evalS over 2^|S| masks,
-- HasRadicals/IndepMultiquad, base case + inductive engine (sqrt_new, generic inverse,
-- indep_multiquad). Mathlib-free. `lake build SounioMultiquadParam`.
lean_lib «SounioMultiquadParam» where

-- Moser spindle over ℚ(√3,√11) (|S|=2, degree 4): instantiates the parametric framework,
-- proves the 7-vertex / 11-edge exact geometry (dist²=144 at ×12 scale) by decide,
-- ¬3-colourable ⇒ χ≥4, and transfers to χ(ℚ(√3,√11)²) ≥ 4 — the base of Madore's
-- multiquadratic line. The real-plane vitrine normalizes this to the standard dist²=1 unit.
-- `lake build SounioMoserSpindleQ311`.
lean_lib «SounioMoserSpindleQ311» where

-- Madore spindle χ(ℝ²) ≥ 4: embeds the ℚ(√3,√11) spindle into Mathlib-free `Real`.
-- `lake build SounioMoserSpindleQ311Real`.
lean_lib «SounioMoserSpindleQ311Real» where

-- Geometry-only Nat-edge adapter for the normalized Moser/Q311 real-plane embedding.
-- Also closes finite zero-distance separation and exposes the full Euclidean wrapper.
lean_lib «SounioMoserSpindleQ311EuclideanGeometry» where

-- Single-entry showcase for Madore χ(ℝ²) ≥ 4 (indep_3_11 + spindle + field-plane + Real).
-- Light import (no SAT certificate). `lake build MadoreSpindleVitrine`.
lean_lib «MadoreSpindleVitrine» where

-- G529 / de Grey radical-support audit: theoremizes the exact coordinate-table support
-- ({3,5,11}, √7 unused). Heavy (imports SounioDeGreyUnitDistance); build on demand with
-- `lake build SounioDeGreyRadicalSupport`.
lean_lib «SounioDeGreyRadicalSupport» where

-- G529 / de Grey χ≥5 on the fresh-order parametric MultiquadField [11,3,5]: packages the
-- degree-8 witness field, canonical [3,5,11] permutation bridge, radS↔evalNum bridge, and
-- q3511_plane_needs_5_colours (re-export of
-- g529_field_plane_chi_ge_5). Heavy (pulls SounioDeGreyChi5Closed + G529 cert).
-- `lake build SounioDeGreyChi5Param`.
lean_lib «SounioDeGreyChi5Param» where

-- χ≥5 UNCONDITIONAL over Mathlib-free ℝ: feeds the discharged SAT leg
-- (DeGrey529.Closed.not_VColourable) into SounioRootedFieldReal.chi_R2_ge_5, closing the
-- QF↪ℝ gap. chi_R2_ge_5_unconditional: the unit-distance graph on the Cauchy-quotient real
-- plane ℝ×ℝ has no proper 4-colouring — χ(ℝ²)≥5, ZERO hypotheses, no Mathlib, no sorry.
-- NOT a default_target (heavy import: pulls in the 30 MB G529 cert via SounioDeGreyChi5Closed,
-- re-checked under native_decide; `lake build SounioDeGreyChi5Real`).
lean_lib «SounioDeGreyChi5Real» where

-- χ(G529) = 5 EXACT: explicit proper 5-colouring of the de Grey unit-distance graph G529,
-- paired with the SAT-leg proof that no 4-colouring exists (SounioSatG529.g529_not_colourable).
-- Mathlib-free, no sorry. Build on demand with `lake build SounioDeGreyChi529Exact`.
lean_lib «SounioDeGreyChi529Exact» where

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

-- Mercyful Learning scheduler correctness (MIMIC-IV vancomycin TDM line,
-- Task 3): constrained argmin reaches the target (anti-Goodhart
-- sufficiency), Goodhart-trap theorem (necessity), naive toxicity
-- minimizer under-doses. Mathlib-free; abstract theorems pure,
-- concrete MIMIC-IV instance via native_decide. Gate:
-- scripts/ci/mercyful_lean_gate.sh.
@[default_target]
lean_lib «SounioMercyfulScheduler» where

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

-- Stage 4 / Float-Real bridge: ℚ-backed epistemic model (PBoxR)
-- with proven ℚ-level theorems (addR WellFormedR preservation,
-- dominance monotonicity) + explicit Float-to-ℚ bridge function
-- and theorem (5 IEEE-754 axioms + 1 zero_toRat axiom).
-- Build: `lake build SounioPBoxSemantics`.
@[default_target]
lean_lib «SounioPBoxSemantics» where

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

-- Value-carrying Knowledge<T> calculus (V2): fixes the scalar-cell subject-
-- reduction gap. Full mechanized type safety — Progress + Preservation — for the
-- epistemic gradual compilation §5.4 metatheory. Mathlib-free, no sorry.
@[default_target]
lean_lib «EpistemicEffectsV2» where

-- Cayley–Dickson erasure ladder: the native-erasure law ker = 2^(n-1) − 4.
-- Algebra-level exact ℤ-rank of the verified cdSigma product certifies the
-- kernel dimensions L4–L8 (native_decide), matching the runtime float-Gauss
-- measurements in examples/cd_l{8,9,10,11}_projective_measurement.sio; the
-- recurrence / half-dimension law proved in core Lean (omega). The closed
-- form ∀n stays a conjecture (see project_cayley_dickson_erasure_ladder).
@[default_target]
lean_lib «SounioCayleyDicksonErasure» where

-- G₂ / V₂(ℝ⁷) skeleton of Cayley–Dickson zero divisors: exact ℤ-rank companion
-- to examples/g2_octonion_derivations.sio. Certifies dim Der(𝕆)=14=dim G₂,
-- Der(𝕆)⊂so(7), the g₂↪Der(𝕊) lift (Aut(𝕆)=G₂ acts on ZD(𝕊)), 𝕆 no-ZD, and
-- the sedenion S³-fiber ker=4. Manifold homeomorphism / V₂(ℝ⁷) cited only.
@[default_target]
lean_lib «SounioG2Derivations» where

-- CD monomial-automorphism tower — third independent kernel checker (native_decide)
-- for the program's FINITE facts: seam is the unique associator-degree arg-max (n=4,5),
-- octonion alternativity Psi_3, block lemma at n=4 (β=0), and the 168 count. Not built by
-- CI (heavy GL(4,2) sweeps); build on demand `lake build SounioCDTowerAutomorphism`.
lean_lib «SounioCDTowerAutomorphism» where

-- FO oral Css algebraic surface parity (residual §5.4 mathematical half of the
-- FO PK method-science handoff). Import ≡ site ≡ method ≡ call-result by `rfl`;
-- default-seed FO freezes as exact ℚ via native_decide. Companion executable
-- certificate: scripts/research/fo_css_surface_parity_cert.py (CI gate always);
-- lake build optional under FO_CSS_LEAN_BUILD=1.
@[default_target]
lean_lib «SounioFoCssSurfaceParity» where

-- FO residual §5.4 semantic bridge (compiler half intermediate): surfaces
-- desugar to one FoExpr AST; FO var is a function of (AST, seeds). Not a
-- Madaros FO_XFER soundness proof — that remains L2-full open (R4 gates).
-- Companion: scripts/research/fo_surface_transfer_cert.py.
@[default_target]
lean_lib «SounioFoSurfaceTransfer» where

-- FO residual §5.4 L2-fragment: Madaros FO bytecode ops 1–6 stack machine for
-- oral Css. Site ≡ import-expanded ≡ method programs interpret to L1 desugar.
-- Does not prove lower.sio emits them — R4 is the executable L2 witness.
-- Companion: scripts/research/fo_bytecode_fragment_cert.py.
@[default_target]
lean_lib «SounioFoBytecodeFragment» where

-- FO residual §5.4 L2 pure-emit: fo_bc_compile_expr pure fragment (lower.sio
-- ~9358–9367) formalised; compile(cssSite)=cssSiteProg; fo_css expand = site.
-- Companion: scripts/research/fo_emit_pure_cert.py.
@[default_target]
lean_lib «SounioFoEmitPure» where

-- FO residual §5.4 L2 registration fragment: multipass FO_XFER expand for
-- oral Css pure helpers (local ≡ import registry).
-- Companion: scripts/research/fo_registration_fragment_cert.py.
@[default_target]
lean_lib «SounioFoRegistrationFragment» where

-- FO residual §5.4 L2 engine-install fragment: multipass register of oral Css
-- pure helpers (forward/reverse/4-pass).
-- Companion: scripts/research/fo_engine_install_fragment_cert.py.
@[default_target]
lean_lib «SounioFoEngineInstallFragment» where

-- FO residual §5.4 L2 method FO_XFER peel (Pk.css / call-result → cssSite).
-- Companion: scripts/research/fo_method_xfer_fragment_cert.py.
@[default_target]
lean_lib «SounioFoMethodXferFragment» where

-- FO residual §5.4 L2 multi-mod prepass model (local ≡ import ≡ union).
-- Companion: scripts/research/fo_multimod_fragment_cert.py.
@[default_target]
lean_lib «SounioFoMultimodFragment» where

-- Oral Css residual §5.4 closeout: docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md
-- Stack gate: scripts/ci/fo_residual4_stack_gate.sh → ORAL_CSS_RESIDUAL4_CLOSED

-- CD-tower ZD fibers, the (★) lane (2026-07-31/08-01). SounioZDCollapse IMPORTS
-- SounioZDFiberAntisym: the collapse law's `hres` hypothesis is discharged by `star_forall`,
-- so the two must build together or the discharge is not checked.
-- Companions: scripts/research/cd_tower_zd_fiber_{antisymmetry_lemma,l1_reduction,
-- l2_switching,v1_reduction,collapse_l1l2}_contract.py
@[default_target]
lean_lib «SounioZDFiberAntisym» where

-- E5 L4 pure-arithmetic package (F closed under (T)+(C)). Separate from the tip:
-- no import of SounioZDFiberAntisym; safe to build in parallel with E4b tiers.
@[default_target]
lean_lib «SounioZDE5Inductive» where

@[default_target]
lean_lib «SounioZDChi» where

@[default_target]
lean_lib «SounioZDCollapse» where

-- Catalysis-mechanism suite: independent Lean4 proof oracle for two
-- enzyme-kinetics algebraic identities (enzyme conservation, Hill
-- half-saturation). Pure Rat-field algebra; no Mathlib, no other
-- Sounio-lean import — see file header for the dependency rationale.
@[default_target]
lean_lib «SounioCatalysisKinetics» where

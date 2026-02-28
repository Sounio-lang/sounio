import OmegaMathlib.EpistemicMapBound

namespace OmegaMathlib

/-!
# Glycolysis Step 1 — Epistemic Reaction Bounds

Formal verification of epistemic invariants for the hexokinase-catalyzed
phosphorylation (glycolysis step 1):

  Glucose + ATP → Glucose-6-Phosphate + ADP

Cross-references the accumulator bound `delta32 < 96` (proved in
`AccumulatorBounds.lean`) for the epistemic reaction module
(`self-hosted/collections/epistemic_reaction.sio`).

## Epistemic reaction invariant

The reaction primitive tracks three counters per species:
  - `variance` (F): incremented on each provenance merge
  - `provenance_counter` (P): always 0 in the current implementation
  - `confidence` (Q): incremented on each provenance merge

After `rxn_apply`, each product's counters are bumped by 1.
The key invariant: `delta32(variance + 1, 0, confidence + 1) < 96`.
Since `delta32` depends only on modular residues, the bound holds
universally — independent of how many reactions have been applied.

## Modules sharing this bound

| Module | Definition | Proved in |
|--------|------------|-----------|
| Hardware accumulator | `delta32` | `AccumulatorBounds.lean` |
| String interner | `internDelta32` | `InternedChemicalGraph.lean` |
| Epistemic ordered map | `epiMapDelta32` | `EpistemicMapBound.lean` |
| Molecule bond graph | `molBondDelta32` | `EpistemicMapBound.lean` |
| Epistemic reaction | `rxnDelta32` | **this file** |
-/

-- ============================================================================
-- REACTION DELTA32
-- ============================================================================

/-- The reaction module uses `delta32` for per-product and per-reaction
    accumulator residual checks. Same formula as the hardware
    accumulator, the interner, the map, and the molecule bond graph. -/
def rxnDelta32 (f p q : ℕ) : ℕ := delta32 f p q

/-- Reaction-level delta32 is strictly less than 96. -/
theorem rxn_delta32_lt_96 (f p q : ℕ) :
    rxnDelta32 f p q < 96 := by
  simpa [rxnDelta32] using delta32_lt_96 f p q

-- ============================================================================
-- CONFIDENCE AND VARIANCE MONOTONICITY
-- ============================================================================

/-- After `rxn_apply`, product confidence is incremented:
    `confidence_post = confidence_pre + 1 ≥ confidence_pre`. -/
theorem confidence_monotonic (c : ℕ) : c ≤ c + 1 :=
  Nat.le_succ c

/-- After `rxn_apply`, product variance is incremented:
    `variance_post = variance_pre + 1 ≥ variance_pre`. -/
theorem variance_monotonic (v : ℕ) : v ≤ v + 1 :=
  Nat.le_succ v

-- ============================================================================
-- POST-APPLY DELTA32 BOUND
-- ============================================================================

/-- After `rxn_apply` bumps variance and confidence by 1, the delta32
    bound still holds. Trivially true since `delta32 < 96` is universal. -/
theorem rxn_apply_preserves_delta32 (v p q : ℕ) :
    rxnDelta32 (v + 1) p (q + 1) < 96 :=
  rxn_delta32_lt_96 (v + 1) p (q + 1)

/-- The accumulator gap bound is preserved after `rxn_apply`. -/
theorem rxn_apply_preserves_gap_bound (v p q : ℕ) :
    0 ≤ gap (v + 1) p (q + 1) ∧ gap (v + 1) p (q + 1) < 3 :=
  accumulator_bound (v + 1) p (q + 1)

-- ============================================================================
-- HEXOKINASE STEP 1 (GLYCOLYSIS)
-- ============================================================================

/-- Hexokinase step 1 configuration:
    2 reactants (glucose, ATP) each with variance=0, confidence=1.
    After `rxn_apply`, products get variance=1, provenance_counter=0,
    confidence=2. -/
def hexokinaseProductDelta32 : ℕ := rxnDelta32 1 0 2

/-- The hexokinase product delta32 equals 6.
    `delta32(1, 0, 2) = 4*(1%8) + 2*(0%16) + (2%32) = 4 + 0 + 2 = 6`. -/
theorem hexokinase_product_delta32_val :
    hexokinaseProductDelta32 = 6 := by
  simp [hexokinaseProductDelta32, rxnDelta32, delta32]

/-- The hexokinase product delta32 is well under the 96 ceiling. -/
theorem hexokinase_product_delta32_lt_96 :
    hexokinaseProductDelta32 < 96 := by
  simp [hexokinase_product_delta32_val]

/-- The hexokinase reaction-level metadata after apply has the same
    delta32 as products: variance=1, provenance=0, confidence=2. -/
theorem hexokinase_rxn_delta32_val :
    rxnDelta32 1 0 2 = 6 := by
  simp [rxnDelta32, delta32]

/-- Hexokinase preserves the hardware accumulator bound for both
    reaction-level and per-product epistemic metadata. -/
theorem hexokinase_preserves_accumulator_bound :
    0 ≤ gap 1 0 2 ∧ gap 1 0 2 < 3 :=
  accumulator_bound 1 0 2

-- ============================================================================
-- ITERATED REACTIONS
-- ============================================================================

/-- After `n` successive reactions starting from (variance=0, confidence=1),
    each product accumulates (variance=n, confidence=n+1).
    The delta32 bound still holds for any number of reactions. -/
theorem iterated_reaction_preserves_delta32 (n : ℕ) :
    rxnDelta32 n 0 (n + 1) < 96 :=
  rxn_delta32_lt_96 n 0 (n + 1)

/-- The gap bound holds for any number of iterated reactions. -/
theorem iterated_reaction_preserves_gap (n : ℕ) :
    0 ≤ gap n 0 (n + 1) ∧ gap n 0 (n + 1) < 3 :=
  accumulator_bound n 0 (n + 1)

-- ============================================================================
-- CONSERVATION CROSS-REFERENCE
-- ============================================================================

/-- The reaction module's `rxn_verify_conservation` checks
    `rxn_delta_32(variance, 0, confidence) < 96` for both the reaction
    root and every product. This theorem confirms that check can never
    fail — the bound holds universally for all ℕ inputs. -/
theorem rxn_conservation_delta32_always_holds (v q : ℕ) :
    rxnDelta32 v 0 q < 96 :=
  rxn_delta32_lt_96 v 0 q

/-- All five epistemic modules use the same `delta32` formula and share
    the same universal `< 96` bound. -/
theorem all_epistemic_modules_share_bound (f p q : ℕ) :
    rxnDelta32 f p q = epiMapDelta32 f p q ∧
    rxnDelta32 f p q = molBondDelta32 f p q ∧
    rxnDelta32 f p q = delta32 f p q := by
  simp [rxnDelta32, epiMapDelta32, molBondDelta32]

/-- Corollary: the reaction bound theorem and the map/molecule bound
    theorems are instances of the same universal `delta32_lt_96`. -/
theorem rxn_bound_eq_mol_bound (f p q : ℕ) :
    rxnDelta32 f p q < 96 ↔ molBondDelta32 f p q < 96 := by
  simp [rxnDelta32, molBondDelta32]

end OmegaMathlib

import OmegaMathlib.AccumulatorBounds

namespace OmegaMathlib

/-!
# Epistemic Collection Accumulator Bounds

Cross-references the accumulator bound `0 ≤ gap < 3` (proved in
`AccumulatorBounds.lean`) for the epistemic ordered map and the
epistemic molecule bond graph.

Both structures use the same `delta32` formula:
  `delta32(F, P, Q) = 4 * (F % 8) + 2 * (P % 16) + (Q % 32)`

where F = variance, P = provenance counter, Q = confidence counter.
The bound `delta32 < 96` (equivalently `gap < 3`) is inherited
directly from the hardware accumulator proof.
-/

/-- The epistemic ordered map uses `delta32` for its per-entry
    accumulator residual. -/
def epiMapDelta32 (f p q : ℕ) : ℕ := delta32 f p q

/-- The map's delta32 residual is strictly less than 96. -/
theorem epi_map_delta32_lt_96 (f p q : ℕ) :
    epiMapDelta32 f p q < 96 := by
  simpa [epiMapDelta32] using delta32_lt_96 f p q

/-- The epistemic molecule bond graph uses `delta32` for its per-bond
    accumulator residual. -/
def molBondDelta32 (f p q : ℕ) : ℕ := delta32 f p q

/-- Each molecule bond's delta32 residual is strictly less than 96. -/
theorem mol_bond_delta32_lt_96 (f p q : ℕ) :
    molBondDelta32 f p q < 96 := by
  simpa [molBondDelta32] using delta32_lt_96 f p q

/-- Both epistemic collections preserve the hardware accumulator bound:
    `0 ≤ gap(F, P, Q) < 3`. -/
theorem epi_collections_preserve_accumulator_bound (f p q : ℕ) :
    0 ≤ gap f p q ∧ gap f p q < 3 :=
  accumulator_bound f p q

end OmegaMathlib

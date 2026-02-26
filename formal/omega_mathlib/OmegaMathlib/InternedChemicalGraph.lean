import OmegaMathlib.AccumulatorBounds

namespace OmegaMathlib

/-- 1/32-scaled accumulator residual used by the interner chemistry checks. -/
def internDelta32 (fidelity provenance quantum : ℕ) : ℕ :=
  delta32 fidelity provenance quantum

/-- Interner-side residual remains under the strict hardware gate ceiling. -/
theorem intern_delta32_lt_96 (fidelity provenance quantum : ℕ) :
    internDelta32 fidelity provenance quantum < 96 := by
  simpa [internDelta32] using delta32_lt_96 fidelity provenance quantum

/-- Re-export the accumulator soundness bound for interned chemical graphs. -/
theorem intern_preserves_accumulator_bound (fidelity provenance quantum : ℕ) :
    0 ≤ gap fidelity provenance quantum ∧ gap fidelity provenance quantum < 3 := by
  simpa using accumulator_bound fidelity provenance quantum

end OmegaMathlib

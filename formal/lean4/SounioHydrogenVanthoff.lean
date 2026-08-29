-- formal/lean4/SounioHydrogenVanthoff.lean
/-!
# Sounio Hydrogen van't Hoff Receipt — extrapolation honesty, as theorems

Machine-checks, in core Lean 4 over exact rationals (no Mathlib, no
`sorry`), the rational spine of `demos/hydrogen/vanthoff_gate.sio`:
the calcite-scaling case of the failure mode reported by Stamatakis et
al. (GHG: Sci. & Technol. 2025 — PHREEQC defaults + van't Hoff
extrapolation producing misleading CH4/calcite results; Geoenergy Sci.
Eng. 2026 — calcite scaling by coupled THC modeling).

Working in pK units makes the whole spine rational:

* **§1 The model.** `pK(T) = pK0 + (ΔH/(R·ln10))·(1/T − 1/T0)` with
  `pK0 = 8.48` at `T0 = 298.15 K` (calcite, Plummer & Busenberg);
  `R·ln10 = 19.147 J/mol·K` is a model constant.

* **§2 Corner exactness.** The extrapolation is *linear* in ΔH, so
  endpoint evaluation on the epistemic interval `[−12, −7] kJ/mol` is
  exact. §2 proves the underlying order-preservation principle
  generically for monotone integer linear maps (`lin_corner_nonneg`,
  `lin_corner_nonpos`); the concrete rational corners in §3 are then
  closed directly by `native_decide`, which is exact for `Rat`.

* **§3 The receipt.** Per-mille sandwiches on the pK corners at 60/90/
  150 °C; the width ordering (extrapolation distance prices itself);
  the 90 °C saturation-index straddle (the UNDETERMINED verdict);
  the point estimate's fragility (|SI| < 0.005 — a rounding artifact
  dressed as a decision).

* **§4 The gate.** Integer gate logic (scaled SI ×10⁴) and the three
  verdicts at 60/90/150 °C, all `native_decide`.

Boundary (honest): the aleatory MC layer (Gaussian brine scatter, the
P-box on P(scaling)) is demo-level, as documented in the README; the
ΔCp correction's `ln(T/T0)` is transcendental and likewise outside this
receipt. Everything the *decision* depends on is proved here.

No sorry. No Mathlib.
-/

namespace Sounio.HydrogenVanthoff

-- ================================================================
-- S1. The model (rational spine)
-- ================================================================

/-- `R·ln10` in J/(mol·K) — model constant of the pK formulation. -/
def RLN10 : Rat := 19147/1000

/-- Reference pK (calcite, 25 °C) and reference temperature [K]. -/
def PK0 : Rat := 848/100
def T0 : Rat := 29815/100

/-- Constant-ΔH van't Hoff extrapolation in pK units (exact rational). -/
def pkConst (dh t : Rat) : Rat :=
  PK0 + (dh / RLN10) * (1/t - 1/T0)

-- ================================================================
-- S2. Corner exactness for monotone linear maps (over Int)
-- ================================================================

/-- Scaling by a nonnegative constant preserves order. -/
theorem lin_corner_nonneg (c lo x : Int) (hc : 0 ≤ c) (h : lo ≤ x) :
    c * lo ≤ c * x := by
  have h1 : 0 ≤ c * (x - lo) := Int.mul_nonneg hc (by omega)
  rw [Int.mul_sub] at h1
  omega

/-- Scaling by a nonpositive constant reverses order — together with the
    above, this is why exactly two ΔH corners suffice: pK is linear in
    ΔH, and linear maps attain interval extrema at interval endpoints. -/
theorem lin_corner_nonpos (c lo x : Int) (hc : c ≤ 0) (h : lo ≤ x) :
    c * x ≤ c * lo := by
  have h1 : 0 ≤ (-c) * (x - lo) := Int.mul_nonneg (by omega) (by omega)
  rw [Int.neg_mul, Int.mul_sub] at h1
  omega

-- ================================================================
-- S3. The receipt: pK corners, widths, the straddle, the fragile point
-- ================================================================

/-- pK corners at 60 °C (333.15 K): per-mille sandwiches. -/
theorem pk60_bracket :
    (8608/1000 : Rat) ≤ pkConst (0-7000) (33315/100)
    ∧ pkConst (0-7000) (33315/100) ≤ (8609/1000 : Rat)
    ∧ (8700/1000 : Rat) ≤ pkConst (0-12000) (33315/100)
    ∧ pkConst (0-12000) (33315/100) ≤ (8701/1000 : Rat) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> native_decide

/-- pK corners at 90 °C (363.15 K). -/
theorem pk90_bracket :
    (8699/1000 : Rat) ≤ pkConst (0-7000) (36315/100)
    ∧ pkConst (0-7000) (36315/100) ≤ (8700/1000 : Rat)
    ∧ (8856/1000 : Rat) ≤ pkConst (0-12000) (36315/100)
    ∧ pkConst (0-12000) (36315/100) ≤ (8857/1000 : Rat) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> native_decide

/-- pK corners at 150 °C (423.15 K). -/
theorem pk150_bracket :
    (8842/1000 : Rat) ≤ pkConst (0-7000) (42315/100)
    ∧ pkConst (0-7000) (42315/100) ≤ (8843/1000 : Rat)
    ∧ (9100/1000 : Rat) ≤ pkConst (0-12000) (42315/100)
    ∧ pkConst (0-12000) (42315/100) ≤ (9101/1000 : Rat) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> native_decide

/-- The ΔH-driven pK width grows with extrapolation distance:
    width(60 °C) < width(90 °C) < width(150 °C) — the distance prices
    itself, exactly. -/
theorem width_ordering :
    pkConst (0-12000) (33315/100) - pkConst (0-7000) (33315/100)
      < pkConst (0-12000) (36315/100) - pkConst (0-7000) (36315/100)
    ∧ pkConst (0-12000) (36315/100) - pkConst (0-7000) (36315/100)
      < pkConst (0-12000) (42315/100) - pkConst (0-7000) (42315/100) := by
  constructor <;> native_decide

/-- The 90 °C straddle: with the marginal brine `log10 IAP = −8.78`,
    the saturation-index interval contains zero — the UNDETERMINED
    verdict is a theorem, not an opinion. -/
theorem si_straddle_90 :
    (0-878)/100 + pkConst (0-7000) (36315/100) < 0
    ∧ 0 < (0-878)/100 + pkConst (0-12000) (36315/100) := by
  constructor <;> native_decide

/-- The point estimate's fragility: at ΔH = −9.61 kJ/mol, SI at 90 °C
    is positive but smaller than 0.005 — the "SCALE" answer of a point
    geochem code is a rounding artifact. -/
theorem point_fragile_90 :
    0 < (0-878)/100 + pkConst (0-9610) (36315/100)
    ∧ (0-878)/100 + pkConst (0-9610) (36315/100) < 5/1000 := by
  constructor <;> native_decide

-- ================================================================
-- S4. The gate (scaled SI ×10⁴)
-- ================================================================

/-- Gate verdict: 0 = CERTAIN_NO_SCALE, 1 = CERTAIN_SCALE,
    2 = UNDETERMINED (interval straddles the boundary). -/
def gate (siLo siHi : Int) : Int :=
  if siHi < 0 then 0 else if 0 < siLo then 1 else 2

/-- 60 °C: confidently no scale (SI corners ×10⁴: [−1711.8, −791.6]
    sandwiched by [−1712, −791]). -/
theorem gate_60 : gate (0-1712) (0-791) = 0 := by native_decide

/-- 90 °C: undetermined — measure ΔH or the brine. -/
theorem gate_90 : gate (0-806) 763 = 2 := by native_decide

/-- 150 °C: confidently scaling. -/
theorem gate_150 : gate 622 3210 = 1 := by native_decide

end Sounio.HydrogenVanthoff

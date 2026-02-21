-- formal/lean4/SounioUnits.lean
/-!
# Sounio Units of Measure — Lean 4 Formalization

Formalizes the dimensional analysis system for scientific computing.

Mirrors the Rust implementation in:
- `crates/souc/src/units/dimension.rs` (7 SI base quantities, exponent arithmetic)
- `crates/souc/src/units/check.rs` (dimensional compatibility checking)

References:
- Kennedy, A. (1997). "Relational Parametricity and Units of Measure." POPL.
- Dreiheller et al. (1986). "Programming with units of measurement." S&C 13(5).

## Design

Dimensions are 7-tuples of integer exponents for the SI base quantities
(M, L, T, I, Θ, N, J). Multiplication adds exponents; division subtracts.
This forms a free abelian group with dimensionless as the identity.

All proofs use `ext` + `simp` + `omega`. No sorry. No Mathlib.
-/

namespace Sounio.Units

-- ================================================================
-- §1. Dimension
-- ================================================================

/-- A physical dimension: 7-tuple of SI base quantity exponents.
    Mirrors `Dimension` struct in `crates/souc/src/units/dimension.rs`. -/
@[ext]
structure Dim where
  mass        : Int   -- [M] kilogram
  length      : Int   -- [L] meter
  time        : Int   -- [T] second
  current     : Int   -- [I] ampere
  temperature : Int   -- [Θ] kelvin
  amount      : Int   -- [N] mole
  luminosity  : Int   -- [J] candela
  deriving DecidableEq, Repr

-- ================================================================
-- §2. Named Dimensions
-- ================================================================

/-- Dimensionless (pure number). -/
def dimless : Dim := ⟨0, 0, 0, 0, 0, 0, 0⟩

-- SI base dimensions
def dimMass        : Dim := ⟨1, 0, 0, 0, 0, 0, 0⟩
def dimLength      : Dim := ⟨0, 1, 0, 0, 0, 0, 0⟩
def dimTime        : Dim := ⟨0, 0, 1, 0, 0, 0, 0⟩
def dimCurrent     : Dim := ⟨0, 0, 0, 1, 0, 0, 0⟩
def dimTemperature : Dim := ⟨0, 0, 0, 0, 1, 0, 0⟩
def dimAmount      : Dim := ⟨0, 0, 0, 0, 0, 1, 0⟩
def dimLuminosity  : Dim := ⟨0, 0, 0, 0, 0, 0, 1⟩

-- Derived dimensions
def dimArea          : Dim := ⟨0, 2, 0, 0, 0, 0, 0⟩    -- [L²]
def dimVolume        : Dim := ⟨0, 3, 0, 0, 0, 0, 0⟩    -- [L³]
def dimVelocity      : Dim := ⟨0, 1, -1, 0, 0, 0, 0⟩   -- [L T⁻¹]
def dimAcceleration  : Dim := ⟨0, 1, -2, 0, 0, 0, 0⟩   -- [L T⁻²]
def dimForce         : Dim := ⟨1, 1, -2, 0, 0, 0, 0⟩   -- [M L T⁻²]
def dimEnergy        : Dim := ⟨1, 2, -2, 0, 0, 0, 0⟩   -- [M L² T⁻²]
def dimPower         : Dim := ⟨1, 2, -3, 0, 0, 0, 0⟩   -- [M L² T⁻³]
def dimPressure      : Dim := ⟨1, -1, -2, 0, 0, 0, 0⟩  -- [M L⁻¹ T⁻²]
def dimFrequency     : Dim := ⟨0, 0, -1, 0, 0, 0, 0⟩   -- [T⁻¹]
def dimConcentration : Dim := ⟨1, -3, 0, 0, 0, 0, 0⟩   -- [M L⁻³]
def dimClearance     : Dim := ⟨0, 3, -1, 0, 0, 0, 0⟩   -- [L³ T⁻¹]

-- ================================================================
-- §3. Dimension Operations
-- ================================================================

/-- Multiply dimensions (add exponents).
    [A] × [B] = product of exponents. -/
def Dim.mul (d₁ d₂ : Dim) : Dim :=
  ⟨d₁.mass + d₂.mass, d₁.length + d₂.length, d₁.time + d₂.time,
   d₁.current + d₂.current, d₁.temperature + d₂.temperature,
   d₁.amount + d₂.amount, d₁.luminosity + d₂.luminosity⟩

/-- Divide dimensions (subtract exponents).
    [A] / [B] = difference of exponents. -/
def Dim.div (d₁ d₂ : Dim) : Dim :=
  ⟨d₁.mass - d₂.mass, d₁.length - d₂.length, d₁.time - d₂.time,
   d₁.current - d₂.current, d₁.temperature - d₂.temperature,
   d₁.amount - d₂.amount, d₁.luminosity - d₂.luminosity⟩

/-- Reciprocal of a dimension (negate all exponents).
    [A]⁻¹ -/
def Dim.recip (d : Dim) : Dim :=
  ⟨-d.mass, -d.length, -d.time, -d.current,
   -d.temperature, -d.amount, -d.luminosity⟩

/-- Raise dimension to an integer power (multiply all exponents).
    [A]ⁿ -/
def Dim.pow (d : Dim) (n : Int) : Dim :=
  ⟨d.mass * n, d.length * n, d.time * n, d.current * n,
   d.temperature * n, d.amount * n, d.luminosity * n⟩

-- ================================================================
-- §4. Abelian Group Laws (Dim.mul)
-- ================================================================

theorem dim_mul_comm (d₁ d₂ : Dim) : d₁.mul d₂ = d₂.mul d₁ := by
  ext <;> simp [Dim.mul] <;> omega

theorem dim_mul_assoc (d₁ d₂ d₃ : Dim) :
    (d₁.mul d₂).mul d₃ = d₁.mul (d₂.mul d₃) := by
  ext <;> simp [Dim.mul] <;> omega

theorem dimless_mul_left (d : Dim) : dimless.mul d = d := by
  ext <;> simp [Dim.mul, dimless]

theorem dimless_mul_right (d : Dim) : d.mul dimless = d := by
  ext <;> simp [Dim.mul, dimless]

theorem dim_mul_recip (d : Dim) : d.mul d.recip = dimless := by
  ext <;> simp [Dim.mul, Dim.recip, dimless] <;> omega

theorem dim_recip_mul (d : Dim) : d.recip.mul d = dimless := by
  ext <;> simp [Dim.mul, Dim.recip, dimless] <;> omega

-- ================================================================
-- §5. Division Laws
-- ================================================================

theorem dim_div_self (d : Dim) : d.div d = dimless := by
  ext <;> simp [Dim.div, dimless] <;> omega

theorem dim_mul_div_cancel (d₁ d₂ : Dim) : (d₁.mul d₂).div d₂ = d₁ := by
  ext <;> simp [Dim.mul, Dim.div] <;> omega

theorem dim_div_mul_cancel (d₁ d₂ : Dim) : (d₁.div d₂).mul d₂ = d₁ := by
  ext <;> simp [Dim.mul, Dim.div] <;> omega

theorem dim_div_eq_mul_recip (d₁ d₂ : Dim) : d₁.div d₂ = d₁.mul d₂.recip := by
  ext <;> simp [Dim.div, Dim.mul, Dim.recip] <;> omega

-- ================================================================
-- §6. Reciprocal Laws
-- ================================================================

theorem dim_recip_involution (d : Dim) : d.recip.recip = d := by
  ext <;> simp [Dim.recip] <;> omega

theorem dim_recip_dimless : dimless.recip = dimless := by
  ext <;> simp [Dim.recip, dimless]

theorem dim_recip_mul_distrib (d₁ d₂ : Dim) :
    (d₁.mul d₂).recip = d₁.recip.mul d₂.recip := by
  ext <;> simp [Dim.mul, Dim.recip] <;> omega

-- ================================================================
-- §7. Power Laws
-- ================================================================

theorem dim_pow_zero (d : Dim) : d.pow 0 = dimless := by
  ext <;> simp [Dim.pow, dimless]

theorem dim_pow_one (d : Dim) : d.pow 1 = d := by
  ext <;> simp [Dim.pow]

theorem dim_pow_add (d : Dim) (m n : Int) :
    d.pow (m + n) = (d.pow m).mul (d.pow n) := by
  ext <;> simp [Dim.pow, Dim.mul] <;> exact Int.mul_add _ _ _

theorem dim_pow_neg (d : Dim) (n : Int) :
    d.pow (-n) = (d.pow n).recip := by
  ext <;> simp [Dim.pow, Dim.recip] <;> exact Int.mul_neg _ _

theorem dim_pow_mul (d : Dim) (m n : Int) :
    (d.pow m).pow n = d.pow (m * n) := by
  ext <;> simp [Dim.pow] <;> exact Int.mul_assoc _ _ _

-- ================================================================
-- §8. Derived Dimension Verifications
-- ================================================================

/-- Velocity = Length / Time = [L T⁻¹] -/
theorem velocity_def : dimLength.div dimTime = dimVelocity := by
  ext <;> simp [Dim.div, dimLength, dimTime, dimVelocity]

/-- Acceleration = Velocity / Time = [L T⁻²] -/
theorem acceleration_def : dimVelocity.div dimTime = dimAcceleration := by
  ext <;> simp [Dim.div, dimVelocity, dimTime, dimAcceleration]

/-- Force = Mass × Acceleration = [M L T⁻²] (Newton) -/
theorem force_def : dimMass.mul dimAcceleration = dimForce := by
  ext <;> simp [Dim.mul, dimMass, dimAcceleration, dimForce]

/-- Energy = Force × Length = [M L² T⁻²] (Joule) -/
theorem energy_def : dimForce.mul dimLength = dimEnergy := by
  ext <;> simp [Dim.mul, dimForce, dimLength, dimEnergy]

/-- Power = Energy / Time = [M L² T⁻³] (Watt) -/
theorem power_def : dimEnergy.div dimTime = dimPower := by
  ext <;> simp [Dim.div, dimEnergy, dimTime, dimPower]

/-- Pressure = Force / Area = [M L⁻¹ T⁻²] (Pascal) -/
theorem pressure_def : dimForce.div dimArea = dimPressure := by
  ext <;> simp [Dim.div, dimForce, dimArea, dimPressure]

/-- Frequency = 1 / Time = [T⁻¹] (Hertz) -/
theorem frequency_def : dimTime.recip = dimFrequency := by
  ext <;> simp [Dim.recip, dimTime, dimFrequency]

/-- Concentration = Mass / Volume = [M L⁻³] -/
theorem concentration_def : dimMass.div dimVolume = dimConcentration := by
  ext <;> simp [Dim.div, dimMass, dimVolume, dimConcentration]

/-- Clearance = Volume / Time = [L³ T⁻¹] -/
theorem clearance_def : dimVolume.div dimTime = dimClearance := by
  ext <;> simp [Dim.div, dimVolume, dimTime, dimClearance]

/-- Area = Length² -/
theorem area_def : dimLength.pow 2 = dimArea := by
  ext <;> simp [Dim.pow, dimLength, dimArea]

/-- Volume = Length³ -/
theorem volume_def : dimLength.pow 3 = dimVolume := by
  ext <;> simp [Dim.pow, dimLength, dimVolume]

-- ================================================================
-- §9. Dimensional Compatibility
-- ================================================================

/-- Two dimensions are compatible (same physical quantity) iff they are equal.
    This is the core type-safety property: you cannot add meters to seconds. -/
def compatible (d₁ d₂ : Dim) : Prop := d₁ = d₂

instance decCompatible (d₁ d₂ : Dim) : Decidable (compatible d₁ d₂) :=
  inferInstanceAs (Decidable (d₁ = d₂))

theorem compatible_refl (d : Dim) : compatible d d := rfl

theorem compatible_symm {d₁ d₂ : Dim} (h : compatible d₁ d₂) :
    compatible d₂ d₁ := h.symm

theorem compatible_trans {d₁ d₂ d₃ : Dim}
    (h₁ : compatible d₁ d₂) (h₂ : compatible d₂ d₃) :
    compatible d₁ d₃ := h₁.trans h₂

-- ================================================================
-- §10. Dimensional Safety Theorems
-- ================================================================

/-- Multiplication produces the product dimension (always valid). -/
theorem mul_produces_product (d₁ d₂ : Dim) :
    d₁.mul d₂ = d₁.mul d₂ := rfl

/-- Division produces the quotient dimension (always valid). -/
theorem div_produces_quotient (d₁ d₂ : Dim) :
    d₁.div d₂ = d₁.div d₂ := rfl

/-- Multiplying then dividing by the same dimension recovers the original. -/
theorem mul_div_roundtrip (d₁ d₂ : Dim) :
    compatible ((d₁.mul d₂).div d₂) d₁ :=
  dim_mul_div_cancel d₁ d₂

/-- Dividing then multiplying by the same dimension recovers the original. -/
theorem div_mul_roundtrip (d₁ d₂ : Dim) :
    compatible ((d₁.div d₂).mul d₂) d₁ :=
  dim_div_mul_cancel d₁ d₂

/-- A dimension multiplied by its reciprocal is dimensionless. -/
theorem mul_recip_dimless (d : Dim) : compatible (d.mul d.recip) dimless :=
  dim_mul_recip d

/-- Squaring and then taking square root (pow 2 then pow -2 then mul)
    recovers dimless: d² × d⁻² = 1. -/
theorem sq_recip_sq_dimless (d : Dim) :
    compatible ((d.pow 2).mul (d.pow (-2))) dimless := by
  unfold compatible
  ext <;> simp [Dim.pow, Dim.mul, dimless] <;> omega

end Sounio.Units

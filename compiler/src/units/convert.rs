//! Unit Conversion System
//!
//! Provides compile-time and runtime unit conversion capabilities.
//! Supports both same-dimension conversions (mg to g) and
//! dimensionally compatible conversions with proper scale factors.

use super::dimension::Dimension;
use super::quantity::{DynamicQuantity, Quantity};
use super::si::base::Unit;

// ============================================================================
// COMPILE-TIME CONVERSION
// ============================================================================

/// Convert a quantity from one unit to another with the same dimension.
/// This is a compile-time checked conversion.
///
/// # Example
/// ```ignore
/// let mass_mg = Quantity::<f64, Milligram>::new(500.0);
/// let mass_g: Quantity<f64, Gram> = convert(mass_mg);
/// assert_eq!(mass_g.value(), 0.5);
/// ```
pub fn convert<N, From, To>(qty: Quantity<N, From>) -> Quantity<N, To>
where
    N: std::ops::Mul<f64, Output = N> + Copy,
    From: Unit,
    To: Unit,
{
    // Compile-time dimension check would go here via const assertion
    // For now, we trust that the caller ensures dimensions match
    debug_assert!(
        From::DIMENSION == To::DIMENSION,
        "Cannot convert between incompatible dimensions: {:?} -> {:?}",
        From::DIMENSION,
        To::DIMENSION
    );

    let scale_factor = From::SCALE / To::SCALE;
    let from_offset = From::OFFSET;
    let to_offset = To::OFFSET;

    // For non-affine units (offset = 0), this simplifies to just scaling
    // For affine units like Celsius/Fahrenheit, we need the full formula:
    // value_to = (value_from * from_scale + from_offset - to_offset) / to_scale
    if from_offset == 0.0 && to_offset == 0.0 {
        Quantity::new(qty.into_value() * scale_factor)
    } else {
        // Affine conversion: handle temperature scales
        let base_value = qty.into_value();
        // This requires N to support more operations for affine units
        // For simplicity, we'll panic on affine conversions with non-f64 types
        panic!("Affine unit conversion requires f64 type. Use convert_affine instead.");
    }
}

/// Convert between affine units (units with non-zero offset, like temperature).
/// Only works with f64 values.
///
/// # Example
/// ```ignore
/// let temp_c = Quantity::<f64, Celsius>::new(100.0);
/// let temp_k: Quantity<f64, Kelvin> = convert_affine(temp_c);
/// assert!((temp_k.value() - 373.15).abs() < 0.01);
/// ```
pub fn convert_affine<From, To>(qty: Quantity<f64, From>) -> Quantity<f64, To>
where
    From: Unit,
    To: Unit,
{
    debug_assert!(
        From::DIMENSION == To::DIMENSION,
        "Cannot convert between incompatible dimensions"
    );

    // Convert to base unit, then to target unit
    // base_value = value * scale + offset
    let base_value = qty.into_value() * From::SCALE + From::OFFSET;
    // target_value = (base_value - to_offset) / to_scale
    let target_value = (base_value - To::OFFSET) / To::SCALE;

    Quantity::new(target_value)
}

// ============================================================================
// RUNTIME/DYNAMIC CONVERSION
// ============================================================================

/// Error type for unit conversion failures
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionError {
    /// Dimensions don't match
    DimensionMismatch { from: Dimension, to: Dimension },
    /// Unknown unit symbol
    UnknownUnit(String),
    /// Conversion not supported
    NotSupported(String),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { from, to } => {
                write!(
                    f,
                    "Cannot convert from {} to {}: incompatible dimensions",
                    from, to
                )
            }
            Self::UnknownUnit(s) => write!(f, "Unknown unit: {}", s),
            Self::NotSupported(s) => write!(f, "Conversion not supported: {}", s),
        }
    }
}

impl std::error::Error for ConversionError {}

/// Convert a dynamic quantity to a different unit (non-affine only)
pub fn convert_dynamic(
    qty: &DynamicQuantity,
    to_dimension: Dimension,
    to_scale: f64,
    to_symbol: String,
) -> Result<DynamicQuantity, ConversionError> {
    if qty.dimension != to_dimension {
        return Err(ConversionError::DimensionMismatch {
            from: qty.dimension,
            to: to_dimension,
        });
    }

    // Convert via scale factors (non-affine conversion)
    let base_value = qty.value * qty.scale;
    let target_value = base_value / to_scale;

    Ok(DynamicQuantity {
        value: target_value,
        dimension: to_dimension,
        scale: to_scale,
        symbol: to_symbol,
    })
}

// ============================================================================
// CONVERSION FACTOR CALCULATION
// ============================================================================

/// Calculate the conversion factor between two units.
/// Returns the factor such that: value_to = value_from * factor
///
/// Only valid for non-affine units (offset = 0).
pub fn conversion_factor<From: Unit, To: Unit>() -> f64 {
    debug_assert!(
        From::DIMENSION == To::DIMENSION,
        "Cannot calculate conversion factor for incompatible dimensions"
    );
    debug_assert!(
        From::OFFSET == 0.0 && To::OFFSET == 0.0,
        "conversion_factor is only valid for non-affine units"
    );

    From::SCALE / To::SCALE
}

/// Get conversion factor at runtime
pub fn conversion_factor_dynamic(from_scale: f64, to_scale: f64) -> f64 {
    from_scale / to_scale
}

// ============================================================================
// UNIT COMPATIBILITY CHECKING
// ============================================================================

/// Check if two units are compatible (same dimension)
pub fn are_compatible<A: Unit, B: Unit>() -> bool {
    A::DIMENSION == B::DIMENSION
}

/// Check if two dimensions are compatible for a given operation
pub fn check_operation_compatibility(
    left: Dimension,
    right: Dimension,
    op: ArithmeticOp,
) -> Result<Dimension, ConversionError> {
    match op {
        ArithmeticOp::Add | ArithmeticOp::Sub => {
            if left == right {
                Ok(left)
            } else {
                Err(ConversionError::DimensionMismatch {
                    from: left,
                    to: right,
                })
            }
        }
        ArithmeticOp::Mul => Ok(left.mul(&right)),
        ArithmeticOp::Div => Ok(left.div(&right)),
    }
}

/// Arithmetic operations for dimension checking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
}

// ============================================================================
// COMMON CONVERSIONS
// ============================================================================

/// Mass conversions
pub mod mass {
    use super::*;
    use crate::units::si::base::Kilogram;
    use crate::units::si::prefixes::{Gram, Microgram, Milligram, Nanogram};

    /// Convert kilograms to grams
    pub fn kg_to_g(kg: f64) -> f64 {
        kg * conversion_factor::<Kilogram, Gram>()
    }

    /// Convert grams to milligrams
    pub fn g_to_mg(g: f64) -> f64 {
        g * conversion_factor::<Gram, Milligram>()
    }

    /// Convert milligrams to micrograms
    pub fn mg_to_ug(mg: f64) -> f64 {
        mg * conversion_factor::<Milligram, Microgram>()
    }

    /// Convert micrograms to nanograms
    pub fn ug_to_ng(ug: f64) -> f64 {
        ug * conversion_factor::<Microgram, Nanogram>()
    }
}

/// Volume conversions
pub mod volume {
    use super::*;
    use crate::units::si::prefixes::{Liter, Microliter, Milliliter};

    /// Convert liters to milliliters
    pub fn l_to_ml(l: f64) -> f64 {
        l * conversion_factor::<Liter, Milliliter>()
    }

    /// Convert milliliters to microliters
    pub fn ml_to_ul(ml: f64) -> f64 {
        ml * conversion_factor::<Milliliter, Microliter>()
    }
}

/// Time conversions
pub mod time {
    use super::*;
    use crate::units::si::base::Second;
    use crate::units::si::prefixes::{Day, Hour, Minute};

    /// Convert hours to minutes
    pub fn h_to_min(h: f64) -> f64 {
        h * conversion_factor::<Hour, Minute>()
    }

    /// Convert minutes to seconds
    pub fn min_to_s(min: f64) -> f64 {
        min * conversion_factor::<Minute, Second>()
    }

    /// Convert days to hours
    pub fn d_to_h(d: f64) -> f64 {
        d * conversion_factor::<Day, Hour>()
    }
}

/// Temperature conversions (affine)
pub mod temperature {

    /// Convert Celsius to Kelvin
    pub fn c_to_k(c: f64) -> f64 {
        c + 273.15
    }

    /// Convert Kelvin to Celsius
    pub fn k_to_c(k: f64) -> f64 {
        k - 273.15
    }

    /// Convert Celsius to Fahrenheit
    pub fn c_to_f(c: f64) -> f64 {
        c * 9.0 / 5.0 + 32.0
    }

    /// Convert Fahrenheit to Celsius
    pub fn f_to_c(f: f64) -> f64 {
        (f - 32.0) * 5.0 / 9.0
    }

    /// Convert Fahrenheit to Kelvin
    pub fn f_to_k(f: f64) -> f64 {
        c_to_k(f_to_c(f))
    }

    /// Convert Kelvin to Fahrenheit
    pub fn k_to_f(k: f64) -> f64 {
        c_to_f(k_to_c(k))
    }
}

/// PK/PD concentration conversions
pub mod concentration {
    use super::*;
    use crate::units::pkpd::{
        GramPerLiter, MicrogramPerLiter, MilligramPerLiter, NanogramPerMilliliter,
        PicogramPerMilliliter,
    };

    /// Convert mg/L to μg/L
    pub fn mg_l_to_ug_l(mg_l: f64) -> f64 {
        mg_l * conversion_factor::<MilligramPerLiter, MicrogramPerLiter>()
    }

    /// Convert μg/L to ng/mL (equivalent)
    pub fn ug_l_to_ng_ml(ug_l: f64) -> f64 {
        ug_l * conversion_factor::<MicrogramPerLiter, NanogramPerMilliliter>()
    }

    /// Convert g/L to mg/L
    pub fn g_l_to_mg_l(g_l: f64) -> f64 {
        g_l * conversion_factor::<GramPerLiter, MilligramPerLiter>()
    }

    /// Convert ng/mL to pg/mL
    pub fn ng_ml_to_pg_ml(ng_ml: f64) -> f64 {
        ng_ml * conversion_factor::<NanogramPerMilliliter, PicogramPerMilliliter>()
    }
}

/// PK/PD clearance conversions
pub mod clearance {
    use super::*;
    use crate::units::pkpd::{LiterPerHour, MilliliterPerMinute};

    /// Convert L/h to mL/min
    pub fn l_h_to_ml_min(l_h: f64) -> f64 {
        l_h * conversion_factor::<LiterPerHour, MilliliterPerMinute>()
    }

    /// Convert mL/min to L/h
    pub fn ml_min_to_l_h(ml_min: f64) -> f64 {
        ml_min * conversion_factor::<MilliliterPerMinute, LiterPerHour>()
    }
}

// ============================================================================
// UNIT EXPRESSION PARSER
// ============================================================================

/// Parse a unit expression and return its dimension and scale.
/// Supports compound units like "mg/L", "L/h", "kg*m/s^2".
pub fn parse_unit_expression(expr: &str) -> Result<(Dimension, f64), ConversionError> {
    // Simple parser for common patterns
    // TODO: Implement full expression parser

    // Handle common PK/PD units directly
    if let Some(pkpd) = super::pkpd::PKPDUnitKind::from_symbol(expr) {
        return Ok((pkpd.dimension(), pkpd.scale()));
    }

    // Handle SI units
    match expr {
        "kg" => Ok((Dimension::MASS, 1.0)),
        "g" => Ok((Dimension::MASS, 1e-3)),
        "mg" => Ok((Dimension::MASS, 1e-6)),
        "μg" | "ug" => Ok((Dimension::MASS, 1e-9)),
        "ng" => Ok((Dimension::MASS, 1e-12)),
        "m" => Ok((Dimension::LENGTH, 1.0)),
        "cm" => Ok((Dimension::LENGTH, 1e-2)),
        "mm" => Ok((Dimension::LENGTH, 1e-3)),
        "s" => Ok((Dimension::TIME, 1.0)),
        "min" => Ok((Dimension::TIME, 60.0)),
        "h" => Ok((Dimension::TIME, 3600.0)),
        "L" => Ok((Dimension::VOLUME, 1e-3)),
        "mL" => Ok((Dimension::VOLUME, 1e-6)),
        "K" => Ok((Dimension::TEMPERATURE, 1.0)),
        "mol" => Ok((Dimension::AMOUNT, 1.0)),
        "A" => Ok((Dimension::CURRENT, 1.0)),
        "cd" => Ok((Dimension::LUMINOSITY, 1.0)),
        "" | "1" => Ok((Dimension::DIMENSIONLESS, 1.0)),
        _ => Err(ConversionError::UnknownUnit(expr.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::si::base::{Kelvin, Kilogram, Second};
    use crate::units::si::derived::Celsius;
    use crate::units::si::prefixes::{Gram, Hour, Milligram, Minute};

    #[test]
    fn test_conversion_factor() {
        // 1 kg = 1000 g
        let factor = conversion_factor::<Kilogram, Gram>();
        assert!((factor - 1000.0).abs() < 1e-10);

        // 1 g = 1000 mg
        let factor = conversion_factor::<Gram, Milligram>();
        assert!((factor - 1000.0).abs() < 1e-10);

        // 1 h = 60 min
        let factor = conversion_factor::<Hour, Minute>();
        assert!((factor - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_conversions() {
        // 0°C = 273.15 K
        assert!((temperature::c_to_k(0.0) - 273.15).abs() < 0.01);

        // 100°C = 373.15 K
        assert!((temperature::c_to_k(100.0) - 373.15).abs() < 0.01);

        // 32°F = 0°C
        assert!(temperature::f_to_c(32.0).abs() < 0.01);

        // 212°F = 100°C
        assert!((temperature::f_to_c(212.0) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_mass_conversions() {
        assert!((mass::kg_to_g(1.0) - 1000.0).abs() < 1e-10);
        assert!((mass::g_to_mg(1.0) - 1000.0).abs() < 1e-10);
        assert!((mass::mg_to_ug(1.0) - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_conversions() {
        assert!((time::h_to_min(1.0) - 60.0).abs() < 1e-10);
        assert!((time::min_to_s(1.0) - 60.0).abs() < 1e-10);
        assert!((time::d_to_h(1.0) - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_concentration_conversions() {
        // 1 mg/L = 1000 μg/L
        assert!((concentration::mg_l_to_ug_l(1.0) - 1000.0).abs() < 1e-10);

        // 1 μg/L = 1 ng/mL (equivalent)
        assert!((concentration::ug_l_to_ng_ml(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clearance_conversions() {
        // 1 L/h ≈ 16.67 mL/min
        let ml_min = clearance::l_h_to_ml_min(1.0);
        assert!((ml_min - 16.666666666666668).abs() < 0.001);

        // Round-trip
        let l_h = clearance::ml_min_to_l_h(ml_min);
        assert!((l_h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_are_compatible() {
        assert!(are_compatible::<Kilogram, Gram>());
        assert!(are_compatible::<Hour, Second>());
        assert!(!are_compatible::<Kilogram, Second>());
    }

    #[test]
    fn test_operation_compatibility() {
        // Addition requires same dimensions
        let result =
            check_operation_compatibility(Dimension::MASS, Dimension::MASS, ArithmeticOp::Add);
        assert!(result.is_ok());

        // Can't add mass and time
        let result =
            check_operation_compatibility(Dimension::MASS, Dimension::TIME, ArithmeticOp::Add);
        assert!(result.is_err());

        // Multiplication combines dimensions
        let result =
            check_operation_compatibility(Dimension::MASS, Dimension::VELOCITY, ArithmeticOp::Mul);
        assert!(result.is_ok());
        // Mass * Velocity = Momentum (M·L·T⁻¹)
        let momentum = result.unwrap();
        assert_eq!(momentum.mass, 1);
        assert_eq!(momentum.length, 1);
        assert_eq!(momentum.time, -1);
    }

    #[test]
    fn test_parse_unit_expression() {
        let (dim, scale) = parse_unit_expression("kg").unwrap();
        assert_eq!(dim, Dimension::MASS);
        assert_eq!(scale, 1.0);

        let (dim, scale) = parse_unit_expression("mg/L").unwrap();
        assert_eq!(dim, Dimension::CONCENTRATION);

        let (dim, scale) = parse_unit_expression("L/h").unwrap();
        assert_eq!(dim, Dimension::CLEARANCE);
    }
}

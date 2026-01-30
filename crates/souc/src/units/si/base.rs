//! SI Base Units
//!
//! The International System of Units defines 7 base units from which
//! all other units are derived.

use crate::units::dimension::Dimension;

/// Trait for all units of measure
///
/// Every unit has:
/// - A dimension (what physical quantity it measures)
/// - A scale factor (relative to SI base unit)
/// - An offset (for affine units like Celsius)
/// - A symbol and name for display
pub trait Unit: Copy + Clone + Default + std::fmt::Debug {
    /// The dimension of this unit
    const DIMENSION: Dimension;

    /// Scale factor relative to SI base unit
    /// e.g., Kilometer has scale 1000.0 (relative to meter)
    const SCALE: f64;

    /// Offset for affine units (e.g., Celsius has offset 273.15)
    const OFFSET: f64 = 0.0;

    /// Unit symbol (e.g., "kg", "m/s", "mg/mL")
    const SYMBOL: &'static str;

    /// Full name (e.g., "kilogram", "meter per second")
    const NAME: &'static str;

    /// Convert value from this unit to SI base
    fn to_base(value: f64) -> f64 {
        value * Self::SCALE + Self::OFFSET
    }

    /// Convert value from SI base to this unit
    fn from_base(value: f64) -> f64 {
        (value - Self::OFFSET) / Self::SCALE
    }

    /// Get dimension at runtime
    fn dimension() -> Dimension {
        Self::DIMENSION
    }

    /// Get scale at runtime
    fn scale() -> f64 {
        Self::SCALE
    }

    /// Get symbol at runtime
    fn symbol() -> &'static str {
        Self::SYMBOL
    }

    /// Get name at runtime
    fn name() -> &'static str {
        Self::NAME
    }
}

// =============================================================================
// SI Base Units
// =============================================================================

/// Kilogram (kg) - SI base unit of mass
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kilogram;

impl Unit for Kilogram {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "kg";
    const NAME: &'static str = "kilogram";
}

/// Meter (m) - SI base unit of length
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Meter;

impl Unit for Meter {
    const DIMENSION: Dimension = Dimension::LENGTH;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "m";
    const NAME: &'static str = "meter";
}

/// Second (s) - SI base unit of time
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Second;

impl Unit for Second {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "s";
    const NAME: &'static str = "second";
}

/// Ampere (A) - SI base unit of electric current
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Ampere;

impl Unit for Ampere {
    const DIMENSION: Dimension = Dimension::CURRENT;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "A";
    const NAME: &'static str = "ampere";
}

/// Kelvin (K) - SI base unit of thermodynamic temperature
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kelvin;

impl Unit for Kelvin {
    const DIMENSION: Dimension = Dimension::TEMPERATURE;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "K";
    const NAME: &'static str = "kelvin";
}

/// Mole (mol) - SI base unit of amount of substance
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Mole;

impl Unit for Mole {
    const DIMENSION: Dimension = Dimension::AMOUNT;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "mol";
    const NAME: &'static str = "mole";
}

/// Candela (cd) - SI base unit of luminous intensity
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Candela;

impl Unit for Candela {
    const DIMENSION: Dimension = Dimension::LUMINOSITY;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "cd";
    const NAME: &'static str = "candela";
}

/// Dimensionless unit (pure number)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Dimensionless;

impl Unit for Dimensionless {
    const DIMENSION: Dimension = Dimension::DIMENSIONLESS;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "";
    const NAME: &'static str = "dimensionless";
}

// =============================================================================
// Type Aliases
// =============================================================================

pub type Kg = Kilogram;
pub type M = Meter;
pub type S = Second;
pub type A = Ampere;
pub type K = Kelvin;
pub type Mol = Mole;
pub type Cd = Candela;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_si_base_dimensions() {
        assert!(Kilogram::DIMENSION.equals(&Dimension::MASS));
        assert!(Meter::DIMENSION.equals(&Dimension::LENGTH));
        assert!(Second::DIMENSION.equals(&Dimension::TIME));
        assert!(Ampere::DIMENSION.equals(&Dimension::CURRENT));
        assert!(Kelvin::DIMENSION.equals(&Dimension::TEMPERATURE));
        assert!(Mole::DIMENSION.equals(&Dimension::AMOUNT));
        assert!(Candela::DIMENSION.equals(&Dimension::LUMINOSITY));
    }

    #[test]
    fn test_si_base_scale() {
        assert_eq!(Kilogram::SCALE, 1.0);
        assert_eq!(Meter::SCALE, 1.0);
        assert_eq!(Second::SCALE, 1.0);
    }

    #[test]
    fn test_dimensionless() {
        assert!(Dimensionless::DIMENSION.is_dimensionless());
    }
}

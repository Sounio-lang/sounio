//! SI Prefixes and Prefixed Units
//!
//! Provides common SI prefixes (milli-, micro-, kilo-, etc.) and
//! pre-defined prefixed units for mass, length, volume, and time.

use super::base::Unit;
use crate::units::dimension::Dimension;

// =============================================================================
// Mass Units with Prefixes
// =============================================================================

/// Gram (g) - 10⁻³ kg
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gram;

impl Unit for Gram {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1e-3;
    const SYMBOL: &'static str = "g";
    const NAME: &'static str = "gram";
}

/// Milligram (mg) - 10⁻⁶ kg
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Milligram;

impl Unit for Milligram {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1e-6;
    const SYMBOL: &'static str = "mg";
    const NAME: &'static str = "milligram";
}

/// Microgram (μg) - 10⁻⁹ kg
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Microgram;

impl Unit for Microgram {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1e-9;
    const SYMBOL: &'static str = "μg";
    const NAME: &'static str = "microgram";
}

/// Nanogram (ng) - 10⁻¹² kg
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nanogram;

impl Unit for Nanogram {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1e-12;
    const SYMBOL: &'static str = "ng";
    const NAME: &'static str = "nanogram";
}

/// Picogram (pg) - 10⁻¹⁵ kg
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Picogram;

impl Unit for Picogram {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1e-15;
    const SYMBOL: &'static str = "pg";
    const NAME: &'static str = "picogram";
}

/// Metric ton (t) - 10³ kg
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MetricTon;

impl Unit for MetricTon {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1e3;
    const SYMBOL: &'static str = "t";
    const NAME: &'static str = "metric ton";
}

// =============================================================================
// Length Units with Prefixes
// =============================================================================

/// Kilometer (km) - 10³ m
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kilometer;

impl Unit for Kilometer {
    const DIMENSION: Dimension = Dimension::LENGTH;
    const SCALE: f64 = 1e3;
    const SYMBOL: &'static str = "km";
    const NAME: &'static str = "kilometer";
}

/// Centimeter (cm) - 10⁻² m
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Centimeter;

impl Unit for Centimeter {
    const DIMENSION: Dimension = Dimension::LENGTH;
    const SCALE: f64 = 1e-2;
    const SYMBOL: &'static str = "cm";
    const NAME: &'static str = "centimeter";
}

/// Millimeter (mm) - 10⁻³ m
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Millimeter;

impl Unit for Millimeter {
    const DIMENSION: Dimension = Dimension::LENGTH;
    const SCALE: f64 = 1e-3;
    const SYMBOL: &'static str = "mm";
    const NAME: &'static str = "millimeter";
}

/// Micrometer (μm) - 10⁻⁶ m
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Micrometer;

impl Unit for Micrometer {
    const DIMENSION: Dimension = Dimension::LENGTH;
    const SCALE: f64 = 1e-6;
    const SYMBOL: &'static str = "μm";
    const NAME: &'static str = "micrometer";
}

/// Nanometer (nm) - 10⁻⁹ m
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nanometer;

impl Unit for Nanometer {
    const DIMENSION: Dimension = Dimension::LENGTH;
    const SCALE: f64 = 1e-9;
    const SYMBOL: &'static str = "nm";
    const NAME: &'static str = "nanometer";
}

/// Angstrom (Å) - 10⁻¹⁰ m
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Angstrom;

impl Unit for Angstrom {
    const DIMENSION: Dimension = Dimension::LENGTH;
    const SCALE: f64 = 1e-10;
    const SYMBOL: &'static str = "Å";
    const NAME: &'static str = "angstrom";
}

// =============================================================================
// Volume Units
// =============================================================================

/// Cubic meter (m³) - SI derived unit of volume
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CubicMeter;

impl Unit for CubicMeter {
    const DIMENSION: Dimension = Dimension::VOLUME;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "m³";
    const NAME: &'static str = "cubic meter";
}

/// Liter (L) - 10⁻³ m³
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Liter;

impl Unit for Liter {
    const DIMENSION: Dimension = Dimension::VOLUME;
    const SCALE: f64 = 1e-3;
    const SYMBOL: &'static str = "L";
    const NAME: &'static str = "liter";
}

/// Deciliter (dL) - 10⁻⁴ m³
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Deciliter;

impl Unit for Deciliter {
    const DIMENSION: Dimension = Dimension::VOLUME;
    const SCALE: f64 = 1e-4;
    const SYMBOL: &'static str = "dL";
    const NAME: &'static str = "deciliter";
}

/// Milliliter (mL) - 10⁻⁶ m³
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Milliliter;

impl Unit for Milliliter {
    const DIMENSION: Dimension = Dimension::VOLUME;
    const SCALE: f64 = 1e-6;
    const SYMBOL: &'static str = "mL";
    const NAME: &'static str = "milliliter";
}

/// Microliter (μL) - 10⁻⁹ m³
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Microliter;

impl Unit for Microliter {
    const DIMENSION: Dimension = Dimension::VOLUME;
    const SCALE: f64 = 1e-9;
    const SYMBOL: &'static str = "μL";
    const NAME: &'static str = "microliter";
}

/// Nanoliter (nL) - 10⁻¹² m³
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nanoliter;

impl Unit for Nanoliter {
    const DIMENSION: Dimension = Dimension::VOLUME;
    const SCALE: f64 = 1e-12;
    const SYMBOL: &'static str = "nL";
    const NAME: &'static str = "nanoliter";
}

// =============================================================================
// Time Units
// =============================================================================

/// Millisecond (ms) - 10⁻³ s
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Millisecond;

impl Unit for Millisecond {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 1e-3;
    const SYMBOL: &'static str = "ms";
    const NAME: &'static str = "millisecond";
}

/// Microsecond (μs) - 10⁻⁶ s
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Microsecond;

impl Unit for Microsecond {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 1e-6;
    const SYMBOL: &'static str = "μs";
    const NAME: &'static str = "microsecond";
}

/// Nanosecond (ns) - 10⁻⁹ s
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nanosecond;

impl Unit for Nanosecond {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 1e-9;
    const SYMBOL: &'static str = "ns";
    const NAME: &'static str = "nanosecond";
}

/// Minute (min) - 60 s
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Minute;

impl Unit for Minute {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 60.0;
    const SYMBOL: &'static str = "min";
    const NAME: &'static str = "minute";
}

/// Hour (h) - 3600 s
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Hour;

impl Unit for Hour {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 3600.0;
    const SYMBOL: &'static str = "h";
    const NAME: &'static str = "hour";
}

/// Day (d) - 86400 s
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Day;

impl Unit for Day {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 86400.0;
    const SYMBOL: &'static str = "d";
    const NAME: &'static str = "day";
}

/// Week (wk) - 604800 s
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Week;

impl Unit for Week {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 604800.0;
    const SYMBOL: &'static str = "wk";
    const NAME: &'static str = "week";
}

/// Year (yr) - ~31557600 s (365.25 days)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Year;

impl Unit for Year {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 31557600.0;
    const SYMBOL: &'static str = "yr";
    const NAME: &'static str = "year";
}

// =============================================================================
// Area Units
// =============================================================================

/// Square meter (m²)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SquareMeter;

impl Unit for SquareMeter {
    const DIMENSION: Dimension = Dimension::AREA;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "m²";
    const NAME: &'static str = "square meter";
}

/// Square centimeter (cm²)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SquareCentimeter;

impl Unit for SquareCentimeter {
    const DIMENSION: Dimension = Dimension::AREA;
    const SCALE: f64 = 1e-4;
    const SYMBOL: &'static str = "cm²";
    const NAME: &'static str = "square centimeter";
}

// =============================================================================
// Type Aliases
// =============================================================================

pub type G = Gram;
pub type Mg = Milligram;
pub type Ug = Microgram;
pub type Ng = Nanogram;
pub type Pg = Picogram;

pub type Km = Kilometer;
pub type Cm = Centimeter;
pub type Mm = Millimeter;
pub type Um = Micrometer;
pub type Nm = Nanometer;

pub type L = Liter;
pub type DL = Deciliter;
pub type ML = Milliliter;
pub type UL = Microliter;
pub type NL = Nanoliter;

pub type Ms = Millisecond;
pub type Us = Microsecond;
pub type Ns = Nanosecond;
pub type Min = Minute;
pub type H = Hour;
pub type D = Day;
pub type Wk = Week;
pub type Yr = Year;

pub type M2 = SquareMeter;
pub type Cm2 = SquareCentimeter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_conversions() {
        // 1000 mg = 1 g
        let mg_to_g = Milligram::SCALE / Gram::SCALE;
        assert!((mg_to_g - 0.001).abs() < 1e-10);

        // 1000 g = 1 kg
        let g_to_kg = Gram::SCALE / 1.0;
        assert!((g_to_kg - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_volume_conversions() {
        // 1000 mL = 1 L
        let ml_to_l = Milliliter::SCALE / Liter::SCALE;
        assert!((ml_to_l - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_time_conversions() {
        // 60 min = 1 h
        let min_to_h = Minute::SCALE / Hour::SCALE;
        assert!((min_to_h - 1.0 / 60.0).abs() < 1e-10);

        // 24 h = 1 d
        let h_to_d = Hour::SCALE / Day::SCALE;
        assert!((h_to_d - 1.0 / 24.0).abs() < 1e-10);
    }
}

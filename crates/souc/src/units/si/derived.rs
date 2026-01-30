//! SI Derived Units
//!
//! Named SI derived units like Newton, Joule, Watt, Pascal, etc.

use super::base::Unit;
use crate::units::dimension::Dimension;

// =============================================================================
// Mechanical Units
// =============================================================================

/// Newton (N) - SI derived unit of force [kg·m/s²]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Newton;

impl Unit for Newton {
    const DIMENSION: Dimension = Dimension::FORCE;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "N";
    const NAME: &'static str = "newton";
}

/// Joule (J) - SI derived unit of energy [kg·m²/s²]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Joule;

impl Unit for Joule {
    const DIMENSION: Dimension = Dimension::ENERGY;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "J";
    const NAME: &'static str = "joule";
}

/// Kilojoule (kJ)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kilojoule;

impl Unit for Kilojoule {
    const DIMENSION: Dimension = Dimension::ENERGY;
    const SCALE: f64 = 1e3;
    const SYMBOL: &'static str = "kJ";
    const NAME: &'static str = "kilojoule";
}

/// Watt (W) - SI derived unit of power [kg·m²/s³]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Watt;

impl Unit for Watt {
    const DIMENSION: Dimension = Dimension::POWER;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "W";
    const NAME: &'static str = "watt";
}

/// Kilowatt (kW)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kilowatt;

impl Unit for Kilowatt {
    const DIMENSION: Dimension = Dimension::POWER;
    const SCALE: f64 = 1e3;
    const SYMBOL: &'static str = "kW";
    const NAME: &'static str = "kilowatt";
}

/// Pascal (Pa) - SI derived unit of pressure [kg/(m·s²)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Pascal;

impl Unit for Pascal {
    const DIMENSION: Dimension = Dimension::PRESSURE;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "Pa";
    const NAME: &'static str = "pascal";
}

/// Kilopascal (kPa)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kilopascal;

impl Unit for Kilopascal {
    const DIMENSION: Dimension = Dimension::PRESSURE;
    const SCALE: f64 = 1e3;
    const SYMBOL: &'static str = "kPa";
    const NAME: &'static str = "kilopascal";
}

/// Bar - 10⁵ Pa
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bar;

impl Unit for Bar {
    const DIMENSION: Dimension = Dimension::PRESSURE;
    const SCALE: f64 = 1e5;
    const SYMBOL: &'static str = "bar";
    const NAME: &'static str = "bar";
}

/// Atmosphere (atm) - 101325 Pa
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Atmosphere;

impl Unit for Atmosphere {
    const DIMENSION: Dimension = Dimension::PRESSURE;
    const SCALE: f64 = 101325.0;
    const SYMBOL: &'static str = "atm";
    const NAME: &'static str = "atmosphere";
}

/// Hertz (Hz) - SI derived unit of frequency [1/s]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Hertz;

impl Unit for Hertz {
    const DIMENSION: Dimension = Dimension::FREQUENCY;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "Hz";
    const NAME: &'static str = "hertz";
}

/// Kilohertz (kHz)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kilohertz;

impl Unit for Kilohertz {
    const DIMENSION: Dimension = Dimension::FREQUENCY;
    const SCALE: f64 = 1e3;
    const SYMBOL: &'static str = "kHz";
    const NAME: &'static str = "kilohertz";
}

/// Megahertz (MHz)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Megahertz;

impl Unit for Megahertz {
    const DIMENSION: Dimension = Dimension::FREQUENCY;
    const SCALE: f64 = 1e6;
    const SYMBOL: &'static str = "MHz";
    const NAME: &'static str = "megahertz";
}

// =============================================================================
// Electrical Units
// =============================================================================

/// Coulomb (C) - SI derived unit of electric charge [A·s]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Coulomb;

impl Unit for Coulomb {
    const DIMENSION: Dimension = Dimension::CHARGE;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "C";
    const NAME: &'static str = "coulomb";
}

/// Volt (V) - SI derived unit of voltage [kg·m²/(A·s³)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Volt;

impl Unit for Volt {
    const DIMENSION: Dimension = Dimension::VOLTAGE;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "V";
    const NAME: &'static str = "volt";
}

/// Millivolt (mV)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Millivolt;

impl Unit for Millivolt {
    const DIMENSION: Dimension = Dimension::VOLTAGE;
    const SCALE: f64 = 1e-3;
    const SYMBOL: &'static str = "mV";
    const NAME: &'static str = "millivolt";
}

/// Ohm (Ω) - SI derived unit of resistance [kg·m²/(A²·s³)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Ohm;

impl Unit for Ohm {
    const DIMENSION: Dimension = Dimension::RESISTANCE;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "Ω";
    const NAME: &'static str = "ohm";
}

// =============================================================================
// Temperature Units (Affine)
// =============================================================================

/// Celsius (°C) - Temperature with offset from Kelvin
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Celsius;

impl Unit for Celsius {
    const DIMENSION: Dimension = Dimension::TEMPERATURE;
    const SCALE: f64 = 1.0;
    const OFFSET: f64 = 273.15; // K = °C + 273.15
    const SYMBOL: &'static str = "°C";
    const NAME: &'static str = "degree Celsius";
}

/// Fahrenheit (°F) - Temperature with offset and scale
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Fahrenheit;

impl Unit for Fahrenheit {
    const DIMENSION: Dimension = Dimension::TEMPERATURE;
    const SCALE: f64 = 5.0 / 9.0; // K = (°F + 459.67) × 5/9
    const OFFSET: f64 = 459.67 * 5.0 / 9.0; // ~255.37
    const SYMBOL: &'static str = "°F";
    const NAME: &'static str = "degree Fahrenheit";
}

// =============================================================================
// Energy Units (Non-SI)
// =============================================================================

/// Calorie (cal) - 4.184 J
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Calorie;

impl Unit for Calorie {
    const DIMENSION: Dimension = Dimension::ENERGY;
    const SCALE: f64 = 4.184;
    const SYMBOL: &'static str = "cal";
    const NAME: &'static str = "calorie";
}

/// Kilocalorie (kcal) - 4184 J
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Kilocalorie;

impl Unit for Kilocalorie {
    const DIMENSION: Dimension = Dimension::ENERGY;
    const SCALE: f64 = 4184.0;
    const SYMBOL: &'static str = "kcal";
    const NAME: &'static str = "kilocalorie";
}

/// Electronvolt (eV) - ~1.602×10⁻¹⁹ J
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Electronvolt;

impl Unit for Electronvolt {
    const DIMENSION: Dimension = Dimension::ENERGY;
    const SCALE: f64 = 1.602176634e-19;
    const SYMBOL: &'static str = "eV";
    const NAME: &'static str = "electronvolt";
}

// =============================================================================
// Velocity Units
// =============================================================================

/// Meters per second (m/s)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MeterPerSecond;

impl Unit for MeterPerSecond {
    const DIMENSION: Dimension = Dimension::VELOCITY;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "m/s";
    const NAME: &'static str = "meter per second";
}

/// Kilometers per hour (km/h)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct KilometerPerHour;

impl Unit for KilometerPerHour {
    const DIMENSION: Dimension = Dimension::VELOCITY;
    const SCALE: f64 = 1000.0 / 3600.0; // ~0.2778 m/s
    const SYMBOL: &'static str = "km/h";
    const NAME: &'static str = "kilometer per hour";
}

// =============================================================================
// Acceleration Units
// =============================================================================

/// Meters per second squared (m/s²)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MeterPerSecondSquared;

impl Unit for MeterPerSecondSquared {
    const DIMENSION: Dimension = Dimension::ACCELERATION;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "m/s²";
    const NAME: &'static str = "meter per second squared";
}

// =============================================================================
// Molar Units
// =============================================================================

/// Molar (M) - mol/L
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Molar;

impl Unit for Molar {
    const DIMENSION: Dimension = Dimension::MOLAR_CONCENTRATION;
    const SCALE: f64 = 1e3; // mol/L = 10³ mol/m³
    const SYMBOL: &'static str = "M";
    const NAME: &'static str = "molar";
}

/// Millimolar (mM) - mmol/L
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Millimolar;

impl Unit for Millimolar {
    const DIMENSION: Dimension = Dimension::MOLAR_CONCENTRATION;
    const SCALE: f64 = 1.0; // mmol/L = mol/m³
    const SYMBOL: &'static str = "mM";
    const NAME: &'static str = "millimolar";
}

/// Micromolar (μM) - μmol/L
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Micromolar;

impl Unit for Micromolar {
    const DIMENSION: Dimension = Dimension::MOLAR_CONCENTRATION;
    const SCALE: f64 = 1e-3;
    const SYMBOL: &'static str = "μM";
    const NAME: &'static str = "micromolar";
}

/// Nanomolar (nM) - nmol/L
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nanomolar;

impl Unit for Nanomolar {
    const DIMENSION: Dimension = Dimension::MOLAR_CONCENTRATION;
    const SCALE: f64 = 1e-6;
    const SYMBOL: &'static str = "nM";
    const NAME: &'static str = "nanomolar";
}

/// Picomolar (pM) - pmol/L
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Picomolar;

impl Unit for Picomolar {
    const DIMENSION: Dimension = Dimension::MOLAR_CONCENTRATION;
    const SCALE: f64 = 1e-9;
    const SYMBOL: &'static str = "pM";
    const NAME: &'static str = "picomolar";
}

// =============================================================================
// Type Aliases
// =============================================================================

pub type N = Newton;
pub type J = Joule;
pub type KJ = Kilojoule;
pub type W = Watt;
pub type KW = Kilowatt;
pub type Pa = Pascal;
pub type KPa = Kilopascal;
pub type Hz = Hertz;
pub type KHz = Kilohertz;
pub type MHz = Megahertz;
pub type C = Coulomb;
pub type V = Volt;
pub type MV = Millivolt;
pub type DegC = Celsius;
pub type DegF = Fahrenheit;
pub type Cal = Calorie;
pub type Kcal = Kilocalorie;
pub type EV = Electronvolt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derived_dimensions() {
        assert!(Newton::DIMENSION.equals(&Dimension::FORCE));
        assert!(Joule::DIMENSION.equals(&Dimension::ENERGY));
        assert!(Watt::DIMENSION.equals(&Dimension::POWER));
        assert!(Pascal::DIMENSION.equals(&Dimension::PRESSURE));
        assert!(Hertz::DIMENSION.equals(&Dimension::FREQUENCY));
    }

    #[test]
    fn test_temperature_conversion() {
        // 0°C = 273.15 K
        let k = Celsius::to_base(0.0);
        assert!((k - 273.15).abs() < 1e-10);

        // 100°C = 373.15 K
        let k = Celsius::to_base(100.0);
        assert!((k - 373.15).abs() < 1e-10);
    }

    #[test]
    fn test_energy_conversion() {
        // 1 kcal = 4184 J
        let j = Kilocalorie::to_base(1.0);
        assert!((j - 4184.0).abs() < 1e-10);
    }
}

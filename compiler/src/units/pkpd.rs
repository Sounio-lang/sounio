//! PK/PD Domain-Specific Units
//!
//! Pharmacokinetic and pharmacodynamic units commonly used in drug development
//! and clinical pharmacology. These include concentrations, clearances, volumes
//! of distribution, rate constants, and exposure metrics.

use super::dimension::Dimension;
use super::si::base::Unit;

// ============================================================================
// CONCENTRATION UNITS
// ============================================================================

/// Milligram per liter (mg/L) - common plasma concentration unit
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MilligramPerLiter;

impl Unit for MilligramPerLiter {
    const DIMENSION: Dimension = Dimension::CONCENTRATION;
    const SCALE: f64 = 1.0; // Base concentration unit for PK
    const SYMBOL: &'static str = "mg/L";
    const NAME: &'static str = "milligram per liter";
}

/// Microgram per liter (μg/L = ng/mL)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MicrogramPerLiter;

impl Unit for MicrogramPerLiter {
    const DIMENSION: Dimension = Dimension::CONCENTRATION;
    const SCALE: f64 = 1e-3; // 1 μg/L = 0.001 mg/L
    const SYMBOL: &'static str = "μg/L";
    const NAME: &'static str = "microgram per liter";
}

/// Nanogram per milliliter (ng/mL) - equivalent to μg/L
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NanogramPerMilliliter;

impl Unit for NanogramPerMilliliter {
    const DIMENSION: Dimension = Dimension::CONCENTRATION;
    const SCALE: f64 = 1e-3; // 1 ng/mL = 1 μg/L = 0.001 mg/L
    const SYMBOL: &'static str = "ng/mL";
    const NAME: &'static str = "nanogram per milliliter";
}

/// Picogram per milliliter (pg/mL)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PicogramPerMilliliter;

impl Unit for PicogramPerMilliliter {
    const DIMENSION: Dimension = Dimension::CONCENTRATION;
    const SCALE: f64 = 1e-6; // 1 pg/mL = 0.000001 mg/L
    const SYMBOL: &'static str = "pg/mL";
    const NAME: &'static str = "picogram per milliliter";
}

/// Gram per liter (g/L)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GramPerLiter;

impl Unit for GramPerLiter {
    const DIMENSION: Dimension = Dimension::CONCENTRATION;
    const SCALE: f64 = 1e3; // 1 g/L = 1000 mg/L
    const SYMBOL: &'static str = "g/L";
    const NAME: &'static str = "gram per liter";
}

// ============================================================================
// CLEARANCE UNITS
// ============================================================================

/// Liter per hour (L/h) - common clearance unit
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LiterPerHour;

impl Unit for LiterPerHour {
    const DIMENSION: Dimension = Dimension::CLEARANCE;
    const SCALE: f64 = 1.0; // Base clearance unit for PK
    const SYMBOL: &'static str = "L/h";
    const NAME: &'static str = "liter per hour";
}

/// Milliliter per minute (mL/min) - clinical clearance unit
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MilliliterPerMinute;

impl Unit for MilliliterPerMinute {
    const DIMENSION: Dimension = Dimension::CLEARANCE;
    const SCALE: f64 = 0.06; // 1 mL/min = 0.06 L/h
    const SYMBOL: &'static str = "mL/min";
    const NAME: &'static str = "milliliter per minute";
}

/// Liter per hour per kilogram (L/h/kg) - weight-normalized clearance
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LiterPerHourPerKilogram;

impl Unit for LiterPerHourPerKilogram {
    // Dimension: L³·T⁻¹·M⁻¹
    const DIMENSION: Dimension = Dimension {
        mass: -1,
        length: 3,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "L/h/kg";
    const NAME: &'static str = "liter per hour per kilogram";
}

/// Milliliter per minute per kilogram (mL/min/kg)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MilliliterPerMinutePerKilogram;

impl Unit for MilliliterPerMinutePerKilogram {
    const DIMENSION: Dimension = Dimension {
        mass: -1,
        length: 3,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };
    const SCALE: f64 = 0.06; // 1 mL/min/kg = 0.06 L/h/kg
    const SYMBOL: &'static str = "mL/min/kg";
    const NAME: &'static str = "milliliter per minute per kilogram";
}

// ============================================================================
// VOLUME OF DISTRIBUTION UNITS
// ============================================================================

/// Liter (L) - absolute volume of distribution
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LiterVd;

impl Unit for LiterVd {
    const DIMENSION: Dimension = Dimension::VOLUME;
    const SCALE: f64 = 1e-3; // 1 L = 0.001 m³
    const SYMBOL: &'static str = "L";
    const NAME: &'static str = "liter (volume of distribution)";
}

/// Liter per kilogram (L/kg) - weight-normalized Vd
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LiterPerKilogram;

impl Unit for LiterPerKilogram {
    // Dimension: L³·M⁻¹
    const DIMENSION: Dimension = Dimension {
        mass: -1,
        length: 3,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };
    const SCALE: f64 = 1e-3; // 1 L/kg in m³/kg
    const SYMBOL: &'static str = "L/kg";
    const NAME: &'static str = "liter per kilogram";
}

// ============================================================================
// RATE CONSTANT UNITS
// ============================================================================

/// Per hour (h⁻¹) - elimination rate constant
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PerHour;

impl Unit for PerHour {
    const DIMENSION: Dimension = Dimension::FREQUENCY;
    const SCALE: f64 = 1.0 / 3600.0; // 1 h⁻¹ in s⁻¹
    const SYMBOL: &'static str = "h⁻¹";
    const NAME: &'static str = "per hour";
}

/// Per minute (min⁻¹)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PerMinute;

impl Unit for PerMinute {
    const DIMENSION: Dimension = Dimension::FREQUENCY;
    const SCALE: f64 = 1.0 / 60.0; // 1 min⁻¹ in s⁻¹
    const SYMBOL: &'static str = "min⁻¹";
    const NAME: &'static str = "per minute";
}

/// Per day (day⁻¹)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PerDay;

impl Unit for PerDay {
    const DIMENSION: Dimension = Dimension::FREQUENCY;
    const SCALE: f64 = 1.0 / 86400.0; // 1 day⁻¹ in s⁻¹
    const SYMBOL: &'static str = "day⁻¹";
    const NAME: &'static str = "per day";
}

// ============================================================================
// DOSE UNITS
// ============================================================================

/// Milligram (mg) - absolute dose
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MilligramDose;

impl Unit for MilligramDose {
    const DIMENSION: Dimension = Dimension::MASS;
    const SCALE: f64 = 1e-6; // 1 mg = 1e-6 kg
    const SYMBOL: &'static str = "mg";
    const NAME: &'static str = "milligram (dose)";
}

/// Milligram per kilogram (mg/kg) - weight-based dose
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MilligramPerKilogram;

impl Unit for MilligramPerKilogram {
    const DIMENSION: Dimension = Dimension::DIMENSIONLESS; // mg/kg is dimensionless (mass/mass)
    const SCALE: f64 = 1e-6; // ratio
    const SYMBOL: &'static str = "mg/kg";
    const NAME: &'static str = "milligram per kilogram";
}

/// Microgram per kilogram (μg/kg)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MicrogramPerKilogram;

impl Unit for MicrogramPerKilogram {
    const DIMENSION: Dimension = Dimension::DIMENSIONLESS;
    const SCALE: f64 = 1e-9;
    const SYMBOL: &'static str = "μg/kg";
    const NAME: &'static str = "microgram per kilogram";
}

// ============================================================================
// INFUSION RATE UNITS
// ============================================================================

/// Milligram per hour (mg/h) - infusion rate
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MilligramPerHour;

impl Unit for MilligramPerHour {
    // Dimension: M·T⁻¹
    const DIMENSION: Dimension = Dimension {
        mass: 1,
        length: 0,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };
    const SCALE: f64 = 1e-6 / 3600.0; // mg/h in kg/s
    const SYMBOL: &'static str = "mg/h";
    const NAME: &'static str = "milligram per hour";
}

/// Microgram per minute (μg/min)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MicrogramPerMinute;

impl Unit for MicrogramPerMinute {
    const DIMENSION: Dimension = Dimension {
        mass: 1,
        length: 0,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };
    const SCALE: f64 = 1e-9 / 60.0; // μg/min in kg/s
    const SYMBOL: &'static str = "μg/min";
    const NAME: &'static str = "microgram per minute";
}

/// Microgram per kilogram per minute (μg/kg/min) - weight-based infusion
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MicrogramPerKilogramPerMinute;

impl Unit for MicrogramPerKilogramPerMinute {
    // Dimension: T⁻¹ (mass cancels)
    const DIMENSION: Dimension = Dimension::FREQUENCY;
    const SCALE: f64 = 1e-9 / 60.0;
    const SYMBOL: &'static str = "μg/kg/min";
    const NAME: &'static str = "microgram per kilogram per minute";
}

// ============================================================================
// EXPOSURE UNITS (AUC)
// ============================================================================

/// Milligram hour per liter (mg·h/L) - AUC unit
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MilligramHourPerLiter;

impl Unit for MilligramHourPerLiter {
    const DIMENSION: Dimension = Dimension::AUC;
    const SCALE: f64 = 1.0; // Base AUC unit
    const SYMBOL: &'static str = "mg·h/L";
    const NAME: &'static str = "milligram hour per liter";
}

/// Microgram hour per liter (μg·h/L)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MicrogramHourPerLiter;

impl Unit for MicrogramHourPerLiter {
    const DIMENSION: Dimension = Dimension::AUC;
    const SCALE: f64 = 1e-3; // 1 μg·h/L = 0.001 mg·h/L
    const SYMBOL: &'static str = "μg·h/L";
    const NAME: &'static str = "microgram hour per liter";
}

/// Nanogram hour per milliliter (ng·h/mL)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NanogramHourPerMilliliter;

impl Unit for NanogramHourPerMilliliter {
    const DIMENSION: Dimension = Dimension::AUC;
    const SCALE: f64 = 1e-3; // 1 ng·h/mL = 1 μg·h/L
    const SYMBOL: &'static str = "ng·h/mL";
    const NAME: &'static str = "nanogram hour per milliliter";
}

// ============================================================================
// HALF-LIFE UNITS
// ============================================================================

/// Hour (h) - half-life unit
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HourHalfLife;

impl Unit for HourHalfLife {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 3600.0; // 1 h = 3600 s
    const SYMBOL: &'static str = "h";
    const NAME: &'static str = "hour (half-life)";
}

/// Minute (min) - short half-life
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MinuteHalfLife;

impl Unit for MinuteHalfLife {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 60.0;
    const SYMBOL: &'static str = "min";
    const NAME: &'static str = "minute (half-life)";
}

/// Day (d) - long half-life
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DayHalfLife;

impl Unit for DayHalfLife {
    const DIMENSION: Dimension = Dimension::TIME;
    const SCALE: f64 = 86400.0;
    const SYMBOL: &'static str = "d";
    const NAME: &'static str = "day (half-life)";
}

// ============================================================================
// BIOAVAILABILITY (dimensionless)
// ============================================================================

/// Fraction (0-1) - bioavailability
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Fraction;

impl Unit for Fraction {
    const DIMENSION: Dimension = Dimension::DIMENSIONLESS;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "";
    const NAME: &'static str = "fraction";
}

/// Percent (%) - bioavailability as percentage
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Percent;

impl Unit for Percent {
    const DIMENSION: Dimension = Dimension::DIMENSIONLESS;
    const SCALE: f64 = 0.01; // 1% = 0.01
    const SYMBOL: &'static str = "%";
    const NAME: &'static str = "percent";
}

// ============================================================================
// MOLAR CONCENTRATION UNITS (for PD)
// ============================================================================

/// Nanomolar (nM) - common for receptor binding
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Nanomolar;

impl Unit for Nanomolar {
    // Dimension: N·L⁻³ (amount per volume)
    const DIMENSION: Dimension = Dimension {
        mass: 0,
        length: -3,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 1,
        luminosity: 0,
    };
    const SCALE: f64 = 1e-9 * 1e3; // nM in mol/m³
    const SYMBOL: &'static str = "nM";
    const NAME: &'static str = "nanomolar";
}

/// Picomolar (pM) - high-affinity binding
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Picomolar;

impl Unit for Picomolar {
    const DIMENSION: Dimension = Dimension {
        mass: 0,
        length: -3,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 1,
        luminosity: 0,
    };
    const SCALE: f64 = 1e-12 * 1e3; // pM in mol/m³
    const SYMBOL: &'static str = "pM";
    const NAME: &'static str = "picomolar";
}

/// Femtomolar (fM) - ultra-high-affinity
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Femtomolar;

impl Unit for Femtomolar {
    const DIMENSION: Dimension = Dimension {
        mass: 0,
        length: -3,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 1,
        luminosity: 0,
    };
    const SCALE: f64 = 1e-15 * 1e3; // fM in mol/m³
    const SYMBOL: &'static str = "fM";
    const NAME: &'static str = "femtomolar";
}

// ============================================================================
// PD EFFECT UNITS
// ============================================================================

/// EC50/IC50 units - typically same as concentration
pub type EC50Unit = Nanomolar;
pub type IC50Unit = Nanomolar;

/// Hill coefficient (dimensionless)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HillCoefficient;

impl Unit for HillCoefficient {
    const DIMENSION: Dimension = Dimension::DIMENSIONLESS;
    const SCALE: f64 = 1.0;
    const SYMBOL: &'static str = "";
    const NAME: &'static str = "Hill coefficient";
}

/// Emax (maximum effect) - often as fraction or percent
pub type Emax = Fraction;

// ============================================================================
// UNIT REGISTRY
// ============================================================================

/// All PK/PD unit types for dynamic lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PKPDUnitKind {
    // Concentrations
    MilligramPerLiter,
    MicrogramPerLiter,
    NanogramPerMilliliter,
    PicogramPerMilliliter,
    GramPerLiter,
    // Clearances
    LiterPerHour,
    MilliliterPerMinute,
    LiterPerHourPerKilogram,
    MilliliterPerMinutePerKilogram,
    // Volumes
    LiterVd,
    LiterPerKilogram,
    // Rate constants
    PerHour,
    PerMinute,
    PerDay,
    // Doses
    MilligramDose,
    MilligramPerKilogram,
    MicrogramPerKilogram,
    // Infusion rates
    MilligramPerHour,
    MicrogramPerMinute,
    MicrogramPerKilogramPerMinute,
    // AUC
    MilligramHourPerLiter,
    MicrogramHourPerLiter,
    NanogramHourPerMilliliter,
    // Half-life
    HourHalfLife,
    MinuteHalfLife,
    DayHalfLife,
    // Dimensionless
    Fraction,
    Percent,
    // Molar
    Nanomolar,
    Picomolar,
    Femtomolar,
    // PD
    HillCoefficient,
}

impl PKPDUnitKind {
    /// Get the symbol for this unit kind
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::MilligramPerLiter => MilligramPerLiter::SYMBOL,
            Self::MicrogramPerLiter => MicrogramPerLiter::SYMBOL,
            Self::NanogramPerMilliliter => NanogramPerMilliliter::SYMBOL,
            Self::PicogramPerMilliliter => PicogramPerMilliliter::SYMBOL,
            Self::GramPerLiter => GramPerLiter::SYMBOL,
            Self::LiterPerHour => LiterPerHour::SYMBOL,
            Self::MilliliterPerMinute => MilliliterPerMinute::SYMBOL,
            Self::LiterPerHourPerKilogram => LiterPerHourPerKilogram::SYMBOL,
            Self::MilliliterPerMinutePerKilogram => MilliliterPerMinutePerKilogram::SYMBOL,
            Self::LiterVd => LiterVd::SYMBOL,
            Self::LiterPerKilogram => LiterPerKilogram::SYMBOL,
            Self::PerHour => PerHour::SYMBOL,
            Self::PerMinute => PerMinute::SYMBOL,
            Self::PerDay => PerDay::SYMBOL,
            Self::MilligramDose => MilligramDose::SYMBOL,
            Self::MilligramPerKilogram => MilligramPerKilogram::SYMBOL,
            Self::MicrogramPerKilogram => MicrogramPerKilogram::SYMBOL,
            Self::MilligramPerHour => MilligramPerHour::SYMBOL,
            Self::MicrogramPerMinute => MicrogramPerMinute::SYMBOL,
            Self::MicrogramPerKilogramPerMinute => MicrogramPerKilogramPerMinute::SYMBOL,
            Self::MilligramHourPerLiter => MilligramHourPerLiter::SYMBOL,
            Self::MicrogramHourPerLiter => MicrogramHourPerLiter::SYMBOL,
            Self::NanogramHourPerMilliliter => NanogramHourPerMilliliter::SYMBOL,
            Self::HourHalfLife => HourHalfLife::SYMBOL,
            Self::MinuteHalfLife => MinuteHalfLife::SYMBOL,
            Self::DayHalfLife => DayHalfLife::SYMBOL,
            Self::Fraction => Fraction::SYMBOL,
            Self::Percent => Percent::SYMBOL,
            Self::Nanomolar => Nanomolar::SYMBOL,
            Self::Picomolar => Picomolar::SYMBOL,
            Self::Femtomolar => Femtomolar::SYMBOL,
            Self::HillCoefficient => HillCoefficient::SYMBOL,
        }
    }

    /// Get the dimension for this unit kind
    pub fn dimension(&self) -> Dimension {
        match self {
            Self::MilligramPerLiter => MilligramPerLiter::DIMENSION,
            Self::MicrogramPerLiter => MicrogramPerLiter::DIMENSION,
            Self::NanogramPerMilliliter => NanogramPerMilliliter::DIMENSION,
            Self::PicogramPerMilliliter => PicogramPerMilliliter::DIMENSION,
            Self::GramPerLiter => GramPerLiter::DIMENSION,
            Self::LiterPerHour => LiterPerHour::DIMENSION,
            Self::MilliliterPerMinute => MilliliterPerMinute::DIMENSION,
            Self::LiterPerHourPerKilogram => LiterPerHourPerKilogram::DIMENSION,
            Self::MilliliterPerMinutePerKilogram => MilliliterPerMinutePerKilogram::DIMENSION,
            Self::LiterVd => LiterVd::DIMENSION,
            Self::LiterPerKilogram => LiterPerKilogram::DIMENSION,
            Self::PerHour => PerHour::DIMENSION,
            Self::PerMinute => PerMinute::DIMENSION,
            Self::PerDay => PerDay::DIMENSION,
            Self::MilligramDose => MilligramDose::DIMENSION,
            Self::MilligramPerKilogram => MilligramPerKilogram::DIMENSION,
            Self::MicrogramPerKilogram => MicrogramPerKilogram::DIMENSION,
            Self::MilligramPerHour => MilligramPerHour::DIMENSION,
            Self::MicrogramPerMinute => MicrogramPerMinute::DIMENSION,
            Self::MicrogramPerKilogramPerMinute => MicrogramPerKilogramPerMinute::DIMENSION,
            Self::MilligramHourPerLiter => MilligramHourPerLiter::DIMENSION,
            Self::MicrogramHourPerLiter => MicrogramHourPerLiter::DIMENSION,
            Self::NanogramHourPerMilliliter => NanogramHourPerMilliliter::DIMENSION,
            Self::HourHalfLife => HourHalfLife::DIMENSION,
            Self::MinuteHalfLife => MinuteHalfLife::DIMENSION,
            Self::DayHalfLife => DayHalfLife::DIMENSION,
            Self::Fraction => Fraction::DIMENSION,
            Self::Percent => Percent::DIMENSION,
            Self::Nanomolar => Nanomolar::DIMENSION,
            Self::Picomolar => Picomolar::DIMENSION,
            Self::Femtomolar => Femtomolar::DIMENSION,
            Self::HillCoefficient => HillCoefficient::DIMENSION,
        }
    }

    /// Get the scale factor for this unit kind
    pub fn scale(&self) -> f64 {
        match self {
            Self::MilligramPerLiter => MilligramPerLiter::SCALE,
            Self::MicrogramPerLiter => MicrogramPerLiter::SCALE,
            Self::NanogramPerMilliliter => NanogramPerMilliliter::SCALE,
            Self::PicogramPerMilliliter => PicogramPerMilliliter::SCALE,
            Self::GramPerLiter => GramPerLiter::SCALE,
            Self::LiterPerHour => LiterPerHour::SCALE,
            Self::MilliliterPerMinute => MilliliterPerMinute::SCALE,
            Self::LiterPerHourPerKilogram => LiterPerHourPerKilogram::SCALE,
            Self::MilliliterPerMinutePerKilogram => MilliliterPerMinutePerKilogram::SCALE,
            Self::LiterVd => LiterVd::SCALE,
            Self::LiterPerKilogram => LiterPerKilogram::SCALE,
            Self::PerHour => PerHour::SCALE,
            Self::PerMinute => PerMinute::SCALE,
            Self::PerDay => PerDay::SCALE,
            Self::MilligramDose => MilligramDose::SCALE,
            Self::MilligramPerKilogram => MilligramPerKilogram::SCALE,
            Self::MicrogramPerKilogram => MicrogramPerKilogram::SCALE,
            Self::MilligramPerHour => MilligramPerHour::SCALE,
            Self::MicrogramPerMinute => MicrogramPerMinute::SCALE,
            Self::MicrogramPerKilogramPerMinute => MicrogramPerKilogramPerMinute::SCALE,
            Self::MilligramHourPerLiter => MilligramHourPerLiter::SCALE,
            Self::MicrogramHourPerLiter => MicrogramHourPerLiter::SCALE,
            Self::NanogramHourPerMilliliter => NanogramHourPerMilliliter::SCALE,
            Self::HourHalfLife => HourHalfLife::SCALE,
            Self::MinuteHalfLife => MinuteHalfLife::SCALE,
            Self::DayHalfLife => DayHalfLife::SCALE,
            Self::Fraction => Fraction::SCALE,
            Self::Percent => Percent::SCALE,
            Self::Nanomolar => Nanomolar::SCALE,
            Self::Picomolar => Picomolar::SCALE,
            Self::Femtomolar => Femtomolar::SCALE,
            Self::HillCoefficient => HillCoefficient::SCALE,
        }
    }

    /// Parse a unit from its symbol
    pub fn from_symbol(s: &str) -> Option<Self> {
        match s {
            "mg/L" => Some(Self::MilligramPerLiter),
            "μg/L" | "ug/L" => Some(Self::MicrogramPerLiter),
            "ng/mL" => Some(Self::NanogramPerMilliliter),
            "pg/mL" => Some(Self::PicogramPerMilliliter),
            "g/L" => Some(Self::GramPerLiter),
            "L/h" => Some(Self::LiterPerHour),
            "mL/min" => Some(Self::MilliliterPerMinute),
            "L/h/kg" => Some(Self::LiterPerHourPerKilogram),
            "mL/min/kg" => Some(Self::MilliliterPerMinutePerKilogram),
            "L/kg" => Some(Self::LiterPerKilogram),
            "h⁻¹" | "1/h" | "/h" => Some(Self::PerHour),
            "min⁻¹" | "1/min" | "/min" => Some(Self::PerMinute),
            "day⁻¹" | "1/day" | "/day" => Some(Self::PerDay),
            "mg/kg" => Some(Self::MilligramPerKilogram),
            "μg/kg" | "ug/kg" => Some(Self::MicrogramPerKilogram),
            "mg/h" => Some(Self::MilligramPerHour),
            "μg/min" | "ug/min" => Some(Self::MicrogramPerMinute),
            "μg/kg/min" | "ug/kg/min" => Some(Self::MicrogramPerKilogramPerMinute),
            "mg·h/L" | "mg*h/L" => Some(Self::MilligramHourPerLiter),
            "μg·h/L" | "ug·h/L" | "μg*h/L" | "ug*h/L" => Some(Self::MicrogramHourPerLiter),
            "ng·h/mL" | "ng*h/mL" => Some(Self::NanogramHourPerMilliliter),
            "%" => Some(Self::Percent),
            "nM" => Some(Self::Nanomolar),
            "pM" => Some(Self::Picomolar),
            "fM" => Some(Self::Femtomolar),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concentration_dimensions() {
        assert_eq!(MilligramPerLiter::DIMENSION, Dimension::CONCENTRATION);
        assert_eq!(NanogramPerMilliliter::DIMENSION, Dimension::CONCENTRATION);
    }

    #[test]
    fn test_clearance_dimensions() {
        assert_eq!(LiterPerHour::DIMENSION, Dimension::CLEARANCE);
        assert_eq!(MilliliterPerMinute::DIMENSION, Dimension::CLEARANCE);
    }

    #[test]
    fn test_concentration_scales() {
        // ng/mL should equal μg/L
        assert_eq!(NanogramPerMilliliter::SCALE, MicrogramPerLiter::SCALE);
        // 1 g/L = 1000 mg/L
        assert_eq!(GramPerLiter::SCALE / MilligramPerLiter::SCALE, 1000.0);
    }

    #[test]
    fn test_clearance_conversion() {
        // 1 mL/min = 0.06 L/h
        let ml_min_to_l_h = MilliliterPerMinute::SCALE / LiterPerHour::SCALE;
        assert!((ml_min_to_l_h - 0.06).abs() < 1e-10);
    }

    #[test]
    fn test_symbol_parsing() {
        assert_eq!(
            PKPDUnitKind::from_symbol("mg/L"),
            Some(PKPDUnitKind::MilligramPerLiter)
        );
        assert_eq!(
            PKPDUnitKind::from_symbol("ng/mL"),
            Some(PKPDUnitKind::NanogramPerMilliliter)
        );
        assert_eq!(
            PKPDUnitKind::from_symbol("L/h"),
            Some(PKPDUnitKind::LiterPerHour)
        );
        assert_eq!(
            PKPDUnitKind::from_symbol("nM"),
            Some(PKPDUnitKind::Nanomolar)
        );
    }

    #[test]
    fn test_auc_dimensions() {
        assert_eq!(MilligramHourPerLiter::DIMENSION, Dimension::AUC);
        // ng·h/mL = μg·h/L in scale
        assert_eq!(
            NanogramHourPerMilliliter::SCALE,
            MicrogramHourPerLiter::SCALE
        );
    }
}

//! Units of Measure System for Sounio
//!
//! Compile-time dimensional analysis for scientific computing.
//! Prevents errors like Mars Climate Orbiter ($125M lost due to unit mismatch)
//! and drug dosing errors (~7,000 deaths/year in US).
//!
//! # Key Features
//!
//! - **Type-level dimensions**: Compile-time checking of dimensional compatibility
//! - **SI base units**: All 7 SI base quantities (M, L, T, I, Θ, N, J)
//! - **PK/PD units**: Comprehensive pharmacokinetic/pharmacodynamic units
//! - **Unit inference**: Automatic propagation through arithmetic
//! - **Integration with Knowledge types**: Epistemic + Physical type safety
//!
//! # Example
//!
//! ```ignore
//! let dose: Quantity<f64, Milligram> = Quantity::new(500.0);
//! let volume: Quantity<f64, Milliliter> = Quantity::new(10.0);
//! let concentration = dose / volume;  // Quantity<f64, MgPerMl>
//!
//! // Compile error: cannot add different dimensions
//! // let wrong = dose + volume;  // ERROR!
//! ```

pub mod check;
pub mod convert;
pub mod dimension;
pub mod epistemic;
pub mod pkpd;
pub mod quantity;
pub mod si;

// Re-exports
pub use check::{UnitChecker, UnitError, UnitType, UnitVar};
pub use convert::{ArithmeticOp, ConversionError, conversion_factor, convert, convert_affine};
pub use dimension::Dimension;
pub use epistemic::{
    ConfidencePropagation, DynamicQuantifiedKnowledge, QuantifiedKnowledge, WithEpistemic,
};
pub use quantity::{DynamicQuantity, Quantity, WithUnit};
pub use si::base::Unit;

/// Prelude for common imports
pub mod prelude {
    pub use super::dimension::Dimension;
    pub use super::epistemic::{QuantifiedKnowledge, WithEpistemic};
    pub use super::quantity::{Quantity, WithUnit};

    // SI base units
    pub use super::si::base::{
        Ampere, Candela, Dimensionless, Kelvin, Kilogram, Meter, Mole, Second, Unit,
    };

    // SI prefixes
    pub use super::si::prefixes::{
        Centimeter, Day, Deciliter, Gram, Hour, Kilometer, Liter, Microgram, Microliter,
        Micrometer, Milligram, Milliliter, Millimeter, Minute, Nanogram, Nanometer, Picogram,
    };

    // SI derived units
    pub use super::si::derived::{
        Celsius, Fahrenheit, Hertz, Joule, Micromolar, Millimolar, Molar, Nanomolar, Newton,
        Pascal, Watt,
    };

    // PK/PD units
    pub use super::pkpd::{
        Fraction, LiterPerHour, LiterPerKilogram, MicrogramPerLiter, MilligramHourPerLiter,
        MilligramPerKilogram, MilligramPerLiter, MilliliterPerMinute, NanogramHourPerMilliliter,
        NanogramPerMilliliter, PerDay, PerHour, PerMinute, Percent,
    };
}

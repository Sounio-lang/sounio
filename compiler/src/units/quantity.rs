//! The Quantity Type: Numeric Value with Compile-Time Units
//!
//! `Quantity<N, U>` represents a numeric value of type `N` with unit `U`.
//! Unit checking is performed at compile time, preventing dimensional errors.

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::dimension::Dimension;
use super::si::base::Unit;

/// A numeric quantity with compile-time unit checking
///
/// # Type Parameters
///
/// * `N` - The numeric type (f64, f32, etc.)
/// * `U` - The unit type (must implement `Unit`)
///
/// # Examples
///
/// ```ignore
/// let mass: Quantity<f64, Kilogram> = Quantity::new(70.0);
/// let height: Quantity<f64, Meter> = Quantity::new(1.75);
///
/// // Unit mismatch is a compile-time error:
/// // let wrong = mass + height;  // ERROR!
///
/// // Derived units work through multiplication/division:
/// let velocity = distance / time;  // Type inferred from operation
/// ```
#[derive(Clone, Copy)]
pub struct Quantity<N, U: Unit> {
    value: N,
    _unit: PhantomData<U>,
}

impl<N, U: Unit> Quantity<N, U> {
    /// Create a new quantity with the given value
    #[inline]
    pub const fn new(value: N) -> Self {
        Self {
            value,
            _unit: PhantomData,
        }
    }

    /// Get a reference to the raw numeric value
    #[inline]
    pub fn value(&self) -> &N {
        &self.value
    }

    /// Get the raw numeric value (consuming)
    #[inline]
    pub fn into_value(self) -> N {
        self.value
    }

    /// Get the unit symbol
    #[inline]
    pub fn symbol() -> &'static str {
        U::SYMBOL
    }

    /// Get the unit name
    #[inline]
    pub fn unit_name() -> &'static str {
        U::NAME
    }

    /// Get the dimension
    #[inline]
    pub fn dimension() -> Dimension {
        U::DIMENSION
    }
}

// =============================================================================
// Same-Unit Arithmetic
// =============================================================================

/// Addition: same units only
impl<N: Add<Output = N>, U: Unit> Add for Quantity<N, U> {
    type Output = Quantity<N, U>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Quantity::new(self.value + rhs.value)
    }
}

/// Subtraction: same units only
impl<N: Sub<Output = N>, U: Unit> Sub for Quantity<N, U> {
    type Output = Quantity<N, U>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Quantity::new(self.value - rhs.value)
    }
}

/// Negation
impl<N: Neg<Output = N>, U: Unit> Neg for Quantity<N, U> {
    type Output = Quantity<N, U>;

    #[inline]
    fn neg(self) -> Self::Output {
        Quantity::new(-self.value)
    }
}

// =============================================================================
// Scalar Multiplication/Division
// =============================================================================

/// Multiplication by scalar (right)
impl<N: Mul<Output = N> + Copy, U: Unit> Mul<N> for Quantity<N, U> {
    type Output = Quantity<N, U>;

    #[inline]
    fn mul(self, rhs: N) -> Self::Output {
        Quantity::new(self.value * rhs)
    }
}

/// Division by scalar
impl<N: Div<Output = N> + Copy, U: Unit> Div<N> for Quantity<N, U> {
    type Output = Quantity<N, U>;

    #[inline]
    fn div(self, rhs: N) -> Self::Output {
        Quantity::new(self.value / rhs)
    }
}

// =============================================================================
// Comparison
// =============================================================================

impl<N: PartialEq, U: Unit> PartialEq for Quantity<N, U> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<N: Eq, U: Unit> Eq for Quantity<N, U> {}

impl<N: PartialOrd, U: Unit> PartialOrd for Quantity<N, U> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<N: Ord, U: Unit> Ord for Quantity<N, U> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

// =============================================================================
// Display and Debug
// =============================================================================

impl<N: fmt::Display, U: Unit> fmt::Display for Quantity<N, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if U::SYMBOL.is_empty() {
            write!(f, "{}", self.value)
        } else {
            write!(f, "{} {}", self.value, U::SYMBOL)
        }
    }
}

impl<N: fmt::Debug, U: Unit> fmt::Debug for Quantity<N, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Quantity")
            .field("value", &self.value)
            .field("unit", &U::SYMBOL)
            .field("dimension", &U::DIMENSION)
            .finish()
    }
}

// =============================================================================
// Default
// =============================================================================

impl<N: Default, U: Unit> Default for Quantity<N, U> {
    fn default() -> Self {
        Self::new(N::default())
    }
}

// =============================================================================
// Float Operations (f64)
// =============================================================================

impl<U: Unit> Quantity<f64, U> {
    /// Absolute value
    #[inline]
    pub fn abs(self) -> Self {
        Quantity::new(self.value.abs())
    }

    /// Check if zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value == 0.0
    }

    /// Check if positive
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.value > 0.0
    }

    /// Check if negative
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.value < 0.0
    }

    /// Check if NaN
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    /// Check if infinite
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.value.is_infinite()
    }

    /// Check if finite
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    /// Floor
    #[inline]
    pub fn floor(self) -> Self {
        Quantity::new(self.value.floor())
    }

    /// Ceiling
    #[inline]
    pub fn ceil(self) -> Self {
        Quantity::new(self.value.ceil())
    }

    /// Round
    #[inline]
    pub fn round(self) -> Self {
        Quantity::new(self.value.round())
    }

    /// Minimum
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Quantity::new(self.value.min(other.value))
    }

    /// Maximum
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Quantity::new(self.value.max(other.value))
    }

    /// Clamp to range
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Quantity::new(self.value.clamp(min.value, max.value))
    }
}

impl<U: Unit> Quantity<f32, U> {
    /// Absolute value
    #[inline]
    pub fn abs(self) -> Self {
        Quantity::new(self.value.abs())
    }

    /// Check if zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value == 0.0
    }

    /// Check if positive
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.value > 0.0
    }

    /// Check if negative
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.value < 0.0
    }
}

// =============================================================================
// Extension Trait for Creating Quantities
// =============================================================================

/// Extension trait for creating quantities from numeric values
///
/// # Example
///
/// ```ignore
/// use sounio::units::prelude::*;
///
/// let mass = 70.0.with_unit::<Kilogram>();
/// let dose = 500.0.with_unit::<Milligram>();
/// ```
pub trait WithUnit: Sized {
    /// Create a quantity with the specified unit
    fn with_unit<U: Unit>(self) -> Quantity<Self, U>;
}

impl WithUnit for f64 {
    #[inline]
    fn with_unit<U: Unit>(self) -> Quantity<f64, U> {
        Quantity::new(self)
    }
}

impl WithUnit for f32 {
    #[inline]
    fn with_unit<U: Unit>(self) -> Quantity<f32, U> {
        Quantity::new(self)
    }
}

impl WithUnit for i64 {
    #[inline]
    fn with_unit<U: Unit>(self) -> Quantity<i64, U> {
        Quantity::new(self)
    }
}

impl WithUnit for i32 {
    #[inline]
    fn with_unit<U: Unit>(self) -> Quantity<i32, U> {
        Quantity::new(self)
    }
}

// =============================================================================
// Type Aliases
// =============================================================================

/// A dimensionless scalar quantity
pub type Scalar<N> = Quantity<N, super::si::base::Dimensionless>;

/// A scalar f64
pub type ScalarF64 = Scalar<f64>;

// =============================================================================
// Runtime Quantity (for dynamic unit handling)
// =============================================================================

/// A quantity with runtime unit information
///
/// Use this when units are not known at compile time (e.g., from user input).
#[derive(Debug, Clone)]
pub struct DynamicQuantity {
    /// The numeric value
    pub value: f64,
    /// The dimension of the unit
    pub dimension: Dimension,
    /// Scale factor to SI base
    pub scale: f64,
    /// Unit symbol for display
    pub symbol: String,
}

impl DynamicQuantity {
    /// Create a new dynamic quantity
    pub fn new(value: f64, dimension: Dimension, scale: f64, symbol: &str) -> Self {
        Self {
            value,
            dimension,
            scale,
            symbol: symbol.to_string(),
        }
    }

    /// Create from a static quantity
    pub fn from_static<U: Unit>(qty: Quantity<f64, U>) -> Self {
        Self {
            value: qty.value,
            dimension: U::DIMENSION,
            scale: U::SCALE,
            symbol: U::SYMBOL.to_string(),
        }
    }

    /// Convert to SI base units
    pub fn to_si_base(&self) -> f64 {
        self.value * self.scale
    }

    /// Check if compatible with another quantity
    pub fn is_compatible(&self, other: &DynamicQuantity) -> bool {
        self.dimension.equals(&other.dimension)
    }

    /// Convert to another unit (if compatible)
    pub fn convert_to(&self, target_scale: f64) -> Option<f64> {
        Some(self.to_si_base() / target_scale)
    }

    /// Add (must have same dimension)
    pub fn add(&self, other: &DynamicQuantity) -> Option<DynamicQuantity> {
        if !self.is_compatible(other) {
            return None;
        }
        // Convert both to SI base, add, then convert back to self's unit
        let sum_si = self.to_si_base() + other.to_si_base();
        Some(DynamicQuantity {
            value: sum_si / self.scale,
            dimension: self.dimension,
            scale: self.scale,
            symbol: self.symbol.clone(),
        })
    }

    /// Multiply
    pub fn mul(&self, other: &DynamicQuantity) -> DynamicQuantity {
        DynamicQuantity {
            value: self.value * other.value,
            dimension: self.dimension.mul(&other.dimension),
            scale: self.scale * other.scale,
            symbol: format!("{}·{}", self.symbol, other.symbol),
        }
    }

    /// Divide
    pub fn div(&self, other: &DynamicQuantity) -> DynamicQuantity {
        DynamicQuantity {
            value: self.value / other.value,
            dimension: self.dimension.div(&other.dimension),
            scale: self.scale / other.scale,
            symbol: format!("{}/{}", self.symbol, other.symbol),
        }
    }
}

impl fmt::Display for DynamicQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.symbol.is_empty() {
            write!(f, "{}", self.value)
        } else {
            write!(f, "{} {}", self.value, self.symbol)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::si::base::*;
    use crate::units::si::prefixes::*;

    #[test]
    fn test_quantity_creation() {
        let mass: Quantity<f64, Kilogram> = Quantity::new(70.0);
        assert_eq!(*mass.value(), 70.0);
        assert_eq!(Quantity::<f64, Kilogram>::symbol(), "kg");
    }

    #[test]
    fn test_same_unit_addition() {
        let m1: Quantity<f64, Kilogram> = Quantity::new(50.0);
        let m2: Quantity<f64, Kilogram> = Quantity::new(20.0);
        let sum = m1 + m2;
        assert_eq!(*sum.value(), 70.0);
    }

    #[test]
    fn test_scalar_multiplication() {
        let mass: Quantity<f64, Kilogram> = Quantity::new(10.0);
        let doubled = mass * 2.0;
        assert_eq!(*doubled.value(), 20.0);
    }

    #[test]
    fn test_with_unit() {
        let mass = 70.0.with_unit::<Kilogram>();
        assert_eq!(*mass.value(), 70.0);
    }

    #[test]
    fn test_display() {
        let mass: Quantity<f64, Kilogram> = Quantity::new(70.0);
        assert_eq!(format!("{}", mass), "70 kg");

        let dose: Quantity<f64, Milligram> = Quantity::new(500.0);
        assert_eq!(format!("{}", dose), "500 mg");
    }

    #[test]
    fn test_comparison() {
        let m1: Quantity<f64, Kilogram> = Quantity::new(50.0);
        let m2: Quantity<f64, Kilogram> = Quantity::new(70.0);
        assert!(m1 < m2);
        assert!(m2 > m1);

        let m3: Quantity<f64, Kilogram> = Quantity::new(50.0);
        assert_eq!(m1, m3);
    }

    #[test]
    fn test_float_operations() {
        let mass: Quantity<f64, Kilogram> = Quantity::new(-10.0);
        assert_eq!(*mass.abs().value(), 10.0);
        assert!(mass.is_negative());
        assert!(!mass.is_positive());
    }

    #[test]
    fn test_dynamic_quantity() {
        let dose = DynamicQuantity::new(500.0, Dimension::MASS, 1e-6, "mg");
        let volume = DynamicQuantity::new(10.0, Dimension::VOLUME, 1e-6, "mL");

        assert!(!dose.is_compatible(&volume));

        let conc = dose.div(&volume);
        assert!(conc.dimension.equals(&Dimension::CONCENTRATION));
    }
}

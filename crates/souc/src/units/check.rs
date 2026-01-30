//! Unit Checker for Type System Integration
//!
//! This module provides the compile-time unit checking infrastructure
//! that integrates with the Sounio type checker. It tracks units
//! through expressions and ensures dimensional consistency.

use std::collections::HashMap;

use super::convert::ConversionError;
use super::dimension::Dimension;

// ============================================================================
// UNIT TYPE REPRESENTATION
// ============================================================================

/// Represents a unit type in the type system
#[derive(Debug, Clone, PartialEq)]
pub enum UnitType {
    /// A concrete unit with known dimension and scale
    Concrete {
        dimension: Dimension,
        scale: f64,
        offset: f64,
        symbol: String,
    },
    /// A unit variable (for inference)
    Variable(UnitVar),
    /// Dimensionless (no unit)
    Dimensionless,
    /// Product of units
    Product(Box<UnitType>, Box<UnitType>),
    /// Quotient of units
    Quotient(Box<UnitType>, Box<UnitType>),
    /// Power of a unit
    Power(Box<UnitType>, i8),
    /// Error/unknown unit
    Error,
}

impl UnitType {
    /// Create a new concrete unit type
    pub fn concrete(dimension: Dimension, scale: f64, symbol: impl Into<String>) -> Self {
        Self::Concrete {
            dimension,
            scale,
            offset: 0.0,
            symbol: symbol.into(),
        }
    }

    /// Create a unit type from a dimension (using base SI units)
    pub fn from_dimension(dim: Dimension) -> Self {
        if dim.is_dimensionless() {
            Self::Dimensionless
        } else {
            Self::Concrete {
                dimension: dim,
                scale: 1.0,
                offset: 0.0,
                symbol: dim.to_string(),
            }
        }
    }

    /// Get the dimension of this unit type
    pub fn dimension(&self) -> Option<Dimension> {
        match self {
            Self::Concrete { dimension, .. } => Some(*dimension),
            Self::Dimensionless => Some(Dimension::DIMENSIONLESS),
            Self::Product(a, b) => {
                let da = a.dimension()?;
                let db = b.dimension()?;
                Some(da.mul(&db))
            }
            Self::Quotient(a, b) => {
                let da = a.dimension()?;
                let db = b.dimension()?;
                Some(da.div(&db))
            }
            Self::Power(base, exp) => {
                let db = base.dimension()?;
                Some(db.pow(*exp))
            }
            Self::Variable(_) | Self::Error => None,
        }
    }

    /// Check if this unit type is dimensionless
    pub fn is_dimensionless(&self) -> bool {
        match self {
            Self::Dimensionless => true,
            Self::Concrete { dimension, .. } => dimension.is_dimensionless(),
            _ => self
                .dimension()
                .map(|d| d.is_dimensionless())
                .unwrap_or(false),
        }
    }

    /// Check if two unit types are compatible (same dimension)
    pub fn is_compatible(&self, other: &Self) -> bool {
        match (self.dimension(), other.dimension()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// Multiply two unit types
    pub fn multiply(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Dimensionless, u) | (u, Self::Dimensionless) => u.clone(),
            (Self::Error, _) | (_, Self::Error) => Self::Error,
            (a, b) => Self::Product(Box::new(a.clone()), Box::new(b.clone())),
        }
    }

    /// Divide two unit types
    pub fn divide(&self, other: &Self) -> Self {
        match (self, other) {
            (u, Self::Dimensionless) => u.clone(),
            (Self::Error, _) | (_, Self::Error) => Self::Error,
            (a, b) => Self::Quotient(Box::new(a.clone()), Box::new(b.clone())),
        }
    }

    /// Raise to a power
    pub fn power(&self, exp: i8) -> Self {
        match self {
            Self::Dimensionless => Self::Dimensionless,
            Self::Error => Self::Error,
            u if exp == 0 => Self::Dimensionless,
            u if exp == 1 => u.clone(),
            u => Self::Power(Box::new(u.clone()), exp),
        }
    }

    /// Simplify the unit type to a canonical form
    pub fn simplify(&self) -> Self {
        match self {
            Self::Product(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                if let (Some(da), Some(db)) = (sa.dimension(), sb.dimension()) {
                    Self::from_dimension(da.mul(&db))
                } else {
                    Self::Product(Box::new(sa), Box::new(sb))
                }
            }
            Self::Quotient(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                if let (Some(da), Some(db)) = (sa.dimension(), sb.dimension()) {
                    Self::from_dimension(da.div(&db))
                } else {
                    Self::Quotient(Box::new(sa), Box::new(sb))
                }
            }
            Self::Power(base, exp) => {
                let sb = base.simplify();
                if let Some(db) = sb.dimension() {
                    Self::from_dimension(db.pow(*exp))
                } else {
                    Self::Power(Box::new(sb), *exp)
                }
            }
            other => other.clone(),
        }
    }
}

impl std::fmt::Display for UnitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Concrete { symbol, .. } => write!(f, "{}", symbol),
            Self::Variable(v) => write!(f, "?{}", v.0),
            Self::Dimensionless => write!(f, "1"),
            Self::Product(a, b) => write!(f, "({} * {})", a, b),
            Self::Quotient(a, b) => write!(f, "({} / {})", a, b),
            Self::Power(base, exp) => write!(f, "{}^{}", base, exp),
            Self::Error => write!(f, "<error>"),
        }
    }
}

// ============================================================================
// UNIT VARIABLES (for inference)
// ============================================================================

/// A unit variable for unit inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitVar(pub u32);

impl UnitVar {
    /// Create a new unit variable
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

// ============================================================================
// UNIT CHECKER
// ============================================================================

/// The unit checker maintains unit information during type checking
pub struct UnitChecker {
    /// Counter for generating fresh unit variables
    next_var: u32,
    /// Substitution map for unit variables
    substitutions: HashMap<UnitVar, UnitType>,
    /// Known units by name
    known_units: HashMap<String, UnitType>,
    /// Errors encountered during checking
    errors: Vec<UnitError>,
}

impl UnitChecker {
    /// Create a new unit checker with default units registered
    pub fn new() -> Self {
        let mut checker = Self {
            next_var: 0,
            substitutions: HashMap::new(),
            known_units: HashMap::new(),
            errors: Vec::new(),
        };
        checker.register_default_units();
        checker
    }

    /// Register the default SI and PK/PD units
    fn register_default_units(&mut self) {
        // SI base units
        self.register_unit("kg", Dimension::MASS, 1.0);
        self.register_unit("m", Dimension::LENGTH, 1.0);
        self.register_unit("s", Dimension::TIME, 1.0);
        self.register_unit("A", Dimension::CURRENT, 1.0);
        self.register_unit("K", Dimension::TEMPERATURE, 1.0);
        self.register_unit("mol", Dimension::AMOUNT, 1.0);
        self.register_unit("cd", Dimension::LUMINOSITY, 1.0);

        // SI prefixes for mass
        self.register_unit("g", Dimension::MASS, 1e-3);
        self.register_unit("mg", Dimension::MASS, 1e-6);
        self.register_unit("μg", Dimension::MASS, 1e-9);
        self.register_unit("ug", Dimension::MASS, 1e-9);
        self.register_unit("ng", Dimension::MASS, 1e-12);
        self.register_unit("pg", Dimension::MASS, 1e-15);

        // SI prefixes for length
        self.register_unit("km", Dimension::LENGTH, 1e3);
        self.register_unit("cm", Dimension::LENGTH, 1e-2);
        self.register_unit("mm", Dimension::LENGTH, 1e-3);
        self.register_unit("μm", Dimension::LENGTH, 1e-6);
        self.register_unit("um", Dimension::LENGTH, 1e-6);
        self.register_unit("nm", Dimension::LENGTH, 1e-9);

        // Time
        self.register_unit("min", Dimension::TIME, 60.0);
        self.register_unit("h", Dimension::TIME, 3600.0);
        self.register_unit("d", Dimension::TIME, 86400.0);
        self.register_unit("day", Dimension::TIME, 86400.0);

        // Volume
        self.register_unit("L", Dimension::VOLUME, 1e-3);
        self.register_unit("mL", Dimension::VOLUME, 1e-6);
        self.register_unit("μL", Dimension::VOLUME, 1e-9);
        self.register_unit("uL", Dimension::VOLUME, 1e-9);

        // PK/PD units
        self.register_unit("mg/L", Dimension::CONCENTRATION, 1.0);
        self.register_unit("μg/L", Dimension::CONCENTRATION, 1e-3);
        self.register_unit("ug/L", Dimension::CONCENTRATION, 1e-3);
        self.register_unit("ng/mL", Dimension::CONCENTRATION, 1e-3);
        self.register_unit("pg/mL", Dimension::CONCENTRATION, 1e-6);

        self.register_unit("L/h", Dimension::CLEARANCE, 1.0);
        self.register_unit("mL/min", Dimension::CLEARANCE, 0.06);

        self.register_unit("mg·h/L", Dimension::AUC, 1.0);
        self.register_unit("mg*h/L", Dimension::AUC, 1.0);
        self.register_unit("μg·h/L", Dimension::AUC, 1e-3);
        self.register_unit("ng·h/mL", Dimension::AUC, 1e-3);

        // Rate constants
        self.register_unit("h⁻¹", Dimension::FREQUENCY, 1.0 / 3600.0);
        self.register_unit("1/h", Dimension::FREQUENCY, 1.0 / 3600.0);
        self.register_unit("/h", Dimension::FREQUENCY, 1.0 / 3600.0);
        self.register_unit("min⁻¹", Dimension::FREQUENCY, 1.0 / 60.0);
        self.register_unit("1/min", Dimension::FREQUENCY, 1.0 / 60.0);

        // Molar concentrations
        self.register_unit("M", Dimension::MOLAR_CONCENTRATION, 1e3);
        self.register_unit("mM", Dimension::MOLAR_CONCENTRATION, 1.0);
        self.register_unit("μM", Dimension::MOLAR_CONCENTRATION, 1e-3);
        self.register_unit("uM", Dimension::MOLAR_CONCENTRATION, 1e-3);
        self.register_unit("nM", Dimension::MOLAR_CONCENTRATION, 1e-6);
        self.register_unit("pM", Dimension::MOLAR_CONCENTRATION, 1e-9);
        self.register_unit("fM", Dimension::MOLAR_CONCENTRATION, 1e-12);
    }

    /// Register a unit
    pub fn register_unit(&mut self, symbol: &str, dimension: Dimension, scale: f64) {
        self.known_units.insert(
            symbol.to_string(),
            UnitType::Concrete {
                dimension,
                scale,
                offset: 0.0,
                symbol: symbol.to_string(),
            },
        );
    }

    /// Look up a unit by symbol
    pub fn lookup_unit(&self, symbol: &str) -> Option<&UnitType> {
        self.known_units.get(symbol)
    }

    /// Generate a fresh unit variable
    pub fn fresh_var(&mut self) -> UnitVar {
        let var = UnitVar(self.next_var);
        self.next_var += 1;
        var
    }

    /// Apply substitutions to a unit type
    pub fn apply(&self, unit: &UnitType) -> UnitType {
        match unit {
            UnitType::Variable(var) => {
                if let Some(subst) = self.substitutions.get(var) {
                    self.apply(subst)
                } else {
                    unit.clone()
                }
            }
            UnitType::Product(a, b) => {
                UnitType::Product(Box::new(self.apply(a)), Box::new(self.apply(b)))
            }
            UnitType::Quotient(a, b) => {
                UnitType::Quotient(Box::new(self.apply(a)), Box::new(self.apply(b)))
            }
            UnitType::Power(base, exp) => UnitType::Power(Box::new(self.apply(base)), *exp),
            other => other.clone(),
        }
    }

    /// Unify two unit types, updating substitutions
    pub fn unify(&mut self, a: &UnitType, b: &UnitType) -> Result<(), UnitError> {
        let a = self.apply(a);
        let b = self.apply(b);

        match (&a, &b) {
            // Same type
            _ if a == b => Ok(()),

            // Variable unification
            (UnitType::Variable(var), other) | (other, UnitType::Variable(var)) => {
                // Occurs check
                if self.occurs_in(*var, other) {
                    return Err(UnitError::InfiniteType(*var));
                }
                self.substitutions.insert(*var, other.clone());
                Ok(())
            }

            // Dimensionless is compatible with itself
            (UnitType::Dimensionless, UnitType::Dimensionless) => Ok(()),

            // Concrete units must have same dimension
            (
                UnitType::Concrete { dimension: d1, .. },
                UnitType::Concrete { dimension: d2, .. },
            ) if d1 == d2 => Ok(()),

            // Structural unification
            (UnitType::Product(a1, a2), UnitType::Product(b1, b2)) => {
                self.unify(a1, b1)?;
                self.unify(a2, b2)
            }
            (UnitType::Quotient(a1, a2), UnitType::Quotient(b1, b2)) => {
                self.unify(a1, b1)?;
                self.unify(a2, b2)
            }
            (UnitType::Power(a1, e1), UnitType::Power(b1, e2)) if e1 == e2 => self.unify(a1, b1),

            // Dimension compatibility check
            _ => {
                let da = a.dimension();
                let db = b.dimension();
                match (da, db) {
                    (Some(da), Some(db)) if da == db => Ok(()),
                    (Some(da), Some(db)) => Err(UnitError::DimensionMismatch {
                        expected: da,
                        found: db,
                    }),
                    _ => Err(UnitError::Incompatible { left: a, right: b }),
                }
            }
        }
    }

    /// Check if a variable occurs in a unit type (for occurs check)
    fn occurs_in(&self, var: UnitVar, unit: &UnitType) -> bool {
        match unit {
            UnitType::Variable(v) => {
                if *v == var {
                    true
                } else if let Some(subst) = self.substitutions.get(v) {
                    self.occurs_in(var, subst)
                } else {
                    false
                }
            }
            UnitType::Product(a, b) | UnitType::Quotient(a, b) => {
                self.occurs_in(var, a) || self.occurs_in(var, b)
            }
            UnitType::Power(base, _) => self.occurs_in(var, base),
            _ => false,
        }
    }

    /// Check that an addition/subtraction has compatible units
    pub fn check_additive(
        &mut self,
        left: &UnitType,
        right: &UnitType,
    ) -> Result<UnitType, UnitError> {
        self.unify(left, right)?;
        Ok(self.apply(left))
    }

    /// Check a multiplication and return the resulting unit
    pub fn check_multiply(&self, left: &UnitType, right: &UnitType) -> UnitType {
        left.multiply(right).simplify()
    }

    /// Check a division and return the resulting unit
    pub fn check_divide(&self, left: &UnitType, right: &UnitType) -> UnitType {
        left.divide(right).simplify()
    }

    /// Check a power operation
    pub fn check_power(&self, base: &UnitType, exp: i8) -> UnitType {
        base.power(exp).simplify()
    }

    /// Check a square root operation
    pub fn check_sqrt(&mut self, unit: &UnitType) -> Result<UnitType, UnitError> {
        let dim = unit.dimension().ok_or(UnitError::CannotInfer)?;

        // Try to take sqrt - returns None if any exponent is odd
        match dim.sqrt() {
            Some(result_dim) => Ok(UnitType::from_dimension(result_dim)),
            None => Err(UnitError::InvalidRoot {
                dimension: dim,
                root: 2,
            }),
        }
    }

    /// Record an error
    pub fn error(&mut self, err: UnitError) {
        self.errors.push(err);
    }

    /// Get all recorded errors
    pub fn errors(&self) -> &[UnitError] {
        &self.errors
    }

    /// Check if any errors were recorded
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Clear all errors
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }
}

impl Default for UnitChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// UNIT ERRORS
// ============================================================================

/// Errors that can occur during unit checking
#[derive(Debug, Clone, PartialEq)]
pub enum UnitError {
    /// Dimension mismatch in operation
    DimensionMismatch {
        expected: Dimension,
        found: Dimension,
    },
    /// Incompatible unit types
    Incompatible { left: UnitType, right: UnitType },
    /// Unknown unit symbol
    UnknownUnit(String),
    /// Cannot take root of unit with odd exponents
    InvalidRoot { dimension: Dimension, root: i8 },
    /// Infinite type (occurs check failed)
    InfiniteType(UnitVar),
    /// Cannot infer unit
    CannotInfer,
    /// Unit required but not provided
    UnitRequired { context: String },
    /// Conversion error
    Conversion(String),
}

impl std::fmt::Display for UnitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            Self::Incompatible { left, right } => {
                write!(f, "incompatible units: {} and {}", left, right)
            }
            Self::UnknownUnit(s) => write!(f, "unknown unit: {}", s),
            Self::InvalidRoot { dimension, root } => {
                write!(f, "cannot take {}th root of dimension {}", root, dimension)
            }
            Self::InfiniteType(var) => {
                write!(
                    f,
                    "infinite type: unit variable ?{} occurs in its own definition",
                    var.0
                )
            }
            Self::CannotInfer => write!(f, "cannot infer unit"),
            Self::UnitRequired { context } => {
                write!(f, "unit annotation required: {}", context)
            }
            Self::Conversion(msg) => write!(f, "conversion error: {}", msg),
        }
    }
}

impl std::error::Error for UnitError {}

impl From<ConversionError> for UnitError {
    fn from(err: ConversionError) -> Self {
        Self::Conversion(err.to_string())
    }
}

// ============================================================================
// DIAGNOSTIC HELPERS
// ============================================================================

/// Suggestions for fixing unit errors
pub fn suggest_fix(error: &UnitError) -> Option<String> {
    match error {
        UnitError::DimensionMismatch { expected, found } => Some(format!(
            "ensure both operands have dimension {}; found {} instead",
            expected, found
        )),
        UnitError::Incompatible { left, right } => Some(
            "convert one operand to match the other's unit, or use explicit conversion".to_string(),
        ),
        UnitError::UnknownUnit(s) => Some(format!(
            "check spelling of unit '{}'; common units: kg, g, mg, m, s, L, mol",
            s
        )),
        UnitError::InvalidRoot { dimension, root } => Some(format!(
            "dimension {} has odd exponents; cannot take {}th root",
            dimension, root
        )),
        UnitError::UnitRequired { context } => {
            Some("add a unit annotation, e.g., `value: f64<mg/L>`".to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_lookup() {
        let checker = UnitChecker::new();

        let kg = checker.lookup_unit("kg").unwrap();
        assert_eq!(kg.dimension(), Some(Dimension::MASS));

        let mg_l = checker.lookup_unit("mg/L").unwrap();
        assert_eq!(mg_l.dimension(), Some(Dimension::CONCENTRATION));
    }

    #[test]
    fn test_unification_same_dimension() {
        let mut checker = UnitChecker::new();

        let mg = checker.lookup_unit("mg").unwrap().clone();
        let g = checker.lookup_unit("g").unwrap().clone();

        // mg and g have same dimension (mass), should unify
        assert!(checker.unify(&mg, &g).is_ok());
    }

    #[test]
    fn test_unification_different_dimension() {
        let mut checker = UnitChecker::new();

        let kg = checker.lookup_unit("kg").unwrap().clone();
        let s = checker.lookup_unit("s").unwrap().clone();

        // kg and s have different dimensions, should fail
        let result = checker.unify(&kg, &s);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_multiply() {
        let checker = UnitChecker::new();

        let m = UnitType::from_dimension(Dimension::LENGTH);
        let s = UnitType::from_dimension(Dimension::TIME);

        // m / s = velocity
        let velocity = checker.check_divide(&m, &s);
        assert_eq!(velocity.dimension(), Some(Dimension::VELOCITY));
    }

    #[test]
    fn test_variable_unification() {
        let mut checker = UnitChecker::new();

        let var = checker.fresh_var();
        let var_type = UnitType::Variable(var);
        let kg = checker.lookup_unit("kg").unwrap().clone();

        // Unify variable with kg
        assert!(checker.unify(&var_type, &kg).is_ok());

        // Variable should now resolve to kg
        let resolved = checker.apply(&var_type);
        assert_eq!(resolved.dimension(), Some(Dimension::MASS));
    }

    #[test]
    fn test_additive_check() {
        let mut checker = UnitChecker::new();

        let mg = checker.lookup_unit("mg").unwrap().clone();
        let g = checker.lookup_unit("g").unwrap().clone();

        // Can add mg + g (same dimension)
        assert!(checker.check_additive(&mg, &g).is_ok());

        let s = checker.lookup_unit("s").unwrap().clone();

        // Cannot add mg + s (different dimensions)
        let mut checker2 = UnitChecker::new();
        let mg2 = checker2.lookup_unit("mg").unwrap().clone();
        let s2 = checker2.lookup_unit("s").unwrap().clone();
        assert!(checker2.check_additive(&mg2, &s2).is_err());
    }

    #[test]
    fn test_sqrt_dimension() {
        let mut checker = UnitChecker::new();

        // Area (m²) can have sqrt taken
        let area = UnitType::from_dimension(Dimension::AREA);
        let result = checker.check_sqrt(&area);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimension(), Some(Dimension::LENGTH));

        // Velocity (m/s) cannot have sqrt taken (odd time exponent)
        let velocity = UnitType::from_dimension(Dimension::VELOCITY);
        let result = checker.check_sqrt(&velocity);
        assert!(result.is_err());
    }
}

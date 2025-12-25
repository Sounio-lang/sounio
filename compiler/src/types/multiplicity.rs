//! Quantitative Type Theory - Multiplicity System
//!
//! This module implements multiplicities from Quantitative Type Theory (QTT),
//! which track how many times a value is used. This enables:
//! - Erasure: Types with multiplicity 0 exist only at compile-time
//! - Linearity: Types with multiplicity 1 must be used exactly once
//! - Unrestricted: Types with multiplicity ω can be used any number of times
//!
//! For Sounio, this is critical: 15M+ ontological types can have multiplicity 0,
//! meaning they guide type checking but generate zero runtime overhead.

use std::fmt;
use std::ops::{Add, Mul};

/// Multiplicities from Quantitative Type Theory
///
/// These form a semiring with addition and multiplication:
/// - 0 + m = m (zero is additive identity)
/// - 1 * m = m (one is multiplicative identity)
/// - 0 * m = 0 (zero annihilates)
/// - ω + m = ω (omega absorbs addition)
/// - ω * m = ω (omega absorbs multiplication, except 0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Multiplicity {
    /// Zero uses - erased at runtime (compile-time only)
    /// Used for ontological types, phantom types, type-level computation
    Zero,

    /// Exactly one use - linear types
    /// Used for resources that must be consumed exactly once
    One,

    /// Unrestricted uses - can be used any number of times
    /// Traditional functional programming semantics
    #[default]
    Many,
}

impl Multiplicity {
    /// Check if this multiplicity allows runtime representation
    ///
    /// Only `One` and `Many` multiplicities exist at runtime.
    /// `Zero` is erased during compilation.
    #[inline]
    pub fn is_runtime_relevant(&self) -> bool {
        !matches!(self, Multiplicity::Zero)
    }

    /// Check if this multiplicity requires exactly one use
    #[inline]
    pub fn is_linear(&self) -> bool {
        matches!(self, Multiplicity::One)
    }

    /// Check if this multiplicity is erased
    #[inline]
    pub fn is_erased(&self) -> bool {
        matches!(self, Multiplicity::Zero)
    }

    /// Check if this multiplicity allows duplication
    #[inline]
    pub fn is_duplicable(&self) -> bool {
        matches!(self, Multiplicity::Many | Multiplicity::Zero)
    }

    /// Check if this multiplicity allows discarding without use
    #[inline]
    pub fn is_droppable(&self) -> bool {
        matches!(self, Multiplicity::Many | Multiplicity::Zero)
    }

    /// Compute the join (least upper bound) of two multiplicities
    ///
    /// Used when a value might be used through different code paths
    pub fn join(self, other: Self) -> Self {
        use Multiplicity::*;
        match (self, other) {
            (Zero, m) | (m, Zero) => m,
            (Many, _) | (_, Many) => Many,
            (One, One) => One,
        }
    }

    /// Compute the meet (greatest lower bound) of two multiplicities
    ///
    /// Used for intersection of usage requirements
    pub fn meet(self, other: Self) -> Self {
        use Multiplicity::*;
        match (self, other) {
            (Many, m) | (m, Many) => m,
            (Zero, _) | (_, Zero) => Zero,
            (One, One) => One,
        }
    }

    /// Scale a multiplicity (used in function application)
    ///
    /// When applying a function n times to an argument used m times,
    /// the total usage is n * m
    pub fn scale(self, other: Self) -> Self {
        self * other
    }

    /// Check if this multiplicity is at least as permissive as another
    ///
    /// Zero ≤ One ≤ Many in terms of what's allowed
    pub fn permits(&self, required: &Self) -> bool {
        use Multiplicity::*;
        match (self, required) {
            (_, Zero) => true,    // Any multiplicity permits zero uses
            (Zero, _) => false,   // Zero doesn't permit any runtime use
            (Many, _) => true,    // Many permits any usage
            (One, One) => true,   // One permits exactly one use
            (One, Many) => false, // One doesn't permit multiple uses
        }
    }
}

impl fmt::Display for Multiplicity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Multiplicity::Zero => write!(f, "0"),
            Multiplicity::One => write!(f, "1"),
            Multiplicity::Many => write!(f, "ω"),
        }
    }
}

/// Semiring addition for multiplicities
///
/// Represents sequential usage: if used m times then n times, total is m + n
impl Add for Multiplicity {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        use Multiplicity::*;
        match (self, other) {
            (Zero, m) | (m, Zero) => m,
            (Many, _) | (_, Many) => Many,
            (One, One) => Many, // 1 + 1 = ω (more than once = unrestricted)
        }
    }
}

/// Semiring multiplication for multiplicities
///
/// Represents nested usage: if used in context of m, n times each = m * n
impl Mul for Multiplicity {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        use Multiplicity::*;
        match (self, other) {
            (Zero, _) | (_, Zero) => Zero,
            (One, m) | (m, One) => m,
            (Many, Many) => Many,
        }
    }
}

/// A type annotated with its multiplicity
///
/// This wraps any type T with usage information from QTT.
/// QType<T> represents "a value of type T that can be used m times"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QType<T> {
    /// The underlying type
    pub ty: T,
    /// How many times this value can be used
    pub multiplicity: Multiplicity,
}

impl<T> QType<T> {
    /// Create a new quantified type with the given multiplicity
    pub fn new(ty: T, multiplicity: Multiplicity) -> Self {
        QType { ty, multiplicity }
    }

    /// Create an erased type (multiplicity 0)
    pub fn erased(ty: T) -> Self {
        QType {
            ty,
            multiplicity: Multiplicity::Zero,
        }
    }

    /// Create a linear type (multiplicity 1)
    pub fn linear(ty: T) -> Self {
        QType {
            ty,
            multiplicity: Multiplicity::One,
        }
    }

    /// Create an unrestricted type (multiplicity ω)
    pub fn unrestricted(ty: T) -> Self {
        QType {
            ty,
            multiplicity: Multiplicity::Many,
        }
    }

    /// Check if this type exists at runtime
    pub fn is_runtime_relevant(&self) -> bool {
        self.multiplicity.is_runtime_relevant()
    }

    /// Map the inner type while preserving multiplicity
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> QType<U> {
        QType {
            ty: f(self.ty),
            multiplicity: self.multiplicity,
        }
    }

    /// Get a reference to the inner type
    pub fn inner(&self) -> &T {
        &self.ty
    }

    /// Get the inner type, consuming self
    pub fn into_inner(self) -> T {
        self.ty
    }
}

impl<T: fmt::Display> fmt::Display for QType<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}·{}", self.multiplicity, self.ty)
    }
}

/// Context for tracking variable multiplicities during type checking
#[derive(Debug, Clone, Default)]
pub struct MultiplicityContext {
    /// Map from variable names to their multiplicities
    bindings: std::collections::HashMap<String, Multiplicity>,
}

impl MultiplicityContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a variable with the given multiplicity
    pub fn bind(&mut self, name: String, multiplicity: Multiplicity) {
        self.bindings.insert(name, multiplicity);
    }

    /// Look up a variable's multiplicity
    pub fn lookup(&self, name: &str) -> Option<Multiplicity> {
        self.bindings.get(name).copied()
    }

    /// Use a variable, returning updated context
    ///
    /// For linear variables, this consumes the binding.
    /// Returns None if the variable doesn't exist or can't be used.
    pub fn use_var(&mut self, name: &str) -> Option<Multiplicity> {
        let mult = self.bindings.get(name).copied()?;

        match mult {
            Multiplicity::Zero => {
                // Erased variables can be "used" any number of times
                // (they don't actually generate code)
                Some(Multiplicity::Zero)
            }
            Multiplicity::One => {
                // Linear variables must be consumed
                self.bindings.remove(name);
                Some(Multiplicity::One)
            }
            Multiplicity::Many => {
                // Unrestricted variables stay in context
                Some(Multiplicity::Many)
            }
        }
    }

    /// Check if all linear variables have been consumed
    pub fn check_consumed(&self) -> Result<(), Vec<String>> {
        let unconsumed: Vec<_> = self
            .bindings
            .iter()
            .filter(|(_, m)| matches!(m, Multiplicity::One))
            .map(|(n, _)| n.clone())
            .collect();

        if unconsumed.is_empty() {
            Ok(())
        } else {
            Err(unconsumed)
        }
    }

    /// Merge two contexts (for join points like if/else branches)
    pub fn merge(&self, other: &Self) -> Self {
        let mut result = MultiplicityContext::new();

        // Union of all bindings, joining multiplicities for shared vars
        for (name, mult) in &self.bindings {
            let joined = other
                .bindings
                .get(name)
                .map(|m| mult.join(*m))
                .unwrap_or(*mult);
            result.bindings.insert(name.clone(), joined);
        }

        for (name, mult) in &other.bindings {
            if !self.bindings.contains_key(name) {
                result.bindings.insert(name.clone(), *mult);
            }
        }

        result
    }

    /// Scale all multiplicities in context
    pub fn scale(&mut self, factor: Multiplicity) {
        for mult in self.bindings.values_mut() {
            *mult = *mult * factor;
        }
    }
}

/// Errors that can occur during multiplicity checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiplicityError {
    /// Attempted to use a linear variable more than once
    LinearVariableReused { name: String },

    /// Linear variable was not consumed
    LinearVariableUnconsumed { name: String },

    /// Attempted to use an erased variable at runtime
    ErasedVariableUsed { name: String },

    /// Multiplicity mismatch in function application
    MultiplicityMismatch {
        expected: Multiplicity,
        found: Multiplicity,
        context: String,
    },

    /// Cannot duplicate a linear value
    CannotDuplicate { name: String },

    /// Cannot drop a linear value
    CannotDrop { name: String },
}

impl fmt::Display for MultiplicityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiplicityError::LinearVariableReused { name } => {
                write!(f, "linear variable '{}' used more than once", name)
            }
            MultiplicityError::LinearVariableUnconsumed { name } => {
                write!(f, "linear variable '{}' must be consumed", name)
            }
            MultiplicityError::ErasedVariableUsed { name } => {
                write!(f, "erased variable '{}' cannot be used at runtime", name)
            }
            MultiplicityError::MultiplicityMismatch {
                expected,
                found,
                context,
            } => {
                write!(
                    f,
                    "multiplicity mismatch in {}: expected {}, found {}",
                    context, expected, found
                )
            }
            MultiplicityError::CannotDuplicate { name } => {
                write!(f, "cannot duplicate linear value '{}'", name)
            }
            MultiplicityError::CannotDrop { name } => {
                write!(
                    f,
                    "cannot drop linear value '{}' without consuming it",
                    name
                )
            }
        }
    }
}

impl std::error::Error for MultiplicityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplicity_semiring_addition() {
        use Multiplicity::*;

        // Zero is additive identity
        assert_eq!(Zero + Zero, Zero);
        assert_eq!(Zero + One, One);
        assert_eq!(Zero + Many, Many);
        assert_eq!(One + Zero, One);
        assert_eq!(Many + Zero, Many);

        // One + One = Many (used more than once)
        assert_eq!(One + One, Many);

        // Many absorbs
        assert_eq!(One + Many, Many);
        assert_eq!(Many + One, Many);
        assert_eq!(Many + Many, Many);
    }

    #[test]
    fn test_multiplicity_semiring_multiplication() {
        use Multiplicity::*;

        // Zero annihilates
        assert_eq!(Zero * Zero, Zero);
        assert_eq!(Zero * One, Zero);
        assert_eq!(Zero * Many, Zero);
        assert_eq!(One * Zero, Zero);
        assert_eq!(Many * Zero, Zero);

        // One is multiplicative identity
        assert_eq!(One * One, One);
        assert_eq!(One * Many, Many);
        assert_eq!(Many * One, Many);

        // Many * Many = Many
        assert_eq!(Many * Many, Many);
    }

    #[test]
    fn test_is_runtime_relevant() {
        assert!(!Multiplicity::Zero.is_runtime_relevant());
        assert!(Multiplicity::One.is_runtime_relevant());
        assert!(Multiplicity::Many.is_runtime_relevant());
    }

    #[test]
    fn test_qtype_creation() {
        let erased = QType::erased("OntologyType");
        assert_eq!(erased.multiplicity, Multiplicity::Zero);
        assert!(!erased.is_runtime_relevant());

        let linear = QType::linear("FileHandle");
        assert_eq!(linear.multiplicity, Multiplicity::One);
        assert!(linear.is_runtime_relevant());

        let unrestricted = QType::unrestricted("Int");
        assert_eq!(unrestricted.multiplicity, Multiplicity::Many);
        assert!(unrestricted.is_runtime_relevant());
    }

    #[test]
    fn test_multiplicity_context_linear() {
        let mut ctx = MultiplicityContext::new();
        ctx.bind("x".to_string(), Multiplicity::One);

        // First use succeeds and consumes
        assert_eq!(ctx.use_var("x"), Some(Multiplicity::One));

        // Second use fails (variable consumed)
        assert_eq!(ctx.use_var("x"), None);
    }

    #[test]
    fn test_multiplicity_context_unrestricted() {
        let mut ctx = MultiplicityContext::new();
        ctx.bind("x".to_string(), Multiplicity::Many);

        // Can use multiple times
        assert_eq!(ctx.use_var("x"), Some(Multiplicity::Many));
        assert_eq!(ctx.use_var("x"), Some(Multiplicity::Many));
        assert_eq!(ctx.use_var("x"), Some(Multiplicity::Many));
    }

    #[test]
    fn test_multiplicity_context_erased() {
        let mut ctx = MultiplicityContext::new();
        ctx.bind("T".to_string(), Multiplicity::Zero);

        // Erased can be "used" any number of times (compile-time only)
        assert_eq!(ctx.use_var("T"), Some(Multiplicity::Zero));
        assert_eq!(ctx.use_var("T"), Some(Multiplicity::Zero));
    }

    #[test]
    fn test_check_consumed() {
        let mut ctx = MultiplicityContext::new();
        ctx.bind("x".to_string(), Multiplicity::One);
        ctx.bind("y".to_string(), Multiplicity::Many);

        // Linear variable not consumed
        assert!(ctx.check_consumed().is_err());

        // Consume linear variable
        ctx.use_var("x");
        assert!(ctx.check_consumed().is_ok());
    }

    #[test]
    fn test_permits() {
        use Multiplicity::*;

        // Zero permits nothing except zero
        assert!(Zero.permits(&Zero));
        assert!(!Zero.permits(&One));
        assert!(!Zero.permits(&Many));

        // One permits zero and one
        assert!(One.permits(&Zero));
        assert!(One.permits(&One));
        assert!(!One.permits(&Many));

        // Many permits everything
        assert!(Many.permits(&Zero));
        assert!(Many.permits(&One));
        assert!(Many.permits(&Many));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Multiplicity::Zero), "0");
        assert_eq!(format!("{}", Multiplicity::One), "1");
        assert_eq!(format!("{}", Multiplicity::Many), "ω");

        let qtype = QType::linear("FileHandle");
        assert_eq!(format!("{}", qtype), "1·FileHandle");
    }
}

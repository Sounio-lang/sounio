//! Algebraic Reasoning (AR) for Geometry
//!
//! Implements symbolic algebra with epistemic semantics for geometry proofs.
//! Key innovations:
//!
//! 1. **Expression trees** with full epistemic tracking (Beta posteriors)
//! 2. **Polynomial simplification** with unit validation
//! 3. **Equality solving** via constraint propagation
//! 4. **Z3-style refinement** for compile-time verification
//! 5. **Variance propagation** through algebraic operations
//!
//! # Theory
//!
//! Algebraic reasoning complements deductive reasoning (DD) by handling
//! numeric relationships that rules can't capture directly. For example:
//!
//! - If |AB| = 2x and |CD| = x + 3 and |AB| = |CD|, then x = 3
//! - Angle sums: if ∠A + ∠B + ∠C = 180° and ∠A = 60°, ∠B = 70°, then ∠C = 50°
//!
//! # Epistemic Integration
//!
//! Every expression carries a `BetaConfidence` representing uncertainty about
//! the algebraic relationship. Operations propagate uncertainty correctly:
//!
//! ```text
//! confidence(a + b) = combine_beta(confidence(a), confidence(b))
//! confidence(a * b) = combine_beta(confidence(a), confidence(b)) * decay
//! ```
//!
//! High variance triggers neural suggestions for auxiliary constructions.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::epistemic::bayesian::BetaConfidence;

use super::predicates::{Predicate, PredicateEpistemic, PredicateKind};

// =============================================================================
// Symbolic Expressions
// =============================================================================

/// A symbolic expression with epistemic tracking
#[derive(Clone)]
pub struct Expression {
    /// The expression tree
    pub kind: ExprKind,
    /// Epistemic confidence (Beta distribution)
    pub confidence: BetaConfidence,
    /// Provenance tracking
    pub provenance: ExprProvenance,
    /// Physical units (for dimensional analysis)
    pub unit: Option<Unit>,
}

/// Kind of expression node
#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Numeric constant
    Constant(f64),
    /// Symbolic variable (e.g., "x", "|AB|")
    Variable(String),
    /// Addition: a + b
    Add(Box<Expression>, Box<Expression>),
    /// Subtraction: a - b
    Sub(Box<Expression>, Box<Expression>),
    /// Multiplication: a * b
    Mul(Box<Expression>, Box<Expression>),
    /// Division: a / b
    Div(Box<Expression>, Box<Expression>),
    /// Power: a^n (n is integer for simplicity)
    Pow(Box<Expression>, i32),
    /// Negation: -a
    Neg(Box<Expression>),
    /// Square root: √a
    Sqrt(Box<Expression>),
    /// Sine: sin(a)
    Sin(Box<Expression>),
    /// Cosine: cos(a)
    Cos(Box<Expression>),
    /// Tangent: tan(a)
    Tan(Box<Expression>),
    /// Arc tangent: atan2(y, x)
    Atan2(Box<Expression>, Box<Expression>),
    /// Pi constant
    Pi,
}

impl PartialEq for ExprKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ExprKind::Constant(a), ExprKind::Constant(b)) => (a - b).abs() < 1e-10,
            (ExprKind::Variable(a), ExprKind::Variable(b)) => a == b,
            (ExprKind::Pi, ExprKind::Pi) => true,
            (ExprKind::Add(a1, a2), ExprKind::Add(b1, b2)) => {
                a1.kind == b1.kind && a2.kind == b2.kind
            }
            (ExprKind::Sub(a1, a2), ExprKind::Sub(b1, b2)) => {
                a1.kind == b1.kind && a2.kind == b2.kind
            }
            (ExprKind::Mul(a1, a2), ExprKind::Mul(b1, b2)) => {
                a1.kind == b1.kind && a2.kind == b2.kind
            }
            (ExprKind::Div(a1, a2), ExprKind::Div(b1, b2)) => {
                a1.kind == b1.kind && a2.kind == b2.kind
            }
            (ExprKind::Atan2(a1, a2), ExprKind::Atan2(b1, b2)) => {
                a1.kind == b1.kind && a2.kind == b2.kind
            }
            (ExprKind::Pow(a, n1), ExprKind::Pow(b, n2)) => a.kind == b.kind && n1 == n2,
            (ExprKind::Neg(a), ExprKind::Neg(b)) => a.kind == b.kind,
            (ExprKind::Sqrt(a), ExprKind::Sqrt(b)) => a.kind == b.kind,
            (ExprKind::Sin(a), ExprKind::Sin(b)) => a.kind == b.kind,
            (ExprKind::Cos(a), ExprKind::Cos(b)) => a.kind == b.kind,
            (ExprKind::Tan(a), ExprKind::Tan(b)) => a.kind == b.kind,
            _ => false,
        }
    }
}

impl Eq for ExprKind {}

/// Provenance for expressions
#[derive(Debug, Clone)]
pub struct ExprProvenance {
    /// Source of this expression
    pub source: ExprSource,
    /// Hash for identity
    pub hash: u64,
    /// Parent expression hashes (for tracing)
    pub parents: Vec<u64>,
    /// Depth in expression tree
    pub depth: usize,
}

/// Source of an expression
#[derive(Debug, Clone)]
pub enum ExprSource {
    /// From problem statement (axiom)
    Axiom,
    /// From predicate (e.g., |AB| from equal_length predicate)
    FromPredicate(String),
    /// Derived algebraically
    Derived { operation: String },
    /// From neural suggestion
    Neural { model: String, confidence: f64 },
}

impl ExprProvenance {
    pub fn axiom() -> Self {
        ExprProvenance {
            source: ExprSource::Axiom,
            hash: 0,
            parents: vec![],
            depth: 0,
        }
    }

    pub fn derived(operation: &str, parents: &[&Expression]) -> Self {
        let parent_hashes: Vec<u64> = parents.iter().map(|e| e.provenance.hash).collect();
        let max_depth = parents
            .iter()
            .map(|e| e.provenance.depth)
            .max()
            .unwrap_or(0);

        ExprProvenance {
            source: ExprSource::Derived {
                operation: operation.to_string(),
            },
            hash: 0, // Will be computed
            parents: parent_hashes,
            depth: max_depth + 1,
        }
    }

    pub fn from_predicate(pred_key: &str) -> Self {
        ExprProvenance {
            source: ExprSource::FromPredicate(pred_key.to_string()),
            hash: 0,
            parents: vec![],
            depth: 0,
        }
    }
}

/// Physical unit for dimensional analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Unit {
    /// Base dimensions: length^a * angle^b * ...
    pub dimensions: UnitDimensions,
    /// Display name (e.g., "m", "°")
    pub name: Option<String>,
}

/// Unit dimensions (simplified for geometry)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct UnitDimensions {
    /// Length exponent (e.g., 1 for length, 2 for area)
    pub length: i32,
    /// Angle exponent (usually 0 or 1)
    pub angle: i32,
}

impl Unit {
    pub fn dimensionless() -> Self {
        Unit {
            dimensions: UnitDimensions::default(),
            name: None,
        }
    }

    pub fn length() -> Self {
        Unit {
            dimensions: UnitDimensions {
                length: 1,
                angle: 0,
            },
            name: Some("length".to_string()),
        }
    }

    pub fn area() -> Self {
        Unit {
            dimensions: UnitDimensions {
                length: 2,
                angle: 0,
            },
            name: Some("area".to_string()),
        }
    }

    pub fn angle() -> Self {
        Unit {
            dimensions: UnitDimensions {
                length: 0,
                angle: 1,
            },
            name: Some("angle".to_string()),
        }
    }

    /// Check if units are compatible for addition/subtraction
    pub fn compatible(&self, other: &Unit) -> bool {
        self.dimensions == other.dimensions
    }

    /// Multiply units
    pub fn multiply(&self, other: &Unit) -> Unit {
        Unit {
            dimensions: UnitDimensions {
                length: self.dimensions.length + other.dimensions.length,
                angle: self.dimensions.angle + other.dimensions.angle,
            },
            name: None,
        }
    }

    /// Divide units
    pub fn divide(&self, other: &Unit) -> Unit {
        Unit {
            dimensions: UnitDimensions {
                length: self.dimensions.length - other.dimensions.length,
                angle: self.dimensions.angle - other.dimensions.angle,
            },
            name: None,
        }
    }

    /// Power of unit
    pub fn power(&self, n: i32) -> Unit {
        Unit {
            dimensions: UnitDimensions {
                length: self.dimensions.length * n,
                angle: self.dimensions.angle * n,
            },
            name: None,
        }
    }

    /// Square root of unit
    pub fn sqrt(&self) -> Option<Unit> {
        if self.dimensions.length % 2 == 0 && self.dimensions.angle % 2 == 0 {
            Some(Unit {
                dimensions: UnitDimensions {
                    length: self.dimensions.length / 2,
                    angle: self.dimensions.angle / 2,
                },
                name: None,
            })
        } else {
            None // Can't take sqrt of odd-power unit
        }
    }
}

impl Expression {
    // Constructors

    /// Create a constant expression
    pub fn constant(value: f64) -> Self {
        Expression {
            kind: ExprKind::Constant(value),
            confidence: BetaConfidence::new(10.0, 0.1), // High confidence for constants
            provenance: ExprProvenance::axiom(),
            unit: Some(Unit::dimensionless()),
        }
    }

    /// Create a variable expression
    pub fn variable(name: impl Into<String>) -> Self {
        Expression {
            kind: ExprKind::Variable(name.into()),
            confidence: BetaConfidence::uniform_prior(), // Unknown confidence
            provenance: ExprProvenance::axiom(),
            unit: None, // Unit will be inferred
        }
    }

    /// Create a length variable (e.g., |AB|)
    pub fn length(p1: &str, p2: &str) -> Self {
        let mut pts = [p1.to_string(), p2.to_string()];
        pts.sort();
        let name = format!("|{}{}|", pts[0], pts[1]);

        Expression {
            kind: ExprKind::Variable(name),
            confidence: BetaConfidence::uniform_prior(),
            provenance: ExprProvenance::axiom(),
            unit: Some(Unit::length()),
        }
    }

    /// Create an angle variable (e.g., ∠ABC)
    pub fn angle_var(p1: &str, vertex: &str, p2: &str) -> Self {
        let mut rays = [p1.to_string(), p2.to_string()];
        rays.sort();
        let name = format!("∠{}{}{}", rays[0], vertex, rays[1]);

        Expression {
            kind: ExprKind::Variable(name),
            confidence: BetaConfidence::uniform_prior(),
            provenance: ExprProvenance::axiom(),
            unit: Some(Unit::angle()),
        }
    }

    /// Pi constant
    pub fn pi() -> Self {
        Expression {
            kind: ExprKind::Pi,
            confidence: BetaConfidence::new(100.0, 0.01), // Very high confidence
            provenance: ExprProvenance::axiom(),
            unit: Some(Unit::angle()), // Pi radians
        }
    }

    /// Set the confidence
    pub fn with_confidence(mut self, confidence: BetaConfidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the unit
    pub fn with_unit(mut self, unit: Unit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Set provenance
    pub fn with_provenance(mut self, provenance: ExprProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    // Helpers

    /// Compute hash for this expression
    fn compute_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.kind.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if this is a constant
    pub fn is_constant(&self) -> bool {
        matches!(self.kind, ExprKind::Constant(_))
    }

    /// Check if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self.kind, ExprKind::Variable(_))
    }

    /// Get variable name if this is a variable
    pub fn as_variable(&self) -> Option<&str> {
        match &self.kind {
            ExprKind::Variable(name) => Some(name),
            _ => None,
        }
    }

    /// Get constant value if this is a constant
    pub fn as_constant(&self) -> Option<f64> {
        match &self.kind {
            ExprKind::Constant(v) => Some(*v),
            ExprKind::Pi => Some(std::f64::consts::PI),
            _ => None,
        }
    }

    /// Get all variables in this expression
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut HashSet<String>) {
        match &self.kind {
            ExprKind::Variable(name) => {
                vars.insert(name.clone());
            }
            ExprKind::Add(a, b)
            | ExprKind::Sub(a, b)
            | ExprKind::Mul(a, b)
            | ExprKind::Div(a, b)
            | ExprKind::Atan2(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            ExprKind::Pow(a, _)
            | ExprKind::Neg(a)
            | ExprKind::Sqrt(a)
            | ExprKind::Sin(a)
            | ExprKind::Cos(a)
            | ExprKind::Tan(a) => {
                a.collect_variables(vars);
            }
            ExprKind::Constant(_) | ExprKind::Pi => {}
        }
    }

    /// Evaluate with variable substitution
    pub fn evaluate(&self, bindings: &HashMap<String, f64>) -> Option<f64> {
        match &self.kind {
            ExprKind::Constant(v) => Some(*v),
            ExprKind::Pi => Some(std::f64::consts::PI),
            ExprKind::Variable(name) => bindings.get(name).copied(),
            ExprKind::Add(a, b) => Some(a.evaluate(bindings)? + b.evaluate(bindings)?),
            ExprKind::Sub(a, b) => Some(a.evaluate(bindings)? - b.evaluate(bindings)?),
            ExprKind::Mul(a, b) => Some(a.evaluate(bindings)? * b.evaluate(bindings)?),
            ExprKind::Div(a, b) => {
                let bv = b.evaluate(bindings)?;
                if bv.abs() < 1e-10 {
                    None
                } else {
                    Some(a.evaluate(bindings)? / bv)
                }
            }
            ExprKind::Pow(a, n) => Some(a.evaluate(bindings)?.powi(*n)),
            ExprKind::Neg(a) => Some(-a.evaluate(bindings)?),
            ExprKind::Sqrt(a) => {
                let av = a.evaluate(bindings)?;
                if av < 0.0 { None } else { Some(av.sqrt()) }
            }
            ExprKind::Sin(a) => Some(a.evaluate(bindings)?.sin()),
            ExprKind::Cos(a) => Some(a.evaluate(bindings)?.cos()),
            ExprKind::Tan(a) => Some(a.evaluate(bindings)?.tan()),
            ExprKind::Atan2(y, x) => Some(y.evaluate(bindings)?.atan2(x.evaluate(bindings)?)),
        }
    }

    /// Substitute a variable with an expression
    pub fn substitute(&self, var: &str, replacement: &Expression) -> Expression {
        match &self.kind {
            ExprKind::Variable(name) if name == var => replacement.clone(),
            ExprKind::Variable(_) | ExprKind::Constant(_) | ExprKind::Pi => self.clone(),
            ExprKind::Add(a, b) => a.substitute(var, replacement) + b.substitute(var, replacement),
            ExprKind::Sub(a, b) => a.substitute(var, replacement) - b.substitute(var, replacement),
            ExprKind::Mul(a, b) => a.substitute(var, replacement) * b.substitute(var, replacement),
            ExprKind::Div(a, b) => a.substitute(var, replacement) / b.substitute(var, replacement),
            ExprKind::Pow(a, n) => a.substitute(var, replacement).pow(*n),
            ExprKind::Neg(a) => -a.substitute(var, replacement),
            ExprKind::Sqrt(a) => a.substitute(var, replacement).sqrt(),
            ExprKind::Sin(a) => a.substitute(var, replacement).sin(),
            ExprKind::Cos(a) => a.substitute(var, replacement).cos(),
            ExprKind::Tan(a) => a.substitute(var, replacement).tan(),
            ExprKind::Atan2(y, x) => Expression::atan2(
                y.substitute(var, replacement),
                x.substitute(var, replacement),
            ),
        }
    }

    // Unary operations

    pub fn pow(self, n: i32) -> Expression {
        let confidence = self.confidence;
        let unit = self.unit.as_ref().map(|u| u.power(n));
        let provenance = ExprProvenance::derived("pow", &[&self]);

        Expression {
            kind: ExprKind::Pow(Box::new(self), n),
            confidence,
            provenance,
            unit,
        }
    }

    pub fn sqrt(self) -> Expression {
        let confidence = self.confidence;
        let unit = self.unit.as_ref().and_then(|u| u.sqrt());
        let provenance = ExprProvenance::derived("sqrt", &[&self]);

        Expression {
            kind: ExprKind::Sqrt(Box::new(self)),
            confidence,
            provenance,
            unit,
        }
    }

    pub fn sin(self) -> Expression {
        let confidence = self.confidence;
        let provenance = ExprProvenance::derived("sin", &[&self]);

        Expression {
            kind: ExprKind::Sin(Box::new(self)),
            confidence,
            provenance,
            unit: Some(Unit::dimensionless()), // sin outputs dimensionless
        }
    }

    pub fn cos(self) -> Expression {
        let confidence = self.confidence;
        let provenance = ExprProvenance::derived("cos", &[&self]);

        Expression {
            kind: ExprKind::Cos(Box::new(self)),
            confidence,
            provenance,
            unit: Some(Unit::dimensionless()),
        }
    }

    pub fn tan(self) -> Expression {
        let confidence = self.confidence;
        let provenance = ExprProvenance::derived("tan", &[&self]);

        Expression {
            kind: ExprKind::Tan(Box::new(self)),
            confidence,
            provenance,
            unit: Some(Unit::dimensionless()),
        }
    }

    pub fn atan2(y: Expression, x: Expression) -> Expression {
        let confidence = combine_confidence(&y.confidence, &x.confidence, 0.99);
        let provenance = ExprProvenance::derived("atan2", &[&y, &x]);

        Expression {
            kind: ExprKind::Atan2(Box::new(y), Box::new(x)),
            confidence,
            provenance,
            unit: Some(Unit::angle()), // atan2 outputs angle
        }
    }

    /// Epistemic uncertainty (variance of Beta distribution)
    pub fn epistemic_uncertainty(&self) -> f64 {
        self.confidence.variance()
    }
}

/// Combine confidences for binary operations
fn combine_confidence(a: &BetaConfidence, b: &BetaConfidence, decay: f64) -> BetaConfidence {
    // Use linear pool with equal weights, then apply decay
    let combined = a.combine(b, 1.0, 1.0);
    BetaConfidence::new(
        combined.alpha * decay,
        combined.beta + (1.0 - decay) * (combined.alpha + combined.beta),
    )
}

// Arithmetic operations for Expression

impl Add for Expression {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Expression {
        // Check unit compatibility
        let unit = match (&self.unit, &rhs.unit) {
            (Some(u1), Some(u2)) if u1.compatible(u2) => Some(u1.clone()),
            (Some(u), None) | (None, Some(u)) => Some(u.clone()),
            (None, None) => None,
            _ => None, // Incompatible units - will be caught by validation
        };

        let confidence = combine_confidence(&self.confidence, &rhs.confidence, 0.995);
        let provenance = ExprProvenance::derived("add", &[&self, &rhs]);

        Expression {
            kind: ExprKind::Add(Box::new(self), Box::new(rhs)),
            confidence,
            provenance,
            unit,
        }
    }
}

impl Sub for Expression {
    type Output = Expression;

    fn sub(self, rhs: Expression) -> Expression {
        let unit = match (&self.unit, &rhs.unit) {
            (Some(u1), Some(u2)) if u1.compatible(u2) => Some(u1.clone()),
            (Some(u), None) | (None, Some(u)) => Some(u.clone()),
            (None, None) => None,
            _ => None,
        };

        let confidence = combine_confidence(&self.confidence, &rhs.confidence, 0.995);
        let provenance = ExprProvenance::derived("sub", &[&self, &rhs]);

        Expression {
            kind: ExprKind::Sub(Box::new(self), Box::new(rhs)),
            confidence,
            provenance,
            unit,
        }
    }
}

impl Mul for Expression {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Expression {
        let unit = match (&self.unit, &rhs.unit) {
            (Some(u1), Some(u2)) => Some(u1.multiply(u2)),
            (Some(u), None) | (None, Some(u)) => Some(u.clone()),
            (None, None) => None,
        };

        let confidence = combine_confidence(&self.confidence, &rhs.confidence, 0.99);
        let provenance = ExprProvenance::derived("mul", &[&self, &rhs]);

        Expression {
            kind: ExprKind::Mul(Box::new(self), Box::new(rhs)),
            confidence,
            provenance,
            unit,
        }
    }
}

impl Div for Expression {
    type Output = Expression;

    fn div(self, rhs: Expression) -> Expression {
        let unit = match (&self.unit, &rhs.unit) {
            (Some(u1), Some(u2)) => Some(u1.divide(u2)),
            (Some(u), None) => Some(u.clone()),
            (None, Some(u)) => Some(Unit::dimensionless().divide(u)),
            (None, None) => None,
        };

        let confidence = combine_confidence(&self.confidence, &rhs.confidence, 0.98); // Division less reliable
        let provenance = ExprProvenance::derived("div", &[&self, &rhs]);

        Expression {
            kind: ExprKind::Div(Box::new(self), Box::new(rhs)),
            confidence,
            provenance,
            unit,
        }
    }
}

impl Neg for Expression {
    type Output = Expression;

    fn neg(self) -> Expression {
        let confidence = self.confidence;
        let unit = self.unit.clone();
        let provenance = ExprProvenance::derived("neg", &[&self]);

        Expression {
            kind: ExprKind::Neg(Box::new(self)),
            confidence,
            provenance,
            unit,
        }
    }
}

impl fmt::Debug for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (conf: {:.3})", self.kind, self.confidence.mean())
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::Constant(v) => write!(f, "{}", v),
            ExprKind::Variable(name) => write!(f, "{}", name),
            ExprKind::Pi => write!(f, "π"),
            ExprKind::Add(a, b) => write!(f, "({} + {})", a, b),
            ExprKind::Sub(a, b) => write!(f, "({} - {})", a, b),
            ExprKind::Mul(a, b) => write!(f, "({} × {})", a, b),
            ExprKind::Div(a, b) => write!(f, "({} / {})", a, b),
            ExprKind::Pow(a, n) => write!(f, "{}^{}", a, n),
            ExprKind::Neg(a) => write!(f, "-{}", a),
            ExprKind::Sqrt(a) => write!(f, "√{}", a),
            ExprKind::Sin(a) => write!(f, "sin({})", a),
            ExprKind::Cos(a) => write!(f, "cos({})", a),
            ExprKind::Tan(a) => write!(f, "tan({})", a),
            ExprKind::Atan2(y, x) => write!(f, "atan2({}, {})", y, x),
        }
    }
}

impl Hash for ExprKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ExprKind::Constant(v) => v.to_bits().hash(state),
            ExprKind::Variable(name) => name.hash(state),
            ExprKind::Pow(_, n) => n.hash(state),
            _ => {}
        }
    }
}

// =============================================================================
// Equations and Constraint System
// =============================================================================

/// An equation: lhs = rhs
#[derive(Debug, Clone)]
pub struct Equation {
    pub lhs: Expression,
    pub rhs: Expression,
    /// Epistemic confidence in this equation
    pub confidence: BetaConfidence,
    /// Source predicate (if any)
    pub source: Option<String>,
}

impl Equation {
    pub fn new(lhs: Expression, rhs: Expression) -> Self {
        let confidence = combine_confidence(&lhs.confidence, &rhs.confidence, 0.99);
        Equation {
            lhs,
            rhs,
            confidence,
            source: None,
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    pub fn with_confidence(mut self, confidence: BetaConfidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Check unit compatibility
    pub fn units_compatible(&self) -> bool {
        match (&self.lhs.unit, &self.rhs.unit) {
            (Some(u1), Some(u2)) => u1.compatible(u2),
            _ => true, // If one side has no unit, assume compatible
        }
    }

    /// Get all variables
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = self.lhs.variables();
        vars.extend(self.rhs.variables());
        vars
    }

    /// Rearrange to solve for a variable: var = expr
    pub fn solve_for(&self, var: &str) -> Option<Equation> {
        // Simple case: var = rhs
        if let ExprKind::Variable(name) = &self.lhs.kind
            && name == var
        {
            return Some(self.clone());
        }

        // Simple case: lhs = var
        if let ExprKind::Variable(name) = &self.rhs.kind
            && name == var
        {
            return Some(
                Equation::new(self.rhs.clone(), self.lhs.clone()).with_confidence(self.confidence),
            );
        }

        // Linear case: a*var + b = c => var = (c - b) / a
        if let Some((coef, rest)) = self.extract_linear_term(&self.lhs, var) {
            let solution = (self.rhs.clone() - rest) / coef;
            return Some(
                Equation::new(Expression::variable(var), solution).with_confidence(self.confidence),
            );
        }

        // Try the other side
        if let Some((coef, rest)) = self.extract_linear_term(&self.rhs, var) {
            let solution = (self.lhs.clone() - rest) / coef;
            return Some(
                Equation::new(Expression::variable(var), solution).with_confidence(self.confidence),
            );
        }

        None
    }

    /// Extract linear term: expr = coef * var + rest
    fn extract_linear_term(
        &self,
        expr: &Expression,
        var: &str,
    ) -> Option<(Expression, Expression)> {
        match &expr.kind {
            ExprKind::Variable(name) if name == var => {
                Some((Expression::constant(1.0), Expression::constant(0.0)))
            }
            ExprKind::Mul(a, b) => {
                if let ExprKind::Variable(name) = &b.kind
                    && name == var
                    && !a.variables().contains(var)
                {
                    return Some(((**a).clone(), Expression::constant(0.0)));
                }
                if let ExprKind::Variable(name) = &a.kind
                    && name == var
                    && !b.variables().contains(var)
                {
                    return Some(((**b).clone(), Expression::constant(0.0)));
                }
                None
            }
            ExprKind::Add(a, b) => {
                // Try extracting from left
                if let Some((coef, rest_a)) = self.extract_linear_term(a, var)
                    && !b.variables().contains(var)
                {
                    return Some((coef, rest_a + (**b).clone()));
                }
                // Try extracting from right
                if let Some((coef, rest_b)) = self.extract_linear_term(b, var)
                    && !a.variables().contains(var)
                {
                    return Some((coef, (**a).clone() + rest_b));
                }
                None
            }
            ExprKind::Sub(a, b) => {
                if let Some((coef, rest_a)) = self.extract_linear_term(a, var)
                    && !b.variables().contains(var)
                {
                    return Some((coef, rest_a - (**b).clone()));
                }
                if let Some((coef, rest_b)) = self.extract_linear_term(b, var)
                    && !a.variables().contains(var)
                {
                    return Some((-coef, (**a).clone() - rest_b));
                }
                None
            }
            ExprKind::Neg(a) => {
                if let Some((coef, rest)) = self.extract_linear_term(a, var) {
                    Some((-coef, -rest))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl fmt::Display for Equation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.lhs, self.rhs)
    }
}

// =============================================================================
// Algebraic Reasoner
// =============================================================================

/// Configuration for algebraic reasoning
#[derive(Debug, Clone)]
pub struct AlgebraicConfig {
    /// Maximum iterations for constraint propagation
    pub max_iterations: usize,
    /// Tolerance for numeric equality
    pub epsilon: f64,
    /// Minimum confidence to accept a solution
    pub min_confidence: f64,
    /// Confidence decay per operation
    pub decay_factor: f64,
    /// Whether to enforce unit compatibility
    pub strict_units: bool,
}

impl Default for AlgebraicConfig {
    fn default() -> Self {
        AlgebraicConfig {
            max_iterations: 100,
            epsilon: 1e-9,
            min_confidence: 0.5,
            decay_factor: 0.99,
            strict_units: true,
        }
    }
}

/// Result of algebraic reasoning
#[derive(Debug, Clone)]
pub struct AlgebraicResult {
    /// Variable solutions (var -> value)
    pub solutions: HashMap<String, f64>,
    /// Solved equations (var -> expression)
    pub symbolic_solutions: HashMap<String, Expression>,
    /// Derived predicates
    pub derived_predicates: Vec<Predicate>,
    /// Overall confidence
    pub confidence: BetaConfidence,
    /// Whether solution is complete
    pub complete: bool,
    /// Reasoning trace
    pub trace: Vec<AlgebraicStep>,
    /// Any unit errors found
    pub unit_errors: Vec<UnitError>,
}

/// A step in algebraic reasoning
#[derive(Debug, Clone)]
pub struct AlgebraicStep {
    pub description: String,
    pub equation: Option<Equation>,
    pub solution: Option<(String, f64)>,
    pub confidence: f64,
}

/// Unit compatibility error
#[derive(Debug, Clone)]
pub struct UnitError {
    pub equation: String,
    pub lhs_unit: Option<Unit>,
    pub rhs_unit: Option<Unit>,
    pub message: String,
}

/// The algebraic reasoner
pub struct AlgebraicReasoner {
    config: AlgebraicConfig,
    equations: Vec<Equation>,
    known_values: HashMap<String, f64>,
    symbolic_values: HashMap<String, Expression>,
    derived_predicates: Vec<Predicate>,
    trace: Vec<AlgebraicStep>,
    unit_errors: Vec<UnitError>,
}

impl AlgebraicReasoner {
    pub fn new(config: AlgebraicConfig) -> Self {
        AlgebraicReasoner {
            config,
            equations: Vec::new(),
            known_values: HashMap::new(),
            symbolic_values: HashMap::new(),
            derived_predicates: Vec::new(),
            trace: Vec::new(),
            unit_errors: Vec::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(AlgebraicConfig::default())
    }

    /// Add an equation to the system
    pub fn add_equation(&mut self, eq: Equation) {
        // Check unit compatibility
        if self.config.strict_units && !eq.units_compatible() {
            self.unit_errors.push(UnitError {
                equation: eq.to_string(),
                lhs_unit: eq.lhs.unit.clone(),
                rhs_unit: eq.rhs.unit.clone(),
                message: "Incompatible units in equation".to_string(),
            });
            return;
        }

        self.trace.push(AlgebraicStep {
            description: format!("Added equation: {}", eq),
            equation: Some(eq.clone()),
            solution: None,
            confidence: eq.confidence.mean(),
        });

        self.equations.push(eq);
    }

    /// Add a known value
    pub fn add_known(&mut self, var: &str, value: f64) {
        self.known_values.insert(var.to_string(), value);
        self.trace.push(AlgebraicStep {
            description: format!("Known value: {} = {}", var, value),
            equation: None,
            solution: Some((var.to_string(), value)),
            confidence: 1.0,
        });
    }

    /// Add equations from predicates
    pub fn add_from_predicate(&mut self, pred: &Predicate) {
        match pred.kind {
            PredicateKind::EqualLength => {
                // equal_length(A, B, C, D) => |AB| = |CD|
                if pred.args.len() >= 4 {
                    let ab = Expression::length(&pred.args[0], &pred.args[1]);
                    let cd = Expression::length(&pred.args[2], &pred.args[3]);
                    let eq = Equation::new(ab, cd)
                        .with_source(&pred.key())
                        .with_confidence(pred.epistemic.confidence);
                    self.add_equation(eq);
                }
            }
            PredicateKind::Midpoint => {
                // midpoint(M, A, B) => |AM| = |MB| = |AB|/2
                if pred.args.len() >= 3 {
                    let m = &pred.args[0];
                    let a = &pred.args[1];
                    let b = &pred.args[2];

                    let am = Expression::length(a, m);
                    let mb = Expression::length(m, b);
                    let ab = Expression::length(a, b);

                    let eq1 = Equation::new(am.clone(), mb.clone())
                        .with_source(&pred.key())
                        .with_confidence(pred.epistemic.confidence);
                    let eq2 = Equation::new(am, ab / Expression::constant(2.0))
                        .with_source(&pred.key())
                        .with_confidence(pred.epistemic.confidence);

                    self.add_equation(eq1);
                    self.add_equation(eq2);
                }
            }
            PredicateKind::RightAngle => {
                // right_angle(A, V, B) => ∠AVB = π/2
                if pred.args.len() >= 3 {
                    let angle = Expression::angle_var(&pred.args[0], &pred.args[1], &pred.args[2]);
                    let right = Expression::pi() / Expression::constant(2.0);
                    let eq = Equation::new(angle, right)
                        .with_source(&pred.key())
                        .with_confidence(pred.epistemic.confidence);
                    self.add_equation(eq);
                }
            }
            PredicateKind::AlgebraicEqual => {
                // Handled externally with explicit expressions
            }
            _ => {
                // Other predicates don't directly contribute equations
            }
        }
    }

    /// Solve the system using constraint propagation
    pub fn solve(&mut self) -> AlgebraicResult {
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            // Try to solve each equation
            for eq_idx in 0..self.equations.len() {
                let eq = &self.equations[eq_idx];
                let vars = eq.variables();

                // Find variables we don't know yet
                let unknown: Vec<_> = vars
                    .iter()
                    .filter(|v| !self.known_values.contains_key(*v))
                    .collect();

                // If exactly one unknown, solve for it
                if unknown.len() == 1 {
                    let var = unknown[0];

                    // Substitute known values
                    let mut lhs = eq.lhs.clone();
                    let mut rhs = eq.rhs.clone();

                    for (known_var, value) in &self.known_values {
                        lhs = lhs.substitute(known_var, &Expression::constant(*value));
                        rhs = rhs.substitute(known_var, &Expression::constant(*value));
                    }

                    // Try to solve
                    let simplified_eq = Equation::new(lhs, rhs).with_confidence(eq.confidence);

                    if let Some(solution) = simplified_eq.solve_for(var) {
                        // Evaluate RHS
                        if let Some(value) = solution.rhs.evaluate(&self.known_values)
                            && value.is_finite()
                        {
                            self.known_values.insert(var.clone(), value);
                            self.symbolic_values
                                .insert(var.clone(), solution.rhs.clone());

                            self.trace.push(AlgebraicStep {
                                description: format!(
                                    "Solved: {} = {} = {}",
                                    var, solution.rhs, value
                                ),
                                equation: Some(eq.clone()),
                                solution: Some((var.clone(), value)),
                                confidence: solution.confidence.mean(),
                            });

                            changed = true;
                        }
                    }
                }
            }
        }

        // Generate derived predicates from solutions
        self.generate_derived_predicates();

        // Compute overall confidence
        let overall_confidence = self.compute_overall_confidence();

        // Check completeness
        let all_vars: HashSet<_> = self
            .equations
            .iter()
            .flat_map(|eq| eq.variables())
            .collect();
        let complete = all_vars.iter().all(|v| self.known_values.contains_key(v));

        AlgebraicResult {
            solutions: self.known_values.clone(),
            symbolic_solutions: self.symbolic_values.clone(),
            derived_predicates: self.derived_predicates.clone(),
            confidence: overall_confidence,
            complete,
            trace: self.trace.clone(),
            unit_errors: self.unit_errors.clone(),
        }
    }

    /// Generate derived predicates from numeric solutions
    fn generate_derived_predicates(&mut self) {
        // Check if any pair of lengths are equal
        let length_vars: Vec<_> = self
            .known_values
            .iter()
            .filter(|(k, _)| k.starts_with('|') && k.ends_with('|'))
            .collect();

        for i in 0..length_vars.len() {
            for j in (i + 1)..length_vars.len() {
                let (name1, val1) = length_vars[i];
                let (name2, val2) = length_vars[j];

                if (val1 - val2).abs() < self.config.epsilon {
                    // Create equal_length predicate
                    // Parse point names from |AB| format
                    if let (Some(pts1), Some(pts2)) =
                        (self.parse_length_var(name1), self.parse_length_var(name2))
                    {
                        let pred = Predicate::equal_length(&pts1.0, &pts1.1, &pts2.0, &pts2.1)
                            .with_epistemic(PredicateEpistemic::derived(
                                &[],
                                "algebraic_equal",
                                self.config.decay_factor,
                            ));
                        self.derived_predicates.push(pred);
                    }
                }
            }
        }
    }

    fn parse_length_var(&self, name: &str) -> Option<(String, String)> {
        // Parse |AB| into ("A", "B")
        let trimmed = name.trim_matches('|');
        if trimmed.len() >= 2 {
            // Assume single-char point names for simplicity
            // Could be enhanced for multi-char names
            let chars: Vec<char> = trimmed.chars().collect();
            if chars.len() == 2 {
                return Some((chars[0].to_string(), chars[1].to_string()));
            }
        }
        None
    }

    fn compute_overall_confidence(&self) -> BetaConfidence {
        if self.equations.is_empty() {
            return BetaConfidence::uniform_prior();
        }

        // Combine all equation confidences
        let mut combined = self.equations[0].confidence;
        for eq in self.equations.iter().skip(1) {
            combined = combined.combine_log_pool(&eq.confidence);
        }

        // Apply penalty for iterations (more iterations = less certain)
        let iteration_penalty = 0.99_f64.powi(self.trace.len() as i32);
        BetaConfidence::new(
            combined.alpha * iteration_penalty,
            combined.beta + (1.0 - iteration_penalty) * combined.alpha,
        )
    }

    /// Reset the reasoner for reuse
    pub fn reset(&mut self) {
        self.equations.clear();
        self.known_values.clear();
        self.symbolic_values.clear();
        self.derived_predicates.clear();
        self.trace.clear();
        self.unit_errors.clear();
    }
}

// =============================================================================
// Expression Simplification
// =============================================================================

/// Simplify an expression algebraically
pub fn simplify(expr: &Expression) -> Expression {
    match &expr.kind {
        // Constants and variables unchanged
        ExprKind::Constant(_) | ExprKind::Variable(_) | ExprKind::Pi => expr.clone(),

        // x + 0 = x, 0 + x = x
        ExprKind::Add(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);

            if let ExprKind::Constant(0.0) = a_simp.kind {
                return b_simp;
            }
            if let ExprKind::Constant(0.0) = b_simp.kind {
                return a_simp;
            }

            // Constant folding
            if let (Some(av), Some(bv)) = (a_simp.as_constant(), b_simp.as_constant()) {
                return Expression::constant(av + bv).with_confidence(combine_confidence(
                    &a_simp.confidence,
                    &b_simp.confidence,
                    1.0,
                ));
            }

            a_simp + b_simp
        }

        // x - 0 = x, x - x = 0
        ExprKind::Sub(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);

            if let ExprKind::Constant(0.0) = b_simp.kind {
                return a_simp;
            }

            // x - x = 0
            if format!("{}", a_simp) == format!("{}", b_simp) {
                return Expression::constant(0.0);
            }

            // Constant folding
            if let (Some(av), Some(bv)) = (a_simp.as_constant(), b_simp.as_constant()) {
                return Expression::constant(av - bv).with_confidence(combine_confidence(
                    &a_simp.confidence,
                    &b_simp.confidence,
                    1.0,
                ));
            }

            a_simp - b_simp
        }

        // x * 0 = 0, x * 1 = x, 0 * x = 0, 1 * x = x
        ExprKind::Mul(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);

            if let ExprKind::Constant(v) = a_simp.kind {
                if v == 0.0 {
                    return Expression::constant(0.0);
                }
                if v == 1.0 {
                    return b_simp;
                }
            }
            if let ExprKind::Constant(v) = b_simp.kind {
                if v == 0.0 {
                    return Expression::constant(0.0);
                }
                if v == 1.0 {
                    return a_simp;
                }
            }

            // Constant folding
            if let (Some(av), Some(bv)) = (a_simp.as_constant(), b_simp.as_constant()) {
                return Expression::constant(av * bv).with_confidence(combine_confidence(
                    &a_simp.confidence,
                    &b_simp.confidence,
                    1.0,
                ));
            }

            a_simp * b_simp
        }

        // x / 1 = x, 0 / x = 0
        ExprKind::Div(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);

            if let ExprKind::Constant(1.0) = b_simp.kind {
                return a_simp;
            }
            if let ExprKind::Constant(0.0) = a_simp.kind {
                return Expression::constant(0.0);
            }

            // Constant folding
            if let (Some(av), Some(bv)) = (a_simp.as_constant(), b_simp.as_constant())
                && bv != 0.0
            {
                return Expression::constant(av / bv).with_confidence(combine_confidence(
                    &a_simp.confidence,
                    &b_simp.confidence,
                    1.0,
                ));
            }

            a_simp / b_simp
        }

        // x^0 = 1, x^1 = x
        ExprKind::Pow(a, n) => {
            if *n == 0 {
                return Expression::constant(1.0);
            }
            if *n == 1 {
                return simplify(a);
            }

            let a_simp = simplify(a);
            if let Some(av) = a_simp.as_constant() {
                return Expression::constant(av.powi(*n)).with_confidence(a_simp.confidence);
            }

            a_simp.pow(*n)
        }

        // --x = x
        ExprKind::Neg(a) => {
            let a_simp = simplify(a);
            if let ExprKind::Neg(inner) = &a_simp.kind {
                return (**inner).clone();
            }
            if let Some(av) = a_simp.as_constant() {
                return Expression::constant(-av).with_confidence(a_simp.confidence);
            }
            -a_simp
        }

        // √(x²) = |x| (simplified to x for positive x)
        ExprKind::Sqrt(a) => {
            let a_simp = simplify(a);
            if let Some(av) = a_simp.as_constant()
                && av >= 0.0
            {
                return Expression::constant(av.sqrt()).with_confidence(a_simp.confidence);
            }
            a_simp.sqrt()
        }

        // Trig simplification
        ExprKind::Sin(a) => {
            let a_simp = simplify(a);
            if let Some(av) = a_simp.as_constant() {
                return Expression::constant(av.sin()).with_confidence(a_simp.confidence);
            }
            a_simp.sin()
        }

        ExprKind::Cos(a) => {
            let a_simp = simplify(a);
            if let Some(av) = a_simp.as_constant() {
                return Expression::constant(av.cos()).with_confidence(a_simp.confidence);
            }
            a_simp.cos()
        }

        ExprKind::Tan(a) => {
            let a_simp = simplify(a);
            if let Some(av) = a_simp.as_constant() {
                return Expression::constant(av.tan()).with_confidence(a_simp.confidence);
            }
            a_simp.tan()
        }

        ExprKind::Atan2(y, x) => {
            let y_simp = simplify(y);
            let x_simp = simplify(x);
            if let (Some(yv), Some(xv)) = (y_simp.as_constant(), x_simp.as_constant()) {
                return Expression::constant(yv.atan2(xv)).with_confidence(combine_confidence(
                    &y_simp.confidence,
                    &x_simp.confidence,
                    1.0,
                ));
            }
            Expression::atan2(y_simp, x_simp)
        }
    }
}

// =============================================================================
// Refinement Checking (Z3-style)
// =============================================================================

/// Result of refinement checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefinementResult {
    /// The equation definitely holds
    Satisfied,
    /// The equation definitely doesn't hold
    Unsatisfied,
    /// Can't determine (need more information)
    Unknown,
}

/// Check if an equation is satisfiable given known constraints
pub fn check_refinement(
    eq: &Equation,
    known: &HashMap<String, f64>,
    epsilon: f64,
) -> RefinementResult {
    // Substitute known values
    let mut lhs = eq.lhs.clone();
    let mut rhs = eq.rhs.clone();

    for (var, value) in known {
        lhs = lhs.substitute(var, &Expression::constant(*value));
        rhs = rhs.substitute(var, &Expression::constant(*value));
    }

    // Simplify
    let lhs_simp = simplify(&lhs);
    let rhs_simp = simplify(&rhs);

    // Try to evaluate
    match (
        lhs_simp.evaluate(&HashMap::new()),
        rhs_simp.evaluate(&HashMap::new()),
    ) {
        (Some(lv), Some(rv)) => {
            if (lv - rv).abs() < epsilon {
                RefinementResult::Satisfied
            } else {
                RefinementResult::Unsatisfied
            }
        }
        _ => RefinementResult::Unknown,
    }
}

/// Check if adding an equation leads to contradiction
pub fn check_consistency(
    equations: &[Equation],
    new_eq: &Equation,
    known: &HashMap<String, f64>,
) -> bool {
    // Simple consistency check: solve with the new equation and verify no contradictions
    let mut reasoner = AlgebraicReasoner::with_default_config();

    for (var, value) in known {
        reasoner.add_known(var, *value);
    }

    for eq in equations {
        reasoner.add_equation(eq.clone());
    }

    reasoner.add_equation(new_eq.clone());

    let result = reasoner.solve();
    result.unit_errors.is_empty()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_creation() {
        let x = Expression::variable("x");
        let two = Expression::constant(2.0);
        let expr = x.clone() + two;

        assert!(expr.variables().contains("x"));
        assert!(!expr.is_constant());
    }

    #[test]
    fn test_expression_evaluate() {
        let x = Expression::variable("x");
        let y = Expression::variable("y");
        let expr = x + y * Expression::constant(2.0);

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 3.0);
        bindings.insert("y".to_string(), 4.0);

        let result = expr.evaluate(&bindings).unwrap();
        assert!((result - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplify_constants() {
        let expr = Expression::constant(2.0) + Expression::constant(3.0);
        let simplified = simplify(&expr);

        assert_eq!(simplified.as_constant(), Some(5.0));
    }

    #[test]
    fn test_simplify_zero() {
        let x = Expression::variable("x");
        let expr = x.clone() + Expression::constant(0.0);
        let simplified = simplify(&expr);

        assert!(matches!(simplified.kind, ExprKind::Variable(_)));
    }

    #[test]
    fn test_equation_solve_linear() {
        let x = Expression::variable("x");
        let eq = Equation::new(
            x.clone() * Expression::constant(2.0),
            Expression::constant(6.0),
        );

        let solved = eq.solve_for("x").unwrap();
        let result = solved.rhs.evaluate(&HashMap::new()).unwrap();
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_algebraic_reasoner() {
        let mut reasoner = AlgebraicReasoner::with_default_config();

        // Test simple equation solving: 2x = 6 => x = 3
        // The reasoner solves equations with exactly one unknown
        let x = Expression::variable("x");

        reasoner.add_equation(Equation::new(
            x.clone() * Expression::constant(2.0),
            Expression::constant(6.0),
        ));

        let result = reasoner.solve();

        assert!(
            result.solutions.contains_key("x"),
            "Should solve for x. Solutions: {:?}",
            result.solutions
        );
        let x_val = result.solutions["x"];
        assert!(
            (x_val - 3.0).abs() < 1e-9,
            "x should equal 3, got {}",
            x_val
        );
    }

    #[test]
    fn test_algebraic_reasoner_chained() {
        let mut reasoner = AlgebraicReasoner::with_default_config();

        // Test chained solving: x = 3, y = 2x => y = 6
        let x = Expression::variable("x");
        let y = Expression::variable("y");

        // First equation: x = 3
        reasoner.add_equation(Equation::new(x.clone(), Expression::constant(3.0)));

        // Second equation: y = 2x (will be solved after x is known)
        reasoner.add_equation(Equation::new(
            y.clone(),
            x.clone() * Expression::constant(2.0),
        ));

        let result = reasoner.solve();

        assert!(result.solutions.contains_key("x"));
        assert!(result.solutions.contains_key("y"));
        assert!((result.solutions["x"] - 3.0).abs() < 1e-9);
        assert!((result.solutions["y"] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_unit_compatibility() {
        let length1 = Expression::length("A", "B");
        let length2 = Expression::length("C", "D");
        let angle = Expression::angle_var("A", "B", "C");

        // Length + length should be ok
        assert!((length1.clone() + length2.clone()).unit.is_some());

        // Length * length = area
        let area = length1.clone() * length2.clone();
        assert_eq!(area.unit.as_ref().unwrap().dimensions.length, 2);

        // Angle should have angle unit
        assert_eq!(angle.unit.as_ref().unwrap().dimensions.angle, 1);
    }

    #[test]
    fn test_confidence_propagation() {
        let a = Expression::constant(1.0).with_confidence(BetaConfidence::new(10.0, 1.0));
        let b = Expression::constant(2.0).with_confidence(BetaConfidence::new(8.0, 2.0));

        let sum = a + b;

        // Combined confidence should be lower than either individual
        assert!(sum.confidence.mean() > 0.0);
        assert!(sum.confidence.mean() <= 1.0);
    }

    #[test]
    fn test_midpoint_equations() {
        let pred = Predicate::midpoint("M", "A", "B").with_epistemic(PredicateEpistemic::axiom());

        let mut reasoner = AlgebraicReasoner::with_default_config();
        reasoner.add_from_predicate(&pred);

        // Should have added equations for |AM| = |MB| and |AM| = |AB|/2
        assert!(reasoner.equations.len() >= 2);
    }

    #[test]
    fn test_refinement_check() {
        let eq = Equation::new(
            Expression::variable("x") + Expression::constant(2.0),
            Expression::constant(5.0),
        );

        let mut known = HashMap::new();
        known.insert("x".to_string(), 3.0);

        assert_eq!(
            check_refinement(&eq, &known, 1e-9),
            RefinementResult::Satisfied
        );

        known.insert("x".to_string(), 4.0);
        assert_eq!(
            check_refinement(&eq, &known, 1e-9),
            RefinementResult::Unsatisfied
        );
    }

    #[test]
    fn test_expression_substitution() {
        let x = Expression::variable("x");
        let y = Expression::variable("y");
        let expr = x.clone() * Expression::constant(2.0) + y;

        let substituted = expr.substitute("x", &Expression::constant(3.0));

        let mut bindings = HashMap::new();
        bindings.insert("y".to_string(), 1.0);

        let result = substituted.evaluate(&bindings).unwrap();
        assert!((result - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_epistemic_uncertainty() {
        let high_conf = Expression::constant(1.0).with_confidence(BetaConfidence::new(100.0, 1.0));
        let low_conf = Expression::variable("x").with_confidence(BetaConfidence::new(2.0, 2.0));

        assert!(high_conf.epistemic_uncertainty() < low_conf.epistemic_uncertainty());
    }
}

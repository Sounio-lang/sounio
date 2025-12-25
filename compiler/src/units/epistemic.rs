//! Integration of Units with Epistemic Types
//!
//! This module provides the bridge between physical types (units of measure)
//! and epistemic types (Knowledge[τ, ε, δ, Φ]). The key type is
//! `QuantifiedKnowledge` which represents a value with both physical
//! dimension and epistemic metadata.
//!
//! # Motivation
//!
//! In scientific computing, values are rarely just numbers with units.
//! They also carry:
//! - **Confidence**: How certain are we? (95% CI, SD, measurement error)
//! - **Provenance**: Where did this value come from? (experiment, literature, model)
//! - **Validation**: What domain constraints apply? (positive, physiological range)
//!
//! # Example
//!
//! ```sounio
//! // A plasma concentration with full epistemic tracking
//! let Cmax: QuantifiedKnowledge[
//!     value = f64,
//!     unit = mg/L,
//!     confidence = 0.95,
//!     source = Measurement(LC-MS/MS),
//!     domain = PKPD:PlasmaConcentration
//! ] = measure_plasma_conc(sample);
//!
//! // Operations preserve both unit and epistemic information
//! let AUC = integrate(concentration_curve, time);
//! // AUC has unit mg·h/L and tracks integration as transformation
//! ```

use std::marker::PhantomData;

use super::dimension::Dimension;
use super::quantity::Quantity;
use super::si::base::Unit;
use crate::epistemic::ContextTime;
use crate::epistemic::confidence::EpistemicStatus;
use crate::epistemic::knowledge::OntologyBinding;
use crate::epistemic::provenance::{Provenance, Transformation};

// ============================================================================
// QUANTIFIED KNOWLEDGE
// ============================================================================

/// A value with both physical unit and epistemic metadata.
///
/// This is the fundamental type for scientific computing in Sounio,
/// combining compile-time unit checking with runtime epistemic tracking.
///
/// # Type Parameters
///
/// - `N`: Numeric type (f64, f32, etc.)
/// - `U`: Unit type (compile-time dimensional analysis)
#[derive(Debug, Clone)]
pub struct QuantifiedKnowledge<N, U: Unit> {
    /// The physical quantity with its unit
    pub quantity: Quantity<N, U>,

    /// Epistemic status (confidence, source, revisability)
    pub epistemic: EpistemicStatus,

    /// Domain ontology binding
    pub domain: Option<OntologyBinding>,

    /// Transformation provenance
    pub provenance: Provenance,

    /// Temporal context
    pub temporal: ContextTime,
}

impl<N, U: Unit> QuantifiedKnowledge<N, U> {
    /// Create new quantified knowledge with full metadata
    pub fn new(
        value: N,
        epistemic: EpistemicStatus,
        domain: Option<OntologyBinding>,
        provenance: Provenance,
        temporal: ContextTime,
    ) -> Self {
        Self {
            quantity: Quantity::new(value),
            epistemic,
            domain,
            provenance,
            temporal,
        }
    }

    /// Create from a quantity with default epistemic status (measurement)
    pub fn from_measurement(quantity: Quantity<N, U>, confidence: f64, instrument: &str) -> Self {
        Self {
            quantity,
            epistemic: EpistemicStatus::from_measurement(confidence, instrument),
            domain: None,
            provenance: Provenance::computed("measurement"),
            temporal: ContextTime::current(),
        }
    }

    /// Create from a quantity with axiomatic status (known constant)
    pub fn from_constant(quantity: Quantity<N, U>) -> Self {
        Self {
            quantity,
            epistemic: EpistemicStatus::axiomatic(),
            domain: None,
            provenance: Provenance::literal(),
            temporal: ContextTime::current(),
        }
    }

    /// Create from a quantity with derived status (computed value)
    pub fn from_derived(quantity: Quantity<N, U>, derivation: &str) -> Self {
        Self {
            quantity,
            epistemic: EpistemicStatus::derived(&[], derivation),
            domain: None,
            provenance: Provenance::computed(derivation),
            temporal: ContextTime::current(),
        }
    }

    /// Get the confidence level
    pub fn confidence(&self) -> f64 {
        self.epistemic.confidence.value()
    }

    /// Get the dimension of the unit
    pub fn dimension(&self) -> Dimension {
        U::DIMENSION
    }

    /// Get the unit symbol
    pub fn unit_symbol(&self) -> &'static str {
        U::SYMBOL
    }

    /// Check if this knowledge is revisable
    pub fn is_revisable(&self) -> bool {
        self.epistemic.revisability.is_revisable()
    }

    /// Apply a transformation, updating provenance and potentially confidence
    pub fn transform(self, transformation: Transformation) -> Self {
        Self {
            quantity: self.quantity,
            epistemic: self.epistemic.propagate(&transformation),
            domain: self.domain,
            provenance: self.provenance.extend(transformation),
            temporal: self.temporal,
        }
    }

    /// Bind to a domain ontology term
    pub fn bind_domain(mut self, binding: OntologyBinding) -> Self {
        self.domain = Some(binding);
        self
    }
}

impl<N: Copy, U: Unit> QuantifiedKnowledge<N, U> {
    /// Get the raw numeric value
    pub fn value(&self) -> N {
        *self.quantity.value()
    }
}

// ============================================================================
// ARITHMETIC OPERATIONS (same unit)
// ============================================================================

impl<N, U> std::ops::Add for QuantifiedKnowledge<N, U>
where
    N: std::ops::Add<Output = N>,
    U: Unit,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Combine epistemic status using derived
        let combined_epistemic =
            EpistemicStatus::derived(&[&self.epistemic, &rhs.epistemic], "addition");

        // Combine provenance
        let combined_provenance = self.provenance.extend(Transformation::function("add"));

        Self {
            quantity: Quantity::new(self.quantity.into_value() + rhs.quantity.into_value()),
            epistemic: combined_epistemic,
            domain: self.domain.or(rhs.domain), // Prefer left's domain
            provenance: combined_provenance,
            temporal: self.temporal, // Keep left's temporal context
        }
    }
}

impl<N, U> std::ops::Sub for QuantifiedKnowledge<N, U>
where
    N: std::ops::Sub<Output = N>,
    U: Unit,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let combined_epistemic =
            EpistemicStatus::derived(&[&self.epistemic, &rhs.epistemic], "subtraction");
        let combined_provenance = self.provenance.extend(Transformation::function("sub"));

        Self {
            quantity: Quantity::new(self.quantity.into_value() - rhs.quantity.into_value()),
            epistemic: combined_epistemic,
            domain: self.domain.or(rhs.domain),
            provenance: combined_provenance,
            temporal: self.temporal, // Keep left's temporal context
        }
    }
}

// ============================================================================
// SCALAR OPERATIONS
// ============================================================================

impl<N, U> std::ops::Mul<N> for QuantifiedKnowledge<N, U>
where
    N: std::ops::Mul<Output = N>,
    U: Unit,
{
    type Output = Self;

    fn mul(self, scalar: N) -> Self::Output {
        Self {
            quantity: Quantity::new(self.quantity.into_value() * scalar),
            epistemic: self.epistemic,
            domain: self.domain,
            provenance: self.provenance,
            temporal: self.temporal,
        }
    }
}

impl<N, U> std::ops::Div<N> for QuantifiedKnowledge<N, U>
where
    N: std::ops::Div<Output = N>,
    U: Unit,
{
    type Output = Self;

    fn div(self, scalar: N) -> Self::Output {
        Self {
            quantity: Quantity::new(self.quantity.into_value() / scalar),
            epistemic: self.epistemic,
            domain: self.domain,
            provenance: self.provenance,
            temporal: self.temporal,
        }
    }
}

// ============================================================================
// DYNAMIC QUANTIFIED KNOWLEDGE
// ============================================================================

/// Runtime-typed quantified knowledge for dynamic unit handling.
///
/// Used when units are not known at compile time, such as when
/// parsing user input or loading from databases.
#[derive(Debug, Clone)]
pub struct DynamicQuantifiedKnowledge {
    /// The numeric value
    pub value: f64,

    /// Dimension of the quantity
    pub dimension: Dimension,

    /// Scale factor relative to SI base
    pub scale: f64,

    /// Unit symbol
    pub symbol: String,

    /// Epistemic status
    pub epistemic: EpistemicStatus,

    /// Domain binding
    pub domain: Option<OntologyBinding>,

    /// Provenance
    pub provenance: Provenance,

    /// Temporal context
    pub temporal: ContextTime,
}

impl DynamicQuantifiedKnowledge {
    /// Create new dynamic quantified knowledge
    pub fn new(
        value: f64,
        dimension: Dimension,
        scale: f64,
        symbol: impl Into<String>,
        epistemic: EpistemicStatus,
    ) -> Self {
        Self {
            value,
            dimension,
            scale,
            symbol: symbol.into(),
            epistemic,
            domain: None,
            provenance: Provenance::literal(),
            temporal: ContextTime::current(),
        }
    }

    /// Create from a static quantified knowledge (type erasure)
    pub fn from_static<N: Into<f64>, U: Unit>(qk: QuantifiedKnowledge<N, U>) -> Self {
        Self {
            value: qk.quantity.into_value().into(),
            dimension: U::DIMENSION,
            scale: U::SCALE,
            symbol: U::SYMBOL.to_string(),
            epistemic: qk.epistemic,
            domain: qk.domain,
            provenance: qk.provenance,
            temporal: qk.temporal,
        }
    }

    /// Get the confidence level
    pub fn confidence(&self) -> f64 {
        self.epistemic.confidence.value()
    }

    /// Check if dimensions match
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.dimension == other.dimension
    }

    /// Convert to different unit (same dimension)
    pub fn convert_to(&self, target_scale: f64, target_symbol: impl Into<String>) -> Option<Self> {
        Some(Self {
            value: self.value * self.scale / target_scale,
            dimension: self.dimension,
            scale: target_scale,
            symbol: target_symbol.into(),
            epistemic: self.epistemic.clone(),
            domain: self.domain.clone(),
            provenance: self.provenance.clone(),
            temporal: self.temporal.clone(),
        })
    }
}

// ============================================================================
// CONFIDENCE PROPAGATION
// ============================================================================

/// Rules for propagating confidence through operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfidencePropagation {
    /// Take minimum confidence of inputs
    Minimum,
    /// Multiply confidences (independent events)
    Multiplicative,
    /// Quadrature (root sum of squares for uncertainties)
    Quadrature,
    /// Custom reduction factor
    CustomFactor(f64),
}

impl ConfidencePropagation {
    /// Apply propagation rule to combine confidences
    pub fn combine(&self, a: f64, b: f64) -> f64 {
        match self {
            Self::Minimum => a.min(b),
            Self::Multiplicative => a * b,
            Self::Quadrature => {
                // For uncertainties: sqrt(u1² + u2²)
                // For confidences: 1 - sqrt((1-c1)² + (1-c2)²)
                let u1 = 1.0 - a;
                let u2 = 1.0 - b;
                (1.0 - (u1 * u1 + u2 * u2).sqrt()).max(0.0)
            }
            Self::CustomFactor(f) => (a * b * f).min(1.0),
        }
    }
}

// ============================================================================
// BUILDER PATTERN
// ============================================================================

/// Builder for constructing QuantifiedKnowledge with fluent API
pub struct QuantifiedKnowledgeBuilder<N, U: Unit> {
    value: N,
    epistemic: Option<EpistemicStatus>,
    domain: Option<OntologyBinding>,
    provenance: Option<Provenance>,
    temporal: Option<ContextTime>,
    _unit: PhantomData<U>,
}

impl<N, U: Unit> QuantifiedKnowledgeBuilder<N, U> {
    /// Create a new builder with the given value
    pub fn new(value: N) -> Self {
        Self {
            value,
            epistemic: None,
            domain: None,
            provenance: None,
            temporal: None,
            _unit: PhantomData,
        }
    }

    /// Mark as measurement with confidence and instrument
    pub fn measured(mut self, confidence: f64, instrument: &str) -> Self {
        self.epistemic = Some(EpistemicStatus::from_measurement(confidence, instrument));
        self.provenance = Some(Provenance::computed("measurement"));
        self
    }

    /// Mark as derived/computed
    pub fn derived(mut self, derivation: &str) -> Self {
        self.epistemic = Some(EpistemicStatus::derived(&[], derivation));
        self.provenance = Some(Provenance::computed(derivation));
        self
    }

    /// Mark as axiomatic (known constant)
    pub fn axiomatic(mut self) -> Self {
        self.epistemic = Some(EpistemicStatus::axiomatic());
        self.provenance = Some(Provenance::literal());
        self
    }

    /// Bind to domain ontology
    pub fn domain(mut self, binding: OntologyBinding) -> Self {
        self.domain = Some(binding);
        self
    }

    /// Set provenance
    pub fn provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Set temporal context
    pub fn temporal(mut self, temporal: ContextTime) -> Self {
        self.temporal = Some(temporal);
        self
    }

    /// Build the QuantifiedKnowledge
    pub fn build(self) -> QuantifiedKnowledge<N, U> {
        QuantifiedKnowledge {
            quantity: Quantity::new(self.value),
            epistemic: self.epistemic.unwrap_or_default(),
            domain: self.domain,
            provenance: self.provenance.unwrap_or_else(Provenance::literal),
            temporal: self.temporal.unwrap_or_else(ContextTime::current),
        }
    }
}

// ============================================================================
// EXTENSION TRAITS
// ============================================================================

/// Extension trait for creating QuantifiedKnowledge from Quantity
pub trait WithEpistemic<N, U: Unit>: Sized {
    /// Wrap in QuantifiedKnowledge with measurement status
    fn measured(self, confidence: f64, instrument: &str) -> QuantifiedKnowledge<N, U>;

    /// Wrap in QuantifiedKnowledge as axiomatic constant
    fn constant(self) -> QuantifiedKnowledge<N, U>;

    /// Wrap in QuantifiedKnowledge as derived value
    fn derived(self, derivation: &str) -> QuantifiedKnowledge<N, U>;

    /// Start building QuantifiedKnowledge with fluent API
    fn with_epistemic(self) -> QuantifiedKnowledgeBuilder<N, U>;
}

impl<N, U: Unit> WithEpistemic<N, U> for Quantity<N, U> {
    fn measured(self, confidence: f64, instrument: &str) -> QuantifiedKnowledge<N, U> {
        QuantifiedKnowledge::from_measurement(self, confidence, instrument)
    }

    fn constant(self) -> QuantifiedKnowledge<N, U> {
        QuantifiedKnowledge::from_constant(self)
    }

    fn derived(self, derivation: &str) -> QuantifiedKnowledge<N, U> {
        QuantifiedKnowledge::from_derived(self, derivation)
    }

    fn with_epistemic(self) -> QuantifiedKnowledgeBuilder<N, U> {
        QuantifiedKnowledgeBuilder::new(self.into_value())
    }
}

// ============================================================================
// DISPLAY
// ============================================================================

impl<N: std::fmt::Display, U: Unit> std::fmt::Display for QuantifiedKnowledge<N, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} (confidence: {:.1}%)",
            self.quantity.value(),
            U::SYMBOL,
            self.confidence() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::si::prefixes::Milligram;

    #[test]
    fn test_quantified_knowledge_creation() {
        let qk: QuantifiedKnowledge<f64, Milligram> = QuantifiedKnowledgeBuilder::new(500.0)
            .measured(0.95, "LC-MS/MS")
            .build();

        assert_eq!(qk.value(), 500.0);
        assert!((qk.confidence() - 0.95).abs() < 0.001);
        assert_eq!(qk.unit_symbol(), "mg");
    }

    #[test]
    fn test_scalar_multiplication() {
        let qk: QuantifiedKnowledge<f64, Milligram> = QuantifiedKnowledgeBuilder::new(100.0)
            .measured(0.95, "scale")
            .build();

        let doubled = qk * 2.0;
        assert_eq!(doubled.value(), 200.0);
        // Confidence should be preserved for scalar ops
        assert!((doubled.confidence() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_confidence_propagation_rules() {
        let min = ConfidencePropagation::Minimum.combine(0.95, 0.90);
        assert!((min - 0.90).abs() < 0.001);

        let mult = ConfidencePropagation::Multiplicative.combine(0.95, 0.90);
        assert!((mult - 0.855).abs() < 0.001);
    }
}

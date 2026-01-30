//! EpistemicValue<RefinementType> Integration
//!
//! This module integrates epistemic uncertainty tracking with refinement types,
//! enabling values that carry both:
//! - Refinement guarantees (the value IS in the specified bounds)
//! - Epistemic uncertainty (our confidence in the measurement/derivation)
//!
//! # Key Innovation
//!
//! When combining epistemic uncertainty with refinement types, we face a fundamental
//! challenge: refinement types provide hard guarantees (`x > 0`), while epistemic
//! uncertainty suggests the value might deviate from its nominal form.
//!
//! Our approach uses **interval-aware epistemic propagation**:
//!
//! 1. **Epistemic Interval Widening**: Low confidence widens the effective bounds
//!    while preserving the refinement guarantee as the outer envelope.
//!
//! 2. **Sound Bound Propagation**: Operations track how uncertainty affects bounds
//!    using affine arithmetic principles.
//!
//! 3. **Confidence-Weighted Verification**: SMT verification considers confidence
//!    when checking refinement predicates.
//!
//! # Example
//!
//! ```sounio
//! // A dose with uncertainty but guaranteed positive
//! let dose: EpistemicRefined[f64, { d | d > 0 && d <= 500 }] =
//!     measure_dose(sample)  // confidence: 0.95
//!
//! // The effective bounds are:
//! // - Refinement guarantee: (0, 500]
//! // - Epistemic interval: nominal +/- uncertainty based on confidence
//! // - Both must be respected for safe extraction
//! ```
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Liquid Types (Rondon et al., 2008) for refinement inference
//! - Abstraction-refinement for hierarchical probabilistic models
//! - Affine arithmetic for correlated uncertainty tracking

use std::collections::HashSet;
use std::fmt;

use super::composition::confidence::ConfidenceValue;
use super::composition::knowledge::{EpistemicValue, OntologyRef};
use super::composition::provenance::ProvenanceNode;
// Note: UncertainValue, UncertaintyModel, IntervalConfig, AffineConfig are available
// for future extension with different uncertainty propagation models

/// A refined epistemic value: combines refinement guarantees with uncertainty
///
/// This is the core type for `EpistemicValue<RefinementType>`.
///
/// # Type Parameters
/// - `T`: The base value type (typically f64 for numerical refinements)
#[derive(Debug, Clone)]
pub struct EpistemicRefinedValue<T> {
    /// The nominal value
    pub nominal: T,

    /// Epistemic confidence [0, 1]
    pub confidence: ConfidenceValue,

    /// The refinement bounds (as an interval)
    pub refinement_bounds: RefinementBounds,

    /// Effective epistemic interval (may be narrower than refinement bounds)
    pub epistemic_interval: EpistemicInterval,

    /// Ontology bindings for domain validation
    pub ontology: HashSet<OntologyRef>,

    /// Provenance tracking
    pub provenance: ProvenanceNode,

    /// The refinement predicate as a string (for display/debugging)
    pub refinement_predicate: String,
}

/// The hard bounds from the refinement type
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementBounds {
    /// Lower bound (None = unbounded below)
    pub lower: Option<f64>,
    /// Whether lower bound is inclusive
    pub lower_inclusive: bool,
    /// Upper bound (None = unbounded above)
    pub upper: Option<f64>,
    /// Whether upper bound is inclusive
    pub upper_inclusive: bool,
}

impl RefinementBounds {
    /// Unbounded (no refinement constraints)
    pub fn unbounded() -> Self {
        Self {
            lower: None,
            lower_inclusive: false,
            upper: None,
            upper_inclusive: false,
        }
    }

    /// Create positive bounds: x > 0
    pub fn positive() -> Self {
        Self {
            lower: Some(0.0),
            lower_inclusive: false,
            upper: None,
            upper_inclusive: false,
        }
    }

    /// Create non-negative bounds: x >= 0
    pub fn non_negative() -> Self {
        Self {
            lower: Some(0.0),
            lower_inclusive: true,
            upper: None,
            upper_inclusive: false,
        }
    }

    /// Create closed interval: lower <= x <= upper
    pub fn closed(lower: f64, upper: f64) -> Self {
        Self {
            lower: Some(lower),
            lower_inclusive: true,
            upper: Some(upper),
            upper_inclusive: true,
        }
    }

    /// Create half-open interval: lower < x <= upper
    pub fn half_open_left(lower: f64, upper: f64) -> Self {
        Self {
            lower: Some(lower),
            lower_inclusive: false,
            upper: Some(upper),
            upper_inclusive: true,
        }
    }

    /// Check if a value satisfies these bounds
    pub fn contains(&self, value: f64) -> bool {
        let lower_ok = match self.lower {
            None => true,
            Some(l) if self.lower_inclusive => value >= l,
            Some(l) => value > l,
        };

        let upper_ok = match self.upper {
            None => true,
            Some(u) if self.upper_inclusive => value <= u,
            Some(u) => value < u,
        };

        lower_ok && upper_ok
    }

    /// Get the effective lower bound as f64
    pub fn effective_lower(&self) -> f64 {
        self.lower.unwrap_or(f64::NEG_INFINITY)
    }

    /// Get the effective upper bound as f64
    pub fn effective_upper(&self) -> f64 {
        self.upper.unwrap_or(f64::INFINITY)
    }

    /// Intersect with another bounds (for combining refinements)
    pub fn intersect(&self, other: &RefinementBounds) -> RefinementBounds {
        let (new_lower, new_lower_incl) = match (self.lower, other.lower) {
            (None, None) => (None, false),
            (Some(l), None) => (Some(l), self.lower_inclusive),
            (None, Some(l)) => (Some(l), other.lower_inclusive),
            (Some(l1), Some(l2)) => {
                if l1 > l2 {
                    (Some(l1), self.lower_inclusive)
                } else if l2 > l1 {
                    (Some(l2), other.lower_inclusive)
                } else {
                    (Some(l1), self.lower_inclusive && other.lower_inclusive)
                }
            }
        };

        let (new_upper, new_upper_incl) = match (self.upper, other.upper) {
            (None, None) => (None, false),
            (Some(u), None) => (Some(u), self.upper_inclusive),
            (None, Some(u)) => (Some(u), other.upper_inclusive),
            (Some(u1), Some(u2)) => {
                if u1 < u2 {
                    (Some(u1), self.upper_inclusive)
                } else if u2 < u1 {
                    (Some(u2), other.upper_inclusive)
                } else {
                    (Some(u1), self.upper_inclusive && other.upper_inclusive)
                }
            }
        };

        RefinementBounds {
            lower: new_lower,
            lower_inclusive: new_lower_incl,
            upper: new_upper,
            upper_inclusive: new_upper_incl,
        }
    }
}

impl fmt::Display for RefinementBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lower = match self.lower {
            None => "-inf".to_string(),
            Some(v) => format!("{:.4}", v),
        };
        let upper = match self.upper {
            None => "+inf".to_string(),
            Some(v) => format!("{:.4}", v),
        };
        let left_bracket = if self.lower_inclusive { '[' } else { '(' };
        let right_bracket = if self.upper_inclusive { ']' } else { ')' };
        write!(f, "{}{}, {}{}", left_bracket, lower, upper, right_bracket)
    }
}

/// The epistemic interval - represents uncertainty about the actual value
///
/// This interval is always contained within (or equal to) the refinement bounds.
/// It represents where we believe the true value lies given our confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicInterval {
    /// Center (nominal) value
    pub center: f64,
    /// Half-width of uncertainty interval
    pub half_width: f64,
}

impl EpistemicInterval {
    /// Create a point interval (no uncertainty)
    pub fn point(value: f64) -> Self {
        Self {
            center: value,
            half_width: 0.0,
        }
    }

    /// Create an interval from center and half-width
    pub fn from_center_width(center: f64, half_width: f64) -> Self {
        Self {
            center,
            half_width: half_width.abs(),
        }
    }

    /// Create from bounds
    pub fn from_bounds(lower: f64, upper: f64) -> Self {
        let center = (lower + upper) / 2.0;
        let half_width = (upper - lower) / 2.0;
        Self { center, half_width }
    }

    /// Create from confidence and relative uncertainty
    ///
    /// Higher confidence = narrower interval
    pub fn from_confidence(center: f64, confidence: f64, base_uncertainty: f64) -> Self {
        // Map confidence to interval width
        // At confidence 1.0 -> point interval
        // At confidence 0.0 -> maximum uncertainty
        let uncertainty_factor = 1.0 - confidence;
        let half_width = center.abs() * base_uncertainty * uncertainty_factor;
        Self { center, half_width }
    }

    /// Get lower bound
    pub fn lower(&self) -> f64 {
        self.center - self.half_width
    }

    /// Get upper bound
    pub fn upper(&self) -> f64 {
        self.center + self.half_width
    }

    /// Check if a value is within this interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower() && value <= self.upper()
    }

    /// Clamp this interval to refinement bounds
    pub fn clamp_to_bounds(&self, bounds: &RefinementBounds) -> Self {
        let lower = self.lower().max(bounds.effective_lower());
        let upper = self.upper().min(bounds.effective_upper());

        // Handle case where clamping inverts the interval
        if lower >= upper {
            let center = (lower + upper) / 2.0;
            Self {
                center,
                half_width: 0.0,
            }
        } else {
            Self::from_bounds(lower, upper)
        }
    }

    /// Add two intervals (interval arithmetic)
    pub fn add(&self, other: &Self) -> Self {
        Self {
            center: self.center + other.center,
            half_width: self.half_width + other.half_width,
        }
    }

    /// Subtract intervals (interval arithmetic)
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            center: self.center - other.center,
            half_width: self.half_width + other.half_width,
        }
    }

    /// Multiply intervals (interval arithmetic)
    pub fn mul(&self, other: &Self) -> Self {
        // For multiplication, we need to consider all corner products
        let corners = [
            self.lower() * other.lower(),
            self.lower() * other.upper(),
            self.upper() * other.lower(),
            self.upper() * other.upper(),
        ];

        let min = corners.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = corners.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self::from_bounds(min, max)
    }

    /// Divide intervals (interval arithmetic)
    pub fn div(&self, other: &Self) -> Option<Self> {
        // Division by interval containing zero is undefined
        if other.lower() <= 0.0 && other.upper() >= 0.0 {
            return None;
        }

        let corners = [
            self.lower() / other.lower(),
            self.lower() / other.upper(),
            self.upper() / other.lower(),
            self.upper() / other.upper(),
        ];

        let min = corners.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = corners.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Some(Self::from_bounds(min, max))
    }

    /// Scale interval by a constant
    pub fn scale(&self, factor: f64) -> Self {
        if factor >= 0.0 {
            Self {
                center: self.center * factor,
                half_width: self.half_width * factor,
            }
        } else {
            Self {
                center: self.center * factor,
                half_width: self.half_width * factor.abs(),
            }
        }
    }
}

impl fmt::Display for EpistemicInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lower(), self.upper())
    }
}

/// Configuration for epistemic-refined operations
#[derive(Debug, Clone)]
pub struct EpistemicRefinedConfig {
    /// Base uncertainty factor (relative to value magnitude)
    pub base_uncertainty: f64,

    /// Minimum confidence for extraction
    pub min_extraction_confidence: f64,

    /// Whether to use affine arithmetic for correlated errors
    pub use_affine_arithmetic: bool,

    /// Maximum interval width before warning
    pub max_interval_width: Option<f64>,

    /// Confidence degradation factor for operations
    pub operation_degradation: f64,
}

impl Default for EpistemicRefinedConfig {
    fn default() -> Self {
        Self {
            base_uncertainty: 0.1, // 10% base uncertainty at confidence 0
            min_extraction_confidence: 0.5,
            use_affine_arithmetic: true,
            max_interval_width: None,
            operation_degradation: 0.99,
        }
    }
}

impl<T: Clone> EpistemicRefinedValue<T> {
    /// Get a reference to the nominal value
    pub fn value(&self) -> &T {
        &self.nominal
    }

    /// Get the confidence level
    pub fn confidence_value(&self) -> f64 {
        self.confidence.value()
    }

    /// Check if this value satisfies its refinement at the epistemic level
    pub fn is_refinement_satisfied(&self) -> bool {
        // Both the epistemic interval bounds must satisfy refinement
        self.refinement_bounds
            .contains(self.epistemic_interval.lower())
            && self
                .refinement_bounds
                .contains(self.epistemic_interval.upper())
    }

    /// Get the effective interval considering both refinement and epistemic bounds
    pub fn effective_interval(&self) -> EpistemicInterval {
        self.epistemic_interval
            .clamp_to_bounds(&self.refinement_bounds)
    }
}

impl EpistemicRefinedValue<f64> {
    /// Create a new epistemic refined value with explicit bounds
    pub fn new(
        value: f64,
        confidence: f64,
        refinement_bounds: RefinementBounds,
        predicate: impl Into<String>,
    ) -> Result<Self, RefinedCreationError> {
        let conf = ConfidenceValue::new(confidence)
            .map_err(|_| RefinedCreationError::InvalidConfidence(confidence))?;

        if !refinement_bounds.contains(value) {
            return Err(RefinedCreationError::ValueViolatesRefinement {
                value,
                bounds: refinement_bounds,
            });
        }

        let config = EpistemicRefinedConfig::default();
        let epistemic_interval =
            EpistemicInterval::from_confidence(value, confidence, config.base_uncertainty)
                .clamp_to_bounds(&refinement_bounds);

        Ok(Self {
            nominal: value,
            confidence: conf,
            refinement_bounds,
            epistemic_interval,
            ontology: HashSet::new(),
            provenance: ProvenanceNode::primitive(),
            refinement_predicate: predicate.into(),
        })
    }

    /// Create a certain (confidence = 1.0) refined value
    pub fn certain(
        value: f64,
        refinement_bounds: RefinementBounds,
    ) -> Result<Self, RefinedCreationError> {
        Self::new(value, 1.0, refinement_bounds, "")
    }

    /// Create from measurement with typical uncertainty
    pub fn from_measurement(
        value: f64,
        measurement_uncertainty: f64,
        confidence: f64,
        refinement_bounds: RefinementBounds,
    ) -> Result<Self, RefinedCreationError> {
        let conf = ConfidenceValue::new(confidence)
            .map_err(|_| RefinedCreationError::InvalidConfidence(confidence))?;

        if !refinement_bounds.contains(value) {
            return Err(RefinedCreationError::ValueViolatesRefinement {
                value,
                bounds: refinement_bounds,
            });
        }

        let epistemic_interval =
            EpistemicInterval::from_center_width(value, measurement_uncertainty)
                .clamp_to_bounds(&refinement_bounds);

        Ok(Self {
            nominal: value,
            confidence: conf,
            refinement_bounds,
            epistemic_interval,
            ontology: HashSet::new(),
            provenance: ProvenanceNode::primitive(),
            refinement_predicate: String::new(),
        })
    }

    /// Add ontology binding
    pub fn with_ontology(mut self, ont_ref: OntologyRef) -> Self {
        self.ontology.insert(ont_ref);
        self
    }

    /// Add provenance
    pub fn with_provenance(mut self, provenance: ProvenanceNode) -> Self {
        self.provenance = provenance;
        self
    }

    /// Map the value while preserving refinement
    ///
    /// The new refinement bounds must be provided as they may change under the operation.
    pub fn map_refined<F>(
        self,
        f: F,
        new_bounds: RefinementBounds,
        confidence_factor: f64,
    ) -> Result<Self, RefinedCreationError>
    where
        F: FnOnce(f64) -> f64,
    {
        let new_value = f(self.nominal);
        let new_confidence = self.confidence.value() * confidence_factor;

        if !new_bounds.contains(new_value) {
            return Err(RefinedCreationError::ValueViolatesRefinement {
                value: new_value,
                bounds: new_bounds,
            });
        }

        let conf = ConfidenceValue::new(new_confidence)
            .map_err(|_| RefinedCreationError::InvalidConfidence(new_confidence))?;

        let config = EpistemicRefinedConfig::default();
        let epistemic_interval =
            EpistemicInterval::from_confidence(new_value, new_confidence, config.base_uncertainty)
                .clamp_to_bounds(&new_bounds);

        Ok(Self {
            nominal: new_value,
            confidence: conf,
            refinement_bounds: new_bounds,
            epistemic_interval,
            ontology: self.ontology,
            provenance: ProvenanceNode::derived("map_refined", vec![self.provenance]),
            refinement_predicate: String::new(),
        })
    }

    /// Add two refined epistemic values
    ///
    /// Uses interval arithmetic for sound uncertainty propagation.
    pub fn add(self, other: Self) -> Self {
        let new_nominal = self.nominal + other.nominal;
        let new_interval = self.epistemic_interval.add(&other.epistemic_interval);

        // Addition of bounded intervals
        let new_bounds = RefinementBounds {
            lower: match (self.refinement_bounds.lower, other.refinement_bounds.lower) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            lower_inclusive: self.refinement_bounds.lower_inclusive
                && other.refinement_bounds.lower_inclusive,
            upper: match (self.refinement_bounds.upper, other.refinement_bounds.upper) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            upper_inclusive: self.refinement_bounds.upper_inclusive
                && other.refinement_bounds.upper_inclusive,
        };

        // Confidence is minimum of operands (conservative)
        let new_conf = self.confidence.value().min(other.confidence.value());

        Self {
            nominal: new_nominal,
            confidence: ConfidenceValue::new(new_conf).unwrap_or(ConfidenceValue::uncertain()),
            refinement_bounds: new_bounds,
            epistemic_interval: new_interval.clamp_to_bounds(&new_bounds),
            ontology: self.ontology.union(&other.ontology).cloned().collect(),
            provenance: ProvenanceNode::derived("add", vec![self.provenance, other.provenance]),
            refinement_predicate: format!(
                "({}) + ({})",
                self.refinement_predicate, other.refinement_predicate
            ),
        }
    }

    /// Multiply two refined epistemic values
    pub fn mul(self, other: Self) -> Self {
        let new_nominal = self.nominal * other.nominal;
        let new_interval = self.epistemic_interval.mul(&other.epistemic_interval);

        // Multiplication of bounded intervals (complex - bounds depend on signs)
        let corners = [
            self.refinement_bounds.effective_lower() * other.refinement_bounds.effective_lower(),
            self.refinement_bounds.effective_lower() * other.refinement_bounds.effective_upper(),
            self.refinement_bounds.effective_upper() * other.refinement_bounds.effective_lower(),
            self.refinement_bounds.effective_upper() * other.refinement_bounds.effective_upper(),
        ];

        let min = corners
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .fold(f64::INFINITY, f64::min);
        let max = corners
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        let new_bounds = if min.is_finite() && max.is_finite() {
            RefinementBounds::closed(min, max)
        } else {
            RefinementBounds::unbounded()
        };

        let new_conf = self.confidence.value().min(other.confidence.value());

        Self {
            nominal: new_nominal,
            confidence: ConfidenceValue::new(new_conf).unwrap_or(ConfidenceValue::uncertain()),
            refinement_bounds: new_bounds,
            epistemic_interval: new_interval.clamp_to_bounds(&new_bounds),
            ontology: self.ontology.union(&other.ontology).cloned().collect(),
            provenance: ProvenanceNode::derived("mul", vec![self.provenance, other.provenance]),
            refinement_predicate: format!(
                "({}) * ({})",
                self.refinement_predicate, other.refinement_predicate
            ),
        }
    }

    /// Extract value if confidence meets threshold and refinement is satisfied
    pub fn extract(&self, min_confidence: f64) -> Option<f64> {
        if self.confidence.value() >= min_confidence && self.is_refinement_satisfied() {
            Some(self.nominal)
        } else {
            None
        }
    }

    /// Extract the effective interval (sound over-approximation)
    pub fn extract_interval(&self, min_confidence: f64) -> Option<(f64, f64)> {
        if self.confidence.value() >= min_confidence {
            let eff = self.effective_interval();
            Some((eff.lower(), eff.upper()))
        } else {
            None
        }
    }

    /// Force extract (bypasses confidence check, still respects refinement)
    pub fn force_extract(&self) -> f64 {
        self.nominal
    }
}

/// Errors that can occur when creating refined epistemic values
#[derive(Debug, Clone)]
pub enum RefinedCreationError {
    /// The value violates the refinement bounds
    ValueViolatesRefinement {
        value: f64,
        bounds: RefinementBounds,
    },
    /// Invalid confidence value (not in [0, 1])
    InvalidConfidence(f64),
    /// The epistemic interval cannot be contained in refinement bounds
    IntervalExceedsBounds {
        interval: EpistemicInterval,
        bounds: RefinementBounds,
    },
}

impl fmt::Display for RefinedCreationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValueViolatesRefinement { value, bounds } => {
                write!(f, "Value {} violates refinement bounds {}", value, bounds)
            }
            Self::InvalidConfidence(c) => {
                write!(f, "Invalid confidence value: {} (must be in [0, 1])", c)
            }
            Self::IntervalExceedsBounds { interval, bounds } => {
                write!(
                    f,
                    "Epistemic interval {} exceeds refinement bounds {}",
                    interval, bounds
                )
            }
        }
    }
}

impl std::error::Error for RefinedCreationError {}

impl fmt::Display for EpistemicRefinedValue<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.4} @ {:.2}% in {} (effective: {})",
            self.nominal,
            self.confidence.value() * 100.0,
            self.refinement_bounds,
            self.effective_interval()
        )
    }
}

/// Convert from regular EpistemicValue to EpistemicRefinedValue
impl EpistemicRefinedValue<f64> {
    /// Create from an EpistemicValue with explicit refinement bounds
    pub fn from_epistemic(
        ev: EpistemicValue<f64>,
        bounds: RefinementBounds,
    ) -> Result<Self, RefinedCreationError> {
        let value = *ev.value();
        if !bounds.contains(value) {
            return Err(RefinedCreationError::ValueViolatesRefinement { value, bounds });
        }

        let config = EpistemicRefinedConfig::default();
        let epistemic_interval = EpistemicInterval::from_confidence(
            value,
            ev.confidence().value(),
            config.base_uncertainty,
        )
        .clamp_to_bounds(&bounds);

        Ok(Self {
            nominal: value,
            confidence: ev.confidence(),
            refinement_bounds: bounds,
            epistemic_interval,
            ontology: ev.ontology().clone(),
            provenance: ev.provenance().clone(),
            refinement_predicate: String::new(),
        })
    }

    /// Convert back to a regular EpistemicValue (loses refinement information)
    pub fn to_epistemic(self) -> EpistemicValue<f64> {
        EpistemicValue::new(
            self.nominal,
            self.confidence,
            self.ontology,
            self.provenance,
        )
    }
}

/// Type alias for common refined epistemic types
pub type PositiveEpistemic = EpistemicRefinedValue<f64>;
pub type BoundedEpistemic = EpistemicRefinedValue<f64>;
pub type ProbabilityEpistemic = EpistemicRefinedValue<f64>;

/// Helper functions for creating common refined epistemic values
pub mod prelude {
    use super::*;

    /// Create a positive epistemic value (x > 0)
    pub fn positive(
        value: f64,
        confidence: f64,
    ) -> Result<PositiveEpistemic, RefinedCreationError> {
        EpistemicRefinedValue::new(value, confidence, RefinementBounds::positive(), "x > 0")
    }

    /// Create a non-negative epistemic value (x >= 0)
    pub fn non_negative(
        value: f64,
        confidence: f64,
    ) -> Result<PositiveEpistemic, RefinedCreationError> {
        EpistemicRefinedValue::new(
            value,
            confidence,
            RefinementBounds::non_negative(),
            "x >= 0",
        )
    }

    /// Create a probability epistemic value (0 <= x <= 1)
    pub fn probability(
        value: f64,
        confidence: f64,
    ) -> Result<ProbabilityEpistemic, RefinedCreationError> {
        EpistemicRefinedValue::new(
            value,
            confidence,
            RefinementBounds::closed(0.0, 1.0),
            "0 <= x <= 1",
        )
    }

    /// Create a bounded epistemic value (lo <= x <= hi)
    pub fn bounded(
        value: f64,
        confidence: f64,
        lo: f64,
        hi: f64,
    ) -> Result<BoundedEpistemic, RefinedCreationError> {
        EpistemicRefinedValue::new(
            value,
            confidence,
            RefinementBounds::closed(lo, hi),
            format!("{} <= x <= {}", lo, hi),
        )
    }

    /// Create a dose value (0 < x <= max_dose)
    pub fn dose(
        value: f64,
        confidence: f64,
        max_dose: f64,
    ) -> Result<BoundedEpistemic, RefinedCreationError> {
        EpistemicRefinedValue::new(
            value,
            confidence,
            RefinementBounds::half_open_left(0.0, max_dose),
            format!("0 < dose <= {}", max_dose),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use super::*;

    #[test]
    fn test_positive_creation() {
        let v = positive(5.0, 0.95).unwrap();
        assert!((v.nominal - 5.0).abs() < f64::EPSILON);
        assert!((v.confidence_value() - 0.95).abs() < 0.001);
        assert!(v.is_refinement_satisfied());
    }

    #[test]
    fn test_positive_creation_fails_for_negative() {
        let result = positive(-5.0, 0.95);
        assert!(result.is_err());
    }

    #[test]
    fn test_probability_bounds() {
        let v = probability(0.5, 0.9).unwrap();
        assert!(v.refinement_bounds.contains(0.5));
        assert!(!v.refinement_bounds.contains(-0.1));
        assert!(!v.refinement_bounds.contains(1.1));
    }

    #[test]
    fn test_interval_arithmetic_add() {
        let a = bounded(10.0, 0.9, 5.0, 15.0).unwrap();
        let b = bounded(20.0, 0.8, 15.0, 25.0).unwrap();

        let sum = a.add(b);

        // Nominal should be sum
        assert!((sum.nominal - 30.0).abs() < f64::EPSILON);

        // Bounds should be sum of bounds
        assert!(sum.refinement_bounds.lower.is_some());
        assert!((sum.refinement_bounds.lower.unwrap() - 20.0).abs() < 0.001);
        assert!((sum.refinement_bounds.upper.unwrap() - 40.0).abs() < 0.001);

        // Confidence should be min
        assert!((sum.confidence_value() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_extraction_with_threshold() {
        let v = positive(10.0, 0.8).unwrap();

        // Should extract at lower threshold
        assert!(v.extract(0.7).is_some());

        // Should not extract at higher threshold
        assert!(v.extract(0.9).is_none());
    }

    #[test]
    fn test_epistemic_interval_widening() {
        let high_conf = positive(10.0, 0.99).unwrap();
        let low_conf = positive(10.0, 0.5).unwrap();

        // Lower confidence should give wider interval
        let high_eff = high_conf.effective_interval();
        let low_eff = low_conf.effective_interval();

        assert!(low_eff.half_width > high_eff.half_width);
    }

    #[test]
    fn test_refinement_bounds_contain() {
        let bounds = RefinementBounds::closed(0.0, 100.0);

        assert!(bounds.contains(0.0));
        assert!(bounds.contains(50.0));
        assert!(bounds.contains(100.0));
        assert!(!bounds.contains(-0.1));
        assert!(!bounds.contains(100.1));
    }

    #[test]
    fn test_interval_clamping() {
        let bounds = RefinementBounds::closed(0.0, 10.0);
        let wide_interval = EpistemicInterval::from_bounds(-5.0, 15.0);

        let clamped = wide_interval.clamp_to_bounds(&bounds);

        assert!((clamped.lower() - 0.0).abs() < 0.001);
        assert!((clamped.upper() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_from_epistemic() {
        let ev = EpistemicValue::with_confidence(50.0, 0.85);
        let refined =
            EpistemicRefinedValue::from_epistemic(ev, RefinementBounds::closed(0.0, 100.0))
                .unwrap();

        assert!((refined.nominal - 50.0).abs() < f64::EPSILON);
        assert!((refined.confidence_value() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_display() {
        let v = bounded(50.0, 0.9, 0.0, 100.0).unwrap();
        let display = format!("{}", v);

        assert!(display.contains("50.0"));
        assert!(display.contains("90.00%"));
    }
}

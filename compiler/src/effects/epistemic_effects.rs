//! Epistemic effect tracking - confidence propagation through effects
//!
//! This module provides infrastructure for tracking how effect operations
//! impact epistemic confidence. When an effect operation occurs, it may
//! degrade confidence based on the nature of the operation.
//!
//! # Design
//!
//! Effect operations can impact confidence in several ways:
//! - IO operations introduce uncertainty from external world
//! - Probabilistic operations degrade confidence through sampling
//! - GPU operations may have numerical precision impact
//! - Pure state operations (Mut) don't affect confidence
//!
//! # Example
//!
//! ```rust,ignore
//! // Effect operations can degrade confidence
//! effect MeasureEffect {
//!     fn read_sensor() -> Knowledge[f64, epsilon *= 0.95]  // 5% confidence loss
//! }
//! ```

use crate::epistemic::{Provenance, Transformation, TransformationKind};
use std::collections::HashMap;

/// How an effect operation impacts confidence
///
/// Modifiers can scale confidence, clamp it to a range, and add provenance tags.
#[derive(Debug, Clone)]
pub struct ConfidenceModifier {
    /// Multiply confidence by this factor (1.0 = no change)
    pub factor: f64,
    /// Minimum confidence floor (0.0 = no floor)
    pub min_confidence: f64,
    /// Maximum confidence ceiling (1.0 = no ceiling)
    pub max_confidence: f64,
    /// Tag to add to provenance chain
    pub provenance_tag: Option<String>,
}

impl ConfidenceModifier {
    /// Identity modifier - no change to confidence
    pub fn identity() -> Self {
        Self {
            factor: 1.0,
            min_confidence: 0.0,
            max_confidence: 1.0,
            provenance_tag: None,
        }
    }

    /// Create a modifier that scales confidence by a factor
    pub fn with_factor(factor: f64) -> Self {
        Self {
            factor,
            ..Self::identity()
        }
    }

    /// Create a modifier with a provenance tag
    pub fn with_provenance(tag: impl Into<String>) -> Self {
        Self {
            provenance_tag: Some(tag.into()),
            ..Self::identity()
        }
    }

    /// Create a modifier with both factor and provenance
    pub fn with_factor_and_provenance(factor: f64, tag: impl Into<String>) -> Self {
        Self {
            factor,
            provenance_tag: Some(tag.into()),
            ..Self::identity()
        }
    }

    /// Set minimum confidence floor
    pub fn with_min(mut self, min: f64) -> Self {
        self.min_confidence = min;
        self
    }

    /// Set maximum confidence ceiling
    pub fn with_max(mut self, max: f64) -> Self {
        self.max_confidence = max;
        self
    }

    /// Apply this modifier to a confidence value
    pub fn apply(&self, confidence: f64) -> f64 {
        (confidence * self.factor)
            .max(self.min_confidence)
            .min(self.max_confidence)
    }

    /// Compose two modifiers (self then other)
    ///
    /// The resulting modifier applies self first, then other.
    /// Factors are multiplied, bounds are intersected.
    pub fn compose(&self, other: &ConfidenceModifier) -> ConfidenceModifier {
        ConfidenceModifier {
            factor: self.factor * other.factor,
            min_confidence: self.min_confidence.max(other.min_confidence),
            max_confidence: self.max_confidence.min(other.max_confidence),
            provenance_tag: match (&self.provenance_tag, &other.provenance_tag) {
                (Some(a), Some(b)) => Some(format!("{} -> {}", a, b)),
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                (None, None) => None,
            },
        }
    }

    /// Convert to a Transformation for provenance tracking
    pub fn to_transformation(&self, effect: &str, operation: &str) -> Transformation {
        let name = self
            .provenance_tag
            .clone()
            .unwrap_or_else(|| format!("{}::{}", effect, operation));
        Transformation::new(&name, TransformationKind::ExternalCall)
            .with_confidence(self.factor)
    }
}

impl Default for ConfidenceModifier {
    fn default() -> Self {
        Self::identity()
    }
}

/// Epistemic impact registry - maps effect operations to their confidence impact
///
/// This registry contains default impacts for built-in effects and allows
/// custom effects to register their epistemic impact.
#[derive(Debug)]
pub struct EpistemicImpactRegistry {
    /// (effect_name, operation_name) -> ConfidenceModifier
    impacts: HashMap<(String, String), ConfidenceModifier>,
}

impl Default for EpistemicImpactRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicImpactRegistry {
    /// Create a new registry with default impacts for built-in effects
    pub fn new() -> Self {
        let mut registry = Self {
            impacts: HashMap::new(),
        };

        // Register default impacts for built-in effects

        // IO operations degrade confidence slightly (external world is uncertain)
        registry.register(
            "IO",
            "read_file",
            ConfidenceModifier::with_factor_and_provenance(0.95, "file_read"),
        );
        registry.register(
            "IO",
            "read_line",
            ConfidenceModifier::with_factor_and_provenance(0.95, "stdin_read"),
        );
        registry.register(
            "IO",
            "write_file",
            ConfidenceModifier::identity(), // Writing doesn't degrade confidence
        );
        registry.register("IO", "print", ConfidenceModifier::identity());

        // Prob operations have variable impact (depends on samples)
        registry.register(
            "Prob",
            "sample",
            ConfidenceModifier {
                factor: 0.9, // Base degradation for sampling
                min_confidence: 0.0,
                max_confidence: 1.0,
                provenance_tag: Some("sampled".into()),
            },
        );
        registry.register(
            "Prob",
            "observe",
            ConfidenceModifier::with_factor_and_provenance(0.95, "observed"),
        );
        registry.register(
            "Prob",
            "factor",
            ConfidenceModifier::with_provenance("likelihood_weighted"),
        );

        // GPU operations may have numerical precision impact
        registry.register(
            "GPU",
            "launch",
            ConfidenceModifier {
                factor: 0.99, // Small numerical precision loss
                min_confidence: 0.0,
                max_confidence: 1.0,
                provenance_tag: Some("gpu_computed".into()),
            },
        );
        registry.register(
            "GPU",
            "sync",
            ConfidenceModifier::identity(), // Sync doesn't affect values
        );

        // Mut (mutable state) doesn't affect confidence
        registry.register("Mut", "get", ConfidenceModifier::identity());
        registry.register("Mut", "set", ConfidenceModifier::identity());

        // Async operations don't inherently degrade confidence
        registry.register("Async", "spawn", ConfidenceModifier::identity());
        registry.register("Async", "await", ConfidenceModifier::identity());

        // Panic operations are exceptional - confidence preserved until panic
        registry.register("Panic", "throw", ConfidenceModifier::identity());
        registry.register("Panic", "catch", ConfidenceModifier::identity());

        // Div (divergence) - no confidence impact
        registry.register("Div", "loop_forever", ConfidenceModifier::identity());

        // Alloc (allocation) - no confidence impact
        registry.register("Alloc", "alloc", ConfidenceModifier::identity());
        registry.register("Alloc", "dealloc", ConfidenceModifier::identity());

        // Network operations have higher uncertainty
        registry.register(
            "Network",
            "fetch",
            ConfidenceModifier {
                factor: 0.90, // Network data is less certain
                min_confidence: 0.0,
                max_confidence: 1.0,
                provenance_tag: Some("network_fetched".into()),
            },
        );
        registry.register(
            "Network",
            "send",
            ConfidenceModifier::identity(), // Sending doesn't degrade
        );

        // Sensor/measurement effects have configurable impact
        registry.register(
            "Sensor",
            "read",
            ConfidenceModifier {
                factor: 0.95,
                min_confidence: 0.0,
                max_confidence: 1.0,
                provenance_tag: Some("sensor_reading".into()),
            },
        );

        // Epistemic effect (meta-effect for epistemic operations)
        registry.register(
            "Epistemic",
            "degrade",
            ConfidenceModifier::with_provenance("explicitly_degraded"),
        );
        registry.register("Epistemic", "assert", ConfidenceModifier::identity());
        registry.register("Epistemic", "revise", ConfidenceModifier::with_provenance("revised"));

        registry
    }

    /// Create an empty registry without defaults
    pub fn empty() -> Self {
        Self {
            impacts: HashMap::new(),
        }
    }

    /// Register an impact for an effect operation
    pub fn register(&mut self, effect: &str, operation: &str, modifier: ConfidenceModifier) {
        self.impacts
            .insert((effect.to_string(), operation.to_string()), modifier);
    }

    /// Get the impact of an effect operation
    ///
    /// Returns identity modifier if the operation is not registered.
    pub fn get_impact(&self, effect: &str, operation: &str) -> ConfidenceModifier {
        self.impacts
            .get(&(effect.to_string(), operation.to_string()))
            .cloned()
            .unwrap_or_else(ConfidenceModifier::identity)
    }

    /// Check if an operation is registered
    pub fn has_impact(&self, effect: &str, operation: &str) -> bool {
        self.impacts
            .contains_key(&(effect.to_string(), operation.to_string()))
    }

    /// Get all registered effects
    pub fn effects(&self) -> impl Iterator<Item = &str> {
        self.impacts.keys().map(|(e, _)| e.as_str())
    }

    /// Get all operations for an effect
    pub fn operations(&self, effect: &str) -> impl Iterator<Item = &str> {
        self.impacts
            .keys()
            .filter(move |(e, _)| e == effect)
            .map(|(_, o)| o.as_str())
    }
}

/// Tracks cumulative epistemic impact through a computation
///
/// This tracker maintains a running tally of confidence degradation
/// as effect operations occur, along with a history for debugging.
#[derive(Debug)]
pub struct EpistemicTracker {
    /// Current cumulative modifier
    pub cumulative: ConfidenceModifier,
    /// History of operations that affected confidence
    pub history: Vec<EpistemicEvent>,
    /// Whether to record history (can be disabled for performance)
    record_history: bool,
}

/// A single epistemic event in the tracking history
#[derive(Debug, Clone)]
pub struct EpistemicEvent {
    /// Effect name
    pub effect: String,
    /// Operation name
    pub operation: String,
    /// Modifier applied
    pub modifier: ConfidenceModifier,
    /// Timestamp (for debugging and ordering)
    pub timestamp: std::time::Instant,
    /// Confidence before this event
    pub confidence_before: f64,
    /// Confidence after this event
    pub confidence_after: f64,
}

impl EpistemicTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self {
            cumulative: ConfidenceModifier::identity(),
            history: Vec::new(),
            record_history: true,
        }
    }

    /// Create a tracker without history recording (for performance)
    pub fn without_history() -> Self {
        Self {
            cumulative: ConfidenceModifier::identity(),
            history: Vec::new(),
            record_history: false,
        }
    }

    /// Record an effect operation's epistemic impact
    pub fn record(&mut self, effect: &str, operation: &str, modifier: ConfidenceModifier) {
        let confidence_before = self.cumulative.factor;
        self.cumulative = self.cumulative.compose(&modifier);
        let confidence_after = self.cumulative.factor;

        if self.record_history {
            self.history.push(EpistemicEvent {
                effect: effect.to_string(),
                operation: operation.to_string(),
                modifier,
                timestamp: std::time::Instant::now(),
                confidence_before,
                confidence_after,
            });
        }
    }

    /// Record using the registry to look up the modifier
    pub fn record_from_registry(
        &mut self,
        effect: &str,
        operation: &str,
        registry: &EpistemicImpactRegistry,
    ) {
        let modifier = registry.get_impact(effect, operation);
        self.record(effect, operation, modifier);
    }

    /// Get the final confidence after all operations
    pub fn final_confidence(&self, initial: f64) -> f64 {
        self.cumulative.apply(initial)
    }

    /// Generate provenance chain from history
    pub fn provenance_chain(&self) -> Vec<String> {
        self.history
            .iter()
            .filter_map(|e| e.modifier.provenance_tag.clone())
            .collect()
    }

    /// Generate transformations for provenance tracking
    pub fn to_transformations(&self) -> Vec<Transformation> {
        self.history
            .iter()
            .map(|e| e.modifier.to_transformation(&e.effect, &e.operation))
            .collect()
    }

    /// Extend provenance with tracked operations
    pub fn extend_provenance(&self, provenance: &Provenance) -> Provenance {
        let mut result = provenance.clone();
        for transformation in self.to_transformations() {
            result = result.extend(transformation);
        }
        result
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.cumulative = ConfidenceModifier::identity();
        self.history.clear();
    }

    /// Get the number of recorded events
    pub fn event_count(&self) -> usize {
        self.history.len()
    }

    /// Check if any operations degraded confidence
    pub fn has_degradation(&self) -> bool {
        self.cumulative.factor < 1.0
    }

    /// Get total confidence factor (product of all modifiers)
    pub fn total_factor(&self) -> f64 {
        self.cumulative.factor
    }
}

impl Default for EpistemicTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration helper: Apply epistemic tracking to a Knowledge value
///
/// This function applies the tracked confidence degradation to a Knowledge
/// value and extends its provenance chain.
pub fn apply_epistemic_tracking(
    knowledge: &crate::epistemic::Knowledge,
    tracker: &EpistemicTracker,
) -> crate::epistemic::Knowledge {
    use crate::epistemic::{Confidence, EpistemicStatus};

    // Calculate new confidence
    let original_confidence = knowledge.epistemic.confidence.value();
    let new_confidence = tracker.final_confidence(original_confidence);

    // Create new epistemic status with updated confidence
    let new_epistemic = EpistemicStatus {
        confidence: Confidence::new(new_confidence),
        revisability: knowledge.epistemic.revisability.clone(),
        source: knowledge.epistemic.source.clone(),
        evidence: knowledge.epistemic.evidence.clone(),
    };

    // Extend provenance with tracked operations
    let new_provenance = tracker.extend_provenance(&knowledge.provenance);

    crate::epistemic::Knowledge {
        content: knowledge.content.clone(),
        temporal: knowledge.temporal.clone(),
        epistemic: new_epistemic,
        domain: knowledge.domain.clone(),
        provenance: new_provenance,
        span: knowledge.span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_modifier_apply() {
        let m = ConfidenceModifier::with_factor(0.9);
        assert!((m.apply(1.0) - 0.9).abs() < 0.001);
        assert!((m.apply(0.5) - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_modifier_with_bounds() {
        let m = ConfidenceModifier::with_factor(0.5).with_min(0.3).with_max(0.8);
        assert!((m.apply(1.0) - 0.5).abs() < 0.001);
        assert!((m.apply(0.4) - 0.3).abs() < 0.001); // Clamped to min
    }

    #[test]
    fn test_modifier_composition() {
        let m1 = ConfidenceModifier::with_factor(0.9);
        let m2 = ConfidenceModifier::with_factor(0.8);
        let composed = m1.compose(&m2);
        assert!((composed.factor - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_modifier_composition_with_provenance() {
        let m1 = ConfidenceModifier::with_factor_and_provenance(0.9, "io");
        let m2 = ConfidenceModifier::with_factor_and_provenance(0.8, "sample");
        let composed = m1.compose(&m2);
        assert_eq!(composed.provenance_tag, Some("io -> sample".to_string()));
    }

    #[test]
    fn test_registry_defaults() {
        let registry = EpistemicImpactRegistry::new();

        // IO operations should have 0.95 factor
        let io_impact = registry.get_impact("IO", "read_file");
        assert!((io_impact.factor - 0.95).abs() < 0.001);

        // Prob sample should have 0.9 factor
        let prob_impact = registry.get_impact("Prob", "sample");
        assert!((prob_impact.factor - 0.9).abs() < 0.001);

        // Mut get should be identity
        let mut_impact = registry.get_impact("Mut", "get");
        assert!((mut_impact.factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_registry_custom_impact() {
        let mut registry = EpistemicImpactRegistry::new();
        registry.register(
            "CustomEffect",
            "custom_op",
            ConfidenceModifier::with_factor(0.75),
        );

        let impact = registry.get_impact("CustomEffect", "custom_op");
        assert!((impact.factor - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_registry_unknown_returns_identity() {
        let registry = EpistemicImpactRegistry::new();
        let impact = registry.get_impact("UnknownEffect", "unknown_op");
        assert!((impact.factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_tracker() {
        let registry = EpistemicImpactRegistry::new();
        let mut tracker = EpistemicTracker::new();

        tracker.record_from_registry("IO", "read_file", &registry);
        tracker.record_from_registry("Prob", "sample", &registry);

        let final_conf = tracker.final_confidence(1.0);
        // 0.95 * 0.9 = 0.855
        assert!((final_conf - 0.855).abs() < 0.001);
    }

    #[test]
    fn test_tracker_provenance_chain() {
        let registry = EpistemicImpactRegistry::new();
        let mut tracker = EpistemicTracker::new();

        tracker.record_from_registry("IO", "read_file", &registry);
        tracker.record_from_registry("Prob", "sample", &registry);

        let chain = tracker.provenance_chain();
        assert_eq!(chain.len(), 2);
        assert!(chain.contains(&"file_read".to_string()));
        assert!(chain.contains(&"sampled".to_string()));
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = EpistemicTracker::new();
        tracker.record("Test", "op", ConfidenceModifier::with_factor(0.5));

        assert!(tracker.has_degradation());
        tracker.reset();
        assert!(!tracker.has_degradation());
        assert_eq!(tracker.event_count(), 0);
    }

    #[test]
    fn test_tracker_without_history() {
        let mut tracker = EpistemicTracker::without_history();
        tracker.record("Test", "op", ConfidenceModifier::with_factor(0.5));

        assert_eq!(tracker.event_count(), 0); // No history recorded
        assert!(tracker.has_degradation()); // But degradation is tracked
    }

    #[test]
    fn test_identity_modifier() {
        let m = ConfidenceModifier::identity();
        assert!((m.apply(0.5) - 0.5).abs() < 0.001);
        assert!(m.provenance_tag.is_none());
    }

    #[test]
    fn test_modifier_to_transformation() {
        let m = ConfidenceModifier::with_factor_and_provenance(0.9, "test_op");
        let t = m.to_transformation("TestEffect", "do_something");
        assert_eq!(t.name, "test_op");
        assert!((t.confidence_factor - 0.9).abs() < 0.001);
    }
}

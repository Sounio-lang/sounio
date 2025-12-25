//! Temporal Knowledge Type
//!
//! Extends the epistemic Knowledge type with temporal dimension.
//!
//! # Type Definition
//!
//! ```text
//! Day 32: Knowledge[τ, ε, δ, Φ]
//! Day 33: TemporalKnowledge[τ, ε, δ, Φ, t]
//!
//! where t : Temporal is the temporal dimension
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! // Create knowledge with exponential decay
//! let clearance = TemporalKnowledge::decaying(
//!     EpistemicValue::with_confidence(5.2, 0.95),
//!     0.3,
//!     TimeUnit::Years,
//! );
//!
//! // Get confidence at current time
//! let now_confidence = clearance.now().core.confidence().value();
//! ```

use super::decay::{DecayFunction, TimeUnit};
use super::types::Temporal;
use crate::epistemic::composition::{ConfidenceValue, EpistemicValue, ProvenanceNode, SourceInfo};
use chrono::{DateTime, Duration, Utc};

/// Temporal Knowledge type - extends EpistemicValue with time dimension
#[derive(Clone, Debug)]
pub struct TemporalKnowledge<T> {
    /// Core epistemic value (from Day 32)
    pub core: EpistemicValue<T>,

    /// Temporal dimension
    pub temporal: Temporal,

    /// History of updates (for HISTORICALLY operator)
    pub history: Option<Vec<TemporalKnowledge<T>>>,
}

impl<T: Clone> TemporalKnowledge<T> {
    /// Create with explicit temporal
    pub fn new(core: EpistemicValue<T>, temporal: Temporal) -> Self {
        TemporalKnowledge {
            core,
            temporal,
            history: None,
        }
    }

    /// Create with history tracking
    pub fn with_history(core: EpistemicValue<T>, temporal: Temporal) -> Self {
        TemporalKnowledge {
            core,
            temporal,
            history: Some(Vec::new()),
        }
    }

    /// Create timeless knowledge (mathematical/physical constants)
    pub fn timeless(core: EpistemicValue<T>) -> Self {
        TemporalKnowledge {
            core,
            temporal: Temporal::Timeless,
            history: None,
        }
    }

    /// Create knowledge at current instant
    pub fn instant(core: EpistemicValue<T>) -> Self {
        TemporalKnowledge {
            core,
            temporal: Temporal::Instant(Utc::now()),
            history: None,
        }
    }

    /// Create with exponential decay
    pub fn decaying(core: EpistemicValue<T>, lambda: f64, time_unit: TimeUnit) -> Self {
        TemporalKnowledge {
            core,
            temporal: Temporal::Decaying {
                created: Utc::now(),
                decay_fn: DecayFunction::Exponential { lambda, time_unit },
            },
            history: None,
        }
    }

    /// Create with half-life based decay
    pub fn with_half_life(core: EpistemicValue<T>, half_life: Duration) -> Self {
        let decay_fn = DecayFunction::from_half_life(half_life, TimeUnit::Days);
        TemporalKnowledge {
            core,
            temporal: Temporal::Decaying {
                created: Utc::now(),
                decay_fn,
            },
            history: None,
        }
    }

    /// Create with fixed validity period
    pub fn valid_for(core: EpistemicValue<T>, duration: Duration) -> Self {
        let now = Utc::now();
        TemporalKnowledge {
            core,
            temporal: Temporal::Interval {
                start: now,
                end: now + duration,
            },
            history: None,
        }
    }

    /// Create with custom decay function
    pub fn with_decay(core: EpistemicValue<T>, decay_fn: DecayFunction) -> Self {
        TemporalKnowledge {
            core,
            temporal: Temporal::Decaying {
                created: Utc::now(),
                decay_fn,
            },
            history: None,
        }
    }

    /// Get the value
    pub fn value(&self) -> &T {
        self.core.value()
    }

    /// Get current confidence (with decay applied)
    pub fn current_confidence(&self) -> ConfidenceValue {
        let decay_factor = self.compute_decay_at(Utc::now());
        let decayed = self.core.confidence().value() * decay_factor;
        ConfidenceValue::new(decayed.clamp(0.0, 1.0)).unwrap_or_else(|_| ConfidenceValue::zero())
    }

    /// Get original (undecayed) confidence
    pub fn original_confidence(&self) -> ConfidenceValue {
        self.core.confidence()
    }

    /// Compute decay factor at a given instant
    pub fn compute_decay_at(&self, instant: DateTime<Utc>) -> f64 {
        match &self.temporal {
            Temporal::Timeless => 1.0,

            Temporal::Instant(created) => {
                let delta = instant.signed_duration_since(*created);
                if delta.num_seconds() < 0 {
                    1.0
                } else {
                    // Default: 10% decay per year
                    let years = delta.num_seconds() as f64 / TimeUnit::Years.to_seconds();
                    (-0.1 * years).exp()
                }
            }

            Temporal::Interval { start, end } => {
                if instant >= *start && instant <= *end {
                    1.0
                } else {
                    0.0
                }
            }

            Temporal::Decaying { created, decay_fn } => {
                let delta = instant.signed_duration_since(*created);
                if delta.num_seconds() < 0 {
                    1.0
                } else {
                    decay_fn.evaluate(delta)
                }
            }

            Temporal::Versioned {
                superseded_by,
                created,
                ..
            } => {
                if superseded_by.is_some() {
                    0.0
                } else {
                    let delta = instant.signed_duration_since(*created);
                    (-0.05 * delta.num_seconds() as f64 / TimeUnit::Years.to_seconds()).exp()
                }
            }
        }
    }

    /// Get the age of this knowledge
    pub fn age(&self) -> Option<Duration> {
        self.temporal
            .creation_time()
            .map(|created| Utc::now().signed_duration_since(created))
    }

    /// Check if knowledge is still valid
    pub fn is_valid(&self) -> bool {
        self.temporal.is_valid_at(Utc::now()) && self.current_confidence().value() > 0.0
    }

    /// Check if knowledge has expired (confidence effectively zero)
    pub fn is_expired(&self) -> bool {
        self.current_confidence().value() < 0.001
    }

    /// Get time until confidence drops below threshold
    pub fn time_to_threshold(&self, threshold: f64) -> Option<Duration> {
        match &self.temporal {
            Temporal::Decaying { decay_fn, .. } => {
                let current = self.core.confidence().value();
                if current <= threshold {
                    return Some(Duration::zero());
                }
                // Need to find t where current * D(t) = threshold
                // D(t) = threshold / current
                let target_decay = threshold / current;
                decay_fn.time_to_threshold(target_decay)
            }
            Temporal::Interval { end, .. } => {
                let now = Utc::now();
                if now >= *end {
                    Some(Duration::zero())
                } else {
                    Some(end.signed_duration_since(now))
                }
            }
            _ => None,
        }
    }

    /// Get suggested revalidation interval
    pub fn suggested_revalidation_interval(&self) -> Duration {
        match &self.temporal {
            Temporal::Decaying { decay_fn, .. } => {
                decay_fn.half_life().unwrap_or(Duration::days(365))
            }
            Temporal::Interval { start, end } => {
                let validity = end.signed_duration_since(*start);
                Duration::seconds((validity.num_seconds() as f64 * 0.8) as i64)
            }
            _ => Duration::days(365),
        }
    }

    /// Update the core value while preserving temporal metadata
    pub fn update_value(&self, new_value: T) -> Self {
        let mut updated = TemporalKnowledge {
            core: EpistemicValue::new(
                new_value,
                self.core.confidence(),
                self.core.ontology().clone(),
                ProvenanceNode::derived("temporal_update", vec![self.core.provenance().clone()]),
            ),
            temporal: self.temporal.clone(),
            history: self.history.clone(),
        };

        // Add current state to history
        if let Some(ref mut history) = updated.history {
            history.push(self.clone());
        }

        updated
    }

    /// Revalidate knowledge with new confidence
    pub fn revalidate(&self, new_confidence: f64) -> Self {
        let mut updated = TemporalKnowledge {
            core: EpistemicValue::new(
                self.core.value().clone(),
                ConfidenceValue::new(new_confidence.clamp(0.0, 1.0))
                    .unwrap_or_else(|_| ConfidenceValue::zero()),
                self.core.ontology().clone(),
                ProvenanceNode::derived_with_metadata(
                    "revalidation",
                    vec![self.core.provenance().clone()],
                    format!("revalidated_at:{}", Utc::now()),
                ),
            ),
            temporal: match &self.temporal {
                Temporal::Decaying { decay_fn, .. } => Temporal::Decaying {
                    created: Utc::now(),
                    decay_fn: decay_fn.clone(),
                },
                other => other.clone(),
            },
            history: self.history.clone(),
        };

        if let Some(ref mut history) = updated.history {
            history.push(self.clone());
        }

        updated
    }

    /// Add ontology reference
    pub fn with_ontology(mut self, ontology: impl Into<String>) -> Self {
        let onto_ref =
            crate::epistemic::composition::knowledge::OntologyRef::new(ontology.into(), "");
        let mut ontology_set = self.core.ontology().clone();
        ontology_set.insert(onto_ref);

        self.core = EpistemicValue::new(
            self.core.value().clone(),
            self.core.confidence(),
            ontology_set,
            self.core.provenance().clone(),
        );

        self
    }
}

/// Convenience constructors for common knowledge types
impl<T: Clone> TemporalKnowledge<T> {
    /// Create from literature source with standard decay
    pub fn from_literature(
        value: T,
        confidence: f64,
        doi: &str,
        publication_date: DateTime<Utc>,
    ) -> Self {
        let core = EpistemicValue::from_source(value, confidence, SourceInfo::from_doi(doi));

        TemporalKnowledge {
            core,
            temporal: Temporal::Decaying {
                created: publication_date,
                decay_fn: super::decay::presets::scientific_literature(),
            },
            history: None,
        }
    }

    /// Create from measurement with appropriate decay
    pub fn from_measurement(
        value: T,
        confidence: f64,
        measurement_id: &str,
        measurement_time: DateTime<Utc>,
        decay_fn: DecayFunction,
    ) -> Self {
        let core = EpistemicValue::from_source(
            value,
            confidence,
            SourceInfo::from_measurement(measurement_id),
        );

        TemporalKnowledge {
            core,
            temporal: Temporal::Decaying {
                created: measurement_time,
                decay_fn,
            },
            history: None,
        }
    }

    /// Create LLM-generated knowledge with standard decay
    pub fn from_llm(value: T, confidence: f64, model: &str, prompt_hash: u64) -> Self {
        let core = EpistemicValue::from_source(
            value,
            confidence,
            SourceInfo::from_llm(model, prompt_hash, confidence),
        );

        TemporalKnowledge {
            core,
            temporal: Temporal::Decaying {
                created: Utc::now(),
                decay_fn: super::decay::presets::llm_generated(),
            },
            history: None,
        }
    }

    /// Create physical constant (timeless)
    pub fn physical_constant(value: T, source: &str) -> Self {
        let core = EpistemicValue::from_source(value, 1.0, SourceInfo::from_doi(source));

        TemporalKnowledge {
            core,
            temporal: Temporal::Timeless,
            history: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeless_knowledge() {
        let k: TemporalKnowledge<f64> =
            TemporalKnowledge::timeless(EpistemicValue::with_confidence(299792458.0, 1.0));

        assert!(k.is_valid());
        assert!(!k.is_expired());
        assert!((k.current_confidence().value() - 1.0).abs() < 1e-10);

        // Timeless never decays
        assert!((k.compute_decay_at(Utc::now() + Duration::days(36500)) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decaying_knowledge() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.2, 0.95),
            1.0,
            TimeUnit::Years,
        );

        // Initially high confidence
        assert!(k.current_confidence().value() > 0.9);

        // After 1 year, confidence should be 0.95 * e^(-1) ≈ 0.35
        let one_year_later = Utc::now() + Duration::days(365);
        let decay = k.compute_decay_at(one_year_later);
        assert!((decay - (-1.0_f64).exp()).abs() < 0.01);
    }

    #[test]
    fn test_valid_for() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::valid_for(
            EpistemicValue::with_confidence(100.0, 0.90),
            Duration::days(30),
        );

        assert!(k.is_valid());

        // After validity period, should be invalid
        let after_expiry = Utc::now() + Duration::days(31);
        assert!((k.compute_decay_at(after_expiry) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_with_half_life() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::with_half_life(
            EpistemicValue::with_confidence(10.0, 1.0),
            Duration::days(30),
        );

        // At half-life, decay should be 0.5
        let at_half_life = Utc::now() + Duration::days(30);
        let decay = k.compute_decay_at(at_half_life);
        assert!((decay - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_update_value() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::with_history(
            EpistemicValue::with_confidence(5.0, 0.9),
            Temporal::now(),
        );

        let updated = k.update_value(6.0);

        assert_eq!(*updated.value(), 6.0);
        assert!(updated.history.is_some());
        assert_eq!(updated.history.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_revalidate() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.0, 0.5),
            1.0,
            TimeUnit::Years,
        );

        let revalidated = k.revalidate(0.95);

        // Confidence should be updated
        assert!((revalidated.original_confidence().value() - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_time_to_threshold() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(10.0, 1.0),
            1.0,
            TimeUnit::Years,
        );

        // Time to 50% (half-life)
        let to_half = k.time_to_threshold(0.5).unwrap();
        let to_half_years = to_half.num_days() as f64 / 365.0;
        assert!((to_half_years - 0.693).abs() < 0.1);
    }

    #[test]
    fn test_physical_constant() {
        let c: TemporalKnowledge<f64> =
            TemporalKnowledge::physical_constant(299792458.0, "CODATA 2018");

        assert!(matches!(c.temporal, Temporal::Timeless));
        assert!((c.current_confidence().value() - 1.0).abs() < 1e-10);
    }
}

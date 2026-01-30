//! Temporal Operators (LTL + S4)
//!
//! Implements temporal logic operators for epistemic knowledge:
//!
//! # Point Operators
//! - `at(t)`: Get knowledge at specific instant
//! - `now()`: Get knowledge at current time
//!
//! # Past Operators (LTL)
//! - `historically()`: Was K always true in the past?
//! - `since(E)`: Has K been true since event E?
//!
//! # Future Operators (LTL)
//! - `eventually(horizon)`: Project K into the future
//! - `always(deadline)`: Assert K remains true
//! - `until(C)`: K remains true until condition C
//!
//! # Semantics
//!
//! ```text
//! (𝔽, w, t) ⊨ φ@t'  ⟺  (𝔽, w, t') ⊨ φ
//! (𝔽, w, t) ⊨ Hφ    ⟺  ∀t' < t. (𝔽, w, t') ⊨ φ
//! (𝔽, w, t) ⊨ Gφ    ⟺  ∀t' ≥ t. (𝔽, w, t') ⊨ φ
//! (𝔽, w, t) ⊨ Fφ    ⟺  ∃t' ≥ t. (𝔽, w, t') ⊨ φ
//! ```

use super::knowledge::TemporalKnowledge;
use super::types::Temporal;
use crate::epistemic::composition::{ConfidenceValue, EpistemicValue, ProvenanceNode};
use chrono::{DateTime, Duration, Utc};

// ═══════════════════════════════════════════════════════════════
// POINT OPERATORS
// ═══════════════════════════════════════════════════════════════

impl<T: Clone> TemporalKnowledge<T> {
    /// AT: Get knowledge at specific instant
    ///
    /// K@t : Knowledge[τ, ε×D(t-t₀), δ, Φ, Instant(t)]
    pub fn at(&self, instant: DateTime<Utc>) -> TemporalKnowledge<T> {
        let decay_factor = self.compute_decay_at(instant);
        let decayed_confidence = self.core.confidence().value() * decay_factor;

        TemporalKnowledge {
            core: EpistemicValue::new(
                self.core.value().clone(),
                ConfidenceValue::new(decayed_confidence.clamp(0.0, 1.0))
                    .unwrap_or_else(|_| ConfidenceValue::zero()),
                self.core.ontology().clone(),
                ProvenanceNode::derived_with_metadata(
                    "temporal_at",
                    vec![self.core.provenance().clone()],
                    format!("evaluated_at:{}", instant),
                ),
            ),
            temporal: Temporal::Instant(instant),
            history: None,
        }
    }

    /// NOW: Convenience for at(Utc::now())
    pub fn now(&self) -> TemporalKnowledge<T> {
        self.at(Utc::now())
    }

    /// PAST: Get knowledge as it was at a past instant
    pub fn at_past(&self, duration_ago: Duration) -> TemporalKnowledge<T> {
        let past_instant = Utc::now() - duration_ago;
        self.at(past_instant)
    }

    /// FUTURE: Get projected knowledge at a future instant
    pub fn at_future(&self, duration_ahead: Duration) -> TemporalKnowledge<T> {
        let future_instant = Utc::now() + duration_ahead;
        self.at(future_instant)
    }
}

// ═══════════════════════════════════════════════════════════════
// PAST OPERATORS
// ═══════════════════════════════════════════════════════════════

/// Result of HISTORICALLY operator
#[derive(Clone, Debug)]
pub struct HistoricalAssessment<T> {
    /// Current knowledge state
    pub current: TemporalKnowledge<T>,

    /// Whether knowledge was always true historically
    pub was_always_true: bool,

    /// Minimum confidence seen in history
    pub historical_confidence: ConfidenceValue,

    /// Number of historical snapshots analyzed
    pub history_depth: usize,

    /// Oldest timestamp in history
    pub earliest_record: Option<DateTime<Utc>>,

    /// Any inconsistencies found
    pub inconsistencies: Vec<HistoricalInconsistency<T>>,
}

/// A historical inconsistency
#[derive(Clone, Debug)]
pub struct HistoricalInconsistency<T> {
    pub timestamp: DateTime<Utc>,
    pub previous_value: T,
    pub discrepancy: String,
}

impl<T: Clone + PartialEq + std::fmt::Debug> TemporalKnowledge<T> {
    /// HISTORICALLY: Was this always true in the past?
    ///
    /// H(K) = ∀t' < now. K@t' is consistent with K
    pub fn historically(&self) -> HistoricalAssessment<T> {
        match &self.history {
            Some(history) if !history.is_empty() => {
                let mut all_consistent = true;
                let mut min_confidence = self.core.confidence().value();
                let mut earliest = None;
                let mut inconsistencies = Vec::new();

                for h in history {
                    // Check consistency
                    if !self.is_consistent_with(h) {
                        all_consistent = false;
                        inconsistencies.push(HistoricalInconsistency {
                            timestamp: h.temporal.effective_instant(),
                            previous_value: h.core.value().clone(),
                            discrepancy: format!(
                                "Value changed from {:?} to {:?}",
                                h.core.value(),
                                self.core.value()
                            ),
                        });
                    }

                    // Track minimum confidence
                    let h_conf = h.core.confidence().value();
                    if h_conf < min_confidence {
                        min_confidence = h_conf;
                    }

                    // Track earliest
                    if let Some(created) = h.temporal.creation_time() {
                        earliest = Some(match earliest {
                            None => created,
                            Some(e) if created < e => created,
                            Some(e) => e,
                        });
                    }
                }

                HistoricalAssessment {
                    current: self.clone(),
                    was_always_true: all_consistent,
                    historical_confidence: ConfidenceValue::new(min_confidence)
                        .unwrap_or_else(|_| ConfidenceValue::zero()),
                    history_depth: history.len(),
                    earliest_record: earliest,
                    inconsistencies,
                }
            }
            _ => HistoricalAssessment {
                current: self.clone(),
                was_always_true: true,
                historical_confidence: self.core.confidence(),
                history_depth: 0,
                earliest_record: self.temporal.creation_time(),
                inconsistencies: Vec::new(),
            },
        }
    }

    fn is_consistent_with(&self, other: &TemporalKnowledge<T>) -> bool {
        self.core.value() == other.core.value()
    }
}

/// Temporal event for SINCE operator
#[derive(Clone, Debug)]
pub struct TemporalEvent<E> {
    /// The event data
    pub event: E,

    /// When the event occurred
    pub timestamp: DateTime<Utc>,

    /// Human-readable description
    pub description: String,
}

impl<E> TemporalEvent<E> {
    /// Create a new temporal event
    pub fn new(event: E, timestamp: DateTime<Utc>, description: impl Into<String>) -> Self {
        TemporalEvent {
            event,
            timestamp,
            description: description.into(),
        }
    }

    /// Create an event that occurred now
    pub fn now(event: E, description: impl Into<String>) -> Self {
        TemporalEvent {
            event,
            timestamp: Utc::now(),
            description: description.into(),
        }
    }

    /// Create an event that occurred in the past
    pub fn past(event: E, duration_ago: Duration, description: impl Into<String>) -> Self {
        TemporalEvent {
            event,
            timestamp: Utc::now() - duration_ago,
            description: description.into(),
        }
    }
}

/// Result of SINCE operator
#[derive(Clone, Debug)]
pub struct SinceAssessment<T, E> {
    /// The triggering event
    pub triggered_by: TemporalEvent<E>,

    /// Whether K was true at the trigger
    pub was_true_at_trigger: bool,

    /// Whether K has been maintained since
    pub maintained_since: bool,

    /// Current knowledge state
    pub current_state: TemporalKnowledge<T>,

    /// Duration since trigger
    pub duration_since: Duration,

    /// Confidence at trigger
    pub confidence_at_trigger: ConfidenceValue,
}

impl<T: Clone + PartialEq + std::fmt::Debug> TemporalKnowledge<T> {
    /// SINCE: Has K been true since event E?
    ///
    /// K Since E = K was true at E and remained true until now
    pub fn since<E: Clone>(&self, event: &TemporalEvent<E>) -> SinceAssessment<T, E> {
        let event_time = event.timestamp;
        let now = Utc::now();

        let k_at_event = self.at(event_time);
        let conf_at_event = k_at_event.core.confidence().value();

        let maintained = match &self.history {
            Some(history) => history
                .iter()
                .filter(|h| {
                    if let Some(t) = h.temporal.creation_time() {
                        t >= event_time && t <= now
                    } else {
                        false
                    }
                })
                .all(|h| self.is_consistent_with(h)),
            None => true,
        };

        SinceAssessment {
            triggered_by: event.clone(),
            was_true_at_trigger: conf_at_event > 0.5,
            maintained_since: maintained,
            current_state: self.clone(),
            duration_since: now.signed_duration_since(event_time),
            confidence_at_trigger: ConfidenceValue::new(conf_at_event)
                .unwrap_or_else(|_| ConfidenceValue::zero()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// FUTURE OPERATORS
// ═══════════════════════════════════════════════════════════════

/// Result of EVENTUALLY operator
#[derive(Clone, Debug)]
pub struct FuturePrediction<T> {
    /// Current knowledge state
    pub current: TemporalKnowledge<T>,

    /// Prediction horizon
    pub horizon: Duration,

    /// Projected confidence at horizon
    pub projected_confidence: ConfidenceValue,

    /// Uncertainty growth factor
    pub uncertainty: f64,

    /// Projected timestamp
    pub projected_at: DateTime<Utc>,
}

impl<T: Clone> TemporalKnowledge<T> {
    /// EVENTUALLY: Project that K will be true at some future time
    ///
    /// F(K, horizon) = K projected to now + horizon
    pub fn eventually(&self, horizon: Duration) -> FuturePrediction<T> {
        let now = Utc::now();
        let future_time = now + horizon;

        let projected_confidence = self.compute_decay_at(future_time);

        // Uncertainty increases with time
        let uncertainty_growth = 1.0 - (-0.1 * horizon.num_days() as f64).exp();
        let adjusted_confidence = projected_confidence * (1.0 - uncertainty_growth);

        FuturePrediction {
            current: self.clone(),
            horizon,
            projected_confidence: ConfidenceValue::new(adjusted_confidence.max(0.0))
                .unwrap_or_else(|_| ConfidenceValue::zero()),
            uncertainty: uncertainty_growth,
            projected_at: future_time,
        }
    }

    /// Project confidence at multiple future points
    pub fn project_trajectory(
        &self,
        steps: usize,
        step_size: Duration,
    ) -> Vec<FuturePrediction<T>> {
        (1..=steps)
            .map(|i| {
                let horizon = step_size * i as i32;
                self.eventually(horizon)
            })
            .collect()
    }
}

/// Result of ALWAYS operator
#[derive(Clone, Debug)]
pub struct ValidityConstraint<T> {
    /// The constrained knowledge
    pub knowledge: TemporalKnowledge<T>,

    /// Deadline for validity (None = indefinite)
    pub valid_until: Option<DateTime<Utc>>,

    /// Whether revalidation is required
    pub requires_revalidation: bool,

    /// Suggested revalidation interval
    pub revalidation_interval: Duration,

    /// Minimum confidence threshold
    pub min_confidence_threshold: f64,
}

impl<T: Clone> TemporalKnowledge<T> {
    /// ALWAYS: Assert that K should remain true
    ///
    /// G(K) = K remains valid (possibly until deadline)
    pub fn always(&self, until: Option<DateTime<Utc>>) -> ValidityConstraint<T> {
        ValidityConstraint {
            knowledge: self.clone(),
            valid_until: until,
            requires_revalidation: match &self.temporal {
                Temporal::Decaying { .. } => true,
                Temporal::Interval { .. } => true,
                Temporal::Timeless => false,
                _ => true,
            },
            revalidation_interval: self.suggested_revalidation_interval(),
            min_confidence_threshold: 0.5,
        }
    }

    /// Create validity constraint with custom threshold
    pub fn always_with_threshold(
        &self,
        until: Option<DateTime<Utc>>,
        min_confidence: f64,
    ) -> ValidityConstraint<T> {
        let mut constraint = self.always(until);
        constraint.min_confidence_threshold = min_confidence.clamp(0.0, 1.0);
        constraint
    }
}

impl<T: Clone> ValidityConstraint<T> {
    /// Check if constraint is currently satisfied
    pub fn is_satisfied(&self) -> bool {
        let current_conf = self.knowledge.current_confidence().value();
        let now = Utc::now();

        // Check deadline
        if let Some(deadline) = self.valid_until
            && now > deadline
        {
            return false;
        }

        // Check confidence threshold
        current_conf >= self.min_confidence_threshold
    }

    /// Get time until revalidation is needed
    pub fn time_until_revalidation(&self) -> Option<Duration> {
        self.knowledge
            .time_to_threshold(self.min_confidence_threshold)
    }

    /// Check if revalidation is overdue
    pub fn needs_revalidation_now(&self) -> bool {
        self.requires_revalidation
            && self.knowledge.current_confidence().value() < self.min_confidence_threshold
    }
}

/// Monitor for UNTIL operator
pub struct UntilMonitor<T: Clone, C> {
    /// The monitored knowledge
    pub knowledge: TemporalKnowledge<T>,

    /// Termination condition
    termination_condition: C,

    /// Check interval
    pub check_interval: Duration,

    /// When monitoring started
    pub started_at: DateTime<Utc>,

    /// Whether terminated
    pub terminated: bool,

    /// Last check timestamp
    pub last_checked: DateTime<Utc>,
}

impl<T: Clone, C: Fn(&T) -> bool> UntilMonitor<T, C> {
    /// Create a new until monitor
    pub fn new(knowledge: TemporalKnowledge<T>, condition: C, check_interval: Duration) -> Self {
        let now = Utc::now();
        UntilMonitor {
            knowledge,
            termination_condition: condition,
            check_interval,
            started_at: now,
            terminated: false,
            last_checked: now,
        }
    }

    /// Check the termination condition
    pub fn check(&mut self) -> UntilStatus<T> {
        if self.terminated {
            return UntilStatus::AlreadyTerminated;
        }

        let now = Utc::now();
        self.last_checked = now;

        if (self.termination_condition)(self.knowledge.core.value()) {
            self.terminated = true;
            UntilStatus::ConditionMet {
                final_state: self.knowledge.now(),
                duration: now.signed_duration_since(self.started_at),
            }
        } else {
            UntilStatus::StillWaiting {
                current_state: self.knowledge.now(),
                elapsed: now.signed_duration_since(self.started_at),
            }
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        Utc::now().signed_duration_since(self.started_at)
    }

    /// Check if it's time for next check
    pub fn should_check(&self) -> bool {
        !self.terminated
            && Utc::now().signed_duration_since(self.last_checked) >= self.check_interval
    }
}

/// Status of UNTIL monitor
#[derive(Clone, Debug)]
pub enum UntilStatus<T> {
    /// Still waiting for condition
    StillWaiting {
        current_state: TemporalKnowledge<T>,
        elapsed: Duration,
    },

    /// Condition was met
    ConditionMet {
        final_state: TemporalKnowledge<T>,
        duration: Duration,
    },

    /// Already terminated
    AlreadyTerminated,
}

impl<T: Clone> UntilStatus<T> {
    /// Check if condition was met
    pub fn is_met(&self) -> bool {
        matches!(self, UntilStatus::ConditionMet { .. })
    }

    /// Check if still waiting
    pub fn is_waiting(&self) -> bool {
        matches!(self, UntilStatus::StillWaiting { .. })
    }
}

impl<T: Clone> TemporalKnowledge<T> {
    /// UNTIL: K remains true until condition C
    ///
    /// K Until C = K holds until C becomes true
    pub fn until<C>(self, condition: C, check_interval: Duration) -> UntilMonitor<T, C>
    where
        C: Fn(&T) -> bool,
    {
        UntilMonitor::new(self, condition, check_interval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::decay::TimeUnit;

    #[test]
    fn test_at_operator() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.0, 0.95),
            1.0,
            TimeUnit::Years,
        );

        // At creation time, confidence unchanged
        let at_now = k.now();
        assert!((at_now.core.confidence().value() - 0.95).abs() < 0.01);

        // At future time, confidence decayed
        let one_year_later = Utc::now() + Duration::days(365);
        let at_future = k.at(one_year_later);
        assert!(at_future.core.confidence().value() < 0.5);
    }

    #[test]
    fn test_historically_with_history() {
        let mut k: TemporalKnowledge<f64> = TemporalKnowledge::with_history(
            EpistemicValue::with_confidence(5.0, 0.9),
            Temporal::now(),
        );

        // Add consistent history
        if let Some(ref mut history) = k.history {
            history.push(TemporalKnowledge::new(
                EpistemicValue::with_confidence(5.0, 0.85),
                Temporal::Instant(Utc::now() - Duration::days(30)),
            ));
        }

        let assessment = k.historically();
        assert!(assessment.was_always_true);
        assert_eq!(assessment.history_depth, 1);
    }

    #[test]
    fn test_historically_with_inconsistency() {
        let mut k: TemporalKnowledge<f64> = TemporalKnowledge::with_history(
            EpistemicValue::with_confidence(5.0, 0.9),
            Temporal::now(),
        );

        // Add inconsistent history (different value)
        if let Some(ref mut history) = k.history {
            history.push(TemporalKnowledge::new(
                EpistemicValue::with_confidence(10.0, 0.85), // Different value!
                Temporal::Instant(Utc::now() - Duration::days(30)),
            ));
        }

        let assessment = k.historically();
        assert!(!assessment.was_always_true);
        assert_eq!(assessment.inconsistencies.len(), 1);
    }

    #[test]
    fn test_since_operator() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.0, 0.95),
            0.1,
            TimeUnit::Years,
        );

        let event =
            TemporalEvent::past("treatment_started", Duration::days(30), "Treatment started");

        let assessment = k.since(&event);
        assert!(assessment.was_true_at_trigger);
        assert!(assessment.maintained_since);
        assert!(assessment.duration_since.num_days() >= 30);
    }

    #[test]
    fn test_eventually_operator() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.0, 0.95),
            1.0,
            TimeUnit::Years,
        );

        let prediction = k.eventually(Duration::days(365));

        // Confidence should decay
        assert!(prediction.projected_confidence.value() < 0.95);

        // Uncertainty should increase
        assert!(prediction.uncertainty > 0.0);
    }

    #[test]
    fn test_always_operator() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.0, 0.95),
            0.1,
            TimeUnit::Years,
        );

        let constraint = k.always(Some(Utc::now() + Duration::days(365)));

        assert!(constraint.requires_revalidation);
        assert!(constraint.is_satisfied());
    }

    #[test]
    fn test_until_operator() {
        let k: TemporalKnowledge<f64> =
            TemporalKnowledge::instant(EpistemicValue::with_confidence(5.0, 0.95));

        let mut monitor = k.until(|v| *v > 10.0, Duration::seconds(1));

        // Initially should be waiting
        let status = monitor.check();
        assert!(status.is_waiting());

        // Not terminated yet
        assert!(!monitor.terminated);
    }

    #[test]
    fn test_project_trajectory() {
        let k: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.0, 1.0),
            1.0,
            TimeUnit::Years,
        );

        let trajectory = k.project_trajectory(5, Duration::days(73));

        assert_eq!(trajectory.len(), 5);

        // Confidence should decrease over trajectory
        for i in 1..trajectory.len() {
            assert!(
                trajectory[i].projected_confidence.value()
                    <= trajectory[i - 1].projected_confidence.value()
            );
        }
    }
}

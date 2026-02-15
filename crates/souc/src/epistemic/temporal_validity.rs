//! Temporal validity enforcement for Knowledge<T>
//!
//! This module wires the existing `ValidityBounds` and `TemporalIndex` types
//! from `temporal.rs` into actual enforcement logic: validation checks,
//! confidence aging models, and temporal comparison utilities.
//!
//! # Aging Models
//!
//! Knowledge confidence degrades over time. This module provides four models:
//!
//! - **ExponentialDecay**: Half-life based decay (`c * 0.5^(t/half_life)`)
//! - **Linear**: Constant rate decay (`c - rate * t`), clamped to 0
//! - **StepFunction**: Sudden drop after threshold
//! - **NoAging**: Confidence is time-invariant

use super::temporal::{ContextTime, TemporalIndex, ValidityBounds};

/// Aging model for confidence decay over time
#[derive(Debug, Clone, PartialEq)]
pub enum AgingModel {
    /// Exponential decay with a half-life in seconds
    ///
    /// After `half_life_seconds`, confidence drops to 50% of original.
    ExponentialDecay { half_life_seconds: u64 },

    /// Linear decay at a constant rate per second
    ///
    /// Confidence = original - (rate_per_second * elapsed), clamped to [0, 1].
    Linear { rate_per_second: f64 },

    /// Step function: confidence is maintained until threshold, then drops
    ///
    /// Before threshold: original confidence.
    /// After threshold: `degraded_confidence`.
    StepFunction {
        threshold_seconds: u64,
        degraded_confidence: f64,
    },

    /// No aging: confidence never degrades
    NoAging,
}

/// Result of temporal validation
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalValidityResult {
    /// Knowledge is temporally valid
    Valid {
        /// Seconds remaining before expiration (None if unbounded)
        remaining_seconds: Option<u64>,
    },

    /// Knowledge has expired
    Expired {
        /// How many seconds ago it expired
        expired_seconds_ago: u64,
    },

    /// Knowledge is valid but confidence has degraded due to aging
    Degraded {
        /// Current (degraded) confidence
        current_confidence: f64,
        /// Original confidence before aging
        original_confidence: f64,
        /// Name of the aging model applied
        model: String,
    },

    /// Temporal validation is not applicable (e.g., compile-time or conditional)
    NotApplicable,
}

/// Validate knowledge against its temporal bounds
///
/// Compares the `current` temporal index against the `bounds` to determine
/// whether the knowledge is still valid, has expired, or is not yet valid.
pub fn validate_temporal(
    bounds: &ValidityBounds,
    current: &TemporalIndex,
) -> TemporalValidityResult {
    match bounds {
        ValidityBounds::Unbounded => TemporalValidityResult::Valid {
            remaining_seconds: None,
        },

        ValidityBounds::Until(end) => {
            match compare_temporal(current, end) {
                Some(diff) if diff <= 0 => {
                    // current <= end: still valid
                    let remaining = (-diff) as u64;
                    TemporalValidityResult::Valid {
                        remaining_seconds: Some(remaining),
                    }
                }
                Some(diff) => {
                    // current > end: expired
                    TemporalValidityResult::Expired {
                        expired_seconds_ago: diff as u64,
                    }
                }
                None => TemporalValidityResult::NotApplicable,
            }
        }

        ValidityBounds::From(start) => {
            match compare_temporal(current, start) {
                Some(diff) if diff >= 0 => {
                    // current >= start: valid (unbounded end)
                    TemporalValidityResult::Valid {
                        remaining_seconds: None,
                    }
                }
                Some(diff) => {
                    // current < start: not yet valid, treat as expired with negative sense
                    TemporalValidityResult::Expired {
                        expired_seconds_ago: (-diff) as u64,
                    }
                }
                None => TemporalValidityResult::NotApplicable,
            }
        }

        ValidityBounds::Range { from, until } => {
            let from_cmp = compare_temporal(current, from);
            let until_cmp = compare_temporal(current, until);

            match (from_cmp, until_cmp) {
                (Some(from_diff), Some(until_diff)) => {
                    if from_diff < 0 {
                        // Before range start
                        TemporalValidityResult::Expired {
                            expired_seconds_ago: (-from_diff) as u64,
                        }
                    } else if until_diff > 0 {
                        // After range end
                        TemporalValidityResult::Expired {
                            expired_seconds_ago: until_diff as u64,
                        }
                    } else {
                        // Within range
                        TemporalValidityResult::Valid {
                            remaining_seconds: Some((-until_diff) as u64),
                        }
                    }
                }
                _ => TemporalValidityResult::NotApplicable,
            }
        }

        ValidityBounds::Conditional { .. } => {
            // Conditional validity requires runtime evaluation
            TemporalValidityResult::NotApplicable
        }
    }
}

/// Check whether a `ContextTime` is currently valid given a `current` temporal index
///
/// This is the high-level entry point for temporal enforcement.
pub fn is_currently_valid(context: &ContextTime, current: &TemporalIndex) -> bool {
    match validate_temporal(&context.validity, current) {
        TemporalValidityResult::Valid { .. } => true,
        TemporalValidityResult::NotApplicable => true, // Conservative: assume valid if can't check
        _ => false,
    }
}

/// Apply an aging model to degrade confidence over elapsed time
///
/// Returns the degraded confidence value, clamped to [0.0, 1.0].
pub fn apply_aging(confidence: f64, elapsed_seconds: u64, model: &AgingModel) -> f64 {
    let result = match model {
        AgingModel::ExponentialDecay { half_life_seconds } => {
            if *half_life_seconds == 0 {
                return 0.0;
            }
            let exponent = elapsed_seconds as f64 / *half_life_seconds as f64;
            confidence * (0.5_f64).powf(exponent)
        }

        AgingModel::Linear { rate_per_second } => {
            confidence - (rate_per_second * elapsed_seconds as f64)
        }

        AgingModel::StepFunction {
            threshold_seconds,
            degraded_confidence,
        } => {
            if elapsed_seconds <= *threshold_seconds {
                confidence
            } else {
                *degraded_confidence
            }
        }

        AgingModel::NoAging => confidence,
    };

    result.clamp(0.0, 1.0)
}

/// Convert a `TemporalIndex` to an approximate epoch (seconds since year 0)
///
/// This provides a comparable scalar for temporal indices that have absolute dates.
/// Returns `None` for CompileTime, Runtime, and Version indices.
pub fn temporal_index_to_epoch(idx: &TemporalIndex) -> Option<u64> {
    match idx {
        TemporalIndex::Absolute { year, month, day } => {
            // Approximate: 365.25 days/year, 30.44 days/month
            let y = *year as u64;
            let m = month.unwrap_or(1) as u64;
            let d = day.unwrap_or(1) as u64;

            let seconds = y * 365 * 86_400
                + (y / 4) * 86_400 // leap year approximation
                + (m - 1) * 30 * 86_400
                + (d - 1) * 86_400;

            Some(seconds)
        }
        TemporalIndex::Relative { base, offset } => {
            temporal_index_to_epoch(base).map(|base_epoch| {
                let offset_seconds = (offset.years as i64 * 365 * 86_400)
                    + (offset.months as i64 * 30 * 86_400)
                    + (offset.days as i64 * 86_400);

                if offset_seconds >= 0 {
                    base_epoch + offset_seconds as u64
                } else {
                    base_epoch.saturating_sub((-offset_seconds) as u64)
                }
            })
        }
        TemporalIndex::CompileTime | TemporalIndex::Runtime | TemporalIndex::Version { .. } => None,
    }
}

/// Compare two temporal indices, returning the difference in seconds (a - b)
///
/// Returns `Some(diff)` where diff > 0 means `a` is after `b`,
/// diff < 0 means `a` is before `b`, and diff == 0 means they are equal.
///
/// Returns `None` if the indices are not comparable.
pub fn compare_temporal(a: &TemporalIndex, b: &TemporalIndex) -> Option<i64> {
    let epoch_a = temporal_index_to_epoch(a)?;
    let epoch_b = temporal_index_to_epoch(b)?;

    Some(epoch_a as i64 - epoch_b as i64)
}

/// Compute remaining validity in seconds
///
/// For bounded validity, returns the number of seconds until expiration.
/// Returns `None` for unbounded, conditional, or incomparable bounds.
pub fn remaining_validity(bounds: &ValidityBounds, current: &TemporalIndex) -> Option<u64> {
    match bounds {
        ValidityBounds::Unbounded => None,

        ValidityBounds::Until(end) => {
            let diff = compare_temporal(current, end)?;
            if diff <= 0 {
                Some((-diff) as u64)
            } else {
                Some(0) // Already expired
            }
        }

        ValidityBounds::From(_) => None, // No upper bound

        ValidityBounds::Range { until, .. } => {
            let diff = compare_temporal(current, until)?;
            if diff <= 0 {
                Some((-diff) as u64)
            } else {
                Some(0) // Already expired
            }
        }

        ValidityBounds::Conditional { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::temporal::{ContextIndex, TemporalOffset};

    // --- validate_temporal ---

    #[test]
    fn test_validate_unbounded() {
        let result =
            validate_temporal(&ValidityBounds::Unbounded, &TemporalIndex::date(2026, 1, 1));
        assert!(matches!(
            result,
            TemporalValidityResult::Valid {
                remaining_seconds: None
            }
        ));
    }

    #[test]
    fn test_validate_until_valid() {
        let bounds = ValidityBounds::Until(TemporalIndex::date(2027, 1, 1));
        let current = TemporalIndex::date(2026, 1, 1);
        let result = validate_temporal(&bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Valid { .. }));
        if let TemporalValidityResult::Valid {
            remaining_seconds: Some(remaining),
        } = result
        {
            assert!(remaining > 0);
        }
    }

    #[test]
    fn test_validate_until_expired() {
        let bounds = ValidityBounds::Until(TemporalIndex::date(2024, 1, 1));
        let current = TemporalIndex::date(2026, 1, 1);
        let result = validate_temporal(&bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Expired { .. }));
    }

    #[test]
    fn test_validate_from_valid() {
        let bounds = ValidityBounds::From(TemporalIndex::date(2020, 1, 1));
        let current = TemporalIndex::date(2026, 1, 1);
        let result = validate_temporal(&bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Valid { .. }));
    }

    #[test]
    fn test_validate_from_not_yet_valid() {
        let bounds = ValidityBounds::From(TemporalIndex::date(2030, 1, 1));
        let current = TemporalIndex::date(2026, 1, 1);
        let result = validate_temporal(&bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Expired { .. }));
    }

    #[test]
    fn test_validate_range_within() {
        let bounds = ValidityBounds::Range {
            from: TemporalIndex::date(2025, 1, 1),
            until: TemporalIndex::date(2027, 1, 1),
        };
        let current = TemporalIndex::date(2026, 6, 1);
        let result = validate_temporal(&bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Valid { .. }));
    }

    #[test]
    fn test_validate_range_before() {
        let bounds = ValidityBounds::Range {
            from: TemporalIndex::date(2025, 1, 1),
            until: TemporalIndex::date(2027, 1, 1),
        };
        let current = TemporalIndex::date(2024, 1, 1);
        let result = validate_temporal(&bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Expired { .. }));
    }

    #[test]
    fn test_validate_range_after() {
        let bounds = ValidityBounds::Range {
            from: TemporalIndex::date(2025, 1, 1),
            until: TemporalIndex::date(2027, 1, 1),
        };
        let current = TemporalIndex::date(2028, 1, 1);
        let result = validate_temporal(&bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Expired { .. }));
    }

    #[test]
    fn test_validate_conditional_not_applicable() {
        let bounds = ValidityBounds::Conditional {
            condition: "lab_certified".into(),
        };
        let result = validate_temporal(&bounds, &TemporalIndex::date(2026, 1, 1));
        assert!(matches!(result, TemporalValidityResult::NotApplicable));
    }

    #[test]
    fn test_validate_compile_time_not_applicable() {
        let bounds = ValidityBounds::Until(TemporalIndex::date(2027, 1, 1));
        let result = validate_temporal(&bounds, &TemporalIndex::CompileTime);
        assert!(matches!(result, TemporalValidityResult::NotApplicable));
    }

    // --- is_currently_valid ---

    #[test]
    fn test_is_currently_valid_unbounded() {
        let ctx = ContextTime::current();
        assert!(is_currently_valid(&ctx, &TemporalIndex::date(2026, 1, 1)));
    }

    #[test]
    fn test_is_currently_valid_expired() {
        let ctx = ContextTime {
            temporal: TemporalIndex::date(2024, 1, 1),
            context: ContextIndex::Unspecified,
            validity: ValidityBounds::Until(TemporalIndex::date(2024, 6, 1)),
        };
        let current = TemporalIndex::date(2026, 1, 1);
        assert!(!is_currently_valid(&ctx, &current));
    }

    // --- apply_aging ---

    #[test]
    fn test_exponential_decay_half_life() {
        let model = AgingModel::ExponentialDecay {
            half_life_seconds: 3600,
        };
        // At exactly one half-life, confidence should be halved
        let result = apply_aging(1.0, 3600, &model);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_decay_two_half_lives() {
        let model = AgingModel::ExponentialDecay {
            half_life_seconds: 3600,
        };
        // At two half-lives: 0.25
        let result = apply_aging(1.0, 7200, &model);
        assert!((result - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_decay_zero_elapsed() {
        let model = AgingModel::ExponentialDecay {
            half_life_seconds: 3600,
        };
        let result = apply_aging(0.9, 0, &model);
        assert!((result - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_linear_decay() {
        let model = AgingModel::Linear {
            rate_per_second: 0.001,
        };
        // After 100 seconds: 0.8 - 0.1 = 0.7
        let result = apply_aging(0.8, 100, &model);
        assert!((result - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_linear_decay_clamps_to_zero() {
        let model = AgingModel::Linear {
            rate_per_second: 0.01,
        };
        // After 200 seconds: 0.5 - 2.0 = -1.5, clamped to 0.0
        let result = apply_aging(0.5, 200, &model);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_step_function_before_threshold() {
        let model = AgingModel::StepFunction {
            threshold_seconds: 3600,
            degraded_confidence: 0.1,
        };
        let result = apply_aging(0.95, 1800, &model);
        assert!((result - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_step_function_after_threshold() {
        let model = AgingModel::StepFunction {
            threshold_seconds: 3600,
            degraded_confidence: 0.1,
        };
        let result = apply_aging(0.95, 7200, &model);
        assert!((result - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_no_aging() {
        let model = AgingModel::NoAging;
        let result = apply_aging(0.85, 999_999, &model);
        assert!((result - 0.85).abs() < 1e-6);
    }

    // --- temporal_index_to_epoch / compare_temporal ---

    #[test]
    fn test_temporal_index_to_epoch_absolute() {
        let idx = TemporalIndex::date(2026, 1, 1);
        let epoch = temporal_index_to_epoch(&idx);
        assert!(epoch.is_some());
        assert!(epoch.unwrap() > 0);
    }

    #[test]
    fn test_temporal_index_to_epoch_compile_time_returns_none() {
        assert!(temporal_index_to_epoch(&TemporalIndex::CompileTime).is_none());
    }

    #[test]
    fn test_temporal_index_to_epoch_version_returns_none() {
        let idx = TemporalIndex::Version {
            ontology: "ChEBI".into(),
            version: "2024.1".into(),
        };
        assert!(temporal_index_to_epoch(&idx).is_none());
    }

    #[test]
    fn test_compare_temporal_ordering() {
        let earlier = TemporalIndex::date(2024, 1, 1);
        let later = TemporalIndex::date(2026, 1, 1);
        let diff = compare_temporal(&earlier, &later);
        assert!(diff.is_some());
        assert!(diff.unwrap() < 0); // earlier is before later
    }

    #[test]
    fn test_compare_temporal_same_date() {
        let a = TemporalIndex::date(2026, 6, 15);
        let b = TemporalIndex::date(2026, 6, 15);
        let diff = compare_temporal(&a, &b);
        assert!(diff.is_some());
        assert_eq!(diff.unwrap(), 0);
    }

    #[test]
    fn test_compare_temporal_incomparable() {
        let a = TemporalIndex::CompileTime;
        let b = TemporalIndex::date(2026, 1, 1);
        assert!(compare_temporal(&a, &b).is_none());
    }

    // --- remaining_validity ---

    #[test]
    fn test_remaining_validity_unbounded() {
        assert!(
            remaining_validity(&ValidityBounds::Unbounded, &TemporalIndex::date(2026, 1, 1))
                .is_none()
        );
    }

    #[test]
    fn test_remaining_validity_until_valid() {
        let bounds = ValidityBounds::Until(TemporalIndex::date(2027, 1, 1));
        let current = TemporalIndex::date(2026, 1, 1);
        let remaining = remaining_validity(&bounds, &current);
        assert!(remaining.is_some());
        assert!(remaining.unwrap() > 0);
    }

    #[test]
    fn test_remaining_validity_until_expired() {
        let bounds = ValidityBounds::Until(TemporalIndex::date(2024, 1, 1));
        let current = TemporalIndex::date(2026, 1, 1);
        let remaining = remaining_validity(&bounds, &current);
        assert_eq!(remaining, Some(0));
    }

    #[test]
    fn test_remaining_validity_conditional() {
        let bounds = ValidityBounds::Conditional {
            condition: "always".into(),
        };
        assert!(remaining_validity(&bounds, &TemporalIndex::date(2026, 1, 1)).is_none());
    }

    // --- Relative temporal index ---

    #[test]
    fn test_relative_temporal_epoch() {
        let base = TemporalIndex::date(2026, 1, 1);
        let relative = TemporalIndex::Relative {
            base: Box::new(base.clone()),
            offset: TemporalOffset::days(30),
        };
        let base_epoch = temporal_index_to_epoch(&base).unwrap();
        let rel_epoch = temporal_index_to_epoch(&relative).unwrap();
        let diff = rel_epoch - base_epoch;
        assert_eq!(diff, 30 * 86_400);
    }

    #[test]
    fn test_exponential_decay_zero_half_life() {
        let model = AgingModel::ExponentialDecay {
            half_life_seconds: 0,
        };
        let result = apply_aging(0.9, 100, &model);
        assert!((result - 0.0).abs() < 1e-6);
    }
}

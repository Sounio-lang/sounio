//! Temporal Composition Rules
//!
//! Extends Day 32's composition algebra with temporal semantics.
//!
//! # Composition Rules
//!
//! ```text
//! TENSOR WITH TIME:
//! ─────────────────
//! (K₁@t₁) ⊗ (K₂@t₂) = result@combine_temporal(t₁, t₂)
//!
//! Rules:
//!   Instant(i₁) × Instant(i₂) → Instant(max(i₁, i₂))
//!   Interval(a₁,b₁) × Interval(a₂,b₂) → Interval(max(a₁,a₂), min(b₁,b₂))
//!   Decaying(c₁,D₁) × Decaying(c₂,D₂) → Decaying(max(c₁,c₂), D₁×D₂)
//!   Timeless × t → t
//!
//! JOIN WITH TIME:
//! ───────────────
//! (K₁@t₁) ⊔ (K₂@t₂)
//!
//! Recency bonus:
//!   recency_bonus(Δt) = 1 + log(1 + Δt.days() / 30)
//!
//! More recent knowledge has increased weight.
//! ```

use super::knowledge::TemporalKnowledge;
use super::types::Temporal;
use crate::epistemic::composition::{
    CombinationStrategy, ConfidenceValue, EpistemicValue, Fusible, ProvenanceNode,
    combine_confidence,
};
use chrono::Utc;
use std::collections::HashSet;

/// Result of temporal join operation
#[derive(Clone, Debug)]
pub enum TemporalJoinResult<T> {
    /// Values agreed, confidence boosted with recency
    Concordant {
        result: TemporalKnowledge<T>,
        recency_bonus: f64,
    },

    /// Minor conflict resolved with penalty and recency
    Resolved {
        result: TemporalKnowledge<T>,
        conflict_level: f64,
        recency_bonus: f64,
    },

    /// Conflict too large to resolve
    Irreconcilable {
        k1: TemporalKnowledge<T>,
        k2: TemporalKnowledge<T>,
        conflict_level: f64,
    },
}

impl<T> TemporalJoinResult<T> {
    /// Check if join was successful
    pub fn is_success(&self) -> bool {
        !matches!(self, TemporalJoinResult::Irreconcilable { .. })
    }

    /// Get the result if successful
    pub fn result(&self) -> Option<&TemporalKnowledge<T>> {
        match self {
            TemporalJoinResult::Concordant { result, .. } => Some(result),
            TemporalJoinResult::Resolved { result, .. } => Some(result),
            TemporalJoinResult::Irreconcilable { .. } => None,
        }
    }

    /// Get the recency bonus applied
    pub fn recency_bonus(&self) -> Option<f64> {
        match self {
            TemporalJoinResult::Concordant { recency_bonus, .. } => Some(*recency_bonus),
            TemporalJoinResult::Resolved { recency_bonus, .. } => Some(*recency_bonus),
            TemporalJoinResult::Irreconcilable { .. } => None,
        }
    }
}

/// Trait for temporal composition operations
pub trait TemporalComposition<T>
where
    T: Clone,
{
    /// Temporal-aware tensor product
    fn tensor_temporal<B: Clone>(self, other: TemporalKnowledge<B>) -> TemporalKnowledge<(T, B)>;

    /// Temporal-aware join with recency bonus
    fn join_temporal(
        self,
        other: TemporalKnowledge<T>,
        conflict_threshold: f64,
    ) -> TemporalJoinResult<T>
    where
        T: Fusible;

    /// Condition on temporal evidence
    fn condition_temporal<E>(
        self,
        evidence: &TemporalKnowledge<E>,
        likelihood: impl Fn(&T, &E) -> f64,
    ) -> TemporalKnowledge<T>;
}

impl<T: Clone> TemporalComposition<T> for TemporalKnowledge<T> {
    /// Temporal-aware tensor product
    ///
    /// Combines independent knowledge, using decay values at the combined
    /// temporal point.
    fn tensor_temporal<B: Clone>(self, other: TemporalKnowledge<B>) -> TemporalKnowledge<(T, B)> {
        let combined_temporal = self.temporal.combine_for_tensor(&other.temporal);
        let effective_time = combined_temporal.effective_instant();

        // Apply decay at the effective time
        let e1 = self.core.confidence().value() * self.compute_decay_at(effective_time);
        let e2 = other.core.confidence().value() * other.compute_decay_at(effective_time);

        // Compute ontology correlation
        let gamma = ontology_correlation(self.core.ontology(), other.core.ontology());
        let combined_confidence = e1 * e2 * gamma;

        // Combine ontologies
        let mut combined_ontology = self.core.ontology().clone();
        combined_ontology.extend(other.core.ontology().iter().cloned());

        TemporalKnowledge {
            core: EpistemicValue::new(
                (self.core.value().clone(), other.core.value().clone()),
                ConfidenceValue::new(combined_confidence.clamp(0.0, 1.0))
                    .unwrap_or_else(|_| ConfidenceValue::zero()),
                combined_ontology,
                ProvenanceNode::derived(
                    "temporal_tensor",
                    vec![
                        self.core.provenance().clone(),
                        other.core.provenance().clone(),
                    ],
                ),
            ),
            temporal: combined_temporal,
            history: None,
        }
    }

    /// Temporal-aware join with recency bonus
    ///
    /// More recent knowledge gets a bonus weight in the merge.
    fn join_temporal(
        self,
        other: TemporalKnowledge<T>,
        conflict_threshold: f64,
    ) -> TemporalJoinResult<T>
    where
        T: Fusible,
    {
        // Determine which is newer
        let t1 = self.temporal.effective_instant();
        let t2 = other.temporal.effective_instant();
        let (newer, older, time_diff) = if t1 >= t2 {
            (&self, &other, t1.signed_duration_since(t2))
        } else {
            (&other, &self, t2.signed_duration_since(t1))
        };

        // Compute recency bonus: 30 days more recent = +69% weight (ln(2))
        let recency_bonus = 1.0 + (1.0 + time_diff.num_days() as f64 / 30.0).ln();

        let w_newer = newer.core.confidence().value() * recency_bonus;
        let w_older = older.core.confidence().value();

        let conflict = newer.core.value().conflict(older.core.value());

        // Combine ontologies
        let mut combined_ontology = newer.core.ontology().clone();
        combined_ontology.extend(older.core.ontology().iter().cloned());

        if conflict.value() < 0.05 {
            // CONCORDANT: Values agree, boost confidence
            let merged_value =
                newer
                    .core
                    .value()
                    .weighted_merge(older.core.value(), w_newer, w_older);

            let boosted = combine_confidence(
                ConfidenceValue::new(w_newer / (w_newer + w_older).max(0.001))
                    .unwrap_or_else(|_| ConfidenceValue::zero()),
                ConfidenceValue::new(w_older / (w_newer + w_older).max(0.001))
                    .unwrap_or_else(|_| ConfidenceValue::zero()),
                &CombinationStrategy::DempsterShafer,
            );

            let result = TemporalKnowledge {
                core: EpistemicValue::new(
                    merged_value,
                    boosted,
                    combined_ontology,
                    ProvenanceNode::derived_with_metadata(
                        "temporal_join_concordant",
                        vec![
                            newer.core.provenance().clone(),
                            older.core.provenance().clone(),
                        ],
                        format!("recency_bonus={:.3}", recency_bonus),
                    ),
                ),
                temporal: newer.temporal.clone(),
                history: Some(vec![older.clone()]),
            };

            TemporalJoinResult::Concordant {
                result,
                recency_bonus,
            }
        } else if conflict.value() < conflict_threshold {
            // RESOLVABLE: Merge with penalty, recency gives extra weight
            let merged_value = newer.core.value().weighted_merge(
                older.core.value(),
                w_newer * (1.0 + recency_bonus * 0.5),
                w_older,
            );

            let avg_conf = (w_newer + w_older) / 2.0;
            let penalized = avg_conf * (1.0 - conflict.value());

            let result = TemporalKnowledge {
                core: EpistemicValue::new(
                    merged_value,
                    ConfidenceValue::new(penalized.clamp(0.0, 1.0))
                        .unwrap_or_else(|_| ConfidenceValue::zero()),
                    combined_ontology,
                    ProvenanceNode::derived_with_metadata(
                        "temporal_join_resolved",
                        vec![
                            newer.core.provenance().clone(),
                            older.core.provenance().clone(),
                        ],
                        format!(
                            "conflict={:.3}, recency_bonus={:.3}",
                            conflict.value(),
                            recency_bonus
                        ),
                    ),
                ),
                temporal: newer.temporal.clone(),
                history: Some(vec![older.clone()]),
            };

            TemporalJoinResult::Resolved {
                result,
                conflict_level: conflict.value(),
                recency_bonus,
            }
        } else {
            // IRRECONCILABLE
            TemporalJoinResult::Irreconcilable {
                k1: self,
                k2: other,
                conflict_level: conflict.value(),
            }
        }
    }

    /// Condition on temporal evidence
    ///
    /// Applies Bayesian update with temporal freshness discount.
    fn condition_temporal<E>(
        self,
        evidence: &TemporalKnowledge<E>,
        likelihood: impl Fn(&T, &E) -> f64,
    ) -> TemporalKnowledge<T> {
        // Compute freshness discount based on evidence age
        let evidence_time = evidence.temporal.effective_instant();
        let prior_time = self.temporal.effective_instant();

        let freshness = if evidence_time < prior_time {
            // Evidence is older than prior - apply discount
            let age = prior_time.signed_duration_since(evidence_time);
            let years = age.num_seconds() as f64 / 31556952.0;
            (-0.3 * years).exp() // 30% discount per year of staleness
        } else {
            1.0 // Fresh evidence
        };

        // Compute likelihood
        let p_e_given_h = likelihood(self.core.value(), evidence.core.value());
        let adjusted_likelihood = 1.0 - (1.0 - p_e_given_h) * freshness;

        // Bayesian update
        let prior = self.core.confidence().value();
        let posterior = if adjusted_likelihood > 0.0 {
            (prior * adjusted_likelihood).clamp(0.0, 1.0)
        } else {
            prior
        };

        // Combine ontologies
        let mut combined_ontology = self.core.ontology().clone();
        combined_ontology.extend(evidence.core.ontology().iter().cloned());

        TemporalKnowledge {
            core: EpistemicValue::new(
                self.core.value().clone(),
                ConfidenceValue::new(posterior).unwrap_or_else(|_| ConfidenceValue::zero()),
                combined_ontology,
                ProvenanceNode::derived_with_metadata(
                    "temporal_condition",
                    vec![
                        self.core.provenance().clone(),
                        evidence.core.provenance().clone(),
                    ],
                    format!("freshness={:.3}", freshness),
                ),
            ),
            temporal: Temporal::Instant(Utc::now()),
            history: self.history.clone(),
        }
    }
}

/// Compute ontology correlation factor
fn ontology_correlation(
    d1: &HashSet<crate::epistemic::composition::knowledge::OntologyRef>,
    d2: &HashSet<crate::epistemic::composition::knowledge::OntologyRef>,
) -> f64 {
    if d1.is_empty() && d2.is_empty() {
        return 1.0;
    }

    let intersection = d1.intersection(d2).count() as f64;
    let union = d1.union(d2).count() as f64;

    if union == 0.0 {
        1.0
    } else {
        let jaccard = intersection / union;
        1.0 - 0.5 * jaccard
    }
}

/// Join multiple temporal knowledge values
pub fn temporal_join_all<T: Clone + Fusible>(
    values: Vec<TemporalKnowledge<T>>,
    conflict_threshold: f64,
) -> Result<TemporalKnowledge<T>, Vec<TemporalKnowledge<T>>> {
    if values.is_empty() {
        panic!("Cannot join empty list");
    }
    if values.len() == 1 {
        return Ok(values.into_iter().next().unwrap());
    }

    let mut iter = values.into_iter();
    let mut acc = iter.next().unwrap();

    for v in iter {
        match acc.join_temporal(v, conflict_threshold) {
            TemporalJoinResult::Concordant { result, .. }
            | TemporalJoinResult::Resolved { result, .. } => {
                acc = result;
            }
            TemporalJoinResult::Irreconcilable { k1, k2, .. } => {
                return Err(vec![k1, k2]);
            }
        }
    }

    Ok(acc)
}

/// Tensor multiple temporal knowledge values
pub fn temporal_tensor_all<T: Clone>(
    values: Vec<TemporalKnowledge<T>>,
) -> TemporalKnowledge<Vec<T>> {
    if values.is_empty() {
        return TemporalKnowledge::timeless(EpistemicValue::with_confidence(Vec::new(), 1.0));
    }

    let mut result_values = Vec::with_capacity(values.len());
    let mut combined_temporal = Temporal::Timeless;
    let mut combined_confidence = 1.0_f64;
    let mut combined_ontology = HashSet::new();
    let mut provenances = Vec::new();

    for v in values {
        let effective_time = combined_temporal.effective_instant();
        let decay = v.compute_decay_at(effective_time);

        combined_confidence *= v.core.confidence().value() * decay;
        combined_temporal = combined_temporal.combine_for_tensor(&v.temporal);
        combined_ontology.extend(v.core.ontology().iter().cloned());
        result_values.push(v.core.value().clone());
        provenances.push(v.core.provenance().clone());
    }

    TemporalKnowledge {
        core: EpistemicValue::new(
            result_values,
            ConfidenceValue::new(combined_confidence.clamp(0.0, 1.0))
                .unwrap_or_else(|_| ConfidenceValue::zero()),
            combined_ontology,
            ProvenanceNode::derived("temporal_tensor_all", provenances),
        ),
        temporal: combined_temporal,
        history: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::decay::TimeUnit;
    use chrono::Duration;

    #[test]
    fn test_temporal_tensor() {
        let k1: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(5.0, 0.9),
            0.1,
            TimeUnit::Years,
        );

        let k2: TemporalKnowledge<f64> = TemporalKnowledge::decaying(
            EpistemicValue::with_confidence(10.0, 0.8),
            0.1,
            TimeUnit::Years,
        );

        let result = k1.tensor_temporal(k2);

        assert_eq!(result.value(), &(5.0, 10.0));
        // Confidence = 0.9 * 0.8 * γ (γ ≈ 1.0 for empty ontologies)
        assert!(result.core.confidence().value() < 0.72 + 0.01);
    }

    #[test]
    fn test_temporal_join_concordant() {
        let k1: TemporalKnowledge<f64> =
            TemporalKnowledge::instant(EpistemicValue::with_confidence(5.0, 0.8));

        let k2: TemporalKnowledge<f64> =
            TemporalKnowledge::instant(EpistemicValue::with_confidence(5.05, 0.75));

        let result = k1.join_temporal(k2, 0.3);

        match result {
            TemporalJoinResult::Concordant {
                result: r,
                recency_bonus,
            } => {
                assert!((*r.value() - 5.0).abs() < 0.1);
                assert!(recency_bonus >= 1.0);
            }
            _ => panic!("Expected concordant join"),
        }
    }

    #[test]
    fn test_temporal_join_with_recency() {
        // Create older knowledge
        let now = Utc::now();
        let k_old: TemporalKnowledge<f64> = TemporalKnowledge {
            core: EpistemicValue::with_confidence(5.0, 0.8),
            temporal: Temporal::Instant(now - Duration::days(60)),
            history: None,
        };

        // Create newer knowledge
        let k_new: TemporalKnowledge<f64> = TemporalKnowledge {
            core: EpistemicValue::with_confidence(6.0, 0.8),
            temporal: Temporal::Instant(now),
            history: None,
        };

        let result = k_old.join_temporal(k_new, 0.3);

        match result {
            TemporalJoinResult::Resolved {
                result: r,
                recency_bonus,
                ..
            } => {
                // Recency bonus should be significant (60 days = ~2 months)
                assert!(recency_bonus > 1.5);
                // Value should be closer to newer (6.0)
                assert!(*r.value() > 5.5);
            }
            _ => panic!("Expected resolved join"),
        }
    }

    #[test]
    fn test_condition_temporal_fresh() {
        let prior: TemporalKnowledge<bool> =
            TemporalKnowledge::instant(EpistemicValue::with_confidence(true, 0.5));

        let evidence: TemporalKnowledge<bool> =
            TemporalKnowledge::instant(EpistemicValue::with_confidence(true, 0.9));

        let posterior = prior.condition_temporal(&evidence, |_, _| 0.9);

        // Confidence should increase
        assert!(posterior.core.confidence().value() > 0.4);
    }

    #[test]
    fn test_condition_temporal_stale() {
        let now = Utc::now();

        // Recent prior
        let prior: TemporalKnowledge<bool> = TemporalKnowledge {
            core: EpistemicValue::with_confidence(true, 0.5),
            temporal: Temporal::Instant(now),
            history: None,
        };

        // Old evidence
        let evidence: TemporalKnowledge<bool> = TemporalKnowledge {
            core: EpistemicValue::with_confidence(true, 0.9),
            temporal: Temporal::Instant(now - Duration::days(365 * 2)), // 2 years old
            history: None,
        };

        let posterior = prior.condition_temporal(&evidence, |_, _| 0.9);

        // Confidence should still increase but less than fresh evidence
        // due to staleness discount
        assert!(posterior.core.confidence().value() > 0.4);
        assert!(posterior.core.confidence().value() < 0.5);
    }

    #[test]
    fn test_temporal_join_all() {
        let values: Vec<TemporalKnowledge<f64>> = vec![
            TemporalKnowledge::instant(EpistemicValue::with_confidence(5.0, 0.9)),
            TemporalKnowledge::instant(EpistemicValue::with_confidence(5.1, 0.85)),
            TemporalKnowledge::instant(EpistemicValue::with_confidence(5.05, 0.88)),
        ];

        let result = temporal_join_all(values, 0.3);

        assert!(result.is_ok());
        let r = result.unwrap();
        assert!((*r.value() - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_temporal_tensor_all() {
        let values: Vec<TemporalKnowledge<f64>> = vec![
            TemporalKnowledge::instant(EpistemicValue::with_confidence(1.0, 0.9)),
            TemporalKnowledge::instant(EpistemicValue::with_confidence(2.0, 0.8)),
            TemporalKnowledge::instant(EpistemicValue::with_confidence(3.0, 0.7)),
        ];

        let result = temporal_tensor_all(values);

        assert_eq!(result.value().len(), 3);
        // Confidence ≈ 0.9 * 0.8 * 0.7 = 0.504
        assert!((result.core.confidence().value() - 0.504).abs() < 0.1);
    }
}

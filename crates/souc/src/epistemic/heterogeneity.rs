//! Epistemic Heterogeneity Handler
//!
//! Resolves conflicts when combining knowledge from sources with different
//! epistemic statuses (confidence levels, evidence types, provenance).
//!
//! # Resolution Strategies
//!
//! 1. **Prioritized**: Higher confidence source wins
//! 2. **Bayesian**: Combine using probabilistic methods
//! 3. **AGM**: Use belief revision theory (Alchourrón-Gärdenfors-Makinson)
//! 4. **Consensus**: Require agreement above threshold
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::epistemic::heterogeneity::{HeterogeneityResolver, Strategy};
//!
//! let resolver = HeterogeneityResolver::new(Strategy::Bayesian);
//! let combined = resolver.resolve(&[status1, status2, status3])?;
//! ```

use std::collections::HashMap;

use super::{Confidence, EpistemicStatus, Evidence, Revisability, Source};

/// Strategy for resolving epistemic heterogeneity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolutionStrategy {
    /// Higher confidence source wins
    Prioritized,
    /// Bayesian combination using log-odds
    #[default]
    Bayesian,
    /// AGM belief revision
    AGM,
    /// Require consensus above threshold
    Consensus { threshold: u8 }, // threshold as percentage (0-100)
    /// Conservative: use minimum confidence
    Conservative,
    /// Optimistic: use maximum confidence
    Optimistic,
    /// Weighted average by evidence strength
    WeightedAverage,
}

/// Configuration for heterogeneity resolution
#[derive(Debug, Clone)]
pub struct HeterogeneityConfig {
    /// Resolution strategy
    pub strategy: ResolutionStrategy,
    /// Minimum confidence difference to consider a conflict
    pub conflict_threshold: f64,
    /// Weight for recency in combination
    pub recency_weight: f64,
    /// Trust weights per source type
    pub source_weights: HashMap<String, f64>,
    /// Whether to preserve all evidence or merge
    pub preserve_evidence: bool,
}

impl Default for HeterogeneityConfig {
    fn default() -> Self {
        Self {
            strategy: ResolutionStrategy::Bayesian,
            conflict_threshold: 0.1,
            recency_weight: 0.0,
            source_weights: HashMap::new(),
            preserve_evidence: true,
        }
    }
}

/// Result of heterogeneity resolution
#[derive(Debug, Clone)]
pub struct ResolutionResult {
    /// Combined epistemic status
    pub status: EpistemicStatus,
    /// Was there a conflict?
    pub had_conflict: bool,
    /// Confidence loss due to heterogeneity (0-1)
    pub confidence_penalty: f64,
    /// Sources that were overridden
    pub overridden_sources: Vec<Source>,
    /// Explanation of resolution
    pub explanation: String,
}

/// Handler for epistemic heterogeneity
pub struct HeterogeneityResolver {
    config: HeterogeneityConfig,
}

impl HeterogeneityResolver {
    /// Create a new resolver with default config
    pub fn new() -> Self {
        Self {
            config: HeterogeneityConfig::default(),
        }
    }

    /// Create with specific strategy
    pub fn with_strategy(strategy: ResolutionStrategy) -> Self {
        Self {
            config: HeterogeneityConfig {
                strategy,
                ..Default::default()
            },
        }
    }

    /// Create with full config
    pub fn with_config(config: HeterogeneityConfig) -> Self {
        Self { config }
    }

    /// Resolve epistemic heterogeneity across multiple statuses
    pub fn resolve(&self, statuses: &[EpistemicStatus]) -> ResolutionResult {
        if statuses.is_empty() {
            return ResolutionResult {
                status: EpistemicStatus::default(),
                had_conflict: false,
                confidence_penalty: 0.0,
                overridden_sources: vec![],
                explanation: "No statuses to resolve".into(),
            };
        }

        if statuses.len() == 1 {
            return ResolutionResult {
                status: statuses[0].clone(),
                had_conflict: false,
                confidence_penalty: 0.0,
                overridden_sources: vec![],
                explanation: "Single source, no resolution needed".into(),
            };
        }

        // Detect conflicts
        let (had_conflict, max_diff) = self.detect_conflict(statuses);

        // Apply resolution strategy
        match self.config.strategy {
            ResolutionStrategy::Prioritized => {
                self.resolve_prioritized(statuses, had_conflict, max_diff)
            }
            ResolutionStrategy::Bayesian => self.resolve_bayesian(statuses, had_conflict, max_diff),
            ResolutionStrategy::AGM => self.resolve_agm(statuses, had_conflict, max_diff),
            ResolutionStrategy::Consensus { threshold } => {
                self.resolve_consensus(statuses, threshold, had_conflict, max_diff)
            }
            ResolutionStrategy::Conservative => {
                self.resolve_conservative(statuses, had_conflict, max_diff)
            }
            ResolutionStrategy::Optimistic => {
                self.resolve_optimistic(statuses, had_conflict, max_diff)
            }
            ResolutionStrategy::WeightedAverage => {
                self.resolve_weighted_average(statuses, had_conflict, max_diff)
            }
        }
    }

    /// Detect if there's a conflict between statuses
    fn detect_conflict(&self, statuses: &[EpistemicStatus]) -> (bool, f64) {
        let mut max_diff = 0.0f64;

        for i in 0..statuses.len() {
            for j in (i + 1)..statuses.len() {
                let diff = (statuses[i].confidence.value() - statuses[j].confidence.value()).abs();
                max_diff = max_diff.max(diff);
            }
        }

        (max_diff > self.config.conflict_threshold, max_diff)
    }

    /// Resolve using priority (highest confidence wins)
    fn resolve_prioritized(
        &self,
        statuses: &[EpistemicStatus],
        had_conflict: bool,
        max_diff: f64,
    ) -> ResolutionResult {
        let best = statuses
            .iter()
            .max_by(|a, b| {
                a.confidence
                    .value()
                    .partial_cmp(&b.confidence.value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let overridden: Vec<Source> = statuses
            .iter()
            .filter(|s| s.confidence.value() < best.confidence.value())
            .map(|s| s.source.clone())
            .collect();

        ResolutionResult {
            status: best.clone(),
            had_conflict,
            confidence_penalty: if had_conflict { max_diff * 0.1 } else { 0.0 },
            overridden_sources: overridden,
            explanation: format!(
                "Selected highest confidence source ({:.2})",
                best.confidence.value()
            ),
        }
    }

    /// Resolve using Bayesian combination
    fn resolve_bayesian(
        &self,
        statuses: &[EpistemicStatus],
        had_conflict: bool,
        max_diff: f64,
    ) -> ResolutionResult {
        // Combine confidence using log-odds
        let mut log_odds_sum = 0.0;

        for status in statuses {
            let p = status.confidence.value().clamp(0.001, 0.999);
            let log_odds = (p / (1.0 - p)).ln();
            log_odds_sum += log_odds;
        }

        let avg_log_odds = log_odds_sum / statuses.len() as f64;
        let combined_p = 1.0 / (1.0 + (-avg_log_odds).exp());
        let combined_confidence = combined_p.clamp(0.0, 1.0);

        // Apply penalty for heterogeneity
        let penalty = if had_conflict { max_diff * 0.15 } else { 0.0 };
        let final_confidence = (combined_confidence - penalty).clamp(0.0, 1.0);

        // Combine evidence
        let combined_evidence: Vec<Evidence> = if self.config.preserve_evidence {
            statuses.iter().flat_map(|s| s.evidence.clone()).collect()
        } else {
            // Keep only strongest evidence
            let max_strength = statuses
                .iter()
                .flat_map(|s| s.evidence.iter())
                .map(|e| e.strength.value())
                .fold(0.0, f64::max);

            statuses
                .iter()
                .flat_map(|s| s.evidence.clone())
                .filter(|e| e.strength.value() >= max_strength * 0.9)
                .collect()
        };

        ResolutionResult {
            status: EpistemicStatus {
                confidence: Confidence::new(final_confidence),
                revisability: self.combine_revisability(statuses),
                source: Source::Derivation("bayesian_combination".into()),
                evidence: combined_evidence,
            },
            had_conflict,
            confidence_penalty: penalty,
            overridden_sources: vec![],
            explanation: format!(
                "Bayesian combination of {} sources: {:.2}",
                statuses.len(),
                final_confidence
            ),
        }
    }

    /// Resolve using AGM belief revision
    fn resolve_agm(
        &self,
        statuses: &[EpistemicStatus],
        had_conflict: bool,
        max_diff: f64,
    ) -> ResolutionResult {
        // AGM revision: newer information takes precedence
        // For now, use the last status as the "newest"
        // In production, would use timestamps

        let revised = statuses.last().unwrap();

        // Apply epistemic entrenchment - more entrenched beliefs resist revision
        // Higher confidence = more entrenched
        let entrenchment_penalty = if had_conflict {
            let prior_confidence: f64 = statuses
                .iter()
                .take(statuses.len() - 1)
                .map(|s| s.confidence.value())
                .sum::<f64>()
                / (statuses.len() - 1) as f64;

            if prior_confidence > revised.confidence.value() {
                // Prior beliefs were more entrenched
                (prior_confidence - revised.confidence.value()) * 0.2
            } else {
                0.0
            }
        } else {
            0.0
        };

        let final_confidence = (revised.confidence.value() - entrenchment_penalty).clamp(0.0, 1.0);

        ResolutionResult {
            status: EpistemicStatus {
                confidence: Confidence::new(final_confidence),
                revisability: revised.revisability.clone(),
                source: revised.source.clone(),
                evidence: if self.config.preserve_evidence {
                    statuses.iter().flat_map(|s| s.evidence.clone()).collect()
                } else {
                    revised.evidence.clone()
                },
            },
            had_conflict,
            confidence_penalty: entrenchment_penalty,
            overridden_sources: statuses
                .iter()
                .take(statuses.len() - 1)
                .map(|s| s.source.clone())
                .collect(),
            explanation: format!(
                "AGM revision: newer belief ({:.2}) with entrenchment penalty {:.2}",
                revised.confidence.value(),
                entrenchment_penalty
            ),
        }
    }

    /// Resolve requiring consensus
    fn resolve_consensus(
        &self,
        statuses: &[EpistemicStatus],
        threshold: u8,
        had_conflict: bool,
        max_diff: f64,
    ) -> ResolutionResult {
        let threshold_f = threshold as f64 / 100.0;

        // Calculate mean confidence
        let mean: f64 =
            statuses.iter().map(|s| s.confidence.value()).sum::<f64>() / statuses.len() as f64;

        // Calculate agreement ratio (how many are within threshold of mean)
        let agreeing = statuses
            .iter()
            .filter(|s| (s.confidence.value() - mean).abs() <= threshold_f)
            .count();

        let agreement_ratio = agreeing as f64 / statuses.len() as f64;

        // Final confidence is mean * agreement ratio
        let final_confidence = mean * agreement_ratio;

        ResolutionResult {
            status: EpistemicStatus {
                confidence: Confidence::new(final_confidence),
                revisability: self.combine_revisability(statuses),
                source: Source::Derivation("consensus".into()),
                evidence: statuses.iter().flat_map(|s| s.evidence.clone()).collect(),
            },
            had_conflict,
            confidence_penalty: 1.0 - agreement_ratio,
            overridden_sources: vec![],
            explanation: format!(
                "Consensus: {:.0}% agreement, mean={:.2}, final={:.2}",
                agreement_ratio * 100.0,
                mean,
                final_confidence
            ),
        }
    }

    /// Conservative resolution (minimum confidence)
    fn resolve_conservative(
        &self,
        statuses: &[EpistemicStatus],
        had_conflict: bool,
        _max_diff: f64,
    ) -> ResolutionResult {
        let min = statuses
            .iter()
            .min_by(|a, b| {
                a.confidence
                    .value()
                    .partial_cmp(&b.confidence.value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        ResolutionResult {
            status: EpistemicStatus {
                confidence: min.confidence,
                revisability: self.combine_revisability(statuses),
                source: Source::Derivation("conservative".into()),
                evidence: statuses.iter().flat_map(|s| s.evidence.clone()).collect(),
            },
            had_conflict,
            confidence_penalty: 0.0,
            overridden_sources: vec![],
            explanation: format!(
                "Conservative: using minimum confidence {:.2}",
                min.confidence.value()
            ),
        }
    }

    /// Optimistic resolution (maximum confidence)
    fn resolve_optimistic(
        &self,
        statuses: &[EpistemicStatus],
        had_conflict: bool,
        _max_diff: f64,
    ) -> ResolutionResult {
        let max = statuses
            .iter()
            .max_by(|a, b| {
                a.confidence
                    .value()
                    .partial_cmp(&b.confidence.value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        ResolutionResult {
            status: EpistemicStatus {
                confidence: max.confidence,
                revisability: self.combine_revisability(statuses),
                source: Source::Derivation("optimistic".into()),
                evidence: statuses.iter().flat_map(|s| s.evidence.clone()).collect(),
            },
            had_conflict,
            confidence_penalty: 0.0,
            overridden_sources: vec![],
            explanation: format!(
                "Optimistic: using maximum confidence {:.2}",
                max.confidence.value()
            ),
        }
    }

    /// Weighted average resolution
    fn resolve_weighted_average(
        &self,
        statuses: &[EpistemicStatus],
        had_conflict: bool,
        max_diff: f64,
    ) -> ResolutionResult {
        // Weight by evidence strength
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for status in statuses {
            let weight = status
                .evidence
                .iter()
                .map(|e| e.strength.value())
                .sum::<f64>()
                .max(0.1); // minimum weight

            weighted_sum += status.confidence.value() * weight;
            weight_total += weight;
        }

        let weighted_avg = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            statuses.iter().map(|s| s.confidence.value()).sum::<f64>() / statuses.len() as f64
        };

        ResolutionResult {
            status: EpistemicStatus {
                confidence: Confidence::new(weighted_avg),
                revisability: self.combine_revisability(statuses),
                source: Source::Derivation("weighted_average".into()),
                evidence: statuses.iter().flat_map(|s| s.evidence.clone()).collect(),
            },
            had_conflict,
            confidence_penalty: if had_conflict { max_diff * 0.05 } else { 0.0 },
            overridden_sources: vec![],
            explanation: format!("Weighted average: {:.2}", weighted_avg),
        }
    }

    /// Combine revisability conditions
    fn combine_revisability(&self, statuses: &[EpistemicStatus]) -> Revisability {
        let all_conditions: Vec<String> = statuses
            .iter()
            .filter_map(|s| {
                if let Revisability::Revisable { conditions } = &s.revisability {
                    Some(conditions.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        if all_conditions.is_empty() {
            // Check if any are non-revisable
            if statuses
                .iter()
                .any(|s| s.revisability == Revisability::NonRevisable)
            {
                Revisability::NonRevisable
            } else {
                Revisability::Revisable { conditions: vec![] }
            }
        } else {
            Revisability::Revisable {
                conditions: all_conditions,
            }
        }
    }

    /// Compute heterogeneity measure between statuses
    pub fn heterogeneity_measure(&self, statuses: &[EpistemicStatus]) -> f64 {
        if statuses.len() < 2 {
            return 0.0;
        }

        let n = statuses.len() as f64;

        // Confidence variance
        let mean_conf: f64 = statuses.iter().map(|s| s.confidence.value()).sum::<f64>() / n;
        let var_conf: f64 = statuses
            .iter()
            .map(|s| (s.confidence.value() - mean_conf).powi(2))
            .sum::<f64>()
            / n;

        // Source diversity (number of unique source types)
        let source_types: std::collections::HashSet<_> = statuses
            .iter()
            .map(|s| std::mem::discriminant(&s.source))
            .collect();
        let source_diversity = source_types.len() as f64 / n;

        // Combine measures
        (var_conf.sqrt() + source_diversity) / 2.0
    }
}

impl Default for HeterogeneityResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_status(confidence: f64) -> EpistemicStatus {
        EpistemicStatus {
            confidence: Confidence::new(confidence),
            ..Default::default()
        }
    }

    #[test]
    fn test_resolve_single() {
        let resolver = HeterogeneityResolver::new();
        let statuses = vec![test_status(0.9)];
        let result = resolver.resolve(&statuses);

        assert!(!result.had_conflict);
        assert_eq!(result.status.confidence.value(), 0.9);
    }

    #[test]
    fn test_resolve_prioritized() {
        let resolver = HeterogeneityResolver::with_strategy(ResolutionStrategy::Prioritized);
        let statuses = vec![test_status(0.7), test_status(0.9), test_status(0.5)];
        let result = resolver.resolve(&statuses);

        assert!(result.status.confidence.value() >= 0.85); // Should pick 0.9
    }

    #[test]
    fn test_resolve_bayesian() {
        let resolver = HeterogeneityResolver::with_strategy(ResolutionStrategy::Bayesian);
        let statuses = vec![test_status(0.8), test_status(0.7)];
        let result = resolver.resolve(&statuses);

        // Bayesian should give something between 0.7 and 0.8
        assert!(result.status.confidence.value() > 0.7);
        assert!(result.status.confidence.value() < 0.8);
    }

    #[test]
    fn test_resolve_conservative() {
        let resolver = HeterogeneityResolver::with_strategy(ResolutionStrategy::Conservative);
        let statuses = vec![test_status(0.9), test_status(0.6)];
        let result = resolver.resolve(&statuses);

        assert_eq!(result.status.confidence.value(), 0.6);
    }

    #[test]
    fn test_resolve_optimistic() {
        let resolver = HeterogeneityResolver::with_strategy(ResolutionStrategy::Optimistic);
        let statuses = vec![test_status(0.9), test_status(0.6)];
        let result = resolver.resolve(&statuses);

        assert_eq!(result.status.confidence.value(), 0.9);
    }

    #[test]
    fn test_conflict_detection() {
        let resolver = HeterogeneityResolver::new();

        // No conflict
        let similar = vec![test_status(0.8), test_status(0.85)];
        let result = resolver.resolve(&similar);
        assert!(!result.had_conflict);

        // Conflict
        let different = vec![test_status(0.9), test_status(0.5)];
        let result = resolver.resolve(&different);
        assert!(result.had_conflict);
    }

    #[test]
    fn test_heterogeneity_measure() {
        let resolver = HeterogeneityResolver::new();

        // Homogeneous
        let homogeneous = vec![test_status(0.8), test_status(0.8), test_status(0.8)];
        let h1 = resolver.heterogeneity_measure(&homogeneous);

        // Heterogeneous
        let heterogeneous = vec![test_status(0.9), test_status(0.5), test_status(0.7)];
        let h2 = resolver.heterogeneity_measure(&heterogeneous);

        assert!(h2 > h1);
    }
}

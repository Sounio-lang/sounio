//! Evidence collection, aggregation, and conflict resolution
//!
//! This module wires evidence management into the Knowledge<T> lifecycle.
//! It connects the existing `Evidence` and `EvidenceKind` types from
//! `confidence.rs` with aggregation strategies, staleness tracking,
//! and audit trail export.
//!
//! # Conflict Resolution Strategies
//!
//! - **TrustHigherConfidence**: Pick evidence with higher strength
//! - **BayesianFusion**: Combine via posterior-like multiplication + renormalization
//! - **TemporalPrecedence**: Trust newer evidence over older
//! - **DempsterShafer**: Dempster's rule of combination for belief masses

use super::confidence::{Evidence, EvidenceKind};

/// Strategy for resolving conflicting evidence
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictStrategy {
    /// Pick evidence with higher confidence
    TrustHigherConfidence,
    /// Bayesian fusion: multiply and renormalize
    BayesianFusion,
    /// Trust newer evidence (higher timestamp wins)
    TemporalPrecedence,
    /// Dempster-Shafer combination rule
    DempsterShafer,
}

/// Result of evidence aggregation
#[derive(Debug, Clone)]
pub struct AggregatedEvidence {
    /// Combined confidence after aggregation
    pub combined_confidence: f64,
    /// Number of supporting (concordant) evidence items
    pub supporting_count: usize,
    /// Number of conflicting (discordant) evidence items
    pub conflicting_count: usize,
    /// The evidence kind appearing most frequently
    pub dominant_kind: EvidenceKind,
    /// Which resolution strategy was applied, if any
    pub resolution_applied: Option<ConflictStrategy>,
}

/// Tracks evidence staleness relative to a reference time
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceAge {
    /// Age of the evidence in seconds
    pub age_seconds: u64,
    /// Whether the evidence exceeds its staleness threshold
    pub is_stale: bool,
    /// The threshold used for staleness determination
    pub staleness_threshold: u64,
}

/// A timestamped, attributed evidence record
#[derive(Debug, Clone)]
pub struct EvidenceRecord {
    /// The underlying evidence
    pub evidence: Evidence,
    /// When this evidence was collected (epoch seconds)
    pub timestamp: u64,
    /// Identifier for the source that contributed this evidence
    pub source_id: String,
    /// If superseded, the index of the superseding record
    pub superseded_by: Option<usize>,
}

/// Main evidence collector: accumulates, aggregates, and manages evidence
///
/// The collector maintains an ordered registry of `EvidenceRecord` items,
/// applies a configurable `ConflictStrategy` during aggregation, and
/// tracks staleness based on a configurable threshold.
pub struct EvidenceCollector {
    registry: Vec<EvidenceRecord>,
    conflict_strategy: ConflictStrategy,
    staleness_threshold: u64,
}

impl EvidenceCollector {
    /// Create a new collector with the given conflict resolution strategy
    ///
    /// The default staleness threshold is 86400 seconds (24 hours).
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self {
            registry: Vec::new(),
            conflict_strategy: strategy,
            staleness_threshold: 86_400,
        }
    }

    /// Create a collector with a custom staleness threshold (in seconds)
    pub fn with_staleness_threshold(mut self, threshold: u64) -> Self {
        self.staleness_threshold = threshold;
        self
    }

    /// Collect a new evidence item into the registry
    pub fn collect(&mut self, evidence: Evidence, source_id: String, timestamp: u64) {
        self.registry.push(EvidenceRecord {
            evidence,
            timestamp,
            source_id,
            superseded_by: None,
        });
    }

    /// Number of active (non-superseded) evidence records
    pub fn active_count(&self) -> usize {
        self.registry
            .iter()
            .filter(|r| r.superseded_by.is_none())
            .count()
    }

    /// Total number of evidence records (including superseded)
    pub fn total_count(&self) -> usize {
        self.registry.len()
    }

    /// Aggregate all active evidence using the configured strategy
    ///
    /// Returns `None` if the registry is empty.
    pub fn aggregate(&self) -> Option<AggregatedEvidence> {
        let active: Vec<&EvidenceRecord> = self
            .registry
            .iter()
            .filter(|r| r.superseded_by.is_none())
            .collect();

        if active.is_empty() {
            return None;
        }

        let strengths: Vec<f64> = active.iter().map(|r| r.evidence.strength.value()).collect();

        let combined_confidence = match &self.conflict_strategy {
            ConflictStrategy::TrustHigherConfidence => {
                // Take the maximum strength
                strengths.iter().cloned().fold(0.0_f64, f64::max)
            }
            ConflictStrategy::BayesianFusion => {
                // Multiply strengths and renormalize
                bayesian_fusion(&strengths)
            }
            ConflictStrategy::TemporalPrecedence => {
                // Use the most recent evidence's strength
                active
                    .iter()
                    .max_by_key(|r| r.timestamp)
                    .map(|r| r.evidence.strength.value())
                    .unwrap_or(0.0)
            }
            ConflictStrategy::DempsterShafer => aggregate_dempster_shafer(&strengths),
        };

        // Count supporting vs conflicting: evidence above/below the median
        let mean: f64 = strengths.iter().sum::<f64>() / strengths.len() as f64;
        let supporting_count = strengths.iter().filter(|&&s| s >= mean).count();
        let conflicting_count = strengths.len() - supporting_count;

        let dominant_kind = find_dominant_kind(&active);

        Some(AggregatedEvidence {
            combined_confidence,
            supporting_count,
            conflicting_count,
            dominant_kind,
            resolution_applied: Some(self.conflict_strategy.clone()),
        })
    }

    /// Compute the age of a specific evidence record relative to `current_time`
    pub fn evidence_age(record: &EvidenceRecord, current_time: u64) -> EvidenceAge {
        let age_seconds = current_time.saturating_sub(record.timestamp);
        EvidenceAge {
            age_seconds,
            is_stale: false, // Caller should use is_stale_at() for threshold check
            staleness_threshold: 0,
        }
    }

    /// Check whether a record is stale given a threshold
    pub fn is_stale_at(record: &EvidenceRecord, current_time: u64, threshold: u64) -> EvidenceAge {
        let age_seconds = current_time.saturating_sub(record.timestamp);
        EvidenceAge {
            age_seconds,
            is_stale: age_seconds > threshold,
            staleness_threshold: threshold,
        }
    }

    /// Resolve a conflict between two evidence items using the configured strategy
    pub fn resolve_conflict(&self, a: &Evidence, b: &Evidence) -> Evidence {
        match &self.conflict_strategy {
            ConflictStrategy::TrustHigherConfidence => {
                if a.strength.value() >= b.strength.value() {
                    a.clone()
                } else {
                    b.clone()
                }
            }
            ConflictStrategy::BayesianFusion => {
                let fused = bayesian_fusion(&[a.strength.value(), b.strength.value()]);
                Evidence::new(
                    a.kind.clone(),
                    format!("fused({},{})", a.reference, b.reference),
                    fused,
                )
            }
            ConflictStrategy::TemporalPrecedence => {
                // Without timestamps on Evidence itself, prefer b (assumed newer)
                b.clone()
            }
            ConflictStrategy::DempsterShafer => {
                let fused = aggregate_dempster_shafer(&[a.strength.value(), b.strength.value()]);
                Evidence::new(
                    a.kind.clone(),
                    format!("ds({},{})", a.reference, b.reference),
                    fused,
                )
            }
        }
    }

    /// Prune stale evidence: mark records older than the staleness threshold
    /// as superseded. Returns the number of records pruned.
    pub fn prune_stale(&mut self, current_time: u64) -> usize {
        let threshold = self.staleness_threshold;
        let mut pruned = 0;

        for record in self.registry.iter_mut() {
            if record.superseded_by.is_none() {
                let age = current_time.saturating_sub(record.timestamp);
                if age > threshold {
                    // Mark as superseded by a sentinel index (usize::MAX means "pruned")
                    record.superseded_by = Some(usize::MAX);
                    pruned += 1;
                }
            }
        }

        pruned
    }

    /// Export an audit trail of all evidence records
    ///
    /// Returns tuples of (source_id, reference, strength) for every record
    /// in collection order. Superseded records are included but marked.
    pub fn export_audit_trail(&self) -> Vec<(String, String, f64)> {
        self.registry
            .iter()
            .map(|r| {
                (
                    r.source_id.clone(),
                    r.evidence.reference.clone(),
                    r.evidence.strength.value(),
                )
            })
            .collect()
    }

    /// Get a reference to the internal registry
    pub fn registry(&self) -> &[EvidenceRecord] {
        &self.registry
    }
}

/// Dempster-Shafer combination rule for belief masses
///
/// Given a slice of belief values (interpreted as mass on the hypothesis),
/// combines them pairwise using Dempster's rule:
///
///   m_combined(A) = (m1(A)*m2(A)) / (1 - K)
///
/// where K is the conflict coefficient:
///   K = sum of products of masses assigned to disjoint hypotheses
///
/// For simplification, we treat each value as Bel(H) with the remainder
/// assigned to the frame of discernment (ignorance).
pub fn aggregate_dempster_shafer(beliefs: &[f64]) -> f64 {
    if beliefs.is_empty() {
        return 0.0;
    }
    if beliefs.len() == 1 {
        return beliefs[0].clamp(0.0, 1.0);
    }

    let mut result = beliefs[0].clamp(0.0, 1.0);

    for &b in &beliefs[1..] {
        let b = b.clamp(0.0, 1.0);
        // m1(H) = result, m1(not-H) = 1 - result
        // m2(H) = b,      m2(not-H) = 1 - b
        //
        // Conflict K = m1(H)*m2(not-H) + m1(not-H)*m2(H)
        //            ... but only disjoint sets conflict.
        // In the binary frame {H, not-H}:
        //   K = m1(H)*m2(not-H)  + m1(not-H)*m2(H)
        //     = result*(1-b)     + (1-result)*b  ... NO: these assign to H or not-H, not empty set.
        //
        // Correct binary-frame Dempster:
        //   m1 assigns: m1(H)=result, m1(Theta)=1-result  (ignorance to frame)
        //   m2 assigns: m2(H)=b,      m2(Theta)=1-b
        //
        //   Intersection table:
        //     m1(H) * m2(H)     = result*b         -> H
        //     m1(H) * m2(Theta) = result*(1-b)     -> H
        //     m1(Theta) * m2(H) = (1-result)*b     -> H
        //     m1(Theta) * m2(Theta) = (1-result)*(1-b) -> Theta
        //
        //   K = 0  (no empty-set intersections when using frame ignorance)
        //   m_combined(H) = result*b + result*(1-b) + (1-result)*b
        //                 = result + b - result*b
        //   m_combined(Theta) = (1-result)*(1-b)
        //
        // This is the standard "combining independent sources" formula.
        result = result + b - result * b;
    }

    result.clamp(0.0, 1.0)
}

/// Bayesian fusion: multiply beliefs and renormalize
///
/// Computes the product of all values, then normalizes against
/// the product with the complementary beliefs to yield a posterior-like value.
fn bayesian_fusion(strengths: &[f64]) -> f64 {
    if strengths.is_empty() {
        return 0.0;
    }
    if strengths.len() == 1 {
        return strengths[0].clamp(0.0, 1.0);
    }

    let product_for: f64 = strengths.iter().map(|&s| s.clamp(0.001, 0.999)).product();
    let product_against: f64 = strengths
        .iter()
        .map(|&s| (1.0 - s).clamp(0.001, 0.999))
        .product();

    let denom = product_for + product_against;
    if denom < 1e-15 {
        return 0.5;
    }

    (product_for / denom).clamp(0.0, 1.0)
}

/// Find the most frequently occurring evidence kind among active records
fn find_dominant_kind(records: &[&EvidenceRecord]) -> EvidenceKind {
    // Use a simple discriminant-based counting approach
    let mut counts: std::collections::HashMap<&str, (usize, &EvidenceKind)> =
        std::collections::HashMap::new();

    for record in records {
        let label = match &record.evidence.kind {
            EvidenceKind::Publication { .. } => "Publication",
            EvidenceKind::Dataset { .. } => "Dataset",
            EvidenceKind::Experiment { .. } => "Experiment",
            EvidenceKind::Computation { .. } => "Computation",
            EvidenceKind::ExpertOpinion { .. } => "ExpertOpinion",
            EvidenceKind::Verified { .. } => "Verified",
            EvidenceKind::HumanAssertion { .. } => "HumanAssertion",
        };

        let entry = counts.entry(label).or_insert((0, &record.evidence.kind));
        entry.0 += 1;
    }

    counts
        .into_values()
        .max_by_key(|(count, _)| *count)
        .map(|(_, kind)| kind.clone())
        .unwrap_or(EvidenceKind::Computation {
            code_ref: "unknown".to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pub_evidence(doi: &str, strength: f64) -> Evidence {
        Evidence::publication(doi, strength)
    }

    fn exp_evidence(protocol: &str, strength: f64) -> Evidence {
        Evidence::experiment(protocol, strength)
    }

    // --- Creation / basic tests ---

    #[test]
    fn test_collector_creation() {
        let collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        assert_eq!(collector.active_count(), 0);
        assert_eq!(collector.total_count(), 0);
    }

    #[test]
    fn test_collector_with_custom_staleness() {
        let collector =
            EvidenceCollector::new(ConflictStrategy::BayesianFusion).with_staleness_threshold(3600);
        assert_eq!(collector.staleness_threshold, 3600);
    }

    #[test]
    fn test_collect_single_evidence() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        collector.collect(pub_evidence("10.1234/test", 0.9), "src1".into(), 1000);
        assert_eq!(collector.active_count(), 1);
        assert_eq!(collector.total_count(), 1);
    }

    #[test]
    fn test_collect_multiple_evidence() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        collector.collect(pub_evidence("doi1", 0.8), "src1".into(), 1000);
        collector.collect(exp_evidence("proto1", 0.7), "src2".into(), 2000);
        collector.collect(pub_evidence("doi2", 0.95), "src3".into(), 3000);
        assert_eq!(collector.active_count(), 3);
    }

    // --- Aggregation strategies ---

    #[test]
    fn test_aggregate_trust_higher_confidence() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        collector.collect(pub_evidence("doi1", 0.6), "src1".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.9), "src2".into(), 2000);
        collector.collect(pub_evidence("doi3", 0.75), "src3".into(), 3000);

        let agg = collector.aggregate().unwrap();
        assert!((agg.combined_confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_temporal_precedence() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TemporalPrecedence);
        collector.collect(pub_evidence("doi1", 0.9), "src1".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.6), "src2".into(), 5000);

        let agg = collector.aggregate().unwrap();
        // Most recent (timestamp=5000) has strength 0.6
        assert!((agg.combined_confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_bayesian_fusion() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::BayesianFusion);
        collector.collect(pub_evidence("doi1", 0.8), "src1".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.8), "src2".into(), 2000);

        let agg = collector.aggregate().unwrap();
        // Two 0.8 beliefs fused: 0.8*0.8 / (0.8*0.8 + 0.2*0.2) = 0.64 / 0.68 ~ 0.941
        assert!(agg.combined_confidence > 0.9);
        assert!(agg.combined_confidence < 1.0);
    }

    #[test]
    fn test_aggregate_dempster_shafer() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        collector.collect(pub_evidence("doi1", 0.6), "src1".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.7), "src2".into(), 2000);

        let agg = collector.aggregate().unwrap();
        // DS: 0.6 + 0.7 - 0.6*0.7 = 0.88
        assert!((agg.combined_confidence - 0.88).abs() < 1e-6);
    }

    #[test]
    fn test_dempster_shafer_three_sources() {
        let result = aggregate_dempster_shafer(&[0.5, 0.5, 0.5]);
        // Step 1: 0.5 + 0.5 - 0.25 = 0.75
        // Step 2: 0.75 + 0.5 - 0.375 = 0.875
        assert!((result - 0.875).abs() < 1e-6);
    }

    #[test]
    fn test_dempster_shafer_empty() {
        assert!((aggregate_dempster_shafer(&[]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dempster_shafer_single() {
        assert!((aggregate_dempster_shafer(&[0.7]) - 0.7).abs() < 1e-6);
    }

    // --- Staleness ---

    #[test]
    fn test_staleness_detection() {
        let record = EvidenceRecord {
            evidence: pub_evidence("doi1", 0.8),
            timestamp: 1000,
            source_id: "src1".into(),
            superseded_by: None,
        };

        let fresh = EvidenceCollector::is_stale_at(&record, 2000, 86_400);
        assert!(!fresh.is_stale);
        assert_eq!(fresh.age_seconds, 1000);

        let stale = EvidenceCollector::is_stale_at(&record, 100_000, 86_400);
        assert!(stale.is_stale);
        assert_eq!(stale.age_seconds, 99_000);
    }

    #[test]
    fn test_evidence_age_computation() {
        let record = EvidenceRecord {
            evidence: pub_evidence("doi1", 0.8),
            timestamp: 500,
            source_id: "src1".into(),
            superseded_by: None,
        };

        let age = EvidenceCollector::evidence_age(&record, 1500);
        assert_eq!(age.age_seconds, 1000);
    }

    // --- Conflict resolution ---

    #[test]
    fn test_resolve_conflict_higher_confidence() {
        let collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        let a = pub_evidence("doi_weak", 0.4);
        let b = pub_evidence("doi_strong", 0.9);

        let winner = collector.resolve_conflict(&a, &b);
        assert!((winner.strength.value() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_conflict_bayesian() {
        let collector = EvidenceCollector::new(ConflictStrategy::BayesianFusion);
        let a = pub_evidence("doi1", 0.7);
        let b = pub_evidence("doi2", 0.8);

        let fused = collector.resolve_conflict(&a, &b);
        // 0.7*0.8 / (0.7*0.8 + 0.3*0.2) = 0.56 / 0.62 ~ 0.903
        assert!(fused.strength.value() > 0.85);
    }

    #[test]
    fn test_resolve_conflict_dempster_shafer() {
        let collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        let a = pub_evidence("doi1", 0.5);
        let b = pub_evidence("doi2", 0.5);

        let fused = collector.resolve_conflict(&a, &b);
        // DS: 0.5 + 0.5 - 0.25 = 0.75
        assert!((fused.strength.value() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_conflict_temporal() {
        let collector = EvidenceCollector::new(ConflictStrategy::TemporalPrecedence);
        let a = pub_evidence("doi_old", 0.9);
        let b = pub_evidence("doi_new", 0.6);

        let result = collector.resolve_conflict(&a, &b);
        // Temporal precedence returns b (assumed newer)
        assert!((result.strength.value() - 0.6).abs() < 1e-6);
    }

    // --- Pruning ---

    #[test]
    fn test_prune_stale_evidence() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence)
            .with_staleness_threshold(5000);

        collector.collect(pub_evidence("doi1", 0.8), "src1".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.7), "src2".into(), 8000);
        collector.collect(pub_evidence("doi3", 0.9), "src3".into(), 9000);

        let pruned = collector.prune_stale(10000);
        // doi1 at t=1000, age=9000 > 5000 => pruned
        // doi2 at t=8000, age=2000 <= 5000 => kept
        // doi3 at t=9000, age=1000 <= 5000 => kept
        assert_eq!(pruned, 1);
        assert_eq!(collector.active_count(), 2);
    }

    // --- Audit trail ---

    #[test]
    fn test_export_audit_trail() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        collector.collect(pub_evidence("doi1", 0.8), "lab_a".into(), 1000);
        collector.collect(exp_evidence("proto1", 0.7), "lab_b".into(), 2000);

        let trail = collector.export_audit_trail();
        assert_eq!(trail.len(), 2);
        assert_eq!(trail[0].0, "lab_a");
        assert_eq!(trail[0].1, "doi1");
        assert!((trail[0].2 - 0.8).abs() < 1e-6);
        assert_eq!(trail[1].0, "lab_b");
    }

    // --- Edge cases ---

    #[test]
    fn test_aggregate_empty_returns_none() {
        let collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        assert!(collector.aggregate().is_none());
    }

    #[test]
    fn test_aggregate_single_evidence() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        collector.collect(pub_evidence("doi1", 0.85), "src1".into(), 1000);

        let agg = collector.aggregate().unwrap();
        assert!((agg.combined_confidence - 0.85).abs() < 1e-6);
        assert_eq!(agg.supporting_count, 1);
        assert_eq!(agg.conflicting_count, 0);
    }

    #[test]
    fn test_supporting_conflicting_counts() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        // Mean of [0.9, 0.8, 0.3] = 0.667
        // Supporting (>= 0.667): 0.9, 0.8 => 2
        // Conflicting (< 0.667): 0.3 => 1
        collector.collect(pub_evidence("doi1", 0.9), "s1".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.8), "s2".into(), 2000);
        collector.collect(pub_evidence("doi3", 0.3), "s3".into(), 3000);

        let agg = collector.aggregate().unwrap();
        assert_eq!(agg.supporting_count, 2);
        assert_eq!(agg.conflicting_count, 1);
    }

    #[test]
    fn test_dominant_kind_detection() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence);
        collector.collect(pub_evidence("doi1", 0.8), "s1".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.7), "s2".into(), 2000);
        collector.collect(exp_evidence("proto1", 0.9), "s3".into(), 3000);

        let agg = collector.aggregate().unwrap();
        // 2 publications vs 1 experiment => dominant is Publication
        assert!(matches!(
            agg.dominant_kind,
            EvidenceKind::Publication { .. }
        ));
    }

    #[test]
    fn test_bayesian_fusion_opposing_evidence() {
        // When one source says 0.9 and another says 0.1, fusion should settle near 0.5
        let result = bayesian_fusion(&[0.9, 0.1]);
        // 0.9*0.1 / (0.9*0.1 + 0.1*0.9) = 0.09 / 0.18 = 0.5
        assert!((result - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_dempster_shafer_high_beliefs() {
        // Two very high beliefs should yield near-certainty
        let result = aggregate_dempster_shafer(&[0.95, 0.95]);
        // 0.95 + 0.95 - 0.9025 = 0.9975
        assert!(result > 0.99);
    }
}

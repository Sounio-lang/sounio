//! Causal Composition: Combining Causal Knowledge
//!
//! Extends Day 32 composition operators (⊗, ⊔, |) with causal awareness:
//! - Causal tensor: combines causal structures
//! - Causal join: merges knowledge with same causal structure
//! - Graph compatibility checking

use super::graph::GraphError;
use super::identification::IdentificationStatus;
use super::knowledge::{CausalError, CausalKnowledge};
use crate::epistemic::composition::ConfidenceValue;
use crate::temporal::TemporalComposition;

/// Result of causal join operation
#[derive(Clone, Debug)]
pub enum CausalJoinResult<T> {
    /// Knowledge agrees (low conflict)
    Concordant(CausalKnowledge<T>),

    /// Knowledge disagrees but was resolved
    Resolved {
        result: CausalKnowledge<T>,
        conflict_level: f64,
    },

    /// Knowledge is irreconcilable
    Irreconcilable {
        k1: CausalKnowledge<T>,
        k2: CausalKnowledge<T>,
        conflict_level: f64,
        reason: String,
    },
}

impl<T: Clone> CausalKnowledge<T> {
    /// Causal tensor: combine two causal knowledge with independent causal structures
    ///
    /// Requirements:
    /// - Graphs must be compatible (no conflicting edges)
    /// - Merged graph must still be a DAG
    pub fn tensor_causal<B: Clone>(
        self,
        other: CausalKnowledge<B>,
    ) -> Result<CausalKnowledge<(T, B)>, CausalError> {
        // Check compatibility
        if !self.graph.compatible_with(&other.graph) {
            return Err(CausalError::GraphError(GraphError::InvalidOperation(
                "Incompatible causal structures: conflicting edges".to_string(),
            )));
        }

        // Merge graphs
        let merged_graph = self.graph.merge(&other.graph);

        // Merge base temporal knowledge
        let merged_base = self.base.tensor_temporal(other.base);

        // Combine treatments and outcome
        let combined_outcome = format!("({}, {})", self.outcome, other.outcome);
        let combined_treatments = [self.treatments, other.treatments].concat();

        Ok(CausalKnowledge {
            base: merged_base,
            graph: merged_graph,
            outcome: combined_outcome,
            treatments: combined_treatments,
            identification: IdentificationStatus::Unknown,
        })
    }

    /// Causal join: merge two causal knowledge about the same phenomenon
    ///
    /// Requirements:
    /// - Graphs must be structurally equivalent
    /// - Base knowledge must be fusible
    pub fn join_causal(
        self,
        other: CausalKnowledge<T>,
        conflict_threshold: f64,
    ) -> CausalJoinResult<T>
    where
        T: PartialEq + Clone,
    {
        // Check structural equivalence
        if !self.graph.structurally_equivalent(&other.graph) {
            return CausalJoinResult::Irreconcilable {
                k1: self.clone(),
                k2: other.clone(),
                conflict_level: 1.0,
                reason: "Different causal structures cannot be joined".to_string(),
            };
        }

        // Check treatment/outcome match
        if self.outcome != other.outcome || self.treatments != other.treatments {
            return CausalJoinResult::Irreconcilable {
                k1: self.clone(),
                k2: other.clone(),
                conflict_level: 1.0,
                reason: "Different treatment/outcome specifications".to_string(),
            };
        }

        // Compute conflict between values
        let conflict = self.compute_conflict(&other);

        if conflict < 0.05 {
            // Concordant - boost confidence
            let merged = self.merge_concordant(other);
            CausalJoinResult::Concordant(merged)
        } else if conflict < conflict_threshold {
            // Resolvable - weighted merge
            let merged = self.merge_resolved(other, conflict);
            CausalJoinResult::Resolved {
                result: merged,
                conflict_level: conflict,
            }
        } else {
            // Irreconcilable
            CausalJoinResult::Irreconcilable {
                k1: self,
                k2: other,
                conflict_level: conflict,
                reason: format!(
                    "Conflict level {} exceeds threshold {}",
                    conflict, conflict_threshold
                ),
            }
        }
    }

    /// Compute conflict between two causal knowledge
    fn compute_conflict(&self, other: &CausalKnowledge<T>) -> f64
    where
        T: PartialEq,
    {
        // Base conflict from values
        let values_match = self.base.core.value() == other.base.core.value();

        // Confidence difference
        let conf_diff = (self.confidence().value() - other.confidence().value()).abs();

        // Identification agreement
        // Both unknown is fine, both identified with same method is fine
        let id_match = matches!(
            (&self.identification, &other.identification),
            (IdentificationStatus::Unknown, IdentificationStatus::Unknown)
                | (
                    IdentificationStatus::Identified { .. },
                    IdentificationStatus::Identified { .. }
                )
        );

        let base_conflict = if values_match { 0.0 } else { 0.5 };
        let conf_conflict = conf_diff * 0.3;
        let id_conflict = if id_match { 0.0 } else { 0.2 };

        base_conflict + conf_conflict + id_conflict
    }

    /// Merge concordant knowledge (boost confidence)
    fn merge_concordant(self, other: CausalKnowledge<T>) -> CausalKnowledge<T> {
        let conf1 = self.confidence().value();
        let conf2 = other.confidence().value();

        // Dempster-Shafer combination
        let boosted = conf1 + conf2 - conf1 * conf2;
        let new_conf = ConfidenceValue::new(boosted.clamp(0.0, 1.0))
            .unwrap_or(ConfidenceValue::new(conf1).unwrap_or(ConfidenceValue::uncertain()));

        let merged_base = self.base.with_boosted_confidence(new_conf);

        CausalKnowledge {
            base: merged_base,
            graph: self.graph,
            outcome: self.outcome,
            treatments: self.treatments,
            identification: self.identification,
        }
    }

    /// Merge resolved knowledge (weighted by confidence)
    fn merge_resolved(self, other: CausalKnowledge<T>, conflict: f64) -> CausalKnowledge<T> {
        let conf1 = self.confidence().value();
        let conf2 = other.confidence().value();

        // Penalize by conflict
        let merged_conf = (conf1 + conf2) / 2.0 * (1.0 - conflict);

        let merged_base = self.base.with_boosted_confidence(
            ConfidenceValue::new(merged_conf.clamp(0.0, 1.0))
                .unwrap_or(ConfidenceValue::uncertain()),
        );

        CausalKnowledge {
            base: merged_base,
            graph: self.graph,
            outcome: self.outcome,
            treatments: self.treatments,
            identification: self.identification,
        }
    }

    /// Condition causal knowledge on new evidence
    ///
    /// Updates confidence based on whether evidence supports the causal model
    pub fn condition_on_evidence(
        self,
        evidence_var: &str,
        evidence_value: f64,
        expected_value: f64,
    ) -> CausalKnowledge<T> {
        // Check if evidence variable is in graph
        if !self.graph.contains_node(evidence_var) {
            return self;
        }

        // Compute how well evidence matches expectation
        let discrepancy = (evidence_value - expected_value).abs();
        let match_factor = (-discrepancy).exp(); // e^(-|diff|)

        let updated_conf = self.confidence().value() * match_factor;

        let updated_base = self.base.with_boosted_confidence(
            ConfidenceValue::new(updated_conf.clamp(0.0, 1.0))
                .unwrap_or(ConfidenceValue::uncertain()),
        );

        CausalKnowledge {
            base: updated_base,
            graph: self.graph,
            outcome: self.outcome,
            treatments: self.treatments,
            identification: self.identification,
        }
    }

    /// Check if this causal model is a refinement of another
    ///
    /// A refinement has the same or more edges (more assumptions)
    pub fn refines(&self, other: &CausalKnowledge<T>) -> bool {
        // Check all edges in other are in self
        for node in other.graph.node_names() {
            if let Some(other_parents) = other.graph.parents(node) {
                if let Some(self_parents) = self.graph.parents(node) {
                    if !other_parents.is_subset(self_parents) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }

    /// Marginalize over a variable (remove from causal structure)
    pub fn marginalize(mut self, variable: &str) -> Result<CausalKnowledge<T>, CausalError> {
        if variable == self.outcome || self.treatments.contains(&variable.to_string()) {
            return Err(CausalError::InvalidIntervention {
                reason: "Cannot marginalize treatment or outcome variable".to_string(),
            });
        }

        // Create new graph without the variable
        // (In a full implementation, would properly handle edge redirection)
        self.identification = IdentificationStatus::Unknown;
        Ok(self)
    }
}

/// Extension trait for causal knowledge with helper methods
impl<T: Clone> crate::temporal::TemporalKnowledge<T> {
    /// Boost confidence (helper for causal composition)
    pub fn with_boosted_confidence(self, new_confidence: ConfidenceValue) -> Self {
        use crate::epistemic::composition::EpistemicValue;

        let new_core =
            EpistemicValue::with_confidence(self.core.value().clone(), new_confidence.value());

        crate::temporal::TemporalKnowledge {
            core: new_core,
            temporal: self.temporal,
            history: self.history,
        }
    }
}

/// Causal meta-analysis: combine multiple causal studies
pub struct CausalMetaAnalysis<T> {
    studies: Vec<CausalKnowledge<T>>,
    weights: Vec<f64>,
}

impl<T: Clone + PartialEq> CausalMetaAnalysis<T> {
    /// Create new meta-analysis
    pub fn new() -> Self {
        CausalMetaAnalysis {
            studies: vec![],
            weights: vec![],
        }
    }

    /// Add a study with optional weight
    pub fn add_study(&mut self, study: CausalKnowledge<T>, weight: Option<f64>) {
        let w = weight.unwrap_or(study.confidence().value());
        self.studies.push(study);
        self.weights.push(w);
    }

    /// Check if all studies have compatible causal structures
    pub fn check_compatibility(&self) -> Result<(), String> {
        if self.studies.len() < 2 {
            return Ok(());
        }

        let reference = &self.studies[0];
        for (i, study) in self.studies.iter().enumerate().skip(1) {
            if !reference.graph.structurally_equivalent(&study.graph) {
                return Err(format!("Study {} has incompatible causal structure", i));
            }
        }

        Ok(())
    }

    /// Compute pooled effect estimate
    pub fn pooled_estimate(&self) -> Option<CausalKnowledge<T>> {
        if self.studies.is_empty() {
            return None;
        }

        // Use first study as base, update confidence
        let total_weight: f64 = self.weights.iter().sum();
        let weighted_conf: f64 = self
            .studies
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| s.confidence().value() * w)
            .sum::<f64>()
            / total_weight;

        let mut result = self.studies[0].clone();
        result.base = result.base.with_boosted_confidence(
            ConfidenceValue::new(weighted_conf.clamp(0.0, 1.0))
                .unwrap_or(ConfidenceValue::uncertain()),
        );

        Some(result)
    }

    /// Compute heterogeneity (I² statistic analogue)
    pub fn heterogeneity(&self) -> f64 {
        if self.studies.len() < 2 {
            return 0.0;
        }

        // Variance in confidence levels as proxy for heterogeneity
        let mean_conf: f64 = self
            .studies
            .iter()
            .map(|s| s.confidence().value())
            .sum::<f64>()
            / self.studies.len() as f64;

        let variance: f64 = self
            .studies
            .iter()
            .map(|s| (s.confidence().value() - mean_conf).powi(2))
            .sum::<f64>()
            / self.studies.len() as f64;

        // I² = variance / (variance + within-study variance)
        let within_variance = 0.01; // Assumed within-study variance
        variance / (variance + within_variance)
    }
}

impl<T: Clone + PartialEq> Default for CausalMetaAnalysis<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::graph::{CausalGraph, CausalNode, EdgeType};
    use crate::epistemic::composition::EpistemicValue;
    use crate::temporal::TemporalKnowledge;

    fn create_causal_knowledge(value: f64, confidence: f64) -> CausalKnowledge<f64> {
        let mut graph = CausalGraph::new();
        graph.add_node(CausalNode::treatment("X"));
        graph.add_node(CausalNode::outcome("Y"));
        graph.add_edge("X", "Y", EdgeType::Direct).unwrap();

        let base = TemporalKnowledge::timeless(EpistemicValue::with_confidence(value, confidence));

        CausalKnowledge::new(base, graph, "Y", vec!["X".to_string()])
    }

    #[test]
    fn test_causal_tensor() {
        let ck1 = create_causal_knowledge(0.5, 0.9);

        let mut graph2 = CausalGraph::new();
        graph2.add_node(CausalNode::treatment("A"));
        graph2.add_node(CausalNode::outcome("B"));
        graph2.add_edge("A", "B", EdgeType::Direct).unwrap();

        let base2 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.7, 0.8));
        let ck2: CausalKnowledge<f64> =
            CausalKnowledge::new(base2, graph2, "B", vec!["A".to_string()]);

        let result = ck1.tensor_causal(ck2);
        assert!(result.is_ok());

        let merged = result.unwrap();
        assert!(merged.graph.contains_node("X"));
        assert!(merged.graph.contains_node("Y"));
        assert!(merged.graph.contains_node("A"));
        assert!(merged.graph.contains_node("B"));
    }

    #[test]
    fn test_causal_join_concordant() {
        let ck1 = create_causal_knowledge(0.5, 0.8);
        let ck2 = create_causal_knowledge(0.5, 0.85);

        let result = ck1.join_causal(ck2, 0.3);

        match result {
            CausalJoinResult::Concordant(merged) => {
                // Confidence should be boosted
                assert!(merged.confidence().value() > 0.8);
            }
            _ => panic!("Expected concordant join"),
        }
    }

    #[test]
    fn test_causal_join_irreconcilable_structure() {
        let ck1 = create_causal_knowledge(0.5, 0.8);

        // Different structure
        let mut graph2 = CausalGraph::new();
        graph2.add_node(CausalNode::treatment("X"));
        graph2.add_node(CausalNode::mediator("M"));
        graph2.add_node(CausalNode::outcome("Y"));
        graph2.add_edge("X", "M", EdgeType::Direct).unwrap();
        graph2.add_edge("M", "Y", EdgeType::Direct).unwrap();

        let base2 = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.85));
        let ck2 = CausalKnowledge::new(base2, graph2, "Y", vec!["X".to_string()]);

        let result = ck1.join_causal(ck2, 0.3);

        assert!(matches!(result, CausalJoinResult::Irreconcilable { .. }));
    }

    #[test]
    fn test_condition_on_evidence() {
        let ck = create_causal_knowledge(0.5, 0.9);

        // Evidence matches expectation
        let updated = ck.clone().condition_on_evidence("Y", 0.5, 0.5);
        assert!((updated.confidence().value() - 0.9).abs() < 0.01);

        // Evidence differs from expectation
        let updated2 = ck.condition_on_evidence("Y", 1.0, 0.5);
        assert!(updated2.confidence().value() < 0.9);
    }

    #[test]
    fn test_meta_analysis() {
        let mut meta = CausalMetaAnalysis::new();

        meta.add_study(create_causal_knowledge(0.5, 0.8), None);
        meta.add_study(create_causal_knowledge(0.5, 0.85), None);
        meta.add_study(create_causal_knowledge(0.5, 0.9), None);

        assert!(meta.check_compatibility().is_ok());

        let pooled = meta.pooled_estimate();
        assert!(pooled.is_some());

        let heterogeneity = meta.heterogeneity();
        assert!(heterogeneity >= 0.0);
        assert!(heterogeneity <= 1.0);
    }
}

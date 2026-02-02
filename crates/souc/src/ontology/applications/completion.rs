// Knowledge Base Completion - Predict missing is-a relations using spectral distance

use crate::ontology::distance::path::HierarchyGraph;
use crate::ontology::distance::{SpectralConfig, SpectralDistance};
use crate::ontology::loader::IRI;
use std::collections::HashSet;

/// Predicted relation with confidence
#[derive(Clone, Debug)]
pub struct CompletionCandidate {
    pub parent: IRI,
    pub child: IRI,
    pub confidence_score: f64,
    pub reasoning: String,
}

impl CompletionCandidate {
    pub fn new(parent: IRI, child: IRI, score: f64, reason: String) -> Self {
        Self {
            parent,
            child,
            confidence_score: score,
            reasoning: reason,
        }
    }
}

/// Knowledge base completion using spectral methods
pub struct KnowledgeBaseCompletion {
    spectral: SpectralDistance,
    known_relations: HashSet<(IRI, IRI)>,
}

impl KnowledgeBaseCompletion {
    pub fn new(config: SpectralConfig) -> Self {
        Self {
            spectral: SpectralDistance::new(config),
            known_relations: HashSet::new(),
        }
    }

    /// Build completion model from hierarchy
    pub fn build(&mut self, hierarchy: &HierarchyGraph) {
        self.spectral.build(hierarchy);

        // Record all existing relations
        for term in hierarchy.all_terms() {
            for parent in hierarchy.get_parents(term) {
                self.known_relations.insert((parent.clone(), term.clone()));
            }
        }
    }

    /// Predict missing is-a relations
    pub fn predict_missing_relations(
        &self,
        hierarchy: &HierarchyGraph,
        top_k: usize,
    ) -> Vec<CompletionCandidate> {
        let mut candidates = Vec::new();
        let all_terms: Vec<_> = hierarchy.all_terms().cloned().collect();

        // Score all potential parent-child pairs
        for (i, child) in all_terms.iter().enumerate() {
            for parent in all_terms.iter().skip(i + 1) {
                // Skip if relation already exists
                if self
                    .known_relations
                    .contains(&(parent.clone(), child.clone()))
                    || self
                        .known_relations
                        .contains(&(child.clone(), parent.clone()))
                {
                    continue;
                }

                // Score candidate relation
                if let Some(distance) = self.spectral.semantic_distance(parent, child) {
                    // Higher distance = less similar = less likely relation
                    let score = 1.0 / (1.0 + distance); // Convert distance to similarity

                    candidates.push(CompletionCandidate::new(
                        parent.clone(),
                        child.clone(),
                        score,
                        format!("Spectral distance: {:.4}", distance),
                    ));
                }
            }
        }

        // Sort by score and return top-k
        candidates.sort_by(|a, b| b.confidence_score.partial_cmp(&a.confidence_score).unwrap());
        candidates.into_iter().take(top_k).collect()
    }

    /// Score a specific candidate relation
    pub fn score_candidate(&self, parent: &IRI, child: &IRI) -> Option<f64> {
        // Skip if relation exists
        if self
            .known_relations
            .contains(&(parent.clone(), child.clone()))
        {
            return Some(1.0);
        }

        // Use spectral distance
        self.spectral
            .semantic_distance(parent, child)
            .map(|distance| 1.0 / (1.0 + distance))
    }

    /// Filter candidates by logical consistency
    pub fn filter_logical_consistency(
        &self,
        candidates: Vec<CompletionCandidate>,
        _hierarchy: &HierarchyGraph,
    ) -> Vec<CompletionCandidate> {
        // Simplified: all candidates pass for now
        // Full implementation would check for cycles, transitivity violations, etc.
        candidates
            .into_iter()
            .filter(|c| c.confidence_score > 0.5)
            .collect()
    }

    /// Check if a relation already exists
    pub fn relation_exists(&self, parent: &IRI, child: &IRI) -> bool {
        self.known_relations
            .contains(&(parent.clone(), child.clone()))
            || self
                .known_relations
                .contains(&(child.clone(), parent.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_candidate_creation() {
        let parent = IRI::new("http://test/parent");
        let child = IRI::new("http://test/child");
        let candidate =
            CompletionCandidate::new(parent.clone(), child.clone(), 0.85, "test".to_string());

        assert_eq!(candidate.parent, parent);
        assert_eq!(candidate.child, child);
        assert!((candidate.confidence_score - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_kb_completion_creation() {
        let config = SpectralConfig::default();
        let completion = KnowledgeBaseCompletion::new(config);
        assert!(completion.known_relations.is_empty());
    }

    #[test]
    fn test_score_candidate() {
        let config = SpectralConfig::default();
        let completion = KnowledgeBaseCompletion::new(config);

        let parent = IRI::new("http://test/parent");
        let child = IRI::new("http://test/child");

        // Score should be between 0 and 1
        if let Some(score) = completion.score_candidate(&parent, &child) {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_filter_logical_consistency() {
        let config = SpectralConfig::default();
        let completion = KnowledgeBaseCompletion::new(config);

        let mut candidates = vec![
            CompletionCandidate::new(
                IRI::new("http://test/p1"),
                IRI::new("http://test/c1"),
                0.85,
                "test".to_string(),
            ),
            CompletionCandidate::new(
                IRI::new("http://test/p2"),
                IRI::new("http://test/c2"),
                0.3,
                "test".to_string(),
            ),
        ];

        let filtered = completion.filter_logical_consistency(candidates, &HierarchyGraph::new());
        assert_eq!(filtered.len(), 1);
    }
}

//! Epistemic Augmentation Engine
//!
//! During ontology loading, each term receives an initial epistemic status
//! based on curation quality, provenance, and semantic analysis.
//!
//! Innovation from "Integrating Ontologies with LLMs" (2025):
//! Use embedding-based clustering to infer implicit relations.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::epistemic::{Confidence, EpistemicStatus, Evidence, EvidenceKind, Revisability, Source};

use super::{CurationStatus, TermEntry};

/// Engine for computing epistemic metadata during ontology load
pub struct EpistemicAugmenter {
    /// Cache for computed embeddings
    embedding_cache: std::collections::HashMap<String, Vec<f32>>,
}

impl EpistemicAugmenter {
    pub fn new() -> Self {
        Self {
            embedding_cache: std::collections::HashMap::new(),
        }
    }

    /// Compute initial epistemic status for a term
    pub fn compute_initial_epistemic(
        &self,
        entry: &TermEntry,
        curation: CurationStatus,
        provenance: &str,
    ) -> EpistemicStatus {
        // Base confidence from curation status
        let base_confidence = curation.base_confidence();

        // Adjust based on definition quality
        let definition_factor = match &entry.definition {
            Some(def) if def.len() > 100 => 1.0, // Good definition
            Some(def) if def.len() > 20 => 0.95, // Minimal definition
            Some(_) => 0.9,                      // Poor definition
            None => 0.8,                         // No definition
        };

        // Adjust based on parent chain (deeper = more specific = higher confidence)
        let hierarchy_factor = match entry.parents.len() {
            0 => 0.95,     // Root term
            1..=3 => 1.0,  // Good specificity
            4..=6 => 1.02, // Very specific
            _ => 1.0,
        };

        let final_confidence =
            (base_confidence * definition_factor * hierarchy_factor).clamp(0.0, 1.0);

        EpistemicStatus {
            confidence: Confidence::new(final_confidence),
            revisability: Revisability::Revisable {
                conditions: vec!["ontology_update".into()],
            },
            source: Source::OntologyAssertion {
                ontology: provenance.to_string(),
                term: entry.id.id.clone(),
            },
            evidence: vec![Evidence {
                kind: EvidenceKind::Publication {
                    doi: self.extract_doi(entry),
                },
                reference: provenance.to_string(),
                strength: Confidence::new(final_confidence),
            }],
        }
    }

    /// Compute embedding vector for semantic similarity
    pub fn compute_embedding(&self, entry: &TermEntry) -> Option<Vec<f32>> {
        // In production: use sentence-transformers or similar
        // For now: simple TF-IDF style embedding

        let text = format!(
            "{} {}",
            entry.id.label.as_deref().unwrap_or(""),
            entry.definition.as_deref().unwrap_or("")
        );

        if text.trim().is_empty() {
            return None;
        }

        // Simplified: hash-based embedding (replace with real model)
        Some(self.simple_hash_embedding(&text))
    }

    fn simple_hash_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; 128];

        for word in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            word.to_lowercase().hash(&mut hasher);
            let hash = hasher.finish();

            // Distribute hash across embedding dimensions
            for i in 0..128 {
                let bit = ((hash >> (i % 64)) & 1) as f32;
                embedding[i] += if bit == 1.0 { 0.1 } else { -0.1 };
            }
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    fn extract_doi(&self, _entry: &TermEntry) -> Option<String> {
        // Would parse from definition or metadata
        None
    }

    /// Compute confidence adjustment based on term characteristics
    pub fn confidence_adjustment(&self, entry: &TermEntry) -> f64 {
        let mut adjustment = 1.0;

        // Terms with more synonyms are often better defined
        // (would need synonym info in TermEntry)

        // Terms with cross-references are more reliable
        // (would need xref info in TermEntry)

        // Very short labels might be abbreviations (less clear)
        if let Some(label) = &entry.id.label
            && label.len() < 3
        {
            adjustment *= 0.95;
        }

        adjustment
    }
}

impl Default for EpistemicAugmenter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::TermId;

    #[test]
    fn test_compute_embedding() {
        let augmenter = EpistemicAugmenter::new();

        let entry = TermEntry {
            id: TermId::new("TEST:001"),
            ontology: "TEST".into(),
            definition: Some("A test term for validation".into()),
            parents: vec![],
        };

        let embedding = augmenter.compute_embedding(&entry);
        assert!(embedding.is_some());

        let emb = embedding.unwrap();
        assert_eq!(emb.len(), 128);

        // Embedding should be normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_initial_epistemic_expert_curated() {
        let augmenter = EpistemicAugmenter::new();

        let entry = TermEntry {
            id: TermId::new("PATO:0000001"),
            ontology: "PATO".into(),
            definition: Some(
                "A dependent entity that inheres in a bearer by virtue of being a quality".into(),
            ),
            parents: vec!["BFO:0000019".into()],
        };

        let status =
            augmenter.compute_initial_epistemic(&entry, CurationStatus::ExpertCurated, "PATO");

        // Expert curated with good definition: 0.98 * 0.95 (mid-length def) * 1.0 ≈ 0.93
        assert!(status.confidence.value() > 0.92);
    }

    #[test]
    fn test_initial_epistemic_auto_generated() {
        let augmenter = EpistemicAugmenter::new();

        let entry = TermEntry {
            id: TermId::new("AUTO:001"),
            ontology: "AUTO".into(),
            definition: None,
            parents: vec![],
        };

        let status =
            augmenter.compute_initial_epistemic(&entry, CurationStatus::AutoGenerated, "AUTO");

        // Auto-generated with no definition should have lower confidence
        assert!(status.confidence.value() < 0.7);
    }
}

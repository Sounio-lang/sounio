//! Cross-Ontology Alignment System
//!
//! This module provides infrastructure for aligning types across different
//! ontologies. This is critical for Sounio' key innovation: type compatibility
//! based on semantic distance across ontology boundaries.
//!
//! # The Problem
//!
//! With 15M+ types from multiple ontologies, the same real-world concept often
//! has different identifiers:
//!
//! - ChEBI:15365 (Aspirin) ≈ DrugBank:DB00945 (Acetylsalicylic acid)
//! - MONDO:0005148 (Type 2 diabetes) ≈ DOID:9352 (Type 2 diabetes mellitus)
//! - SNOMED:73211009 (Diabetes mellitus) ≈ ICD10:E11 (Type 2 diabetes)
//!
//! # Solution Layers
//!
//! 1. **SSSOM Mappings**: Curated cross-references between ontologies
//! 2. **UMLS CUI Bridging**: Unified Medical Language System concept IDs
//! 3. **Embedding Similarity**: SapBERT vectors for semantic matching
//! 4. **Transitive Closure**: A ≈ B, B ≈ C → A ≈ C
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                 AlignmentIndex                       │
//! │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │
//! │  │   SSSOM     │ │    UMLS     │ │   Embedding   │  │
//! │  │  Mappings   │ │  CUI Bridge │ │   Similarity  │  │
//! │  └──────┬──────┘ └──────┬──────┘ └───────┬───────┘  │
//! │         │               │                │          │
//! │         └───────────────┼────────────────┘          │
//! │                         ▼                           │
//! │              ┌─────────────────┐                    │
//! │              │  Unified Lookup │                    │
//! │              └─────────────────┘                    │
//! └─────────────────────────────────────────────────────┘
//! ```

pub mod cui;
pub mod loom;
pub mod unified;

use crate::ontology::distance::sssom::MappingPredicate;
use crate::ontology::loader::IRI;

pub use cui::{CUIBridge, CUIMapping, UMLSConcept};
pub use loom::{LOOMClient, LOOMMapping};
pub use unified::{AlignmentIndex, AlignmentMethod, AlignmentResult};

/// Confidence threshold for accepting a mapping
pub const MIN_MAPPING_CONFIDENCE: f64 = 0.5;

/// Maximum transitive hops to consider
pub const MAX_TRANSITIVE_HOPS: usize = 3;

/// Alignment source indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlignmentSource {
    /// Direct SSSOM mapping
    SSSOM,
    /// UMLS CUI bridging
    UMLS,
    /// BioPortal LOOM
    LOOM,
    /// Embedding similarity
    Embedding,
    /// Transitive inference
    Transitive,
    /// Manual curation
    Manual,
}

impl AlignmentSource {
    /// Get reliability score for this source
    pub fn reliability(&self) -> f64 {
        match self {
            Self::Manual => 1.0,
            Self::SSSOM => 0.95,
            Self::UMLS => 0.90,
            Self::LOOM => 0.85,
            Self::Embedding => 0.70,
            Self::Transitive => 0.60,
        }
    }
}

/// A cross-ontology alignment between two terms
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Source term
    pub source: IRI,
    /// Target term
    pub target: IRI,
    /// Mapping predicate (exact, close, narrow, broad, related)
    pub predicate: MappingPredicate,
    /// Confidence in the alignment [0, 1]
    pub confidence: f64,
    /// How the alignment was derived
    pub source_type: AlignmentSource,
    /// Provenance chain (for transitive alignments)
    pub provenance: Vec<IRI>,
    /// Bridging concept (e.g., UMLS CUI)
    pub bridge: Option<String>,
}

impl Alignment {
    /// Create a direct alignment
    pub fn direct(
        source: IRI,
        target: IRI,
        predicate: MappingPredicate,
        confidence: f64,
        source_type: AlignmentSource,
    ) -> Self {
        Self {
            source,
            target,
            predicate,
            confidence,
            source_type,
            provenance: Vec::new(),
            bridge: None,
        }
    }

    /// Create an alignment via UMLS CUI bridge
    pub fn via_cui(source: IRI, target: IRI, cui: String, confidence: f64) -> Self {
        Self {
            source,
            target,
            predicate: MappingPredicate::ExactMatch,
            confidence,
            source_type: AlignmentSource::UMLS,
            provenance: Vec::new(),
            bridge: Some(cui),
        }
    }

    /// Create a transitive alignment
    pub fn transitive(source: IRI, target: IRI, confidence: f64, provenance: Vec<IRI>) -> Self {
        Self {
            source,
            target,
            predicate: MappingPredicate::CloseMatch,
            confidence,
            source_type: AlignmentSource::Transitive,
            provenance,
            bridge: None,
        }
    }

    /// Compute effective confidence (confidence * source reliability)
    pub fn effective_confidence(&self) -> f64 {
        self.confidence * self.source_type.reliability()
    }

    /// Convert to semantic distance [0, 1]
    pub fn to_distance(&self) -> f64 {
        // Distance = 1 - (predicate_weight * effective_confidence)
        let predicate_weight = self.predicate.semantic_weight();
        1.0 - (predicate_weight * self.effective_confidence())
    }
}

/// Statistics for alignment index
#[derive(Debug, Clone, Default)]
pub struct AlignmentStats {
    pub total_alignments: usize,
    pub sssom_alignments: usize,
    pub umls_alignments: usize,
    pub loom_alignments: usize,
    pub embedding_alignments: usize,
    pub transitive_alignments: usize,
    pub unique_terms: usize,
    pub ontology_pairs: usize,
}

impl AlignmentStats {
    pub fn summary(&self) -> String {
        format!(
            "Alignments: {} total ({} SSSOM, {} UMLS, {} LOOM, {} embedding, {} transitive), {} terms, {} ontology pairs",
            self.total_alignments,
            self.sssom_alignments,
            self.umls_alignments,
            self.loom_alignments,
            self.embedding_alignments,
            self.transitive_alignments,
            self.unique_terms,
            self.ontology_pairs,
        )
    }
}

/// Ontology pair for tracking cross-ontology mappings
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OntologyPair {
    /// First ontology (lexicographically smaller)
    pub first: String,
    /// Second ontology
    pub second: String,
}

impl OntologyPair {
    pub fn new(a: &str, b: &str) -> Self {
        if a <= b {
            Self {
                first: a.to_string(),
                second: b.to_string(),
            }
        } else {
            Self {
                first: b.to_string(),
                second: a.to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_source_reliability() {
        assert!(AlignmentSource::Manual.reliability() > AlignmentSource::SSSOM.reliability());
        assert!(AlignmentSource::SSSOM.reliability() > AlignmentSource::Embedding.reliability());
    }

    #[test]
    fn test_alignment_to_distance() {
        let exact = Alignment::direct(
            IRI::new("test:a"),
            IRI::new("test:b"),
            MappingPredicate::ExactMatch,
            1.0,
            AlignmentSource::SSSOM,
        );

        // ExactMatch weight = 1.0, confidence = 1.0, SSSOM reliability = 0.95
        // Distance = 1 - (1.0 * 1.0 * 0.95) = 0.05
        assert!((exact.to_distance() - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_ontology_pair_normalization() {
        let pair1 = OntologyPair::new("MONDO", "DOID");
        let pair2 = OntologyPair::new("DOID", "MONDO");
        assert_eq!(pair1, pair2);
    }
}

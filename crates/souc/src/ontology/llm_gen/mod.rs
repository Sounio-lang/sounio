//! LLM-Augmented Ontology Generation
//!
//! This module provides automatic ontology generation and enrichment using LLMs.
//! It bridges the gap between natural language domain descriptions and formal
//! ontological structures.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    OntologyGenerator                                 │
//! │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
//! │  │ Term         │   │ Taxonomy     │   │ Relation     │            │
//! │  │ Extractor    │──▶│ Builder      │──▶│ Extractor    │            │
//! │  └──────────────┘   └──────────────┘   └──────────────┘            │
//! │         │                  │                  │                     │
//! │         ▼                  ▼                  ▼                     │
//! │  ┌──────────────────────────────────────────────────────────┐      │
//! │  │              GeneratedOntologyFragment                    │      │
//! │  │  - classes: Vec<GeneratedClass>                          │      │
//! │  │  - properties: Vec<GeneratedProperty>                    │      │
//! │  │  - axioms: Vec<GeneratedAxiom>                           │      │
//! │  │  - provenance: LLMProvenance                             │      │
//! │  └──────────────────────────────────────────────────────────┘      │
//! │                            │                                        │
//! │                            ▼                                        │
//! │  ┌──────────────────────────────────────────────────────────┐      │
//! │  │              OntologyValidator                            │      │
//! │  │  - Consistency checks                                     │      │
//! │  │  - BFO alignment verification                             │      │
//! │  │  - Naming convention checks                               │      │
//! │  └──────────────────────────────────────────────────────────┘      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::llm_gen::{OntologyGenerator, GenerationConfig};
//! use sounio::llm::LLMClientRegistry;
//!
//! let registry = LLMClientRegistry::from_env();
//! let generator = OntologyGenerator::new(registry);
//!
//! // Generate from natural language
//! let fragment = generator.generate_from_text(
//!     "Porous scaffolds support bone regeneration through cell migration",
//!     "biomaterials",
//! )?;
//!
//! // Validate the generated ontology
//! let validation = generator.validate(&fragment)?;
//! if validation.is_valid() {
//!     // Merge into existing ontology
//! }
//! ```

pub mod generator;
pub mod validator;

pub use generator::{
    ExtractedRelation, ExtractedTerm, GeneratedAxiom, GeneratedClass, GeneratedOntologyFragment,
    GeneratedProperty, GenerationConfig, GenerationStats, LLMProvenance, OntologyGenerator,
    TaxonomicRelation,
};
pub use validator::{
    OntologyValidator, ValidationConfig, ValidationIssue, ValidationResult, ValidationSeverity,
};

use crate::llm::LLMError;
use std::fmt;

/// Errors that can occur during ontology generation
#[derive(Debug, Clone)]
pub enum GenerationError {
    /// LLM query failed
    LLMError(String),
    /// Failed to parse LLM response
    ParseError(String),
    /// Generated ontology is invalid
    ValidationError(String),
    /// No terms could be extracted
    NoTermsExtracted,
    /// Ontology generation not available (feature not enabled)
    NotAvailable,
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenerationError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            GenerationError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            GenerationError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            GenerationError::NoTermsExtracted => write!(f, "No terms could be extracted from text"),
            GenerationError::NotAvailable => {
                write!(f, "Ontology generation requires the 'llm' feature")
            }
        }
    }
}

impl std::error::Error for GenerationError {}

impl From<LLMError> for GenerationError {
    fn from(err: LLMError) -> Self {
        GenerationError::LLMError(err.to_string())
    }
}

/// Result type for generation operations
pub type GenerationResult<T> = Result<T, GenerationError>;

//! Knowledge[τ, ε, δ, Φ] - The epistemic type primitive
//!
//! This is the foundational type that makes Sounio unique:
//! Every value knows where it came from, how certain it is,
//! what domain validates it, and how it was transformed.
//!
//! # Core Innovation
//!
//! Type checking in Sounio IS ontological reasoning:
//! - A value of type `Knowledge[δ: ChEBI:aspirin]`
//! - Can be used where `Knowledge[δ: ChEBI:drug]` is expected
//! - Because aspirin is a subclass of drug in the ChEBI ontology
//!
//! # Example
//!
//! ```sounio
//! let mass: Knowledge[
//!     content = f64,
//!     τ = (2024, Lab, Experiment),
//!     ε = (confidence: 0.95, source: Measurement),
//!     δ = PATO:mass,
//!     Φ = [sensor → calibration → conversion]
//! ] = measure_mass(sample);
//! ```

use super::{ContextTime, EpistemicStatus, Provenance, Transformation};
use crate::common::Span;
use crate::types::Type;

/// The Knowledge type - first-class epistemic primitive
///
/// This is the central type that enables ontology-driven type checking.
/// Every Knowledge value carries complete epistemic metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Knowledge {
    /// The wrapped content type (e.g., f64, String, custom struct)
    pub content: Box<Type>,

    /// τ: Temporal context index
    /// Tracks when and in what context this knowledge is valid
    pub temporal: ContextTime,

    /// ε: Epistemic status (confidence, revisability, source)
    /// Tracks how certain we are and where this came from
    pub epistemic: EpistemicStatus,

    /// δ: Domain ontology binding
    /// Links this value to an ontology term for semantic validation
    pub domain: OntologyBinding,

    /// Φ: Functor trace (transformation provenance)
    /// Complete history of how this value was derived
    pub provenance: Provenance,

    /// Source location in the code
    pub span: Span,
}

/// Binding to an ontology term
///
/// This is how values are linked to ontological concepts.
/// The binding enables semantic type checking based on ontology relationships.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OntologyBinding {
    /// Ontology prefix (e.g., "BFO", "PATO", "ChEBI")
    pub ontology: OntologyRef,

    /// Term identifier within the ontology
    pub term: TermId,

    /// Optional constraint/refinement on the binding
    pub constraint: Option<OntologyConstraint>,
}

/// Reference to an ontology in the hierarchy
///
/// Sounio uses a 4-layer ontology architecture:
/// - L1: Primitive (BFO, RO, COB) - ~850 terms, compiled into compiler
/// - L2: Foundation (PATO, UO, IAO, Schema.org, FHIR) - ~8,000 terms, shipped with stdlib
/// - L3: Domain (ChEBI, GO, DOID, etc.) - ~500,000 terms, lazy loaded
/// - L4: Federated (BioPortal resolution) - ~15,000,000 terms, runtime resolution
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OntologyRef {
    /// L1: Primitive (BFO, RO, COB) - always available
    Primitive(PrimitiveOntology),

    /// L2: Foundation (PATO, UO, IAO, Schema.org, FHIR)
    Foundation(FoundationOntology),

    /// L3: Domain (ChEBI, GO, DOID, etc.) - lazy loaded
    Domain(DomainOntology),

    /// L4: Federated (BioPortal resolution)
    Federated(FederatedRef),
}

/// L1 Primitive ontologies - compiled into the compiler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveOntology {
    /// Basic Formal Ontology 2020 (36 classes)
    /// The upper-level ontology providing foundational categories
    BFO,
    /// Relation Ontology (~600 relations)
    /// Standard relations for linking ontology terms
    RO,
    /// Core Ontology for Biology and Biomedicine (~200 classes)
    /// Bridge between BFO and domain ontologies
    COB,
}

impl PrimitiveOntology {
    /// Get the standard prefix for this ontology
    pub fn prefix(&self) -> &'static str {
        match self {
            PrimitiveOntology::BFO => "BFO",
            PrimitiveOntology::RO => "RO",
            PrimitiveOntology::COB => "COB",
        }
    }

    /// Get the full IRI namespace
    pub fn namespace(&self) -> &'static str {
        match self {
            PrimitiveOntology::BFO => "http://purl.obolibrary.org/obo/BFO_",
            PrimitiveOntology::RO => "http://purl.obolibrary.org/obo/RO_",
            PrimitiveOntology::COB => "http://purl.obolibrary.org/obo/COB_",
        }
    }
}

/// L2 Foundation ontologies - shipped with stdlib
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FoundationOntology {
    /// Phenotype And Trait Ontology (~2,500 terms)
    PATO,
    /// Units of Measurement Ontology (~1,000 terms)
    UO,
    /// Information Artifact Ontology (~300 terms)
    IAO,
    /// Schema.org vocabulary (~2,850 types)
    SchemaOrg,
    /// FHIR R5 resources (~1,150 resources)
    FHIR,
}

impl FoundationOntology {
    /// Get the standard prefix for this ontology
    pub fn prefix(&self) -> &'static str {
        match self {
            FoundationOntology::PATO => "PATO",
            FoundationOntology::UO => "UO",
            FoundationOntology::IAO => "IAO",
            FoundationOntology::SchemaOrg => "schema",
            FoundationOntology::FHIR => "fhir",
        }
    }
}

/// L3 Domain ontologies - lazy loaded on demand
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DomainOntology {
    /// OBO Foundry ID (e.g., "chebi", "go", "doid")
    pub id: String,
    /// Version (semantic versioning)
    pub version: Option<String>,
}

/// L4 Federated reference - runtime resolution via BioPortal
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FederatedRef {
    /// BioPortal ontology acronym
    pub acronym: String,
    /// Specific version or "latest"
    pub version: String,
}

/// Term identifier within an ontology
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TermId {
    /// Numeric or alphanumeric ID (e.g., "0000001", "C12345")
    pub id: String,
    /// Human-readable label (cached for display)
    pub label: Option<String>,
}

impl TermId {
    /// Create a new term ID
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
        }
    }

    /// Create a term ID with a label
    pub fn with_label(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: Some(label.into()),
        }
    }
}

/// Constraint on ontology binding
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OntologyConstraint {
    /// Must be subclass of given term
    SubclassOf(TermId),
    /// Must have specific relation to term
    RelatedTo { relation: TermId, target: TermId },
    /// Must satisfy intersection of constraints
    Intersection(Vec<OntologyConstraint>),
    /// Must satisfy union of constraints
    Union(Vec<OntologyConstraint>),
}

impl Knowledge {
    /// Create a new Knowledge type with all indices specified
    pub fn new(
        content: Type,
        temporal: ContextTime,
        epistemic: EpistemicStatus,
        domain: OntologyBinding,
        provenance: Provenance,
        span: Span,
    ) -> Self {
        Self {
            content: Box::new(content),
            temporal,
            epistemic,
            domain,
            provenance,
            span,
        }
    }

    /// Create Knowledge with default epistemic status (for literals)
    ///
    /// Literals are axiomatic - certain, non-revisable, from definition.
    pub fn from_literal(content: Type, domain: OntologyBinding, span: Span) -> Self {
        Self {
            content: Box::new(content),
            temporal: ContextTime::current(),
            epistemic: EpistemicStatus::axiomatic(),
            domain,
            provenance: Provenance::literal(),
            span,
        }
    }

    /// Transform this knowledge through a functor, updating provenance
    ///
    /// When knowledge passes through a function, its provenance is extended
    /// and its confidence may be affected.
    pub fn transform(&self, transformation: Transformation) -> Self {
        Self {
            content: self.content.clone(),
            temporal: self.temporal.clone(),
            epistemic: self.epistemic.propagate(&transformation),
            domain: self.domain.clone(),
            provenance: self.provenance.extend(transformation),
            span: self.span,
        }
    }

    /// Check if this knowledge is compatible with target domain
    ///
    /// This is the core of ontological type checking.
    pub fn compatible_with(&self, target: &OntologyBinding) -> CompatibilityResult {
        // Same binding is always compatible
        if self.domain == *target {
            return CompatibilityResult::Compatible;
        }

        // Different ontology refs need translation
        if std::mem::discriminant(&self.domain.ontology) != std::mem::discriminant(&target.ontology)
        {
            // Could be compatible via translation functor
            return CompatibilityResult::RequiresTranslation(TranslationPath {
                steps: vec![TranslationStep {
                    from: self.domain.clone(),
                    to: target.clone(),
                    via: TermId::new("skos:closeMatch"),
                }],
                confidence_loss: 0.05, // Default 5% confidence loss for translation
            });
        }

        // Same ontology family - check subclass relationship
        // This would call into the ontology resolver
        CompatibilityResult::Incompatible(IncompatibilityReason::NoPath)
    }

    /// Get the confidence level
    pub fn confidence(&self) -> f64 {
        self.epistemic.confidence.value()
    }

    /// Check if this knowledge is revisable
    pub fn is_revisable(&self) -> bool {
        self.epistemic.revisability.is_revisable()
    }

    /// Get a display-friendly representation of the domain
    pub fn domain_display(&self) -> String {
        let prefix = match &self.domain.ontology {
            OntologyRef::Primitive(p) => p.prefix().to_string(),
            OntologyRef::Foundation(f) => f.prefix().to_string(),
            OntologyRef::Domain(d) => d.id.to_uppercase(),
            OntologyRef::Federated(f) => f.acronym.clone(),
        };

        if let Some(label) = &self.domain.term.label {
            format!("{}:{} ({})", prefix, self.domain.term.id, label)
        } else {
            format!("{}:{}", prefix, self.domain.term.id)
        }
    }
}

/// Result of compatibility check
#[derive(Debug, Clone)]
pub enum CompatibilityResult {
    /// Fully compatible (same or subtype in ontology)
    Compatible,
    /// Compatible via translation functor
    RequiresTranslation(TranslationPath),
    /// Incompatible with explanation
    Incompatible(IncompatibilityReason),
}

/// Path for translating between ontologies
#[derive(Debug, Clone)]
pub struct TranslationPath {
    /// Steps in the translation
    pub steps: Vec<TranslationStep>,
    /// Total confidence loss through translation
    pub confidence_loss: f64,
}

/// Single step in a translation path
#[derive(Debug, Clone)]
pub struct TranslationStep {
    /// Source binding
    pub from: OntologyBinding,
    /// Target binding
    pub to: OntologyBinding,
    /// Mapping relation (e.g., skos:exactMatch, skos:closeMatch)
    pub via: TermId,
}

/// Reason for incompatibility
#[derive(Debug, Clone)]
pub enum IncompatibilityReason {
    /// No ontological path exists between terms
    NoPath,
    /// Domains are explicitly disjoint (owl:disjointWith)
    Disjoint { evidence: TermId },
    /// Constraint violation
    ConstraintViolation { constraint: String, reason: String },
}

/// The KnowledgeType enum for the type system
///
/// Supports concrete knowledge with all indices, existential quantification
/// over some indices, and universal quantification for polymorphism.
#[derive(Debug, Clone, PartialEq)]
pub enum KnowledgeType {
    /// Concrete knowledge with all indices specified
    Concrete(Knowledge),

    /// Existential knowledge (some indices unknown)
    /// Represents knowledge where we don't know all the epistemic details
    Existential {
        content: Box<Type>,
        known_indices: KnownIndices,
    },

    /// Universal knowledge (polymorphic over indices)
    /// For functions that work with any epistemic status
    Universal {
        content: Box<Type>,
        quantified: QuantifiedIndices,
    },
}

/// Known indices for existential knowledge types
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KnownIndices {
    pub temporal: Option<ContextTime>,
    pub epistemic: Option<EpistemicStatus>,
    pub domain: Option<OntologyBinding>,
    pub provenance: Option<Provenance>,
}

/// Quantified indices for universal knowledge types
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QuantifiedIndices {
    pub temporal_var: Option<String>,
    pub epistemic_var: Option<String>,
    pub domain_var: Option<String>,
    pub provenance_var: Option<String>,
}

impl KnowledgeType {
    /// Get the content type, regardless of knowledge variant
    pub fn content_type(&self) -> &Type {
        match self {
            KnowledgeType::Concrete(k) => &k.content,
            KnowledgeType::Existential { content, .. } => content,
            KnowledgeType::Universal { content, .. } => content,
        }
    }

    /// Check if this is concrete (all indices known)
    pub fn is_concrete(&self) -> bool {
        matches!(self, KnowledgeType::Concrete(_))
    }

    /// Get confidence if available
    pub fn confidence(&self) -> Option<f64> {
        match self {
            KnowledgeType::Concrete(k) => Some(k.confidence()),
            KnowledgeType::Existential { known_indices, .. } => known_indices
                .epistemic
                .as_ref()
                .map(|e| e.confidence.value()),
            KnowledgeType::Universal { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_id_creation() {
        let term = TermId::new("0000001");
        assert_eq!(term.id, "0000001");
        assert!(term.label.is_none());

        let term_with_label = TermId::with_label("0000001", "entity");
        assert_eq!(term_with_label.label, Some("entity".to_string()));
    }

    #[test]
    fn test_primitive_ontology_prefix() {
        assert_eq!(PrimitiveOntology::BFO.prefix(), "BFO");
        assert_eq!(PrimitiveOntology::RO.prefix(), "RO");
        assert_eq!(PrimitiveOntology::COB.prefix(), "COB");
    }

    #[test]
    fn test_ontology_binding_equality() {
        let binding1 = OntologyBinding {
            ontology: OntologyRef::Primitive(PrimitiveOntology::BFO),
            term: TermId::new("0000001"),
            constraint: None,
        };

        let binding2 = OntologyBinding {
            ontology: OntologyRef::Primitive(PrimitiveOntology::BFO),
            term: TermId::new("0000001"),
            constraint: None,
        };

        assert_eq!(binding1, binding2);
    }
}

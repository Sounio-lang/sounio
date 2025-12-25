//! Type system for the Sounio language
//!
//! This module implements D's advanced type system including:
//! - Core types (primitives, references, functions)
//! - Ownership and borrowing (linear, affine types)
//! - Algebraic effects
//! - Refinement types
//! - Units of measure with inference
//! - Epistemic types (Knowledge[T, ε, Φ, τ])
//! - Quantitative Type Theory (multiplicities for erasure)

pub mod core;
pub mod effects;
pub mod epistemic;
pub mod erasure;
pub mod multiplicity;
pub mod ontology_erasure;
pub mod ownership;
pub mod refinement;
pub mod semantic;
pub mod unit_infer;
pub mod units;

pub use self::core::*;
pub use effects::*;
pub use ownership::*;

// Don't use glob re-export for these to avoid ambiguous `medical` module conflict
pub use refinement::{
    ArithOp, CompareOp, Predicate, RefinedType, RefinementChecker, RefinementResult,
};
pub use unit_infer::{UnitExpr, UnitInference, UnitInferenceError, UnitVar};
pub use units::{Unit, UnitChecker, UnitError, UnitOp};

// Epistemic types - the key differentiator of Sounio
pub use epistemic::{
    ConfidenceAnalysisError, ConfidenceBound, ConfidenceFlowAnalysis, ConfidenceOp,
    EpistemicCheckResult, EpistemicChecker, KnowledgeType, PropagationRule, PropagationRules,
    ProvenanceConstraint, TemporalConstraint,
};

// Semantic types - ontological type checking
pub use semantic::{
    SemanticCompatibility, SemanticType, SemanticTypeBuilder, SemanticTypeChecker,
    SemanticTypeConfig, SemanticTypeError, SemanticTypeStats,
};

// Quantitative Type Theory - multiplicities for erasure semantics
pub use multiplicity::{Multiplicity, MultiplicityContext, MultiplicityError, QType};

// Erasure analysis
pub use erasure::{
    ErasureAnalyzer, ErasureCategory, ErasureConfig, ErasureInfo, ErasureSet, ErasureStats,
};
pub use ontology_erasure::{
    CompilationErasure, ErasedRepresentation, FunctionErasureInfo, OntologicalType,
    OntologyErasureAnalyzer, OntologyErasureStats,
};

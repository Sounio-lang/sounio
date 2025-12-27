//! Epistemic type system for Sounio
//!
//! This module implements Knowledge as a first-class type with:
//! - Temporal indexing (τ) for context-dependent typing
//! - Epistemic status (ε) for confidence and revisability tracking
//! - Domain binding (δ) for ontology-validated types
//! - Functor trace (Φ) for complete provenance
//!
//! # The Paradigm Shift
//!
//! Traditional languages: Types are syntactic constraints
//! Sounio: Types are ontological assertions about reality
//!
//! Every value in Sounio carries its epistemic history.
//!
//! # Example
//!
//! ```sounio
//! let result: Knowledge[
//!     content = f64,
//!     τ = (2024, Lab, Experiment),
//!     ε = (confidence: 0.95, source: Measurement),
//!     δ = PATO:mass,
//!     Φ = [sensor → calibration → conversion]
//! ] = measure_mass(sample);
//! ```
//!
//! # Knowledge Type Structure
//!
//! ```text
//! Knowledge[τ, ε, δ, Φ]
//! │        │  │  │  └── Φ: Functor trace (transformation provenance)
//! │        │  │  └───── δ: Domain ontology (which ontology validates this)
//! │        │  └──────── ε: Epistemic status (confidence, revisability, source)
//! │        └─────────── τ: Context-time (temporal indexing for type evolution)
//! └──────────────────── Knowledge: First-class epistemic primitive
//! ```

pub mod agents;
pub mod bayesian;
pub mod beta_knowledge;
pub mod composition;
pub mod confidence;
pub mod evolution;
pub mod firewall;
pub mod heterogeneity;
pub mod knowledge;
pub mod merkle;
pub mod models;
pub mod operations;
pub mod provenance;
pub mod temporal;
pub mod time_travel;

// Uncertainty promotion lattice and KEC auto-selection
pub mod kec;
pub mod promotion;

// Epistemic + Refinement Type integration
pub mod refined_epistemic;

pub use confidence::{Confidence, EpistemicStatus, Evidence, EvidenceKind, Revisability, Source};
pub use heterogeneity::{
    HeterogeneityConfig, HeterogeneityResolver, ResolutionResult, ResolutionStrategy,
};
pub use knowledge::{
    CompatibilityResult, DomainOntology, FederatedRef, FoundationOntology, IncompatibilityReason,
    Knowledge, KnowledgeType, KnownIndices, OntologyBinding, OntologyConstraint, OntologyRef,
    PrimitiveOntology, QuantifiedIndices, TermId, TranslationPath, TranslationStep,
};
pub use operations::{
    EpistemicConstraint, InspectField, InspectOp, KnowledgeOp, MergeOp, MergeStrategy, QueryOp,
    RelationalConstraint, ReviseOp, RevisionStrategy, TranslateOp, TranslateOptions,
    assert_knowledge, query_knowledge, revise_knowledge, translate_knowledge,
};
pub use provenance::{
    FunctorTrace, Origin, Provenance, Transformation, TransformationKind, TransformationMetadata,
};
pub use temporal::{ContextIndex, ContextTime, TemporalIndex, TemporalOffset, ValidityBounds};

// New modules for advanced epistemic computing
pub use bayesian::{
    BayesianFusionResult, BeliefMass, BetaBound, BetaConfidence, BetaConstraintResult,
    DSTCombinationResult, EvidenceAssessment, HierarchicalPrior, OntologyDomain, SourceReliability,
    check_beta_constraint, combine_epistemic_beta, combine_epistemic_beta_with_prior,
    combine_epistemic_hierarchical, dempster_combine, dempster_combine_multiple,
};
pub use firewall::{EpistemicFirewall, FirewallConfig, FirewallMode, FirewallViolation};
pub use merkle::{
    AuditTrail, Hash256, MerkleProvenanceDAG, MerkleProvenanceNode, OperationKind,
    ProvenanceMetadata, ProvenanceOperation, ProvenanceSignature,
};
pub use models::{
    AffineConfig, BayesianConfig, BinaryOp, CombinationRule, DempsterShaferConfig, EpistemicConfig,
    FuzzyConfig, IntervalConfig, ProbabilisticConfig, UncertainValue, UncertaintyModel,
    propagate_binary,
};

// Beta-epistemic knowledge types (revolutionary full-distribution epistemic computing)
pub use beta_knowledge::{
    ActiveInferenceMetrics, BetaEpistemicStatus, BetaKnowledge, DecayModel, PriorType,
    SourcePriorType, exploration_priorities, variance_penalty,
};

// Time-travel debugging for epistemic provenance
pub use time_travel::{
    BreakAction, BreakCondition, BreakpointManager, ConfidenceDelta, CustodyRecord,
    DegradationReason, DegradingOperation, EpistemicBreakpoint, EpistemicSnapshot,
    FDAComplianceProof, ProofVerificationError, TimeTravelResult, TimelineGraph, TimelineState,
    VerificationResult, verify_external,
};

// KEC auto-selection for optimal uncertainty model
pub use kec::{
    ComplexityMetrics, KECConfig, KECResult, KECSelector, UncertaintyMetrics, auto_select_model,
    select_for_operation,
};

// Uncertainty promotion lattice
pub use promotion::{Promotable, PromotedValue, Promoter, PromotionLattice, UncertaintyLevel};

// Epistemic + Refinement Type integration (Issue #8)
pub use refined_epistemic::{
    BoundedEpistemic, EpistemicInterval, EpistemicRefinedConfig, EpistemicRefinedValue,
    PositiveEpistemic, ProbabilityEpistemic, RefinedCreationError, RefinementBounds,
};

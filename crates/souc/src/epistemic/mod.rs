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
pub mod gaussian_process;
pub mod heterogeneity;
pub mod information_geometry;
pub mod knowledge;
pub mod levinson_durbin;
pub mod mcmc;
pub mod mcmc_diagnostics;
pub mod merkle;
pub mod models;
pub mod nonlinear_propagation;
pub mod operations;
pub mod pce;
pub mod provenance;
pub mod temporal;
pub mod time_travel;
pub mod unscented_transform;
pub mod wasserstein;

// Uncertainty promotion lattice and KEC auto-selection
pub mod kec;
pub mod promotion;

// Epistemic + Refinement Type integration
pub mod refined_epistemic;

// SIMD-accelerated epistemic operations
pub mod simd_knowledge;

// Evidence collection and temporal validity enforcement
pub mod evidence_collector;
pub mod temporal_validity;

// Cross-cutting integration tests
#[cfg(test)]
mod integration_tests;

pub use confidence::{
    Confidence, EpistemicStatus, Evidence, EvidenceKind, KnightianMode, Revisability, Source,
};
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

// Information geometry on type spaces (Fisher metric, natural gradient)
pub use firewall::{EpistemicFirewall, FirewallConfig, FirewallMode, FirewallViolation};
pub use information_geometry::{AlphaConnection, FisherMatrix, NaturalGradientOptimizer};

// Optimal transport (Wasserstein-2 distance for type composition)
pub use merkle::{
    AuditTrail, Hash256, MerkleProvenanceDAG, MerkleProvenanceNode, OperationKind,
    ProvenanceMetadata, ProvenanceOperation, ProvenanceSignature,
};
pub use models::{
    AffineConfig, BayesianConfig, BinaryOp, CombinationRule, DempsterShaferConfig, EpistemicConfig,
    FuzzyConfig, IntervalConfig, ProbabilisticConfig, UncertainValue, UncertaintyModel,
    propagate_binary,
};
// Nonlinear confidence propagation (sin, cos, exp, log, sqrt, pow, control flow)
pub use nonlinear_propagation::{
    ConfidenceFlowResult, ConfidenceWarning, FunctionConfidenceSummary, NonlinearOp,
    NonlinearPropagationRule, PropagationChainBuilder, propagate_conditional, propagate_loop,
    propagate_nonlinear, propagate_with_rule,
};

pub use wasserstein::{
    WassersteinBarycenter, WassersteinDistance, transport_cost, wasserstein_barycenter,
    wasserstein2_distance,
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

// Polynomial Chaos Expansion for advanced UQ
pub use pce::{
    MultiIndex, PCEConfig, PCEExpansion, PolynomialFamily, SobolIndices, TruncationScheme,
};

// Unscented Transform for nonlinear uncertainty propagation
pub use unscented_transform::{
    ComparisonResult, SigmaPointMethod, UTConfig, UTCrossPropagationResult, UTPropagationResult,
    UnscentedTransform,
};

// MCMC samplers for Bayesian epistemic inference
pub use mcmc::{HamiltonianMC, MCMCConfig, MCMCResult, MetropolisHastings, NUTS, compute_r_hat};

// Gaussian Process regression for surrogate modeling and epistemic UQ
pub use gaussian_process::{GPConfig, GPPrediction, GaussianProcess, Kernel};

// SIMD-accelerated epistemic knowledge vectors
pub use simd_knowledge::{
    KnowledgeVec4, KnowledgeVec8, batch_add_uncertainties, batch_weighted_mean,
};

pub mod physical_law;
pub mod physical_law_synthesis;
pub mod grand_challenges;

pub use physical_law::{
    BoundaryType, ComplexityMeasure, ConservationQuantity, ConstitutiveRelation, CouplingStrategy,
    DomainOfValidity, EquilibriumType, FieldTheory, HybridComposition, HybridOperator,
    LawParameter, LawType, LearnedOperator, LearningMethod, LossFunction, NeuralArchitecture,
    OperatorTarget, PacBounds, ParameterConstraint, ParameterEpistemic, PhysicalLaw, PhysicalLawType,
    PhysicalProvenance, RegimeConstraint, ScaleConstraint, SpatialConstraint, TrainingConfig,
    TrainingProvenance, UqMethod, ValidationMetrics,
};

pub use physical_law_synthesis::{
    AutoPinNConfig, AutoPinNResult, EquationParser, LawSynthesisError, PdePattern, PdeTemplate,
    PhysicsInformedSynthesizer, SynthesisStrategy, auto_build_pinn, synthesize_from_equations,
};

pub use grand_challenges::{StabilityProof, SOUNIO_PROOF_DSL};

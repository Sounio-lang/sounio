//! PhysicalLaw[Λ, ε, Φ, P] - Epistemic types for Physics-Informed Neural Networks
//!
//! This module extends Sounio's epistemic type system to support Physics-Informed
//! Neural Networks (PINNs) by distinguishing between:
//! - Axiomatic physical laws (verified, immutable)
//! - Learned operators (data-driven, uncertain)
//! - Hybrid compositions (physics-guided learning)
//!
//! # Core Innovation
//!
//! PINNs in Sounio are not just "neural networks with loss terms"—they are
//! epistemically typed computational graphs where every operator carries
//! uncertainty quantification and physical validity constraints.
//!
//! # Example
//!
//! ```sounio
//! // Axiomatic: Navier-Stokes equations (100% confidence, verified)
//! let ns_law: PhysicalLaw[NavierStokes, ε = 1.0, Φ = Axiomatic] = 
//!     PhysicalLaw::from_pde(NavierStokes::incompressible());
//!
//! // Learned: Neural operator for turbulence closure (data-driven, uncertain)
//! let turbulence_model: LearnedOperator[RANS, ε >= 0.85, Φ = PAC(δ=0.05)] =
//!     LearnedOperator::train(rans_data, epochs: 10000);
//!
//! // Hybrid: Physics-informed neural operator
//! let pino: HybridOperator[FlowField] = 
//!     HybridOperator::compose(ns_law, turbulence_model);
//!
//! // Solve with automatic uncertainty propagation
//! let solution: Knowledge[FlowField, ε >= 0.80] = 
//!     pino.solve(initial_conditions, boundary_conditions);
//! ```

use crate::common::Span;
use crate::epistemic::{Confidence, EpistemicStatus, Provenance};
use crate::types::Type;

/// PhysicalLaw[Λ, ε, Φ, P] - The fundamental type for physics constraints
///
/// Λ (Lambda): The physical law being represented (PDE, conservation law, etc.)
/// ε (Epsilon): Confidence in the law's applicability to this context
/// Φ (Phi): Provenance—where did this law come from?
/// P (Pi): Parameters—free parameters that may be learned or measured
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalLaw {
    /// The physical law being represented
    pub law_type: LawType,

    /// Epistemic status (confidence, revisability, source)
    pub epistemic: EpistemicStatus,

    /// Provenance tracking the law's origin and validation
    pub provenance: PhysicalProvenance,

    /// Free parameters that may be learned or measured
    pub parameters: Vec<LawParameter>,

    /// Domain of validity (spatial, temporal, regime constraints)
    pub domain_of_validity: DomainOfValidity,

    /// Source location in code
    pub span: Span,
}

/// Types of physical laws supported by the system
#[derive(Debug, Clone, PartialEq)]
pub enum LawType {
    /// Conservation laws (mass, momentum, energy)
    Conservation(ConservationQuantity),

    /// Constitutive relations (stress-strain, equation of state)
    Constitutive(ConstitutiveRelation),

    /// Field equations (Maxwell, Schrödinger, Einstein)
    FieldEquation(FieldTheory),

    /// Rate equations (chemical kinetics, radioactive decay)
    RateEquation(String), // Process name

    /// Equilibrium conditions (thermodynamic, mechanical)
    Equilibrium(EquilibriumType),

    /// Boundary/interface conditions
    BoundaryCondition(BoundaryType),

    /// Learned surrogate for a physical process
    LearnedSurrogate {
        /// Name of the learned model
        name: String,
        /// Underlying physical process being approximated
        approximates: Box<LawType>,
        /// Training data provenance
        training_data: Provenance,
    },
}

/// Conservation quantities for conservation laws
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConservationQuantity {
    Mass,
    Momentum,
    Energy,
    AngularMomentum,
    ElectricCharge,
    BaryonNumber,
    LeptonNumber,
}

impl ConservationQuantity {
    /// Get the mathematical form of the conservation law
    pub fn divergence_form(&self) -> &'static str {
        match self {
            ConservationQuantity::Mass => "∂ρ/∂t + ∇·(ρv) = 0",
            ConservationQuantity::Momentum => "∂(ρv)/∂t + ∇·(ρv⊗v) = ∇·σ + ρf",
            ConservationQuantity::Energy => "∂E/∂t + ∇·((E+p)v) = ∇·(σ·v) + ρf·v",
            _ => "∂u/∂t + ∇·F(u) = S(u)",
        }
    }
}

/// Constitutive relations (material/field behavior)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstitutiveRelation {
    /// Linear elasticity: σ = C:ε
    LinearElasticity,
    /// Newtonian fluid: σ = -pI + 2μD
    NewtonianFluid,
    /// Ideal gas: p = ρRT
    IdealGas,
    /// Non-Newtonian fluid (learned or empirical)
    NonNewtonian { model: String },
    /// Electromagnetic constitutive: D = εE, B = μH
    Electromagnetic,
    /// Custom learned constitutive relation
    Learned { name: String, basis: String },
}

/// Field theories
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldTheory {
    /// Maxwell's equations
    Maxwell,
    /// Schrödinger equation
    Schrodinger,
    /// Einstein field equations (general relativity)
    Einstein,
    /// Yang-Mills gauge theory
    YangMills { gauge_group: String },
    /// Scalar field (Klein-Gordon, etc.)
    ScalarField { potential: String },
}

/// Types of equilibrium
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EquilibriumType {
    Thermodynamic,
    Mechanical,
    Chemical,
    Phase { phases: Vec<String> },
}

/// Boundary condition types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BoundaryType {
    Dirichlet,
    Neumann,
    Robin,
    Periodic,
    Absorbing,
    Inflow,
    Outflow { characteristic: String },
}

/// Provenance specific to physical laws
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalProvenance {
    /// Axiomatic: derived from first principles, mathematically proven
    Axiomatic {
        /// The fundamental principle (Noether's theorem, etc.)
        principle: String,
        /// Mathematical derivation reference
        derivation: String,
    },

    /// Empirical: validated by experimental data
    Empirical {
        /// Experimental dataset reference
        experiment: String,
        /// Validation confidence level
        validation_confidence: f64,
        /// Reproducibility score
        reproducibility: f64,
    },

    /// Learned: data-driven approximation
    Learned {
        /// Training methodology
        method: LearningMethod,
        /// Training data provenance
        data: Box<Provenance>,
        /// PAC bounds if available
        pac_bounds: Option<PacBounds>,
        /// Cross-validation score
        cv_score: f64,
    },

    /// Hybrid: combination of physics and learning
    Hybrid {
        /// The physical component
        physics: Box<PhysicalProvenance>,
        /// The learned component
        learned: Box<PhysicalProvenance>,
        /// Composition strategy
        composition: HybridComposition,
    },

    /// Assumed: working hypothesis, not yet validated
    Assumed {
        /// Rationale for assumption
        rationale: String,
        /// Confidence in assumption
        confidence: f64,
    },
}

/// Learning methods for physical operators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LearningMethod {
    /// Physics-Informed Neural Networks (Raissi et al.)
    PINN,
    /// Neural Operators (FNO, DeepONet)
    NeuralOperator { architecture: String },
    /// Gaussian Process regression
    GaussianProcess,
    /// Symbolic regression (SR)
    SymbolicRegression,
    /// Reinforcement learning
    ReinforcementLearning { algorithm: String },
}

/// PAC (Probably Approximately Correct) learning bounds
#[derive(Debug, Clone, PartialEq)]
pub struct PacBounds {
    /// Confidence parameter (1 - δ)
    pub confidence: f64,
    /// Accuracy parameter (ε)
    pub accuracy: f64,
    /// Sample complexity
    pub sample_size: u64,
    /// VC dimension or Rademacher complexity
    pub complexity_measure: ComplexityMeasure,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityMeasure {
    VcDimension(u64),
    Rademacher(f64),
    CoveringNumber { epsilon: f64, number: u64 },
}

/// Composition strategies for hybrid models
#[derive(Debug, Clone, PartialEq)]
pub enum HybridComposition {
    /// Physics residual as regularization: L = L_data + λ*L_physics
    ResidualRegularization { weight: f64 },
    /// Physics-constrained architecture (e.g., divergence-free networks)
    ArchitectureConstraint,
    /// Multi-fidelity: coarse physics + learned correction
    MultiFidelity,
    /// Domain decomposition: physics in some regions, learning in others
    DomainDecomposition { interface: String },
}

/// Parameters of a physical law that may be learned or measured
#[derive(Debug, Clone, PartialEq)]
pub struct LawParameter {
    /// Parameter name
    pub name: String,
    /// Physical dimension (using Sounio's unit system)
    pub dimension: String, // Would be Unit type in full implementation
    /// Current value (if known)
    pub value: Option<f64>,
    /// Epistemic status of this parameter
    pub epistemic: ParameterEpistemic,
    /// Bounds/constraints on the parameter
    pub constraints: Vec<ParameterConstraint>,
}

/// Epistemic status of a parameter
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterEpistemic {
    /// Known constant (e.g., speed of light)
    Constant { value: f64, uncertainty: f64 },
    /// Measured value with uncertainty
    Measured {
        value: f64,
        standard_error: f64,
        confidence_interval: (f64, f64),
        n_samples: u64,
    },
    /// Learned from data
    Learned {
        point_estimate: f64,
        posterior_variance: f64,
        credible_interval: (f64, f64),
    },
    /// Calibrated via inverse problem
    Calibrated {
        initial_guess: f64,
        calibration_data: String,
        final_value: f64,
        misfit: f64,
    },
}

/// Constraints on parameters
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterConstraint {
    /// Physical bound (e.g., viscosity > 0)
    PhysicalBound { lower: Option<f64>, upper: Option<f64> },
    /// Thermodynamic consistency (e.g., second law)
    ThermodynamicConsistency,
    /// Dimensional consistency
    DimensionalConsistency,
    /// Empirical correlation
    EmpiricalCorrelation { relation: String },
}

/// Temporal validity constraint for physical laws
#[derive(Debug, Clone, PartialEq, Default)]
pub enum TemporalConstraint {
    /// Knowledge valid for at most N seconds
    MaxAge(u64),
    /// Knowledge valid after timestamp
    ValidAfter(u64),
    /// Knowledge valid within date range
    ValidBetween(u64, u64),
    /// Always valid (no temporal constraint)
    #[default]
    Eternal,
}

/// Domain of validity for a physical law
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DomainOfValidity {
    /// Spatial domain constraints
    pub spatial: Vec<SpatialConstraint>,
    /// Temporal validity
    pub temporal: TemporalConstraint,
    /// Physical regime (laminar/turbulent, relativistic/classical, etc.)
    pub regime: Vec<RegimeConstraint>,
    /// Scale constraints (micro/meso/macro)
    pub scale: Vec<ScaleConstraint>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpatialConstraint {
    BoundingBox { x: (f64, f64), y: (f64, f64), z: (f64, f64) },
    Region { ontology_ref: String }, // Reference to spatial ontology
    ExcludedRegion { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegimeConstraint {
    ReynoldsNumber { min: Option<f64>, max: Option<f64> },
    MachNumber { min: Option<f64>, max: Option<f64> },
    KnudsenNumber { min: Option<f64>, max: Option<f64> },
    Custom { parameter: String, range: (f64, f64) },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScaleConstraint {
    Continuum,      // Kn << 1
    Mesoscopic,     // Intermediate
    Molecular,      // Kn >> 1
    Quantum,        // ħ effects dominant
    Relativistic,   // c effects dominant
}

// =============================================================================
// LearnedOperator - Neural operators with epistemic tracking
// =============================================================================

/// LearnedOperator[O, ε, Φ] - Data-driven physical operators
///
/// Unlike black-box neural networks, LearnedOperators track:
/// - What physical quantity they approximate
/// - Confidence bounds from PAC learning
/// - Domain of applicability
/// - Uncertainty quantification
#[derive(Debug, Clone, PartialEq)]
pub struct LearnedOperator {
    /// What this operator approximates
    pub target: OperatorTarget,

    /// Neural architecture used
    pub architecture: NeuralArchitecture,

    /// Epistemic status (confidence from training)
    pub epistemic: EpistemicStatus,

    /// Training provenance
    pub training: TrainingProvenance,

    /// Domain where the operator is valid
    pub domain: DomainOfValidity,

    /// Uncertainty quantification method
    pub uncertainty: UqMethod,
}

/// What the learned operator approximates
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperatorTarget {
    /// Direct approximation of a field
    Field { quantity: String, domain: String },
    /// Closure model (turbulence, subgrid, etc.)
    Closure { for_model: String, term: String },
    /// Constitutive relation
    Constitutive { input: String, output: String },
    /// Source term
    Source { process: String },
    /// Boundary condition
    Boundary { condition_type: String },
    /// Inverse mapping (e.g., parameter inference)
    Inverse { forward_model: String },
}

/// Neural architectures for physical operators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NeuralArchitecture {
    /// Fourier Neural Operator (Li et al. 2021)
    FNO { modes: u32, width: u32 },
    /// DeepONet (Lu et al. 2021)
    DeepONet { branch_dim: u32, trunk_dim: u32 },
    /// Physics-Informed Neural Network
    PINN { layers: Vec<u32>, activation: String },
    /// Graph Neural Network
    GNN { message_passing_steps: u32 },
    /// Transformer-based operator
    Transformer { heads: u32, layers: u32 },
    /// Custom architecture
    Custom { name: String, description: String },
}

/// Training provenance for learned operators
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingProvenance {
    /// Dataset used for training
    pub dataset: String,
    /// Data provenance
    pub data_source: Provenance,
    /// Training configuration
    pub config: TrainingConfig,
    /// Validation metrics
    pub validation: ValidationMetrics,
    /// Model checkpoint reference
    pub checkpoint: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingConfig {
    pub optimizer: String,
    pub learning_rate: f64,
    pub epochs: u32,
    pub batch_size: u32,
    pub loss_function: LossFunction,
    pub physics_constraints: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LossFunction {
    MSE,
    MAE,
    Huber { delta: f64 },
    PhysicsInformed { data_weight: f64, physics_weight: f64 },
    Spectral { cutoff: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationMetrics {
    pub train_error: f64,
    pub validation_error: f64,
    pub test_error: f64,
    pub generalization_gap: f64,
    pub physics_residual: f64,
}

/// Uncertainty quantification methods
#[derive(Debug, Clone, PartialEq)]
pub enum UqMethod {
    /// Ensemble methods
    Ensemble { n_models: u32 },
    /// Bayesian neural networks
    BNN { inference: String },
    /// Monte Carlo Dropout
    MCDropout { drop_rate: f64 },
    /// Deep Ensembles with adversarial training
    DeepEnsemble,
    /// Spectral normalization bounds
    SpectralNormalization,
    /// Conformal prediction
    Conformal { coverage: f64 },
}

// =============================================================================
// HybridOperator - Physics-guided learning
// =============================================================================

/// HybridOperator - Composition of physical laws and learned operators
///
/// This is the core abstraction for physics-informed machine learning in Sounio.
/// It enables seamless composition of axiomatic physics with data-driven components
/// while tracking epistemic uncertainty throughout.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridOperator {
    /// Name/identifier for this hybrid model
    pub name: String,

    /// The physical component (may be None for pure data-driven)
    pub physics: Option<PhysicalLaw>,

    /// The learned component (may be None for pure physics)
    pub learned: Option<LearnedOperator>,

    /// Composition strategy
    pub composition: HybridComposition,

    /// Coupling between components
    pub coupling: CouplingStrategy,

    /// Overall epistemic status (computed from components)
    pub epistemic: EpistemicStatus,
}

/// How physics and learning are coupled
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingStrategy {
    /// Sequential: physics → learning or learning → physics
    Sequential { order: Vec<String> },
    /// Iterative: alternating physics and learning steps
    Iterative { max_iterations: u32, tolerance: f64 },
    /// Coupled: simultaneous solution
    Coupled { coupling_strength: f64 },
    /// Hierarchical: multi-resolution composition
    Hierarchical { levels: Vec<String> },
}

// =============================================================================
// PhysicalLaw Implementation
// =============================================================================

impl PhysicalLaw {
    /// Create a new axiomatic physical law
    pub fn axiomatic(law_type: LawType, principle: impl Into<String>) -> Self {
        Self {
            law_type,
            epistemic: EpistemicStatus::axiomatic(),
            provenance: PhysicalProvenance::Axiomatic {
                principle: principle.into(),
                derivation: String::new(),
            },
            parameters: Vec::new(),
            domain_of_validity: DomainOfValidity::default(),
            span: Span::default(),
        }
    }

    /// Create a physical law from empirical validation
    pub fn empirical(
        law_type: LawType,
        experiment: impl Into<String>,
        validation_confidence: f64,
    ) -> Self {
        Self {
            law_type,
            epistemic: EpistemicStatus {
                confidence: Confidence::new(validation_confidence),
                ..EpistemicStatus::default()
            },
            provenance: PhysicalProvenance::Empirical {
                experiment: experiment.into(),
                validation_confidence,
                reproducibility: validation_confidence,
            },
            parameters: Vec::new(),
            domain_of_validity: DomainOfValidity::default(),
            span: Span::default(),
        }
    }

    /// Create a learned surrogate for a physical law
    pub fn learned_surrogate(
        name: impl Into<String>,
        approximates: LawType,
        training_data: Provenance,
        cv_score: f64,
    ) -> Self {
        let name = name.into();
        Self {
            law_type: LawType::LearnedSurrogate {
                name: name.clone(),
                approximates: Box::new(approximates),
                training_data: training_data.clone(),
            },
            epistemic: EpistemicStatus {
                confidence: Confidence::new(cv_score),
                ..EpistemicStatus::default()
            },
            provenance: PhysicalProvenance::Learned {
                method: LearningMethod::PINN,
                data: Box::new(training_data),
                pac_bounds: None,
                cv_score,
            },
            parameters: Vec::new(),
            domain_of_validity: DomainOfValidity::default(),
            span: Span::default(),
        }
    }

    /// Add a parameter to this law
    pub fn with_parameter(mut self, param: LawParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set the domain of validity
    pub fn with_domain(mut self, domain: DomainOfValidity) -> Self {
        self.domain_of_validity = domain;
        self
    }

    /// Check if this law is axiomatic (100% confidence)
    pub fn is_axiomatic(&self) -> bool {
        matches!(self.provenance, PhysicalProvenance::Axiomatic { .. })
    }

    /// Check if this law is learned
    pub fn is_learned(&self) -> bool {
        matches!(self.provenance, PhysicalProvenance::Learned { .. })
    }

    /// Get confidence level
    pub fn confidence(&self) -> f64 {
        self.epistemic.confidence.value()
    }

    /// Compose with a learned operator to create a hybrid
    pub fn compose_with(self, learned: LearnedOperator, strategy: HybridComposition) -> HybridOperator {
        let confidence = self.confidence() * learned.confidence();
        HybridOperator {
            name: format!("Hybrid_{}", self.law_type_name()),
            physics: Some(self),
            learned: Some(learned),
            composition: strategy,
            coupling: CouplingStrategy::Coupled { coupling_strength: 0.5 },
            epistemic: EpistemicStatus {
                confidence: Confidence::new(confidence),
                ..EpistemicStatus::default()
            },
        }
    }

    fn law_type_name(&self) -> String {
        match &self.law_type {
            LawType::Conservation(q) => format!("{:?}", q),
            LawType::Constitutive(c) => format!("{:?}", c),
            LawType::FieldEquation(f) => format!("{:?}", f),
            LawType::RateEquation(s) => s.clone(),
            LawType::Equilibrium(e) => format!("{:?}", e),
            LawType::BoundaryCondition(b) => format!("{:?}", b),
            LawType::LearnedSurrogate { name, .. } => name.clone(),
        }
    }
}

// =============================================================================
// LearnedOperator Implementation
// =============================================================================

impl LearnedOperator {
    /// Create a new learned operator with FNO architecture
    pub fn fno(
        target: OperatorTarget,
        modes: u32,
        width: u32,
        dataset: impl Into<String>,
    ) -> Self {
        Self {
            target,
            architecture: NeuralArchitecture::FNO { modes, width },
            epistemic: EpistemicStatus::default(),
            training: TrainingProvenance {
                dataset: dataset.into(),
                data_source: Provenance::computed("synthetic"),
                config: TrainingConfig {
                    optimizer: "Adam".to_string(),
                    learning_rate: 1e-3,
                    epochs: 1000,
                    batch_size: 32,
                    loss_function: LossFunction::MSE,
                    physics_constraints: Vec::new(),
                },
                validation: ValidationMetrics {
                    train_error: 0.0,
                    validation_error: 0.0,
                    test_error: 0.0,
                    generalization_gap: 0.0,
                    physics_residual: 0.0,
                },
                checkpoint: String::new(),
            },
            domain: DomainOfValidity::default(),
            uncertainty: UqMethod::DeepEnsemble,
        }
    }

    /// Get confidence from validation metrics
    pub fn confidence(&self) -> f64 {
        1.0 - self.training.validation.test_error.min(1.0)
    }

    /// Add physics constraints to training
    pub fn with_physics_constraint(mut self, constraint: impl Into<String>) -> Self {
        self.training.config.physics_constraints.push(constraint.into());
        self
    }

    /// Set uncertainty quantification method
    pub fn with_uq(mut self, uq: UqMethod) -> Self {
        self.uncertainty = uq;
        self
    }
}

// =============================================================================
// HybridOperator Implementation
// =============================================================================

impl HybridOperator {
    /// Create a new hybrid operator from physics and learned components
    pub fn compose(physics: PhysicalLaw, learned: LearnedOperator) -> Self {
        let confidence = physics.confidence() * learned.confidence();
        Self {
            name: format!("{}_{}", physics.law_type_name(), learned.target_name()),
            physics: Some(physics),
            learned: Some(learned),
            composition: HybridComposition::ResidualRegularization { weight: 1.0 },
            coupling: CouplingStrategy::Coupled { coupling_strength: 0.5 },
            epistemic: EpistemicStatus {
                confidence: Confidence::new(confidence),
                ..EpistemicStatus::default()
            },
        }
    }

    /// Create physics-only operator
    pub fn physics_only(physics: PhysicalLaw) -> Self {
        Self {
            name: physics.law_type_name(),
            physics: Some(physics),
            learned: None,
            composition: HybridComposition::ArchitectureConstraint,
            coupling: CouplingStrategy::Sequential { order: vec!["physics".to_string()] },
            epistemic: EpistemicStatus::axiomatic(),
        }
    }

    /// Create learned-only operator (pure data-driven)
    pub fn learned_only(learned: LearnedOperator) -> Self {
        let confidence = learned.confidence();
        Self {
            name: learned.target_name(),
            physics: None,
            learned: Some(learned),
            composition: HybridComposition::ResidualRegularization { weight: 0.0 },
            coupling: CouplingStrategy::Sequential { order: vec!["learned".to_string()] },
            epistemic: EpistemicStatus {
                confidence: Confidence::new(confidence),
                ..EpistemicStatus::default()
            },
        }
    }

    /// Set composition strategy
    pub fn with_composition(mut self, composition: HybridComposition) -> Self {
        self.composition = composition;
        self
    }

    /// Get overall confidence
    pub fn confidence(&self) -> f64 {
        self.epistemic.confidence.value()
    }
}

impl LearnedOperator {
    fn target_name(&self) -> String {
        format!("{:?}", self.target)
    }
}

// =============================================================================
// Type System Integration
// =============================================================================

/// PhysicalLawType for the Sounio type system
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalLawType {
    /// Concrete physical law
    Concrete(PhysicalLaw),
    /// Existential (some parameters unknown)
    Existential {
        law_kind: Box<LawType>,
        known_parameters: Vec<String>,
    },
    /// Universal (polymorphic over physical regimes)
    Universal {
        law_template: Box<LawType>,
        regime_var: String,
    },
}

/// Integration with Knowledge types
impl PhysicalLaw {
    /// Convert to Knowledge type for values computed by this law
    pub fn to_knowledge_type(&self, content_type: Type) -> crate::types::KnowledgeType {
        crate::types::KnowledgeType::from_pac_learning(
            content_type,
            self.confidence(),
            1.0 - self.confidence(), // epsilon as error bound
            0, // Would track actual sample size
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axiomatic_law_creation() {
        let ns = PhysicalLaw::axiomatic(
            LawType::Conservation(ConservationQuantity::Momentum),
            "Noether's theorem: spatial translation symmetry",
        );

        assert!(ns.is_axiomatic());
        assert_eq!(ns.confidence(), 1.0);
        assert!(matches!(
            ns.provenance,
            PhysicalProvenance::Axiomatic { principle, .. } if principle.contains("Noether")
        ));
    }

    #[test]
    fn test_learned_operator_confidence() {
        let fno = LearnedOperator::fno(
            OperatorTarget::Closure { for_model: "RANS".to_string(), term: "turbulence".to_string() },
            12,
            64,
            "turbulence_dataset_v2",
        );

        assert!(fno.confidence() >= 0.0 && fno.confidence() <= 1.0);
    }

    #[test]
    fn test_hybrid_composition_confidence() {
        let physics = PhysicalLaw::axiomatic(
            LawType::Conservation(ConservationQuantity::Mass),
            "Mass conservation",
        );

        let learned = LearnedOperator::fno(
            OperatorTarget::Closure { for_model: "NS".to_string(), term: "viscosity".to_string() },
            16,
            32,
            "flow_data",
        );

        let hybrid = HybridOperator::compose(physics, learned);

        // Hybrid confidence should be product of components
        assert!(hybrid.confidence() <= 1.0);
        assert!(hybrid.physics.is_some());
        assert!(hybrid.learned.is_some());
    }

    #[test]
    fn test_conservation_divergence_form() {
        let mass = ConservationQuantity::Mass;
        assert!(mass.divergence_form().contains("∂ρ/∂t"));

        let momentum = ConservationQuantity::Momentum;
        assert!(momentum.divergence_form().contains("∇·(ρv⊗v)"));
    }

    #[test]
    fn test_domain_of_validity() {
        let domain = DomainOfValidity {
            spatial: vec![SpatialConstraint::BoundingBox {
                x: (0.0, 1.0),
                y: (0.0, 1.0),
                z: (0.0, 0.0),
            }],
            temporal: TemporalConstraint::MaxAge(3600),
            regime: vec![RegimeConstraint::ReynoldsNumber {
                min: Some(100.0),
                max: Some(10000.0),
            }],
            scale: vec![ScaleConstraint::Continuum],
        };

        assert_eq!(domain.spatial.len(), 1);
        assert_eq!(domain.regime.len(), 1);
    }
}

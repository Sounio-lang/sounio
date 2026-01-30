//! Beta-Epistemic Knowledge Types
//!
//! Revolutionary extension of Sounio epistemic types with full Beta posterior
//! distribution instead of scalar confidence. This enables true uncertainty
//! quantification where variance is a first-class metric for "how much I know
//! that I don't know."
//!
//! # Key Innovation
//!
//! Traditional epistemic systems: confidence = 0.8 (a point estimate)
//! Beta-epistemic systems: confidence ~ Beta(8, 2) with mean=0.8, variance=0.015
//!
//! The variance tells us HOW CERTAIN we are about being 80% confident.
//! Low variance (many observations) → we trust the 80%
//! High variance (few observations) → we should explore more (active inference)
//!
//! # Active Inference Integration
//!
//! The variance directly enables active inference:
//! - High variance → high expected information gain → should acquire more data
//! - Variance reduction is an optimization objective
//! - Ignorance-driven exploration becomes native to the type system
//!
//! # Hierarchical Combination
//!
//! When combining knowledge from multiple sources:
//! - Sources with lower variance are weighted more heavily
//! - Hierarchical priors allow domain-specific weighting
//! - Provenance tracking enables source-specific decay models

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use super::bayesian::BetaConfidence;
use super::confidence::{Evidence, Revisability, Source};
use super::merkle::{MerkleProvenanceNode, OperationKind, ProvenanceOperation};
use super::provenance::{Transformation, TransformationKind};

// =============================================================================
// Decay Models - Pluggable confidence decay strategies
// =============================================================================

/// Decay model for confidence propagation through transformations
///
/// Different operations have different reliability characteristics.
/// Decay models capture this domain knowledge.
#[derive(Debug, Clone, PartialEq)]
pub enum DecayModel {
    /// Linear decay: new_conf = old_conf * factor
    Linear { factor: f64 },

    /// Exponential decay: new_conf = old_conf^depth
    Exponential { base: f64 },

    /// Logarithmic decay: slower decay for well-established knowledge
    Logarithmic { rate: f64 },

    /// Bayesian decay: update based on transformation likelihood
    Bayesian {
        /// Prior belief in transformation correctness
        prior_success: f64,
        /// Strength of prior (pseudo-observations)
        prior_strength: f64,
    },

    /// No decay (perfect transformation)
    Perfect,

    /// Custom decay function
    Custom {
        name: String,
        /// Parameters for the custom function
        params: HashMap<String, f64>,
    },
}

impl Default for DecayModel {
    fn default() -> Self {
        DecayModel::Linear { factor: 0.95 }
    }
}

impl DecayModel {
    /// Apply decay to a BetaConfidence
    pub fn apply(&self, conf: &BetaConfidence, depth: usize) -> BetaConfidence {
        match self {
            DecayModel::Linear { factor } => {
                // Linear decay: scale mean while increasing variance
                let decay = factor.powi(depth as i32);
                let new_alpha = conf.alpha * decay;
                let new_beta = conf.beta + (1.0 - decay) * conf.alpha;
                BetaConfidence::new(new_alpha.max(0.5), new_beta.max(0.5))
            }

            DecayModel::Exponential { base } => {
                // Exponential decay: faster falloff for uncertain transformations
                let decay = base.powf(depth as f64);
                let effective_n = conf.sample_size() * decay;
                BetaConfidence::from_confidence(conf.mean() * decay, effective_n.max(2.0))
            }

            DecayModel::Logarithmic { rate } => {
                // Logarithmic decay: slow decay for established knowledge
                let decay = 1.0 / (1.0 + rate * (depth as f64).ln_1p());
                BetaConfidence::from_confidence(conf.mean() * decay, conf.sample_size())
            }

            DecayModel::Bayesian {
                prior_success,
                prior_strength,
            } => {
                // Bayesian decay: treat transformation as another observation
                let transformation_success = *prior_success;
                let new_alpha = conf.alpha + transformation_success * prior_strength;
                let new_beta = conf.beta + (1.0 - transformation_success) * prior_strength;
                BetaConfidence::new(new_alpha, new_beta)
            }

            DecayModel::Perfect => *conf,

            DecayModel::Custom { .. } => {
                // Custom decay - default to linear for now
                DecayModel::Linear { factor: 0.95 }.apply(conf, depth)
            }
        }
    }

    /// Get the expected variance increase from this decay model
    pub fn expected_variance_increase(&self, depth: usize) -> f64 {
        match self {
            DecayModel::Linear { factor } => (1.0 - factor.powi(depth as i32)) * 0.1,
            DecayModel::Exponential { base } => (1.0 - base.powf(depth as f64)) * 0.15,
            DecayModel::Logarithmic { rate } => rate * (depth as f64).ln_1p() * 0.05,
            DecayModel::Bayesian { prior_strength, .. } => 1.0 / (prior_strength + depth as f64),
            DecayModel::Perfect => 0.0,
            DecayModel::Custom { .. } => 0.1 * depth as f64,
        }
    }
}

// =============================================================================
// Prior Types - Pluggable priors for hierarchical Bayesian inference
// =============================================================================

/// Prior distribution type for hierarchical combination
#[derive(Debug, Clone, PartialEq, Default)]
pub enum PriorType {
    /// Uniform prior Beta(1, 1) - maximum ignorance
    #[default]
    Uniform,

    /// Jeffreys prior Beta(0.5, 0.5) - uninformative
    Jeffreys,

    /// Weak prior centered at value
    Weak { center: f64, strength: f64 },

    /// Strong prior from domain knowledge
    Strong { center: f64, strength: f64 },

    /// Source-driven prior based on source reliability
    SourceDriven {
        source_type: SourcePriorType,
        base_strength: f64,
    },

    /// Empirical prior from historical data
    Empirical {
        alpha: f64,
        beta: f64,
        source: String,
    },

    /// Hierarchical prior with multiple levels
    Hierarchical {
        levels: Vec<BetaConfidence>,
        weights: Vec<f64>,
    },
}

/// Source-specific prior types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourcePriorType {
    /// Experimental measurement - high reliability
    Experimental,
    /// Computational result - medium-high reliability
    Computational,
    /// Model prediction - medium reliability
    ModelPrediction,
    /// Human assertion - variable reliability
    HumanAssertion,
    /// External source - low-medium reliability
    External,
    /// Unknown - low reliability
    Unknown,
}

impl PriorType {
    /// Convert to BetaConfidence
    pub fn to_beta(&self) -> BetaConfidence {
        match self {
            PriorType::Uniform => BetaConfidence::uniform_prior(),
            PriorType::Jeffreys => BetaConfidence::jeffreys_prior(),
            PriorType::Weak { center, strength } => BetaConfidence::weak_prior(*center, *strength),
            PriorType::Strong { center, strength } => {
                BetaConfidence::strong_prior(*center, *strength)
            }
            PriorType::SourceDriven {
                source_type,
                base_strength,
            } => {
                let (center, multiplier) = match source_type {
                    SourcePriorType::Experimental => (0.95, 2.0),
                    SourcePriorType::Computational => (0.90, 1.5),
                    SourcePriorType::ModelPrediction => (0.75, 1.0),
                    SourcePriorType::HumanAssertion => (0.70, 0.8),
                    SourcePriorType::External => (0.60, 0.6),
                    SourcePriorType::Unknown => (0.50, 0.5),
                };
                BetaConfidence::weak_prior(center, base_strength * multiplier)
            }
            PriorType::Empirical { alpha, beta, .. } => BetaConfidence::new(*alpha, *beta),
            PriorType::Hierarchical { levels, weights } => {
                // Weighted combination of hierarchical levels
                if levels.is_empty() {
                    return BetaConfidence::uniform_prior();
                }

                let total_weight: f64 = weights.iter().sum();
                let mut combined_alpha = 0.0;
                let mut combined_beta = 0.0;

                for (level, weight) in levels.iter().zip(weights.iter()) {
                    let normalized_weight = weight / total_weight;
                    combined_alpha += level.alpha * normalized_weight;
                    combined_beta += level.beta * normalized_weight;
                }

                BetaConfidence::new(combined_alpha.max(0.5), combined_beta.max(0.5))
            }
        }
    }

    /// Get the effective sample size of this prior
    pub fn effective_sample_size(&self) -> f64 {
        self.to_beta().sample_size()
    }
}

// =============================================================================
// BetaEpistemicStatus - Full Beta posterior epistemic status
// =============================================================================

/// Complete epistemic status with Beta posterior confidence
///
/// This is the revolutionary replacement for scalar confidence.
/// Every piece of knowledge now carries its full uncertainty distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct BetaEpistemicStatus {
    /// Full Beta posterior confidence distribution
    pub confidence: BetaConfidence,

    /// Can this be revised with new evidence?
    pub revisability: Revisability,

    /// Where did this knowledge originate?
    pub source: Source,

    /// Evidence chain supporting this knowledge
    pub evidence: Vec<Evidence>,

    /// Decay model for propagation
    pub decay_model: DecayModel,

    /// Prior type used for this knowledge
    pub prior: PriorType,

    /// Transformation depth (for decay tracking)
    pub transformation_depth: usize,

    /// Active inference metrics
    pub active_inference: ActiveInferenceMetrics,
}

/// Metrics for active inference / ignorance-driven exploration
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ActiveInferenceMetrics {
    /// Expected information gain from acquiring more data
    pub expected_info_gain: f64,

    /// Variance reduction achieved so far
    pub variance_reduction: f64,

    /// Number of observations contributing to this knowledge
    pub observation_count: usize,

    /// Entropy of the confidence distribution
    pub entropy: f64,

    /// Free energy (for active inference optimization)
    pub free_energy: f64,
}

impl ActiveInferenceMetrics {
    /// Compute metrics from a BetaConfidence
    pub fn from_beta(beta: &BetaConfidence) -> Self {
        let variance = beta.variance();
        let entropy = beta.entropy();

        // Expected information gain approximation
        // Higher variance = more potential information gain
        let expected_info_gain = variance.sqrt() * 2.0;

        // Free energy (negative log evidence + KL divergence from prior)
        // Simplified: entropy + (1 - mean)^2 as a proxy for model complexity
        let free_energy = entropy + (1.0 - beta.mean()).powi(2);

        Self {
            expected_info_gain,
            variance_reduction: 0.0, // Needs comparison to compute
            observation_count: (beta.sample_size() - 2.0).max(0.0) as usize,
            entropy,
            free_energy,
        }
    }

    /// Update metrics after new observation
    pub fn update_with_observation(
        &mut self,
        old_beta: &BetaConfidence,
        new_beta: &BetaConfidence,
    ) {
        let old_variance = old_beta.variance();
        let new_variance = new_beta.variance();

        self.variance_reduction = old_variance - new_variance;
        self.observation_count += 1;
        self.entropy = new_beta.entropy();
        self.expected_info_gain = new_variance.sqrt() * 2.0;
        self.free_energy = self.entropy + (1.0 - new_beta.mean()).powi(2);
    }

    /// Should we acquire more data? (Active inference decision)
    pub fn should_acquire_data(&self, threshold: f64) -> bool {
        self.expected_info_gain > threshold
    }

    /// How much should we prioritize this for exploration?
    pub fn exploration_priority(&self) -> f64 {
        // Higher variance + lower observation count = higher priority
        self.expected_info_gain * (1.0 + 1.0 / (self.observation_count as f64 + 1.0))
    }
}

impl BetaEpistemicStatus {
    /// Create axiomatic knowledge - certain, non-revisable, from definition
    pub fn axiomatic() -> Self {
        let confidence = BetaConfidence::new(1000.0, 1.0); // Very high alpha
        Self {
            confidence,
            revisability: Revisability::NonRevisable,
            source: Source::Axiom,
            evidence: vec![],
            decay_model: DecayModel::Perfect,
            prior: PriorType::Strong {
                center: 1.0,
                strength: 1000.0,
            },
            transformation_depth: 0,
            active_inference: ActiveInferenceMetrics::from_beta(&confidence),
        }
    }

    /// Create empirical knowledge from observations
    pub fn empirical(successes: f64, failures: f64, source: Source) -> Self {
        let prior = PriorType::SourceDriven {
            source_type: source_to_prior_type(&source),
            base_strength: 2.0,
        };
        let prior_beta = prior.to_beta();
        let confidence =
            BetaConfidence::new(prior_beta.alpha + successes, prior_beta.beta + failures);

        Self {
            confidence,
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".into()],
            },
            source,
            evidence: vec![],
            decay_model: DecayModel::Linear { factor: 0.95 },
            prior,
            transformation_depth: 0,
            active_inference: ActiveInferenceMetrics::from_beta(&confidence),
        }
    }

    /// Create from scalar confidence with effective sample size
    pub fn from_confidence(mean: f64, effective_n: f64, source: Source) -> Self {
        let confidence = BetaConfidence::from_confidence(mean, effective_n);
        let prior = PriorType::Uniform;

        Self {
            confidence,
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".into()],
            },
            source,
            evidence: vec![],
            decay_model: DecayModel::default(),
            prior,
            transformation_depth: 0,
            active_inference: ActiveInferenceMetrics::from_beta(&confidence),
        }
    }

    /// Create with full control over all parameters
    pub fn new(
        confidence: BetaConfidence,
        revisability: Revisability,
        source: Source,
        decay_model: DecayModel,
        prior: PriorType,
    ) -> Self {
        Self {
            confidence,
            revisability,
            source,
            evidence: vec![],
            decay_model,
            prior,
            transformation_depth: 0,
            active_inference: ActiveInferenceMetrics::from_beta(&confidence),
        }
    }

    /// Get the mean confidence (point estimate)
    pub fn mean(&self) -> f64 {
        self.confidence.mean()
    }

    /// Get the variance (uncertainty about confidence)
    pub fn variance(&self) -> f64 {
        self.confidence.variance()
    }

    /// Get the standard deviation
    pub fn std_dev(&self) -> f64 {
        self.confidence.std_dev()
    }

    /// Get epistemic uncertainty (sqrt of variance)
    /// This is "how much I know that I don't know"
    pub fn epistemic_uncertainty(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get 95% credible interval
    pub fn credible_interval_95(&self) -> (f64, f64) {
        self.confidence.credible_interval(0.95)
    }

    /// Update with new observation
    pub fn observe(&mut self, success: bool) {
        let old_beta = self.confidence;

        if success {
            self.confidence =
                BetaConfidence::new(self.confidence.alpha + 1.0, self.confidence.beta);
        } else {
            self.confidence =
                BetaConfidence::new(self.confidence.alpha, self.confidence.beta + 1.0);
        }

        self.active_inference
            .update_with_observation(&old_beta, &self.confidence);
    }

    /// Update with weighted evidence
    pub fn observe_weighted(&mut self, success_weight: f64, failure_weight: f64) {
        let old_beta = self.confidence;

        self.confidence = BetaConfidence::new(
            self.confidence.alpha + success_weight,
            self.confidence.beta + failure_weight,
        );

        self.active_inference
            .update_with_observation(&old_beta, &self.confidence);
    }

    /// Propagate through a transformation
    pub fn propagate(&self, transformation: &Transformation) -> Self {
        let new_depth = self.transformation_depth + 1;
        let new_confidence = self.decay_model.apply(&self.confidence, new_depth);

        Self {
            confidence: new_confidence,
            revisability: self.revisability.clone(),
            source: Source::Transformation {
                original: Box::new(self.source.clone()),
                via: transformation.name().to_string(),
            },
            evidence: self.evidence.clone(),
            decay_model: self.decay_model.clone(),
            prior: self.prior.clone(),
            transformation_depth: new_depth,
            active_inference: ActiveInferenceMetrics::from_beta(&new_confidence),
        }
    }

    /// Combine with another epistemic status (for conjunction)
    pub fn combine(&self, other: &BetaEpistemicStatus) -> Self {
        // Use variance-weighted combination: lower variance = higher weight
        let self_weight = 1.0 / (self.variance() + 0.001);
        let other_weight = 1.0 / (other.variance() + 0.001);
        let total_weight = self_weight + other_weight;

        let combined_alpha = (self.confidence.alpha * self_weight
            + other.confidence.alpha * other_weight)
            / total_weight;
        let combined_beta = (self.confidence.beta * self_weight
            + other.confidence.beta * other_weight)
            / total_weight;

        let confidence = BetaConfidence::new(combined_alpha, combined_beta);

        // Revisability: revisable if any source is revisable
        let revisability = if self.revisability.is_revisable() || other.revisability.is_revisable()
        {
            Revisability::Revisable {
                conditions: vec!["revision of either source".into()],
            }
        } else {
            Revisability::NonRevisable
        };

        Self {
            confidence,
            revisability,
            source: Source::Derivation("combined".to_string()),
            evidence: [self.evidence.clone(), other.evidence.clone()].concat(),
            decay_model: self.decay_model.clone(), // Use self's decay model
            prior: PriorType::Hierarchical {
                levels: vec![self.prior.to_beta(), other.prior.to_beta()],
                weights: vec![self_weight, other_weight],
            },
            transformation_depth: self.transformation_depth.max(other.transformation_depth),
            active_inference: ActiveInferenceMetrics::from_beta(&confidence),
        }
    }

    /// Combine multiple sources hierarchically with a prior
    pub fn combine_hierarchical(sources: &[&BetaEpistemicStatus], prior: PriorType) -> Self {
        if sources.is_empty() {
            return Self::from_confidence(0.5, 2.0, Source::Unknown);
        }

        if sources.len() == 1 {
            return sources[0].clone();
        }

        // Compute variance-based weights
        let weights: Vec<f64> = sources
            .iter()
            .map(|s| 1.0 / (s.variance() + 0.001))
            .collect();
        let total_weight: f64 = weights.iter().sum();

        // Combine alphas and betas with weights
        let mut combined_alpha = prior.to_beta().alpha;
        let mut combined_beta = prior.to_beta().beta;

        for (source, weight) in sources.iter().zip(weights.iter()) {
            let normalized = weight / total_weight;
            combined_alpha += (source.confidence.alpha - 1.0) * normalized;
            combined_beta += (source.confidence.beta - 1.0) * normalized;
        }

        let confidence = BetaConfidence::new(combined_alpha.max(0.5), combined_beta.max(0.5));

        // Combine evidence
        let evidence: Vec<Evidence> = sources.iter().flat_map(|s| s.evidence.clone()).collect();

        // Max transformation depth
        let max_depth = sources
            .iter()
            .map(|s| s.transformation_depth)
            .max()
            .unwrap_or(0);

        Self {
            confidence,
            revisability: Revisability::Revisable {
                conditions: vec!["revision of any source".into()],
            },
            source: Source::Derivation("hierarchical_combination".to_string()),
            evidence,
            decay_model: DecayModel::default(),
            prior,
            transformation_depth: max_depth,
            active_inference: ActiveInferenceMetrics::from_beta(&confidence),
        }
    }

    /// Add evidence to this status
    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence.push(evidence);
        self
    }

    /// Set custom decay model
    pub fn with_decay_model(mut self, decay_model: DecayModel) -> Self {
        self.decay_model = decay_model;
        self
    }

    /// Check if we should acquire more data (active inference)
    pub fn should_explore(&self, info_gain_threshold: f64) -> bool {
        self.active_inference
            .should_acquire_data(info_gain_threshold)
    }

    /// Get exploration priority for active learning
    pub fn exploration_priority(&self) -> f64 {
        self.active_inference.exploration_priority()
    }

    /// Compute variance penalty for training loss
    pub fn variance_penalty(&self, lambda: f64) -> f64 {
        lambda * self.variance()
    }
}

impl Default for BetaEpistemicStatus {
    fn default() -> Self {
        Self::from_confidence(0.5, 2.0, Source::Unknown)
    }
}

impl fmt::Display for BetaEpistemicStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.credible_interval_95();
        write!(
            f,
            "ε(μ={:.3}, σ={:.3}, [{:.3}, {:.3}], n={:.1})",
            self.mean(),
            self.std_dev(),
            lo,
            hi,
            self.confidence.sample_size()
        )
    }
}

// =============================================================================
// BetaKnowledge - Knowledge with full Beta epistemic status
// =============================================================================

/// Knowledge value with full Beta posterior epistemic status
///
/// This is the revolutionary type that makes Sounio unique:
/// Every value knows not just its confidence, but the full uncertainty
/// distribution over that confidence.
#[derive(Debug, Clone)]
pub struct BetaKnowledge<T> {
    /// The wrapped value
    pub value: T,

    /// Full Beta epistemic status
    pub epistemic: BetaEpistemicStatus,

    /// Merkle provenance (for cryptographic audit trail)
    pub provenance: MerkleProvenanceNode,
}

impl<T: Clone> BetaKnowledge<T> {
    /// Create new knowledge with empirical confidence
    pub fn new(value: T, successes: f64, failures: f64, source: Source) -> Self {
        Self {
            value,
            epistemic: BetaEpistemicStatus::empirical(successes, failures, source),
            provenance: MerkleProvenanceNode::root(ProvenanceOperation::new(
                "beta_knowledge_creation",
                OperationKind::Literal,
            )),
        }
    }

    /// Create axiomatic knowledge (certain)
    pub fn axiomatic(value: T) -> Self {
        Self {
            value,
            epistemic: BetaEpistemicStatus::axiomatic(),
            provenance: MerkleProvenanceNode::root(ProvenanceOperation::new(
                "beta_knowledge_creation",
                OperationKind::Literal,
            )),
        }
    }

    /// Create from scalar confidence
    pub fn from_confidence(value: T, confidence: f64, effective_n: f64) -> Self {
        Self {
            value,
            epistemic: BetaEpistemicStatus::from_confidence(
                confidence,
                effective_n,
                Source::Unknown,
            ),
            provenance: MerkleProvenanceNode::root(ProvenanceOperation::new(
                "beta_knowledge_creation",
                OperationKind::Literal,
            )),
        }
    }

    /// Map the value while propagating epistemic status
    pub fn map<U: Clone, F: FnOnce(T) -> U>(
        self,
        f: F,
        transformation_name: &str,
    ) -> BetaKnowledge<U> {
        let transformation = Transformation::new(transformation_name, TransformationKind::Function);

        BetaKnowledge {
            value: f(self.value),
            epistemic: self.epistemic.propagate(&transformation),
            provenance: self.provenance.extend(transformation_name),
        }
    }

    /// Flat map (monadic bind)
    pub fn flat_map<U: Clone, F: FnOnce(T) -> BetaKnowledge<U>>(
        self,
        f: F,
        transformation_name: &str,
    ) -> BetaKnowledge<U> {
        let result = f(self.value);
        let transformation = Transformation::new(transformation_name, TransformationKind::Function);

        // Combine epistemic statuses
        let combined = self
            .epistemic
            .propagate(&transformation)
            .combine(&result.epistemic);

        BetaKnowledge {
            value: result.value,
            epistemic: combined,
            provenance: self
                .provenance
                .merge(&result.provenance, transformation_name),
        }
    }

    /// Get the mean confidence
    pub fn confidence(&self) -> f64 {
        self.epistemic.mean()
    }

    /// Get the variance (uncertainty about confidence)
    pub fn variance(&self) -> f64 {
        self.epistemic.variance()
    }

    /// Get the epistemic uncertainty
    pub fn epistemic_uncertainty(&self) -> f64 {
        self.epistemic.epistemic_uncertainty()
    }

    /// Get 95% credible interval
    pub fn credible_interval(&self) -> (f64, f64) {
        self.epistemic.credible_interval_95()
    }

    /// Update with new observation
    pub fn observe(&mut self, success: bool) {
        self.epistemic.observe(success);
    }

    /// Should we explore to reduce uncertainty?
    pub fn should_explore(&self, threshold: f64) -> bool {
        self.epistemic.should_explore(threshold)
    }

    /// Get exploration priority for active learning
    pub fn exploration_priority(&self) -> f64 {
        self.epistemic.exploration_priority()
    }
}

// =============================================================================
// Arithmetic Operations with Beta Propagation
// =============================================================================

impl<T: Add<Output = T> + Clone> Add for BetaKnowledge<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.value.clone() + rhs.value.clone();
        let epistemic = self.epistemic.combine(&rhs.epistemic);
        let provenance = self.provenance.merge(&rhs.provenance, "add");

        BetaKnowledge {
            value,
            epistemic,
            provenance,
        }
    }
}

impl<T: Sub<Output = T> + Clone> Sub for BetaKnowledge<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.value.clone() - rhs.value.clone();
        let epistemic = self.epistemic.combine(&rhs.epistemic);
        let provenance = self.provenance.merge(&rhs.provenance, "sub");

        BetaKnowledge {
            value,
            epistemic,
            provenance,
        }
    }
}

impl<T: Mul<Output = T> + Clone> Mul for BetaKnowledge<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let value = self.value.clone() * rhs.value.clone();
        let epistemic = self.epistemic.combine(&rhs.epistemic);
        let provenance = self.provenance.merge(&rhs.provenance, "mul");

        BetaKnowledge {
            value,
            epistemic,
            provenance,
        }
    }
}

impl<T: Div<Output = T> + Clone> Div for BetaKnowledge<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let value = self.value.clone() / rhs.value.clone();
        // Division increases uncertainty more than other operations
        let mut epistemic = self.epistemic.combine(&rhs.epistemic);
        epistemic.confidence = BetaConfidence::new(
            epistemic.confidence.alpha * 0.9,
            epistemic.confidence.beta * 1.1,
        );
        let provenance = self.provenance.merge(&rhs.provenance, "div");

        BetaKnowledge {
            value,
            epistemic,
            provenance,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert Source to SourcePriorType
fn source_to_prior_type(source: &Source) -> SourcePriorType {
    match source {
        Source::Axiom => SourcePriorType::Experimental, // Axioms are highly reliable
        Source::Measurement { .. } => SourcePriorType::Experimental,
        Source::Derivation(_) => SourcePriorType::Computational,
        Source::ModelPrediction { .. } => SourcePriorType::ModelPrediction,
        Source::HumanAssertion { .. } => SourcePriorType::HumanAssertion,
        Source::External { .. } => SourcePriorType::External,
        Source::OntologyAssertion { .. } => SourcePriorType::External,
        Source::Transformation { .. } => SourcePriorType::Computational,
        Source::Unknown => SourcePriorType::Unknown,
    }
}

/// Compute variance penalty for training loss
///
/// This is the key function for integrating Beta epistemic types
/// with neural network training. The variance penalty encourages
/// networks to minimize ignorance (reduce variance).
pub fn variance_penalty(knowledges: &[&BetaEpistemicStatus], lambda: f64) -> f64 {
    knowledges.iter().map(|k| k.variance_penalty(lambda)).sum()
}

/// Compute exploration priorities for active learning
///
/// Returns a list of (index, priority) pairs sorted by priority (highest first)
pub fn exploration_priorities<T: Clone>(knowledges: &[BetaKnowledge<T>]) -> Vec<(usize, f64)> {
    let mut priorities: Vec<(usize, f64)> = knowledges
        .iter()
        .enumerate()
        .map(|(i, k)| (i, k.exploration_priority()))
        .collect();

    priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    priorities
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_epistemic_status_creation() {
        let status = BetaEpistemicStatus::empirical(8.0, 2.0, Source::Axiom);
        // With source-driven prior from Axiom (mapped to Experimental),
        // the mean should be high (>0.7) due to 8 successes vs 2 failures
        assert!(status.mean() > 0.7);
        assert!(status.mean() < 1.0);
        assert!(status.variance() > 0.0);
    }

    #[test]
    fn test_beta_epistemic_status_axiomatic() {
        let status = BetaEpistemicStatus::axiomatic();
        assert!(status.mean() > 0.99);
        assert!(status.variance() < 0.001);
    }

    #[test]
    fn test_decay_model_linear() {
        let conf = BetaConfidence::new(10.0, 2.0);
        let model = DecayModel::Linear { factor: 0.9 };
        let decayed = model.apply(&conf, 1);
        assert!(decayed.mean() < conf.mean());
    }

    #[test]
    fn test_prior_types() {
        let uniform = PriorType::Uniform.to_beta();
        assert!((uniform.mean() - 0.5).abs() < 0.001);

        let jeffreys = PriorType::Jeffreys.to_beta();
        assert!((jeffreys.mean() - 0.5).abs() < 0.001);

        let weak = PriorType::Weak {
            center: 0.8,
            strength: 5.0,
        }
        .to_beta();
        assert!((weak.mean() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_observation_update() {
        let mut status = BetaEpistemicStatus::from_confidence(0.5, 2.0, Source::Unknown);
        let old_variance = status.variance();

        // Observe 10 successes
        for _ in 0..10 {
            status.observe(true);
        }

        // Mean should increase, variance should decrease
        assert!(status.mean() > 0.5);
        assert!(status.variance() < old_variance);
    }

    #[test]
    fn test_combine_hierarchical() {
        let s1 = BetaEpistemicStatus::empirical(8.0, 2.0, Source::Axiom);
        let s2 = BetaEpistemicStatus::empirical(6.0, 4.0, Source::Axiom);
        let s3 = BetaEpistemicStatus::empirical(9.0, 1.0, Source::Axiom);

        let combined =
            BetaEpistemicStatus::combine_hierarchical(&[&s1, &s2, &s3], PriorType::Uniform);

        // Combined should be weighted toward lower variance sources
        assert!(combined.mean() > 0.0);
        assert!(combined.mean() < 1.0);
    }

    #[test]
    fn test_active_inference_metrics() {
        let status = BetaEpistemicStatus::from_confidence(0.5, 2.0, Source::Unknown);

        // High variance → should explore
        assert!(status.should_explore(0.01));

        let certain = BetaEpistemicStatus::axiomatic();
        // Low variance → should not explore
        assert!(!certain.should_explore(0.01));
    }

    #[test]
    fn test_beta_knowledge_arithmetic() {
        let k1 = BetaKnowledge::from_confidence(10.0f64, 0.9, 10.0);
        let k2 = BetaKnowledge::from_confidence(5.0f64, 0.8, 10.0);

        let sum = k1.clone() + k2.clone();
        assert!((sum.value - 15.0).abs() < 0.001);

        let prod = k1 * k2;
        assert!((prod.value - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_variance_penalty() {
        let s1 = BetaEpistemicStatus::from_confidence(0.5, 2.0, Source::Unknown);
        let s2 = BetaEpistemicStatus::axiomatic();

        let penalty = variance_penalty(&[&s1, &s2], 0.1);

        // Penalty should be dominated by high-variance source
        assert!(penalty > s2.variance_penalty(0.1));
    }

    #[test]
    fn test_exploration_priorities() {
        let k1 = BetaKnowledge::from_confidence(0.5f64, 0.5, 2.0); // High variance
        let k2 = BetaKnowledge::from_confidence(0.5f64, 0.5, 100.0); // Low variance

        let priorities = exploration_priorities(&[k1, k2]);

        // First item should be the high-variance one
        assert_eq!(priorities[0].0, 0);
        assert!(priorities[0].1 > priorities[1].1);
    }

    #[test]
    fn test_credible_interval() {
        let status = BetaEpistemicStatus::empirical(80.0, 20.0, Source::Axiom);
        let (lo, hi) = status.credible_interval_95();

        assert!(lo < status.mean());
        assert!(hi > status.mean());
        assert!(lo >= 0.0);
        assert!(hi <= 1.0);
    }
}

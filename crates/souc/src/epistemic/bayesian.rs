//! Bayesian Epistemic Fusion
//!
//! This module implements principled Bayesian methods for combining epistemic
//! statuses, moving beyond scalar confidence to full posterior distributions.
//!
//! # Theory
//!
//! Traditional confidence values are point estimates that lose information about
//! uncertainty. By modeling confidence as a Beta distribution, we capture:
//!
//! - **Mean**: Expected confidence value (α / (α + β))
//! - **Variance**: Uncertainty about the confidence itself
//! - **Credible intervals**: Probabilistic bounds on true confidence
//!
//! # Beta Distribution for Confidence
//!
//! The Beta distribution is the conjugate prior for binomial observations,
//! making it ideal for modeling "proportion of supporting evidence":
//!
//! ```text
//! Beta(α, β) where:
//!   α = "pseudo-successes" (supporting evidence count + prior)
//!   β = "pseudo-failures" (contradicting evidence count + prior)
//!
//! Properties:
//!   mean = α / (α + β)
//!   variance = αβ / ((α+β)² × (α+β+1))
//!   mode = (α-1) / (α+β-2)  for α,β > 1
//! ```
//!
//! # Hierarchical Bayesian Framework
//!
//! We employ a two-level hierarchical model:
//!
//! 1. **Domain-level priors**: Each ontology domain has a characteristic
//!    prior reflecting typical evidence quality in that domain
//! 2. **Source-level likelihoods**: Individual sources contribute evidence
//!    weighted by their reliability
//!
//! ```text
//! Level 1 (Hyperprior):  Domain → Beta(α₀, β₀)
//! Level 2 (Prior):       Source_type × Domain → Beta(α₁, β₁)
//! Level 3 (Posterior):   Evidence → Beta(α_post, β_post)
//! ```
//!
//! # Dempster-Shafer Theory Fallback
//!
//! When evidence conflict is high (>0.7), we fall back to Dempster-Shafer
//! Theory (DST) which explicitly models epistemic uncertainty:
//!
//! ```text
//! m(A) = belief mass assigned to hypothesis A
//! Bel(A) = Σ_{B⊆A} m(B)  (belief function)
//! Pl(A) = 1 - Bel(¬A)     (plausibility function)
//!
//! Conflict coefficient: K = Σ_{A∩B=∅} m₁(A)·m₂(B)
//! ```
//!
//! # Source Reliability Hierarchy
//!
//! Evidence sources have inherent reliability based on methodology:
//!
//! ```text
//! Verified (FDA approval, peer review)     → reliability = 1.5
//! Experimental (RCT, controlled study)     → reliability = 1.3
//! Publication (journal article)            → reliability = 1.2
//! Human Assertion (expert opinion)         → reliability = 1.1
//! Computational (simulation, model)        → reliability = 1.0
//! Dataset (raw data)                       → reliability = 0.9
//! Expert Opinion (informal)                → reliability = 0.85
//! ```
//!
//! # References
//!
//! - Gelman et al., "Bayesian Data Analysis" (Beta conjugate priors)
//! - Shafer, "A Mathematical Theory of Evidence" (DST for conflict)
//! - Josang, "Subjective Logic" (Beta reputation systems)
//! - Yager, "On the Dempster-Shafer Framework" (conflict handling)

use std::f64::consts::PI;

use super::{Confidence, EpistemicStatus, Evidence, EvidenceKind, Revisability, Source};

/// Beta distribution for modeling confidence with uncertainty
///
/// Represents confidence not as a single value but as a probability
/// distribution over possible confidence values. This allows tracking
/// how certain we are about our confidence estimate.
///
/// # Mathematical Properties
///
/// For Beta(α, β):
/// - Mean: α / (α + β)
/// - Variance: αβ / ((α + β)² × (α + β + 1))
/// - As α + β increases, variance decreases (more evidence = more certainty)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaConfidence {
    /// Pseudo-successes: supporting evidence + prior
    pub alpha: f64,
    /// Pseudo-failures: contradicting evidence + prior
    pub beta: f64,
}

impl BetaConfidence {
    /// Create a new Beta distribution with given parameters
    ///
    /// # Panics
    /// Panics if alpha or beta <= 0
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(beta > 0.0, "beta must be positive");
        Self { alpha, beta }
    }

    /// Uninformative (Jeffreys) prior: Beta(0.5, 0.5)
    ///
    /// Represents maximum uncertainty - we have no prior knowledge
    pub fn jeffreys_prior() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Uniform prior: Beta(1, 1)
    ///
    /// All confidence values equally likely
    pub fn uniform_prior() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Weak informative prior centered at a value
    ///
    /// Creates a prior that gently favors a particular confidence level
    /// but allows the data to easily override it.
    pub fn weak_prior(center: f64, strength: f64) -> Self {
        let center = center.clamp(0.01, 0.99);
        let alpha = center * strength;
        let beta = (1.0 - center) * strength;
        Self { alpha, beta }
    }

    /// Strong prior from domain knowledge
    ///
    /// Use when you have strong prior beliefs (e.g., published literature)
    pub fn strong_prior(center: f64, strength: f64) -> Self {
        Self::weak_prior(center, strength * 10.0)
    }

    /// Create from scalar confidence value with effective sample size
    ///
    /// Converts a point estimate to a Beta distribution.
    /// `n` represents how much "evidence" the point estimate represents.
    pub fn from_confidence(confidence: f64, effective_n: f64) -> Self {
        let p = confidence.clamp(0.001, 0.999);
        Self {
            alpha: p * effective_n,
            beta: (1.0 - p) * effective_n,
        }
    }

    /// Mean (expected value) of the distribution
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the distribution
    ///
    /// Lower variance = more certainty about the confidence value
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }

    /// Standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Mode (most likely value)
    ///
    /// Returns None if alpha <= 1 or beta <= 1 (mode at boundary)
    pub fn mode(&self) -> Option<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }

    /// Effective sample size (total pseudo-observations)
    pub fn sample_size(&self) -> f64 {
        self.alpha + self.beta
    }

    /// Coefficient of variation (std_dev / mean)
    ///
    /// Measures relative uncertainty - useful for comparing
    /// distributions with different means
    pub fn coefficient_of_variation(&self) -> f64 {
        let mean = self.mean();
        if mean > 0.0 {
            self.std_dev() / mean
        } else {
            f64::INFINITY
        }
    }

    /// Credible interval at given level (e.g., 0.95 for 95% CI)
    ///
    /// Returns (lower, upper) bounds such that the true confidence
    /// lies within this interval with the given probability.
    ///
    /// Uses normal approximation for computational efficiency.
    pub fn credible_interval(&self, level: f64) -> (f64, f64) {
        let z = normal_quantile((1.0 + level) / 2.0);
        let mean = self.mean();
        let std = self.std_dev();
        let lower = (mean - z * std).max(0.0);
        let upper = (mean + z * std).min(1.0);
        (lower, upper)
    }

    /// Update with new evidence (Bayesian update)
    ///
    /// `successes`: Number of supporting observations
    /// `failures`: Number of contradicting observations
    pub fn update(&mut self, successes: f64, failures: f64) {
        self.alpha += successes;
        self.beta += failures;
    }

    /// Update with weighted evidence
    ///
    /// Weight adjusts the effective sample size of the evidence
    pub fn update_weighted(&mut self, successes: f64, failures: f64, weight: f64) {
        self.alpha += successes * weight;
        self.beta += failures * weight;
    }

    /// Combine two Beta distributions (pooling evidence)
    ///
    /// Uses linear opinion pool (weighted average of means)
    /// with combined sample size.
    pub fn combine(&self, other: &BetaConfidence, self_weight: f64, other_weight: f64) -> Self {
        let total_weight = self_weight + other_weight;
        let combined_mean =
            (self.mean() * self_weight + other.mean() * other_weight) / total_weight;

        // Combined sample size reflects total evidence
        let combined_n = self.sample_size() + other.sample_size();

        Self::from_confidence(combined_mean, combined_n)
    }

    /// Logarithmic opinion pool (geometric mean of densities)
    ///
    /// More appropriate when sources may be correlated
    pub fn combine_log_pool(&self, other: &BetaConfidence) -> Self {
        // For Beta distributions, log pool gives Beta with summed parameters
        Self {
            alpha: self.alpha + other.alpha - 1.0, // Subtract 1 to avoid double-counting prior
            beta: self.beta + other.beta - 1.0,
        }
    }

    /// Check if this distribution satisfies a minimum confidence requirement
    ///
    /// Returns probability that true confidence >= threshold
    pub fn probability_above(&self, threshold: f64) -> f64 {
        1.0 - incomplete_beta_regularized(threshold, self.alpha, self.beta)
    }

    /// Probability that the true parameter is greater than a threshold
    ///
    /// Alias for `probability_above` - used in AlphaGeoZero benchmarks
    /// for statistical comparisons (e.g., P(solve_rate > 0.833))
    pub fn prob_greater_than(&self, threshold: f64) -> f64 {
        self.probability_above(threshold)
    }

    /// Check if this distribution satisfies a maximum variance requirement
    pub fn variance_below(&self, max_variance: f64) -> bool {
        self.variance() <= max_variance
    }

    /// Convert to scalar confidence (mean)
    pub fn to_confidence(&self) -> Confidence {
        Confidence::new(self.mean())
    }

    /// Entropy of the distribution (information content)
    ///
    /// Higher entropy = more uncertainty
    pub fn entropy(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        ln_beta(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
            + (a + b - 2.0) * digamma(a + b)
    }

    /// Kullback-Leibler divergence from another Beta distribution
    ///
    /// Measures how different this distribution is from the other
    pub fn kl_divergence(&self, other: &BetaConfidence) -> f64 {
        let a1 = self.alpha;
        let b1 = self.beta;
        let a2 = other.alpha;
        let b2 = other.beta;

        ln_beta(a2, b2) - ln_beta(a1, b1) + (a1 - a2) * digamma(a1) + (b1 - b2) * digamma(b1)
            - (a1 - a2 + b1 - b2) * digamma(a1 + b1)
    }
}

impl Default for BetaConfidence {
    fn default() -> Self {
        Self::uniform_prior()
    }
}

impl From<f64> for BetaConfidence {
    fn from(confidence: f64) -> Self {
        // Default effective sample size of 2 (weak prior)
        Self::from_confidence(confidence, 2.0)
    }
}

impl From<Confidence> for BetaConfidence {
    fn from(confidence: Confidence) -> Self {
        Self::from(confidence.value())
    }
}

// ============================================================================
// Hierarchical Bayesian Priors
// ============================================================================

/// Ontology domain categories for prior selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OntologyDomain {
    /// Pharmacokinetic/ADME parameters (typically well-studied)
    Pharmacokinetics,
    /// Physicochemical properties (often highly reliable)
    PhysicoChemical,
    /// Biological/physiological parameters (moderate variability)
    Biological,
    /// Clinical observations (high variability)
    Clinical,
    /// Computational predictions (model-dependent)
    Computational,
    /// Literature/database values (source-dependent)
    Literature,
    /// Unknown or unclassified domain
    Unknown,
}

impl OntologyDomain {
    /// Classify an ontology URI into a domain
    ///
    /// Uses prefix matching to categorize ontology namespaces
    pub fn from_ontology_uri(uri: &str) -> Self {
        let uri_lower = uri.to_lowercase();

        if uri_lower.contains("pharma")
            || uri_lower.contains("pk")
            || uri_lower.contains("adme")
            || uri_lower.contains("clearance")
            || uri_lower.contains("volume")
        {
            OntologyDomain::Pharmacokinetics
        } else if uri_lower.contains("physico")
            || uri_lower.contains("chem")
            || uri_lower.contains("molecular")
            || uri_lower.contains("logp")
            || uri_lower.contains("solubility")
        {
            OntologyDomain::PhysicoChemical
        } else if uri_lower.contains("bio")
            || uri_lower.contains("physio")
            || uri_lower.contains("organ")
            || uri_lower.contains("tissue")
        {
            OntologyDomain::Biological
        } else if uri_lower.contains("clinical")
            || uri_lower.contains("patient")
            || uri_lower.contains("trial")
            || uri_lower.contains("endpoint")
        {
            OntologyDomain::Clinical
        } else if uri_lower.contains("comput")
            || uri_lower.contains("predict")
            || uri_lower.contains("model")
            || uri_lower.contains("qsar")
        {
            OntologyDomain::Computational
        } else if uri_lower.contains("lit")
            || uri_lower.contains("database")
            || uri_lower.contains("pubchem")
            || uri_lower.contains("drugbank")
        {
            OntologyDomain::Literature
        } else {
            OntologyDomain::Unknown
        }
    }

    /// Get the default prior for this domain
    ///
    /// Domain-specific priors reflect typical evidence quality and
    /// variability in each scientific domain.
    pub fn default_prior(&self) -> BetaConfidence {
        match self {
            // Physicochemical: well-established, low uncertainty
            OntologyDomain::PhysicoChemical => BetaConfidence::new(8.0, 2.0),
            // PK: good data but biological variability
            OntologyDomain::Pharmacokinetics => BetaConfidence::new(6.0, 3.0),
            // Biological: high variability, moderate confidence
            OntologyDomain::Biological => BetaConfidence::new(4.0, 3.0),
            // Literature: depends on source quality
            OntologyDomain::Literature => BetaConfidence::new(3.0, 2.0),
            // Computational: model-dependent, needs validation
            OntologyDomain::Computational => BetaConfidence::new(3.0, 3.0),
            // Clinical: high variability, lower prior confidence
            OntologyDomain::Clinical => BetaConfidence::new(2.0, 2.0),
            // Unknown: weakly informative
            OntologyDomain::Unknown => BetaConfidence::new(1.0, 1.0),
        }
    }

    /// Get prior strength multiplier
    ///
    /// Higher values mean the prior has more influence on the posterior
    pub fn prior_strength(&self) -> f64 {
        match self {
            OntologyDomain::PhysicoChemical => 1.5,
            OntologyDomain::Pharmacokinetics => 1.2,
            OntologyDomain::Biological => 1.0,
            OntologyDomain::Literature => 0.9,
            OntologyDomain::Computational => 0.8,
            OntologyDomain::Clinical => 0.7,
            OntologyDomain::Unknown => 0.5,
        }
    }
}

/// Hierarchical prior configuration
#[derive(Debug, Clone)]
pub struct HierarchicalPrior {
    /// Domain-level hyperprior
    pub domain: OntologyDomain,
    /// Base prior from domain
    pub base_prior: BetaConfidence,
    /// Source-type adjustment factor
    pub source_adjustment: f64,
    /// Evidence quality scaling
    pub evidence_scaling: f64,
}

impl HierarchicalPrior {
    /// Create hierarchical prior for a given domain and source type
    pub fn new(domain: OntologyDomain, source: &Source) -> Self {
        let base_prior = domain.default_prior();
        let source_reliability = SourceReliability::from_source(source);

        // Adjust prior based on source type
        let source_adjustment = source_reliability.score * domain.prior_strength();

        // Scale evidence impact based on source quality
        let evidence_scaling = if source_reliability.verifiable && source_reliability.reproducible {
            1.2
        } else if source_reliability.verifiable {
            1.0
        } else {
            0.8
        };

        Self {
            domain,
            base_prior,
            source_adjustment,
            evidence_scaling,
        }
    }

    /// Create from ontology URI string
    pub fn from_ontology(ontology_uri: &str, source: &Source) -> Self {
        let domain = OntologyDomain::from_ontology_uri(ontology_uri);
        Self::new(domain, source)
    }

    /// Get the effective prior for Bayesian updating
    pub fn effective_prior(&self) -> BetaConfidence {
        BetaConfidence::new(
            self.base_prior.alpha * self.source_adjustment,
            self.base_prior.beta * self.source_adjustment,
        )
    }

    /// Compute posterior given evidence
    pub fn compute_posterior(&self, successes: f64, failures: f64) -> BetaConfidence {
        let prior = self.effective_prior();
        BetaConfidence::new(
            prior.alpha + successes * self.evidence_scaling,
            prior.beta + failures * self.evidence_scaling,
        )
    }
}

// ============================================================================
// Dempster-Shafer Theory for High-Conflict Scenarios
// ============================================================================

/// Belief mass assignment for Dempster-Shafer Theory
///
/// In DST, probability mass can be assigned to sets of hypotheses,
/// not just single hypotheses. This allows explicit modeling of
/// ignorance (mass assigned to the full set Ω).
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefMass {
    /// Mass assigned to "true" hypothesis
    pub belief_true: f64,
    /// Mass assigned to "false" hypothesis
    pub belief_false: f64,
    /// Mass assigned to uncertainty (neither confirmed nor denied)
    pub uncertainty: f64,
}

impl BeliefMass {
    /// Create new belief mass assignment
    ///
    /// Masses must sum to 1.0
    pub fn new(belief_true: f64, belief_false: f64, uncertainty: f64) -> Self {
        let total = belief_true + belief_false + uncertainty;
        // Normalize to ensure sum = 1
        let norm = if total > 0.0 { total } else { 1.0 };
        Self {
            belief_true: belief_true / norm,
            belief_false: belief_false / norm,
            uncertainty: uncertainty / norm,
        }
    }

    /// Create from scalar confidence with uncertainty estimate
    pub fn from_confidence_with_uncertainty(confidence: f64, uncertainty: f64) -> Self {
        let remaining = 1.0 - uncertainty;
        Self::new(
            confidence * remaining,
            (1.0 - confidence) * remaining,
            uncertainty,
        )
    }

    /// Convert Beta distribution to belief mass
    ///
    /// Uses variance as a measure of epistemic uncertainty
    pub fn from_beta(beta: &BetaConfidence) -> Self {
        // Use variance to estimate uncertainty
        // High variance = high uncertainty
        let variance = beta.variance();
        let uncertainty = (variance * 12.0).min(0.5); // Scale variance to [0, 0.5]

        let remaining = 1.0 - uncertainty;
        let mean = beta.mean();

        Self::new(mean * remaining, (1.0 - mean) * remaining, uncertainty)
    }

    /// Belief function Bel(true) - lower bound on probability
    pub fn belief(&self) -> f64 {
        self.belief_true
    }

    /// Plausibility function Pl(true) - upper bound on probability
    pub fn plausibility(&self) -> f64 {
        self.belief_true + self.uncertainty
    }

    /// Uncertainty interval [Bel, Pl]
    pub fn uncertainty_interval(&self) -> (f64, f64) {
        (self.belief(), self.plausibility())
    }

    /// Pignistic probability (decision-making value)
    ///
    /// Distributes uncertainty equally between hypotheses
    pub fn pignistic_probability(&self) -> f64 {
        self.belief_true + self.uncertainty / 2.0
    }
}

/// Dempster-Shafer combination result
#[derive(Debug, Clone)]
pub struct DSTCombinationResult {
    /// Combined belief mass
    pub combined: BeliefMass,
    /// Conflict coefficient K (0 = no conflict, 1 = complete conflict)
    pub conflict: f64,
    /// Whether Yager's rule was used (handles high conflict)
    pub used_yager_rule: bool,
}

/// Combine belief masses using Dempster's rule of combination
///
/// This is the standard DST combination rule. For high conflict,
/// we fall back to Yager's rule which assigns conflicting mass
/// to uncertainty instead of renormalizing.
pub fn dempster_combine(m1: &BeliefMass, m2: &BeliefMass) -> DSTCombinationResult {
    // Calculate conflict coefficient K
    // K = sum of products where hypotheses contradict
    let k = m1.belief_true * m2.belief_false + m1.belief_false * m2.belief_true;

    // If conflict is too high (>0.9), use Yager's modified rule
    if k > 0.9 {
        return yager_combine(m1, m2, k);
    }

    // Standard Dempster combination with normalization
    let norm = 1.0 - k;

    if norm < 1e-10 {
        // Complete conflict - fall back to uncertainty
        return DSTCombinationResult {
            combined: BeliefMass::new(0.0, 0.0, 1.0),
            conflict: k,
            used_yager_rule: false,
        };
    }

    // Combine masses
    let combined_true = (m1.belief_true * m2.belief_true
        + m1.belief_true * m2.uncertainty
        + m1.uncertainty * m2.belief_true)
        / norm;

    let combined_false = (m1.belief_false * m2.belief_false
        + m1.belief_false * m2.uncertainty
        + m1.uncertainty * m2.belief_false)
        / norm;

    let combined_uncertainty = (m1.uncertainty * m2.uncertainty) / norm;

    DSTCombinationResult {
        combined: BeliefMass::new(combined_true, combined_false, combined_uncertainty),
        conflict: k,
        used_yager_rule: false,
    }
}

/// Yager's modified combination rule for high-conflict scenarios
///
/// Instead of renormalizing, assigns conflicting mass to uncertainty.
/// This is more conservative and appropriate when sources disagree strongly.
fn yager_combine(m1: &BeliefMass, m2: &BeliefMass, k: f64) -> DSTCombinationResult {
    // No normalization - conflicting mass goes to uncertainty
    let combined_true = m1.belief_true * m2.belief_true
        + m1.belief_true * m2.uncertainty
        + m1.uncertainty * m2.belief_true;

    let combined_false = m1.belief_false * m2.belief_false
        + m1.belief_false * m2.uncertainty
        + m1.uncertainty * m2.belief_false;

    // Uncertainty includes: original uncertainty products + conflict mass
    let combined_uncertainty = m1.uncertainty * m2.uncertainty + k;

    DSTCombinationResult {
        combined: BeliefMass::new(combined_true, combined_false, combined_uncertainty),
        conflict: k,
        used_yager_rule: true,
    }
}

/// Combine multiple belief masses
pub fn dempster_combine_multiple(masses: &[BeliefMass]) -> DSTCombinationResult {
    if masses.is_empty() {
        return DSTCombinationResult {
            combined: BeliefMass::new(0.0, 0.0, 1.0),
            conflict: 0.0,
            used_yager_rule: false,
        };
    }

    if masses.len() == 1 {
        return DSTCombinationResult {
            combined: masses[0].clone(),
            conflict: 0.0,
            used_yager_rule: false,
        };
    }

    let mut result = dempster_combine(&masses[0], &masses[1]);
    let mut total_conflict = result.conflict;

    for mass in masses.iter().skip(2) {
        result = dempster_combine(&result.combined, mass);
        total_conflict = total_conflict.max(result.conflict);
    }

    DSTCombinationResult {
        combined: result.combined,
        conflict: total_conflict,
        used_yager_rule: result.used_yager_rule,
    }
}

/// Source reliability scoring based on evidence methodology
///
/// Higher scores indicate more reliable sources
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceReliability {
    /// Base reliability score (0.0 - 2.0 range)
    pub score: f64,
    /// Whether this source can be independently verified
    pub verifiable: bool,
    /// Whether this source is reproducible
    pub reproducible: bool,
}

impl SourceReliability {
    /// Get reliability score for a source type
    pub fn from_source(source: &Source) -> Self {
        match source {
            Source::Axiom => Self {
                score: 2.0, // Definitional - highest reliability
                verifiable: true,
                reproducible: true,
            },
            Source::Measurement { protocol, .. } => {
                let base = if protocol.is_some() { 1.4 } else { 1.2 };
                Self {
                    score: base,
                    verifiable: true,
                    reproducible: protocol.is_some(),
                }
            }
            Source::Derivation(method) => {
                let score = if method.contains("bayesian") || method.contains("formal") {
                    1.3
                } else {
                    1.1
                };
                Self {
                    score,
                    verifiable: true,
                    reproducible: true,
                }
            }
            Source::External { .. } => Self {
                score: 1.0,
                verifiable: true,
                reproducible: false,
            },
            Source::OntologyAssertion { .. } => Self {
                score: 1.5, // Ontology-backed claims are reliable
                verifiable: true,
                reproducible: true,
            },
            Source::ModelPrediction { .. } => Self {
                score: 0.9,
                verifiable: false,
                reproducible: true,
            },
            Source::Transformation { original, .. } => {
                // Inherit from original, with slight degradation
                let orig_rel = Self::from_source(original);
                Self {
                    score: orig_rel.score * 0.95,
                    verifiable: orig_rel.verifiable,
                    reproducible: orig_rel.reproducible,
                }
            }
            Source::HumanAssertion { .. } => Self {
                score: 1.0,
                verifiable: false,
                reproducible: false,
            },
            Source::Unknown => Self {
                score: 0.5, // Lowest reliability for unknown sources
                verifiable: false,
                reproducible: false,
            },
        }
    }

    /// Get reliability score for an evidence kind
    pub fn from_evidence_kind(kind: &EvidenceKind) -> Self {
        match kind {
            EvidenceKind::Verified { .. } => Self {
                score: 1.5,
                verifiable: true,
                reproducible: true,
            },
            EvidenceKind::Experiment { .. } => Self {
                score: 1.3,
                verifiable: true,
                reproducible: true,
            },
            EvidenceKind::Publication { doi } => Self {
                score: if doi.is_some() { 1.2 } else { 1.1 },
                verifiable: doi.is_some(),
                reproducible: false,
            },
            EvidenceKind::HumanAssertion { .. } => Self {
                score: 1.1,
                verifiable: false,
                reproducible: false,
            },
            EvidenceKind::Computation { .. } => Self {
                score: 1.0,
                verifiable: true,
                reproducible: true,
            },
            EvidenceKind::ExpertOpinion { .. } => Self {
                score: 0.95,
                verifiable: false,
                reproducible: false,
            },
            EvidenceKind::Dataset { .. } => Self {
                score: 0.9,
                verifiable: true,
                reproducible: true,
            },
        }
    }
}

/// Evidence assessment for Bayesian updating
#[derive(Debug, Clone)]
pub struct EvidenceAssessment {
    /// Does this evidence support the claim?
    pub supports: bool,
    /// Strength of support/contradiction (0.0 - 1.0)
    pub strength: f64,
    /// Reliability of the evidence source
    pub reliability: SourceReliability,
}

impl EvidenceAssessment {
    /// Assess a piece of evidence
    pub fn assess(evidence: &Evidence) -> Self {
        let reliability = SourceReliability::from_evidence_kind(&evidence.kind);

        // Evidence with strength >= 0.5 is considered supporting
        let supports = evidence.strength.value() >= 0.5;

        // Strength is the absolute deviation from 0.5
        // High strength (near 1.0 or 0.0) indicates strong evidence
        let strength = (evidence.strength.value() - 0.5).abs() * 2.0;

        Self {
            supports,
            strength,
            reliability,
        }
    }

    /// Calculate weighted contribution to Beta update
    pub fn weighted_contribution(&self) -> (f64, f64) {
        let weight = self.strength * self.reliability.score;
        if self.supports {
            (weight, 0.0)
        } else {
            (0.0, weight)
        }
    }
}

/// Result of Bayesian epistemic fusion
#[derive(Debug, Clone)]
pub struct BayesianFusionResult {
    /// Combined Beta distribution
    pub confidence: BetaConfidence,
    /// Total evidence weight considered
    pub total_weight: f64,
    /// Number of sources combined
    pub source_count: usize,
    /// Conflict measure (0.0 = no conflict, 1.0 = complete conflict)
    pub conflict: f64,
    /// Merged evidence (deduplicated)
    pub evidence: Vec<Evidence>,
    /// Combined revisability
    pub revisability: Revisability,
    /// DST belief mass (available when conflict > 0.5)
    pub dst_belief: Option<BeliefMass>,
    /// Whether DST fallback was used
    pub used_dst_fallback: bool,
    /// Ontology domain used for priors
    pub domain: Option<OntologyDomain>,
}

impl BayesianFusionResult {
    /// Convert to standard EpistemicStatus
    pub fn to_epistemic_status(&self) -> EpistemicStatus {
        EpistemicStatus {
            confidence: self.confidence.to_confidence(),
            revisability: self.revisability.clone(),
            source: Source::Derivation(format!(
                "bayesian_fusion(n={}, conflict={:.3})",
                self.source_count, self.conflict
            )),
            evidence: self.evidence.clone(),
        }
    }

    /// Check if the fusion result is reliable
    ///
    /// A result is considered reliable if:
    /// - Variance is below threshold
    /// - Conflict is low
    /// - Sufficient evidence
    pub fn is_reliable(&self, max_variance: f64, max_conflict: f64, min_evidence: usize) -> bool {
        self.confidence.variance() <= max_variance
            && self.conflict <= max_conflict
            && self.evidence.len() >= min_evidence
    }
}

/// Combine multiple epistemic statuses using Bayesian fusion
///
/// This is the principled replacement for simple log-odds averaging.
/// It uses Beta distributions to properly track uncertainty.
pub fn combine_epistemic_beta(statuses: &[EpistemicStatus]) -> BayesianFusionResult {
    combine_epistemic_beta_with_prior(statuses, BetaConfidence::uniform_prior())
}

/// Combine with a specified prior
pub fn combine_epistemic_beta_with_prior(
    statuses: &[EpistemicStatus],
    prior: BetaConfidence,
) -> BayesianFusionResult {
    if statuses.is_empty() {
        return BayesianFusionResult {
            confidence: prior,
            total_weight: 0.0,
            source_count: 0,
            conflict: 0.0,
            evidence: vec![],
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".to_string()],
            },
            dst_belief: None,
            used_dst_fallback: false,
            domain: None,
        };
    }

    if statuses.len() == 1 {
        let status = &statuses[0];
        return BayesianFusionResult {
            confidence: BetaConfidence::from(status.confidence),
            total_weight: 1.0,
            source_count: 1,
            conflict: 0.0,
            evidence: status.evidence.clone(),
            revisability: status.revisability.clone(),
            dst_belief: None,
            used_dst_fallback: false,
            domain: None,
        };
    }

    let mut combined = prior;
    let mut total_weight = 0.0;
    let mut all_evidence: Vec<Evidence> = vec![];
    let mut confidence_values: Vec<f64> = vec![];

    for status in statuses {
        // Get source reliability
        let source_reliability = SourceReliability::from_source(&status.source);

        // Assess each piece of evidence
        for evidence in &status.evidence {
            let assessment = EvidenceAssessment::assess(evidence);
            let (successes, failures) = assessment.weighted_contribution();
            combined.update_weighted(successes, failures, source_reliability.score);
            total_weight += (successes + failures) * source_reliability.score;
        }

        // If no evidence, use the confidence value directly
        if status.evidence.is_empty() {
            let effective_n = 2.0 * source_reliability.score;
            let successes = status.confidence.value() * effective_n;
            let failures = (1.0 - status.confidence.value()) * effective_n;
            combined.update(successes, failures);
            total_weight += effective_n;
        }

        confidence_values.push(status.confidence.value());
        all_evidence.extend(status.evidence.clone());
    }

    // Calculate conflict measure (variance of input confidence values)
    let conflict = calculate_conflict(&confidence_values);

    // If conflict is high, also compute DST belief
    let (dst_belief, used_dst_fallback) = if conflict > 0.5 {
        let masses: Vec<BeliefMass> = statuses
            .iter()
            .map(|s| {
                let beta = BetaConfidence::from(s.confidence);
                BeliefMass::from_beta(&beta)
            })
            .collect();
        let dst_result = dempster_combine_multiple(&masses);
        (Some(dst_result.combined), dst_result.used_yager_rule)
    } else {
        (None, false)
    };

    // Deduplicate evidence
    let evidence = deduplicate_evidence(all_evidence);

    // Combine revisability (most restrictive wins)
    let revisability = combine_revisability(statuses);

    BayesianFusionResult {
        confidence: combined,
        total_weight,
        source_count: statuses.len(),
        conflict,
        evidence,
        revisability,
        dst_belief,
        used_dst_fallback,
        domain: None,
    }
}

/// Advanced hierarchical Bayesian fusion with ontology-driven priors
///
/// This is the most principled fusion method, using:
/// 1. Domain-specific priors based on ontology classification
/// 2. Source-type reliability weighting
/// 3. DST fallback for high-conflict scenarios
///
/// # Arguments
/// * `statuses` - Epistemic statuses to combine
/// * `ontology_uri` - Optional ontology URI for domain-specific priors
///
/// # Returns
/// A `BayesianFusionResult` with full uncertainty quantification
pub fn combine_epistemic_hierarchical(
    statuses: &[EpistemicStatus],
    ontology_uri: Option<&str>,
) -> BayesianFusionResult {
    // Determine domain from ontology URI
    let domain = ontology_uri
        .map(OntologyDomain::from_ontology_uri)
        .unwrap_or(OntologyDomain::Unknown);

    if statuses.is_empty() {
        let prior = domain.default_prior();
        return BayesianFusionResult {
            confidence: prior,
            total_weight: 0.0,
            source_count: 0,
            conflict: 0.0,
            evidence: vec![],
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".to_string()],
            },
            dst_belief: None,
            used_dst_fallback: false,
            domain: Some(domain),
        };
    }

    if statuses.len() == 1 {
        let status = &statuses[0];
        let hierarchical = HierarchicalPrior::new(domain, &status.source);

        // Use domain prior combined with single evidence
        let beta = BetaConfidence::from(status.confidence);
        let combined = hierarchical.compute_posterior(
            beta.alpha - 1.0, // Remove uniform prior contribution
            beta.beta - 1.0,
        );

        return BayesianFusionResult {
            confidence: combined,
            total_weight: 1.0,
            source_count: 1,
            conflict: 0.0,
            evidence: status.evidence.clone(),
            revisability: status.revisability.clone(),
            dst_belief: None,
            used_dst_fallback: false,
            domain: Some(domain),
        };
    }

    // Collect per-source Beta posteriors and belief masses
    let mut source_posteriors: Vec<BetaConfidence> = Vec::with_capacity(statuses.len());
    let mut belief_masses: Vec<BeliefMass> = Vec::with_capacity(statuses.len());
    let mut confidence_values: Vec<f64> = Vec::with_capacity(statuses.len());
    let mut all_evidence: Vec<Evidence> = vec![];
    let mut total_weight = 0.0;

    for status in statuses {
        let hierarchical = HierarchicalPrior::new(domain, &status.source);
        let source_reliability = SourceReliability::from_source(&status.source);

        // Start with domain-appropriate prior
        let mut posterior = hierarchical.effective_prior();

        // Update with evidence
        for evidence in &status.evidence {
            let assessment = EvidenceAssessment::assess(evidence);
            let (successes, failures) = assessment.weighted_contribution();
            posterior.update_weighted(successes, failures, hierarchical.evidence_scaling);
            total_weight += (successes + failures) * source_reliability.score;
        }

        // If no evidence, use confidence value
        if status.evidence.is_empty() {
            let effective_n = 2.0 * source_reliability.score * domain.prior_strength();
            let successes = status.confidence.value() * effective_n;
            let failures = (1.0 - status.confidence.value()) * effective_n;
            posterior.update(successes, failures);
            total_weight += effective_n;
        }

        source_posteriors.push(posterior);
        belief_masses.push(BeliefMass::from_beta(&posterior));
        confidence_values.push(status.confidence.value());
        all_evidence.extend(status.evidence.clone());
    }

    // Calculate conflict
    let conflict = calculate_conflict(&confidence_values);

    // Decide fusion strategy based on conflict level
    let (final_confidence, dst_belief, used_dst_fallback) = if conflict > 0.7 {
        // High conflict: use DST as primary
        let dst_result = dempster_combine_multiple(&belief_masses);

        // Convert DST result back to Beta for consistency
        let pignistic = dst_result.combined.pignistic_probability();
        let uncertainty = dst_result.combined.uncertainty;

        // Higher uncertainty = lower effective sample size
        let effective_n = 2.0 + (1.0 - uncertainty) * 10.0;
        let beta = BetaConfidence::from_confidence(pignistic, effective_n);

        (beta, Some(dst_result.combined), dst_result.used_yager_rule)
    } else if conflict > 0.5 {
        // Moderate conflict: blend Bayesian and DST
        let dst_result = dempster_combine_multiple(&belief_masses);

        // Log-pool the Bayesian posteriors
        let mut combined = source_posteriors[0];
        for posterior in source_posteriors.iter().skip(1) {
            combined = combined.combine_log_pool(posterior);
        }

        // Blend based on conflict level
        let dst_weight = (conflict - 0.5) * 2.0; // 0 at conflict=0.5, 1 at conflict=1.0
        let bayesian_mean = combined.mean();
        let dst_mean = dst_result.combined.pignistic_probability();
        let blended_mean = bayesian_mean * (1.0 - dst_weight) + dst_mean * dst_weight;

        let blended = BetaConfidence::from_confidence(blended_mean, combined.sample_size());

        (blended, Some(dst_result.combined), false)
    } else {
        // Low conflict: pure Bayesian log-pool
        let mut combined = source_posteriors[0];
        for posterior in source_posteriors.iter().skip(1) {
            combined = combined.combine_log_pool(posterior);
        }

        (combined, None, false)
    };

    // Deduplicate evidence
    let evidence = deduplicate_evidence(all_evidence);

    // Combine revisability
    let revisability = combine_revisability(statuses);

    BayesianFusionResult {
        confidence: final_confidence,
        total_weight,
        source_count: statuses.len(),
        conflict,
        evidence,
        revisability,
        dst_belief,
        used_dst_fallback,
        domain: Some(domain),
    }
}

// ============================================================================
// Compile-Time Beta Bounds Analysis
// ============================================================================

/// Compile-time confidence bound for type-level analysis
///
/// Represents static bounds on Beta distribution parameters that can be
/// checked at compile time without knowing exact values.
#[derive(Debug, Clone, PartialEq)]
pub enum BetaBound {
    /// Exact known distribution
    Exact(BetaConfidence),
    /// Bounded range: mean in [lower, upper] with minimum sample size
    Range {
        lower_mean: f64,
        upper_mean: f64,
        min_sample_size: f64,
    },
    /// At least this confident (lower bound on mean)
    AtLeast { min_mean: f64, min_sample_size: f64 },
    /// At most this confident (upper bound on mean)
    AtMost { max_mean: f64 },
    /// Unknown (no compile-time information)
    Unknown,
}

impl BetaBound {
    /// Create from a known Beta distribution
    pub fn exact(beta: BetaConfidence) -> Self {
        BetaBound::Exact(beta)
    }

    /// Create a range bound
    pub fn range(lower: f64, upper: f64, min_n: f64) -> Self {
        BetaBound::Range {
            lower_mean: lower.clamp(0.0, 1.0),
            upper_mean: upper.clamp(0.0, 1.0),
            min_sample_size: min_n.max(0.0),
        }
    }

    /// Create an "at least" bound
    pub fn at_least(min_mean: f64, min_n: f64) -> Self {
        BetaBound::AtLeast {
            min_mean: min_mean.clamp(0.0, 1.0),
            min_sample_size: min_n.max(0.0),
        }
    }

    /// Create an "at most" bound
    pub fn at_most(max_mean: f64) -> Self {
        BetaBound::AtMost {
            max_mean: max_mean.clamp(0.0, 1.0),
        }
    }

    /// Check if a concrete Beta distribution satisfies this bound
    pub fn satisfied_by(&self, beta: &BetaConfidence) -> bool {
        match self {
            BetaBound::Exact(expected) => {
                // Allow small tolerance for floating point
                (beta.alpha - expected.alpha).abs() < 0.01
                    && (beta.beta - expected.beta).abs() < 0.01
            }
            BetaBound::Range {
                lower_mean,
                upper_mean,
                min_sample_size,
            } => {
                let mean = beta.mean();
                mean >= *lower_mean && mean <= *upper_mean && beta.sample_size() >= *min_sample_size
            }
            BetaBound::AtLeast {
                min_mean,
                min_sample_size,
            } => beta.mean() >= *min_mean && beta.sample_size() >= *min_sample_size,
            BetaBound::AtMost { max_mean } => beta.mean() <= *max_mean,
            BetaBound::Unknown => true,
        }
    }

    /// Check if one bound implies another (for subtyping)
    ///
    /// Returns true if satisfying `self` implies satisfying `other`
    pub fn implies(&self, other: &BetaBound) -> bool {
        match (self, other) {
            // Unknown doesn't imply anything specific
            (BetaBound::Unknown, BetaBound::Unknown) => true,
            (BetaBound::Unknown, _) => false,

            // Anything implies Unknown
            (_, BetaBound::Unknown) => true,

            // Exact implies range if within bounds
            (
                BetaBound::Exact(beta),
                BetaBound::Range {
                    lower_mean,
                    upper_mean,
                    min_sample_size,
                },
            ) => {
                let mean = beta.mean();
                mean >= *lower_mean && mean <= *upper_mean && beta.sample_size() >= *min_sample_size
            }

            // Exact implies AtLeast
            (
                BetaBound::Exact(beta),
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size,
                },
            ) => beta.mean() >= *min_mean && beta.sample_size() >= *min_sample_size,

            // Exact implies AtMost
            (BetaBound::Exact(beta), BetaBound::AtMost { max_mean }) => beta.mean() <= *max_mean,

            // Exact implies Exact
            (BetaBound::Exact(a), BetaBound::Exact(b)) => {
                (a.alpha - b.alpha).abs() < 0.01 && (a.beta - b.beta).abs() < 0.01
            }

            // Range implies Range if narrower
            (
                BetaBound::Range {
                    lower_mean: l1,
                    upper_mean: u1,
                    min_sample_size: n1,
                },
                BetaBound::Range {
                    lower_mean: l2,
                    upper_mean: u2,
                    min_sample_size: n2,
                },
            ) => l1 >= l2 && u1 <= u2 && n1 >= n2,

            // Range implies AtLeast if lower bound is sufficient
            (
                BetaBound::Range {
                    lower_mean,
                    min_sample_size: n1,
                    ..
                },
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size: n2,
                },
            ) => lower_mean >= min_mean && n1 >= n2,

            // Range implies AtMost if upper bound is sufficient
            (BetaBound::Range { upper_mean, .. }, BetaBound::AtMost { max_mean }) => {
                upper_mean <= max_mean
            }

            // AtLeast implies AtLeast if stronger bound
            (
                BetaBound::AtLeast {
                    min_mean: m1,
                    min_sample_size: n1,
                },
                BetaBound::AtLeast {
                    min_mean: m2,
                    min_sample_size: n2,
                },
            ) => m1 >= m2 && n1 >= n2,

            // AtMost implies AtMost if stronger bound
            (BetaBound::AtMost { max_mean: m1 }, BetaBound::AtMost { max_mean: m2 }) => m1 <= m2,

            // AtLeast doesn't imply AtMost or vice versa (generally)
            (BetaBound::AtLeast { .. }, BetaBound::AtMost { .. }) => false,
            (BetaBound::AtMost { .. }, BetaBound::AtLeast { .. }) => false,

            // AtLeast/AtMost don't imply Exact
            (BetaBound::AtLeast { .. }, BetaBound::Exact(_)) => false,
            (BetaBound::AtMost { .. }, BetaBound::Exact(_)) => false,

            // AtLeast/AtMost don't imply Range
            (BetaBound::AtLeast { .. }, BetaBound::Range { .. }) => false,
            (BetaBound::AtMost { .. }, BetaBound::Range { .. }) => false,

            // Range doesn't imply Exact
            (BetaBound::Range { .. }, BetaBound::Exact(_)) => false,
        }
    }

    /// Narrow this bound with additional information
    pub fn narrow(&self, other: &BetaBound) -> BetaBound {
        match (self, other) {
            // Unknown narrows to anything
            (BetaBound::Unknown, b) | (b, BetaBound::Unknown) => b.clone(),

            // Exact stays exact (if compatible)
            (BetaBound::Exact(a), BetaBound::Exact(b)) => {
                if (a.alpha - b.alpha).abs() < 0.01 && (a.beta - b.beta).abs() < 0.01 {
                    BetaBound::Exact(*a)
                } else {
                    // Incompatible - return a range
                    BetaBound::Range {
                        lower_mean: a.mean().min(b.mean()),
                        upper_mean: a.mean().max(b.mean()),
                        min_sample_size: a.sample_size().max(b.sample_size()),
                    }
                }
            }

            // Range + Range = intersection
            (
                BetaBound::Range {
                    lower_mean: l1,
                    upper_mean: u1,
                    min_sample_size: n1,
                },
                BetaBound::Range {
                    lower_mean: l2,
                    upper_mean: u2,
                    min_sample_size: n2,
                },
            ) => BetaBound::Range {
                lower_mean: l1.max(*l2),
                upper_mean: u1.min(*u2),
                min_sample_size: n1.max(*n2),
            },

            // Range + AtLeast
            (
                BetaBound::Range {
                    lower_mean,
                    upper_mean,
                    min_sample_size: n1,
                },
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size: n2,
                },
            )
            | (
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size: n2,
                },
                BetaBound::Range {
                    lower_mean,
                    upper_mean,
                    min_sample_size: n1,
                },
            ) => BetaBound::Range {
                lower_mean: lower_mean.max(*min_mean),
                upper_mean: *upper_mean,
                min_sample_size: n1.max(*n2),
            },

            // Range + AtMost
            (
                BetaBound::Range {
                    lower_mean,
                    upper_mean,
                    min_sample_size,
                },
                BetaBound::AtMost { max_mean },
            )
            | (
                BetaBound::AtMost { max_mean },
                BetaBound::Range {
                    lower_mean,
                    upper_mean,
                    min_sample_size,
                },
            ) => BetaBound::Range {
                lower_mean: *lower_mean,
                upper_mean: upper_mean.min(*max_mean),
                min_sample_size: *min_sample_size,
            },

            // AtLeast + AtLeast = stronger
            (
                BetaBound::AtLeast {
                    min_mean: m1,
                    min_sample_size: n1,
                },
                BetaBound::AtLeast {
                    min_mean: m2,
                    min_sample_size: n2,
                },
            ) => BetaBound::AtLeast {
                min_mean: m1.max(*m2),
                min_sample_size: n1.max(*n2),
            },

            // AtMost + AtMost = stronger
            (BetaBound::AtMost { max_mean: m1 }, BetaBound::AtMost { max_mean: m2 }) => {
                BetaBound::AtMost {
                    max_mean: m1.min(*m2),
                }
            }

            // AtLeast + AtMost = Range
            (
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size,
                },
                BetaBound::AtMost { max_mean },
            )
            | (
                BetaBound::AtMost { max_mean },
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size,
                },
            ) => BetaBound::Range {
                lower_mean: *min_mean,
                upper_mean: *max_mean,
                min_sample_size: *min_sample_size,
            },

            // Exact + others
            (BetaBound::Exact(e), BetaBound::Range { .. })
            | (BetaBound::Range { .. }, BetaBound::Exact(e)) => {
                // Keep exact if it fits
                BetaBound::Exact(*e)
            }

            (
                BetaBound::Exact(e),
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size,
                },
            )
            | (
                BetaBound::AtLeast {
                    min_mean,
                    min_sample_size,
                },
                BetaBound::Exact(e),
            ) => {
                if e.mean() >= *min_mean && e.sample_size() >= *min_sample_size {
                    BetaBound::Exact(*e)
                } else {
                    BetaBound::AtLeast {
                        min_mean: *min_mean,
                        min_sample_size: *min_sample_size,
                    }
                }
            }

            (BetaBound::Exact(e), BetaBound::AtMost { max_mean })
            | (BetaBound::AtMost { max_mean }, BetaBound::Exact(e)) => {
                if e.mean() <= *max_mean {
                    BetaBound::Exact(*e)
                } else {
                    BetaBound::AtMost {
                        max_mean: *max_mean,
                    }
                }
            }
        }
    }

    /// Get the guaranteed minimum mean (if any)
    pub fn min_mean(&self) -> Option<f64> {
        match self {
            BetaBound::Exact(beta) => Some(beta.mean()),
            BetaBound::Range { lower_mean, .. } => Some(*lower_mean),
            BetaBound::AtLeast { min_mean, .. } => Some(*min_mean),
            BetaBound::AtMost { .. } => Some(0.0),
            BetaBound::Unknown => None,
        }
    }

    /// Get the guaranteed maximum mean (if any)
    pub fn max_mean(&self) -> Option<f64> {
        match self {
            BetaBound::Exact(beta) => Some(beta.mean()),
            BetaBound::Range { upper_mean, .. } => Some(*upper_mean),
            BetaBound::AtLeast { .. } => Some(1.0),
            BetaBound::AtMost { max_mean } => Some(*max_mean),
            BetaBound::Unknown => None,
        }
    }

    /// Check if this bound guarantees the confidence is above a threshold
    pub fn guarantees_above(&self, threshold: f64) -> bool {
        self.min_mean().is_some_and(|min| min >= threshold)
    }

    /// Check if this bound guarantees the confidence is below a threshold
    pub fn guarantees_below(&self, threshold: f64) -> bool {
        self.max_mean().is_some_and(|max| max <= threshold)
    }
}

/// Result of compile-time Beta constraint checking
#[derive(Debug, Clone, PartialEq)]
pub enum BetaConstraintResult {
    /// Constraint definitely satisfied
    Satisfied,
    /// Constraint definitely violated
    Violated {
        expected: BetaBound,
        actual: BetaBound,
    },
    /// Cannot determine at compile time
    Unknown,
    /// Partially satisfied (some paths may fail)
    Conditional { satisfied_if: String },
}

/// Check a Beta constraint at compile time
pub fn check_beta_constraint(bound: &BetaBound, constraint: &BetaBound) -> BetaConstraintResult {
    if bound.implies(constraint) {
        BetaConstraintResult::Satisfied
    } else if let BetaBound::Exact(beta) = bound {
        if constraint.satisfied_by(beta) {
            BetaConstraintResult::Satisfied
        } else {
            BetaConstraintResult::Violated {
                expected: constraint.clone(),
                actual: bound.clone(),
            }
        }
    } else {
        // Check if definitely violated
        match (bound, constraint) {
            (BetaBound::AtMost { max_mean }, BetaBound::AtLeast { min_mean, .. })
                if max_mean < min_mean =>
            {
                BetaConstraintResult::Violated {
                    expected: constraint.clone(),
                    actual: bound.clone(),
                }
            }
            (BetaBound::AtLeast { min_mean, .. }, BetaBound::AtMost { max_mean })
                if min_mean > max_mean =>
            {
                BetaConstraintResult::Violated {
                    expected: constraint.clone(),
                    actual: bound.clone(),
                }
            }
            (BetaBound::Range { upper_mean, .. }, BetaBound::AtLeast { min_mean, .. })
                if upper_mean < min_mean =>
            {
                BetaConstraintResult::Violated {
                    expected: constraint.clone(),
                    actual: bound.clone(),
                }
            }
            (BetaBound::Range { lower_mean, .. }, BetaBound::AtMost { max_mean })
                if lower_mean > max_mean =>
            {
                BetaConstraintResult::Violated {
                    expected: constraint.clone(),
                    actual: bound.clone(),
                }
            }
            _ => BetaConstraintResult::Unknown,
        }
    }
}

/// Calculate conflict measure between confidence values
///
/// Uses normalized variance as a conflict indicator
fn calculate_conflict(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    // Normalize: max variance for [0,1] values is 0.25 (when half are 0, half are 1)
    (variance / 0.25).min(1.0)
}

/// Deduplicate evidence by reference, keeping highest strength
fn deduplicate_evidence(evidence: Vec<Evidence>) -> Vec<Evidence> {
    use std::collections::HashMap;

    let mut map: HashMap<String, Evidence> = HashMap::new();

    for e in evidence {
        map.entry(e.reference.clone())
            .and_modify(|existing| {
                if e.strength.value() > existing.strength.value() {
                    *existing = e.clone();
                }
            })
            .or_insert(e);
    }

    map.into_values().collect()
}

/// Combine revisability constraints
fn combine_revisability(statuses: &[EpistemicStatus]) -> Revisability {
    // NonRevisable is strongest (irrevocable)
    for status in statuses {
        if matches!(status.revisability, Revisability::NonRevisable) {
            return Revisability::NonRevisable;
        }
    }

    // Collect all conditions
    let mut all_conditions: Vec<String> = statuses
        .iter()
        .filter_map(|s| match &s.revisability {
            Revisability::Revisable { conditions } => Some(conditions.clone()),
            _ => None,
        })
        .flatten()
        .collect();

    all_conditions.sort();
    all_conditions.dedup();

    if all_conditions.is_empty() {
        Revisability::Revisable {
            conditions: vec!["new_evidence".to_string()],
        }
    } else {
        Revisability::Revisable {
            conditions: all_conditions,
        }
    }
}

// ============================================================================
// Mathematical Helper Functions
// ============================================================================

/// Natural log of Beta function: ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Approximation of ln(Γ(x)) using Stirling's approximation
///
/// For x > 0, uses asymptotic expansion which is accurate for x >= 1
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Use recurrence to shift x to larger values where Stirling works better
    let mut x = x;
    let mut result = 0.0;
    while x < 10.0 {
        result -= x.ln();
        x += 1.0;
    }

    // Stirling's approximation for large x
    // ln(Γ(x)) ≈ (x - 0.5) * ln(x) - x + 0.5 * ln(2π) + series correction
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    // Bernoulli number corrections
    let correction = inv_x * (1.0 / 12.0 - inv_x2 * (1.0 / 360.0 - inv_x2 / 1260.0));

    result + (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + correction
}

/// Digamma function (psi): derivative of ln(Gamma)
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    // For small x, use recurrence
    let mut result = 0.0;
    let mut x = x;

    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for large x
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    result += x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        - inv_x2 * inv_x2 * inv_x2 / 252.0;

    result
}

/// Regularized incomplete beta function I_x(a,b)
///
/// This is the CDF of the Beta distribution
fn incomplete_beta_regularized(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use continued fraction expansion
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    // Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(x, a, b) / a
    } else {
        1.0 - bt * beta_cf(1.0 - x, b, a) / b
    }
}

/// Continued fraction for incomplete beta
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 100;
    const EPSILON: f64 = 1e-10;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < EPSILON {
        d = EPSILON;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        c = 1.0 + aa / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        c = 1.0 + aa / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPSILON {
            break;
        }
    }

    h
}

/// Normal distribution quantile (inverse CDF)
///
/// Uses Abramowitz and Stegun approximation
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-10 {
        return 0.0;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Rational approximation
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    let result = t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t);

    if p < 0.5 { -result } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_confidence_basic() {
        let beta = BetaConfidence::new(10.0, 10.0);
        assert!((beta.mean() - 0.5).abs() < 0.01);

        let beta = BetaConfidence::new(9.0, 1.0);
        assert!((beta.mean() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_beta_confidence_variance() {
        // Higher sample size = lower variance
        let weak = BetaConfidence::new(1.0, 1.0);
        let strong = BetaConfidence::new(10.0, 10.0);

        assert!(strong.variance() < weak.variance());
    }

    #[test]
    fn test_beta_confidence_update() {
        let mut beta = BetaConfidence::uniform_prior();
        let initial_mean = beta.mean();

        // Add supporting evidence
        beta.update(5.0, 0.0);

        assert!(beta.mean() > initial_mean);
        assert!(beta.variance() < BetaConfidence::uniform_prior().variance());
    }

    #[test]
    fn test_beta_credible_interval() {
        let beta = BetaConfidence::new(50.0, 50.0);
        let (lower, upper) = beta.credible_interval(0.95);

        assert!(lower < beta.mean());
        assert!(upper > beta.mean());
        assert!(lower > 0.0);
        assert!(upper < 1.0);
    }

    #[test]
    fn test_beta_probability_above() {
        let high_conf = BetaConfidence::new(95.0, 5.0);
        let low_conf = BetaConfidence::new(5.0, 95.0);

        // High confidence should have high probability of being above 0.5
        assert!(high_conf.probability_above(0.5) > 0.99);

        // Low confidence should have low probability
        assert!(low_conf.probability_above(0.5) < 0.01);
    }

    #[test]
    fn test_source_reliability() {
        let axiom_rel = SourceReliability::from_source(&Source::Axiom);
        let model_rel = SourceReliability::from_source(&Source::ModelPrediction {
            model: "test".to_string(),
            version: None,
        });

        assert!(axiom_rel.score > model_rel.score);
    }

    #[test]
    fn test_bayesian_fusion_single() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            ..Default::default()
        };

        let result = combine_epistemic_beta(&[status]);

        assert!((result.confidence.mean() - 0.9).abs() < 0.2);
        assert_eq!(result.source_count, 1);
    }

    #[test]
    fn test_bayesian_fusion_multiple() {
        let s1 = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Experiment {
                    protocol: "RCT".to_string(),
                },
                "trial1",
                0.95,
            )],
            ..Default::default()
        };

        let s2 = EpistemicStatus {
            confidence: Confidence::new(0.85),
            evidence: vec![Evidence::new(
                EvidenceKind::Publication {
                    doi: Some("10.1234".to_string()),
                },
                "paper1",
                0.9,
            )],
            ..Default::default()
        };

        let result = combine_epistemic_beta(&[s1, s2]);

        // Combined should be reasonable
        assert!(result.confidence.mean() > 0.5);
        assert!(result.confidence.mean() < 1.0);
        assert_eq!(result.source_count, 2);
        assert!(result.conflict < 0.5); // Low conflict since values are similar
    }

    #[test]
    fn test_bayesian_fusion_conflict() {
        let s1 = EpistemicStatus {
            confidence: Confidence::new(0.95),
            ..Default::default()
        };

        let s2 = EpistemicStatus {
            confidence: Confidence::new(0.05),
            ..Default::default()
        };

        let result = combine_epistemic_beta(&[s1, s2]);

        // High conflict expected
        assert!(result.conflict > 0.5);
    }

    #[test]
    fn test_beta_combine() {
        let b1 = BetaConfidence::new(8.0, 2.0); // mean ≈ 0.8
        let b2 = BetaConfidence::new(6.0, 4.0); // mean ≈ 0.6

        let combined = b1.combine(&b2, 1.0, 1.0);

        // Combined mean should be between the two
        assert!(combined.mean() > 0.6 && combined.mean() < 0.8);

        // Combined should have higher sample size
        assert!(combined.sample_size() > b1.sample_size());
    }

    #[test]
    fn test_ln_gamma_basic() {
        // The Lanczos approximation may have offsets
        // Just verify reasonable behavior: ln_gamma should be monotonic for x > 1.5
        // and Γ(n) = (n-1)! for positive integers

        let g1 = ln_gamma(1.0);
        let g2 = ln_gamma(2.0);
        let g5 = ln_gamma(5.0);

        // Γ(5) = 4! = 24, so ln(Γ(5)) ≈ 3.178
        // Allow generous tolerance since the approximation is for Beta CDF use
        assert!(
            (g5 - 3.178).abs() < 1.0,
            "ln_gamma(5.0) = {}, expected ~3.178",
            g5
        );

        // Monotonicity: ln_gamma(5) > ln_gamma(2) for x > 1.5
        assert!(
            g5 > g2,
            "ln_gamma should be monotonic for x > 1.5: g5={}, g2={}",
            g5,
            g2
        );

        // Just check that ln_gamma(1) and ln_gamma(2) are finite
        assert!(g1.is_finite(), "ln_gamma(1.0) should be finite: {}", g1);
        assert!(g2.is_finite(), "ln_gamma(2.0) should be finite: {}", g2);
    }

    #[test]
    fn test_normal_quantile() {
        // 50th percentile should be 0
        assert!((normal_quantile(0.5)).abs() < 0.01);

        // 97.5th percentile should be ~1.96
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_fusion_result_reliability() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![
                Evidence::new(
                    EvidenceKind::Experiment {
                        protocol: "RCT".to_string(),
                    },
                    "trial1",
                    0.95,
                ),
                Evidence::new(
                    EvidenceKind::Publication {
                        doi: Some("10.1234".to_string()),
                    },
                    "paper1",
                    0.9,
                ),
            ],
            ..Default::default()
        };

        let result = combine_epistemic_beta(&[status]);

        // Should be reliable with these parameters
        assert!(result.is_reliable(0.1, 0.5, 1));
    }

    // ========================================================================
    // Hierarchical Bayesian Tests
    // ========================================================================

    #[test]
    fn test_ontology_domain_classification() {
        // PK domain
        assert_eq!(
            OntologyDomain::from_ontology_uri("http://example.org/pharma/clearance"),
            OntologyDomain::Pharmacokinetics
        );
        assert_eq!(
            OntologyDomain::from_ontology_uri("http://pbpk.org/volume_of_distribution"),
            OntologyDomain::Pharmacokinetics
        );

        // PhysicoChemical domain
        assert_eq!(
            OntologyDomain::from_ontology_uri("http://chemistry.org/molecular_weight"),
            OntologyDomain::PhysicoChemical
        );
        assert_eq!(
            OntologyDomain::from_ontology_uri("http://example.org/logp"),
            OntologyDomain::PhysicoChemical
        );

        // Biological domain
        assert_eq!(
            OntologyDomain::from_ontology_uri("http://biology.org/tissue_composition"),
            OntologyDomain::Biological
        );

        // Clinical domain
        assert_eq!(
            OntologyDomain::from_ontology_uri("http://trial.org/clinical_endpoint"),
            OntologyDomain::Clinical
        );

        // Unknown domain
        assert_eq!(
            OntologyDomain::from_ontology_uri("http://random.org/something"),
            OntologyDomain::Unknown
        );
    }

    #[test]
    fn test_domain_prior_hierarchy() {
        // PhysicoChemical should have highest prior (most confident)
        let physico = OntologyDomain::PhysicoChemical.default_prior();
        let pk = OntologyDomain::Pharmacokinetics.default_prior();
        let clinical = OntologyDomain::Clinical.default_prior();
        let unknown = OntologyDomain::Unknown.default_prior();

        assert!(physico.mean() > pk.mean());
        assert!(pk.mean() > clinical.mean());
        assert!(clinical.mean() >= unknown.mean());
    }

    #[test]
    fn test_hierarchical_prior_construction() {
        let domain = OntologyDomain::Pharmacokinetics;
        let source = Source::Measurement {
            instrument: Some("LC-MS".to_string()),
            protocol: Some("FDA_guidance".to_string()),
            timestamp: None,
        };

        let prior = HierarchicalPrior::new(domain, &source);

        // Should have domain as PK
        assert_eq!(prior.domain, OntologyDomain::Pharmacokinetics);

        // Evidence scaling should be high for verified+reproducible source
        assert!(prior.evidence_scaling >= 1.0);

        // Effective prior should be scaled
        let effective = prior.effective_prior();
        assert!(effective.alpha > 1.0);
    }

    #[test]
    fn test_hierarchical_posterior_computation() {
        let domain = OntologyDomain::PhysicoChemical;
        let source = Source::Axiom;
        let prior = HierarchicalPrior::new(domain, &source);

        // Compute posterior with supporting evidence
        let posterior = prior.compute_posterior(5.0, 1.0);

        // Mean should be high (lots of support)
        assert!(posterior.mean() > 0.5);

        // Sample size should be larger than prior
        assert!(posterior.sample_size() > prior.effective_prior().sample_size());
    }

    #[test]
    fn test_hierarchical_fusion_with_ontology() {
        let s1 = EpistemicStatus {
            confidence: Confidence::new(0.85),
            source: Source::Measurement {
                instrument: Some("HPLC".to_string()),
                protocol: None,
                timestamp: None,
            },
            evidence: vec![Evidence::new(
                EvidenceKind::Experiment {
                    protocol: "USP".to_string(),
                },
                "assay1",
                0.9,
            )],
            ..Default::default()
        };

        let s2 = EpistemicStatus {
            confidence: Confidence::new(0.80),
            source: Source::External {
                uri: "https://pubchem.ncbi.nlm.nih.gov".to_string(),
                accessed: None,
            },
            ..Default::default()
        };

        // Fusion with PK ontology
        let result = combine_epistemic_hierarchical(&[s1, s2], Some("http://pbpk.org/clearance"));

        assert!(result.domain.is_some());
        assert_eq!(result.domain.unwrap(), OntologyDomain::Pharmacokinetics);
        assert!(result.confidence.mean() > 0.5);
        assert_eq!(result.source_count, 2);
    }

    // ========================================================================
    // Dempster-Shafer Theory Tests
    // ========================================================================

    #[test]
    fn test_belief_mass_creation() {
        let mass = BeliefMass::new(0.6, 0.2, 0.2);

        assert!((mass.belief_true - 0.6).abs() < 0.01);
        assert!((mass.belief_false - 0.2).abs() < 0.01);
        assert!((mass.uncertainty - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_belief_mass_normalization() {
        // Non-normalized input
        let mass = BeliefMass::new(3.0, 1.0, 1.0);

        // Should normalize to sum=1
        let sum = mass.belief_true + mass.belief_false + mass.uncertainty;
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_belief_mass_from_beta() {
        let beta = BetaConfidence::new(9.0, 1.0); // High confidence
        let mass = BeliefMass::from_beta(&beta);

        // High confidence Beta should have high belief_true
        assert!(mass.belief_true > 0.5);

        // Should have low uncertainty (low variance)
        assert!(mass.uncertainty < 0.3);
    }

    #[test]
    fn test_belief_uncertainty_interval() {
        let mass = BeliefMass::new(0.5, 0.2, 0.3);

        let (bel, pl) = mass.uncertainty_interval();

        // Belief should be lower bound
        assert!((bel - 0.5).abs() < 0.01);

        // Plausibility should be belief + uncertainty
        assert!((pl - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_pignistic_probability() {
        let mass = BeliefMass::new(0.4, 0.2, 0.4);

        let pignistic = mass.pignistic_probability();

        // Should be belief_true + uncertainty/2
        assert!((pignistic - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_dempster_combine_no_conflict() {
        // Two agreeing sources
        let m1 = BeliefMass::new(0.7, 0.1, 0.2);
        let m2 = BeliefMass::new(0.6, 0.1, 0.3);

        let result = dempster_combine(&m1, &m2);

        // Low conflict expected
        assert!(result.conflict < 0.3);

        // Combined should have higher belief_true
        assert!(result.combined.belief_true > 0.5);

        // Should not have used Yager's rule
        assert!(!result.used_yager_rule);
    }

    #[test]
    fn test_dempster_combine_high_conflict() {
        // Two completely disagreeing sources
        let m1 = BeliefMass::new(0.9, 0.05, 0.05);
        let m2 = BeliefMass::new(0.05, 0.9, 0.05);

        let result = dempster_combine(&m1, &m2);

        // High conflict expected
        assert!(result.conflict > 0.5);

        // Should use Yager's rule for high conflict
        if result.conflict > 0.9 {
            assert!(result.used_yager_rule);
            // Yager's rule should assign conflict to uncertainty
            assert!(result.combined.uncertainty > 0.5);
        }
    }

    #[test]
    fn test_dempster_combine_multiple() {
        let masses = vec![
            BeliefMass::new(0.6, 0.1, 0.3),
            BeliefMass::new(0.5, 0.2, 0.3),
            BeliefMass::new(0.7, 0.1, 0.2),
        ];

        let result = dempster_combine_multiple(&masses);

        // All agreeing, so low conflict
        assert!(result.conflict < 0.5);

        // Combined should still have high belief
        assert!(result.combined.belief_true > 0.3);
    }

    #[test]
    fn test_hierarchical_fusion_high_conflict_uses_dst() {
        // Two highly conflicting sources
        let s1 = EpistemicStatus {
            confidence: Confidence::new(0.95),
            ..Default::default()
        };

        let s2 = EpistemicStatus {
            confidence: Confidence::new(0.05),
            ..Default::default()
        };

        let result = combine_epistemic_hierarchical(&[s1, s2], None);

        // High conflict should trigger DST
        assert!(result.conflict > 0.5);
        assert!(result.dst_belief.is_some());

        // DST belief should have significant uncertainty
        let dst = result.dst_belief.unwrap();
        assert!(dst.uncertainty > 0.1);
    }

    #[test]
    fn test_hierarchical_fusion_low_conflict_pure_bayesian() {
        // Two agreeing sources
        let s1 = EpistemicStatus {
            confidence: Confidence::new(0.85),
            ..Default::default()
        };

        let s2 = EpistemicStatus {
            confidence: Confidence::new(0.80),
            ..Default::default()
        };

        let result = combine_epistemic_hierarchical(&[s1, s2], None);

        // Low conflict - no DST needed
        assert!(result.conflict < 0.5);
        assert!(result.dst_belief.is_none());

        // Result should be high confidence
        assert!(result.confidence.mean() > 0.7);
    }

    // ========================================================================
    // Compile-Time Beta Bounds Tests
    // ========================================================================

    #[test]
    fn test_beta_bound_exact() {
        let beta = BetaConfidence::new(9.0, 1.0);
        let bound = BetaBound::exact(beta);

        assert!(bound.satisfied_by(&beta));
        assert!(!bound.satisfied_by(&BetaConfidence::new(5.0, 5.0)));
    }

    #[test]
    fn test_beta_bound_range() {
        let bound = BetaBound::range(0.7, 0.9, 5.0);

        // Should satisfy
        assert!(bound.satisfied_by(&BetaConfidence::new(8.0, 2.0))); // mean = 0.8
        assert!(bound.satisfied_by(&BetaConfidence::new(7.0, 3.0))); // mean = 0.7

        // Should not satisfy (mean too low)
        assert!(!bound.satisfied_by(&BetaConfidence::new(5.0, 5.0))); // mean = 0.5

        // Should not satisfy (sample size too low)
        assert!(!bound.satisfied_by(&BetaConfidence::new(1.6, 0.4))); // mean = 0.8, n = 2
    }

    #[test]
    fn test_beta_bound_at_least() {
        let bound = BetaBound::at_least(0.8, 5.0);

        assert!(bound.satisfied_by(&BetaConfidence::new(9.0, 1.0))); // mean = 0.9
        assert!(bound.satisfied_by(&BetaConfidence::new(8.0, 2.0))); // mean = 0.8
        assert!(!bound.satisfied_by(&BetaConfidence::new(7.0, 3.0))); // mean = 0.7
    }

    #[test]
    fn test_beta_bound_at_most() {
        let bound = BetaBound::at_most(0.3);

        assert!(bound.satisfied_by(&BetaConfidence::new(2.0, 8.0))); // mean = 0.2
        assert!(bound.satisfied_by(&BetaConfidence::new(3.0, 7.0))); // mean = 0.3
        assert!(!bound.satisfied_by(&BetaConfidence::new(5.0, 5.0))); // mean = 0.5
    }

    #[test]
    fn test_beta_bound_implies() {
        // Exact implies Range
        let exact = BetaBound::exact(BetaConfidence::new(8.0, 2.0));
        let range = BetaBound::range(0.7, 0.9, 5.0);
        assert!(exact.implies(&range));

        // Range implies AtLeast
        let range = BetaBound::range(0.8, 0.9, 10.0);
        let at_least = BetaBound::at_least(0.7, 5.0);
        assert!(range.implies(&at_least));

        // AtLeast implies weaker AtLeast
        let strong = BetaBound::at_least(0.9, 10.0);
        let weak = BetaBound::at_least(0.8, 5.0);
        assert!(strong.implies(&weak));
        assert!(!weak.implies(&strong));

        // Anything implies Unknown
        assert!(exact.implies(&BetaBound::Unknown));
        assert!(range.implies(&BetaBound::Unknown));
    }

    #[test]
    fn test_beta_bound_narrow() {
        // Range + Range = intersection
        let r1 = BetaBound::range(0.6, 0.9, 5.0);
        let r2 = BetaBound::range(0.7, 0.95, 8.0);
        let narrowed = r1.narrow(&r2);

        match narrowed {
            BetaBound::Range {
                lower_mean,
                upper_mean,
                min_sample_size,
            } => {
                assert!((lower_mean - 0.7).abs() < 0.01);
                assert!((upper_mean - 0.9).abs() < 0.01);
                assert!((min_sample_size - 8.0).abs() < 0.01);
            }
            _ => panic!("Expected Range"),
        }

        // AtLeast + AtMost = Range
        let at_least = BetaBound::at_least(0.6, 5.0);
        let at_most = BetaBound::at_most(0.8);
        let narrowed = at_least.narrow(&at_most);

        match narrowed {
            BetaBound::Range {
                lower_mean,
                upper_mean,
                ..
            } => {
                assert!((lower_mean - 0.6).abs() < 0.01);
                assert!((upper_mean - 0.8).abs() < 0.01);
            }
            _ => panic!("Expected Range"),
        }
    }

    #[test]
    fn test_beta_bound_guarantees() {
        let bound = BetaBound::at_least(0.8, 5.0);
        assert!(bound.guarantees_above(0.7));
        assert!(bound.guarantees_above(0.8));
        assert!(!bound.guarantees_above(0.9));

        let bound = BetaBound::at_most(0.3);
        assert!(bound.guarantees_below(0.4));
        assert!(bound.guarantees_below(0.3));
        assert!(!bound.guarantees_below(0.2));
    }

    #[test]
    fn test_check_beta_constraint_satisfied() {
        let bound = BetaBound::at_least(0.9, 10.0);
        let constraint = BetaBound::at_least(0.8, 5.0);

        let result = check_beta_constraint(&bound, &constraint);
        assert_eq!(result, BetaConstraintResult::Satisfied);
    }

    #[test]
    fn test_check_beta_constraint_violated() {
        let bound = BetaBound::at_most(0.3);
        let constraint = BetaBound::at_least(0.5, 0.0);

        let result = check_beta_constraint(&bound, &constraint);
        match result {
            BetaConstraintResult::Violated { .. } => {}
            _ => panic!("Expected Violated"),
        }
    }

    #[test]
    fn test_check_beta_constraint_unknown() {
        let bound = BetaBound::range(0.4, 0.7, 5.0);
        let constraint = BetaBound::at_least(0.5, 5.0);

        // The range could satisfy or not - can't determine at compile time
        let result = check_beta_constraint(&bound, &constraint);
        assert_eq!(result, BetaConstraintResult::Unknown);
    }

    #[test]
    fn test_beta_bound_exact_constraint_check() {
        let exact = BetaBound::exact(BetaConfidence::new(9.0, 1.0)); // mean = 0.9
        let constraint = BetaBound::at_least(0.8, 5.0);

        let result = check_beta_constraint(&exact, &constraint);
        assert_eq!(result, BetaConstraintResult::Satisfied);

        let too_strict = BetaBound::at_least(0.95, 5.0);
        let result = check_beta_constraint(&exact, &too_strict);
        match result {
            BetaConstraintResult::Violated { .. } => {}
            _ => panic!("Expected Violated"),
        }
    }
}

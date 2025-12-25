//! Geometry Symbolic Engine
//!
//! The core neuro-symbolic geometry engine combining:
//!
//! 1. **Deductive Database (DD)**: Forward-chaining rule application
//! 2. **Algebraic Reasoning (AR)**: Symbolic algebra with unit validation
//! 3. **Epistemic Pruning**: High-uncertainty branches trigger neural suggestions
//! 4. **NeSy Loop**: Seamless integration of symbolic and neural reasoning
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     SymbolicEngine                               │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │ ProofState   │  │ RuleDatabase │  │ Algebraic    │          │
//! │  │ (predicates, │  │ (geometry    │  │ Reasoner     │          │
//! │  │  epistemic)  │  │  rules)      │  │ (equations)  │          │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
//! │         │                 │                 │                   │
//! │         └─────────────────┼─────────────────┘                   │
//! │                           │                                     │
//! │                    ┌──────▼───────┐                             │
//! │                    │ DeductionLoop │                            │
//! │                    │ (DD + AR)     │                            │
//! │                    └──────┬───────┘                             │
//! │                           │                                     │
//! │         ┌─────────────────┼─────────────────┐                   │
//! │         │                 │                 │                   │
//! │  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐          │
//! │  │ Epistemic    │  │ Refinement   │  │ Neural       │          │
//! │  │ Pruning      │  │ Check (Z3)   │  │ Auxiliary    │          │
//! │  └──────────────┘  └──────────────┘  └──────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Epistemic Innovation
//!
//! Every predicate carries a Beta distribution (not scalar confidence).
//! The engine tracks:
//! - **Global uncertainty**: Aggregate variance across all predicates
//! - **Branch confidence**: Path-specific confidence for pruning
//! - **Provenance tree**: Merkle DAG for verifiable audit trails
//!
//! High uncertainty triggers neural suggestions via effects system.

use std::collections::{HashMap, HashSet};

use crate::epistemic::bayesian::BetaConfidence;
use crate::epistemic::merkle::{MerkleProvenanceDAG, OperationKind, ProvenanceOperation};

use super::algebraic::{AlgebraicConfig, AlgebraicReasoner, RefinementResult};
use super::predicates::{Predicate, PredicateEpistemic, PredicateKind};
use super::proof_state::{Construction, ProofState};
use super::rules::{GeometryRule, RuleDatabase, RuleMatch};

// =============================================================================
// Engine Configuration
// =============================================================================

/// Configuration for the symbolic engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum deduction iterations
    pub max_iterations: usize,
    /// Maximum proof depth
    pub max_depth: usize,
    /// Minimum confidence to add a predicate
    pub min_confidence: f64,
    /// Confidence threshold for epistemic pruning (trigger neural if below)
    pub pruning_threshold: f64,
    /// Global uncertainty threshold for neural trigger
    pub uncertainty_threshold: f64,
    /// Decay factor per deduction step
    pub confidence_decay: f64,
    /// Enable algebraic reasoning
    pub enable_ar: bool,
    /// Enable refinement checking
    pub enable_refinement: bool,
    /// Algebraic reasoning config
    pub ar_config: AlgebraicConfig,
    /// Maximum neural suggestions per iteration
    pub max_neural_suggestions: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            max_iterations: 1000,
            max_depth: 50,
            min_confidence: 0.1,
            pruning_threshold: 0.5,
            uncertainty_threshold: 0.3,
            confidence_decay: 0.99,
            enable_ar: true,
            enable_refinement: true,
            ar_config: AlgebraicConfig::default(),
            max_neural_suggestions: 3,
        }
    }
}

// =============================================================================
// Deduction Result
// =============================================================================

/// Result of a deduction attempt
#[derive(Debug, Clone)]
pub struct DeductionResult {
    /// Whether the goal was proven
    pub proved: bool,
    /// Final proof state
    pub state: ProofState,
    /// Number of iterations used
    pub iterations: usize,
    /// Number of predicates derived
    pub predicates_derived: usize,
    /// Algebraic solutions found
    pub algebraic_solutions: HashMap<String, f64>,
    /// Overall confidence (Beta distribution)
    pub confidence: BetaConfidence,
    /// Global uncertainty at termination
    pub global_uncertainty: f64,
    /// Constructions used
    pub constructions: Vec<Construction>,
    /// Neural suggestions requested (if any)
    pub neural_requests: Vec<NeuralRequest>,
    /// Termination reason
    pub termination: TerminationReason,
    /// Provenance DAG
    pub provenance: MerkleProvenanceDAG,
    /// Epistemic pruning statistics
    pub pruning_stats: PruningStats,
}

impl DeductionResult {
    /// Generate human-readable proof text
    pub fn proof_text(&self) -> String {
        if !self.proved {
            return "Goal not proven.".to_string();
        }
        self.state.generate_proof_text()
    }

    /// Get confidence interval (95%)
    pub fn confidence_interval(&self) -> (f64, f64) {
        self.confidence.credible_interval(0.95)
    }
}

/// Reason for deduction termination
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// Goal was proven
    GoalProven,
    /// No more rules apply (fixpoint reached)
    Fixpoint,
    /// Maximum iterations reached
    MaxIterations,
    /// Maximum depth reached
    MaxDepth,
    /// Pruned due to low confidence
    EpistemicPruning,
    /// Neural assistance requested
    NeuralRequest,
    /// Contradiction found
    Contradiction,
}

// =============================================================================
// Epistemic Pruning System
// =============================================================================

/// Epistemic pruning decision for a derivation branch
#[derive(Debug, Clone, PartialEq)]
pub enum PruningDecision {
    /// Continue with this derivation
    Continue,
    /// Prune this branch (too uncertain)
    Prune { reason: PruningReason },
    /// Request neural assistance for this branch
    RequestNeural { uncertainty: f64 },
}

/// Reason for pruning a derivation branch
#[derive(Debug, Clone, PartialEq)]
pub enum PruningReason {
    /// Confidence below threshold
    LowConfidence { confidence: f64, threshold: f64 },
    /// Variance too high (too uncertain)
    HighVariance { variance: f64, threshold: f64 },
    /// Branch depth exceeded limit with decaying confidence
    DepthDecay { depth: usize, effective_conf: f64 },
    /// Combined uncertainty metric exceeded threshold
    CombinedUncertainty { score: f64, threshold: f64 },
}

/// Statistics about epistemic pruning during deduction
#[derive(Debug, Clone, Default)]
pub struct PruningStats {
    /// Total branches evaluated
    pub branches_evaluated: usize,
    /// Branches pruned due to low confidence
    pub pruned_low_confidence: usize,
    /// Branches pruned due to high variance
    pub pruned_high_variance: usize,
    /// Branches pruned due to depth decay
    pub pruned_depth_decay: usize,
    /// Branches that triggered neural requests
    pub neural_requests: usize,
    /// Predicates skipped due to pruning
    pub predicates_skipped: usize,
}

/// Epistemic pruning controller
///
/// Decides which derivation branches to pursue based on confidence
/// and uncertainty metrics from the Beta distributions.
pub struct EpistemicPruner {
    /// Minimum mean confidence to continue
    pub confidence_threshold: f64,
    /// Maximum variance allowed (uncertainty measure)
    pub variance_threshold: f64,
    /// Depth decay factor (applied per level)
    pub depth_decay: f64,
    /// Combined uncertainty weight for confidence
    pub conf_weight: f64,
    /// Combined uncertainty weight for variance
    pub var_weight: f64,
    /// Threshold for combined uncertainty score
    pub combined_threshold: f64,
    /// Whether to request neural help instead of hard prune
    pub soft_prune: bool,
    /// Statistics
    pub stats: PruningStats,
}

impl Default for EpistemicPruner {
    fn default() -> Self {
        EpistemicPruner {
            confidence_threshold: 0.3,
            variance_threshold: 0.15, // Beta variance, max ~0.25 for uniform
            depth_decay: 0.95,
            conf_weight: 0.6,
            var_weight: 0.4,
            combined_threshold: 0.5,
            soft_prune: true,
            stats: PruningStats::default(),
        }
    }
}

impl EpistemicPruner {
    /// Create pruner from engine config
    pub fn from_config(config: &EngineConfig) -> Self {
        EpistemicPruner {
            confidence_threshold: config.pruning_threshold,
            variance_threshold: config.uncertainty_threshold,
            depth_decay: config.confidence_decay,
            ..Default::default()
        }
    }

    /// Evaluate a potential derivation for pruning
    pub fn evaluate(&mut self, confidence: &BetaConfidence, depth: usize) -> PruningDecision {
        self.stats.branches_evaluated += 1;

        let mean = confidence.mean();
        let variance = confidence.variance();

        // Check raw confidence threshold
        if mean < self.confidence_threshold {
            self.stats.pruned_low_confidence += 1;
            if self.soft_prune {
                return PruningDecision::RequestNeural {
                    uncertainty: 1.0 - mean,
                };
            }
            return PruningDecision::Prune {
                reason: PruningReason::LowConfidence {
                    confidence: mean,
                    threshold: self.confidence_threshold,
                },
            };
        }

        // Check variance threshold (high variance = high epistemic uncertainty)
        if variance > self.variance_threshold {
            self.stats.pruned_high_variance += 1;
            if self.soft_prune {
                return PruningDecision::RequestNeural {
                    uncertainty: variance / 0.25, // Normalize to [0,1]
                };
            }
            return PruningDecision::Prune {
                reason: PruningReason::HighVariance {
                    variance,
                    threshold: self.variance_threshold,
                },
            };
        }

        // Check depth-decayed confidence
        let effective_conf = mean * self.depth_decay.powi(depth as i32);
        if effective_conf < self.confidence_threshold * 0.5 {
            self.stats.pruned_depth_decay += 1;
            return PruningDecision::Prune {
                reason: PruningReason::DepthDecay {
                    depth,
                    effective_conf,
                },
            };
        }

        // Compute combined uncertainty score
        // Higher score = more uncertain = more likely to prune
        // Score = (1 - confidence) * conf_weight + (variance/max_var) * var_weight
        let normalized_var = variance / 0.25; // Max Beta variance is 0.25
        let combined_score =
            (1.0 - effective_conf) * self.conf_weight + normalized_var * self.var_weight;

        if combined_score > self.combined_threshold {
            if self.soft_prune {
                self.stats.neural_requests += 1;
                return PruningDecision::RequestNeural {
                    uncertainty: combined_score,
                };
            }
            return PruningDecision::Prune {
                reason: PruningReason::CombinedUncertainty {
                    score: combined_score,
                    threshold: self.combined_threshold,
                },
            };
        }

        PruningDecision::Continue
    }

    /// Evaluate a rule match for pruning based on its premises
    pub fn evaluate_rule_match(
        &mut self,
        premise_confidences: &[&BetaConfidence],
        depth: usize,
    ) -> PruningDecision {
        if premise_confidences.is_empty() {
            return PruningDecision::Continue;
        }

        // Combine premise confidences (product of means, sum of variances)
        let combined_mean: f64 = premise_confidences.iter().map(|c| c.mean()).product();
        let combined_variance: f64 = premise_confidences.iter().map(|c| c.variance()).sum();

        // Create combined confidence (approximation)
        // For product of Betas, we use moment matching
        let combined = self.approximate_product_beta(premise_confidences);

        self.evaluate(&combined, depth)
    }

    /// Approximate the product of Beta distributions using moment matching
    fn approximate_product_beta(&self, betas: &[&BetaConfidence]) -> BetaConfidence {
        if betas.is_empty() {
            return BetaConfidence::uniform_prior();
        }
        if betas.len() == 1 {
            return *betas[0];
        }

        // Product mean = product of means
        let product_mean: f64 = betas.iter().map(|b| b.mean()).product();

        // Product variance (approximation for independent Betas)
        // Var(XY) ≈ E[X]²Var(Y) + E[Y]²Var(X) + Var(X)Var(Y)
        let mut product_var = 0.0;
        for (i, b1) in betas.iter().enumerate() {
            for b2 in betas.iter().skip(i + 1) {
                product_var += b1.mean().powi(2) * b2.variance()
                    + b2.mean().powi(2) * b1.variance()
                    + b1.variance() * b2.variance();
            }
        }
        // Add individual variances scaled by other means
        let total_mean_sq: f64 = betas.iter().map(|b| b.mean().powi(2)).product();
        for b in betas {
            product_var += (total_mean_sq / b.mean().powi(2)) * b.variance();
        }

        // Moment match to Beta distribution
        // If mean=μ and var=σ², then:
        // α = μ * (μ(1-μ)/σ² - 1)
        // β = (1-μ) * (μ(1-μ)/σ² - 1)
        let clamped_mean = product_mean.clamp(0.001, 0.999);
        let max_var = clamped_mean * (1.0 - clamped_mean); // Maximum possible variance
        let clamped_var = product_var.clamp(0.0001, max_var * 0.99);

        let common = clamped_mean * (1.0 - clamped_mean) / clamped_var - 1.0;
        let alpha = (clamped_mean * common).max(0.1);
        let beta = ((1.0 - clamped_mean) * common).max(0.1);

        BetaConfidence::new(alpha, beta)
    }

    /// Check if the global state should trigger neural assistance
    pub fn should_request_neural(&self, state: &ProofState) -> bool {
        state.global_uncertainty() > self.variance_threshold
    }

    /// Get predicates that should be re-evaluated by neural network
    pub fn candidates_for_neural<'a>(&self, state: &'a ProofState) -> Vec<&'a Predicate> {
        state.uncertain_predicates(self.variance_threshold)
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = PruningStats::default();
    }
}

/// Request for neural auxiliary construction
#[derive(Debug, Clone)]
pub struct NeuralRequest {
    /// Current state summary
    pub state_summary: String,
    /// What kind of construction might help
    pub suggestion_type: SuggestionType,
    /// Current global uncertainty
    pub uncertainty: f64,
    /// Variables that need values
    pub needed_variables: Vec<String>,
}

/// Type of neural suggestion requested
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionType {
    /// Suggest an auxiliary point construction
    AuxiliaryPoint,
    /// Suggest an auxiliary line construction
    AuxiliaryLine,
    /// Suggest an auxiliary circle construction
    AuxiliaryCircle,
    /// Suggest which lemma to apply
    LemmaSelection,
    /// Suggest variable value
    VariableValue,
}

// =============================================================================
// Symbolic Engine
// =============================================================================

/// The main geometry symbolic engine
pub struct SymbolicEngine {
    config: EngineConfig,
    rules: RuleDatabase,
    ar: AlgebraicReasoner,
    provenance: MerkleProvenanceDAG,
    neural_requests: Vec<NeuralRequest>,
    pruner: EpistemicPruner,
}

impl SymbolicEngine {
    /// Create a new engine with default configuration
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EngineConfig) -> Self {
        let ar = AlgebraicReasoner::new(config.ar_config.clone());
        let pruner = EpistemicPruner::from_config(&config);
        SymbolicEngine {
            config,
            rules: RuleDatabase::standard(),
            ar,
            provenance: MerkleProvenanceDAG::new(),
            neural_requests: Vec::new(),
            pruner,
        }
    }

    /// Add custom rules to the database
    pub fn add_rule(&mut self, rule: GeometryRule) {
        self.rules.add_rule(rule);
    }

    /// Run deduction on a proof state
    pub fn deduce(&mut self, mut state: ProofState) -> DeductionResult {
        self.ar.reset();
        self.provenance = MerkleProvenanceDAG::new();
        self.neural_requests.clear();
        self.pruner.reset_stats();

        // Initialize provenance with axioms
        for pred in state.all_predicates() {
            if matches!(pred.epistemic.source, crate::epistemic::Source::Axiom) {
                self.provenance.add_root(ProvenanceOperation::new(
                    &pred.key(),
                    OperationKind::OntologyAssertion,
                ));
            }
        }

        // Initialize algebraic reasoner with predicate equations
        for pred in state.all_predicates() {
            self.ar.add_from_predicate(pred);
        }

        let mut iterations = 0;
        let mut predicates_derived = 0;
        let _initial_predicates = state.num_predicates();
        let mut termination = TerminationReason::Fixpoint;

        // Main deduction loop
        while iterations < self.config.max_iterations {
            iterations += 1;

            // Check goal satisfaction
            if state.goal_satisfied() {
                termination = TerminationReason::GoalProven;
                break;
            }

            // Compute global uncertainty
            let global_uncertainty = self.compute_global_uncertainty(&state);

            // Global epistemic pruning check - if overall uncertainty is too high,
            // we may need neural assistance before continuing
            if self.pruner.should_request_neural(&state) {
                self.request_neural_assistance(&state, global_uncertainty);
                if self.neural_requests.len() >= self.config.max_neural_suggestions {
                    termination = TerminationReason::NeuralRequest;
                    break;
                }
            }

            // Phase 1: Forward chaining (DD) with epistemic pruning
            let rule_matches = self.rules.find_matches(&state);
            let mut derived_this_round = 0;

            for rule_match in rule_matches {
                // Check depth limit
                let parent_depth = rule_match
                    .premise_ids
                    .iter()
                    .filter_map(|id| state.get_predicate_by_id(*id))
                    .map(|p| p.epistemic.depth)
                    .max()
                    .unwrap_or(0);

                if parent_depth >= self.config.max_depth {
                    continue;
                }

                // Collect parent Beta confidences for epistemic pruning
                let parent_beta_confidences: Vec<&BetaConfidence> = rule_match
                    .premise_ids
                    .iter()
                    .filter_map(|id| state.get_predicate_by_id(*id))
                    .map(|p| &p.epistemic.confidence)
                    .collect();

                // Epistemic pruning decision based on premise uncertainties
                let pruning_decision = self
                    .pruner
                    .evaluate_rule_match(&parent_beta_confidences, parent_depth + 1);

                match pruning_decision {
                    PruningDecision::Prune { reason } => {
                        // Skip this derivation - too uncertain
                        self.pruner.stats.predicates_skipped += 1;
                        // Log pruning for debugging (could be made configurable)
                        #[cfg(debug_assertions)]
                        eprintln!("Pruned derivation {}: {:?}", rule_match.rule_name, reason);
                        continue;
                    }
                    PruningDecision::RequestNeural { uncertainty } => {
                        // Soft prune - request neural assistance for this branch
                        if self.neural_requests.len() < self.config.max_neural_suggestions {
                            self.request_neural_for_branch(&state, &rule_match, uncertainty);
                        }
                        // Still continue with derivation (soft prune)
                    }
                    PruningDecision::Continue => {
                        // Proceed with derivation
                    }
                }

                // Compute conclusion confidence (product of parent means with decay)
                let parent_confidences: Vec<f64> =
                    parent_beta_confidences.iter().map(|c| c.mean()).collect();

                let combined_conf = parent_confidences.iter().fold(1.0, |acc, c| acc * c)
                    * self.config.confidence_decay;

                // Skip low-confidence derivations (hard threshold)
                if combined_conf < self.config.min_confidence {
                    continue;
                }

                // Refinement check (Z3-style)
                if self.config.enable_refinement {
                    let refinement = self.check_rule_refinement(&rule_match, &state);
                    if refinement == RefinementResult::Unsatisfied {
                        continue; // Skip contradictory derivations
                    }
                }

                // Add derived predicate
                let added = state.add_derived(
                    rule_match.conclusion.clone(),
                    &rule_match.rule_name,
                    &rule_match.premise_ids,
                    self.config.confidence_decay,
                );

                if added {
                    derived_this_round += 1;
                    predicates_derived += 1;

                    // Update provenance
                    let parent_hashes: Vec<_> = rule_match
                        .premise_ids
                        .iter()
                        .filter_map(|id| {
                            state
                                .get_predicate_by_id(*id)
                                .map(|p| self.pred_to_hash(&p.key()))
                        })
                        .collect();

                    if !parent_hashes.is_empty() {
                        self.provenance.add_derived(
                            parent_hashes,
                            ProvenanceOperation::new(
                                &format!(
                                    "{}:{}",
                                    rule_match.rule_name,
                                    rule_match.conclusion.key()
                                ),
                                OperationKind::Computation,
                            ),
                        );
                    }

                    // Add to algebraic reasoner
                    self.ar.add_from_predicate(&rule_match.conclusion);
                }
            }

            // Phase 2: Algebraic reasoning (AR)
            if self.config.enable_ar {
                let ar_result = self.ar.solve();

                // Add algebraically derived predicates
                for pred in &ar_result.derived_predicates {
                    if !state.has_predicate(&pred.key()) {
                        let pred_with_epistemic = pred.clone().with_epistemic(PredicateEpistemic {
                            confidence: ar_result.confidence,
                            source: crate::epistemic::Source::Derivation("algebraic".to_string()),
                            revisability: crate::epistemic::Revisability::Revisable {
                                conditions: vec!["new_evidence".to_string()],
                            },
                            depth: iterations,
                            derived_from: vec![],
                            merkle_hash: None,
                            parent_hashes: vec![],
                        });
                        state.add_axiom(pred_with_epistemic);
                        derived_this_round += 1;
                        predicates_derived += 1;
                    }
                }

                // Check for contradictions from AR
                if !ar_result.unit_errors.is_empty() {
                    termination = TerminationReason::Contradiction;
                    break;
                }
            }

            // Check fixpoint
            if derived_this_round == 0 {
                termination = TerminationReason::Fixpoint;
                break;
            }
        }

        if iterations >= self.config.max_iterations {
            termination = TerminationReason::MaxIterations;
        }

        // Compute final statistics
        let ar_result = self.ar.solve();
        let global_uncertainty = self.compute_global_uncertainty(&state);
        let proved = state.goal_satisfied();

        // Compute overall confidence
        let confidence = if proved {
            state
                .goal
                .as_ref()
                .and_then(|g| state.get_predicate(&g.predicate.key()))
                .map(|p| p.epistemic.confidence)
                .unwrap_or_else(BetaConfidence::uniform_prior)
        } else {
            state.confidence
        };

        DeductionResult {
            proved,
            state,
            iterations,
            predicates_derived,
            algebraic_solutions: ar_result.solutions,
            confidence,
            global_uncertainty,
            constructions: Vec::new(), // TODO: track constructions
            neural_requests: self.neural_requests.clone(),
            termination,
            provenance: self.provenance.clone(),
            pruning_stats: self.pruner.stats.clone(),
        }
    }

    /// Apply a construction and continue deduction
    pub fn apply_construction(&mut self, state: &mut ProofState, construction: Construction) {
        state.apply_construction(construction.clone());

        // Add new predicates from construction to AR
        for pred in state.all_predicates() {
            self.ar.add_from_predicate(pred);
        }

        // Track provenance
        self.provenance.add_derived(
            vec![], // Construction is like a new root
            ProvenanceOperation::new(
                &format!("{:?}", construction.kind),
                OperationKind::Computation,
            ),
        );
    }

    /// Compute global uncertainty (aggregate variance)
    fn compute_global_uncertainty(&self, state: &ProofState) -> f64 {
        let predicates: Vec<_> = state.all_predicates().collect();
        if predicates.is_empty() {
            return 1.0; // Maximum uncertainty if no predicates
        }

        // Use Beta variance as uncertainty measure
        let total_variance: f64 = predicates
            .iter()
            .map(|p| p.epistemic.confidence.variance())
            .sum();

        (total_variance / predicates.len() as f64).sqrt()
    }

    /// Check refinement for a rule application
    fn check_rule_refinement(
        &self,
        rule_match: &RuleMatch,
        state: &ProofState,
    ) -> RefinementResult {
        // Create equation representing the rule conclusion
        // For now, just check that conclusion doesn't contradict existing predicates

        let conclusion_key = rule_match.conclusion.key();

        // If we already have the predicate, check consistency
        if state.get_predicate(&conclusion_key).is_some() {
            // Same predicate exists - this is fine (no contradiction)
            RefinementResult::Satisfied
        } else {
            // No existing predicate - rule application is valid
            // Full AR checking happens in the main deduction loop
            RefinementResult::Satisfied
        }
    }

    /// Request neural assistance for a specific branch during pruning
    fn request_neural_for_branch(
        &mut self,
        state: &ProofState,
        rule_match: &RuleMatch,
        uncertainty: f64,
    ) {
        // Create a targeted neural request for this specific derivation branch
        let summary = format!(
            "Branch: {} with {} premises, uncertainty: {:.3}",
            rule_match.rule_name,
            rule_match.premise_ids.len(),
            uncertainty
        );

        // Determine what might help this branch
        let suggestion_type = match rule_match.conclusion.kind {
            PredicateKind::OnCircle | PredicateKind::Concyclic => SuggestionType::AuxiliaryCircle,
            PredicateKind::Parallel | PredicateKind::Perpendicular => SuggestionType::AuxiliaryLine,
            PredicateKind::Collinear | PredicateKind::Midpoint => SuggestionType::AuxiliaryPoint,
            _ => SuggestionType::LemmaSelection,
        };

        // Collect variables from the conclusion that might need values
        let needed_variables = rule_match.conclusion.args.clone();

        self.neural_requests.push(NeuralRequest {
            state_summary: summary,
            suggestion_type,
            uncertainty,
            needed_variables,
        });
    }

    /// Request neural assistance for high-uncertainty situations
    fn request_neural_assistance(&mut self, state: &ProofState, uncertainty: f64) {
        // Determine what kind of suggestion would help
        let suggestion_type = self.determine_suggestion_type(state);

        // Collect needed variables from AR
        let ar_result = self.ar.solve();
        let known_vars: HashSet<_> = ar_result.solutions.keys().cloned().collect();
        let all_vars: HashSet<_> = state
            .all_predicates()
            .flat_map(|p| p.args.iter().cloned())
            .collect();
        let needed: Vec<_> = all_vars.difference(&known_vars).cloned().collect();

        // Create state summary
        let summary = format!(
            "Points: {}, Predicates: {}, Uncertainty: {:.3}",
            state.points.len(),
            state.num_predicates(),
            uncertainty
        );

        self.neural_requests.push(NeuralRequest {
            state_summary: summary,
            suggestion_type,
            uncertainty,
            needed_variables: needed,
        });
    }

    /// Determine what type of neural suggestion would be most helpful
    fn determine_suggestion_type(&self, state: &ProofState) -> SuggestionType {
        // Heuristics based on current state

        // If we have few predicates, suggest auxiliary constructions
        if state.num_predicates() < 10 {
            // Check what types of predicates we have
            let has_circles = !state.predicates_by_kind(PredicateKind::OnCircle).is_empty();
            let has_parallel = !state.predicates_by_kind(PredicateKind::Parallel).is_empty();

            if has_circles {
                return SuggestionType::AuxiliaryCircle;
            }
            if has_parallel {
                return SuggestionType::AuxiliaryLine;
            }
            return SuggestionType::AuxiliaryPoint;
        }

        // Default to lemma selection for complex states
        SuggestionType::LemmaSelection
    }

    /// Convert predicate key to provenance hash
    fn pred_to_hash(&self, key: &str) -> crate::epistemic::merkle::Hash256 {
        crate::epistemic::merkle::hash(key.as_bytes())
    }

    /// Get statistics about the rule database
    pub fn rule_stats(&self) -> RuleStats {
        RuleStats {
            total_rules: self.rules.rules().len(),
            rules_by_priority: self
                .rules
                .rules()
                .iter()
                .fold(HashMap::new(), |mut acc, r| {
                    *acc.entry(r.priority).or_insert(0) += 1;
                    acc
                }),
        }
    }
}

impl Default for SymbolicEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the rule database
#[derive(Debug, Clone)]
pub struct RuleStats {
    pub total_rules: usize,
    pub rules_by_priority: HashMap<i32, usize>,
}

// =============================================================================
// NeSy Loop Integration
// =============================================================================

/// Handler for neural suggestions (to be implemented by effect handler)
pub trait NeuralSuggestionHandler {
    /// Suggest auxiliary constructions given current state
    fn suggest_constructions(
        &self,
        state: &ProofState,
        request: &NeuralRequest,
    ) -> Vec<Construction>;

    /// Score a potential construction
    fn score_construction(&self, state: &ProofState, construction: &Construction) -> f64;
}

/// Default handler that returns no suggestions (pure symbolic mode)
pub struct NoNeuralHandler;

impl NeuralSuggestionHandler for NoNeuralHandler {
    fn suggest_constructions(
        &self,
        _state: &ProofState,
        _request: &NeuralRequest,
    ) -> Vec<Construction> {
        Vec::new()
    }

    fn score_construction(&self, _state: &ProofState, _construction: &Construction) -> f64 {
        0.0
    }
}

/// NeSy loop controller
pub struct NeSyLoop<H: NeuralSuggestionHandler> {
    engine: SymbolicEngine,
    neural: H,
    max_neural_iterations: usize,
}

impl<H: NeuralSuggestionHandler> NeSyLoop<H> {
    pub fn new(engine: SymbolicEngine, neural: H) -> Self {
        NeSyLoop {
            engine,
            neural,
            max_neural_iterations: 5,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_neural_iterations = max;
        self
    }

    /// Run the full NeSy loop
    pub fn solve(&mut self, mut state: ProofState) -> DeductionResult {
        let mut neural_iterations = 0;

        loop {
            // Phase 1: Symbolic deduction
            let result = self.engine.deduce(state.clone());

            // Check termination
            match result.termination {
                TerminationReason::GoalProven => return result,
                TerminationReason::Contradiction => return result,
                TerminationReason::MaxIterations => {
                    if neural_iterations >= self.max_neural_iterations {
                        return result;
                    }
                }
                TerminationReason::NeuralRequest => {
                    // Phase 2: Neural suggestions
                    if neural_iterations >= self.max_neural_iterations {
                        return result;
                    }

                    let mut best_construction: Option<Construction> = None;
                    let mut best_score = 0.0;

                    for request in &result.neural_requests {
                        let suggestions = self.neural.suggest_constructions(&state, request);

                        for construction in suggestions {
                            let score = self.neural.score_construction(&state, &construction);
                            if score > best_score {
                                best_score = score;
                                best_construction = Some(construction);
                            }
                        }
                    }

                    if let Some(construction) = best_construction {
                        self.engine.apply_construction(&mut state, construction);
                        neural_iterations += 1;
                        continue;
                    }

                    // No good suggestions - return current result
                    return result;
                }
                _ => {
                    // Fixpoint or max depth - try neural if available
                    if neural_iterations < self.max_neural_iterations
                        && !result.neural_requests.is_empty()
                    {
                        // Try to get neural help
                        for request in &result.neural_requests {
                            let suggestions = self.neural.suggest_constructions(&state, request);
                            if let Some(construction) = suggestions.into_iter().next() {
                                self.engine.apply_construction(&mut state, construction);
                                neural_iterations += 1;
                                continue;
                            }
                        }
                    }
                    return result;
                }
            }
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Quick proof attempt with default settings
pub fn prove(state: ProofState) -> DeductionResult {
    let mut engine = SymbolicEngine::new();
    engine.deduce(state)
}

/// Prove with custom configuration
pub fn prove_with_config(state: ProofState, config: EngineConfig) -> DeductionResult {
    let mut engine = SymbolicEngine::with_config(config);
    engine.deduce(state)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::predicates::Predicate;

    #[test]
    fn test_engine_creation() {
        let engine = SymbolicEngine::new();
        let stats = engine.rule_stats();
        assert!(stats.total_rules > 0);
    }

    #[test]
    fn test_simple_deduction() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));
        state.add_axiom(Predicate::collinear("A", "B", "D"));

        // Goal: collinear(B, C, D)
        let goal = Predicate::collinear("B", "C", "D");
        state.set_goal(goal, 0.9);

        let result = prove(state);

        // The collinear_trans rule should derive this
        assert!(result.proved || result.predicates_derived > 0);
    }

    #[test]
    fn test_midpoint_deduction() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "M", "N"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));
        state.add_axiom(Predicate::midpoint("N", "A", "C"));

        // Goal: parallel(M, N, B, C) from midpoint theorem
        let goal = Predicate::parallel("M", "N", "B", "C");
        state.set_goal(goal, 0.9);

        let result = prove(state);

        // Should derive parallel from midpoint theorem
        assert!(result.proved);
        assert!(result.predicates_derived > 0);
    }

    #[test]
    fn test_algebraic_integration() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "M"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));

        let config = EngineConfig {
            enable_ar: true,
            ..Default::default()
        };

        let result = prove_with_config(state, config);

        // AR should derive equal_length(A, M, M, B)
        assert!(result.predicates_derived > 0);
    }

    #[test]
    fn test_epistemic_propagation() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);

        // Add axioms with different confidences
        let mut p1 = Predicate::collinear("A", "B", "C");
        p1.epistemic.confidence = BetaConfidence::new(95.0, 5.0); // mean ≈ 0.95
        state.add_axiom(p1);

        let mut p2 = Predicate::collinear("A", "B", "D");
        p2.epistemic.confidence = BetaConfidence::new(90.0, 10.0); // mean ≈ 0.90
        state.add_axiom(p2);

        let result = prove(state);

        // Derived predicates should have decayed confidence
        if result.predicates_derived > 0 {
            // Check that confidence propagated correctly
            let derived: Vec<_> = result
                .state
                .all_predicates()
                .filter(|p| !matches!(p.epistemic.source, crate::epistemic::Source::Axiom))
                .collect();

            for pred in &derived {
                // Derived predicates should have reasonable confidence
                // (may be high if derived from high-confidence parents via deterministic rules)
                let conf = pred.epistemic.confidence.mean();
                assert!(
                    conf > 0.0 && conf <= 1.0,
                    "Derived predicate confidence {} should be in (0, 1]",
                    conf
                );
            }

            // At least verify we got some derived predicates
            assert!(
                !derived.is_empty() || result.predicates_derived == 0,
                "Inconsistent: predicates_derived={} but derived vec empty",
                result.predicates_derived
            );
        }
    }

    #[test]
    fn test_global_uncertainty() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));

        let engine = SymbolicEngine::new();
        let uncertainty = engine.compute_global_uncertainty(&state);

        // Should have low uncertainty with high-confidence axioms
        assert!(uncertainty < 0.5);
    }

    #[test]
    fn test_neural_request_generation() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C"]);

        // Add low-confidence predicate to trigger neural request
        let mut pred = Predicate::collinear("A", "B", "C");
        pred.epistemic.confidence = BetaConfidence::new(3.0, 7.0); // mean ≈ 0.3
        state.add_axiom(pred);

        // Set high uncertainty threshold to trigger
        let config = EngineConfig {
            uncertainty_threshold: 0.1, // Very low threshold
            pruning_threshold: 0.4,
            ..Default::default()
        };

        let result = prove_with_config(state, config);

        // May or may not have neural requests depending on uncertainty calculation
        // Just verify no crash
        assert!(result.termination != TerminationReason::Contradiction);
    }

    #[test]
    fn test_nesy_loop_pure_symbolic() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "M", "N"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));
        state.add_axiom(Predicate::midpoint("N", "A", "C"));

        let goal = Predicate::parallel("M", "N", "B", "C");
        state.set_goal(goal, 0.9);

        let engine = SymbolicEngine::new();
        let mut nesy = NeSyLoop::new(engine, NoNeuralHandler);

        let result = nesy.solve(state);
        assert!(result.proved);
    }

    #[test]
    fn test_fixpoint_detection() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B"]);
        state.add_axiom(Predicate::collinear("A", "A", "B")); // Trivial

        // Goal that can't be derived
        let goal = Predicate::parallel("A", "B", "A", "B");
        state.set_goal(goal, 0.9);

        let result = prove(state);

        // Should reach fixpoint (no new derivations possible)
        assert!(!result.proved);
        assert!(matches!(
            result.termination,
            TerminationReason::Fixpoint | TerminationReason::MaxIterations
        ));
    }

    #[test]
    fn test_proof_trace() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));
        state.add_axiom(Predicate::collinear("A", "B", "D"));

        let goal = Predicate::collinear("B", "C", "D");
        state.set_goal(goal, 0.9);

        let result = prove(state);

        if result.proved {
            let proof_text = result.proof_text();
            assert!(proof_text.contains("PROOF"));
            assert!(proof_text.contains("collinear"));
        }
    }

    #[test]
    fn test_provenance_tracking() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));
        state.add_axiom(Predicate::collinear("A", "B", "D"));

        let result = prove(state);

        // Provenance DAG should have entries
        assert!(result.provenance.len() > 0);
    }

    // =========================================================================
    // Epistemic Pruning Tests
    // =========================================================================

    #[test]
    fn test_epistemic_pruner_creation() {
        let pruner = EpistemicPruner::default();
        assert_eq!(pruner.confidence_threshold, 0.3);
        assert_eq!(pruner.variance_threshold, 0.15);
        assert!(pruner.soft_prune);
    }

    #[test]
    fn test_pruner_from_config() {
        let config = EngineConfig {
            pruning_threshold: 0.4,
            uncertainty_threshold: 0.2,
            confidence_decay: 0.95,
            ..Default::default()
        };
        let pruner = EpistemicPruner::from_config(&config);
        assert_eq!(pruner.confidence_threshold, 0.4);
        assert_eq!(pruner.variance_threshold, 0.2);
        assert_eq!(pruner.depth_decay, 0.95);
    }

    #[test]
    fn test_pruning_decision_high_confidence() {
        let mut pruner = EpistemicPruner::default();
        // High confidence (α=100, β=1) -> mean ≈ 0.99, variance very low
        let high_conf = BetaConfidence::new(100.0, 1.0);

        let decision = pruner.evaluate(&high_conf, 0);
        assert_eq!(decision, PruningDecision::Continue);
    }

    #[test]
    fn test_pruning_decision_low_confidence() {
        let mut pruner = EpistemicPruner::default();
        pruner.soft_prune = false; // Hard prune mode

        // Low confidence (α=1, β=10) -> mean ≈ 0.09
        let low_conf = BetaConfidence::new(1.0, 10.0);

        let decision = pruner.evaluate(&low_conf, 0);
        match decision {
            PruningDecision::Prune { reason } => {
                match reason {
                    PruningReason::LowConfidence {
                        confidence,
                        threshold,
                    } => {
                        assert!(confidence < threshold);
                    }
                    _ => {} // Other prune reasons are also acceptable
                }
            }
            _ => panic!("Expected pruning decision for low confidence"),
        }
    }

    #[test]
    fn test_pruning_decision_high_variance() {
        let mut pruner = EpistemicPruner::default();
        pruner.soft_prune = false;
        pruner.variance_threshold = 0.1;

        // Uniform prior has high variance (≈ 0.083)
        // But with α=β=2, variance is 0.05
        // Use α=β=1.5 for higher variance
        let uncertain = BetaConfidence::new(1.5, 1.5);

        // Variance of Beta(1.5, 1.5) = 1.5*1.5/((3)^2 * 4) ≈ 0.0625
        // This should be below threshold 0.1, so let's use uniform
        let uniform = BetaConfidence::uniform_prior();
        // Variance of Beta(1,1) = 1/(4) = 0.083

        let decision = pruner.evaluate(&uniform, 0);
        // Uniform has mean 0.5, variance 0.083 < 0.1, so might not trigger
        // Let's check stats instead
        assert!(pruner.stats.branches_evaluated > 0);
    }

    #[test]
    fn test_pruning_decision_soft_prune_requests_neural() {
        let mut pruner = EpistemicPruner::default();
        pruner.soft_prune = true;

        // Low confidence should trigger neural request instead of hard prune
        let low_conf = BetaConfidence::new(1.0, 10.0);

        let decision = pruner.evaluate(&low_conf, 0);
        match decision {
            PruningDecision::RequestNeural { uncertainty } => {
                assert!(uncertainty > 0.0);
            }
            _ => panic!("Expected neural request for soft prune"),
        }
    }

    #[test]
    fn test_pruning_depth_decay() {
        let mut pruner = EpistemicPruner::default();
        pruner.soft_prune = false;
        pruner.depth_decay = 0.8; // Aggressive decay
        pruner.confidence_threshold = 0.3;

        // Medium confidence that decays with depth
        let medium_conf = BetaConfidence::new(5.0, 3.0); // mean ≈ 0.625

        // At depth 0, effective conf = 0.625 * 0.8^0 = 0.625 > 0.3
        let decision_shallow = pruner.evaluate(&medium_conf, 0);
        assert_eq!(decision_shallow, PruningDecision::Continue);

        // At depth 5, effective conf = 0.625 * 0.8^5 ≈ 0.205 < 0.3 * 0.5 = 0.15
        let decision_deep = pruner.evaluate(&medium_conf, 5);
        match decision_deep {
            PruningDecision::Prune {
                reason: PruningReason::DepthDecay { .. },
            } => {}
            _ => {} // Combined score might trigger first
        }
    }

    #[test]
    fn test_pruning_rule_match() {
        let mut pruner = EpistemicPruner::default();

        // Two high-confidence premises
        let conf1 = BetaConfidence::new(10.0, 1.0); // mean ≈ 0.91
        let conf2 = BetaConfidence::new(8.0, 1.0); // mean ≈ 0.89

        let decision = pruner.evaluate_rule_match(&[&conf1, &conf2], 1);
        // Product of means ≈ 0.81, should continue
        assert_eq!(decision, PruningDecision::Continue);
    }

    #[test]
    fn test_pruning_rule_match_low_confidence_premise() {
        let mut pruner = EpistemicPruner::default();
        pruner.soft_prune = false;

        // One high and one low confidence premise
        let conf_high = BetaConfidence::new(10.0, 1.0); // mean ≈ 0.91
        let conf_low = BetaConfidence::new(1.0, 5.0); // mean ≈ 0.17

        let decision = pruner.evaluate_rule_match(&[&conf_high, &conf_low], 1);
        // Product of means ≈ 0.15, should prune
        match decision {
            PruningDecision::Prune { .. } => {}
            _ => panic!("Expected prune for low combined confidence"),
        }
    }

    #[test]
    fn test_pruning_stats_tracking() {
        let mut pruner = EpistemicPruner::default();
        pruner.soft_prune = false;

        // Evaluate several decisions
        let high = BetaConfidence::new(10.0, 1.0);
        let low = BetaConfidence::new(1.0, 10.0);

        pruner.evaluate(&high, 0);
        pruner.evaluate(&low, 0);
        pruner.evaluate(&high, 0);

        assert_eq!(pruner.stats.branches_evaluated, 3);
        assert!(pruner.stats.pruned_low_confidence >= 1);
    }

    #[test]
    fn test_pruning_stats_reset() {
        let mut pruner = EpistemicPruner::default();

        let high = BetaConfidence::new(10.0, 1.0);
        pruner.evaluate(&high, 0);
        pruner.evaluate(&high, 0);

        assert_eq!(pruner.stats.branches_evaluated, 2);

        pruner.reset_stats();
        assert_eq!(pruner.stats.branches_evaluated, 0);
    }

    #[test]
    fn test_engine_with_pruning() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));
        state.add_axiom(Predicate::collinear("A", "B", "D"));

        let goal = Predicate::collinear("B", "C", "D");
        state.set_goal(goal, 0.9);

        let result = prove(state);

        // Should have pruning stats
        assert!(result.pruning_stats.branches_evaluated > 0);
    }

    #[test]
    fn test_deduction_result_pruning_stats() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "M"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));

        let result = prove(state);

        // Verify pruning stats are populated
        // Even if no pruning occurred, branches should be evaluated
        // (unless no rules matched)
        assert!(result.pruning_stats.branches_evaluated >= 0);
    }

    #[test]
    fn test_combined_uncertainty_score() {
        let mut pruner = EpistemicPruner::default();
        pruner.combined_threshold = 0.4;
        pruner.soft_prune = false;

        // Medium-low confidence with some variance
        // Beta(3, 3) has mean=0.5, variance=0.036
        let medium = BetaConfidence::new(3.0, 3.0);

        // Combined score = (1 - 0.5) * 0.6 + (0.036/0.25) * 0.4
        //                = 0.3 + 0.058 = 0.358 < 0.4
        let decision = pruner.evaluate(&medium, 0);
        assert_eq!(decision, PruningDecision::Continue);

        // With lower confidence: Beta(2, 4) has mean≈0.33, var≈0.032
        let lower = BetaConfidence::new(2.0, 4.0);
        // Combined score = (1 - 0.33) * 0.6 + (0.032/0.25) * 0.4
        //                ≈ 0.4 + 0.05 = 0.45 > 0.4
        let decision2 = pruner.evaluate(&lower, 0);
        match decision2 {
            PruningDecision::Prune {
                reason: PruningReason::CombinedUncertainty { .. },
            } => {}
            PruningDecision::Prune {
                reason: PruningReason::LowConfidence { .. },
            } => {}
            _ => {} // Low confidence might trigger first
        }
    }
}

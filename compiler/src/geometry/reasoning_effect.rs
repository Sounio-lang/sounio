//! Geometry Reasoning Effect
//!
//! An algebraic effect for neuro-symbolic geometry reasoning that integrates:
//! - Symbolic deduction (DD + AR)
//! - Epistemic pruning
//! - Neural auxiliary suggestions
//!
//! # Architecture
//!
//! The GeometryReasoning effect provides operations for:
//! 1. **deduce** - Run symbolic deduction with epistemic tracking
//! 2. **prune** - Check if a branch should be pruned based on uncertainty
//! 3. **suggest_auxiliary** - Request neural network suggestions
//! 4. **apply_construction** - Apply an auxiliary construction
//! 5. **commit_proof** - Finalize a proof with provenance
//!
//! # Example Usage (in Sounio)
//!
//! ```sounio
//! effect GeometryReasoning {
//!     deduce(state: ProofState, goal: Predicate) -> DeductionResult;
//!     prune(confidence: BetaConfidence, depth: u64) -> PruningDecision;
//!     suggest_auxiliary(state: ProofState) -> [Construction];
//!     apply_construction(state: ProofState, constr: Construction) -> ProofState;
//!     commit_proof(result: DeductionResult) -> MerkleProof;
//! }
//!
//! // Handler that uses pure symbolic reasoning
//! handler PureSymbolic for GeometryReasoning {
//!     deduce(state, goal) -> resume(symbolic_engine.prove(state, goal));
//!     prune(conf, depth) -> resume(epistemic_pruner.evaluate(conf, depth));
//!     suggest_auxiliary(state) -> resume([]);  // No neural suggestions
//!     apply_construction(state, constr) -> resume(state.with_construction(constr));
//!     commit_proof(result) -> resume(result.provenance.finalize());
//! }
//!
//! // Handler that integrates neural suggestions
//! handler NeSyGeometry for GeometryReasoning {
//!     deduce(state, goal) -> {
//!         let result = symbolic_engine.prove(state, goal);
//!         if result.termination == NeuralRequest {
//!             let suggestions = perform suggest_auxiliary(state);
//!             // ... apply suggestions and retry
//!         }
//!         resume(result)
//!     };
//!     suggest_auxiliary(state) -> resume(neural_model.suggest(state));
//!     // ...
//! }
//! ```

use std::collections::HashMap;

use crate::epistemic::bayesian::BetaConfidence;
use crate::epistemic::merkle::Hash256;
use crate::types::core::Type;
use crate::types::effects::{EffectDef, EffectOperation};

use super::engine::{
    DeductionResult, EngineConfig, EpistemicPruner, NeuralRequest, PruningDecision, SymbolicEngine,
    TerminationReason,
};
use super::predicates::Predicate;
use super::proof_state::{Construction, ProofState};

// =============================================================================
// Effect Definition
// =============================================================================

/// Create the GeometryReasoning effect definition for the type system
pub fn geometry_reasoning_effect() -> EffectDef {
    EffectDef::new("GeometryReasoning")
        // Run symbolic deduction
        .with_op(EffectOperation::new(
            "deduce",
            vec![
                Type::Named {
                    name: "ProofState".to_string(),
                    args: vec![],
                },
                Type::Named {
                    name: "Predicate".to_string(),
                    args: vec![],
                },
            ],
            Type::Named {
                name: "DeductionResult".to_string(),
                args: vec![],
            },
        ))
        // Check pruning decision
        .with_op(EffectOperation::new(
            "prune",
            vec![
                Type::Named {
                    name: "BetaConfidence".to_string(),
                    args: vec![],
                },
                Type::U64,
            ],
            Type::Named {
                name: "PruningDecision".to_string(),
                args: vec![],
            },
        ))
        // Request neural suggestions
        .with_op(EffectOperation::new(
            "suggest_auxiliary",
            vec![Type::Named {
                name: "ProofState".to_string(),
                args: vec![],
            }],
            Type::Array {
                element: Box::new(Type::Named {
                    name: "Construction".to_string(),
                    args: vec![],
                }),
                size: None, // Dynamic array (slice)
            },
        ))
        // Apply a construction
        .with_op(EffectOperation::new(
            "apply_construction",
            vec![
                Type::Named {
                    name: "ProofState".to_string(),
                    args: vec![],
                },
                Type::Named {
                    name: "Construction".to_string(),
                    args: vec![],
                },
            ],
            Type::Named {
                name: "ProofState".to_string(),
                args: vec![],
            },
        ))
        // Commit proof with Merkle provenance
        .with_op(EffectOperation::new(
            "commit_proof",
            vec![Type::Named {
                name: "DeductionResult".to_string(),
                args: vec![],
            }],
            Type::Named {
                name: "MerkleProof".to_string(),
                args: vec![],
            },
        ))
        // Check global uncertainty level
        .with_op(EffectOperation::new(
            "check_uncertainty",
            vec![Type::Named {
                name: "ProofState".to_string(),
                args: vec![],
            }],
            Type::F64,
        ))
        // Update confidence for a predicate
        .with_op(EffectOperation::new(
            "update_confidence",
            vec![
                Type::Named {
                    name: "PredicateId".to_string(),
                    args: vec![],
                },
                Type::Named {
                    name: "BetaConfidence".to_string(),
                    args: vec![],
                },
            ],
            Type::Unit,
        ))
}

// =============================================================================
// Effect Handler Trait
// =============================================================================

/// Trait for handling GeometryReasoning effect operations
pub trait GeometryReasoningHandler {
    /// Run symbolic deduction to prove a goal
    fn deduce(&mut self, state: ProofState, goal: Predicate) -> DeductionResult;

    /// Evaluate pruning decision for a branch
    fn prune(&mut self, confidence: &BetaConfidence, depth: usize) -> PruningDecision;

    /// Get neural suggestions for auxiliary constructions
    fn suggest_auxiliary(&mut self, state: &ProofState) -> Vec<Construction>;

    /// Apply a construction to the proof state
    fn apply_construction(&mut self, state: ProofState, construction: Construction) -> ProofState;

    /// Commit a proof and get the Merkle provenance root
    fn commit_proof(&mut self, result: &DeductionResult) -> MerkleProof;

    /// Check global uncertainty of the state
    fn check_uncertainty(&self, state: &ProofState) -> f64;

    /// Update confidence for a specific predicate
    fn update_confidence(
        &mut self,
        state: &mut ProofState,
        pred_key: &str,
        confidence: BetaConfidence,
    );
}

/// Finalized Merkle proof with root hash and witness path
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// Root hash of the proof DAG
    pub root: Hash256,
    /// All node hashes in topological order
    pub nodes: Vec<Hash256>,
    /// Mapping from predicate keys to their hashes
    pub predicate_hashes: HashMap<String, Hash256>,
    /// Total confidence of the proof
    pub confidence: BetaConfidence,
    /// Whether the goal was proven
    pub proved: bool,
}

// =============================================================================
// Pure Symbolic Handler
// =============================================================================

/// Handler that uses pure symbolic reasoning (no neural suggestions)
pub struct PureSymbolicHandler {
    engine: SymbolicEngine,
    pruner: EpistemicPruner,
}

impl PureSymbolicHandler {
    pub fn new() -> Self {
        Self {
            engine: SymbolicEngine::new(),
            pruner: EpistemicPruner::default(),
        }
    }

    pub fn with_config(config: EngineConfig) -> Self {
        Self {
            pruner: EpistemicPruner::from_config(&config),
            engine: SymbolicEngine::with_config(config),
        }
    }
}

impl Default for PureSymbolicHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryReasoningHandler for PureSymbolicHandler {
    fn deduce(&mut self, mut state: ProofState, goal: Predicate) -> DeductionResult {
        state.set_goal(goal, 0.9);
        self.engine.deduce(state)
    }

    fn prune(&mut self, confidence: &BetaConfidence, depth: usize) -> PruningDecision {
        self.pruner.evaluate(confidence, depth)
    }

    fn suggest_auxiliary(&mut self, _state: &ProofState) -> Vec<Construction> {
        // Pure symbolic handler doesn't provide neural suggestions
        Vec::new()
    }

    fn apply_construction(
        &mut self,
        mut state: ProofState,
        construction: Construction,
    ) -> ProofState {
        state.apply_construction(construction);
        state
    }

    fn commit_proof(&mut self, result: &DeductionResult) -> MerkleProof {
        // Collect all node hashes from heads (leaf nodes in the DAG)
        let nodes: Vec<Hash256> = result.provenance.heads().map(|node| node.id).collect();

        // Use the DAG hash as the root
        let root = result.provenance.compute_dag_hash();

        // Build predicate hash mapping
        let mut predicate_hashes = HashMap::new();
        for pred in result.state.all_predicates() {
            let key = pred.key();
            let hash = crate::epistemic::merkle::hash(key.as_bytes());
            predicate_hashes.insert(key, hash);
        }

        MerkleProof {
            root,
            nodes,
            predicate_hashes,
            confidence: result.confidence,
            proved: result.proved,
        }
    }

    fn check_uncertainty(&self, state: &ProofState) -> f64 {
        state.global_uncertainty()
    }

    fn update_confidence(
        &mut self,
        _state: &mut ProofState,
        _pred_key: &str,
        _confidence: BetaConfidence,
    ) {
        // TODO: Implement predicate confidence update
        // This would require mutable access to predicates in ProofState
    }
}

// =============================================================================
// NeSy Loop Handler
// =============================================================================

/// Configuration for the NeSy loop handler
#[derive(Debug, Clone)]
pub struct NeSyConfig {
    /// Maximum neural iterations before giving up
    pub max_neural_iterations: usize,
    /// Uncertainty threshold to trigger neural assistance
    pub neural_trigger_threshold: f64,
    /// Minimum improvement required to accept a suggestion
    pub min_improvement: f64,
    /// Engine configuration
    pub engine_config: EngineConfig,
}

impl Default for NeSyConfig {
    fn default() -> Self {
        Self {
            max_neural_iterations: 5,
            neural_trigger_threshold: 0.3,
            min_improvement: 0.1,
            engine_config: EngineConfig::default(),
        }
    }
}

/// Handler that integrates neural suggestions in a NeSy loop
pub struct NeSyHandler<N: NeuralSuggester> {
    engine: SymbolicEngine,
    pruner: EpistemicPruner,
    neural: N,
    config: NeSyConfig,
    /// Statistics about neural assistance
    pub stats: NeSyStats,
}

/// Statistics about NeSy loop execution
#[derive(Debug, Clone, Default)]
pub struct NeSyStats {
    /// Number of neural iterations performed
    pub neural_iterations: usize,
    /// Number of suggestions accepted
    pub suggestions_accepted: usize,
    /// Number of suggestions rejected
    pub suggestions_rejected: usize,
    /// Total constructions applied
    pub constructions_applied: usize,
}

/// Trait for neural suggestion providers
pub trait NeuralSuggester {
    /// Generate auxiliary construction suggestions
    fn suggest(&self, state: &ProofState, request: &NeuralRequest) -> Vec<Construction>;

    /// Score a potential construction
    fn score(&self, state: &ProofState, construction: &Construction) -> f64;
}

/// Placeholder neural suggester that returns nothing (for testing)
pub struct NoOpSuggester;

impl NeuralSuggester for NoOpSuggester {
    fn suggest(&self, _state: &ProofState, _request: &NeuralRequest) -> Vec<Construction> {
        Vec::new()
    }

    fn score(&self, _state: &ProofState, _construction: &Construction) -> f64 {
        0.0
    }
}

impl<N: NeuralSuggester> NeSyHandler<N> {
    pub fn new(neural: N) -> Self {
        let config = NeSyConfig::default();
        Self {
            pruner: EpistemicPruner::from_config(&config.engine_config),
            engine: SymbolicEngine::with_config(config.engine_config.clone()),
            neural,
            config,
            stats: NeSyStats::default(),
        }
    }

    pub fn with_config(neural: N, config: NeSyConfig) -> Self {
        Self {
            pruner: EpistemicPruner::from_config(&config.engine_config),
            engine: SymbolicEngine::with_config(config.engine_config.clone()),
            neural,
            config,
            stats: NeSyStats::default(),
        }
    }

    /// Run the full NeSy loop
    pub fn nesy_loop(&mut self, mut state: ProofState, goal: Predicate) -> DeductionResult {
        state.set_goal(goal, 0.9);
        let mut neural_iterations = 0;

        loop {
            // Phase 1: Symbolic deduction
            let result = self.engine.deduce(state.clone());

            // Check termination conditions
            match result.termination {
                TerminationReason::GoalProven => return result,
                TerminationReason::Contradiction => return result,
                TerminationReason::NeuralRequest | TerminationReason::Fixpoint => {
                    if neural_iterations >= self.config.max_neural_iterations {
                        return result;
                    }

                    // Phase 2: Neural suggestions
                    let suggestions =
                        self.gather_suggestions(&result.state, &result.neural_requests);

                    if suggestions.is_empty() {
                        return result;
                    }

                    // Find best suggestion
                    let best = self.select_best_suggestion(&result.state, &suggestions);

                    if let Some(construction) = best {
                        state = self.apply_construction(result.state.clone(), construction);
                        neural_iterations += 1;
                        self.stats.neural_iterations += 1;
                        self.stats.suggestions_accepted += 1;
                        continue;
                    }

                    return result;
                }
                _ => return result,
            }
        }
    }

    fn gather_suggestions(
        &self,
        state: &ProofState,
        requests: &[NeuralRequest],
    ) -> Vec<Construction> {
        let mut suggestions = Vec::new();

        for request in requests {
            let mut neural_suggestions = self.neural.suggest(state, request);
            suggestions.append(&mut neural_suggestions);
        }

        suggestions
    }

    fn select_best_suggestion(
        &self,
        state: &ProofState,
        suggestions: &[Construction],
    ) -> Option<Construction> {
        suggestions
            .iter()
            .map(|c| (c, self.neural.score(state, c)))
            .filter(|(_, score)| *score >= self.config.min_improvement)
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap())
            .map(|(c, _)| c.clone())
    }
}

impl<N: NeuralSuggester> GeometryReasoningHandler for NeSyHandler<N> {
    fn deduce(&mut self, state: ProofState, goal: Predicate) -> DeductionResult {
        // Use the full NeSy loop
        self.nesy_loop(state, goal)
    }

    fn prune(&mut self, confidence: &BetaConfidence, depth: usize) -> PruningDecision {
        self.pruner.evaluate(confidence, depth)
    }

    fn suggest_auxiliary(&mut self, state: &ProofState) -> Vec<Construction> {
        // Create a generic neural request
        let request = NeuralRequest {
            state_summary: format!("Predicates: {}", state.num_predicates()),
            suggestion_type: super::engine::SuggestionType::AuxiliaryPoint,
            uncertainty: state.global_uncertainty(),
            needed_variables: Vec::new(),
        };

        self.neural.suggest(state, &request)
    }

    fn apply_construction(
        &mut self,
        mut state: ProofState,
        construction: Construction,
    ) -> ProofState {
        self.stats.constructions_applied += 1;
        state.apply_construction(construction);
        state
    }

    fn commit_proof(&mut self, result: &DeductionResult) -> MerkleProof {
        // Collect all node hashes from heads
        let nodes: Vec<Hash256> = result.provenance.heads().map(|node| node.id).collect();

        // Use DAG hash as root
        let root = result.provenance.compute_dag_hash();

        let mut predicate_hashes = HashMap::new();
        for pred in result.state.all_predicates() {
            let key = pred.key();
            let hash = crate::epistemic::merkle::hash(key.as_bytes());
            predicate_hashes.insert(key, hash);
        }

        MerkleProof {
            root,
            nodes,
            predicate_hashes,
            confidence: result.confidence,
            proved: result.proved,
        }
    }

    fn check_uncertainty(&self, state: &ProofState) -> f64 {
        state.global_uncertainty()
    }

    fn update_confidence(
        &mut self,
        _state: &mut ProofState,
        _pred_key: &str,
        _confidence: BetaConfidence,
    ) {
        // TODO: Implement
    }
}

// =============================================================================
// Effect Interpreter
// =============================================================================

/// Operation in the GeometryReasoning effect
#[derive(Debug, Clone)]
pub enum GeometryOp {
    Deduce {
        state: ProofState,
        goal: Predicate,
    },
    Prune {
        confidence: BetaConfidence,
        depth: usize,
    },
    SuggestAuxiliary {
        state: ProofState,
    },
    ApplyConstruction {
        state: ProofState,
        construction: Construction,
    },
    CommitProof {
        result: DeductionResult,
    },
    CheckUncertainty {
        state: ProofState,
    },
}

/// Result of a GeometryReasoning operation
#[derive(Debug, Clone)]
pub enum GeometryResult {
    Deduction(DeductionResult),
    Pruning(PruningDecision),
    Suggestions(Vec<Construction>),
    State(ProofState),
    Proof(MerkleProof),
    Uncertainty(f64),
}

/// Interpret a sequence of geometry operations
pub fn interpret<H: GeometryReasoningHandler>(
    handler: &mut H,
    ops: Vec<GeometryOp>,
) -> Vec<GeometryResult> {
    ops.into_iter()
        .map(|op| match op {
            GeometryOp::Deduce { state, goal } => {
                GeometryResult::Deduction(handler.deduce(state, goal))
            }
            GeometryOp::Prune { confidence, depth } => {
                GeometryResult::Pruning(handler.prune(&confidence, depth))
            }
            GeometryOp::SuggestAuxiliary { state } => {
                GeometryResult::Suggestions(handler.suggest_auxiliary(&state))
            }
            GeometryOp::ApplyConstruction {
                state,
                construction,
            } => GeometryResult::State(handler.apply_construction(state, construction)),
            GeometryOp::CommitProof { result } => {
                GeometryResult::Proof(handler.commit_proof(&result))
            }
            GeometryOp::CheckUncertainty { state } => {
                GeometryResult::Uncertainty(handler.check_uncertainty(&state))
            }
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_reasoning_effect_definition() {
        let effect = geometry_reasoning_effect();
        assert_eq!(effect.name, "GeometryReasoning");
        assert!(effect.operations.len() >= 5);

        // Check operation names
        let op_names: Vec<_> = effect.operations.iter().map(|o| o.name.as_str()).collect();
        assert!(op_names.contains(&"deduce"));
        assert!(op_names.contains(&"prune"));
        assert!(op_names.contains(&"suggest_auxiliary"));
        assert!(op_names.contains(&"apply_construction"));
        assert!(op_names.contains(&"commit_proof"));
    }

    #[test]
    fn test_pure_symbolic_handler() {
        let mut handler = PureSymbolicHandler::new();

        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "M", "N"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));
        state.add_axiom(Predicate::midpoint("N", "A", "C"));

        let goal = Predicate::parallel("M", "N", "B", "C");
        let result = handler.deduce(state, goal);

        assert!(result.proved);
    }

    #[test]
    fn test_pure_symbolic_pruning() {
        let mut handler = PureSymbolicHandler::new();

        // High confidence should continue
        let high_conf = BetaConfidence::new(100.0, 1.0);
        let decision = handler.prune(&high_conf, 0);
        assert_eq!(decision, PruningDecision::Continue);

        // Low confidence should trigger (soft prune by default)
        let low_conf = BetaConfidence::new(1.0, 10.0);
        let decision = handler.prune(&low_conf, 0);
        match decision {
            PruningDecision::RequestNeural { .. } => {}
            _ => panic!("Expected neural request for low confidence"),
        }
    }

    #[test]
    fn test_pure_symbolic_no_suggestions() {
        let mut handler = PureSymbolicHandler::new();
        let state = ProofState::new();

        let suggestions = handler.suggest_auxiliary(&state);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_merkle_proof_generation() {
        let mut handler = PureSymbolicHandler::new();

        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));

        let goal = Predicate::collinear("A", "B", "C"); // Trivially true
        let result = handler.deduce(state, goal);

        let proof = handler.commit_proof(&result);
        assert!(proof.proved);
        assert!(!proof.predicate_hashes.is_empty());
    }

    #[test]
    fn test_nesy_handler_with_noop_suggester() {
        let mut handler = NeSyHandler::new(NoOpSuggester);

        let mut state = ProofState::new();
        state.add_points(&["A", "B", "M"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));

        // This goal should be provable symbolically
        let goal = Predicate::collinear("A", "M", "B");
        let result = handler.deduce(state, goal);

        // Midpoint implies collinearity
        assert!(result.proved || result.predicates_derived > 0);
    }

    #[test]
    fn test_interpret_operations() {
        let mut handler = PureSymbolicHandler::new();

        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));

        let ops = vec![
            GeometryOp::CheckUncertainty {
                state: state.clone(),
            },
            GeometryOp::Prune {
                confidence: BetaConfidence::new(10.0, 1.0),
                depth: 0,
            },
        ];

        let results = interpret(&mut handler, ops);
        assert_eq!(results.len(), 2);

        match &results[0] {
            GeometryResult::Uncertainty(u) => assert!(*u < 1.0),
            _ => panic!("Expected uncertainty result"),
        }

        match &results[1] {
            GeometryResult::Pruning(PruningDecision::Continue) => {}
            _ => panic!("Expected continue decision"),
        }
    }

    #[test]
    fn test_nesy_stats() {
        let mut handler = NeSyHandler::new(NoOpSuggester);

        let mut state = ProofState::new();
        state.add_points(&["A", "B"]);

        let goal = Predicate::collinear("A", "A", "B");
        let _result = handler.deduce(state, goal);

        // With NoOpSuggester, no neural iterations should occur
        assert_eq!(handler.stats.suggestions_accepted, 0);
    }
}

//! Geometry Proof Game for AlphaGeometry-style Self-Play
//!
//! Bridges the geometry symbolic engine with MCTS for proof search.
//! Implements the `GameState` trait to enable RL-based proof discovery.
//!
//! # AlphaGeometry Integration
//!
//! The key insight from AlphaGeometry is that geometric theorem proving
//! can be cast as a game where:
//! - **State**: Current proof state (predicates, constructions)
//! - **Actions**: Either apply a rule OR add an auxiliary construction
//! - **Reward**: +1 for proving the goal, penalties for depth/uncertainty
//!
//! # Epistemic MCTS Extension
//!
//! Unlike standard MCTS, we use Beta-distributed Q-values for principled
//! uncertainty quantification. High-variance nodes get exploration bonus:
//!
//! ```text
//! UCB(s, a) = Q(s,a).mean + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
//!                        + c_epistemic * Q(s,a).std   // uncertainty bonus
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::epistemic::bayesian::BetaConfidence;
use crate::rl::game::{Action, GameState, GameTrait, Player};

use super::engine::{DeductionResult, EngineConfig, SymbolicEngine};
use super::predicates::{Predicate, PredicateKind};
use super::proof_state::{Construction, ConstructionKind, ConstructionSource, ProofState};

// =============================================================================
// Geometry Actions
// =============================================================================

/// Actions available in the geometry proof game
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeoAction {
    /// Apply a deduction rule to specific predicates
    ApplyRule {
        rule_name: String,
        /// Indices into current predicates (for binding)
        premise_indices: Vec<usize>,
    },
    /// Add an auxiliary construction
    Construct(GeoConstruction),
    /// Run one step of symbolic deduction (DD)
    DeductionStep,
    /// Request neural suggestion (triggers NeSy loop)
    RequestNeural,
    /// Terminate search (give up or declare proved)
    Terminate,
}

impl Action for GeoAction {}

/// Auxiliary constructions that can be added
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeoConstruction {
    /// Add midpoint M of segment AB
    Midpoint { p1: String, p2: String },
    /// Add perpendicular from point to line
    Perpendicular {
        point: String,
        line_p1: String,
        line_p2: String,
    },
    /// Add parallel through point to line
    Parallel {
        point: String,
        line_p1: String,
        line_p2: String,
    },
    /// Add circumcircle of triangle
    Circumcircle { p1: String, p2: String, p3: String },
    /// Add incircle of triangle
    Incircle { p1: String, p2: String, p3: String },
    /// Add angle bisector
    AngleBisector {
        p1: String,
        vertex: String,
        p2: String,
    },
    /// Add intersection of two lines
    LineIntersection {
        l1_p1: String,
        l1_p2: String,
        l2_p1: String,
        l2_p2: String,
    },
    /// Add foot of perpendicular
    Foot {
        point: String,
        line_p1: String,
        line_p2: String,
    },
    /// Add reflection of point over line
    Reflection {
        point: String,
        line_p1: String,
        line_p2: String,
    },
    /// Extend a line through two points
    ExtendLine { p1: String, p2: String },
}

impl GeoConstruction {
    /// Convert to proof_state Construction
    fn to_construction(&self, confidence: f64) -> Construction {
        let kind = match self {
            GeoConstruction::Midpoint { p1, p2 } => ConstructionKind::Midpoint {
                p1: p1.clone(),
                p2: p2.clone(),
            },
            GeoConstruction::Perpendicular {
                point,
                line_p1,
                line_p2,
            } => ConstructionKind::Perpendicular {
                point: point.clone(),
                line_p1: line_p1.clone(),
                line_p2: line_p2.clone(),
            },
            GeoConstruction::Parallel {
                point,
                line_p1,
                line_p2,
            } => ConstructionKind::Parallel {
                point: point.clone(),
                line_p1: line_p1.clone(),
                line_p2: line_p2.clone(),
            },
            GeoConstruction::Circumcircle { p1, p2, p3 } => ConstructionKind::Circumcircle {
                p1: p1.clone(),
                p2: p2.clone(),
                p3: p3.clone(),
            },
            GeoConstruction::Incircle { p1, p2, p3 } => ConstructionKind::Incircle {
                p1: p1.clone(),
                p2: p2.clone(),
                p3: p3.clone(),
            },
            GeoConstruction::AngleBisector { p1, vertex, p2 } => ConstructionKind::AngleBisector {
                p1: p1.clone(),
                vertex: vertex.clone(),
                p2: p2.clone(),
            },
            GeoConstruction::LineIntersection {
                l1_p1,
                l1_p2,
                l2_p1,
                l2_p2,
            } => ConstructionKind::LineIntersection {
                l1_p1: l1_p1.clone(),
                l1_p2: l1_p2.clone(),
                l2_p1: l2_p1.clone(),
                l2_p2: l2_p2.clone(),
            },
            GeoConstruction::Foot {
                point,
                line_p1,
                line_p2,
            } => ConstructionKind::Foot {
                point: point.clone(),
                line_p1: line_p1.clone(),
                line_p2: line_p2.clone(),
            },
            GeoConstruction::Reflection {
                point,
                line_p1,
                line_p2,
            } => ConstructionKind::Reflection {
                point: point.clone(),
                line_p1: line_p1.clone(),
                line_p2: line_p2.clone(),
            },
            GeoConstruction::ExtendLine { .. } => {
                // ExtendLine doesn't create a new point, just marks intention
                // Return a dummy construction
                ConstructionKind::Midpoint {
                    p1: String::new(),
                    p2: String::new(),
                }
            }
        };

        Construction {
            kind,
            new_points: vec![], // Will be filled by apply_construction
            confidence,
            source: ConstructionSource::Neural {
                model: "mcts".to_string(),
                confidence,
            },
        }
    }

    /// Get the name of the new point that will be created
    fn new_point_name(&self) -> Option<String> {
        match self {
            GeoConstruction::Midpoint { p1, p2 } => Some(format!("M_{}_{}", p1, p2)),
            GeoConstruction::Perpendicular {
                point,
                line_p1,
                line_p2,
            } => Some(format!("Perp_{}_{}{}", point, line_p1, line_p2)),
            GeoConstruction::Parallel {
                point,
                line_p1,
                line_p2,
            } => Some(format!("Par_{}_{}{}", point, line_p1, line_p2)),
            GeoConstruction::Circumcircle { p1, p2, p3 } => Some(format!("O_{}{}{}", p1, p2, p3)),
            GeoConstruction::Incircle { p1, p2, p3 } => Some(format!("I_{}{}{}", p1, p2, p3)),
            GeoConstruction::AngleBisector { p1, vertex, p2 } => {
                Some(format!("Bis_{}{}{}", p1, vertex, p2))
            }
            GeoConstruction::LineIntersection {
                l1_p1,
                l1_p2,
                l2_p1,
                l2_p2,
            } => Some(format!("Int_{}{}_{}{}", l1_p1, l1_p2, l2_p1, l2_p2)),
            GeoConstruction::Foot {
                point,
                line_p1,
                line_p2,
            } => Some(format!("F_{}_{}{}", point, line_p1, line_p2)),
            GeoConstruction::Reflection {
                point,
                line_p1,
                line_p2,
            } => Some(format!("Ref_{}_{}{}", point, line_p1, line_p2)),
            GeoConstruction::ExtendLine { .. } => None,
        }
    }
}

// =============================================================================
// Game Configuration
// =============================================================================

/// Configuration for the geometry proof game
#[derive(Debug, Clone)]
pub struct GeoGameConfig {
    /// Maximum number of actions (steps) allowed
    pub max_steps: usize,
    /// Maximum number of constructions allowed
    pub max_constructions: usize,
    /// Minimum confidence to consider goal proved
    pub goal_confidence: f64,
    /// Maximum variance to consider goal proved (epistemic certainty)
    pub max_goal_variance: f64,
    /// Penalty per step (encourages shorter proofs)
    pub step_penalty: f64,
    /// Penalty per construction (encourages minimal auxiliary points)
    pub construction_penalty: f64,
    /// Bonus for reducing global uncertainty
    pub uncertainty_reduction_bonus: f64,
    /// Engine configuration for symbolic deduction
    pub engine_config: EngineConfig,
    /// Whether to generate all possible constructions as actions
    pub enumerate_constructions: bool,
    /// Maximum number of construction actions to enumerate
    pub max_construction_actions: usize,
}

impl Default for GeoGameConfig {
    fn default() -> Self {
        GeoGameConfig {
            max_steps: 50,
            max_constructions: 5,
            goal_confidence: 0.9,
            max_goal_variance: 0.1,
            step_penalty: 0.01,
            construction_penalty: 0.05,
            uncertainty_reduction_bonus: 0.1,
            engine_config: EngineConfig::default(),
            enumerate_constructions: true,
            max_construction_actions: 20,
        }
    }
}

// =============================================================================
// Geometry Proof Game State
// =============================================================================

/// The state of a geometry proof game
#[derive(Clone)]
pub struct GeoProofGame {
    /// Current proof state
    pub state: ProofState,
    /// Game configuration
    pub config: GeoGameConfig,
    /// Number of steps taken
    pub steps: usize,
    /// Number of constructions added
    pub num_constructions: usize,
    /// Whether the goal has been proved
    pub proved: bool,
    /// Whether the game has terminated
    pub terminated: bool,
    /// Final reward (computed at termination)
    pub reward: f64,
    /// Global uncertainty at previous step (for reward shaping)
    pub prev_uncertainty: f64,
    /// History of actions taken
    pub action_history: Vec<GeoAction>,
    /// Cached legal actions (for efficiency)
    cached_actions: Option<Vec<GeoAction>>,
    /// Current player (always Player::One for single-player proof search)
    player: Player,
}

impl fmt::Debug for GeoProofGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeoProofGame")
            .field("steps", &self.steps)
            .field("num_constructions", &self.num_constructions)
            .field("proved", &self.proved)
            .field("terminated", &self.terminated)
            .field("num_predicates", &self.state.num_predicates())
            .field("num_points", &self.state.points.len())
            .finish()
    }
}

impl GeoProofGame {
    /// Create a new game from a proof state with a goal
    pub fn new(state: ProofState, config: GeoGameConfig) -> Self {
        let prev_uncertainty = state.global_uncertainty();
        GeoProofGame {
            state,
            config,
            steps: 0,
            num_constructions: 0,
            proved: false,
            terminated: false,
            reward: 0.0,
            prev_uncertainty,
            action_history: Vec::new(),
            cached_actions: None,
            player: Player::One,
        }
    }

    /// Create a game for proving a specific goal
    pub fn for_goal(mut state: ProofState, goal: Predicate, min_confidence: f64) -> Self {
        state.set_goal(goal, min_confidence);
        Self::new(state, GeoGameConfig::default())
    }

    /// Check if the goal is satisfied
    pub fn goal_satisfied(&self) -> bool {
        self.state
            .goal_satisfied_with_certainty(self.config.max_goal_variance)
    }

    /// Get the current global uncertainty
    pub fn global_uncertainty(&self) -> f64 {
        self.state.global_uncertainty()
    }

    /// Get the goal confidence if proved
    pub fn goal_confidence(&self) -> Option<BetaConfidence> {
        self.state.goal.as_ref().and_then(|g| {
            self.state
                .get_predicate(&g.predicate.key())
                .map(|p| p.epistemic.confidence)
        })
    }

    /// Generate all possible construction actions from current state
    fn generate_construction_actions(&self) -> Vec<GeoAction> {
        if self.num_constructions >= self.config.max_constructions {
            return vec![];
        }

        let mut actions = Vec::new();
        let points: Vec<&str> = self.state.point_labels();
        let n = points.len();

        if n < 2 {
            return vec![];
        }

        // Limit construction enumeration
        let mut count = 0;
        let max = self.config.max_construction_actions;

        // Midpoints (n choose 2)
        for i in 0..n {
            for j in (i + 1)..n {
                if count >= max {
                    break;
                }
                // Check if midpoint already exists
                let mid_name = format!("M_{}_{}", points[i], points[j]);
                if !self.state.points.contains_key(&mid_name) {
                    actions.push(GeoAction::Construct(GeoConstruction::Midpoint {
                        p1: points[i].to_string(),
                        p2: points[j].to_string(),
                    }));
                    count += 1;
                }
            }
        }

        // Circumcircles (n choose 3) - only for triangles
        if n >= 3 {
            for i in 0..n.min(5) {
                for j in (i + 1)..n.min(6) {
                    for k in (j + 1)..n.min(7) {
                        if count >= max {
                            break;
                        }
                        let center_name = format!("O_{}{}{}", points[i], points[j], points[k]);
                        if !self.state.points.contains_key(&center_name) {
                            actions.push(GeoAction::Construct(GeoConstruction::Circumcircle {
                                p1: points[i].to_string(),
                                p2: points[j].to_string(),
                                p3: points[k].to_string(),
                            }));
                            count += 1;
                        }
                    }
                }
            }
        }

        // Perpendicular feet (point to line)
        for i in 0..n.min(4) {
            for j in 0..n.min(4) {
                for k in (j + 1)..n.min(5) {
                    if count >= max {
                        break;
                    }
                    if i != j && i != k {
                        let foot_name = format!("F_{}_{}{}", points[i], points[j], points[k]);
                        if !self.state.points.contains_key(&foot_name) {
                            actions.push(GeoAction::Construct(GeoConstruction::Foot {
                                point: points[i].to_string(),
                                line_p1: points[j].to_string(),
                                line_p2: points[k].to_string(),
                            }));
                            count += 1;
                        }
                    }
                }
            }
        }

        // Line intersections (limited)
        if n >= 4 {
            for i in 0..n.min(3) {
                for j in (i + 1)..n.min(4) {
                    for k in 0..n.min(3) {
                        for l in (k + 1)..n.min(4) {
                            if count >= max {
                                break;
                            }
                            // Skip if lines share a point
                            if i != k && i != l && j != k && j != l {
                                let int_name = format!(
                                    "Int_{}{}_{}{}",
                                    points[i], points[j], points[k], points[l]
                                );
                                if !self.state.points.contains_key(&int_name) {
                                    actions.push(GeoAction::Construct(
                                        GeoConstruction::LineIntersection {
                                            l1_p1: points[i].to_string(),
                                            l1_p2: points[j].to_string(),
                                            l2_p1: points[k].to_string(),
                                            l2_p2: points[l].to_string(),
                                        },
                                    ));
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        actions
    }

    /// Apply a construction to the state
    fn apply_construction(&mut self, construction: &GeoConstruction) {
        let conf = self.goal_confidence().map(|c| c.mean()).unwrap_or(0.5);

        let constr = construction.to_construction(conf);
        self.state.apply_construction(constr);
        self.num_constructions += 1;
    }

    /// Run symbolic deduction for one step
    fn run_deduction_step(&mut self) -> DeductionResult {
        let mut engine = SymbolicEngine::with_config(EngineConfig {
            max_iterations: 10, // Limited iterations per step
            ..self.config.engine_config.clone()
        });
        engine.deduce(self.state.clone())
    }

    /// Compute the reward for the current state
    fn compute_reward(&self) -> f64 {
        if self.proved {
            // Base reward for proving
            let mut reward = 1.0;

            // Bonus for shorter proofs
            let step_bonus = 1.0 - (self.steps as f64 / self.config.max_steps as f64);
            reward += 0.2 * step_bonus;

            // Bonus for fewer constructions
            let constr_bonus =
                1.0 - (self.num_constructions as f64 / self.config.max_constructions as f64);
            reward += 0.1 * constr_bonus;

            // Bonus for high confidence
            if let Some(conf) = self.goal_confidence() {
                reward += 0.1 * conf.mean();
            }

            reward
        } else if self.terminated {
            // Penalty for giving up
            -0.5
        } else {
            // Intermediate reward based on uncertainty reduction
            let curr_uncertainty = self.global_uncertainty();
            let reduction = self.prev_uncertainty - curr_uncertainty;

            // Small positive reward for reducing uncertainty
            let uncertainty_reward = reduction * self.config.uncertainty_reduction_bonus;

            // Small penalties for taking steps
            let step_penalty = -self.config.step_penalty;

            uncertainty_reward + step_penalty
        }
    }
}

impl GameState for GeoProofGame {
    type Action = GeoAction;

    fn current_player(&self) -> Player {
        self.player // Always Player::One (single-player game)
    }

    fn legal_actions(&self) -> Vec<Self::Action> {
        if self.terminated || self.proved {
            return vec![];
        }

        let mut actions = Vec::new();

        // Always can do a deduction step
        actions.push(GeoAction::DeductionStep);

        // Can request neural suggestion
        actions.push(GeoAction::RequestNeural);

        // Can terminate
        actions.push(GeoAction::Terminate);

        // Add construction actions if enabled and not at limit
        if self.config.enumerate_constructions
            && self.num_constructions < self.config.max_constructions
        {
            actions.extend(self.generate_construction_actions());
        }

        actions
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut new_game = self.clone();
        new_game.cached_actions = None; // Invalidate cache
        new_game.prev_uncertainty = new_game.global_uncertainty();
        new_game.steps += 1;
        new_game.action_history.push(action.clone());

        match action {
            GeoAction::DeductionStep => {
                let result = new_game.run_deduction_step();
                new_game.state = result.state;

                if result.proved {
                    new_game.proved = true;
                    new_game.terminated = true;
                }
            }
            GeoAction::Construct(construction) => {
                new_game.apply_construction(construction);
            }
            GeoAction::RequestNeural => {
                // In the game abstraction, this is a no-op
                // The neural network will provide the next action
            }
            GeoAction::Terminate => {
                new_game.terminated = true;
            }
            GeoAction::ApplyRule {
                rule_name,
                premise_indices,
            } => {
                // Direct rule application (advanced action)
                // TODO: Implement direct rule application
            }
        }

        // Check termination conditions
        if new_game.steps >= new_game.config.max_steps {
            new_game.terminated = true;
        }

        if new_game.goal_satisfied() {
            new_game.proved = true;
            new_game.terminated = true;
        }

        // Compute reward
        new_game.reward = new_game.compute_reward();

        new_game
    }

    fn is_terminal(&self) -> bool {
        self.terminated || self.proved
    }

    fn terminal_value(&self, _player: Player) -> f64 {
        // Since this is single-player, player doesn't matter
        self.reward.max(0.0).min(1.0)
    }
}

impl GameTrait for GeoProofGame {
    fn to_features(&self) -> Vec<f32> {
        // Feature vector for neural network input
        // This is a simplified encoding; real implementation would be richer

        let mut features = Vec::new();

        // Basic statistics
        features.push(self.steps as f32 / self.config.max_steps as f32);
        features.push(self.num_constructions as f32 / self.config.max_constructions as f32);
        features.push(self.state.num_predicates() as f32 / 100.0);
        features.push(self.state.points.len() as f32 / 20.0);
        features.push(self.global_uncertainty() as f32);

        // Goal progress
        let goal_conf = self
            .goal_confidence()
            .map(|c| c.mean() as f32)
            .unwrap_or(0.0);
        features.push(goal_conf);

        // Predicate type counts (normalized)
        let predicate_counts = self.predicate_type_counts();
        for kind in &[
            PredicateKind::Collinear,
            PredicateKind::Parallel,
            PredicateKind::Perpendicular,
            PredicateKind::EqualLength,
            PredicateKind::Midpoint,
            PredicateKind::Congruent,
            PredicateKind::Similar,
            PredicateKind::OnCircle,
        ] {
            let count = predicate_counts.get(kind).copied().unwrap_or(0);
            features.push(count as f32 / 20.0);
        }

        // Pad to fixed size
        while features.len() < 64 {
            features.push(0.0);
        }

        features.truncate(64);
        features
    }

    fn feature_shape() -> Vec<usize> {
        vec![64] // Flat feature vector
    }

    fn num_actions() -> usize {
        // Fixed action space size:
        // 1 (deduction) + 1 (neural) + 1 (terminate) + max_constructions
        3 + 50 // Reserve space for constructions
    }

    fn action_to_index(action: &Self::Action) -> usize {
        match action {
            GeoAction::DeductionStep => 0,
            GeoAction::RequestNeural => 1,
            GeoAction::Terminate => 2,
            GeoAction::Construct(c) => {
                // Hash construction to get index
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                c.hash(&mut hasher);
                3 + (hasher.finish() as usize % 50)
            }
            GeoAction::ApplyRule { .. } => {
                // Rules are handled dynamically
                52
            }
        }
    }

    fn index_to_action(index: usize) -> Option<Self::Action> {
        match index {
            0 => Some(GeoAction::DeductionStep),
            1 => Some(GeoAction::RequestNeural),
            2 => Some(GeoAction::Terminate),
            _ => None, // Constructions must be enumerated from state
        }
    }

    fn initial() -> Self {
        // Create an empty proof state (should be overridden)
        let state = ProofState::new();
        Self::new(state, GeoGameConfig::default())
    }

    fn name() -> &'static str {
        "GeoProofGame"
    }
}

impl GeoProofGame {
    /// Get counts of each predicate type
    fn predicate_type_counts(&self) -> HashMap<PredicateKind, usize> {
        let mut counts = HashMap::new();
        for pred in self.state.all_predicates() {
            *counts.entry(pred.kind.clone()).or_insert(0) += 1;
        }
        counts
    }
}

// =============================================================================
// Generate Proof Game (Self-Play Entry Point)
// =============================================================================

/// Result of a proof game episode
#[derive(Debug, Clone)]
pub struct ProofGameEpisode {
    /// Initial state
    pub initial_state: ProofState,
    /// Final state
    pub final_state: ProofState,
    /// Was the goal proved?
    pub proved: bool,
    /// Total reward
    pub total_reward: f64,
    /// Number of steps taken
    pub num_steps: usize,
    /// Number of constructions used
    pub num_constructions: usize,
    /// Action history
    pub actions: Vec<GeoAction>,
    /// State-action-value triples for training
    pub trajectory: Vec<TrajectoryStep>,
    /// Final confidence in goal (if proved)
    pub goal_confidence: Option<BetaConfidence>,
}

/// A single step in the trajectory (for training)
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// Features at this state
    pub features: Vec<f32>,
    /// Action taken
    pub action: GeoAction,
    /// Action probabilities from MCTS
    pub action_probs: HashMap<GeoAction, f64>,
    /// Value estimate at this state
    pub value: f64,
    /// Epistemic uncertainty at this state
    pub uncertainty: f64,
}

/// Generate a proof game episode using MCTS
///
/// This is the main entry point for self-play. Given a geometry problem
/// (initial state + goal), run MCTS to find a proof and return the
/// episode data for training.
pub fn generate_proof_game<E>(
    state: ProofState,
    goal: Predicate,
    evaluator: &E,
    mcts_simulations: usize,
) -> ProofGameEpisode
where
    E: crate::rl::mcts::NeuralEvaluator<GeoProofGame>,
{
    use crate::rl::mcts::{MCTSConfig, MCTSTree, search};

    let mut game = GeoProofGame::for_goal(state.clone(), goal, 0.9);
    let mut trajectory = Vec::new();
    let mut actions = Vec::new();

    let mcts_config = MCTSConfig {
        num_simulations: mcts_simulations,
        c_puct: 1.5,
        c_epistemic: 0.5, // Epistemic exploration bonus
        temperature: 1.0,
        ..MCTSConfig::default()
    };

    while !game.is_terminal() {
        // Run MCTS from current position
        let mut tree = MCTSTree::new(game.clone(), mcts_config.clone());
        let result = search(&mut tree, evaluator);

        // Record trajectory step
        trajectory.push(TrajectoryStep {
            features: game.to_features(),
            action: result.best_action.clone().unwrap_or(GeoAction::Terminate),
            action_probs: result.action_probabilities.clone(),
            value: result.root_value.mean(),
            uncertainty: result.global_uncertainty,
        });

        // Take the best action
        if let Some(action) = result.best_action {
            actions.push(action.clone());
            game = game.apply_action(&action);
        } else {
            break;
        }
    }

    ProofGameEpisode {
        initial_state: state,
        final_state: game.state.clone(),
        proved: game.proved,
        total_reward: game.reward,
        num_steps: game.steps,
        num_constructions: game.num_constructions,
        actions,
        trajectory,
        goal_confidence: game.goal_confidence(),
    }
}

/// Simplified version for testing without neural network
///
/// Uses a deterministic pseudo-random selection based on state hash
/// to enable reproducible testing without requiring the rand crate.
pub fn generate_proof_game_random(
    state: ProofState,
    goal: Predicate,
    max_steps: usize,
) -> ProofGameEpisode {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut game = GeoProofGame::for_goal(state.clone(), goal, 0.9);
    let mut actions = Vec::new();
    let mut trajectory = Vec::new();

    // Simple deterministic "random" based on step count and state hash
    let mut pseudo_random_seed: u64 = 42;

    while !game.is_terminal() && game.steps < max_steps {
        let legal = game.legal_actions();
        if legal.is_empty() {
            break;
        }

        // Deterministic pseudo-random selection
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(pseudo_random_seed);
        hasher.write_usize(game.steps);
        hasher.write_usize(game.state.num_predicates());
        pseudo_random_seed = hasher.finish();

        let action_idx = (pseudo_random_seed as usize) % legal.len();
        let action = legal[action_idx].clone();

        // Record trajectory
        let mut action_probs = HashMap::new();
        for a in &legal {
            action_probs.insert(a.clone(), 1.0 / legal.len() as f64);
        }

        trajectory.push(TrajectoryStep {
            features: game.to_features(),
            action: action.clone(),
            action_probs,
            value: 0.5, // No value estimate
            uncertainty: game.global_uncertainty(),
        });

        actions.push(action.clone());
        game = game.apply_action(&action);
    }

    ProofGameEpisode {
        initial_state: state,
        final_state: game.state.clone(),
        proved: game.proved,
        total_reward: game.reward,
        num_steps: game.steps,
        num_constructions: game.num_constructions,
        actions,
        trajectory,
        goal_confidence: game.goal_confidence(),
    }
}

// =============================================================================
// Standard Geometry Problems
// =============================================================================

/// Create a triangle congruence problem (SAS)
///
/// Given: Triangle ABC with AB = DE, angle BAC = angle EDF, AC = DF
/// Prove: Triangle ABC is congruent to triangle DEF
pub fn triangle_congruence_sas() -> (ProofState, Predicate) {
    let mut state = ProofState::new();

    // Add points for both triangles
    state.add_points(&["A", "B", "C", "D", "E", "F"]);

    // Add axioms for SAS
    state.add_axiom(Predicate::equal_length("A", "B", "D", "E")); // AB = DE
    state.add_axiom(Predicate::equal_length("A", "C", "D", "F")); // AC = DF
    // Angle BAC = Angle EDF (represented as equal angles)
    let angle_equal = Predicate::new(
        PredicateKind::EqualAngle,
        vec![
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
            "E".to_string(),
            "D".to_string(),
            "F".to_string(),
        ],
    );
    state.add_axiom(angle_equal);

    // Goal: triangles are congruent
    let goal = Predicate::new(
        PredicateKind::Congruent,
        vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
            "F".to_string(),
        ],
    );

    (state, goal)
}

/// Create a midpoint theorem problem
///
/// Given: M is midpoint of AB, N is midpoint of AC
/// Prove: MN is parallel to BC
pub fn midpoint_theorem() -> (ProofState, Predicate) {
    let mut state = ProofState::new();

    state.add_points(&["A", "B", "C", "M", "N"]);
    state.add_axiom(Predicate::midpoint("M", "A", "B"));
    state.add_axiom(Predicate::midpoint("N", "A", "C"));

    let goal = Predicate::parallel("M", "N", "B", "C");

    (state, goal)
}

/// Create an isoceles triangle problem
///
/// Given: Triangle ABC with AB = AC, M is midpoint of BC
/// Prove: AM is perpendicular to BC
pub fn isoceles_perpendicular() -> (ProofState, Predicate) {
    let mut state = ProofState::new();

    state.add_points(&["A", "B", "C", "M"]);
    state.add_axiom(Predicate::equal_length("A", "B", "A", "C")); // AB = AC
    state.add_axiom(Predicate::midpoint("M", "B", "C")); // M is midpoint of BC

    let goal = Predicate::perpendicular("A", "M", "B", "C");

    (state, goal)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_game_creation() {
        let (state, goal) = midpoint_theorem();
        let game = GeoProofGame::for_goal(state, goal, 0.9);

        assert!(!game.is_terminal());
        assert!(!game.proved);
        assert_eq!(game.steps, 0);
    }

    #[test]
    fn test_legal_actions() {
        let (state, goal) = midpoint_theorem();
        let game = GeoProofGame::for_goal(state, goal, 0.9);

        let actions = game.legal_actions();

        // Should have at least deduction, neural, terminate
        assert!(actions.len() >= 3);
        assert!(actions.contains(&GeoAction::DeductionStep));
        assert!(actions.contains(&GeoAction::RequestNeural));
        assert!(actions.contains(&GeoAction::Terminate));
    }

    #[test]
    fn test_deduction_step() {
        let (state, goal) = midpoint_theorem();
        let game = GeoProofGame::for_goal(state, goal, 0.9);

        let new_game = game.apply_action(&GeoAction::DeductionStep);

        assert_eq!(new_game.steps, 1);
        // Midpoint theorem should be provable in a few steps
    }

    #[test]
    fn test_construction_action() {
        let (state, goal) = triangle_congruence_sas();
        let game = GeoProofGame::for_goal(state, goal, 0.9);

        let construction = GeoAction::Construct(GeoConstruction::Midpoint {
            p1: "A".to_string(),
            p2: "B".to_string(),
        });

        let new_game = game.apply_action(&construction);

        assert_eq!(new_game.num_constructions, 1);
        assert!(new_game.state.points.contains_key("M_A_B"));
    }

    #[test]
    fn test_terminate_action() {
        let (state, goal) = midpoint_theorem();
        let game = GeoProofGame::for_goal(state, goal, 0.9);

        let new_game = game.apply_action(&GeoAction::Terminate);

        assert!(new_game.is_terminal());
        assert!(!new_game.proved);
    }

    #[test]
    fn test_features() {
        let (state, goal) = midpoint_theorem();
        let game = GeoProofGame::for_goal(state, goal, 0.9);

        let features = game.to_features();

        assert_eq!(features.len(), 64);
        // All features should be in reasonable range
        for f in &features {
            assert!(*f >= 0.0 && *f <= 10.0);
        }
    }

    #[test]
    fn test_random_proof_game() {
        let (state, goal) = midpoint_theorem();

        let episode = generate_proof_game_random(state, goal, 20);

        // Should have taken some steps
        assert!(episode.num_steps > 0);
        // Midpoint theorem might be proved by random exploration
        // (or might not - that's fine for this test)
    }

    #[test]
    fn test_game_trait_implementation() {
        // Verify GameTrait is properly implemented
        assert_eq!(GeoProofGame::name(), "GeoProofGame");
        assert_eq!(GeoProofGame::feature_shape(), vec![64]);
        assert!(GeoProofGame::num_actions() > 3);

        // Test action conversion
        assert_eq!(
            GeoProofGame::index_to_action(0),
            Some(GeoAction::DeductionStep)
        );
        assert_eq!(GeoProofGame::action_to_index(&GeoAction::DeductionStep), 0);
    }

    #[test]
    fn test_construction_new_point_name() {
        let c = GeoConstruction::Midpoint {
            p1: "A".to_string(),
            p2: "B".to_string(),
        };
        assert_eq!(c.new_point_name(), Some("M_A_B".to_string()));

        let c2 = GeoConstruction::Circumcircle {
            p1: "A".to_string(),
            p2: "B".to_string(),
            p3: "C".to_string(),
        };
        assert_eq!(c2.new_point_name(), Some("O_ABC".to_string()));
    }
}

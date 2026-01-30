//! Geometry Self-Play Training Loop
//!
//! AlphaGeometry-style self-play training with epistemic enhancements.
//! This module provides the complete training loop for learning to prove
//! geometry theorems through self-play.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Geometry Self-Play Loop                    │
//! │  ┌───────────┐     ┌───────────┐     ┌───────────┐         │
//! │  │ Problem   │ --> │ MCTS      │ --> │ Trajectory │         │
//! │  │ Generator │     │ Search    │     │ Collection │         │
//! │  └───────────┘     └───────────┘     └───────────┘         │
//! │                                            │                 │
//! │                                            v                 │
//! │  ┌───────────┐     ┌───────────┐     ┌───────────┐         │
//! │  │ Improved  │ <-- │ Neural    │ <-- │ Training  │         │
//! │  │ Network   │     │ Training  │     │ Examples  │         │
//! │  └───────────┘     └───────────┘     └───────────┘         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Epistemic Innovation
//!
//! Unlike standard AlphaZero, we use:
//! - Beta-distributed Q-values for uncertainty quantification
//! - Variance penalty in loss function (discourage overconfident predictions)
//! - Epistemic exploration bonus in MCTS

use std::collections::HashMap;

use crate::epistemic::bayesian::BetaConfidence;
use crate::rl::game::{GameState, GameTrait};
use crate::rl::mcts::{MCTSConfig, MCTSTree, NeuralEvaluator, search};

use super::geo_game::{
    GeoAction, GeoGameConfig, GeoProofGame, ProofGameEpisode, TrajectoryStep,
    generate_proof_game_random, isoceles_perpendicular, midpoint_theorem, triangle_congruence_sas,
};
use super::predicates::Predicate;
use super::proof_state::ProofState;

// =============================================================================
// Training Configuration
// =============================================================================

/// Configuration for geometry self-play training
#[derive(Debug, Clone)]
pub struct GeoTrainingConfig {
    /// MCTS configuration
    pub mcts_config: MCTSConfig,
    /// Game configuration
    pub game_config: GeoGameConfig,
    /// Number of self-play games per training iteration
    pub games_per_iteration: usize,
    /// Number of training iterations
    pub num_iterations: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Weight for policy loss
    pub policy_loss_weight: f64,
    /// Weight for value loss
    pub value_loss_weight: f64,
    /// Weight for variance penalty (epistemic regularization)
    pub variance_penalty_weight: f64,
    /// Temperature schedule: (step, temperature)
    pub temperature_schedule: Vec<(usize, f64)>,
    /// Minimum confidence to consider a proof valid
    pub min_proof_confidence: f64,
    /// Whether to save checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint directory
    pub checkpoint_dir: String,
}

impl Default for GeoTrainingConfig {
    fn default() -> Self {
        GeoTrainingConfig {
            mcts_config: MCTSConfig {
                num_simulations: 100,
                c_puct: 1.5,
                c_epistemic: 0.5, // Epistemic exploration bonus
                temperature: 1.0,
                ..MCTSConfig::default()
            },
            game_config: GeoGameConfig::default(),
            games_per_iteration: 50,
            num_iterations: 100,
            learning_rate: 0.001,
            batch_size: 32,
            policy_loss_weight: 1.0,
            value_loss_weight: 1.0,
            variance_penalty_weight: 0.1, // Penalize high uncertainty
            temperature_schedule: vec![
                (0, 1.0),   // High exploration initially
                (50, 0.5),  // Medium exploration
                (100, 0.1), // Low exploration late
            ],
            min_proof_confidence: 0.9,
            save_checkpoints: false,
            checkpoint_dir: "./checkpoints".to_string(),
        }
    }
}

impl GeoTrainingConfig {
    /// Get temperature for a given step
    pub fn temperature_for_step(&self, step: usize) -> f64 {
        let mut temp = 1.0;
        for &(threshold, t) in &self.temperature_schedule {
            if step >= threshold {
                temp = t;
            }
        }
        temp
    }
}

// =============================================================================
// Training Example
// =============================================================================

/// A training example from self-play
#[derive(Debug, Clone)]
pub struct GeoTrainingExample {
    /// State features
    pub features: Vec<f32>,
    /// Target policy (from MCTS visit distribution)
    pub policy_target: Vec<f32>,
    /// Target value (final game outcome)
    pub value_target: f32,
    /// Epistemic variance at this state (for variance penalty)
    pub variance: f32,
    /// Weight for this example (optional curriculum learning)
    pub weight: f32,
}

impl GeoTrainingExample {
    /// Create from a trajectory step
    pub fn from_step(step: &TrajectoryStep, outcome_value: f32) -> Self {
        // Convert action probabilities to policy vector
        let mut policy_target = vec![0.0; GeoProofGame::num_actions()];
        for (action, &prob) in &step.action_probs {
            let idx = GeoProofGame::action_to_index(action);
            if idx < policy_target.len() {
                policy_target[idx] = prob as f32;
            }
        }

        GeoTrainingExample {
            features: step.features.clone(),
            policy_target,
            value_target: outcome_value,
            variance: step.uncertainty as f32,
            weight: 1.0,
        }
    }
}

// =============================================================================
// Training Statistics
// =============================================================================

/// Statistics from a training run
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Total games played
    pub games_played: usize,
    /// Games where proof was found
    pub proofs_found: usize,
    /// Average steps to proof (for successful proofs)
    pub avg_proof_steps: f64,
    /// Average constructions used
    pub avg_constructions: f64,
    /// Average final confidence
    pub avg_confidence: f64,
    /// Total training examples generated
    pub total_examples: usize,
    /// Average policy loss
    pub avg_policy_loss: f64,
    /// Average value loss
    pub avg_value_loss: f64,
    /// Average variance penalty
    pub avg_variance_penalty: f64,
    /// Total training time (seconds)
    pub training_time_secs: f64,
}

impl TrainingStats {
    /// Update stats from an episode
    pub fn record_episode(&mut self, episode: &ProofGameEpisode) {
        self.games_played += 1;

        if episode.proved {
            self.proofs_found += 1;
            self.avg_proof_steps = (self.avg_proof_steps * (self.proofs_found - 1) as f64
                + episode.num_steps as f64)
                / self.proofs_found as f64;
        }

        self.avg_constructions = (self.avg_constructions * (self.games_played - 1) as f64
            + episode.num_constructions as f64)
            / self.games_played as f64;

        if let Some(conf) = &episode.goal_confidence {
            self.avg_confidence = (self.avg_confidence * (self.games_played - 1) as f64
                + conf.mean())
                / self.games_played as f64;
        }

        self.total_examples += episode.trajectory.len();
    }

    /// Compute proof success rate
    pub fn success_rate(&self) -> f64 {
        if self.games_played == 0 {
            0.0
        } else {
            self.proofs_found as f64 / self.games_played as f64
        }
    }
}

// =============================================================================
// Geometry Problem Generator
// =============================================================================

/// Types of geometry problems that can be generated
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemType {
    /// Midpoint theorem: M, N midpoints => MN || BC
    MidpointTheorem,
    /// Triangle congruence (SAS)
    TriangleCongruenceSAS,
    /// Isoceles triangle perpendicular
    IsocelesPerpendicular,
    /// Random problem from problem database
    Random,
}

/// A geometry problem for training
#[derive(Debug, Clone)]
pub struct GeoProblem {
    /// Problem type
    pub problem_type: ProblemType,
    /// Initial proof state
    pub state: ProofState,
    /// Goal to prove
    pub goal: Predicate,
    /// Difficulty estimate (0-1)
    pub difficulty: f64,
    /// Problem name/description
    pub name: String,
}

impl GeoProblem {
    /// Create a standard problem by type
    pub fn standard(problem_type: ProblemType) -> Self {
        match problem_type {
            ProblemType::MidpointTheorem => {
                let (state, goal) = midpoint_theorem();
                GeoProblem {
                    problem_type,
                    state,
                    goal,
                    difficulty: 0.3,
                    name: "Midpoint Theorem".to_string(),
                }
            }
            ProblemType::TriangleCongruenceSAS => {
                let (state, goal) = triangle_congruence_sas();
                GeoProblem {
                    problem_type,
                    state,
                    goal,
                    difficulty: 0.5,
                    name: "Triangle Congruence (SAS)".to_string(),
                }
            }
            ProblemType::IsocelesPerpendicular => {
                let (state, goal) = isoceles_perpendicular();
                GeoProblem {
                    problem_type,
                    state,
                    goal,
                    difficulty: 0.4,
                    name: "Isoceles Perpendicular".to_string(),
                }
            }
            ProblemType::Random => {
                // Default to midpoint theorem for now
                // Real implementation would sample from problem database
                Self::standard(ProblemType::MidpointTheorem)
            }
        }
    }

    /// Get all standard problems
    pub fn all_standard() -> Vec<Self> {
        vec![
            Self::standard(ProblemType::MidpointTheorem),
            Self::standard(ProblemType::TriangleCongruenceSAS),
            Self::standard(ProblemType::IsocelesPerpendicular),
        ]
    }
}

/// Problem generator for self-play
pub struct ProblemGenerator {
    /// Available problems
    problems: Vec<GeoProblem>,
    /// Current problem index
    current_idx: usize,
    /// Whether to cycle through problems
    cycle: bool,
}

impl ProblemGenerator {
    /// Create with standard problems
    pub fn standard() -> Self {
        ProblemGenerator {
            problems: GeoProblem::all_standard(),
            current_idx: 0,
            cycle: true,
        }
    }

    /// Create with custom problems
    pub fn with_problems(problems: Vec<GeoProblem>) -> Self {
        ProblemGenerator {
            problems,
            current_idx: 0,
            cycle: true,
        }
    }

    /// Get next problem
    pub fn next(&mut self) -> Option<GeoProblem> {
        if self.problems.is_empty() {
            return None;
        }

        let problem = self.problems[self.current_idx].clone();
        self.current_idx += 1;

        if self.cycle && self.current_idx >= self.problems.len() {
            self.current_idx = 0;
        }

        Some(problem)
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }
}

// =============================================================================
// Geometry Neural Network Interface
// =============================================================================

/// Trait for geometry proof neural networks
pub trait GeoNeuralNetwork: Send + Sync {
    /// Forward pass: state -> (policy, value)
    fn forward(&self, state: &GeoProofGame) -> (Vec<f32>, f32);

    /// Get policy and value as Beta distributions
    fn forward_epistemic(&self, state: &GeoProofGame) -> (Vec<f32>, BetaConfidence) {
        let (policy, value) = self.forward(state);
        // Convert scalar value to Beta distribution
        let beta_value = BetaConfidence::from_confidence(value as f64, 10.0);
        (policy, beta_value)
    }

    /// Train on batch of examples
    fn train(&mut self, examples: &[GeoTrainingExample], learning_rate: f64) -> f64;

    /// Save model to file
    fn save(&self, path: &str) -> Result<(), String>;

    /// Load model from file
    fn load(&mut self, path: &str) -> Result<(), String>;
}

/// Uniform (random) network for baseline/testing
#[derive(Clone)]
pub struct UniformGeoNetwork;

impl UniformGeoNetwork {
    pub fn new() -> Self {
        UniformGeoNetwork
    }
}

impl Default for UniformGeoNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl GeoNeuralNetwork for UniformGeoNetwork {
    fn forward(&self, state: &GeoProofGame) -> (Vec<f32>, f32) {
        let num_actions = GeoProofGame::num_actions();
        let legal = state.legal_actions();

        // Uniform policy over legal actions
        let mut policy = vec![0.0; num_actions];
        if !legal.is_empty() {
            let prob = 1.0 / legal.len() as f32;
            for action in legal {
                let idx = GeoProofGame::action_to_index(&action);
                if idx < num_actions {
                    policy[idx] = prob;
                }
            }
        }

        // Value = 0.5 (uncertain)
        (policy, 0.5)
    }

    fn train(&mut self, _examples: &[GeoTrainingExample], _learning_rate: f64) -> f64 {
        // Uniform network doesn't learn
        0.0
    }

    fn save(&self, _path: &str) -> Result<(), String> {
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<(), String> {
        Ok(())
    }
}

/// Adapter to use GeoNeuralNetwork with MCTS
pub struct GeoNetworkEvaluator<'a, N: GeoNeuralNetwork> {
    network: &'a N,
}

impl<'a, N: GeoNeuralNetwork> GeoNetworkEvaluator<'a, N> {
    pub fn new(network: &'a N) -> Self {
        GeoNetworkEvaluator { network }
    }
}

impl<'a, N: GeoNeuralNetwork> NeuralEvaluator<GeoProofGame> for GeoNetworkEvaluator<'a, N> {
    fn evaluate(&self, state: &GeoProofGame) -> (HashMap<GeoAction, f64>, f64) {
        let (policy_vec, value) = self.network.forward(state);

        // Convert policy vector to action map
        let mut policy = HashMap::new();
        for action in state.legal_actions() {
            let idx = GeoProofGame::action_to_index(&action);
            if idx < policy_vec.len() {
                policy.insert(action, policy_vec[idx] as f64);
            }
        }

        // Normalize
        let total: f64 = policy.values().sum();
        if total > 0.0 {
            for v in policy.values_mut() {
                *v /= total;
            }
        }

        (policy, value as f64)
    }
}

// =============================================================================
// Self-Play Training Loop
// =============================================================================

/// The main geometry self-play trainer
pub struct GeoSelfPlayTrainer<N: GeoNeuralNetwork> {
    /// Neural network
    pub network: N,
    /// Training configuration
    pub config: GeoTrainingConfig,
    /// Problem generator
    pub problem_generator: ProblemGenerator,
    /// Training statistics
    pub stats: TrainingStats,
    /// Collected training examples
    examples: Vec<GeoTrainingExample>,
    /// Current iteration
    iteration: usize,
}

impl<N: GeoNeuralNetwork> GeoSelfPlayTrainer<N> {
    /// Create a new trainer
    pub fn new(network: N, config: GeoTrainingConfig) -> Self {
        GeoSelfPlayTrainer {
            network,
            config,
            problem_generator: ProblemGenerator::standard(),
            stats: TrainingStats::default(),
            examples: Vec::new(),
            iteration: 0,
        }
    }

    /// Create with custom problem generator
    pub fn with_problems(network: N, config: GeoTrainingConfig, problems: Vec<GeoProblem>) -> Self {
        GeoSelfPlayTrainer {
            network,
            config,
            problem_generator: ProblemGenerator::with_problems(problems),
            stats: TrainingStats::default(),
            examples: Vec::new(),
            iteration: 0,
        }
    }

    /// Run one iteration of self-play + training
    pub fn run_iteration(&mut self) -> IterationResult {
        let start_time = std::time::Instant::now();
        let mut iteration_examples = Vec::new();
        let mut iteration_episodes = Vec::new();

        // Generate self-play games
        for _ in 0..self.config.games_per_iteration {
            if let Some(problem) = self.problem_generator.next() {
                let episode = self.play_episode(&problem);

                // Collect training examples from trajectory
                let outcome_value = if episode.proved {
                    1.0
                } else {
                    -0.5 // Penalty for not proving
                };

                for step in &episode.trajectory {
                    iteration_examples.push(GeoTrainingExample::from_step(step, outcome_value));
                }

                self.stats.record_episode(&episode);
                iteration_episodes.push(episode);
            }
        }

        // Train on collected examples
        let train_loss = if !iteration_examples.is_empty() {
            self.train_on_examples(&iteration_examples)
        } else {
            0.0
        };

        // Update iteration counter
        self.iteration += 1;
        let elapsed = start_time.elapsed().as_secs_f64();
        self.stats.training_time_secs += elapsed;

        IterationResult {
            iteration: self.iteration,
            games_played: iteration_episodes.len(),
            proofs_found: iteration_episodes.iter().filter(|e| e.proved).count(),
            examples_generated: iteration_examples.len(),
            training_loss: train_loss,
            elapsed_secs: elapsed,
        }
    }

    /// Play a single episode
    fn play_episode(&self, problem: &GeoProblem) -> ProofGameEpisode {
        let evaluator = GeoNetworkEvaluator::new(&self.network);

        let mut game = GeoProofGame::for_goal(
            problem.state.clone(),
            problem.goal.clone(),
            self.config.min_proof_confidence,
        );

        let mut trajectory = Vec::new();
        let mut actions = Vec::new();

        // Adjust temperature for current iteration
        let mut mcts_config = self.config.mcts_config.clone();
        mcts_config.temperature = self.config.temperature_for_step(self.iteration);

        while !game.is_terminal() {
            // Run MCTS
            let mut tree = MCTSTree::new(game.clone(), mcts_config.clone());
            let result = search(&mut tree, &evaluator);

            // Record trajectory step
            trajectory.push(TrajectoryStep {
                features: game.to_features(),
                action: result.best_action.clone().unwrap_or(GeoAction::Terminate),
                action_probs: result.action_probabilities.clone(),
                value: result.root_value.mean(),
                uncertainty: result.global_uncertainty,
            });

            // Take action
            if let Some(action) = result.best_action {
                actions.push(action.clone());
                game = game.apply_action(&action);
            } else {
                break;
            }
        }

        ProofGameEpisode {
            initial_state: problem.state.clone(),
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

    /// Train on batch of examples
    fn train_on_examples(&mut self, examples: &[GeoTrainingExample]) -> f64 {
        // Shuffle and batch
        let mut total_loss = 0.0;
        let num_batches = examples.len().div_ceil(self.config.batch_size);

        for batch_start in (0..examples.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(examples.len());
            let batch = &examples[batch_start..batch_end];

            let loss = self.network.train(batch, self.config.learning_rate);
            total_loss += loss;
        }

        if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        }
    }

    /// Run full training loop
    pub fn train(&mut self) -> TrainingStats {
        for iter in 0..self.config.num_iterations {
            let result = self.run_iteration();

            // Log progress (could use proper logging)
            if (iter + 1) % 10 == 0 {
                println!(
                    "Iteration {}/{}: {} games, {} proofs found, loss: {:.4}",
                    iter + 1,
                    self.config.num_iterations,
                    result.games_played,
                    result.proofs_found,
                    result.training_loss
                );
            }

            // Save checkpoint
            if self.config.save_checkpoints && (iter + 1) % 50 == 0 {
                let path = format!("{}/checkpoint_{}.bin", self.config.checkpoint_dir, iter + 1);
                let _ = self.network.save(&path);
            }
        }

        self.stats.clone()
    }

    /// Get current statistics
    pub fn statistics(&self) -> &TrainingStats {
        &self.stats
    }
}

/// Result of a single training iteration
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// Iteration number
    pub iteration: usize,
    /// Games played this iteration
    pub games_played: usize,
    /// Proofs found this iteration
    pub proofs_found: usize,
    /// Training examples generated
    pub examples_generated: usize,
    /// Training loss
    pub training_loss: f64,
    /// Time elapsed (seconds)
    pub elapsed_secs: f64,
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Run self-play training with default settings
pub fn train_geometry_prover(num_iterations: usize) -> TrainingStats {
    let network = UniformGeoNetwork::new();
    let config = GeoTrainingConfig {
        num_iterations,
        games_per_iteration: 10, // Fewer games per iteration for testing
        mcts_config: MCTSConfig {
            num_simulations: 50, // Fewer simulations for speed
            ..MCTSConfig::default()
        },
        ..Default::default()
    };

    let mut trainer = GeoSelfPlayTrainer::new(network, config);
    trainer.train()
}

/// Quick test on triangle congruence
pub fn test_triangle_congruence() -> ProofGameEpisode {
    let (state, goal) = triangle_congruence_sas();
    generate_proof_game_random(state, goal, 30)
}

/// Quick test on midpoint theorem
pub fn test_midpoint_theorem() -> ProofGameEpisode {
    let (state, goal) = midpoint_theorem();
    generate_proof_game_random(state, goal, 30)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = GeoTrainingConfig::default();
        assert!(config.num_iterations > 0);
        assert!(config.games_per_iteration > 0);
        assert!(config.variance_penalty_weight > 0.0);
    }

    #[test]
    fn test_temperature_schedule() {
        let config = GeoTrainingConfig::default();

        assert!((config.temperature_for_step(0) - 1.0).abs() < 0.01);
        assert!(config.temperature_for_step(100) < 1.0);
    }

    #[test]
    fn test_problem_generator() {
        let mut generator = ProblemGenerator::standard();

        let p1 = generator.next();
        assert!(p1.is_some());

        let p2 = generator.next();
        assert!(p2.is_some());

        // Should cycle
        for _ in 0..10 {
            assert!(generator.next().is_some());
        }
    }

    #[test]
    fn test_geo_problem_standard() {
        let problem = GeoProblem::standard(ProblemType::MidpointTheorem);
        assert_eq!(problem.name, "Midpoint Theorem");
        assert!(problem.difficulty > 0.0);
    }

    #[test]
    fn test_uniform_network() {
        let network = UniformGeoNetwork::new();
        let (state, goal) = midpoint_theorem();
        let game = GeoProofGame::for_goal(state, goal, 0.9);

        let (policy, value) = network.forward(&game);

        assert_eq!(policy.len(), GeoProofGame::num_actions());
        assert!((value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_training_example_creation() {
        let step = TrajectoryStep {
            features: vec![0.1, 0.2, 0.3],
            action: GeoAction::DeductionStep,
            action_probs: {
                let mut m = HashMap::new();
                m.insert(GeoAction::DeductionStep, 0.8);
                m.insert(GeoAction::Terminate, 0.2);
                m
            },
            value: 0.6,
            uncertainty: 0.1,
        };

        let example = GeoTrainingExample::from_step(&step, 1.0);

        assert_eq!(example.features.len(), 3);
        assert!((example.value_target - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_trainer_creation() {
        let network = UniformGeoNetwork::new();
        let config = GeoTrainingConfig::default();
        let trainer = GeoSelfPlayTrainer::new(network, config);

        assert_eq!(trainer.iteration, 0);
        assert_eq!(trainer.stats.games_played, 0);
    }

    #[test]
    fn test_single_iteration() {
        let network = UniformGeoNetwork::new();
        let config = GeoTrainingConfig {
            num_iterations: 1,
            games_per_iteration: 2,
            mcts_config: MCTSConfig {
                num_simulations: 5,
                ..MCTSConfig::fast()
            },
            ..Default::default()
        };

        let mut trainer = GeoSelfPlayTrainer::new(network, config);
        let result = trainer.run_iteration();

        assert_eq!(result.iteration, 1);
        assert_eq!(result.games_played, 2);
    }

    #[test]
    fn test_training_stats_update() {
        let mut stats = TrainingStats::default();

        let episode = ProofGameEpisode {
            initial_state: ProofState::new(),
            final_state: ProofState::new(),
            proved: true,
            total_reward: 1.0,
            num_steps: 10,
            num_constructions: 2,
            actions: vec![],
            trajectory: vec![],
            goal_confidence: Some(BetaConfidence::new(90.0, 10.0)),
        };

        stats.record_episode(&episode);

        assert_eq!(stats.games_played, 1);
        assert_eq!(stats.proofs_found, 1);
        assert!((stats.avg_proof_steps - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_convenience_functions() {
        let episode = test_midpoint_theorem();
        assert!(episode.num_steps > 0);

        let episode2 = test_triangle_congruence();
        assert!(episode2.num_steps > 0);
    }
}

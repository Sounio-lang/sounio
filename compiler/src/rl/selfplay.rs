//! Self-Play Training Loop
//!
//! AlphaZero-style self-play with epistemic enhancements.
//!
//! # Key Features
//!
//! 1. **Epistemic trajectories**: Track uncertainty through games
//! 2. **Variance penalty**: High uncertainty = higher loss
//! 3. **Provenance tracking**: Full replay explainability
//! 4. **Effect integration**: Self-play as composable effect

use std::collections::HashMap;
use std::hash::Hash;

use crate::epistemic::bayesian::BetaConfidence;
use crate::epistemic::merkle::{MerkleProvenanceDAG, OperationKind, ProvenanceOperation};

use super::game::{GameOutcome, GameTrait, Player};
use super::mcts::{MCTSConfig, MCTSStats, MCTSTree, search};
use super::neural::{NeuralConfig, NeuralEval, NeuralNetwork, TrainingExample};

// =============================================================================
// Configuration
// =============================================================================

/// Self-play configuration
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    /// MCTS config
    pub mcts: MCTSConfig,
    /// Neural config
    pub neural: NeuralConfig,
    /// Number of games per iteration
    pub games_per_iteration: usize,
    /// Temperature schedule: (move_number, temperature)
    pub temperature_schedule: Vec<(usize, f64)>,
    /// Whether to add Dirichlet noise at root
    pub add_noise: bool,
    /// Resign threshold (value below which to resign)
    pub resign_threshold: Option<f64>,
    /// Maximum game length
    pub max_game_length: usize,
    /// Minimum uncertainty to record example (skip very certain positions)
    pub min_uncertainty: f64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            mcts: MCTSConfig::default(),
            neural: NeuralConfig::default(),
            games_per_iteration: 100,
            temperature_schedule: vec![
                (0, 1.0),  // Exploration early
                (30, 0.1), // Exploit later
            ],
            add_noise: true,
            resign_threshold: Some(0.05),
            max_game_length: 500,
            min_uncertainty: 0.0,
        }
    }
}

impl SelfPlayConfig {
    /// Get temperature for a given move number
    pub fn temperature_for_move(&self, move_num: usize) -> f64 {
        let mut temp = 1.0;
        for &(threshold, t) in &self.temperature_schedule {
            if move_num >= threshold {
                temp = t;
            }
        }
        temp
    }
}

// =============================================================================
// Trajectory
// =============================================================================

/// A single step in a trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryStep<G: GameTrait> {
    /// State before action
    pub state: G,
    /// Action taken
    pub action: G::Action,
    /// Policy from MCTS (visit distribution)
    pub policy: HashMap<G::Action, f64>,
    /// Value estimate before action
    pub value: BetaConfidence,
    /// MCTS statistics
    pub mcts_stats: MCTSStats,
    /// Epistemic uncertainty at this step
    pub uncertainty: f64,
    /// Move number
    pub move_num: usize,
}

/// A complete game trajectory
#[derive(Debug, Clone)]
pub struct Trajectory<G: GameTrait> {
    /// Steps in the game
    pub steps: Vec<TrajectoryStep<G>>,
    /// Final outcome
    pub outcome: GameOutcome,
    /// Total moves played
    pub length: usize,
    /// Average uncertainty across trajectory
    pub avg_uncertainty: f64,
    /// Provenance DAG
    pub provenance: MerkleProvenanceDAG,
    /// Whether game was resigned
    pub resigned: bool,
    /// Player who resigned (if any)
    pub resigned_by: Option<Player>,
}

impl<G: GameTrait> Trajectory<G> {
    /// Create new empty trajectory
    pub fn new() -> Self {
        let mut provenance = MerkleProvenanceDAG::new();
        provenance.add_root(ProvenanceOperation::new(
            "game_start",
            OperationKind::Computation,
        ));

        Trajectory {
            steps: Vec::new(),
            outcome: GameOutcome::InProgress,
            length: 0,
            avg_uncertainty: 0.0,
            provenance,
            resigned: false,
            resigned_by: None,
        }
    }

    /// Add a step
    pub fn add_step(&mut self, step: TrajectoryStep<G>) {
        self.length = step.move_num + 1;
        self.steps.push(step);

        // Update average uncertainty
        if !self.steps.is_empty() {
            self.avg_uncertainty =
                self.steps.iter().map(|s| s.uncertainty).sum::<f64>() / self.steps.len() as f64;
        }
    }

    /// Finalize with outcome
    pub fn finalize(&mut self, outcome: GameOutcome) {
        self.outcome = outcome;
    }

    /// Convert to training examples
    pub fn to_training_examples(&self) -> Vec<TrainingExample<G>>
    where
        G: Clone,
    {
        let outcome_value = match self.outcome {
            GameOutcome::Win(Player::One) => 1.0,
            GameOutcome::Win(Player::Two) => -1.0,
            GameOutcome::Draw => 0.0,
            GameOutcome::InProgress => 0.0,
        };

        self.steps
            .iter()
            .map(|step| {
                // Create policy target vector
                let mut policy_target = vec![0.0; G::num_actions()];
                for (action, &prob) in &step.policy {
                    let idx = G::action_to_index(action);
                    policy_target[idx] = prob as f32;
                }

                // Value from perspective of player to move
                let value_target = if step.state.current_player() == Player::One {
                    outcome_value
                } else {
                    -outcome_value
                };

                TrainingExample {
                    state: step.state.clone(),
                    policy_target,
                    value_target: value_target as f32,
                    weight: 1.0,
                    policy_variance: None, // Could add from MCTS epistemic
                    value_variance: Some(step.uncertainty as f32),
                }
            })
            .collect()
    }
}

impl<G: GameTrait> Default for Trajectory<G> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Self-Play Engine
// =============================================================================

/// Neural evaluator adapter for MCTS
pub struct NeuralEvaluator<'a, G: GameTrait, N: NeuralNetwork<G>> {
    network: &'a N,
    _phantom: std::marker::PhantomData<G>,
}

impl<'a, G: GameTrait, N: NeuralNetwork<G>> NeuralEvaluator<'a, G, N> {
    pub fn new(network: &'a N) -> Self {
        NeuralEvaluator {
            network,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, G, N> super::mcts::NeuralEvaluator<G> for NeuralEvaluator<'a, G, N>
where
    G: GameTrait + Clone,
    G::Action: Clone + Eq + Hash,
    N: NeuralNetwork<G>,
{
    fn evaluate(&self, state: &G) -> (HashMap<G::Action, f64>, f64) {
        let pv = self.network.forward(state);
        let eval: NeuralEval<G::Action> = pv.to_eval(state, self.network.version());
        (eval.policy, eval.value)
    }
}

/// Play a single self-play game
pub fn play_game<G, N>(network: &N, config: &SelfPlayConfig) -> Trajectory<G>
where
    G: GameTrait + Clone,
    G::Action: Clone + Eq + Hash,
    N: NeuralNetwork<G>,
{
    let mut trajectory = Trajectory::new();
    let mut state = G::initial();
    let mut move_num = 0;

    let evaluator = NeuralEvaluator::new(network);

    while !state.is_terminal() && move_num < config.max_game_length {
        // Create MCTS tree
        let mut mcts_config = config.mcts.clone();
        mcts_config.temperature = config.temperature_for_move(move_num);

        let mut tree = MCTSTree::new(state.clone(), mcts_config);

        // Run search
        let result = search(&mut tree, &evaluator);

        // Check for resignation
        if let Some(threshold) = config.resign_threshold
            && result.root_value.mean() < threshold
        {
            trajectory.resigned = true;
            trajectory.resigned_by = Some(state.current_player());
            trajectory.finalize(GameOutcome::Win(state.current_player().opponent()));
            return trajectory;
        }

        // Select action
        let action = if let Some(action) = result.best_action.clone() {
            action
        } else {
            break;
        };

        // Record step
        let step = TrajectoryStep {
            state: state.clone(),
            action: action.clone(),
            policy: result.action_probabilities.clone(),
            value: result.root_value,
            mcts_stats: result.stats.clone(),
            uncertainty: result.global_uncertainty,
            move_num,
        };
        trajectory.add_step(step);

        // Apply action
        state = state.apply_action(&action);
        move_num += 1;
    }

    // Finalize
    trajectory.finalize(state.outcome());
    trajectory
}

/// Generate multiple self-play games
pub fn generate_games<G, N>(
    network: &N,
    config: &SelfPlayConfig,
    num_games: usize,
) -> Vec<Trajectory<G>>
where
    G: GameTrait + Clone,
    G::Action: Clone + Eq + Hash,
    N: NeuralNetwork<G>,
{
    (0..num_games).map(|_| play_game(network, config)).collect()
}

/// Statistics from self-play
#[derive(Debug, Clone, Default)]
pub struct SelfPlayStats {
    pub games_played: usize,
    pub total_moves: usize,
    pub avg_game_length: f64,
    pub p1_wins: usize,
    pub p2_wins: usize,
    pub draws: usize,
    pub resignations: usize,
    pub avg_uncertainty: f64,
}

impl SelfPlayStats {
    /// Compute from trajectories
    pub fn from_trajectories<G: GameTrait>(trajectories: &[Trajectory<G>]) -> Self {
        let mut stats = SelfPlayStats::default();

        for traj in trajectories {
            stats.games_played += 1;
            stats.total_moves += traj.length;

            match traj.outcome {
                GameOutcome::Win(Player::One) => stats.p1_wins += 1,
                GameOutcome::Win(Player::Two) => stats.p2_wins += 1,
                GameOutcome::Draw => stats.draws += 1,
                _ => {}
            }

            if traj.resigned {
                stats.resignations += 1;
            }

            stats.avg_uncertainty += traj.avg_uncertainty;
        }

        if stats.games_played > 0 {
            stats.avg_game_length = stats.total_moves as f64 / stats.games_played as f64;
            stats.avg_uncertainty /= stats.games_played as f64;
        }

        stats
    }
}

// =============================================================================
// Training Loop (Placeholder)
// =============================================================================

/// Training iteration result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub total_loss: f64,
    pub examples_trained: usize,
}

/// Run one training iteration (placeholder - would integrate with actual optimizer)
pub fn train_iteration<G: GameTrait + Clone>(
    _examples: &[TrainingExample<G>],
    config: &NeuralConfig,
) -> TrainingResult {
    // Placeholder - actual implementation would:
    // 1. Batch examples
    // 2. Forward pass
    // 3. Compute loss
    // 4. Backward pass
    // 5. Update weights

    TrainingResult {
        policy_loss: 0.0,
        value_loss: 0.0,
        total_loss: 0.0,
        examples_trained: 0,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::game::TicTacToeState;
    use crate::rl::neural::UniformNetwork;

    #[test]
    fn test_trajectory_creation() {
        let traj = Trajectory::<TicTacToeState>::new();
        assert_eq!(traj.length, 0);
        assert!(traj.steps.is_empty());
    }

    #[test]
    fn test_temperature_schedule() {
        let config = SelfPlayConfig::default();

        assert!((config.temperature_for_move(0) - 1.0).abs() < 0.01);
        assert!((config.temperature_for_move(35) - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_play_game() {
        let network = UniformNetwork::new::<TicTacToeState>();
        let config = SelfPlayConfig {
            mcts: MCTSConfig {
                num_simulations: 10,
                ..MCTSConfig::fast()
            },
            ..Default::default()
        };

        let trajectory = play_game::<TicTacToeState, _>(&network, &config);

        // Game should complete
        assert!(trajectory.length > 0);
        assert!(!matches!(trajectory.outcome, GameOutcome::InProgress));
    }

    #[test]
    fn test_to_training_examples() {
        let network = UniformNetwork::new::<TicTacToeState>();
        let config = SelfPlayConfig {
            mcts: MCTSConfig {
                num_simulations: 10,
                ..MCTSConfig::fast()
            },
            ..Default::default()
        };

        let trajectory = play_game::<TicTacToeState, _>(&network, &config);
        let examples = trajectory.to_training_examples();

        assert_eq!(examples.len(), trajectory.length);

        for example in &examples {
            assert_eq!(example.policy_target.len(), 9);
            assert!(example.value_target >= -1.0 && example.value_target <= 1.0);
        }
    }

    #[test]
    fn test_selfplay_stats() {
        let network = UniformNetwork::new::<TicTacToeState>();
        let config = SelfPlayConfig {
            mcts: MCTSConfig {
                num_simulations: 5,
                ..MCTSConfig::fast()
            },
            ..Default::default()
        };

        let trajectories = generate_games::<TicTacToeState, _>(&network, &config, 3);
        let stats = SelfPlayStats::from_trajectories(&trajectories);

        assert_eq!(stats.games_played, 3);
        assert!(stats.p1_wins + stats.p2_wins + stats.draws == 3);
    }
}

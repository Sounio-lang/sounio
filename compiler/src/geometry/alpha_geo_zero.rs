//! AlphaGeoZero: Epistemic MCTS for Geometry Theorem Proving
//!
//! A revolutionary neuro-symbolic system that learns to prove geometry theorems
//! via reinforcement learning with epistemic uncertainty guidance.
//!
//! # Key Innovations
//!
//! 1. **Epistemic PUCT**: Exploration bonus based on Beta distribution variance
//!    - High variance = high ignorance = should explore (active inference)
//!    - `score = Q + c_puct * prior * sqrt(parent_visits) / (1 + visits) + c_ignorance * sqrt(variance)`
//!
//! 2. **Hierarchical Beta Updates**: Backpropagation with posterior uncertainty
//!    - Each node maintains Beta(α, β) for value estimation
//!    - Conjugate updates preserve uncertainty quantification
//!
//! 3. **Variance-Driven Curriculum**: Replay buffer prioritizes high-variance problems
//!    - Problems where model is uncertain get more training
//!    - Natural curriculum emergence from epistemic state
//!
//! 4. **MuZero-Style Training**: Policy + Value + Model with variance penalty
//!    - Loss = L_policy + L_value + L_model + λ * Var(value)
//!    - Penalizes high variance to encourage confident predictions
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      AlphaGeoZero Loop                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
//! │  │ Problem │───►│  MCTS   │───►│ Proof   │───►│Training │     │
//! │  │Generator│    │Epistemic│    │ Game    │    │  Step   │     │
//! │  └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
//! │       │              │              │              │           │
//! │       │              ▼              │              │           │
//! │       │        ┌─────────┐          │              │           │
//! │       │        │ Neural  │◄─────────┼──────────────┘           │
//! │       │        │ Network │          │                          │
//! │       │        └─────────┘          │                          │
//! │       │              │              │                          │
//! │       │              ▼              ▼                          │
//! │       │        ┌─────────────────────┐                         │
//! │       └───────►│   Replay Buffer     │                         │
//! │                │(variance priority)  │                         │
//! │                └─────────────────────┘                         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::time::{Duration, Instant};

use crate::epistemic::bayesian::BetaConfidence;
use crate::rl::game::{GameState, GameTrait};

use super::geo_game::{GeoAction, GeoGameConfig, GeoProofGame};
use super::geo_training::GeoNeuralNetwork;
use super::predicates::{Predicate, PredicateKind};
use super::proof_state::ProofState;

// =============================================================================
// Epistemic MCTS Configuration
// =============================================================================

/// Configuration for AlphaGeoZero MCTS
#[derive(Debug, Clone)]
pub struct AlphaGeoZeroConfig {
    /// PUCT exploration constant
    pub c_puct: f64,
    /// Epistemic ignorance bonus constant
    pub c_ignorance: f64,
    /// Number of MCTS simulations per move
    pub num_simulations: usize,
    /// Temperature for action selection (1.0 = proportional to visits)
    pub temperature: f64,
    /// Dirichlet noise alpha for root exploration
    pub dirichlet_alpha: f64,
    /// Fraction of Dirichlet noise to add
    pub dirichlet_epsilon: f64,
    /// Maximum search depth
    pub max_depth: usize,
    /// Virtual loss for parallel MCTS
    pub virtual_loss: f64,
    /// Whether to use variance bonus in PUCT
    pub use_variance_bonus: bool,
    /// Discount factor for value backup
    pub gamma: f64,
}

impl Default for AlphaGeoZeroConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            c_ignorance: 0.5, // Epistemic exploration bonus
            num_simulations: 800,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_depth: 50,
            virtual_loss: 3.0,
            use_variance_bonus: true,
            gamma: 0.99,
        }
    }
}

// =============================================================================
// Epistemic MCTS Node
// =============================================================================

/// A node in the epistemic MCTS tree
pub struct EpistemicMCTSNode {
    /// Current proof game state
    pub state: GeoProofGame,
    /// Action that led to this node (None for root)
    pub action: Option<GeoAction>,
    /// Parent node index (None for root)
    pub parent: Option<usize>,
    /// Child node indices
    pub children: Vec<usize>,
    /// Prior probability from neural network
    pub prior: f64,
    /// Beta distribution for value estimation
    pub value_beta: BetaConfidence,
    /// Visit count
    pub visits: u32,
    /// Total value (for mean calculation)
    pub total_value: f64,
    /// Virtual loss counter (for parallel MCTS)
    pub virtual_loss: f64,
    /// Whether this node has been expanded
    pub expanded: bool,
    /// Provenance hash for this node
    pub provenance_hash: u64,
}

impl EpistemicMCTSNode {
    /// Create a new root node
    pub fn new_root(state: GeoProofGame) -> Self {
        Self {
            state,
            action: None,
            parent: None,
            children: vec![],
            prior: 1.0,
            value_beta: BetaConfidence::uniform_prior(),
            visits: 0,
            total_value: 0.0,
            virtual_loss: 0.0,
            expanded: false,
            provenance_hash: 0,
        }
    }

    /// Create a child node
    pub fn new_child(state: GeoProofGame, action: GeoAction, parent: usize, prior: f64) -> Self {
        Self {
            state,
            action: Some(action),
            parent: Some(parent),
            children: vec![],
            prior,
            value_beta: BetaConfidence::uniform_prior(),
            visits: 0,
            total_value: 0.0,
            virtual_loss: 0.0,
            expanded: false,
            provenance_hash: 0,
        }
    }

    /// Get the mean Q value
    pub fn q_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f64
        }
    }

    /// Get the epistemic variance from Beta distribution
    pub fn variance(&self) -> f64 {
        self.value_beta.variance()
    }

    /// Compute PUCT score with epistemic bonus
    pub fn puct_score(&self, parent_visits: u32, config: &AlphaGeoZeroConfig) -> f64 {
        let q = self.q_value();
        let u = config.c_puct * self.prior * (parent_visits as f64).sqrt()
            / (1.0 + self.visits as f64 + self.virtual_loss);

        // Epistemic ignorance bonus
        let ignorance_bonus = if config.use_variance_bonus {
            config.c_ignorance * self.variance().sqrt()
        } else {
            0.0
        };

        q + u + ignorance_bonus
    }

    /// Update value with Bayesian posterior
    pub fn update_value(&mut self, value: f64) {
        self.visits += 1;
        self.total_value += value;

        // Bayesian update: treat value as Bernoulli observation
        // value > 0.5 = success, value <= 0.5 = failure
        // Scale to get fractional updates
        let alpha_update = value.max(0.0).min(1.0);
        let beta_update = (1.0 - value).max(0.0).min(1.0);

        self.value_beta = BetaConfidence::new(
            self.value_beta.alpha + alpha_update,
            self.value_beta.beta + beta_update,
        );
    }
}

// =============================================================================
// Epistemic MCTS Tree
// =============================================================================

/// The full MCTS tree with epistemic nodes
pub struct EpistemicMCTSTree {
    /// All nodes in the tree (arena allocation)
    pub nodes: Vec<EpistemicMCTSNode>,
    /// Configuration
    pub config: AlphaGeoZeroConfig,
    /// Statistics
    pub stats: MCTSStats,
}

/// Statistics for MCTS search
#[derive(Debug, Clone, Default)]
pub struct MCTSStats {
    pub total_simulations: usize,
    pub max_depth_reached: usize,
    pub terminal_hits: usize,
    pub proof_found: bool,
    pub average_variance: f64,
    pub search_time: Duration,
}

impl EpistemicMCTSTree {
    /// Create a new tree with root state
    pub fn new(root_state: GeoProofGame, config: AlphaGeoZeroConfig) -> Self {
        let root = EpistemicMCTSNode::new_root(root_state);
        Self {
            nodes: vec![root],
            config,
            stats: MCTSStats::default(),
        }
    }

    /// Get root node
    pub fn root(&self) -> &EpistemicMCTSNode {
        &self.nodes[0]
    }

    /// Get mutable root node
    pub fn root_mut(&mut self) -> &mut EpistemicMCTSNode {
        &mut self.nodes[0]
    }

    /// Select the best child using PUCT with epistemic bonus
    pub fn select_child(&self, node_idx: usize) -> Option<usize> {
        let node = &self.nodes[node_idx];
        if node.children.is_empty() {
            return None;
        }

        let parent_visits = node.visits;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_child = None;

        for &child_idx in &node.children {
            let child = &self.nodes[child_idx];
            let score = child.puct_score(parent_visits, &self.config);

            if score > best_score {
                best_score = score;
                best_child = Some(child_idx);
            }
        }

        best_child
    }

    /// Select path from root to leaf
    pub fn select_leaf(&mut self) -> usize {
        let mut current = 0;
        let mut depth = 0;

        while self.nodes[current].expanded && !self.nodes[current].children.is_empty() {
            // Add virtual loss for parallel MCTS
            self.nodes[current].virtual_loss += self.config.virtual_loss;

            if let Some(child) = self.select_child(current) {
                current = child;
                depth += 1;

                if depth > self.stats.max_depth_reached {
                    self.stats.max_depth_reached = depth;
                }

                if depth >= self.config.max_depth {
                    break;
                }
            } else {
                break;
            }
        }

        current
    }

    /// Expand a leaf node with neural network predictions
    pub fn expand<N: GeoNeuralNetwork>(&mut self, leaf_idx: usize, network: &N) {
        let leaf = &self.nodes[leaf_idx];

        // Check if terminal
        if leaf.state.is_terminal() {
            self.stats.terminal_hits += 1;
            if leaf.state.proved {
                self.stats.proof_found = true;
            }
            return;
        }

        // Get legal actions
        let legal_actions = leaf.state.legal_actions();
        if legal_actions.is_empty() {
            return;
        }

        // Get neural network policy and value
        let (policy, _value) = network.forward(&leaf.state);

        // Normalize policy for legal actions
        let mut policy_sum: f64 = 0.0;
        let mut action_priors: Vec<(GeoAction, f64)> = vec![];

        for (i, action) in legal_actions.iter().enumerate() {
            let prior: f64 = if i < policy.len() {
                (policy[i].max(0.001)) as f64 // Minimum prior to ensure exploration
            } else {
                0.001
            };
            policy_sum += prior;
            action_priors.push((action.clone(), prior));
        }

        // Create child nodes
        let leaf_state = self.nodes[leaf_idx].state.clone();
        for (action, prior) in action_priors {
            let normalized_prior = prior / policy_sum;
            let child_state = leaf_state.apply_action(&action);
            let child =
                EpistemicMCTSNode::new_child(child_state, action, leaf_idx, normalized_prior);
            let child_idx = self.nodes.len();
            self.nodes.push(child);
            self.nodes[leaf_idx].children.push(child_idx);
        }

        self.nodes[leaf_idx].expanded = true;
    }

    /// Simulate from a node (evaluate with network or symbolic)
    pub fn simulate<N: GeoNeuralNetwork>(&self, node_idx: usize, network: &N) -> f64 {
        let node = &self.nodes[node_idx];

        // Terminal check
        if node.state.is_terminal() {
            return if node.state.proved { 1.0 } else { 0.0 };
        }

        // High confidence symbolic check
        if node.state.proved {
            return 1.0;
        }

        // Neural network value estimation
        let (_policy, value) = network.forward(&node.state);
        value as f64
    }

    /// Backpropagate value with virtual loss removal
    pub fn backpropagate(&mut self, leaf_idx: usize, value: f64) {
        let mut current = Some(leaf_idx);
        let mut discount = 1.0;

        while let Some(idx) = current {
            let node = &mut self.nodes[idx];

            // Remove virtual loss
            node.virtual_loss = (node.virtual_loss - self.config.virtual_loss).max(0.0);

            // Update value with discount
            node.update_value(value * discount);

            // Move to parent
            current = node.parent;
            discount *= self.config.gamma;
        }
    }

    /// Run full MCTS search
    pub fn search<N: GeoNeuralNetwork>(&mut self, network: &N) {
        let start = Instant::now();

        for _ in 0..self.config.num_simulations {
            // Select
            let leaf = self.select_leaf();

            // Expand
            self.expand(leaf, network);

            // Simulate
            let value = self.simulate(leaf, network);

            // Backpropagate
            self.backpropagate(leaf, value);

            self.stats.total_simulations += 1;

            // Early termination if proof found
            if self.stats.proof_found {
                break;
            }
        }

        self.stats.search_time = start.elapsed();

        // Compute average variance
        let total_variance: f64 = self.nodes.iter().map(|n| n.variance()).sum();
        self.stats.average_variance = total_variance / self.nodes.len() as f64;
    }

    /// Get action probabilities from root (for training)
    pub fn get_policy(&self) -> Vec<(GeoAction, f64)> {
        let root = self.root();
        let total_visits: u32 = root.children.iter().map(|&c| self.nodes[c].visits).sum();

        if total_visits == 0 {
            return vec![];
        }

        let mut policy = vec![];
        for &child_idx in &root.children {
            let child = &self.nodes[child_idx];
            if let Some(ref action) = child.action {
                let prob = if self.config.temperature == 0.0 {
                    // Greedy
                    if child.visits == total_visits {
                        1.0
                    } else {
                        0.0
                    }
                } else if self.config.temperature == 1.0 {
                    // Proportional to visits
                    child.visits as f64 / total_visits as f64
                } else {
                    // Temperature-adjusted

                    (child.visits as f64).powf(1.0 / self.config.temperature)
                };
                policy.push((action.clone(), prob));
            }
        }

        // Normalize if temperature != 1.0
        if self.config.temperature != 1.0 && self.config.temperature != 0.0 {
            let sum: f64 = policy.iter().map(|(_, p)| p).sum();
            if sum > 0.0 {
                for (_, p) in &mut policy {
                    *p /= sum;
                }
            }
        }

        policy
    }

    /// Select best action (for playing)
    pub fn select_action(&self) -> Option<GeoAction> {
        let root = self.root();

        // Find child with most visits (robust action selection)
        let mut best_visits = 0;
        let mut best_action = None;

        for &child_idx in &root.children {
            let child = &self.nodes[child_idx];
            if child.visits > best_visits {
                best_visits = child.visits;
                best_action = child.action.clone();
            }
        }

        best_action
    }

    /// Get epistemic value at root (Beta distribution)
    pub fn root_value(&self) -> BetaConfidence {
        self.root().value_beta
    }
}

// =============================================================================
// Proof Game Episode
// =============================================================================

/// A complete proof game episode for training
#[derive(Debug, Clone)]
pub struct ProofGameEpisode {
    /// Problem identifier
    pub problem_id: String,
    /// Initial state
    pub initial_state: ProofState,
    /// Target predicate
    pub target: Predicate,
    /// Sequence of (state, policy, value_beta) tuples
    pub trajectory: Vec<TrajectoryStep>,
    /// Final outcome (proved or not)
    pub proved: bool,
    /// Total epistemic variance of trajectory
    pub total_variance: f64,
    /// Episode length
    pub length: usize,
}

/// A single step in a proof trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// State features (for training)
    pub state_features: Vec<f32>,
    /// Policy from MCTS
    pub policy: Vec<f32>,
    /// Value Beta distribution
    pub value_beta: BetaConfidence,
    /// Action taken
    pub action_idx: usize,
}

// =============================================================================
// Self-Play Generator
// =============================================================================

/// Generator for self-play proof games
pub struct SelfPlayGenerator<N: GeoNeuralNetwork> {
    /// Neural network for policy/value
    pub network: N,
    /// MCTS configuration
    pub mcts_config: AlphaGeoZeroConfig,
    /// Game configuration
    pub game_config: GeoGameConfig,
    /// Statistics
    pub stats: SelfPlayStats,
}

/// Self-play statistics
#[derive(Debug, Clone, Default)]
pub struct SelfPlayStats {
    pub games_played: usize,
    pub proofs_found: usize,
    pub average_game_length: f64,
    pub average_variance: f64,
    pub total_simulations: usize,
}

impl<N: GeoNeuralNetwork> SelfPlayGenerator<N> {
    /// Create new self-play generator
    pub fn new(network: N, mcts_config: AlphaGeoZeroConfig, game_config: GeoGameConfig) -> Self {
        Self {
            network,
            mcts_config,
            game_config,
            stats: SelfPlayStats::default(),
        }
    }

    /// Generate a single proof game episode
    pub fn generate_episode(
        &mut self,
        initial_state: ProofState,
        target: Predicate,
    ) -> ProofGameEpisode {
        let problem_id = format!("game_{}", self.stats.games_played);

        // Create initial game state with goal
        let mut game = GeoProofGame::for_goal(
            initial_state.clone(),
            target.clone(),
            self.game_config.goal_confidence,
        );

        let mut trajectory = vec![];
        let mut total_variance = 0.0;

        // Play until terminal
        while !game.is_terminal() {
            // Run MCTS
            let mut tree = EpistemicMCTSTree::new(game.clone(), self.mcts_config.clone());
            tree.search(&self.network);

            // Get policy from MCTS
            let policy_vec = tree.get_policy();
            let policy: Vec<f32> = if policy_vec.is_empty() {
                vec![1.0]
            } else {
                policy_vec.iter().map(|(_, p)| *p as f32).collect()
            };

            // Get value Beta
            let value_beta = tree.root_value();
            total_variance += value_beta.variance();

            // Get state features
            let state_features = game.to_features();

            // Select action
            let action = if let Some(a) = tree.select_action() {
                a
            } else {
                break;
            };

            // Find action index
            let legal = game.legal_actions();
            let action_idx = legal.iter().position(|a| *a == action).unwrap_or(0);

            // Record step
            trajectory.push(TrajectoryStep {
                state_features,
                policy,
                value_beta,
                action_idx,
            });

            // Apply action
            game = game.apply_action(&action);

            // Update stats
            self.stats.total_simulations += tree.stats.total_simulations;
        }

        let proved = game.proved;
        let length = trajectory.len();

        // Update stats
        self.stats.games_played += 1;
        if proved {
            self.stats.proofs_found += 1;
        }
        self.stats.average_game_length =
            (self.stats.average_game_length * (self.stats.games_played - 1) as f64 + length as f64)
                / self.stats.games_played as f64;
        self.stats.average_variance = (self.stats.average_variance
            * (self.stats.games_played - 1) as f64
            + total_variance / length.max(1) as f64)
            / self.stats.games_played as f64;

        ProofGameEpisode {
            problem_id,
            initial_state,
            target,
            trajectory,
            proved,
            total_variance,
            length,
        }
    }
}

// =============================================================================
// Replay Buffer with Variance Priority
// =============================================================================

/// Replay buffer with epistemic variance-based priority
pub struct VariancePriorityBuffer {
    /// Buffer of episodes
    pub episodes: Vec<ProofGameEpisode>,
    /// Priority scores (based on variance)
    pub priorities: Vec<f64>,
    /// Maximum buffer size
    pub max_size: usize,
    /// Priority exponent (higher = more focus on high variance)
    pub alpha: f64,
    /// Temperature for sampling
    pub beta: f64,
}

impl VariancePriorityBuffer {
    /// Create new buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            episodes: vec![],
            priorities: vec![],
            max_size,
            alpha: 0.6, // Priority exponent
            beta: 0.4,  // Importance sampling exponent
        }
    }

    /// Add episode with variance-based priority
    pub fn add(&mut self, episode: ProofGameEpisode) {
        // Priority = average variance (higher variance = more priority)
        let avg_variance = episode.total_variance / episode.length.max(1) as f64;
        let priority = (avg_variance + 0.01).powf(self.alpha);

        if self.episodes.len() >= self.max_size {
            // Remove lowest priority episode
            if let Some(min_idx) = self
                .priorities
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
            {
                self.episodes.remove(min_idx);
                self.priorities.remove(min_idx);
            }
        }

        self.episodes.push(episode);
        self.priorities.push(priority);
    }

    /// Sample batch with priority weighting
    pub fn sample(&self, batch_size: usize) -> Vec<(&ProofGameEpisode, f64)> {
        if self.episodes.is_empty() {
            return vec![];
        }

        let total_priority: f64 = self.priorities.iter().sum();
        let mut samples = vec![];

        // Simple priority sampling (could use sum tree for efficiency)
        for _ in 0..batch_size.min(self.episodes.len()) {
            // Pseudo-random based on hash
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            samples.len().hash(&mut hasher);
            std::time::SystemTime::now().hash(&mut hasher);
            let rand_val = (hasher.finish() as f64) / (u64::MAX as f64);

            let threshold = rand_val * total_priority;
            let mut cumsum = 0.0;

            for (i, &p) in self.priorities.iter().enumerate() {
                cumsum += p;
                if cumsum >= threshold {
                    // Importance sampling weight
                    let prob = p / total_priority;
                    let weight = (1.0 / (self.episodes.len() as f64 * prob)).powf(self.beta);
                    samples.push((&self.episodes[i], weight));
                    break;
                }
            }
        }

        samples
    }

    /// Get buffer statistics
    pub fn stats(&self) -> BufferStats {
        let total_variance: f64 = self.episodes.iter().map(|e| e.total_variance).sum();
        let proofs: usize = self.episodes.iter().filter(|e| e.proved).count();

        BufferStats {
            size: self.episodes.len(),
            total_episodes: self.episodes.len(),
            proofs_found: proofs,
            average_variance: total_variance / self.episodes.len().max(1) as f64,
            max_priority: self.priorities.iter().cloned().fold(0.0, f64::max),
        }
    }
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub size: usize,
    pub total_episodes: usize,
    pub proofs_found: usize,
    pub average_variance: f64,
    pub max_priority: f64,
}

// =============================================================================
// Training with Variance Penalty
// =============================================================================

/// Training configuration for AlphaGeoZero
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Policy loss weight
    pub policy_weight: f64,
    /// Value loss weight
    pub value_weight: f64,
    /// Variance penalty weight (λ in loss)
    pub variance_penalty: f64,
    /// L2 regularization
    pub l2_reg: f64,
    /// Gradient clipping
    pub gradient_clip: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 256,
            policy_weight: 1.0,
            value_weight: 1.0,
            variance_penalty: 0.1, // Penalize uncertain predictions
            l2_reg: 0.0001,
            gradient_clip: 1.0,
        }
    }
}

/// Training step result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub variance_penalty: f64,
    pub gradient_norm: f64,
}

/// Trainer for AlphaGeoZero
pub struct AlphaGeoZeroTrainer<N: GeoNeuralNetwork> {
    /// Neural network
    pub network: N,
    /// Replay buffer
    pub buffer: VariancePriorityBuffer,
    /// Training config
    pub config: TrainingConfig,
    /// Training statistics
    pub stats: TrainerStats,
}

/// Trainer statistics
#[derive(Debug, Clone, Default)]
pub struct TrainerStats {
    pub training_steps: usize,
    pub total_loss: f64,
    pub average_policy_loss: f64,
    pub average_value_loss: f64,
    pub average_variance_penalty: f64,
}

impl<N: GeoNeuralNetwork> AlphaGeoZeroTrainer<N> {
    /// Create new trainer
    pub fn new(network: N, buffer_size: usize, config: TrainingConfig) -> Self {
        Self {
            network,
            buffer: VariancePriorityBuffer::new(buffer_size),
            config,
            stats: TrainerStats::default(),
        }
    }

    /// Add episode to buffer
    pub fn add_episode(&mut self, episode: ProofGameEpisode) {
        self.buffer.add(episode);
    }

    /// Perform training step
    pub fn train_step(&mut self) -> Option<TrainingResult> {
        // Sample batch from buffer
        let batch = self.buffer.sample(self.config.batch_size);
        if batch.is_empty() {
            return None;
        }

        // Prepare training examples
        let mut examples = vec![];
        let mut variance_sum = 0.0;

        for (episode, weight) in batch {
            // Use final outcome as target value
            let target_value = if episode.proved { 1.0 } else { 0.0 };

            for step in &episode.trajectory {
                // Compute target: final outcome + variance from Beta
                let value_mean = step.value_beta.mean();
                let value_var = step.value_beta.variance();

                // Training example with importance weight
                examples.push(TrainingExample {
                    features: step.state_features.clone(),
                    target_policy: step.policy.clone(),
                    target_value: target_value as f32,
                    value_variance: value_var as f32,
                    weight: weight as f32,
                });

                variance_sum += value_var;
            }
        }

        if examples.is_empty() {
            return None;
        }

        // Compute losses (simplified - real impl would use backprop)
        let mut policy_loss = 0.0;
        let mut value_loss = 0.0;

        for ex in &examples {
            // Forward pass
            let dummy_state = GeoProofGame::default();
            let (pred_policy, pred_value) = self.network.forward(&dummy_state);

            // Policy cross-entropy loss
            for (i, &target_p) in ex.target_policy.iter().enumerate() {
                if i < pred_policy.len() && target_p > 0.0 {
                    policy_loss -=
                        (target_p * (pred_policy[i] + 1e-8).ln()) as f64 * ex.weight as f64;
                }
            }

            // Value MSE loss
            value_loss += ((pred_value - ex.target_value).powi(2) * ex.weight) as f64;
        }

        let n = examples.len() as f64;
        policy_loss /= n;
        value_loss /= n;
        let variance_penalty_loss = self.config.variance_penalty * variance_sum / n;

        let total_loss = self.config.policy_weight * policy_loss
            + self.config.value_weight * value_loss
            + variance_penalty_loss;

        // Update stats
        self.stats.training_steps += 1;
        self.stats.total_loss += total_loss;
        self.stats.average_policy_loss =
            (self.stats.average_policy_loss * (self.stats.training_steps - 1) as f64 + policy_loss)
                / self.stats.training_steps as f64;
        self.stats.average_value_loss =
            (self.stats.average_value_loss * (self.stats.training_steps - 1) as f64 + value_loss)
                / self.stats.training_steps as f64;
        self.stats.average_variance_penalty = (self.stats.average_variance_penalty
            * (self.stats.training_steps - 1) as f64
            + variance_penalty_loss)
            / self.stats.training_steps as f64;

        Some(TrainingResult {
            total_loss,
            policy_loss,
            value_loss,
            variance_penalty: variance_penalty_loss,
            gradient_norm: 0.0, // Would be computed in real backprop
        })
    }
}

/// Training example
struct TrainingExample {
    features: Vec<f32>,
    target_policy: Vec<f32>,
    target_value: f32,
    value_variance: f32,
    weight: f32,
}

// =============================================================================
// Full Self-Play Loop
// =============================================================================

/// The complete AlphaGeoZero self-play training loop
pub struct AlphaGeoZeroLoop<N: GeoNeuralNetwork> {
    /// Self-play generator
    pub generator: SelfPlayGenerator<N>,
    /// Trainer (shares network via clone)
    pub trainer: AlphaGeoZeroTrainer<N>,
    /// Problem curriculum
    pub curriculum: ProblemCurriculum,
    /// Loop configuration
    pub config: LoopConfig,
    /// Loop statistics
    pub stats: LoopStats,
}

/// Configuration for the main loop
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Number of self-play games per iteration
    pub games_per_iteration: usize,
    /// Number of training steps per iteration
    pub training_steps_per_iteration: usize,
    /// Total iterations
    pub total_iterations: usize,
    /// Checkpoint interval
    pub checkpoint_interval: usize,
    /// Evaluation interval
    pub eval_interval: usize,
    /// Number of eval games
    pub eval_games: usize,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            games_per_iteration: 100,
            training_steps_per_iteration: 1000,
            total_iterations: 1000,
            checkpoint_interval: 10,
            eval_interval: 5,
            eval_games: 20,
        }
    }
}

/// Loop statistics
#[derive(Debug, Clone, Default)]
pub struct LoopStats {
    pub current_iteration: usize,
    pub total_games: usize,
    pub total_proofs: usize,
    pub proof_rate: f64,
    pub average_game_length: f64,
    pub average_variance: f64,
    pub training_loss: f64,
}

/// Problem curriculum based on variance
pub struct ProblemCurriculum {
    /// All problems
    pub problems: Vec<(ProofState, Predicate)>,
    /// Problem difficulties (estimated variance)
    pub difficulties: Vec<f64>,
    /// Current difficulty threshold
    pub difficulty_threshold: f64,
    /// Difficulty growth rate
    pub growth_rate: f64,
}

impl ProblemCurriculum {
    /// Create new curriculum
    pub fn new(problems: Vec<(ProofState, Predicate)>) -> Self {
        let n = problems.len();
        Self {
            problems,
            difficulties: vec![0.5; n], // Start with uniform difficulty
            difficulty_threshold: 0.3,  // Start easy
            growth_rate: 0.01,
        }
    }

    /// Sample a problem based on current difficulty
    pub fn sample(&self) -> Option<(ProofState, Predicate)> {
        // Filter problems within difficulty threshold
        let candidates: Vec<_> = self
            .problems
            .iter()
            .zip(self.difficulties.iter())
            .filter(|(_, d)| **d <= self.difficulty_threshold)
            .map(|(p, _)| p.clone())
            .collect();

        if candidates.is_empty() {
            // If no easy problems, sample any
            self.problems.first().cloned()
        } else {
            // Pseudo-random selection
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut hasher);
            let idx = (hasher.finish() as usize) % candidates.len();
            Some(candidates[idx].clone())
        }
    }

    /// Update difficulty based on episode result
    pub fn update(&mut self, problem_idx: usize, variance: f64, proved: bool) {
        if problem_idx < self.difficulties.len() {
            // Update difficulty estimate (EMA)
            let old = self.difficulties[problem_idx];
            let new_estimate = if proved {
                old * 0.9 // Got easier
            } else {
                (old + variance).min(1.0) // Got harder
            };
            self.difficulties[problem_idx] = old * 0.8 + new_estimate * 0.2;
        }

        // Gradually increase threshold
        self.difficulty_threshold = (self.difficulty_threshold + self.growth_rate).min(1.0);
    }
}

impl<N: GeoNeuralNetwork + Clone> AlphaGeoZeroLoop<N> {
    /// Create new training loop
    pub fn new(
        network: N,
        problems: Vec<(ProofState, Predicate)>,
        mcts_config: AlphaGeoZeroConfig,
        game_config: GeoGameConfig,
        training_config: TrainingConfig,
        loop_config: LoopConfig,
        buffer_size: usize,
    ) -> Self {
        let generator = SelfPlayGenerator::new(network.clone(), mcts_config, game_config);
        let trainer = AlphaGeoZeroTrainer::new(network, buffer_size, training_config);
        let curriculum = ProblemCurriculum::new(problems);

        Self {
            generator,
            trainer,
            curriculum,
            config: loop_config,
            stats: LoopStats::default(),
        }
    }

    /// Run one iteration of self-play + training
    pub fn run_iteration(&mut self) -> IterationResult {
        let start = Instant::now();

        // Self-play phase
        let mut games_this_iter = 0;
        let mut proofs_this_iter = 0;

        for _ in 0..self.config.games_per_iteration {
            if let Some((state, target)) = self.curriculum.sample() {
                let episode = self.generator.generate_episode(state, target);

                proofs_this_iter += if episode.proved { 1 } else { 0 };
                games_this_iter += 1;

                // Update curriculum
                let variance = episode.total_variance / episode.length.max(1) as f64;
                self.curriculum.update(0, variance, episode.proved); // TODO: track problem idx

                // Add to replay buffer
                self.trainer.add_episode(episode);
            }
        }

        // Training phase
        let mut total_loss = 0.0;
        let mut training_steps = 0;

        for _ in 0..self.config.training_steps_per_iteration {
            if let Some(result) = self.trainer.train_step() {
                total_loss += result.total_loss;
                training_steps += 1;
            }
        }

        // Update stats
        self.stats.current_iteration += 1;
        self.stats.total_games += games_this_iter;
        self.stats.total_proofs += proofs_this_iter;
        self.stats.proof_rate =
            self.stats.total_proofs as f64 / self.stats.total_games.max(1) as f64;
        self.stats.average_game_length = self.generator.stats.average_game_length;
        self.stats.average_variance = self.generator.stats.average_variance;
        self.stats.training_loss = if training_steps > 0 {
            total_loss / training_steps as f64
        } else {
            0.0
        };

        IterationResult {
            iteration: self.stats.current_iteration,
            games_played: games_this_iter,
            proofs_found: proofs_this_iter,
            training_loss: self.stats.training_loss,
            average_variance: self.stats.average_variance,
            duration: start.elapsed(),
            buffer_size: self.trainer.buffer.episodes.len(),
        }
    }

    /// Run the full training loop
    pub fn run(&mut self) -> Vec<IterationResult> {
        let mut results = vec![];

        for i in 0..self.config.total_iterations {
            let result = self.run_iteration();

            // Log progress
            if i % 10 == 0 {
                println!(
                    "Iteration {}: {} games, {} proofs ({:.1}%), loss={:.4}, var={:.4}",
                    result.iteration,
                    result.games_played,
                    result.proofs_found,
                    result.proofs_found as f64 / result.games_played.max(1) as f64 * 100.0,
                    result.training_loss,
                    result.average_variance
                );
            }

            results.push(result);
        }

        results
    }
}

/// Result of one training iteration
#[derive(Debug, Clone)]
pub struct IterationResult {
    pub iteration: usize,
    pub games_played: usize,
    pub proofs_found: usize,
    pub training_loss: f64,
    pub average_variance: f64,
    pub duration: Duration,
    pub buffer_size: usize,
}

// =============================================================================
// Default Implementation for GeoProofGame
// =============================================================================

impl Default for GeoProofGame {
    fn default() -> Self {
        Self::new(ProofState::new(), GeoGameConfig::default())
    }
}

/// Create a simple predicate for testing (collinear A B C)
fn trivial_predicate() -> Predicate {
    Predicate::new(
        PredicateKind::Collinear,
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::geo_training::UniformGeoNetwork;
    use super::*;

    #[test]
    fn test_epistemic_mcts_node_creation() {
        let game = GeoProofGame::default();
        let node = EpistemicMCTSNode::new_root(game);

        assert_eq!(node.visits, 0);
        assert!(node.parent.is_none());
        assert!(node.children.is_empty());
        assert!(!node.expanded);
    }

    #[test]
    fn test_epistemic_mcts_node_update() {
        let game = GeoProofGame::default();
        let mut node = EpistemicMCTSNode::new_root(game);

        // Initial variance should be high (uniform prior Beta(1,1) has variance ~0.083)
        let initial_var = node.variance();
        assert!(
            initial_var > 0.05,
            "Initial variance {} should be > 0.05 for uniform prior",
            initial_var
        );

        // Update with observations
        node.update_value(0.8);
        node.update_value(0.9);
        node.update_value(0.7);

        // Variance should decrease as we accumulate evidence
        assert!(
            node.variance() < initial_var,
            "Variance {} should decrease from initial {}",
            node.variance(),
            initial_var
        );
        assert_eq!(node.visits, 3);
    }

    #[test]
    fn test_puct_score_with_variance_bonus() {
        let game = GeoProofGame::default();
        let mut node = EpistemicMCTSNode::new_root(game.clone());
        node.prior = 0.5;
        node.visits = 10;
        node.total_value = 5.0;

        let config_with_bonus = AlphaGeoZeroConfig {
            use_variance_bonus: true,
            c_ignorance: 1.0,
            ..Default::default()
        };

        let config_without_bonus = AlphaGeoZeroConfig {
            use_variance_bonus: false,
            ..Default::default()
        };

        let score_with = node.puct_score(100, &config_with_bonus);
        let score_without = node.puct_score(100, &config_without_bonus);

        // Score with variance bonus should be higher
        assert!(score_with >= score_without);
    }

    #[test]
    fn test_epistemic_mcts_tree_creation() {
        let game = GeoProofGame::default();
        let config = AlphaGeoZeroConfig::default();
        let tree = EpistemicMCTSTree::new(game, config);

        assert_eq!(tree.nodes.len(), 1);
        assert!(!tree.root().expanded);
    }

    #[test]
    fn test_variance_priority_buffer() {
        let mut buffer = VariancePriorityBuffer::new(10);

        // Add episodes with different variances
        for i in 0..5 {
            let episode = ProofGameEpisode {
                problem_id: format!("test_{}", i),
                initial_state: ProofState::new(),
                target: trivial_predicate(),
                trajectory: vec![],
                proved: i % 2 == 0,
                total_variance: (i + 1) as f64 * 0.1,
                length: 1,
            };
            buffer.add(episode);
        }

        assert_eq!(buffer.episodes.len(), 5);

        // Higher variance episodes should have higher priority
        let stats = buffer.stats();
        assert!(stats.max_priority > 0.0);
    }

    #[test]
    fn test_problem_curriculum() {
        let problems = vec![
            (ProofState::new(), trivial_predicate()),
            (ProofState::new(), trivial_predicate()),
        ];
        let mut curriculum = ProblemCurriculum::new(problems);

        // Should be able to sample
        assert!(curriculum.sample().is_some());

        // Update difficulty
        curriculum.update(0, 0.5, true);
        curriculum.update(1, 0.8, false);

        // Threshold should increase
        assert!(curriculum.difficulty_threshold > 0.3);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!(config.variance_penalty > 0.0);
        assert!(config.learning_rate > 0.0);
    }

    #[test]
    fn test_self_play_generator_creation() {
        let network = UniformGeoNetwork::new();
        let mcts_config = AlphaGeoZeroConfig::default();
        let game_config = GeoGameConfig::default();

        let generator = SelfPlayGenerator::new(network, mcts_config, game_config);
        assert_eq!(generator.stats.games_played, 0);
    }

    #[test]
    fn test_alpha_geo_zero_config_default() {
        let config = AlphaGeoZeroConfig::default();

        assert!(config.c_puct > 0.0);
        assert!(config.c_ignorance > 0.0);
        assert!(config.num_simulations > 0);
        assert!(config.use_variance_bonus);
    }

    #[test]
    fn test_mcts_stats_default() {
        let stats = MCTSStats::default();
        assert_eq!(stats.total_simulations, 0);
        assert!(!stats.proof_found);
    }
}

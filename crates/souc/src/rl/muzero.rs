//! MuZero-Style Latent Dynamics with Epistemic States
//!
//! MuZero learns a world model in latent space rather than using explicit rules.
//! Our innovation: latent states carry epistemic uncertainty (Beta distributions).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     Epistemic MuZero                                 │
//! │                                                                     │
//! │  Observation ──► Representation ──► LatentState[ε]                  │
//! │                       h(o)              s₀                          │
//! │                                         │                           │
//! │                                         ▼                           │
//! │                              ┌──────────────────┐                   │
//! │                              │ Prediction f(s) │                    │
//! │                              │  • policy π     │                    │
//! │                              │  • value v      │                    │
//! │                              │  • uncertainty ε│                    │
//! │                              └──────────────────┘                   │
//! │                                         │                           │
//! │                    Action ──► Dynamics g(s,a) ──► LatentState[ε']   │
//! │                                                        s₁          │
//! │                                                                     │
//! │  Key Innovation: Each LatentState carries BetaConfidence            │
//! │  - Model uncertainty propagates through dynamics                    │
//! │  - MCTS explores high-uncertainty branches                          │
//! │  - Training penalizes variance in addition to value loss            │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Epistemic Innovation
//!
//! Standard MuZero has no notion of model uncertainty. Our version:
//!
//! 1. **Latent states are distributions**, not points
//!    - Each dimension has mean + variance
//!    - Variance tracks model's confidence in that feature
//!
//! 2. **Dynamics model outputs uncertainty**
//!    - Predicts how confident it is about state transition
//!    - Uncertainty increases for out-of-distribution states
//!
//! 3. **MCTS uses uncertainty for exploration**
//!    - High model uncertainty → explore more
//!    - Low uncertainty → trust the model, exploit
//!
//! 4. **Training includes uncertainty calibration**
//!    - Penalize overconfident wrong predictions
//!    - Encourage conservative uncertainty in novel states

use std::collections::HashMap;
use std::hash::Hash;

use crate::epistemic::bayesian::BetaConfidence;
use crate::epistemic::merkle::{Hash256, MerkleProvenanceDAG};

use super::game::{Action, GameState, Player};

// =============================================================================
// Latent State with Epistemic Uncertainty
// =============================================================================

/// A latent state representation with epistemic uncertainty
///
/// Each dimension of the latent state is a distribution (mean + variance),
/// not a point estimate. This allows the model to express "I don't know"
/// about certain features.
#[derive(Debug, Clone)]
pub struct LatentState {
    /// Latent representation (mean values)
    pub features: Vec<f64>,
    /// Epistemic uncertainty per feature (variance)
    pub uncertainties: Vec<f64>,
    /// Overall state confidence (Beta distribution)
    pub confidence: BetaConfidence,
    /// Depth in the search tree (for dynamics rollout)
    pub depth: usize,
    /// Hash for provenance tracking
    pub hash: Option<Hash256>,
    /// Parent state hash (for Merkle tree)
    pub parent_hash: Option<Hash256>,
    /// Action that led to this state
    pub action_taken: Option<usize>,
}

impl LatentState {
    /// Create a new latent state with given features
    pub fn new(features: Vec<f64>, initial_confidence: BetaConfidence) -> Self {
        let dim = features.len();
        LatentState {
            features,
            uncertainties: vec![0.1; dim], // Start with moderate uncertainty
            confidence: initial_confidence,
            depth: 0,
            hash: None,
            parent_hash: None,
            action_taken: None,
        }
    }

    /// Create with explicit uncertainties
    pub fn with_uncertainties(features: Vec<f64>, uncertainties: Vec<f64>) -> Self {
        assert_eq!(features.len(), uncertainties.len());
        // Compute confidence from uncertainties
        let mean_uncertainty: f64 = uncertainties.iter().sum::<f64>() / uncertainties.len() as f64;
        let confidence = BetaConfidence::from_confidence(1.0 - mean_uncertainty.min(1.0), 10.0);

        LatentState {
            features,
            uncertainties,
            confidence,
            depth: 0,
            hash: None,
            parent_hash: None,
            action_taken: None,
        }
    }

    /// Create initial state from observation encoding
    pub fn from_observation(features: Vec<f64>) -> Self {
        // Observations have high confidence (we observe them directly)
        Self::new(features, BetaConfidence::new(50.0, 1.0))
    }

    /// Get dimensionality
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Get mean uncertainty across all features
    pub fn mean_uncertainty(&self) -> f64 {
        if self.uncertainties.is_empty() {
            return 1.0;
        }
        self.uncertainties.iter().sum::<f64>() / self.uncertainties.len() as f64
    }

    /// Get maximum uncertainty (most uncertain feature)
    pub fn max_uncertainty(&self) -> f64 {
        self.uncertainties.iter().cloned().fold(0.0, f64::max)
    }

    /// Check if state is highly uncertain
    pub fn is_uncertain(&self, threshold: f64) -> bool {
        self.mean_uncertainty() > threshold || self.confidence.variance() > threshold
    }

    /// Compute hash for this state
    pub fn compute_hash(&mut self) {
        use crate::epistemic::merkle::hash;

        let mut data = Vec::new();

        // Include features
        for f in &self.features {
            data.extend_from_slice(&f.to_le_bytes());
        }

        // Include depth
        data.extend_from_slice(&self.depth.to_le_bytes());

        // Include parent hash if present
        if let Some(parent) = &self.parent_hash {
            data.extend_from_slice(parent.as_bytes());
        }

        // Include action
        if let Some(action) = self.action_taken {
            data.extend_from_slice(&action.to_le_bytes());
        }

        self.hash = Some(hash(&data));
    }

    /// Get or compute hash
    pub fn get_hash(&mut self) -> Hash256 {
        if self.hash.is_none() {
            self.compute_hash();
        }
        self.hash.unwrap()
    }
}

// =============================================================================
// Model Predictions with Epistemic Uncertainty
// =============================================================================

/// Prediction output from the neural network
#[derive(Debug, Clone)]
pub struct EpistemicPrediction {
    /// Policy (action probabilities)
    pub policy: Vec<f64>,
    /// Value estimate
    pub value: f64,
    /// Value uncertainty (Beta distribution)
    pub value_confidence: BetaConfidence,
    /// Reward prediction (for dynamics)
    pub reward: f64,
    /// Reward uncertainty
    pub reward_confidence: BetaConfidence,
}

impl EpistemicPrediction {
    /// Create a uniform prediction (maximum uncertainty)
    pub fn uniform(num_actions: usize) -> Self {
        EpistemicPrediction {
            policy: vec![1.0 / num_actions as f64; num_actions],
            value: 0.0,
            value_confidence: BetaConfidence::uniform_prior(),
            reward: 0.0,
            reward_confidence: BetaConfidence::uniform_prior(),
        }
    }

    /// Get the most probable action
    pub fn best_action(&self) -> usize {
        self.policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Sample an action according to policy
    pub fn sample_action(&self, temperature: f64) -> usize {
        if temperature <= 0.0 {
            return self.best_action();
        }

        // Apply temperature
        let logits: Vec<f64> = self
            .policy
            .iter()
            .map(|p| (p.max(1e-8)).ln() / temperature)
            .collect();

        // Softmax
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        let probs: Vec<f64> = logits
            .iter()
            .map(|l| (l - max_logit).exp() / exp_sum)
            .collect();

        // Sample (simple linear search)
        let r: f64 = rand_simple();
        let mut cumsum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        probs.len() - 1
    }
}

/// Simple deterministic "random" for reproducibility in tests
fn rand_simple() -> f64 {
    use std::cell::RefCell;
    thread_local! {
        static STATE: RefCell<u64> = const { RefCell::new(12345) };
    }
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state as f64) / (u64::MAX as f64)
    })
}

// =============================================================================
// Dynamics Transition with Uncertainty Propagation
// =============================================================================

/// Result of applying dynamics model
#[derive(Debug, Clone)]
pub struct DynamicsResult {
    /// Next latent state
    pub next_state: LatentState,
    /// Predicted reward
    pub reward: f64,
    /// Confidence in the transition
    pub transition_confidence: BetaConfidence,
}

impl DynamicsResult {
    /// Create a terminal state (no further transitions)
    pub fn terminal(state: LatentState, final_reward: f64) -> Self {
        DynamicsResult {
            next_state: state,
            reward: final_reward,
            transition_confidence: BetaConfidence::new(100.0, 1.0), // Certain (it's terminal)
        }
    }
}

// =============================================================================
// MuZero Model Interface (to be implemented by neural network)
// =============================================================================

/// Trait for the MuZero model components
pub trait MuZeroModel: Clone {
    /// Number of actions in the action space
    fn num_actions(&self) -> usize;

    /// Latent state dimensionality
    fn latent_dim(&self) -> usize;

    /// Encode observation to latent state: h(o) → s
    fn represent(&self, observation: &[f64]) -> LatentState;

    /// Predict policy and value from latent state: f(s) → (π, v)
    fn predict(&self, state: &LatentState) -> EpistemicPrediction;

    /// Apply dynamics to get next state: g(s, a) → (s', r)
    fn dynamics(&self, state: &LatentState, action: usize) -> DynamicsResult;

    /// Check if state is terminal
    fn is_terminal(&self, state: &LatentState) -> bool;

    /// Get uncertainty penalty for training
    fn uncertainty_penalty(&self, state: &LatentState) -> f64 {
        state.mean_uncertainty() * 0.1 // Default: 10% of uncertainty
    }
}

// =============================================================================
// Epistemic MCTS in Latent Space
// =============================================================================

/// Configuration for latent MCTS
#[derive(Debug, Clone)]
pub struct LatentMCTSConfig {
    /// Number of simulations per move
    pub num_simulations: usize,
    /// Exploration constant (c_puct)
    pub c_puct: f64,
    /// Epistemic exploration bonus
    pub c_epistemic: f64,
    /// Discount factor
    pub gamma: f64,
    /// Maximum search depth
    pub max_depth: usize,
    /// Uncertainty threshold for pruning
    pub uncertainty_threshold: f64,
    /// Temperature for action selection
    pub temperature: f64,
    /// Whether to use Dirichlet noise at root
    pub use_dirichlet_noise: bool,
    /// Dirichlet alpha parameter
    pub dirichlet_alpha: f64,
    /// Noise fraction
    pub noise_fraction: f64,
}

impl Default for LatentMCTSConfig {
    fn default() -> Self {
        LatentMCTSConfig {
            num_simulations: 800,
            c_puct: 1.25,
            c_epistemic: 0.5,
            gamma: 0.997,
            max_depth: 50,
            uncertainty_threshold: 0.5,
            temperature: 1.0,
            use_dirichlet_noise: true,
            dirichlet_alpha: 0.3,
            noise_fraction: 0.25,
        }
    }
}

/// Node in the latent MCTS tree
#[derive(Debug, Clone)]
pub struct LatentMCTSNode {
    /// Latent state at this node
    pub state: LatentState,
    /// Visit count
    pub visits: u64,
    /// Value estimate (Beta distribution)
    pub value: BetaConfidence,
    /// Prior probability from policy network
    pub prior: f64,
    /// Children indexed by action
    pub children: HashMap<usize, LatentMCTSNode>,
    /// Reward received transitioning to this node
    pub reward: f64,
    /// Whether this is a terminal node
    pub is_terminal: bool,
    /// Sum of values for mean calculation
    pub value_sum: f64,
}

impl LatentMCTSNode {
    /// Create root node from latent state
    pub fn root(state: LatentState) -> Self {
        LatentMCTSNode {
            state,
            visits: 0,
            value: BetaConfidence::uniform_prior(),
            prior: 1.0,
            children: HashMap::new(),
            reward: 0.0,
            is_terminal: false,
            value_sum: 0.0,
        }
    }

    /// Create child node
    pub fn child(state: LatentState, prior: f64, reward: f64, is_terminal: bool) -> Self {
        LatentMCTSNode {
            state,
            visits: 0,
            value: BetaConfidence::uniform_prior(),
            prior,
            children: HashMap::new(),
            reward,
            is_terminal,
            value_sum: 0.0,
        }
    }

    /// Get Q value (mean of Beta distribution)
    pub fn q_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f64
        }
    }

    /// Get epistemic uncertainty (variance)
    pub fn epistemic_uncertainty(&self) -> f64 {
        self.value.variance() + self.state.mean_uncertainty()
    }

    /// PUCT score with epistemic bonus
    pub fn puct_score(&self, parent_visits: u64, config: &LatentMCTSConfig) -> f64 {
        let q = self.q_value();
        let exploration = config.c_puct
            * self.prior
            * ((parent_visits as f64).sqrt() / (1.0 + self.visits as f64));
        let epistemic_bonus = config.c_epistemic * self.epistemic_uncertainty();

        q + exploration + epistemic_bonus
    }

    /// Check if this node should be pruned due to uncertainty
    pub fn should_prune(&self, config: &LatentMCTSConfig) -> bool {
        self.state.mean_uncertainty() > config.uncertainty_threshold && self.visits > 10 // Only prune after some exploration
    }

    /// Expand this node using the model
    pub fn expand<M: MuZeroModel>(&mut self, model: &M) {
        if self.is_terminal || !self.children.is_empty() {
            return;
        }

        let prediction = model.predict(&self.state);

        for action in 0..model.num_actions() {
            let dynamics_result = model.dynamics(&self.state, action);
            let is_terminal = model.is_terminal(&dynamics_result.next_state);

            let child = LatentMCTSNode::child(
                dynamics_result.next_state,
                prediction.policy[action],
                dynamics_result.reward,
                is_terminal,
            );

            self.children.insert(action, child);
        }
    }

    /// Select best child according to PUCT
    pub fn select_child(&self, config: &LatentMCTSConfig) -> Option<usize> {
        if self.children.is_empty() {
            return None;
        }

        self.children
            .iter()
            .filter(|(_, child)| !child.should_prune(config))
            .max_by(|(_, a), (_, b)| {
                let score_a = a.puct_score(self.visits, config);
                let score_b = b.puct_score(self.visits, config);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(action, _)| *action)
    }

    /// Backpropagate value
    pub fn backpropagate(&mut self, value: f64) {
        self.visits += 1;
        self.value_sum += value;

        // Update Beta distribution
        // Treat value as success rate
        let success = if value > 0.0 { value } else { 0.0 };
        let failure = if value < 1.0 { 1.0 - value } else { 0.0 };
        self.value.update(success, failure);
    }
}

/// Latent MCTS tree
pub struct LatentMCTSTree<M: MuZeroModel> {
    /// Root node
    pub root: LatentMCTSNode,
    /// Model for predictions and dynamics
    pub model: M,
    /// Configuration
    pub config: LatentMCTSConfig,
    /// Provenance DAG
    pub provenance: MerkleProvenanceDAG,
}

impl<M: MuZeroModel> LatentMCTSTree<M> {
    /// Create a new tree from observation
    pub fn new(observation: &[f64], model: M, config: LatentMCTSConfig) -> Self {
        let latent_state = model.represent(observation);
        let root = LatentMCTSNode::root(latent_state);

        LatentMCTSTree {
            root,
            model,
            config,
            provenance: MerkleProvenanceDAG::new(),
        }
    }

    /// Create from existing latent state
    pub fn from_latent(state: LatentState, model: M, config: LatentMCTSConfig) -> Self {
        let root = LatentMCTSNode::root(state);

        LatentMCTSTree {
            root,
            model,
            config,
            provenance: MerkleProvenanceDAG::new(),
        }
    }

    /// Run MCTS search
    pub fn search(&mut self) {
        // Expand root if needed
        self.root.expand(&self.model);

        for _ in 0..self.config.num_simulations {
            self.simulate();
        }
    }

    /// Run a single simulation
    fn simulate(&mut self) {
        let mut path: Vec<usize> = Vec::new();
        let mut node = &mut self.root;
        let mut depth = 0;

        // Selection: traverse tree using PUCT
        while !node.children.is_empty() && depth < self.config.max_depth {
            if let Some(action) = node.select_child(&self.config) {
                path.push(action);
                node = node.children.get_mut(&action).unwrap();
                depth += 1;

                if node.is_terminal {
                    break;
                }
            } else {
                break; // All children pruned
            }
        }

        // Expansion: expand leaf if not terminal
        if !node.is_terminal && node.children.is_empty() {
            node.expand(&self.model);
        }

        // Evaluation: get value from prediction network
        let value = if node.is_terminal {
            // Use the reward as terminal value
            node.reward
        } else {
            let prediction = self.model.predict(&node.state);
            prediction.value
        };

        // Backpropagation: update values along path
        self.backpropagate(&path, value);
    }

    /// Backpropagate value along path
    fn backpropagate(&mut self, path: &[usize], value: f64) {
        // First, collect rewards along the path to compute discounted values
        let rewards = self.collect_rewards_along_path(path);

        // Compute discounted values from end to start
        let mut values = Vec::with_capacity(path.len() + 1);
        let mut v = value;
        values.push(v);

        for &reward in rewards.iter().rev() {
            v = reward + self.config.gamma * v;
            values.push(v);
        }
        values.reverse();

        // Now backpropagate values along path
        Self::backpropagate_node(&mut self.root, path, &values, 0);
    }

    /// Collect rewards along the path (read-only traversal)
    fn collect_rewards_along_path(&self, path: &[usize]) -> Vec<f64> {
        let mut rewards = Vec::with_capacity(path.len());
        let mut node = &self.root;

        for &action in path {
            if let Some(child) = node.children.get(&action) {
                rewards.push(child.reward);
                node = child;
            } else {
                break;
            }
        }

        rewards
    }

    /// Recursive helper for backpropagation to avoid borrow issues
    fn backpropagate_node(node: &mut LatentMCTSNode, path: &[usize], values: &[f64], depth: usize) {
        if depth < values.len() {
            node.backpropagate(values[depth]);
        }

        if let Some((&action, rest)) = path.split_first()
            && let Some(child) = node.children.get_mut(&action)
        {
            Self::backpropagate_node(child, rest, values, depth + 1);
        }
    }

    /// Get action probabilities (visit counts normalized)
    pub fn get_policy(&self) -> Vec<f64> {
        let mut policy = vec![0.0; self.model.num_actions()];
        let total_visits: u64 = self.root.children.values().map(|c| c.visits).sum();

        if total_visits == 0 {
            // Uniform if no visits
            let uniform = 1.0 / self.model.num_actions() as f64;
            return vec![uniform; self.model.num_actions()];
        }

        for (&action, child) in &self.root.children {
            policy[action] = child.visits as f64 / total_visits as f64;
        }

        policy
    }

    /// Get best action
    pub fn best_action(&self) -> usize {
        self.root
            .children
            .iter()
            .max_by_key(|(_, child)| child.visits)
            .map(|(&action, _)| action)
            .unwrap_or(0)
    }

    /// Sample action according to temperature
    pub fn sample_action(&self, temperature: f64) -> usize {
        if temperature <= 0.0 {
            return self.best_action();
        }

        let policy = self.get_policy();
        let pred = EpistemicPrediction {
            policy,
            value: 0.0,
            value_confidence: BetaConfidence::uniform_prior(),
            reward: 0.0,
            reward_confidence: BetaConfidence::uniform_prior(),
        };

        pred.sample_action(temperature)
    }

    /// Get search statistics
    pub fn stats(&self) -> LatentSearchStats {
        let total_visits: u64 = self.root.children.values().map(|c| c.visits).sum();
        let max_depth = self.compute_max_depth(&self.root, 0);
        let mean_uncertainty = self.compute_mean_uncertainty(&self.root);

        LatentSearchStats {
            total_simulations: self.root.visits,
            total_child_visits: total_visits,
            max_depth,
            mean_uncertainty,
            root_value: self.root.q_value(),
            root_confidence: self.root.value,
        }
    }

    fn compute_max_depth(&self, node: &LatentMCTSNode, current: usize) -> usize {
        let mut max = current;
        for child in node.children.values() {
            let child_depth = self.compute_max_depth(child, current + 1);
            max = max.max(child_depth);
        }
        max
    }

    fn compute_mean_uncertainty(&self, node: &LatentMCTSNode) -> f64 {
        let mut sum = node.state.mean_uncertainty();
        let mut count = 1;

        for child in node.children.values() {
            sum += self.compute_mean_uncertainty(child);
            count += 1;
        }

        sum / count as f64
    }

    /// Advance tree after taking an action (reuse subtree)
    pub fn advance(&mut self, action: usize) {
        if let Some(child) = self.root.children.remove(&action) {
            self.root = child;
        } else {
            // Action not in tree, need to recompute
            let dynamics_result = self.model.dynamics(&self.root.state, action);
            self.root = LatentMCTSNode::root(dynamics_result.next_state);
        }
    }
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct LatentSearchStats {
    pub total_simulations: u64,
    pub total_child_visits: u64,
    pub max_depth: usize,
    pub mean_uncertainty: f64,
    pub root_value: f64,
    pub root_confidence: BetaConfidence,
}

// =============================================================================
// Training Target Generation
// =============================================================================

/// Training target for MuZero
#[derive(Debug, Clone)]
pub struct MuZeroTarget {
    /// Observation (input to representation function)
    pub observation: Vec<f64>,
    /// Target policy (from MCTS)
    pub policy: Vec<f64>,
    /// Target value
    pub value: f64,
    /// Sequence of actions taken
    pub actions: Vec<usize>,
    /// Sequence of rewards received
    pub rewards: Vec<f64>,
    /// Sequence of values (for TD targets)
    pub values: Vec<f64>,
    /// Uncertainties at each step
    pub uncertainties: Vec<f64>,
}

impl MuZeroTarget {
    /// Create from a game trajectory
    pub fn from_trajectory<S: GameState>(
        observations: Vec<Vec<f64>>,
        policies: Vec<Vec<f64>>,
        actions: Vec<usize>,
        rewards: Vec<f64>,
        outcome: f64,
        gamma: f64,
    ) -> Vec<Self> {
        let n = observations.len();
        let mut targets = Vec::with_capacity(n);

        // Compute returns (discounted sum of future rewards + final value)
        let mut returns = vec![0.0; n];
        returns[n - 1] = outcome;
        for i in (0..n - 1).rev() {
            returns[i] = rewards[i] + gamma * returns[i + 1];
        }

        for i in 0..n {
            // Look-ahead for K steps (MuZero uses K=5)
            let k = 5.min(n - i);
            let action_seq: Vec<usize> = actions[i..i + k].to_vec();
            let reward_seq: Vec<f64> = rewards[i..i + k].to_vec();
            let value_seq: Vec<f64> = returns[i..i + k].to_vec();

            targets.push(MuZeroTarget {
                observation: observations[i].clone(),
                policy: policies[i].clone(),
                value: returns[i],
                actions: action_seq,
                rewards: reward_seq,
                values: value_seq,
                uncertainties: vec![0.0; k], // To be filled by model
            });
        }

        targets
    }
}

// =============================================================================
// Dummy Model for Testing
// =============================================================================

/// Simple dummy model for testing
#[derive(Debug, Clone)]
pub struct DummyMuZeroModel {
    pub num_actions: usize,
    pub latent_dim: usize,
}

impl DummyMuZeroModel {
    pub fn new(num_actions: usize, latent_dim: usize) -> Self {
        DummyMuZeroModel {
            num_actions,
            latent_dim,
        }
    }
}

impl MuZeroModel for DummyMuZeroModel {
    fn num_actions(&self) -> usize {
        self.num_actions
    }

    fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    fn represent(&self, observation: &[f64]) -> LatentState {
        // Just use observation as latent state (identity)
        let features = if observation.len() >= self.latent_dim {
            observation[..self.latent_dim].to_vec()
        } else {
            let mut f = observation.to_vec();
            f.resize(self.latent_dim, 0.0);
            f
        };

        LatentState::from_observation(features)
    }

    fn predict(&self, state: &LatentState) -> EpistemicPrediction {
        // Uniform policy, zero value
        EpistemicPrediction::uniform(self.num_actions)
    }

    fn dynamics(&self, state: &LatentState, action: usize) -> DynamicsResult {
        // Identity dynamics with small perturbation
        let mut next_features = state.features.clone();
        if action < next_features.len() {
            next_features[action] += 0.1;
        }

        let mut next_state = LatentState::with_uncertainties(
            next_features,
            state.uncertainties.iter().map(|u| u * 1.1).collect(), // Increase uncertainty
        );
        next_state.depth = state.depth + 1;
        next_state.parent_hash = state.hash;
        next_state.action_taken = Some(action);

        DynamicsResult {
            next_state,
            reward: 0.0,
            transition_confidence: BetaConfidence::new(10.0, 1.0),
        }
    }

    fn is_terminal(&self, state: &LatentState) -> bool {
        state.depth >= 10 // Terminal after 10 steps
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_creation() {
        let state = LatentState::new(vec![1.0, 2.0, 3.0], BetaConfidence::new(10.0, 1.0));
        assert_eq!(state.dim(), 3);
        assert!(state.confidence.mean() > 0.9);
    }

    #[test]
    fn test_latent_state_uncertainty() {
        let state = LatentState::with_uncertainties(vec![1.0, 2.0], vec![0.1, 0.3]);
        assert!((state.mean_uncertainty() - 0.2).abs() < 0.01);
        assert!((state.max_uncertainty() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_latent_state_hash() {
        let mut state1 = LatentState::new(vec![1.0, 2.0], BetaConfidence::uniform_prior());
        let mut state2 = LatentState::new(vec![1.0, 2.0], BetaConfidence::uniform_prior());

        let hash1 = state1.get_hash();
        let hash2 = state2.get_hash();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_epistemic_prediction() {
        let pred = EpistemicPrediction::uniform(4);
        assert_eq!(pred.policy.len(), 4);
        assert!((pred.policy[0] - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_dummy_model() {
        let model = DummyMuZeroModel::new(4, 8);
        assert_eq!(model.num_actions(), 4);
        assert_eq!(model.latent_dim(), 8);

        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let state = model.represent(&obs);
        assert_eq!(state.dim(), 8);

        let pred = model.predict(&state);
        assert_eq!(pred.policy.len(), 4);
    }

    #[test]
    fn test_dynamics() {
        let model = DummyMuZeroModel::new(4, 8);
        let obs = vec![0.0; 8];
        let state = model.represent(&obs);

        let result = model.dynamics(&state, 0);
        assert_eq!(result.next_state.depth, 1);
        assert!(result.next_state.features[0] > 0.0);
    }

    #[test]
    fn test_latent_mcts_node() {
        let state = LatentState::new(vec![1.0, 2.0], BetaConfidence::new(10.0, 1.0));
        let mut node = LatentMCTSNode::root(state);

        assert_eq!(node.visits, 0);
        assert_eq!(node.q_value(), 0.0);

        node.backpropagate(0.8);
        assert_eq!(node.visits, 1);
        assert!((node.q_value() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_latent_mcts_tree() {
        let model = DummyMuZeroModel::new(4, 8);
        let config = LatentMCTSConfig {
            num_simulations: 10,
            ..Default::default()
        };

        let obs = vec![0.0; 8];
        let mut tree = LatentMCTSTree::new(&obs, model, config);

        tree.search();

        let stats = tree.stats();
        assert!(stats.total_simulations > 0);

        let policy = tree.get_policy();
        assert_eq!(policy.len(), 4);
    }

    #[test]
    fn test_latent_mcts_action_selection() {
        let model = DummyMuZeroModel::new(4, 8);
        let config = LatentMCTSConfig {
            num_simulations: 50,
            ..Default::default()
        };

        let obs = vec![0.0; 8];
        let mut tree = LatentMCTSTree::new(&obs, model, config);

        tree.search();

        let best = tree.best_action();
        assert!(best < 4);

        let sampled = tree.sample_action(1.0);
        assert!(sampled < 4);
    }

    #[test]
    fn test_tree_advance() {
        let model = DummyMuZeroModel::new(4, 8);
        let config = LatentMCTSConfig {
            num_simulations: 20,
            ..Default::default()
        };

        let obs = vec![0.0; 8];
        let mut tree = LatentMCTSTree::new(&obs, model, config);

        tree.search();
        let initial_depth = tree.root.state.depth;

        tree.advance(0);
        assert_eq!(tree.root.state.depth, initial_depth + 1);
    }

    #[test]
    fn test_uncertainty_propagation() {
        let model = DummyMuZeroModel::new(4, 8);
        let obs = vec![0.0; 8];
        let state = model.represent(&obs);

        // Apply dynamics multiple times
        let mut current = state;
        for i in 0..5 {
            let result = model.dynamics(&current, 0);
            // Uncertainty should increase
            assert!(result.next_state.mean_uncertainty() >= current.mean_uncertainty());
            current = result.next_state;
        }
    }

    #[test]
    fn test_muzero_target() {
        let observations = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let policies = vec![vec![0.5, 0.5], vec![0.6, 0.4], vec![0.7, 0.3]];
        let actions = vec![0, 1, 0];
        let rewards = vec![0.0, 0.1, 0.0];
        let outcome = 1.0;

        let targets = MuZeroTarget::from_trajectory::<DummyGameState>(
            observations,
            policies,
            actions,
            rewards,
            outcome,
            0.99,
        );

        assert_eq!(targets.len(), 3);
        assert!(targets[0].value > targets[2].value); // Earlier states should have higher return
    }
}

// Dummy action type for testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DummyAction(pub usize);

// Action is a marker trait with only bounds (Debug + Clone + Eq + Hash + Send + Sync)
impl Action for DummyAction {}

/// Dummy game state for testing MuZero
#[derive(Debug, Clone)]
pub struct DummyGameState {
    pub step: usize,
    pub max_steps: usize,
    pub num_actions: usize,
}

impl DummyGameState {
    pub fn new(max_steps: usize, num_actions: usize) -> Self {
        DummyGameState {
            step: 0,
            max_steps,
            num_actions,
        }
    }
}

impl GameState for DummyGameState {
    type Action = DummyAction;

    fn legal_actions(&self) -> Vec<Self::Action> {
        (0..self.num_actions).map(DummyAction).collect()
    }

    fn apply_action(&self, _action: &Self::Action) -> Self {
        DummyGameState {
            step: self.step + 1,
            ..*self
        }
    }

    fn is_terminal(&self) -> bool {
        self.step >= self.max_steps
    }

    fn current_player(&self) -> Player {
        if self.step.is_multiple_of(2) {
            Player::One
        } else {
            Player::Two
        }
    }

    fn terminal_value(&self, player: Player) -> f64 {
        if !self.is_terminal() {
            return 0.5;
        }
        // Dummy: Player One always wins
        match player {
            Player::One => 1.0,
            Player::Two => 0.0,
        }
    }
}

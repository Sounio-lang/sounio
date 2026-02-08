//! Epistemic Monte Carlo Tree Search (MCTS)
//!
//! MCTS with Beta-distributed Q-values for principled uncertainty quantification.
//!
//! # Key Innovations
//!
//! 1. **Q as Beta**: Q-values are Beta(α, β) distributions, not scalars
//! 2. **Epistemic PUCT**: Selection bonus for high-variance (uncertain) nodes
//! 3. **Hierarchical backprop**: Bayesian update of Q distributions
//! 4. **Provenance tracking**: Every node has Merkle hash for explainability
//!
//! # PUCT Formula (Epistemic Extension)
//!
//! ```text
//! UCB(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
//!                    + c_epistemic * σ(Q(s,a))  // uncertainty bonus!
//! ```
//!
//! The epistemic bonus encourages exploration of uncertain branches,
//! implementing active inference: "reduce ignorance first".

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::epistemic::bayesian::BetaConfidence;
use crate::epistemic::merkle::{MerkleProvenanceDAG, OperationKind, ProvenanceOperation};

use super::game::{Action, GameState, Player};

// =============================================================================
// Configuration
// =============================================================================

/// MCTS configuration
#[derive(Debug, Clone)]
pub struct MCTSConfig {
    /// Number of simulations per search
    pub num_simulations: usize,
    /// PUCT exploration constant (c_puct)
    pub c_puct: f64,
    /// Epistemic exploration constant (c_epistemic)
    pub c_epistemic: f64,
    /// Temperature for action selection (τ)
    pub temperature: f64,
    /// Dirichlet noise alpha for root exploration
    pub dirichlet_alpha: f64,
    /// Fraction of noise to add at root
    pub noise_fraction: f64,
    /// Minimum visits before expanding
    pub min_visits_to_expand: u32,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Uncertainty threshold for neural re-evaluation
    pub uncertainty_threshold: f64,
    /// Decay factor for backpropagation
    pub backprop_decay: f64,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig {
            num_simulations: 800,
            c_puct: 1.5,
            c_epistemic: 0.5,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            noise_fraction: 0.25,
            min_visits_to_expand: 1,
            max_depth: 100,
            uncertainty_threshold: 0.3,
            backprop_decay: 0.99,
        }
    }
}

impl MCTSConfig {
    /// Config for chess-like games
    pub fn chess() -> Self {
        MCTSConfig {
            num_simulations: 800,
            c_puct: 2.5,
            c_epistemic: 0.3,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            noise_fraction: 0.25,
            ..Default::default()
        }
    }

    /// Config for Go-like games
    pub fn go() -> Self {
        MCTSConfig {
            num_simulations: 1600,
            c_puct: 2.0,
            c_epistemic: 0.4,
            temperature: 1.0,
            dirichlet_alpha: 0.03,
            noise_fraction: 0.25,
            ..Default::default()
        }
    }

    /// Config for fast play (fewer simulations)
    pub fn fast() -> Self {
        MCTSConfig {
            num_simulations: 100,
            c_puct: 1.5,
            c_epistemic: 0.5,
            temperature: 0.5,
            ..Default::default()
        }
    }
}

// =============================================================================
// MCTS Node
// =============================================================================

/// A node in the MCTS tree with epistemic Q-values
#[derive(Debug, Clone)]
pub struct MCTSNode<S: GameState> {
    /// Game state at this node
    pub state: S,
    /// Player to move
    pub player: Player,
    /// Prior probability from policy network (as Beta)
    pub prior: BetaConfidence,
    /// Q-value distribution (epistemic!)
    pub value: BetaConfidence,
    /// Visit count
    pub visits: u64,
    /// Sum of values (for mean calculation fallback)
    pub value_sum: f64,
    /// Children indexed by action
    pub children: HashMap<S::Action, MCTSNode<S>>,
    /// Whether this node has been expanded
    pub is_expanded: bool,
    /// Whether this is a terminal state
    pub is_terminal: bool,
    /// Terminal value (if terminal)
    pub terminal_value: Option<f64>,
    /// Depth in tree
    pub depth: usize,
    /// Provenance hash for explainability
    pub provenance_hash: u64,
    /// Parent action (how we got here)
    pub parent_action: Option<S::Action>,
}

impl<S: GameState> MCTSNode<S>
where
    S::Action: Clone + Hash + Eq,
{
    /// Create a new root node
    pub fn root(state: S) -> Self {
        let player = state.current_player();
        let is_terminal = state.is_terminal();
        let terminal_value = if is_terminal {
            Some(state.terminal_value(player))
        } else {
            None
        };

        MCTSNode {
            state,
            player,
            prior: BetaConfidence::uniform_prior(),
            value: BetaConfidence::uniform_prior(),
            visits: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            is_expanded: false,
            is_terminal,
            terminal_value,
            depth: 0,
            provenance_hash: 0,
            parent_action: None,
        }
    }

    /// Create a child node
    pub fn child(state: S, prior: BetaConfidence, parent_action: S::Action, depth: usize) -> Self {
        let player = state.current_player();
        let is_terminal = state.is_terminal();
        let terminal_value = if is_terminal {
            Some(state.terminal_value(player))
        } else {
            None
        };

        // Compute provenance hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_action.hash(&mut hasher);
        depth.hash(&mut hasher);
        let provenance_hash = hasher.finish();

        MCTSNode {
            state,
            player,
            prior,
            value: BetaConfidence::uniform_prior(),
            visits: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            is_expanded: false,
            is_terminal,
            terminal_value,
            depth,
            provenance_hash,
            parent_action: Some(parent_action),
        }
    }

    /// Get mean Q-value
    pub fn q_mean(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value.mean()
        }
    }

    /// Get Q-value variance (epistemic uncertainty)
    pub fn q_variance(&self) -> f64 {
        self.value.variance()
    }

    /// Get epistemic uncertainty (standard deviation)
    pub fn epistemic_uncertainty(&self) -> f64 {
        self.value.variance().sqrt()
    }

    /// Compute PUCT score for selection
    pub fn puct_score(&self, parent_visits: u64, config: &MCTSConfig) -> f64 {
        let q = self.q_mean();

        // Standard PUCT term
        let exploration = config.c_puct
            * self.prior.mean()
            * ((parent_visits as f64).sqrt() / (1.0 + self.visits as f64));

        // Epistemic bonus: explore uncertain nodes
        let epistemic_bonus = config.c_epistemic * self.epistemic_uncertainty();

        q + exploration + epistemic_bonus
    }

    /// Update node with backpropagated value
    pub fn update(&mut self, value: f64, config: &MCTSConfig) {
        self.visits += 1;
        self.value_sum += value;

        // Bayesian update of Beta distribution
        // Treat value as evidence: high value = success, low = failure
        let success = value.max(0.0).min(1.0);
        let failure = 1.0 - success;

        // Weight by decay factor
        let weight = config.backprop_decay;
        self.value.update_weighted(success, failure, weight);
    }

    /// Get the best child by visit count
    pub fn best_child_by_visits(&self) -> Option<(&S::Action, &MCTSNode<S>)> {
        self.children.iter().max_by_key(|(_, child)| child.visits)
    }

    /// Get the best child by Q-value
    pub fn best_child_by_value(&self) -> Option<(&S::Action, &MCTSNode<S>)> {
        self.children.iter().max_by(|(_, a), (_, b)| {
            a.q_mean()
                .partial_cmp(&b.q_mean())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get action probabilities (policy) based on visit counts
    pub fn action_probabilities(&self, temperature: f64) -> HashMap<S::Action, f64> {
        if self.children.is_empty() {
            return HashMap::new();
        }

        let visits: Vec<_> = self
            .children
            .iter()
            .map(|(a, c)| (a.clone(), c.visits as f64))
            .collect();

        let total: f64 = if temperature == 0.0 {
            // Deterministic: pick max
            let max_visits = visits.iter().map(|(_, v)| *v).fold(0.0, f64::max);
            visits.iter().filter(|(_, v)| *v == max_visits).count() as f64
        } else {
            // Temperature-scaled softmax
            visits.iter().map(|(_, v)| v.powf(1.0 / temperature)).sum()
        };

        visits
            .into_iter()
            .map(|(a, v)| {
                let prob = if temperature == 0.0 {
                    let max_visits = self
                        .children
                        .values()
                        .map(|c| c.visits as f64)
                        .fold(0.0, f64::max);
                    if v == max_visits { 1.0 / total } else { 0.0 }
                } else {
                    v.powf(1.0 / temperature) / total
                };
                (a, prob)
            })
            .collect()
    }

    /// Check if this node needs neural re-evaluation (high uncertainty)
    pub fn needs_reevaluation(&self, threshold: f64) -> bool {
        self.epistemic_uncertainty() > threshold && self.visits > 0
    }
}

// =============================================================================
// MCTS Tree
// =============================================================================

/// The MCTS tree with epistemic tracking
pub struct MCTSTree<S: GameState> {
    /// Root node
    pub root: MCTSNode<S>,
    /// Configuration
    pub config: MCTSConfig,
    /// Global epistemic variance (aggregate uncertainty)
    pub global_variance: f64,
    /// Provenance DAG for full traceability
    pub provenance: MerkleProvenanceDAG,
    /// Statistics
    pub stats: MCTSStats,
}

/// MCTS statistics
#[derive(Debug, Clone, Default)]
pub struct MCTSStats {
    pub total_simulations: usize,
    pub total_expansions: usize,
    pub max_depth_reached: usize,
    pub high_uncertainty_triggers: usize,
    pub neural_evaluations: usize,
}

impl<S: GameState + Clone> MCTSTree<S>
where
    S::Action: Clone + Hash + Eq,
{
    /// Create a new tree from root state
    pub fn new(state: S, config: MCTSConfig) -> Self {
        let root = MCTSNode::root(state);
        let mut provenance = MerkleProvenanceDAG::new();
        provenance.add_root(ProvenanceOperation::new(
            "mcts_root",
            OperationKind::Computation,
        ));

        MCTSTree {
            root,
            config,
            global_variance: 1.0, // Start with high uncertainty
            provenance,
            stats: MCTSStats::default(),
        }
    }

    /// Select a leaf node for expansion using PUCT
    pub fn select(&self) -> (Vec<S::Action>, *const MCTSNode<S>) {
        let mut path = Vec::new();
        let mut node = &self.root;

        while !node.children.is_empty() && !node.is_terminal {
            // Find child with highest PUCT score
            let parent_visits = node.visits;
            let best = node.children.iter().max_by(|(_, a), (_, b)| {
                let score_a = a.puct_score(parent_visits, &self.config);
                let score_b = b.puct_score(parent_visits, &self.config);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some((action, child)) = best {
                path.push(action.clone());
                node = child;
            } else {
                break;
            }
        }

        (path, node as *const _)
    }

    /// Expand a leaf node with policy priors
    pub fn expand(&mut self, path: &[S::Action], policy: HashMap<S::Action, f64>) {
        let node = self.get_node_mut(path);

        if node.is_expanded || node.is_terminal {
            return;
        }

        let legal_actions = node.state.legal_actions();
        let depth = node.depth + 1;

        for action in legal_actions {
            let new_state = node.state.apply_action(&action);

            // Get prior from policy, default to uniform
            let prior_value = policy
                .get(&action)
                .copied()
                .unwrap_or(1.0 / (policy.len().max(1) as f64));
            let prior = BetaConfidence::from_confidence(prior_value, 2.0);

            let child = MCTSNode::child(new_state, prior, action.clone(), depth);
            node.children.insert(action, child);
        }

        node.is_expanded = true;
        self.stats.total_expansions += 1;
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);
    }

    /// Backpropagate value through path
    pub fn backpropagate(&mut self, path: &[S::Action], value: f64) {
        // Update root
        self.root.update(value, &self.config);

        // Update path nodes with alternating perspective
        let mut current_value = value;
        let mut node = &mut self.root;

        for action in path {
            if let Some(child) = node.children.get_mut(action) {
                // Flip value for opponent
                current_value = 1.0 - current_value;
                child.update(current_value, &self.config);
                node = child;
            } else {
                break;
            }
        }

        // Update global variance
        self.global_variance = self.compute_global_variance();

        self.stats.total_simulations += 1;
    }

    /// Get a node by path (immutable)
    pub fn get_node(&self, path: &[S::Action]) -> &MCTSNode<S> {
        let mut node = &self.root;
        for action in path {
            if let Some(child) = node.children.get(action) {
                node = child;
            } else {
                return node;
            }
        }
        node
    }

    /// Get a node by path (mutable)
    pub fn get_node_mut(&mut self, path: &[S::Action]) -> &mut MCTSNode<S> {
        let mut node = &mut self.root;
        for action in path {
            if node.children.contains_key(action) {
                node = node.children.get_mut(action).unwrap();
            } else {
                return node;
            }
        }
        node
    }

    /// Compute global epistemic variance
    fn compute_global_variance(&self) -> f64 {
        let mut total_variance = 0.0;
        let mut count = 0;

        self.traverse_nodes(&self.root, &mut |node| {
            if node.visits > 0 {
                total_variance += node.q_variance();
                count += 1;
            }
        });

        if count > 0 {
            (total_variance / count as f64).sqrt()
        } else {
            1.0
        }
    }

    /// Traverse all nodes
    fn traverse_nodes<F>(&self, node: &MCTSNode<S>, f: &mut F)
    where
        F: FnMut(&MCTSNode<S>),
    {
        f(node);
        for child in node.children.values() {
            self.traverse_nodes(child, f);
        }
    }

    /// Get the best action from root
    pub fn best_action(&self) -> Option<S::Action> {
        self.root.best_child_by_visits().map(|(a, _)| a.clone())
    }

    /// Get search result with full statistics
    pub fn search_result(&self) -> SearchResult<S::Action> {
        let action_probs = self.root.action_probabilities(self.config.temperature);
        let best_action = self.best_action();

        SearchResult {
            best_action,
            action_probabilities: action_probs,
            root_value: self.root.value,
            root_visits: self.root.visits,
            global_uncertainty: self.global_variance,
            stats: self.stats.clone(),
        }
    }

    /// Advance tree by applying action (for reuse)
    pub fn advance(&mut self, action: &S::Action) -> bool {
        if let Some(child) = self.root.children.remove(action) {
            self.root = child;
            self.root.parent_action = None;
            self.root.depth = 0;
            true
        } else {
            false
        }
    }
}

/// Result of MCTS search
#[derive(Debug, Clone)]
pub struct SearchResult<A: Action> {
    /// Best action to take
    pub best_action: Option<A>,
    /// Action probabilities (policy target for training)
    pub action_probabilities: HashMap<A, f64>,
    /// Root Q-value distribution
    pub root_value: BetaConfidence,
    /// Total visits at root
    pub root_visits: u64,
    /// Global epistemic uncertainty
    pub global_uncertainty: f64,
    /// Statistics
    pub stats: MCTSStats,
}

impl<A: Action> SearchResult<A> {
    /// Get confidence interval for root value
    pub fn value_confidence_interval(&self, level: f64) -> (f64, f64) {
        self.root_value.credible_interval(level)
    }
}

// =============================================================================
// MCTS Search Function
// =============================================================================

/// Trait for neural network evaluation
pub trait NeuralEvaluator<S: GameState> {
    /// Evaluate state, returning (policy, value)
    fn evaluate(&self, state: &S) -> (HashMap<S::Action, f64>, f64);
}

/// Dummy evaluator that returns uniform policy and 0.5 value
pub struct UniformEvaluator;

impl<S: GameState> NeuralEvaluator<S> for UniformEvaluator
where
    S::Action: Clone + Eq + Hash,
{
    fn evaluate(&self, state: &S) -> (HashMap<S::Action, f64>, f64) {
        let actions = state.legal_actions();
        let prob = 1.0 / actions.len().max(1) as f64;
        let policy: HashMap<S::Action, f64> = actions.into_iter().map(|a| (a, prob)).collect();
        (policy, 0.5)
    }
}

/// Run MCTS search
pub fn search<S, E>(tree: &mut MCTSTree<S>, evaluator: &E) -> SearchResult<S::Action>
where
    S: GameState + Clone,
    S::Action: Clone + Hash + Eq,
    E: NeuralEvaluator<S>,
{
    let num_sims = tree.config.num_simulations;

    for _ in 0..num_sims {
        // Selection
        let (path, leaf_ptr) = tree.select();

        // Get leaf node state for evaluation
        let leaf = tree.get_node(&path);

        if leaf.is_terminal {
            // Terminal node - backprop terminal value
            let value = leaf.terminal_value.unwrap_or(0.5);
            tree.backpropagate(&path, value);
            continue;
        }

        // Neural evaluation
        let (policy, value) = evaluator.evaluate(&leaf.state);
        tree.stats.neural_evaluations += 1;

        // Expansion
        tree.expand(&path, policy);

        // Backpropagation
        tree.backpropagate(&path, value);

        // Check for high uncertainty trigger
        if tree.global_variance > tree.config.uncertainty_threshold {
            tree.stats.high_uncertainty_triggers += 1;
        }
    }

    tree.search_result()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test game: count to 10
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct CountState {
        count: i32,
        player: Player,
    }

    impl GameState for CountState {
        type Action = CountAction;

        fn current_player(&self) -> Player {
            self.player
        }

        fn legal_actions(&self) -> Vec<Self::Action> {
            if self.count >= 10 {
                vec![]
            } else {
                vec![CountAction::Add1, CountAction::Add2]
            }
        }

        fn apply_action(&self, action: &Self::Action) -> Self {
            let delta = match action {
                CountAction::Add1 => 1,
                CountAction::Add2 => 2,
            };
            CountState {
                count: self.count + delta,
                player: self.player.opponent(),
            }
        }

        fn is_terminal(&self) -> bool {
            self.count >= 10
        }

        fn terminal_value(&self, player: Player) -> f64 {
            if self.count >= 10 {
                if self.player == player { 0.0 } else { 1.0 }
            } else {
                0.5
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum CountAction {
        Add1,
        Add2,
    }

    impl Action for CountAction {}

    #[test]
    fn test_mcts_node_creation() {
        let state = CountState {
            count: 0,
            player: Player::One,
        };
        let node = MCTSNode::<CountState>::root(state);

        assert_eq!(node.visits, 0);
        assert!(!node.is_terminal);
        assert!(!node.is_expanded);
    }

    #[test]
    fn test_mcts_tree_creation() {
        let state = CountState {
            count: 0,
            player: Player::One,
        };
        let tree = MCTSTree::<CountState>::new(state, MCTSConfig::fast());

        assert_eq!(tree.root.visits, 0);
    }

    #[test]
    fn test_mcts_search() {
        let state = CountState {
            count: 0,
            player: Player::One,
        };
        let mut tree = MCTSTree::new(
            state,
            MCTSConfig {
                num_simulations: 50,
                ..MCTSConfig::fast()
            },
        );

        let result = search(&mut tree, &UniformEvaluator);

        assert!(result.best_action.is_some());
        assert!(result.root_visits > 0);
        assert!(!result.action_probabilities.is_empty());
    }

    #[test]
    fn test_epistemic_uncertainty() {
        let state = CountState {
            count: 0,
            player: Player::One,
        };
        let mut node = MCTSNode::<CountState>::root(state);

        // Initial uncertainty should be high
        let initial_uncertainty = node.epistemic_uncertainty();

        // Update with some values
        let config = MCTSConfig::default();
        node.update(0.7, &config);
        node.update(0.8, &config);
        node.update(0.75, &config);

        // Uncertainty should decrease with more observations
        let final_uncertainty = node.epistemic_uncertainty();
        assert!(final_uncertainty < initial_uncertainty);
    }

    #[test]
    fn test_puct_score() {
        let state = CountState {
            count: 0,
            player: Player::One,
        };
        let mut node = MCTSNode::<CountState>::root(state);
        node.prior = BetaConfidence::from_confidence(0.5, 5.0);

        let config = MCTSConfig::default();

        // Unvisited node should have exploration bonus
        let score_unvisited = node.puct_score(100, &config);

        node.visits = 10;
        node.update(0.6, &config);

        let score_visited = node.puct_score(100, &config);

        // Unvisited should have higher exploration bonus
        assert!(score_unvisited > score_visited || node.q_mean() > 0.5);
    }

    #[test]
    fn test_action_probabilities() {
        let state = CountState {
            count: 0,
            player: Player::One,
        };
        let mut tree = MCTSTree::new(
            state,
            MCTSConfig {
                num_simulations: 100,
                ..MCTSConfig::fast()
            },
        );

        let _ = search(&mut tree, &UniformEvaluator);

        let probs = tree.root.action_probabilities(1.0);

        // Probabilities should sum to ~1
        let total: f64 = probs.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}

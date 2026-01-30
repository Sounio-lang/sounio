//! Neural Network Integration for RL
//!
//! Policy-value networks with epistemic output (Beta distributions).
//!
//! # Key Features
//!
//! 1. **Epistemic outputs**: Policy and value as Beta distributions
//! 2. **Effect integration**: Neural evaluation as differentiable effect
//! 3. **GPU-ready**: Designed for kernel execution
//! 4. **Provenance**: Track which model version made predictions

use std::collections::HashMap;

use crate::epistemic::bayesian::BetaConfidence;

use super::game::{Action, GameTrait};

// =============================================================================
// Configuration
// =============================================================================

/// Neural network configuration
#[derive(Debug, Clone)]
pub struct NeuralConfig {
    /// Number of residual blocks
    pub num_blocks: usize,
    /// Number of filters per conv layer
    pub num_filters: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Policy loss weight
    pub policy_weight: f64,
    /// Value loss weight
    pub value_weight: f64,
    /// Epistemic variance penalty weight
    pub variance_penalty: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to output epistemic (Beta) predictions
    pub epistemic_output: bool,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        NeuralConfig {
            num_blocks: 19,
            num_filters: 256,
            learning_rate: 0.001,
            weight_decay: 1e-4,
            policy_weight: 1.0,
            value_weight: 1.0,
            variance_penalty: 0.1,
            batch_size: 256,
            epistemic_output: true,
        }
    }
}

impl NeuralConfig {
    /// Smaller network for testing
    pub fn small() -> Self {
        NeuralConfig {
            num_blocks: 4,
            num_filters: 64,
            batch_size: 32,
            ..Default::default()
        }
    }

    /// Config for Tic-Tac-Toe
    pub fn tictactoe() -> Self {
        NeuralConfig {
            num_blocks: 2,
            num_filters: 32,
            batch_size: 64,
            ..Default::default()
        }
    }
}

// =============================================================================
// Neural Evaluation Result
// =============================================================================

/// Result of neural network evaluation
#[derive(Debug, Clone)]
pub struct NeuralEval<A: Action> {
    /// Policy: probability distribution over actions
    pub policy: HashMap<A, f64>,
    /// Policy with epistemic uncertainty (Beta distributions)
    pub policy_epistemic: HashMap<A, BetaConfidence>,
    /// Value estimate (scalar)
    pub value: f64,
    /// Value with epistemic uncertainty
    pub value_epistemic: BetaConfidence,
    /// Model version/hash for provenance
    pub model_version: u64,
}

impl<A: Action> NeuralEval<A> {
    /// Create from scalar outputs (convert to Beta)
    pub fn from_scalars(policy: HashMap<A, f64>, value: f64, model_version: u64) -> Self
    where
        A: Clone,
    {
        // Convert policy to epistemic with weak prior
        let policy_epistemic: HashMap<A, BetaConfidence> = policy
            .iter()
            .map(|(a, &p)| (a.clone(), BetaConfidence::from_confidence(p, 5.0)))
            .collect();

        // Convert value to epistemic
        let value_epistemic = BetaConfidence::from_confidence(
            (value + 1.0) / 2.0, // Convert [-1, 1] to [0, 1]
            10.0,
        );

        NeuralEval {
            policy,
            policy_epistemic,
            value,
            value_epistemic,
            model_version,
        }
    }

    /// Create with full epistemic outputs
    pub fn epistemic(
        policy_epistemic: HashMap<A, BetaConfidence>,
        value_epistemic: BetaConfidence,
        model_version: u64,
    ) -> Self
    where
        A: Clone,
    {
        let policy: HashMap<A, f64> = policy_epistemic
            .iter()
            .map(|(a, beta)| (a.clone(), beta.mean()))
            .collect();
        let value = value_epistemic.mean() * 2.0 - 1.0; // Convert [0, 1] to [-1, 1]

        NeuralEval {
            policy,
            policy_epistemic,
            value,
            value_epistemic,
            model_version,
        }
    }

    /// Get policy entropy (for exploration bonus)
    pub fn policy_entropy(&self) -> f64 {
        -self
            .policy
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Get epistemic uncertainty of value
    pub fn value_uncertainty(&self) -> f64 {
        self.value_epistemic.variance().sqrt()
    }

    /// Get average epistemic uncertainty of policy
    pub fn policy_uncertainty(&self) -> f64 {
        if self.policy_epistemic.is_empty() {
            return 1.0;
        }

        let total: f64 = self
            .policy_epistemic
            .values()
            .map(|beta| beta.variance().sqrt())
            .sum();
        total / self.policy_epistemic.len() as f64
    }
}

/// Policy-value pair (simplified)
#[derive(Debug, Clone)]
pub struct PolicyValue {
    /// Policy logits (pre-softmax)
    pub policy_logits: Vec<f32>,
    /// Value (tanh output, in [-1, 1])
    pub value: f32,
    /// Optional: Alpha parameter for policy Beta
    pub policy_alpha: Option<Vec<f32>>,
    /// Optional: Beta parameter for policy Beta
    pub policy_beta: Option<Vec<f32>>,
    /// Optional: Alpha for value Beta
    pub value_alpha: Option<f32>,
    /// Optional: Beta for value Beta
    pub value_beta: Option<f32>,
}

impl PolicyValue {
    /// Convert to NeuralEval for a specific game
    pub fn to_eval<G: GameTrait>(&self, state: &G, model_version: u64) -> NeuralEval<G::Action>
    where
        G::Action: Clone,
    {
        // Apply softmax to get probabilities
        let max_logit = self
            .policy_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = self
            .policy_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        // Get legal actions mask
        let mask = state.action_mask();

        // Build policy HashMap
        let mut policy = HashMap::new();
        let mut policy_epistemic = HashMap::new();

        let mut masked_sum = 0.0;
        for i in 0..G::num_actions() {
            if mask[i] > 0.0 {
                masked_sum += exp_logits[i];
            }
        }

        for i in 0..G::num_actions() {
            if let Some(action) = G::index_to_action(i)
                && mask[i] > 0.0
            {
                let prob = (exp_logits[i] / masked_sum.max(1e-8)) as f64;
                policy.insert(action.clone(), prob);

                // Create epistemic from alpha/beta if available
                let beta_conf =
                    if let (Some(alphas), Some(betas)) = (&self.policy_alpha, &self.policy_beta) {
                        BetaConfidence::new(alphas[i] as f64, betas[i] as f64)
                    } else {
                        BetaConfidence::from_confidence(prob, 5.0)
                    };
                policy_epistemic.insert(action, beta_conf);
            }
        }

        // Value epistemic
        let value_epistemic = if let (Some(alpha), Some(beta)) = (self.value_alpha, self.value_beta)
        {
            BetaConfidence::new(alpha as f64, beta as f64)
        } else {
            BetaConfidence::from_confidence((self.value as f64 + 1.0) / 2.0, 10.0)
        };

        NeuralEval {
            policy,
            policy_epistemic,
            value: self.value as f64,
            value_epistemic,
            model_version,
        }
    }
}

// =============================================================================
// Neural Network Trait
// =============================================================================

/// Trait for neural networks
pub trait NeuralNetwork<G: GameTrait>: Send + Sync {
    /// Forward pass: state -> (policy, value)
    fn forward(&self, state: &G) -> PolicyValue;

    /// Batch forward pass
    fn forward_batch(&self, states: &[G]) -> Vec<PolicyValue>;

    /// Get model version hash
    fn version(&self) -> u64;

    /// Get configuration
    fn config(&self) -> &NeuralConfig;
}

/// Dummy network that returns uniform policy
pub struct UniformNetwork {
    config: NeuralConfig,
    version: u64,
    num_actions: usize,
}

impl UniformNetwork {
    pub fn new<G: GameTrait>() -> Self {
        UniformNetwork {
            config: NeuralConfig::default(),
            version: 0,
            num_actions: G::num_actions(),
        }
    }
}

impl<G: GameTrait> NeuralNetwork<G> for UniformNetwork {
    fn forward(&self, _state: &G) -> PolicyValue {
        let logit = 1.0 / self.num_actions as f32;
        PolicyValue {
            policy_logits: vec![logit; self.num_actions],
            value: 0.0,
            policy_alpha: None,
            policy_beta: None,
            value_alpha: None,
            value_beta: None,
        }
    }

    fn forward_batch(&self, states: &[G]) -> Vec<PolicyValue> {
        states.iter().map(|s| self.forward(s)).collect()
    }

    fn version(&self) -> u64 {
        self.version
    }

    fn config(&self) -> &NeuralConfig {
        &self.config
    }
}

// =============================================================================
// Training Data
// =============================================================================

/// A single training example
#[derive(Debug, Clone)]
pub struct TrainingExample<G: GameTrait> {
    /// Game state
    pub state: G,
    /// Target policy (from MCTS visit counts)
    pub policy_target: Vec<f32>,
    /// Target value (game outcome)
    pub value_target: f32,
    /// Weight for this example
    pub weight: f32,
    /// Epistemic: policy variance from MCTS
    pub policy_variance: Option<Vec<f32>>,
    /// Epistemic: value variance from MCTS
    pub value_variance: Option<f32>,
}

/// Loss components
#[derive(Debug, Clone, Default)]
pub struct LossComponents {
    /// Policy cross-entropy loss
    pub policy_loss: f64,
    /// Value MSE loss
    pub value_loss: f64,
    /// Epistemic variance penalty
    pub variance_penalty: f64,
    /// L2 regularization
    pub l2_loss: f64,
    /// Total weighted loss
    pub total_loss: f64,
}

impl LossComponents {
    /// Compute total with weights from config
    pub fn compute_total(&mut self, config: &NeuralConfig) {
        self.total_loss = config.policy_weight * self.policy_loss
            + config.value_weight * self.value_loss
            + config.variance_penalty * self.variance_penalty
            + config.weight_decay * self.l2_loss;
    }
}

/// Compute loss for a batch
pub fn compute_loss<G: GameTrait>(
    predictions: &[PolicyValue],
    targets: &[TrainingExample<G>],
    config: &NeuralConfig,
) -> LossComponents {
    let mut loss = LossComponents::default();
    let n = predictions.len() as f64;

    for (pred, target) in predictions.iter().zip(targets.iter()) {
        // Policy loss: cross-entropy
        let max_logit = pred
            .policy_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = pred
            .policy_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;

        for (i, &target_prob) in target.policy_target.iter().enumerate() {
            if target_prob > 0.0 {
                let log_prob = pred.policy_logits[i] - log_sum_exp;
                loss.policy_loss -= (target_prob as f64) * (log_prob as f64);
            }
        }

        // Value loss: MSE
        let value_diff = pred.value - target.value_target;
        loss.value_loss += (value_diff * value_diff) as f64;

        // Variance penalty: penalize high uncertainty
        if let Some(ref var) = target.value_variance {
            loss.variance_penalty += *var as f64;
        }
    }

    loss.policy_loss /= n;
    loss.value_loss /= n;
    loss.variance_penalty /= n;

    loss.compute_total(config);
    loss
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::game::{TicTacToeAction, TicTacToeState};

    #[test]
    fn test_neural_eval_from_scalars() {
        let mut policy = HashMap::new();
        policy.insert(TicTacToeAction(0), 0.3);
        policy.insert(TicTacToeAction(1), 0.7);

        let eval = NeuralEval::from_scalars(policy, 0.5, 1);

        assert!((eval.value - 0.5).abs() < 0.01);
        assert!(eval.policy_epistemic.contains_key(&TicTacToeAction(0)));
    }

    #[test]
    fn test_policy_entropy() {
        let mut policy = HashMap::new();
        // Uniform distribution has maximum entropy
        for i in 0..9 {
            policy.insert(TicTacToeAction(i), 1.0 / 9.0);
        }

        let eval = NeuralEval::from_scalars(policy, 0.0, 1);
        let entropy = eval.policy_entropy();

        // Entropy of uniform over 9 options = ln(9) ≈ 2.197
        assert!((entropy - 9.0_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_policy_value_to_eval() {
        let state = TicTacToeState::new();
        let pv = PolicyValue {
            policy_logits: vec![1.0; 9],
            value: 0.0,
            policy_alpha: None,
            policy_beta: None,
            value_alpha: None,
            value_beta: None,
        };

        let eval: NeuralEval<TicTacToeAction> = pv.to_eval(&state, 1);

        // All legal, should be roughly uniform
        assert_eq!(eval.policy.len(), 9);
        for &prob in eval.policy.values() {
            assert!((prob - 1.0 / 9.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_uniform_network() {
        let network = UniformNetwork::new::<TicTacToeState>();
        let state = TicTacToeState::new();

        let pv = network.forward(&state);

        assert_eq!(pv.policy_logits.len(), 9);
        assert!((pv.value).abs() < 0.01);
    }

    #[test]
    fn test_epistemic_uncertainty() {
        let mut policy_epistemic = HashMap::new();
        // High uncertainty (low sample size)
        policy_epistemic.insert(TicTacToeAction(0), BetaConfidence::new(1.0, 1.0));
        // Low uncertainty (high sample size)
        policy_epistemic.insert(TicTacToeAction(1), BetaConfidence::new(50.0, 50.0));

        let value_epistemic = BetaConfidence::new(5.0, 5.0);

        let eval = NeuralEval::epistemic(policy_epistemic, value_epistemic, 1);

        // Policy uncertainty should be moderate (average of high and low)
        let uncertainty = eval.policy_uncertainty();
        assert!(uncertainty > 0.0);
        assert!(uncertainty < 0.5);
    }
}

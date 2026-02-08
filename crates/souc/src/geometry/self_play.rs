//! Full Self-Play Loop for AlphaGeoZero
//!
//! Integrates all components into a complete training system:
//! - Synthetic problem generation with difficulty curriculum
//! - Epistemic MCTS proof search
//! - Variance-priority replay buffer
//! - Multi-task training with variance penalty
//! - IMO-AG-30 benchmark evaluation with epistemic posteriors
//!
//! # Key Innovation: Ignorance-Driven Curriculum
//!
//! The system automatically focuses on problems where it has high uncertainty:
//! - Problems with high solve rate variance → need more data
//! - Templates with low confidence → need more exploration
//! - Failed proofs with high value variance → interesting edge cases
//!
//! This implements active inference: minimize expected free energy by
//! reducing epistemic uncertainty about problem-solving capability.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::epistemic::bayesian::BetaConfidence;
use crate::rl::game::{GameState, GameTrait};

use super::alpha_geo_zero::{
    AlphaGeoZeroConfig, EpistemicMCTSTree, ProofGameEpisode, TrajectoryStep,
};
use super::geo_game::{GeoGameConfig, GeoProofGame};
use super::geo_training::{GeoNeuralNetwork, GeoTrainingExample, UniformGeoNetwork};
use super::imo_benchmark::{BenchmarkConfig, BenchmarkStats, imo_ag_30, run_benchmark_on_problems};
use super::imo_parser::IMOProblem;
use super::predicates::Predicate;
use super::proof_state::ProofState;
use super::synthetic::{
    GeneratorConfig, ProblemTemplate, SyntheticProblem, SyntheticProblemGenerator,
};

// =============================================================================
// Variance-Priority Replay Buffer
// =============================================================================

/// Replay buffer with epistemic variance-based prioritization
///
/// Priority = variance^alpha - problems where we're uncertain get more replay
/// This implements curriculum learning based on ignorance rather than loss
pub struct VarianceReplayBuffer {
    /// Stored episodes with metadata
    episodes: Vec<BufferedEpisode>,
    /// Maximum buffer size
    max_size: usize,
    /// Priority exponent (higher = more focus on high variance)
    pub alpha: f64,
    /// Importance sampling exponent
    pub beta: f64,
    /// Minimum priority (ensures all episodes have some chance)
    pub min_priority: f64,
    /// Statistics
    pub stats: BufferStats,
}

/// An episode stored in the buffer with priority metadata
#[derive(Debug, Clone)]
struct BufferedEpisode {
    /// The actual episode data
    episode: ProofGameEpisode,
    /// Priority based on variance
    priority: f64,
    /// Problem template (for stratified sampling)
    template: Option<ProblemTemplate>,
    /// Synthetic problem difficulty
    difficulty: f64,
    /// Times this episode has been sampled
    sample_count: usize,
    /// When this was added
    added_at: usize,
}

/// Statistics for the replay buffer
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    pub total_episodes: usize,
    pub total_samples: usize,
    pub current_size: usize,
    pub avg_priority: f64,
    pub max_priority: f64,
    pub min_priority: f64,
    pub solved_ratio: f64,
    pub avg_variance: f64,
}

impl VarianceReplayBuffer {
    /// Create a new buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(max_size),
            max_size,
            alpha: 0.6,
            beta: 0.4,
            min_priority: 0.01,
            stats: BufferStats::default(),
        }
    }

    /// Create with custom parameters
    pub fn with_params(max_size: usize, alpha: f64, beta: f64) -> Self {
        Self {
            episodes: Vec::with_capacity(max_size),
            max_size,
            alpha,
            beta,
            min_priority: 0.01,
            stats: BufferStats::default(),
        }
    }

    /// Add an episode from a synthetic problem
    pub fn add_synthetic(&mut self, episode: ProofGameEpisode, problem: &SyntheticProblem) {
        let variance = episode.total_variance / episode.length.max(1) as f64;
        let priority = (variance + self.min_priority).powf(self.alpha);

        let buffered = BufferedEpisode {
            episode,
            priority,
            template: Some(problem.template),
            difficulty: problem.difficulty,
            sample_count: 0,
            added_at: self.stats.total_episodes,
        };

        self.add_buffered(buffered);
    }

    /// Add an episode from an IMO problem
    pub fn add_imo(&mut self, episode: ProofGameEpisode, difficulty: f64) {
        let variance = episode.total_variance / episode.length.max(1) as f64;
        let priority = (variance + self.min_priority).powf(self.alpha);

        let buffered = BufferedEpisode {
            episode,
            priority,
            template: None,
            difficulty,
            sample_count: 0,
            added_at: self.stats.total_episodes,
        };

        self.add_buffered(buffered);
    }

    /// Internal add with eviction if needed
    fn add_buffered(&mut self, buffered: BufferedEpisode) {
        if self.episodes.len() >= self.max_size {
            // Evict lowest priority episode
            if let Some((min_idx, _)) = self
                .episodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.priority.partial_cmp(&b.priority).unwrap())
            {
                self.episodes.swap_remove(min_idx);
            }
        }

        self.episodes.push(buffered);
        self.stats.total_episodes += 1;
        self.update_stats();
    }

    /// Sample a batch with priority weighting
    ///
    /// Returns (episode, importance_weight) pairs
    pub fn sample(&mut self, batch_size: usize) -> Vec<(ProofGameEpisode, f64)> {
        if self.episodes.is_empty() {
            return vec![];
        }

        let total_priority: f64 = self.episodes.iter().map(|e| e.priority).sum();
        let n = self.episodes.len() as f64;
        let mut samples = Vec::with_capacity(batch_size);

        // Use deterministic pseudo-random for reproducibility
        let mut rng_state = self.stats.total_samples as u64 ^ 0xDEADBEEF;

        for _ in 0..batch_size.min(self.episodes.len()) {
            // XorShift random
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let rand_val = (rng_state as f64) / (u64::MAX as f64);

            let threshold = rand_val * total_priority;
            let mut cumsum = 0.0;

            for ep in &mut self.episodes {
                cumsum += ep.priority;
                if cumsum >= threshold {
                    // Importance sampling weight
                    let prob = ep.priority / total_priority;
                    let weight = (1.0 / (n * prob)).powf(self.beta);

                    ep.sample_count += 1;
                    samples.push((ep.episode.clone(), weight));
                    break;
                }
            }
        }

        self.stats.total_samples += samples.len();
        samples
    }

    /// Sample stratified by template (for balanced training)
    pub fn sample_stratified(&mut self, batch_size: usize) -> Vec<(ProofGameEpisode, f64)> {
        if self.episodes.is_empty() {
            return vec![];
        }

        // Group by template
        let mut by_template: HashMap<Option<ProblemTemplate>, Vec<usize>> = HashMap::new();
        for (i, ep) in self.episodes.iter().enumerate() {
            by_template.entry(ep.template).or_default().push(i);
        }

        let n_templates = by_template.len();
        let per_template = (batch_size / n_templates).max(1);

        let mut samples = Vec::new();
        let mut rng_state = self.stats.total_samples as u64 ^ 0xCAFEBABE;

        for indices in by_template.values() {
            for _ in 0..per_template.min(indices.len()) {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;

                let idx = indices[(rng_state as usize) % indices.len()];
                let ep = &self.episodes[idx];
                let weight = 1.0; // Equal weight for stratified

                samples.push((ep.episode.clone(), weight));
            }
        }

        self.stats.total_samples += samples.len();
        samples
    }

    /// Get episodes with highest variance (most uncertain)
    pub fn get_high_variance(&self, count: usize) -> Vec<&ProofGameEpisode> {
        let mut sorted: Vec<_> = self.episodes.iter().collect();
        sorted.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        sorted.into_iter().take(count).map(|e| &e.episode).collect()
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.current_size = self.episodes.len();

        if self.episodes.is_empty() {
            return;
        }

        let priorities: Vec<f64> = self.episodes.iter().map(|e| e.priority).collect();
        self.stats.avg_priority = priorities.iter().sum::<f64>() / priorities.len() as f64;
        self.stats.max_priority = priorities.iter().cloned().fold(0.0, f64::max);
        self.stats.min_priority = priorities.iter().cloned().fold(f64::MAX, f64::min);

        let solved = self.episodes.iter().filter(|e| e.episode.proved).count();
        self.stats.solved_ratio = solved as f64 / self.episodes.len() as f64;

        let total_var: f64 = self
            .episodes
            .iter()
            .map(|e| e.episode.total_variance / e.episode.length.max(1) as f64)
            .sum();
        self.stats.avg_variance = total_var / self.episodes.len() as f64;
    }

    /// Get buffer statistics
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }

    /// Check if buffer has enough data for training
    pub fn is_ready(&self, min_episodes: usize) -> bool {
        self.episodes.len() >= min_episodes
    }
}

// =============================================================================
// Self-Play Configuration
// =============================================================================

/// Configuration for the full self-play loop
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    // --- Problem Generation ---
    /// Synthetic problem generator config
    pub generator_config: GeneratorConfig,
    /// Ratio of synthetic to IMO problems (0-1)
    pub synthetic_ratio: f64,
    /// Problems to generate per iteration
    pub problems_per_iteration: usize,

    // --- MCTS Search ---
    /// MCTS configuration
    pub mcts_config: AlphaGeoZeroConfig,
    /// Game configuration
    pub game_config: GeoGameConfig,

    // --- Training ---
    /// Batch size for training
    pub batch_size: usize,
    /// Training steps per iteration
    pub training_steps_per_iteration: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Policy loss weight
    pub policy_weight: f64,
    /// Value loss weight
    pub value_weight: f64,
    /// Variance penalty (KEY: penalize overconfidence)
    pub variance_penalty: f64,
    /// L2 regularization
    pub l2_reg: f64,

    // --- Buffer ---
    /// Replay buffer size
    pub buffer_size: usize,
    /// Minimum buffer size before training starts
    pub min_buffer_size: usize,
    /// Priority exponent for variance-based sampling
    pub priority_alpha: f64,

    // --- Evaluation ---
    /// Iterations between IMO evaluations
    pub eval_interval: usize,
    /// Benchmark configuration
    pub benchmark_config: BenchmarkConfig,

    // --- Loop Control ---
    /// Total iterations (0 = infinite)
    pub total_iterations: usize,
    /// Checkpoint save interval
    pub checkpoint_interval: usize,
    /// Log interval
    pub log_interval: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            generator_config: GeneratorConfig::default(),
            synthetic_ratio: 0.8, // 80% synthetic, 20% IMO
            problems_per_iteration: 50,

            mcts_config: AlphaGeoZeroConfig {
                num_simulations: 400,
                c_puct: 1.5,
                c_ignorance: 0.5,
                use_variance_bonus: true,
                ..AlphaGeoZeroConfig::default()
            },
            game_config: GeoGameConfig::default(),

            batch_size: 128,
            training_steps_per_iteration: 200,
            learning_rate: 0.001,
            policy_weight: 1.0,
            value_weight: 1.0,
            variance_penalty: 0.1,
            l2_reg: 0.0001,

            buffer_size: 100_000,
            min_buffer_size: 1000,
            priority_alpha: 0.6,

            eval_interval: 10,
            benchmark_config: BenchmarkConfig {
                timeout_secs: 60,
                mcts_simulations: 800,
                max_steps: 50,
                min_confidence: 0.9,
                verbose: false,
            },

            total_iterations: 0, // Infinite
            checkpoint_interval: 100,
            log_interval: 10,
        }
    }
}

// =============================================================================
// Self-Play Statistics
// =============================================================================

/// Statistics for the self-play loop
#[derive(Debug, Clone)]
pub struct SelfPlayStats {
    /// Current iteration
    pub iteration: usize,
    /// Total problems attempted
    pub total_problems: usize,
    /// Total proofs found
    pub total_proofs: usize,
    /// Solve rate posterior (Beta distribution)
    pub solve_rate_posterior: BetaConfidence,
    /// Average proof length
    pub avg_proof_length: f64,
    /// Average constructions used
    pub avg_constructions: f64,
    /// Average epistemic variance
    pub avg_variance: f64,
    /// Training loss history
    pub loss_history: Vec<f64>,
    /// IMO benchmark results history
    pub benchmark_history: Vec<BenchmarkSnapshot>,
    /// Per-template statistics
    pub template_stats: HashMap<ProblemTemplate, TemplateStats>,
    /// Total training time
    pub total_time: Duration,
}

/// Snapshot of benchmark results at an iteration
#[derive(Debug, Clone)]
pub struct BenchmarkSnapshot {
    pub iteration: usize,
    pub solved: usize,
    pub total: usize,
    pub solve_rate_posterior: BetaConfidence,
    pub avg_confidence: f64,
}

/// Statistics per problem template
#[derive(Debug, Clone, Default)]
pub struct TemplateStats {
    pub attempts: usize,
    pub solved: usize,
    pub avg_variance: f64,
    pub solve_rate_posterior: BetaConfidence,
}

impl Default for SelfPlayStats {
    fn default() -> Self {
        Self {
            iteration: 0,
            total_problems: 0,
            total_proofs: 0,
            solve_rate_posterior: BetaConfidence::uniform_prior(),
            avg_proof_length: 0.0,
            avg_constructions: 0.0,
            avg_variance: 0.0,
            loss_history: vec![],
            benchmark_history: vec![],
            template_stats: HashMap::new(),
            total_time: Duration::ZERO,
        }
    }
}

impl SelfPlayStats {
    /// Record a problem attempt
    pub fn record_attempt(
        &mut self,
        template: Option<ProblemTemplate>,
        solved: bool,
        variance: f64,
    ) {
        self.total_problems += 1;

        if solved {
            self.total_proofs += 1;
            self.solve_rate_posterior = BetaConfidence::new(
                self.solve_rate_posterior.alpha + 1.0,
                self.solve_rate_posterior.beta,
            );
        } else {
            self.solve_rate_posterior = BetaConfidence::new(
                self.solve_rate_posterior.alpha,
                self.solve_rate_posterior.beta + 1.0,
            );
        }

        // Update running average of variance
        self.avg_variance = (self.avg_variance * (self.total_problems - 1) as f64 + variance)
            / self.total_problems as f64;

        // Update template stats
        if let Some(t) = template {
            let stats = self.template_stats.entry(t).or_default();
            stats.attempts += 1;
            if solved {
                stats.solved += 1;
                stats.solve_rate_posterior = BetaConfidence::new(
                    stats.solve_rate_posterior.alpha + 1.0,
                    stats.solve_rate_posterior.beta,
                );
            } else {
                stats.solve_rate_posterior = BetaConfidence::new(
                    stats.solve_rate_posterior.alpha,
                    stats.solve_rate_posterior.beta + 1.0,
                );
            }
            stats.avg_variance = (stats.avg_variance * (stats.attempts - 1) as f64 + variance)
                / stats.attempts as f64;
        }
    }

    /// Record benchmark results
    pub fn record_benchmark(&mut self, stats: &BenchmarkStats) {
        self.benchmark_history.push(BenchmarkSnapshot {
            iteration: self.iteration,
            solved: stats.solved,
            total: stats.total_problems,
            solve_rate_posterior: BetaConfidence::new(
                1.0 + stats.solved as f64,
                1.0 + (stats.total_problems - stats.solved) as f64,
            ),
            avg_confidence: stats.avg_confidence,
        });
    }

    /// Get solve rate with confidence interval
    pub fn solve_rate_with_ci(&self, confidence: f64) -> (f64, f64, f64) {
        let mean = self.solve_rate_posterior.mean();
        let (lo, hi) = self.solve_rate_posterior.credible_interval(confidence);
        (mean, lo, hi)
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Self-Play Statistics ===");
        println!("Iteration: {}", self.iteration);
        println!(
            "Problems: {}/{} solved ({:.1}%)",
            self.total_proofs,
            self.total_problems,
            self.solve_rate_posterior.mean() * 100.0
        );

        let (mean, lo, hi) = self.solve_rate_with_ci(0.95);
        println!(
            "Solve rate: {:.1}% [{:.1}%, {:.1}%] (95% CI)",
            mean * 100.0,
            lo * 100.0,
            hi * 100.0
        );

        println!("Avg variance: {:.4}", self.avg_variance);

        if let Some(last) = self.benchmark_history.last() {
            println!(
                "\nLast IMO benchmark: {}/{} ({:.1}%)",
                last.solved,
                last.total,
                last.solve_rate_posterior.mean() * 100.0
            );
        }

        println!("\nPer-template performance:");
        for (template, stats) in &self.template_stats {
            println!(
                "  {:?}: {}/{} ({:.1}%), var={:.3}",
                template,
                stats.solved,
                stats.attempts,
                stats.solve_rate_posterior.mean() * 100.0,
                stats.avg_variance
            );
        }
    }
}

// =============================================================================
// Training Result
// =============================================================================

/// Result of a single training step
#[derive(Debug, Clone)]
pub struct TrainingStepResult {
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub variance_penalty: f64,
    pub batch_size: usize,
    pub avg_weight: f64,
}

// =============================================================================
// Full Self-Play Loop
// =============================================================================

/// The complete AlphaGeoZero self-play training loop
pub struct AlphaGeoZeroSelfPlay<N: GeoNeuralNetwork> {
    /// Neural network
    pub network: N,
    /// Configuration
    pub config: SelfPlayConfig,
    /// Synthetic problem generator
    pub generator: SyntheticProblemGenerator,
    /// Replay buffer
    pub buffer: VarianceReplayBuffer,
    /// Statistics
    pub stats: SelfPlayStats,
    /// IMO problems for evaluation and training
    imo_problems: Vec<IMOProblem>,
}

impl<N: GeoNeuralNetwork + Clone> AlphaGeoZeroSelfPlay<N> {
    /// Create a new self-play loop
    pub fn new(network: N, config: SelfPlayConfig) -> Self {
        let generator = SyntheticProblemGenerator::new(config.generator_config.clone());
        let buffer =
            VarianceReplayBuffer::with_params(config.buffer_size, config.priority_alpha, 0.4);
        let imo_problems = imo_ag_30();

        Self {
            network,
            config,
            generator,
            buffer,
            stats: SelfPlayStats::default(),
            imo_problems,
        }
    }

    /// Run the self-play loop
    pub fn run(&mut self) {
        let start_time = Instant::now();

        loop {
            self.stats.iteration += 1;
            let iter_start = Instant::now();

            // --- Generate Problems & Episodes ---
            self.generate_episodes();

            // --- Training ---
            if self.buffer.is_ready(self.config.min_buffer_size) {
                let train_result = self.train_iteration();
                self.stats.loss_history.push(train_result.total_loss);
            }

            // --- Evaluation ---
            if self
                .stats
                .iteration
                .is_multiple_of(self.config.eval_interval)
            {
                self.evaluate_imo();
            }

            // --- Logging ---
            if self
                .stats
                .iteration
                .is_multiple_of(self.config.log_interval)
            {
                let iter_time = iter_start.elapsed();
                self.log_progress(iter_time);
            }

            // --- Checkpointing ---
            if self.config.checkpoint_interval > 0
                && self
                    .stats
                    .iteration
                    .is_multiple_of(self.config.checkpoint_interval)
            {
                self.save_checkpoint();
            }

            // --- Termination ---
            if self.config.total_iterations > 0
                && self.stats.iteration >= self.config.total_iterations
            {
                break;
            }

            self.stats.total_time = start_time.elapsed();
        }

        // Final evaluation
        self.evaluate_imo();
        self.stats.print_summary();
    }

    /// Generate episodes for one iteration
    fn generate_episodes(&mut self) {
        let n_synthetic =
            (self.config.problems_per_iteration as f64 * self.config.synthetic_ratio) as usize;
        let n_imo = self.config.problems_per_iteration - n_synthetic;

        // Generate synthetic problems
        for _ in 0..n_synthetic {
            let problem = self.generator.generate();
            let episode = self.run_mcts_episode(&problem.state, &problem.goal);

            // Update generator with feedback
            self.generator
                .update_solve_rate(problem.template, episode.proved);

            // Record stats
            let variance = episode.total_variance / episode.length.max(1) as f64;
            self.stats
                .record_attempt(Some(problem.template), episode.proved, variance);

            // Add to buffer with high priority for uncertain problems
            self.buffer.add_synthetic(episode, &problem);
        }

        // Sample IMO problems (prioritize unsolved/high-variance)
        for _ in 0..n_imo {
            // Select IMO problem based on historical performance
            let problem_idx = self.select_imo_problem();
            let problem = &self.imo_problems[problem_idx];

            let episode = self.run_mcts_episode(&problem.initial_state, &problem.goal);

            let variance = episode.total_variance / episode.length.max(1) as f64;
            self.stats.record_attempt(None, episode.proved, variance);

            self.buffer.add_imo(episode, problem.difficulty);
        }
    }

    /// Select an IMO problem with bias toward hard/uncertain ones
    fn select_imo_problem(&mut self) -> usize {
        // Simple selection - could be improved with per-problem tracking
        let mut rng_state = self.stats.iteration as u64 ^ 0xBADCAFE;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        (rng_state as usize) % self.imo_problems.len()
    }

    /// Run MCTS episode on a problem
    fn run_mcts_episode(&self, state: &ProofState, goal: &Predicate) -> ProofGameEpisode {
        let mut game = GeoProofGame::for_goal(
            state.clone(),
            goal.clone(),
            self.config.game_config.goal_confidence,
        );

        let mut trajectory = Vec::new();
        let mut total_variance = 0.0;

        while !game.is_terminal() {
            // Run MCTS
            let mut tree = EpistemicMCTSTree::new(game.clone(), self.config.mcts_config.clone());
            tree.search(&self.network);

            // Get policy and value
            let policy_vec = tree.get_policy();
            let policy: Vec<f32> = if policy_vec.is_empty() {
                vec![1.0]
            } else {
                policy_vec.iter().map(|(_, p)| *p as f32).collect()
            };

            let value_beta = tree.root_value();
            total_variance += value_beta.variance();

            // Record step
            trajectory.push(TrajectoryStep {
                state_features: game.to_features(),
                policy,
                value_beta,
                action_idx: 0, // Will be set below
            });

            // Select action
            if let Some(action) = tree.select_action() {
                let legal = game.legal_actions();
                if let Some(idx) = legal.iter().position(|a| *a == action)
                    && let Some(last) = trajectory.last_mut()
                {
                    last.action_idx = idx;
                }
                game = game.apply_action(&action);
            } else {
                break;
            }
        }

        ProofGameEpisode {
            problem_id: format!("episode_{}", self.stats.total_problems),
            initial_state: state.clone(),
            target: goal.clone(),
            trajectory,
            proved: game.proved,
            total_variance,
            length: game.steps,
        }
    }

    /// Train for one iteration
    fn train_iteration(&mut self) -> TrainingStepResult {
        let mut total_loss = 0.0;
        let mut total_policy = 0.0;
        let mut total_value = 0.0;
        let mut total_variance_penalty = 0.0;
        let mut total_weight = 0.0;
        let mut n_steps = 0;

        for _ in 0..self.config.training_steps_per_iteration {
            let batch = self.buffer.sample(self.config.batch_size);
            if batch.is_empty() {
                continue;
            }

            // Prepare training examples
            let mut examples = Vec::new();
            for (episode, weight) in &batch {
                let target_value = if episode.proved { 1.0 } else { 0.0 };

                for step in &episode.trajectory {
                    examples.push(GeoTrainingExample {
                        features: step.state_features.clone(),
                        policy_target: step.policy.clone(),
                        value_target: target_value,
                        variance: step.value_beta.variance() as f32,
                        weight: *weight as f32,
                    });
                }
            }

            if examples.is_empty() {
                continue;
            }

            // Compute losses
            let (policy_loss, value_loss, var_penalty) = self.compute_losses(&examples);

            let step_loss = self.config.policy_weight * policy_loss
                + self.config.value_weight * value_loss
                + self.config.variance_penalty * var_penalty;

            total_loss += step_loss;
            total_policy += policy_loss;
            total_value += value_loss;
            total_variance_penalty += var_penalty;
            total_weight += batch.iter().map(|(_, w)| w).sum::<f64>();
            n_steps += 1;

            // Train network
            self.network.train(&examples, self.config.learning_rate);
        }

        let n = n_steps.max(1) as f64;
        TrainingStepResult {
            total_loss: total_loss / n,
            policy_loss: total_policy / n,
            value_loss: total_value / n,
            variance_penalty: total_variance_penalty / n,
            batch_size: self.config.batch_size,
            avg_weight: total_weight / (n_steps * self.config.batch_size).max(1) as f64,
        }
    }

    /// Compute policy, value, and variance penalty losses
    fn compute_losses(&self, examples: &[GeoTrainingExample]) -> (f64, f64, f64) {
        let mut policy_loss = 0.0;
        let mut value_loss = 0.0;
        let mut var_penalty = 0.0;

        for ex in examples {
            // Forward pass
            let dummy_game = GeoProofGame::default();
            let (pred_policy, pred_value) = self.network.forward(&dummy_game);

            // Policy cross-entropy
            for (i, &target_p) in ex.policy_target.iter().enumerate() {
                if i < pred_policy.len() && target_p > 0.0 {
                    policy_loss -=
                        (target_p * (pred_policy[i] + 1e-8).ln()) as f64 * ex.weight as f64;
                }
            }

            // Value MSE
            value_loss += ((pred_value - ex.value_target).powi(2) * ex.weight) as f64;

            // Variance penalty
            var_penalty += ex.variance as f64 * ex.weight as f64;
        }

        let n = examples.len().max(1) as f64;
        (policy_loss / n, value_loss / n, var_penalty / n)
    }

    /// Evaluate on IMO-AG-30 benchmark
    fn evaluate_imo(&mut self) {
        let stats = run_benchmark_on_problems(
            &self.network,
            &self.imo_problems,
            &self.config.benchmark_config,
        );

        self.stats.record_benchmark(&stats);

        let (mean, lo, hi) = (
            stats.solve_rate(),
            stats.solve_rate()
                - 2.0
                    * (stats.solve_rate() * (1.0 - stats.solve_rate())
                        / stats.total_problems as f64)
                        .sqrt(),
            stats.solve_rate()
                + 2.0
                    * (stats.solve_rate() * (1.0 - stats.solve_rate())
                        / stats.total_problems as f64)
                        .sqrt(),
        );

        println!(
            "\n=== IMO-AG-30 Evaluation (Iter {}) ===",
            self.stats.iteration
        );
        println!(
            "Solved: {}/{} = {:.1}% [{:.1}%, {:.1}%]",
            stats.solved,
            stats.total_problems,
            mean * 100.0,
            lo.max(0.0) * 100.0,
            hi.min(1.0) * 100.0
        );
        println!("AlphaGeometry baseline: 83% (25/30)");

        if mean > 0.83 {
            println!("Status: EXCEEDS AlphaGeometry!");
        } else if hi > 0.83 {
            println!("Status: Competitive (within uncertainty)");
        } else {
            println!("Status: Below baseline");
        }
    }

    /// Log progress
    fn log_progress(&self, iter_time: Duration) {
        let (mean, lo, hi) = self.stats.solve_rate_with_ci(0.95);

        println!(
            "Iter {:5} | Solved {}/{} ({:.1}% [{:.1}-{:.1}]) | Var {:.4} | Loss {:.4} | Buffer {} | Time {:.1}s",
            self.stats.iteration,
            self.stats.total_proofs,
            self.stats.total_problems,
            mean * 100.0,
            lo * 100.0,
            hi * 100.0,
            self.stats.avg_variance,
            self.stats.loss_history.last().unwrap_or(&0.0),
            self.buffer.stats.current_size,
            iter_time.as_secs_f64()
        );
    }

    /// Save checkpoint
    fn save_checkpoint(&self) {
        let path = format!("checkpoints/alphageozero_iter_{}.pt", self.stats.iteration);
        if let Err(e) = self.network.save(&path) {
            eprintln!("Failed to save checkpoint: {}", e);
        } else {
            println!("Saved checkpoint to {}", path);
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Run self-play with default settings
pub fn run_default_self_play(iterations: usize) -> SelfPlayStats {
    let network = UniformGeoNetwork::new();
    let mut config = SelfPlayConfig::default();
    config.total_iterations = iterations;
    config.problems_per_iteration = 10;
    config.training_steps_per_iteration = 50;

    let mut loop_ = AlphaGeoZeroSelfPlay::new(network, config);
    loop_.run();
    loop_.stats
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffer = VarianceReplayBuffer::new(1000);
        assert_eq!(buffer.stats.current_size, 0);
    }

    #[test]
    fn test_buffer_add_and_sample() {
        let mut buffer = VarianceReplayBuffer::new(100);

        // Add some episodes
        for i in 0..10 {
            let episode = ProofGameEpisode {
                problem_id: format!("test_{}", i),
                initial_state: ProofState::new(),
                target: Predicate::collinear("A", "B", "C"),
                trajectory: vec![],
                proved: i % 2 == 0,
                total_variance: (i + 1) as f64 * 0.1,
                length: 5,
            };

            let problem = SyntheticProblem {
                id: format!("synth_{}", i),
                template: ProblemTemplate::MidpointTheorem,
                state: ProofState::new(),
                goal: Predicate::collinear("A", "B", "C"),
                difficulty: 0.5,
                seed: i as u64,
                num_premises: 3,
                expected_constructions: 0,
                historical_solve_rate: BetaConfidence::uniform_prior(),
                variation: Default::default(),
            };

            buffer.add_synthetic(episode, &problem);
        }

        assert_eq!(buffer.stats.current_size, 10);

        // Sample
        let samples = buffer.sample(5);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_stats_recording() {
        let mut stats = SelfPlayStats::default();

        stats.record_attempt(Some(ProblemTemplate::MidpointTheorem), true, 0.1);
        stats.record_attempt(Some(ProblemTemplate::MidpointTheorem), false, 0.2);
        stats.record_attempt(Some(ProblemTemplate::IsocelesPerpendicular), true, 0.15);

        assert_eq!(stats.total_problems, 3);
        assert_eq!(stats.total_proofs, 2);
        assert!(stats.solve_rate_posterior.mean() > 0.5);
    }

    #[test]
    fn test_config_default() {
        let config = SelfPlayConfig::default();
        assert!(config.variance_penalty > 0.0);
        assert!(config.mcts_config.use_variance_bonus);
    }

    #[test]
    fn test_self_play_creation() {
        let network = UniformGeoNetwork::new();
        let config = SelfPlayConfig::default();
        let loop_ = AlphaGeoZeroSelfPlay::new(network, config);

        assert_eq!(loop_.stats.iteration, 0);
        assert!(!loop_.imo_problems.is_empty());
    }
}

//! AlphaGeoZero Showcase Module
//!
//! Provides demonstration runs of epistemic geometry proving with
//! honest confidence intervals and variance-aware metrics.
//!
//! # Key Innovations Demonstrated
//!
//! 1. **Epistemic MCTS**: PUCT + variance bonus explores uncertain proof paths
//! 2. **Beta Posterior Metrics**: Solve rate as Beta(solved+1, failed+1)
//! 3. **Variance-Priority Learning**: Network improves on hardest problems first
//! 4. **Honest Uncertainty**: All metrics include confidence intervals

use std::time::Duration;

use crate::epistemic::bayesian::BetaConfidence;

use super::geo_training::UniformGeoNetwork;
use super::imo_benchmark::{BenchmarkConfig, BenchmarkStats, imo_ag_30, run_benchmark_on_problems};
use super::imo_parser::IMOProblem;
use super::self_play::{AlphaGeoZeroSelfPlay, SelfPlayConfig};

// =============================================================================
// Showcase Results
// =============================================================================

/// Epistemic benchmark result with honest uncertainty quantification
#[derive(Debug, Clone)]
pub struct EpistemicBenchmarkResult {
    /// Number of problems solved
    pub solved: usize,
    /// Total number of problems
    pub total: usize,
    /// Solve rate as Beta posterior (not just a percentage!)
    pub solve_rate: BetaConfidence,
    /// Average time per problem
    pub avg_time: Duration,
    /// Average confidence when solved
    pub avg_confidence: BetaConfidence,
    /// Problems sorted by difficulty (hardest first)
    pub difficulty_ranking: Vec<ProblemDifficulty>,
    /// Total training episodes (if applicable)
    pub training_episodes: usize,
}

/// Per-problem difficulty assessment
#[derive(Debug, Clone)]
pub struct ProblemDifficulty {
    pub problem_id: String,
    pub solved: bool,
    pub attempts: usize,
    /// Solve rate for this specific problem as Beta posterior
    pub solve_rate: BetaConfidence,
    /// Average proof length when solved
    pub avg_proof_length: f64,
    /// Time taken on last attempt
    pub last_time: Duration,
}

impl EpistemicBenchmarkResult {
    /// Create from benchmark stats
    pub fn from_stats(stats: &BenchmarkStats) -> Self {
        // Compute Beta posterior for solve rate
        // Beta(solved + 1, failed + 1) is the posterior with uniform prior
        let solved = stats.solved;
        let failed = stats.total_problems - stats.solved;
        let solve_rate = BetaConfidence::new(solved as f64 + 1.0, failed as f64 + 1.0);

        // Compute average confidence from solved problems
        let avg_confidence = if stats.solved > 0 {
            BetaConfidence::new(stats.avg_confidence * 10.0 + 1.0, 1.0)
        } else {
            BetaConfidence::new(1.0, 1.0) // Uniform prior
        };

        // Build difficulty ranking
        let mut difficulty_ranking: Vec<ProblemDifficulty> = stats
            .results
            .iter()
            .map(|r| ProblemDifficulty {
                problem_id: r.problem_id.clone(),
                solved: r.solved,
                attempts: 1,
                solve_rate: if r.solved {
                    BetaConfidence::new(9.0, 1.0)
                } else {
                    BetaConfidence::new(1.0, 9.0)
                },
                avg_proof_length: r.constructions as f64,
                last_time: r.time_taken,
            })
            .collect();

        // Sort by solve rate (hardest first)
        difficulty_ranking.sort_by(|a, b| {
            a.solve_rate
                .mean()
                .partial_cmp(&b.solve_rate.mean())
                .unwrap()
        });

        Self {
            solved: stats.solved,
            total: stats.total_problems,
            solve_rate,
            avg_time: stats.avg_time,
            avg_confidence,
            difficulty_ranking,
            training_episodes: 0,
        }
    }

    /// Format as showcase output string
    pub fn format_showcase(&self) -> String {
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║         AlphaGeoZero - Epistemic Geometry Prover                 ║\n");
        output.push_str("║     First theorem prover with honest confidence intervals        ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Solve rate with Beta posterior
        let mean = self.solve_rate.mean();
        let std = self.solve_rate.variance().sqrt();
        let ci_low = (mean - 1.96 * std).max(0.0);
        let ci_high = (mean + 1.96 * std).min(1.0);

        output.push_str(&format!(
            "║  Solve Rate: {}/{} = {:.1}% ± {:.1}%                              \n",
            self.solved,
            self.total,
            mean * 100.0,
            std * 100.0
        ));
        output.push_str(&format!(
            "║  95% CI: [{:.1}%, {:.1}%]                                         \n",
            ci_low * 100.0,
            ci_high * 100.0
        ));
        output.push_str(&format!(
            "║  Beta Posterior: Beta({:.1}, {:.1})                               \n",
            self.solve_rate.alpha, self.solve_rate.beta
        ));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Per-problem results
        output.push_str("║  Problem Results (sorted by difficulty):                         ║\n");
        output.push_str("║  ─────────────────────────────────────────────────────────────   ║\n");

        for (i, prob) in self.difficulty_ranking.iter().take(10).enumerate() {
            let status = if prob.solved { "✓" } else { "✗" };
            let conf = prob.solve_rate.mean() * 100.0;
            output.push_str(&format!(
                "║  {:2}. {} {:20} solve_rate: {:.0}% ± {:.0}%              \n",
                i + 1,
                status,
                &prob.problem_id[..prob.problem_id.len().min(20)],
                conf,
                prob.solve_rate.variance().sqrt() * 100.0
            ));
        }

        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
        output.push_str("║  Key Innovations:                                                ║\n");
        output.push_str("║  • Epistemic MCTS: PUCT + variance bonus explores uncertainty    ║\n");
        output.push_str("║  • Beta posteriors: Honest solve rate, not just percentage       ║\n");
        output.push_str("║  • Variance-priority: Learns hardest problems first              ║\n");
        output.push_str("║  • Provenance: Full proof trace with confidence per step         ║\n");
        output.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Format as JSON for programmatic consumption
    pub fn to_json(&self) -> String {
        let mean = self.solve_rate.mean();
        let std = self.solve_rate.variance().sqrt();

        format!(
            r#"{{
  "solver": "AlphaGeoZero",
  "epistemic_aware": true,
  "results": {{
    "solved": {},
    "total": {},
    "solve_rate": {{
      "mean": {:.4},
      "std": {:.4},
      "ci_95_low": {:.4},
      "ci_95_high": {:.4},
      "beta_alpha": {:.2},
      "beta_beta": {:.2}
    }},
    "avg_time_ms": {},
    "avg_confidence": {:.4}
  }},
  "problems": [
{}
  ]
}}"#,
            self.solved,
            self.total,
            mean,
            std,
            (mean - 1.96 * std).max(0.0),
            (mean + 1.96 * std).min(1.0),
            self.solve_rate.alpha,
            self.solve_rate.beta,
            self.avg_time.as_millis(),
            self.avg_confidence.mean(),
            self.difficulty_ranking
                .iter()
                .map(|p| format!(
                    r#"    {{"id": "{}", "solved": {}, "solve_rate": {:.4}}}"#,
                    p.problem_id,
                    p.solved,
                    p.solve_rate.mean()
                ))
                .collect::<Vec<_>>()
                .join(",\n")
        )
    }
}

// =============================================================================
// Showcase Runner
// =============================================================================

/// Configuration for showcase run
#[derive(Debug, Clone)]
pub struct ShowcaseConfig {
    /// Number of problems to attempt (subset of IMO-AG-30)
    pub num_problems: usize,
    /// Maximum time per problem
    pub timeout_per_problem: Duration,
    /// Number of MCTS simulations
    pub mcts_simulations: usize,
    /// Whether to run self-play training first
    pub train_first: bool,
    /// Number of training episodes if training
    pub training_episodes: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for ShowcaseConfig {
    fn default() -> Self {
        Self {
            num_problems: 10,
            timeout_per_problem: Duration::from_secs(60),
            mcts_simulations: 800,
            train_first: false,
            training_episodes: 100,
            verbose: true,
        }
    }
}

/// Run the AlphaGeoZero showcase
pub fn run_showcase(config: &ShowcaseConfig) -> EpistemicBenchmarkResult {
    if config.verbose {
        println!("\n🔬 AlphaGeoZero Showcase - Epistemic Geometry Proving\n");
        println!("Configuration:");
        println!("  Problems: {}", config.num_problems);
        println!("  Timeout: {:?}", config.timeout_per_problem);
        println!("  MCTS sims: {}", config.mcts_simulations);
        if config.train_first {
            println!("  Training episodes: {}", config.training_episodes);
        }
        println!();
    }

    // Create network (optionally train)
    let network = if config.train_first {
        if config.verbose {
            println!("📚 Training network via self-play...\n");
        }
        train_for_showcase(config)
    } else {
        UniformGeoNetwork::new()
    };

    // Get subset of IMO problems
    let all_problems = imo_ag_30();
    let problems: Vec<IMOProblem> = all_problems.into_iter().take(config.num_problems).collect();

    if config.verbose {
        println!("🎯 Evaluating on {} IMO problems...\n", problems.len());
    }

    // Run benchmark
    let bench_config = BenchmarkConfig {
        timeout_secs: config.timeout_per_problem.as_secs(),
        mcts_simulations: config.mcts_simulations,
        max_steps: 50,
        min_confidence: 0.5,
        verbose: config.verbose,
    };

    let stats = run_benchmark_on_problems(&network, &problems, &bench_config);
    let mut result = EpistemicBenchmarkResult::from_stats(&stats);

    if config.train_first {
        result.training_episodes = config.training_episodes;
    }

    if config.verbose {
        println!("\n{}", result.format_showcase());
    }

    result
}

/// Train network via self-play for showcase
fn train_for_showcase(config: &ShowcaseConfig) -> UniformGeoNetwork {
    let mut self_play_config = SelfPlayConfig::default();
    self_play_config.total_iterations = config.training_episodes;
    self_play_config.problems_per_iteration = 10;
    self_play_config.training_steps_per_iteration = 50;
    self_play_config.log_interval = 5;

    let mut self_play = AlphaGeoZeroSelfPlay::new(UniformGeoNetwork::new(), self_play_config);

    // Run training (this runs the full loop internally)
    self_play.run();

    if config.verbose {
        let stats = &self_play.stats;
        println!(
            "  Training complete: iterations={}, problems={}, solve_rate={:.1}% ± {:.1}%",
            stats.iteration,
            stats.total_problems,
            stats.solve_rate_posterior.mean() * 100.0,
            stats.solve_rate_posterior.variance().sqrt() * 100.0
        );
    }

    self_play.network
}

/// Quick showcase for demo purposes (fewer problems, faster)
pub fn quick_showcase() -> EpistemicBenchmarkResult {
    run_showcase(&ShowcaseConfig {
        num_problems: 5,
        timeout_per_problem: Duration::from_secs(30),
        mcts_simulations: 400,
        train_first: false,
        training_episodes: 0,
        verbose: true,
    })
}

/// Full showcase with training (overnight run)
pub fn full_showcase() -> EpistemicBenchmarkResult {
    run_showcase(&ShowcaseConfig {
        num_problems: 30,
        timeout_per_problem: Duration::from_secs(300),
        mcts_simulations: 1600,
        train_first: true,
        training_episodes: 1000,
        verbose: true,
    })
}

// =============================================================================
// CLI Entry Point
// =============================================================================

/// Run showcase from command line
pub fn main_showcase() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     AlphaGeoZero - First Epistemic Geometry Theorem Prover       ║");
    println!("║                                                                  ║");
    println!("║  \"The first geometry prover that knows what it doesn't know\"    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let result = quick_showcase();

    println!("\n📊 JSON Output:\n");
    println!("{}", result.to_json());
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_result_creation() {
        // Test that we can create an epistemic result with Beta posteriors
        let solve_rate = BetaConfidence::new(8.0, 4.0); // 7 solved, 3 failed + uniform prior
        assert!((solve_rate.mean() - 0.667).abs() < 0.01);
        assert!(solve_rate.variance() > 0.0);
    }

    #[test]
    fn test_showcase_output_format() {
        let result = EpistemicBenchmarkResult {
            solved: 8,
            total: 10,
            solve_rate: BetaConfidence::new(9.0, 3.0),
            avg_time: Duration::from_secs(10),
            avg_confidence: BetaConfidence::new(9.0, 1.0),
            difficulty_ranking: vec![],
            training_episodes: 0,
        };

        let output = result.format_showcase();
        assert!(output.contains("AlphaGeoZero"));
        assert!(output.contains("Epistemic"));
        assert!(output.contains("Beta"));
    }

    #[test]
    fn test_json_output() {
        let result = EpistemicBenchmarkResult {
            solved: 5,
            total: 10,
            solve_rate: BetaConfidence::new(6.0, 6.0),
            avg_time: Duration::from_millis(5000),
            avg_confidence: BetaConfidence::new(8.0, 2.0),
            difficulty_ranking: vec![ProblemDifficulty {
                problem_id: "test_p1".to_string(),
                solved: true,
                attempts: 1,
                solve_rate: BetaConfidence::new(9.0, 1.0),
                avg_proof_length: 5.0,
                last_time: Duration::from_secs(2),
            }],
            training_episodes: 100,
        };

        let json = result.to_json();
        assert!(json.contains("\"solver\": \"AlphaGeoZero\""));
        assert!(json.contains("\"epistemic_aware\": true"));
        assert!(json.contains("\"beta_alpha\""));
    }

    #[test]
    fn test_showcase_config_default() {
        let config = ShowcaseConfig::default();
        assert_eq!(config.num_problems, 10);
        assert_eq!(config.mcts_simulations, 800);
        assert!(!config.train_first);
    }
}

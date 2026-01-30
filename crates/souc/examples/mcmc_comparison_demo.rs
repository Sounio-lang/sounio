//! MCMC Sampler Comparison Demo
//!
//! Demonstrates Task 1.4: MCMC methods for Bayesian epistemic inference.
//!
//! This example compares three MCMC algorithms:
//! 1. Metropolis-Hastings (MH): Random walk with adaptive proposals
//! 2. Hamiltonian Monte Carlo (HMC): Gradient-guided exploration
//! 3. NUTS: No-U-Turn Sampler with automatic tuning
//!
//! We sample from various posterior distributions and compare:
//! - Acceptance rates
//! - Effective sample size (ESS)
//! - Posterior accuracy
//! - Computational efficiency

use sounio::epistemic::mcmc::{compute_r_hat, HamiltonianMC, MCMCConfig, MetropolisHastings, NUTS};

fn main() {
    println!("=== MCMC Sampler Comparison for Bayesian Inference ===\n");

    // Example 1: Univariate normal posterior
    demo_univariate_normal();

    // Example 2: Bivariate correlated normal
    demo_bivariate_correlated();

    // Example 3: Non-Gaussian posterior (mixture)
    demo_mixture_posterior();

    // Example 4: Hierarchical model
    demo_hierarchical_model();

    // Example 5: Multiple chains for convergence diagnostics
    demo_multiple_chains();
}

fn demo_univariate_normal() {
    println!("--- Example 1: Univariate Normal Posterior ---");
    println!("Posterior: p(θ|data) ∝ N(θ; μ=2, σ²=1)");
    println!("True posterior: N(2, 1)\n");

    // Log-posterior: p(θ|data) ∝ exp(-0.5(θ-2)²)
    let log_posterior = |theta: &[f64]| -0.5 * (theta[0] - 2.0).powi(2);
    let grad_log_posterior = |theta: &[f64]| vec![-(theta[0] - 2.0)];

    let n_samples = 5000;
    let initial = vec![0.0];

    // Metropolis-Hastings
    println!("Metropolis-Hastings:");
    let config_mh = MCMCConfig::default();
    let mut mh = MetropolisHastings::new(config_mh, 1);
    let result_mh = mh.sample(&log_posterior, &initial, n_samples);
    print_results("MH", &result_mh, 2.0, 1.0);

    // Hamiltonian Monte Carlo
    println!("\nHamiltonian Monte Carlo:");
    let config_hmc = MCMCConfig::for_hmc();
    let mut hmc = HamiltonianMC::new(config_hmc, 1);
    let result_hmc = hmc.sample(&log_posterior, &grad_log_posterior, &initial, n_samples);
    print_results("HMC", &result_hmc, 2.0, 1.0);

    // NUTS
    println!("\nNUTS (No-U-Turn Sampler):");
    let config_nuts = MCMCConfig::for_nuts();
    let mut nuts = NUTS::new(config_nuts, 1);
    let result_nuts = nuts.sample(&log_posterior, &grad_log_posterior, &initial, n_samples);
    print_results("NUTS", &result_nuts, 2.0, 1.0);

    println!("\n{}", "=".repeat(70));
    println!();
}

fn demo_bivariate_correlated() {
    println!("--- Example 2: Bivariate Correlated Normal ---");
    println!("Posterior: N([1, 2], [[1, 0.7], [0.7, 1]])");
    println!("Correlation ρ = 0.7 makes it challenging for random walk\n");

    // Precision matrix (inverse covariance)
    // For Σ = [[1, 0.7], [0.7, 1]], det=0.51, Σ⁻¹ = [[1.96, -1.37], [-1.37, 1.96]]
    let log_posterior = |theta: &[f64]| {
        let x = theta[0] - 1.0;
        let y = theta[1] - 2.0;
        -0.5 * (1.96 * x * x - 2.0 * 1.37 * x * y + 1.96 * y * y)
    };

    let grad_log_posterior = |theta: &[f64]| {
        let x = theta[0] - 1.0;
        let y = theta[1] - 2.0;
        vec![-(1.96 * x - 1.37 * y), -(-1.37 * x + 1.96 * y)]
    };

    let n_samples = 5000;
    let initial = vec![0.0, 0.0];

    // Metropolis-Hastings
    println!("Metropolis-Hastings:");
    let mut config_mh = MCMCConfig::default();
    config_mh.initial_step_size = 0.3; // Tune for correlation
    let mut mh = MetropolisHastings::new(config_mh, 2);
    let result_mh = mh.sample(&log_posterior, &initial, n_samples);
    println!(
        "  Posterior mean: [{:.3}, {:.3}] (true: [1, 2])",
        result_mh.mean()[0],
        result_mh.mean()[1]
    );
    println!("  Acceptance rate: {:.3}", result_mh.accept_rate);
    println!("  ESS: [{:.1}, {:.1}]", result_mh.ess[0], result_mh.ess[1]);

    // HMC
    println!("\nHamiltonian Monte Carlo:");
    let config_hmc = MCMCConfig::for_hmc();
    let mut hmc = HamiltonianMC::new(config_hmc, 2);
    let result_hmc = hmc.sample(&log_posterior, &grad_log_posterior, &initial, n_samples);
    println!(
        "  Posterior mean: [{:.3}, {:.3}] (true: [1, 2])",
        result_hmc.mean()[0],
        result_hmc.mean()[1]
    );
    println!("  Acceptance rate: {:.3}", result_hmc.accept_rate);
    println!(
        "  ESS: [{:.1}, {:.1}]",
        result_hmc.ess[0], result_hmc.ess[1]
    );

    // NUTS
    println!("\nNUTS:");
    let config_nuts = MCMCConfig::for_nuts();
    let mut nuts = NUTS::new(config_nuts, 2);
    let result_nuts = nuts.sample(&log_posterior, &grad_log_posterior, &initial, n_samples);
    println!(
        "  Posterior mean: [{:.3}, {:.3}] (true: [1, 2])",
        result_nuts.mean()[0],
        result_nuts.mean()[1]
    );
    println!("  Acceptance rate: {:.3}", result_nuts.accept_rate);
    println!(
        "  ESS: [{:.1}, {:.1}]",
        result_nuts.ess[0], result_nuts.ess[1]
    );

    println!("\nConclusion:");
    println!("  HMC and NUTS explore correlated spaces more efficiently than MH");
    println!("  NUTS adapts trajectory length automatically\n");

    println!("{}", "=".repeat(70));
    println!();
}

fn demo_mixture_posterior() {
    println!("--- Example 3: Non-Gaussian Mixture Posterior ---");
    println!("Posterior: 0.6*N(-2, 0.5²) + 0.4*N(3, 0.8²)");
    println!("Multimodal distribution tests sampler exploration\n");

    let log_posterior = |theta: &[f64]| {
        let x = theta[0];

        // Log of mixture: log(0.6*N1 + 0.4*N2)
        let log_n1 = -0.5 * ((x + 2.0) / 0.5).powi(2)
            - libm::log(0.5)
            - 0.5 * libm::log(2.0 * 3.14159265359);
        let log_n2 = -0.5 * ((x - 3.0) / 0.8).powi(2)
            - libm::log(0.8)
            - 0.5 * libm::log(2.0 * 3.14159265359);
        let log_w1 = libm::log(0.6);
        let log_w2 = libm::log(0.4);

        // log(a*exp(x) + b*exp(y)) = max(x,y) + log(a*exp(x-max) + b*exp(y-max))
        let max_log = if log_w1 + log_n1 > log_w2 + log_n2 {
            log_w1 + log_n1
        } else {
            log_w2 + log_n2
        };

        max_log
            + libm::log(libm::exp(log_w1 + log_n1 - max_log) + libm::exp(log_w2 + log_n2 - max_log))
    };

    // Numerical gradient (mixture gradient is complex)
    let grad_log_posterior = |theta: &[f64]| {
        let eps = 1e-7;
        let f0 = log_posterior(theta);
        let theta_plus = vec![theta[0] + eps];
        let f_plus = log_posterior(&theta_plus);
        vec![(f_plus - f0) / eps]
    };

    let n_samples = 10000; // More samples for multimodal
    let initial = vec![0.0]; // Start in middle

    // MH
    println!("Metropolis-Hastings:");
    let mut config_mh = MCMCConfig::default();
    config_mh.n_warmup = 2000; // Longer warmup for exploration
    let mut mh = MetropolisHastings::new(config_mh, 1);
    let result_mh = mh.sample(&log_posterior, &initial, n_samples);
    println!("  Mean: {:.3}", result_mh.mean()[0]);
    println!("  Std: {:.3}", result_mh.std_dev()[0]);
    println!("  Acceptance: {:.3}", result_mh.accept_rate);

    // HMC
    println!("\nHamiltonian Monte Carlo:");
    let mut config_hmc = MCMCConfig::for_hmc();
    config_hmc.n_warmup = 2000;
    let mut hmc = HamiltonianMC::new(config_hmc, 1);
    let result_hmc = hmc.sample(&log_posterior, &grad_log_posterior, &initial, n_samples);
    println!("  Mean: {:.3}", result_hmc.mean()[0]);
    println!("  Std: {:.3}", result_hmc.std_dev()[0]);
    println!("  Acceptance: {:.3}", result_hmc.accept_rate);

    // NUTS
    println!("\nNUTS:");
    let mut config_nuts = MCMCConfig::for_nuts();
    config_nuts.n_warmup = 2000;
    let mut nuts = NUTS::new(config_nuts, 1);
    let result_nuts = nuts.sample(&log_posterior, &grad_log_posterior, &initial, n_samples);
    println!("  Mean: {:.3}", result_nuts.mean()[0]);
    println!("  Std: {:.3}", result_nuts.std_dev()[0]);
    println!("  Acceptance: {:.3}", result_nuts.accept_rate);

    println!("\nNote: Multimodal distributions require careful initialization");
    println!("      and sufficient warmup for proper exploration.\n");

    println!("{}", "=".repeat(70));
    println!();
}

fn demo_hierarchical_model() {
    println!("--- Example 4: Hierarchical Model ---");
    println!("Model: y ~ N(μ, σ²), μ ~ N(0, 10²), σ ~ HalfNormal(5)");
    println!("Data: y = [1.2, 2.3, 1.8, 2.1, 1.9] (observed)\n");

    let data = vec![1.2, 2.3, 1.8, 2.1, 1.9];

    // Log-posterior: p(μ, log_σ | data)
    // Using log_σ for unconstrained sampling
    let log_posterior = |params: &[f64]| {
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = libm::exp(log_sigma);

        // Likelihood: Π N(yi; μ, σ²)
        let log_likelihood: f64 = data
            .iter()
            .map(|&y| {
                -0.5 * ((y - mu) / sigma).powi(2)
                    - libm::log(sigma)
                    - 0.5 * libm::log(2.0 * 3.14159265359)
            })
            .sum();

        // Prior: μ ~ N(0, 100), log_σ ~ N(log(5), 1²)
        let log_prior_mu = -0.5 * (mu / 10.0).powi(2);
        let log_prior_sigma = -0.5 * (log_sigma - libm::log(5.0)).powi(2);

        // Jacobian for log_σ transformation
        let log_jacobian = log_sigma;

        log_likelihood + log_prior_mu + log_prior_sigma + log_jacobian
    };

    let grad_log_posterior = |params: &[f64]| {
        let eps = 1e-7;
        let f0 = log_posterior(params);

        let mut grad = Vec::new();
        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += eps;
            let f_plus = log_posterior(&params_plus);
            grad.push((f_plus - f0) / eps);
        }
        grad
    };

    let n_samples = 3000;
    let initial = vec![0.0, libm::log(1.0)]; // [μ, log_σ]

    println!("NUTS (best for hierarchical models):");
    let config_nuts = MCMCConfig::for_nuts();
    let mut nuts = NUTS::new(config_nuts, 2);
    let result_nuts = nuts.sample(&log_posterior, &grad_log_posterior, &initial, n_samples);

    let mean_mu = result_nuts.mean()[0];
    let mean_log_sigma = result_nuts.mean()[1];
    let mean_sigma = libm::exp(mean_log_sigma);

    println!(
        "  Posterior μ: {:.3} ± {:.3}",
        mean_mu,
        result_nuts.std_dev()[0]
    );
    println!(
        "  Posterior σ: {:.3} (from log_σ = {:.3})",
        mean_sigma, mean_log_sigma
    );
    println!("  Acceptance rate: {:.3}", result_nuts.accept_rate);
    println!(
        "  ESS: [{:.1}, {:.1}]",
        result_nuts.ess[0], result_nuts.ess[1]
    );

    let ci = result_nuts.credible_interval(0.95);
    println!("\n  95% Credible Intervals:");
    println!("    μ: [{:.3}, {:.3}]", ci[0].0, ci[0].1);
    println!(
        "    σ: [{:.3}, {:.3}]",
        libm::exp(ci[1].0),
        libm::exp(ci[1].1)
    );

    println!("\n{}", "=".repeat(70));
    println!();
}

fn demo_multiple_chains() {
    println!("--- Example 5: Multiple Chains & Convergence Diagnostics ---");
    println!("Run 4 parallel chains and compute R-hat for convergence\n");

    let log_posterior = |theta: &[f64]| -0.5 * theta[0].powi(2);
    let grad_log_posterior = |theta: &[f64]| vec![-theta[0]];

    let n_chains = 4;
    let n_samples = 2000;

    println!(
        "Running {} chains with different initializations...",
        n_chains
    );

    let mut all_chains = Vec::new();
    let initials = vec![-2.0, -1.0, 1.0, 2.0]; // Different starting points

    for (i, &init) in initials.iter().enumerate() {
        let config = MCMCConfig::for_nuts();
        let mut nuts = NUTS::new(config, 1);
        let result = nuts.sample(&log_posterior, &grad_log_posterior, &[init], n_samples);
        all_chains.push(result.samples.clone());

        println!(
            "  Chain {}: mean={:.3}, ESS={:.1}, accept={:.3}",
            i + 1,
            result.mean()[0],
            result.ess[0],
            result.accept_rate
        );
    }

    // Compute R-hat
    let r_hat = compute_r_hat(&all_chains);
    println!("\nGelman-Rubin diagnostic (R-hat): {:.4}", r_hat[0]);

    if r_hat[0] < 1.1 {
        println!("  ✓ Chains have converged (R-hat < 1.1)");
    } else {
        println!("  ✗ Chains have NOT converged (R-hat >= 1.1)");
        println!("     Recommend longer warmup or more samples");
    }

    println!("\n{}", "=".repeat(70));
    println!();
}

// =============================================================================
// Helper Functions
// =============================================================================

fn print_results(
    name: &str,
    result: &sounio::epistemic::mcmc::MCMCResult,
    true_mean: f64,
    true_std: f64,
) {
    let mean = result.mean()[0];
    let std = result.std_dev()[0];

    println!("  Posterior mean: {:.3} (true: {:.1})", mean, true_mean);
    println!("  Posterior std:  {:.3} (true: {:.1})", std, true_std);
    println!("  Acceptance rate: {:.3}", result.accept_rate);
    println!("  ESS: {:.1}", result.ess[0]);
    println!("  Final step size: {:.4}", result.final_step_size);

    let ci = result.credible_interval(0.95);
    println!("  95% CI: [{:.3}, {:.3}]", ci[0].0, ci[0].1);

    // Error analysis
    let mean_error = (mean - true_mean).abs();
    let std_error = (std - true_std).abs();
    println!(
        "  Mean error: {:.4}, Std error: {:.4}",
        mean_error, std_error
    );
}

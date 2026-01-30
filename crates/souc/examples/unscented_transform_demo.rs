//! Unscented Transform Demo
//!
//! Demonstrates Task 1.2: UT for nonlinear uncertainty propagation.
//!
//! This example shows:
//! - Sigma point generation and propagation
//! - Comparison with first-order Taylor (GUM)
//! - Comparison with Monte Carlo reference
//! - Highly nonlinear transformations (trig, exponential, polynomial)
//! - Multi-dimensional propagation
//!
//! The Unscented Transform captures uncertainty to 3rd order vs 1st order
//! for linearization (GUM), making it superior for nonlinear systems.

use sounio::epistemic::unscented_transform::{UTConfig, UnscentedTransform};
use std::f64::consts::PI;

fn main() {
    println!("=== Unscented Transform for Nonlinear Uncertainty Propagation ===\n");

    // Example 1: Highly nonlinear trigonometric function
    demo_trigonometric();

    // Example 2: Exponential nonlinearity
    demo_exponential();

    // Example 3: Polynomial nonlinearity
    demo_polynomial();

    // Example 4: Multi-dimensional propagation
    demo_multivariate();

    // Example 5: Cross-covariance for filtering
    demo_cross_covariance();

    // Example 6: Comparison with Taylor approximation
    demo_ut_vs_taylor();
}

fn demo_trigonometric() {
    println!("--- Example 1: Trigonometric Nonlinearity ---");
    println!("y = sin(x), x ~ N(π/4, 0.1²)");
    println!("Highly nonlinear due to curvature\n");

    let config = UTConfig::default();
    let mean = vec![PI / 4.0]; // 45 degrees
    let cov = vec![vec![0.01]]; // Variance = 0.1²

    let ut = UnscentedTransform::new(config, mean, cov);

    let result = ut.propagate(|x| vec![libm::sin(x[0])]);

    println!("Input:");
    println!("  Mean: {:.6}", ut.mean[0]);
    println!("  Std Dev: {:.6}", ut.std_devs()[0]);

    println!("\nOutput (UT):");
    println!("  Mean: {:.6}", result.mean[0]);
    println!("  Std Dev: {:.6}", result.std_devs()[0]);

    // Analytical values for comparison
    let analytical_mean = libm::sin(PI / 4.0); // ≈ 0.7071
    println!("\nAnalytical (at mean): {:.6}", analytical_mean);

    // Monte Carlo reference
    let mc_mean = monte_carlo_mean(|x| libm::sin(x), PI / 4.0, 0.1, 100000);
    let mc_std = monte_carlo_std(|x| libm::sin(x), PI / 4.0, 0.1, 100000);
    println!("\nMonte Carlo (100k samples):");
    println!("  Mean: {:.6}", mc_mean);
    println!("  Std Dev: {:.6}", mc_std);
    println!();
}

fn demo_exponential() {
    println!("--- Example 2: Exponential Nonlinearity ---");
    println!("y = exp(x), x ~ N(0, 0.5²)");
    println!("Extremely nonlinear - exponential growth\n");

    let config = UTConfig::default();
    let mean = vec![0.0];
    let cov = vec![vec![0.25]]; // Variance = 0.5²

    let ut = UnscentedTransform::new(config, mean, cov);

    let result = ut.propagate(|x| vec![libm::exp(x[0])]);

    println!("Input:");
    println!("  Mean: {:.6}", ut.mean[0]);
    println!("  Std Dev: {:.6}", ut.std_devs()[0]);

    println!("\nOutput (UT):");
    println!("  Mean: {:.6}", result.mean[0]);
    println!("  Std Dev: {:.6}", result.std_devs()[0]);

    // For log-normal: E[exp(X)] = exp(μ + σ²/2)
    let lognormal_mean = libm::exp(0.0 + 0.25 / 2.0);
    println!("\nLog-normal analytical: {:.6}", lognormal_mean);

    let mc_mean = monte_carlo_mean(|x| libm::exp(x), 0.0, 0.5, 100000);
    let mc_std = monte_carlo_std(|x| libm::exp(x), 0.0, 0.5, 100000);
    println!("\nMonte Carlo (100k samples):");
    println!("  Mean: {:.6}", mc_mean);
    println!("  Std Dev: {:.6}", mc_std);
    println!();
}

fn demo_polynomial() {
    println!("--- Example 3: Polynomial Nonlinearity ---");
    println!("y = x³ - 2x² + x, x ~ N(1, 0.2²)");
    println!("Cubic nonlinearity with multiple turning points\n");

    let config = UTConfig::default();
    let mean = vec![1.0];
    let cov = vec![vec![0.04]]; // Variance = 0.2²

    let ut = UnscentedTransform::new(config, mean, cov);

    let f = |x: &[f64]| vec![x[0].powi(3) - 2.0 * x[0].powi(2) + x[0]];
    let result = ut.propagate(&f);

    println!("Input:");
    println!("  Mean: {:.6}", ut.mean[0]);
    println!("  Std Dev: {:.6}", ut.std_devs()[0]);

    println!("\nOutput (UT):");
    println!("  Mean: {:.6}", result.mean[0]);
    println!("  Std Dev: {:.6}", result.std_devs()[0]);

    // Evaluate at mean
    let at_mean = f(&ut.mean);
    println!("\nFunction at mean: {:.6}", at_mean[0]);

    let mc_mean = monte_carlo_mean(|x| x.powi(3) - 2.0 * x.powi(2) + x, 1.0, 0.2, 100000);
    let mc_std = monte_carlo_std(|x| x.powi(3) - 2.0 * x.powi(2) + x, 1.0, 0.2, 100000);
    println!("\nMonte Carlo (100k samples):");
    println!("  Mean: {:.6}", mc_mean);
    println!("  Std Dev: {:.6}", mc_std);
    println!();
}

fn demo_multivariate() {
    println!("--- Example 4: Multivariate Propagation ---");
    println!("f(x,y) = [x·sin(y), exp(x+y)]");
    println!("x ~ N(1, 0.1²), y ~ N(π/6, 0.05²), uncorrelated\n");

    let config = UTConfig::default();
    let mean = vec![1.0, PI / 6.0];
    let cov = vec![
        vec![0.01, 0.0],   // Var(x) = 0.1², no correlation
        vec![0.0, 0.0025], // Var(y) = 0.05²
    ];

    let ut = UnscentedTransform::new(config, mean, cov);

    let result = ut.propagate(|x| vec![x[0] * libm::sin(x[1]), libm::exp(x[0] + x[1])]);

    println!("Input:");
    println!("  x: μ={:.6}, σ={:.6}", ut.mean[0], ut.std_devs()[0]);
    println!("  y: μ={:.6}, σ={:.6}", ut.mean[1], ut.std_devs()[1]);

    println!("\nOutput (UT):");
    println!("  z₁ = x·sin(y):");
    println!("    Mean: {:.6}", result.mean[0]);
    println!("    Std Dev: {:.6}", result.std_devs()[0]);
    println!("  z₂ = exp(x+y):");
    println!("    Mean: {:.6}", result.mean[1]);
    println!("    Std Dev: {:.6}", result.std_devs()[1]);

    // Correlation
    let corr = result.correlation_matrix();
    println!("\nOutput correlation:");
    println!("  ρ(z₁,z₂) = {:.6}", corr[0][1]);
    println!();
}

fn demo_cross_covariance() {
    println!("--- Example 5: Cross-Covariance (for Kalman Filtering) ---");
    println!("y = [x₁ + x₂, x₁·x₂]");
    println!("x ~ N([1,2], [[0.1, 0.05], [0.05, 0.2]])\n");

    let config = UTConfig::default();
    let mean = vec![1.0, 2.0];
    let cov = vec![vec![0.1, 0.05], vec![0.05, 0.2]];

    let ut = UnscentedTransform::new(config, mean, cov);

    let result = ut.propagate_with_cross(|x| vec![x[0] + x[1], x[0] * x[1]]);

    println!("Output mean:");
    println!("  y₁: {:.6}", result.mean[0]);
    println!("  y₂: {:.6}", result.mean[1]);

    println!("\nOutput covariance:");
    println!("  Var(y₁): {:.6}", result.covariance[0][0]);
    println!("  Var(y₂): {:.6}", result.covariance[1][1]);
    println!("  Cov(y₁,y₂): {:.6}", result.covariance[0][1]);

    println!("\nCross-covariance Cov(x,y):");
    println!("  Cov(x₁,y₁): {:.6}", result.cross_covariance[0][0]);
    println!("  Cov(x₁,y₂): {:.6}", result.cross_covariance[0][1]);
    println!("  Cov(x₂,y₁): {:.6}", result.cross_covariance[1][0]);
    println!("  Cov(x₂,y₂): {:.6}", result.cross_covariance[1][1]);
    println!();
}

fn demo_ut_vs_taylor() {
    println!("--- Example 6: UT vs First-Order Taylor (GUM) ---");
    println!("y = x², x ~ N(2, 1²)");
    println!("Quadratic nonlinearity - Taylor approximation has error\n");

    let config = UTConfig::default();
    let mean = vec![2.0];
    let cov = vec![vec![1.0]]; // Large variance to show difference

    let ut = UnscentedTransform::new(config, mean, cov);

    // UT propagation
    let f = |x: &[f64]| vec![x[0] * x[0]];
    let ut_result = ut.propagate(&f);

    // First-order Taylor: E[y] ≈ f(μ), Var[y] ≈ (df/dx)² Var[x]
    // For y = x²: df/dx = 2x, so at x=2: df/dx = 4
    let taylor_mean = f(&ut.mean);
    let jacobian = vec![vec![2.0 * ut.mean[0]]]; // df/dx = 2x at x=μ
    let taylor_var = jacobian[0][0].powi(2) * ut.covariance[0][0]; // J·Σ·Jᵀ

    println!("UT Results:");
    println!("  Mean: {:.6}", ut_result.mean[0]);
    println!("  Variance: {:.6}", ut_result.covariance[0][0]);
    println!("  Std Dev: {:.6}", ut_result.std_devs()[0]);

    println!("\nFirst-Order Taylor (GUM):");
    println!("  Mean: {:.6}", taylor_mean[0]);
    println!("  Variance: {:.6}", taylor_var);
    println!("  Std Dev: {:.6}", libm::sqrt(taylor_var));

    // Analytical for x² with x ~ N(μ, σ²):
    // E[X²] = μ² + σ²
    // Var[X²] = 2σ⁴ + 4μ²σ²
    let analytical_mean = ut.mean[0].powi(2) + ut.covariance[0][0];
    let analytical_var =
        2.0 * ut.covariance[0][0].powi(2) + 4.0 * ut.mean[0].powi(2) * ut.covariance[0][0];

    println!("\nAnalytical (exact for Gaussian):");
    println!("  Mean: {:.6}", analytical_mean);
    println!("  Variance: {:.6}", analytical_var);
    println!("  Std Dev: {:.6}", libm::sqrt(analytical_var));

    // Monte Carlo validation
    let mc_mean = monte_carlo_mean(
        |x| x * x,
        ut.mean[0],
        libm::sqrt(ut.covariance[0][0]),
        100000,
    );
    let mc_var = monte_carlo_variance(
        |x| x * x,
        ut.mean[0],
        libm::sqrt(ut.covariance[0][0]),
        100000,
    );

    println!("\nMonte Carlo (100k samples):");
    println!("  Mean: {:.6}", mc_mean);
    println!("  Variance: {:.6}", mc_var);
    println!("  Std Dev: {:.6}", libm::sqrt(mc_var));

    println!("\nError Analysis:");
    println!(
        "  UT mean error: {:.6}",
        (ut_result.mean[0] - analytical_mean).abs()
    );
    println!(
        "  Taylor mean error: {:.6}",
        (taylor_mean[0] - analytical_mean).abs()
    );
    println!(
        "  UT variance error: {:.6}",
        (ut_result.covariance[0][0] - analytical_var).abs()
    );
    println!(
        "  Taylor variance error: {:.6}",
        (taylor_var - analytical_var).abs()
    );

    println!("\nConclusion:");
    if (ut_result.mean[0] - analytical_mean).abs() < (taylor_mean[0] - analytical_mean).abs() {
        println!("  ✓ UT has lower mean error than Taylor");
    }
    if (ut_result.covariance[0][0] - analytical_var).abs() < (taylor_var - analytical_var).abs() {
        println!("  ✓ UT has lower variance error than Taylor");
    }
    println!();
}

// =============================================================================
// Monte Carlo Reference Implementation
// =============================================================================

fn monte_carlo_mean<F>(f: F, mean: f64, std_dev: f64, n_samples: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut sum = 0.0;

    for _ in 0..n_samples {
        let x = mean + std_dev * sample_normal();
        sum += f(x);
    }

    sum / n_samples as f64
}

fn monte_carlo_std<F>(f: F, mean: f64, std_dev: f64, n_samples: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let var = monte_carlo_variance(f, mean, std_dev, n_samples);
    libm::sqrt(var)
}

fn monte_carlo_variance<F>(f: F, mean: f64, std_dev: f64, n_samples: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_samples {
        let x = mean + std_dev * sample_normal();
        let y = f(x);
        sum += y;
        sum_sq += y * y;
    }

    let mc_mean = sum / n_samples as f64;
    sum_sq / n_samples as f64 - mc_mean * mc_mean
}

/// Sample from standard normal using Box-Muller transform
fn sample_normal() -> f64 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        );
    }

    SEED.with(|seed| {
        // Box-Muller transform
        let s = seed.get();

        // Generate two uniform random numbers
        let next1 = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let next2 = next1.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed.set(next2);

        let u1 = next1 as f64 / u64::MAX as f64;
        let u2 = next2 as f64 / u64::MAX as f64;

        // Box-Muller
        let r = libm::sqrt(-2.0 * libm::log(u1));
        let theta = 2.0 * PI * u2;
        r * libm::cos(theta)
    })
}

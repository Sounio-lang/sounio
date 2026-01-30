//! Polynomial Chaos Expansion Demo with Sobol Sensitivity Analysis
//!
//! Demonstrates Task 1.1: PCE with Sobol Indices for variance-based sensitivity.
//!
//! This example shows:
//! - PCE coefficient computation via Gauss quadrature
//! - Mean and variance estimation from expansion
//! - Sobol indices for global sensitivity analysis
//! - Comparison with Monte Carlo reference
//! - Benchmark functions: Ishigami and Sobol G-function

use sounio::epistemic::pce::{PCEConfig, PCEExpansion, PolynomialFamily, TruncationScheme};
use std::f64::consts::PI;

fn main() {
    println!("=== Polynomial Chaos Expansion with Sobol Sensitivity Demo ===\n");

    // Example 1: Ishigami function (classic UQ benchmark)
    demo_ishigami();

    // Example 2: Sobol G-function (variable importance)
    demo_sobol_g();

    // Example 3: Simple polynomial for verification
    demo_polynomial();

    // Example 4: Additive model (independence check)
    demo_additive();

    // Example 5: Interaction effects
    demo_interactions();
}

fn demo_ishigami() {
    println!("--- Example 1: Ishigami Function ---");
    println!("f(x1,x2,x3) = sin(x1) + 7sin²(x2) + 0.1x3⁴sin(x1)");
    println!("Domain: [-π, π]³\n");

    let config = PCEConfig {
        max_degree: 6,
        family: PolynomialFamily::Legendre, // Uniform distribution
        n_quadrature_points: 15,
        sparse_grid: false,
        truncation: TruncationScheme::TotalDegree,
    };

    let mut pce = PCEExpansion::new(config, 3);

    // Ishigami function: transform from [-1,1] to [-π,π]
    let ishigami = |xi: &[f64]| -> f64 {
        let x1 = xi[0] * PI;
        let x2 = xi[1] * PI;
        let x3 = xi[2] * PI;

        libm::sin(x1) + 7.0 * libm::sin(x2).powi(2) + 0.1 * x3.powi(4) * libm::sin(x1)
    };

    pce.compute_coefficients(ishigami);

    println!("PCE Results:");
    println!("  Mean: {:.6}", pce.mean());
    println!("  Variance: {:.6}", pce.variance());
    println!("  Std Dev: {:.6}", pce.variance().sqrt());

    let indices = pce.sobol_indices();
    println!("\nSobol Indices:");
    println!("  S1 (x1): {:.4}", indices.first_order[0]);
    println!("  S2 (x2): {:.4}", indices.first_order[1]);
    println!("  S3 (x3): {:.4}", indices.first_order[2]);
    println!("  ST1 (x1 total): {:.4}", indices.total[0]);
    println!("  ST2 (x2 total): {:.4}", indices.total[1]);
    println!("  ST3 (x3 total): {:.4}", indices.total[2]);

    println!("\nAnalytical Reference (from literature):");
    println!("  Mean: 3.5000");
    println!("  Variance: 13.8447");
    println!("  S1: 0.3139");
    println!("  S2: 0.4424");
    println!("  S3: 0.0000");
    println!("  ST1: 0.5576");
    println!("  ST2: 0.4424");
    println!("  ST3: 0.2437");
    println!();

    // Monte Carlo reference
    let mc_mean = monte_carlo_mean(&ishigami, 3, 100000);
    let mc_var = monte_carlo_variance(&ishigami, 3, 100000);
    println!("Monte Carlo Reference (100k samples):");
    println!("  Mean: {:.6}", mc_mean);
    println!("  Variance: {:.6}", mc_var);
    println!();
}

fn demo_sobol_g() {
    println!("--- Example 2: Sobol G-Function ---");
    println!("f(x) = Π (|4xi - 2| + ai) / (1 + ai)");
    println!("Domain: [0, 1]^d with a = [0, 1, 4.5, 9]\n");

    let a = vec![0.0, 1.0, 4.5, 9.0];

    let config = PCEConfig {
        max_degree: 4,
        family: PolynomialFamily::Legendre,
        n_quadrature_points: 10,
        sparse_grid: false,
        truncation: TruncationScheme::TotalDegree,
    };

    let mut pce = PCEExpansion::new(config, a.len());

    // Sobol G-function: transform from [-1,1] to [0,1]
    let sobol_g = |xi: &[f64]| -> f64 {
        let mut product = 1.0;

        for (i, &ai) in a.iter().enumerate() {
            let x = (xi[i] + 1.0) / 2.0; // [-1,1] → [0,1]
            product *= (libm::fabs(4.0 * x - 2.0) + ai) / (1.0 + ai);
        }

        product
    };

    pce.compute_coefficients(sobol_g);

    println!("PCE Results:");
    println!("  Mean: {:.6}", pce.mean());
    println!("  Variance: {:.6}", pce.variance());

    let indices = pce.sobol_indices();
    println!("\nSobol First-Order Indices:");
    for (i, &s) in indices.first_order.iter().enumerate() {
        println!("  S{} (a={}): {:.4}", i + 1, a[i], s);
    }

    println!("\nSobol Total Indices:");
    for (i, &st) in indices.total.iter().enumerate() {
        println!("  ST{} (a={}): {:.4}", i + 1, a[i], st);
    }

    println!("\nInterpretation:");
    println!("  - Smaller ai → higher importance");
    println!("  - a=0 (i=0) should dominate");
    println!("  - Importance order: a₀ > a₁ > a₂ > a₃");
    println!();
}

fn demo_polynomial() {
    println!("--- Example 3: Simple Polynomial ---");
    println!("f(x1,x2) = x1² + 2x1x2 + 3x2²");
    println!("Domain: [-1, 1]²\n");

    let config = PCEConfig {
        max_degree: 2,
        family: PolynomialFamily::Legendre,
        n_quadrature_points: 5,
        sparse_grid: false,
        truncation: TruncationScheme::TotalDegree,
    };

    let mut pce = PCEExpansion::new(config, 2);

    let polynomial = |xi: &[f64]| -> f64 {
        let x1 = xi[0];
        let x2 = xi[1];
        x1 * x1 + 2.0 * x1 * x2 + 3.0 * x2 * x2
    };

    pce.compute_coefficients(polynomial);

    println!("PCE Results:");
    println!("  Mean: {:.6}", pce.mean());
    println!("  Variance: {:.6}", pce.variance());

    // Analytical for uniform on [-1,1]: E[x²] = 1/3
    let expected_mean = 1.0 / 3.0 + 3.0 / 3.0; // E[x1²] + E[x2²]
    println!("\nExpected mean (E[x1²] + 3E[x2²]): {:.6}", expected_mean);
    println!("Error: {:.6}", (pce.mean() - expected_mean).abs());

    let indices = pce.sobol_indices();
    println!("\nSobol Indices:");
    println!("  S1: {:.4}", indices.first_order[0]);
    println!("  S2: {:.4}", indices.first_order[1]);
    println!("  ST1: {:.4}", indices.total[0]);
    println!("  ST2: {:.4}", indices.total[1]);
    println!();
}

fn demo_additive() {
    println!("--- Example 4: Additive Model (Independence) ---");
    println!("f(x1,x2,x3) = sin(x1) + x2² + exp(x3)");
    println!("Domain: [-1, 1]³\n");

    let config = PCEConfig {
        max_degree: 5,
        family: PolynomialFamily::Legendre,
        n_quadrature_points: 10,
        sparse_grid: false,
        truncation: TruncationScheme::TotalDegree,
    };

    let mut pce = PCEExpansion::new(config, 3);

    let additive = |xi: &[f64]| -> f64 { libm::sin(xi[0]) + xi[1] * xi[1] + libm::exp(xi[2]) };

    pce.compute_coefficients(additive);

    println!("PCE Results:");
    println!("  Mean: {:.6}", pce.mean());
    println!("  Variance: {:.6}", pce.variance());

    let indices = pce.sobol_indices();
    println!("\nSobol First-Order Indices:");
    println!("  S1 (sin): {:.4}", indices.first_order[0]);
    println!("  S2 (x²): {:.4}", indices.first_order[1]);
    println!("  S3 (exp): {:.4}", indices.first_order[2]);

    println!("\nSobol Total Indices:");
    println!("  ST1: {:.4}", indices.total[0]);
    println!("  ST2: {:.4}", indices.total[1]);
    println!("  ST3: {:.4}", indices.total[2]);

    let sum_first_order: f64 = indices.first_order.iter().sum();
    println!("\nAdditive model check:");
    println!("  Sum of first-order indices: {:.4}", sum_first_order);
    println!("  Should be ~1.0 for additive model");
    println!(
        "  First ≈ Total? {}",
        if (indices.first_order[0] - indices.total[0]).abs() < 0.05 {
            "YES (no interactions)"
        } else {
            "NO (has interactions)"
        }
    );
    println!();
}

fn demo_interactions() {
    println!("--- Example 5: Interaction Effects ---");
    println!("f(x1,x2) = x1 + x2 + 5x1x2");
    println!("Domain: [-1, 1]²\n");

    let config = PCEConfig {
        max_degree: 2,
        family: PolynomialFamily::Legendre,
        n_quadrature_points: 5,
        sparse_grid: false,
        truncation: TruncationScheme::TotalDegree,
    };

    let mut pce = PCEExpansion::new(config, 2);

    let interaction = |xi: &[f64]| -> f64 { xi[0] + xi[1] + 5.0 * xi[0] * xi[1] };

    pce.compute_coefficients(interaction);

    println!("PCE Results:");
    println!("  Mean: {:.6}", pce.mean());
    println!("  Variance: {:.6}", pce.variance());

    let indices = pce.sobol_indices();
    println!("\nSobol Indices:");
    println!("  S1 (first-order): {:.4}", indices.first_order[0]);
    println!("  S2 (first-order): {:.4}", indices.first_order[1]);
    println!("  ST1 (total): {:.4}", indices.total[0]);
    println!("  ST2 (total): {:.4}", indices.total[1]);

    let interaction_12 =
        indices.total[0] + indices.total[1] - indices.first_order[0] - indices.first_order[1];
    println!("\nInteraction term S12:");
    println!("  S12 = ST1 + ST2 - S1 - S2 = {:.4}", interaction_12);
    println!("  Strong interaction due to 5x1x2 term");
    println!();
}

// =============================================================================
// Monte Carlo Reference Implementation
// =============================================================================

fn monte_carlo_mean<F>(f: &F, n_dims: usize, n_samples: usize) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut sum = 0.0;

    for _ in 0..n_samples {
        let xi: Vec<f64> = (0..n_dims).map(|_| uniform_random()).collect();
        sum += f(&xi);
    }

    sum / n_samples as f64
}

fn monte_carlo_variance<F>(f: &F, n_dims: usize, n_samples: usize) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mean = monte_carlo_mean(f, n_dims, n_samples);
    let mut sum_sq_dev = 0.0;

    for _ in 0..n_samples {
        let xi: Vec<f64> = (0..n_dims).map(|_| uniform_random()).collect();
        let val = f(&xi);
        sum_sq_dev += (val - mean).powi(2);
    }

    sum_sq_dev / n_samples as f64
}

/// Simple pseudo-random uniform on [-1, 1]
/// (Not cryptographically secure - for demo only)
fn uniform_random() -> f64 {
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
        let s = seed.get();
        // Linear congruential generator
        let next = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed.set(next);

        // Map to [-1, 1]
        2.0 * (next as f64 / u64::MAX as f64) - 1.0
    })
}

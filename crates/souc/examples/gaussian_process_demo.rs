//! Gaussian Process Regression Demo
//!
//! Demonstrates GP-based surrogate modeling and uncertainty quantification for:
//! 1. 1D function approximation with predictive uncertainty
//! 2. Kernel function comparison (RBF, Matérn, Periodic)
//! 3. Hyperparameter optimization effects
//! 4. Interpolation vs extrapolation uncertainty behavior
//! 5. Multi-dimensional regression
//!
//! Run with: cargo run --example gaussian_process_demo

use souc::epistemic::gaussian_process::{GPConfig, GaussianProcess, Kernel};

fn main() {
    println!("=== Gaussian Process Regression Demo ===\n");

    demo_1d_regression();
    demo_kernel_comparison();
    demo_hyperparameter_optimization();
    demo_interpolation_vs_extrapolation();
    demo_multidimensional();
}

/// Demo 1: Basic 1D function approximation
fn demo_1d_regression() {
    println!("--- Demo 1: 1D Function Approximation ---");
    println!("Fitting GP to f(x) = sin(x) + noise\n");

    // Generate training data: f(x) = sin(x) with noise
    let x_train: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
        vec![6.0],
    ];
    let y_train: Vec<f64> = x_train
        .iter()
        .map(|x| libm::sin(x[0]) + 0.1 * (x[0] * 0.7).sin())
        .collect();

    // Configure GP with RBF kernel
    let config = GPConfig {
        kernel: Kernel::rbf(1.0, 1.0),
        noise_variance: 0.01,
        optimize_hyperparameters: true,
        max_iterations: 50,
        ..Default::default()
    };

    let mut gp = GaussianProcess::new(config);
    gp.fit(&x_train, &y_train);

    // Test points (including interpolation and mild extrapolation)
    let x_test: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.5]).collect();

    let predictions = gp.predict(&x_test);

    println!("  x     | True f(x) | GP Mean | GP Std | 95% CI Width");
    println!("--------|-----------|---------|--------|-------------");
    for (i, x) in x_test.iter().enumerate() {
        let true_val = libm::sin(x[0]);
        let mean = predictions.mean[i];
        let std = predictions.std_dev[i];
        let ci_width = 1.96 * std; // 95% confidence interval half-width

        println!(
            " {:5.2} | {:9.4} | {:7.4} | {:6.4} | {:11.4}",
            x[0], true_val, mean, std, ci_width
        );
    }
    println!();
}

/// Demo 2: Comparison of different kernel functions
fn demo_kernel_comparison() {
    println!("--- Demo 2: Kernel Function Comparison ---");
    println!("Comparing RBF, Matérn-3/2, and Matérn-5/2 on noisy data\n");

    // Generate training data with noise
    let x_train: Vec<Vec<f64>> = vec![vec![0.0], vec![1.5], vec![3.0], vec![4.5], vec![6.0]];
    let y_train: Vec<f64> = vec![0.0, 0.9, 0.1, -0.8, -0.3];

    let kernels = vec![
        ("RBF", Kernel::rbf(1.5, 1.0)),
        ("Matérn-3/2", Kernel::matern32(1.5, 1.0)),
        ("Matérn-5/2", Kernel::matern52(1.5, 1.0)),
    ];

    let x_test: Vec<Vec<f64>> = (0..13).map(|i| vec![i as f64 * 0.5]).collect();

    for (name, kernel) in kernels {
        let config = GPConfig {
            kernel,
            noise_variance: 0.05,
            optimize_hyperparameters: false,
            ..Default::default()
        };

        let mut gp = GaussianProcess::new(config);
        gp.fit(&x_train, &y_train);
        let predictions = gp.predict(&x_test);

        println!("{} kernel predictions:", name);
        println!("  x   | Mean  | Std Dev");
        println!("------|-------|--------");
        for (i, x) in x_test.iter().enumerate() {
            println!(
                " {:4.1} | {:5.2} | {:6.3}",
                x[0], predictions.mean[i], predictions.std_dev[i]
            );
        }
        println!();
    }
}

/// Demo 3: Effect of hyperparameter optimization
fn demo_hyperparameter_optimization() {
    println!("--- Demo 3: Hyperparameter Optimization ---");
    println!("Comparing manual vs optimized hyperparameters\n");

    // Training data
    let x_train: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y_train: Vec<f64> = vec![0.0, 1.0, 0.5, -0.5, 0.0];

    // Without optimization (manual hyperparameters)
    let config_manual = GPConfig {
        kernel: Kernel::rbf(0.5, 1.0), // Potentially suboptimal
        noise_variance: 0.01,
        optimize_hyperparameters: false,
        ..Default::default()
    };

    let mut gp_manual = GaussianProcess::new(config_manual);
    gp_manual.fit(&x_train, &y_train);

    // With optimization
    let config_optimized = GPConfig {
        kernel: Kernel::rbf(1.0, 1.0), // Will be optimized
        noise_variance: 0.01,
        optimize_hyperparameters: true,
        max_iterations: 50,
        ..Default::default()
    };

    let mut gp_optimized = GaussianProcess::new(config_optimized);
    gp_optimized.fit(&x_train, &y_train);

    let x_test: Vec<Vec<f64>> = (0..9).map(|i| vec![i as f64 * 0.5]).collect();

    let pred_manual = gp_manual.predict(&x_test);
    let pred_optimized = gp_optimized.predict(&x_test);

    println!("  x   | Manual Mean | Manual Std | Optimized Mean | Optimized Std");
    println!("------|-------------|------------|----------------|---------------");
    for (i, x) in x_test.iter().enumerate() {
        println!(
            " {:4.1} | {:11.4} | {:10.4} | {:14.4} | {:13.4}",
            x[0],
            pred_manual.mean[i],
            pred_manual.std_dev[i],
            pred_optimized.mean[i],
            pred_optimized.std_dev[i]
        );
    }
    println!("\nNote: Optimized GP should have better marginal likelihood.");
    println!();
}

/// Demo 4: Interpolation vs Extrapolation uncertainty
fn demo_interpolation_vs_extrapolation() {
    println!("--- Demo 4: Interpolation vs Extrapolation Uncertainty ---");
    println!("GP uncertainty grows in regions without training data\n");

    // Training data clustered around x = 2-4
    let x_train: Vec<Vec<f64>> = vec![vec![2.0], vec![2.5], vec![3.0], vec![3.5], vec![4.0]];
    let y_train: Vec<f64> = vec![0.5, 0.8, 1.0, 0.9, 0.6];

    let config = GPConfig {
        kernel: Kernel::rbf(0.5, 1.0),
        noise_variance: 0.01,
        optimize_hyperparameters: false,
        ..Default::default()
    };

    let mut gp = GaussianProcess::new(config);
    gp.fit(&x_train, &y_train);

    // Test points spanning interpolation and extrapolation regions
    let x_test: Vec<Vec<f64>> = (0..13).map(|i| vec![i as f64 * 0.5]).collect();
    let predictions = gp.predict(&x_test);

    println!("  x   | Region         | Mean  | Std Dev | Uncertainty");
    println!("------|----------------|-------|---------|-------------");
    for (i, x) in x_test.iter().enumerate() {
        let region = if x[0] >= 2.0 && x[0] <= 4.0 {
            "Interpolation"
        } else {
            "Extrapolation"
        };
        let uncertainty_level = if predictions.std_dev[i] > 0.5 {
            "HIGH"
        } else if predictions.std_dev[i] > 0.2 {
            "Medium"
        } else {
            "Low"
        };

        println!(
            " {:4.1} | {:14} | {:5.2} | {:7.3} | {}",
            x[0], region, predictions.mean[i], predictions.std_dev[i], uncertainty_level
        );
    }
    println!("\nObservation: Uncertainty increases away from training data (extrapolation).");
    println!();
}

/// Demo 5: Multi-dimensional regression
fn demo_multidimensional() {
    println!("--- Demo 5: Multi-Dimensional Regression ---");
    println!("Fitting GP to 2D function: f(x,y) = sin(x) * cos(y)\n");

    // Generate 2D training data
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    // Grid of training points
    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64 * 0.5;
            let y = j as f64 * 0.5;
            x_train.push(vec![x, y]);
            y_train.push(libm::sin(x) * libm::cos(y));
        }
    }

    let config = GPConfig {
        kernel: Kernel::rbf(1.0, 1.0),
        noise_variance: 0.001,
        optimize_hyperparameters: true,
        max_iterations: 30,
        ..Default::default()
    };

    let mut gp = GaussianProcess::new(config);
    gp.fit(&x_train, &y_train);

    // Test on a few points
    let x_test = vec![
        vec![0.5, 0.5],
        vec![1.0, 1.0],
        vec![1.5, 0.5],
        vec![0.5, 1.5],
    ];

    let predictions = gp.predict(&x_test);

    println!("   (x, y)    | True f(x,y) | GP Mean | GP Std | Error");
    println!("-------------|-------------|---------|--------|-------");
    for (i, xy) in x_test.iter().enumerate() {
        let true_val = libm::sin(xy[0]) * libm::cos(xy[1]);
        let error = libm::fabs(true_val - predictions.mean[i]);

        println!(
            " ({:3.1}, {:3.1}) | {:11.4} | {:7.4} | {:6.4} | {:5.4}",
            xy[0], xy[1], true_val, predictions.mean[i], predictions.std_dev[i], error
        );
    }
    println!("\nMulti-dimensional GP captures complex surface structure.");
    println!();
}

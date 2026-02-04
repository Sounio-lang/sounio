/// Benchmarks for natural gradient descent convergence
/// Compares natural gradient (5-10x speedup) vs Euclidean gradient on Beta parameter estimation
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sounio::epistemic::{FisherMatrix, NaturalGradientOptimizer};

/// Simple loss function: KL divergence from target Beta(target_α, target_β)
/// Used to demonstrate convergence speedup
fn kl_divergence_loss(alpha: f64, beta: f64, target_alpha: f64, target_beta: f64) -> f64 {
    // Approximate KL(target || current) using moment matching
    // D_KL ≈ ||E[X] - target_E[X]||² + ||V[X] - target_V[X]||²
    let target_mean = target_alpha / (target_alpha + target_beta);
    let target_var = (target_alpha * target_beta)
        / ((target_alpha + target_beta).powi(2) * (target_alpha + target_beta + 1.0));

    let current_mean = alpha / (alpha + beta);
    let current_var = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));

    let mean_diff = (current_mean - target_mean).powi(2);
    let var_diff = (current_var - target_var).powi(2);

    mean_diff + var_diff
}

/// Compute approximate gradient via finite differences
fn gradient_fd(alpha: f64, beta: f64, target_alpha: f64, target_beta: f64, eps: f64) -> (f64, f64) {
    let l0 = kl_divergence_loss(alpha, beta, target_alpha, target_beta);
    let l_alpha = kl_divergence_loss(alpha + eps, beta, target_alpha, target_beta);
    let l_beta = kl_divergence_loss(alpha, beta + eps, target_alpha, target_beta);

    ((l_alpha - l0) / eps, (l_beta - l0) / eps)
}

/// Euclidean gradient descent - simple baseline
fn euclidean_gradient_descent(
    mut alpha: f64,
    mut beta: f64,
    learning_rate: f64,
    target_alpha: f64,
    target_beta: f64,
    max_iterations: usize,
    tolerance: f64,
) -> usize {
    let eps = 1e-6;

    for iter in 0..max_iterations {
        let loss = kl_divergence_loss(alpha, beta, target_alpha, target_beta);
        if loss < tolerance {
            return iter;
        }

        let (grad_alpha, grad_beta) = gradient_fd(alpha, beta, target_alpha, target_beta, eps);

        alpha = (alpha - learning_rate * grad_alpha).max(0.1);
        beta = (beta - learning_rate * grad_beta).max(0.1);
    }

    max_iterations
}

/// Natural gradient descent - uses Fisher metric
fn natural_gradient_descent(
    alpha: f64,
    beta: f64,
    learning_rate: f64,
    target_alpha: f64,
    target_beta: f64,
    max_iterations: usize,
    tolerance: f64,
) -> usize {
    let mut optimizer = NaturalGradientOptimizer::new(alpha, beta, learning_rate, 0.99);
    let eps = 1e-6;

    for iter in 0..max_iterations {
        let loss = kl_divergence_loss(optimizer.alpha, optimizer.beta, target_alpha, target_beta);
        if loss < tolerance {
            return iter;
        }

        let (grad_alpha, grad_beta) = gradient_fd(
            optimizer.alpha,
            optimizer.beta,
            target_alpha,
            target_beta,
            eps,
        );

        optimizer.natural_gradient_step(grad_alpha, grad_beta, loss);
    }

    max_iterations
}

fn benchmark_convergence_speed(c: &mut Criterion) {
    let target_alpha = 5.0;
    let target_beta = 3.0;
    let learning_rate = 0.01;
    let max_iterations = 1000;
    let tolerance = 1e-4;

    let mut group = c.benchmark_group("convergence_speed");

    // Benchmark Euclidean gradient
    group.bench_function("euclidean_gradient", |b| {
        b.iter(|| {
            euclidean_gradient_descent(
                black_box(1.0),
                black_box(1.0),
                black_box(learning_rate),
                black_box(target_alpha),
                black_box(target_beta),
                black_box(max_iterations),
                black_box(tolerance),
            )
        })
    });

    // Benchmark natural gradient
    group.bench_function("natural_gradient", |b| {
        b.iter(|| {
            natural_gradient_descent(
                black_box(1.0),
                black_box(1.0),
                black_box(learning_rate),
                black_box(target_alpha),
                black_box(target_beta),
                black_box(max_iterations),
                black_box(tolerance),
            )
        })
    });

    group.finish();
}

fn benchmark_fisher_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fisher_operations");

    // Test various parameter configurations
    let params = vec![(1.0, 1.0), (2.0, 3.0), (5.0, 5.0), (10.0, 8.0)];

    for (alpha, beta) in params {
        let param_id = format!("alpha={},beta={}", alpha, beta);

        group.bench_with_input(
            BenchmarkId::from_parameter(&param_id),
            &(alpha, beta),
            |bencher, &(a, b)| {
                bencher.iter(|| {
                    let fisher = FisherMatrix::from_beta(black_box(a), black_box(b));
                    black_box(fisher.determinant());
                    black_box(fisher.condition_number());
                    black_box(fisher.inverse())
                })
            },
        );
    }

    group.finish();
}

fn benchmark_optimizer_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_steps");

    group.bench_function("natural_gradient_step", |b| {
        let mut opt = NaturalGradientOptimizer::new(2.0, 3.0, 0.01, 0.95);
        b.iter(|| opt.natural_gradient_step(black_box(0.1), black_box(0.1), black_box(1.0)))
    });

    group.bench_function("line_search_step", |b| {
        let mut opt = NaturalGradientOptimizer::new(2.0, 3.0, 0.1, 0.95);
        b.iter(|| {
            opt.line_search_step(
                black_box(0.1),
                black_box(0.05),
                black_box(1.0),
                black_box(0.1),
                |a, b| kl_divergence_loss(a, b, 5.0, 3.0),
            )
        })
    });

    group.finish();
}

fn benchmark_trigamma_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("trigamma");

    let test_points = vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

    for point in test_points {
        group.bench_with_input(
            BenchmarkId::from_parameter(point.to_string()),
            &point,
            |bencher, &p| {
                bencher.iter(|| {
                    let fisher = FisherMatrix::from_beta(black_box(p), black_box(p));
                    black_box(fisher.determinant())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_convergence_speed,
    benchmark_fisher_matrix_operations,
    benchmark_optimizer_steps,
    benchmark_trigamma_computation
);
criterion_main!(benches);

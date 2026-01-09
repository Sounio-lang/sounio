# Variational Inference

Variational Inference (VI) is an alternative to MCMC that approximates posteriors by optimization rather than sampling. VI trades exactness for speed, making it suitable for large datasets and complex models.

## Overview

Instead of sampling from the posterior P(theta | data), VI finds the closest distribution q(theta) from a tractable family Q:

```
q*(theta) = argmin_{q in Q} KL(q(theta) || P(theta | data))
```

This optimization perspective enables gradient-based methods and scalability to big data.

## The Evidence Lower Bound (ELBO)

Since KL divergence requires the intractable P(data), we optimize the equivalent Evidence Lower Bound (ELBO):

```
ELBO(q) = E_q[log P(data, theta)] - E_q[log q(theta)]
        = E_q[log P(data | theta)] + E_q[log P(theta)] - E_q[log q(theta)]
        = Likelihood term - KL(q || prior)

Maximizing ELBO = Minimizing KL(q || posterior)
```

```sio
use std::prob::{Normal, sample_normal, logpdf_normal}

/// Compute ELBO for Gaussian variational family
fn compute_elbo(
    log_joint: fn([f64]) -> f64,
    q_means: [f64],
    q_stds: [f64],
    n_samples: i64
) -> f64 with Prob {
    var elbo_sum = 0.0

    var s: i64 = 0
    while s < n_samples {
        // Sample from q(theta)
        var theta: [f64] = []
        var entropy = 0.0
        var j: i64 = 0
        while j < q_means.len() {
            let z = sample_normal(0.0, 1.0)
            let t = q_means[j] + q_stds[j] * z
            theta = theta ++ [t]
            // Entropy of normal: 0.5 * log(2*pi*e*sigma^2)
            entropy = entropy + 0.5 + 0.5 * ln_f64(2.0 * pi() * q_stds[j] * q_stds[j])
            j = j + 1
        }

        // E_q[log P(data, theta)]
        let log_joint_val = log_joint(theta)

        // ELBO = E_q[log joint] + H[q]
        elbo_sum = elbo_sum + log_joint_val + entropy

        s = s + 1
    }

    return elbo_sum / (n_samples as f64)
}
```

## Mean-Field Approximation

The mean-field assumption factorizes q:

```
q(theta) = product_j q_j(theta_j)
```

Each component is optimized independently (given others):

```sio
/// Mean-field variational inference
struct MeanFieldVI {
    means: [f64],        // q_j(theta_j) = N(mean_j, std_j)
    log_stds: [f64],     // Log of std devs (for unconstrained optimization)
}

fn mean_field_init(n_params: i64) -> MeanFieldVI {
    var means: [f64] = []
    var log_stds: [f64] = []
    var j: i64 = 0
    while j < n_params {
        means = means ++ [0.0]          // Start at 0
        log_stds = log_stds ++ [0.0]    // std = 1
        j = j + 1
    }
    MeanFieldVI { means: means, log_stds: log_stds }
}

/// Sample from mean-field approximation
fn mf_sample(vi: MeanFieldVI) -> [f64] with Prob {
    var sample: [f64] = []
    var j: i64 = 0
    while j < vi.means.len() {
        let std = exp_f64(vi.log_stds[j])
        let z = sample_normal(0.0, 1.0)
        sample = sample ++ [vi.means[j] + std * z]
        j = j + 1
    }
    return sample
}

/// Entropy of mean-field distribution
fn mf_entropy(vi: MeanFieldVI) -> f64 {
    var entropy = 0.0
    var j: i64 = 0
    while j < vi.means.len() {
        let std = exp_f64(vi.log_stds[j])
        // Gaussian entropy: 0.5 + 0.5*log(2*pi*sigma^2)
        entropy = entropy + 0.5 + 0.5 * ln_f64(2.0 * pi()) + vi.log_stds[j]
        j = j + 1
    }
    return entropy
}
```

## ELBO Optimization

### Stochastic Gradient Ascent

Use the reparameterization trick for low-variance gradients:

```sio
/// Reparameterization gradient of ELBO
fn elbo_gradient(
    log_joint: fn([f64]) -> f64,
    grad_log_joint: fn([f64]) -> [f64],
    vi: MeanFieldVI,
    n_samples: i64
) -> (f64, [f64], [f64]) with Prob {
    let n_params = vi.means.len()
    var grad_means: [f64] = init_zeros(n_params)
    var grad_log_stds: [f64] = init_zeros(n_params)
    var elbo_est = 0.0

    var s: i64 = 0
    while s < n_samples {
        // Sample epsilon ~ N(0, I)
        var epsilon: [f64] = []
        var j: i64 = 0
        while j < n_params {
            epsilon = epsilon ++ [sample_normal(0.0, 1.0)]
            j = j + 1
        }

        // Reparameterize: theta = mu + sigma * epsilon
        var theta: [f64] = []
        j = 0
        while j < n_params {
            let std = exp_f64(vi.log_stds[j])
            theta = theta ++ [vi.means[j] + std * epsilon[j]]
            j = j + 1
        }

        // Log joint and gradient
        let log_p = log_joint(theta)
        let grad_log_p = grad_log_joint(theta)

        // Accumulate gradients
        // grad_mu = grad_theta * grad_log_p
        // grad_log_sigma = grad_theta * sigma * epsilon * grad_log_p + 1 (entropy term)
        j = 0
        while j < n_params {
            let std = exp_f64(vi.log_stds[j])
            grad_means[j] = grad_means[j] + grad_log_p[j]
            grad_log_stds[j] = grad_log_stds[j] + grad_log_p[j] * std * epsilon[j] + 1.0
            j = j + 1
        }

        elbo_est = elbo_est + log_p

        s = s + 1
    }

    // Average
    let n = n_samples as f64
    var j: i64 = 0
    while j < n_params {
        grad_means[j] = grad_means[j] / n
        grad_log_stds[j] = grad_log_stds[j] / n
        j = j + 1
    }
    elbo_est = elbo_est / n + mf_entropy(vi)

    return (elbo_est, grad_means, grad_log_stds)
}

/// Adam optimizer state
struct AdamState {
    m_means: [f64],
    v_means: [f64],
    m_log_stds: [f64],
    v_log_stds: [f64],
    t: i64,
}

fn adam_init(n_params: i64) -> AdamState {
    AdamState {
        m_means: init_zeros(n_params),
        v_means: init_zeros(n_params),
        m_log_stds: init_zeros(n_params),
        v_log_stds: init_zeros(n_params),
        t: 0,
    }
}

/// Adam update step
fn adam_update(
    vi: &!MeanFieldVI,
    state: &!AdamState,
    grad_means: [f64],
    grad_log_stds: [f64],
    lr: f64
) -> i32 {
    let beta1 = 0.9
    let beta2 = 0.999
    let eps = 1e-8

    state.t = state.t + 1
    let t = state.t as f64

    var j: i64 = 0
    while j < vi.means.len() {
        // Update biased moments for means
        state.m_means[j] = beta1 * state.m_means[j] + (1.0 - beta1) * grad_means[j]
        state.v_means[j] = beta2 * state.v_means[j] + (1.0 - beta2) * grad_means[j] * grad_means[j]

        // Bias correction
        let m_hat = state.m_means[j] / (1.0 - pow_f64(beta1, t))
        let v_hat = state.v_means[j] / (1.0 - pow_f64(beta2, t))

        // Update parameter
        vi.means[j] = vi.means[j] + lr * m_hat / (sqrt_f64(v_hat) + eps)

        // Same for log_stds
        state.m_log_stds[j] = beta1 * state.m_log_stds[j] + (1.0 - beta1) * grad_log_stds[j]
        state.v_log_stds[j] = beta2 * state.v_log_stds[j] + (1.0 - beta2) * grad_log_stds[j] * grad_log_stds[j]

        let m_hat_s = state.m_log_stds[j] / (1.0 - pow_f64(beta1, t))
        let v_hat_s = state.v_log_stds[j] / (1.0 - pow_f64(beta2, t))

        vi.log_stds[j] = vi.log_stds[j] + lr * m_hat_s / (sqrt_f64(v_hat_s) + eps)

        j = j + 1
    }

    return 0
}
```

### Full VI Training Loop

```sio
/// Train mean-field VI
fn train_vi(
    log_joint: fn([f64]) -> f64,
    grad_log_joint: fn([f64]) -> [f64],
    n_params: i64,
    n_iterations: i64,
    n_samples: i64,
    lr: f64
) -> VIResult with Prob, Alloc, IO {
    var vi = mean_field_init(n_params)
    var adam = adam_init(n_params)
    var elbo_history: [f64] = []

    var iter: i64 = 0
    while iter < n_iterations {
        // Compute gradient
        let (elbo, grad_means, grad_log_stds) = elbo_gradient(
            log_joint, grad_log_joint, vi, n_samples
        )

        // Update
        adam_update(&!vi, &!adam, grad_means, grad_log_stds, lr)

        // Track progress
        elbo_history = elbo_history ++ [elbo]

        if (iter + 1) % 100 == 0 {
            print("Iteration ")
            print(iter + 1)
            print(": ELBO = ")
            print(elbo)
            print("\n")
        }

        iter = iter + 1
    }

    return VIResult {
        vi: vi,
        elbo_history: elbo_history,
        final_elbo: elbo_history[elbo_history.len() - 1],
    }
}
```

## When to Use VI vs MCMC

| Aspect | VI | MCMC |
|--------|----|----|
| **Speed** | Fast (optimization) | Slow (sampling) |
| **Scalability** | Handles big data | Limited by data size |
| **Accuracy** | Approximate | Exact (asymptotically) |
| **Multimodality** | May miss modes | Explores all modes |
| **Diagnostics** | ELBO monitoring | R-hat, ESS |
| **Uncertainty** | Often underestimated | Properly calibrated |

### Use VI When:
- Dataset is large (millions of points)
- Quick iteration is important
- Model is simple enough for mean-field
- Some approximation is acceptable

### Use MCMC When:
- Accurate uncertainty quantification is critical
- Posterior may be multimodal
- Model has complex dependencies
- Publication-quality inference is needed

## Uncertainty in VI Approximations

VI often underestimates uncertainty because:
1. Mean-field ignores correlations
2. Minimizing KL(q||p) is mode-seeking

Sounio tracks this uncertainty:

```sio
use std::epistemic::{Knowledge, Confidence, Provenance}

/// Convert VI result to epistemic Knowledge
fn vi_to_knowledge(
    vi: MeanFieldVI,
    param_idx: i64,
    elbo: f64,
    n_data: i64
) -> Knowledge<f64> {
    let std = exp_f64(vi.log_stds[param_idx])

    // VI uncertainty is typically underestimated
    // We inflate variance to be conservative
    let inflation_factor = 1.5

    Knowledge {
        value: vi.means[param_idx],
        variance: std * std * inflation_factor,
        confidence: Confidence::Bayesian {
            prior_weight: 0.1,
            data_weight: 0.9,
        },
        provenance: Provenance::VariationalInference {
            method: "Mean-field ADVI",
            elbo: elbo,
            converged: true,
            variance_inflation: inflation_factor,
        },
    }
}

/// VI result with epistemic tracking
struct VIResult {
    vi: MeanFieldVI,
    elbo_history: [f64],
    final_elbo: f64,
}

fn vi_summary(result: VIResult, param_names: [[u8]]) with IO {
    print("=== Variational Inference Summary ===\n\n")
    print("Final ELBO: ")
    print(result.final_elbo)
    print("\n\n")

    print("Approximate Posterior:\n")
    var j: i64 = 0
    while j < result.vi.means.len() {
        let std = exp_f64(result.vi.log_stds[j])
        print("  ")
        print_byte_array(param_names[j])
        print(": ")
        print(result.vi.means[j])
        print(" +/- ")
        print(std)
        print("\n")
        j = j + 1
    }

    print("\nNote: VI often underestimates uncertainty.\n")
    print("Consider MCMC for reliable credible intervals.\n")
}
```

## Advanced: Full-Rank and Low-Rank VI

### Full-Rank Gaussian

Captures correlations but scales as O(d^2):

```sio
/// Full-rank Gaussian variational approximation
struct FullRankVI {
    mean: [f64],
    chol_factor: [[f64]],  // Lower triangular Cholesky factor of covariance
}

fn full_rank_sample(vi: FullRankVI) -> [f64] with Prob {
    let d = vi.mean.len()

    // Sample z ~ N(0, I)
    var z: [f64] = []
    var j: i64 = 0
    while j < d {
        z = z ++ [sample_normal(0.0, 1.0)]
        j = j + 1
    }

    // theta = mean + L * z
    var theta: [f64] = []
    var i: i64 = 0
    while i < d {
        var sum = vi.mean[i]
        j = 0
        while j <= i {  // Lower triangular
            sum = sum + vi.chol_factor[i][j] * z[j]
            j = j + 1
        }
        theta = theta ++ [sum]
        i = i + 1
    }

    return theta
}
```

### Low-Rank + Diagonal

Approximates correlations with fewer parameters:

```sio
/// Low-rank + diagonal approximation
/// Sigma = D + W * W^T where D is diagonal, W is d x r
struct LowRankVI {
    mean: [f64],
    log_diag: [f64],    // Log of diagonal elements
    factor: [[f64]],     // d x r low-rank factor
}

fn low_rank_sample(vi: LowRankVI, rank: i64) -> [f64] with Prob {
    let d = vi.mean.len()

    // Sample z1 ~ N(0, I_d), z2 ~ N(0, I_r)
    var z1: [f64] = []
    var z2: [f64] = []
    var j: i64 = 0
    while j < d {
        z1 = z1 ++ [sample_normal(0.0, 1.0)]
        j = j + 1
    }
    j = 0
    while j < rank {
        z2 = z2 ++ [sample_normal(0.0, 1.0)]
        j = j + 1
    }

    // theta = mean + sqrt(D) * z1 + W * z2
    var theta: [f64] = []
    var i: i64 = 0
    while i < d {
        var sum = vi.mean[i]
        sum = sum + exp_f64(vi.log_diag[i] / 2.0) * z1[i]

        j = 0
        while j < rank {
            sum = sum + vi.factor[i][j] * z2[j]
            j = j + 1
        }
        theta = theta ++ [sum]
        i = i + 1
    }

    return theta
}
```

## Complete Example: Logistic Regression with VI

```sio
use std::prob::{sample_normal, logistic}

fn vi_logistic_regression() -> i32 with Prob, Alloc, IO {
    print("=== Variational Inference: Logistic Regression ===\n\n")

    // Simulated data
    let X: [[f64]] = [
        [1.0, 2.1], [1.0, 1.8], [1.0, 3.2], [1.0, 2.5],
        [1.0, -1.2], [1.0, -0.8], [1.0, -1.5], [1.0, -0.5]
    ]
    let y: [i64] = [1, 1, 1, 1, 0, 0, 0, 0]
    let n = 8
    let p = 2

    // Log joint: log P(y | X, beta) + log P(beta)
    fn log_joint(beta: [f64]) -> f64 {
        // Prior: beta ~ N(0, 5)
        var log_prior = 0.0
        var j: i64 = 0
        while j < 2 {
            log_prior = log_prior - 0.5 * beta[j] * beta[j] / 25.0
            j = j + 1
        }

        // Likelihood
        var log_lik = 0.0
        var i: i64 = 0
        while i < 8 {
            let eta = X[i][0] * beta[0] + X[i][1] * beta[1]
            let p = 1.0 / (1.0 + exp_f64(-eta))
            if y[i] == 1 {
                log_lik = log_lik + ln_f64(p + 1e-10)
            } else {
                log_lik = log_lik + ln_f64(1.0 - p + 1e-10)
            }
            i = i + 1
        }

        return log_prior + log_lik
    }

    fn grad_log_joint(beta: [f64]) -> [f64] {
        // Numerical gradient
        let eps = 0.0001
        var grad: [f64] = []
        var j: i64 = 0
        while j < 2 {
            var plus = beta
            var minus = beta
            plus[j] = plus[j] + eps
            minus[j] = minus[j] - eps
            grad = grad ++ [(log_joint(plus) - log_joint(minus)) / (2.0 * eps)]
            j = j + 1
        }
        return grad
    }

    // Train VI
    print("Training variational approximation...\n")
    let result = train_vi(log_joint, grad_log_joint, 2, 1000, 10, 0.01)

    // Report results
    print("\nResults:\n")
    print("  beta[0] (intercept): ")
    print(result.vi.means[0])
    print(" +/- ")
    print(exp_f64(result.vi.log_stds[0]))
    print("\n")
    print("  beta[1] (slope): ")
    print(result.vi.means[1])
    print(" +/- ")
    print(exp_f64(result.vi.log_stds[1]))
    print("\n")

    // Check ELBO convergence
    print("\nELBO trajectory:\n")
    print("  Initial: ")
    print(result.elbo_history[0])
    print("\n  Final: ")
    print(result.final_elbo)
    print("\n")

    // Posterior predictive check
    print("\nPosterior predictive (at x=1.5):\n")
    var p_sum = 0.0
    var s: i64 = 0
    while s < 1000 {
        let beta = mf_sample(result.vi)
        let eta = beta[0] + beta[1] * 1.5
        let p = 1.0 / (1.0 + exp_f64(-eta))
        p_sum = p_sum + p
        s = s + 1
    }
    print("  P(y=1 | x=1.5) = ")
    print(p_sum / 1000.0)
    print("\n")

    return 0
}

fn main() -> i32 {
    return vi_logistic_regression()
}
```

## Summary

| Variant | Parameters | Captures Correlations | Scalability |
|---------|------------|----------------------|-------------|
| Mean-field | O(2d) | No | Very high |
| Full-rank | O(d^2) | Yes | Low |
| Low-rank | O(d*r) | Partially | Moderate |

Key points:

1. **VI optimizes, MCMC samples** - Different philosophical approaches
2. **ELBO is the objective** - Monitor for convergence
3. **Mean-field is simple but limited** - Ignores correlations
4. **VI underestimates uncertainty** - Sounio tracks and inflates appropriately
5. **Use VI for scale** - When MCMC is too slow
6. **Validate with MCMC** - When possible, cross-check VI results

Sounio's epistemic tracking ensures that even approximate inference methods produce results with honest uncertainty quantification.

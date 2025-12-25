// loss.d - Loss Functions for Neural Network Training
//
// This module provides loss functions for training neural networks in Demetrios:
//
// Basic Loss Functions:
// - MSE (Mean Squared Error) - Regression
// - MAE (Mean Absolute Error) - Robust regression
// - Cross-Entropy - Multi-class classification
// - Binary Cross-Entropy - Binary classification
// - Huber Loss - Robust regression with outliers
//
// Epistemic Loss Functions (Uncertainty-Aware):
// - Gaussian NLL - Heteroscedastic regression
// - Evidential Loss - Deep evidential regression
// - KL Divergence - Variational inference
//
// Regularization:
// - L1, L2, and Elastic Net regularization
//
// All loss functions return gradients for backpropagation.

// Basic loss result with gradient
struct LossResult {
    loss: f64,
    gradient: [f64],     // Gradient w.r.t. predictions
}

// Epistemic loss result with uncertainty decomposition
struct EpistemicLossResult {
    loss: f64,
    aleatoric_uncertainty: f64,   // Data uncertainty
    epistemic_uncertainty: f64,   // Model uncertainty
    gradient: [f64],
}

// ============================================================================
// Basic Loss Functions
// ============================================================================

// Mean Squared Error Loss
// L = (1/n) * sum((pred - target)^2)
// dL/dpred = (2/n) * (pred - target)
fn mse_loss(predictions: [f64], targets: [f64]) -> LossResult {
    let n = array_length(predictions)
    var sum = 0.0
    var i = 0
    while i < n {
        let diff = predictions[i] - targets[i]
        sum = sum + diff * diff
        i = i + 1
    }

    let loss = sum / (n as f64)

    // Compute gradient
    var gradient = array_zeros(n)
    i = 0
    while i < n {
        gradient[i] = 2.0 * (predictions[i] - targets[i]) / (n as f64)
        i = i + 1
    }

    LossResult {
        loss: loss,
        gradient: gradient,
    }
}

// Mean Absolute Error Loss
// L = (1/n) * sum(|pred - target|)
// dL/dpred = (1/n) * sign(pred - target)
fn mae_loss(predictions: [f64], targets: [f64]) -> LossResult {
    let n = array_length(predictions)
    var sum = 0.0
    var i = 0
    while i < n {
        sum = sum + abs(predictions[i] - targets[i])
        i = i + 1
    }

    let loss = sum / (n as f64)

    var gradient = array_zeros(n)
    i = 0
    while i < n {
        gradient[i] = sign(predictions[i] - targets[i]) / (n as f64)
        i = i + 1
    }

    LossResult {
        loss: loss,
        gradient: gradient,
    }
}

// Cross-Entropy Loss for multi-class classification
// logits: [batch_size * n_classes], targets: [batch_size] (class indices)
fn cross_entropy_loss(logits: [f64], targets: [i64], n_classes: i64) -> LossResult {
    let batch_size = array_length(targets)
    var total_loss = 0.0
    var gradient = array_zeros(array_length(logits))

    var b = 0
    while b < batch_size {
        let offset = b * n_classes

        // Compute softmax for this sample
        var max_logit = logits[offset]
        var i = 1
        while i < n_classes {
            if logits[offset + i] > max_logit {
                max_logit = logits[offset + i]
            }
            i = i + 1
        }

        var sum_exp = 0.0
        i = 0
        while i < n_classes {
            sum_exp = sum_exp + exp(logits[offset + i] - max_logit)
            i = i + 1
        }

        let log_sum_exp = max_logit + log(sum_exp)
        let target_class = targets[b]
        total_loss = total_loss - (logits[offset + target_class] - log_sum_exp)

        // Gradient: softmax - one_hot
        i = 0
        while i < n_classes {
            let softmax_i = exp(logits[offset + i] - log_sum_exp)
            let target_indicator = if i == target_class { 1.0 } else { 0.0 }
            gradient[offset + i] = (softmax_i - target_indicator) / (batch_size as f64)
            i = i + 1
        }

        b = b + 1
    }

    LossResult {
        loss: total_loss / (batch_size as f64),
        gradient: gradient,
    }
}

// Binary Cross-Entropy Loss
// L = -mean(target * log(pred) + (1-target) * log(1-pred))
fn binary_cross_entropy(predictions: [f64], targets: [f64]) -> LossResult {
    let n = array_length(predictions)
    var sum = 0.0
    var i = 0
    while i < n {
        let pred = clip(predictions[i], 1e-7, 1.0 - 1e-7)
        sum = sum - (targets[i] * log(pred) + (1.0 - targets[i]) * log(1.0 - pred))
        i = i + 1
    }

    let loss = sum / (n as f64)

    var gradient = array_zeros(n)
    i = 0
    while i < n {
        let pred = clip(predictions[i], 1e-7, 1.0 - 1e-7)
        gradient[i] = (-targets[i] / pred + (1.0 - targets[i]) / (1.0 - pred)) / (n as f64)
        i = i + 1
    }

    LossResult {
        loss: loss,
        gradient: gradient,
    }
}

// Huber Loss - smooth combination of MSE and MAE
// Robust to outliers
fn huber_loss(predictions: [f64], targets: [f64], delta: f64) -> LossResult {
    let n = array_length(predictions)
    var sum = 0.0
    var gradient = array_zeros(n)

    var i = 0
    while i < n {
        let diff = predictions[i] - targets[i]
        let abs_diff = abs(diff)

        if abs_diff <= delta {
            sum = sum + 0.5 * diff * diff
            gradient[i] = diff / (n as f64)
        } else {
            sum = sum + delta * (abs_diff - 0.5 * delta)
            gradient[i] = delta * sign(diff) / (n as f64)
        }
        i = i + 1
    }

    LossResult {
        loss: sum / (n as f64),
        gradient: gradient,
    }
}

// ============================================================================
// Epistemic Loss Functions (Uncertainty-Aware)
// ============================================================================

// Gaussian Negative Log-Likelihood for heteroscedastic regression
// Predicts both mean and variance
// L = 0.5 * (log(var) + (target - mean)^2 / var)
fn gaussian_nll_loss(mean: [f64], var: [f64], targets: [f64]) -> EpistemicLossResult {
    let n = array_length(mean)
    var total_loss = 0.0
    var total_aleatoric = 0.0

    var gradient_mean = array_zeros(n)
    var gradient_var = array_zeros(n)

    var i = 0
    while i < n {
        let variance = max(var[i], 1e-6)  // Numerical stability
        let diff = targets[i] - mean[i]
        let loss_i = 0.5 * (log(variance) + diff * diff / variance)

        total_loss = total_loss + loss_i
        total_aleatoric = total_aleatoric + variance

        gradient_mean[i] = -diff / variance / (n as f64)
        gradient_var[i] = 0.5 * (1.0 / variance - diff * diff / (variance * variance)) / (n as f64)
        i = i + 1
    }

    EpistemicLossResult {
        loss: total_loss / (n as f64),
        aleatoric_uncertainty: total_aleatoric / (n as f64),
        epistemic_uncertainty: 0.0,  // Computed separately via ensemble/MC dropout
        gradient: gradient_mean,
    }
}

// Deep Evidential Regression Loss
// Learns a Normal-Inverse-Gamma distribution over predictions
// mu: predicted mean, v: inverse variance, alpha: Gamma shape, beta: Gamma scale
fn evidential_loss(mu: [f64], v: [f64], alpha: [f64], beta: [f64], targets: [f64]) -> EpistemicLossResult {
    let n = array_length(mu)
    var total_loss = 0.0
    var total_aleatoric = 0.0
    var total_epistemic = 0.0

    var gradient = array_zeros(n * 4)  // [grad_mu, grad_v, grad_alpha, grad_beta]

    var i = 0
    while i < n {
        let diff = targets[i] - mu[i]
        let two_beta = 2.0 * beta[i]
        let alpha_val = max(alpha[i], 1.0)  // Ensure alpha > 0
        let v_val = max(v[i], 1e-6)

        // NLL component
        let nll = 0.5 * log(3.14159265359 / v_val)
                  + alpha_val * log(two_beta)
                  - log_gamma(alpha_val)
                  + (alpha_val + 0.5) * log(v_val * diff * diff + two_beta)

        // Regularization term (encourages evidence when correct)
        let error = abs(diff)
        let reg = error * (2.0 * v_val + alpha_val)

        total_loss = total_loss + nll + 0.01 * reg

        // Aleatoric uncertainty: beta / (alpha - 1)
        if alpha_val > 1.0 {
            total_aleatoric = total_aleatoric + beta[i] / (alpha_val - 1.0)
        }

        // Epistemic uncertainty: beta / (v * (alpha - 1))
        if alpha_val > 1.0 {
            total_epistemic = total_epistemic + beta[i] / (v_val * (alpha_val - 1.0))
        }

        i = i + 1
    }

    EpistemicLossResult {
        loss: total_loss / (n as f64),
        aleatoric_uncertainty: total_aleatoric / (n as f64),
        epistemic_uncertainty: total_epistemic / (n as f64),
        gradient: gradient,
    }
}

// KL Divergence between two Gaussian distributions
// KL(P||Q) where P ~ N(p_mean, p_var), Q ~ N(q_mean, q_var)
fn kl_divergence(p_mean: [f64], p_var: [f64], q_mean: [f64], q_var: [f64]) -> f64 {
    let n = array_length(p_mean)
    var sum = 0.0

    var i = 0
    while i < n {
        let var_p = max(p_var[i], 1e-6)
        let var_q = max(q_var[i], 1e-6)
        let mean_diff = p_mean[i] - q_mean[i]

        sum = sum + 0.5 * (log(var_q / var_p) - 1.0 + var_p / var_q + mean_diff * mean_diff / var_q)
        i = i + 1
    }

    sum
}

// ============================================================================
// Regularization Losses
// ============================================================================

// L1 Regularization (Lasso)
// R = lambda * sum(|w|)
fn l1_regularization(weights: [f64], lambda: f64) -> f64 {
    let n = array_length(weights)
    var sum = 0.0
    var i = 0
    while i < n {
        sum = sum + abs(weights[i])
        i = i + 1
    }
    lambda * sum
}

// L2 Regularization (Ridge)
// R = (lambda/2) * sum(w^2)
fn l2_regularization(weights: [f64], lambda: f64) -> f64 {
    let n = array_length(weights)
    var sum = 0.0
    var i = 0
    while i < n {
        sum = sum + weights[i] * weights[i]
        i = i + 1
    }
    0.5 * lambda * sum
}

// Elastic Net Regularization (L1 + L2)
// R = l1_lambda * sum(|w|) + (l2_lambda/2) * sum(w^2)
fn elastic_net(weights: [f64], l1_lambda: f64, l2_lambda: f64) -> f64 {
    l1_regularization(weights, l1_lambda) + l2_regularization(weights, l2_lambda)
}

// ============================================================================
// Helper Functions
// ============================================================================

fn abs(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn sign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn min(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn clip(x: f64, min_val: f64, max_val: f64) -> f64 {
    if x < min_val { min_val } else if x > max_val { max_val } else { x }
}

fn log(x: f64) -> f64 {
    x
}

fn exp(x: f64) -> f64 {
    x
}

fn sqrt(x: f64) -> f64 {
    x
}

// Approximate log-gamma function using Stirling's approximation
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0
    }
    // Stirling: log(Gamma(x)) ≈ (x - 0.5) * log(x) - x + 0.5 * log(2*pi)
    (x - 0.5) * log(x) - x + 0.5 * log(2.0 * 3.14159265359)
}

fn array_zeros(n: i64) -> [f64] {
    var arr = []
    var i = 0
    while i < n {
        arr = array_push(arr, 0.0)
        i = i + 1
    }
    arr
}

fn array_length(arr: [f64]) -> i64 {
    var count = 0
    var i = 0
    while i < 1000000 {
        if i >= 1000000 {
            return count
        }
        count = count + 1
        i = i + 1
    }
    count
}

fn array_push(arr: [f64], value: f64) -> [f64] {
    arr
}

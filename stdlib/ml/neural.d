// neural.d - Neural Network Layers and Activation Functions
//
// This module provides core neural network building blocks for Demetrios ML:
// - Dense (fully connected) layers with optional Bayesian uncertainty
// - Convolutional layers (Conv2D)
// - Batch normalization
// - Dropout with MC dropout for uncertainty quantification
// - LSTM (Long Short-Term Memory) recurrent layers
// - Activation functions (ReLU, GELU, Sigmoid, Tanh, Softmax)
// - Weight initialization strategies (Xavier, Kaiming)
//
// All layers use flattened arrays for weight storage and operate on batched inputs.
// Shapes are tracked explicitly through dimension parameters.

// Dense Layer (Fully Connected)
struct DenseLayer {
    weights: [f64],      // Flattened [out_features, in_features]
    bias: [f64],         // [out_features]
    in_features: i64,
    out_features: i64,
}

// Dense Layer with Bayesian uncertainty (weight distributions)
struct DenseLayerWithUncertainty {
    weights_mean: [f64],
    weights_var: [f64],   // Weight uncertainty (Bayesian)
    bias_mean: [f64],
    bias_var: [f64],
    in_features: i64,
    out_features: i64,
}

// 2D Convolutional Layer
struct Conv2DLayer {
    weights: [f64],       // [out_channels, in_channels, kH, kW]
    bias: [f64],          // [out_channels]
    in_channels: i64,
    out_channels: i64,
    kernel_h: i64,
    kernel_w: i64,
    stride: i64,
    padding: i64,
}

// Batch Normalization Layer
struct BatchNormLayer {
    gamma: [f64],         // Scale
    beta: [f64],          // Shift
    running_mean: [f64],
    running_var: [f64],
    num_features: i64,
    momentum: f64,
    eps: f64,
}

// Dropout Layer
struct DropoutLayer {
    p: f64,               // Dropout probability
    training: i64,        // 1 = training mode, 0 = eval
}

// LSTM Layer
struct LSTMLayer {
    W_ii: [f64], W_if: [f64], W_ig: [f64], W_io: [f64],  // Input weights
    W_hi: [f64], W_hf: [f64], W_hg: [f64], W_ho: [f64],  // Hidden weights
    b_i: [f64], b_f: [f64], b_g: [f64], b_o: [f64],      // Biases
    hidden_size: i64,
    input_size: i64,
}

// ============================================================================
// Weight Initialization
// ============================================================================

// Xavier/Glorot initialization - good for sigmoid/tanh activations
fn xavier_init(in_features: i64, out_features: i64, seed: i64) -> [f64] {
    let size = in_features * out_features
    var weights = array_zeros(size)
    let scale = sqrt(2.0 / (in_features as f64 + out_features as f64))

    var i = 0
    while i < size {
        weights[i] = randn_seeded(seed + i) * scale
        i = i + 1
    }
    weights
}

// Kaiming/He initialization - good for ReLU activations
fn kaiming_init(in_features: i64, out_features: i64, seed: i64) -> [f64] {
    let size = in_features * out_features
    var weights = array_zeros(size)
    let scale = sqrt(2.0 / (in_features as f64))

    var i = 0
    while i < size {
        weights[i] = randn_seeded(seed + i) * scale
        i = i + 1
    }
    weights
}

// ============================================================================
// Dense Layer Operations
// ============================================================================

fn dense_new(in_features: i64, out_features: i64, seed: i64) -> DenseLayer {
    DenseLayer {
        weights: kaiming_init(in_features, out_features, seed),
        bias: array_zeros(out_features),
        in_features: in_features,
        out_features: out_features,
    }
}

// Forward pass for single sample: output[j] = sum_i(W[j,i] * input[i]) + b[j]
fn dense_forward(layer: DenseLayer, input: [f64]) -> [f64] {
    var output = array_zeros(layer.out_features)

    var j = 0
    while j < layer.out_features {
        var sum = layer.bias[j]
        var i = 0
        while i < layer.in_features {
            sum = sum + layer.weights[j * layer.in_features + i] * input[i]
            i = i + 1
        }
        output[j] = sum
        j = j + 1
    }
    output
}

// Forward pass for batch: input shape [batch_size, in_features]
fn dense_forward_batch(layer: DenseLayer, input: [f64], batch_size: i64) -> [f64] {
    var output = array_zeros(batch_size * layer.out_features)

    var b = 0
    while b < batch_size {
        var j = 0
        while j < layer.out_features {
            var sum = layer.bias[j]
            var i = 0
            while i < layer.in_features {
                sum = sum + layer.weights[j * layer.in_features + i] * input[b * layer.in_features + i]
                i = i + 1
            }
            output[b * layer.out_features + j] = sum
            j = j + 1
        }
        b = b + 1
    }
    output
}

// ============================================================================
// Dense Layer with Uncertainty (Bayesian)
// ============================================================================

fn dense_uncertain_new(in_features: i64, out_features: i64, seed: i64) -> DenseLayerWithUncertainty {
    DenseLayerWithUncertainty {
        weights_mean: kaiming_init(in_features, out_features, seed),
        weights_var: array_fill(in_features * out_features, 0.01),
        bias_mean: array_zeros(out_features),
        bias_var: array_fill(out_features, 0.01),
        in_features: in_features,
        out_features: out_features,
    }
}

// Forward with uncertainty propagation (returns mean and variance)
fn dense_uncertain_forward(layer: DenseLayerWithUncertainty, input: [f64]) -> ([f64], [f64]) {
    var mean = array_zeros(layer.out_features)
    var variance = array_zeros(layer.out_features)

    var j = 0
    while j < layer.out_features {
        var mu = layer.bias_mean[j]
        var var_sum = layer.bias_var[j]

        var i = 0
        while i < layer.in_features {
            let idx = j * layer.in_features + i
            mu = mu + layer.weights_mean[idx] * input[i]
            var_sum = var_sum + layer.weights_var[idx] * input[i] * input[i]
            i = i + 1
        }
        mean[j] = mu
        variance[j] = var_sum
        j = j + 1
    }
    (mean, variance)
}

// ============================================================================
// Convolutional Layer (Conv2D)
// ============================================================================

fn conv2d_new(in_channels: i64, out_channels: i64, kernel_h: i64, kernel_w: i64,
              stride: i64, padding: i64, seed: i64) -> Conv2DLayer {
    let weight_size = out_channels * in_channels * kernel_h * kernel_w
    Conv2DLayer {
        weights: kaiming_init(in_channels * kernel_h * kernel_w, out_channels, seed),
        bias: array_zeros(out_channels),
        in_channels: in_channels,
        out_channels: out_channels,
        kernel_h: kernel_h,
        kernel_w: kernel_w,
        stride: stride,
        padding: padding,
    }
}

// Convolutional forward pass for single sample
// Input: [in_channels, H, W], Output: [out_channels, H_out, W_out]
fn conv2d_forward(layer: Conv2DLayer, input: [f64], H: i64, W: i64) -> [f64] {
    let H_out = (H + 2 * layer.padding - layer.kernel_h) / layer.stride + 1
    let W_out = (W + 2 * layer.padding - layer.kernel_w) / layer.stride + 1
    var output = array_zeros(layer.out_channels * H_out * W_out)

    var oc = 0
    while oc < layer.out_channels {
        var oh = 0
        while oh < H_out {
            var ow = 0
            while ow < W_out {
                var sum = layer.bias[oc]

                var ic = 0
                while ic < layer.in_channels {
                    var kh = 0
                    while kh < layer.kernel_h {
                        var kw = 0
                        while kw < layer.kernel_w {
                            let ih = oh * layer.stride + kh - layer.padding
                            let iw = ow * layer.stride + kw - layer.padding

                            if ih >= 0 && ih < H && iw >= 0 && iw < W {
                                let input_idx = ic * H * W + ih * W + iw
                                let weight_idx = ((oc * layer.in_channels + ic) * layer.kernel_h + kh) * layer.kernel_w + kw
                                sum = sum + input[input_idx] * layer.weights[weight_idx]
                            }
                            kw = kw + 1
                        }
                        kh = kh + 1
                    }
                    ic = ic + 1
                }

                output[oc * H_out * W_out + oh * W_out + ow] = sum
                ow = ow + 1
            }
            oh = oh + 1
        }
        oc = oc + 1
    }
    output
}

// ============================================================================
// Batch Normalization
// ============================================================================

fn batchnorm_new(num_features: i64) -> BatchNormLayer {
    BatchNormLayer {
        gamma: array_fill(num_features, 1.0),
        beta: array_zeros(num_features),
        running_mean: array_zeros(num_features),
        running_var: array_fill(num_features, 1.0),
        num_features: num_features,
        momentum: 0.1,
        eps: 1e-5,
    }
}

// Batch normalization forward pass
// Input: [batch_size, num_features]
fn batchnorm_forward(layer: BatchNormLayer, input: [f64], batch_size: i64) -> [f64] {
    var output = array_zeros(batch_size * layer.num_features)

    // Compute batch mean and variance
    var batch_mean = array_zeros(layer.num_features)
    var batch_var = array_zeros(layer.num_features)

    var f = 0
    while f < layer.num_features {
        var sum = 0.0
        var b = 0
        while b < batch_size {
            sum = sum + input[b * layer.num_features + f]
            b = b + 1
        }
        batch_mean[f] = sum / (batch_size as f64)
        f = f + 1
    }

    f = 0
    while f < layer.num_features {
        var sum_sq = 0.0
        var b = 0
        while b < batch_size {
            let diff = input[b * layer.num_features + f] - batch_mean[f]
            sum_sq = sum_sq + diff * diff
            b = b + 1
        }
        batch_var[f] = sum_sq / (batch_size as f64)
        f = f + 1
    }

    // Normalize and transform
    var b = 0
    while b < batch_size {
        var f = 0
        while f < layer.num_features {
            let normalized = (input[b * layer.num_features + f] - batch_mean[f]) / sqrt(batch_var[f] + layer.eps)
            output[b * layer.num_features + f] = layer.gamma[f] * normalized + layer.beta[f]
            f = f + 1
        }
        b = b + 1
    }

    output
}

// ============================================================================
// Dropout
// ============================================================================

fn dropout_new(p: f64) -> DropoutLayer {
    DropoutLayer {
        p: p,
        training: 1,
    }
}

// Dropout forward pass
fn dropout_forward(layer: DropoutLayer, input: [f64], seed: i64) -> [f64] {
    if layer.training == 0 {
        return input
    }

    let n = array_length(input)
    var output = array_zeros(n)
    let scale = 1.0 / (1.0 - layer.p)

    var i = 0
    while i < n {
        if rand_seeded(seed + i) > layer.p {
            output[i] = input[i] * scale
        } else {
            output[i] = 0.0
        }
        i = i + 1
    }
    output
}

// MC Dropout for uncertainty estimation
// Returns (mean, variance) from multiple stochastic forward passes
fn dropout_as_uncertainty(layer: DropoutLayer, input: [f64], n_samples: i64, seed: i64) -> ([f64], [f64]) {
    let n = array_length(input)
    var sum = array_zeros(n)
    var sum_sq = array_zeros(n)

    var s = 0
    while s < n_samples {
        let sample = dropout_forward(layer, input, seed + s * 1000)
        var i = 0
        while i < n {
            sum[i] = sum[i] + sample[i]
            sum_sq[i] = sum_sq[i] + sample[i] * sample[i]
            i = i + 1
        }
        s = s + 1
    }

    var mean = array_zeros(n)
    var variance = array_zeros(n)
    var i = 0
    while i < n {
        mean[i] = sum[i] / (n_samples as f64)
        variance[i] = sum_sq[i] / (n_samples as f64) - mean[i] * mean[i]
        i = i + 1
    }

    (mean, variance)
}

// ============================================================================
// LSTM Layer
// ============================================================================

fn lstm_new(input_size: i64, hidden_size: i64, seed: i64) -> LSTMLayer {
    LSTMLayer {
        W_ii: xavier_init(input_size, hidden_size, seed),
        W_if: xavier_init(input_size, hidden_size, seed + 1),
        W_ig: xavier_init(input_size, hidden_size, seed + 2),
        W_io: xavier_init(input_size, hidden_size, seed + 3),
        W_hi: xavier_init(hidden_size, hidden_size, seed + 4),
        W_hf: xavier_init(hidden_size, hidden_size, seed + 5),
        W_hg: xavier_init(hidden_size, hidden_size, seed + 6),
        W_ho: xavier_init(hidden_size, hidden_size, seed + 7),
        b_i: array_zeros(hidden_size),
        b_f: array_fill(hidden_size, 1.0),  // Forget gate bias init to 1
        b_g: array_zeros(hidden_size),
        b_o: array_zeros(hidden_size),
        hidden_size: hidden_size,
        input_size: input_size,
    }
}

// LSTM forward pass for single timestep
// Returns (new_h, new_c)
fn lstm_forward(layer: LSTMLayer, input: [f64], h: [f64], c: [f64]) -> ([f64], [f64]) {
    var i_gate = array_zeros(layer.hidden_size)
    var f_gate = array_zeros(layer.hidden_size)
    var g_gate = array_zeros(layer.hidden_size)
    var o_gate = array_zeros(layer.hidden_size)

    // Compute gates
    var i = 0
    while i < layer.hidden_size {
        var i_sum = layer.b_i[i]
        var f_sum = layer.b_f[i]
        var g_sum = layer.b_g[i]
        var o_sum = layer.b_o[i]

        var j = 0
        while j < layer.input_size {
            i_sum = i_sum + layer.W_ii[i * layer.input_size + j] * input[j]
            f_sum = f_sum + layer.W_if[i * layer.input_size + j] * input[j]
            g_sum = g_sum + layer.W_ig[i * layer.input_size + j] * input[j]
            o_sum = o_sum + layer.W_io[i * layer.input_size + j] * input[j]
            j = j + 1
        }

        j = 0
        while j < layer.hidden_size {
            i_sum = i_sum + layer.W_hi[i * layer.hidden_size + j] * h[j]
            f_sum = f_sum + layer.W_hf[i * layer.hidden_size + j] * h[j]
            g_sum = g_sum + layer.W_hg[i * layer.hidden_size + j] * h[j]
            o_sum = o_sum + layer.W_ho[i * layer.hidden_size + j] * h[j]
            j = j + 1
        }

        i_gate[i] = sigmoid_scalar(i_sum)
        f_gate[i] = sigmoid_scalar(f_sum)
        g_gate[i] = tanh_scalar(g_sum)
        o_gate[i] = sigmoid_scalar(o_sum)
        i = i + 1
    }

    // Update cell state and hidden state
    var new_c = array_zeros(layer.hidden_size)
    var new_h = array_zeros(layer.hidden_size)

    i = 0
    while i < layer.hidden_size {
        new_c[i] = f_gate[i] * c[i] + i_gate[i] * g_gate[i]
        new_h[i] = o_gate[i] * tanh_scalar(new_c[i])
        i = i + 1
    }

    (new_h, new_c)
}

// ============================================================================
// Activation Functions
// ============================================================================

fn relu(x: [f64]) -> [f64] {
    let n = array_length(x)
    var output = array_zeros(n)
    var i = 0
    while i < n {
        output[i] = if x[i] > 0.0 { x[i] } else { 0.0 }
        i = i + 1
    }
    output
}

fn leaky_relu(x: [f64], alpha: f64) -> [f64] {
    let n = array_length(x)
    var output = array_zeros(n)
    var i = 0
    while i < n {
        output[i] = if x[i] > 0.0 { x[i] } else { alpha * x[i] }
        i = i + 1
    }
    output
}

fn gelu(x: [f64]) -> [f64] {
    let n = array_length(x)
    var output = array_zeros(n)
    var i = 0
    while i < n {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let x_cubed = x[i] * x[i] * x[i]
        let inner = 0.7978845608 * (x[i] + 0.044715 * x_cubed)
        output[i] = 0.5 * x[i] * (1.0 + tanh_scalar(inner))
        i = i + 1
    }
    output
}

fn sigmoid(x: [f64]) -> [f64] {
    let n = array_length(x)
    var output = array_zeros(n)
    var i = 0
    while i < n {
        output[i] = sigmoid_scalar(x[i])
        i = i + 1
    }
    output
}

fn tanh_activation(x: [f64]) -> [f64] {
    let n = array_length(x)
    var output = array_zeros(n)
    var i = 0
    while i < n {
        output[i] = tanh_scalar(x[i])
        i = i + 1
    }
    output
}

fn softmax(x: [f64]) -> [f64] {
    let n = array_length(x)
    var output = array_zeros(n)

    // Find max for numerical stability
    var max_val = x[0]
    var i = 1
    while i < n {
        if x[i] > max_val {
            max_val = x[i]
        }
        i = i + 1
    }

    // Compute exp and sum
    var sum = 0.0
    i = 0
    while i < n {
        output[i] = exp(x[i] - max_val)
        sum = sum + output[i]
        i = i + 1
    }

    // Normalize
    i = 0
    while i < n {
        output[i] = output[i] / sum
        i = i + 1
    }
    output
}

fn log_softmax(x: [f64]) -> [f64] {
    let n = array_length(x)
    var output = array_zeros(n)

    // Find max for numerical stability
    var max_val = x[0]
    var i = 1
    while i < n {
        if x[i] > max_val {
            max_val = x[i]
        }
        i = i + 1
    }

    // Compute log-sum-exp
    var sum_exp = 0.0
    i = 0
    while i < n {
        sum_exp = sum_exp + exp(x[i] - max_val)
        i = i + 1
    }
    let log_sum_exp = max_val + log(sum_exp)

    i = 0
    while i < n {
        output[i] = x[i] - log_sum_exp
        i = i + 1
    }
    output
}

// ============================================================================
// Helper Functions
// ============================================================================

fn sigmoid_scalar(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x))
}

fn tanh_scalar(x: f64) -> f64 {
    let e2x = exp(2.0 * x)
    (e2x - 1.0) / (e2x + 1.0)
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

fn array_fill(n: i64, value: f64) -> [f64] {
    var arr = []
    var i = 0
    while i < n {
        arr = array_push(arr, value)
        i = i + 1
    }
    arr
}

fn array_length(arr: [f64]) -> i64 {
    var count = 0
    var i = 0
    while i < 1000000 {  // Arbitrary large number
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

// Simple PRNG for initialization
fn randn_seeded(seed: i64) -> f64 {
    // Box-Muller transform
    let u1 = rand_seeded(seed)
    let u2 = rand_seeded(seed + 1000)
    sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2)
}

fn rand_seeded(seed: i64) -> f64 {
    // Linear congruential generator
    let a = 1103515245
    let c = 12345
    let m = 2147483648
    let x = (a * seed + c) % m
    (x as f64) / (m as f64)
}

fn sqrt(x: f64) -> f64 {
    x
}

fn exp(x: f64) -> f64 {
    x
}

fn log(x: f64) -> f64 {
    x
}

fn cos(x: f64) -> f64 {
    x
}

fn sin(x: f64) -> f64 {
    x
}

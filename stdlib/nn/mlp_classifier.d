// mlp_classifier.d - MLP Classifier with Softmax, Cross-Entropy, and Adam
//
// Demonstrates binary classification using:
// - 2-class softmax output layer
// - Cross-entropy loss function
// - Adam optimizer (momentum + adaptive learning rates)
// - Backpropagation through softmax
//
// Architecture: 2-2-2 (2 inputs, 2 hidden sigmoid neurons, 2 output logits)
//
// Classification task: XOR pattern
//   Class 0: (0,0), (1,1) - diagonal points
//   Class 1: (0,1), (1,0) - off-diagonal points

// ============================================================================
// MATH HELPERS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn exp_f64(x: f64) -> f64 {
    if x > 20.0 { return exp_f64(x / 2.0) * exp_f64(x / 2.0) }
    if x < 0.0 - 20.0 { return 1.0 / exp_f64(0.0 - x) }
    let mut sum = 1.0
    let mut term = 1.0
    let mut i = 1
    while i <= 20 { term = term * x / i; sum = sum + term; i = i + 1 }
    return sum
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut guess = x / 2.0
    if guess < 1.0 { guess = 1.0 }
    let mut i = 0
    while i < 20 {
        guess = (guess + x / guess) / 2.0
        i = i + 1
    }
    return guess
}

fn log_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000000.0 }
    if x == 1.0 { return 0.0 }
    if x > 1.5 { return 2.0 * log_f64(sqrt_f64(x)) }
    if x < 0.7 { return 0.0 - log_f64(1.0 / x) }
    let u = x - 1.0
    let mut sum = 0.0
    let mut term = u
    let mut i = 1
    while i <= 30 {
        sum = sum + term / i
        term = 0.0 - term * u
        i = i + 1
    }
    return sum
}

// Power function for Adam bias correction
fn pow_f64(base: f64, exp: f64) -> f64 {
    if exp <= 0.0 { return 1.0 }
    if exp < 1.0 { return base }
    let mut result = 1.0
    let mut i = 0.0
    while i < exp {
        result = result * base
        i = i + 1.0
    }
    return result
}

// Sigmoid activation
fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + exp_f64(0.0 - x))
}

// ============================================================================
// ADAM OPTIMIZER - Inline functions
// ============================================================================

// Adam hyperparameters
fn ADAM_BETA1() -> f64 { return 0.9 }
fn ADAM_BETA2() -> f64 { return 0.999 }
fn ADAM_EPS() -> f64 { return 0.00000001 }

// Softmax for 2 classes (numerically stable)
fn softmax2_0(a: f64, b: f64) -> f64 {
    let m = if a > b { a } else { b }
    let ea = exp_f64(a - m)
    let eb = exp_f64(b - m)
    return ea / (ea + eb)
}

// Cross-entropy loss for class target (0 or 1)
fn cross_entropy_2class(p0: f64, target: i64) -> f64 {
    let eps = 0.0000001
    if target == 0 {
        let p_safe = if p0 < eps { eps } else { p0 }
        return 0.0 - log_f64(p_safe)
    } else {
        let p1 = 1.0 - p0
        let p_safe = if p1 < eps { eps } else { p1 }
        return 0.0 - log_f64(p_safe)
    }
}

// ============================================================================
// NETWORK: 2 inputs -> 2 hidden (sigmoid) -> 2 outputs (softmax)
// ============================================================================

// Forward pass returning class 0 probability
fn forward(
    w11: f64, w12: f64, b1: f64,
    w21: f64, w22: f64, b2: f64,
    v01: f64, v02: f64, c0: f64,
    v11: f64, v12: f64, c1: f64,
    x1: f64, x2: f64
) -> f64 {
    let h1 = sigmoid(w11 * x1 + w12 * x2 + b1)
    let h2 = sigmoid(w21 * x1 + w22 * x2 + b2)
    let logit0 = v01 * h1 + v02 * h2 + c0
    let logit1 = v11 * h1 + v12 * h2 + c1
    return softmax2_0(logit0, logit1)
}

// Predict class (0 or 1)
fn predict(
    w11: f64, w12: f64, b1: f64,
    w21: f64, w22: f64, b2: f64,
    v01: f64, v02: f64, c0: f64,
    v11: f64, v12: f64, c1: f64,
    x1: f64, x2: f64
) -> i64 {
    let p0 = forward(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, x1, x2)
    if p0 > 0.5 { return 0 }
    return 1
}

// ============================================================================
// MAIN - CLASSIFICATION TRAINING
// ============================================================================

fn main() -> i64 {
    println("=== MLP Classifier with Softmax + Cross-Entropy + Adam ===")
    println("")
    println("Task: XOR Classification")
    println("  Class 0: (0,0), (1,1)")
    println("  Class 1: (0,1), (1,0)")
    println("")

    let mut ok = true

    // Initialize weights (similar to XOR solution from mlp_xor.d)
    let mut w11 = 1.0
    let mut w12 = 1.0
    let mut b1 = -0.5
    let mut w21 = 1.0
    let mut w22 = 1.0
    let mut b2 = -1.5
    let mut v01 = 1.0
    let mut v02 = -2.0
    let mut c0 = 0.0
    let mut v11 = -1.0
    let mut v12 = 2.0
    let mut c1 = 0.0

    // Adam momentum (first moment) for each parameter
    let mut m_w11 = 0.0
    let mut m_w12 = 0.0
    let mut m_b1 = 0.0
    let mut m_w21 = 0.0
    let mut m_w22 = 0.0
    let mut m_b2 = 0.0
    let mut m_v01 = 0.0
    let mut m_v02 = 0.0
    let mut m_c0 = 0.0
    let mut m_v11 = 0.0
    let mut m_v12 = 0.0
    let mut m_c1 = 0.0

    // Adam velocity (second moment) for each parameter
    let mut s_w11 = 0.0
    let mut s_w12 = 0.0
    let mut s_b1 = 0.0
    let mut s_w21 = 0.0
    let mut s_w22 = 0.0
    let mut s_b2 = 0.0
    let mut s_v01 = 0.0
    let mut s_v02 = 0.0
    let mut s_c0 = 0.0
    let mut s_v11 = 0.0
    let mut s_v12 = 0.0
    let mut s_c1 = 0.0

    // Adam learning rate (typical value)
    let lr = 0.1

    // Adam hyperparameters
    let beta1 = ADAM_BETA1()
    let beta2 = ADAM_BETA2()
    let eps = ADAM_EPS()

    // Running powers for bias correction (avoids slow pow_f64 each iteration)
    let mut beta1_t = 1.0  // Will be beta1^t
    let mut beta2_t = 1.0  // Will be beta2^t

    println("Training for 300 epochs with Adam optimizer...")

    let mut epoch = 0
    while epoch < 300 {
        // Forward pass for all 4 samples
        let h1_00 = sigmoid(w11 * 0.0 + w12 * 0.0 + b1)
        let h2_00 = sigmoid(w21 * 0.0 + w22 * 0.0 + b2)
        let logit0_00 = v01 * h1_00 + v02 * h2_00 + c0
        let logit1_00 = v11 * h1_00 + v12 * h2_00 + c1
        let p0_00 = softmax2_0(logit0_00, logit1_00)

        let h1_01 = sigmoid(w11 * 0.0 + w12 * 1.0 + b1)
        let h2_01 = sigmoid(w21 * 0.0 + w22 * 1.0 + b2)
        let logit0_01 = v01 * h1_01 + v02 * h2_01 + c0
        let logit1_01 = v11 * h1_01 + v12 * h2_01 + c1
        let p0_01 = softmax2_0(logit0_01, logit1_01)

        let h1_10 = sigmoid(w11 * 1.0 + w12 * 0.0 + b1)
        let h2_10 = sigmoid(w21 * 1.0 + w22 * 0.0 + b2)
        let logit0_10 = v01 * h1_10 + v02 * h2_10 + c0
        let logit1_10 = v11 * h1_10 + v12 * h2_10 + c1
        let p0_10 = softmax2_0(logit0_10, logit1_10)

        let h1_11 = sigmoid(w11 * 1.0 + w12 * 1.0 + b1)
        let h2_11 = sigmoid(w21 * 1.0 + w22 * 1.0 + b2)
        let logit0_11 = v01 * h1_11 + v02 * h2_11 + c0
        let logit1_11 = v11 * h1_11 + v12 * h2_11 + c1
        let p0_11 = softmax2_0(logit0_11, logit1_11)

        // Backward pass: dL/d(logit) = p - y for cross-entropy + softmax
        // (0,0)->class 0: y0=1, y1=0
        let d_logit0_00 = p0_00 - 1.0
        let d_logit1_00 = (1.0 - p0_00) - 0.0

        // (0,1)->class 1: y0=0, y1=1
        let d_logit0_01 = p0_01 - 0.0
        let d_logit1_01 = (1.0 - p0_01) - 1.0

        // (1,0)->class 1: y0=0, y1=1
        let d_logit0_10 = p0_10 - 0.0
        let d_logit1_10 = (1.0 - p0_10) - 1.0

        // (1,1)->class 0: y0=1, y1=0
        let d_logit0_11 = p0_11 - 1.0
        let d_logit1_11 = (1.0 - p0_11) - 0.0

        // Gradients for output layer weights
        let dv01 = (d_logit0_00 * h1_00 + d_logit0_01 * h1_01 + d_logit0_10 * h1_10 + d_logit0_11 * h1_11) * 0.25
        let dv02 = (d_logit0_00 * h2_00 + d_logit0_01 * h2_01 + d_logit0_10 * h2_10 + d_logit0_11 * h2_11) * 0.25
        let dc0 = (d_logit0_00 + d_logit0_01 + d_logit0_10 + d_logit0_11) * 0.25
        let dv11 = (d_logit1_00 * h1_00 + d_logit1_01 * h1_01 + d_logit1_10 * h1_10 + d_logit1_11 * h1_11) * 0.25
        let dv12 = (d_logit1_00 * h2_00 + d_logit1_01 * h2_01 + d_logit1_10 * h2_10 + d_logit1_11 * h2_11) * 0.25
        let dc1 = (d_logit1_00 + d_logit1_01 + d_logit1_10 + d_logit1_11) * 0.25

        // Backprop to hidden layer
        let d_h1_00 = d_logit0_00 * v01 + d_logit1_00 * v11
        let d_h2_00 = d_logit0_00 * v02 + d_logit1_00 * v12
        let d_h1_01 = d_logit0_01 * v01 + d_logit1_01 * v11
        let d_h2_01 = d_logit0_01 * v02 + d_logit1_01 * v12
        let d_h1_10 = d_logit0_10 * v01 + d_logit1_10 * v11
        let d_h2_10 = d_logit0_10 * v02 + d_logit1_10 * v12
        let d_h1_11 = d_logit0_11 * v01 + d_logit1_11 * v11
        let d_h2_11 = d_logit0_11 * v02 + d_logit1_11 * v12

        // Through sigmoid: d_z = d_h * h * (1-h)
        let d_z1_00 = d_h1_00 * h1_00 * (1.0 - h1_00)
        let d_z2_00 = d_h2_00 * h2_00 * (1.0 - h2_00)
        let d_z1_01 = d_h1_01 * h1_01 * (1.0 - h1_01)
        let d_z2_01 = d_h2_01 * h2_01 * (1.0 - h2_01)
        let d_z1_10 = d_h1_10 * h1_10 * (1.0 - h1_10)
        let d_z2_10 = d_h2_10 * h2_10 * (1.0 - h2_10)
        let d_z1_11 = d_h1_11 * h1_11 * (1.0 - h1_11)
        let d_z2_11 = d_h2_11 * h2_11 * (1.0 - h2_11)

        // Hidden layer gradients (x1, x2 for each sample)
        // (0,0): x1=0, x2=0
        // (0,1): x1=0, x2=1
        // (1,0): x1=1, x2=0
        // (1,1): x1=1, x2=1
        let dw11 = (d_z1_00 * 0.0 + d_z1_01 * 0.0 + d_z1_10 * 1.0 + d_z1_11 * 1.0) * 0.25
        let dw12 = (d_z1_00 * 0.0 + d_z1_01 * 1.0 + d_z1_10 * 0.0 + d_z1_11 * 1.0) * 0.25
        let db1 = (d_z1_00 + d_z1_01 + d_z1_10 + d_z1_11) * 0.25
        let dw21 = (d_z2_00 * 0.0 + d_z2_01 * 0.0 + d_z2_10 * 1.0 + d_z2_11 * 1.0) * 0.25
        let dw22 = (d_z2_00 * 0.0 + d_z2_01 * 1.0 + d_z2_10 * 0.0 + d_z2_11 * 1.0) * 0.25
        let db2 = (d_z2_00 + d_z2_01 + d_z2_10 + d_z2_11) * 0.25

        // Update running powers for bias correction (O(1) instead of O(t))
        beta1_t = beta1_t * beta1
        beta2_t = beta2_t * beta2

        // Bias correction denominators
        let bc1 = 1.0 - beta1_t
        let bc2 = 1.0 - beta2_t

        // Adam update for w11
        m_w11 = beta1 * m_w11 + (1.0 - beta1) * dw11
        s_w11 = beta2 * s_w11 + (1.0 - beta2) * dw11 * dw11
        w11 = w11 - lr * (m_w11 / bc1) / (sqrt_f64(s_w11 / bc2) + eps)

        // Adam update for w12
        m_w12 = beta1 * m_w12 + (1.0 - beta1) * dw12
        s_w12 = beta2 * s_w12 + (1.0 - beta2) * dw12 * dw12
        w12 = w12 - lr * (m_w12 / bc1) / (sqrt_f64(s_w12 / bc2) + eps)

        // Adam update for b1
        m_b1 = beta1 * m_b1 + (1.0 - beta1) * db1
        s_b1 = beta2 * s_b1 + (1.0 - beta2) * db1 * db1
        b1 = b1 - lr * (m_b1 / bc1) / (sqrt_f64(s_b1 / bc2) + eps)

        // Adam update for w21
        m_w21 = beta1 * m_w21 + (1.0 - beta1) * dw21
        s_w21 = beta2 * s_w21 + (1.0 - beta2) * dw21 * dw21
        w21 = w21 - lr * (m_w21 / bc1) / (sqrt_f64(s_w21 / bc2) + eps)

        // Adam update for w22
        m_w22 = beta1 * m_w22 + (1.0 - beta1) * dw22
        s_w22 = beta2 * s_w22 + (1.0 - beta2) * dw22 * dw22
        w22 = w22 - lr * (m_w22 / bc1) / (sqrt_f64(s_w22 / bc2) + eps)

        // Adam update for b2
        m_b2 = beta1 * m_b2 + (1.0 - beta1) * db2
        s_b2 = beta2 * s_b2 + (1.0 - beta2) * db2 * db2
        b2 = b2 - lr * (m_b2 / bc1) / (sqrt_f64(s_b2 / bc2) + eps)

        // Adam update for v01
        m_v01 = beta1 * m_v01 + (1.0 - beta1) * dv01
        s_v01 = beta2 * s_v01 + (1.0 - beta2) * dv01 * dv01
        v01 = v01 - lr * (m_v01 / bc1) / (sqrt_f64(s_v01 / bc2) + eps)

        // Adam update for v02
        m_v02 = beta1 * m_v02 + (1.0 - beta1) * dv02
        s_v02 = beta2 * s_v02 + (1.0 - beta2) * dv02 * dv02
        v02 = v02 - lr * (m_v02 / bc1) / (sqrt_f64(s_v02 / bc2) + eps)

        // Adam update for c0
        m_c0 = beta1 * m_c0 + (1.0 - beta1) * dc0
        s_c0 = beta2 * s_c0 + (1.0 - beta2) * dc0 * dc0
        c0 = c0 - lr * (m_c0 / bc1) / (sqrt_f64(s_c0 / bc2) + eps)

        // Adam update for v11
        m_v11 = beta1 * m_v11 + (1.0 - beta1) * dv11
        s_v11 = beta2 * s_v11 + (1.0 - beta2) * dv11 * dv11
        v11 = v11 - lr * (m_v11 / bc1) / (sqrt_f64(s_v11 / bc2) + eps)

        // Adam update for v12
        m_v12 = beta1 * m_v12 + (1.0 - beta1) * dv12
        s_v12 = beta2 * s_v12 + (1.0 - beta2) * dv12 * dv12
        v12 = v12 - lr * (m_v12 / bc1) / (sqrt_f64(s_v12 / bc2) + eps)

        // Adam update for c1
        m_c1 = beta1 * m_c1 + (1.0 - beta1) * dc1
        s_c1 = beta2 * s_c1 + (1.0 - beta2) * dc1 * dc1
        c1 = c1 - lr * (m_c1 / bc1) / (sqrt_f64(s_c1 / bc2) + eps)

        // Print loss at key epochs
        if epoch == 0 {
            let loss = (cross_entropy_2class(p0_00, 0) + cross_entropy_2class(p0_11, 0) +
                        cross_entropy_2class(p0_01, 1) + cross_entropy_2class(p0_10, 1)) * 0.25
            println("  Epoch 0, Loss = ")
            println(loss)
        }
        if epoch == 50 {
            let loss = (cross_entropy_2class(p0_00, 0) + cross_entropy_2class(p0_11, 0) +
                        cross_entropy_2class(p0_01, 1) + cross_entropy_2class(p0_10, 1)) * 0.25
            println("  Epoch 50, Loss = ")
            println(loss)
        }
        if epoch == 150 {
            let loss = (cross_entropy_2class(p0_00, 0) + cross_entropy_2class(p0_11, 0) +
                        cross_entropy_2class(p0_01, 1) + cross_entropy_2class(p0_10, 1)) * 0.25
            println("  Epoch 150, Loss = ")
            println(loss)
        }

        epoch = epoch + 1
    }

    // Final predictions
    let p00 = forward(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 0.0, 0.0)
    let p01 = forward(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 0.0, 1.0)
    let p10 = forward(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 1.0, 0.0)
    let p11 = forward(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 1.0, 1.0)

    let final_loss = (cross_entropy_2class(p00, 0) + cross_entropy_2class(p11, 0) +
                      cross_entropy_2class(p01, 1) + cross_entropy_2class(p10, 1)) * 0.25
    println("  Epoch 300, Loss = ")
    println(final_loss)
    println("")

    // Show predictions
    println("Predictions (P(class=0)):")
    println("  (0,0) -> P(0) = ")
    println(p00)
    println("    predicted = ")
    println(predict(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 0.0, 0.0))
    println("    expected = 0")

    println("  (1,1) -> P(0) = ")
    println(p11)
    println("    predicted = ")
    println(predict(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 1.0, 1.0))
    println("    expected = 0")

    println("  (0,1) -> P(0) = ")
    println(p01)
    println("    predicted = ")
    println(predict(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 0.0, 1.0))
    println("    expected = 1")

    println("  (1,0) -> P(0) = ")
    println(p10)
    println("    predicted = ")
    println(predict(w11, w12, b1, w21, w22, b2, v01, v02, c0, v11, v12, c1, 1.0, 0.0))
    println("    expected = 1")
    println("")

    // Verify predictions
    if p00 < 0.5 { ok = false; println("FAIL: (0,0) should have P(0) > 0.5") }
    if p11 < 0.5 { ok = false; println("FAIL: (1,1) should have P(0) > 0.5") }
    if p01 > 0.5 { ok = false; println("FAIL: (0,1) should have P(0) < 0.5") }
    if p10 > 0.5 { ok = false; println("FAIL: (1,0) should have P(0) < 0.5") }
    if final_loss > 0.5 { ok = false; println("FAIL: Final loss should be < 0.5") }

    if ok {
        println("ALL TESTS PASSED")
        return 0
    } else {
        println("SOME TESTS FAILED")
        return 1
    }
}

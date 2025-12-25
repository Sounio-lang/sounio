// activation.d - Neural network activation functions and derivatives
//
// Provides common activation functions used in neural networks:
// - ReLU: Rectified Linear Unit - max(0, x)
// - Sigmoid: σ(x) = 1 / (1 + e^(-x))
// - Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
//
// Each function has a corresponding derivative for backpropagation.
//
// All functions are scalar (f64 -> f64) for simplicity.

// ============================================================================
// MATH HELPERS
// ============================================================================

fn abs_val(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0 - x
    }
    return x
}

// Exponential function using Taylor series
fn exp_val(x: f64) -> f64 {
    // For large |x|, use recursion to avoid overflow
    if x > 20.0 {
        let half = exp_val(x / 2.0)
        return half * half
    }
    if x < 0.0 - 20.0 {
        return 1.0 / exp_val(0.0 - x)
    }

    // Taylor series: e^x = 1 + x + x^2/2! + x^3/3! + ...
    let mut sum = 1.0
    let mut term = 1.0
    let mut count = 1
    while count <= 20 {
        term = term * x / count
        sum = sum + term
        count = count + 1
    }
    return sum
}

// Max of two values
fn max_val(a: f64, b: f64) -> f64 {
    if a > b { return a }
    return b
}

// Min of two values
fn min_val(a: f64, b: f64) -> f64 {
    if a < b { return a }
    return b
}

// ============================================================================
// RELU - RECTIFIED LINEAR UNIT
// ============================================================================

// ReLU: f(x) = max(0, x)
// Advantages: Simple, fast, no vanishing gradient for x > 0
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        return x
    }
    return 0.0
}

// ReLU derivative: f'(x) = 1 if x > 0, else 0
// Note: At x=0, technically undefined, but we use 0 by convention
fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0
    }
    return 0.0
}

// Leaky ReLU: f(x) = x if x > 0, else alpha * x
// Helps prevent "dying ReLU" problem
fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        return x
    }
    return alpha * x
}

// Leaky ReLU derivative
fn leaky_relu_deriv(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        return 1.0
    }
    return alpha
}

// ============================================================================
// SIGMOID - LOGISTIC FUNCTION
// ============================================================================

// Sigmoid: σ(x) = 1 / (1 + e^(-x))
// Range: (0, 1)
// Used for: Binary classification, gates in LSTMs
fn sigmoid(x: f64) -> f64 {
    // Numerically stable version:
    // If x >= 0: 1 / (1 + exp(-x))
    // If x < 0: exp(x) / (1 + exp(x))
    if x >= 0.0 {
        return 1.0 / (1.0 + exp_val(0.0 - x))
    } else {
        let ex = exp_val(x)
        return ex / (1.0 + ex)
    }
}

// Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
// This is why sigmoid is convenient: derivative in terms of function value
fn sigmoid_deriv(x: f64) -> f64 {
    let s = sigmoid(x)
    return s * (1.0 - s)
}

// Sigmoid derivative when you already have σ(x)
fn sigmoid_deriv_from_output(sigmoid_val: f64) -> f64 {
    return sigmoid_val * (1.0 - sigmoid_val)
}

// ============================================================================
// TANH - HYPERBOLIC TANGENT
// ============================================================================

// Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
// Range: (-1, 1)
// Centered at 0 (unlike sigmoid), often works better than sigmoid
fn tanh_val(x: f64) -> f64 {
    // Numerically stable version:
    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1) for x >= 0
    // tanh(x) = -tanh(-x) for x < 0
    if x >= 0.0 {
        if x > 20.0 { return 1.0 }  // Saturates for large x
        let e2x = exp_val(2.0 * x)
        return (e2x - 1.0) / (e2x + 1.0)
    } else {
        return 0.0 - tanh_val(0.0 - x)
    }
}

// Tanh derivative: tanh'(x) = 1 - tanh(x)^2
fn tanh_deriv(x: f64) -> f64 {
    let t = tanh_val(x)
    return 1.0 - t * t
}

// Tanh derivative when you already have tanh(x)
fn tanh_deriv_from_output(tanh_output: f64) -> f64 {
    return 1.0 - tanh_output * tanh_output
}

// ============================================================================
// SOFTPLUS - SMOOTH RELU
// ============================================================================

// Softplus: f(x) = ln(1 + e^x)
// Smooth approximation of ReLU
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        return x  // For large x, ln(1 + e^x) ≈ x
    }
    return ln_val(1.0 + exp_val(x))
}

// Softplus derivative: f'(x) = 1 / (1 + e^(-x)) = sigmoid(x)
fn softplus_deriv(x: f64) -> f64 {
    return sigmoid(x)
}

// ============================================================================
// NATURAL LOGARITHM (needed for softplus)
// ============================================================================

// Natural log using Taylor series around x=1
fn ln_val(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 999999.0 }  // Undefined for x <= 0
    if x == 1.0 { return 0.0 }

    // For x far from 1, use ln(x) = ln(x/e^k) + k where k brings x/e^k close to 1
    if x > 10.0 {
        return 2.302585093 + ln_val(x / 10.0)  // ln(10) ≈ 2.302585093
    }
    if x < 0.1 {
        return 0.0 - 2.302585093 + ln_val(x * 10.0)
    }

    // Taylor series: ln(x) = ln(1 + y) = y - y^2/2 + y^3/3 - y^4/4 + ...
    // where y = x - 1
    let y_val = x - 1.0
    let mut sum = 0.0
    let mut term = y_val
    let mut count = 1
    while count <= 20 {
        if count % 2 == 1 {
            sum = sum + term / count
        } else {
            sum = sum - term / count
        }
        term = term * y_val
        count = count + 1
    }
    return sum
}

// ============================================================================
// ELU - EXPONENTIAL LINEAR UNIT
// ============================================================================

// ELU: f(x) = x if x > 0, else alpha * (e^x - 1)
// Smoother than ReLU, can produce negative outputs
fn elu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        return x
    }
    return alpha * (exp_val(x) - 1.0)
}

// ELU derivative
fn elu_deriv(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        return 1.0
    }
    return alpha * exp_val(x)
}

// ============================================================================
// SWISH / SILU - SELF-GATED ACTIVATION
// ============================================================================

// Swish: f(x) = x * sigmoid(x)
// Also called SiLU (Sigmoid Linear Unit)
// Used in many modern architectures
fn swish(x: f64) -> f64 {
    return x * sigmoid(x)
}

// Swish derivative: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
fn swish_deriv(x: f64) -> f64 {
    let s = sigmoid(x)
    let f = x * s
    return f + s * (1.0 - f)
}

// ============================================================================
// IDENTITY - LINEAR ACTIVATION
// ============================================================================

// Identity: f(x) = x
// Used in regression output layers
fn identity(x: f64) -> f64 {
    return x
}

// Identity derivative: f'(x) = 1
fn identity_deriv(x: f64) -> f64 {
    return 1.0
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    println("=== Demetrios Activation Functions Test ===")
    println("")

    // Test ReLU
    println("Testing ReLU:")
    let relu_pos = relu(5.0)
    let relu_neg = relu(0.0 - 3.0)
    println("  relu(5.0) = ")
    println(relu_pos)
    println("  relu(-3.0) = ")
    println(relu_neg)
    let relu_deriv_pos = relu_deriv(5.0)
    let relu_deriv_neg = relu_deriv(0.0 - 3.0)
    println("  relu'(5.0) = ")
    println(relu_deriv_pos)
    println("  relu'(-3.0) = ")
    println(relu_deriv_neg)
    println("")

    // Test Sigmoid
    println("Testing Sigmoid:")
    let sig0 = sigmoid(0.0)
    let sig_pos = sigmoid(2.0)
    let sig_neg = sigmoid(0.0 - 2.0)
    println("  sigmoid(0.0) = ")
    println(sig0)
    println("  sigmoid(2.0) = ")
    println(sig_pos)
    println("  sigmoid(-2.0) = ")
    println(sig_neg)
    let sig_deriv0 = sigmoid_deriv(0.0)
    println("  sigmoid'(0.0) = ")
    println(sig_deriv0)
    println("")

    // Test Tanh
    println("Testing Tanh:")
    let tanh0 = tanh_val(0.0)
    let tanh_pos = tanh_val(1.0)
    let tanh_neg = tanh_val(0.0 - 1.0)
    println("  tanh(0.0) = ")
    println(tanh0)
    println("  tanh(1.0) = ")
    println(tanh_pos)
    println("  tanh(-1.0) = ")
    println(tanh_neg)
    let tanh_deriv0 = tanh_deriv(0.0)
    println("  tanh'(0.0) = ")
    println(tanh_deriv0)
    println("")

    // Test Leaky ReLU
    println("Testing Leaky ReLU (alpha=0.01):")
    let lrelu_pos = leaky_relu(5.0, 0.01)
    let lrelu_neg = leaky_relu(0.0 - 3.0, 0.01)
    println("  leaky_relu(5.0) = ")
    println(lrelu_pos)
    println("  leaky_relu(-3.0) = ")
    println(lrelu_neg)
    println("")

    // Test Swish
    println("Testing Swish:")
    let swish0 = swish(0.0)
    let swish1 = swish(1.0)
    println("  swish(0.0) = ")
    println(swish0)
    println("  swish(1.0) = ")
    println(swish1)
    println("")

    // Verify expected values
    let relu_pos_expected = 5.0
    let relu_neg_expected = 0.0
    let relu_err = abs_val(relu_pos - relu_pos_expected) + abs_val(relu_neg - relu_neg_expected)

    let sig0_expected = 0.5
    let sig0_err = abs_val(sig0 - sig0_expected)

    let sig_deriv0_expected = 0.25  // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
    let sig_deriv0_err = abs_val(sig_deriv0 - sig_deriv0_expected)

    let tanh0_expected = 0.0
    let tanh0_err = abs_val(tanh0 - tanh0_expected)

    let tanh_deriv0_expected = 1.0  // tanh'(0) = 1 - 0^2 = 1
    let tanh_deriv0_err = abs_val(tanh_deriv0 - tanh_deriv0_expected)

    let lrelu_neg_expected = 0.0 - 0.03  // -3.0 * 0.01
    let lrelu_err = abs_val(lrelu_neg - lrelu_neg_expected)

    let swish0_expected = 0.0  // 0 * sigmoid(0) = 0
    let swish0_err = abs_val(swish0 - swish0_expected)

    let all_err = relu_err + sig0_err + sig_deriv0_err + tanh0_err + tanh_deriv0_err + lrelu_err + swish0_err

    if all_err < 0.001 {
        println("TEST PASSED: All activation functions correct")
        return 0
    } else {
        println("TEST FAILED: Activation function errors")
        println("  relu_err = ")
        println(relu_err)
        println("  sig0_err = ")
        println(sig0_err)
        println("  sig_deriv0_err = ")
        println(sig_deriv0_err)
        println("  tanh0_err = ")
        println(tanh0_err)
        println("  tanh_deriv0_err = ")
        println(tanh_deriv0_err)
        println("  lrelu_err = ")
        println(lrelu_err)
        println("  swish0_err = ")
        println(swish0_err)
        return 1
    }
}

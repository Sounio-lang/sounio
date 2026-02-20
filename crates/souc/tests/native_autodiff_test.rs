//! Integration tests for native autodiff runtime functions
//!
//! Tests forward-mode and reverse-mode automatic differentiation
//! using the C-compatible runtime functions.

use sounio::backend::native::autodiff_runtime::*;

/// Test dual number addition
#[test]
fn test_dual_add() {
    let a = Dual::variable(3.0); // x = 3, dx/dx = 1
    let b = Dual::constant(5.0); // 5, d5/dx = 0
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_add(&a, &b, &mut result);
    }

    // (3 + 5, 1 + 0) = (8, 1)
    assert!((result.value - 8.0).abs() < 1e-10);
    assert!((result.derivative - 1.0).abs() < 1e-10);
}

/// Test dual number multiplication (product rule)
#[test]
fn test_dual_mul() {
    let a = Dual::variable(2.0); // x = 2, dx/dx = 1
    let b = Dual::variable(3.0); // x = 3, dx/dx = 1
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_mul(&a, &b, &mut result);
    }

    // Product rule: (a*b)' = a'*b + a*b'
    // (2*3, 1*3 + 2*1) = (6, 5)
    assert!((result.value - 6.0).abs() < 1e-10);
    assert!((result.derivative - 5.0).abs() < 1e-10);
}

/// Test dual number division (quotient rule)
#[test]
fn test_dual_div() {
    let a = Dual::variable(6.0); // x = 6, dx/dx = 1
    let b = Dual::constant(2.0); // 2, d2/dx = 0
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_div(&a, &b, &mut result);
    }

    // Quotient rule: (a/b)' = (a'*b - a*b') / b²
    // (6/2, (1*2 - 6*0) / 4) = (3, 0.5)
    assert!((result.value - 3.0).abs() < 1e-10);
    assert!((result.derivative - 0.5).abs() < 1e-10);
}

/// Test dual number exponential
#[test]
fn test_dual_exp() {
    let x = Dual::variable(2.0); // x = 2, dx/dx = 1
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_exp(&x, &mut result);
    }

    // d/dx(exp(x)) = exp(x) * x'
    // (exp(2), exp(2) * 1) = (e², e²)
    let expected_value = 2.0_f64.exp();
    assert!((result.value - expected_value).abs() < 1e-10);
    assert!((result.derivative - expected_value).abs() < 1e-10);
}

/// Test dual number logarithm
#[test]
fn test_dual_log() {
    let x = Dual::variable(2.0); // x = 2, dx/dx = 1
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_log(&x, &mut result);
    }

    // d/dx(ln(x)) = x' / x
    // (ln(2), 1/2) = (ln(2), 0.5)
    let expected_value = 2.0_f64.ln();
    assert!((result.value - expected_value).abs() < 1e-10);
    assert!((result.derivative - 0.5).abs() < 1e-10);
}

/// Test dual number sine
#[test]
fn test_dual_sin() {
    let x = Dual::variable(0.0); // x = 0, dx/dx = 1
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_sin(&x, &mut result);
    }

    // d/dx(sin(x)) = cos(x) * x'
    // (sin(0), cos(0) * 1) = (0, 1)
    assert!((result.value - 0.0).abs() < 1e-10);
    assert!((result.derivative - 1.0).abs() < 1e-10);
}

/// Test dual number cosine
#[test]
fn test_dual_cos() {
    let x = Dual::variable(0.0); // x = 0, dx/dx = 1
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_cos(&x, &mut result);
    }

    // d/dx(cos(x)) = -sin(x) * x'
    // (cos(0), -sin(0) * 1) = (1, 0)
    assert!((result.value - 1.0).abs() < 1e-10);
    assert!((result.derivative - 0.0).abs() < 1e-10);
}

/// Test dual number power
#[test]
fn test_dual_pow() {
    let x = Dual::variable(2.0); // x = 2, dx/dx = 1
    let power = 3.0;
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_pow(&x, power, &mut result);
    }

    // d/dx(x^p) = p * x^(p-1) * x'
    // (2³, 3 * 2² * 1) = (8, 12)
    assert!((result.value - 8.0).abs() < 1e-10);
    assert!((result.derivative - 12.0).abs() < 1e-10);
}

/// Test dual number square root
#[test]
fn test_dual_sqrt() {
    let x = Dual::variable(4.0); // x = 4, dx/dx = 1
    let mut result = Dual::constant(0.0);

    unsafe {
        sounio_dual_sqrt(&x, &mut result);
    }

    // d/dx(sqrt(x)) = x' / (2 * sqrt(x))
    // (sqrt(4), 1 / (2 * 2)) = (2, 0.25)
    assert!((result.value - 2.0).abs() < 1e-10);
    assert!((result.derivative - 0.25).abs() < 1e-10);
}

/// Test forward-mode autodiff with a simple function
/// f(x) = x², so f'(x) = 2x
#[test]
fn test_forward_mode_simple() {
    extern "C" fn square(x: *const f64, y: *mut f64) {
        unsafe {
            let x_val = *x;
            *y = x_val * x_val;
        }
    }

    let x = 3.0;
    let mut gradient = 0.0;

    unsafe {
        let result = sounio_autodiff_forward(square, x, &mut gradient);

        // f(3) = 9
        assert!((result - 9.0).abs() < 1e-6);

        // f'(3) = 6
        assert!((gradient - 6.0).abs() < 1e-6);
    }
}

/// Test forward-mode autodiff with exponential function
/// f(x) = exp(x), so f'(x) = exp(x)
#[test]
fn test_forward_mode_exp() {
    extern "C" fn exp_func(x: *const f64, y: *mut f64) {
        unsafe {
            *y = (*x).exp();
        }
    }

    let x = 1.0;
    let mut gradient = 0.0;

    unsafe {
        let result = sounio_autodiff_forward(exp_func, x, &mut gradient);

        // f(1) = e
        let expected = std::f64::consts::E;
        assert!((result - expected).abs() < 1e-6);

        // f'(1) = e
        assert!((gradient - expected).abs() < 1e-6);
    }
}

/// Test composite function: f(x) = x² + 2x + 1
/// f'(x) = 2x + 2
#[test]
fn test_forward_mode_composite() {
    extern "C" fn quadratic(x: *const f64, y: *mut f64) {
        unsafe {
            let x_val = *x;
            *y = x_val * x_val + 2.0 * x_val + 1.0;
        }
    }

    let x = 2.0;
    let mut gradient = 0.0;

    unsafe {
        let result = sounio_autodiff_forward(quadratic, x, &mut gradient);

        // f(2) = 4 + 4 + 1 = 9
        assert!((result - 9.0).abs() < 1e-6);

        // f'(2) = 4 + 2 = 6
        assert!((gradient - 6.0).abs() < 1e-6);
    }
}

// =============================================================================
// Reverse-mode (tape-based) integration tests
// =============================================================================

/// Test reverse-mode: f(x,y) = x*y + x
/// df/dx = y + 1, df/dy = x
#[test]
fn test_reverse_mode_basic() {
    unsafe {
        let tape = sounio_tape_create(2);
        let x = sounio_tape_input(tape, 0, 3.0);
        let y = sounio_tape_input(tape, 1, 5.0);
        let xy = sounio_tape_mul(tape, x, y);
        let z = sounio_tape_add(tape, xy, x);

        assert!((sounio_tape_value(tape, z) - 18.0).abs() < 1e-10);

        let mut grads = [0.0f64; 2];
        sounio_autodiff_reverse(tape, 1.0, grads.as_mut_ptr(), 2);

        assert!((grads[0] - 6.0).abs() < 1e-10); // df/dx = y + 1
        assert!((grads[1] - 3.0).abs() < 1e-10); // df/dy = x

        sounio_tape_destroy(tape);
    }
}

/// Test reverse-mode: f(x) = exp(x²)
/// f'(x) = 2x * exp(x²)
#[test]
fn test_reverse_mode_exp_of_square() {
    unsafe {
        let tape = sounio_tape_create(1);
        let x = sounio_tape_input(tape, 0, 1.5);
        let x_sq = sounio_tape_mul(tape, x, x);
        let z = sounio_tape_exp(tape, x_sq);

        let expected_val = (1.5_f64 * 1.5).exp();
        assert!((sounio_tape_value(tape, z) - expected_val).abs() < 1e-8);

        let mut grad = 0.0f64;
        sounio_autodiff_reverse(tape, 1.0, &mut grad, 1);

        let expected_grad = 2.0 * 1.5 * expected_val;
        assert!((grad - expected_grad).abs() < 1e-8);

        sounio_tape_destroy(tape);
    }
}

/// Test reverse-mode: Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²
/// At (1,1): gradient is (0, 0) (minimum)
#[test]
fn test_reverse_mode_rosenbrock_minimum() {
    unsafe {
        let tape = sounio_tape_create(2);
        let x = sounio_tape_input(tape, 0, 1.0);
        let y = sounio_tape_input(tape, 1, 1.0);
        let one = sounio_tape_const(tape, 1.0);
        let hundred = sounio_tape_const(tape, 100.0);

        let one_minus_x = sounio_tape_sub(tape, one, x);
        let term1 = sounio_tape_mul(tape, one_minus_x, one_minus_x);

        let x_sq = sounio_tape_mul(tape, x, x);
        let y_minus_xsq = sounio_tape_sub(tape, y, x_sq);
        let t2_sq = sounio_tape_mul(tape, y_minus_xsq, y_minus_xsq);
        let term2 = sounio_tape_mul(tape, hundred, t2_sq);

        let z = sounio_tape_add(tape, term1, term2);

        // At the minimum (1,1), f = 0
        assert!((sounio_tape_value(tape, z) - 0.0).abs() < 1e-10);

        let mut grads = [0.0f64; 2];
        sounio_autodiff_reverse(tape, 1.0, grads.as_mut_ptr(), 2);

        // Gradient at the minimum should be zero
        assert!((grads[0]).abs() < 1e-10);
        assert!((grads[1]).abs() < 1e-10);

        sounio_tape_destroy(tape);
    }
}

/// Test reverse-mode: verify forward/reverse agreement
/// f(x) = sin(x) at x=1.0 → both should give cos(1)
#[test]
fn test_forward_reverse_agreement() {
    let x_val = 1.0;

    // Forward mode
    let x_dual = Dual::variable(x_val);
    let mut forward_result = Dual::constant(0.0);
    unsafe {
        sounio_dual_sin(&x_dual, &mut forward_result);
    }

    // Reverse mode
    unsafe {
        let tape = sounio_tape_create(1);
        let x = sounio_tape_input(tape, 0, x_val);
        let _z = sounio_tape_sin(tape, x);

        let mut reverse_grad = 0.0f64;
        sounio_autodiff_reverse(tape, 1.0, &mut reverse_grad, 1);

        // Both should give cos(1)
        assert!(
            (forward_result.derivative - reverse_grad).abs() < 1e-10,
            "Forward ({}) and reverse ({}) disagree",
            forward_result.derivative,
            reverse_grad
        );

        sounio_tape_destroy(tape);
    }
}

/// Test NativeTape Rust API directly for gradient accumulation
/// f(x) = x * x * x (built as x*x then result*x)
/// f'(x) = 3x²
#[test]
fn test_reverse_mode_gradient_accumulation() {
    let mut tape = NativeTape::new(1);
    let x = tape.record_input(0, 2.0);
    let x_sq = tape.record_binary(NativeTapeOp::Mul, x, x);
    let z = tape.record_binary(NativeTapeOp::Mul, x_sq, x);

    assert!((tape.value(z) - 8.0).abs() < 1e-10);

    let grads = tape.backward(z, 1.0);
    // f'(2) = 3 * 4 = 12
    assert!((grads[0] - 12.0).abs() < 1e-10);
}

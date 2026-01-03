//! Integration tests for native autodiff runtime functions
//!
//! Tests forward-mode and reverse-mode automatic differentiation
//! using the C-compatible runtime functions.

use sounio::backend::native::autodiff_runtime::*;

/// Test dual number addition
#[test]
fn test_dual_add() {
    let a = Dual::variable(3.0); // x = 3, dx/dx = 1
    let b = Dual::constant(5.0);  // 5, d5/dx = 0
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
    let b = Dual::constant(2.0);  // 2, d2/dx = 0
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

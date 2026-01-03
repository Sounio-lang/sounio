//! Integration tests for native uncertainty propagation runtime functions
//!
//! Tests uncertainty propagation using GUM (Guide to the Expression of Uncertainty
//! in Measurement) rules for uncertain<T> types.

use sounio::backend::native::uncertain_runtime::*;

/// Test uncertainty propagation: addition
/// For independent variables: u²(a+b) = u²(a) + u²(b)
#[test]
fn test_uncertain_add() {
    let a = Uncertain::new(10.0, 0.3);
    let b = Uncertain::new(5.0, 0.4);
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_add(&a, &b, &mut result);
    }

    // Value: 10.0 + 5.0 = 15.0
    assert!((result.value - 15.0).abs() < 1e-10);
    
    // Uncertainty: sqrt(0.3² + 0.4²) = sqrt(0.09 + 0.16) = sqrt(0.25) = 0.5
    assert!((result.uncertainty - 0.5).abs() < 1e-10);
}

/// Test uncertainty propagation: subtraction
#[test]
fn test_uncertain_sub() {
    let a = Uncertain::new(10.0, 0.3);
    let b = Uncertain::new(5.0, 0.4);
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_sub(&a, &b, &mut result);
    }

    // Value: 10.0 - 5.0 = 5.0
    assert!((result.value - 5.0).abs() < 1e-10);
    
    // Uncertainty: sqrt(0.3² + 0.4²) = 0.5
    assert!((result.uncertainty - 0.5).abs() < 1e-10);
}

/// Test uncertainty propagation: multiplication
/// For independent variables: u²(a*b) = (a*b)² * (u²(a)/a² + u²(b)/b²)
#[test]
fn test_uncertain_mul() {
    let a = Uncertain::new(10.0, 0.1); // 1% relative error
    let b = Uncertain::new(5.0, 0.05); // 1% relative error
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_mul(&a, &b, &mut result);
    }

    // Value: 10.0 * 5.0 = 50.0
    assert!((result.value - 50.0).abs() < 1e-10);
    
    // Relative error: sqrt((0.1/10)² + (0.05/5)²) = sqrt(0.01 + 0.01) = sqrt(0.02) ≈ 0.1414
    // Absolute error: 50.0 * 0.1414 ≈ 7.07
    // But our implementation uses: |a*b| * sqrt((u(a)/a)² + (u(b)/b)²)
    // = 50.0 * sqrt((0.1/10)² + (0.05/5)²) = 50.0 * sqrt(0.01 + 0.01) = 50.0 * sqrt(0.02)
    let rel_unc_a: f64 = 0.1 / 10.0;
    let rel_unc_b: f64 = 0.05 / 5.0;
    let expected_uncertainty: f64 = 50.0 * (rel_unc_a * rel_unc_a + rel_unc_b * rel_unc_b).sqrt();
    assert!((result.uncertainty - expected_uncertainty).abs() < 0.01, 
            "uncertainty: got {}, expected {}", result.uncertainty, expected_uncertainty);
}

/// Test uncertainty propagation: division
#[test]
fn test_uncertain_div() {
    let a = Uncertain::new(10.0, 0.1);
    let b = Uncertain::new(5.0, 0.05);
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_div(&a, &b, &mut result);
    }

    // Value: 10.0 / 5.0 = 2.0
    assert!((result.value - 2.0).abs() < 1e-10);
    
    // Same relative error propagation as multiplication
    let rel_unc_a: f64 = 0.1 / 10.0;
    let rel_unc_b: f64 = 0.05 / 5.0;
    let expected_uncertainty: f64 = 2.0 * (rel_unc_a * rel_unc_a + rel_unc_b * rel_unc_b).sqrt();
    assert!((result.uncertainty - expected_uncertainty).abs() < 0.01,
            "uncertainty: got {}, expected {}", result.uncertainty, expected_uncertainty);
}

/// Test uncertainty propagation: power
/// u(x^p) = |p * x^(p-1)| * u(x)
#[test]
fn test_uncertain_pow() {
    let x = Uncertain::new(2.0, 0.1);
    let power = 3.0;
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_pow(&x, power, &mut result);
    }

    // Value: 2.0³ = 8.0
    assert!((result.value - 8.0).abs() < 1e-10);
    
    // Uncertainty: |3 * 2²| * 0.1 = 12 * 0.1 = 1.2
    assert!((result.uncertainty - 1.2).abs() < 1e-10);
}

/// Test uncertainty propagation: square root
/// u(sqrt(x)) = u(x) / (2 * sqrt(x))
#[test]
fn test_uncertain_sqrt() {
    let x = Uncertain::new(4.0, 0.04);
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_sqrt(&x, &mut result);
    }

    // Value: sqrt(4.0) = 2.0
    assert!((result.value - 2.0).abs() < 1e-10);
    
    // Uncertainty: 0.04 / (2 * 2.0) = 0.04 / 4.0 = 0.01
    assert!((result.uncertainty - 0.01).abs() < 1e-10);
}

/// Test uncertainty propagation: exponential
/// u(exp(x)) = exp(x) * u(x)
#[test]
fn test_uncertain_exp() {
    let x = Uncertain::new(1.0, 0.01);
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_exp(&x, &mut result);
    }

    // Value: exp(1.0) = e
    let expected_value = std::f64::consts::E;
    assert!((result.value - expected_value).abs() < 1e-10);
    
    // Uncertainty: e * 0.01 ≈ 0.02718
    let expected_uncertainty = expected_value * 0.01;
    assert!((result.uncertainty - expected_uncertainty).abs() < 1e-10);
}

/// Test uncertainty propagation: natural logarithm
/// u(ln(x)) = u(x) / x
#[test]
fn test_uncertain_log() {
    let x = Uncertain::new(2.0, 0.02);
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_log(&x, &mut result);
    }

    // Value: ln(2.0)
    let expected_value = 2.0_f64.ln();
    assert!((result.value - expected_value).abs() < 1e-10);
    
    // Uncertainty: 0.02 / 2.0 = 0.01
    assert!((result.uncertainty - 0.01).abs() < 1e-10);
}

/// Test uncertainty propagation: scaling
/// u(alpha * x) = |alpha| * u(x)
#[test]
fn test_uncertain_scale() {
    let x = Uncertain::new(5.0, 0.1);
    let alpha = 2.5;
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_scale(&x, alpha, &mut result);
    }

    // Value: 2.5 * 5.0 = 12.5
    assert!((result.value - 12.5).abs() < 1e-10);
    
    // Uncertainty: 2.5 * 0.1 = 0.25
    assert!((result.uncertainty - 0.25).abs() < 1e-10);
}

/// Test combining multiple uncertain measurements
/// Uses inverse-variance weighting
#[test]
fn test_uncertain_combine() {
    // Three measurements of the same quantity with different uncertainties
    let measurements = vec![
        Uncertain::new(10.0, 0.1),  // High precision
        Uncertain::new(10.1, 0.2),  // Medium precision
        Uncertain::new(9.9, 0.3),   // Low precision
    ];
    let mut result = Uncertain::constant(0.0);

    unsafe {
        sounio_uncertain_combine(measurements.as_ptr(), 3, &mut result);
    }

    // Combined value should be weighted average (closer to 10.0 due to higher weight)
    assert!(result.value > 9.95 && result.value < 10.05);
    
    // Combined uncertainty should be less than any individual uncertainty
    assert!(result.uncertainty < 0.1);
    assert!(result.uncertainty > 0.0);
}

/// Test composite uncertainty propagation: (a + b) * c
#[test]
fn test_uncertain_composite() {
    let a = Uncertain::new(3.0, 0.1);
    let b = Uncertain::new(2.0, 0.1);
    let c = Uncertain::new(5.0, 0.2);
    
    // First: a + b
    let mut sum = Uncertain::constant(0.0);
    unsafe {
        sounio_uncertain_add(&a, &b, &mut sum);
    }
    
    // Then: sum * c
    let mut result = Uncertain::constant(0.0);
    unsafe {
        sounio_uncertain_mul(&sum, &c, &mut result);
    }
    
    // Value: (3.0 + 2.0) * 5.0 = 25.0
    assert!((result.value - 25.0).abs() < 1e-10);
    
    // Uncertainty should be propagated correctly through both operations
    assert!(result.uncertainty > 0.0);
}

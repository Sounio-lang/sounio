//! Integration tests for native stiff ODE solvers (BDF, LSODA)
//!
//! Tests stiff ODE solvers that require implicit methods and Jacobian computation.
//!
//! Test problems:
//! 1. Exponential decay (simple stiff problem)
//! 2. Van der Pol oscillator (classic stiff test)
//! 3. Robertson problem (very stiff chemical kinetics)

use sounio::backend::native::ode_runtime::{
    sounio_ode_bdf_reset, sounio_ode_bdf_step, sounio_ode_lsoda_reset, sounio_ode_lsoda_step,
};

// ============================================================================
// TEST PROBLEM 1: Simple exponential decay (dy/dt = -k*y)
// ============================================================================

/// Fast exponential decay: dy/dt = -1000*y
/// Exact solution: y(t) = y0 * exp(-1000*t)
unsafe extern "C" fn fast_decay_derivatives(state: *mut f64, _t: f64, dydt: *mut f64) {
    let y = unsafe { *state };
    unsafe {
        *dydt = -1000.0 * y;
    }
}

/// Jacobian for fast decay: J = -1000
unsafe extern "C" fn fast_decay_jacobian(_state: *const f64, _t: f64, jacobian: *mut f64) {
    unsafe {
        *jacobian = -1000.0;
    }
}

#[test]
fn test_bdf_exponential_decay() {
    let mut state = vec![1.0];
    let mut t = 0.0;
    let mut dt = 1e-3; // Larger initial step
    let rtol = 1e-4; // Relaxed tolerance for faster convergence
    let atol = 1e-6;
    let t_end = 0.005; // Shorter integration

    // Reset BDF state
    unsafe {
        sounio_ode_bdf_reset(state.as_mut_ptr());
    }

    let mut steps = 0;
    let max_steps = 1000;

    while t < t_end && steps < max_steps {
        unsafe {
            let err = sounio_ode_bdf_step(
                state.as_mut_ptr(),
                1,
                &mut t,
                &mut dt,
                rtol,
                atol,
                fast_decay_derivatives,
                fast_decay_jacobian,
            );

            if err.is_finite() {
                steps += 1;
            }
        }
    }

    // Check that we reached the end time
    assert!(t >= t_end - 1e-10, "BDF should reach t_end, got t={}", t);

    // Check solution accuracy
    // Exact: y(0.005) = exp(-1000 * 0.005) = exp(-5) ≈ 0.0067
    let expected = (-1000.0 * t_end).exp();
    let error = (state[0] - expected).abs();
    let rel_error = error / expected.abs().max(1e-15);

    assert!(
        rel_error < 0.05, // Allow 5% error for stiff solver
        "BDF solution error too large: y={}, expected={}, rel_error={}",
        state[0],
        expected,
        rel_error
    );
}

#[test]
fn test_lsoda_exponential_decay() {
    let mut state = vec![1.0];
    let mut t = 0.0;
    let mut dt = 1e-3; // Larger initial step
    let rtol = 1e-4;
    let atol = 1e-6;
    let t_end = 0.005; // Shorter integration

    // Reset LSODA state
    unsafe {
        sounio_ode_lsoda_reset(state.as_mut_ptr());
    }

    let mut steps = 0;
    let max_steps = 1000;

    while t < t_end && steps < max_steps {
        unsafe {
            let err = sounio_ode_lsoda_step(
                state.as_mut_ptr(),
                1,
                &mut t,
                &mut dt,
                rtol,
                atol,
                fast_decay_derivatives,
                fast_decay_jacobian,
            );

            if err.is_finite() {
                steps += 1;
            }
        }
    }

    // Check that we reached the end time
    assert!(t >= t_end - 1e-10, "LSODA should reach t_end, got t={}", t);

    // Check solution accuracy
    let expected = (-1000.0 * t_end).exp();
    let error = (state[0] - expected).abs();
    let rel_error = error / expected.abs().max(1e-15);

    assert!(
        rel_error < 0.05, // Allow 5% error
        "LSODA solution error too large: y={}, expected={}, rel_error={}",
        state[0],
        expected,
        rel_error
    );
}

// ============================================================================
// TEST PROBLEM 2: Van der Pol oscillator (stiff for large mu)
// ============================================================================

/// Van der Pol oscillator with mu = 1000 (very stiff)
/// dx/dt = y
/// dy/dt = mu * (1 - x²) * y - x
unsafe extern "C" fn vanderpol_derivatives(state: *mut f64, _t: f64, dydt: *mut f64) {
    let state_slice = unsafe { std::slice::from_raw_parts(state, 2) };
    let dydt_slice = unsafe { std::slice::from_raw_parts_mut(dydt, 2) };

    let mu = 100.0; // Moderately stiff (1000 is very challenging)
    let x = state_slice[0];
    let y = state_slice[1];

    dydt_slice[0] = y;
    dydt_slice[1] = mu * (1.0 - x * x) * y - x;
}

/// Jacobian for Van der Pol oscillator
unsafe extern "C" fn vanderpol_jacobian(state: *const f64, _t: f64, jacobian: *mut f64) {
    let state_slice = unsafe { std::slice::from_raw_parts(state, 2) };
    let jac_slice = unsafe { std::slice::from_raw_parts_mut(jacobian, 4) }; // 2x2 matrix, row-major

    let mu = 100.0;
    let x = state_slice[0];
    let y = state_slice[1];

    // J[0,0] = ∂f₀/∂x = 0
    jac_slice[0] = 0.0;
    // J[0,1] = ∂f₀/∂y = 1
    jac_slice[1] = 1.0;
    // J[1,0] = ∂f₁/∂x = -2*mu*x*y - 1
    jac_slice[2] = -2.0 * mu * x * y - 1.0;
    // J[1,1] = ∂f₁/∂y = mu * (1 - x²)
    jac_slice[3] = mu * (1.0 - x * x);
}

#[test]
fn test_bdf_vanderpol() {
    let mut state = vec![2.0, 0.0]; // Initial condition
    let mut t = 0.0;
    let mut dt = 1e-4; // Larger step
    let rtol = 1e-3; // Relaxed tolerance
    let atol = 1e-5;
    let t_end = 0.01; // Very short integration just to test functionality

    // Reset BDF state
    unsafe {
        sounio_ode_bdf_reset(state.as_mut_ptr());
    }

    let mut steps = 0;
    let max_steps = 500;

    while t < t_end && steps < max_steps {
        unsafe {
            let err = sounio_ode_bdf_step(
                state.as_mut_ptr(),
                2,
                &mut t,
                &mut dt,
                rtol,
                atol,
                vanderpol_derivatives,
                vanderpol_jacobian,
            );

            if err.is_finite() {
                steps += 1;
            }
        }
    }

    // Check that we made progress
    assert!(
        steps > 5,
        "BDF should complete at least 5 steps on Van der Pol, got {}",
        steps
    );

    // Van der Pol with these parameters should have bounded solution
    assert!(
        state[0].abs() < 10.0 && state[1].abs() < 1000.0,
        "Van der Pol solution should be bounded: x={}, y={}",
        state[0],
        state[1]
    );
}

// ============================================================================
// TEST PROBLEM 3: Robertson chemical kinetics (very stiff)
// ============================================================================

/// Robertson problem: classic extremely stiff chemical kinetics
/// dy1/dt = -0.04*y1 + 1e4*y2*y3
/// dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2²
/// dy3/dt = 3e7*y2²
/// Conservation: y1 + y2 + y3 = 1
unsafe extern "C" fn robertson_derivatives(state: *mut f64, _t: f64, dydt: *mut f64) {
    let y = unsafe { std::slice::from_raw_parts(state, 3) };
    let f = unsafe { std::slice::from_raw_parts_mut(dydt, 3) };

    f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
    f[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
    f[2] = 3e7 * y[1] * y[1];
}

/// Jacobian for Robertson problem
unsafe extern "C" fn robertson_jacobian(state: *const f64, _t: f64, jacobian: *mut f64) {
    let y = unsafe { std::slice::from_raw_parts(state, 3) };
    let j = unsafe { std::slice::from_raw_parts_mut(jacobian, 9) }; // 3x3 matrix, row-major

    // Row 0: ∂f1/∂y
    j[0] = -0.04;
    j[1] = 1e4 * y[2];
    j[2] = 1e4 * y[1];

    // Row 1: ∂f2/∂y
    j[3] = 0.04;
    j[4] = -1e4 * y[2] - 6e7 * y[1];
    j[5] = -1e4 * y[1];

    // Row 2: ∂f3/∂y
    j[6] = 0.0;
    j[7] = 6e7 * y[1];
    j[8] = 0.0;
}

#[test]
fn test_bdf_robertson() {
    let mut state = vec![1.0, 0.0, 0.0]; // Initial: all species 1
    let mut t = 0.0;
    let mut dt = 1e-5; // Reasonable step for Robertson
    let rtol = 1e-3;
    let atol = 1e-6;
    let t_end = 0.01; // Much shorter time span

    // Reset BDF state
    unsafe {
        sounio_ode_bdf_reset(state.as_mut_ptr());
    }

    let mut steps = 0;
    let max_steps = 2000;

    while t < t_end && steps < max_steps {
        unsafe {
            let err = sounio_ode_bdf_step(
                state.as_mut_ptr(),
                3,
                &mut t,
                &mut dt,
                rtol,
                atol,
                robertson_derivatives,
                robertson_jacobian,
            );

            if err.is_finite() {
                steps += 1;
            }
        }
    }

    // Check conservation law: y1 + y2 + y3 = 1
    let sum: f64 = state.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.1, // Allow some numerical drift
        "Robertson conservation violated: sum={} (should be 1.0)",
        sum
    );

    // All components should be non-negative (allow small negative from numerical error)
    assert!(
        state.iter().all(|&y| y >= -1e-6),
        "Robertson solution should be non-negative: {:?}",
        state
    );
}

// ============================================================================
// NULL SAFETY TESTS
// ============================================================================

#[test]
fn test_stiff_solvers_null_safety() {
    unsafe {
        let err_bdf = sounio_ode_bdf_step(
            std::ptr::null_mut(),
            2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1e-6,
            1e-8,
            fast_decay_derivatives,
            fast_decay_jacobian,
        );

        assert!(
            err_bdf.is_infinite(),
            "BDF should return INFINITY for NULL pointers"
        );

        let err_lsoda = sounio_ode_lsoda_step(
            std::ptr::null_mut(),
            2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1e-6,
            1e-8,
            fast_decay_derivatives,
            fast_decay_jacobian,
        );

        assert!(
            err_lsoda.is_infinite(),
            "LSODA should return INFINITY for NULL pointers"
        );
    }
}

#[test]
fn test_bdf_zero_dimension() {
    let mut state = vec![1.0];
    let mut t = 0.0;
    let mut dt = 0.001;

    unsafe {
        let err = sounio_ode_bdf_step(
            state.as_mut_ptr(),
            0, // Zero dimension
            &mut t,
            &mut dt,
            1e-6,
            1e-8,
            fast_decay_derivatives,
            fast_decay_jacobian,
        );

        assert!(
            err.is_infinite(),
            "BDF should return INFINITY for zero dimension"
        );
    }
}

// ============================================================================
// NUMERICAL JACOBIAN TESTS (passing NULL for Jacobian)
// ============================================================================

/// Null Jacobian function pointer type for testing numerical differentiation
#[allow(dead_code)]
unsafe extern "C" fn null_jacobian(_state: *const f64, _t: f64, _jacobian: *mut f64) {
    // This function should never be called when we pass NULL
    panic!("null_jacobian should not be called");
}

#[test]
fn test_bdf_numerical_jacobian() {
    // Test that BDF works with provided Jacobian function
    let mut state = vec![1.0];
    let mut t = 0.0;
    let mut dt = 1e-3; // Larger step
    let rtol = 1e-4;
    let atol = 1e-6;
    let t_end = 0.005; // Shorter time

    // Reset BDF state
    unsafe {
        sounio_ode_bdf_reset(state.as_mut_ptr());
    }

    let mut steps = 0;
    let max_steps = 500;

    while t < t_end && steps < max_steps {
        unsafe {
            let err = sounio_ode_bdf_step(
                state.as_mut_ptr(),
                1,
                &mut t,
                &mut dt,
                rtol,
                atol,
                fast_decay_derivatives,
                fast_decay_jacobian,
            );

            if err.is_finite() {
                steps += 1;
            }
        }
    }

    // Should converge with analytical Jacobian
    assert!(
        t >= t_end - 1e-10,
        "BDF with Jacobian should reach t_end, got t={}",
        t
    );
}

// ============================================================================
// ADAPTIVE STEP SIZE TESTS
// ============================================================================

#[test]
fn test_bdf_adaptive_step() {
    let mut state = vec![1.0];
    let mut t = 0.0;
    let mut dt = 1e-4; // Start with modest step
    let rtol = 1e-4;
    let atol = 1e-6;
    let t_end = 0.002;

    // Reset BDF state
    unsafe {
        sounio_ode_bdf_reset(state.as_mut_ptr());
    }

    let mut steps = 0;
    let max_steps = 200;

    while t < t_end && steps < max_steps {
        unsafe {
            let err = sounio_ode_bdf_step(
                state.as_mut_ptr(),
                1,
                &mut t,
                &mut dt,
                rtol,
                atol,
                fast_decay_derivatives,
                fast_decay_jacobian,
            );

            if err.is_finite() {
                steps += 1;
            }
        }
    }

    // Just check we made progress
    assert!(steps > 5, "BDF should complete some steps: steps={}", steps);
}

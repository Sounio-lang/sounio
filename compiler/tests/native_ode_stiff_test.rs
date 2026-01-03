//! Integration tests for native stiff ODE solvers (BDF, LSODA)
//!
//! Tests stiff ODE solvers that require implicit methods and Jacobian computation.

use sounio::backend::native::ode_runtime::{
    sounio_ode_bdf_step, sounio_ode_lsoda_step, DerivativeFn, JacobianFn,
};

/// Van der Pol oscillator (stiff system)
/// dx/dt = y
/// dy/dt = mu * (1 - x²) * y - x
/// where mu is a parameter (large mu makes it stiff)
unsafe extern "C" fn vanderpol_derivatives(
    state: *mut f64,
    _t: f64,
    dydt: *mut f64,
) {
    let state_slice = std::slice::from_raw_parts(state, 2);
    let dydt_slice = std::slice::from_raw_parts_mut(dydt, 2);
    
    let mu = 1000.0; // Large mu makes it very stiff
    let x = state_slice[0];
    let y = state_slice[1];
    
    dydt_slice[0] = y;
    dydt_slice[1] = mu * (1.0 - x * x) * y - x;
}

/// Jacobian for Van der Pol oscillator
unsafe extern "C" fn vanderpol_jacobian(
    state: *const f64,
    _t: f64,
    jacobian: *mut f64,
) {
    let state_slice = std::slice::from_raw_parts(state, 2);
    let jac_slice = std::slice::from_raw_parts_mut(jacobian, 4); // 2x2 matrix, row-major
    
    let mu = 1000.0;
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

/// Test BDF step (placeholder - implementation returns INFINITY for now)
#[test]
fn test_bdf_step_placeholder() {
    let mut state = vec![1.0, 0.0];
    let mut t = 0.0;
    let mut dt = 0.001;
    let rtol = 1e-6;
    let atol = 1e-8;

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

        // BDF is not yet implemented, so it returns INFINITY
        assert!(err.is_infinite(), "BDF should return INFINITY (not yet implemented)");
    }
}

/// Test LSODA step (placeholder - implementation returns INFINITY for now)
#[test]
fn test_lsoda_step_placeholder() {
    let mut state = vec![1.0, 0.0];
    let mut t = 0.0;
    let mut dt = 0.001;
    let rtol = 1e-6;
    let atol = 1e-8;

    unsafe {
        let err = sounio_ode_lsoda_step(
            state.as_mut_ptr(),
            2,
            &mut t,
            &mut dt,
            rtol,
            atol,
            vanderpol_derivatives,
            vanderpol_jacobian,
        );

        // LSODA is not yet implemented, so it returns INFINITY
        assert!(err.is_infinite(), "LSODA should return INFINITY (not yet implemented)");
    }
}

/// Test that BDF and LSODA handle NULL pointers gracefully
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
            vanderpol_derivatives,
            vanderpol_jacobian,
        );
        
        assert!(err_bdf.is_infinite(), "BDF should return INFINITY for NULL pointers");
        
        let err_lsoda = sounio_ode_lsoda_step(
            std::ptr::null_mut(),
            2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1e-6,
            1e-8,
            vanderpol_derivatives,
            vanderpol_jacobian,
        );
        
        assert!(err_lsoda.is_infinite(), "LSODA should return INFINITY for NULL pointers");
    }
}

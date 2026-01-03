//! C-Compatible ODE Runtime Functions
//!
//! This module provides C-compatible wrappers for adaptive ODE solvers
//! that can be called from assembly code. These functions follow the
//! System V ABI calling convention for x86-64.
//!
//! The functions are designed to be called from the native backend's
//! assembly runtime, providing adaptive step size control and error
//! estimation for complex ODE methods.


/// C-compatible function pointer type for ODE derivatives
/// Signature: void derivatives(double* state, double t, double* dydt)
pub type DerivativeFn = unsafe extern "C" fn(*mut f64, f64, *mut f64);

/// Single DoPri5 adaptive step
///
/// C signature:
/// ```c
/// double sounio_ode_dopri5_step(
///     double* state,      // RDI: state vector (modified in-place)
///     int n,              // RSI: dimension
///     double* t,          // RDX: pointer to current time (updated)
///     double* dt,         // RCX: pointer to step size (updated)
///     double rtol,        // XMM0: relative tolerance
///     double atol,        // XMM1: absolute tolerance
///     DerivativeFn f      // R8: derivatives function pointer
/// );
/// ```
/// Returns: error estimate (in XMM0)
///
/// This function performs a single adaptive step using Dormand-Prince 5(4).
/// If the step is accepted, state, t, and dt are updated. Otherwise, dt is
/// reduced and the step is rejected.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_ode_dopri5_step(
    state: *mut f64,
    n: i32,
    t: *mut f64,
    dt: *mut f64,
    rtol: f64,
    atol: f64,
    derivatives: DerivativeFn,
) -> f64 {
    if state.is_null() || t.is_null() || dt.is_null() || n <= 0 {
        return f64::INFINITY; // Error indicator
    }

    let n = n as usize;
    let current_t = unsafe { *t };
    let current_dt = unsafe { *dt };

    // Perform single DoPri5 step
    // Dormand-Prince coefficients
    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;
    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;
    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;
    const A71: f64 = 35.0 / 384.0;
    const A73: f64 = 500.0 / 1113.0;
    const A74: f64 = 125.0 / 192.0;
    const A75: f64 = -2187.0 / 6784.0;
    const A76: f64 = 11.0 / 84.0;

    // Error coefficients
    const E1: f64 = 71.0 / 57600.0;
    const E3: f64 = -71.0 / 16695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17253.0 / 339200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    let step = current_dt;
    let state_slice = unsafe { std::slice::from_raw_parts_mut(state, n) };
    let y: Vec<f64> = state_slice.to_vec();

    // Allocate workspace for stages
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut k7 = vec![0.0; n];
    let mut y_temp = vec![0.0; n];
    let mut y_new = vec![0.0; n];

    // Stage 1: k1 = f(t, y)
    unsafe {
        derivatives(state, current_t, k1.as_mut_ptr());
    }

    // Stage 2
    for i in 0..n {
        y_temp[i] = y[i] + step * A21 * k1[i];
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + step / 5.0, k2.as_mut_ptr());
    }

    // Stage 3
    for i in 0..n {
        y_temp[i] = y[i] + step * (A31 * k1[i] + A32 * k2[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + 3.0 * step / 10.0, k3.as_mut_ptr());
    }

    // Stage 4
    for i in 0..n {
        y_temp[i] = y[i] + step * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + 4.0 * step / 5.0, k4.as_mut_ptr());
    }

    // Stage 5
    for i in 0..n {
        y_temp[i] = y[i] + step * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + 8.0 * step / 9.0, k5.as_mut_ptr());
    }

    // Stage 6
    for i in 0..n {
        y_temp[i] = y[i] + step * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + step, k6.as_mut_ptr());
    }

    // Stage 7 (5th order solution)
    for i in 0..n {
        y_new[i] = y[i] + step * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
    }
    unsafe {
        derivatives(y_new.as_mut_ptr(), current_t + step, k7.as_mut_ptr());
    }

    // Error estimate
    let mut err = 0.0;
    for i in 0..n {
        let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
        let ei = step * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        err += (ei / sc).powi(2);
    }
    err = (err / n as f64).sqrt();

    if err <= 1.0 {
        // Accept step
        for i in 0..n {
            state_slice[i] = y_new[i];
        }
        unsafe { *t = current_t + step };
        
        // Step size control
        let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 5.0 };
        unsafe { *dt = step * factor.clamp(0.2, 5.0) };
        
        err // Return error estimate
    } else {
        // Reject step - reduce dt
        let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 0.5 };
        unsafe { *dt = step * factor.clamp(0.1, 0.9) };
        f64::INFINITY // Error indicator
    }
}

/// Single CashKarp adaptive step
///
/// C signature:
/// ```c
/// double sounio_ode_cashkarp_step(
///     double* state,      // RDI: state vector (modified in-place)
///     int n,              // RSI: dimension
///     double* t,          // RDX: pointer to current time (updated)
///     double* dt,         // RCX: pointer to step size (updated)
///     double rtol,        // XMM0: relative tolerance
///     double atol,        // XMM1: absolute tolerance
///     DerivativeFn f      // R8: derivatives function pointer
/// );
/// ```
/// Returns: error estimate (in XMM0)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_ode_cashkarp_step(
    state: *mut f64,
    n: i32,
    t: *mut f64,
    dt: *mut f64,
    rtol: f64,
    atol: f64,
    derivatives: DerivativeFn,
) -> f64 {
    if state.is_null() || t.is_null() || dt.is_null() || n <= 0 {
        return f64::INFINITY;
    }

    let n = n as usize;
    let current_t = unsafe { *t };
    let current_dt = unsafe { *dt };

    // Cash-Karp coefficients (Butcher tableau)
    // Stage times
    const C2: f64 = 1.0 / 5.0;
    const C3: f64 = 3.0 / 10.0;
    const C4: f64 = 3.0 / 5.0;
    const C5: f64 = 1.0;
    const C6: f64 = 7.0 / 8.0;

    // Butcher tableau coefficients
    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 3.0 / 10.0;
    const A42: f64 = -9.0 / 10.0;
    const A43: f64 = 6.0 / 5.0;
    const A51: f64 = -11.0 / 54.0;
    const A52: f64 = 5.0 / 2.0;
    const A53: f64 = -70.0 / 27.0;
    const A54: f64 = 35.0 / 27.0;
    const A61: f64 = 1631.0 / 55296.0;
    const A62: f64 = 175.0 / 512.0;
    const A63: f64 = 575.0 / 13824.0;
    const A64: f64 = 44275.0 / 110592.0;
    const A65: f64 = 253.0 / 4096.0;

    // 5th order weights
    const B1: f64 = 37.0 / 378.0;
    const B3: f64 = 250.0 / 621.0;
    const B4: f64 = 125.0 / 594.0;
    const B6: f64 = 512.0 / 1771.0;

    // 4th order weights (for error estimation)
    const B1_4: f64 = 2825.0 / 27648.0;
    const B3_4: f64 = 18575.0 / 48384.0;
    const B4_4: f64 = 13525.0 / 55296.0;
    const B5_4: f64 = 277.0 / 14336.0;
    const B6_4: f64 = 1.0 / 4.0;

    // Error coefficients (difference between 5th and 4th order)
    const E1: f64 = B1 - B1_4;
    const E3: f64 = B3 - B3_4;
    const E4: f64 = B4 - B4_4;
    const E5: f64 = -B5_4;  // B5 is 0 for 5th order
    const E6: f64 = B6 - B6_4;

    let step = current_dt;
    let state_slice = unsafe { std::slice::from_raw_parts_mut(state, n) };
    let y: Vec<f64> = state_slice.to_vec();

    // Allocate workspace for stages
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut y_temp = vec![0.0; n];
    let mut y_new = vec![0.0; n];

    // Stage 1: k1 = f(t, y)
    unsafe {
        derivatives(state, current_t, k1.as_mut_ptr());
    }

    // Stage 2
    for i in 0..n {
        y_temp[i] = y[i] + step * A21 * k1[i];
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + C2 * step, k2.as_mut_ptr());
    }

    // Stage 3
    for i in 0..n {
        y_temp[i] = y[i] + step * (A31 * k1[i] + A32 * k2[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + C3 * step, k3.as_mut_ptr());
    }

    // Stage 4
    for i in 0..n {
        y_temp[i] = y[i] + step * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + C4 * step, k4.as_mut_ptr());
    }

    // Stage 5
    for i in 0..n {
        y_temp[i] = y[i] + step * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + C5 * step, k5.as_mut_ptr());
    }

    // Stage 6
    for i in 0..n {
        y_temp[i] = y[i] + step * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + C6 * step, k6.as_mut_ptr());
    }

    // 5th order solution
    for i in 0..n {
        y_new[i] = y[i] + step * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B6 * k6[i]);
    }

    // Error estimate
    let mut err = 0.0;
    for i in 0..n {
        let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
        let ei = step * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i]);
        err += (ei / sc).powi(2);
    }
    err = (err / n as f64).sqrt();

    if err <= 1.0 {
        // Accept step
        for i in 0..n {
            state_slice[i] = y_new[i];
        }
        unsafe { *t = current_t + step };
        
        // Step size control
        let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 5.0 };
        unsafe { *dt = step * factor.clamp(0.2, 5.0) };
        
        err // Return error estimate
    } else {
        // Reject step - reduce dt
        let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 0.5 };
        unsafe { *dt = step * factor.clamp(0.1, 0.9) };
        f64::INFINITY // Error indicator
    }
}

/// Generic ODE step dispatcher
///
/// C signature:
/// ```c
/// double sounio_ode_step(
///     int method,         // RDI: method ID (0=Euler, 1=RK4, 2=DoPri5, 3=CashKarp)
///     double* state,      // RSI: state vector
///     int n,              // RDX: dimension
///     double* t,          // RCX: pointer to current time
///     double* dt,         // R8: pointer to step size
///     double rtol,        // XMM0: relative tolerance
///     double atol,        // XMM1: absolute tolerance
///     DerivativeFn f      // Stack: derivatives function pointer
/// );
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_ode_step(
    method: i32,
    state: *mut f64,
    n: i32,
    t: *mut f64,
    dt: *mut f64,
    rtol: f64,
    atol: f64,
    derivatives: DerivativeFn,
) -> f64 {
    match method {
        0 => {
            // Euler - would need separate implementation
            f64::INFINITY
        }
        1 => {
            // RK4 - would need separate implementation
            f64::INFINITY
        }
        2 => unsafe { sounio_ode_dopri5_step(state, n, t, dt, rtol, atol, derivatives) },
        3 => unsafe { sounio_ode_cashkarp_step(state, n, t, dt, rtol, atol, derivatives) },
        _ => f64::INFINITY,
    }
}

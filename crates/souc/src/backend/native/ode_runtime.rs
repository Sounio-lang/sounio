//! C-Compatible ODE Runtime Functions
//!
//! This module provides C-compatible wrappers for adaptive ODE solvers
//! that can be called from assembly code. These functions follow the
//! System V ABI calling convention for x86-64.
//!
//! The functions are designed to be called from the native backend's
//! assembly runtime, providing adaptive step size control and error
//! estimation for complex ODE methods.

#![allow(unsafe_op_in_unsafe_fn)]

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
        derivatives(
            y_temp.as_mut_ptr(),
            current_t + 3.0 * step / 10.0,
            k3.as_mut_ptr(),
        );
    }

    // Stage 4
    for i in 0..n {
        y_temp[i] = y[i] + step * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
    }
    unsafe {
        derivatives(
            y_temp.as_mut_ptr(),
            current_t + 4.0 * step / 5.0,
            k4.as_mut_ptr(),
        );
    }

    // Stage 5
    for i in 0..n {
        y_temp[i] = y[i] + step * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
    }
    unsafe {
        derivatives(
            y_temp.as_mut_ptr(),
            current_t + 8.0 * step / 9.0,
            k5.as_mut_ptr(),
        );
    }

    // Stage 6
    for i in 0..n {
        y_temp[i] =
            y[i] + step * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
    }
    unsafe {
        derivatives(y_temp.as_mut_ptr(), current_t + step, k6.as_mut_ptr());
    }

    // Stage 7 (5th order solution)
    for i in 0..n {
        y_new[i] =
            y[i] + step * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
    }
    unsafe {
        derivatives(y_new.as_mut_ptr(), current_t + step, k7.as_mut_ptr());
    }

    // Error estimate
    let mut err = 0.0;
    for i in 0..n {
        let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
        let ei =
            step * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
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
    const E5: f64 = -B5_4; // B5 is 0 for 5th order
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
        y_temp[i] =
            y[i] + step * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
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

/// C-compatible function pointer type for ODE Jacobian
/// Signature: void jacobian(double* state, double t, double* jacobian_matrix)
/// Jacobian is stored row-major: jacobian[i*n + j] = ∂f_i/∂y_j
pub type JacobianFn = unsafe extern "C" fn(*const f64, f64, *mut f64);

// ============================================================================
// BDF (Backward Differentiation Formula) Implementation
// ============================================================================
//
// BDF is a family of implicit multi-step methods for stiff ODEs.
// BDF-k formula: sum_{j=0}^{k} alpha_j * y_{n-j} = h * beta * f(t_{n+1}, y_{n+1})
//
// Key characteristics:
// - Implicit: requires solving nonlinear system at each step
// - A-stable for orders 1-2, A(alpha)-stable for orders 3-5
// - Excellent for stiff problems (large eigenvalue spread)
// - Variable order (1-5) and step size for efficiency

use std::collections::HashMap;
use std::sync::Mutex;

/// BDF solver state (maintains history across calls)
struct BDFState {
    /// Previous solution values (most recent first)
    history_y: Vec<Vec<f64>>,
    /// Previous time points
    history_t: Vec<f64>,
    /// Previous step sizes
    history_h: Vec<f64>,
    /// Current order (1-5)
    order: usize,
    /// Number of steps at current order
    steps_at_order: usize,
    /// Last Jacobian matrix (for reuse)
    jacobian_cache: Vec<f64>,
    /// Last time Jacobian was computed
    jacobian_t: f64,
    /// LU factorization of iteration matrix
    lu_matrix: Vec<f64>,
    lu_pivot: Vec<usize>,
    /// Whether LU is valid
    lu_valid: bool,
}

impl BDFState {
    fn new(n: usize) -> Self {
        Self {
            history_y: Vec::with_capacity(6),
            history_t: Vec::with_capacity(6),
            history_h: Vec::with_capacity(6),
            order: 1,
            steps_at_order: 0,
            jacobian_cache: vec![0.0; n * n],
            jacobian_t: f64::NEG_INFINITY,
            lu_matrix: vec![0.0; n * n],
            lu_pivot: vec![0; n],
            lu_valid: false,
        }
    }

    fn reset(&mut self, n: usize) {
        self.history_y.clear();
        self.history_t.clear();
        self.history_h.clear();
        self.order = 1;
        self.steps_at_order = 0;
        self.jacobian_cache = vec![0.0; n * n];
        self.jacobian_t = f64::NEG_INFINITY;
        self.lu_matrix = vec![0.0; n * n];
        self.lu_pivot = vec![0; n];
        self.lu_valid = false;
    }
}

/// Global BDF state storage (keyed by state pointer address for thread safety)
static BDF_STATES: Mutex<Option<HashMap<usize, BDFState>>> = Mutex::new(None);

/// BDF coefficients for orders 1-5
/// Returns (alpha, beta, error_coefficient)
fn bdf_coefficients(order: usize) -> (Vec<f64>, f64, f64) {
    match order {
        1 => {
            // BDF1 (Backward Euler): y_{n+1} - y_n = h*f(t_{n+1}, y_{n+1})
            (vec![1.0, -1.0], 1.0, 0.5)
        }
        2 => {
            // BDF2: (3/2)y_{n+1} - 2y_n + (1/2)y_{n-1} = h*f(t_{n+1}, y_{n+1})
            (vec![3.0 / 2.0, -2.0, 1.0 / 2.0], 1.0, 2.0 / 9.0)
        }
        3 => {
            // BDF3: (11/6)y_{n+1} - 3y_n + (3/2)y_{n-1} - (1/3)y_{n-2} = h*f
            (
                vec![11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0],
                1.0,
                3.0 / 22.0,
            )
        }
        4 => {
            // BDF4
            (
                vec![25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0],
                1.0,
                12.0 / 125.0,
            )
        }
        5 => {
            // BDF5
            (
                vec![137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
                1.0,
                10.0 / 137.0,
            )
        }
        _ => panic!("BDF order must be 1-5"),
    }
}

/// LU decomposition with partial pivoting
fn lu_decompose(a: &mut [f64], pivot: &mut [usize], n: usize) -> bool {
    for i in 0..n {
        pivot[i] = i;
    }

    for k in 0..n {
        // Find pivot
        let mut max_val = a[k * n + k].abs();
        let mut max_idx = k;
        for i in (k + 1)..n {
            let val = a[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < 1e-15 {
            return false; // Singular matrix
        }

        // Swap rows
        if max_idx != k {
            pivot.swap(k, max_idx);
            for j in 0..n {
                let tmp = a[k * n + j];
                a[k * n + j] = a[max_idx * n + j];
                a[max_idx * n + j] = tmp;
            }
        }

        // Elimination
        for i in (k + 1)..n {
            a[i * n + k] /= a[k * n + k];
            for j in (k + 1)..n {
                a[i * n + j] -= a[i * n + k] * a[k * n + j];
            }
        }
    }
    true
}

/// Solve LU * x = b
fn lu_solve(lu: &[f64], pivot: &[usize], b: &[f64], x: &mut [f64], n: usize) {
    // Apply permutation and forward substitution
    for i in 0..n {
        x[i] = b[pivot[i]];
    }
    for i in 0..n {
        for j in 0..i {
            x[i] -= lu[i * n + j] * x[j];
        }
    }

    // Backward substitution
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu[i * n + j] * x[j];
        }
        x[i] /= lu[i * n + i];
    }
}

/// Compute numerical Jacobian using finite differences
unsafe fn compute_numerical_jacobian(
    derivatives: DerivativeFn,
    state: *const f64,
    t: f64,
    jac: &mut [f64],
    n: usize,
) {
    let eps = 1e-8;
    let mut f0 = vec![0.0; n];
    let mut f1 = vec![0.0; n];
    let mut y_pert = vec![0.0; n];

    // Base derivatives
    unsafe { std::ptr::copy_nonoverlapping(state, y_pert.as_mut_ptr(), n) };
    derivatives(y_pert.as_mut_ptr(), t, f0.as_mut_ptr());

    // Perturb each component
    for j in 0..n {
        let y_j = *state.add(j);
        let dy = eps * (1.0 + y_j.abs());
        y_pert[j] = y_j + dy;

        derivatives(y_pert.as_mut_ptr(), t, f1.as_mut_ptr());

        for i in 0..n {
            jac[i * n + j] = (f1[i] - f0[i]) / dy;
        }

        y_pert[j] = y_j;
    }
}

/// Single BDF step for stiff ODEs
///
/// C signature:
/// ```c
/// double sounio_ode_bdf_step(
///     double* state,      // RDI: state vector (modified in-place)
///     int n,              // RSI: dimension
///     double* t,          // RDX: pointer to current time (updated)
///     double* dt,         // RCX: pointer to step size (updated)
///     double rtol,        // XMM0: relative tolerance
///     double atol,        // XMM1: absolute tolerance
///     DerivativeFn f,     // R8: derivatives function pointer
///     JacobianFn jac      // R9: jacobian function pointer (can be NULL for numerical)
/// );
/// ```
/// Returns: error estimate (in XMM0), INFINITY if step rejected
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_ode_bdf_step(
    state: *mut f64,
    n: i32,
    t: *mut f64,
    dt: *mut f64,
    rtol: f64,
    atol: f64,
    derivatives: DerivativeFn,
    jacobian: JacobianFn,
) -> f64 {
    if state.is_null() || t.is_null() || dt.is_null() || n <= 0 {
        return f64::INFINITY;
    }

    let n = n as usize;
    let current_t = *t;
    let h = *dt;
    let state_key = state as usize;

    // Get or create BDF state
    let mut states_guard = BDF_STATES.lock().unwrap();
    if states_guard.is_none() {
        *states_guard = Some(HashMap::new());
    }
    let states = states_guard.as_mut().unwrap();

    if !states.contains_key(&state_key) {
        states.insert(state_key, BDFState::new(n));
    }
    let bdf_state = states.get_mut(&state_key).unwrap();

    // Copy current state
    let state_slice = std::slice::from_raw_parts_mut(state, n);
    let y_current: Vec<f64> = state_slice.to_vec();

    // Initialize history if empty (first call)
    if bdf_state.history_y.is_empty() {
        bdf_state.history_y.push(y_current.clone());
        bdf_state.history_t.push(current_t);
        bdf_state.order = 1;
    }

    // Determine usable order based on history
    let usable_order = bdf_state.order.min(bdf_state.history_y.len()).min(5);
    let (alpha, beta, error_coef) = bdf_coefficients(usable_order);

    // Target time
    let t_new = current_t + h;

    // Compute or update Jacobian
    let jacobian_age = (current_t - bdf_state.jacobian_t).abs();
    let need_new_jacobian = !bdf_state.lu_valid || jacobian_age > h * 10.0;

    if need_new_jacobian {
        // Compute Jacobian
        // Note: In Rust, function pointers cannot be null, so we always use the provided Jacobian.
        // If the caller wants numerical Jacobian, they should pass a Jacobian function that
        // computes it numerically, or we can detect a dummy function in the future.
        jacobian(state, current_t, bdf_state.jacobian_cache.as_mut_ptr());
        bdf_state.jacobian_t = current_t;

        // Form iteration matrix: M = alpha[0]*I - h*beta*J
        let gamma = h * beta / alpha[0];
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                bdf_state.lu_matrix[idx] =
                    if i == j { 1.0 } else { 0.0 } - gamma * bdf_state.jacobian_cache[idx];
            }
        }

        // LU decomposition
        bdf_state.lu_valid = lu_decompose(&mut bdf_state.lu_matrix, &mut bdf_state.lu_pivot, n);

        if !bdf_state.lu_valid {
            // Singular matrix - reduce step size
            *dt = h * 0.25;
            return f64::INFINITY;
        }
    }

    // Predictor: extrapolate from history
    let mut y_pred = y_current.clone();
    if bdf_state.history_y.len() >= 2 {
        // Simple linear extrapolation
        let y0 = &bdf_state.history_y[0];
        let y1 = &bdf_state.history_y[1];
        for i in 0..n {
            y_pred[i] = y0[i] + (y0[i] - y1[i]);
        }
    }

    // Newton iteration
    let mut y_new = y_pred.clone();
    let mut converged = false;
    let max_newton_iterations = 10;
    let newton_tol = 1e-10;
    let mut residual = vec![0.0; n];
    let mut delta = vec![0.0; n];
    let mut f_new = vec![0.0; n];

    for newton_iter in 0..max_newton_iterations {
        // Compute f(t_new, y_new)
        derivatives(y_new.as_mut_ptr(), t_new, f_new.as_mut_ptr());

        // Compute residual: alpha[0]*y_new - sum(alpha[j]*y_j) - h*beta*f(t_new, y_new)
        for i in 0..n {
            residual[i] = alpha[0] * y_new[i];

            // Add history terms
            for (j, y_hist) in bdf_state.history_y.iter().enumerate().take(usable_order) {
                if j + 1 < alpha.len() {
                    residual[i] += alpha[j + 1] * y_hist[i];
                }
            }

            residual[i] -= h * beta * f_new[i];
        }

        // Solve M * delta = residual
        lu_solve(
            &bdf_state.lu_matrix,
            &bdf_state.lu_pivot,
            &residual,
            &mut delta,
            n,
        );

        // Update and check convergence
        let mut max_delta: f64 = 0.0;
        for i in 0..n {
            y_new[i] -= delta[i];
            let scale = atol + rtol * y_new[i].abs();
            max_delta = max_delta.max(delta[i].abs() / scale);
        }

        if max_delta < newton_tol {
            converged = true;
            break;
        }

        // Check for divergence
        if newton_iter > 2 && max_delta > 1.0 {
            break;
        }
    }

    if !converged {
        // Newton failed to converge - reduce step size and order
        *dt = h * 0.25;
        if bdf_state.order > 1 {
            bdf_state.order -= 1;
            bdf_state.steps_at_order = 0;
        }
        bdf_state.lu_valid = false;
        return f64::INFINITY;
    }

    // Estimate error (difference from predictor scaled by error coefficient)
    let mut error_norm = 0.0;
    for i in 0..n {
        let diff = y_new[i] - y_pred[i];
        let scale = atol + rtol * y_new[i].abs().max(y_pred[i].abs());
        error_norm += (diff / scale).powi(2);
    }
    error_norm = (error_norm / n as f64).sqrt() * error_coef;

    if error_norm <= 1.0 {
        // Accept step - update state and history
        for i in 0..n {
            state_slice[i] = y_new[i];
        }
        *t = t_new;

        // Update history (insert at front, limit size)
        bdf_state.history_y.insert(0, y_new);
        bdf_state.history_t.insert(0, t_new);
        bdf_state.history_h.insert(0, h);
        if bdf_state.history_y.len() > 6 {
            bdf_state.history_y.pop();
            bdf_state.history_t.pop();
            bdf_state.history_h.pop();
        }

        // Order and step size control
        bdf_state.steps_at_order += 1;

        // Consider increasing order if error is small and we have enough history
        if error_norm < 0.1
            && bdf_state.order < 5
            && bdf_state.history_y.len() > bdf_state.order
            && bdf_state.steps_at_order >= 3
        {
            bdf_state.order += 1;
            bdf_state.steps_at_order = 0;
            bdf_state.lu_valid = false;
        }

        // Step size adjustment
        let safety = 0.9;
        let order_exp = 1.0 / (usable_order as f64 + 1.0);
        let factor = safety * error_norm.powf(-order_exp);
        *dt = h * factor.clamp(0.2, 5.0);

        error_norm
    } else {
        // Reject step - reduce step size
        let safety = 0.9;
        let order_exp = 1.0 / (usable_order as f64 + 1.0);
        let factor = safety * error_norm.powf(-order_exp);
        *dt = h * factor.clamp(0.1, 0.9);

        // Consider reducing order if struggling
        if bdf_state.order > 1 && error_norm > 10.0 {
            bdf_state.order -= 1;
            bdf_state.steps_at_order = 0;
        }

        bdf_state.lu_valid = false;
        f64::INFINITY
    }
}

// ============================================================================
// LSODA Implementation (Automatic Stiff/Non-stiff Switching)
// ============================================================================
//
// LSODA automatically switches between:
// - Adams-Moulton methods (non-stiff, up to order 12)
// - BDF methods (stiff, up to order 5)
//
// The switching is based on monitoring:
// 1. Step size behavior (small steps suggest stiffness)
// 2. Jacobian eigenvalue estimates
// 3. Convergence of Newton iteration

/// Method type for LSODA
#[derive(Clone, Copy, PartialEq)]
enum LSODAMethod {
    Adams, // Non-stiff (explicit predictor-corrector)
    BDF,   // Stiff (implicit)
}

/// LSODA solver state
struct LSODAState {
    /// Current method type
    method: LSODAMethod,
    /// History of y values (most recent first)
    history_y: Vec<Vec<f64>>,
    /// History of f values (most recent first)
    history_f: Vec<Vec<f64>>,
    /// History of time points
    history_t: Vec<f64>,
    /// Current order
    order: usize,
    /// Steps since last method switch
    steps_since_switch: usize,
    /// Steps at current order
    steps_at_order: usize,
    /// Jacobian cache (for BDF)
    jacobian_cache: Vec<f64>,
    jacobian_t: f64,
    /// LU factorization (for BDF)
    lu_matrix: Vec<f64>,
    lu_pivot: Vec<usize>,
    lu_valid: bool,
    /// Stiffness detection counters
    stiff_count: usize,
    nonstiff_count: usize,
}

impl LSODAState {
    fn new(n: usize) -> Self {
        Self {
            method: LSODAMethod::Adams,
            history_y: Vec::with_capacity(13),
            history_f: Vec::with_capacity(13),
            history_t: Vec::with_capacity(13),
            order: 1,
            steps_since_switch: 0,
            steps_at_order: 0,
            jacobian_cache: vec![0.0; n * n],
            jacobian_t: f64::NEG_INFINITY,
            lu_matrix: vec![0.0; n * n],
            lu_pivot: vec![0; n],
            lu_valid: false,
            stiff_count: 0,
            nonstiff_count: 0,
        }
    }
}

/// Global LSODA state storage
static LSODA_STATES: Mutex<Option<HashMap<usize, LSODAState>>> = Mutex::new(None);

/// Adams-Bashforth predictor coefficients (up to order 4 for simplicity)
fn adams_bashforth_coefficients(order: usize) -> Vec<f64> {
    match order {
        1 => vec![1.0],
        2 => vec![3.0 / 2.0, -1.0 / 2.0],
        3 => vec![23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
        4 => vec![55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
        _ => vec![1.0],
    }
}

/// Adams-Moulton corrector coefficients (up to order 4)
fn adams_moulton_coefficients(order: usize) -> Vec<f64> {
    match order {
        1 => vec![1.0 / 2.0, 1.0 / 2.0],
        2 => vec![5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0],
        3 => vec![9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0],
        4 => vec![
            251.0 / 720.0,
            646.0 / 720.0,
            -264.0 / 720.0,
            106.0 / 720.0,
            -19.0 / 720.0,
        ],
        _ => vec![1.0 / 2.0, 1.0 / 2.0],
    }
}

/// Single LSODA step (automatic stiff/non-stiff switching)
///
/// C signature:
/// ```c
/// double sounio_ode_lsoda_step(
///     double* state,
///     int n,
///     double* t,
///     double* dt,
///     double rtol,
///     double atol,
///     DerivativeFn f,
///     JacobianFn jac
/// );
/// ```
/// Returns: error estimate (in XMM0)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_ode_lsoda_step(
    state: *mut f64,
    n: i32,
    t: *mut f64,
    dt: *mut f64,
    rtol: f64,
    atol: f64,
    derivatives: DerivativeFn,
    jacobian: JacobianFn,
) -> f64 {
    if state.is_null() || t.is_null() || dt.is_null() || n <= 0 {
        return f64::INFINITY;
    }

    let n = n as usize;
    let current_t = *t;
    let h = *dt;
    let state_key = state as usize;

    // Get or create LSODA state
    let mut states_guard = LSODA_STATES.lock().unwrap();
    if states_guard.is_none() {
        *states_guard = Some(HashMap::new());
    }
    let states = states_guard.as_mut().unwrap();

    if !states.contains_key(&state_key) {
        states.insert(state_key, LSODAState::new(n));
    }
    let lsoda_state = states.get_mut(&state_key).unwrap();

    // Copy current state
    let state_slice = std::slice::from_raw_parts_mut(state, n);
    let y_current: Vec<f64> = state_slice.to_vec();

    // Compute current derivatives
    let mut f_current = vec![0.0; n];
    derivatives(state, current_t, f_current.as_mut_ptr());

    // Initialize history if empty
    if lsoda_state.history_y.is_empty() {
        lsoda_state.history_y.push(y_current.clone());
        lsoda_state.history_f.push(f_current.clone());
        lsoda_state.history_t.push(current_t);
    }

    let t_new = current_t + h;

    // Perform step based on current method
    let (y_new, f_new, error_norm, step_accepted) = match lsoda_state.method {
        LSODAMethod::Adams => {
            // Adams predictor-corrector step
            lsoda_adams_step(
                &y_current,
                &f_current,
                current_t,
                h,
                rtol,
                atol,
                derivatives,
                lsoda_state,
                n,
            )
        }
        LSODAMethod::BDF => {
            // BDF step
            lsoda_bdf_step(
                &y_current,
                current_t,
                h,
                rtol,
                atol,
                derivatives,
                jacobian,
                lsoda_state,
                n,
            )
        }
    };

    if step_accepted {
        // Update state
        for i in 0..n {
            state_slice[i] = y_new[i];
        }
        *t = t_new;

        // Update history
        lsoda_state.history_y.insert(0, y_new);
        lsoda_state.history_f.insert(0, f_new);
        lsoda_state.history_t.insert(0, t_new);

        // Limit history size based on method
        let max_history = if lsoda_state.method == LSODAMethod::Adams {
            13
        } else {
            6
        };
        while lsoda_state.history_y.len() > max_history {
            lsoda_state.history_y.pop();
            lsoda_state.history_f.pop();
            lsoda_state.history_t.pop();
        }

        lsoda_state.steps_since_switch += 1;
        lsoda_state.steps_at_order += 1;

        // Stiffness detection (every 10 steps)
        if lsoda_state.steps_since_switch >= 10 {
            let is_stiff = detect_stiffness_lsoda(state, current_t, h, derivatives, n);

            if is_stiff && lsoda_state.method == LSODAMethod::Adams {
                lsoda_state.stiff_count += 1;
                lsoda_state.nonstiff_count = 0;
                if lsoda_state.stiff_count >= 3 {
                    // Switch to BDF
                    lsoda_state.method = LSODAMethod::BDF;
                    lsoda_state.order = 1;
                    lsoda_state.steps_since_switch = 0;
                    lsoda_state.steps_at_order = 0;
                    lsoda_state.lu_valid = false;
                    lsoda_state.stiff_count = 0;
                }
            } else if !is_stiff && lsoda_state.method == LSODAMethod::BDF {
                lsoda_state.nonstiff_count += 1;
                lsoda_state.stiff_count = 0;
                if lsoda_state.nonstiff_count >= 5 {
                    // Switch to Adams
                    lsoda_state.method = LSODAMethod::Adams;
                    lsoda_state.order = 1;
                    lsoda_state.steps_since_switch = 0;
                    lsoda_state.steps_at_order = 0;
                    lsoda_state.nonstiff_count = 0;
                }
            } else {
                lsoda_state.stiff_count = 0;
                lsoda_state.nonstiff_count = 0;
            }
        }

        // Order control
        let max_order = if lsoda_state.method == LSODAMethod::Adams {
            4
        } else {
            5
        };
        if error_norm < 0.1
            && lsoda_state.order < max_order
            && lsoda_state.history_y.len() > lsoda_state.order
            && lsoda_state.steps_at_order >= 3
        {
            lsoda_state.order += 1;
            lsoda_state.steps_at_order = 0;
            lsoda_state.lu_valid = false;
        }

        // Step size control
        let safety = 0.9;
        let order_exp = 1.0 / (lsoda_state.order as f64 + 1.0);
        let factor = safety * error_norm.powf(-order_exp);
        *dt = h * factor.clamp(0.2, 5.0);

        error_norm
    } else {
        // Step rejected - reduce step size
        let safety = 0.9;
        let order_exp = 1.0 / (lsoda_state.order as f64 + 1.0);
        let factor = safety * error_norm.powf(-order_exp);
        *dt = h * factor.clamp(0.1, 0.9);

        // Reduce order if struggling
        if lsoda_state.order > 1 && error_norm > 10.0 {
            lsoda_state.order -= 1;
            lsoda_state.steps_at_order = 0;
        }

        // If Adams is failing repeatedly, try BDF
        if lsoda_state.method == LSODAMethod::Adams {
            lsoda_state.stiff_count += 1;
            if lsoda_state.stiff_count >= 5 {
                lsoda_state.method = LSODAMethod::BDF;
                lsoda_state.order = 1;
                lsoda_state.steps_since_switch = 0;
                lsoda_state.stiff_count = 0;
                lsoda_state.lu_valid = false;
            }
        }

        f64::INFINITY
    }
}

/// Adams predictor-corrector step
unsafe fn lsoda_adams_step(
    y: &[f64],
    f: &[f64],
    t: f64,
    h: f64,
    rtol: f64,
    atol: f64,
    derivatives: DerivativeFn,
    state: &LSODAState,
    n: usize,
) -> (Vec<f64>, Vec<f64>, f64, bool) {
    let order = state.order.min(state.history_f.len()).min(4);

    // Adams-Bashforth predictor
    let ab_coef = adams_bashforth_coefficients(order);
    let mut y_pred = y.to_vec();

    for i in 0..n {
        for (j, coef) in ab_coef.iter().enumerate() {
            if j < state.history_f.len() {
                y_pred[i] += h * coef * state.history_f[j][i];
            }
        }
    }

    // Evaluate at predicted point
    let mut f_pred = vec![0.0; n];
    derivatives(y_pred.as_mut_ptr(), t + h, f_pred.as_mut_ptr());

    // Adams-Moulton corrector
    let am_coef = adams_moulton_coefficients(order);
    let mut y_new = y.to_vec();

    y_new.iter_mut().enumerate().for_each(|(i, yi)| {
        *yi += h * am_coef[0] * f_pred[i];
        for (j, coef) in am_coef.iter().enumerate().skip(1) {
            if j - 1 < state.history_f.len() {
                *yi += h * coef * state.history_f[j - 1][i];
            }
        }
    });

    // Final evaluation
    let mut f_new = vec![0.0; n];
    derivatives(y_new.as_mut_ptr(), t + h, f_new.as_mut_ptr());

    // Error estimate (difference between predictor and corrector)
    let mut error_norm = 0.0;
    for i in 0..n {
        let diff = (y_new[i] - y_pred[i]).abs();
        let scale = atol + rtol * y_new[i].abs().max(y_pred[i].abs());
        error_norm += (diff / scale).powi(2);
    }
    error_norm = (error_norm / n as f64).sqrt();

    let accepted = error_norm <= 1.0;
    (y_new, f_new, error_norm, accepted)
}

/// BDF step for LSODA
unsafe fn lsoda_bdf_step(
    y: &[f64],
    t: f64,
    h: f64,
    rtol: f64,
    atol: f64,
    derivatives: DerivativeFn,
    jacobian: JacobianFn,
    state: &mut LSODAState,
    n: usize,
) -> (Vec<f64>, Vec<f64>, f64, bool) {
    let order = state.order.min(state.history_y.len()).min(5);
    let (alpha, beta, error_coef) = bdf_coefficients(order);

    let t_new = t + h;

    // Update Jacobian if needed
    let jacobian_age = (t - state.jacobian_t).abs();
    if !state.lu_valid || jacobian_age > h * 10.0 {
        // Use provided Jacobian function
        jacobian(y.as_ptr(), t, state.jacobian_cache.as_mut_ptr());
        state.jacobian_t = t;

        // Form iteration matrix
        let gamma = h * beta / alpha[0];
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                state.lu_matrix[idx] =
                    if i == j { 1.0 } else { 0.0 } - gamma * state.jacobian_cache[idx];
            }
        }

        state.lu_valid = lu_decompose(&mut state.lu_matrix, &mut state.lu_pivot, n);
        if !state.lu_valid {
            return (y.to_vec(), vec![0.0; n], f64::INFINITY, false);
        }
    }

    // Predictor
    let mut y_pred = y.to_vec();
    if state.history_y.len() >= 2 {
        let y0 = &state.history_y[0];
        let y1 = &state.history_y[1];
        for i in 0..n {
            y_pred[i] = y0[i] + (y0[i] - y1[i]);
        }
    }

    // Newton iteration
    let mut y_new = y_pred.clone();
    let mut converged = false;
    let mut residual = vec![0.0; n];
    let mut delta = vec![0.0; n];
    let mut f_new = vec![0.0; n];

    for newton_iter in 0..10 {
        derivatives(y_new.as_mut_ptr(), t_new, f_new.as_mut_ptr());

        for i in 0..n {
            residual[i] = alpha[0] * y_new[i];
            for (j, y_hist) in state.history_y.iter().enumerate().take(order) {
                if j + 1 < alpha.len() {
                    residual[i] += alpha[j + 1] * y_hist[i];
                }
            }
            residual[i] -= h * beta * f_new[i];
        }

        lu_solve(&state.lu_matrix, &state.lu_pivot, &residual, &mut delta, n);

        let mut max_delta: f64 = 0.0;
        for i in 0..n {
            y_new[i] -= delta[i];
            let scale = atol + rtol * y_new[i].abs();
            max_delta = max_delta.max(delta[i].abs() / scale);
        }

        if max_delta < 1e-10 {
            converged = true;
            break;
        }

        if newton_iter > 2 && max_delta > 1.0 {
            break;
        }
    }

    if !converged {
        return (y.to_vec(), vec![0.0; n], f64::INFINITY, false);
    }

    // Final evaluation
    derivatives(y_new.as_mut_ptr(), t_new, f_new.as_mut_ptr());

    // Error estimate
    let mut error_norm = 0.0;
    for i in 0..n {
        let diff = y_new[i] - y_pred[i];
        let scale = atol + rtol * y_new[i].abs().max(y_pred[i].abs());
        error_norm += (diff / scale).powi(2);
    }
    error_norm = (error_norm / n as f64).sqrt() * error_coef;

    let accepted = error_norm <= 1.0;
    (y_new, f_new, error_norm, accepted)
}

/// Detect stiffness using spectral radius estimate
unsafe fn detect_stiffness_lsoda(
    state: *const f64,
    t: f64,
    h: f64,
    derivatives: DerivativeFn,
    n: usize,
) -> bool {
    // Estimate spectral radius using power iteration approximation
    let eps = 1e-8;
    let mut f0 = vec![0.0; n];
    let mut f1 = vec![0.0; n];
    let mut y_pert = vec![0.0; n];
    let v = vec![1.0 / (n as f64).sqrt(); n];

    std::ptr::copy_nonoverlapping(state, y_pert.as_mut_ptr(), n);
    derivatives(y_pert.as_mut_ptr(), t, f0.as_mut_ptr());

    // Perturb in direction of v
    for i in 0..n {
        y_pert[i] = unsafe { *state.add(i) } + eps * v[i];
    }
    derivatives(y_pert.as_mut_ptr(), t, f1.as_mut_ptr());

    // Approximate J*v
    let mut jv_norm = 0.0;
    for i in 0..n {
        let jv_i = (f1[i] - f0[i]) / eps;
        jv_norm += jv_i * jv_i;
    }
    let eigenvalue_estimate = jv_norm.sqrt();

    // System is stiff if h * |lambda| >> 1
    eigenvalue_estimate * h > 5.0
}

/// Reset BDF state for a state pointer (call when starting new integration)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_ode_bdf_reset(state: *mut f64) {
    if !state.is_null() {
        let state_key = state as usize;
        if let Ok(mut guard) = BDF_STATES.lock() {
            if let Some(states) = guard.as_mut() {
                states.remove(&state_key);
            }
        }
    }
}

/// Reset LSODA state for a state pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sounio_ode_lsoda_reset(state: *mut f64) {
    if !state.is_null() {
        let state_key = state as usize;
        if let Ok(mut guard) = LSODA_STATES.lock() {
            if let Some(states) = guard.as_mut() {
                states.remove(&state_key);
            }
        }
    }
}

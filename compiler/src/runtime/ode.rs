//! ODE (Ordinary Differential Equations) Solver Runtime
//!
//! This module provides native ODE solvers for Sounio, enabling
//! scientific simulation of dynamical systems.
//!
//! Supported solvers:
//! - Euler: Simple forward Euler (first-order)
//! - RK4: Classic 4th-order Runge-Kutta (fixed step)
//! - RK45/Dormand-Prince: Adaptive step size (5th-order with 4th-order error estimate)

use std::fmt;

/// ODE solution containing time points and state values
#[derive(Debug, Clone)]
pub struct ODESolution {
    /// Time points
    pub t: Vec<f64>,
    /// State values at each time point (each entry is a vector of state variables)
    pub y: Vec<Vec<f64>>,
    /// Solver statistics
    pub stats: SolverStats,
}

impl ODESolution {
    /// Create a new empty solution
    pub fn new() -> Self {
        Self {
            t: Vec::new(),
            y: Vec::new(),
            stats: SolverStats::default(),
        }
    }

    /// Interpolate solution at a specific time point
    pub fn y_at(&self, t_query: f64) -> Option<Vec<f64>> {
        if self.t.is_empty() {
            return None;
        }

        // Find the interval containing t_query
        if t_query <= self.t[0] {
            return Some(self.y[0].clone());
        }
        if t_query >= *self.t.last().unwrap() {
            return Some(self.y.last().unwrap().clone());
        }

        // Binary search for the interval
        let idx = self.t.partition_point(|&x| x < t_query);
        if idx == 0 {
            return Some(self.y[0].clone());
        }

        // Linear interpolation
        let t0 = self.t[idx - 1];
        let t1 = self.t[idx];
        let alpha = (t_query - t0) / (t1 - t0);

        let y0 = &self.y[idx - 1];
        let y1 = &self.y[idx];

        let result: Vec<f64> = y0
            .iter()
            .zip(y1.iter())
            .map(|(&a, &b)| a + alpha * (b - a))
            .collect();

        Some(result)
    }

    /// Get the final state
    pub fn final_state(&self) -> Option<&Vec<f64>> {
        self.y.last()
    }

    /// Get the number of time points
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// Check if solution is empty
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
}

impl Default for ODESolution {
    fn default() -> Self {
        Self::new()
    }
}

/// Solver statistics
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Number of function evaluations
    pub n_evals: usize,
    /// Number of accepted steps
    pub n_steps: usize,
    /// Number of rejected steps (for adaptive methods)
    pub n_rejected: usize,
    /// Total solve time in seconds
    pub solve_time: f64,
}

/// Solver options
#[derive(Debug, Clone)]
pub struct SolverOptions {
    /// Absolute tolerance (for adaptive methods)
    pub abstol: f64,
    /// Relative tolerance (for adaptive methods)
    pub reltol: f64,
    /// Maximum step size
    pub max_step: f64,
    /// Minimum step size
    pub min_step: f64,
    /// Initial step size (0 for automatic)
    pub initial_step: f64,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Store dense output for interpolation
    pub dense: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            abstol: 1e-6,
            reltol: 1e-3,
            max_step: f64::INFINITY,
            min_step: 1e-14,
            initial_step: 0.0,
            max_steps: 100_000,
            dense: true,
        }
    }
}

impl SolverOptions {
    /// Create options with specified tolerances
    pub fn with_tolerances(abstol: f64, reltol: f64) -> Self {
        Self {
            abstol,
            reltol,
            ..Default::default()
        }
    }
}

/// Solver method enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMethod {
    /// Forward Euler (1st order, fixed step)
    Euler,
    /// Classic Runge-Kutta (4th order, fixed step)
    RK4,
    /// Dormand-Prince (5th order, adaptive step)
    RK45,
    /// Alias for RK45
    DoPri5,
}

impl SolverMethod {
    /// Parse solver method from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "euler" => Some(SolverMethod::Euler),
            "rk4" => Some(SolverMethod::RK4),
            "rk45" | "dopri5" | "dp5" => Some(SolverMethod::RK45),
            _ => None,
        }
    }
}

impl fmt::Display for SolverMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverMethod::Euler => write!(f, "Euler"),
            SolverMethod::RK4 => write!(f, "RK4"),
            SolverMethod::RK45 | SolverMethod::DoPri5 => write!(f, "RK45 (Dormand-Prince)"),
        }
    }
}

/// ODE trait for defining differential equations
pub trait ODE {
    /// Compute the derivative dy/dt = f(t, y)
    fn derivative(&self, t: f64, y: &[f64], dydt: &mut [f64]);

    /// Get the number of state variables
    fn dimension(&self) -> usize;
}

/// Solve an ODE using the specified method
pub fn solve<F>(
    f: F,
    y0: &[f64],
    t_span: (f64, f64),
    method: SolverMethod,
    options: &SolverOptions,
) -> ODESolution
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    match method {
        SolverMethod::Euler => solve_euler(f, y0, t_span, options),
        SolverMethod::RK4 => solve_rk4(f, y0, t_span, options),
        SolverMethod::RK45 | SolverMethod::DoPri5 => solve_rk45(f, y0, t_span, options),
    }
}

/// Forward Euler solver (1st order, fixed step)
pub fn solve_euler<F>(f: F, y0: &[f64], t_span: (f64, f64), options: &SolverOptions) -> ODESolution
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let start_time = std::time::Instant::now();
    let (t0, tf) = t_span;
    let n = y0.len();

    // Determine step size
    let h = if options.initial_step > 0.0 {
        options.initial_step
    } else {
        (tf - t0) / 1000.0
    }
    .min(options.max_step);

    let mut solution = ODESolution::new();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut dydt = vec![0.0; n];

    solution.t.push(t);
    solution.y.push(y.clone());

    while t < tf && solution.stats.n_steps < options.max_steps {
        let step = h.min(tf - t);

        // Compute derivative
        f(t, &y, &mut dydt);
        solution.stats.n_evals += 1;

        // Euler step: y_new = y + h * f(t, y)
        for i in 0..n {
            y[i] += step * dydt[i];
        }
        t += step;

        solution.t.push(t);
        solution.y.push(y.clone());
        solution.stats.n_steps += 1;
    }

    solution.stats.solve_time = start_time.elapsed().as_secs_f64();
    solution
}

/// Classic RK4 solver (4th order, fixed step)
pub fn solve_rk4<F>(f: F, y0: &[f64], t_span: (f64, f64), options: &SolverOptions) -> ODESolution
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let start_time = std::time::Instant::now();
    let (t0, tf) = t_span;
    let n = y0.len();

    // Determine step size
    let h = if options.initial_step > 0.0 {
        options.initial_step
    } else {
        (tf - t0) / 1000.0
    }
    .min(options.max_step);

    let mut solution = ODESolution::new();
    let mut t = t0;
    let mut y = y0.to_vec();

    // Temporary vectors
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut y_temp = vec![0.0; n];

    solution.t.push(t);
    solution.y.push(y.clone());

    while t < tf && solution.stats.n_steps < options.max_steps {
        let step = h.min(tf - t);
        let half_step = step / 2.0;

        // k1 = f(t, y)
        f(t, &y, &mut k1);

        // k2 = f(t + h/2, y + h/2 * k1)
        for i in 0..n {
            y_temp[i] = y[i] + half_step * k1[i];
        }
        f(t + half_step, &y_temp, &mut k2);

        // k3 = f(t + h/2, y + h/2 * k2)
        for i in 0..n {
            y_temp[i] = y[i] + half_step * k2[i];
        }
        f(t + half_step, &y_temp, &mut k3);

        // k4 = f(t + h, y + h * k3)
        for i in 0..n {
            y_temp[i] = y[i] + step * k3[i];
        }
        f(t + step, &y_temp, &mut k4);

        solution.stats.n_evals += 4;

        // y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in 0..n {
            y[i] += step / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += step;

        solution.t.push(t);
        solution.y.push(y.clone());
        solution.stats.n_steps += 1;
    }

    solution.stats.solve_time = start_time.elapsed().as_secs_f64();
    solution
}

/// Dormand-Prince RK45 solver (5th order, adaptive step)
pub fn solve_rk45<F>(f: F, y0: &[f64], t_span: (f64, f64), options: &SolverOptions) -> ODESolution
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let start_time = std::time::Instant::now();
    let (t0, tf) = t_span;
    let n = y0.len();

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

    // Error coefficients (difference between 5th and 4th order)
    const E1: f64 = 71.0 / 57600.0;
    const E3: f64 = -71.0 / 16695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17253.0 / 339200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    let mut solution = ODESolution::new();
    let mut t = t0;
    let mut y = y0.to_vec();

    // Initial step size
    let mut h = if options.initial_step > 0.0 {
        options.initial_step
    } else {
        (tf - t0) / 100.0
    }
    .min(options.max_step);

    // Temporary vectors
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut k7 = vec![0.0; n];
    let mut y_temp = vec![0.0; n];
    let mut y_new = vec![0.0; n];

    solution.t.push(t);
    solution.y.push(y.clone());

    // k1 for first step
    f(t, &y, &mut k1);
    solution.stats.n_evals += 1;

    while t < tf && solution.stats.n_steps < options.max_steps {
        let step = h.min(tf - t);

        // Stage 2
        for i in 0..n {
            y_temp[i] = y[i] + step * A21 * k1[i];
        }
        f(t + step / 5.0, &y_temp, &mut k2);

        // Stage 3
        for i in 0..n {
            y_temp[i] = y[i] + step * (A31 * k1[i] + A32 * k2[i]);
        }
        f(t + 3.0 * step / 10.0, &y_temp, &mut k3);

        // Stage 4
        for i in 0..n {
            y_temp[i] = y[i] + step * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        f(t + 4.0 * step / 5.0, &y_temp, &mut k4);

        // Stage 5
        for i in 0..n {
            y_temp[i] = y[i] + step * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        f(t + 8.0 * step / 9.0, &y_temp, &mut k5);

        // Stage 6
        for i in 0..n {
            y_temp[i] =
                y[i] + step * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        f(t + step, &y_temp, &mut k6);

        // Stage 7 (5th order solution)
        for i in 0..n {
            y_new[i] =
                y[i] + step * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
        }
        f(t + step, &y_new, &mut k7);

        solution.stats.n_evals += 6;

        // Error estimate
        let mut err = 0.0;
        for i in 0..n {
            let sc = options.abstol + options.reltol * y[i].abs().max(y_new[i].abs());
            let ei = step
                * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            err += (ei / sc).powi(2);
        }
        err = (err / n as f64).sqrt();

        if err <= 1.0 {
            // Accept step
            t += step;
            y.clone_from(&y_new);
            k1.clone_from(&k7); // FSAL property

            solution.t.push(t);
            solution.y.push(y.clone());
            solution.stats.n_steps += 1;
        } else {
            solution.stats.n_rejected += 1;
        }

        // Step size control
        let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 5.0 };
        h = step * factor.clamp(0.2, 5.0);
        h = h.clamp(options.min_step, options.max_step);
    }

    solution.stats.solve_time = start_time.elapsed().as_secs_f64();
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let f = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        };

        let solution = solve_euler(f, &[1.0], (0.0, 1.0), &SolverOptions::default());

        let y_final = solution.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 0.1); // Euler has large error
    }

    #[test]
    fn test_rk4_exponential_decay() {
        let f = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        };

        let solution = solve_rk4(f, &[1.0], (0.0, 1.0), &SolverOptions::default());

        let y_final = solution.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 1e-4);
    }

    #[test]
    fn test_rk45_exponential_decay() {
        let f = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        };

        let options = SolverOptions::with_tolerances(1e-8, 1e-6);
        let solution = solve_rk45(f, &[1.0], (0.0, 1.0), &options);

        let y_final = solution.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 1e-6);
    }

    #[test]
    fn test_interpolation() {
        let f = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        };

        let solution = solve_rk4(f, &[1.0], (0.0, 1.0), &SolverOptions::default());

        // Interpolate at t = 0.5
        let y_half = solution.y_at(0.5).unwrap()[0];
        let expected = (-0.5_f64).exp();
        assert!((y_half - expected).abs() < 0.01);
    }

    #[test]
    fn test_lotka_volterra() {
        // Lotka-Volterra predator-prey model
        let alpha = 1.1;
        let beta = 0.4;
        let gamma = 0.4;
        let delta = 0.1;

        let f = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
            let prey = y[0];
            let pred = y[1];
            dydt[0] = alpha * prey - beta * prey * pred;
            dydt[1] = delta * prey * pred - gamma * pred;
        };

        let options = SolverOptions::with_tolerances(1e-6, 1e-4);
        let solution = solve_rk45(f, &[10.0, 5.0], (0.0, 50.0), &options);

        // Check solution exists and is reasonable
        assert!(solution.len() > 10);
        let final_state = solution.final_state().unwrap();
        assert!(final_state[0] > 0.0); // Prey should be positive
        assert!(final_state[1] > 0.0); // Predator should be positive
    }
}

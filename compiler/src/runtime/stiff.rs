//! Stiff ODE Solvers for Sounio
//!
//! This module implements specialized solvers for stiff ODEs:
//! - BDF (Backward Differentiation Formulas) up to order 5
//! - LSODA (automatic stiff/non-stiff switching)
//! - Implicit Euler
//! - Rosenbrock methods
//!
//! Stiff ODEs have widely varying time scales and require implicit methods.

use std::collections::VecDeque;

/// Newton iteration configuration
#[derive(Debug, Clone)]
pub struct NewtonConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub damping: f64,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 1e-10,
            damping: 1.0,
        }
    }
}

/// BDF solver configuration
#[derive(Debug, Clone)]
pub struct BDFConfig {
    pub max_order: usize, // 1-5
    pub initial_step: f64,
    pub min_step: f64,
    pub max_step: f64,
    pub rtol: f64,
    pub atol: f64,
    pub newton: NewtonConfig,
}

impl Default for BDFConfig {
    fn default() -> Self {
        Self {
            max_order: 5,
            initial_step: 1e-3,
            min_step: 1e-12,
            max_step: 1.0,
            rtol: 1e-6,
            atol: 1e-9,
            newton: NewtonConfig::default(),
        }
    }
}

/// BDF coefficients for orders 1-5
pub struct BDFCoefficients {
    /// Alpha coefficients for y values
    pub alpha: Vec<f64>,
    /// Beta coefficient for f(y_n+1)
    pub beta: f64,
    /// Error coefficient
    pub error_coef: f64,
}

impl BDFCoefficients {
    pub fn order(k: usize) -> Self {
        match k {
            1 => Self {
                // BDF1 = Backward Euler: y_n+1 - y_n = h*f(y_n+1)
                alpha: vec![1.0, -1.0],
                beta: 1.0,
                error_coef: 0.5,
            },
            2 => Self {
                // BDF2: (3/2)y_n+1 - 2y_n + (1/2)y_n-1 = h*f(y_n+1)
                alpha: vec![3.0 / 2.0, -2.0, 1.0 / 2.0],
                beta: 1.0,
                error_coef: 2.0 / 9.0,
            },
            3 => Self {
                // BDF3
                alpha: vec![11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0],
                beta: 1.0,
                error_coef: 3.0 / 22.0,
            },
            4 => Self {
                // BDF4
                alpha: vec![25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0],
                beta: 1.0,
                error_coef: 12.0 / 125.0,
            },
            5 => Self {
                // BDF5
                alpha: vec![137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
                beta: 1.0,
                error_coef: 10.0 / 137.0,
            },
            _ => panic!("BDF order must be 1-5"),
        }
    }
}

/// Solution from stiff solver
#[derive(Debug, Clone)]
pub struct StiffSolution {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub stats: SolverStats,
}

/// Solver statistics
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub steps: usize,
    pub rejected_steps: usize,
    pub function_evals: usize,
    pub jacobian_evals: usize,
    pub lu_decompositions: usize,
    pub newton_iterations: usize,
    pub order_changes: usize,
    pub stiffness_switches: usize, // For LSODA
}

/// BDF (Backward Differentiation Formula) solver
///
/// Implicit multi-step method excellent for stiff problems.
/// Automatically adjusts order (1-5) and step size.
pub struct BDFSolver<F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    f: F,
    jacobian: J,
    config: BDFConfig,
}

impl<F, J> BDFSolver<F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    pub fn new(f: F, jacobian: J) -> Self {
        Self {
            f,
            jacobian,
            config: BDFConfig::default(),
        }
    }

    pub fn with_config(mut self, config: BDFConfig) -> Self {
        self.config = config;
        self
    }

    /// Solve the ODE system from t0 to tf
    pub fn solve(&self, y0: &[f64], t0: f64, tf: f64) -> StiffSolution {
        let n = y0.len();
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut h = self.config.initial_step;
        let mut order = 1;

        // History for multi-step methods
        let mut history: VecDeque<(f64, Vec<f64>)> = VecDeque::new();
        history.push_back((t, y.clone()));

        let mut t_out = vec![t];
        let mut y_out = vec![y.clone()];
        let mut stats = SolverStats::default();

        while t < tf {
            // Adjust step to not overshoot
            if t + h > tf {
                h = tf - t;
            }

            // Determine usable order based on history
            let usable_order = order.min(history.len());

            // BDF step with Newton iteration
            match self.bdf_step(&y, t, h, usable_order, &history, &mut stats) {
                Ok((y_new, error)) => {
                    // Error control
                    let tol = self.compute_tolerance(&y_new);
                    let error_ratio = error / tol;

                    if error_ratio <= 1.0 {
                        // Accept step
                        t += h;
                        y = y_new;
                        stats.steps += 1;

                        // Update history
                        history.push_front((t, y.clone()));
                        if history.len() > 6 {
                            history.pop_back();
                        }

                        t_out.push(t);
                        y_out.push(y.clone());

                        // Adjust step size and order
                        let (new_h, new_order) =
                            self.adjust_step_order(h, order, error_ratio, &history);

                        if new_order != order {
                            stats.order_changes += 1;
                        }

                        h = new_h.clamp(self.config.min_step, self.config.max_step);
                        order = new_order.clamp(1, self.config.max_order);
                    } else {
                        // Reject step
                        stats.rejected_steps += 1;
                        h *= 0.5_f64.max(0.9 / error_ratio.powf(1.0 / (order as f64 + 1.0)));
                        h = h.max(self.config.min_step);

                        // Reduce order if struggling
                        if order > 1 {
                            order -= 1;
                            stats.order_changes += 1;
                        }
                    }
                }
                Err(_) => {
                    // Newton failed to converge, reduce step
                    stats.rejected_steps += 1;
                    h *= 0.25;
                    h = h.max(self.config.min_step);

                    if order > 1 {
                        order -= 1;
                        stats.order_changes += 1;
                    }
                }
            }
        }

        StiffSolution {
            t: t_out,
            y: y_out,
            stats,
        }
    }

    fn bdf_step(
        &self,
        _y: &[f64],
        t: f64,
        h: f64,
        order: usize,
        history: &VecDeque<(f64, Vec<f64>)>,
        stats: &mut SolverStats,
    ) -> Result<(Vec<f64>, f64), &'static str> {
        let n = history[0].1.len();
        let coef = BDFCoefficients::order(order);

        // Predictor: extrapolate from history
        let mut y_pred = vec![0.0; n];
        for i in 0..n {
            for (j, (_, y_hist)) in history.iter().enumerate().take(order) {
                if j == 0 {
                    y_pred[i] = y_hist[i];
                } else {
                    // Simple extrapolation
                    y_pred[i] += (y_hist[i] - history[j - 1].1[i]) * (1.0 - j as f64);
                }
            }
        }

        // Compute Jacobian at predicted point
        let t_new = t + h;
        let jac = (self.jacobian)(t_new, &y_pred);
        stats.jacobian_evals += 1;

        // Form iteration matrix: I - h*beta/alpha[0] * J
        let gamma = h * coef.beta / coef.alpha[0];
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = if i == j { 1.0 } else { 0.0 } - gamma * jac[i][j];
            }
        }

        // LU decomposition
        let lu = lu_decompose(&matrix)?;
        stats.lu_decompositions += 1;

        // Newton iteration
        let mut y_new = y_pred.clone();
        for _ in 0..self.config.newton.max_iterations {
            // Compute residual: alpha[0]*y_new - sum(alpha[j]*y_j) - h*beta*f(t_new, y_new)
            let f_new = (self.f)(t_new, &y_new);
            stats.function_evals += 1;

            let mut residual = vec![0.0; n];
            for i in 0..n {
                residual[i] = coef.alpha[0] * y_new[i];
                for (j, (_, y_hist)) in history.iter().enumerate().take(order) {
                    if j + 1 < coef.alpha.len() {
                        residual[i] += coef.alpha[j + 1] * y_hist[i];
                    }
                }
                residual[i] -= h * coef.beta * f_new[i];
            }

            // Solve for correction
            let delta = lu_solve(&lu, &residual)?;
            stats.newton_iterations += 1;

            // Apply damped correction
            let mut max_delta: f64 = 0.0;
            for i in 0..n {
                let d = self.config.newton.damping * delta[i];
                y_new[i] -= d;
                max_delta = max_delta.max(d.abs() / (y_new[i].abs() + self.config.atol));
            }

            if max_delta < self.config.newton.tolerance {
                // Converged - estimate error
                let error = self.estimate_error(&y_new, history, order, &coef);
                return Ok((y_new, error));
            }
        }

        Err("Newton iteration failed to converge")
    }

    fn estimate_error(
        &self,
        y_new: &[f64],
        history: &VecDeque<(f64, Vec<f64>)>,
        order: usize,
        coef: &BDFCoefficients,
    ) -> f64 {
        // Error estimate based on difference from lower order
        let mut error = 0.0;
        for i in 0..y_new.len() {
            let mut diff = y_new[i];
            for (j, (_, y_hist)) in history.iter().enumerate().take(order) {
                if j < coef.alpha.len() - 1 {
                    diff -= y_hist[i];
                }
            }
            let scale = y_new[i].abs() + self.config.atol;
            error += (diff / scale).powi(2);
        }
        (error / y_new.len() as f64).sqrt() * coef.error_coef
    }

    fn compute_tolerance(&self, y: &[f64]) -> f64 {
        let mut tol = 0.0;
        for &yi in y {
            tol += (self.config.rtol * yi.abs() + self.config.atol).powi(2);
        }
        (tol / y.len() as f64).sqrt()
    }

    fn adjust_step_order(
        &self,
        h: f64,
        order: usize,
        error_ratio: f64,
        history: &VecDeque<(f64, Vec<f64>)>,
    ) -> (f64, usize) {
        // Step size adjustment
        let safety = 0.9;
        let factor = safety / error_ratio.powf(1.0 / (order as f64 + 1.0));
        let new_h = h * factor.clamp(0.2, 5.0);

        // Order adjustment (simplified)
        let new_order =
            if error_ratio < 0.1 && order < self.config.max_order && history.len() > order {
                order + 1
            } else if error_ratio > 0.5 && order > 1 {
                order - 1
            } else {
                order
            };

        (new_h, new_order)
    }
}

/// LSODA-style solver with automatic stiff/non-stiff detection
///
/// Switches between Adams methods (non-stiff) and BDF (stiff).
pub struct LSODASolver<F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    f: F,
    jacobian: J,
    config: LSODAConfig,
}

/// LSODA configuration
#[derive(Debug, Clone)]
pub struct LSODAConfig {
    pub initial_step: f64,
    pub min_step: f64,
    pub max_step: f64,
    pub rtol: f64,
    pub atol: f64,
    pub max_adams_order: usize, // 1-12
    pub max_bdf_order: usize,   // 1-5
    pub stiffness_test_interval: usize,
}

impl Default for LSODAConfig {
    fn default() -> Self {
        Self {
            initial_step: 1e-3,
            min_step: 1e-12,
            max_step: 1.0,
            rtol: 1e-6,
            atol: 1e-9,
            max_adams_order: 12,
            max_bdf_order: 5,
            stiffness_test_interval: 10,
        }
    }
}

/// Method type for LSODA
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MethodType {
    Adams, // Non-stiff (explicit)
    BDF,   // Stiff (implicit)
}

impl<F, J> LSODASolver<F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    pub fn new(f: F, jacobian: J) -> Self {
        Self {
            f,
            jacobian,
            config: LSODAConfig::default(),
        }
    }

    pub fn with_config(mut self, config: LSODAConfig) -> Self {
        self.config = config;
        self
    }

    /// Solve with automatic stiff/non-stiff switching
    pub fn solve(&self, y0: &[f64], t0: f64, tf: f64) -> StiffSolution {
        let n = y0.len();
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut h = self.config.initial_step;

        let mut method = MethodType::Adams;
        let mut order = 1;
        let mut steps_since_switch = 0;

        let mut history: VecDeque<(f64, Vec<f64>, Vec<f64>)> = VecDeque::new();
        let f0 = (self.f)(t, &y);
        history.push_back((t, y.clone(), f0));

        let mut t_out = vec![t];
        let mut y_out = vec![y.clone()];
        let mut stats = SolverStats::default();

        while t < tf {
            if t + h > tf {
                h = tf - t;
            }

            // Perform step based on method type
            let step_result = match method {
                MethodType::Adams => self.adams_step(&y, t, h, order, &history, &mut stats),
                MethodType::BDF => self.bdf_step_lsoda(&y, t, h, order, &history, &mut stats),
            };

            match step_result {
                Ok((y_new, f_new, error)) => {
                    let tol = self.compute_tolerance(&y_new);
                    let error_ratio = error / tol;

                    if error_ratio <= 1.0 {
                        t += h;
                        y = y_new;
                        stats.steps += 1;
                        steps_since_switch += 1;

                        history.push_front((t, y.clone(), f_new));
                        if history.len() > 13 {
                            history.pop_back();
                        }

                        t_out.push(t);
                        y_out.push(y.clone());

                        // Stiffness detection
                        if steps_since_switch >= self.config.stiffness_test_interval {
                            let is_stiff = self.detect_stiffness(&y, t, h, &mut stats);

                            if is_stiff && method == MethodType::Adams {
                                method = MethodType::BDF;
                                order = 1;
                                steps_since_switch = 0;
                                stats.stiffness_switches += 1;
                            } else if !is_stiff && method == MethodType::BDF {
                                method = MethodType::Adams;
                                order = 1;
                                steps_since_switch = 0;
                                stats.stiffness_switches += 1;
                            }
                        }

                        // Adjust step and order
                        let max_order = match method {
                            MethodType::Adams => self.config.max_adams_order,
                            MethodType::BDF => self.config.max_bdf_order,
                        };

                        let (new_h, new_order) = self.adjust_step_order_lsoda(
                            h,
                            order,
                            error_ratio,
                            history.len(),
                            max_order,
                        );

                        if new_order != order {
                            stats.order_changes += 1;
                        }

                        h = new_h.clamp(self.config.min_step, self.config.max_step);
                        order = new_order;
                    } else {
                        stats.rejected_steps += 1;
                        h *= 0.5_f64.max(0.9 / error_ratio.powf(1.0 / (order as f64 + 1.0)));
                        h = h.max(self.config.min_step);
                    }
                }
                Err(_) => {
                    stats.rejected_steps += 1;
                    h *= 0.25;
                    h = h.max(self.config.min_step);

                    // Switch to BDF if Adams is failing
                    if method == MethodType::Adams {
                        method = MethodType::BDF;
                        order = 1;
                        steps_since_switch = 0;
                        stats.stiffness_switches += 1;
                    }
                }
            }
        }

        StiffSolution {
            t: t_out,
            y: y_out,
            stats,
        }
    }

    fn adams_step(
        &self,
        y: &[f64],
        t: f64,
        h: f64,
        order: usize,
        history: &VecDeque<(f64, Vec<f64>, Vec<f64>)>,
        stats: &mut SolverStats,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), &'static str> {
        let n = y.len();

        // Adams-Bashforth predictor coefficients (simplified for lower orders)
        let ab_coef: Vec<f64> = match order.min(history.len()) {
            1 => vec![1.0],
            2 => vec![3.0 / 2.0, -1.0 / 2.0],
            3 => vec![23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
            4 => vec![55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
            _ => vec![1.0],
        };

        // Predictor step
        let mut y_pred = y.to_vec();
        for i in 0..n {
            for (j, (_, _, f_hist)) in history.iter().enumerate().take(ab_coef.len()) {
                y_pred[i] += h * ab_coef[j] * f_hist[i];
            }
        }

        // Evaluate at predicted point
        let f_pred = (self.f)(t + h, &y_pred);
        stats.function_evals += 1;

        // Adams-Moulton corrector (one iteration)
        let am_coef: Vec<f64> = match order.min(history.len()) {
            1 => vec![1.0 / 2.0, 1.0 / 2.0],
            2 => vec![5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0],
            3 => vec![9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0],
            _ => vec![1.0 / 2.0, 1.0 / 2.0],
        };

        let mut y_new = y.to_vec();
        for i in 0..n {
            y_new[i] += h * am_coef[0] * f_pred[i];
            for (j, (_, _, f_hist)) in history.iter().enumerate().take(am_coef.len() - 1) {
                y_new[i] += h * am_coef[j + 1] * f_hist[i];
            }
        }

        let f_new = (self.f)(t + h, &y_new);
        stats.function_evals += 1;

        // Error estimate
        let mut error = 0.0;
        for i in 0..n {
            let diff = (y_new[i] - y_pred[i]).abs();
            let scale = y_new[i].abs() + self.config.atol;
            error += (diff / scale).powi(2);
        }
        error = (error / n as f64).sqrt();

        Ok((y_new, f_new, error))
    }

    fn bdf_step_lsoda(
        &self,
        _y: &[f64],
        t: f64,
        h: f64,
        order: usize,
        history: &VecDeque<(f64, Vec<f64>, Vec<f64>)>,
        stats: &mut SolverStats,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), &'static str> {
        let n = history[0].1.len();
        let usable_order = order.min(history.len()).min(5);
        let coef = BDFCoefficients::order(usable_order);

        // Predictor
        let y_pred = history[0].1.clone();

        // Jacobian
        let t_new = t + h;
        let jac = (self.jacobian)(t_new, &y_pred);
        stats.jacobian_evals += 1;

        // Form matrix
        let gamma = h * coef.beta / coef.alpha[0];
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = if i == j { 1.0 } else { 0.0 } - gamma * jac[i][j];
            }
        }

        let lu = lu_decompose(&matrix)?;
        stats.lu_decompositions += 1;

        // Newton iteration
        let mut y_new = y_pred.clone();
        for _ in 0..10 {
            let f_new = (self.f)(t_new, &y_new);
            stats.function_evals += 1;

            let mut residual = vec![0.0; n];
            for i in 0..n {
                residual[i] = coef.alpha[0] * y_new[i];
                for (j, (_, y_hist, _)) in history.iter().enumerate().take(usable_order) {
                    if j + 1 < coef.alpha.len() {
                        residual[i] += coef.alpha[j + 1] * y_hist[i];
                    }
                }
                residual[i] -= h * coef.beta * f_new[i];
            }

            let delta = lu_solve(&lu, &residual)?;
            stats.newton_iterations += 1;

            let mut max_delta: f64 = 0.0;
            for i in 0..n {
                y_new[i] -= delta[i];
                max_delta = max_delta.max(delta[i].abs() / (y_new[i].abs() + self.config.atol));
            }

            if max_delta < 1e-10 {
                let f_final = (self.f)(t_new, &y_new);
                stats.function_evals += 1;

                // Error estimate
                let mut error = 0.0;
                for i in 0..n {
                    let diff = (y_new[i] - y_pred[i]).abs();
                    let scale = y_new[i].abs() + self.config.atol;
                    error += (diff / scale).powi(2);
                }
                error = (error / n as f64).sqrt() * coef.error_coef;

                return Ok((y_new, f_final, error));
            }
        }

        Err("Newton failed to converge")
    }

    fn detect_stiffness(&self, y: &[f64], t: f64, h: f64, stats: &mut SolverStats) -> bool {
        // Estimate dominant eigenvalue of Jacobian
        let jac = (self.jacobian)(t, y);
        stats.jacobian_evals += 1;

        // Approximate spectral radius using power iteration (1 step)
        let n = y.len();
        let v = vec![1.0 / (n as f64).sqrt(); n];

        // One matrix-vector multiply
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += jac[i][j] * v[j];
            }
        }

        let eigenvalue_estimate: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();

        // System is stiff if h * |lambda| >> 1
        eigenvalue_estimate * h > 5.0
    }

    fn compute_tolerance(&self, y: &[f64]) -> f64 {
        let mut tol = 0.0;
        for &yi in y {
            tol += (self.config.rtol * yi.abs() + self.config.atol).powi(2);
        }
        (tol / y.len() as f64).sqrt()
    }

    fn adjust_step_order_lsoda(
        &self,
        h: f64,
        order: usize,
        error_ratio: f64,
        history_len: usize,
        max_order: usize,
    ) -> (f64, usize) {
        let safety = 0.9;
        let factor = safety / error_ratio.powf(1.0 / (order as f64 + 1.0));
        let new_h = h * factor.clamp(0.2, 5.0);

        let new_order = if error_ratio < 0.1 && order < max_order && history_len > order {
            order + 1
        } else if error_ratio > 0.5 && order > 1 {
            order - 1
        } else {
            order
        };

        (new_h, new_order)
    }
}

/// Implicit Euler solver (simplest stiff solver)
pub fn implicit_euler<F, J>(
    f: F,
    jacobian: J,
    y0: &[f64],
    t_span: (f64, f64),
    dt: f64,
) -> StiffSolution
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let n = y0.len();
    let mut t = t_span.0;
    let mut y = y0.to_vec();

    let mut t_out = vec![t];
    let mut y_out = vec![y.clone()];
    let mut stats = SolverStats::default();

    while t < t_span.1 {
        let h = dt.min(t_span.1 - t);
        let t_new = t + h;

        // Newton iteration for y_new = y + h * f(t_new, y_new)
        let mut y_new = y.clone();

        for _ in 0..20 {
            let f_val = f(t_new, &y_new);
            stats.function_evals += 1;

            let jac = jacobian(t_new, &y_new);
            stats.jacobian_evals += 1;

            // Residual: y_new - y - h*f(t_new, y_new)
            let mut residual = vec![0.0; n];
            for i in 0..n {
                residual[i] = y_new[i] - y[i] - h * f_val[i];
            }

            // Matrix: I - h*J
            let mut matrix = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    matrix[i][j] = if i == j { 1.0 } else { 0.0 } - h * jac[i][j];
                }
            }

            if let Ok(lu) = lu_decompose(&matrix) {
                stats.lu_decompositions += 1;
                if let Ok(delta) = lu_solve(&lu, &residual) {
                    stats.newton_iterations += 1;

                    let mut max_delta: f64 = 0.0;
                    for i in 0..n {
                        y_new[i] -= delta[i];
                        max_delta = max_delta.max(delta[i].abs());
                    }

                    if max_delta < 1e-12 {
                        break;
                    }
                }
            }
        }

        t = t_new;
        y = y_new;
        stats.steps += 1;

        t_out.push(t);
        y_out.push(y.clone());
    }

    StiffSolution {
        t: t_out,
        y: y_out,
        stats,
    }
}

/// Rosenbrock method (linearly implicit)
///
/// Good for moderately stiff problems, only requires one Jacobian per step.
/// Rosenbrock method (linearly implicit Euler variant)
///
/// Simplified L-stable method for stiff problems.
pub fn rosenbrock<F, J>(f: F, jacobian: J, y0: &[f64], t_span: (f64, f64), dt: f64) -> StiffSolution
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let n = y0.len();
    let mut t = t_span.0;
    let mut y = y0.to_vec();

    let mut t_out = vec![t];
    let mut y_out = vec![y.clone()];
    let mut stats = SolverStats::default();

    // Rosenbrock-Euler (ROS1): simplest L-stable Rosenbrock method
    // Solves: (I - h*J) * k = h * f(y)
    // Updates: y_new = y + k

    while t < t_span.1 {
        let h = dt.min(t_span.1 - t);

        let f0 = f(t, &y);
        let jac = jacobian(t, &y);
        stats.function_evals += 1;
        stats.jacobian_evals += 1;

        // Form (I - h*J)
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = if i == j { 1.0 } else { 0.0 } - h * jac[i][j];
            }
        }

        // RHS = h * f(y)
        let rhs: Vec<f64> = f0.iter().map(|&fi| h * fi).collect();

        if let Ok(lu) = lu_decompose(&matrix) {
            stats.lu_decompositions += 1;

            if let Ok(k) = lu_solve(&lu, &rhs) {
                // Update: y_new = y + k
                for i in 0..n {
                    y[i] += k[i];
                }
            }
        }

        t += h;
        stats.steps += 1;
        t_out.push(t);
        y_out.push(y.clone());
    }

    StiffSolution {
        t: t_out,
        y: y_out,
        stats,
    }
}

// ============================================================================
// Linear Algebra Utilities
// ============================================================================

/// LU decomposition result
struct LU {
    matrix: Vec<Vec<f64>>,
    pivot: Vec<usize>,
}

/// LU decomposition with partial pivoting
fn lu_decompose(a: &[Vec<f64>]) -> Result<LU, &'static str> {
    let n = a.len();
    let mut lu: Vec<Vec<f64>> = a.to_vec();
    let mut pivot: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = lu[k][k].abs();
        let mut max_idx = k;
        for i in (k + 1)..n {
            if lu[i][k].abs() > max_val {
                max_val = lu[i][k].abs();
                max_idx = i;
            }
        }

        if max_val < 1e-15 {
            return Err("Singular matrix");
        }

        // Swap rows
        if max_idx != k {
            lu.swap(k, max_idx);
            pivot.swap(k, max_idx);
        }

        // Elimination
        for i in (k + 1)..n {
            lu[i][k] /= lu[k][k];
            for j in (k + 1)..n {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }

    Ok(LU { matrix: lu, pivot })
}

/// Solve LU * x = b
fn lu_solve(lu: &LU, b: &[f64]) -> Result<Vec<f64>, &'static str> {
    let n = b.len();
    let mut x: Vec<f64> = lu.pivot.iter().map(|&i| b[i]).collect();

    // Forward substitution
    for i in 0..n {
        for j in 0..i {
            x[i] -= lu.matrix[i][j] * x[j];
        }
    }

    // Backward substitution
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu.matrix[i][j] * x[j];
        }
        if lu.matrix[i][i].abs() < 1e-15 {
            return Err("Singular matrix");
        }
        x[i] /= lu.matrix[i][i];
    }

    Ok(x)
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Solve a stiff ODE using BDF with automatic Jacobian
pub fn solve_stiff<F>(f: F, y0: &[f64], t_span: (f64, f64), rtol: f64, atol: f64) -> StiffSolution
where
    F: Fn(f64, &[f64]) -> Vec<f64> + Copy,
{
    let n = y0.len();

    // Numerical Jacobian using copy of f
    let jacobian = move |t: f64, y: &[f64]| -> Vec<Vec<f64>> {
        let eps = 1e-8;
        let f0 = f(t, y);
        let mut jac = vec![vec![0.0; n]; n];
        let mut y_pert = y.to_vec();

        for j in 0..n {
            let dy = eps * (1.0 + y[j].abs());
            y_pert[j] = y[j] + dy;
            let f1 = f(t, &y_pert);
            for i in 0..n {
                jac[i][j] = (f1[i] - f0[i]) / dy;
            }
            y_pert[j] = y[j];
        }
        jac
    };

    let config = BDFConfig {
        rtol,
        atol,
        ..Default::default()
    };

    let solver = BDFSolver::new(f, jacobian).with_config(config);
    solver.solve(y0, t_span.0, t_span.1)
}

/// Solve using LSODA with automatic method switching
pub fn solve_lsoda<F>(f: F, y0: &[f64], t_span: (f64, f64), rtol: f64, atol: f64) -> StiffSolution
where
    F: Fn(f64, &[f64]) -> Vec<f64> + Copy,
{
    let n = y0.len();

    let jacobian = move |t: f64, y: &[f64]| -> Vec<Vec<f64>> {
        let eps = 1e-8;
        let f0 = f(t, y);
        let mut jac = vec![vec![0.0; n]; n];
        let mut y_pert = y.to_vec();

        for j in 0..n {
            let dy = eps * (1.0 + y[j].abs());
            y_pert[j] = y[j] + dy;
            let f1 = f(t, &y_pert);
            for i in 0..n {
                jac[i][j] = (f1[i] - f0[i]) / dy;
            }
            y_pert[j] = y[j];
        }
        jac
    };

    let config = LSODAConfig {
        rtol,
        atol,
        ..Default::default()
    };

    let solver = LSODASolver::new(f, jacobian).with_config(config);
    solver.solve(y0, t_span.0, t_span.1)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_euler_simple() {
        // dy/dt = -y, y(0) = 1, solution: y = e^(-t)
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let j = |_t: f64, _y: &[f64]| vec![vec![-1.0]];

        let sol = implicit_euler(f, j, &[1.0], (0.0, 1.0), 0.01);

        let y_final = sol.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();

        assert!((y_final - expected).abs() < 0.01);
    }

    #[test]
    fn test_bdf_stiff_system() {
        // Robertson problem (classic stiff test)
        // dy1/dt = -0.04*y1 + 1e4*y2*y3
        // dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
        // dy3/dt = 3e7*y2^2

        let f = |_t: f64, y: &[f64]| {
            vec![
                -0.04 * y[0] + 1e4 * y[1] * y[2],
                0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                3e7 * y[1] * y[1],
            ]
        };

        let j = |_t: f64, y: &[f64]| {
            vec![
                vec![-0.04, 1e4 * y[2], 1e4 * y[1]],
                vec![0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
                vec![0.0, 6e7 * y[1], 0.0],
            ]
        };

        let config = BDFConfig {
            rtol: 1e-4,
            atol: 1e-8,
            ..Default::default()
        };

        let solver = BDFSolver::new(f, j).with_config(config);
        let sol = solver.solve(&[1.0, 0.0, 0.0], 0.0, 0.1);

        // Check conservation (y1 + y2 + y3 = 1)
        let y_final = sol.y.last().unwrap();
        let sum: f64 = y_final.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_rosenbrock() {
        // Simple decay - test that solver runs and produces decreasing values
        let f = |_t: f64, y: &[f64]| vec![-2.0 * y[0]];
        let j = |_t: f64, _y: &[f64]| vec![vec![-2.0]];

        let sol = rosenbrock(f, j, &[1.0], (0.0, 1.0), 0.05);

        // Check solver completed
        assert!(sol.t.len() > 1);
        assert!(sol.stats.steps > 0);

        // Solution should decay monotonically
        let y_final = sol.y.last().unwrap()[0];
        assert!(y_final < 1.0, "Solution should decay");
        assert!(y_final > 0.0, "Solution should stay positive");
    }

    #[test]
    fn test_lsoda_switching() {
        // System that transitions from non-stiff to stiff
        let f = |t: f64, y: &[f64]| {
            let k = if t < 0.5 { 1.0 } else { 100.0 };
            vec![-k * y[0]]
        };

        let j = |t: f64, _y: &[f64]| {
            let k = if t < 0.5 { 1.0 } else { 100.0 };
            vec![vec![-k]]
        };

        let solver = LSODASolver::new(f, j);
        let sol = solver.solve(&[1.0], 0.0, 1.0);

        // Should complete without error
        assert!(sol.t.last().unwrap() >= &1.0);
    }

    #[test]
    fn test_solve_stiff_convenience() {
        let f = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];

        let sol = solve_stiff(f, &[1.0], (0.0, 0.2), 1e-4, 1e-8);

        let y_final = sol.y.last().unwrap()[0];
        let expected = (-10.0_f64).exp(); // e^(-50*0.2) = e^(-10)

        assert!((y_final - expected).abs() < 1e-3);
    }

    #[test]
    fn test_lu_decomposition() {
        let a = vec![
            vec![2.0, 1.0, 1.0],
            vec![4.0, 3.0, 3.0],
            vec![8.0, 7.0, 9.0],
        ];

        let lu = lu_decompose(&a).unwrap();
        let b = vec![4.0, 10.0, 24.0];
        let x = lu_solve(&lu, &b).unwrap();

        // x should be [1, 1, 1]
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bdf_coefficients() {
        let c1 = BDFCoefficients::order(1);
        assert_eq!(c1.alpha.len(), 2);

        let c5 = BDFCoefficients::order(5);
        assert_eq!(c5.alpha.len(), 6);
    }
}

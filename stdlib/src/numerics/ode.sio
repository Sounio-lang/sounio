//! Ordinary Differential Equation solvers

use linalg::{Vector, Matrix}

/// ODE system: dy/dt = f(t, y)
pub trait ODESystem {
    /// Evaluate the right-hand side
    fn eval(&self, t: f64, y: &Vector<f64>, dydt: &!Vector<f64>);

    /// Dimension of the system
    fn dim(&self) -> usize;

    /// Optional Jacobian: df/dy
    fn jacobian(&self, t: f64, y: &Vector<f64>) -> Option<Matrix<f64>> {
        None
    }
}

/// ODE solver configuration
pub struct ODEConfig {
    /// Absolute tolerance
    pub atol: f64,

    /// Relative tolerance
    pub rtol: f64,

    /// Maximum step size
    pub max_step: f64,

    /// Minimum step size
    pub min_step: f64,

    /// Maximum number of steps
    pub max_steps: usize,

    /// Initial step size (0 = auto)
    pub initial_step: f64,
}

impl Default for ODEConfig {
    fn default() -> Self {
        ODEConfig {
            atol: 1e-6,
            rtol: 1e-3,
            max_step: f64::INFINITY,
            min_step: 1e-12,
            max_steps: 100000,
            initial_step: 0.0,
        }
    }
}

/// ODE solution trajectory
pub struct ODESolution {
    /// Time points
    pub t: Vector<f64>,

    /// Solution at each time point (rows = time, cols = state)
    pub y: Matrix<f64>,

    /// Number of function evaluations
    pub nfev: usize,

    /// Number of Jacobian evaluations
    pub njev: usize,

    /// Whether solution terminated successfully
    pub success: bool,

    /// Termination message
    pub message: string,
}

/// Runge-Kutta-Fehlberg 4(5) method (adaptive step size)
pub struct RKF45<S: ODESystem> {
    system: S,
    config: ODEConfig,
}

impl<S: ODESystem> RKF45<S> {
    pub fn new(system: S, config: ODEConfig) -> Self {
        RKF45 { system, config }
    }

    /// Integrate from t0 to tf
    pub fn integrate(
        &self,
        y0: &Vector<f64>,
        t0: f64,
        tf: f64,
    ) -> ODESolution with Alloc {
        let mut t = t0;
        let mut y = y0.clone();
        let n = self.system.dim();

        // Storage for trajectory
        let mut t_hist = Vec::new();
        let mut y_hist = Vec::new();

        t_hist.push(t);
        y_hist.push(y.clone());

        // Initial step size
        let mut h = if self.config.initial_step > 0.0 {
            self.config.initial_step
        } else {
            self.estimate_initial_step(&y, t0, tf)
        };

        // RKF45 coefficients (Butcher tableau)
        let a2 = 1.0/4.0;
        let a3 = 3.0/8.0;
        let a4 = 12.0/13.0;
        let a5 = 1.0;
        let a6 = 1.0/2.0;

        let b21 = 1.0/4.0;
        let b31 = 3.0/32.0;    let b32 = 9.0/32.0;
        let b41 = 1932.0/2197.0; let b42 = -7200.0/2197.0; let b43 = 7296.0/2197.0;
        let b51 = 439.0/216.0; let b52 = -8.0;          let b53 = 3680.0/513.0;   let b54 = -845.0/4104.0;
        let b61 = -8.0/27.0;   let b62 = 2.0;           let b63 = -3544.0/2565.0; let b64 = 1859.0/4104.0; let b65 = -11.0/40.0;

        // 4th order coefficients
        let c1 = 25.0/216.0;   let c3 = 1408.0/2565.0;  let c4 = 2197.0/4104.0;  let c5 = -1.0/5.0;

        // 5th order coefficients (for error estimation)
        let d1 = 16.0/135.0;   let d3 = 6656.0/12825.0; let d4 = 28561.0/56430.0; let d5 = -9.0/50.0; let d6 = 2.0/55.0;

        // Temporary vectors
        let mut k1 = Vector::new(n);
        let mut k2 = Vector::new(n);
        let mut k3 = Vector::new(n);
        let mut k4 = Vector::new(n);
        let mut k5 = Vector::new(n);
        let mut k6 = Vector::new(n);
        let mut y_temp = Vector::new(n);
        let mut y4 = Vector::new(n);
        let mut y5 = Vector::new(n);

        let mut nfev = 0;
        let mut step_count = 0;

        while t < tf && step_count < self.config.max_steps {
            // Ensure we don't overshoot
            if t + h > tf {
                h = tf - t;
            }

            // k1 = f(t, y)
            self.system.eval(t, &y, &!k1);
            nfev += 1;

            // k2 = f(t + a2*h, y + h*b21*k1)
            for i in 0..n {
                y_temp[i] = y[i] + h * b21 * k1[i];
            }
            self.system.eval(t + a2*h, &y_temp, &!k2);
            nfev += 1;

            // k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
            for i in 0..n {
                y_temp[i] = y[i] + h * (b31*k1[i] + b32*k2[i]);
            }
            self.system.eval(t + a3*h, &y_temp, &!k3);
            nfev += 1;

            // k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
            for i in 0..n {
                y_temp[i] = y[i] + h * (b41*k1[i] + b42*k2[i] + b43*k3[i]);
            }
            self.system.eval(t + a4*h, &y_temp, &!k4);
            nfev += 1;

            // k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
            for i in 0..n {
                y_temp[i] = y[i] + h * (b51*k1[i] + b52*k2[i] + b53*k3[i] + b54*k4[i]);
            }
            self.system.eval(t + a5*h, &y_temp, &!k5);
            nfev += 1;

            // k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
            for i in 0..n {
                y_temp[i] = y[i] + h * (b61*k1[i] + b62*k2[i] + b63*k3[i] + b64*k4[i] + b65*k5[i]);
            }
            self.system.eval(t + a6*h, &y_temp, &!k6);
            nfev += 1;

            // 4th order solution
            for i in 0..n {
                y4[i] = y[i] + h * (c1*k1[i] + c3*k3[i] + c4*k4[i] + c5*k5[i]);
            }

            // 5th order solution
            for i in 0..n {
                y5[i] = y[i] + h * (d1*k1[i] + d3*k3[i] + d4*k4[i] + d5*k5[i] + d6*k6[i]);
            }

            // Error estimate
            let mut err = 0.0;
            for i in 0..n {
                let scale = self.config.atol + self.config.rtol * y[i].abs().max(y5[i].abs());
                let e = (y5[i] - y4[i]).abs() / scale;
                err = err.max(e);
            }

            // Accept or reject step
            if err <= 1.0 {
                // Accept step
                t += h;
                y = y5.clone();

                t_hist.push(t);
                y_hist.push(y.clone());
                step_count += 1;
            }

            // Adjust step size
            let safety = 0.9;
            let p_grow = -0.2;
            let p_shrink = -0.25;

            if err > 0.0 {
                let factor = if err <= 1.0 {
                    safety * err.powf(p_grow)
                } else {
                    safety * err.powf(p_shrink)
                };
                h *= factor.clamp(0.1, 5.0);
            }

            h = h.clamp(self.config.min_step, self.config.max_step);
        }

        // Build solution
        let nt = t_hist.len();
        let mut t_vec = Vector::new(nt);
        let mut y_mat = Matrix::new(nt, n);

        for i in 0..nt {
            t_vec[i] = t_hist[i];
            for j in 0..n {
                y_mat[(i, j)] = y_hist[i][j];
            }
        }

        ODESolution {
            t: t_vec,
            y: y_mat,
            nfev,
            njev: 0,
            success: t >= tf - 1e-10,
            message: if step_count >= self.config.max_steps {
                "maximum steps exceeded".to_string()
            } else {
                "success".to_string()
            },
        }
    }

    fn estimate_initial_step(&self, y0: &Vector<f64>, t0: f64, tf: f64) -> f64 {
        let n = self.system.dim();
        let mut f0 = Vector::new(n);
        self.system.eval(t0, y0, &!f0);

        let d0 = y0.norm();
        let d1 = f0.norm();

        let h0 = if d0 < 1e-5 || d1 < 1e-5 {
            1e-6
        } else {
            0.01 * d0 / d1
        };

        h0.min((tf - t0) / 10.0)
    }
}

/// Backward Differentiation Formula (BDF) for stiff ODEs
pub struct BDF<S: ODESystem> {
    system: S,
    config: ODEConfig,
    order: usize,
}

impl<S: ODESystem> BDF<S> {
    pub fn new(system: S, config: ODEConfig) -> Self {
        BDF { system, config, order: 5 }
    }

    /// Integrate stiff ODE
    pub fn integrate(
        &self,
        y0: &Vector<f64>,
        t0: f64,
        tf: f64,
    ) -> ODESolution with Alloc {
        // BDF coefficients for orders 1-5
        let bdf_coeffs = [
            vec![1.0, -1.0],                                    // BDF1 (backward Euler)
            vec![3.0/2.0, -2.0, 1.0/2.0],                       // BDF2
            vec![11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0],            // BDF3
            vec![25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0],      // BDF4
            vec![137.0/60.0, -5.0, 5.0, -10.0/3.0, 5.0/4.0, -1.0/5.0], // BDF5
        ];

        let n = self.system.dim();
        let mut t = t0;
        let mut y = y0.clone();

        // History for multistep methods
        let mut y_history: Vec<Vector<f64>> = vec![y.clone()];
        let mut t_history: Vec<f64> = vec![t];

        // Storage for trajectory
        let mut t_out = Vec::new();
        let mut y_out = Vec::new();

        t_out.push(t);
        y_out.push(y.clone());

        let mut h = self.config.initial_step.max(1e-6);
        let mut nfev = 0;
        let mut njev = 0;
        let mut current_order = 1;

        while t < tf {
            if t + h > tf {
                h = tf - t;
            }

            // Predictor (extrapolation)
            let mut y_pred = y.clone();

            // Corrector (Newton iteration)
            let coeffs = &bdf_coeffs[current_order - 1];
            let beta = h / coeffs[0];

            // Newton iteration for implicit equation
            let max_newton = 10;
            let newton_tol = 1e-10;

            for _ in 0..max_newton {
                let mut f = Vector::new(n);
                self.system.eval(t + h, &y_pred, &!f);
                nfev += 1;

                // Residual: y_new - beta*f - sum(coeffs[i]*y_history[i])
                let mut residual = Vector::new(n);
                for i in 0..n {
                    residual[i] = y_pred[i] - beta * f[i];
                    for (j, &c) in coeffs[1..].iter().enumerate() {
                        if j < y_history.len() {
                            let idx = y_history.len() - 1 - j;
                            residual[i] += c * y_history[idx][i];
                        }
                    }
                }

                // Check convergence
                if residual.norm() < newton_tol {
                    break;
                }

                // Newton update (simplified, would use Jacobian)
                if let Some(jac) = self.system.jacobian(t + h, &y_pred) {
                    njev += 1;
                    // Solve (I - beta*J) * delta = residual
                    let mut a = Matrix::eye(n);
                    for i in 0..n {
                        for j in 0..n {
                            a[(i, j)] -= beta * jac[(i, j)];
                        }
                    }
                    if let Ok(delta) = linalg::solve(&a, &residual) {
                        for i in 0..n {
                            y_pred[i] -= delta[i];
                        }
                    }
                } else {
                    // Simplified Newton without Jacobian
                    for i in 0..n {
                        y_pred[i] -= 0.5 * residual[i];
                    }
                }
            }

            // Accept step
            t += h;
            y = y_pred;

            // Update history
            y_history.push(y.clone());
            t_history.push(t);

            // Keep limited history
            if y_history.len() > 6 {
                y_history.remove(0);
                t_history.remove(0);
            }

            t_out.push(t);
            y_out.push(y.clone());

            // Increase order gradually
            if current_order < self.order && y_history.len() > current_order {
                current_order += 1;
            }
        }

        // Build solution
        let nt = t_out.len();
        let mut t_vec = Vector::new(nt);
        let mut y_mat = Matrix::new(nt, n);

        for i in 0..nt {
            t_vec[i] = t_out[i];
            for j in 0..n {
                y_mat[(i, j)] = y_out[i][j];
            }
        }

        ODESolution {
            t: t_vec,
            y: y_mat,
            nfev,
            njev,
            success: true,
            message: "success".to_string(),
        }
    }
}

/// Convenience function to integrate ODE
pub fn odeint<S: ODESystem>(
    system: S,
    y0: &Vector<f64>,
    t_span: (f64, f64),
    method: &str,
) -> ODESolution with Alloc {
    let config = ODEConfig::default();

    match method {
        "RK45" | "rk45" => {
            let solver = RKF45::new(system, config);
            solver.integrate(y0, t_span.0, t_span.1)
        }
        "BDF" | "bdf" => {
            let solver = BDF::new(system, config);
            solver.integrate(y0, t_span.0, t_span.1)
        }
        _ => panic!("unknown method: {}", method),
    }
}

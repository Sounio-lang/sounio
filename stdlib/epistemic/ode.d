//! stdlib/epistemic/ode.d
//!
//! Epistemic ODE Solvers — Demetrios v0.79
//!
//! Every integration step propagates uncertainty.
//! Every parameter carries its variance through time.
//! Every solution knows its numerical error bounds.
//!
//! # Philosophy
//!
//! Traditional ODE solvers treat initial conditions and parameters as exact.
//! In reality:
//! - Initial conditions come from measurements with noise
//! - Parameters are estimated with uncertainty
//! - Numerical integration introduces truncation error
//! - Chaotic systems amplify small uncertainties exponentially
//!
//! Demetrios ODE solvers make this EXPLICIT by propagating covariance
//! matrices alongside the state trajectory.
//!
//! # Quick Example
//!
//! ```demetrios
//! use epistemic::ode::{ODESystem, RK45, solve_ivp_uncertain}
//! use epistemic::knowledge::Knowledge
//!
//! // Exponential decay with uncertain rate
//! let k = Knowledge::measured(0.5, 0.05, "decay_rate")
//!
//! let system = ODESystem::new(
//!     |t, y, k| [-k * y[0]],
//!     k.value,
//! )
//!
//! let y0 = [Knowledge::measured(1.0, 0.01, "initial")]
//! let result = solve_ivp_uncertain(system, (0.0, 10.0), y0, RK45::default())
//!
//! println("Final: {} ± {}", result.y_final.y[0], result.y_final.std(0))
//! println("Uncertainty growth: {}x", result.error_assessment.uncertainty_growth_rate)
//! ```

module epistemic::ode

use core::prelude::*
use core::option::{Option, Some, None}
use core::result::{Result, Ok, Err}
use epistemic::knowledge::{Knowledge, BetaConfidence, Source, Provenance}
use epistemic::linalg::{EVector, EMatrix}
use units::{h, s, dimensionless}

// ============================================================================
// ODE SYSTEM DEFINITION
// ============================================================================

/// Right-hand side function type: f(t, y, params) -> dy/dt
pub type RHSFn<const N: usize, P> = fn(f64, [f64; N], P) -> [f64; N]

/// Jacobian function type: J(t, y, params) -> ∂f/∂y
pub type JacobianFn<const N: usize, P> = fn(f64, [f64; N], P) -> [[f64; N]; N]

/// Parameter sensitivity function type: ∂f/∂p
pub type SensitivityFn<const N: usize, const P_DIM: usize, P> = fn(f64, [f64; N], P) -> [[f64; N]; P_DIM]

/// ODE system with optional Jacobian and sensitivity information
pub struct ODESystem<const N: usize, P> {
    /// Right-hand side function dy/dt = f(t, y, p)
    pub rhs: RHSFn<N, P>,
    /// Parameters for the system
    pub params: P,
    /// Optional analytical Jacobian (if None, uses finite differences)
    pub jacobian: Option<JacobianFn<N, P>>,
    /// Optional mass matrix for DAEs: M dy/dt = f(t, y)
    pub mass: Option<[[f64; N]; N]>,
    /// Whether the system is stiff
    pub is_stiff: bool,
}

impl<const N: usize, P> ODESystem<N, P> {
    /// Create a new ODE system with just the RHS
    pub fn new(rhs: RHSFn<N, P>, params: P) -> Self {
        ODESystem {
            rhs,
            params,
            jacobian: None,
            mass: None,
            is_stiff: false,
        }
    }

    /// Add analytical Jacobian for improved accuracy
    pub fn with_jacobian(mut self, jac: JacobianFn<N, P>) -> Self {
        self.jacobian = Some(jac)
        self
    }

    /// Mark system as stiff (will prefer implicit methods)
    pub fn stiff(mut self) -> Self {
        self.is_stiff = true
        self
    }

    /// Add mass matrix for differential-algebraic equations
    pub fn with_mass(mut self, mass: [[f64; N]; N]) -> Self {
        self.mass = Some(mass)
        self
    }

    /// Evaluate the RHS at given point
    pub fn eval(self, t: f64, y: [f64; N]) -> [f64; N] {
        (self.rhs)(t, y, self.params)
    }

    /// Compute Jacobian (analytical or finite difference)
    pub fn compute_jacobian(self, t: f64, y: [f64; N]) -> [[f64; N]; N] {
        match self.jacobian {
            Some(jac) => jac(t, y, self.params),
            None => finite_difference_jacobian(self.rhs, t, y, self.params),
        }
    }
}

// ============================================================================
// EPISTEMIC STATE
// ============================================================================

/// State vector with full covariance matrix for uncertainty tracking
pub struct EState<const N: usize> {
    /// Mean state values
    pub y: [f64; N],
    /// Covariance matrix (uncertainty)
    pub covariance: [[f64; N]; N],
    /// Current time
    pub t: f64,
    /// Local truncation error estimate
    pub local_error: [f64; N],
    /// Provenance tracking
    pub provenance: Provenance,
}

impl<const N: usize> EState<N> {
    /// Create state with exact (zero variance) initial conditions
    pub fn exact(y0: [f64; N], t0: f64) -> Self with Alloc {
        EState {
            y: y0,
            covariance: [[0.0; N]; N],
            t: t0,
            local_error: [0.0; N],
            provenance: Provenance::new("initial_condition"),
        }
    }

    /// Create state with uncertain initial conditions
    pub fn uncertain(y0: [Knowledge<f64>; N], t0: f64) -> Self with Alloc {
        let mut y = [0.0; N]
        let mut cov = [[0.0; N]; N]
        let mut prov = Provenance::new("uncertain_ic")

        for i in 0..N {
            y[i] = y0[i].value
            cov[i][i] = y0[i].variance()
            prov = prov.merge(y0[i].provenance)
        }

        EState {
            y,
            covariance: cov,
            t: t0,
            local_error: [0.0; N],
            provenance: prov,
        }
    }

    /// Create state with full covariance specification
    pub fn with_covariance(y0: [f64; N], cov: [[f64; N]; N], t0: f64) -> Self with Alloc {
        EState {
            y: y0,
            covariance: cov,
            t: t0,
            local_error: [0.0; N],
            provenance: Provenance::new("initial_with_covariance"),
        }
    }

    /// Get element as Knowledge value
    pub fn get(self, i: usize) -> Knowledge<f64> {
        Knowledge::from_gaussian(
            self.y[i],
            self.covariance[i][i],
            self.provenance.clone(),
        )
    }

    /// Get standard deviation of element
    pub fn std(self, i: usize) -> f64 {
        sqrt_f64(self.covariance[i][i])
    }

    /// Total uncertainty (trace of covariance)
    pub fn total_uncertainty(self) -> f64 {
        let mut trace = 0.0
        for i in 0..N {
            trace += self.covariance[i][i]
        }
        trace
    }

    /// Maximum variance across all states
    pub fn max_variance(self) -> f64 {
        let mut max_val = 0.0
        for i in 0..N {
            if self.covariance[i][i] > max_val {
                max_val = self.covariance[i][i]
            }
        }
        max_val
    }

    /// Clone the state
    pub fn clone(self) -> Self with Alloc {
        EState {
            y: self.y,
            covariance: self.covariance,
            t: self.t,
            local_error: self.local_error,
            provenance: self.provenance.clone(),
        }
    }

    /// Correlation matrix
    pub fn correlation(self) -> [[f64; N]; N] {
        let mut corr = [[0.0; N]; N]
        for i in 0..N {
            for j in 0..N {
                let si = sqrt_f64(self.covariance[i][i])
                let sj = sqrt_f64(self.covariance[j][j])
                if si > 1e-15 && sj > 1e-15 {
                    corr[i][j] = self.covariance[i][j] / (si * sj)
                } else {
                    corr[i][j] = if i == j { 1.0 } else { 0.0 }
                }
            }
        }
        corr
    }
}

// ============================================================================
// SOLVER OPTIONS
// ============================================================================

/// Configuration options for ODE solvers
pub struct SolverOptions {
    /// Relative tolerance for error control
    pub rtol: f64,
    /// Absolute tolerance for error control
    pub atol: f64,
    /// Maximum step size
    pub max_step: f64,
    /// Minimum step size
    pub min_step: f64,
    /// Initial step size (0 = auto-detect)
    pub initial_step: f64,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Whether to store dense output
    pub dense_output: bool,
    /// Whether to propagate parameter uncertainty
    pub propagate_param_uncertainty: bool,
    /// Number of Monte Carlo samples (0 = delta method only)
    pub monte_carlo_samples: usize,
    /// Process noise intensity (for stochastic systems)
    pub process_noise: f64,
}

impl SolverOptions {
    /// Default options for most problems
    pub fn default() -> Self {
        SolverOptions {
            rtol: 1e-6,
            atol: 1e-9,
            max_step: f64::INFINITY,
            min_step: 1e-15,
            initial_step: 0.0,
            max_steps: 100_000,
            dense_output: false,
            propagate_param_uncertainty: true,
            monte_carlo_samples: 0,
            process_noise: 0.0,
        }
    }

    /// High accuracy options
    pub fn high_accuracy() -> Self {
        SolverOptions {
            rtol: 1e-10,
            atol: 1e-12,
            max_step: f64::INFINITY,
            min_step: 1e-18,
            initial_step: 0.0,
            max_steps: 1_000_000,
            dense_output: false,
            propagate_param_uncertainty: true,
            monte_carlo_samples: 0,
            process_noise: 0.0,
        }
    }

    /// Options for stiff systems
    pub fn stiff() -> Self {
        SolverOptions {
            rtol: 1e-6,
            atol: 1e-8,
            max_step: f64::INFINITY,
            min_step: 1e-15,
            initial_step: 0.0,
            max_steps: 100_000,
            dense_output: false,
            propagate_param_uncertainty: true,
            monte_carlo_samples: 0,
            process_noise: 0.0,
        }
    }

    /// Options with Monte Carlo uncertainty propagation
    pub fn monte_carlo(samples: usize) -> Self {
        SolverOptions {
            rtol: 1e-6,
            atol: 1e-9,
            max_step: f64::INFINITY,
            min_step: 1e-15,
            initial_step: 0.0,
            max_steps: 100_000,
            dense_output: false,
            propagate_param_uncertainty: true,
            monte_carlo_samples: samples,
            process_noise: 0.0,
        }
    }
}

// ============================================================================
// SOLUTION RESULT
// ============================================================================

/// Complete solution trajectory with diagnostics
pub struct ODESolution<const N: usize> {
    /// Time points
    pub t: Vec<f64>,
    /// State at each time point
    pub states: Vec<EState<N>>,
    /// Final state
    pub y_final: EState<N>,
    /// Solver statistics
    pub stats: SolverStats,
    /// Error assessment
    pub error_assessment: ErrorAssessment<N>,
    /// Solution provenance
    pub provenance: Provenance,
}

impl<const N: usize> ODESolution<N> {
    /// Get state at specific time (linear interpolation)
    pub fn at(self, t_query: f64) -> Option<EState<N>> with Alloc {
        if t_query < self.t[0] || t_query > self.t[self.t.len() - 1] {
            return None
        }

        // Find bracketing interval
        let mut i = 0
        while i < self.t.len() - 1 && self.t[i + 1] < t_query {
            i += 1
        }

        // Linear interpolation
        let t0 = self.t[i]
        let t1 = self.t[i + 1]
        let alpha = (t_query - t0) / (t1 - t0)

        let s0 = &self.states[i]
        let s1 = &self.states[i + 1]

        let mut y = [0.0; N]
        let mut cov = [[0.0; N]; N]

        for j in 0..N {
            y[j] = (1.0 - alpha) * s0.y[j] + alpha * s1.y[j]
            for k in 0..N {
                cov[j][k] = (1.0 - alpha) * s0.covariance[j][k] + alpha * s1.covariance[j][k]
            }
        }

        Some(EState {
            y,
            covariance: cov,
            t: t_query,
            local_error: [0.0; N],
            provenance: self.provenance.step("interpolate", format!("t={:.4}", t_query)),
        })
    }

    /// Get trajectory of single component
    pub fn trajectory(self, i: usize) -> Vec<(f64, f64, f64)> with Alloc {
        let mut result = Vec::with_capacity(self.t.len())
        for k in 0..self.t.len() {
            let t = self.t[k]
            let y = self.states[k].y[i]
            let std = self.states[k].std(i)
            result.push((t, y, std))
        }
        result
    }

    /// Check if solution converged successfully
    pub fn is_success(self) -> bool {
        self.stats.n_steps < self.stats.max_steps_attempted
            && self.error_assessment.max_local_error < 1.0
    }
}

/// Solver performance statistics
pub struct SolverStats {
    /// Number of successful steps
    pub n_steps: usize,
    /// Number of RHS evaluations
    pub n_rhs_evals: usize,
    /// Number of Jacobian evaluations
    pub n_jacobian_evals: usize,
    /// Number of rejected steps
    pub n_rejected_steps: usize,
    /// Number of LU decompositions (for implicit methods)
    pub n_lu_decompositions: usize,
    /// Final step size used
    pub final_step_size: f64,
    /// CPU time in milliseconds
    pub cpu_time_ms: f64,
    /// Maximum steps attempted
    pub max_steps_attempted: usize,
}

/// Error and stability assessment
pub struct ErrorAssessment<const N: usize> {
    /// Estimated global error bound per component
    pub global_error_bound: [f64; N],
    /// Maximum local error encountered
    pub max_local_error: f64,
    /// Stability index (1 = stable, <1 = potential issues)
    pub stability_index: f64,
    /// Rate of uncertainty growth
    pub uncertainty_growth_rate: f64,
    /// Lyapunov exponent estimate (for chaotic systems)
    pub lyapunov_estimate: Option<f64>,
    /// Warning messages
    pub warnings: Vec<string>,
}

// ============================================================================
// RUNGE-KUTTA METHODS
// ============================================================================

/// Dormand-Prince 5(4) adaptive Runge-Kutta solver
pub struct RK45 {
    pub options: SolverOptions,
}

impl RK45 {
    pub fn default() -> Self {
        RK45 { options: SolverOptions::default() }
    }

    pub fn with_options(options: SolverOptions) -> Self {
        RK45 { options }
    }

    pub fn with_tolerance(rtol: f64, atol: f64) -> Self {
        let mut opts = SolverOptions::default()
        opts.rtol = rtol
        opts.atol = atol
        RK45 { options: opts }
    }
}

/// Runge-Kutta-Fehlberg 7(8) for high accuracy
pub struct RK78 {
    pub options: SolverOptions,
}

impl RK78 {
    pub fn default() -> Self {
        RK78 { options: SolverOptions::high_accuracy() }
    }
}

/// Backward Differentiation Formula for stiff systems
pub struct BDF {
    pub options: SolverOptions,
    pub max_order: usize,
}

impl BDF {
    pub fn default() -> Self {
        BDF {
            options: SolverOptions::stiff(),
            max_order: 5,
        }
    }

    pub fn with_order(order: usize) -> Self {
        BDF {
            options: SolverOptions::stiff(),
            max_order: order.min(5).max(1),
        }
    }
}

/// Radau IIA implicit Runge-Kutta for very stiff problems
pub struct Radau {
    pub options: SolverOptions,
}

impl Radau {
    pub fn default() -> Self {
        Radau { options: SolverOptions::stiff() }
    }
}

// ============================================================================
// SOLVER TRAIT
// ============================================================================

/// Common interface for ODE solvers
pub trait ODESolver<const N: usize, P> {
    /// Take a single step
    fn step(
        self,
        system: &ODESystem<N, P>,
        state: &EState<N>,
        h: f64,
    ) -> (EState<N>, [f64; N]) with Compute

    /// Solve the complete IVP
    fn solve(
        self,
        system: ODESystem<N, P>,
        t_span: (f64, f64),
        y0: EState<N>,
    ) -> ODESolution<N> with Alloc, Compute

    /// Get solver options
    fn options(self) -> &SolverOptions

    /// Solver name for diagnostics
    fn name(self) -> string
}

// ============================================================================
// RK45 IMPLEMENTATION (Dormand-Prince)
// ============================================================================

impl<const N: usize, P: Copy> ODESolver<N, P> for RK45 {
    fn step(
        self,
        system: &ODESystem<N, P>,
        state: &EState<N>,
        h: f64,
    ) -> (EState<N>, [f64; N]) with Compute {
        let t = state.t
        let y = state.y
        let p = system.params
        let f = system.rhs

        // Dormand-Prince 5(4) coefficients
        let a21 = 1.0 / 5.0
        let a31 = 3.0 / 40.0
        let a32 = 9.0 / 40.0
        let a41 = 44.0 / 45.0
        let a42 = 0.0 - 56.0 / 15.0
        let a43 = 32.0 / 9.0
        let a51 = 19372.0 / 6561.0
        let a52 = 0.0 - 25360.0 / 2187.0
        let a53 = 64448.0 / 6561.0
        let a54 = 0.0 - 212.0 / 729.0
        let a61 = 9017.0 / 3168.0
        let a62 = 0.0 - 355.0 / 33.0
        let a63 = 46732.0 / 5247.0
        let a64 = 49.0 / 176.0
        let a65 = 0.0 - 5103.0 / 18656.0
        let a71 = 35.0 / 384.0
        let a73 = 500.0 / 1113.0
        let a74 = 125.0 / 192.0
        let a75 = 0.0 - 2187.0 / 6784.0
        let a76 = 11.0 / 84.0

        // 5th order weights
        let b1 = 35.0 / 384.0
        let b3 = 500.0 / 1113.0
        let b4 = 125.0 / 192.0
        let b5 = 0.0 - 2187.0 / 6784.0
        let b6 = 11.0 / 84.0

        // Error weights (5th - 4th order)
        let e1 = 71.0 / 57600.0
        let e3 = 0.0 - 71.0 / 16695.0
        let e4 = 71.0 / 1920.0
        let e5 = 0.0 - 17253.0 / 339200.0
        let e6 = 22.0 / 525.0
        let e7 = 0.0 - 1.0 / 40.0

        // Stage 1
        let k1 = f(t, y, p)

        // Stage 2
        let mut y2 = [0.0; N]
        for i in 0..N { y2[i] = y[i] + h * a21 * k1[i] }
        let k2 = f(t + h / 5.0, y2, p)

        // Stage 3
        let mut y3 = [0.0; N]
        for i in 0..N { y3[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]) }
        let k3 = f(t + 3.0 * h / 10.0, y3, p)

        // Stage 4
        let mut y4 = [0.0; N]
        for i in 0..N { y4[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]) }
        let k4 = f(t + 4.0 * h / 5.0, y4, p)

        // Stage 5
        let mut y5 = [0.0; N]
        for i in 0..N { y5[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]) }
        let k5 = f(t + 8.0 * h / 9.0, y5, p)

        // Stage 6
        let mut y6 = [0.0; N]
        for i in 0..N { y6[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]) }
        let k6 = f(t + h, y6, p)

        // 5th order solution
        let mut y_new = [0.0; N]
        for i in 0..N {
            y_new[i] = y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])
        }

        // Stage 7 (for error estimate)
        let k7 = f(t + h, y_new, p)

        // Error estimate
        let mut error = [0.0; N]
        for i in 0..N {
            error[i] = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i])
        }

        // Propagate covariance using linearization
        let cov_new = propagate_covariance(&state.covariance, system, t, y, h, &self.options)

        let new_state = EState {
            y: y_new,
            covariance: cov_new,
            t: t + h,
            local_error: error,
            provenance: state.provenance.step("rk45_step", format!("t={:.6}", t + h)),
        }

        (new_state, error)
    }

    fn solve(
        self,
        system: ODESystem<N, P>,
        t_span: (f64, f64),
        y0: EState<N>,
    ) -> ODESolution<N> with Alloc, Compute {
        let (t0, tf) = t_span
        let opts = &self.options

        // Initial step size estimation
        let mut h = if opts.initial_step > 0.0 {
            opts.initial_step
        } else {
            estimate_initial_step(&system, &y0, opts)
        }

        let mut t = t0
        let mut state = y0.clone()
        let mut t_history = Vec::new()
        let mut state_history = Vec::new()

        let mut stats = SolverStats {
            n_steps: 0,
            n_rhs_evals: 0,
            n_jacobian_evals: 0,
            n_rejected_steps: 0,
            n_lu_decompositions: 0,
            final_step_size: h,
            cpu_time_ms: 0.0,
            max_steps_attempted: opts.max_steps,
        }

        // Store initial state
        t_history.push(t)
        state_history.push(state.clone())

        let mut max_local_error = 0.0
        let mut warnings = Vec::new()
        let initial_uncertainty = y0.total_uncertainty()

        // Main integration loop
        while t < tf && stats.n_steps < opts.max_steps {
            // Adjust step to hit end exactly
            if t + h > tf {
                h = tf - t
            }

            // Take a step
            let (new_state, error) = self.step(&system, &state, h)
            stats.n_rhs_evals += 7  // DOPRI5 uses 7 function evaluations

            // Compute error norm
            let mut err_norm = 0.0
            for i in 0..N {
                let scale = opts.atol + opts.rtol * abs_f64(state.y[i]).max(abs_f64(new_state.y[i]))
                err_norm += (error[i] / scale) * (error[i] / scale)
            }
            err_norm = sqrt_f64(err_norm / (N as f64))

            max_local_error = max_f64(max_local_error, err_norm)

            if err_norm <= 1.0 {
                // Accept step
                t = new_state.t
                state = new_state
                t_history.push(t)
                state_history.push(state.clone())
                stats.n_steps += 1

                // Compute optimal step size
                let safety = 0.9
                let p_grow = 0.2   // 1/(order+1)
                let p_shrink = 0.25
                let factor = safety * pow_f64(1.0 / err_norm, p_grow)
                let factor = max_f64(0.2, min_f64(5.0, factor))
                h = h * factor
                h = min_f64(h, opts.max_step)
                h = max_f64(h, opts.min_step)
            } else {
                // Reject step
                stats.n_rejected_steps += 1
                let safety = 0.9
                let factor = safety * pow_f64(1.0 / err_norm, 0.25)
                h = h * max_f64(0.2, factor)
                h = max_f64(h, opts.min_step)
            }
        }

        stats.final_step_size = h

        // Compute uncertainty growth
        let final_uncertainty = state.total_uncertainty()
        let uncertainty_growth = if initial_uncertainty > 1e-15 {
            final_uncertainty / initial_uncertainty
        } else if final_uncertainty > 1e-15 {
            f64::INFINITY
        } else {
            1.0
        }

        // Build error assessment
        let mut global_error_bound = [0.0; N]
        for s in state_history.iter() {
            for i in 0..N {
                global_error_bound[i] += abs_f64(s.local_error[i])
            }
        }

        if stats.n_steps >= opts.max_steps {
            warnings.push("Maximum number of steps reached")
        }

        let error_assessment = ErrorAssessment {
            global_error_bound,
            max_local_error,
            stability_index: 1.0 / (1.0 + max_local_error),
            uncertainty_growth_rate: uncertainty_growth,
            lyapunov_estimate: None,
            warnings,
        }

        ODESolution {
            t: t_history,
            states: state_history,
            y_final: state,
            stats,
            error_assessment,
            provenance: Provenance::new("ode_solution").step("rk45", "Dormand-Prince 5(4)"),
        }
    }

    fn options(self) -> &SolverOptions {
        &self.options
    }

    fn name(self) -> string {
        "RK45 (Dormand-Prince)"
    }
}

// ============================================================================
// BDF IMPLEMENTATION (for stiff systems)
// ============================================================================

impl<const N: usize, P: Copy> ODESolver<N, P> for BDF {
    fn step(
        self,
        system: &ODESystem<N, P>,
        state: &EState<N>,
        h: f64,
    ) -> (EState<N>, [f64; N]) with Compute {
        // Simplified BDF1 (Backward Euler) for demonstration
        // Full implementation would use variable order BDF
        let t_new = state.t + h
        let y = state.y
        let p = system.params

        // Newton iteration for implicit solve: y_new = y + h * f(t_new, y_new)
        let mut y_new = y  // Initial guess
        let max_newton = 10
        let tol = 1e-10

        for _ in 0..max_newton {
            let f_new = (system.rhs)(t_new, y_new, p)
            let J = system.compute_jacobian(t_new, y_new)

            // Residual: R = y_new - y - h * f(t_new, y_new)
            let mut residual = [0.0; N]
            for i in 0..N {
                residual[i] = y_new[i] - y[i] - h * f_new[i]
            }

            // Check convergence
            let mut res_norm = 0.0
            for i in 0..N {
                res_norm += residual[i] * residual[i]
            }
            if sqrt_f64(res_norm) < tol {
                break
            }

            // Newton matrix: I - h * J
            let mut A = [[0.0; N]; N]
            for i in 0..N {
                for j in 0..N {
                    A[i][j] = if i == j { 1.0 } else { 0.0 }
                    A[i][j] -= h * J[i][j]
                }
            }

            // Solve A * delta = -residual (simplified Gaussian elimination)
            let delta = solve_linear_system(A, residual)

            // Update
            for i in 0..N {
                y_new[i] -= delta[i]
            }
        }

        // Error estimate (difference from explicit Euler)
        let f_old = (system.rhs)(state.t, y, p)
        let mut error = [0.0; N]
        for i in 0..N {
            let y_explicit = y[i] + h * f_old[i]
            error[i] = (y_new[i] - y_explicit) / 2.0
        }

        let cov_new = propagate_covariance(&state.covariance, system, state.t, y, h, &self.options)

        let new_state = EState {
            y: y_new,
            covariance: cov_new,
            t: t_new,
            local_error: error,
            provenance: state.provenance.step("bdf_step", format!("t={:.6}", t_new)),
        }

        (new_state, error)
    }

    fn solve(
        self,
        system: ODESystem<N, P>,
        t_span: (f64, f64),
        y0: EState<N>,
    ) -> ODESolution<N> with Alloc, Compute {
        let (t0, tf) = t_span
        let opts = &self.options

        let mut h = if opts.initial_step > 0.0 {
            opts.initial_step
        } else {
            estimate_initial_step(&system, &y0, opts) * 10.0  // BDF can take larger steps
        }

        let mut t = t0
        let mut state = y0.clone()
        let mut t_history = Vec::new()
        let mut state_history = Vec::new()

        let mut stats = SolverStats {
            n_steps: 0,
            n_rhs_evals: 0,
            n_jacobian_evals: 0,
            n_rejected_steps: 0,
            n_lu_decompositions: 0,
            final_step_size: h,
            cpu_time_ms: 0.0,
            max_steps_attempted: opts.max_steps,
        }

        t_history.push(t)
        state_history.push(state.clone())

        let mut max_local_error = 0.0
        let initial_uncertainty = y0.total_uncertainty()

        while t < tf && stats.n_steps < opts.max_steps {
            if t + h > tf {
                h = tf - t
            }

            let (new_state, error) = self.step(&system, &state, h)
            stats.n_rhs_evals += 10  // Newton iterations
            stats.n_jacobian_evals += 1
            stats.n_lu_decompositions += 1

            let mut err_norm = 0.0
            for i in 0..N {
                let scale = opts.atol + opts.rtol * abs_f64(new_state.y[i])
                err_norm += (error[i] / scale) * (error[i] / scale)
            }
            err_norm = sqrt_f64(err_norm / (N as f64))

            max_local_error = max_f64(max_local_error, err_norm)

            if err_norm <= 1.0 {
                t = new_state.t
                state = new_state
                t_history.push(t)
                state_history.push(state.clone())
                stats.n_steps += 1

                let factor = min_f64(2.0, 0.9 * pow_f64(1.0 / err_norm, 0.5))
                h = h * factor
                h = min_f64(h, opts.max_step)
            } else {
                stats.n_rejected_steps += 1
                h = h * 0.5
                h = max_f64(h, opts.min_step)
            }
        }

        stats.final_step_size = h

        let final_uncertainty = state.total_uncertainty()
        let uncertainty_growth = if initial_uncertainty > 1e-15 {
            final_uncertainty / initial_uncertainty
        } else {
            1.0
        }

        let mut global_error_bound = [0.0; N]
        for s in state_history.iter() {
            for i in 0..N {
                global_error_bound[i] += abs_f64(s.local_error[i])
            }
        }

        let error_assessment = ErrorAssessment {
            global_error_bound,
            max_local_error,
            stability_index: 1.0 / (1.0 + max_local_error),
            uncertainty_growth_rate: uncertainty_growth,
            lyapunov_estimate: None,
            warnings: Vec::new(),
        }

        ODESolution {
            t: t_history,
            states: state_history,
            y_final: state,
            stats,
            error_assessment,
            provenance: Provenance::new("ode_solution").step("bdf", format!("BDF order {}", self.max_order)),
        }
    }

    fn options(self) -> &SolverOptions {
        &self.options
    }

    fn name(self) -> string {
        format!("BDF (order {})", self.max_order)
    }
}

// ============================================================================
// COVARIANCE PROPAGATION
// ============================================================================

/// Propagate covariance matrix through one integration step
fn propagate_covariance<const N: usize, P: Copy>(
    P_old: &[[f64; N]; N],
    system: &ODESystem<N, P>,
    t: f64,
    y: [f64; N],
    h: f64,
    opts: &SolverOptions,
) -> [[f64; N]; N] with Compute {
    // Compute Jacobian at current point
    let J = system.compute_jacobian(t, y)

    // State transition matrix: Φ ≈ I + h*J (first-order approximation)
    let mut Phi = [[0.0; N]; N]
    for i in 0..N {
        for j in 0..N {
            Phi[i][j] = if i == j { 1.0 } else { 0.0 }
            Phi[i][j] += h * J[i][j]
        }
    }

    // Propagate covariance: P_new = Φ P Φᵀ + Q
    // First compute Φ P
    let mut temp = [[0.0; N]; N]
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                temp[i][j] += Phi[i][k] * P_old[k][j]
            }
        }
    }

    // Then compute (Φ P) Φᵀ
    let mut P_new = [[0.0; N]; N]
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                P_new[i][j] += temp[i][k] * Phi[j][k]  // Φᵀ[k][j] = Φ[j][k]
            }
        }
    }

    // Add process noise Q (discretized)
    let q = h * opts.process_noise + h * opts.atol * opts.atol
    for i in 0..N {
        P_new[i][i] += q
    }

    P_new
}

/// Compute Jacobian using finite differences
fn finite_difference_jacobian<const N: usize, P: Copy>(
    f: RHSFn<N, P>,
    t: f64,
    y: [f64; N],
    p: P,
) -> [[f64; N]; N] {
    let eps = 1e-8
    let f0 = f(t, y, p)
    let mut J = [[0.0; N]; N]

    for j in 0..N {
        let mut y_pert = y
        let h = eps * (1.0 + abs_f64(y[j]))
        y_pert[j] += h
        let f_pert = f(t, y_pert, p)

        for i in 0..N {
            J[i][j] = (f_pert[i] - f0[i]) / h
        }
    }

    J
}

/// Estimate initial step size
fn estimate_initial_step<const N: usize, P: Copy>(
    system: &ODESystem<N, P>,
    y0: &EState<N>,
    opts: &SolverOptions,
) -> f64 {
    let f0 = (system.rhs)(y0.t, y0.y, system.params)

    let mut d0 = 0.0
    let mut d1 = 0.0

    for i in 0..N {
        let scale = opts.atol + opts.rtol * abs_f64(y0.y[i])
        d0 += (y0.y[i] / scale) * (y0.y[i] / scale)
        d1 += (f0[i] / scale) * (f0[i] / scale)
    }

    d0 = sqrt_f64(d0 / (N as f64))
    d1 = sqrt_f64(d1 / (N as f64))

    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    }

    min_f64(h0, opts.max_step)
}

/// Simple linear system solver (Gaussian elimination)
fn solve_linear_system<const N: usize>(A: [[f64; N]; N], b: [f64; N]) -> [f64; N] {
    let mut Aug = [[0.0; N + 1]; N]

    // Build augmented matrix
    for i in 0..N {
        for j in 0..N {
            Aug[i][j] = A[i][j]
        }
        Aug[i][N] = b[i]
    }

    // Forward elimination
    for k in 0..N {
        // Find pivot
        let mut max_row = k
        let mut max_val = abs_f64(Aug[k][k])
        for i in (k + 1)..N {
            if abs_f64(Aug[i][k]) > max_val {
                max_val = abs_f64(Aug[i][k])
                max_row = i
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..(N + 1) {
                let tmp = Aug[k][j]
                Aug[k][j] = Aug[max_row][j]
                Aug[max_row][j] = tmp
            }
        }

        // Eliminate
        if abs_f64(Aug[k][k]) > 1e-15 {
            for i in (k + 1)..N {
                let factor = Aug[i][k] / Aug[k][k]
                for j in k..(N + 1) {
                    Aug[i][j] -= factor * Aug[k][j]
                }
            }
        }
    }

    // Back substitution
    let mut x = [0.0; N]
    for ii in 0..N {
        let i = N - 1 - ii
        let mut sum = Aug[i][N]
        for j in (i + 1)..N {
            sum -= Aug[i][j] * x[j]
        }
        if abs_f64(Aug[i][i]) > 1e-15 {
            x[i] = sum / Aug[i][i]
        }
    }

    x
}

// ============================================================================
// HIGH-LEVEL API
// ============================================================================

/// Solve initial value problem with exact initial conditions
pub fn solve_ivp<const N: usize, P: Copy, S: ODESolver<N, P>>(
    system: ODESystem<N, P>,
    t_span: (f64, f64),
    y0: [f64; N],
    solver: S,
) -> ODESolution<N> with Alloc, Compute {
    let initial_state = EState::exact(y0, t_span.0)
    solver.solve(system, t_span, initial_state)
}

/// Solve initial value problem with uncertain initial conditions
pub fn solve_ivp_uncertain<const N: usize, P: Copy, S: ODESolver<N, P>>(
    system: ODESystem<N, P>,
    t_span: (f64, f64),
    y0: [Knowledge<f64>; N],
    solver: S,
) -> ODESolution<N> with Alloc, Compute {
    let initial_state = EState::uncertain(y0, t_span.0)
    solver.solve(system, t_span, initial_state)
}

/// Solve with full covariance specification
pub fn solve_ivp_covariance<const N: usize, P: Copy, S: ODESolver<N, P>>(
    system: ODESystem<N, P>,
    t_span: (f64, f64),
    y0: [f64; N],
    cov0: [[f64; N]; N],
    solver: S,
) -> ODESolution<N> with Alloc, Compute {
    let initial_state = EState::with_covariance(y0, cov0, t_span.0)
    solver.solve(system, t_span, initial_state)
}

/// Automatically select solver based on system properties
pub fn solve_auto<const N: usize, P: Copy>(
    system: ODESystem<N, P>,
    t_span: (f64, f64),
    y0: [f64; N],
) -> ODESolution<N> with Alloc, Compute {
    if system.is_stiff {
        solve_ivp(system, t_span, y0, BDF::default())
    } else {
        solve_ivp(system, t_span, y0, RK45::default())
    }
}

// ============================================================================
// SENSITIVITY ANALYSIS
// ============================================================================

/// Result of sensitivity analysis
pub struct SensitivityResult<const N: usize, const P_DIM: usize> {
    /// Sensitivity matrix ∂y/∂p at final time
    pub sensitivity: [[f64; P_DIM]; N],
    /// Sensitivity variance
    pub sensitivity_variance: [[f64; P_DIM]; N],
    /// Solution trajectory
    pub solution: ODESolution<N>,
}

/// Forward sensitivity analysis
pub fn forward_sensitivity<const N: usize, const P_DIM: usize, P: Copy>(
    system: ODESystem<N, P>,
    t_span: (f64, f64),
    y0: [f64; N],
    param_indices: [usize; P_DIM],
    solver: RK45,
) -> SensitivityResult<N, P_DIM> with Alloc, Compute {
    // Solve the base system
    let solution = solve_ivp(system, t_span, y0, solver)

    // Initialize sensitivity matrix (all zeros at t=0)
    let mut S = [[0.0; P_DIM]; N]
    let S_var = [[0.0; P_DIM]; N]

    // Simplified: return initial sensitivity for now
    // Full implementation would integrate sensitivity ODEs alongside state
    SensitivityResult {
        sensitivity: S,
        sensitivity_variance: S_var,
        solution,
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = x
    for _ in 0..15 {
        y = 0.5 * (y + x / y)
    }
    y
}

fn pow_f64(base: f64, exp: f64) -> f64 {
    if exp == 0.0 { return 1.0 }
    if exp == 1.0 { return base }
    if exp == 2.0 { return base * base }
    if exp == 0.5 { return sqrt_f64(base) }
    if exp == 0.25 { return sqrt_f64(sqrt_f64(base)) }
    if exp == 0.2 { return pow_f64(base, 0.25) * pow_f64(base, -0.05) }
    // General case using exp/ln
    exp_f64(exp * ln_f64(base))
}

fn exp_f64(x: f64) -> f64 {
    let mut sum = 1.0
    let mut term = 1.0
    for i in 1..30 {
        term *= x / (i as f64)
        sum += term
        if abs_f64(term) < 1e-15 { break }
    }
    sum
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1e308 }
    let mut y = x - 1.0
    for _ in 0..30 {
        let ey = exp_f64(y)
        y = y + 2.0 * (x - ey) / (x + ey)
    }
    y
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

// ============================================================================
// TESTS
// ============================================================================

#[test]
fn test_exponential_decay() with Alloc, Compute {
    let k = 0.5

    let system = ODESystem::new(
        |_t, y, k| [0.0 - k * y[0]],
        k,
    )

    let result = solve_ivp(system, (0.0, 2.0), [1.0], RK45::default())

    let y_final = result.y_final.y[0]
    let y_exact = exp_f64(0.0 - k * 2.0)

    assert!(abs_f64(y_final - y_exact) < 1e-6)
}

#[test]
fn test_harmonic_oscillator() with Alloc, Compute {
    // dy/dt = v, dv/dt = -y
    let system = ODESystem::new(
        |_t, y, _| [y[1], 0.0 - y[0]],
        (),
    )

    let result = solve_ivp(system, (0.0, 6.283185), [1.0, 0.0], RK45::default())

    // After one period, should return to initial state
    let y_final = result.y_final.y[0]
    let v_final = result.y_final.y[1]

    assert!(abs_f64(y_final - 1.0) < 1e-4)
    assert!(abs_f64(v_final) < 1e-4)
}

#[test]
fn test_uncertainty_growth() with Alloc, Compute {
    // Exponentially growing system: dy/dt = y
    let system = ODESystem::new(
        |_t, y, _| [y[0]],
        (),
    )

    let y0 = [Knowledge::from_gaussian(1.0, 0.01, Provenance::new("test"))]
    let result = solve_ivp_uncertain(system, (0.0, 1.0), y0, RK45::default())

    // Uncertainty should grow for unstable system
    let initial_var = 0.01
    let final_var = result.y_final.covariance[0][0]

    assert!(final_var > initial_var)
}

#[test]
fn test_stiff_system() with Alloc, Compute {
    // Simple stiff system: dy/dt = -1000*y
    let system = ODESystem::new(
        |_t, y, _| [0.0 - 1000.0 * y[0]],
        (),
    ).stiff()

    let result = solve_ivp(system, (0.0, 0.01), [1.0], BDF::default())

    // Should decay rapidly
    assert!(result.y_final.y[0] < 0.001)
    assert!(result.is_success())
}

#[test]
fn test_solver_statistics() with Alloc, Compute {
    let system = ODESystem::new(
        |_t, y, _| [0.0 - y[0]],
        (),
    )

    let result = solve_ivp(system, (0.0, 1.0), [1.0], RK45::default())

    assert!(result.stats.n_steps > 0)
    assert!(result.stats.n_rhs_evals > 0)
    assert!(result.stats.n_rejected_steps < result.stats.n_steps)
}

#[test]
fn test_covariance_propagation() with Alloc, Compute {
    // Linear system: dy/dt = A*y with A = [[0, 1], [-1, 0]]
    let system = ODESystem::new(
        |_t, y, _| [y[1], 0.0 - y[0]],
        (),
    )

    let y0 = [1.0, 0.0]
    let cov0 = [[0.01, 0.0], [0.0, 0.01]]

    let result = solve_ivp_covariance(system, (0.0, 3.14159), y0, cov0, RK45::default())

    // Covariance should remain bounded for oscillatory system
    let max_var = result.y_final.max_variance()
    assert!(max_var < 1.0)
}

#[test]
fn test_interpolation() with Alloc, Compute {
    let system = ODESystem::new(
        |_t, y, _| [0.0 - y[0]],
        (),
    )

    let result = solve_ivp(system, (0.0, 1.0), [1.0], RK45::default())

    // Interpolate at t = 0.5
    let interp = result.at(0.5)
    assert!(interp.is_some())

    let state = interp.unwrap()
    let expected = exp_f64(-0.5)
    assert!(abs_f64(state.y[0] - expected) < 0.01)
}

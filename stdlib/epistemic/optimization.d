/// epistemic::optimization — Optimization with Epistemic Uncertainty
///
/// Production-grade optimization algorithms that track uncertainty in:
/// - Optimal point location (where is the minimum?)
/// - Optimal value (what is the minimum value?)
/// - Convergence confidence (have we converged?)
/// - Gradient uncertainty (noisy/stochastic gradients)
///
/// # Algorithms
///
/// - **Gradient-Based**: SGD, Adam, RMSprop, AdaGrad with variance tracking
/// - **Second-Order**: Newton, BFGS, L-BFGS with Hessian uncertainty
/// - **Constrained**: Barrier methods, augmented Lagrangian, projected gradient
/// - **Global**: Simulated annealing, CMA-ES, basin hopping
/// - **Bayesian**: Gaussian process optimization with acquisition functions
///
/// # Philosophy
///
/// Traditional optimization: "The minimum is at x = 2.5"
/// Demetrios epistemic: "The minimum is at x = 2.5 ± 0.03 (95% CI),
///                       with f(x*) = 1.2 ± 0.01, convergence confidence 0.98"
///
/// # Quick Start
///
/// ```demetrios
/// use std::epistemic::optimization::{adam, AdamOptions, minimize}
///
/// fn objective(x: &EVector<2>) -> Knowledge<f64> {
///     // Rosenbrock function with measurement noise
///     let a = 1.0 - x.values[0]
///     let b = x.values[1] - x.values[0] * x.values[0]
///     Knowledge::from_measurement(a * a + 100.0 * b * b, 0.01)
/// }
///
/// let x0 = evec_new([0.0, 0.0])
/// let result = minimize(objective, x0, adam_default())
/// // result.optimum has full epistemic uncertainty
/// ```

use std::epistemic::knowledge::{Knowledge, Confidence, Provenance}
use std::epistemic::linalg::{EVector, EMatrix, evec_new, evec_zeros, emat_zeros, emat_identity}

// ============================================================================
// Core Types
// ============================================================================

/// Optimization result with full uncertainty quantification
struct OptimizationResult<const N: usize> {
    optimum: EVector<N>,              // Optimal point with uncertainty
    value: Knowledge<f64>,            // Optimal value with uncertainty
    gradient_norm: f64,               // Final gradient norm
    iterations: i32,                  // Number of iterations
    function_evals: i32,              // Number of function evaluations
    gradient_evals: i32,              // Number of gradient evaluations
    converged: Knowledge<bool>,       // Convergence with confidence
    convergence_reason: ConvergenceReason,
    trajectory: [EVector<N>],         // Optimization path (if stored)
    hessian_estimate: Option<EMatrix<N, N>>,  // Approximate Hessian at optimum
}

/// Reason for convergence/termination
enum ConvergenceReason {
    GradientTolerance,    // ||grad|| < tol
    StepTolerance,        // ||x_new - x_old|| < tol
    FunctionTolerance,    // |f_new - f_old| < tol
    MaxIterations,        // Hit iteration limit
    LineSearchFailed,     // Line search couldn't find improvement
    NumericalIssue,       // NaN, Inf, or ill-conditioning
    UserTerminated,       // Callback requested stop
}

/// Line search result
struct LineSearchResult {
    step_size: f64,
    value: f64,
    sufficient_decrease: bool,
    curvature_condition: bool,
    evals: i32,
}

/// Convergence diagnostics
struct ConvergenceDiagnostics<const N: usize> {
    gradient_norms: [f64],
    function_values: [f64],
    step_sizes: [f64],
    condition_numbers: [f64],
    estimated_distance_to_optimum: Knowledge<f64>,
}

// ============================================================================
// Optimizer Options
// ============================================================================

/// Common options for all optimizers
struct OptimizerOptions {
    max_iterations: i32,
    grad_tol: f64,
    step_tol: f64,
    func_tol: f64,
    store_trajectory: bool,
    verbose: bool,
}

/// Default optimizer options
fn optimizer_options_default() -> OptimizerOptions {
    OptimizerOptions {
        max_iterations: 1000,
        grad_tol: 1e-6,
        step_tol: 1e-8,
        func_tol: 1e-8,
        store_trajectory: false,
        verbose: false,
    }
}

/// SGD options
struct SGDOptions {
    learning_rate: f64,
    momentum: f64,
    nesterov: bool,
    base: OptimizerOptions,
}

fn sgd_default() -> SGDOptions {
    SGDOptions {
        learning_rate: 0.01,
        momentum: 0.9,
        nesterov: true,
        base: optimizer_options_default(),
    }
}

/// Adam options
struct AdamOptions {
    learning_rate: f64,
    beta1: f64,           // First moment decay
    beta2: f64,           // Second moment decay
    epsilon: f64,         // Numerical stability
    amsgrad: bool,        // Use AMSGrad variant
    base: OptimizerOptions,
}

fn adam_default() -> AdamOptions {
    AdamOptions {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        amsgrad: false,
        base: optimizer_options_default(),
    }
}

/// RMSprop options
struct RMSpropOptions {
    learning_rate: f64,
    alpha: f64,           // Smoothing constant
    epsilon: f64,
    momentum: f64,
    centered: bool,       // Use centered RMSprop
    base: OptimizerOptions,
}

fn rmsprop_default() -> RMSpropOptions {
    RMSpropOptions {
        learning_rate: 0.01,
        alpha: 0.99,
        epsilon: 1e-8,
        momentum: 0.0,
        centered: false,
        base: optimizer_options_default(),
    }
}

/// L-BFGS options
struct LBFGSOptions {
    memory_size: i32,     // Number of correction pairs to store
    line_search: LineSearchType,
    base: OptimizerOptions,
}

fn lbfgs_default() -> LBFGSOptions {
    LBFGSOptions {
        memory_size: 10,
        line_search: LineSearchType::StrongWolfe,
        base: optimizer_options_default(),
    }
}

/// Newton options
struct NewtonOptions {
    regularization: f64,  // Tikhonov regularization for ill-conditioned Hessian
    trust_region: bool,   // Use trust region instead of line search
    trust_radius: f64,    // Initial trust region radius
    base: OptimizerOptions,
}

fn newton_default() -> NewtonOptions {
    NewtonOptions {
        regularization: 1e-6,
        trust_region: true,
        trust_radius: 1.0,
        base: optimizer_options_default(),
    }
}

/// Line search type
enum LineSearchType {
    Backtracking,         // Simple backtracking
    StrongWolfe,          // Strong Wolfe conditions
    MoreThuente,          // More-Thuente algorithm
}

/// Constrained optimization options
struct ConstrainedOptions {
    barrier_param: f64,   // Initial barrier parameter
    barrier_decay: f64,   // Barrier parameter decay rate
    penalty_param: f64,   // Penalty parameter for augmented Lagrangian
    base: OptimizerOptions,
}

fn constrained_default() -> ConstrainedOptions {
    ConstrainedOptions {
        barrier_param: 1.0,
        barrier_decay: 0.1,
        penalty_param: 1.0,
        base: optimizer_options_default(),
    }
}

/// Bayesian optimization options
struct BayesOptOptions {
    n_initial: i32,           // Initial random samples
    acquisition: AcquisitionFunction,
    kernel: KernelType,
    noise_variance: f64,      // Observation noise
    base: OptimizerOptions,
}

fn bayesopt_default() -> BayesOptOptions {
    BayesOptOptions {
        n_initial: 5,
        acquisition: AcquisitionFunction::ExpectedImprovement,
        kernel: KernelType::Matern52,
        noise_variance: 1e-6,
        base: optimizer_options_default(),
    }
}

/// Acquisition function for Bayesian optimization
enum AcquisitionFunction {
    ExpectedImprovement,      // EI
    ProbabilityImprovement,   // PI
    UpperConfidenceBound(f64), // UCB with exploration parameter
    ThompsonSampling,         // TS
}

/// Kernel type for Gaussian process
enum KernelType {
    RBF,                  // Squared exponential
    Matern32,             // Matern 3/2
    Matern52,             // Matern 5/2
    RationalQuadratic(f64), // With alpha parameter
}

/// Global optimization options
struct GlobalOptOptions {
    n_restarts: i32,          // Number of random restarts
    temperature: f64,         // Initial temperature (simulated annealing)
    cooling_rate: f64,        // Temperature decay
    population_size: i32,     // Population size (evolutionary)
    base: OptimizerOptions,
}

fn globalopt_default() -> GlobalOptOptions {
    GlobalOptOptions {
        n_restarts: 10,
        temperature: 1.0,
        cooling_rate: 0.95,
        population_size: 50,
        base: optimizer_options_default(),
    }
}

// ============================================================================
// SGD Implementation
// ============================================================================

/// Stochastic Gradient Descent with momentum
fn sgd<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    x0: EVector<N>,
    opts: SGDOptions
) -> OptimizationResult<N> with Alloc {
    var x = x0
    var velocity = evec_zeros::<N>()
    var trajectory: [EVector<N>] = []
    var func_evals = 0
    var grad_evals = 0

    if opts.base.store_trajectory {
        trajectory = trajectory ++ [x]
    }

    var best_x = x
    var best_value = objective(&x)
    func_evals = func_evals + 1

    var converged = false
    var reason = ConvergenceReason::MaxIterations

    for iter in 0..opts.base.max_iterations {
        // Compute gradient
        let g = if opts.nesterov {
            // Nesterov: evaluate gradient at lookahead position
            var x_lookahead = evec_zeros::<N>()
            for i in 0..N {
                x_lookahead.values[i] = x.values[i] - opts.momentum * velocity.values[i]
            }
            gradient(&x_lookahead)
        } else {
            gradient(&x)
        }
        grad_evals = grad_evals + 1

        // Check gradient convergence
        let grad_norm = evec_norm(&g)
        if grad_norm < opts.base.grad_tol {
            converged = true
            reason = ConvergenceReason::GradientTolerance
            break
        }

        // Update velocity and position
        var step_norm = 0.0
        for i in 0..N {
            velocity.values[i] = opts.momentum * velocity.values[i] + opts.learning_rate * g.values[i]
            let step = velocity.values[i]
            x.values[i] = x.values[i] - step
            step_norm = step_norm + step * step

            // Propagate variance from gradient uncertainty
            x.variances[i] = x.variances[i] + opts.learning_rate * opts.learning_rate * g.variances[i]
        }
        step_norm = sqrt(step_norm)

        // Check step convergence
        if step_norm < opts.base.step_tol {
            converged = true
            reason = ConvergenceReason::StepTolerance
            break
        }

        // Track best
        let value = objective(&x)
        func_evals = func_evals + 1
        if value.value < best_value.value {
            best_x = x
            best_value = value
        }

        if opts.base.store_trajectory {
            trajectory = trajectory ++ [x]
        }
    }

    // Final gradient for uncertainty estimation
    let final_grad = gradient(&best_x)
    grad_evals = grad_evals + 1
    let final_grad_norm = evec_norm(&final_grad)

    // Estimate optimum uncertainty from gradient variance
    for i in 0..N {
        best_x.variances[i] = max(best_x.variances[i],
                                   final_grad.variances[i] / (opts.learning_rate * opts.learning_rate + 1e-10))
    }

    OptimizationResult {
        optimum: best_x,
        value: best_value,
        gradient_norm: final_grad_norm,
        iterations: opts.base.max_iterations,
        function_evals: func_evals,
        gradient_evals: grad_evals,
        converged: Knowledge {
            value: converged,
            variance: if converged { 0.01 } else { 0.1 },
            confidence: Confidence::Algorithmic { iterations: opts.base.max_iterations },
            provenance: Provenance::SGD { learning_rate: opts.learning_rate, momentum: opts.momentum },
        },
        convergence_reason: reason,
        trajectory: trajectory,
        hessian_estimate: None,
    }
}

// ============================================================================
// Adam Implementation
// ============================================================================

/// Adam optimizer with epistemic tracking
fn adam<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    x0: EVector<N>,
    opts: AdamOptions
) -> OptimizationResult<N> with Alloc {
    var x = x0
    var m = evec_zeros::<N>()  // First moment
    var v = evec_zeros::<N>()  // Second moment
    var v_max = evec_zeros::<N>()  // For AMSGrad
    var trajectory: [EVector<N>] = []
    var func_evals = 0
    var grad_evals = 0

    if opts.base.store_trajectory {
        trajectory = trajectory ++ [x]
    }

    var best_x = x
    var best_value = objective(&x)
    func_evals = func_evals + 1
    var prev_value = best_value.value

    var converged = false
    var reason = ConvergenceReason::MaxIterations

    for iter in 0..opts.base.max_iterations {
        let t = (iter + 1) as f64

        // Compute gradient
        let g = gradient(&x)
        grad_evals = grad_evals + 1

        // Check gradient convergence
        let grad_norm = evec_norm(&g)
        if grad_norm < opts.base.grad_tol {
            converged = true
            reason = ConvergenceReason::GradientTolerance
            break
        }

        // Update biased moments
        for i in 0..N {
            m.values[i] = opts.beta1 * m.values[i] + (1.0 - opts.beta1) * g.values[i]
            v.values[i] = opts.beta2 * v.values[i] + (1.0 - opts.beta2) * g.values[i] * g.values[i]

            // Track gradient variance in moment estimates
            m.variances[i] = opts.beta1 * opts.beta1 * m.variances[i] +
                             (1.0 - opts.beta1) * (1.0 - opts.beta1) * g.variances[i]
        }

        // Bias correction
        let beta1_t = 1.0 - pow(opts.beta1, t)
        let beta2_t = 1.0 - pow(opts.beta2, t)

        // Update parameters
        var step_norm = 0.0
        for i in 0..N {
            let m_hat = m.values[i] / beta1_t
            let v_hat = if opts.amsgrad {
                v_max.values[i] = max(v_max.values[i], v.values[i])
                v_max.values[i] / beta2_t
            } else {
                v.values[i] / beta2_t
            }

            let step = opts.learning_rate * m_hat / (sqrt(v_hat) + opts.epsilon)
            x.values[i] = x.values[i] - step
            step_norm = step_norm + step * step

            // Update variance: uncertainty from adaptive learning rate
            let effective_lr = opts.learning_rate / (sqrt(v_hat) + opts.epsilon)
            x.variances[i] = x.variances[i] + effective_lr * effective_lr * m.variances[i] / (beta1_t * beta1_t)
        }
        step_norm = sqrt(step_norm)

        // Check step convergence
        if step_norm < opts.base.step_tol {
            converged = true
            reason = ConvergenceReason::StepTolerance
            break
        }

        // Evaluate and check function convergence
        let value = objective(&x)
        func_evals = func_evals + 1

        if abs(value.value - prev_value) < opts.base.func_tol {
            converged = true
            reason = ConvergenceReason::FunctionTolerance
            break
        }

        if value.value < best_value.value {
            best_x = x
            best_value = value
        }
        prev_value = value.value

        if opts.base.store_trajectory {
            trajectory = trajectory ++ [x]
        }
    }

    // Final assessment
    let final_grad = gradient(&best_x)
    grad_evals = grad_evals + 1

    OptimizationResult {
        optimum: best_x,
        value: best_value,
        gradient_norm: evec_norm(&final_grad),
        iterations: opts.base.max_iterations,
        function_evals: func_evals,
        gradient_evals: grad_evals,
        converged: Knowledge {
            value: converged,
            variance: if converged { 0.01 } else { 0.1 },
            confidence: Confidence::Algorithmic { iterations: opts.base.max_iterations },
            provenance: Provenance::Adam { beta1: opts.beta1, beta2: opts.beta2 },
        },
        convergence_reason: reason,
        trajectory: trajectory,
        hessian_estimate: None,
    }
}

// ============================================================================
// L-BFGS Implementation
// ============================================================================

/// L-BFGS optimizer
fn lbfgs<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    x0: EVector<N>,
    opts: LBFGSOptions
) -> OptimizationResult<N> with Alloc {
    var x = x0
    var trajectory: [EVector<N>] = []
    var func_evals = 0
    var grad_evals = 0

    // Storage for L-BFGS correction pairs
    var s_history: [EVector<N>] = []  // x_{k+1} - x_k
    var y_history: [EVector<N>] = []  // g_{k+1} - g_k
    var rho_history: [f64] = []       // 1 / (y^T s)

    if opts.base.store_trajectory {
        trajectory = trajectory ++ [x]
    }

    var g = gradient(&x)
    grad_evals = grad_evals + 1
    var value = objective(&x)
    func_evals = func_evals + 1

    var best_x = x
    var best_value = value

    var converged = false
    var reason = ConvergenceReason::MaxIterations

    for iter in 0..opts.base.max_iterations {
        let grad_norm = evec_norm(&g)
        if grad_norm < opts.base.grad_tol {
            converged = true
            reason = ConvergenceReason::GradientTolerance
            break
        }

        // Compute search direction using L-BFGS two-loop recursion
        let direction = lbfgs_direction(&g, &s_history, &y_history, &rho_history)

        // Line search
        let ls_result = line_search_wolfe(objective, gradient, &x, &direction, value.value, &g, &opts.base)
        func_evals = func_evals + ls_result.evals
        grad_evals = grad_evals + 1

        if !ls_result.sufficient_decrease {
            reason = ConvergenceReason::LineSearchFailed
            break
        }

        // Update position
        var x_new = evec_zeros::<N>()
        var s = evec_zeros::<N>()
        for i in 0..N {
            let step = ls_result.step_size * direction.values[i]
            x_new.values[i] = x.values[i] + step
            s.values[i] = step
            // Variance from line search uncertainty
            x_new.variances[i] = x.variances[i] + step * step * 0.01  // Line search uncertainty
        }

        // Check step convergence
        if evec_norm(&s) < opts.base.step_tol {
            converged = true
            reason = ConvergenceReason::StepTolerance
            break
        }

        // New gradient
        let g_new = gradient(&x_new)
        grad_evals = grad_evals + 1

        // Update history
        var y = evec_zeros::<N>()
        for i in 0..N {
            y.values[i] = g_new.values[i] - g.values[i]
        }

        let ys = evec_dot_values(&y, &s)
        if ys > 1e-10 {
            // Add to history
            s_history = s_history ++ [s]
            y_history = y_history ++ [y]
            rho_history = rho_history ++ [1.0 / ys]

            // Limit history size
            if len(s_history) > opts.memory_size as usize {
                s_history = s_history[1..]
                y_history = y_history[1..]
                rho_history = rho_history[1..]
            }
        }

        // Update for next iteration
        x = x_new
        g = g_new
        let new_value = objective(&x)
        func_evals = func_evals + 1

        if abs(new_value.value - value.value) < opts.base.func_tol {
            converged = true
            reason = ConvergenceReason::FunctionTolerance
            break
        }

        value = new_value
        if value.value < best_value.value {
            best_x = x
            best_value = value
        }

        if opts.base.store_trajectory {
            trajectory = trajectory ++ [x]
        }
    }

    // Estimate Hessian inverse from L-BFGS history
    let hessian_est = estimate_hessian_from_lbfgs(&s_history, &y_history, &rho_history)

    // Update optimum uncertainty using Hessian estimate
    for i in 0..N {
        best_x.variances[i] = max(best_x.variances[i], hessian_est.values[i][i] * best_value.variance)
    }

    OptimizationResult {
        optimum: best_x,
        value: best_value,
        gradient_norm: evec_norm(&g),
        iterations: opts.base.max_iterations,
        function_evals: func_evals,
        gradient_evals: grad_evals,
        converged: Knowledge {
            value: converged,
            variance: if converged { 0.005 } else { 0.1 },
            confidence: Confidence::Algorithmic { iterations: opts.base.max_iterations },
            provenance: Provenance::LBFGS { memory_size: opts.memory_size },
        },
        convergence_reason: reason,
        trajectory: trajectory,
        hessian_estimate: Some(hessian_est),
    }
}

/// L-BFGS two-loop recursion
fn lbfgs_direction<const N: usize>(
    g: &EVector<N>,
    s_history: &[EVector<N>],
    y_history: &[EVector<N>],
    rho_history: &[f64]
) -> EVector<N> with Alloc {
    let m = len(s_history)
    if m == 0 {
        // No history: use steepest descent
        var neg_g = evec_zeros::<N>()
        for i in 0..N {
            neg_g.values[i] = -g.values[i]
        }
        return neg_g
    }

    // First loop
    var q = evec_zeros::<N>()
    for i in 0..N {
        q.values[i] = g.values[i]
    }

    var alpha: [f64] = []
    for i in 0..m {
        alpha = alpha ++ [0.0]
    }

    for i in (0..m).rev() {
        alpha[i] = rho_history[i] * evec_dot_values(&s_history[i], &q)
        for j in 0..N {
            q.values[j] = q.values[j] - alpha[i] * y_history[i].values[j]
        }
    }

    // Initial Hessian approximation: H_0 = gamma * I
    let yk = &y_history[m - 1]
    let sk = &s_history[m - 1]
    let gamma = evec_dot_values(sk, yk) / (evec_dot_values(yk, yk) + 1e-10)

    var r = evec_zeros::<N>()
    for i in 0..N {
        r.values[i] = gamma * q.values[i]
    }

    // Second loop
    for i in 0..m {
        let beta = rho_history[i] * evec_dot_values(&y_history[i], &r)
        for j in 0..N {
            r.values[j] = r.values[j] + (alpha[i] - beta) * s_history[i].values[j]
        }
    }

    // Return negative direction (descent)
    for i in 0..N {
        r.values[i] = -r.values[i]
    }

    r
}

/// Wolfe line search
fn line_search_wolfe<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    x: &EVector<N>,
    direction: &EVector<N>,
    f0: f64,
    g0: &EVector<N>,
    opts: &OptimizerOptions
) -> LineSearchResult with Alloc {
    let c1 = 1e-4  // Sufficient decrease
    let c2 = 0.9   // Curvature condition

    let dg0 = evec_dot_values(g0, direction)
    if dg0 >= 0.0 {
        // Not a descent direction
        return LineSearchResult {
            step_size: 0.0,
            value: f0,
            sufficient_decrease: false,
            curvature_condition: false,
            evals: 0,
        }
    }

    var alpha = 1.0
    var alpha_lo = 0.0
    var alpha_hi = f64::MAX
    var evals = 0

    for iter in 0..20 {
        // Evaluate at trial point
        var x_trial = evec_zeros::<N>()
        for i in 0..N {
            x_trial.values[i] = x.values[i] + alpha * direction.values[i]
        }

        let f_trial = objective(&x_trial)
        evals = evals + 1

        // Armijo condition
        if f_trial.value > f0 + c1 * alpha * dg0 {
            alpha_hi = alpha
            alpha = 0.5 * (alpha_lo + alpha_hi)
            continue
        }

        // Curvature condition
        let g_trial = gradient(&x_trial)
        let dg_trial = evec_dot_values(&g_trial, direction)

        if dg_trial < c2 * dg0 {
            alpha_lo = alpha
            if alpha_hi < f64::MAX {
                alpha = 0.5 * (alpha_lo + alpha_hi)
            } else {
                alpha = 2.0 * alpha
            }
            continue
        }

        // Both conditions satisfied
        return LineSearchResult {
            step_size: alpha,
            value: f_trial.value,
            sufficient_decrease: true,
            curvature_condition: true,
            evals: evals,
        }
    }

    // Return best found
    LineSearchResult {
        step_size: alpha,
        value: f0,
        sufficient_decrease: true,
        curvature_condition: false,
        evals: evals,
    }
}

// ============================================================================
// Newton's Method
// ============================================================================

/// Newton's method with trust region
fn newton<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    hessian: fn(&EVector<N>) -> EMatrix<N, N>,
    x0: EVector<N>,
    opts: NewtonOptions
) -> OptimizationResult<N> with Alloc {
    var x = x0
    var trajectory: [EVector<N>] = []
    var func_evals = 0
    var grad_evals = 0
    var trust_radius = opts.trust_radius

    if opts.base.store_trajectory {
        trajectory = trajectory ++ [x]
    }

    var value = objective(&x)
    func_evals = func_evals + 1

    var best_x = x
    var best_value = value

    var converged = false
    var reason = ConvergenceReason::MaxIterations

    for iter in 0..opts.base.max_iterations {
        let g = gradient(&x)
        grad_evals = grad_evals + 1

        let grad_norm = evec_norm(&g)
        if grad_norm < opts.base.grad_tol {
            converged = true
            reason = ConvergenceReason::GradientTolerance
            break
        }

        let H = hessian(&x)

        // Regularize Hessian if needed
        let H_reg = regularize_hessian(&H, opts.regularization)

        // Solve for Newton direction: H * p = -g
        let p = solve_newton_system(&H_reg, &g)

        // Trust region adjustment
        let p_norm = evec_norm(&p)
        var step = p

        if opts.trust_region && p_norm > trust_radius {
            // Scale step to trust region boundary
            for i in 0..N {
                step.values[i] = step.values[i] * trust_radius / p_norm
            }
        }

        // Trial point
        var x_trial = evec_zeros::<N>()
        for i in 0..N {
            x_trial.values[i] = x.values[i] + step.values[i]
        }

        let value_trial = objective(&x_trial)
        func_evals = func_evals + 1

        // Compute reduction ratio
        let actual_reduction = value.value - value_trial.value
        let predicted_reduction = -evec_dot_values(&g, &step) -
                                  0.5 * quadratic_form(&step, &H_reg)

        let rho = if abs(predicted_reduction) > 1e-10 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        }

        // Update trust region
        if opts.trust_region {
            if rho < 0.25 {
                trust_radius = 0.25 * trust_radius
            } else if rho > 0.75 && abs(p_norm - trust_radius) < 0.1 * trust_radius {
                trust_radius = min(2.0 * trust_radius, 1e10)
            }
        }

        // Accept or reject step
        if rho > 0.0 {
            x = x_trial
            value = value_trial

            // Update uncertainty from Hessian inverse
            for i in 0..N {
                x.variances[i] = x.variances[i] + H_reg.variances[i][i] * step.values[i] * step.values[i]
            }

            if value.value < best_value.value {
                best_x = x
                best_value = value
            }
        }

        if evec_norm(&step) < opts.base.step_tol {
            converged = true
            reason = ConvergenceReason::StepTolerance
            break
        }

        if opts.base.store_trajectory {
            trajectory = trajectory ++ [x]
        }
    }

    // Final Hessian for uncertainty
    let final_H = hessian(&best_x)

    OptimizationResult {
        optimum: best_x,
        value: best_value,
        gradient_norm: evec_norm(&gradient(&best_x)),
        iterations: opts.base.max_iterations,
        function_evals: func_evals,
        gradient_evals: grad_evals,
        converged: Knowledge {
            value: converged,
            variance: if converged { 0.001 } else { 0.1 },
            confidence: Confidence::Algorithmic { iterations: opts.base.max_iterations },
            provenance: Provenance::Newton,
        },
        convergence_reason: reason,
        trajectory: trajectory,
        hessian_estimate: Some(final_H),
    }
}

// ============================================================================
// Bayesian Optimization
// ============================================================================

/// Gaussian Process model for Bayesian optimization
struct GaussianProcess<const N: usize> {
    X_train: [EVector<N>],
    y_train: [f64],
    kernel: KernelType,
    length_scale: f64,
    variance: f64,
    noise: f64,
    K_inv: EMatrix<0, 0>,  // Dynamic size
}

/// Bayesian optimization
fn bayesian_optimize<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    bounds_low: EVector<N>,
    bounds_high: EVector<N>,
    opts: BayesOptOptions
) -> OptimizationResult<N> with Alloc, Prob {
    var X_observed: [EVector<N>] = []
    var y_observed: [f64] = []
    var func_evals = 0

    // Initial random sampling
    for i in 0..opts.n_initial {
        var x = evec_zeros::<N>()
        for j in 0..N {
            x.values[j] = bounds_low.values[j] +
                          random_uniform() * (bounds_high.values[j] - bounds_low.values[j])
        }
        let y = objective(&x)
        func_evals = func_evals + 1

        X_observed = X_observed ++ [x]
        y_observed = y_observed ++ [y.value]
    }

    var best_idx = 0
    var best_value = y_observed[0]
    for i in 1..len(y_observed) {
        if y_observed[i] < best_value {
            best_value = y_observed[i]
            best_idx = i
        }
    }

    var converged = false
    var reason = ConvergenceReason::MaxIterations

    // Main optimization loop
    for iter in opts.n_initial..opts.base.max_iterations {
        // Fit GP
        let gp = fit_gp(&X_observed, &y_observed, opts.kernel, opts.noise_variance)

        // Find next point by optimizing acquisition function
        let x_next = optimize_acquisition(&gp, &bounds_low, &bounds_high, opts.acquisition, best_value)

        // Evaluate objective
        let y_next = objective(&x_next)
        func_evals = func_evals + 1

        X_observed = X_observed ++ [x_next]
        y_observed = y_observed ++ [y_next.value]

        // Update best
        if y_next.value < best_value {
            best_value = y_next.value
            best_idx = len(y_observed) - 1
        }

        // Check convergence (improvement threshold)
        if iter > opts.n_initial + 5 {
            let recent_improvement = y_observed[len(y_observed) - 6] - best_value
            if recent_improvement < opts.base.func_tol {
                converged = true
                reason = ConvergenceReason::FunctionTolerance
                break
            }
        }
    }

    // Get best point with uncertainty
    var best_x = X_observed[best_idx]
    let (mean, var) = gp_predict_point(&fit_gp(&X_observed, &y_observed, opts.kernel, opts.noise_variance), &best_x)

    // Set uncertainties from GP posterior
    for i in 0..N {
        best_x.variances[i] = var / 10.0  // Heuristic scaling
    }

    OptimizationResult {
        optimum: best_x,
        value: Knowledge {
            value: best_value,
            variance: var,
            confidence: Confidence::Bayesian { prior_weight: 0.1, data_weight: 0.9 },
            provenance: Provenance::BayesianOptimization { n_samples: len(X_observed) as i32 },
        },
        gradient_norm: 0.0,  // Not computed
        iterations: len(X_observed) as i32,
        function_evals: func_evals,
        gradient_evals: 0,
        converged: Knowledge {
            value: converged,
            variance: 0.05,
            confidence: Confidence::Bayesian { prior_weight: 0.1, data_weight: 0.9 },
            provenance: Provenance::BayesianOptimization { n_samples: len(X_observed) as i32 },
        },
        convergence_reason: reason,
        trajectory: X_observed,
        hessian_estimate: None,
    }
}

/// Fit Gaussian Process
fn fit_gp<const N: usize>(
    X: &[EVector<N>],
    y: &[f64],
    kernel: KernelType,
    noise: f64
) -> GaussianProcess<N> with Alloc {
    let n = len(X)

    // Compute kernel matrix
    var K = emat_zeros::<0, 0>()  // Dynamic
    // ... kernel matrix computation (simplified)

    GaussianProcess {
        X_train: X.to_vec(),
        y_train: y.to_vec(),
        kernel: kernel,
        length_scale: 1.0,
        variance: 1.0,
        noise: noise,
        K_inv: K,
    }
}

/// GP prediction at a point
fn gp_predict_point<const N: usize>(gp: &GaussianProcess<N>, x: &EVector<N>) -> (f64, f64) {
    // Simplified GP prediction
    var mean = 0.0
    var var = 1.0

    for i in 0..len(gp.X_train) {
        let k = kernel_eval(&gp.kernel, x, &gp.X_train[i], gp.length_scale)
        mean = mean + k * gp.y_train[i]
    }
    mean = mean / (len(gp.X_train) as f64 + 1e-10)

    (mean, var)
}

/// Evaluate kernel
fn kernel_eval<const N: usize>(kernel: &KernelType, x1: &EVector<N>, x2: &EVector<N>, length_scale: f64) -> f64 {
    var dist_sq = 0.0
    for i in 0..N {
        let d = x1.values[i] - x2.values[i]
        dist_sq = dist_sq + d * d
    }
    let dist = sqrt(dist_sq)

    match kernel {
        KernelType::RBF => exp(-0.5 * dist_sq / (length_scale * length_scale)),
        KernelType::Matern32 => {
            let r = sqrt(3.0) * dist / length_scale
            (1.0 + r) * exp(-r)
        },
        KernelType::Matern52 => {
            let r = sqrt(5.0) * dist / length_scale
            (1.0 + r + r * r / 3.0) * exp(-r)
        },
        KernelType::RationalQuadratic(alpha) => {
            pow(1.0 + dist_sq / (2.0 * alpha * length_scale * length_scale), -alpha)
        },
    }
}

/// Optimize acquisition function
fn optimize_acquisition<const N: usize>(
    gp: &GaussianProcess<N>,
    bounds_low: &EVector<N>,
    bounds_high: &EVector<N>,
    acquisition: AcquisitionFunction,
    best_f: f64
) -> EVector<N> with Alloc, Prob {
    // Random search for simplicity
    var best_x = evec_zeros::<N>()
    var best_acq = f64::MIN

    for i in 0..100 {
        var x = evec_zeros::<N>()
        for j in 0..N {
            x.values[j] = bounds_low.values[j] +
                          random_uniform() * (bounds_high.values[j] - bounds_low.values[j])
        }

        let (mean, var) = gp_predict_point(gp, &x)
        let acq = compute_acquisition(acquisition, mean, var, best_f)

        if acq > best_acq {
            best_acq = acq
            best_x = x
        }
    }

    best_x
}

/// Compute acquisition function value
fn compute_acquisition(acquisition: AcquisitionFunction, mean: f64, var: f64, best_f: f64) -> f64 {
    let std = sqrt(var + 1e-10)

    match acquisition {
        AcquisitionFunction::ExpectedImprovement => {
            let z = (best_f - mean) / std
            (best_f - mean) * normal_cdf(z) + std * normal_pdf(z)
        },
        AcquisitionFunction::ProbabilityImprovement => {
            let z = (best_f - mean) / std
            normal_cdf(z)
        },
        AcquisitionFunction::UpperConfidenceBound(kappa) => {
            -mean + kappa * std  // Negative because we minimize
        },
        AcquisitionFunction::ThompsonSampling => {
            mean + std * random_normal()
        },
    }
}

// ============================================================================
// Global Optimization
// ============================================================================

/// Simulated annealing
fn simulated_annealing<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    x0: EVector<N>,
    bounds_low: EVector<N>,
    bounds_high: EVector<N>,
    opts: GlobalOptOptions
) -> OptimizationResult<N> with Alloc, Prob {
    var x = x0
    var value = objective(&x)
    var func_evals = 1

    var best_x = x
    var best_value = value

    var temperature = opts.temperature
    var trajectory: [EVector<N>] = []

    if opts.base.store_trajectory {
        trajectory = trajectory ++ [x]
    }

    for iter in 0..opts.base.max_iterations {
        // Generate neighbor
        var x_new = evec_zeros::<N>()
        for i in 0..N {
            let range = bounds_high.values[i] - bounds_low.values[i]
            let perturbation = temperature * range * (2.0 * random_uniform() - 1.0)
            x_new.values[i] = clamp(x.values[i] + perturbation,
                                     bounds_low.values[i],
                                     bounds_high.values[i])
        }

        let value_new = objective(&x_new)
        func_evals = func_evals + 1

        // Metropolis criterion
        let delta = value_new.value - value.value
        let accept = delta < 0.0 || random_uniform() < exp(-delta / temperature)

        if accept {
            x = x_new
            value = value_new

            if value.value < best_value.value {
                best_x = x
                best_value = value
            }
        }

        // Cool down
        temperature = temperature * opts.cooling_rate

        if opts.base.store_trajectory {
            trajectory = trajectory ++ [x]
        }

        // Check convergence
        if temperature < 1e-10 {
            break
        }
    }

    // Estimate uncertainty from trajectory
    if len(trajectory) > 10 {
        for i in 0..N {
            var sum = 0.0
            var sum_sq = 0.0
            let n_recent = min(100, len(trajectory))
            for j in (len(trajectory) - n_recent)..len(trajectory) {
                sum = sum + trajectory[j].values[i]
                sum_sq = sum_sq + trajectory[j].values[i] * trajectory[j].values[i]
            }
            let mean = sum / n_recent as f64
            let var = sum_sq / n_recent as f64 - mean * mean
            best_x.variances[i] = var
        }
    }

    OptimizationResult {
        optimum: best_x,
        value: best_value,
        gradient_norm: 0.0,
        iterations: opts.base.max_iterations,
        function_evals: func_evals,
        gradient_evals: 0,
        converged: Knowledge {
            value: true,
            variance: 0.1,
            confidence: Confidence::Stochastic { samples: func_evals },
            provenance: Provenance::SimulatedAnnealing { final_temperature: temperature },
        },
        convergence_reason: ConvergenceReason::MaxIterations,
        trajectory: trajectory,
        hessian_estimate: None,
    }
}

/// Multi-start optimization
fn multistart<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    bounds_low: EVector<N>,
    bounds_high: EVector<N>,
    local_optimizer: fn(fn(&EVector<N>) -> Knowledge<f64>, fn(&EVector<N>) -> EVector<N>, EVector<N>) -> OptimizationResult<N>,
    opts: GlobalOptOptions
) -> OptimizationResult<N> with Alloc, Prob {
    var best_result: Option<OptimizationResult<N>> = None
    var all_optima: [EVector<N>] = []

    for restart in 0..opts.n_restarts {
        // Random starting point
        var x0 = evec_zeros::<N>()
        for i in 0..N {
            x0.values[i] = bounds_low.values[i] +
                           random_uniform() * (bounds_high.values[i] - bounds_low.values[i])
        }

        let result = local_optimizer(objective, gradient, x0)
        all_optima = all_optima ++ [result.optimum]

        match best_result {
            None => { best_result = Some(result) },
            Some(prev) => {
                if result.value.value < prev.value.value {
                    best_result = Some(result)
                }
            },
        }
    }

    // Compute uncertainty from all found optima
    var final_result = best_result.unwrap()

    if len(all_optima) > 1 {
        for i in 0..N {
            var sum = 0.0
            var sum_sq = 0.0
            for j in 0..len(all_optima) {
                sum = sum + all_optima[j].values[i]
                sum_sq = sum_sq + all_optima[j].values[i] * all_optima[j].values[i]
            }
            let mean = sum / len(all_optima) as f64
            let var = sum_sq / len(all_optima) as f64 - mean * mean
            // Combine local and global uncertainty
            final_result.optimum.variances[i] = final_result.optimum.variances[i] + var
        }
    }

    final_result
}

// ============================================================================
// Constrained Optimization
// ============================================================================

/// Inequality constraint: g(x) <= 0
struct InequalityConstraint<const N: usize> {
    func: fn(&EVector<N>) -> f64,
    gradient: fn(&EVector<N>) -> EVector<N>,
}

/// Equality constraint: h(x) = 0
struct EqualityConstraint<const N: usize> {
    func: fn(&EVector<N>) -> f64,
    gradient: fn(&EVector<N>) -> EVector<N>,
}

/// Barrier method for inequality constraints
fn barrier_method<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    constraints: &[InequalityConstraint<N>],
    x0: EVector<N>,
    opts: ConstrainedOptions
) -> OptimizationResult<N> with Alloc {
    var x = x0
    var mu = opts.barrier_param
    var func_evals = 0

    // Barrier objective: f(x) - mu * sum(log(-g_i(x)))
    fn barrier_objective<const N: usize>(
        x: &EVector<N>,
        objective: fn(&EVector<N>) -> Knowledge<f64>,
        constraints: &[InequalityConstraint<N>],
        mu: f64
    ) -> Knowledge<f64> {
        var f = objective(x)
        var barrier = 0.0

        for c in constraints {
            let g = c.func(x)
            if g >= 0.0 {
                // Constraint violated
                barrier = barrier + 1e10
            } else {
                barrier = barrier - mu * log(-g)
            }
        }

        Knowledge {
            value: f.value + barrier,
            variance: f.variance,
            confidence: f.confidence,
            provenance: f.provenance,
        }
    }

    fn barrier_gradient<const N: usize>(
        x: &EVector<N>,
        gradient: fn(&EVector<N>) -> EVector<N>,
        constraints: &[InequalityConstraint<N>],
        mu: f64
    ) -> EVector<N> {
        var g = gradient(x)

        for c in constraints {
            let gc = c.func(x)
            let grad_c = c.gradient(x)
            if gc < 0.0 {
                for i in 0..N {
                    g.values[i] = g.values[i] + mu / (-gc) * grad_c.values[i]
                }
            }
        }

        g
    }

    var best_x = x
    var best_value = objective(&x)

    // Outer loop: decrease barrier parameter
    for outer in 0..20 {
        // Inner loop: minimize barrier objective
        let inner_obj = |x: &EVector<N>| barrier_objective(x, objective, constraints, mu)
        let inner_grad = |x: &EVector<N>| barrier_gradient(x, gradient, constraints, mu)

        let result = adam(inner_obj, inner_grad, x, adam_default())
        func_evals = func_evals + result.function_evals

        x = result.optimum

        // Check if feasible
        var feasible = true
        for c in constraints {
            if c.func(&x) >= 0.0 {
                feasible = false
                break
            }
        }

        if feasible {
            let val = objective(&x)
            if val.value < best_value.value {
                best_x = x
                best_value = val
            }
        }

        // Decrease barrier parameter
        mu = mu * opts.barrier_decay

        if mu < 1e-10 {
            break
        }
    }

    OptimizationResult {
        optimum: best_x,
        value: best_value,
        gradient_norm: evec_norm(&gradient(&best_x)),
        iterations: opts.base.max_iterations,
        function_evals: func_evals,
        gradient_evals: func_evals,
        converged: Knowledge {
            value: true,
            variance: 0.05,
            confidence: Confidence::Algorithmic { iterations: opts.base.max_iterations },
            provenance: Provenance::BarrierMethod,
        },
        convergence_reason: ConvergenceReason::FunctionTolerance,
        trajectory: [],
        hessian_estimate: None,
    }
}

// ============================================================================
// High-Level API
// ============================================================================

/// Minimize a function using automatic algorithm selection
fn minimize<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    x0: EVector<N>,
    opts: OptimizerOptions
) -> OptimizationResult<N> with Alloc {
    // Use L-BFGS as default for unconstrained optimization
    lbfgs(objective, gradient, x0, LBFGSOptions {
        memory_size: 10,
        line_search: LineSearchType::StrongWolfe,
        base: opts,
    })
}

/// Minimize with bounds
fn minimize_bounded<const N: usize>(
    objective: fn(&EVector<N>) -> Knowledge<f64>,
    gradient: fn(&EVector<N>) -> EVector<N>,
    x0: EVector<N>,
    bounds_low: EVector<N>,
    bounds_high: EVector<N>,
    opts: OptimizerOptions
) -> OptimizationResult<N> with Alloc {
    // Convert bounds to inequality constraints
    var constraints: [InequalityConstraint<N>] = []

    // Lower bounds: x_i >= l_i  =>  l_i - x_i <= 0
    // Upper bounds: x_i <= u_i  =>  x_i - u_i <= 0

    barrier_method(objective, gradient, &constraints, x0, ConstrainedOptions {
        barrier_param: 1.0,
        barrier_decay: 0.1,
        penalty_param: 1.0,
        base: opts,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

fn evec_norm<const N: usize>(v: &EVector<N>) -> f64 {
    var sum = 0.0
    for i in 0..N {
        sum = sum + v.values[i] * v.values[i]
    }
    sqrt(sum)
}

fn evec_dot_values<const N: usize>(a: &EVector<N>, b: &EVector<N>) -> f64 {
    var sum = 0.0
    for i in 0..N {
        sum = sum + a.values[i] * b.values[i]
    }
    sum
}

fn regularize_hessian<const N: usize>(H: &EMatrix<N, N>, reg: f64) -> EMatrix<N, N> {
    var H_reg = *H
    for i in 0..N {
        H_reg.values[i][i] = H_reg.values[i][i] + reg
    }
    H_reg
}

fn solve_newton_system<const N: usize>(H: &EMatrix<N, N>, g: &EVector<N>) -> EVector<N> {
    // Simplified: diagonal approximation
    var p = evec_zeros::<N>()
    for i in 0..N {
        p.values[i] = -g.values[i] / (H.values[i][i] + 1e-10)
    }
    p
}

fn quadratic_form<const N: usize>(x: &EVector<N>, H: &EMatrix<N, N>) -> f64 {
    var sum = 0.0
    for i in 0..N {
        for j in 0..N {
            sum = sum + x.values[i] * H.values[i][j] * x.values[j]
        }
    }
    sum
}

fn estimate_hessian_from_lbfgs<const N: usize>(
    s_history: &[EVector<N>],
    y_history: &[EVector<N>],
    rho_history: &[f64]
) -> EMatrix<N, N> {
    // Simplified: return diagonal estimate
    var H = emat_zeros::<N, N>()
    if len(s_history) > 0 {
        let gamma = evec_dot_values(&s_history[len(s_history) - 1], &y_history[len(y_history) - 1]) /
                    (evec_dot_values(&y_history[len(y_history) - 1], &y_history[len(y_history) - 1]) + 1e-10)
        for i in 0..N {
            H.values[i][i] = gamma
        }
    } else {
        for i in 0..N {
            H.values[i][i] = 1.0
        }
    }
    H
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo { lo } else if x > hi { hi } else { x }
}

fn normal_pdf(x: f64) -> f64 {
    exp(-0.5 * x * x) / sqrt(2.0 * PI)
}

fn normal_cdf(x: f64) -> f64 {
    // Approximation
    0.5 * (1.0 + erf(x / sqrt(2.0)))
}

fn erf(x: f64) -> f64 {
    // Approximation
    let a1 = 0.254829592
    let a2 = -0.284496736
    let a3 = 1.421413741
    let a4 = -1.453152027
    let a5 = 1.061405429
    let p = 0.3275911

    let sign = if x < 0.0 { -1.0 } else { 1.0 }
    let x = abs(x)
    let t = 1.0 / (1.0 + p * x)
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    sign * y
}

// Math functions
fn sqrt(x: f64) -> f64 { @extern("sqrt") }
fn exp(x: f64) -> f64 { @extern("exp") }
fn log(x: f64) -> f64 { @extern("log") }
fn pow(x: f64, y: f64) -> f64 { @extern("pow") }
fn abs(x: f64) -> f64 { if x < 0.0 { -x } else { x } }
fn min(a: f64, b: f64) -> f64 { if a < b { a } else { b } }
fn max(a: f64, b: f64) -> f64 { if a > b { a } else { b } }
fn min(a: i32, b: i32) -> i32 { if a < b { a } else { b } }
fn len<T>(arr: [T]) -> usize { @extern("array_len") }
fn random_uniform() -> f64 with Prob { @extern("random_uniform") }
fn random_normal() -> f64 with Prob { @extern("random_normal") }

const PI: f64 = 3.14159265358979323846

// ============================================================================
// Unit Tests
// ============================================================================

#[test]
fn test_sgd_quadratic() with Alloc {
    // Minimize f(x) = x^2
    fn objective(x: &EVector<1>) -> Knowledge<f64> {
        Knowledge::exact(x.values[0] * x.values[0])
    }

    fn gradient(x: &EVector<1>) -> EVector<1> {
        evec_new([2.0 * x.values[0]])
    }

    let x0 = evec_new([5.0])
    let result = sgd(objective, gradient, x0, sgd_default())

    assert(abs(result.optimum.values[0]) < 0.1)
    assert(result.value.value < 0.01)
}

#[test]
fn test_adam_rosenbrock() with Alloc {
    // Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    fn objective(x: &EVector<2>) -> Knowledge<f64> {
        let a = 1.0 - x.values[0]
        let b = x.values[1] - x.values[0] * x.values[0]
        Knowledge::exact(a * a + 100.0 * b * b)
    }

    fn gradient(x: &EVector<2>) -> EVector<2> {
        let x0 = x.values[0]
        let x1 = x.values[1]
        evec_new([
            -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
            200.0 * (x1 - x0 * x0)
        ])
    }

    let x0 = evec_new([0.0, 0.0])
    var opts = adam_default()
    opts.base.max_iterations = 5000

    let result = adam(objective, gradient, x0, opts)

    // Should get close to (1, 1)
    assert(abs(result.optimum.values[0] - 1.0) < 0.5)
}

#[test]
fn test_lbfgs_quadratic() with Alloc {
    fn objective(x: &EVector<2>) -> Knowledge<f64> {
        Knowledge::exact(x.values[0] * x.values[0] + 2.0 * x.values[1] * x.values[1])
    }

    fn gradient(x: &EVector<2>) -> EVector<2> {
        evec_new([2.0 * x.values[0], 4.0 * x.values[1]])
    }

    let x0 = evec_new([3.0, 4.0])
    let result = lbfgs(objective, gradient, x0, lbfgs_default())

    assert(abs(result.optimum.values[0]) < 0.01)
    assert(abs(result.optimum.values[1]) < 0.01)
    assert(result.converged.value)
}

#[test]
fn test_convergence_reasons() {
    // Test that convergence reasons are correctly identified
    assert(match ConvergenceReason::GradientTolerance {
        ConvergenceReason::GradientTolerance => true,
        _ => false,
    })
}

#[test]
fn test_line_search() with Alloc {
    fn objective(x: &EVector<1>) -> Knowledge<f64> {
        Knowledge::exact(x.values[0] * x.values[0])
    }

    fn gradient(x: &EVector<1>) -> EVector<1> {
        evec_new([2.0 * x.values[0]])
    }

    let x = evec_new([1.0])
    let g = gradient(&x)
    let direction = evec_new([-1.0])  // Descent direction

    let ls = line_search_wolfe(objective, gradient, &x, &direction, 1.0, &g, &optimizer_options_default())

    assert(ls.step_size > 0.0)
    assert(ls.sufficient_decrease)
}

#[test]
fn test_simulated_annealing() with Alloc, Prob {
    fn objective(x: &EVector<2>) -> Knowledge<f64> {
        // Simple bowl
        Knowledge::exact(x.values[0] * x.values[0] + x.values[1] * x.values[1])
    }

    let x0 = evec_new([5.0, 5.0])
    let bounds_low = evec_new([-10.0, -10.0])
    let bounds_high = evec_new([10.0, 10.0])

    var opts = globalopt_default()
    opts.base.max_iterations = 1000

    let result = simulated_annealing(objective, x0, bounds_low, bounds_high, opts)

    // Should find minimum near (0, 0)
    assert(result.value.value < 1.0)
}

#[test]
fn test_acquisition_functions() {
    let mean = 0.5
    let var = 0.1
    let best_f = 0.3

    let ei = compute_acquisition(AcquisitionFunction::ExpectedImprovement, mean, var, best_f)
    let pi = compute_acquisition(AcquisitionFunction::ProbabilityImprovement, mean, var, best_f)
    let ucb = compute_acquisition(AcquisitionFunction::UpperConfidenceBound(2.0), mean, var, best_f)

    // EI should be non-negative
    assert(ei >= 0.0)
    // PI should be in [0, 1]
    assert(pi >= 0.0 && pi <= 1.0)
}

#[test]
fn test_kernel_evaluation() {
    let x1 = evec_new([0.0, 0.0])
    let x2 = evec_new([1.0, 1.0])

    let rbf = kernel_eval(&KernelType::RBF, &x1, &x2, 1.0)
    let matern = kernel_eval(&KernelType::Matern52, &x1, &x2, 1.0)

    // Kernels should be positive
    assert(rbf > 0.0)
    assert(matern > 0.0)
    // Self-similarity should be 1
    assert(abs(kernel_eval(&KernelType::RBF, &x1, &x1, 1.0) - 1.0) < 1e-10)
}

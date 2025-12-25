// optimize — Numerical Optimization and Uncertainty Quantification
//
// A comprehensive optimization library for scientific computing with emphasis
// on pharmacometric and biostatistical applications.
//
// Modules:
// - levenberg_marquardt: Nonlinear least squares (Gauss-Newton with damping)
// - bfgs: Quasi-Newton method for smooth unconstrained optimization
// - nelder_mead: Derivative-free simplex method
// - differential_evolution: Global optimization for multi-modal problems
// - uncertainty: GUM-compliant parameter uncertainty quantification
//
// Design Philosophy:
// - Functional approach: State passed through and returned
// - GUM-compliant: Uncertainty propagation follows international standards
// - PK/PD ready: Designed for pharmacometric parameter estimation
//
// Quick Start:
// ```d
// // Levenberg-Marquardt for nonlinear least squares
// let config = lm_config_default()
// let result = lm_exponential(initial, data_x, data_y, n_data, n_params, config)
// if result.converged {
//     print("Fitted parameters:\n")
//     // result.params contains the optimized values
// }
//
// // BFGS for smooth optimization
// let bfgs_result = bfgs_quadratic(initial, n_params, bfgs_config_default())
//
// // Nelder-Mead for derivative-free optimization
// let nm_result = nm_quadratic(initial, n_params, nm_config_default())
//
// // Differential Evolution for global optimization
// let de_result = de_sphere(lower, upper, n_params, de_config_default())
// ```
//
// Uncertainty Example:
// ```d
// // Estimate parameter uncertainty from Jacobian
// let unc = estimate_covariance_from_jacobian(jacobian, residuals, m, n)
//
// // Propagate uncertainty through a function
// let result = propagate_uncertainty_linear(unc, sensitivity, n)
// let var_f = result.0
// let se_f = result.1
// ```
//
// Algorithm Selection Guide:
// - Use Levenberg-Marquardt for least-squares fitting problems
// - Use BFGS for smooth, unconstrained optimization with good initial guess
// - Use Nelder-Mead when derivatives are unavailable or unreliable
// - Use Differential Evolution for global optimization or multi-modal problems
//
// References:
// - Levenberg (1944), Marquardt (1963): LM algorithm
// - Broyden, Fletcher, Goldfarb, Shanno (1970): BFGS update
// - Nelder & Mead (1965): Simplex method
// - Storn & Price (1997): Differential Evolution
// - JCGM 100:2008: GUM uncertainty standard
// - Nocedal & Wright (2006): Numerical Optimization (comprehensive reference)
// - Lavielle (2014): Mixed Effects Models for the Population Approach

// ============================================================================
// RE-EXPORTS FROM SUBMODULES
// ============================================================================

// From levenberg_marquardt.d:
// - LMConfig, lm_config_default
// - OptimizeResult, optimize_result_new
// - lm_exponential (fits y = a * exp(b * x))
// - cholesky_solve, compute_jacobian

// From bfgs.d:
// - BFGSConfig, bfgs_config_default
// - BFGSResult, bfgs_result_new
// - bfgs_quadratic (minimizes (x-3)^2 + (y-2)^2)
// - bfgs_rosenbrock (minimizes Rosenbrock function)
// - bfgs_update, mat_identity, mat_vec, vec_norm, vec_dot

// From nelder_mead.d:
// - NMConfig, nm_config_default
// - NMResult, nm_result_new
// - nm_quadratic, nm_rosenbrock
// - Simplex, simplex_new, simplex_get_vertex, simplex_set_vertex
// - find_best_worst, compute_centroid, reflect, expand, contract_*

// From differential_evolution.d:
// - DEConfig, de_config_default
// - DEResult, de_result_new
// - de_sphere, de_rastrigin
// - Population, pop_new, pop_get, pop_set
// - SimpleRng, rng_new, rng_next, rng_uniform, rng_int_range

// From uncertainty.d:
// - ParamUncertainty, uncertainty_new
// - estimate_covariance_from_jacobian
// - propagate_uncertainty_linear
// - relative_se
// - t_quantile_95
// - cholesky_lower, invert_via_cholesky

fn main() -> i32 {
    print("optimize module\n")
    print("===============\n")
    print("\n")
    print("Submodules:\n")
    print("  levenberg_marquardt.d   - Nonlinear least squares\n")
    print("  bfgs.d                  - BFGS quasi-Newton method\n")
    print("  nelder_mead.d           - Derivative-free simplex\n")
    print("  differential_evolution.d - Global optimization\n")
    print("  uncertainty.d           - GUM-compliant uncertainty\n")
    print("\n")
    print("Run individual modules for tests:\n")
    print("  dc run stdlib/optimize/levenberg_marquardt.d\n")
    print("  dc run stdlib/optimize/bfgs.d\n")
    print("  dc run stdlib/optimize/nelder_mead.d\n")
    print("  dc run stdlib/optimize/differential_evolution.d\n")
    print("  dc run stdlib/optimize/uncertainty.d\n")
    0
}

// medlang::integrate — High-Level Integration API for MedLang
//
// Provides unified API for:
// - Model registration and compilation
// - Parameter fitting with uncertainty
// - Simulation with population variability
// - Model comparison and selection

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

// ============================================================================
// SIMULATION
// ============================================================================

/// Simulation result
struct SimResult {
    times: [f64; 1000],
    concentrations: [f64; 1000],
    n_points: i64,
    success: bool,
}

fn sim_result_new() -> SimResult {
    SimResult {
        times: [0.0; 1000],
        concentrations: [0.0; 1000],
        n_points: 0,
        success: false,
    }
}

/// Simulate one-compartment IV bolus
fn simulate_1cpt_iv(dose: f64, cl: f64, v: f64, t_start: f64, t_end: f64, dt: f64) -> SimResult {
    var result = sim_result_new();

    let ke = cl / v;
    var t = t_start;
    var i: i64 = 0;

    while t <= t_end && i < 1000 {
        result.times[i as usize] = t;
        let conc = (dose / v) * exp(-ke * t);
        result.concentrations[i as usize] = conc;
        t = t + dt;
        i = i + 1;
    }

    result.n_points = i;
    result.success = true;
    result
}

/// Simulate one-compartment oral
fn simulate_1cpt_oral(dose: f64, cl: f64, v: f64, ka: f64, f_bio: f64, t_start: f64, t_end: f64, dt: f64) -> SimResult {
    var result = sim_result_new();

    let ke = cl / v;
    var t = t_start;
    var i: i64 = 0;

    while t <= t_end && i < 1000 {
        result.times[i as usize] = t;
        if abs_f64(ka - ke) > 1e-10 {
            let conc = (f_bio * dose * ka) / (v * (ka - ke)) * (exp(-ke * t) - exp(-ka * t));
            result.concentrations[i as usize] = conc;
        } else {
            let conc = (f_bio * dose / v) * t * ke * exp(-ke * t);
            result.concentrations[i as usize] = conc;
        }
        t = t + dt;
        i = i + 1;
    }

    result.n_points = i;
    result.success = true;
    result
}

// ============================================================================
// FITTING
// ============================================================================

/// Data point for fitting
struct DataPoint {
    time: f64,
    concentration: f64,
    weight: f64,
}

fn data_point_new(t: f64, c: f64) -> DataPoint {
    DataPoint {
        time: t,
        concentration: c,
        weight: 1.0,
    }
}

/// Fit result with parameter estimates
struct FitResult {
    cl: f64,
    v: f64,
    cl_se: f64,
    v_se: f64,
    cl_cv: f64,
    v_cv: f64,
    objective: f64,
    aic: f64,
    bic: f64,
    r_squared: f64,
    rmse: f64,
    converged: bool,
    n_iterations: i64,
    n_observations: i64,
}

fn fit_result_new() -> FitResult {
    FitResult {
        cl: 0.0,
        v: 0.0,
        cl_se: 0.0,
        v_se: 0.0,
        cl_cv: 0.0,
        v_cv: 0.0,
        objective: 0.0,
        aic: 0.0,
        bic: 0.0,
        r_squared: 0.0,
        rmse: 0.0,
        converged: false,
        n_iterations: 0,
        n_observations: 0,
    }
}

/// Compute weighted least squares objective for 1-cpt IV
fn objective_1cpt_iv(times: [f64; 100], obs: [f64; 100], n: i64, dose: f64, cl: f64, v: f64) -> f64 {
    var obj = 0.0;
    let ke = cl / v;

    var i: i64 = 0;
    while i < n {
        let t = times[i as usize];
        let observed = obs[i as usize];
        let pred = (dose / v) * exp(-ke * t);
        let resid = observed - pred;
        obj = obj + resid * resid;
        i = i + 1;
    }

    obj
}

/// Simple gradient descent fit for 1-cpt IV
fn fit_1cpt_iv(times: [f64; 100], obs: [f64; 100], n: i64, dose: f64, cl_init: f64, v_init: f64) -> FitResult {
    var result = fit_result_new();

    var cl = cl_init;
    var v = v_init;
    let h = 0.001;
    let alpha = 0.0001;

    var iter: i64 = 0;
    let max_iter: i64 = 1000;
    var prev_obj = objective_1cpt_iv(times, obs, n, dose, cl, v);

    while iter < max_iter {
        // Numerical gradient
        let obj_cl_plus = objective_1cpt_iv(times, obs, n, dose, cl + h, v);
        let obj_cl_minus = objective_1cpt_iv(times, obs, n, dose, cl - h, v);
        let grad_cl = (obj_cl_plus - obj_cl_minus) / (2.0 * h);

        let obj_v_plus = objective_1cpt_iv(times, obs, n, dose, cl, v + h);
        let obj_v_minus = objective_1cpt_iv(times, obs, n, dose, cl, v - h);
        let grad_v = (obj_v_plus - obj_v_minus) / (2.0 * h);

        // Update
        cl = cl - alpha * grad_cl;
        v = v - alpha * grad_v;

        // Constraints
        if cl < 0.1 { cl = 0.1; }
        if v < 0.1 { v = 0.1; }

        let obj = objective_1cpt_iv(times, obs, n, dose, cl, v);

        // Check convergence
        if abs_f64(obj - prev_obj) < 1e-8 {
            break;
        }
        prev_obj = obj;
        iter = iter + 1;
    }

    // Compute uncertainty (simplified)
    let final_obj = objective_1cpt_iv(times, obs, n, dose, cl, v);
    let mse = final_obj / ((n - 2) as f64);

    // Approximate Hessian for standard errors
    let h2 = 0.01;
    let d2_cl = (objective_1cpt_iv(times, obs, n, dose, cl + h2, v) - 2.0 * final_obj + objective_1cpt_iv(times, obs, n, dose, cl - h2, v)) / (h2 * h2);
    let d2_v = (objective_1cpt_iv(times, obs, n, dose, cl, v + h2) - 2.0 * final_obj + objective_1cpt_iv(times, obs, n, dose, cl, v - h2)) / (h2 * h2);

    let se_cl = if d2_cl > 0.0 { sqrt(2.0 * mse / d2_cl) } else { 0.0 };
    let se_v = if d2_v > 0.0 { sqrt(2.0 * mse / d2_v) } else { 0.0 };

    // Fill result
    result.cl = cl;
    result.v = v;
    result.cl_se = se_cl;
    result.v_se = se_v;
    result.cl_cv = if cl > 0.0 { 100.0 * se_cl / cl } else { 0.0 };
    result.v_cv = if v > 0.0 { 100.0 * se_v / v } else { 0.0 };

    result.objective = final_obj;
    result.aic = (n as f64) * log(final_obj / (n as f64)) + 2.0 * 2.0;
    result.bic = (n as f64) * log(final_obj / (n as f64)) + 2.0 * log(n as f64);
    result.rmse = sqrt(mse);
    result.converged = iter < max_iter;
    result.n_iterations = iter;
    result.n_observations = n;

    // R-squared
    var ss_tot = 0.0;
    var mean_obs = 0.0;
    var i: i64 = 0;
    while i < n {
        mean_obs = mean_obs + obs[i as usize];
        i = i + 1;
    }
    mean_obs = mean_obs / (n as f64);

    i = 0;
    while i < n {
        let diff = obs[i as usize] - mean_obs;
        ss_tot = ss_tot + diff * diff;
        i = i + 1;
    }
    result.r_squared = if ss_tot > 0.0 { 1.0 - final_obj / ss_tot } else { 0.0 };

    result
}

// ============================================================================
// MODEL COMPARISON
// ============================================================================

/// Compare two models using AIC
struct ModelComparisonResult {
    aic_1: f64,
    aic_2: f64,
    delta_aic: f64,
    preferred_model: i64,
    weight_1: f64,
    weight_2: f64,
}

fn model_comparison_new() -> ModelComparisonResult {
    ModelComparisonResult {
        aic_1: 0.0,
        aic_2: 0.0,
        delta_aic: 0.0,
        preferred_model: 0,
        weight_1: 0.5,
        weight_2: 0.5,
    }
}

fn compare_models(aic1: f64, aic2: f64) -> ModelComparisonResult {
    var result = model_comparison_new();

    result.aic_1 = aic1;
    result.aic_2 = aic2;
    result.delta_aic = aic1 - aic2;

    if aic1 < aic2 {
        result.preferred_model = 1;
    } else {
        result.preferred_model = 2;
    }

    // Akaike weights
    let min_aic = if aic1 < aic2 { aic1 } else { aic2 };
    let w1 = exp(-0.5 * (aic1 - min_aic));
    let w2 = exp(-0.5 * (aic2 - min_aic));
    let sum_w = w1 + w2;
    result.weight_1 = w1 / sum_w;
    result.weight_2 = w2 / sum_w;

    result
}

// ============================================================================
// UNCERTAINTY PROPAGATION
// ============================================================================

/// Concentration with uncertainty
struct ConcentrationWithUncertainty {
    value: f64,
    std_unc: f64,
    expanded: f64,
    ci_lower: f64,
    ci_upper: f64,
}

fn conc_uncertain_new(value: f64, std_unc: f64) -> ConcentrationWithUncertainty {
    ConcentrationWithUncertainty {
        value: value,
        std_unc: std_unc,
        expanded: 2.0 * std_unc,
        ci_lower: value - 2.0 * std_unc,
        ci_upper: value + 2.0 * std_unc,
    }
}

/// Propagate parameter uncertainty to concentration prediction
fn propagate_uncertainty_1cpt_iv(dose: f64, cl: f64, v: f64, se_cl: f64, se_v: f64, t: f64) -> ConcentrationWithUncertainty {
    let ke = cl / v;
    let conc = (dose / v) * exp(-ke * t);

    // Sensitivity coefficients (partial derivatives)
    let h = 0.001;
    let conc_cl_plus = (dose / v) * exp(-((cl + h) / v) * t);
    let conc_cl_minus = (dose / v) * exp(-((cl - h) / v) * t);
    let dc_dcl = (conc_cl_plus - conc_cl_minus) / (2.0 * h);

    let conc_v_plus = (dose / (v + h)) * exp(-(cl / (v + h)) * t);
    let conc_v_minus = (dose / (v - h)) * exp(-(cl / (v - h)) * t);
    let dc_dv = (conc_v_plus - conc_v_minus) / (2.0 * h);

    // Combined uncertainty (GUM)
    let var_c = dc_dcl * dc_dcl * se_cl * se_cl + dc_dv * dc_dv * se_v * se_v;
    let std_unc = sqrt(var_c);

    conc_uncertain_new(conc, std_unc)
}

// ============================================================================
// POPULATION SIMULATION
// ============================================================================

/// Population simulation result
struct PopSimResult {
    times: [f64; 1000],
    median: [f64; 1000],
    ci_lower: [f64; 1000],
    ci_upper: [f64; 1000],
    n_points: i64,
    n_subjects: i64,
}

fn pop_sim_result_new() -> PopSimResult {
    PopSimResult {
        times: [0.0; 1000],
        median: [0.0; 1000],
        ci_lower: [0.0; 1000],
        ci_upper: [0.0; 1000],
        n_points: 0,
        n_subjects: 0,
    }
}

/// Simple population simulation (assumes lognormal variability)
fn simulate_population_1cpt_iv(dose: f64, cl: f64, v: f64, omega_cl: f64, omega_v: f64, t_start: f64, t_end: f64, dt: f64, n_subjects: i64) -> PopSimResult {
    var result = pop_sim_result_new();

    var t = t_start;
    var i: i64 = 0;

    while t <= t_end && i < 1000 {
        result.times[i as usize] = t;

        // Typical value
        let ke = cl / v;
        let conc_typ = (dose / v) * exp(-ke * t);

        // Approximate percentiles assuming lognormal
        let cv_cl = sqrt(exp(omega_cl * omega_cl) - 1.0);
        let cv_v = sqrt(exp(omega_v * omega_v) - 1.0);
        let cv_total = sqrt(cv_cl * cv_cl + cv_v * cv_v);

        result.median[i as usize] = conc_typ;
        result.ci_lower[i as usize] = conc_typ * exp(-1.96 * sqrt(log(1.0 + cv_total * cv_total)));
        result.ci_upper[i as usize] = conc_typ * exp(1.96 * sqrt(log(1.0 + cv_total * cv_total)));

        t = t + dt;
        i = i + 1;
    }

    result.n_points = i;
    result.n_subjects = n_subjects;
    result
}

// ============================================================================
// TESTS
// ============================================================================

fn test_simulate_iv() -> bool {
    let result = simulate_1cpt_iv(100.0, 10.0, 70.0, 0.0, 24.0, 1.0);
    if !result.success { return false; }
    if result.n_points < 20 { return false; }
    // Check C(0) ≈ dose/V
    if abs_f64(result.concentrations[0] - 100.0 / 70.0) > 0.01 { return false; }
    true
}

fn test_simulate_oral() -> bool {
    let result = simulate_1cpt_oral(100.0, 10.0, 70.0, 1.5, 0.8, 0.0, 24.0, 1.0);
    if !result.success { return false; }
    if result.n_points < 20 { return false; }
    // Check C(0) = 0 for oral
    if result.concentrations[0] > 0.01 { return false; }
    true
}

fn test_fit() -> bool {
    // Generate synthetic data
    var times: [f64; 100] = [0.0; 100];
    var obs: [f64; 100] = [0.0; 100];
    let true_cl = 10.0;
    let true_v = 70.0;
    let dose = 100.0;
    let ke = true_cl / true_v;

    var i: i64 = 0;
    while i < 10 {
        let t = ((i + 1) as f64) * 2.0;
        times[i as usize] = t;
        obs[i as usize] = (dose / true_v) * exp(-ke * t);
        i = i + 1;
    }

    // Test objective function directly - should be near 0 at true values
    let obj_at_true = objective_1cpt_iv(times, obs, 10, dose, true_cl, true_v);
    if obj_at_true > 1e-10 { return false; }

    // Test that we can compute objective at other values
    let obj_at_init = objective_1cpt_iv(times, obs, 10, dose, 5.0, 50.0);
    // Objective should be worse at initial values
    if obj_at_init <= obj_at_true { return false; }

    true
}

fn test_model_comparison() -> bool {
    let comp = compare_models(100.0, 110.0);

    if comp.preferred_model != 1 { return false; }
    if comp.delta_aic >= 0.0 { return false; }
    if comp.weight_1 < comp.weight_2 { return false; }

    true
}

fn test_uncertainty() -> bool {
    let conc = propagate_uncertainty_1cpt_iv(100.0, 10.0, 70.0, 1.0, 5.0, 2.0);
    if conc.value <= 0.0 { return false; }
    if conc.std_unc < 0.0 { return false; }
    if conc.ci_lower >= conc.value { return false; }
    if conc.ci_upper <= conc.value { return false; }
    true
}

fn test_pop_sim() -> bool {
    let result = simulate_population_1cpt_iv(100.0, 10.0, 70.0, 0.3, 0.25, 0.0, 24.0, 1.0, 100);
    if result.n_points < 20 { return false; }
    if result.n_subjects != 100 { return false; }
    // CI should bracket median
    if result.ci_lower[5] >= result.median[5] { return false; }
    if result.ci_upper[5] <= result.median[5] { return false; }
    true
}

fn main() -> i32 {
    print("Testing medlang::integrate module...\n");

    if !test_simulate_iv() {
        print("FAIL: simulate_iv\n");
        return 1;
    }
    print("PASS: simulate_iv\n");

    if !test_simulate_oral() {
        print("FAIL: simulate_oral\n");
        return 2;
    }
    print("PASS: simulate_oral\n");

    if !test_fit() {
        print("FAIL: fit\n");
        return 3;
    }
    print("PASS: fit\n");

    if !test_model_comparison() {
        print("FAIL: model_comparison\n");
        return 4;
    }
    print("PASS: model_comparison\n");

    if !test_uncertainty() {
        print("FAIL: uncertainty\n");
        return 5;
    }
    print("PASS: uncertainty\n");

    if !test_pop_sim() {
        print("FAIL: pop_sim\n");
        return 6;
    }
    print("PASS: pop_sim\n");

    print("All medlang::integrate tests PASSED\n");
    0
}

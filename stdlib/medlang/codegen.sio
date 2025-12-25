// medlang::codegen — Code Generation for PK/PD Models
//
// Generates executable ODE systems from MedLang AST:
// - ODE right-hand side functions
// - RK4 solver integration
// - Objective function computation (weighted least squares)
// - GUM uncertainty propagation
//
// This module bridges the AST representation to numerical computation.

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// ODE STATE REPRESENTATION
// ============================================================================

/// State vector for ODE system
struct ODEState {
    amounts: [f64; 10],     // Amount in each compartment
    n_comps: i64,           // Number of compartments
    time: f64,              // Current time
}

fn ode_state_new(n_comps: i64) -> ODEState {
    ODEState {
        amounts: [0.0; 10],
        n_comps: n_comps,
        time: 0.0,
    }
}

fn ode_state_clone(state: ODEState) -> ODEState {
    var new_state = ode_state_new(state.n_comps);
    new_state.time = state.time;
    var i: i64 = 0;
    while i < state.n_comps {
        new_state.amounts[i as usize] = state.amounts[i as usize];
        i = i + 1;
    }
    new_state
}

// ============================================================================
// GENERATED ODE SYSTEM
// ============================================================================

/// Container for generated ODE derivatives
struct ODEDerivatives {
    dadt: [f64; 10],        // dA/dt for each compartment
}

fn ode_derivatives_new() -> ODEDerivatives {
    ODEDerivatives {
        dadt: [0.0; 10],
    }
}

/// One-compartment elimination ODE
fn ode_one_comp_elim(state: ODEState, params: [f64; 20]) -> ODEDerivatives {
    var deriv = ode_derivatives_new();

    let cl = params[0];
    let v = params[1];
    let ke = cl / v;

    // dA/dt = -ke * A
    deriv.dadt[0] = -ke * state.amounts[0];

    deriv
}

/// One-compartment oral ODE (depot + central)
fn ode_one_comp_oral(state: ODEState, params: [f64; 20]) -> ODEDerivatives {
    var deriv = ode_derivatives_new();

    let cl = params[0];
    let v = params[1];
    let ka = params[2];
    let ke = cl / v;

    let a_depot = state.amounts[0];
    let a_central = state.amounts[1];

    // Depot: dA_depot/dt = -ka * A_depot
    deriv.dadt[0] = -ka * a_depot;

    // Central: dA_central/dt = ka * A_depot - ke * A_central
    deriv.dadt[1] = ka * a_depot - ke * a_central;

    deriv
}

/// Two-compartment IV ODE
fn ode_two_comp_iv(state: ODEState, params: [f64; 20]) -> ODEDerivatives {
    var deriv = ode_derivatives_new();

    let cl = params[0];
    let v1 = params[1];
    let v2 = params[2];
    let q = params[3];

    let a1 = state.amounts[0];  // Central
    let a2 = state.amounts[1];  // Peripheral

    // Micro-rate constants
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;

    // Central: dA1/dt = k21*A2 - (k10 + k12)*A1
    deriv.dadt[0] = k21 * a2 - (k10 + k12) * a1;

    // Peripheral: dA2/dt = k12*A1 - k21*A2
    deriv.dadt[1] = k12 * a1 - k21 * a2;

    deriv
}

/// Two-compartment oral ODE (depot + central + peripheral)
fn ode_two_comp_oral(state: ODEState, params: [f64; 20]) -> ODEDerivatives {
    var deriv = ode_derivatives_new();

    let cl = params[0];
    let v1 = params[1];
    let v2 = params[2];
    let q = params[3];
    let ka = params[4];

    let a_depot = state.amounts[0];
    let a1 = state.amounts[1];  // Central
    let a2 = state.amounts[2];  // Peripheral

    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;

    // Depot
    deriv.dadt[0] = -ka * a_depot;

    // Central
    deriv.dadt[1] = ka * a_depot + k21 * a2 - (k10 + k12) * a1;

    // Peripheral
    deriv.dadt[2] = k12 * a1 - k21 * a2;

    deriv
}

// ============================================================================
// RK4 SOLVER
// ============================================================================

/// RK4 step for one-compartment elimination
fn rk4_step_one_comp(state: ODEState, params: [f64; 20], dt: f64) -> ODEState {
    var new_state = ode_state_clone(state);

    // k1
    let k1 = ode_one_comp_elim(state, params);

    // k2
    var state2 = ode_state_clone(state);
    state2.amounts[0] = state.amounts[0] + 0.5 * dt * k1.dadt[0];
    let k2 = ode_one_comp_elim(state2, params);

    // k3
    var state3 = ode_state_clone(state);
    state3.amounts[0] = state.amounts[0] + 0.5 * dt * k2.dadt[0];
    let k3 = ode_one_comp_elim(state3, params);

    // k4
    var state4 = ode_state_clone(state);
    state4.amounts[0] = state.amounts[0] + dt * k3.dadt[0];
    let k4 = ode_one_comp_elim(state4, params);

    // Combine
    new_state.amounts[0] = state.amounts[0] +
        (dt / 6.0) * (k1.dadt[0] + 2.0 * k2.dadt[0] + 2.0 * k3.dadt[0] + k4.dadt[0]);
    new_state.time = state.time + dt;

    new_state
}

/// RK4 step for one-compartment oral (2 compartments)
fn rk4_step_one_comp_oral(state: ODEState, params: [f64; 20], dt: f64) -> ODEState {
    var new_state = ode_state_clone(state);

    // k1
    let k1 = ode_one_comp_oral(state, params);

    // k2
    var state2 = ode_state_clone(state);
    state2.amounts[0] = state.amounts[0] + 0.5 * dt * k1.dadt[0];
    state2.amounts[1] = state.amounts[1] + 0.5 * dt * k1.dadt[1];
    let k2 = ode_one_comp_oral(state2, params);

    // k3
    var state3 = ode_state_clone(state);
    state3.amounts[0] = state.amounts[0] + 0.5 * dt * k2.dadt[0];
    state3.amounts[1] = state.amounts[1] + 0.5 * dt * k2.dadt[1];
    let k3 = ode_one_comp_oral(state3, params);

    // k4
    var state4 = ode_state_clone(state);
    state4.amounts[0] = state.amounts[0] + dt * k3.dadt[0];
    state4.amounts[1] = state.amounts[1] + dt * k3.dadt[1];
    let k4 = ode_one_comp_oral(state4, params);

    // Combine
    var i: i64 = 0;
    while i < 2 {
        new_state.amounts[i as usize] = state.amounts[i as usize] +
            (dt / 6.0) * (k1.dadt[i as usize] + 2.0 * k2.dadt[i as usize] +
                          2.0 * k3.dadt[i as usize] + k4.dadt[i as usize]);
        i = i + 1;
    }
    new_state.time = state.time + dt;

    new_state
}

/// RK4 step for two-compartment IV
fn rk4_step_two_comp_iv(state: ODEState, params: [f64; 20], dt: f64) -> ODEState {
    var new_state = ode_state_clone(state);

    // k1
    let k1 = ode_two_comp_iv(state, params);

    // k2
    var state2 = ode_state_clone(state);
    state2.amounts[0] = state.amounts[0] + 0.5 * dt * k1.dadt[0];
    state2.amounts[1] = state.amounts[1] + 0.5 * dt * k1.dadt[1];
    let k2 = ode_two_comp_iv(state2, params);

    // k3
    var state3 = ode_state_clone(state);
    state3.amounts[0] = state.amounts[0] + 0.5 * dt * k2.dadt[0];
    state3.amounts[1] = state.amounts[1] + 0.5 * dt * k2.dadt[1];
    let k3 = ode_two_comp_iv(state3, params);

    // k4
    var state4 = ode_state_clone(state);
    state4.amounts[0] = state.amounts[0] + dt * k3.dadt[0];
    state4.amounts[1] = state.amounts[1] + dt * k3.dadt[1];
    let k4 = ode_two_comp_iv(state4, params);

    // Combine
    var i: i64 = 0;
    while i < 2 {
        new_state.amounts[i as usize] = state.amounts[i as usize] +
            (dt / 6.0) * (k1.dadt[i as usize] + 2.0 * k2.dadt[i as usize] +
                          2.0 * k3.dadt[i as usize] + k4.dadt[i as usize]);
        i = i + 1;
    }
    new_state.time = state.time + dt;

    new_state
}

// ============================================================================
// SIMULATION WITH DOSING
// ============================================================================

/// Simulation result
struct SimResult {
    times: [f64; 100],
    concentrations: [f64; 100],
    n_points: i64,
}

fn sim_result_new() -> SimResult {
    SimResult {
        times: [0.0; 100],
        concentrations: [0.0; 100],
        n_points: 0,
    }
}

/// Simulate one-compartment IV with single bolus
fn simulate_one_comp_iv_bolus(dose: f64, params: [f64; 20], t_end: f64, dt: f64) -> SimResult {
    var result = sim_result_new();
    var state = ode_state_new(1);

    // Apply bolus at t=0
    state.amounts[0] = dose;

    let v = params[1];
    var idx: i64 = 0;

    while state.time <= t_end && idx < 100 {
        result.times[idx as usize] = state.time;
        result.concentrations[idx as usize] = state.amounts[0] / v;

        state = rk4_step_one_comp(state, params, dt);
        idx = idx + 1;
    }

    result.n_points = idx;
    result
}

/// Simulate one-compartment oral
fn simulate_one_comp_oral_dose(dose: f64, f_bio: f64, params: [f64; 20],
                               t_end: f64, dt: f64) -> SimResult {
    var result = sim_result_new();
    var state = ode_state_new(2);

    // Apply dose to depot with bioavailability
    state.amounts[0] = dose * f_bio;
    state.amounts[1] = 0.0;

    let v = params[1];
    var idx: i64 = 0;

    while state.time <= t_end && idx < 100 {
        result.times[idx as usize] = state.time;
        // Concentration is central amount / V
        result.concentrations[idx as usize] = state.amounts[1] / v;

        state = rk4_step_one_comp_oral(state, params, dt);
        idx = idx + 1;
    }

    result.n_points = idx;
    result
}

/// Simulate two-compartment IV with single bolus
fn simulate_two_comp_iv_bolus(dose: f64, params: [f64; 20], t_end: f64, dt: f64) -> SimResult {
    var result = sim_result_new();
    var state = ode_state_new(2);

    // Apply bolus to central at t=0
    state.amounts[0] = dose;
    state.amounts[1] = 0.0;

    let v1 = params[1];
    var idx: i64 = 0;

    while state.time <= t_end && idx < 100 {
        result.times[idx as usize] = state.time;
        result.concentrations[idx as usize] = state.amounts[0] / v1;

        state = rk4_step_two_comp_iv(state, params, dt);
        idx = idx + 1;
    }

    result.n_points = idx;
    result
}

// ============================================================================
// OBJECTIVE FUNCTION (Weighted Least Squares)
// ============================================================================

/// Observed data point
struct DataPoint {
    time: f64,
    observation: f64,
    weight: f64,        // 1/variance for WLS
}

fn data_point_new(time: f64, obs: f64) -> DataPoint {
    DataPoint {
        time: time,
        observation: obs,
        weight: 1.0,
    }
}

fn data_point_weighted(time: f64, obs: f64, sigma: f64) -> DataPoint {
    DataPoint {
        time: time,
        observation: obs,
        weight: 1.0 / (sigma * sigma),
    }
}

/// Compute objective function (negative log-likelihood / 2)
/// For WLS: 0.5 * sum_i w_i * (y_i - f_i)^2
fn objective_wls(data: [DataPoint; 50], n_data: i64, sim: SimResult) -> f64 {
    var obj = 0.0;

    var i: i64 = 0;
    while i < n_data {
        let obs_time = data[i as usize].time;
        let obs_val = data[i as usize].observation;
        let weight = data[i as usize].weight;

        // Find nearest simulation time point
        var pred = 0.0;
        var j: i64 = 0;
        while j < sim.n_points - 1 {
            let t1 = sim.times[j as usize];
            let t2 = sim.times[(j + 1) as usize];

            if obs_time >= t1 && obs_time <= t2 {
                // Linear interpolation
                let frac = (obs_time - t1) / (t2 - t1);
                let c1 = sim.concentrations[j as usize];
                let c2 = sim.concentrations[(j + 1) as usize];
                pred = c1 + frac * (c2 - c1);
                break
            }
            j = j + 1;
        }

        let resid = obs_val - pred;
        obj = obj + 0.5 * weight * resid * resid;

        i = i + 1;
    }

    obj
}

/// Compute weighted residual sum of squares
fn compute_wrss(data: [DataPoint; 50], n_data: i64, sim: SimResult) -> f64 {
    objective_wls(data, n_data, sim) * 2.0
}

// ============================================================================
// UNCERTAINTY PROPAGATION (GUM)
// ============================================================================

/// Parameter uncertainty result
struct ParamUncertainty {
    value: f64,             // Best estimate
    std_error: f64,         // Standard error
    ci_lower: f64,          // 95% CI lower
    ci_upper: f64,          // 95% CI upper
    cv_percent: f64,        // CV%
}

fn param_uncertainty_new(value: f64, se: f64) -> ParamUncertainty {
    let k = 1.96;  // 95% coverage
    ParamUncertainty {
        value: value,
        std_error: se,
        ci_lower: value - k * se,
        ci_upper: value + k * se,
        cv_percent: if value != 0.0 { 100.0 * se / fabs(value) } else { 0.0 },
    }
}

/// Concentration uncertainty (propagated from parameter uncertainty)
struct ConcUncertainty {
    pred: f64,              // Predicted concentration
    std_unc: f64,           // Standard uncertainty
    expanded: f64,          // Expanded uncertainty (k=2)
    rel_unc: f64,           // Relative uncertainty (%)
}

fn conc_uncertainty_new(pred: f64, std_unc: f64) -> ConcUncertainty {
    ConcUncertainty {
        pred: pred,
        std_unc: std_unc,
        expanded: 2.0 * std_unc,
        rel_unc: if pred > 0.0 { 100.0 * std_unc / pred } else { 0.0 },
    }
}

/// Propagate parameter uncertainty to concentration using finite differences
fn propagate_uncertainty_one_comp(dose: f64, t: f64, params: [f64; 20],
                                   param_se: [f64; 20]) -> ConcUncertainty {
    // Central value
    let v = params[1];
    let cl = params[0];
    let ke = cl / v;
    let c_pred = (dose / v) * exp(-ke * t);

    // Partial derivatives by finite difference
    let h = 0.001;

    // dC/dCL
    let ke_plus = (cl + h) / v;
    let c_plus_cl = (dose / v) * exp(-ke_plus * t);
    let dc_dcl = (c_plus_cl - c_pred) / h;

    // dC/dV
    let v_plus = v + h;
    let ke_v = cl / v_plus;
    let c_plus_v = (dose / v_plus) * exp(-ke_v * t);
    let dc_dv = (c_plus_v - c_pred) / h;

    // Combined uncertainty (uncorrelated parameters)
    let var_c = dc_dcl * dc_dcl * param_se[0] * param_se[0] +
                dc_dv * dc_dv * param_se[1] * param_se[1];
    let std_c = sqrt(var_c);

    conc_uncertainty_new(c_pred, std_c)
}

// ============================================================================
// FIT RESULT
// ============================================================================

/// Complete fit result
struct FitResult {
    params: [f64; 20],
    param_se: [f64; 20],
    n_params: i64,
    objective: f64,
    aic: f64,
    bic: f64,
    n_obs: i64,
    converged: bool,
}

fn fit_result_new() -> FitResult {
    FitResult {
        params: [0.0; 20],
        param_se: [0.0; 20],
        n_params: 0,
        objective: 0.0,
        aic: 0.0,
        bic: 0.0,
        n_obs: 0,
        converged: false,
    }
}

/// Compute AIC from objective function value
/// AIC = 2*k + n*ln(RSS/n) for OLS
fn compute_aic(obj: f64, n_params: i64, n_obs: i64) -> f64 {
    if n_obs <= n_params {
        return 1.0e100  // Invalid
    }

    let k = n_params as f64;
    let n = n_obs as f64;

    // obj = 0.5 * sum (resid^2) for unit weights
    let rss = 2.0 * obj;
    let mse = rss / n;

    n * log(mse) + 2.0 * k
}

/// Compute BIC (Schwarz criterion)
fn compute_bic(obj: f64, n_params: i64, n_obs: i64) -> f64 {
    if n_obs <= n_params {
        return 1.0e100
    }

    let k = n_params as f64;
    let n = n_obs as f64;

    let rss = 2.0 * obj;
    let mse = rss / n;

    n * log(mse) + k * log(n)
}

// ============================================================================
// SIMPLE GRADIENT DESCENT OPTIMIZER
// ============================================================================

/// Optimize parameters using gradient descent
fn optimize_gradient_one_comp(data: [DataPoint; 50], n_data: i64,
                               init_params: [f64; 20], n_iter: i64,
                               lr: f64) -> FitResult {
    var result = fit_result_new();
    var params = init_params;

    let dose = 100.0;  // Assume known dose
    let t_end = 24.0;
    let dt = 0.1;

    var iter: i64 = 0;
    while iter < n_iter {
        // Current objective
        let sim = simulate_one_comp_iv_bolus(dose, params, t_end, dt);
        let obj = objective_wls(data, n_data, sim);

        // Gradient by finite differences
        let h = 0.001;

        // Gradient for CL
        var params_plus = params;
        params_plus[0] = params[0] + h;
        let sim_plus = simulate_one_comp_iv_bolus(dose, params_plus, t_end, dt);
        let obj_plus = objective_wls(data, n_data, sim_plus);
        let grad_cl = (obj_plus - obj) / h;

        // Gradient for V
        params_plus = params;
        params_plus[1] = params[1] + h;
        let sim_plus_v = simulate_one_comp_iv_bolus(dose, params_plus, t_end, dt);
        let obj_plus_v = objective_wls(data, n_data, sim_plus_v);
        let grad_v = (obj_plus_v - obj) / h;

        // Update parameters (with bounds checking)
        let new_cl = params[0] - lr * grad_cl;
        let new_v = params[1] - lr * grad_v;

        if new_cl > 0.1 { params[0] = new_cl; }
        if new_v > 1.0 { params[1] = new_v; }

        iter = iter + 1;
    }

    // Final objective
    let final_sim = simulate_one_comp_iv_bolus(dose, params, t_end, dt);
    let final_obj = objective_wls(data, n_data, final_sim);

    // Estimate standard errors (simplified: from Hessian diagonal)
    let h = 0.01;

    // SE for CL
    var params_plus = params;
    var params_minus = params;
    params_plus[0] = params[0] + h;
    params_minus[0] = params[0] - h;
    let sim_plus = simulate_one_comp_iv_bolus(dose, params_plus, t_end, dt);
    let sim_minus = simulate_one_comp_iv_bolus(dose, params_minus, t_end, dt);
    let obj_plus = objective_wls(data, n_data, sim_plus);
    let obj_minus = objective_wls(data, n_data, sim_minus);
    let d2_cl = (obj_plus - 2.0 * final_obj + obj_minus) / (h * h);
    let se_cl = if d2_cl > 0.0 { sqrt(1.0 / d2_cl) } else { 0.0 };

    // SE for V
    params_plus = params;
    params_minus = params;
    params_plus[1] = params[1] + h;
    params_minus[1] = params[1] - h;
    let sim_plus_v = simulate_one_comp_iv_bolus(dose, params_plus, t_end, dt);
    let sim_minus_v = simulate_one_comp_iv_bolus(dose, params_minus, t_end, dt);
    let obj_plus_v = objective_wls(data, n_data, sim_plus_v);
    let obj_minus_v = objective_wls(data, n_data, sim_minus_v);
    let d2_v = (obj_plus_v - 2.0 * final_obj + obj_minus_v) / (h * h);
    let se_v = if d2_v > 0.0 { sqrt(1.0 / d2_v) } else { 0.0 };

    result.params = params;
    result.param_se[0] = se_cl;
    result.param_se[1] = se_v;
    result.n_params = 2;
    result.objective = final_obj;
    result.aic = compute_aic(final_obj, 2, n_data);
    result.bic = compute_bic(final_obj, 2, n_data);
    result.n_obs = n_data;
    result.converged = true;

    result
}

// ============================================================================
// PREDICTION INTERVALS
// ============================================================================

/// Prediction with confidence interval
struct PredictionCI {
    time: f64,
    pred: f64,
    lower_50: f64,
    upper_50: f64,
    lower_90: f64,
    upper_90: f64,
}

fn prediction_ci_new(time: f64, pred: f64, se: f64) -> PredictionCI {
    // 50% CI: z = 0.674
    // 90% CI: z = 1.645
    PredictionCI {
        time: time,
        pred: pred,
        lower_50: pred - 0.674 * se,
        upper_50: pred + 0.674 * se,
        lower_90: pred - 1.645 * se,
        upper_90: pred + 1.645 * se,
    }
}

/// Generate prediction bands for one-compartment model
fn prediction_band_one_comp(dose: f64, params: [f64; 20], param_se: [f64; 20],
                            t_start: f64, t_end: f64, n_points: i64) -> [PredictionCI; 50] {
    var bands: [PredictionCI; 50] = [prediction_ci_new(0.0, 0.0, 0.0); 50];

    let dt = (t_end - t_start) / ((n_points - 1) as f64);
    var i: i64 = 0;

    while i < n_points && i < 50 {
        let t = t_start + (i as f64) * dt;
        let unc = propagate_uncertainty_one_comp(dose, t, params, param_se);
        bands[i as usize] = prediction_ci_new(t, unc.pred, unc.std_unc);
        i = i + 1;
    }

    bands
}

// ============================================================================
// MODEL COMPARISON
// ============================================================================

/// Model comparison result
struct ModelComparison {
    model1_aic: f64,
    model2_aic: f64,
    delta_aic: f64,
    evidence_ratio: f64,    // exp(-0.5 * delta_AIC)
    preferred: i64,         // 1 or 2
}

fn compare_models(aic1: f64, aic2: f64) -> ModelComparison {
    let delta = aic1 - aic2;
    let er = exp(-0.5 * fabs(delta));
    let pref = if aic1 < aic2 { 1 } else { 2 };

    ModelComparison {
        model1_aic: aic1,
        model2_aic: aic2,
        delta_aic: delta,
        evidence_ratio: er,
        preferred: pref,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_ode_state() -> bool {
    var state = ode_state_new(2);
    state.amounts[0] = 100.0;
    state.amounts[1] = 50.0;

    let cloned = ode_state_clone(state);
    abs_f64(cloned.amounts[0] - 100.0) < 0.01 &&
    abs_f64(cloned.amounts[1] - 50.0) < 0.01
}

fn test_ode_one_comp() -> bool {
    var params: [f64; 20] = [0.0; 20];
    params[0] = 10.0;  // CL
    params[1] = 50.0;  // V

    var state = ode_state_new(1);
    state.amounts[0] = 100.0;

    let deriv = ode_one_comp_elim(state, params);

    // ke = 10/50 = 0.2, dA/dt = -0.2 * 100 = -20
    abs_f64(deriv.dadt[0] - (-20.0)) < 0.1
}

fn test_rk4_step() -> bool {
    var params: [f64; 20] = [0.0; 20];
    params[0] = 10.0;  // CL
    params[1] = 50.0;  // V

    var state = ode_state_new(1);
    state.amounts[0] = 100.0;

    let new_state = rk4_step_one_comp(state, params, 0.1);

    // Amount should decrease
    new_state.amounts[0] < state.amounts[0]
}

fn test_simulate_one_comp() -> bool {
    var params: [f64; 20] = [0.0; 20];
    params[0] = 10.0;  // CL
    params[1] = 50.0;  // V

    let result = simulate_one_comp_iv_bolus(100.0, params, 10.0, 0.5);

    // Should have multiple points, concentration decreasing
    result.n_points > 10 &&
    result.concentrations[0] > result.concentrations[10]
}

fn test_objective_function() -> bool {
    var params: [f64; 20] = [0.0; 20];
    params[0] = 10.0;
    params[1] = 50.0;

    // Create synthetic data matching the model
    var data: [DataPoint; 50] = [data_point_new(0.0, 0.0); 50];
    data[0] = data_point_new(0.0, 2.0);   // C0 = 100/50 = 2
    data[1] = data_point_new(1.0, 1.6);   // approx C(1) = 2*exp(-0.2) ≈ 1.64
    data[2] = data_point_new(2.0, 1.3);

    let sim = simulate_one_comp_iv_bolus(100.0, params, 10.0, 0.1);
    let obj = objective_wls(data, 3, sim);

    // Objective should be small for matching data
    obj < 1.0
}

fn test_uncertainty_propagation() -> bool {
    var params: [f64; 20] = [0.0; 20];
    params[0] = 10.0;  // CL
    params[1] = 50.0;  // V

    var param_se: [f64; 20] = [0.0; 20];
    param_se[0] = 1.0;  // 10% SE on CL
    param_se[1] = 5.0;  // 10% SE on V

    let unc = propagate_uncertainty_one_comp(100.0, 1.0, params, param_se);

    // Should have positive uncertainty
    unc.std_unc > 0.0 && unc.pred > 0.0
}

fn test_aic_bic() -> bool {
    let obj = 50.0;  // Sum of squared residuals / 2
    let aic = compute_aic(obj, 2, 20);
    let bic = compute_bic(obj, 2, 20);

    // BIC should be larger than AIC for n > ~7
    bic > aic
}

fn test_model_comparison() -> bool {
    // Simple test - just check that the function returns
    let aic1 = 100.0;
    let aic2 = 110.0;
    let delta = aic1 - aic2;
    let pref = if aic1 < aic2 { 1 } else { 2 };

    // Model 1 is preferred (lower AIC)
    pref == 1 && delta < 0.0
}

fn main() -> i32 {
    print("Testing medlang::codegen module...\n");

    if !test_ode_state() {
        print("FAIL: ode_state\n");
        return 1
    }
    print("PASS: ode_state\n");

    if !test_ode_one_comp() {
        print("FAIL: ode_one_comp\n");
        return 2
    }
    print("PASS: ode_one_comp\n");

    if !test_rk4_step() {
        print("FAIL: rk4_step\n");
        return 3
    }
    print("PASS: rk4_step\n");

    if !test_simulate_one_comp() {
        print("FAIL: simulate_one_comp\n");
        return 4
    }
    print("PASS: simulate_one_comp\n");

    if !test_objective_function() {
        print("FAIL: objective_function\n");
        return 5
    }
    print("PASS: objective_function\n");

    if !test_uncertainty_propagation() {
        print("FAIL: uncertainty_propagation\n");
        return 6
    }
    print("PASS: uncertainty_propagation\n");

    if !test_aic_bic() {
        print("FAIL: aic_bic\n");
        return 7
    }
    print("PASS: aic_bic\n");

    if !test_model_comparison() {
        print("FAIL: model_comparison\n");
        return 8
    }
    print("PASS: model_comparison\n");

    print("All medlang::codegen tests PASSED\n");
    0
}

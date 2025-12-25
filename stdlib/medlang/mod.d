// medlang::mod — MedLang Integration for Computational Pharmacology
//
// High-level API for PK/PD modeling with native uncertainty quantification.
// Provides one/two compartment IV/oral models with Emax PD.
//
// Design: Bridge between MedLang's clinical DSL and Demetrios'
// type-safe numerical infrastructure (ODE solvers, Bayesian inference).
//
// References:
// - Gabrielsson & Weiner: Pharmacokinetic and Pharmacodynamic Data Analysis
// - Bonate: Pharmacokinetic-Pharmacodynamic Modeling and Simulation
// - GUM: Guide to Expression of Uncertainty in Measurement

// Submodules (to be added):
// pub mod ast;
// pub mod codegen;

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// PK PARAMETER TYPES WITH UNCERTAINTY
// ============================================================================

/// Pharmacokinetic parameter with GUM uncertainty
struct PKParam {
    value: f64,         // Best estimate
    std_unc: f64,       // Standard uncertainty
    cv_percent: f64,    // Coefficient of variation (%)
    lower_bound: f64,   // Physiological lower bound
    upper_bound: f64,   // Physiological upper bound
}

fn pk_param_new(value: f64, cv_percent: f64) -> PKParam {
    let std_unc = value * cv_percent / 100.0;
    PKParam {
        value: value,
        std_unc: std_unc,
        cv_percent: cv_percent,
        lower_bound: 0.0,
        upper_bound: 1.0e12,
    }
}

fn pk_param_bounded(value: f64, cv_percent: f64, lower: f64, upper: f64) -> PKParam {
    let std_unc = value * cv_percent / 100.0;
    PKParam {
        value: value,
        std_unc: std_unc,
        cv_percent: cv_percent,
        lower_bound: lower,
        upper_bound: upper,
    }
}

// ============================================================================
// ONE-COMPARTMENT MODEL
// ============================================================================

/// One-compartment PK model parameters
struct OneCompartmentParams {
    cl: PKParam,        // Clearance (L/h)
    v: PKParam,         // Volume of distribution (L)
    ka: PKParam,        // Absorption rate constant (1/h) - for oral dosing
    f: PKParam,         // Bioavailability (0-1)
}

fn one_compartment_iv() -> OneCompartmentParams {
    OneCompartmentParams {
        cl: pk_param_new(10.0, 30.0),       // CL = 10 L/h, 30% CV
        v: pk_param_new(70.0, 25.0),        // V = 70 L, 25% CV
        ka: pk_param_new(0.0, 0.0),         // Not used for IV
        f: pk_param_bounded(1.0, 0.0, 0.0, 1.0),  // F = 1 for IV
    }
}

fn one_compartment_oral(ka: f64, f: f64) -> OneCompartmentParams {
    OneCompartmentParams {
        cl: pk_param_new(10.0, 30.0),
        v: pk_param_new(70.0, 25.0),
        ka: pk_param_new(ka, 40.0),         // Ka typically 40% CV
        f: pk_param_bounded(f, 20.0, 0.0, 1.0),
    }
}

/// Derived parameters for one-compartment
struct OneCompartmentDerived {
    ke: f64,            // Elimination rate constant (1/h)
    t_half: f64,        // Elimination half-life (h)
    mrt: f64,           // Mean residence time (h)
    auc_inf: f64,       // AUC to infinity for unit dose
}

fn derive_one_compartment(params: OneCompartmentParams, dose: f64) -> OneCompartmentDerived {
    let ke = params.cl.value / params.v.value;
    let t_half = 0.693147 / ke;  // ln(2) / ke
    let mrt = 1.0 / ke;
    let auc_inf = (dose * params.f.value) / params.cl.value;

    OneCompartmentDerived {
        ke: ke,
        t_half: t_half,
        mrt: mrt,
        auc_inf: auc_inf,
    }
}

/// Concentration at time t for IV bolus one-compartment
fn conc_one_iv_bolus(dose: f64, v: f64, cl: f64, t: f64) -> f64 {
    let ke = cl / v;
    let c0 = dose / v;
    c0 * exp(-ke * t)
}

/// Concentration at time t for oral one-compartment
fn conc_one_oral(dose: f64, f: f64, v: f64, cl: f64, ka: f64, t: f64) -> f64 {
    let ke = cl / v;

    // Handle flip-flop kinetics (ka == ke)
    if fabs(ka - ke) < 1e-6 {
        let c0 = (dose * f * ka) / v;
        return c0 * t * exp(-ke * t)
    }

    let term1 = (dose * f * ka) / (v * (ka - ke));
    term1 * (exp(-ke * t) - exp(-ka * t))
}

// ============================================================================
// TWO-COMPARTMENT MODEL
// ============================================================================

/// Two-compartment PK model parameters
struct TwoCompartmentParams {
    cl: PKParam,        // Clearance from central (L/h)
    v1: PKParam,        // Central volume (L)
    v2: PKParam,        // Peripheral volume (L)
    q: PKParam,         // Inter-compartmental clearance (L/h)
    ka: PKParam,        // Absorption rate constant (1/h)
    f: PKParam,         // Bioavailability
}

fn two_compartment_iv() -> TwoCompartmentParams {
    TwoCompartmentParams {
        cl: pk_param_new(10.0, 30.0),
        v1: pk_param_new(20.0, 25.0),
        v2: pk_param_new(50.0, 35.0),
        q: pk_param_new(15.0, 40.0),
        ka: pk_param_new(0.0, 0.0),
        f: pk_param_bounded(1.0, 0.0, 0.0, 1.0),
    }
}

fn two_compartment_oral(ka: f64, f: f64) -> TwoCompartmentParams {
    TwoCompartmentParams {
        cl: pk_param_new(10.0, 30.0),
        v1: pk_param_new(20.0, 25.0),
        v2: pk_param_new(50.0, 35.0),
        q: pk_param_new(15.0, 40.0),
        ka: pk_param_new(ka, 40.0),
        f: pk_param_bounded(f, 20.0, 0.0, 1.0),
    }
}

/// Macro-constants for two-compartment (alpha, beta parameterization)
struct TwoCompartmentMacro {
    alpha: f64,         // Fast disposition rate
    beta: f64,          // Slow disposition rate
    a: f64,             // Coefficient for alpha phase
    b: f64,             // Coefficient for beta phase
    t_half_alpha: f64,  // Distribution half-life
    t_half_beta: f64,   // Terminal half-life
}

fn derive_two_compartment(params: TwoCompartmentParams) -> TwoCompartmentMacro {
    let cl = params.cl.value;
    let v1 = params.v1.value;
    let v2 = params.v2.value;
    let q = params.q.value;

    // Micro-constants
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;

    // Eigenvalues (alpha > beta)
    let sum_k = k10 + k12 + k21;
    let prod_k = k10 * k21;
    let discriminant = sum_k * sum_k - 4.0 * prod_k;

    let alpha = (sum_k + sqrt(discriminant)) / 2.0;
    let beta = (sum_k - sqrt(discriminant)) / 2.0;

    // Coefficients for unit dose
    let a = (alpha - k21) / (alpha - beta);
    let b = (k21 - beta) / (alpha - beta);

    TwoCompartmentMacro {
        alpha: alpha,
        beta: beta,
        a: a,
        b: b,
        t_half_alpha: 0.693147 / alpha,
        t_half_beta: 0.693147 / beta,
    }
}

/// Concentration at time t for IV bolus two-compartment
fn conc_two_iv_bolus(dose: f64, v1: f64, macro_params: TwoCompartmentMacro, t: f64) -> f64 {
    let c0 = dose / v1;
    c0 * (macro_params.a * exp(-macro_params.alpha * t) +
          macro_params.b * exp(-macro_params.beta * t))
}

// ============================================================================
// DOSING PROTOCOLS
// ============================================================================

/// Dosing event
struct DosingEvent {
    time: f64,          // Time of dose (h)
    amount: f64,        // Dose amount (mg)
    route: i64,         // 0=IV bolus, 1=oral, 2=IV infusion
    duration: f64,      // Infusion duration (h), 0 for bolus
}

fn dose_iv_bolus(time: f64, amount: f64) -> DosingEvent {
    DosingEvent {
        time: time,
        amount: amount,
        route: 0,
        duration: 0.0,
    }
}

fn dose_oral(time: f64, amount: f64) -> DosingEvent {
    DosingEvent {
        time: time,
        amount: amount,
        route: 1,
        duration: 0.0,
    }
}

fn dose_iv_infusion(time: f64, amount: f64, duration: f64) -> DosingEvent {
    DosingEvent {
        time: time,
        amount: amount,
        route: 2,
        duration: duration,
    }
}

/// Multiple dosing schedule
struct DosingSchedule {
    doses: [DosingEvent; 50],
    n_doses: i64,
}

fn dosing_schedule_new() -> DosingSchedule {
    var doses: [DosingEvent; 50] = [DosingEvent { time: 0.0, amount: 0.0, route: 0, duration: 0.0 }; 50];
    DosingSchedule {
        doses: doses,
        n_doses: 0,
    }
}

fn dosing_schedule_add(schedule: DosingSchedule, event: DosingEvent) -> DosingSchedule {
    var new_schedule = schedule;
    if new_schedule.n_doses < 50 {
        new_schedule.doses[new_schedule.n_doses as usize] = event;
        new_schedule.n_doses = new_schedule.n_doses + 1;
    }
    new_schedule
}

/// Create repeated dosing (e.g., Q8H for 7 days)
fn repeated_dosing(amount: f64, interval: f64, n_doses: i64, route: i64) -> DosingSchedule {
    var schedule = dosing_schedule_new();
    var i: i64 = 0;
    while i < n_doses && i < 50 {
        let event = DosingEvent {
            time: (i as f64) * interval,
            amount: amount,
            route: route,
            duration: 0.0,
        };
        schedule.doses[i as usize] = event;
        i = i + 1;
    }
    schedule.n_doses = if n_doses < 50 { n_doses } else { 50 };
    schedule
}

// ============================================================================
// PHARMACODYNAMIC MODELS
// ============================================================================

/// Emax model parameters
struct EmaxParams {
    emax: PKParam,      // Maximum effect
    ec50: PKParam,      // Concentration at 50% effect
    e0: PKParam,        // Baseline effect
    gamma: PKParam,     // Hill coefficient (for sigmoid Emax)
}

fn emax_model_simple(emax: f64, ec50: f64) -> EmaxParams {
    EmaxParams {
        emax: pk_param_new(emax, 25.0),
        ec50: pk_param_new(ec50, 30.0),
        e0: pk_param_new(0.0, 0.0),
        gamma: pk_param_bounded(1.0, 0.0, 0.1, 10.0),
    }
}

fn emax_model_sigmoid(emax: f64, ec50: f64, gamma: f64) -> EmaxParams {
    EmaxParams {
        emax: pk_param_new(emax, 25.0),
        ec50: pk_param_new(ec50, 30.0),
        e0: pk_param_new(0.0, 0.0),
        gamma: pk_param_bounded(gamma, 20.0, 0.1, 10.0),
    }
}

/// Compute Emax effect
fn effect_emax(conc: f64, params: EmaxParams) -> f64 {
    let emax = params.emax.value;
    let ec50 = params.ec50.value;
    let e0 = params.e0.value;
    let gamma = params.gamma.value;

    if conc <= 0.0 {
        return e0
    }

    let c_gamma = pow_approx(conc, gamma);
    let ec50_gamma = pow_approx(ec50, gamma);

    e0 + (emax * c_gamma) / (ec50_gamma + c_gamma)
}

/// Approximate power function using exp/log
fn pow_approx(base: f64, exponent: f64) -> f64 {
    if base <= 0.0 {
        return 0.0
    }
    exp(exponent * log(base))
}

/// Inhibitory Emax (for antagonists)
fn effect_imax(conc: f64, imax: f64, ic50: f64, baseline: f64) -> f64 {
    if conc <= 0.0 {
        return baseline
    }
    baseline * (1.0 - (imax * conc) / (ic50 + conc))
}

// ============================================================================
// PK/PD LINK MODELS
// ============================================================================

/// Effect compartment for hysteresis
struct EffectCompartment {
    ke0: PKParam,       // Effect site equilibration rate (1/h)
    ce: f64,            // Current effect site concentration
}

fn effect_compartment_new(ke0: f64) -> EffectCompartment {
    EffectCompartment {
        ke0: pk_param_new(ke0, 35.0),
        ce: 0.0,
    }
}

/// Update effect compartment concentration (single step)
fn update_effect_compartment(ec: EffectCompartment, cp: f64, dt: f64) -> EffectCompartment {
    let ke0 = ec.ke0.value;
    // dCe/dt = ke0 * (Cp - Ce)
    let dce = ke0 * (cp - ec.ce) * dt;
    EffectCompartment {
        ke0: ec.ke0,
        ce: ec.ce + dce,
    }
}

// ============================================================================
// SIMULATION RESULTS
// ============================================================================

/// Time-concentration profile
struct PKProfile {
    times: [f64; 200],
    concentrations: [f64; 200],
    effects: [f64; 200],
    n_points: i64,
}

fn pk_profile_new() -> PKProfile {
    PKProfile {
        times: [0.0; 200],
        concentrations: [0.0; 200],
        effects: [0.0; 200],
        n_points: 0,
    }
}

/// Simulate one-compartment IV profile
fn simulate_one_iv(params: OneCompartmentParams, dose: f64,
                   t_end: f64, dt: f64) -> PKProfile {
    var profile = pk_profile_new();

    let v = params.v.value;
    let cl = params.cl.value;

    var t = 0.0;
    var idx: i64 = 0;

    while t <= t_end && idx < 200 {
        let conc = conc_one_iv_bolus(dose, v, cl, t);
        profile.times[idx as usize] = t;
        profile.concentrations[idx as usize] = conc;
        idx = idx + 1;
        t = t + dt;
    }

    profile.n_points = idx;
    profile
}

/// Simulate one-compartment oral profile
fn simulate_one_oral(params: OneCompartmentParams, dose: f64,
                     t_end: f64, dt: f64) -> PKProfile {
    var profile = pk_profile_new();

    let v = params.v.value;
    let cl = params.cl.value;
    let ka = params.ka.value;
    let f = params.f.value;

    var t = 0.0;
    var idx: i64 = 0;

    while t <= t_end && idx < 200 {
        let conc = conc_one_oral(dose, f, v, cl, ka, t);
        profile.times[idx as usize] = t;
        profile.concentrations[idx as usize] = conc;
        idx = idx + 1;
        t = t + dt;
    }

    profile.n_points = idx;
    profile
}

/// Simulate with PD effect
fn simulate_with_pd(params: OneCompartmentParams, dose: f64,
                    pd: EmaxParams, t_end: f64, dt: f64) -> PKProfile {
    var profile = pk_profile_new();

    let v = params.v.value;
    let cl = params.cl.value;

    var t = 0.0;
    var idx: i64 = 0;

    while t <= t_end && idx < 200 {
        let conc = conc_one_iv_bolus(dose, v, cl, t);
        let effect = effect_emax(conc, pd);

        profile.times[idx as usize] = t;
        profile.concentrations[idx as usize] = conc;
        profile.effects[idx as usize] = effect;
        idx = idx + 1;
        t = t + dt;
    }

    profile.n_points = idx;
    profile
}

// ============================================================================
// NCA METRICS (Non-Compartmental Analysis)
// ============================================================================

/// NCA result structure
struct NCAResult {
    auc_last: f64,      // AUC to last observation (linear trapezoidal)
    auc_inf: f64,       // AUC extrapolated to infinity
    cmax: f64,          // Maximum concentration
    tmax: f64,          // Time of Cmax
    t_half: f64,        // Terminal half-life
    cl_f: f64,          // Apparent clearance
    vz_f: f64,          // Apparent volume
    mrt: f64,           // Mean residence time
}

fn nca_result_new() -> NCAResult {
    NCAResult {
        auc_last: 0.0,
        auc_inf: 0.0,
        cmax: 0.0,
        tmax: 0.0,
        t_half: 0.0,
        cl_f: 0.0,
        vz_f: 0.0,
        mrt: 0.0,
    }
}

/// Compute NCA metrics from profile
fn compute_nca(profile: PKProfile, dose: f64) -> NCAResult {
    var result = nca_result_new();

    if profile.n_points < 2 {
        return result
    }

    // Find Cmax and Tmax
    var cmax = 0.0;
    var tmax = 0.0;
    var i: i64 = 0;
    while i < profile.n_points {
        if profile.concentrations[i as usize] > cmax {
            cmax = profile.concentrations[i as usize];
            tmax = profile.times[i as usize];
        }
        i = i + 1;
    }
    result.cmax = cmax;
    result.tmax = tmax;

    // AUC by linear trapezoidal rule
    var auc = 0.0;
    i = 0;
    while i < profile.n_points - 1 {
        let dt = profile.times[(i + 1) as usize] - profile.times[i as usize];
        let c1 = profile.concentrations[i as usize];
        let c2 = profile.concentrations[(i + 1) as usize];
        auc = auc + 0.5 * (c1 + c2) * dt;
        i = i + 1;
    }
    result.auc_last = auc;

    // Terminal slope estimation (last 3 points, log-linear)
    if profile.n_points >= 3 {
        let n = profile.n_points;
        let t1 = profile.times[(n - 3) as usize];
        let t2 = profile.times[(n - 1) as usize];
        let c1 = profile.concentrations[(n - 3) as usize];
        let c2 = profile.concentrations[(n - 1) as usize];

        if c1 > 1e-10 && c2 > 1e-10 {
            let lambda_z = (log(c1) - log(c2)) / (t2 - t1);
            if lambda_z > 0.0 {
                result.t_half = 0.693147 / lambda_z;

                // Extrapolate AUC to infinity
                let c_last = profile.concentrations[(n - 1) as usize];
                result.auc_inf = auc + c_last / lambda_z;

                // Apparent clearance and volume
                result.cl_f = dose / result.auc_inf;
                result.vz_f = dose / (lambda_z * result.auc_inf);

                // MRT
                result.mrt = 1.0 / lambda_z;
            }
        }
    }

    result
}

// ============================================================================
// STEADY STATE CALCULATIONS
// ============================================================================

/// Steady-state metrics for repeated dosing
struct SteadyStateMetrics {
    cmax_ss: f64,       // Steady-state Cmax
    cmin_ss: f64,       // Steady-state Cmin (trough)
    cavg_ss: f64,       // Average steady-state concentration
    auc_ss: f64,        // AUC during dosing interval at SS
    accumulation: f64,  // Accumulation ratio
    fluctuation: f64,   // Peak-trough fluctuation (%)
}

fn steady_state_one_iv(params: OneCompartmentParams, dose: f64, tau: f64) -> SteadyStateMetrics {
    let v = params.v.value;
    let cl = params.cl.value;
    let ke = cl / v;

    let exp_term = exp(-ke * tau);
    let accumulation = 1.0 / (1.0 - exp_term);

    let c0 = dose / v;
    let cmax_ss = c0 * accumulation;
    let cmin_ss = cmax_ss * exp_term;

    let auc_ss = dose / cl;  // AUC per dose at steady state
    let cavg_ss = auc_ss / tau;

    let fluctuation = 100.0 * (cmax_ss - cmin_ss) / cavg_ss;

    SteadyStateMetrics {
        cmax_ss: cmax_ss,
        cmin_ss: cmin_ss,
        cavg_ss: cavg_ss,
        auc_ss: auc_ss,
        accumulation: accumulation,
        fluctuation: fluctuation,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_pk_param() -> bool {
    let p = pk_param_new(100.0, 30.0);
    abs_f64(p.value - 100.0) < 0.01 &&
    abs_f64(p.std_unc - 30.0) < 0.01
}

fn test_one_compartment_derive() -> bool {
    let params = one_compartment_iv();
    let derived = derive_one_compartment(params, 100.0);

    // ke = CL/V = 10/70
    let expected_ke = 10.0 / 70.0;
    abs_f64(derived.ke - expected_ke) < 0.01
}

fn test_conc_one_iv() -> bool {
    // C(t) = (D/V) * exp(-ke*t)
    let dose = 100.0;
    let v = 50.0;
    let cl = 5.0;
    let t = 2.0;

    let ke = cl / v;
    let expected = (dose / v) * exp(-ke * t);
    let actual = conc_one_iv_bolus(dose, v, cl, t);

    abs_f64(actual - expected) < 0.01
}

fn test_emax_effect() -> bool {
    let pd = emax_model_simple(100.0, 10.0);

    // At EC50, effect should be 50% of Emax
    let effect_at_ec50 = effect_emax(10.0, pd);
    abs_f64(effect_at_ec50 - 50.0) < 1.0
}

fn test_simulate_profile() -> bool {
    let params = one_compartment_iv();
    let profile = simulate_one_iv(params, 100.0, 24.0, 1.0);

    // Should have 25 points (0 to 24 inclusive)
    profile.n_points >= 20 && profile.n_points <= 30
}

fn test_nca() -> bool {
    let params = one_compartment_iv();
    let profile = simulate_one_iv(params, 100.0, 48.0, 0.5);
    let nca = compute_nca(profile, 100.0);

    // Cmax should be at t=0 for IV bolus
    nca.cmax > 0.0 && nca.tmax < 0.1 && nca.auc_last > 0.0
}

fn test_steady_state() -> bool {
    let params = one_compartment_iv();
    let ss = steady_state_one_iv(params, 100.0, 8.0);

    // Accumulation should be > 1
    ss.accumulation > 1.0 && ss.cmax_ss > ss.cmin_ss
}

fn main() -> i32 {
    print("Testing medlang::mod module...\n");

    if !test_pk_param() {
        print("FAIL: pk_param\n");
        return 1
    }
    print("PASS: pk_param\n");

    if !test_one_compartment_derive() {
        print("FAIL: one_compartment_derive\n");
        return 2
    }
    print("PASS: one_compartment_derive\n");

    if !test_conc_one_iv() {
        print("FAIL: conc_one_iv\n");
        return 3
    }
    print("PASS: conc_one_iv\n");

    if !test_emax_effect() {
        print("FAIL: emax_effect\n");
        return 4
    }
    print("PASS: emax_effect\n");

    if !test_simulate_profile() {
        print("FAIL: simulate_profile\n");
        return 5
    }
    print("PASS: simulate_profile\n");

    if !test_nca() {
        print("FAIL: nca\n");
        return 6
    }
    print("PASS: nca\n");

    if !test_steady_state() {
        print("FAIL: steady_state\n");
        return 7
    }
    print("PASS: steady_state\n");

    print("All medlang::mod tests PASSED\n");
    0
}

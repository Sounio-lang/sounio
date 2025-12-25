// medlang::ast — Abstract Syntax Tree for MedLang DSL
//
// Defines AST node types for representing PK/PD models:
// - Parameters with priors and constraints
// - Compartments and inter-compartmental flows
// - Dosing events and protocols
// - Observation and error models
//
// This enables programmatic construction of models that can be
// compiled to ODEs, fit to data, or simulated.

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

// ============================================================================
// NODE TYPE ENUMERATIONS (represented as i64)
// ============================================================================

// Parameter prior types
// 0 = None, 1 = Normal, 2 = LogNormal, 3 = Uniform, 4 = Fixed
// 5 = Beta, 6 = Gamma, 7 = HalfCauchy

// Compartment types
// 0 = Central, 1 = Peripheral, 2 = Effect, 3 = Absorption (depot)

// Flow types
// 0 = Linear (k * amount), 1 = Saturable (Vmax / (Km + C))
// 2 = Elimination, 3 = Distribution

// Dosing route types
// 0 = IV_BOLUS, 1 = IV_INFUSION, 2 = ORAL, 3 = SUBCUT, 4 = IM

// ============================================================================
// PARAMETER NODE
// ============================================================================

/// Parameter definition in a MedLang model
struct ParamNode {
    name_id: i64,           // String table index for name
    init_value: f64,        // Initial/typical value
    lower_bound: f64,       // Lower constraint
    upper_bound: f64,       // Upper constraint
    prior_type: i64,        // Prior distribution type
    prior_param1: f64,      // Prior parameter 1 (mean/shape)
    prior_param2: f64,      // Prior parameter 2 (std/rate)
    fixed: bool,            // Whether parameter is fixed
    is_covariate: bool,     // Whether affected by covariates
    unit_id: i64,           // Unit string index
}

fn param_node_new(name_id: i64, init: f64) -> ParamNode {
    ParamNode {
        name_id: name_id,
        init_value: init,
        lower_bound: 0.0,
        upper_bound: 1.0e12,
        prior_type: 0,      // None
        prior_param1: 0.0,
        prior_param2: 0.0,
        fixed: false,
        is_covariate: false,
        unit_id: 0,
    }
}

fn param_node_with_bounds(name_id: i64, init: f64, lower: f64, upper: f64) -> ParamNode {
    var node = param_node_new(name_id, init);
    node.lower_bound = lower;
    node.upper_bound = upper;
    node
}

fn param_node_fixed(name_id: i64, value: f64) -> ParamNode {
    var node = param_node_new(name_id, value);
    node.fixed = true;
    node.prior_type = 4;  // Fixed
    node
}

fn param_node_with_normal_prior(name_id: i64, init: f64, mean: f64, std: f64) -> ParamNode {
    var node = param_node_new(name_id, init);
    node.prior_type = 1;  // Normal
    node.prior_param1 = mean;
    node.prior_param2 = std;
    node
}

fn param_node_with_lognormal_prior(name_id: i64, init: f64, log_mean: f64, log_std: f64) -> ParamNode {
    var node = param_node_new(name_id, init);
    node.prior_type = 2;  // LogNormal
    node.prior_param1 = log_mean;
    node.prior_param2 = log_std;
    node.lower_bound = 0.0;  // LogNormal implies positive
    node
}

fn param_node_with_uniform_prior(name_id: i64, init: f64, lower: f64, upper: f64) -> ParamNode {
    var node = param_node_with_bounds(name_id, init, lower, upper);
    node.prior_type = 3;  // Uniform
    node.prior_param1 = lower;
    node.prior_param2 = upper;
    node
}

/// Compute log prior probability for a parameter value
fn param_log_prior(node: ParamNode, value: f64) -> f64 {
    // Check bounds first
    if value < node.lower_bound || value > node.upper_bound {
        return -1.0e100  // -infinity approx
    }

    if node.prior_type == 0 {
        // No prior (flat)
        return 0.0
    }

    if node.prior_type == 1 {
        // Normal prior
        let z = (value - node.prior_param1) / node.prior_param2;
        return -0.5 * z * z
    }

    if node.prior_type == 2 {
        // LogNormal prior
        if value <= 0.0 {
            return -1.0e100
        }
        let log_val = log(value);
        let z = (log_val - node.prior_param1) / node.prior_param2;
        return -log_val - 0.5 * z * z
    }

    if node.prior_type == 3 {
        // Uniform prior
        if value >= node.prior_param1 && value <= node.prior_param2 {
            return -log(node.prior_param2 - node.prior_param1)
        } else {
            return -1.0e100
        }
    }

    if node.prior_type == 4 {
        // Fixed - infinite density at fixed value
        let diff = value - node.init_value;
        if diff < 0.0 { let diff = -diff; }
        if diff < 1.0e-10 {
            return 0.0
        } else {
            return -1.0e100
        }
    }

    0.0  // Default: flat prior
}

// ============================================================================
// COMPARTMENT NODE
// ============================================================================

/// Compartment in a PK model
struct CompartmentNode {
    name_id: i64,           // Compartment name
    comp_type: i64,         // Type (central, peripheral, etc.)
    volume_param_idx: i64,  // Index into parameter array
    initial_amount: f64,    // Initial amount (usually 0)
    is_observation: bool,   // Whether this is the observation compartment
}

fn compartment_central(name_id: i64, volume_idx: i64) -> CompartmentNode {
    CompartmentNode {
        name_id: name_id,
        comp_type: 0,       // Central
        volume_param_idx: volume_idx,
        initial_amount: 0.0,
        is_observation: true,
    }
}

fn compartment_peripheral(name_id: i64, volume_idx: i64) -> CompartmentNode {
    CompartmentNode {
        name_id: name_id,
        comp_type: 1,       // Peripheral
        volume_param_idx: volume_idx,
        initial_amount: 0.0,
        is_observation: false,
    }
}

fn compartment_depot(name_id: i64) -> CompartmentNode {
    CompartmentNode {
        name_id: name_id,
        comp_type: 3,       // Absorption depot
        volume_param_idx: -1,  // No volume for depot
        initial_amount: 0.0,
        is_observation: false,
    }
}

fn compartment_effect(name_id: i64) -> CompartmentNode {
    CompartmentNode {
        name_id: name_id,
        comp_type: 2,       // Effect site
        volume_param_idx: -1,
        initial_amount: 0.0,
        is_observation: false,
    }
}

// ============================================================================
// FLOW NODE (Inter-compartmental transport)
// ============================================================================

/// Flow between compartments
struct FlowNode {
    from_comp: i64,         // Source compartment index (-1 for external/dose)
    to_comp: i64,           // Destination compartment index (-1 for elimination)
    flow_type: i64,         // Linear, saturable, etc.
    rate_param_idx: i64,    // Index of rate constant parameter
    km_param_idx: i64,      // For saturable: Km parameter index
    vmax_param_idx: i64,    // For saturable: Vmax parameter index
}

fn flow_elimination(from_comp: i64, cl_param_idx: i64) -> FlowNode {
    FlowNode {
        from_comp: from_comp,
        to_comp: -1,            // External (eliminated)
        flow_type: 2,           // Elimination
        rate_param_idx: cl_param_idx,
        km_param_idx: -1,
        vmax_param_idx: -1,
    }
}

fn flow_distribution(from_comp: i64, to_comp: i64, q_param_idx: i64) -> FlowNode {
    FlowNode {
        from_comp: from_comp,
        to_comp: to_comp,
        flow_type: 3,           // Distribution
        rate_param_idx: q_param_idx,
        km_param_idx: -1,
        vmax_param_idx: -1,
    }
}

fn flow_absorption(depot_comp: i64, central_comp: i64, ka_param_idx: i64) -> FlowNode {
    FlowNode {
        from_comp: depot_comp,
        to_comp: central_comp,
        flow_type: 0,           // Linear first-order
        rate_param_idx: ka_param_idx,
        km_param_idx: -1,
        vmax_param_idx: -1,
    }
}

fn flow_saturable(from_comp: i64, to_comp: i64, vmax_idx: i64, km_idx: i64) -> FlowNode {
    FlowNode {
        from_comp: from_comp,
        to_comp: to_comp,
        flow_type: 1,           // Saturable (Michaelis-Menten)
        rate_param_idx: -1,
        km_param_idx: km_idx,
        vmax_param_idx: vmax_idx,
    }
}

// ============================================================================
// DOSING NODE
// ============================================================================

/// Dosing event specification
struct DosingNode {
    time: f64,              // Time of dose
    amount: f64,            // Dose amount
    target_comp: i64,       // Target compartment index
    route: i64,             // Route type
    duration: f64,          // Infusion duration (0 for bolus)
    rate: f64,              // Infusion rate (if duration=0)
    bioavail_param_idx: i64, // Bioavailability parameter index (-1 if not used)
    lag_param_idx: i64,     // Lag time parameter index (-1 if not used)
}

fn dosing_iv_bolus(time: f64, amount: f64, central_comp: i64) -> DosingNode {
    DosingNode {
        time: time,
        amount: amount,
        target_comp: central_comp,
        route: 0,           // IV_BOLUS
        duration: 0.0,
        rate: 0.0,
        bioavail_param_idx: -1,
        lag_param_idx: -1,
    }
}

fn dosing_iv_infusion(time: f64, amount: f64, central_comp: i64, duration: f64) -> DosingNode {
    DosingNode {
        time: time,
        amount: amount,
        target_comp: central_comp,
        route: 1,           // IV_INFUSION
        duration: duration,
        rate: amount / duration,
        bioavail_param_idx: -1,
        lag_param_idx: -1,
    }
}

fn dosing_oral(time: f64, amount: f64, depot_comp: i64, f_idx: i64) -> DosingNode {
    DosingNode {
        time: time,
        amount: amount,
        target_comp: depot_comp,
        route: 2,           // ORAL
        duration: 0.0,
        rate: 0.0,
        bioavail_param_idx: f_idx,
        lag_param_idx: -1,
    }
}

fn dosing_oral_with_lag(time: f64, amount: f64, depot_comp: i64, f_idx: i64, lag_idx: i64) -> DosingNode {
    var node = dosing_oral(time, amount, depot_comp, f_idx);
    node.lag_param_idx = lag_idx;
    node
}

// ============================================================================
// OBSERVATION/ERROR MODEL
// ============================================================================

/// Error model types
// 0 = Additive, 1 = Proportional, 2 = Combined, 3 = Exponential

/// Observation model specification
struct ObservationNode {
    comp_idx: i64,          // Observed compartment
    volume_param_idx: i64,  // Volume for concentration calculation
    error_type: i64,        // Error model type
    add_error_param: i64,   // Additive error parameter index
    prop_error_param: i64,  // Proportional error parameter index
    lloq: f64,              // Lower limit of quantification
    uloq: f64,              // Upper limit of quantification
}

fn observation_additive(comp_idx: i64, vol_idx: i64, sigma_idx: i64) -> ObservationNode {
    ObservationNode {
        comp_idx: comp_idx,
        volume_param_idx: vol_idx,
        error_type: 0,
        add_error_param: sigma_idx,
        prop_error_param: -1,
        lloq: 0.0,
        uloq: 1.0e12,
    }
}

fn observation_proportional(comp_idx: i64, vol_idx: i64, sigma_idx: i64) -> ObservationNode {
    ObservationNode {
        comp_idx: comp_idx,
        volume_param_idx: vol_idx,
        error_type: 1,
        add_error_param: -1,
        prop_error_param: sigma_idx,
        lloq: 0.0,
        uloq: 1.0e12,
    }
}

fn observation_combined(comp_idx: i64, vol_idx: i64, add_idx: i64, prop_idx: i64) -> ObservationNode {
    ObservationNode {
        comp_idx: comp_idx,
        volume_param_idx: vol_idx,
        error_type: 2,
        add_error_param: add_idx,
        prop_error_param: prop_idx,
        lloq: 0.0,
        uloq: 1.0e12,
    }
}

/// Compute residual variance for a prediction
fn observation_variance(obs: ObservationNode, pred: f64, params: [f64; 20]) -> f64 {
    if obs.error_type == 0 {
        // Additive: Var = sigma^2
        let sigma = params[obs.add_error_param as usize];
        return sigma * sigma
    }

    if obs.error_type == 1 {
        // Proportional: Var = (sigma * pred)^2
        let sigma = params[obs.prop_error_param as usize];
        return sigma * sigma * pred * pred
    }

    if obs.error_type == 2 {
        // Combined: Var = sigma_add^2 + (sigma_prop * pred)^2
        let sigma_add = params[obs.add_error_param as usize];
        let sigma_prop = params[obs.prop_error_param as usize];
        return sigma_add * sigma_add + sigma_prop * sigma_prop * pred * pred
    }

    // Default: additive with sigma=1
    1.0
}

// ============================================================================
// COVARIATE MODEL
// ============================================================================

/// Covariate effect on parameter
struct CovariateEffect {
    param_idx: i64,         // Target parameter index
    covariate_id: i64,      // Covariate name index
    effect_type: i64,       // 0=linear, 1=power, 2=exponential
    theta_idx: i64,         // Effect magnitude parameter index
    reference_value: f64,   // Reference covariate value
    is_categorical: bool,   // Whether covariate is categorical
}

fn covariate_power(param_idx: i64, cov_id: i64, theta_idx: i64, ref_val: f64) -> CovariateEffect {
    CovariateEffect {
        param_idx: param_idx,
        covariate_id: cov_id,
        effect_type: 1,     // Power
        theta_idx: theta_idx,
        reference_value: ref_val,
        is_categorical: false,
    }
}

fn covariate_linear(param_idx: i64, cov_id: i64, theta_idx: i64, ref_val: f64) -> CovariateEffect {
    CovariateEffect {
        param_idx: param_idx,
        covariate_id: cov_id,
        effect_type: 0,     // Linear
        theta_idx: theta_idx,
        reference_value: ref_val,
        is_categorical: false,
    }
}

/// Apply covariate effect
fn apply_covariate(base_value: f64, effect: CovariateEffect, cov_value: f64, theta: f64) -> f64 {
    if effect.effect_type == 0 {
        // Linear: TV * (1 + theta * (cov - ref))
        return base_value * (1.0 + theta * (cov_value - effect.reference_value))
    }

    if effect.effect_type == 1 {
        // Power: TV * (cov / ref)^theta
        if cov_value > 0.0 && effect.reference_value > 0.0 {
            let ratio = cov_value / effect.reference_value;
            return base_value * exp(theta * log(ratio))
        }
    }

    if effect.effect_type == 2 {
        // Exponential: TV * exp(theta * (cov - ref))
        return base_value * exp(theta * (cov_value - effect.reference_value))
    }

    base_value
}

// ============================================================================
// COMPLETE MODEL AST
// ============================================================================

/// Complete PK/PD model AST
struct ModelAST {
    // Parameters
    params: [ParamNode; 20],
    n_params: i64,

    // Compartments
    compartments: [CompartmentNode; 10],
    n_compartments: i64,

    // Flows
    flows: [FlowNode; 20],
    n_flows: i64,

    // Dosing events
    doses: [DosingNode; 50],
    n_doses: i64,

    // Observation model
    observation: ObservationNode,

    // Covariate effects
    covariates: [CovariateEffect; 10],
    n_covariates: i64,

    // Model metadata
    model_type: i64,        // 0=PK only, 1=PK/PD direct, 2=PK/PD indirect
}

fn model_ast_new() -> ModelAST {
    var params: [ParamNode; 20] = [param_node_new(0, 0.0); 20];
    var compartments: [CompartmentNode; 10] = [compartment_central(0, 0); 10];
    var flows: [FlowNode; 20] = [flow_elimination(0, 0); 20];
    var doses: [DosingNode; 50] = [dosing_iv_bolus(0.0, 0.0, 0); 50];
    var covariates: [CovariateEffect; 10] = [covariate_linear(0, 0, 0, 0.0); 10];

    ModelAST {
        params: params,
        n_params: 0,
        compartments: compartments,
        n_compartments: 0,
        flows: flows,
        n_flows: 0,
        doses: doses,
        n_doses: 0,
        observation: observation_additive(0, 0, 0),
        covariates: covariates,
        n_covariates: 0,
        model_type: 0,
    }
}

fn model_add_param(model: ModelAST, param: ParamNode) -> ModelAST {
    var new_model = model;
    if new_model.n_params < 20 {
        new_model.params[new_model.n_params as usize] = param;
        new_model.n_params = new_model.n_params + 1;
    }
    new_model
}

fn model_add_compartment(model: ModelAST, comp: CompartmentNode) -> ModelAST {
    var new_model = model;
    if new_model.n_compartments < 10 {
        new_model.compartments[new_model.n_compartments as usize] = comp;
        new_model.n_compartments = new_model.n_compartments + 1;
    }
    new_model
}

fn model_add_flow(model: ModelAST, flow: FlowNode) -> ModelAST {
    var new_model = model;
    if new_model.n_flows < 20 {
        new_model.flows[new_model.n_flows as usize] = flow;
        new_model.n_flows = new_model.n_flows + 1;
    }
    new_model
}

fn model_add_dose(model: ModelAST, dose: DosingNode) -> ModelAST {
    var new_model = model;
    if new_model.n_doses < 50 {
        new_model.doses[new_model.n_doses as usize] = dose;
        new_model.n_doses = new_model.n_doses + 1;
    }
    new_model
}

// ============================================================================
// PRE-BUILT MODEL TEMPLATES
// ============================================================================

/// Create one-compartment IV model AST
fn template_one_comp_iv() -> ModelAST {
    var model = model_ast_new();

    // Parameters: CL, V
    model = model_add_param(model, param_node_with_lognormal_prior(0, 10.0, 2.3, 0.5));  // CL
    model = model_add_param(model, param_node_with_lognormal_prior(1, 70.0, 4.25, 0.4)); // V
    model = model_add_param(model, param_node_new(2, 0.1));  // sigma (error)

    // Compartments: Central
    model = model_add_compartment(model, compartment_central(0, 1));  // V is param 1

    // Flows: Elimination
    model = model_add_flow(model, flow_elimination(0, 0));  // CL is param 0

    // Observation model
    model.observation = observation_proportional(0, 1, 2);

    model
}

/// Create two-compartment IV model AST
fn template_two_comp_iv() -> ModelAST {
    var model = model_ast_new();

    // Parameters: CL, V1, V2, Q
    model = model_add_param(model, param_node_with_lognormal_prior(0, 10.0, 2.3, 0.5));  // CL
    model = model_add_param(model, param_node_with_lognormal_prior(1, 20.0, 3.0, 0.4));  // V1
    model = model_add_param(model, param_node_with_lognormal_prior(2, 50.0, 3.9, 0.5));  // V2
    model = model_add_param(model, param_node_with_lognormal_prior(3, 15.0, 2.7, 0.5));  // Q
    model = model_add_param(model, param_node_new(4, 0.1));  // sigma

    // Compartments: Central, Peripheral
    model = model_add_compartment(model, compartment_central(0, 1));
    model = model_add_compartment(model, compartment_peripheral(1, 2));

    // Flows: Elimination from central, distribution
    model = model_add_flow(model, flow_elimination(0, 0));
    model = model_add_flow(model, flow_distribution(0, 1, 3));
    model = model_add_flow(model, flow_distribution(1, 0, 3));

    // Observation
    model.observation = observation_proportional(0, 1, 4);

    model
}

/// Create one-compartment oral model AST
fn template_one_comp_oral() -> ModelAST {
    var model = model_ast_new();

    // Parameters: CL, V, Ka, F
    model = model_add_param(model, param_node_with_lognormal_prior(0, 10.0, 2.3, 0.5));  // CL
    model = model_add_param(model, param_node_with_lognormal_prior(1, 70.0, 4.25, 0.4)); // V
    model = model_add_param(model, param_node_with_lognormal_prior(2, 1.5, 0.4, 0.6));   // Ka
    model = model_add_param(model, param_node_with_uniform_prior(3, 0.8, 0.0, 1.0));     // F
    model = model_add_param(model, param_node_new(4, 0.15));  // sigma

    // Compartments: Depot, Central
    model = model_add_compartment(model, compartment_depot(0));
    model = model_add_compartment(model, compartment_central(1, 1));

    // Flows: Absorption, Elimination
    model = model_add_flow(model, flow_absorption(0, 1, 2));  // Ka is param 2
    model = model_add_flow(model, flow_elimination(1, 0));    // CL is param 0

    // Observation
    model.observation = observation_proportional(1, 1, 4);

    model
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_param_node() -> bool {
    let p = param_node_new(0, 100.0);
    abs_f64(p.init_value - 100.0) < 0.01 && !p.fixed
}

fn test_param_with_prior() -> bool {
    let p = param_node_with_normal_prior(0, 10.0, 10.0, 2.0);
    p.prior_type == 1 && abs_f64(p.prior_param1 - 10.0) < 0.01
}

fn test_log_prior_normal() -> bool {
    let p = param_node_with_normal_prior(0, 10.0, 10.0, 1.0);

    // At mean, log prior should be 0 (up to constant)
    let lp_at_mean = param_log_prior(p, 10.0);

    // One sigma away should be -0.5
    let lp_one_sigma = param_log_prior(p, 11.0);

    abs_f64(lp_at_mean - 0.0) < 0.01 && abs_f64(lp_one_sigma - (-0.5)) < 0.01
}

fn test_compartment_node() -> bool {
    let central = compartment_central(0, 1);
    central.comp_type == 0 && central.is_observation
}

fn test_flow_node() -> bool {
    let elim = flow_elimination(0, 1);
    elim.to_comp == -1 && elim.flow_type == 2
}

fn test_dosing_node() -> bool {
    let dose = dosing_iv_bolus(0.0, 100.0, 0);
    dose.route == 0 && abs_f64(dose.amount - 100.0) < 0.01
}

fn test_model_template() -> bool {
    let model = template_one_comp_iv();
    model.n_params >= 3 && model.n_compartments >= 1 && model.n_flows >= 1
}

fn test_covariate_power() -> bool {
    let effect = covariate_power(0, 0, 1, 70.0);

    // Weight 80kg, ref 70kg, theta 0.75
    let adjusted = apply_covariate(10.0, effect, 80.0, 0.75);

    // Expected: 10 * (80/70)^0.75 ≈ 10 * 1.107 ≈ 11.07
    adjusted > 10.5 && adjusted < 12.0
}

fn main() -> i32 {
    print("Testing medlang::ast module...\n");

    if !test_param_node() {
        print("FAIL: param_node\n");
        return 1
    }
    print("PASS: param_node\n");

    if !test_param_with_prior() {
        print("FAIL: param_with_prior\n");
        return 2
    }
    print("PASS: param_with_prior\n");

    if !test_log_prior_normal() {
        print("FAIL: log_prior_normal\n");
        return 3
    }
    print("PASS: log_prior_normal\n");

    if !test_compartment_node() {
        print("FAIL: compartment_node\n");
        return 4
    }
    print("PASS: compartment_node\n");

    if !test_flow_node() {
        print("FAIL: flow_node\n");
        return 5
    }
    print("PASS: flow_node\n");

    if !test_dosing_node() {
        print("FAIL: dosing_node\n");
        return 6
    }
    print("PASS: dosing_node\n");

    if !test_model_template() {
        print("FAIL: model_template\n");
        return 7
    }
    print("PASS: model_template\n");

    if !test_covariate_power() {
        print("FAIL: covariate_power\n");
        return 8
    }
    print("PASS: covariate_power\n");

    print("All medlang::ast tests PASSED\n");
    0
}

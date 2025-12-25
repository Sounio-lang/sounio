// simulation.d - Main PBPK Simulation Module
//
// This is the TOP-LEVEL module that brings everything together.
//
// Architecture:
// 1. SimulationConfig - User-facing configuration for simulations
// 2. SimulationResult - Complete PK metrics output
// 3. run_pbpk_simulation() - Main entry point orchestrating all steps
// 4. Wrapper functions - Convenient APIs for common scenarios
// 5. Patient scaling - Allometric scaling for weight/age
// 6. Drug parameterization - Kp calculation via Rodgers-Rowland
// 7. PK metrics - Cmax, AUC, t1/2, CL, Vdss calculations
// 8. Validation - Compare against observed data
// 9. Examples - Hardcoded simulations for testing

use core::pbpk_params::{PBPKParams, PatientData, DrugProperties};
use core::rodgers_rowland::{AllKpValues, predict_all_kp, predict_vdss};
use core::ode_pbpk::{PBPKState, simulate_pbpk, create_iv_bolus_state, create_oral_dose_state};

// ============================================================================
// CONFIGURATION STRUCTURES
// ============================================================================

/// SimulationConfig - User-facing configuration for PBPK simulation
struct SimulationConfig {
    t_end: f64@h,           // Simulation end time (e.g., 24h)
    dt: f64@h,              // Time step for ODE integration (e.g., 0.01h)
    dose: f64@mg,           // Drug dose (mg)
    route: i32,             // 0 = IV bolus, 1 = oral
    n_doses: i32,           // Number of doses (for multiple dosing)
    dosing_interval: f64@h, // Interval between doses (h)
}

impl SimulationConfig {
    /// Create default configuration for single IV bolus
    fn default_iv(dose: f64@mg, t_end: f64@h) -> Self {
        SimulationConfig {
            t_end: t_end,
            dt: 0.01@h,  // 0.01h = 36 seconds (high accuracy)
            dose: dose,
            route: 0,    // IV
            n_doses: 1,
            dosing_interval: 0.0@h
        }
    }

    /// Create default configuration for single oral dose
    fn default_oral(dose: f64@mg, t_end: f64@h) -> Self {
        SimulationConfig {
            t_end: t_end,
            dt: 0.01@h,
            dose: dose,
            route: 1,    // Oral
            n_doses: 1,
            dosing_interval: 0.0@h
        }
    }

    /// Create configuration for multiple dosing
    fn multiple_dose(dose: f64@mg, interval: f64@h, n_doses: i32, t_end: f64@h, route: i32) -> Self {
        SimulationConfig {
            t_end: t_end,
            dt: 0.01@h,
            dose: dose,
            route: route,
            n_doses: n_doses,
            dosing_interval: interval
        }
    }
}

// ============================================================================
// RESULT STRUCTURES
// ============================================================================

/// SimulationResult - Complete PK metrics from simulation
struct SimulationResult {
    // Primary PK parameters
    cmax_plasma: f64@mg_per_L,      // Maximum plasma concentration
    tmax: f64@h,                     // Time to Cmax
    auc_0_inf: f64@mg_h_per_L,      // Area under curve (0 to infinity)
    half_life: f64@h,                // Elimination half-life
    clearance: f64@L_per_h,          // Total body clearance
    vdss: f64@L,                     // Volume of distribution at steady state

    // Final state (for multiple dosing or extended simulations)
    final_state: PBPKState,

    // Success flag
    success: bool,
}

impl SimulationResult {
    /// Create empty result (for error cases)
    fn empty() -> Self {
        SimulationResult {
            cmax_plasma: 0.0@mg_per_L,
            tmax: 0.0@h,
            auc_0_inf: 0.0@mg_h_per_L,
            half_life: 0.0@h,
            clearance: 0.0@L_per_h,
            vdss: 0.0@L,
            final_state: zero_state(),
            success: false
        }
    }
}

// ============================================================================
// PATIENT SCALING FUNCTIONS
// ============================================================================

/// Scale PBPK parameters for patient weight and age
///
/// Scaling rules:
/// - Volumes: weight^1.0 (isometric)
/// - Flows: weight^0.75 (allometric, metabolic scaling)
/// - GFR: adjusted for age (Cockcroft-Gault-like)
fn scale_params_for_patient(base_params: PBPKParams, patient: PatientData) -> PBPKParams {
    // Reference weight (70 kg adult male)
    let ref_weight = 70.0@kg;
    
    // Weight scaling factors
    let volume_scale = patient.weight / ref_weight;              // Linear scaling
    let flow_scale = pow(patient.weight / ref_weight, 0.75);     // Allometric scaling

    // Age adjustment for renal function (Cockcroft-Gault)
    // GFR_adj = GFR × (1 - 0.01 × (age - 30)) for age > 30
    let age_factor = if patient.age > 30.0 {
        1.0 - 0.01 * (patient.age - 30.0)
    } else {
        1.0
    };
    let age_factor_clamped = if age_factor < 0.3 { 0.3 } else { age_factor }; // Minimum 30% GFR

    // Sex adjustment for flows (females ~10% lower cardiac output)
    let sex_factor = if patient.sex { 1.0 } else { 0.9 };

    // Combined flow adjustment
    let flow_adjustment = flow_scale * sex_factor;

    // Scale volumes
    let scaled_params = PBPKParams {
        v_blood: base_params.v_blood * volume_scale,
        v_liver: base_params.v_liver * volume_scale,
        v_kidney: base_params.v_kidney * volume_scale,
        v_brain: base_params.v_brain * volume_scale,
        v_heart: base_params.v_heart * volume_scale,
        v_lung: base_params.v_lung * volume_scale,
        v_muscle: base_params.v_muscle * volume_scale,
        v_adipose: base_params.v_adipose * volume_scale,
        v_gut: base_params.v_gut * volume_scale,
        v_skin: base_params.v_skin * volume_scale,
        v_bone: base_params.v_bone * volume_scale,
        v_spleen: base_params.v_spleen * volume_scale,
        v_pancreas: base_params.v_pancreas * volume_scale,
        v_rest: base_params.v_rest * volume_scale,

        // Scale flows
        q_blood: base_params.q_blood * flow_adjustment,
        q_liver: base_params.q_liver * flow_adjustment,
        q_kidney: base_params.q_kidney * flow_adjustment * age_factor_clamped,  // Age affects kidney
        q_brain: base_params.q_brain * flow_adjustment,
        q_heart: base_params.q_heart * flow_adjustment,
        q_lung: base_params.q_lung * flow_adjustment,
        q_muscle: base_params.q_muscle * flow_adjustment,
        q_adipose: base_params.q_adipose * flow_adjustment,
        q_gut: base_params.q_gut * flow_adjustment,
        q_skin: base_params.q_skin * flow_adjustment,
        q_bone: base_params.q_bone * flow_adjustment,
        q_spleen: base_params.q_spleen * flow_adjustment,
        q_pancreas: base_params.q_pancreas * flow_adjustment,
        q_rest: base_params.q_rest * flow_adjustment,

        // Clearances (keep from drug properties, will be set later)
        clearance_hepatic: base_params.clearance_hepatic,
        clearance_renal: base_params.clearance_renal * age_factor_clamped,  // Age affects renal CL

        // Kp values (keep from base)
        kp_blood: base_params.kp_blood,
        kp_liver: base_params.kp_liver,
        kp_kidney: base_params.kp_kidney,
        kp_brain: base_params.kp_brain,
        kp_heart: base_params.kp_heart,
        kp_lung: base_params.kp_lung,
        kp_muscle: base_params.kp_muscle,
        kp_adipose: base_params.kp_adipose,
        kp_gut: base_params.kp_gut,
        kp_skin: base_params.kp_skin,
        kp_bone: base_params.kp_bone,
        kp_spleen: base_params.kp_spleen,
        kp_pancreas: base_params.kp_pancreas,
        kp_rest: base_params.kp_rest,

        // Physiological parameters (unchanged)
        fu_plasma: base_params.fu_plasma,
        hematocrit: base_params.hematocrit,
        bp_ratio: base_params.bp_ratio
    };

    return scaled_params;
}

// ============================================================================
// DRUG PARAMETERIZATION
// ============================================================================

/// Calculate all Kp values using Rodgers-Rowland
fn calculate_drug_params(drug: DrugProperties) -> AllKpValues {
    return predict_all_kp(&drug);
}

/// Estimate clearance from drug properties and patient
///
/// CL_total = CL_hepatic + CL_renal
///
/// CL_hepatic = fm_hepatic × Q_liver × fu_plasma × CLint / (Q_liver + fu_plasma × CLint)
/// CL_renal = fm_renal × GFR × fu_plasma
fn estimate_clearance(
    drug: DrugProperties,
    patient: PatientData,
    fm_hepatic: f64,
    fm_renal: f64,
    clint: f64@L_per_h
) -> (f64@L_per_h, f64@L_per_h) {
    // Reference GFR (120 mL/min = 7.2 L/h)
    let gfr_base = 7.2@L_per_h;
    
    // Age-adjusted GFR
    let age_factor = if patient.age > 30.0 {
        1.0 - 0.01 * (patient.age - 30.0)
    } else {
        1.0
    };
    let gfr = gfr_base * age_factor;

    // Hepatic clearance (well-stirred model)
    // CL_h = Q_h × fu × CLint / (Q_h + fu × CLint)
    let q_liver = 97.2@L_per_h;  // Typical liver blood flow
    let fu = drug.fu;
    let cl_hepatic = fm_hepatic * (q_liver * fu * clint) / (q_liver + fu * clint);

    // Renal clearance
    // CL_r = GFR × fu × fm_renal
    let cl_renal = fm_renal * gfr * fu;

    return (cl_hepatic, cl_renal);
}

// ============================================================================
// PK METRIC CALCULATIONS
// ============================================================================

/// Calculate elimination half-life from two concentration points
///
/// t1/2 = 0.693 × (t_late - t_early) / ln(C_early / C_late)
///
/// Assumes exponential decay in elimination phase
fn calculate_half_life_from_curve(
    c_early: f64@mg_per_L,
    c_late: f64@mg_per_L,
    t_early: f64@h,
    t_late: f64@h
) -> f64@h {
    if c_early <= 0.0@mg_per_L || c_late <= 0.0@mg_per_L {
        return 0.0@h;  // Invalid data
    }

    if c_late >= c_early {
        return 0.0@h;  // Not in elimination phase
    }

    let ratio = c_early / c_late;
    let ln_ratio = log(ratio);
    let delta_t = t_late - t_early;

    return 0.693 * delta_t / ln_ratio;
}

/// Calculate total body clearance from AUC
///
/// CL = F × Dose / AUC₀₋∞
fn calculate_clearance_from_auc(
    dose: f64@mg,
    auc: f64@mg_h_per_L,
    f_oral: f64
) -> f64@L_per_h {
    if auc <= 0.0@mg_h_per_L {
        return 0.0@L_per_h;
    }

    return (f_oral * dose) / auc;
}

/// Calculate volume of distribution at steady state
///
/// Vdss = CL × t1/2 / 0.693
fn calculate_vdss_from_cl_thalf(
    cl: f64@L_per_h,
    t_half: f64@h
) -> f64@L {
    return cl * t_half / 0.693;
}

// ============================================================================
// MAIN SIMULATION FUNCTION
// ============================================================================

/// Main PBPK simulation orchestrator
///
/// Steps:
/// 1. Create base PBPK parameters
/// 2. Scale for patient weight/age/sex
/// 3. Calculate Kp values using Rodgers-Rowland
/// 4. Set drug-specific clearances
/// 5. Initialize state (IV or oral)
/// 6. Run ODE integration
/// 7. Calculate PK metrics (Cmax, AUC, t1/2, CL, Vdss)
/// 8. Return results
fn run_pbpk_simulation(
    drug: DrugProperties,
    patient: PatientData,
    config: SimulationConfig
) -> SimulationResult {
    // Step 1: Create base PBPK parameters (70kg reference)
    let base_params = create_default_pbpk_params();

    // Step 2: Scale for patient
    let mut params = scale_params_for_patient(base_params, patient);

    // Step 3: Calculate Kp values using Rodgers-Rowland
    let kps = calculate_drug_params(drug);
    
    // Update Kp values in params
    params.kp_adipose = kps.kp_adipose;
    params.kp_bone = kps.kp_bone;
    params.kp_brain = kps.kp_brain;
    params.kp_gut = kps.kp_gut;
    params.kp_heart = kps.kp_heart;
    params.kp_kidney = kps.kp_kidney;
    params.kp_liver = kps.kp_liver;
    params.kp_lung = kps.kp_lung;
    params.kp_muscle = kps.kp_muscle;
    params.kp_skin = kps.kp_skin;
    params.kp_spleen = kps.kp_spleen;
    params.kp_pancreas = kps.kp_pancreas;
    params.kp_rest = 1.0;  // Default for rest compartment

    // Step 4: Set drug-specific clearances (from drug properties)
    params.clearance_hepatic = drug.cl_hepatic;
    params.clearance_renal = drug.cl_renal;

    // Step 5: Initialize state based on route
    let mut state = if config.route == 0 {
        // IV bolus
        create_iv_bolus_state(config.dose, params)
    } else {
        // Oral
        create_oral_dose_state(config.dose, drug.f_oral)
    };

    // Step 6: Run simulation with detailed tracking for PK metrics
    let mut t = 0.0@h;
    let mut cmax = 0.0@mg_per_L;
    let mut tmax = 0.0@h;
    let mut auc = 0.0@mg_h_per_L;
    let mut c_prev = state.c_blood;

    // For half-life calculation: sample at 25% and 75% of simulation time
    let t_early = config.t_end * 0.25;
    let t_late = config.t_end * 0.75;
    let mut c_early = 0.0@mg_per_L;
    let mut c_late = 0.0@mg_per_L;

    while t < config.t_end {
        // Take single RK4 step
        state = rk4_step_pbpk(state, params, drug, t, config.dt);
        t = t + config.dt;

        // Track Cmax and Tmax
        if state.c_blood > cmax {
            cmax = state.c_blood;
            tmax = t;
        }

        // Calculate AUC (trapezoidal rule)
        auc = auc + ((c_prev + state.c_blood) / 2.0) * config.dt;
        c_prev = state.c_blood;

        // Sample for half-life calculation
        if t >= t_early && c_early == 0.0@mg_per_L {
            c_early = state.c_blood;
        }
        if t >= t_late && c_late == 0.0@mg_per_L {
            c_late = state.c_blood;
        }
    }

    // Step 7: Calculate PK metrics
    let half_life = calculate_half_life_from_curve(c_early, c_late, t_early, t_late);
    let f_bioavail = if config.route == 0 { 1.0 } else { drug.f_oral };
    let clearance = calculate_clearance_from_auc(config.dose, auc, f_bioavail);
    let vdss = calculate_vdss_from_cl_thalf(clearance, half_life);

    // Step 8: Return results
    return SimulationResult {
        cmax_plasma: cmax,
        tmax: tmax,
        auc_0_inf: auc,
        half_life: half_life,
        clearance: clearance,
        vdss: vdss,
        final_state: state,
        success: true
    };
}

// ============================================================================
// WRAPPER FUNCTIONS (Convenient APIs)
// ============================================================================

/// Simulate IV bolus dosing
fn simulate_iv_bolus(
    drug: DrugProperties,
    patient: PatientData,
    dose: f64@mg,
    t_end: f64@h
) -> SimulationResult {
    let config = SimulationConfig::default_iv(dose, t_end);
    return run_pbpk_simulation(drug, patient, config);
}

/// Simulate oral dosing
fn simulate_oral(
    drug: DrugProperties,
    patient: PatientData,
    dose: f64@mg,
    t_end: f64@h
) -> SimulationResult {
    let config = SimulationConfig::default_oral(dose, t_end);
    return run_pbpk_simulation(drug, patient, config);
}

/// Simulate multiple dosing (IV or oral)
fn simulate_multiple_dose(
    drug: DrugProperties,
    patient: PatientData,
    dose: f64@mg,
    interval: f64@h,
    n_doses: i32,
    t_end: f64@h,
    route: i32
) -> SimulationResult {
    let config = SimulationConfig::multiple_dose(dose, interval, n_doses, t_end, route);
    
    // For multiple dosing, we need to add doses at intervals
    // This is simplified - full implementation would add doses at t = 0, interval, 2*interval, etc.
    // For now, simulate single dose (TODO: implement multiple dosing loop)
    
    return run_pbpk_simulation(drug, patient, config);
}

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/// ValidationResult - Comparison against observed data
struct ValidationResult {
    fold_error_cmax: f64,
    fold_error_auc: f64,
    fold_error_thalf: f64,
    within_2fold: bool,
}

/// Validate simulation results against observed PK data
///
/// Fold error = Predicted / Observed
/// Success criteria: 0.5 ≤ FE ≤ 2.0 (within 2-fold)
fn validate_against_observed(
    result: SimulationResult,
    cmax_obs: f64@mg_per_L,
    auc_obs: f64@mg_h_per_L,
    thalf_obs: f64@h
) -> ValidationResult {
    let fe_cmax = if cmax_obs > 0.0@mg_per_L {
        result.cmax_plasma / cmax_obs
    } else {
        0.0
    };

    let fe_auc = if auc_obs > 0.0@mg_h_per_L {
        result.auc_0_inf / auc_obs
    } else {
        0.0
    };

    let fe_thalf = if thalf_obs > 0.0@h {
        result.half_life / thalf_obs
    } else {
        0.0
    };

    // Check if all within 2-fold
    let within_2fold = (
        fe_cmax >= 0.5 && fe_cmax <= 2.0 &&
        fe_auc >= 0.5 && fe_auc <= 2.0 &&
        fe_thalf >= 0.5 && fe_thalf <= 2.0
    );

    return ValidationResult {
        fold_error_cmax: fe_cmax,
        fold_error_auc: fe_auc,
        fold_error_thalf: fe_thalf,
        within_2fold: within_2fold
    };
}

// ============================================================================
// EXAMPLE SIMULATIONS (Hardcoded)
// ============================================================================

/// Example: Midazolam 2mg IV bolus in healthy adult
///
/// Midazolam is a CYP3A4 substrate, lipophilic benzodiazepine
/// Expected PK:
/// - Cmax: ~100-150 ng/mL (after 2mg IV)
/// - t1/2: ~2-3 hours
/// - CL: ~25-40 L/h
/// - Vdss: ~1-2 L/kg (~70-140 L for 70kg)
fn example_midazolam_simulation() -> SimulationResult {
    // Patient: 70kg male, 35 years old, healthy
    let patient = create_patient(35.0, 70.0@kg, true);

    // Midazolam drug properties
    let midazolam = DrugProperties {
        name: "Midazolam".to_string(),
        mw: 325.77,
        logp: 3.89,      // Lipophilic
        pka: 6.15,       // Weak base
        fu: 0.03,        // 3% unbound (highly protein bound)
        bp_ratio: 0.65,  // Preferentially in plasma
        is_base: true,
        
        // PK parameters
        cl_hepatic: 30.0@L_per_h,   // High hepatic clearance (CYP3A4)
        cl_renal: 1.0@L_per_h,      // Minimal renal clearance
        ka: 1.5@per_h,              // Fast absorption (if oral)
        f_oral: 0.4,                // 40% oral bioavailability (high first-pass)
        
        // Kp values (will be calculated by Rodgers-Rowland)
        kp_liver: 3.0,
        kp_kidney: 2.5,
        kp_brain: 2.0,
        kp_heart: 2.0,
        kp_lung: 2.5,
        kp_muscle: 1.5,
        kp_adipose: 5.0,  // Lipophilic accumulation
        kp_gut: 2.5,
        kp_skin: 1.8,
        kp_bone: 0.5,
        kp_spleen: 2.5,
        kp_pancreas: 2.0
    };

    // Simulate 2mg IV bolus over 24 hours
    return simulate_iv_bolus(midazolam, patient, 2.0@mg, 24.0@h);
}

/// Example: Metformin 500mg oral in type 2 diabetes patient
///
/// Metformin is a hydrophilic biguanide, minimal metabolism
/// Expected PK:
/// - Cmax: ~1-2 μg/mL (1000-2000 ng/mL)
/// - Tmax: ~2-3 hours
/// - t1/2: ~4-6 hours
/// - CL: ~400-600 mL/min (~24-36 L/h)
/// - Primarily renal elimination
fn example_metformin_simulation() -> SimulationResult {
    // Patient: 70kg male, 50 years old, type 2 diabetes
    let mut patient = create_patient(50.0, 70.0@kg, true);
    patient.disease_state = DIABETES;

    // Metformin drug properties
    let metformin = DrugProperties {
        name: "Metformin".to_string(),
        mw: 129.16,
        logp: -1.43,     // Hydrophilic
        pka: 12.4,       // Strong base (ionized at physiological pH)
        fu: 1.0,         // Not protein bound
        bp_ratio: 1.08,  // Slightly higher in blood
        is_base: true,
        
        // PK parameters
        cl_hepatic: 0.0@L_per_h,    // No hepatic metabolism
        cl_renal: 30.0@L_per_h,     // Primarily renal elimination
        ka: 0.8@per_h,              // Moderate absorption
        f_oral: 0.5,                // 50% oral bioavailability
        
        // Kp values (hydrophilic drug - limited tissue distribution)
        kp_liver: 4.0,   // OCT1 transporter - high liver uptake
        kp_kidney: 5.0,  // OCT2/MATE - high kidney uptake
        kp_brain: 0.3,   // Poor CNS penetration
        kp_heart: 1.5,
        kp_lung: 1.2,
        kp_muscle: 2.0,  // GLUT4 - therapeutic target
        kp_adipose: 0.5, // Poor lipid partitioning
        kp_gut: 3.0,     // GI absorption site
        kp_skin: 0.8,
        kp_bone: 0.3,
        kp_spleen: 1.5,
        kp_pancreas: 2.0
    };

    // Simulate 500mg oral dose over 24 hours
    return simulate_oral(metformin, patient, 500.0@mg, 24.0@h);
}

// ============================================================================
// OUTPUT FUNCTIONS
// ============================================================================

/// Print simulation results in human-readable format
fn print_simulation_results(result: SimulationResult) -> i32 {
    if !result.success {
        println!("Simulation FAILED");
        return 1;
    }

    println!("========================================");
    println!("PBPK Simulation Results");
    println!("========================================");
    println!("");
    
    println!("Primary PK Parameters:");
    println!("  Cmax (plasma):  {:.2} mg/L", result.cmax_plasma);
    println!("  Tmax:           {:.2} h", result.tmax);
    println!("  AUC(0-∞):       {:.2} mg·h/L", result.auc_0_inf);
    println!("  Half-life:      {:.2} h", result.half_life);
    println!("  Clearance:      {:.2} L/h", result.clearance);
    println!("  Vdss:           {:.2} L", result.vdss);
    println!("");
    
    println!("Final Concentrations (mg/L):");
    println!("  Blood:    {:.4}", result.final_state.c_blood);
    println!("  Liver:    {:.4}", result.final_state.c_liver);
    println!("  Kidney:   {:.4}", result.final_state.c_kidney);
    println!("  Brain:    {:.4}", result.final_state.c_brain);
    println!("  Heart:    {:.4}", result.final_state.c_heart);
    println!("  Muscle:   {:.4}", result.final_state.c_muscle);
    println!("  Adipose:  {:.4}", result.final_state.c_adipose);
    println!("========================================");

    return 0;
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

pub fn main() {
    println!("Darwin PBPK Platform - Main Simulation Module");
    println!("==============================================");
    println!("");

    // Example 1: Midazolam IV
    println!("Example 1: Midazolam 2mg IV bolus");
    println!("----------------------------------");
    let midazolam_result = example_midazolam_simulation();
    print_simulation_results(midazolam_result);
    println!("");

    // Example 2: Metformin oral
    println!("Example 2: Metformin 500mg oral");
    println!("--------------------------------");
    let metformin_result = example_metformin_simulation();
    print_simulation_results(metformin_result);
    println!("");

    println!("All simulations completed successfully!");
}

// ode_pbpk.d - Integrated 14-Compartment PBPK ODE Solver
// 
// This is the HEART of the Darwin PBPK system - the actual differential equations
// that govern drug distribution, metabolism, and elimination across physiological compartments.
//
// Architecture:
// 1. PBPKState - Current concentrations in all 14 compartments + gut lumen
// 2. PBPKDerivatives - dC/dt for each compartment (rates of change)
// 3. pbpk_ode() - Core ODE function implementing physiological equations
// 4. RK4 integrator - 4th-order Runge-Kutta for numerical integration
// 5. Initialization functions - IV bolus vs oral dosing
// 6. PK metrics - AUC, Cmax calculation
//
// Physiological Principles:
// - Flow-limited compartments: Q × (C_arterial - C_venous)
// - Venous mixing: Blood receives all tissue outflows
// - Arterial blood: Comes from lung (oxygenated)
// - Clearance: Liver (metabolic) + Kidney (renal)
// - Absorption: First-order from gut lumen

import "../units.d";
import "params.d";
import "drug.d";

// ============================================================================
// STATE STRUCTURES
// ============================================================================

/// PBPKState - Current drug concentrations in all compartments
/// This represents the complete state vector x(t) of the PBPK system
struct PBPKState {
    // Central compartment
    c_blood: f64@mg_per_L;
    
    // Clearance organs
    c_liver: f64@mg_per_L;
    c_kidney: f64@mg_per_L;
    
    // Highly perfused organs
    c_brain: f64@mg_per_L;
    c_heart: f64@mg_per_L;
    c_lung: f64@mg_per_L;
    
    // Poorly perfused tissues
    c_muscle: f64@mg_per_L;
    c_adipose: f64@mg_per_L;
    
    // GI tract
    c_gut: f64@mg_per_L;
    
    // Other tissues
    c_skin: f64@mg_per_L;
    c_bone: f64@mg_per_L;
    c_spleen: f64@mg_per_L;
    c_pancreas: f64@mg_per_L;
    
    // Rest of body (lumped compartment)
    c_rest: f64@mg_per_L;
    
    // Gut lumen (amount, not concentration)
    a_gut_lumen: f64@mg;
}

/// PBPKDerivatives - Rates of change for all compartments
/// This represents dx/dt = f(x, t) in the ODE system
struct PBPKDerivatives {
    dc_blood_dt: f64@mg_per_L_per_h;
    dc_liver_dt: f64@mg_per_L_per_h;
    dc_kidney_dt: f64@mg_per_L_per_h;
    dc_brain_dt: f64@mg_per_L_per_h;
    dc_heart_dt: f64@mg_per_L_per_h;
    dc_lung_dt: f64@mg_per_L_per_h;
    dc_muscle_dt: f64@mg_per_L_per_h;
    dc_adipose_dt: f64@mg_per_L_per_h;
    dc_gut_dt: f64@mg_per_L_per_h;
    dc_skin_dt: f64@mg_per_L_per_h;
    dc_bone_dt: f64@mg_per_L_per_h;
    dc_spleen_dt: f64@mg_per_L_per_h;
    dc_pancreas_dt: f64@mg_per_L_per_h;
    dc_rest_dt: f64@mg_per_L_per_h;
    da_gut_lumen_dt: f64@mg_per_h;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate arterial concentration (output from lungs)
/// C_arterial = C_lung / Kp_lung
fn arterial_concentration(c_lung: f64@mg_per_L, kp_lung: f64) -> f64@mg_per_L {
    return c_lung / kp_lung;
}

/// Calculate unbound (free) concentration
/// C_unbound = C_total × fu
fn unbound_concentration(c_total: f64@mg_per_L, fu: f64) -> f64@mg_per_L {
    return c_total * fu;
}

/// Calculate venous return concentration (flow-weighted average)
/// C_venous_return = Σ(Q_i × C_i / Kp_i) / CO
fn venous_return_concentration(
    state: PBPKState,
    params: PBPKParams,
    drug: DrugProperties
) -> f64@mg_per_L {
    let q_liver = params.q_liver;
    let q_kidney = params.q_kidney;
    let q_brain = params.q_brain;
    let q_heart = params.q_heart;
    let q_muscle = params.q_muscle;
    let q_adipose = params.q_adipose;
    let q_gut = params.q_gut;
    let q_skin = params.q_skin;
    let q_bone = params.q_bone;
    let q_spleen = params.q_spleen;
    let q_pancreas = params.q_pancreas;
    let q_rest = params.q_rest;
    
    let kp_liver = drug.kp_liver;
    let kp_kidney = drug.kp_kidney;
    let kp_brain = drug.kp_brain;
    let kp_heart = drug.kp_heart;
    let kp_muscle = drug.kp_muscle;
    let kp_adipose = drug.kp_adipose;
    let kp_gut = drug.kp_gut;
    let kp_skin = drug.kp_skin;
    let kp_bone = drug.kp_bone;
    let kp_spleen = drug.kp_spleen;
    let kp_pancreas = drug.kp_pancreas;
    let kp_rest = drug.kp_rest;
    
    // Sum of flow-weighted venous concentrations
    let sum = (
        q_liver * (state.c_liver / kp_liver) +
        q_kidney * (state.c_kidney / kp_kidney) +
        q_brain * (state.c_brain / kp_brain) +
        q_heart * (state.c_heart / kp_heart) +
        q_muscle * (state.c_muscle / kp_muscle) +
        q_adipose * (state.c_adipose / kp_adipose) +
        q_gut * (state.c_gut / kp_gut) +
        q_skin * (state.c_skin / kp_skin) +
        q_bone * (state.c_bone / kp_bone) +
        q_spleen * (state.c_spleen / kp_spleen) +
        q_pancreas * (state.c_pancreas / kp_pancreas) +
        q_rest * (state.c_rest / kp_rest)
    );
    
    return sum / params.cardiac_output;
}

// ============================================================================
// CORE ODE FUNCTION
// ============================================================================

/// Core PBPK ODE function: dx/dt = f(x, t)
/// 
/// Implements the 14-compartment physiological equations:
/// 
/// Flow-limited tissues:
///   dC_tissue/dt = (Q_tissue / V_tissue) × (C_arterial - C_tissue / Kp_tissue)
/// 
/// Liver (with hepatic clearance):
///   dC_liver/dt = (Q_liver / V_liver) × (C_arterial - C_liver / Kp_liver)
///                 - (CL_hepatic / V_liver) × C_liver_unbound
/// 
/// Kidney (with renal clearance):
///   dC_kidney/dt = (Q_kidney / V_kidney) × (C_arterial - C_kidney / Kp_kidney)
///                  - (CL_renal / V_kidney) × C_kidney_unbound
/// 
/// Gut (with absorption):
///   dC_gut/dt = (Q_gut / V_gut) × (C_arterial - C_gut / Kp_gut)
///               + (ka / V_gut) × A_gut_lumen
///   dA_gut_lumen/dt = -ka × A_gut_lumen
/// 
/// Lung (receives venous return):
///   dC_lung/dt = (Q_lung / V_lung) × (C_venous_return - C_lung / Kp_lung)
/// 
/// Blood (venous mixing from all tissues):
///   dC_blood/dt = (CO / V_blood) × (C_venous_return - C_blood)
fn pbpk_ode(
    state: PBPKState,
    params: PBPKParams,
    drug: DrugProperties,
    t: f64@h
) -> PBPKDerivatives {
    // Calculate arterial concentration (from lungs)
    let c_arterial = arterial_concentration(state.c_lung, drug.kp_lung);
    
    // Calculate venous return concentration (to lungs)
    let c_venous_return = venous_return_concentration(state, params, drug);
    
    // Cardiac output
    let co = params.cardiac_output;
    
    // ========================================================================
    // BLOOD COMPARTMENT
    // ========================================================================
    // Blood receives venous return from all tissues
    let dc_blood_dt = (co / params.v_blood) * (c_venous_return - state.c_blood);
    
    // ========================================================================
    // LUNG COMPARTMENT
    // ========================================================================
    // Lung receives venous return and outputs arterial blood
    let dc_lung_dt = (co / params.v_lung) * 
                     (c_venous_return - state.c_lung / drug.kp_lung);
    
    // ========================================================================
    // LIVER COMPARTMENT (with hepatic clearance)
    // ========================================================================
    let c_liver_unbound = unbound_concentration(state.c_liver, drug.fu_plasma);
    let hepatic_clearance_term = (drug.cl_hepatic / params.v_liver) * c_liver_unbound;
    
    let dc_liver_dt = (params.q_liver / params.v_liver) * 
                      (c_arterial - state.c_liver / drug.kp_liver) -
                      hepatic_clearance_term;
    
    // ========================================================================
    // KIDNEY COMPARTMENT (with renal clearance)
    // ========================================================================
    let c_kidney_unbound = unbound_concentration(state.c_kidney, drug.fu_plasma);
    let renal_clearance_term = (drug.cl_renal / params.v_kidney) * c_kidney_unbound;
    
    let dc_kidney_dt = (params.q_kidney / params.v_kidney) * 
                       (c_arterial - state.c_kidney / drug.kp_kidney) -
                       renal_clearance_term;
    
    // ========================================================================
    // GUT COMPARTMENT (with oral absorption)
    // ========================================================================
    let absorption_rate = drug.ka * state.a_gut_lumen;
    let absorption_term = absorption_rate / params.v_gut;
    
    let dc_gut_dt = (params.q_gut / params.v_gut) * 
                    (c_arterial - state.c_gut / drug.kp_gut) +
                    absorption_term;
    
    let da_gut_lumen_dt = -absorption_rate;
    
    // ========================================================================
    // BRAIN COMPARTMENT (highly perfused)
    // ========================================================================
    let dc_brain_dt = (params.q_brain / params.v_brain) * 
                      (c_arterial - state.c_brain / drug.kp_brain);
    
    // ========================================================================
    // HEART COMPARTMENT (highly perfused)
    // ========================================================================
    let dc_heart_dt = (params.q_heart / params.v_heart) * 
                      (c_arterial - state.c_heart / drug.kp_heart);
    
    // ========================================================================
    // MUSCLE COMPARTMENT (poorly perfused, large volume)
    // ========================================================================
    let dc_muscle_dt = (params.q_muscle / params.v_muscle) * 
                       (c_arterial - state.c_muscle / drug.kp_muscle);
    
    // ========================================================================
    // ADIPOSE COMPARTMENT (poorly perfused, lipophilic drugs)
    // ========================================================================
    let dc_adipose_dt = (params.q_adipose / params.v_adipose) * 
                        (c_arterial - state.c_adipose / drug.kp_adipose);
    
    // ========================================================================
    // SKIN COMPARTMENT
    // ========================================================================
    let dc_skin_dt = (params.q_skin / params.v_skin) * 
                     (c_arterial - state.c_skin / drug.kp_skin);
    
    // ========================================================================
    // BONE COMPARTMENT
    // ========================================================================
    let dc_bone_dt = (params.q_bone / params.v_bone) * 
                     (c_arterial - state.c_bone / drug.kp_bone);
    
    // ========================================================================
    // SPLEEN COMPARTMENT
    // ========================================================================
    let dc_spleen_dt = (params.q_spleen / params.v_spleen) * 
                       (c_arterial - state.c_spleen / drug.kp_spleen);
    
    // ========================================================================
    // PANCREAS COMPARTMENT
    // ========================================================================
    let dc_pancreas_dt = (params.q_pancreas / params.v_pancreas) * 
                         (c_arterial - state.c_pancreas / drug.kp_pancreas);
    
    // ========================================================================
    // REST COMPARTMENT (lumped remaining tissues)
    // ========================================================================
    let dc_rest_dt = (params.q_rest / params.v_rest) * 
                     (c_arterial - state.c_rest / drug.kp_rest);
    
    // Return derivatives struct
    return PBPKDerivatives {
        dc_blood_dt: dc_blood_dt,
        dc_liver_dt: dc_liver_dt,
        dc_kidney_dt: dc_kidney_dt,
        dc_brain_dt: dc_brain_dt,
        dc_heart_dt: dc_heart_dt,
        dc_lung_dt: dc_lung_dt,
        dc_muscle_dt: dc_muscle_dt,
        dc_adipose_dt: dc_adipose_dt,
        dc_gut_dt: dc_gut_dt,
        dc_skin_dt: dc_skin_dt,
        dc_bone_dt: dc_bone_dt,
        dc_spleen_dt: dc_spleen_dt,
        dc_pancreas_dt: dc_pancreas_dt,
        dc_rest_dt: dc_rest_dt,
        da_gut_lumen_dt: da_gut_lumen_dt
    };
}

// ============================================================================
// RK4 INTEGRATOR
// ============================================================================

/// Add state and scaled derivative (helper for RK4)
fn add_scaled_derivative(
    state: PBPKState,
    deriv: PBPKDerivatives,
    scale: f64@h
) -> PBPKState {
    return PBPKState {
        c_blood: state.c_blood + deriv.dc_blood_dt * scale,
        c_liver: state.c_liver + deriv.dc_liver_dt * scale,
        c_kidney: state.c_kidney + deriv.dc_kidney_dt * scale,
        c_brain: state.c_brain + deriv.dc_brain_dt * scale,
        c_heart: state.c_heart + deriv.dc_heart_dt * scale,
        c_lung: state.c_lung + deriv.dc_lung_dt * scale,
        c_muscle: state.c_muscle + deriv.dc_muscle_dt * scale,
        c_adipose: state.c_adipose + deriv.dc_adipose_dt * scale,
        c_gut: state.c_gut + deriv.dc_gut_dt * scale,
        c_skin: state.c_skin + deriv.dc_skin_dt * scale,
        c_bone: state.c_bone + deriv.dc_bone_dt * scale,
        c_spleen: state.c_spleen + deriv.dc_spleen_dt * scale,
        c_pancreas: state.c_pancreas + deriv.dc_pancreas_dt * scale,
        c_rest: state.c_rest + deriv.dc_rest_dt * scale,
        a_gut_lumen: state.a_gut_lumen + deriv.da_gut_lumen_dt * scale
    };
}

/// Single RK4 integration step
/// 
/// 4th-order Runge-Kutta method:
///   k1 = f(x_n, t_n)
///   k2 = f(x_n + k1·dt/2, t_n + dt/2)
///   k3 = f(x_n + k2·dt/2, t_n + dt/2)
///   k4 = f(x_n + k3·dt, t_n + dt)
///   x_{n+1} = x_n + (k1 + 2k2 + 2k3 + k4) · dt/6
fn rk4_step_pbpk(
    state: PBPKState,
    params: PBPKParams,
    drug: DrugProperties,
    t: f64@h,
    dt: f64@h
) -> PBPKState {
    // k1 = f(x_n, t_n)
    let k1 = pbpk_ode(state, params, drug, t);
    
    // k2 = f(x_n + k1·dt/2, t_n + dt/2)
    let state_k2 = add_scaled_derivative(state, k1, dt / 2.0);
    let k2 = pbpk_ode(state_k2, params, drug, t + dt / 2.0);
    
    // k3 = f(x_n + k2·dt/2, t_n + dt/2)
    let state_k3 = add_scaled_derivative(state, k2, dt / 2.0);
    let k3 = pbpk_ode(state_k3, params, drug, t + dt / 2.0);
    
    // k4 = f(x_n + k3·dt, t_n + dt)
    let state_k4 = add_scaled_derivative(state, k3, dt);
    let k4 = pbpk_ode(state_k4, params, drug, t + dt);
    
    // Weighted sum: (k1 + 2k2 + 2k3 + k4) / 6
    let deriv_weighted = PBPKDerivatives {
        dc_blood_dt: (k1.dc_blood_dt + 2.0 * k2.dc_blood_dt + 2.0 * k3.dc_blood_dt + k4.dc_blood_dt) / 6.0,
        dc_liver_dt: (k1.dc_liver_dt + 2.0 * k2.dc_liver_dt + 2.0 * k3.dc_liver_dt + k4.dc_liver_dt) / 6.0,
        dc_kidney_dt: (k1.dc_kidney_dt + 2.0 * k2.dc_kidney_dt + 2.0 * k3.dc_kidney_dt + k4.dc_kidney_dt) / 6.0,
        dc_brain_dt: (k1.dc_brain_dt + 2.0 * k2.dc_brain_dt + 2.0 * k3.dc_brain_dt + k4.dc_brain_dt) / 6.0,
        dc_heart_dt: (k1.dc_heart_dt + 2.0 * k2.dc_heart_dt + 2.0 * k3.dc_heart_dt + k4.dc_heart_dt) / 6.0,
        dc_lung_dt: (k1.dc_lung_dt + 2.0 * k2.dc_lung_dt + 2.0 * k3.dc_lung_dt + k4.dc_lung_dt) / 6.0,
        dc_muscle_dt: (k1.dc_muscle_dt + 2.0 * k2.dc_muscle_dt + 2.0 * k3.dc_muscle_dt + k4.dc_muscle_dt) / 6.0,
        dc_adipose_dt: (k1.dc_adipose_dt + 2.0 * k2.dc_adipose_dt + 2.0 * k3.dc_adipose_dt + k4.dc_adipose_dt) / 6.0,
        dc_gut_dt: (k1.dc_gut_dt + 2.0 * k2.dc_gut_dt + 2.0 * k3.dc_gut_dt + k4.dc_gut_dt) / 6.0,
        dc_skin_dt: (k1.dc_skin_dt + 2.0 * k2.dc_skin_dt + 2.0 * k3.dc_skin_dt + k4.dc_skin_dt) / 6.0,
        dc_bone_dt: (k1.dc_bone_dt + 2.0 * k2.dc_bone_dt + 2.0 * k3.dc_bone_dt + k4.dc_bone_dt) / 6.0,
        dc_spleen_dt: (k1.dc_spleen_dt + 2.0 * k2.dc_spleen_dt + 2.0 * k3.dc_spleen_dt + k4.dc_spleen_dt) / 6.0,
        dc_pancreas_dt: (k1.dc_pancreas_dt + 2.0 * k2.dc_pancreas_dt + 2.0 * k3.dc_pancreas_dt + k4.dc_pancreas_dt) / 6.0,
        dc_rest_dt: (k1.dc_rest_dt + 2.0 * k2.dc_rest_dt + 2.0 * k3.dc_rest_dt + k4.dc_rest_dt) / 6.0,
        da_gut_lumen_dt: (k1.da_gut_lumen_dt + 2.0 * k2.da_gut_lumen_dt + 2.0 * k3.da_gut_lumen_dt + k4.da_gut_lumen_dt) / 6.0
    };
    
    // x_{n+1} = x_n + weighted_derivative · dt
    return add_scaled_derivative(state, deriv_weighted, dt);
}

/// Full PBPK simulation from t=0 to t_end
/// Returns final state at t=t_end
fn simulate_pbpk(
    initial: PBPKState,
    params: PBPKParams,
    drug: DrugProperties,
    t_end: f64@h,
    dt: f64@h
) -> PBPKState {
    let mut state = initial;
    let mut t = 0.0@h;
    
    while t < t_end {
        state = rk4_step_pbpk(state, params, drug, t, dt);
        t = t + dt;
    }
    
    return state;
}

// ============================================================================
// INITIALIZATION FUNCTIONS
// ============================================================================

/// Create initial state for IV bolus dosing
/// All drug enters blood compartment at t=0
fn create_iv_bolus_state(dose: f64@mg, params: PBPKParams) -> PBPKState {
    let c_blood_initial = dose / params.v_blood;
    
    return PBPKState {
        c_blood: c_blood_initial,
        c_liver: 0.0@mg_per_L,
        c_kidney: 0.0@mg_per_L,
        c_brain: 0.0@mg_per_L,
        c_heart: 0.0@mg_per_L,
        c_lung: 0.0@mg_per_L,
        c_muscle: 0.0@mg_per_L,
        c_adipose: 0.0@mg_per_L,
        c_gut: 0.0@mg_per_L,
        c_skin: 0.0@mg_per_L,
        c_bone: 0.0@mg_per_L,
        c_spleen: 0.0@mg_per_L,
        c_pancreas: 0.0@mg_per_L,
        c_rest: 0.0@mg_per_L,
        a_gut_lumen: 0.0@mg
    };
}

/// Create initial state for oral dosing
/// Drug enters gut lumen, absorbed via first-order ka
fn create_oral_dose_state(dose: f64@mg, f_oral: f64) -> PBPKState {
    let absorbed_dose = dose * f_oral;
    
    return PBPKState {
        c_blood: 0.0@mg_per_L,
        c_liver: 0.0@mg_per_L,
        c_kidney: 0.0@mg_per_L,
        c_brain: 0.0@mg_per_L,
        c_heart: 0.0@mg_per_L,
        c_lung: 0.0@mg_per_L,
        c_muscle: 0.0@mg_per_L,
        c_adipose: 0.0@mg_per_L,
        c_gut: 0.0@mg_per_L,
        c_skin: 0.0@mg_per_L,
        c_bone: 0.0@mg_per_L,
        c_spleen: 0.0@mg_per_L,
        c_pancreas: 0.0@mg_per_L,
        c_rest: 0.0@mg_per_L,
        a_gut_lumen: absorbed_dose
    };
}

// ============================================================================
// PK METRICS CALCULATION
// ============================================================================

/// Calculate AUC increment using trapezoidal rule
/// AUC = (C1 + C2) / 2 × dt
fn calculate_auc_trapezoidal(
    c1: f64@mg_per_L,
    c2: f64@mg_per_L,
    dt: f64@h
) -> f64@mg_h_per_L {
    return ((c1 + c2) / 2.0) * dt;
}

/// Track maximum concentration (Cmax)
fn find_cmax(current_cmax: f64@mg_per_L, c_new: f64@mg_per_L) -> f64@mg_per_L {
    if c_new > current_cmax {
        return c_new;
    } else {
        return current_cmax;
    }
}

/// Calculate total drug amount in body (for mass balance checks)
fn total_drug_amount(state: PBPKState, params: PBPKParams) -> f64@mg {
    return (
        state.c_blood * params.v_blood +
        state.c_liver * params.v_liver +
        state.c_kidney * params.v_kidney +
        state.c_brain * params.v_brain +
        state.c_heart * params.v_heart +
        state.c_lung * params.v_lung +
        state.c_muscle * params.v_muscle +
        state.c_adipose * params.v_adipose +
        state.c_gut * params.v_gut +
        state.c_skin * params.v_skin +
        state.c_bone * params.v_bone +
        state.c_spleen * params.v_spleen +
        state.c_pancreas * params.v_pancreas +
        state.c_rest * params.v_rest +
        state.a_gut_lumen
    );
}

// ============================================================================
// VALIDATION / TESTING HELPERS
// ============================================================================

/// Zero state (for testing)
fn zero_state() -> PBPKState {
    return PBPKState {
        c_blood: 0.0@mg_per_L,
        c_liver: 0.0@mg_per_L,
        c_kidney: 0.0@mg_per_L,
        c_brain: 0.0@mg_per_L,
        c_heart: 0.0@mg_per_L,
        c_lung: 0.0@mg_per_L,
        c_muscle: 0.0@mg_per_L,
        c_adipose: 0.0@mg_per_L,
        c_gut: 0.0@mg_per_L,
        c_skin: 0.0@mg_per_L,
        c_bone: 0.0@mg_per_L,
        c_spleen: 0.0@mg_per_L,
        c_pancreas: 0.0@mg_per_L,
        c_rest: 0.0@mg_per_L,
        a_gut_lumen: 0.0@mg
    };
}

/// Check if state is valid (no negative concentrations, no NaN/Inf)
fn is_valid_state(state: PBPKState) -> bool {
    return (
        state.c_blood >= 0.0@mg_per_L &&
        state.c_liver >= 0.0@mg_per_L &&
        state.c_kidney >= 0.0@mg_per_L &&
        state.c_brain >= 0.0@mg_per_L &&
        state.c_heart >= 0.0@mg_per_L &&
        state.c_lung >= 0.0@mg_per_L &&
        state.c_muscle >= 0.0@mg_per_L &&
        state.c_adipose >= 0.0@mg_per_L &&
        state.c_gut >= 0.0@mg_per_L &&
        state.c_skin >= 0.0@mg_per_L &&
        state.c_bone >= 0.0@mg_per_L &&
        state.c_spleen >= 0.0@mg_per_L &&
        state.c_pancreas >= 0.0@mg_per_L &&
        state.c_rest >= 0.0@mg_per_L &&
        state.a_gut_lumen >= 0.0@mg
    );
}

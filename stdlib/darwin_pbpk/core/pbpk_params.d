// Darwin PBPK Core Module - PBPK Parameters and Data Structures
// Demetrios Implementation with Unit Annotations


// =============================================================================
// Unit Type Aliases (leveraging Demetrios' unit system)
// =============================================================================

type L = f64;          // Volume (liters)
type h = f64;          // Time (hours)
type mg = f64;         // Mass (milligrams)
type kg = f64;         // Body mass (kilograms)
type L_per_h = f64;    // Clearance, blood flow (L/h)
type mg_per_L = f64;   // Concentration (mg/L)
type mg_h_per_L = f64; // AUC (mg·h/L)

// =============================================================================
// Constants for 14-Organ PBPK Model
// =============================================================================

const NUM_ORGANS: i32 = 14;

// Organ indices (enumeration)
const BLOOD: i32 = 0;
const LIVER: i32 = 1;
const KIDNEY: i32 = 2;
const BRAIN: i32 = 3;
const HEART: i32 = 4;
const LUNG: i32 = 5;
const MUSCLE: i32 = 6;
const ADIPOSE: i32 = 7;
const GUT: i32 = 8;
const SKIN: i32 = 9;
const BONE: i32 = 10;
const SPLEEN: i32 = 11;
const PANCREAS: i32 = 12;
const REST: i32 = 13;

// =============================================================================
// PBPKParams Struct - Main PBPK Parameter Container
// =============================================================================

struct PBPKParams {
    // Organ volumes (L) - 14 organs
    v_blood: f64@L,
    v_liver: f64@L,
    v_kidney: f64@L,
    v_brain: f64@L,
    v_heart: f64@L,
    v_lung: f64@L,
    v_muscle: f64@L,
    v_adipose: f64@L,
    v_gut: f64@L,
    v_skin: f64@L,
    v_bone: f64@L,
    v_spleen: f64@L,
    v_pancreas: f64@L,
    v_rest: f64@L,

    // Blood flow rates (L/h) - 14 organs
    q_blood: f64@L_per_h,
    q_liver: f64@L_per_h,
    q_kidney: f64@L_per_h,
    q_brain: f64@L_per_h,
    q_heart: f64@L_per_h,
    q_lung: f64@L_per_h,
    q_muscle: f64@L_per_h,
    q_adipose: f64@L_per_h,
    q_gut: f64@L_per_h,
    q_skin: f64@L_per_h,
    q_bone: f64@L_per_h,
    q_spleen: f64@L_per_h,
    q_pancreas: f64@L_per_h,
    q_rest: f64@L_per_h,

    // Clearance parameters (L/h)
    clearance_hepatic: f64@L_per_h,
    clearance_renal: f64@L_per_h,

    // Partition coefficients (Kp, dimensionless) - 14 organs
    kp_blood: f64,
    kp_liver: f64,
    kp_kidney: f64,
    kp_brain: f64,
    kp_heart: f64,
    kp_lung: f64,
    kp_muscle: f64,
    kp_adipose: f64,
    kp_gut: f64,
    kp_skin: f64,
    kp_bone: f64,
    kp_spleen: f64,
    kp_pancreas: f64,
    kp_rest: f64,

    // Physiological parameters (dimensionless)
    fu_plasma: f64,      // Fraction unbound in plasma
    hematocrit: f64,     // Hematocrit (0-1)
    bp_ratio: f64,       // Blood:plasma ratio
}

// =============================================================================
// PatientData Struct - Patient Demographics and Clinical State
// =============================================================================

struct PatientData {
    age: f64,           // Age in years
    weight: f64@kg,     // Body weight in kg
    height: f64,        // Height in cm
    sex: bool,          // true = male, false = female
    disease_state: i32, // Disease state enum (0=healthy, 1=renal, 2=hepatic, etc.)
}

// Disease state enumeration values
const HEALTHY: i32 = 0;
const RENAL_IMPAIRMENT: i32 = 1;
const HEPATIC_IMPAIRMENT: i32 = 2;
const CARDIAC_FAILURE: i32 = 3;
const DIABETES: i32 = 4;

// =============================================================================
// DrugProperties Struct - Drug-Specific Physicochemical Properties
// =============================================================================

struct DrugProperties {
    mw: f64,         // Molecular weight (g/mol)
    logp: f64,       // Lipophilicity (log P)
    pka: f64,        // Acid dissociation constant
    fu: f64,         // Fraction unbound (0-1)
    bp_ratio: f64,   // Blood:plasma ratio
    is_base: bool,   // true if base, false if acid
}

// =============================================================================
// Helper Functions
// =============================================================================

// Create default PBPK parameters for a 70kg male adult
fn create_default_pbpk_params() -> PBPKParams {
    return PBPKParams {
        // Default organ volumes (L) - based on ICRP reference adult male
        v_blood: 5.6,
        v_liver: 1.8,
        v_kidney: 0.31,
        v_brain: 1.45,
        v_heart: 0.33,
        v_lung: 1.17,
        v_muscle: 29.0,
        v_adipose: 14.5,
        v_gut: 1.65,
        v_skin: 3.3,
        v_bone: 4.9,
        v_spleen: 0.19,
        v_pancreas: 0.14,
        v_rest: 3.5,

        // Default blood flow rates (L/h) - based on cardiac output ~6.5 L/min
        q_blood: 390.0,
        q_liver: 97.2,   // 25% of CO (includes portal + hepatic artery)
        q_kidney: 74.1,  // 19% of CO
        q_brain: 46.8,   // 12% of CO
        q_heart: 15.6,   // 4% of CO
        q_lung: 390.0,   // 100% of CO (series circulation)
        q_muscle: 70.2,  // 18% of CO
        q_adipose: 19.5, // 5% of CO
        q_gut: 70.2,     // 18% of CO (includes portal)
        q_skin: 19.5,    // 5% of CO
        q_bone: 19.5,    // 5% of CO
        q_spleen: 11.7,  // 3% of CO
        q_pancreas: 3.9, // 1% of CO
        q_rest: 19.5,    // 5% of CO

        // Default clearance (L/h)
        clearance_hepatic: 10.0,
        clearance_renal: 5.0,

        // Default partition coefficients (typical for neutral lipophilic drug)
        kp_blood: 1.0,
        kp_liver: 3.0,
        kp_kidney: 2.5,
        kp_brain: 1.5,
        kp_heart: 2.0,
        kp_lung: 2.5,
        kp_muscle: 1.2,
        kp_adipose: 5.0,
        kp_gut: 2.5,
        kp_skin: 1.8,
        kp_bone: 0.5,
        kp_spleen: 2.5,
        kp_pancreas: 2.0,
        kp_rest: 1.5,

        // Default physiological parameters
        fu_plasma: 0.1,    // 10% unbound (typical for protein-bound drug)
        hematocrit: 0.45,  // 45% (normal adult)
        bp_ratio: 1.0,     // Equal distribution in blood and plasma
    };
}

// Create a patient profile with basic demographics
fn create_patient(age: f64, weight: f64@kg, sex: bool) -> PatientData {
    let height: f64;
    
    // Estimate height based on sex and typical BMI
    if sex {
        height = 175.0;  // Default male height (cm)
    } else {
        height = 162.0;  // Default female height (cm)
    }
    
    return PatientData {
        age: age,
        weight: weight,
        height: height,
        sex: sex,
        disease_state: HEALTHY,
    };
}

// Validate PBPK parameters for physiological plausibility
fn validate_pbpk_params(params: PBPKParams) -> bool {
    // Check organ volumes are positive
    if params.v_blood <= 0.0 { return false; }
    if params.v_liver <= 0.0 { return false; }
    if params.v_kidney <= 0.0 { return false; }
    if params.v_brain <= 0.0 { return false; }
    if params.v_heart <= 0.0 { return false; }
    if params.v_lung <= 0.0 { return false; }
    if params.v_muscle <= 0.0 { return false; }
    if params.v_adipose <= 0.0 { return false; }
    if params.v_gut <= 0.0 { return false; }
    if params.v_skin <= 0.0 { return false; }
    if params.v_bone <= 0.0 { return false; }
    if params.v_spleen <= 0.0 { return false; }
    if params.v_pancreas <= 0.0 { return false; }
    if params.v_rest <= 0.0 { return false; }

    // Check blood flows are positive
    if params.q_blood <= 0.0 { return false; }
    if params.q_liver <= 0.0 { return false; }
    if params.q_kidney <= 0.0 { return false; }
    if params.q_brain <= 0.0 { return false; }
    if params.q_heart <= 0.0 { return false; }
    if params.q_lung <= 0.0 { return false; }
    if params.q_muscle <= 0.0 { return false; }
    if params.q_adipose <= 0.0 { return false; }
    if params.q_gut <= 0.0 { return false; }
    if params.q_skin <= 0.0 { return false; }
    if params.q_bone <= 0.0 { return false; }
    if params.q_spleen <= 0.0 { return false; }
    if params.q_pancreas <= 0.0 { return false; }
    if params.q_rest <= 0.0 { return false; }

    // Check clearances are non-negative
    if params.clearance_hepatic < 0.0 { return false; }
    if params.clearance_renal < 0.0 { return false; }

    // Check partition coefficients are positive
    if params.kp_blood <= 0.0 { return false; }
    if params.kp_liver <= 0.0 { return false; }
    if params.kp_kidney <= 0.0 { return false; }
    if params.kp_brain <= 0.0 { return false; }
    if params.kp_heart <= 0.0 { return false; }
    if params.kp_lung <= 0.0 { return false; }
    if params.kp_muscle <= 0.0 { return false; }
    if params.kp_adipose <= 0.0 { return false; }
    if params.kp_gut <= 0.0 { return false; }
    if params.kp_skin <= 0.0 { return false; }
    if params.kp_bone <= 0.0 { return false; }
    if params.kp_spleen <= 0.0 { return false; }
    if params.kp_pancreas <= 0.0 { return false; }
    if params.kp_rest <= 0.0 { return false; }

    // Check fu_plasma is between 0 and 1
    if params.fu_plasma < 0.0 || params.fu_plasma > 1.0 { return false; }

    // Check hematocrit is between 0 and 1
    if params.hematocrit < 0.0 || params.hematocrit > 1.0 { return false; }

    // Check bp_ratio is positive
    if params.bp_ratio <= 0.0 { return false; }

    return true;
}

// Calculate total body volume from organ volumes
fn calculate_total_volume(params: PBPKParams) -> f64@L {
    return params.v_blood + params.v_liver + params.v_kidney + 
           params.v_brain + params.v_heart + params.v_lung + 
           params.v_muscle + params.v_adipose + params.v_gut + 
           params.v_skin + params.v_bone + params.v_spleen + 
           params.v_pancreas + params.v_rest;
}

// Calculate cardiac output from blood flows (excluding lung which is series)
fn calculate_cardiac_output(params: PBPKParams) -> f64@L_per_h {
    return params.q_liver + params.q_kidney + params.q_brain + 
           params.q_heart + params.q_muscle + params.q_adipose + 
           params.q_gut + params.q_skin + params.q_bone + 
           params.q_spleen + params.q_pancreas + params.q_rest;
}

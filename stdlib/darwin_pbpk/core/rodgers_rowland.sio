// rodgers_rowland.d
// Rodgers-Rowland mechanistic tissue:plasma partition coefficient model
// Literature: Rodgers & Rowland (2006-2007) J Pharm Sci

use core::math::{pow, log10};

// ============================================================================
// TISSUE COMPOSITION DATA
// ============================================================================

/// Tissue-specific composition parameters for Kp prediction
struct TissueComposition {
    name: String,
    f_water: f64,              // Total water fraction (extracellular + intracellular)
    f_ew: f64,                 // Extracellular water fraction
    f_iw: f64,                 // Intracellular water fraction
    f_neutral_lipid: f64,      // Neutral lipid fraction
    f_phospholipid: f64,       // Total phospholipid fraction
    f_acidic_phospholipid: f64, // Acidic phospholipid fraction
    f_protein: f64,            // Protein fraction (tissue-specific binding)
    pH_iw: f64,                // Intracellular water pH
}

impl TissueComposition {
    fn new(
        name: String,
        f_water: f64,
        f_ew: f64,
        f_iw: f64,
        f_nl: f64,
        f_pl: f64,
        f_ap: f64,
        f_prot: f64,
        ph_iw: f64
    ) -> Self {
        TissueComposition {
            name,
            f_water,
            f_ew,
            f_iw,
            f_neutral_lipid: f_nl,
            f_phospholipid: f_pl,
            f_acidic_phospholipid: f_ap,
            f_protein: f_prot,
            pH_iw: ph_iw,
        }
    }
}

// ============================================================================
// HARDCODED HUMAN TISSUE DATA
// From Rodgers & Rowland (2006), Poulin & Theil (2002)
// ============================================================================

fn get_adipose_composition() -> TissueComposition {
    TissueComposition::new(
        "adipose".to_string(),
        0.18,   // f_water
        0.135,  // f_ew (extracellular)
        0.045,  // f_iw (intracellular)
        0.79,   // f_neutral_lipid
        0.002,  // f_phospholipid
        0.0004, // f_acidic_phospholipid
        0.001,  // f_protein
        7.0     // pH_iw
    )
}

fn get_bone_composition() -> TissueComposition {
    TissueComposition::new(
        "bone".to_string(),
        0.439,  // f_water
        0.329,  // f_ew
        0.110,  // f_iw
        0.074,  // f_neutral_lipid
        0.011,  // f_phospholipid
        0.002,  // f_acidic_phospholipid
        0.070,  // f_protein
        7.0     // pH_iw
    )
}

fn get_brain_composition() -> TissueComposition {
    TissueComposition::new(
        "brain".to_string(),
        0.78,   // f_water
        0.162,  // f_ew
        0.618,  // f_iw
        0.039,  // f_neutral_lipid
        0.054,  // f_phospholipid
        0.012,  // f_acidic_phospholipid
        0.016,  // f_protein
        7.0     // pH_iw
    )
}

fn get_gut_composition() -> TissueComposition {
    TissueComposition::new(
        "gut".to_string(),
        0.76,   // f_water
        0.282,  // f_ew
        0.478,  // f_iw
        0.035,  // f_neutral_lipid
        0.025,  // f_phospholipid
        0.005,  // f_acidic_phospholipid
        0.016,  // f_protein
        7.0     // pH_iw
    )
}

fn get_heart_composition() -> TissueComposition {
    TissueComposition::new(
        "heart".to_string(),
        0.79,   // f_water
        0.320,  // f_ew
        0.470,  // f_iw
        0.017,  // f_neutral_lipid
        0.029,  // f_phospholipid
        0.006,  // f_acidic_phospholipid
        0.157,  // f_protein
        7.0     // pH_iw
    )
}

fn get_kidney_composition() -> TissueComposition {
    TissueComposition::new(
        "kidney".to_string(),
        0.79,   // f_water
        0.273,  // f_ew
        0.517,  // f_iw
        0.012,  // f_neutral_lipid
        0.029,  // f_phospholipid
        0.006,  // f_acidic_phospholipid
        0.130,  // f_protein
        7.0     // pH_iw
    )
}

fn get_liver_composition() -> TissueComposition {
    TissueComposition::new(
        "liver".to_string(),
        0.76,   // f_water
        0.161,  // f_ew
        0.599,  // f_iw
        0.035,  // f_neutral_lipid
        0.025,  // f_phospholipid
        0.005,  // f_acidic_phospholipid
        0.086,  // f_protein
        7.0     // pH_iw
    )
}

fn get_lung_composition() -> TissueComposition {
    TissueComposition::new(
        "lung".to_string(),
        0.80,   // f_water
        0.336,  // f_ew
        0.464,  // f_iw
        0.013,  // f_neutral_lipid
        0.027,  // f_phospholipid
        0.006,  // f_acidic_phospholipid
        0.100,  // f_protein
        7.0     // pH_iw
    )
}

fn get_muscle_composition() -> TissueComposition {
    TissueComposition::new(
        "muscle".to_string(),
        0.76,   // f_water
        0.118,  // f_ew
        0.642,  // f_iw
        0.010,  // f_neutral_lipid
        0.009,  // f_phospholipid
        0.001,  // f_acidic_phospholipid
        0.190,  // f_protein
        7.0     // pH_iw
    )
}

fn get_skin_composition() -> TissueComposition {
    TissueComposition::new(
        "skin".to_string(),
        0.72,   // f_water
        0.382,  // f_ew
        0.338,  // f_iw
        0.015,  // f_neutral_lipid
        0.012,  // f_phospholipid
        0.003,  // f_acidic_phospholipid
        0.116,  // f_protein
        7.0     // pH_iw
    )
}

fn get_spleen_composition() -> TissueComposition {
    TissueComposition::new(
        "spleen".to_string(),
        0.79,   // f_water
        0.207,  // f_ew
        0.583,  // f_iw
        0.012,  // f_neutral_lipid
        0.029,  // f_phospholipid
        0.006,  // f_acidic_phospholipid
        0.130,  // f_protein
        7.0     // pH_iw
    )
}

fn get_pancreas_composition() -> TissueComposition {
    TissueComposition::new(
        "pancreas".to_string(),
        0.76,   // f_water
        0.282,  // f_ew
        0.478,  // f_iw
        0.035,  // f_neutral_lipid
        0.025,  // f_phospholipid
        0.005,  // f_acidic_phospholipid
        0.016,  // f_protein
        7.0     // pH_iw
    )
}

// ============================================================================
// DRUG PROPERTIES STRUCT
// ============================================================================

/// Drug physicochemical properties required for Kp prediction
struct DrugProperties {
    name: String,
    logp: f64,              // Octanol:water partition coefficient (log scale)
    pka: f64,               // Dissociation constant
    is_base: bool,          // true = base, false = acid/neutral
    fu_plasma: f64,         // Fraction unbound in plasma (0-1)
    blood_plasma_ratio: f64, // Blood:plasma concentration ratio
}

impl DrugProperties {
    fn new(
        name: String,
        logp: f64,
        pka: f64,
        is_base: bool,
        fu_plasma: f64,
        bp_ratio: f64
    ) -> Self {
        DrugProperties {
            name,
            logp,
            pka,
            is_base,
            fu_plasma,
            blood_plasma_ratio: bp_ratio,
        }
    }
}

// ============================================================================
// ALL Kp VALUES OUTPUT
// ============================================================================

struct AllKpValues {
    kp_adipose: f64,
    kp_bone: f64,
    kp_brain: f64,
    kp_gut: f64,
    kp_heart: f64,
    kp_kidney: f64,
    kp_liver: f64,
    kp_lung: f64,
    kp_muscle: f64,
    kp_skin: f64,
    kp_spleen: f64,
    kp_pancreas: f64,
}

impl AllKpValues {
    fn new() -> Self {
        AllKpValues {
            kp_adipose: 0.0,
            kp_bone: 0.0,
            kp_brain: 0.0,
            kp_gut: 0.0,
            kp_heart: 0.0,
            kp_kidney: 0.0,
            kp_liver: 0.0,
            kp_lung: 0.0,
            kp_muscle: 0.0,
            kp_skin: 0.0,
            kp_spleen: 0.0,
            kp_pancreas: 0.0,
        }
    }
}

// ============================================================================
// IONIZATION AND PARTITIONING CALCULATIONS
// ============================================================================

/// Calculate ionization ratio (ionized/unionized) at given pH
/// For bases: ratio = 10^(pKa - pH)
/// For acids: ratio = 10^(pH - pKa)
fn calculate_ionization_ratio(pka: f64, pH: f64, is_base: bool) -> f64 {
    if is_base {
        pow(10.0, pka - pH)
    } else {
        pow(10.0, pH - pka)
    }
}

/// Calculate fraction unbound in tissue
/// Based on tissue protein binding and lipid partitioning
fn calculate_fut(
    fu_plasma: f64,
    tissue: &TissueComposition,
    logp: f64,
    pka: f64,
    is_base: bool
) -> f64 {
    // Partition coefficient (linear scale)
    let p = pow(10.0, logp);
    
    // Ionization at plasma pH (assume 7.4)
    let pH_plasma = 7.4;
    let ion_ratio_plasma = calculate_ionization_ratio(pka, pH_plasma, is_base);
    
    // Ionization at tissue pH
    let ion_ratio_tissue = calculate_ionization_ratio(pka, tissue.pH_iw, is_base);
    
    // Tissue binding parameters
    let ka = 0.5; // Association constant for acidic phospholipids (L/kg)
    let kn = 0.3 * p + 0.7; // Neutral lipid partitioning
    
    // Total tissue binding
    let tissue_binding = tissue.f_ew 
        + (tissue.f_iw / (1.0 + ion_ratio_tissue))
        + (kn * tissue.f_neutral_lipid)
        + ((0.3 * p + 0.7) * tissue.f_phospholipid);
    
    // Plasma binding (assume standard albumin binding)
    let plasma_binding = 1.0;
    
    // Fraction unbound in tissue
    let fut = fu_plasma * (plasma_binding / tissue_binding);
    
    // Clamp between 0.001 and 1.0
    if fut < 0.001 {
        0.001
    } else if fut > 1.0 {
        1.0
    } else {
        fut
    }
}

/// Main Rodgers-Rowland Kp calculation
/// Returns tissue:plasma partition coefficient (dimensionless)
fn calculate_kp_rodgers_rowland(
    drug: &DrugProperties,
    tissue: &TissueComposition
) -> f64 {
    let p = pow(10.0, drug.logp); // Partition coefficient (linear)
    
    // Plasma pH
    let pH_plasma = 7.4;
    
    // Ionization ratios
    let ion_plasma = calculate_ionization_ratio(drug.pka, pH_plasma, drug.is_base);
    let ion_tissue = calculate_ionization_ratio(drug.pka, tissue.pH_iw, drug.is_base);
    
    // Partition coefficients for different compartments
    let ka = 0.5;  // Acidic phospholipid association
    let kn = 0.3 * p + 0.7; // Neutral lipid partition
    let kp_np = 0.3 * p + 0.7; // Neutral phospholipid partition
    
    // Calculate Kpu (partition coefficient of unionized drug)
    let kpu = (tissue.f_ew + tissue.f_iw + (kn * tissue.f_neutral_lipid) 
              + (kp_np * tissue.f_phospholipid)) / drug.fu_plasma;
    
    // Different equations for bases vs acids/neutrals
    let kp = if drug.is_base {
        // For bases: enhanced tissue partitioning due to ion trapping
        let kp_base = kpu * (tissue.f_ew + (tissue.f_iw * (1.0 + ion_tissue) / (1.0 + ion_plasma)))
                     + (kn * tissue.f_neutral_lipid)
                     + (kp_np * tissue.f_phospholipid)
                     + (ka * tissue.f_acidic_phospholipid * (1.0 + ion_tissue));
        
        kp_base / drug.fu_plasma
    } else {
        // For acids/neutrals: standard partitioning
        let kp_acid = tissue.f_ew + (tissue.f_iw / (1.0 + ion_tissue))
                     + (kn * tissue.f_neutral_lipid)
                     + (kp_np * (tissue.f_phospholipid - tissue.f_acidic_phospholipid))
                     + ((ka * (1.0 + ion_tissue)) * tissue.f_acidic_phospholipid);
        
        (drug.fu_plasma * kp_acid) / calculate_fut(
            drug.fu_plasma,
            tissue,
            drug.logp,
            drug.pka,
            drug.is_base
        )
    };
    
    // Return Kp (minimum 0.01 to avoid numerical issues)
    if kp < 0.01 {
        0.01
    } else {
        kp
    }
}

// ============================================================================
// BATCH PREDICTION FUNCTIONS
// ============================================================================

/// Predict Kp for all 12 major tissues
fn predict_all_kp(drug: &DrugProperties) -> AllKpValues {
    let mut kps = AllKpValues::new();
    
    kps.kp_adipose = calculate_kp_rodgers_rowland(drug, &get_adipose_composition());
    kps.kp_bone = calculate_kp_rodgers_rowland(drug, &get_bone_composition());
    kps.kp_brain = calculate_kp_rodgers_rowland(drug, &get_brain_composition());
    kps.kp_gut = calculate_kp_rodgers_rowland(drug, &get_gut_composition());
    kps.kp_heart = calculate_kp_rodgers_rowland(drug, &get_heart_composition());
    kps.kp_kidney = calculate_kp_rodgers_rowland(drug, &get_kidney_composition());
    kps.kp_liver = calculate_kp_rodgers_rowland(drug, &get_liver_composition());
    kps.kp_lung = calculate_kp_rodgers_rowland(drug, &get_lung_composition());
    kps.kp_muscle = calculate_kp_rodgers_rowland(drug, &get_muscle_composition());
    kps.kp_skin = calculate_kp_rodgers_rowland(drug, &get_skin_composition());
    kps.kp_spleen = calculate_kp_rodgers_rowland(drug, &get_spleen_composition());
    kps.kp_pancreas = calculate_kp_rodgers_rowland(drug, &get_pancreas_composition());
    
    kps
}

/// Calculate volume of distribution at steady state (Vdss)
/// Vdss = Vp + sum(Vt × Kp × fu_p / fu_t) for all tissues
/// 
/// Standard tissue volumes (L/kg body weight):
/// - Plasma: 0.0436
/// - Adipose: 0.2142 (variable with BMI)
/// - Bone: 0.0857
/// - Brain: 0.0200
/// - Gut: 0.0171
/// - Heart: 0.0047
/// - Kidney: 0.0044
/// - Liver: 0.0257
/// - Lung: 0.0076
/// - Muscle: 0.4000
/// - Skin: 0.0371
/// - Spleen: 0.0026
/// - Pancreas: 0.0014
fn predict_vdss(drug: &DrugProperties, patient_weight_kg: f64) -> f64 {
    let kps = predict_all_kp(drug);
    
    // Tissue volumes as fraction of body weight (L/kg)
    let v_plasma = 0.0436;
    let v_adipose = 0.2142;
    let v_bone = 0.0857;
    let v_brain = 0.0200;
    let v_gut = 0.0171;
    let v_heart = 0.0047;
    let v_kidney = 0.0044;
    let v_liver = 0.0257;
    let v_lung = 0.0076;
    let v_muscle = 0.4000;
    let v_skin = 0.0371;
    let v_spleen = 0.0026;
    let v_pancreas = 0.0014;
    
    // Calculate Vdss (L)
    let vdss = v_plasma * patient_weight_kg +
               (v_adipose * kps.kp_adipose +
                v_bone * kps.kp_bone +
                v_brain * kps.kp_brain +
                v_gut * kps.kp_gut +
                v_heart * kps.kp_heart +
                v_kidney * kps.kp_kidney +
                v_liver * kps.kp_liver +
                v_lung * kps.kp_lung +
                v_muscle * kps.kp_muscle +
                v_skin * kps.kp_skin +
                v_spleen * kps.kp_spleen +
                v_pancreas * kps.kp_pancreas) * patient_weight_kg * drug.fu_plasma;
    
    vdss
}

// ============================================================================
// EXAMPLE USAGE
// ============================================================================

fn example_metformin() {
    // Metformin: hydrophilic base
    let metformin = DrugProperties::new(
        "Metformin".to_string(),
        -1.43,  // logP (hydrophilic)
        12.4,   // pKa (strong base)
        true,   // is_base
        1.0,    // fu_plasma (not protein bound)
        1.08    // blood:plasma ratio
    );
    
    let kps = predict_all_kp(&metformin);
    let vdss = predict_vdss(&metformin, 70.0); // 70 kg patient
    
    println!("Metformin Kp predictions:");
    println!("  Liver Kp: {}", kps.kp_liver);
    println!("  Kidney Kp: {}", kps.kp_kidney);
    println!("  Muscle Kp: {}", kps.kp_muscle);
    println!("  Vdss: {} L", vdss);
}

fn example_propranolol() {
    // Propranolol: lipophilic base
    let propranolol = DrugProperties::new(
        "Propranolol".to_string(),
        3.48,   // logP (lipophilic)
        9.42,   // pKa (weak base)
        true,   // is_base
        0.07,   // fu_plasma (highly protein bound)
        0.89    // blood:plasma ratio
    );
    
    let kps = predict_all_kp(&propranolol);
    let vdss = predict_vdss(&propranolol, 70.0);
    
    println!("Propranolol Kp predictions:");
    println!("  Liver Kp: {}", kps.kp_liver);
    println!("  Brain Kp: {}", kps.kp_brain);
    println!("  Adipose Kp: {}", kps.kp_adipose);
    println!("  Vdss: {} L", vdss);
}

// ============================================================================
// PUBLIC API EXPORTS
// ============================================================================

pub fn main() {
    println!("Rodgers-Rowland PBPK Partition Coefficient Module");
    println!("==================================================");
    println!("");
    
    example_metformin();
    println!("");
    example_propranolol();
}

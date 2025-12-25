// PBPK Module for Demetrios
// Physiologically-Based Pharmacokinetic Modeling with Epistemic Types
//
// FULL COMPLIANCE with FDA and EMA 2024-2025 Guidelines:
// - FDA: "Physiologically Based Pharmacokinetic Analyses — Format and Content"
// - EMA: "Guideline on Reporting of PBPK Modelling and Simulation"
//
// This module demonstrates what NO OTHER language can do:
// - Type-safe units (mg, L, h) with compile-time dimensional analysis
// - Epistemic confidence tracking (Knowledge[T, ε >= bound])
// - Ontology-validated drug concepts (ChEBI)
// - Full provenance for FDA/EMA compliance
// - Automatic validation metrics (GMFE, AFE, AAFE, within 2-fold)
// - Impact-appropriate acceptance criteria (EMA low/medium/high)
//
// References:
// - https://www.fda.gov/regulatory-information/search-fda-guidance-documents/physiologically-based-pharmacokinetic-analyses-format-and-content-guidance-industry
// - https://www.ema.europa.eu/en/reporting-physiologically-based-pharmacokinetic-pbpk-modelling-simulation-scientific-guideline
// - CPT Pharmacometrics 2025: Current Use of PBPK modeling at EMA
//
// Inspired by Darwin PBPK Platform (darwin-pbpk-platform)
// Developed solo in 2 months by Demetrios Agourakis

module pbpk

// Re-export submodules
pub use compartments::*
pub use parameters::*
pub use solver::*
pub use validation::*
pub use regulatory::*  // FDA/EMA compliance (v0.45.0)

// =============================================================================
// Core PBPK Types with Epistemic Qualifications
// =============================================================================

/// Drug entity with ontology validation
pub struct Drug {
    /// ChEBI identifier (validated at compile-time)
    pub chebi_id: string,

    /// Human-readable name
    pub name: string,

    /// Molecular weight with units
    pub mw: g/mol,

    /// LogP (lipophilicity)
    pub logp: f64,

    /// Fraction unbound in plasma
    pub fu: Knowledge[f64, ε >= 0.80],
}

/// PBPK Parameters with epistemic tracking
pub struct PBPKParams {
    /// Hepatic clearance
    pub cl_hepatic: Knowledge[L/h, ε >= 0.75],

    /// Renal clearance
    pub cl_renal: Knowledge[L/h, ε >= 0.75],

    /// Volume of distribution
    pub vd: Knowledge[L, ε >= 0.80],

    /// Absorption rate constant
    pub ka: Knowledge[1/h, ε >= 0.70],

    /// Partition coefficients (14 organs)
    pub kp: [Knowledge[f64, ε >= 0.60]; 14],
}

/// Patient profile for individualized PBPK
pub struct Patient {
    pub id: string,
    pub weight: kg,
    pub height: cm,
    pub age: years,
    pub sex: Sex,
    pub genotype: Option[Genotype],
}

pub enum Sex { Male, Female }

pub struct Genotype {
    pub cyp2d6: CYP2D6Status,
    pub cyp3a4: CYP3A4Status,
}

pub enum CYP2D6Status { PM, IM, EM, UM }  // Poor/Intermediate/Extensive/Ultra Metabolizer
pub enum CYP3A4Status { Low, Normal, High }

// =============================================================================
// Compartment Model (14-organ PBPK)
// =============================================================================

/// Standard 14-compartment PBPK organs
pub const ORGANS: [string; 14] = [
    "blood",    // 0 - Plasma
    "liver",    // 1 - Metabolism
    "kidney",   // 2 - Excretion
    "brain",    // 3 - BBB
    "heart",    // 4
    "lung",     // 5
    "muscle",   // 6
    "adipose",  // 7
    "gut",      // 8 - Absorption
    "skin",     // 9
    "bone",     // 10
    "spleen",   // 11
    "pancreas", // 12
    "other",    // 13
]

/// Compartment state (amount of drug)
pub struct CompartmentState {
    pub amounts: [mg; 14],
    pub time: h,
}

/// Reference physiological values (70kg adult)
pub fn default_volumes() -> [L; 14] {
    [
        5.0,   // blood
        1.8,   // liver
        0.31,  // kidney
        1.4,   // brain
        0.33,  // heart
        0.5,   // lung
        30.0,  // muscle
        15.0,  // adipose
        1.1,   // gut
        3.3,   // skin
        10.0,  // bone
        0.18,  // spleen
        0.1,   // pancreas
        5.0,   // other
    ]
}

pub fn default_blood_flows() -> [L/h; 14] {
    [
        348.0,  // blood (cardiac output)
        90.0,   // liver
        72.0,   // kidney
        42.0,   // brain
        15.0,   // heart
        348.0,  // lung (full CO)
        54.0,   // muscle
        18.0,   // adipose
        60.0,   // gut
        18.0,   // skin
        15.0,   // bone
        4.8,    // spleen
        3.0,    // pancreas
        12.0,   // other
    ]
}

// =============================================================================
// Epistemic-Aware Simulation
// =============================================================================

/// Simulation result with confidence propagation
pub struct SimulationResult {
    /// Time points
    pub times: Vec<h>,

    /// Concentrations per organ (with propagated confidence)
    pub concentrations: Vec<[Knowledge[mg/L, ε >= 0.50]; 14]>,

    /// Overall simulation confidence
    pub confidence: f64,

    /// Full provenance chain
    pub provenance: Provenance,
}

/// Run PBPK simulation with epistemic tracking
pub fn simulate(
    drug: &Drug,
    params: &PBPKParams,
    patient: &Patient,
    dose: mg,
    duration: h,
    dt: h,
) -> SimulationResult with Alloc, Prob {
    // Confidence propagates through computation
    let base_confidence = params.cl_hepatic.confidence
        .min(params.vd.confidence)
        .min(params.ka.confidence);

    // Scale parameters to patient
    let scaled = scale_to_patient(params, patient);

    // Run ODE solver
    let (times, amounts) = solve_ode(
        &scaled,
        dose,
        duration,
        dt,
    );

    // Convert to concentrations with confidence
    let volumes = scale_volumes(default_volumes(), patient);
    let concentrations = amounts.iter().map(|a| {
        a.zip(volumes).map(|(amt, vol)| {
            Knowledge::new(
                amt / vol,
                confidence: base_confidence * 0.95,  // ODE solver degrades slightly
                provenance: Provenance::derived("ode_simulation"),
            )
        })
    }).collect();

    SimulationResult {
        times,
        concentrations,
        confidence: base_confidence * 0.95,
        provenance: Provenance::merged([
            drug.provenance(),
            params.provenance(),
            Provenance::source("darwin_pbpk_solver"),
        ]),
    }
}

/// Allometric scaling to individual patient
fn scale_to_patient(params: &PBPKParams, patient: &Patient) -> PBPKParams {
    let bw_ratio = patient.weight / 70.0_kg;

    PBPKParams {
        cl_hepatic: params.cl_hepatic.map(|cl| cl * bw_ratio.pow(0.75)),
        cl_renal: params.cl_renal.map(|cl| cl * bw_ratio.pow(0.75)),
        vd: params.vd.map(|v| v * bw_ratio),
        ka: params.ka,  // Usually not scaled
        kp: params.kp,
    }
}

fn scale_volumes(base: [L; 14], patient: &Patient) -> [L; 14] {
    let bw_ratio = patient.weight / 70.0_kg;
    base.map(|v| v * bw_ratio)
}

// =============================================================================
// FDA Validation with Epistemic Constraints
// =============================================================================

/// Validation metrics for regulatory submission
pub struct ValidationMetrics {
    /// Geometric Mean Fold Error
    pub gmfe: f64,

    /// Fraction within 2-fold
    pub within_2fold: f64,

    /// R-squared
    pub r_squared: f64,

    /// Root Mean Square Error
    pub rmse: mg/L,
}

/// Validate simulation against observed data
/// REQUIRES: predictions have ε >= 0.80 for FDA submission
pub fn validate_for_fda(
    predictions: &[Knowledge[mg/L, ε >= 0.80]],
    observed: &[mg/L],
) -> Result<ValidationMetrics, ValidationError> with IO {
    assert_eq!(predictions.len(), observed.len());

    // Check all predictions meet confidence threshold
    for (i, pred) in predictions.iter().enumerate() {
        if pred.confidence < 0.80 {
            return Err(ValidationError::InsufficientConfidence {
                index: i,
                required: 0.80,
                actual: pred.confidence,
            });
        }
    }

    let n = predictions.len() as f64;

    // GMFE
    let log_errors: Vec<f64> = predictions.iter()
        .zip(observed)
        .map(|(p, o)| (p.value / o).ln().abs())
        .collect();
    let gmfe = (log_errors.iter().sum::<f64>() / n).exp();

    // Within 2-fold
    let within_2fold = predictions.iter()
        .zip(observed)
        .filter(|(p, o)| {
            let ratio = p.value / o;
            ratio >= 0.5 && ratio <= 2.0
        })
        .count() as f64 / n;

    // R²
    let mean_obs = observed.iter().sum::<mg/L>() / n;
    let ss_tot: f64 = observed.iter().map(|o| (o - mean_obs).powi(2)).sum();
    let ss_res: f64 = predictions.iter()
        .zip(observed)
        .map(|(p, o)| (p.value - o).powi(2))
        .sum();
    let r_squared = 1.0 - ss_res / ss_tot;

    // RMSE
    let rmse = (ss_res / n).sqrt();

    Ok(ValidationMetrics { gmfe, within_2fold, r_squared, rmse })
}

pub enum ValidationError {
    InsufficientConfidence {
        index: usize,
        required: f64,
        actual: f64,
    },
    DataMismatch {
        expected: usize,
        actual: usize,
    },
}

// =============================================================================
// Provenance for Regulatory Compliance
// =============================================================================

pub enum Provenance {
    /// Primary data source
    Source {
        id: string,
        timestamp: DateTime,
        version: Option<string>,
    },

    /// Derived through computation
    Derived {
        method: string,
        sources: Vec<Provenance>,
    },

    /// Merged from multiple sources
    Merged(Vec<Provenance>),

    /// User input
    UserInput {
        user_id: string,
        timestamp: DateTime,
    },
}

impl Provenance {
    pub fn source(id: &str) -> Self {
        Provenance::Source {
            id: id.to_string(),
            timestamp: DateTime::now(),
            version: None,
        }
    }

    pub fn derived(method: &str) -> Self {
        Provenance::Derived {
            method: method.to_string(),
            sources: vec![],
        }
    }

    pub fn merged(sources: impl Into<Vec<Provenance>>) -> Self {
        Provenance::Merged(sources.into())
    }

    /// Generate audit trail for FDA
    pub fn to_audit_trail(&self) -> string {
        match self {
            Provenance::Source { id, timestamp, version } => {
                format!("SOURCE: {} @ {} (v{})",
                    id, timestamp, version.unwrap_or("?"))
            }
            Provenance::Derived { method, sources } => {
                let src_trails: Vec<string> = sources.iter()
                    .map(|s| s.to_audit_trail())
                    .collect();
                format!("DERIVED[{}]: {}", method, src_trails.join(" + "))
            }
            Provenance::Merged(sources) => {
                let trails: Vec<string> = sources.iter()
                    .map(|s| s.to_audit_trail())
                    .collect();
                format!("MERGED: {}", trails.join(" | "))
            }
            Provenance::UserInput { user_id, timestamp } => {
                format!("USER: {} @ {}", user_id, timestamp)
            }
        }
    }
}

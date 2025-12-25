// covariate.d - Covariate Models for Population PK/PBPK
//
// Implements allometric scaling and covariate effects following
// MedLang Track D Pharmacometrics QSP Specification.
//
// Key features:
// - Allometric scaling (weight-based parameter adjustment)
// - Renal function covariate (creatinine clearance)
// - Hepatic function covariate (Child-Pugh)
// - Genotype effects (CYP450 polymorphisms)
// - Age-based adjustments
//
// Reference: FDA Guidance on Population PK, EMA Guidelines
// Inspired by MedLang (github.com/agourakis82/medlang)
//
// Module: pbpk::covariate (for future module system)

// =============================================================================
// MATH HELPERS (must be defined first - no forward declarations)
// =============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn exp_f64(x: f64) -> f64 {
    if x > 20.0 { return exp_f64(x / 2.0) * exp_f64(x / 2.0) }
    if x < 0.0 - 20.0 { return 1.0 / exp_f64(0.0 - x) }
    let mut sum = 1.0
    let mut term = 1.0
    let mut i = 1
    while i <= 15 {
        term = term * x / i
        sum = sum + term
        i = i + 1
    }
    return sum
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000000.0 }
    let e = 2.718281828459045
    let mut val = x
    let mut k = 0.0
    while val > e {
        val = val / e
        k = k + 1.0
    }
    while val < 1.0 / e {
        val = val * e
        k = k - 1.0
    }
    let u = (val - 1.0) / (val + 1.0)
    let u2 = u * u
    let mut sum = u
    let mut term = u
    term = term * u2
    sum = sum + term / 3.0
    term = term * u2
    sum = sum + term / 5.0
    term = term * u2
    sum = sum + term / 7.0
    term = term * u2
    sum = sum + term / 9.0
    return 2.0 * sum + k
}

fn pow_f64(x: f64, n: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    return exp_f64(n * ln_f64(x))
}

// =============================================================================
// ALLOMETRIC SCALING
// =============================================================================

// Standard allometric exponents (FDA/EMA recommended)
fn allometric_exp_clearance() -> f64 { return 0.75 }  // 3/4 power law
fn allometric_exp_volume() -> f64 { return 1.0 }      // Linear with weight
fn allometric_exp_flow() -> f64 { return 0.75 }       // Same as clearance

// Reference body weight (standard adult)
fn reference_weight() -> f64 { return 70.0 }          // 70 kg

// Allometric scaling for clearance parameters
// CL_individual = CL_pop * (WT/70)^0.75
fn allometric_clearance(
    cl_pop: f64,
    weight: f64,
    ref_wt: f64,
    exponent: f64
) -> f64 {
    let wt_ratio = weight / ref_wt
    return cl_pop * pow_f64(wt_ratio, exponent)
}

// Allometric scaling for volume parameters
// V_individual = V_pop * (WT/70)^1.0
fn allometric_volume(
    v_pop: f64,
    weight: f64,
    ref_wt: f64,
    exponent: f64
) -> f64 {
    let wt_ratio = weight / ref_wt
    return v_pop * pow_f64(wt_ratio, exponent)
}

// Scale clearance using standard 0.75 exponent
fn scale_clearance(cl_pop: f64, weight: f64) -> f64 {
    return allometric_clearance(cl_pop, weight, 70.0, 0.75)
}

// Scale volume using standard 1.0 exponent
fn scale_volume(v_pop: f64, weight: f64) -> f64 {
    return allometric_volume(v_pop, weight, 70.0, 1.0)
}

// Scale blood flow using 0.75 exponent (same as clearance)
fn scale_blood_flow(q_pop: f64, weight: f64) -> f64 {
    return allometric_clearance(q_pop, weight, 70.0, 0.75)
}

// =============================================================================
// RENAL FUNCTION COVARIATES
// =============================================================================

// Estimate creatinine clearance using Cockcroft-Gault equation
// CrCL (mL/min) = [(140 - age) * weight] / (72 * SCr) * [0.85 if female]
fn cockcroft_gault(
    age: f64,          // years
    weight: f64,       // kg
    scr: f64,          // serum creatinine mg/dL
    is_female: bool
) -> f64 {
    let base = (140.0 - age) * weight / (72.0 * scr)
    if is_female {
        return base * 0.85
    }
    return base
}

// Estimate GFR using CKD-EPI equation (2021)
// More accurate than Cockcroft-Gault for most populations
fn ckd_epi_2021(
    scr: f64,          // serum creatinine mg/dL
    age: f64,          // years
    is_female: bool
) -> f64 {
    let kappa = if is_female { 0.7 } else { 0.9 }
    let alpha = if is_female { 0.0 - 0.241 } else { 0.0 - 0.302 }

    let scr_kappa = scr / kappa
    let min_term = if scr_kappa < 1.0 { scr_kappa } else { 1.0 }
    let max_term = if scr_kappa > 1.0 { scr_kappa } else { 1.0 }

    let base = 142.0 * pow_f64(min_term, alpha) * pow_f64(max_term, 0.0 - 1.200)
    let age_factor = pow_f64(0.9938, age)
    let sex_factor = if is_female { 1.012 } else { 1.0 }

    return base * age_factor * sex_factor
}

// Adjust clearance for renal function
// For drugs with significant renal elimination
fn renal_adjustment(
    cl_pop: f64,       // Population clearance (100% renal function)
    crcl: f64,         // Patient's creatinine clearance (mL/min)
    ref_crcl: f64,     // Reference CrCL (typically 120 mL/min)
    fe: f64            // Fraction excreted unchanged in urine
) -> f64 {
    // CL = CL_nonrenal + CL_renal * (CrCL/CrCL_ref)
    let cl_renal = cl_pop * fe
    let cl_nonrenal = cl_pop * (1.0 - fe)
    return cl_nonrenal + cl_renal * (crcl / ref_crcl)
}

// =============================================================================
// HEPATIC FUNCTION COVARIATES
// =============================================================================

// Child-Pugh score categories
fn child_pugh_a() -> i32 { return 5 }   // 5-6 points: Mild
fn child_pugh_b() -> i32 { return 7 }   // 7-9 points: Moderate
fn child_pugh_c() -> i32 { return 10 }  // 10-15 points: Severe

// Adjust hepatic clearance for Child-Pugh class
// Based on typical recommendations in drug labels
fn hepatic_adjustment(
    cl_pop: f64,       // Population clearance
    child_pugh: i32,   // Child-Pugh score (5-15)
    fh: f64            // Fraction cleared hepatically
) -> f64 {
    // Adjustment factors based on FDA guidance
    let adjustment = if child_pugh <= 6 {
        1.0                // Child-Pugh A: no adjustment
    } else {
        if child_pugh <= 9 {
            0.5            // Child-Pugh B: 50% reduction
        } else {
            0.25           // Child-Pugh C: 75% reduction
        }
    }

    let cl_hepatic = cl_pop * fh
    let cl_other = cl_pop * (1.0 - fh)
    return cl_other + cl_hepatic * adjustment
}

// =============================================================================
// GENOTYPE EFFECTS (CYP450 POLYMORPHISMS)
// =============================================================================

// CYP2D6 metabolizer phenotype effects
// Returns multiplication factor for CYP2D6-mediated clearance
fn cyp2d6_effect(phenotype: i32) -> f64 {
    // 0=PM, 1=IM, 2=EM, 3=UM
    if phenotype == 0 { return 0.1 }    // Poor metabolizer: 10% activity
    if phenotype == 1 { return 0.5 }    // Intermediate: 50% activity
    if phenotype == 2 { return 1.0 }    // Extensive (normal): 100%
    if phenotype == 3 { return 2.0 }    // Ultra-rapid: 200%
    return 1.0  // Default to normal
}

// CYP3A4 activity effect
// Returns multiplication factor for CYP3A4-mediated clearance
fn cyp3a4_effect(activity_level: i32) -> f64 {
    // 0=Low, 1=Normal, 2=High (induced)
    if activity_level == 0 { return 0.5 }    // Low activity
    if activity_level == 1 { return 1.0 }    // Normal
    if activity_level == 2 { return 2.0 }    // Induced (e.g., rifampin)
    return 1.0
}

// Adjust clearance for CYP2D6 phenotype
fn adjust_for_cyp2d6(
    cl_pop: f64,        // Population clearance
    phenotype: i32,     // 0=PM, 1=IM, 2=EM, 3=UM
    fm_cyp2d6: f64      // Fraction metabolized by CYP2D6
) -> f64 {
    let factor = cyp2d6_effect(phenotype)
    let cl_cyp2d6 = cl_pop * fm_cyp2d6
    let cl_other = cl_pop * (1.0 - fm_cyp2d6)
    return cl_other + cl_cyp2d6 * factor
}

// =============================================================================
// AGE-BASED ADJUSTMENTS
// =============================================================================

// Age-based clearance adjustment (maturation and aging)
// Based on sigmoid Emax model for pediatric, linear decline for geriatric
fn age_adjustment(
    cl_adult: f64,      // Adult clearance (reference)
    age: f64,           // Patient age in years
    pma50: f64,         // Post-menstrual age at 50% maturation (pediatric)
    hill: f64           // Hill coefficient for maturation curve
) -> f64 {
    if age < 18.0 {
        // Pediatric: sigmoid maturation
        // Assumes PMA = age + 0.75 years (40 weeks gestation)
        let pma = age + 0.75
        let maturation = pow_f64(pma, hill) / (pow_f64(pma50, hill) + pow_f64(pma, hill))
        return cl_adult * maturation
    } else {
        if age > 65.0 {
            // Geriatric: ~1% decline per year after 65
            let decline = 1.0 - 0.01 * (age - 65.0)
            let min_decline = 0.5  // Floor at 50% of adult
            if decline < min_decline {
                return cl_adult * min_decline
            }
            return cl_adult * decline
        } else {
            // Adult: no adjustment
            return cl_adult
        }
    }
}

// =============================================================================
// COMPOSITE COVARIATE MODELS
// =============================================================================

// Full covariate model for clearance
// Combines allometric, renal, hepatic, and genotype effects
struct CovariateFactors {
    weight: f64,
    crcl: f64,
    child_pugh: i32,
    cyp2d6: i32,
    age: f64
}

struct DrugCharacteristics {
    fe: f64,            // Fraction excreted renally
    fh: f64,            // Fraction cleared hepatically
    fm_cyp2d6: f64      // Fraction metabolized by CYP2D6
}

// Calculate individual clearance from population value
fn individual_clearance(
    cl_pop: f64,
    cov: CovariateFactors,
    drug: DrugCharacteristics
) -> f64 {
    // Start with allometric scaling
    let cl_scaled = scale_clearance(cl_pop, cov.weight)

    // Apply renal adjustment
    let cl_renal_adj = renal_adjustment(cl_scaled, cov.crcl, 120.0, drug.fe)

    // Apply hepatic adjustment
    let cl_hepatic_adj = hepatic_adjustment(cl_renal_adj, cov.child_pugh, drug.fh)

    // Apply CYP2D6 effect
    let cl_cyp_adj = adjust_for_cyp2d6(cl_hepatic_adj, cov.cyp2d6, drug.fm_cyp2d6)

    // Apply age adjustment (simplified)
    let cl_final = age_adjustment(cl_cyp_adj, cov.age, 0.5, 3.0)

    return cl_final
}

// =============================================================================
// TESTS
// =============================================================================

fn main() -> i32 {
    println("=== Covariate Model Tests ===")
    println("")

    // Test 1: Allometric scaling
    println("Test 1: Allometric scaling")
    let cl_pop = 10.0  // 10 L/h
    let cl_80kg = scale_clearance(cl_pop, 80.0)
    let cl_50kg = scale_clearance(cl_pop, 50.0)
    println("  CL_pop = 10 L/h")
    println("  CL @ 80kg = ")
    println(cl_80kg)
    println("  CL @ 50kg = ")
    println(cl_50kg)
    // Expected: 80kg -> ~10.7 L/h, 50kg -> ~8.4 L/h
    println("")

    // Test 2: Cockcroft-Gault
    println("Test 2: Cockcroft-Gault CrCL")
    let test_crcl = cockcroft_gault(40.0, 70.0, 1.0, false)
    println("  40yo, 70kg male, SCr=1.0: CrCL = ")
    println(test_crcl)
    // Expected: (140-40)*70/(72*1.0) = 97.2 mL/min
    println("")

    // Test 3: Renal adjustment
    println("Test 3: Renal clearance adjustment")
    let cl_normal = 10.0
    let cl_impaired = renal_adjustment(cl_normal, 30.0, 120.0, 0.6)
    println("  CL_pop=10, CrCL=30 (vs 120), fe=0.6")
    println("  CL_adjusted = ")
    println(cl_impaired)
    // Expected: 4.0 + 6.0*(30/120) = 4.0 + 1.5 = 5.5 L/h
    println("")

    // Test 4: CYP2D6 effect
    println("Test 4: CYP2D6 phenotype effect")
    let cl_em = adjust_for_cyp2d6(10.0, 2, 0.8)  // EM
    let cl_pm = adjust_for_cyp2d6(10.0, 0, 0.8)  // PM
    println("  CL_pop=10, fm_CYP2D6=0.8")
    println("  CL @ EM (normal) = ")
    println(cl_em)
    println("  CL @ PM (poor) = ")
    println(cl_pm)
    // Expected: EM -> 10.0, PM -> 2.0 + 8.0*0.1 = 2.8
    println("")

    // Validation - use nested ifs (no && operator)
    // Expected: 80kg -> ~11.05 L/h (from (80/70)^0.75 * 10)
    let err1 = abs_f64(cl_80kg - 11.05)
    let err2 = abs_f64(test_crcl - 97.22)
    let err3 = abs_f64(cl_impaired - 5.5)
    let err4 = abs_f64(cl_pm - 2.8)

    if err1 < 0.1 {
        if err2 < 0.1 {
            if err3 < 0.1 {
                if err4 < 0.1 {
                    println("ALL TESTS PASSED")
                    return 0
                }
            }
        }
    }

    println("SOME TESTS FAILED")
    println("  err1 = ")
    println(err1)
    println("  err2 = ")
    println(err2)
    println("  err3 = ")
    println(err3)
    println("  err4 = ")
    println(err4)
    return 1
}

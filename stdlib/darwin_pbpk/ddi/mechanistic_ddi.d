// mechanistic_ddi.d
// Mechanistic Drug-Drug Interaction (DDI) Module
// CRITICAL MODULE FOR CLINICAL SAFETY PREDICTION
//
// Implements FDA guidance-compliant DDI prediction models:
// - Enzyme inhibition (competitive, noncompetitive, uncompetitive, mixed, TDI)
// - Enzyme induction (PXR, CAR, AhR pathways)
// - Transporter-mediated DDI (OATP, P-gp)
// - AUC ratio prediction (basic and mechanistic)
// - Risk classification (weak/moderate/strong)
//
// References:
// - FDA Drug Interaction Studies (2020)
// - EMA Guideline on DDI Studies (2012)
// - Rostami-Hodjegan A. (2012) J Pharm Sci

module mechanistic_ddi;

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

/// Enzyme inhibition parameters
struct EnzymeInhibition {
    ki: f64@mg_per_L;           // Inhibition constant
    inhibition_type: i32;        // 0=competitive, 1=noncompetitive, 2=uncompetitive, 3=mixed
    is_tdi: bool;                // Time-dependent inhibition (mechanism-based)
    kinact: f64;                 // Inactivation rate constant (1/h) for TDI
    alpha: f64;                  // Mixed inhibition parameter (α > 1 favors ESI binding)
}

/// Enzyme induction parameters
struct EnzymeInduction {
    ec50: f64@mg_per_L;         // Half-maximal induction concentration
    emax: f64;                   // Maximal fold induction (Emax = 10 means 10-fold increase)
    pathway: i32;                // 0=PXR (CYP3A4), 1=CAR (CYP2B6), 2=AhR (CYP1A2)
    hill_coefficient: f64;       // Hill coefficient (typically 1-3)
}

/// CYP isoform-specific inhibition data
struct CYPInhibitionData {
    cyp3a4_ki: f64@mg_per_L;
    cyp2d6_ki: f64@mg_per_L;
    cyp2c9_ki: f64@mg_per_L;
    cyp2c19_ki: f64@mg_per_L;
    cyp1a2_ki: f64@mg_per_L;
}

/// Mixed inhibition result (affects both Km and Vmax)
struct MixedResult {
    km_app: f64@mg_per_L;       // Apparent Km
    vmax_app: f64;               // Apparent Vmax (fold change)
}

/// DDI prediction result
struct DDIPrediction {
    auc_ratio: f64;              // AUCR = AUC_with_inhibitor / AUC_without
    cmax_ratio: f64;             // Cmax ratio
    risk_category: i32;          // 0=none, 1=weak, 2=moderate, 3=strong
    mechanism: i32;              // 0=inhibition, 1=induction, 2=transporter
}

// ============================================================================
// ENZYME INHIBITION MODELS
// ============================================================================

/// Competitive inhibition: increases apparent Km
/// Km_app = Km × (1 + [I]/Ki)
/// Returns: fold-increase in Km
fn competitive_inhibition(km: f64@mg_per_L, i: f64@mg_per_L, ki: f64@mg_per_L) -> f64 {
    let fold_increase: f64 = 1.0 + (i / ki);
    return fold_increase;
}

/// Noncompetitive inhibition: decreases Vmax
/// Vmax_app = Vmax / (1 + [I]/Ki)
/// Returns: fold-decrease in Vmax (fraction remaining)
fn noncompetitive_inhibition(vmax: f64, i: f64@mg_per_L, ki: f64@mg_per_L) -> f64 {
    let vmax_app: f64 = vmax / (1.0 + (i / ki));
    return vmax_app;
}

/// Uncompetitive inhibition: binds only to ES complex
/// Km_app = Km / (1 + [I]/Ki)
/// Vmax_app = Vmax / (1 + [I]/Ki)
/// Returns: fold-decrease factor
fn uncompetitive_inhibition(i: f64@mg_per_L, ki: f64@mg_per_L) -> f64 {
    let factor: f64 = 1.0 / (1.0 + (i / ki));
    return factor;
}

/// Mixed inhibition: affects both Km and Vmax
/// α = Ki(ESI) / Ki(EI) (alpha > 1 favors ESI, alpha < 1 favors EI)
/// Km_app = Km × (1 + [I]/Ki) / (1 + [I]/(α×Ki))
/// Vmax_app = Vmax / (1 + [I]/(α×Ki))
fn mixed_inhibition(km: f64@mg_per_L, vmax: f64, i: f64@mg_per_L, ki: f64@mg_per_L, alpha: f64) -> MixedResult {
    let i_over_ki: f64 = i / ki;
    let i_over_alpha_ki: f64 = i / (alpha * ki);
    
    let km_app: f64@mg_per_L = km * (1.0 + i_over_ki) / (1.0 + i_over_alpha_ki);
    let vmax_app: f64 = vmax / (1.0 + i_over_alpha_ki);
    
    let result: MixedResult = MixedResult {
        km_app: km_app,
        vmax_app: vmax_app
    };
    
    return result;
}

/// Mechanism-based (time-dependent) inhibition
/// Enzyme activity: A(t) = A0 × exp(-kinact × [I]/(Ki + [I]) × t - kdeg × t)
/// Returns: fraction of enzyme remaining
fn mechanism_based_inhibition(kinact: f64, ki: f64@mg_per_L, i: f64@mg_per_L, kdeg: f64, time: f64@h) -> f64 {
    let kinact_app: f64 = kinact * i / (ki + i);
    let total_rate: f64 = kinact_app + kdeg;
    let fraction_remaining: f64 = exp(-total_rate * time);
    
    return fraction_remaining;
}

// ============================================================================
// AUC RATIO PREDICTION (FDA METHODS)
// ============================================================================

/// Basic static DDI model (FDA 2020)
/// AUCR = 1 / [fm_CYP × (1/(1 + [I]/Ki)) + (1 - fm_CYP)]
/// For strong inhibition (fm ~ 1): AUCR ≈ 1 + [I]/Ki
///
/// Args:
///   fm_cyp: Fraction metabolized by the affected CYP (0-1)
///   ki: Inhibition constant
///   i_max: Systemic Cmax of inhibitor (use unbound if Ki is from microsomal data)
fn predict_auc_ratio_basic(fm_cyp: f64, ki: f64@mg_per_L, i_max: f64@mg_per_L) -> f64 {
    let inhibition_factor: f64 = 1.0 / (1.0 + (i_max / ki));
    let auc_ratio: f64 = 1.0 / (fm_cyp * inhibition_factor + (1.0 - fm_cyp));
    
    return auc_ratio;
}

/// Mechanistic liver DDI model with inlet concentration
/// For hepatic metabolism, use inlet (portal vein) concentration:
/// I_inlet = I_max × (1 + ka × Dose × Fa × Fg / (Qh × fu_p))
///
/// Args:
///   fm_cyp: Fraction metabolized by affected CYP
///   ki: Inhibition constant
///   i_inlet_max: Maximum inhibitor concentration at liver inlet
///   fu_plasma: Fraction unbound in plasma
///   fu_liver: Fraction unbound in liver (typically fu_p / fu_mic)
fn predict_auc_ratio_liver(
    fm_cyp: f64,
    ki: f64@mg_per_L,
    i_inlet_max: f64@mg_per_L,
    fu_plasma: f64,
    fu_liver: f64
) -> f64 {
    // Adjust Ki for protein binding
    let ki_adj: f64@mg_per_L = ki * fu_liver / fu_plasma;
    
    let inhibition_factor: f64 = 1.0 / (1.0 + (i_inlet_max / ki_adj));
    let auc_ratio: f64 = 1.0 / (fm_cyp * inhibition_factor + (1.0 - fm_cyp));
    
    return auc_ratio;
}

/// Calculate inlet concentration for liver DDI
/// I_inlet,max = Cmax + (ka × Dose × Fa × Fg) / (Qh × RB)
///
/// Args:
///   cmax: Systemic Cmax of inhibitor
///   ka: Absorption rate constant (1/h)
///   dose: Oral dose (mg)
///   fa: Fraction absorbed
///   fg: Intestinal availability
///   qh: Hepatic blood flow (L/h) - typically 90 L/h for 70kg human
///   rb: Blood-to-plasma ratio
fn calculate_inlet_concentration(
    cmax: f64@mg_per_L,
    ka: f64,
    dose: f64,
    fa: f64,
    fg: f64,
    qh: f64,
    rb: f64
) -> f64@mg_per_L {
    let gut_contribution: f64@mg_per_L = (ka * dose * fa * fg) / (qh * rb);
    let i_inlet: f64@mg_per_L = cmax + gut_contribution;
    
    return i_inlet;
}

// ============================================================================
// ENZYME INDUCTION MODELS
// ============================================================================

/// Calculate enzyme induction effect (Emax model)
/// Fold_induction = 1 + Emax × [C]^n / (EC50^n + [C]^n)
/// where n is the Hill coefficient (default = 1)
///
/// Returns: fold-increase in enzyme activity (e.g., 3.0 = 3-fold induction)
fn calculate_induction_effect(
    c_hepatocyte: f64@mg_per_L,
    ec50: f64@mg_per_L,
    emax: f64
) -> f64 {
    let fold_induction: f64 = 1.0 + (emax * c_hepatocyte) / (ec50 + c_hepatocyte);
    return fold_induction;
}

/// Induction with Hill coefficient (sigmoidal)
fn calculate_induction_hill(
    c_hepatocyte: f64@mg_per_L,
    ec50: f64@mg_per_L,
    emax: f64,
    hill: f64
) -> f64 {
    let c_pow: f64 = pow(c_hepatocyte, hill);
    let ec50_pow: f64 = pow(ec50, hill);
    
    let fold_induction: f64 = 1.0 + (emax * c_pow) / (ec50_pow + c_pow);
    return fold_induction;
}

/// Net effect when both induction and inhibition occur
/// Induction increases enzyme synthesis, inhibition decreases activity
/// Net AUCR can be complex: measure both separately then combine
///
/// Simplified: AUCR_net = AUCR_inhibition / fold_induction
fn net_effect_induction_inhibition(induction_fold: f64, inhibition_aucr: f64) -> f64 {
    let net_aucr: f64 = inhibition_aucr / induction_fold;
    return net_aucr;
}

// ============================================================================
// TRANSPORTER-MEDIATED DDI
// ============================================================================

/// OATP1B1/1B3 inhibition effect on hepatic uptake
/// Inhibition of hepatic uptake increases systemic exposure
/// AUCR = 1 + (fm_OATP / fu_p) × [I_portal] / Ki
///
/// Args:
///   fm_oatp: Fraction of hepatic uptake via OATP (typically 0.3-0.9 for statins)
///   fu_p: Fraction unbound in plasma
///   i_portal: Inhibitor concentration in portal vein
///   ki_oatp: Ki for OATP inhibition
fn oatp_inhibition_effect(
    fm_oatp: f64,
    fu_p: f64,
    i_portal: f64@mg_per_L,
    ki_oatp: f64@mg_per_L
) -> f64 {
    let auc_ratio: f64 = 1.0 + (fm_oatp / fu_p) * (i_portal / ki_oatp);
    return auc_ratio;
}

/// P-glycoprotein (P-gp/MDR1) inhibition at gut
/// Increases oral bioavailability by reducing efflux
/// Fg_ratio = (1 + [I_gut]/Ki_pgp) / (1 + baseline_activity)
///
/// Simplified: AUCR ≈ 1 + [I_gut]/Ki_pgp (for high baseline efflux)
fn pgp_inhibition_effect(i_gut: f64@mg_per_L, ki_pgp: f64@mg_per_L) -> f64 {
    let auc_ratio: f64 = 1.0 + (i_gut / ki_pgp);
    return auc_ratio;
}

/// Estimate gut concentration for P-gp DDI
/// I_gut ≈ Dose / (250 mL) for immediate release formulation
/// Use actual Fa × Dose / Volume_gut for more precision
fn estimate_gut_concentration(dose: f64, volume_gut: f64) -> f64@mg_per_L {
    let c_gut: f64@mg_per_L = dose / volume_gut;
    return c_gut;
}

// ============================================================================
// DDI RISK CLASSIFICATION (FDA GUIDANCE)
// ============================================================================

/// Classify DDI risk based on AUC ratio
/// FDA categories:
///   Weak: 1.25 - 2x
///   Moderate: 2 - 5x
///   Strong: ≥ 5x
///   None: < 1.25x
///
/// Returns: 0=none, 1=weak, 2=moderate, 3=strong
fn classify_ddi_risk(auc_ratio: f64) -> i32 {
    if auc_ratio < 1.25 {
        return 0;  // No interaction
    } else if auc_ratio < 2.0 {
        return 1;  // Weak
    } else if auc_ratio < 5.0 {
        return 2;  // Moderate
    } else {
        return 3;  // Strong
    }
}

/// Classify induction risk (AUC decrease)
/// Strong induction: AUCR < 0.2 (80% decrease)
/// Moderate: 0.2 - 0.5
/// Weak: 0.5 - 0.8
fn classify_induction_risk(auc_ratio: f64) -> i32 {
    if auc_ratio >= 0.8 {
        return 0;  // No significant induction
    } else if auc_ratio >= 0.5 {
        return 1;  // Weak
    } else if auc_ratio >= 0.2 {
        return 2;  // Moderate
    } else {
        return 3;  // Strong
    }
}

// ============================================================================
// CLINICAL EXAMPLES (HARDCODED REFERENCE DATA)
// ============================================================================

/// Ketoconazole + Midazolam: Strong CYP3A4 inhibition
/// Clinical AUCR: ~15x (10-16x range)
/// Ketoconazole: potent CYP3A4 inhibitor (Ki ~0.015 μM)
fn example_ketoconazole_midazolam() -> DDIPrediction {
    let fm_cyp3a4: f64 = 0.95;  // Midazolam almost exclusively CYP3A4
    let ki_keto: f64@mg_per_L = 0.008;  // 0.008 mg/L = 0.015 μM
    let cmax_keto: f64@mg_per_L = 6.0;   // Typical Cmax 400mg dose
    
    let auc_ratio: f64 = predict_auc_ratio_basic(fm_cyp3a4, ki_keto, cmax_keto);
    let risk: i32 = classify_ddi_risk(auc_ratio);
    
    let result: DDIPrediction = DDIPrediction {
        auc_ratio: auc_ratio,
        cmax_ratio: auc_ratio * 0.8,  // Approximate
        risk_category: risk,
        mechanism: 0  // Inhibition
    };
    
    return result;
}

/// Fluconazole + Warfarin: Moderate CYP2C9 inhibition
/// Clinical AUCR: ~2x (S-warfarin)
/// Fluconazole: CYP2C9 inhibitor (Ki ~5 μM)
fn example_fluconazole_warfarin() -> DDIPrediction {
    let fm_cyp2c9: f64 = 0.9;  // S-warfarin metabolism
    let ki_fluc: f64@mg_per_L = 1.5;  // ~5 μM
    let cmax_fluc: f64@mg_per_L = 6.0;  // 200mg dose
    
    let auc_ratio: f64 = predict_auc_ratio_basic(fm_cyp2c9, ki_fluc, cmax_fluc);
    let risk: i32 = classify_ddi_risk(auc_ratio);
    
    let result: DDIPrediction = DDIPrediction {
        auc_ratio: auc_ratio,
        cmax_ratio: auc_ratio * 0.9,
        risk_category: risk,
        mechanism: 0
    };
    
    return result;
}

/// Rifampin + Midazolam: Strong CYP3A4 induction
/// Clinical AUCR: ~0.04x (96% decrease)
/// Rifampin: potent PXR-mediated CYP3A4 inducer
fn example_rifampin_midazolam() -> DDIPrediction {
    // Induction typically modeled as fold-increase in Vmax/CLint
    let fold_induction: f64 = 10.0;  // 10-fold increase in CYP3A4
    let fm_cyp3a4: f64 = 0.95;
    
    // AUCR ≈ 1 / fold_induction for high fm
    let auc_ratio: f64 = 1.0 / fold_induction;
    let risk: i32 = classify_induction_risk(auc_ratio);
    
    let result: DDIPrediction = DDIPrediction {
        auc_ratio: auc_ratio,
        cmax_ratio: auc_ratio * 1.2,  // Cmax less affected
        risk_category: risk,
        mechanism: 1  // Induction
    };
    
    return result;
}

/// Cyclosporine + Rosuvastatin: OATP1B1 inhibition
/// Clinical AUCR: ~7x
/// Cyclosporine: potent OATP1B1/1B3 inhibitor
fn example_cyclosporine_rosuvastatin() -> DDIPrediction {
    let fm_oatp: f64 = 0.8;  // Rosuvastatin highly dependent on OATP uptake
    let fu_p: f64 = 0.1;     // 10% unbound
    let ki_oatp: f64@mg_per_L = 0.1;  // Very potent
    let i_portal: f64@mg_per_L = 2.0;  // High local concentration
    
    let auc_ratio: f64 = oatp_inhibition_effect(fm_oatp, fu_p, i_portal, ki_oatp);
    let risk: i32 = classify_ddi_risk(auc_ratio);
    
    let result: DDIPrediction = DDIPrediction {
        auc_ratio: auc_ratio,
        cmax_ratio: auc_ratio,
        risk_category: risk,
        mechanism: 2  // Transporter
    };
    
    return result;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Convert Ki from μM to mg/L using molecular weight
fn ki_from_micromolar(ki_um: f64, mw: f64) -> f64@mg_per_L {
    let ki_mg_l: f64@mg_per_L = ki_um * mw / 1000.0;
    return ki_mg_l;
}

/// Calculate R-value (FDA preliminary cutoff)
/// R = 1 + [I]/Ki
/// If R > 1.1: potential DDI, requires clinical study
fn calculate_r_value(i: f64@mg_per_L, ki: f64@mg_per_L) -> f64 {
    let r: f64 = 1.0 + (i / ki);
    return r;
}

/// Create default CYP inhibition data (no inhibition)
fn create_default_cyp_data() -> CYPInhibitionData {
    let data: CYPInhibitionData = CYPInhibitionData {
        cyp3a4_ki: 1000.0,  // Very high = no inhibition
        cyp2d6_ki: 1000.0,
        cyp2c9_ki: 1000.0,
        cyp2c19_ki: 1000.0,
        cyp1a2_ki: 1000.0
    };
    return data;
}

/// Helper: exp function (placeholder - implement in stdlib)
fn exp(x: f64) -> f64 {
    // Exponential function - should be implemented in math stdlib
    // For now, approximate or link to Rust std::f64::exp
    return 2.718281828459045 ** x;  // Using power operator
}

/// Helper: pow function
fn pow(base: f64, exponent: f64) -> f64 {
    return base ** exponent;
}

// ============================================================================
// COMMENTS ON CLINICAL USE
// ============================================================================

// CRITICAL NOTES:
//
// 1. **In vitro-to-in vivo extrapolation (IVIVE)**:
//    - Ki from microsomes: use unbound plasma concentration
//    - Ki from hepatocytes: use total plasma concentration
//    - Always specify which [I] is used (Cmax,u vs Cmax,total)
//
// 2. **FDA R-value cutoffs** (2020 guidance):
//    - R = 1 + [I]/Ki
//    - For reversible inhibition: R > 1.02 requires clinical study
//    - For TDI: R > 1.25 at hepatic inlet
//    - For induction: Emax/EC50 ratio
//
// 3. **Protein binding corrections**:
//    - Always critical for lipophilic drugs
//    - fu,mic typically ~3x fu,plasma for acidic/neutral drugs
//
// 4. **Time-dependent effects**:
//    - TDI: max effect after several doses (enzyme resynthesis ~24-48h)
//    - Induction: steady-state after 7-14 days
//    - Always specify time point for predictions
//
// 5. **Multiple pathways**:
//    - Sum fm values: fm_CYP3A4 + fm_CYP2D6 + ... + fm_renal = 1.0
//    - Use network model for complex DDI (e.g., dual inhibition)
//
// 6. **Validation**:
//    - Predicted AUCR within 2-fold of clinical = acceptable
//    - Strong inhibitors: often underpredict (use higher [I])
//    - Inducers: high variability (PXR polymorphisms)


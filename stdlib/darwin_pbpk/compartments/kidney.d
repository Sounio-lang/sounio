// kidney.d - Advanced Kidney Compartment Module for Darwin PBPK
// Implements comprehensive renal physiology, clearance mechanisms, and nephrotoxicity
// Version: 1.0.0

module darwin_pbpk::compartments::kidney;

// ============================================================================
// RENAL PHYSIOLOGY STRUCTURES
// ============================================================================

/// Kidney physiological parameters for PBPK modeling
/// Reference: Davies & Morris (1993), Price et al. (2003)
struct KidneyPhysiology {
    volume: f64@L,                    // 0.31 L for both kidneys (0.44% BW)
    blood_flow: f64@L_per_h,          // 1200 L/h = 20 L/min (20-25% cardiac output)
    gfr: f64@L_per_h,                 // 7.5 L/h = 125 mL/min (normal adult)
    filtration_fraction: f64,          // 0.2 (GFR/renal plasma flow)
    nephron_count: f64,                // 1e6 per kidney (2e6 total)
    plasma_flow: f64@L_per_h,         // Derived: blood_flow × (1 - hematocrit)
    urine_flow: f64@L_per_h,          // 0.06 L/h = 1 mL/min (0.5-2 L/day)
    urine_pH: f64,                     // 5.5-7.0 (typically 6.0)
}

impl KidneyPhysiology {
    /// Create standard adult kidney physiology
    /// Reference: ICRP Publication 89 (2002)
    fn new_adult(body_weight: f64@kg, hematocrit: f64) -> KidneyPhysiology {
        let volume = 0.31@L;  // 0.0044 * BW for allometric scaling
        let blood_flow = 1200.0@L_per_h;  // 20 L/min
        let gfr = 7.5@L_per_h;  // 125 mL/min
        let plasma_flow = blood_flow * (1.0 - hematocrit);
        
        KidneyPhysiology {
            volume: volume,
            blood_flow: blood_flow,
            gfr: gfr,
            filtration_fraction: 0.2,
            nephron_count: 2.0e6,
            plasma_flow: plasma_flow,
            urine_flow: 0.06@L_per_h,  // 1.5 L/day
            urine_pH: 6.0,
        }
    }
    
    /// Allometric scaling for different body weights
    /// Reference: Mahmood (2007) - Allometric scaling in drug development
    fn scale_for_weight(bw: f64@kg, hematocrit: f64) -> KidneyPhysiology {
        let bw_ref = 70.0@kg;
        let exponent_volume = 1.0;  // Linear for volume
        let exponent_flow = 0.75;   // 3/4 power law for flow
        
        let volume = 0.31@L * pow(bw / bw_ref, exponent_volume);
        let blood_flow = 1200.0@L_per_h * pow(bw / bw_ref, exponent_flow);
        let gfr = 7.5@L_per_h * pow(bw / bw_ref, exponent_flow);
        let plasma_flow = blood_flow * (1.0 - hematocrit);
        
        KidneyPhysiology {
            volume: volume,
            blood_flow: blood_flow,
            gfr: gfr,
            filtration_fraction: 0.2,
            nephron_count: 2.0e6,
            plasma_flow: plasma_flow,
            urine_flow: 0.06@L_per_h * pow(bw / bw_ref, exponent_flow),
            urine_pH: 6.0,
        }
    }
}

/// Renal transporter abundances (pmol/mg protein)
/// Critical for drug-drug interactions and secretion clearance
/// Reference: Prasad et al. (2016), Drozdzik et al. (2014)
struct RenalTransporters {
    // Basolateral uptake transporters (blood → cell)
    oat1_abundance: f64,      // SLC22A6 - PAH, NSAIDs, antivirals (acyclovir, tenofovir)
    oat3_abundance: f64,      // SLC22A8 - Statins, loop diuretics, methotrexate
    oct2_abundance: f64,      // SLC22A2 - Metformin, cisplatin, cimetidine
    
    // Apical efflux transporters (cell → urine)
    mate1_abundance: f64,     // SLC47A1 - H+/drug exchanger, OCT2 substrate efflux
    mate2k_abundance: f64,    // SLC47A2 - Kidney-specific MATE
    pgp_abundance: f64,       // ABCB1 - Digoxin, cyclosporine, tacrolimus
    mrp2_abundance: f64,      // ABCC2 - Conjugated metabolites, methotrexate
    mrp4_abundance: f64,      // ABCC4 - Nucleoside analogs, diuretics
    
    // Basolateral efflux (cell → blood) - for reabsorption
    mrp3_abundance: f64,      // ABCC3 - Glucuronides
    mrp6_abundance: f64,      // ABCC6 - Conjugates
}

impl RenalTransporters {
    /// Create standard adult renal transporter profile
    /// Reference: Prasad et al. (2016) - Proteomics data
    fn new_adult() -> RenalTransporters {
        RenalTransporters {
            // Uptake (basolateral)
            oat1_abundance: 12.5,    // High expression
            oat3_abundance: 18.3,    // Highest renal SLC
            oct2_abundance: 8.7,     // Moderate-high
            
            // Efflux (apical)
            mate1_abundance: 6.2,
            mate2k_abundance: 4.8,   // Kidney-specific
            pgp_abundance: 3.5,
            mrp2_abundance: 9.1,     // High for conjugates
            mrp4_abundance: 5.3,
            
            // Basolateral efflux
            mrp3_abundance: 2.1,
            mrp6_abundance: 1.4,
        }
    }
    
    /// Pediatric transporter ontogeny (age in years)
    /// Reference: Brouwer et al. (2015), Cheung et al. (2019)
    fn scale_for_age(age: f64) -> RenalTransporters {
        let adult = RenalTransporters::new_adult();
        
        // Maturation function: abundance = adult × (age^n / (TM50^n + age^n))
        let tm50_oat = 0.5;      // OATs mature early (neonatal)
        let tm50_oct = 1.0;      // OCT2 matures by 1 year
        let tm50_mate = 2.0;     // MATEs mature by 2 years
        let tm50_pgp = 0.3;      // P-gp expressed early
        let tm50_mrp = 1.5;      // MRPs mature by 1.5 years
        
        let hill = 4.0;  // Hill coefficient
        
        fn maturation_factor(age: f64, tm50: f64, hill: f64) -> f64 {
            pow(age, hill) / (pow(tm50, hill) + pow(age, hill))
        }
        
        RenalTransporters {
            oat1_abundance: adult.oat1_abundance * maturation_factor(age, tm50_oat, hill),
            oat3_abundance: adult.oat3_abundance * maturation_factor(age, tm50_oat, hill),
            oct2_abundance: adult.oct2_abundance * maturation_factor(age, tm50_oct, hill),
            mate1_abundance: adult.mate1_abundance * maturation_factor(age, tm50_mate, hill),
            mate2k_abundance: adult.mate2k_abundance * maturation_factor(age, tm50_mate, hill),
            pgp_abundance: adult.pgp_abundance * maturation_factor(age, tm50_pgp, hill),
            mrp2_abundance: adult.mrp2_abundance * maturation_factor(age, tm50_mrp, hill),
            mrp4_abundance: adult.mrp4_abundance * maturation_factor(age, tm50_mrp, hill),
            mrp3_abundance: adult.mrp3_abundance * maturation_factor(age, tm50_mrp, hill),
            mrp6_abundance: adult.mrp6_abundance * maturation_factor(age, tm50_mrp, hill),
        }
    }
}

// ============================================================================
// RENAL CLEARANCE MECHANISMS
// ============================================================================

/// Calculate glomerular filtration clearance
/// CL_filt = GFR × fu (only unbound drug is filtered)
/// Reference: Rowland & Tozer (2011) - Clinical Pharmacokinetics
fn calculate_filtration_clearance(gfr: f64@L_per_h, fu: f64) -> f64@L_per_h {
    return gfr * fu;
}

/// Calculate active tubular secretion clearance (Michaelis-Menten)
/// Used for OAT1/OAT3/OCT2-mediated secretion
/// Reference: Scotcher et al. (2016), Huang et al. (2018)
fn calculate_secretion_clearance(
    km: f64@mg_per_L,           // Michaelis constant
    vmax: f64@mg_per_h,         // Maximum secretion rate
    c_unbound: f64@mg_per_L,    // Unbound plasma concentration
    q_kidney: f64@L_per_h       // Renal plasma flow
) -> f64@L_per_h {
    // Michaelis-Menten: CL_sec = Vmax / (Km + Cu)
    // Limited by renal plasma flow (physiological maximum)
    let cl_int = vmax / (km + c_unbound);
    
    // Well-stirred model with flow limitation
    let cl_sec = (q_kidney * cl_int) / (q_kidney + cl_int);
    
    return cl_sec;
}

/// Calculate passive tubular reabsorption (pH-dependent)
/// Reference: Weiner & Hamm (2007) - Molecular mechanisms of renal acidification
fn calculate_reabsorption(
    fraction_reabsorbed: f64,    // Intrinsic reabsorption fraction
    urine_flow: f64@L_per_h,     // Urine production rate
    fu: f64,                      // Fraction unbound
    pka: f64,                     // Drug pKa
    urine_pH: f64,                // Urine pH (5.5-7.0)
    is_base: bool                 // True for bases, false for acids
) -> f64 {
    // Henderson-Hasselbalch equation for ionization
    let ph_diff = urine_pH - pka;
    let ionization_ratio = if is_base {
        // For bases: ratio of ionized/unionized = 10^(pH - pKa)
        pow(10.0, ph_diff)
    } else {
        // For acids: ratio of ionized/unionized = 10^(pKa - pH)
        pow(10.0, -ph_diff)
    };
    
    // Fraction unionized (lipophilic, can be reabsorbed)
    let f_unionized = 1.0 / (1.0 + ionization_ratio);
    
    // Effective reabsorption depends on unionized fraction
    // Ionized drugs are trapped in urine (no reabsorption)
    let effective_reabsorption = fraction_reabsorbed * f_unionized;
    
    return effective_reabsorption;
}

/// Calculate total renal clearance
/// CL_renal = CL_filt + CL_sec - CL_reab
/// Or equivalently: CL_renal = (CL_filt + CL_sec) × (1 - F_reab)
/// Reference: Pang & Rowland (1977) - Hepatic clearance theory applied to kidney
fn calculate_total_renal_clearance(
    cl_filt: f64@L_per_h,
    cl_sec: f64@L_per_h,
    fraction_reabsorbed: f64
) -> f64@L_per_h {
    let cl_renal = (cl_filt + cl_sec) * (1.0 - fraction_reabsorbed);
    return cl_renal;
}

// ============================================================================
// RENAL IMPAIRMENT & DISEASE STATES
// ============================================================================

/// Scale GFR for age-related decline
/// GFR declines ~1 mL/min per year after age 40
/// Reference: Cockcroft & Gault (1976), Davies & Shock (1950)
fn scale_gfr_for_age(gfr_young: f64@L_per_h, age: f64) -> f64@L_per_h {
    if age <= 40.0 {
        return gfr_young;
    } else {
        // Decline: 1 mL/min/year = 0.06 L/h/year
        let years_over_40 = age - 40.0;
        let decline = 0.06@L_per_h * years_over_40;
        let gfr_aged = gfr_young - decline;
        
        // Minimum GFR of 1.8 L/h (30 mL/min) for stage 3b CKD
        return max(gfr_aged, 1.8@L_per_h);
    }
}

/// Scale clearance for chronic kidney disease (CKD)
/// CKD stages: G1 (>90), G2 (60-89), G3a (45-59), G3b (30-44), G4 (15-29), G5 (<15 mL/min)
/// Reference: KDIGO 2012 Clinical Practice Guideline
fn scale_clearance_for_ckd(
    cl_normal: f64@L_per_h,      // Clearance in normal renal function
    gfr_patient: f64@L_per_h,    // Patient's GFR
    gfr_normal: f64@L_per_h,     // Normal GFR (7.5 L/h = 125 mL/min)
    fe_renal: f64                 // Fraction excreted unchanged in urine
) -> f64@L_per_h {
    // Scaling factor based on GFR ratio
    let gfr_ratio = gfr_patient / gfr_normal;
    
    // Only renal clearance component is affected by CKD
    // Non-renal clearance (hepatic, other) remains unchanged
    let cl_renal = cl_normal * fe_renal;
    let cl_non_renal = cl_normal * (1.0 - fe_renal);
    
    // Scale renal clearance by GFR ratio
    let cl_renal_ckd = cl_renal * gfr_ratio;
    
    // Total clearance in CKD
    let cl_ckd = cl_renal_ckd + cl_non_renal;
    
    return cl_ckd;
}

/// Cockcroft-Gault equation for creatinine clearance estimation
/// CrCL (mL/min) = [(140 - age) × BW] / (72 × SCr) × (0.85 if female)
/// Reference: Cockcroft & Gault (1976)
fn cockcroft_gault(
    age: f64,                // Years
    weight: f64@kg,          // Body weight (kg)
    creatinine: f64,         // Serum creatinine (mg/dL)
    is_male: bool            // Sex
) -> f64@L_per_h {
    let age_factor = 140.0 - age;
    let weight_kg = weight;
    
    // Cockcroft-Gault formula (mL/min)
    let crcl_ml_min = (age_factor * weight_kg) / (72.0 * creatinine);
    
    // Adjust for females (85% of male value)
    let crcl_ml_min_adjusted = if is_male {
        crcl_ml_min
    } else {
        crcl_ml_min * 0.85
    };
    
    // Convert mL/min to L/h
    let crcl_L_per_h = crcl_ml_min_adjusted * 0.06@L_per_h;
    
    return crcl_L_per_h;
}

/// MDRD equation (Modification of Diet in Renal Disease)
/// More accurate for CKD patients than Cockcroft-Gault
/// Reference: Levey et al. (1999)
fn mdrd_egfr(
    age: f64,
    creatinine: f64,        // mg/dL
    is_male: bool,
    is_african_american: bool
) -> f64@L_per_h {
    // MDRD: GFR = 186 × SCr^(-1.154) × age^(-0.203) × (0.742 if female) × (1.212 if AA)
    let base = 186.0 * pow(creatinine, -1.154) * pow(age, -0.203);
    
    let sex_factor = if is_male { 1.0 } else { 0.742 };
    let race_factor = if is_african_american { 1.212 } else { 1.0 };
    
    let egfr_ml_min = base * sex_factor * race_factor;
    
    // Convert to L/h
    return egfr_ml_min * 0.06@L_per_h;
}

/// CKD-EPI equation (2009) - Current gold standard
/// More accurate across GFR range than MDRD
/// Reference: Levey et al. (2009)
fn ckd_epi_egfr(
    age: f64,
    creatinine: f64,        // mg/dL
    is_male: bool,
    is_african_american: bool
) -> f64@L_per_h {
    let kappa = if is_male { 0.9 } else { 0.7 };
    let alpha = if is_male { -0.411 } else { -0.329 };
    let sex_factor = if is_male { 1.0 } else { 1.018 };
    let race_factor = if is_african_american { 1.159 } else { 1.0 };
    
    let cr_ratio = creatinine / kappa;
    let cr_term = if cr_ratio <= 1.0 {
        pow(cr_ratio, alpha)
    } else {
        pow(cr_ratio, -1.209)
    };
    
    let egfr_ml_min = 141.0 * cr_term * pow(0.993, age) * sex_factor * race_factor;
    
    return egfr_ml_min * 0.06@L_per_h;
}

// ============================================================================
// PROXIMAL TUBULE LYSOSOMAL TRAPPING
// ============================================================================

/// Calculate lysosomal trapping in proximal tubule cells
/// Critical for aminoglycosides (gentamicin, tobramycin) nephrotoxicity
/// Reference: Sandoval et al. (2012), Laurent et al. (2005)
fn calculate_pt_lysosomal_trapping(pka: f64, is_base: bool) -> f64 {
    let ph_lysosome = 4.8;   // Acidic lysosomal pH
    let ph_cytosol = 7.2;    // Cytosolic pH
    
    if !is_base {
        // Lysosomal trapping only relevant for bases
        return 1.0;
    }
    
    // Henderson-Hasselbalch for lysosomes
    let ph_diff_lyso = ph_lysosome - pka;
    let ionized_lyso = 1.0 / (1.0 + pow(10.0, ph_diff_lyso));
    
    // Henderson-Hasselbalch for cytosol
    let ph_diff_cyto = ph_cytosol - pka;
    let ionized_cyto = 1.0 / (1.0 + pow(10.0, ph_diff_cyto));
    
    // Trapping factor: ratio of ionized fractions
    // Higher = more drug trapped in lysosomes
    let trapping_factor = ionized_lyso / ionized_cyto;
    
    return trapping_factor;
}

/// Calculate proximal tubule accumulation index
/// High accumulation → nephrotoxicity risk
fn calculate_pt_accumulation(
    cl_uptake: f64@L_per_h,     // OAT/OCT-mediated uptake
    cl_efflux: f64@L_per_h,     // MATE/MRP-mediated efflux
    v_pt: f64@L                  // Proximal tubule cell volume
) -> f64 {
    // Accumulation factor = uptake / efflux
    let accumulation = cl_uptake / cl_efflux;
    
    return accumulation;
}

// ============================================================================
// NEPHROTOXICITY RISK ASSESSMENT
// ============================================================================

/// Calculate kidney accumulation factor
/// High values indicate nephrotoxicity risk
/// Reference: Nolin et al. (2008), Muller et al. (2017)
fn calculate_accumulation_factor(
    cl_renal: f64@L_per_h,
    v_kidney: f64@L
) -> f64 {
    // Accumulation = Kel × Vkidney / CLrenal
    // Higher values = more drug accumulation
    let kel = 0.1;  // Typical kidney elimination rate (can be derived)
    let accumulation = (kel * v_kidney) / cl_renal;
    
    return accumulation;
}

/// Assess nephrotoxicity risk
/// Threshold typically 10-fold for clinical concern
fn is_nephrotoxic_risk(accumulation: f64, threshold: f64) -> bool {
    return accumulation > threshold;
}

/// Calculate tubular necrosis risk score
/// Based on cisplatin/aminoglycoside toxicity models
/// Reference: Pabla & Dong (2008) - Cisplatin nephrotoxicity mechanisms
fn calculate_tubular_necrosis_risk(
    c_kidney: f64@mg_per_L,     // Kidney concentration
    c_toxic: f64@mg_per_L,      // Toxic threshold concentration
    duration: f64@h,            // Exposure duration
    pt_trapping: f64            // Lysosomal trapping factor
) -> f64 {
    // Risk increases with concentration, duration, and trapping
    let concentration_ratio = c_kidney / c_toxic;
    let time_factor = duration / 24.0@h;  // Days of exposure
    
    let risk_score = concentration_ratio * time_factor * pt_trapping;
    
    return risk_score;
}

// ============================================================================
// DRUG-SPECIFIC RENAL CLEARANCE PROFILES
// ============================================================================

/// Metformin renal clearance (OCT2/MATE substrate)
/// Reference: Zamek-Gliszczynski et al. (2013)
fn metformin_renal_clearance(
    gfr: f64@L_per_h,
    oct2_activity: f64,
    mate1_activity: f64,
    fu: f64
) -> f64@L_per_h {
    let cl_filt = calculate_filtration_clearance(gfr, fu);
    
    // Active secretion via OCT2 (basolateral) and MATE1 (apical)
    // Metformin: CL_sec ~450 L/h (very high renal clearance)
    let vmax_oct2 = 1000.0@mg_per_h * oct2_activity;
    let km_oct2 = 1.0@mg_per_L;
    let c_plasma = 0.5@mg_per_L;  // Typical Cmax
    
    let cl_sec = calculate_secretion_clearance(km_oct2, vmax_oct2, c_plasma, 780.0@L_per_h);
    
    // Minimal reabsorption (hydrophilic cation)
    let cl_renal = cl_filt + cl_sec;
    
    return cl_renal;
}

/// Gentamicin renal clearance (lysosomal trapping model)
/// Reference: Mingeot-Leclercq & Tulkens (1999)
fn gentamicin_renal_clearance(
    gfr: f64@L_per_h,
    fu: f64,
    pt_megalin_activity: f64  // Megalin-mediated uptake
) -> f64@L_per_h {
    // Gentamicin: filtered + reabsorbed via megalin → lysosomal trapping
    let cl_filt = calculate_filtration_clearance(gfr, fu);
    
    // Net clearance reduced due to proximal tubule reabsorption
    // ~90% reabsorbed via megalin receptor
    let reabsorption_fraction = 0.9 * pt_megalin_activity;
    
    let cl_renal = cl_filt * (1.0 - reabsorption_fraction);
    
    return cl_renal;
}

/// Tenofovir renal clearance (OAT1/MRP2/MRP4 substrate)
/// Reference: Ray et al. (2006), Kiser et al. (2008)
fn tenofovir_renal_clearance(
    gfr: f64@L_per_h,
    oat1_activity: f64,
    mrp4_activity: f64,
    fu: f64
) -> f64@L_per_h {
    let cl_filt = calculate_filtration_clearance(gfr, fu);
    
    // Active secretion via OAT1 (high affinity)
    let vmax_oat1 = 500.0@mg_per_h * oat1_activity;
    let km_oat1 = 0.1@mg_per_L;
    let c_plasma = 0.3@mg_per_L;
    
    let cl_sec = calculate_secretion_clearance(km_oat1, vmax_oat1, c_plasma, 780.0@L_per_h);
    
    // MRP4-mediated apical efflux (rate-limiting)
    let efflux_factor = 0.7 * mrp4_activity;  // 70% efficiency
    
    let cl_renal = (cl_filt + cl_sec) * efflux_factor;
    
    return cl_renal;
}

// ============================================================================
// URINE pH MANIPULATION EFFECTS
// ============================================================================

/// Calculate effect of urine alkalinization on weak acid excretion
/// Used clinically for salicylate, methotrexate, barbiturate overdose
/// Reference: Proudfoot et al. (2004) - Position paper on urine alkalinization
fn urine_alkalinization_effect(
    pka: f64,
    baseline_pH: f64,
    target_pH: f64,
    fu: f64,
    gfr: f64@L_per_h
) -> f64@L_per_h {
    // Baseline reabsorption at normal pH
    let baseline_ionized = 1.0 / (1.0 + pow(10.0, pka - baseline_pH));
    let baseline_reabsorption = 1.0 - baseline_ionized;
    
    // Reabsorption after alkalinization
    let alkalinized_ionized = 1.0 / (1.0 + pow(10.0, pka - target_pH));
    let alkalinized_reabsorption = 1.0 - alkalinized_ionized;
    
    // Clearance change
    let cl_filt = calculate_filtration_clearance(gfr, fu);
    let cl_baseline = cl_filt * (1.0 - baseline_reabsorption);
    let cl_alkalinized = cl_filt * (1.0 - alkalinized_reabsorption);
    
    return cl_alkalinized;
}

/// Calculate effect of urine acidification on weak base excretion
/// Used for amphetamine, phencyclidine toxicity
fn urine_acidification_effect(
    pka: f64,
    baseline_pH: f64,
    target_pH: f64,
    fu: f64,
    gfr: f64@L_per_h
) -> f64@L_per_h {
    // For bases: ionization = 1 / (1 + 10^(pH - pKa))
    let baseline_ionized = 1.0 / (1.0 + pow(10.0, baseline_pH - pka));
    let baseline_reabsorption = 1.0 - baseline_ionized;
    
    let acidified_ionized = 1.0 / (1.0 + pow(10.0, target_pH - pka));
    let acidified_reabsorption = 1.0 - acidified_ionized;
    
    let cl_filt = calculate_filtration_clearance(gfr, fu);
    let cl_acidified = cl_filt * (1.0 - acidified_reabsorption);
    
    return cl_acidified;
}

// ============================================================================
// RENAL REPLACEMENT THERAPY (DIALYSIS)
// ============================================================================

/// Calculate hemodialysis clearance
/// Reference: Heintz et al. (2009) - Drug dosing in CRRT
fn hemodialysis_clearance(
    qb: f64@L_per_h,           // Blood flow rate (typically 12-18 L/h)
    qd: f64@L_per_h,           // Dialysate flow rate (typically 30 L/h)
    koa: f64@L_per_h,          // Mass transfer coefficient
    fu: f64,                    // Fraction unbound
    molecular_weight: f64       // Daltons
) -> f64@L_per_h {
    // Only unbound drug is dialyzable
    // Small molecules (<500 Da) dialyze efficiently
    let size_factor = if molecular_weight < 500.0 {
        1.0
    } else if molecular_weight < 1000.0 {
        0.5
    } else {
        0.1  // Large molecules poorly dialyzed
    };
    
    // Dialyzer clearance (two-resistance model)
    let cl_d = (qb * qd * koa) / (qb * qd + qb * koa + qd * koa);
    
    let cl_hd = cl_d * fu * size_factor;
    
    return cl_hd;
}

/// Calculate peritoneal dialysis clearance
/// Reference: Daugirdas et al. (2007) - Handbook of Dialysis
fn peritoneal_dialysis_clearance(
    dialysate_volume: f64@L,    // Typical 2 L per exchange
    dwell_time: f64@h,          // 4-6 hours
    exchanges_per_day: f64,     // 4-5 exchanges
    fu: f64,
    peritoneal_permeability: f64  // Drug-specific (0.1-0.8)
) -> f64@L_per_h {
    // Clearance per exchange
    let cl_per_exchange = (dialysate_volume * peritoneal_permeability * fu) / dwell_time;
    
    // Total daily clearance
    let cl_pd = cl_per_exchange * exchanges_per_day / 24.0;
    
    return cl_pd;
}

// ============================================================================
// EXPORT MODULE INTERFACE
// ============================================================================

export {
    KidneyPhysiology,
    RenalTransporters,
    calculate_filtration_clearance,
    calculate_secretion_clearance,
    calculate_reabsorption,
    calculate_total_renal_clearance,
    scale_gfr_for_age,
    scale_clearance_for_ckd,
    cockcroft_gault,
    mdrd_egfr,
    ckd_epi_egfr,
    calculate_pt_lysosomal_trapping,
    calculate_pt_accumulation,
    calculate_accumulation_factor,
    is_nephrotoxic_risk,
    calculate_tubular_necrosis_risk,
    metformin_renal_clearance,
    gentamicin_renal_clearance,
    tenofovir_renal_clearance,
    urine_alkalinization_effect,
    urine_acidification_effect,
    hemodialysis_clearance,
    peritoneal_dialysis_clearance,
};

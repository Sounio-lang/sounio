// liver.d - Advanced Hepatic Compartment Module for Darwin PBPK
// Comprehensive liver physiology, transporter-enzyme kinetics, and clearance models
// Version: 1.0.0
// Date: 2025-12-08

module darwin_pbpk.compartments.liver;

use std.math;

// =============================================================================
// 1. LIVER PHYSIOLOGY
// =============================================================================

/// Comprehensive liver physiological parameters
/// Reference: Davies & Morris 1993, Yang et al. 2007
struct LiverPhysiology {
    /// Total liver volume (typical adult: 1.8 L)
    volume: f64@L,
    
    /// Total hepatic blood flow (25% of cardiac output)
    /// Typical: 1500 L/h = 25 mL/min/kg for 70 kg adult
    blood_flow_total: f64@L_per_h,
    
    /// Portal vein contribution to total flow
    /// Typical: 75% of total hepatic blood flow
    portal_fraction: f64,
    
    /// Hepatic artery contribution to total flow
    /// Typical: 25% of total hepatic blood flow
    arterial_fraction: f64,
    
    /// Hepatocellularity (cells per gram of liver tissue)
    /// Typical: 120 × 10^6 cells/g
    hepatocellularity: f64,
    
    /// Total liver weight (typical adult: 1.8 kg)
    liver_weight: f64@kg,
    
    /// Microsomal protein per gram liver (mg/g)
    /// Typical: 45 mg microsomal protein/g liver
    microsomal_protein_per_g: f64,
    
    /// Cytosolic protein per gram liver (mg/g)
    /// Typical: 108 mg cytosolic protein/g liver
    cytosolic_protein_per_g: f64
}

/// Create reference adult liver physiology
fn create_reference_liver() -> LiverPhysiology {
    LiverPhysiology {
        volume: 1.8@L,
        blood_flow_total: 1500.0@L_per_h,
        portal_fraction: 0.75,
        arterial_fraction: 0.25,
        hepatocellularity: 120e6,
        liver_weight: 1.8@kg,
        microsomal_protein_per_g: 45.0,
        cytosolic_protein_per_g: 108.0
    }
}

/// Scale liver physiology by body weight using allometric scaling
/// Reference: West et al. 1997 (allometric exponent 0.75 for blood flow)
fn scale_liver_by_weight(reference: LiverPhysiology, weight_kg: f64) -> LiverPhysiology {
    let scale_volume = weight_kg / 70.0;
    let scale_flow = pow(weight_kg / 70.0, 0.75);
    
    LiverPhysiology {
        volume: reference.volume * scale_volume,
        blood_flow_total: reference.blood_flow_total * scale_flow,
        portal_fraction: reference.portal_fraction,
        arterial_fraction: reference.arterial_fraction,
        hepatocellularity: reference.hepatocellularity,
        liver_weight: reference.liver_weight * scale_volume,
        microsomal_protein_per_g: reference.microsomal_protein_per_g,
        cytosolic_protein_per_g: reference.cytosolic_protein_per_g
    }
}

// =============================================================================
// 2. LIVER COMPOSITION (for Kp calculation)
// =============================================================================

/// Liver tissue composition for partition coefficient prediction
/// Reference: Rodgers & Rowland 2006, Poulin & Theil 2002
struct LiverComposition {
    /// Fraction water content
    f_water: f64,
    
    /// Fraction neutral lipids (triglycerides + cholesterol esters)
    f_neutral_lipid: f64,
    
    /// Fraction phospholipids (membrane components)
    f_phospholipid: f64,
    
    /// Fraction acidic phospholipids (important for cationic drugs)
    f_acidic_phospholipid: f64,
    
    /// Lysosomal fraction (critical for basic drug trapping)
    /// Liver has significant lysosomal content
    lysosomal_fraction: f64,
    
    /// Interstitial fluid fraction
    f_interstitial: f64,
    
    /// Intracellular fluid fraction
    f_intracellular: f64
}

/// Create reference liver tissue composition
fn create_reference_liver_composition() -> LiverComposition {
    LiverComposition {
        f_water: 0.76,
        f_neutral_lipid: 0.035,
        f_phospholipid: 0.025,
        f_acidic_phospholipid: 0.005,
        lysosomal_fraction: 0.025,
        f_interstitial: 0.15,
        f_intracellular: 0.61
    }
}

// =============================================================================
// 3. HEPATIC TRANSPORTERS
// =============================================================================

/// Hepatic transporter abundances
/// Reference: Prasad & Unadkat 2014, Ohtsuki et al. 2012
struct HepaticTransporters {
    /// OATP1B1 (SLCO1B1) - Major organic anion uptake
    /// Substrates: statins, rifampin, bosentan
    /// Typical: 8.5 pmol/mg membrane protein
    oatp1b1_abundance: f64,
    
    /// OATP1B3 (SLCO1B3) - Organic anion uptake
    /// Substrates: statins, methotrexate, telmisartan
    /// Typical: 2.1 pmol/mg membrane protein
    oatp1b3_abundance: f64,
    
    /// OCT1 (SLC22A1) - Organic cation uptake
    /// Substrates: metformin, lamivudine, imatinib
    /// Typical: 4.5 pmol/mg membrane protein
    oct1_abundance: f64,
    
    /// NTCP (SLC10A1) - Bile acid uptake
    /// Typical: 3.8 pmol/mg membrane protein
    ntcp_abundance: f64,
    
    /// P-glycoprotein (MDR1/ABCB1) - Biliary excretion
    /// Substrates: digoxin, verapamil, cyclosporine
    /// Typical: 1.5 pmol/mg membrane protein
    pgp_abundance: f64,
    
    /// BCRP (ABCG2) - Biliary/basolateral efflux
    /// Substrates: rosuvastatin, sulfasalazine, nitrofurantoin
    /// Typical: 2.8 pmol/mg membrane protein
    bcrp_abundance: f64,
    
    /// MRP2 (ABCC2) - Canalicular organic anion efflux
    /// Substrates: methotrexate, pravastatin, indinavir
    /// Typical: 3.2 pmol/mg membrane protein
    mrp2_abundance: f64,
    
    /// BSEP (ABCB11) - Bile salt export pump
    /// Typical: 5.1 pmol/mg membrane protein
    bsep_abundance: f64,
    
    /// MRP3 (ABCC3) - Basolateral efflux (backup pathway)
    /// Typical: 1.2 pmol/mg membrane protein
    mrp3_abundance: f64
}

/// Create reference hepatic transporter abundances
fn create_reference_transporters() -> HepaticTransporters {
    HepaticTransporters {
        oatp1b1_abundance: 8.5,
        oatp1b3_abundance: 2.1,
        oct1_abundance: 4.5,
        ntcp_abundance: 3.8,
        pgp_abundance: 1.5,
        bcrp_abundance: 2.8,
        mrp2_abundance: 3.2,
        bsep_abundance: 5.1,
        mrp3_abundance: 1.2
    }
}

// =============================================================================
// 4. CYP ENZYMES
// =============================================================================

/// Cytochrome P450 enzyme abundances
/// Reference: Achour et al. 2014, Rodrigues 1999
struct CYPEnzymes {
    /// CYP3A4 - Most abundant CYP (30%% of total hepatic CYP)
    /// Substrates: midazolam, simvastatin, cyclosporine, >50%% of drugs
    /// Typical: 108 pmol/mg microsomal protein
    cyp3a4_abundance: f64,
    
    /// CYP2D6 - High variability due to polymorphisms
    /// Substrates: codeine, dextromethorphan, metoprolol
    /// Typical: 10 pmol/mg microsomal protein (extensive metabolizers)
    cyp2d6_abundance: f64,
    
    /// CYP2C9 - Warfarin, NSAIDs metabolism
    /// Substrates: warfarin, phenytoin, tolbutamide
    /// Typical: 96 pmol/mg microsomal protein
    cyp2c9_abundance: f64,
    
    /// CYP2C19 - Proton pump inhibitor metabolism
    /// Substrates: omeprazole, clopidogrel, diazepam
    /// Typical: 19 pmol/mg microsomal protein
    cyp2c19_abundance: f64,
    
    /// CYP1A2 - Caffeine metabolism
    /// Substrates: caffeine, theophylline, clozapine
    /// Typical: 52 pmol/mg microsomal protein
    cyp1a2_abundance: f64,
    
    /// CYP2E1 - Ethanol, acetaminophen metabolism
    /// Typical: 49 pmol/mg microsomal protein
    cyp2e1_abundance: f64,
    
    /// CYP2C8 - Paclitaxel, amodiaquine metabolism
    /// Typical: 64 pmol/mg microsomal protein
    cyp2c8_abundance: f64,
    
    /// CYP2B6 - Efavirenz, bupropion metabolism
    /// Typical: 39 pmol/mg microsomal protein
    cyp2b6_abundance: f64
}

/// Create reference CYP enzyme abundances (extensive metabolizer phenotype)
fn create_reference_cyp_enzymes() -> CYPEnzymes {
    CYPEnzymes {
        cyp3a4_abundance: 108.0,
        cyp2d6_abundance: 10.0,
        cyp2c9_abundance: 96.0,
        cyp2c19_abundance: 19.0,
        cyp1a2_abundance: 52.0,
        cyp2e1_abundance: 49.0,
        cyp2c8_abundance: 64.0,
        cyp2b6_abundance: 39.0
    }
}

// =============================================================================
// 5. HEPATIC CLEARANCE MODELS
// =============================================================================

/// Well-stirred model (venous equilibration model)
/// Assumes instantaneous mixing in liver
/// Reference: Rowland & Tozer 1980
/// 
/// CL_h = Q_h × f_u × CL_int / (Q_h + f_u × CL_int)
fn well_stirred_model(cl_int: f64@L_per_h, q_liver: f64@L_per_h, fu: f64) -> f64@L_per_h {
    let numerator = q_liver * fu * cl_int;
    let denominator = q_liver + fu * cl_int;
    numerator / denominator
}

/// Parallel tube model (sinusoidal perfusion model)
/// Assumes plug flow through liver sinusoids
/// More accurate for high extraction ratio drugs
/// Reference: Winkler et al. 1973
/// 
/// CL_h = Q_h × (1 - exp(-f_u × CL_int / Q_h))
fn parallel_tube_model(cl_int: f64@L_per_h, q_liver: f64@L_per_h, fu: f64) -> f64@L_per_h {
    q_liver * (1.0 - exp(-fu * cl_int / q_liver))
}

/// Dispersion model (accounts for axial dispersion)
/// Most physiologically accurate model
/// Reference: Roberts & Rowland 1986
/// 
/// Args:
///   cl_int: Intrinsic clearance
///   q_liver: Hepatic blood flow
///   fu: Fraction unbound in blood
///   dn: Dispersion number (0.17 typical for liver)
fn dispersion_model(cl_int: f64@L_per_h, q_liver: f64@L_per_h, fu: f64, dn: f64) -> f64@L_per_h {
    let rn = fu * cl_int / q_liver;
    let a = sqrt(1.0 + 4.0 * rn * dn);
    let b = (1.0 + a) / (2.0 * dn);
    let c = (1.0 - a) / (2.0 * dn);
    let e_h = (4.0 * a) / ((1.0 + a) * (1.0 + a) * exp(b) - (1.0 - a) * (1.0 - a) * exp(c));
    q_liver * e_h
}

// =============================================================================
// 6. LYSOSOMAL TRAPPING
// =============================================================================

/// Calculate lysosomal trapping ratio for basic drugs
/// Lysosomes are acidic (pH ~4.8) and can trap weak bases
/// Reference: Trapp et al. 2008, Schmitt 2008
/// 
/// For bases: K_lyso = [drug]_lyso / [drug]_cyto
/// K_lyso = (1 + 10^(pH_lyso - pKa)) / (1 + 10^(pH_cyto - pKa))
fn calculate_lysosomal_trapping(pH_lysosome: f64, pH_cytosol: f64, pka: f64, is_base: bool) -> f64 {
    if is_base {
        let numerator = 1.0 + pow(10.0, pH_lysosome - pka);
        let denominator = 1.0 + pow(10.0, pH_cytosol - pka);
        numerator / denominator
    } else {
        let numerator = 1.0 + pow(10.0, pka - pH_lysosome);
        let denominator = 1.0 + pow(10.0, pka - pH_cytosol);
        numerator / denominator
    }
}

/// Adjust liver partition coefficient for lysosomal trapping
/// K_p_liver_adj = K_p_liver + f_lyso × (K_lyso - 1) × K_p_liver
fn adjust_kp_for_lysosomes(kp_liver: f64, lysosomal_fraction: f64, k_lysosomal: f64) -> f64 {
    kp_liver * (1.0 + lysosomal_fraction * (k_lysosomal - 1.0))
}

/// Full lysosomal trapping calculation for liver
fn liver_lysosomal_correction(kp_liver: f64, pka: f64, is_base: bool, composition: LiverComposition) -> f64 {
    let ph_lyso = 4.8;
    let ph_cyto = 7.0;
    let k_lyso = calculate_lysosomal_trapping(ph_lyso, ph_cyto, pka, is_base);
    adjust_kp_for_lysosomes(kp_liver, composition.lysosomal_fraction, k_lyso)
}

// =============================================================================
// 7. HEPATIC ZONATION
// =============================================================================

/// Hepatic zonation structure
/// Zone 1 (periportal): High O2, oxidative metabolism, lower CYP3A4
/// Zone 3 (centrilobular): Low O2, reductive metabolism, higher CYP3A4
/// Reference: Jungermann & Kietzmann 2000
struct HepaticZonation {
    zone1_fraction: f64,
    zone2_fraction: f64,
    zone3_fraction: f64,
    cyp3a4_gradient: f64,
    o2_gradient: f64
}

/// Create reference hepatic zonation
fn create_reference_zonation() -> HepaticZonation {
    HepaticZonation {
        zone1_fraction: 0.35,
        zone2_fraction: 0.35,
        zone3_fraction: 0.30,
        cyp3a4_gradient: 2.5,
        o2_gradient: 2.0
    }
}

/// Calculate zonal clearance accounting for enzyme distribution
fn calculate_zonal_clearance(
    cl_int_zone1: f64@L_per_h,
    cl_int_zone3: f64@L_per_h,
    zonation: HepaticZonation
) -> f64@L_per_h {
    let cl_int_zone2 = (cl_int_zone1 + cl_int_zone3) / 2.0;
    let cl_int_total = 
        cl_int_zone1 * zonation.zone1_fraction +
        cl_int_zone2 * zonation.zone2_fraction +
        cl_int_zone3 * zonation.zone3_fraction;
    cl_int_total
}

// =============================================================================
// 8. TRANSPORTER-MEDIATED UPTAKE
// =============================================================================

/// Michaelis-Menten kinetics for transporter-mediated uptake
/// v = V_max × C / (K_m + C)
fn calculate_active_uptake(km: f64@mg_per_L, vmax: f64, c_unbound: f64@mg_per_L) -> f64 {
    (vmax * c_unbound) / (km + c_unbound)
}

/// OATP-mediated hepatic uptake clearance
/// Reference: Varma et al. 2012 (Extended Clearance Model)
fn calculate_oatp_mediated_clearance(
    oatp_km: f64@mg_per_L,
    oatp_vmax: f64,
    fu: f64,
    liver_weight: f64@kg,
    microsomal_protein: f64
) -> f64@L_per_h {
    let vmax_liver = oatp_vmax * microsomal_protein * (liver_weight * 1000.0) * 60.0 * 1e-12;
    let cl_uptake = vmax_liver / oatp_km;
    cl_uptake * fu
}

// =============================================================================
// 9. EXTRACTION RATIO
// =============================================================================

/// Calculate hepatic extraction ratio
/// E = CL_h / Q_h
/// 
/// Interpretation:
///   E < 0.3: Low extraction (capacity-limited clearance)
///   0.3 < E < 0.7: Intermediate extraction
///   E > 0.7: High extraction (flow-limited clearance)
fn calculate_extraction_ratio(cl_hepatic: f64@L_per_h, q_liver: f64@L_per_h) -> f64 {
    cl_hepatic / q_liver
}

// =============================================================================
// 10. FIRST-PASS EFFECT
// =============================================================================

/// Calculate hepatic bioavailability (fraction escaping first-pass metabolism)
/// F_H = 1 - E
/// 
/// For oral administration: F = F_a × F_g × F_H
fn calculate_fh(extraction_ratio: f64) -> f64 {
    1.0 - extraction_ratio
}

/// Calculate overall oral bioavailability
fn calculate_oral_bioavailability(f_absorbed: f64, f_gut: f64, extraction_ratio: f64) -> f64 {
    let f_hepatic = calculate_fh(extraction_ratio);
    f_absorbed * f_gut * f_hepatic
}

// =============================================================================
// 11. INTRINSIC CLEARANCE FROM IN VITRO DATA
// =============================================================================

/// Scale intrinsic clearance from microsomal data to whole liver
/// Reference: Obach et al. 1997
/// 
/// CL_int,liver = CL_int,mic × mg microsomal protein/g liver × g liver
fn scale_microsomal_to_liver(
    cl_int_mic: f64,
    microsomal_protein: f64,
    liver_weight: f64@kg
) -> f64@L_per_h {
    let cl_int_liver = cl_int_mic * microsomal_protein * 1000.0 * 60.0 / 1e6;
    cl_int_liver * liver_weight
}

/// Scale intrinsic clearance from hepatocyte data
/// CL_int,liver = CL_int,hep × cells/g liver × g liver
fn scale_hepatocyte_to_liver(
    cl_int_hep: f64,
    hepatocellularity: f64,
    liver_weight: f64@kg
) -> f64@L_per_h {
    let cl_int_liver = cl_int_hep * hepatocellularity * 1000.0 * 60.0 / 1e6;
    cl_int_liver * liver_weight
}

// =============================================================================
// EXPORT PUBLIC API
// =============================================================================

pub use {
    LiverPhysiology,
    create_reference_liver,
    scale_liver_by_weight,
    LiverComposition,
    create_reference_liver_composition,
    HepaticTransporters,
    create_reference_transporters,
    CYPEnzymes,
    create_reference_cyp_enzymes,
    well_stirred_model,
    parallel_tube_model,
    dispersion_model,
    calculate_lysosomal_trapping,
    liver_lysosomal_correction,
    HepaticZonation,
    calculate_zonal_clearance,
    calculate_active_uptake,
    calculate_oatp_mediated_clearance,
    calculate_extraction_ratio,
    calculate_fh,
    calculate_oral_bioavailability,
    scale_microsomal_to_liver,
    scale_hepatocyte_to_liver
};

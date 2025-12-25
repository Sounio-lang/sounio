# Advanced Liver Compartment Module - Demetrios

**Location**: 
**Size**: 17KB (480 lines)
**Version**: 1.0.0
**Date**: 2025-12-08

## Overview

Comprehensive hepatic compartment module for Darwin PBPK Platform implementing advanced liver physiology, transporter-enzyme kinetics, and clearance models.

## Module Structure

### 1. LiverPhysiology (Lines 16-47)
**Struct** with 8 fields:
-  - Total liver volume (1.8 L typical)
-  - 1500 L/h (25% cardiac output)
-  - 0.75 (portal vein contribution)
-  - 0.25 (hepatic artery)
-  - 120×10^6 cells/g
-  - 1.8 kg
-  - 45 mg/g
-  - 108 mg/g

**Functions**:
-  - Creates adult reference liver
-  - Allometric scaling (exponent 0.75)

### 2. LiverComposition (Lines 87-121)
**Struct** for Kp calculation (Rodgers & Rowland 2006):
- 
- 
- 
- 
-  - Critical for basic drug trapping!
- 
- 

**Function**:
- 

### 3. HepaticTransporters (Lines 127-187)
**Struct** with 9 transporter abundances (pmol/mg protein):

**Uptake transporters (basolateral)**:
-  - Statins, rifampin (SLCO1B1)
-  - Statins, methotrexate (SLCO1B3)
-  - Metformin, imatinib (SLC22A1)
-  - Bile acids (SLC10A1)

**Efflux transporters (canalicular)**:
-  - Digoxin, verapamil (MDR1/ABCB1)
-  - Rosuvastatin (ABCG2)
-  - Methotrexate (ABCC2)
-  - Bile salts (ABCB11)
-  - Backup pathway (ABCC3)

**Function**:
- 

### 4. CYPEnzymes (Lines 193-245)
**Struct** with 8 CYP isoform abundances (pmol/mg microsomal protein):
-  - Most abundant (30% total CYP, >50% drugs)
-  - Codeine, dextromethorphan (EM phenotype)
-  - Warfarin, NSAIDs
-  - Omeprazole, clopidogrel
-  - Caffeine, theophylline
-  - Ethanol, acetaminophen
-  - Paclitaxel
-  - Efavirenz, bupropion

**Function**:
- 

### 5. Hepatic Clearance Models (Lines 251-288)

#### well_stirred_model()

- Assumes instantaneous mixing
- Good for low extraction drugs (E < 0.3)
- Reference: Rowland & Tozer 1980

#### parallel_tube_model()

- Assumes plug flow through sinusoids
- Better for high extraction drugs (E > 0.7)
- Reference: Winkler et al. 1973

#### dispersion_model()

- Most physiologically accurate
- Uses dispersion number
- Reference: Roberts & Rowland 1986

### 6. Lysosomal Trapping (Lines 294-335)

**Critical for basic drugs** (amiodarone, chloroquine, etc.)

#### calculate_lysosomal_trapping()


#### liver_lysosomal_correction()
Full correction for liver Kp including 2.5% lysosomal fraction

**Reference**: Trapp et al. 2008, Schmitt 2008

### 7. Hepatic Zonation (Lines 341-365)

**Struct HepaticZonation**:
-  - Periportal (high O2, oxidative)
-  - Intermediate
-  - Centrilobular (low O2, high CYP3A4)
-  - Zone 3 has 2.5× more CYP3A4
-  - Zone 1 has 2× more O2

#### calculate_zonal_clearance()
Weighted average clearance across zones

**Reference**: Jungermann & Kietzmann 2000

### 8. Transporter-Mediated Uptake (Lines 371-390)

#### calculate_active_uptake()

Michaelis-Menten kinetics

#### calculate_oatp_mediated_clearance()
Extended clearance model for OATP transporters
- Scales V_max to whole liver
- Accounts for protein binding

**Reference**: Varma et al. 2012

### 9. Extraction Ratio (Lines 396-408)

#### calculate_extraction_ratio()


**Interpretation**:
- E < 0.3: Low extraction (capacity-limited)
- 0.3 < E < 0.7: Intermediate
- E > 0.7: High extraction (flow-limited)

### 10. First-Pass Effect (Lines 414-429)

#### calculate_fh()

Hepatic bioavailability

#### calculate_oral_bioavailability()

Overall oral bioavailability:
- F_a = fraction absorbed
- F_g = fraction escaping gut metabolism
- F_H = fraction escaping hepatic first-pass

### 11. In Vitro → In Vivo Scaling (Lines 435-462)

#### scale_microsomal_to_liver()

Scales from microsomal data (μL/min/mg protein)

#### scale_hepatocyte_to_liver()

Scales from hepatocyte data (μL/min/million cells)

**Reference**: Obach et al. 1997

## Exported API (Lines 468-480)

All 25 public items:

**Structs**:
- LiverPhysiology
- LiverComposition
- HepaticTransporters
- CYPEnzymes
- HepaticZonation

**Creation Functions**:
- create_reference_liver()
- create_reference_liver_composition()
- create_reference_transporters()
- create_reference_cyp_enzymes()

**Clearance Models**:
- well_stirred_model()
- parallel_tube_model()
- dispersion_model()

**Lysosomal Trapping**:
- calculate_lysosomal_trapping()
- liver_lysosomal_correction()

**Zonation**:
- calculate_zonal_clearance()

**Transporters**:
- calculate_active_uptake()
- calculate_oatp_mediated_clearance()

**Extraction & Bioavailability**:
- calculate_extraction_ratio()
- calculate_fh()
- calculate_oral_bioavailability()

**IVIVE Scaling**:
- scale_microsomal_to_liver()
- scale_hepatocyte_to_liver()

**Allometric Scaling**:
- scale_liver_by_weight()

## Key Features

1. **Comprehensive physiology**: Portal + arterial blood flow, hepatocellularity, protein content
2. **Transporter abundances**: 9 major hepatic transporters (OATP, OCT, P-gp, BCRP, MRP, BSEP)
3. **CYP enzyme panel**: 8 major CYP isoforms with literature-based abundances
4. **Multiple clearance models**: Well-stirred, parallel tube, dispersion
5. **Lysosomal trapping**: Critical for basic drugs (pH 4.8 vs 7.0)
6. **Hepatic zonation**: Accounts for periportal-centrilobular gradient
7. **OATP kinetics**: Michaelis-Menten with extended clearance model
8. **Extraction ratio classification**: Low/intermediate/high
9. **First-pass metabolism**: Full F = F_a × F_g × F_H model
10. **IVIVE scaling**: Microsomal and hepatocyte data scaling

## Literature References

- Davies & Morris 1993 (liver physiology)
- Yang et al. 2007 (organ volumes)
- West et al. 1997 (allometric scaling)
- Rodgers & Rowland 2006 (tissue composition, Kp)
- Poulin & Theil 2002 (partition coefficients)
- Prasad & Unadkat 2014 (transporter abundances)
- Ohtsuki et al. 2012 (transporter proteomics)
- Achour et al. 2014 (CYP abundances)
- Rodrigues 1999 (CYP kinetics)
- Rowland & Tozer 1980 (well-stirred model)
- Winkler et al. 1973 (parallel tube model)
- Roberts & Rowland 1986 (dispersion model)
- Trapp et al. 2008 (lysosomal trapping)
- Schmitt 2008 (tissue distribution)
- Jungermann & Kietzmann 2000 (hepatic zonation)
- Varma et al. 2012 (extended clearance model)
- Obach et al. 1997 (IVIVE scaling)

## Usage Examples



## Integration with Darwin PBPK

This module is designed to integrate with:
- 
- Julia implementation: 

## Status

✅ Complete implementation of all 10 requested features
✅ 25 exported functions/structs
✅ Literature-based default values
✅ Dimensional analysis with units (@L, @L_per_h, @kg, @mg_per_L)
✅ Comprehensive documentation

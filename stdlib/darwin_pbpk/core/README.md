# Darwin PBPK Core Module - Demetrios Implementation

This module provides the core PBPK (Physiologically-Based Pharmacokinetic) data structures and functions for the Darwin PBPK Platform, implemented in the Demetrios language.

## File: pbpk_params.d

### Overview
A comprehensive PBPK parameter module with unit-annotated types for 14-organ physiological modeling.

### Contents

#### 1. Unit Type Aliases
-  - Volume (liters)
-  - Time (hours)
-  - Mass (milligrams)
-  - Body mass (kilograms)
-  - Clearance, blood flow (L/h)
-  - Concentration (mg/L)
-  - AUC (mg·h/L)

#### 2. Constants
- 
- Organ indices: BLOOD, LIVER, KIDNEY, BRAIN, HEART, LUNG, MUSCLE, ADIPOSE, GUT, SKIN, BONE, SPLEEN, PANCREAS, REST

#### 3. Data Structures

##### PBPKParams
Main PBPK parameter container with:
- 14 organ volumes (v_blood, v_liver, etc.) - 
- 14 blood flow rates (q_blood, q_liver, etc.) - 
- Hepatic and renal clearance - 
- 14 partition coefficients (kp_blood, kp_liver, etc.) -  (dimensionless)
- Physiological parameters: fu_plasma, hematocrit, bp_ratio

##### PatientData
Patient demographics and clinical state:
- age (years)
- weight ()
- height (cm)
- sex (bool: true=male, false=female)
- disease_state (i32 enum: HEALTHY, RENAL_IMPAIRMENT, HEPATIC_IMPAIRMENT, CARDIAC_FAILURE, DIABETES)

##### DrugProperties
Drug physicochemical properties:
- mw (molecular weight, g/mol)
- logp (lipophilicity)
- pka (acid dissociation constant)
- fu (fraction unbound, 0-1)
- bp_ratio (blood:plasma ratio)
- is_base (bool)

#### 4. Helper Functions

##### 
Creates default PBPK parameters for a 70kg reference adult male based on ICRP data.

Default values include:
- Organ volumes from ICRP reference adult male
- Blood flows based on cardiac output ~6.5 L/min (390 L/h)
- Typical clearance values (hepatic: 10 L/h, renal: 5 L/h)
- Generic partition coefficients for neutral lipophilic drug
- Standard physiological parameters (fu_plasma: 0.1, hematocrit: 0.45, bp_ratio: 1.0)

##### 
Creates a patient profile with basic demographics. Height is estimated based on sex.

##### 
Validates PBPK parameters for physiological plausibility:
- All volumes and blood flows are positive
- Clearances are non-negative
- Partition coefficients are positive
- fu_plasma and hematocrit are in [0,1]
- bp_ratio is positive

##### 
Calculates total body volume by summing all organ volumes.

##### 
Calculates cardiac output by summing all organ blood flows (excluding lung which is in series).

## Usage Example



## Integration
This module is designed to integrate with:
- Demetrios ODE solver for PBPK simulations
- MedLang DSL for pharmacometric modeling
- Darwin PBPK Platform's Julia implementation

## File Statistics
- **Lines:** 301
- **Structs:** 3 (PBPKParams, PatientData, DrugProperties)
- **Functions:** 5 helper functions
- **Constants:** 19 (organ indices + disease states)

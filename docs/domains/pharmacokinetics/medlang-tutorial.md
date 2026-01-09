# MedLang Tutorial

MedLang is Sounio's domain-specific language for pharmacometric model specification. It provides a declarative syntax for defining compartmental models, dosing regimens, observation models, and population structures.

## Overview

MedLang extends Sounio with pharmacometric primitives:

- **Models**: Compartmental structures with parameters and flows
- **Parameters**: Fixed and random effects with priors
- **Compartments**: State containers with volumes
- **Flows**: Mass transport between compartments
- **Observations**: Measurement models with error structures
- **Protocols**: Dosing regimens and study designs

## Basic Model Syntax

### Defining a Model

A MedLang model is defined using the `model` keyword:

```sio
model ModelName {
    // Parameters
    param param_name ~ Distribution(value, omega: variability)

    // Compartments
    compartment CompartmentName(volume: V)

    // Flows
    flow FlowName = rate_expression -> TargetCompartment

    // Observations
    observe ObsName = expression with ErrorModel(sigma)
}
```

### Parameters

Parameters can be fixed or random (with inter-individual variability):

```sio
// Fixed parameter
param F = 1.0  // Bioavailability (unitless)

// Random parameter with log-normal distribution
param CL ~ LogNormal(10.0 L/h, omega: 0.3)  // Clearance

// Random parameter with normal distribution
param EMAX ~ Normal(100.0, omega: 0.2)

// Parameter with bounds
param KA ~ LogNormal(1.5 1/h, omega: 0.4) where KA > 0.0

// Parameter with covariate effect
param CL ~ LogNormal(
    theta1 * (WT/70)^theta2,  // Typical value with allometric scaling
    omega: 0.3
)
```

### Compartments

Compartments represent pharmacokinetic spaces:

```sio
// Compartment with volume parameter
compartment Central(volume: V1)

// Compartment with explicit volume
compartment Peripheral(volume: V2)

// Depot compartment (no volume - amount only)
compartment Depot()

// Effect compartment
compartment Effect(volume: VE, equilibration: KEO)
```

### Flows

Flows define mass transport between compartments:

```sio
// First-order elimination
flow Elimination = CL/V * Central.amount -> External

// First-order absorption
flow Absorption = KA * Depot.amount -> Central

// Distribution to peripheral
flow Distribution12 = Q/V1 * Central.amount -> Peripheral
flow Distribution21 = Q/V2 * Peripheral.amount -> Central

// Zero-order input (infusion)
flow Infusion = RATE -> Central where t >= TINF_START && t < TINF_END

// Saturable elimination (Michaelis-Menten)
flow Elimination = VMAX * Central.C / (KM + Central.C) -> External
```

### Observations

Observation models connect predictions to data:

```sio
// Concentration observation with proportional error
observe CP = Central.C with Proportional(sigma: 0.1)

// With additive error
observe PD_RESPONSE = EMAX * Central.C / (EC50 + Central.C) with Additive(sigma: 5.0)

// With combined error
observe CONC = Central.C with Combined(sigma_add: 0.1, sigma_prop: 0.1)

// Log-transformed observation
observe LOGCP = log(Central.C) with Additive(sigma: 0.15)
```

## Complete Examples

### One-Compartment IV Model

```sio
use medlang::*

/// One-compartment model for IV bolus administration
model OneCompartmentIV {
    // Population parameters
    param CL ~ LogNormal(10.0 L/h, omega: 0.3)   // Clearance
    param V ~ LogNormal(70.0 L, omega: 0.25)     // Volume of distribution

    // Covariates
    covariate WT: kg    // Body weight
    covariate AGE: years

    // Covariate effects (allometric scaling)
    let CL_i = CL * (WT/70.0)^0.75
    let V_i = V * (WT/70.0)^1.0

    // Compartment
    compartment Central(volume: V_i)

    // Elimination flow
    flow Elimination = CL_i/V_i * Central.amount -> External

    // Observation
    observe CP = Central.C with Proportional(sigma: 0.1)

    // Derived parameters for output
    derive KE = CL_i / V_i               // Elimination rate constant
    derive THALF = 0.693 / KE            // Half-life
    derive AUC = Dose / CL_i             // Area under curve
}
```

### Two-Compartment Oral Model

```sio
use medlang::*

/// Two-compartment model with first-order oral absorption
model TwoCompartmentOral {
    // Structural parameters
    param KA ~ LogNormal(1.5 1/h, omega: 0.4)    // Absorption rate
    param CL ~ LogNormal(15.0 L/h, omega: 0.3)   // Central clearance
    param V1 ~ LogNormal(50.0 L, omega: 0.25)    // Central volume
    param V2 ~ LogNormal(100.0 L, omega: 0.3)    // Peripheral volume
    param Q ~ LogNormal(10.0 L/h, omega: 0.35)   // Inter-compartmental clearance
    param F = 0.8                                  // Bioavailability

    // Compartments
    compartment Depot()                           // GI depot
    compartment Central(volume: V1)               // Plasma
    compartment Peripheral(volume: V2)            // Tissue

    // Dosing
    dose(amt: Dose, route: Oral) -> Depot with bioavailability: F

    // Flows
    flow Absorption = KA * Depot.amount -> Central
    flow Distribution_CP = Q/V1 * Central.amount -> Peripheral
    flow Distribution_PC = Q/V2 * Peripheral.amount -> Central
    flow Elimination = CL/V1 * Central.amount -> External

    // Observations
    observe CONC = Central.C with Combined(sigma_add: 0.05, sigma_prop: 0.1)

    // Micro-constants
    derive K10 = CL / V1
    derive K12 = Q / V1
    derive K21 = Q / V2

    // Macro-constants
    derive ALPHA = 0.5 * (K12 + K21 + K10 + sqrt((K12 + K21 + K10)^2 - 4*K21*K10))
    derive BETA = 0.5 * (K12 + K21 + K10 - sqrt((K12 + K21 + K10)^2 - 4*K21*K10))
    derive T_HALF_ALPHA = 0.693 / ALPHA
    derive T_HALF_BETA = 0.693 / BETA
}
```

## Dosing Specification

### Single Dose

```sio
// IV bolus
dose(amt: 100.0 mg, route: IV) -> Central at t = 0.0

// Oral dose
dose(amt: 500.0 mg, route: Oral) -> Depot at t = 0.0

// Subcutaneous with bioavailability
dose(amt: 40.0 mg, route: SC) -> Depot with bioavailability: 0.64
```

### Multiple Doses

```sio
// BID dosing for 7 days
dosing_regimen {
    route: Oral
    amt: 250.0 mg
    frequency: BID
    duration: 7 days
}

// QD dosing
dosing_regimen {
    route: IV
    amt: 100.0 mg
    frequency: QD
    start: 0.0 h
    n_doses: 10
}

// Custom intervals
dosing_regimen {
    route: Oral
    amt: 500.0 mg
    times: [0.0, 8.0, 16.0, 24.0] h
}
```

### IV Infusion

```sio
// 1-hour infusion
infusion(amt: 500.0 mg, duration: 1.0 h) -> Central at t = 0.0

// Continuous infusion at fixed rate
infusion(rate: 10.0 mg/h) -> Central from t = 0.0 to t = 24.0 h

// Loading dose followed by maintenance infusion
dose(amt: 1000.0 mg, route: IV) -> Central at t = 0.0
infusion(rate: 50.0 mg/h) -> Central from t = 0.0 to t = 168.0 h
```

## Covariate Modeling

MedLang supports incorporating patient covariates:

```sio
model CovariateExample {
    // Covariates
    covariate WT: kg           // Body weight
    covariate AGE: years       // Age
    covariate SEX: i32         // 0=male, 1=female
    covariate CRCL: mL/min     // Creatinine clearance

    // Reference values
    const WT_REF = 70.0 kg
    const AGE_REF = 40.0 years
    const CRCL_REF = 100.0 mL/min

    // Base parameters
    param TVCL ~ LogNormal(10.0 L/h, omega: 0.0)   // Typical CL
    param TVV ~ LogNormal(70.0 L, omega: 0.0)      // Typical V

    // Covariate effects (thetas)
    param THETA_WT_CL = 0.75    // Allometric exponent for CL
    param THETA_WT_V = 1.0      // Allometric exponent for V
    param THETA_AGE = -0.01     // Age effect on CL
    param THETA_SEX = -0.15     // Female effect on CL
    param THETA_CRCL = 0.5      // Renal function effect

    // Inter-individual variability
    param ETA_CL ~ Normal(0.0, omega: 0.3)
    param ETA_V ~ Normal(0.0, omega: 0.25)

    // Individual parameters with covariate effects
    let CL_CRCL = 1.0 + THETA_CRCL * (CRCL - CRCL_REF) / CRCL_REF
    let CL_i = TVCL * (WT/WT_REF)^THETA_WT_CL *
               exp(THETA_AGE * (AGE - AGE_REF)) *
               (1.0 + THETA_SEX * SEX) *
               CL_CRCL *
               exp(ETA_CL)

    let V_i = TVV * (WT/WT_REF)^THETA_WT_V * exp(ETA_V)

    // Model structure
    compartment Central(volume: V_i)
    flow Elimination = CL_i / V_i * Central.amount -> External
    observe CP = Central.C with Proportional(sigma: 0.1)
}
```

## Running Simulations

### Single Subject Simulation

```sio
use medlang::*

fn main() {
    // Define the model
    let model = OneCompartmentIV::new()

    // Set parameters
    let params = ModelParams {
        CL: 10.0 L/h,
        V: 70.0 L
    }

    // Define dosing
    let dose = Dose::iv_bolus(amt: 500.0 mg, time: 0.0 h)

    // Simulation options
    let options = SimOptions {
        t_end: 24.0 h,
        dt: 0.1 h,
        solver: SolverMethod::Tsit5
    }

    // Run simulation
    let result = simulate(model, params, dose, options)?

    // Access results
    for point in result.observations {
        println("t={:.1}h: CP={:.2} mg/L", point.time, point.value)
    }

    println("Cmax: {:.2} mg/L at t={:.1}h", result.cmax, result.tmax)
    println("AUC(0-24): {:.2} mg*h/L", result.auc_0_24)
}
```

### Multiple Subject Simulation

```sio
use medlang::*
use medlang::population::*

fn simulate_population() {
    let model = TwoCompartmentOral::new()

    // Population parameters
    let pop_params = PopulationParams {
        theta: [15.0, 50.0, 100.0, 10.0, 1.5, 0.8],  // CL, V1, V2, Q, KA, F
        omega: omega_matrix_diagonal(&vec![0.09, 0.0625, 0.09, 0.1225, 0.16]),
        sigma: residual_combined(0.05, 0.1)
    }

    // Subject covariates
    let subjects = vec![
        Patient { id: 1, weight: 65.0 kg, age: 35 years },
        Patient { id: 2, weight: 80.0 kg, age: 55 years },
        Patient { id: 3, weight: 70.0 kg, age: 45 years }
    ]

    // Dosing regimen
    let regimen = DosingRegimen::oral_qd(amt: 500.0 mg, n_doses: 7)

    // Simulate population
    let results = simulate_population(
        model: model,
        population: pop_params,
        subjects: subjects,
        dosing: regimen,
        t_end: 168.0 h,
        n_samples: 500
    )?

    // Summary statistics
    for subject in results.subjects {
        println("Subject {}: Cmax={:.1}+/-{:.1}, AUC={:.0}+/-{:.0}",
            subject.id,
            subject.cmax_mean, subject.cmax_sd,
            subject.auc_mean, subject.auc_sd)
    }
}
```

## Observation Models in Detail

### Error Model Types

```sio
// Additive error: Y = F + eps
// eps ~ N(0, sigma_add^2)
observe Y = PRED with Additive(sigma: 0.5)

// Proportional error: Y = F * (1 + eps)
// eps ~ N(0, sigma_prop^2)
observe Y = PRED with Proportional(sigma: 0.1)

// Combined error: Y = F * (1 + eps_prop) + eps_add
observe Y = PRED with Combined(sigma_add: 0.1, sigma_prop: 0.1)

// Power error: SD = sigma * |F|^power
observe Y = PRED with Power(sigma: 0.1, power: 0.5)

// Log-additive: log(Y) = log(F) + eps
observe LOGY = log(PRED) with Additive(sigma: 0.2)
```

### Handling Below Quantification Limit (BQL)

```sio
// Define BQL handling
observe CP = Central.C with Proportional(sigma: 0.1) {
    lloq: 0.1 mg/L,
    bql_method: M3  // Likelihood-based
}

// Alternative BQL methods
// M1: BQL = LLOQ/2
// M3: Likelihood-based (recommended)
// M4: Likelihood-based with conditional
```

## Model Library

MedLang includes pre-built model templates:

```sio
use medlang::library::*

// One-compartment IV
let model1 = OneCompIV::new()

// One-compartment oral with first-order absorption
let model2 = OneCompOral::new()

// Two-compartment IV
let model3 = TwoCompIV::new()

// Two-compartment oral
let model4 = TwoCompOral::new()

// Three-compartment IV
let model5 = ThreeCompIV::new()

// Target-mediated drug disposition (TMDD)
let model6 = TMDD_QSS::new()  // Quasi-steady-state approximation

// Indirect response models
let model7 = IndirectResponse::stimulation_kin()
let model8 = IndirectResponse::inhibition_kout()
```

## Next Steps

- [PBPK Modeling](pbpk-modeling.md) - Physiologically-based pharmacokinetic models
- [Population PK](population-pk.md) - Mixed-effects modeling
- [Dosing Protocols](dosing-protocols.md) - Advanced dosing specification

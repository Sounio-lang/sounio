# PBPK Modeling in Sounio

Physiologically-based pharmacokinetic (PBPK) modeling represents drug disposition using anatomically and physiologically realistic compartments. Sounio provides comprehensive support for PBPK modeling through the `pbpk` and `darwin_pbpk` modules.

## What is PBPK?

PBPK models differ from empirical compartmental models by:

1. **Anatomical basis**: Compartments represent actual organs/tissues
2. **Physiological parameters**: Blood flows, tissue volumes from measured data
3. **Mechanistic clearance**: Enzyme kinetics, transporter-mediated processes
4. **Predictive capability**: Extrapolation across species, populations, doses

### Advantages of PBPK

- Predict PK in untested scenarios (pediatrics, disease, DDI)
- Mechanistic understanding of drug distribution
- Support regulatory submissions (FDA/EMA guidance)
- Species scaling for early drug development

## PBPK Model Structure

### Standard 14-Compartment Model

Sounio's standard PBPK model includes:

```
                    +----------+
                    |   Lung   |<----+
                    +----------+     |
                         |           |
                         v           |
  +------+    +----------+    +----------+    +----------+
  | Dose |--->|  Blood   |--->|  Heart   |--->| Arterial |
  +------+    | (Venous) |    +----------+    |   Pool   |
              +----------+                    +----------+
                   ^                               |
                   |     +-------------------------+
                   |     |     |     |     |     |
                   |     v     v     v     v     v
              +----+---+---+---+---+---+---+---+---+
              |Liver|Kid|Brain|Musc|Adip|Gut|Skin|...|
              +----+---+---+---+---+---+---+---+---+
```

### Tissue Compartments

```sio
use pbpk::types::*

/// 14-compartment PBPK structure
struct PBPKState {
    // Central compartments
    c_blood: mg/L,          // Venous blood
    c_arterial: mg/L,       // Arterial blood

    // Tissue compartments (drug amount in tissue)
    a_liver: mg,
    a_kidney: mg,
    a_brain: mg,
    a_heart: mg,
    a_lung: mg,
    a_muscle: mg,
    a_adipose: mg,
    a_gut: mg,
    a_skin: mg,
    a_bone: mg,
    a_spleen: mg,
    a_pancreas: mg,
    a_rest: mg,             // "Other" tissues

    // Depot for oral dosing
    a_depot: mg,

    // Cumulative clearance
    a_eliminated: mg,
}
```

### Physiological Parameters

```sio
use pbpk::*
use darwin_pbpk::*

/// Reference human (70 kg adult male)
struct PBPKParams {
    // Tissue volumes (L)
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

    // Blood flows (L/h)
    q_blood: f64@L_per_h,   // Cardiac output
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

    // Clearances
    clearance_hepatic: f64@L_per_h,
    clearance_renal: f64@L_per_h,

    // Drug-specific tissue partition coefficients
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

    // Binding parameters
    fu_plasma: f64,         // Fraction unbound in plasma
    hematocrit: f64,        // Blood hematocrit
    bp_ratio: f64,          // Blood:plasma ratio
}
```

## Tissue-Plasma Partition Coefficients

The tissue:plasma partition coefficient (Kp) determines drug distribution. Sounio implements the Rodgers-Rowland method for Kp prediction:

### Rodgers-Rowland Method

```sio
use pbpk::rodgers_rowland::*

/// Drug physicochemical properties
struct DrugProperties {
    name: string,
    mw: f64,            // Molecular weight (g/mol)
    logp: f64,          // Octanol-water partition coefficient
    pka: f64,           // Ionization constant
    fu: f64,            // Fraction unbound in plasma
    bp_ratio: f64,      // Blood:plasma ratio
    is_base: bool,      // True for basic drugs

    // Clearance parameters
    cl_hepatic: f64@L_per_h,
    cl_renal: f64@L_per_h,
    ka: f64@per_h,      // Absorption rate
    f_oral: f64,        // Oral bioavailability
}

/// Predict all Kp values using Rodgers-Rowland
fn predict_all_kp(drug: &DrugProperties) -> AllKpValues {
    let kps = AllKpValues {
        kp_adipose: predict_kp_adipose(drug.logp, drug.fu),
        kp_bone: predict_kp_bone(drug.pka, drug.fu, drug.is_base),
        kp_brain: predict_kp_brain(drug.logp, drug.pka, drug.fu, drug.is_base),
        kp_gut: predict_kp_perfused(drug.logp, drug.fu),
        kp_heart: predict_kp_perfused(drug.logp, drug.fu),
        kp_kidney: predict_kp_kidney(drug.logp, drug.fu, drug.is_base),
        kp_liver: predict_kp_liver(drug.logp, drug.fu),
        kp_lung: predict_kp_lung(drug.logp, drug.pka, drug.fu, drug.is_base),
        kp_muscle: predict_kp_muscle(drug.logp, drug.fu),
        kp_skin: predict_kp_skin(drug.logp, drug.fu),
        kp_spleen: predict_kp_perfused(drug.logp, drug.fu),
        kp_pancreas: predict_kp_perfused(drug.logp, drug.fu),
    }
    return kps
}
```

### Kp Prediction Equations

For neutral and acidic drugs:
```
Kp = (fu_p/fu_t) * (Vw + Vnl * Knl + Vph * Kph)
```

For basic drugs (tissue binding correction):
```
Kp = (fu_p/fu_t) * (Vw + Vnl * Knl + Vph * Kph * Ka_ratio)
```

Where:
- `fu_p` = fraction unbound in plasma
- `fu_t` = fraction unbound in tissue
- `Vw` = fractional water volume
- `Vnl` = fractional neutral lipid volume
- `Vph` = fractional phospholipid volume
- `Knl`, `Kph` = partition coefficients

## Patient Scaling

### Allometric Scaling

Sounio scales PBPK parameters from reference to individual patients:

```sio
use darwin_pbpk::simulation::*

/// Patient data for scaling
struct PatientData {
    age: f64,           // Years
    weight: f64@kg,
    sex: bool,          // true = male
    disease_state: i32, // 0 = healthy
}

/// Scale parameters for patient
fn scale_params_for_patient(base: PBPKParams, patient: PatientData) -> PBPKParams {
    let ref_weight = 70.0@kg

    // Volume scaling: weight^1.0 (isometric)
    let volume_scale = patient.weight / ref_weight

    // Flow scaling: weight^0.75 (allometric, metabolic)
    let flow_scale = pow(patient.weight / ref_weight, 0.75)

    // Age adjustment for renal function (Cockcroft-Gault)
    let age_factor = if patient.age > 30.0 {
        1.0 - 0.01 * (patient.age - 30.0)
    } else {
        1.0
    }
    let age_factor = max(age_factor, 0.3)  // Minimum 30%

    // Sex adjustment (females ~10% lower cardiac output)
    let sex_factor = if patient.sex { 1.0 } else { 0.9 }

    // Scale volumes
    var scaled = base
    scaled.v_liver = base.v_liver * volume_scale
    scaled.v_kidney = base.v_kidney * volume_scale
    // ... all volumes

    // Scale flows
    let flow_adj = flow_scale * sex_factor
    scaled.q_liver = base.q_liver * flow_adj
    scaled.q_kidney = base.q_kidney * flow_adj * age_factor  // Age affects kidney
    // ... all flows

    // Clearances
    scaled.clearance_renal = base.clearance_renal * age_factor

    return scaled
}
```

### Species Scaling (Darwin PBPK)

For preclinical to clinical extrapolation:

```sio
use darwin_pbpk::species::*

/// Species enumeration
enum Species {
    Human,
    Rat,
    Mouse,
    Dog,
    Monkey,
    Minipig,
}

/// Scale from one species to another
fn cross_species_scale(
    params: PBPKParams,
    from_species: Species,
    to_species: Species
) -> PBPKParams {
    // Get species-specific scaling factors
    let bw_from = body_weight(from_species)
    let bw_to = body_weight(to_species)

    // Allometric exponents
    let vol_exp = 1.0    // Volume scales linearly
    let flow_exp = 0.75  // Flow scales sublinearly
    let cl_exp = 0.75    // Clearance scales with metabolic rate

    let ratio = bw_to / bw_from

    var scaled = params
    // Scale volumes
    scaled.v_liver = params.v_liver * pow(ratio, vol_exp)
    // ...

    // Scale flows
    scaled.q_liver = params.q_liver * pow(ratio, flow_exp)
    // ...

    // Scale clearances
    scaled.clearance_hepatic = params.clearance_hepatic * pow(ratio, cl_exp)

    return scaled
}
```

## ODE System for PBPK

The PBPK model is solved as a system of ODEs:

```sio
use ode::bdf::*

/// PBPK differential equations
fn pbpk_rhs(t: f64, state: &[f64], dydt: &![f64], params: &PBPKParams) {
    // Unpack state
    let c_blood = state[0]
    let a_liver = state[1]
    // ... etc

    // Tissue concentrations (amount / volume / Kp for unbound)
    let c_liver = a_liver / params.v_liver
    let c_liver_free = c_liver / params.kp_liver

    // Mass balance equations

    // Venous blood: sum of all tissue efflux - cardiac output
    let q_total = params.q_liver + params.q_kidney + params.q_muscle +
                  params.q_adipose + params.q_brain + params.q_gut +
                  params.q_skin + params.q_bone

    let c_venous_in = (params.q_liver * c_liver_free +
                       params.q_kidney * c_kidney_free +
                       // ... other tissues
                      ) / q_total

    dydt[0] = (c_venous_in - c_blood) * params.q_blood / params.v_blood

    // Liver: hepatic artery + portal vein - hepatic vein - metabolism
    let q_hepatic = params.q_liver + params.q_gut + params.q_spleen + params.q_pancreas
    let c_arterial = c_blood  // Simplified
    let hepatic_extraction = params.clearance_hepatic / q_hepatic
    dydt[1] = params.q_liver * c_arterial +
              (params.q_gut * c_gut_free + params.q_spleen * c_spleen_free +
               params.q_pancreas * c_pancreas_free) -
              q_hepatic * c_liver_free -
              params.clearance_hepatic * c_liver_free * params.fu_plasma

    // Kidney: flow in - flow out - renal clearance
    dydt[2] = params.q_kidney * (c_arterial - c_kidney_free) -
              params.clearance_renal * c_kidney_free * params.fu_plasma

    // Other tissues: flow-limited distribution
    // dA/dt = Q * (C_arterial - C_tissue/Kp)
    dydt[3] = params.q_brain * (c_arterial - c_brain_free)  // Brain
    dydt[4] = params.q_muscle * (c_arterial - c_muscle_free)  // Muscle
    // ... etc
}
```

### Solver Selection

For PBPK models, stiff ODE solvers are recommended due to fast blood circulation:

```sio
use ode::bdf::*

fn run_pbpk_simulation(
    drug: DrugProperties,
    patient: PatientData,
    dose: f64@mg,
    t_end: f64@h
) -> SimulationResult {
    // Initialize parameters
    let base_params = create_default_pbpk_params()
    var params = scale_params_for_patient(base_params, patient)

    // Calculate Kp values
    let kps = predict_all_kp(&drug)
    params.kp_liver = kps.kp_liver
    // ... set all Kp values

    // Initial state (IV bolus to blood)
    var y0: [f64; 16] = [0.0; 16]
    y0[0] = dose / params.v_blood  // Initial blood concentration

    // BDF solver configuration for stiff system
    var config = bdf_config_default()
    config.rtol = 1e-6
    config.atol = 1e-9
    config.max_order = 5

    // Define RHS wrapper
    fn rhs(t: f64, y: &[f64], dydt: &![f64]) {
        pbpk_rhs(t, y, dydt, &params)
    }

    // Solve
    let result = bdf_solve(rhs, &y0, 0.0, t_end as f64, &config)

    return process_results(result, params)
}
```

## PK Metrics Calculation

### Cmax, Tmax, AUC

```sio
use darwin_pbpk::simulation::*

/// Calculate PK metrics from simulation
fn calculate_pk_metrics(
    time_course: &Vec<(f64, f64)>,  // (time, concentration)
    dose: f64@mg
) -> PKMetrics {
    var cmax = 0.0@mg_per_L
    var tmax = 0.0@h
    var auc = 0.0@mg_h_per_L
    var prev_t = 0.0
    var prev_c = 0.0

    for (t, c) in time_course {
        // Track Cmax
        if c > cmax {
            cmax = c
            tmax = t
        }

        // Trapezoidal AUC
        auc = auc + (prev_c + c) / 2.0 * (t - prev_t)
        prev_t = t
        prev_c = c
    }

    // Half-life (from terminal phase)
    let (c_early, c_late, t_early, t_late) = terminal_phase_points(time_course)
    let half_life = 0.693 * (t_late - t_early) / ln(c_early / c_late)

    // Clearance
    let clearance = dose / auc

    // Volume at steady state
    let vdss = clearance * half_life / 0.693

    return PKMetrics {
        cmax_plasma: cmax,
        tmax: tmax,
        auc_0_inf: auc,
        half_life: half_life,
        clearance: clearance,
        vdss: vdss
    }
}
```

## Complete Example: Midazolam PBPK

```sio
use darwin_pbpk::*

fn main() {
    println("=== Midazolam PBPK Simulation ===")

    // Patient: 70kg male, 35 years old
    let patient = PatientData {
        age: 35.0,
        weight: 70.0@kg,
        sex: true,  // Male
        disease_state: 0  // Healthy
    }

    // Midazolam properties
    let midazolam = DrugProperties {
        name: "Midazolam",
        mw: 325.77,
        logp: 3.89,          // Lipophilic
        pka: 6.15,           // Weak base
        fu: 0.03,            // 3% unbound (highly protein bound)
        bp_ratio: 0.65,      // Preferentially in plasma
        is_base: true,

        cl_hepatic: 30.0@L_per_h,  // High CYP3A4 clearance
        cl_renal: 1.0@L_per_h,     // Minimal
        ka: 1.5@per_h,
        f_oral: 0.4                 // 40% (high first-pass)
    }

    // Run simulation
    let result = run_pbpk_simulation(
        drug: midazolam,
        patient: patient,
        dose: 2.0@mg,
        route: Route::IV,
        t_end: 24.0@h
    )

    // Print results
    println("Primary PK Parameters:")
    println("  Cmax (plasma):  {:.2} mg/L", result.cmax_plasma)
    println("  Tmax:           {:.2} h", result.tmax)
    println("  AUC(0-inf):     {:.2} mg*h/L", result.auc_0_inf)
    println("  Half-life:      {:.2} h", result.half_life)
    println("  Clearance:      {:.2} L/h", result.clearance)
    println("  Vdss:           {:.2} L", result.vdss)

    // Tissue concentrations at end
    println("\nFinal Tissue Concentrations (mg/L):")
    println("  Blood:    {:.4}", result.final_state.c_blood)
    println("  Liver:    {:.4}", result.final_state.c_liver)
    println("  Brain:    {:.4}", result.final_state.c_brain)
    println("  Adipose:  {:.4}", result.final_state.c_adipose)
}
```

## Validation Against Clinical Data

```sio
use pbpk::regulatory::*

/// Validate PBPK against observed clinical data
fn validate_pbpk(
    result: SimulationResult,
    observed: ClinicalData
) -> ValidationResult {
    // Fold error calculations
    let fe_cmax = result.cmax_plasma / observed.cmax
    let fe_auc = result.auc_0_inf / observed.auc
    let fe_thalf = result.half_life / observed.half_life

    // Check 2-fold acceptance criterion
    let within_2fold = (
        fe_cmax >= 0.5 && fe_cmax <= 2.0 &&
        fe_auc >= 0.5 && fe_auc <= 2.0 &&
        fe_thalf >= 0.5 && fe_thalf <= 2.0
    )

    println("Validation Results:")
    println("  Cmax fold error: {:.2}", fe_cmax)
    println("  AUC fold error:  {:.2}", fe_auc)
    println("  t1/2 fold error: {:.2}", fe_thalf)
    println("  Within 2-fold:   {}", within_2fold)

    return ValidationResult {
        fold_error_cmax: fe_cmax,
        fold_error_auc: fe_auc,
        fold_error_thalf: fe_thalf,
        within_2fold: within_2fold
    }
}
```

## Next Steps

- [Population PK](population-pk.md) - Adding variability to PBPK models
- [Regulatory Compliance](regulatory-compliance.md) - FDA/EMA submission requirements

# Pharmacokinetics Recipes

Practical recipes for building and using pharmacokinetic models in Sounio.

## One-Compartment IV Bolus Model

### Problem

You need to model the plasma concentration following an IV bolus dose using a simple one-compartment model.

### Solution

Using the MedLang DSL:

```sio
use medlang::pk::one_compartment::*

model OneCompartmentIV {
    // Parameters with uncertainty (Knowledge<T>)
    param CL: Knowledge<L/h> ~ LogNormal(
        mean: 10.0 L/h,
        omega: 0.30  // 30% CV
    )

    param V: Knowledge<L> ~ LogNormal(
        mean: 50.0 L,
        omega: 0.25  // 25% CV
    )

    // Compartment
    compartment Central {
        volume: V
    }

    // Flow: Central -> Elimination
    flow Central -> Elimination {
        rate: CL
    }

    // IV dose into Central
    dose IV {
        into: Central
    }

    // Observable: plasma concentration
    observe Cp: Concentration = Central.concentration
}
```

Manual implementation using epistemic types:

```sio
use epistemic::core::*

struct OneCompartmentParams {
    cl: EpistemicValue,  // Clearance (L/h)
    v: EpistemicValue,   // Volume of distribution (L)
}

// Create parameters with uncertainty
fn create_params() -> OneCompartmentParams {
    OneCompartmentParams {
        cl: epistemic_std(10.0, 3.0, 0.95),   // CL = 10 +/- 3 L/h
        v: epistemic_std(50.0, 12.5, 0.95),  // V = 50 +/- 12.5 L
    }
}

// Concentration at time t after IV bolus dose
fn concentration_iv_bolus(
    dose: f64,
    params: OneCompartmentParams,
    t: f64
) -> EpistemicValue {
    // ke = CL / V
    let ke = div_epistemic(params.cl, params.v)

    // C(t) = (Dose / V) * exp(-ke * t)
    let c0 = epistemic_exact(dose, 1.0)
    let c0_over_v = div_epistemic(c0, params.v)

    // For exp(-ke*t), we need to propagate uncertainty through exp
    let ke_t = mul_epistemic(ke, epistemic_exact(t, 1.0))

    // Approximate exponential decay with uncertainty propagation
    let decay_factor = exp_decay(ke_t)

    return mul_epistemic(c0_over_v, decay_factor)
}

fn exp_decay(x: EpistemicValue) -> EpistemicValue {
    // exp(-x) with uncertainty propagation
    let value = exp_f64(0.0 - x.value)
    let uncertainty = value * get_std_uncertainty(x)

    return epistemic_std(value, uncertainty, x.conf * 0.95)
}
```

### Discussion

The one-compartment IV bolus model assumes:
- Instantaneous distribution throughout the body
- First-order elimination
- Linear pharmacokinetics

The elimination rate constant ke = CL / V, and the half-life t1/2 = 0.693 / ke.

When using epistemic types, uncertainty propagates automatically through all calculations, giving you confidence intervals on predicted concentrations.

---

## Two-Compartment Model with Oral Absorption

### Problem

You need to model a drug with two-compartment kinetics and first-order oral absorption.

### Solution

Using the MedLang DSL:

```sio
use medlang::pk::two_compartment::*

model TwoCompartmentOral {
    // Parameters
    param ka: Knowledge<1/h> ~ LogNormal(mean: 1.0 1/h, omega: 0.40)
    param CL: Knowledge<L/h> ~ LogNormal(mean: 10.0 L/h, omega: 0.30)
    param V1: Knowledge<L> ~ LogNormal(mean: 50.0 L, omega: 0.25)
    param V2: Knowledge<L> ~ LogNormal(mean: 100.0 L, omega: 0.30)
    param Q: Knowledge<L/h> ~ LogNormal(mean: 5.0 L/h, omega: 0.40)
    param F: Knowledge<fraction> ~ Beta(mean: 0.80, alpha: 2.0, beta: 0.5)

    // Compartments
    compartment Gut {
        transit_time: 1.0 / ka
    }

    compartment Central {
        volume: V1
    }

    compartment Peripheral {
        volume: V2
    }

    // Flows
    flow Gut -> Central {
        rate: ka * F
    }

    flow Central -> Elimination {
        rate: CL
    }

    flow Central <-> Peripheral {
        rate: Q
    }

    // Oral dose into Gut
    dose Oral {
        into: Gut
    }

    // Observable
    observe Cp: Concentration = Central.concentration
}
```

### Discussion

The two-compartment model accounts for:
- Distribution phase (alpha phase)
- Elimination phase (beta phase)
- Peripheral tissue binding/distribution

Key parameters:
- CL: Clearance from central compartment
- V1: Central compartment volume
- V2: Peripheral compartment volume
- Q: Inter-compartmental clearance
- ka: Absorption rate constant
- F: Bioavailability

The bidirectional flow (`<->`) indicates equilibrium between compartments.

---

## Population PK with Random Effects

### Problem

You want to model inter-individual variability using a population PK approach.

### Solution

```sio
use medlang::population::model::*
use medlang::population::estimation::*

// Define population model structure
let pop_model = population_model_pk_1cmt_oral()

// Define fixed effects (typical values)
let theta_cl = fixed_effect_log(0, 10.0)   // Typical CL = 10 L/h
let theta_v = fixed_effect_log(1, 50.0)    // Typical V = 50 L
let theta_ka = fixed_effect_log(2, 1.0)    // Typical ka = 1 h^-1
let theta_f = fixed_effect_logit(3, 0.8, 0.0, 1.0)  // Typical F = 0.8

// Define priors on random effects (omega values)
let omega_cl = prior_half_normal(0.3)  // ~30% CV on CL
let omega_v = prior_half_normal(0.25)  // ~25% CV on V
let omega_ka = prior_half_normal(0.4)  // ~40% CV on ka

// Define covariate effects
let wt_on_cl = covariate_power(0, 0, 0.75, 70.0)  // CL scales with weight^0.75
let age_on_cl = covariate_linear(0, 1, -0.01, 40.0)  // CL decreases 1% per year above 40

// Compute individual parameters
fn compute_individual(
    thetas: &Vec<f64>,
    etas: &Vec<f64>,
    covariates: SubjectCovariates
) -> Vec<f64> {
    // Get population parameters
    var ind_params = compute_individual_parameters(
        thetas,
        etas,
        &vec![1, 1, 1, 2]  // Log transforms for CL, V, ka; logit for F
    )

    // Apply covariate effects
    ind_params[0] = apply_covariate_effect(&wt_on_cl, ind_params[0], covariates.weight)
    ind_params[0] = apply_covariate_effect(&age_on_cl, ind_params[0], covariates.age)

    return ind_params
}
```

### Discussion

Population PK models separate:

1. **Fixed effects (theta)**: Typical values in the population
2. **Random effects (eta)**: Individual deviations from typical values
3. **Residual error**: Measurement/model error

The individual parameter is typically:
```
theta_i = theta_pop * exp(eta_i)  // Log-normal distribution
```

This ensures positivity and gives symmetric variability on the log scale.

Covariate effects explain part of the inter-individual variability:
- **Power model**: `CL_i = CL_pop * (WT/70)^0.75`
- **Linear model**: `CL_i = CL_pop * (1 + theta * (COV - ref))`
- **Exponential model**: `CL_i = CL_pop * exp(theta * COV)`

---

## Dosing Optimization with Uncertainty

### Problem

You want to find the optimal dose that achieves a target concentration with high probability, accounting for PK uncertainty.

### Solution

```sio
use epistemic::core::*
use epistemic::montecarlo::*

struct DosingResult {
    dose: f64,
    predicted_cmax: EpistemicValue,
    prob_in_range: f64,
}

fn optimize_dose(
    params: OneCompartmentParams,
    target_min: f64,
    target_max: f64,
    confidence_threshold: f64
) -> DosingResult {
    // Search over dose range
    var best_dose = 0.0
    var best_prob = 0.0

    for dose in [100.0, 200.0, 300.0, 400.0, 500.0] {
        // Predict Cmax with uncertainty
        let cmax = predict_cmax(dose, params)

        // Compute probability of being in therapeutic range
        let prob = probability_in_range(cmax, target_min, target_max)

        if prob > best_prob {
            best_prob = prob
            best_dose = dose
        }
    }

    let final_cmax = predict_cmax(best_dose, params)

    return DosingResult {
        dose: best_dose,
        predicted_cmax: final_cmax,
        prob_in_range: best_prob,
    }
}

fn predict_cmax(dose: f64, params: OneCompartmentParams) -> EpistemicValue {
    // For IV bolus: Cmax = Dose / V
    let dose_ev = epistemic_exact(dose, 1.0)
    return div_epistemic(dose_ev, params.v)
}

fn probability_in_range(value: EpistemicValue, min: f64, max: f64) -> f64 {
    // Using normal approximation
    let mean = value.value
    let std = get_std_uncertainty(value)

    if std < 1.0e-10 {
        if mean >= min && mean <= max { return 1.0 }
        return 0.0
    }

    // P(min < X < max) = Phi((max-mean)/std) - Phi((min-mean)/std)
    let z_max = (max - mean) / std
    let z_min = (min - mean) / std

    let p_max = normal_cdf(z_max)
    let p_min = normal_cdf(z_min)

    return p_max - p_min
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / 1.4142135623730951))
}
```

### Discussion

Dose optimization with uncertainty accounts for:
- PK parameter uncertainty
- Inter-individual variability
- Target concentration range

The goal is to find a dose where:
- P(C_target_min < C < C_target_max) > confidence_threshold

This probabilistic approach is more robust than point estimates because it explicitly considers the risk of under- or over-dosing.

---

## Parameter Sensitivity Analysis

### Problem

You want to understand which PK parameters have the greatest impact on model predictions.

### Solution

```sio
use epistemic::core::*

struct SensitivityResult {
    param_name: string,
    sensitivity_coefficient: f64,
    elasticity: f64,
}

fn sensitivity_analysis(
    base_params: OneCompartmentParams,
    dose: f64,
    time: f64,
    perturbation: f64  // e.g., 0.01 for 1%
) -> Vec<SensitivityResult> {
    var results: Vec<SensitivityResult> = vec![]

    // Base prediction
    let base_conc = concentration_iv_bolus(dose, base_params, time)
    let base_value = base_conc.value

    // Sensitivity to CL
    let cl_high = OneCompartmentParams {
        cl: epistemic_std(base_params.cl.value * (1.0 + perturbation),
                         get_std_uncertainty(base_params.cl), base_params.cl.conf),
        v: base_params.v,
    }
    let conc_cl_high = concentration_iv_bolus(dose, cl_high, time)
    let s_cl = (conc_cl_high.value - base_value) / (perturbation * base_value)

    results.push(SensitivityResult {
        param_name: "CL",
        sensitivity_coefficient: s_cl,
        elasticity: s_cl * base_params.cl.value / base_value,
    })

    // Sensitivity to V
    let v_high = OneCompartmentParams {
        cl: base_params.cl,
        v: epistemic_std(base_params.v.value * (1.0 + perturbation),
                        get_std_uncertainty(base_params.v), base_params.v.conf),
    }
    let conc_v_high = concentration_iv_bolus(dose, v_high, time)
    let s_v = (conc_v_high.value - base_value) / (perturbation * base_value)

    results.push(SensitivityResult {
        param_name: "V",
        sensitivity_coefficient: s_v,
        elasticity: s_v * base_params.v.value / base_value,
    })

    return results
}
```

### Global Sensitivity with Sobol Indices

For more comprehensive analysis, use Sobol sensitivity indices:

```sio
use epistemic::sobol::*

fn global_sensitivity(
    params: OneCompartmentParams,
    dose: f64,
    times: &[f64],
    n_samples: i64
) -> SobolResult {
    // Define parameter ranges
    let ranges = [
        (params.cl.value * 0.5, params.cl.value * 1.5),  // CL range
        (params.v.value * 0.5, params.v.value * 1.5),    // V range
    ]

    // Run Sobol analysis
    return sobol_analyze(
        |p| concentration_at_times(dose, p[0], p[1], times),
        ranges,
        n_samples
    )
}
```

### Discussion

Sensitivity analysis reveals:

1. **Local sensitivity**: How much output changes for small parameter perturbation
2. **Elasticity**: Normalized sensitivity (% change in output per % change in input)
3. **Global sensitivity (Sobol)**: Total effect considering full parameter range

For a one-compartment model:
- At early times: Concentration is most sensitive to V (determines Cmax)
- At late times: Concentration is most sensitive to CL (determines elimination)
- ke = CL/V: Both parameters contribute to elimination rate

---

## See Also

- [Uncertainty Recipes](uncertainty-recipes.md) for general epistemic patterns
- [MedLang Documentation](../domains/medlang.md) for full DSL reference
- [Population Modeling](../domains/population-pk.md) for advanced techniques

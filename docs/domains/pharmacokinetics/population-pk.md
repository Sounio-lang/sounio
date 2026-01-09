# Population PK in Sounio

Population pharmacokinetics (PopPK) uses mixed-effects modeling to characterize drug disposition across patient populations, accounting for both typical behavior and individual variability. Sounio's `medlang::population` module provides comprehensive support for PopPK analysis.

## Population PK Concepts

### The Mixed-Effects Model

Population PK models decompose individual parameters into:

1. **Fixed effects (theta)**: Population typical values
2. **Random effects (eta)**: Inter-individual variability (IIV)
3. **Residual error (epsilon)**: Unexplained variability

```
Parameter_individual = theta_population * exp(eta)
Observation = Prediction * (1 + epsilon)
```

### Hierarchical Structure

Sounio implements a three-level hierarchy:

```sio
// Level 1: Individual (deterministic) - ODE model with fixed theta
let individual_params = bind_params(population_theta, patient_covariates, eta)

// Level 2: Population (hierarchical) - theta_i ~ Population distribution
// eta ~ MVN(0, Omega)

// Level 3: Observation (measurement) - y ~ ErrorModel(prediction)
// y = f(theta_i, t) + g(theta_i, t) * epsilon
```

## Inter-Individual Variability (IIV)

### Omega Matrix

The omega matrix captures variance and covariance of random effects:

```sio
use medlang::population::variability::*

// Create diagonal omega matrix (independent parameters)
var variances: Vec<f64> = vec![]
variances.push(0.09)    // 30% CV for CL
variances.push(0.0625)  // 25% CV for V
variances.push(0.16)    // 40% CV for Ka

let omega = omega_matrix_diagonal(&variances)

// Access elements
let var_cl = omega_get(&omega, 0, 0)    // 0.09
let var_v = omega_get(&omega, 1, 1)     // 0.0625
let cov_cl_v = omega_get(&omega, 0, 1)  // 0.0 (diagonal)
```

### Full Omega with Correlations

```sio
// Create full omega matrix with correlations
var omega = omega_matrix_new(3)

// Set variances (diagonal)
omega_set(&omega, 0, 0, 0.09)    // omega_CL^2
omega_set(&omega, 1, 1, 0.0625)  // omega_V^2
omega_set(&omega, 2, 2, 0.16)    // omega_Ka^2

// Set covariances (off-diagonal) - CL and V often correlated
let rho_cl_v = 0.3  // Correlation
let cov_cl_v = rho_cl_v * sqrt(0.09) * sqrt(0.0625)  // 0.0225
omega_set(&omega, 0, 1, cov_cl_v)  // Also sets (1, 0)
```

### Computing CV% from Omega

For log-normal parameters:
```sio
// CV% = sqrt(exp(omega^2) - 1) * 100
let cv_percent = omega_cv_percent(&omega, 0)  // ~30.5% for omega=0.3
```

### Correlation from Omega

```sio
let correlation = omega_correlation(&omega, 0, 1)
// corr = cov(CL, V) / sqrt(var(CL) * var(V))
```

## Fixed Effects Structure

### One-Compartment Model

```sio
use medlang::population::*

/// Population fixed effects for one-compartment model
struct OneCompFixedEffects {
    cl_pop: f64,    // Population clearance (L/h)
    v_pop: f64,     // Population volume (L)
    ka_pop: f64     // Population absorption rate (1/h)
}

/// Random effects (eta values) for one-compartment
struct OneCompRandomEffects {
    eta_cl: f64,    // IIV on clearance
    eta_v: f64,     // IIV on volume
    eta_ka: f64     // IIV on absorption
}

/// Individual parameters after binding
struct OneCompIndividual {
    cl: f64,        // Individual clearance
    v: f64,         // Individual volume
    ka: f64,        // Individual absorption rate
    ke: f64         // Derived: elimination rate = CL/V
}
```

### Two-Compartment Model

```sio
struct TwoCompFixedEffects {
    cl_pop: f64,    // Central clearance (L/h)
    v1_pop: f64,    // Central volume (L)
    v2_pop: f64,    // Peripheral volume (L)
    q_pop: f64,     // Inter-compartmental clearance (L/h)
    ka_pop: f64     // Absorption rate (1/h)
}

struct TwoCompRandomEffects {
    eta_cl: f64,
    eta_v1: f64,
    eta_v2: f64,
    eta_q: f64,
    eta_ka: f64
}
```

## Parameter Binding

Individual parameters are derived from population values, covariates, and random effects:

```sio
use medlang::population::*

/// Patient covariates
struct PatientCovariates {
    id: i32,
    weight: f64,        // kg
    height: f64,        // cm
    age: f64,           // years
    sex: i32,           // 0=male, 1=female
    crcl: f64,          // mL/min (creatinine clearance)
    albumin: f64,       // g/dL
    cyp2d6: i32,        // 0=PM, 1=IM, 2=EM, 3=UM
    cyp3a4: i32         // 0=Low, 1=Normal, 2=High
}

/// Bind population parameters to individual
/// theta_i = theta_pop * cov_effect * exp(eta_i)
fn bind_one_comp_params(
    pop: OneCompFixedEffects,
    cov: PatientCovariates,
    eta: OneCompRandomEffects
) -> OneCompIndividual {
    // Allometric weight scaling
    let wt_cl = pow(cov.weight / 70.0, 0.75)  // CL scales as BW^0.75
    let wt_v = cov.weight / 70.0               // V scales linearly

    // Calculate individual parameters
    let cl = pop.cl_pop * wt_cl * exp(eta.eta_cl)
    let v = pop.v_pop * wt_v * exp(eta.eta_v)
    let ka = pop.ka_pop * exp(eta.eta_ka)

    return OneCompIndividual {
        cl: cl,
        v: v,
        ka: ka,
        ke: cl / v
    }
}
```

## Residual Error Models

Residual error accounts for unexplained variability:

### Error Model Types

```sio
use medlang::population::variability::*

// Additive error: Y = IPRED + epsilon
// SD = sigma_add (constant)
let error_add = residual_additive(sigma: 0.5)

// Proportional error: Y = IPRED * (1 + epsilon)
// SD = sigma_prop * |IPRED|
let error_prop = residual_proportional(sigma: 0.1)

// Combined error: Y = IPRED * (1 + eps_prop) + eps_add
// SD = sqrt(sigma_add^2 + sigma_prop^2 * IPRED^2)
let error_comb = residual_combined(sigma_add: 0.1, sigma_prop: 0.1)

// Power error: SD = sigma * |IPRED|^power
let error_power = residual_power(sigma: 0.1, power: 0.5)
```

### Computing Standard Deviation

```sio
fn residual_std(error: &ResidualError, prediction: f64) -> f64 {
    if error.model_type == 0 {  // Additive
        return error.sigma_add
    }
    if error.model_type == 1 {  // Proportional
        return error.sigma_prop * abs(prediction)
    }
    if error.model_type == 2 {  // Combined
        let add_sq = error.sigma_add * error.sigma_add
        let prop_sq = error.sigma_prop * error.sigma_prop * prediction * prediction
        return sqrt(add_sq + prop_sq)
    }
    if error.model_type == 3 {  // Power
        return error.sigma_add * pow(abs(prediction), error.power)
    }
    return error.sigma_add
}
```

### Weighted Residuals

```sio
// Weighted residual (WRES)
fn weighted_residual(error: &ResidualError, observed: f64, predicted: f64) -> f64 {
    let sd = residual_std(error, predicted)
    return (observed - predicted) / sd
}

// Conditional weighted residual (CWRES)
fn conditional_weighted_residual(
    error: &ResidualError,
    observed: f64,
    ipred: f64,       // Individual prediction
    pred: f64,        // Population prediction
    h: f64,           // Derivative
    omega: f64        // Random effect variance
) -> f64 {
    let g = residual_std(error, ipred) / residual_std(error, pred)
    let var_approx = g * g + h * h * omega
    return (observed - ipred) / (residual_std(error, ipred) * sqrt(var_approx))
}
```

## Random Effect Sampling

### Uncorrelated Sampling

```sio
use medlang::population::*

/// Sample random effects (Box-Muller)
fn sample_normal(mean: f64, sd: f64, u1: f64, u2: f64) -> f64 {
    let z = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    return mean + sd * z
}

/// Sample independent random effects
fn sample_one_comp_eta(
    omega: OneCompOmega,
    u1: f64, u2: f64, u3: f64, u4: f64, u5: f64, u6: f64
) -> OneCompRandomEffects {
    return OneCompRandomEffects {
        eta_cl: sample_normal(0.0, omega.omega_cl, u1, u2),
        eta_v: sample_normal(0.0, omega.omega_v, u3, u4),
        eta_ka: sample_normal(0.0, omega.omega_ka, u5, u6)
    }
}
```

### Correlated Sampling (Cholesky)

For correlated random effects, use Cholesky decomposition:

```sio
use medlang::population::variability::*

/// Cholesky decomposition of omega matrix
fn cholesky_decompose(omega: &OmegaMatrix) -> OmegaMatrix {
    let n = omega.dimension
    var L = omega_matrix_new(n)

    for i in 0..n {
        for j in 0..=i {
            var sum = 0.0

            if j == i {
                // Diagonal element
                for k in 0..j {
                    let l_jk = omega_get(&L, j, k)
                    sum = sum + l_jk * l_jk
                }
                let diag = omega_get(omega, i, i) - sum
                if diag > 0.0 {
                    omega_set(&L, i, j, sqrt(diag))
                }
            } else {
                // Off-diagonal element
                for k in 0..j {
                    let l_ik = omega_get(&L, i, k)
                    let l_jk = omega_get(&L, j, k)
                    sum = sum + l_ik * l_jk
                }
                let l_jj = omega_get(&L, j, j)
                if l_jj > 0.0 {
                    omega_set(&L, i, j, (omega_get(omega, i, j) - sum) / l_jj)
                }
            }
        }
    }
    return L
}

/// Sample correlated random effects: eta = L * z
fn sample_correlated_etas(L: &OmegaMatrix, z: &Vec<f64>) -> Vec<f64> {
    let n = L.dimension as usize
    var eta: Vec<f64> = vec![]

    for i in 0..n {
        var sum = 0.0
        for j in 0..=i {
            sum = sum + omega_get(L, i as i64, j as i64) * z[j]
        }
        eta.push(sum)
    }
    return eta
}
```

## Population Simulation

### Simulating a Population

```sio
use medlang::population::*

fn simulate_population(
    pop: OneCompFixedEffects,
    omega: OneCompOmega,
    base_cov: PatientCovariates,
    n_subjects: i32,
    seed: i64
) -> PopulationResults {
    var results: Vec<IndividualResult> = vec![]

    for i in 0..n_subjects {
        // Generate random numbers
        let r1 = pseudo_random(seed + i * 6)
        let r2 = pseudo_random(seed + i * 6 + 1)
        let r3 = pseudo_random(seed + i * 6 + 2)
        let r4 = pseudo_random(seed + i * 6 + 3)
        let r5 = pseudo_random(seed + i * 6 + 4)
        let r6 = pseudo_random(seed + i * 6 + 5)

        // Sample random effects
        let eta = sample_one_comp_eta(omega, r1, r2, r3, r4, r5, r6)

        // Bind to individual parameters
        let ind = bind_one_comp_params(pop, base_cov, eta)

        // Run simulation for this individual
        let sim_result = simulate_one_compartment(ind, dose, t_end)

        results.push(IndividualResult {
            id: i,
            params: ind,
            eta: eta,
            cmax: sim_result.cmax,
            auc: sim_result.auc
        })
    }

    return PopulationResults {
        individuals: results,
        summary: compute_summary(&results)
    }
}
```

### Epistemic Integration

Sounio uniquely integrates epistemic uncertainty with population variability:

```sio
use epistemic::*
use medlang::population::*

fn simulate_epistemic_population(
    pop: EpistemicPopParams,  // Parameters with confidence
    omega: OmegaMatrix,
    n_subjects: i32
) -> EpistemicPopResults {
    // Each parameter carries confidence
    let cl_pop = pop.cl  // Knowledge[L/h, confidence >= 0.75]

    var results: Vec<EpistemicIndividual> = vec![]

    for i in 0..n_subjects {
        // Sample random effects
        let eta = sample_etas(omega)

        // Individual parameters inherit base confidence
        let cl_i = Knowledge::new(
            value: cl_pop.value * wt_scale * exp(eta.cl),
            confidence: cl_pop.confidence * 0.95,  // Slight reduction
            source: "Population simulation"
        )

        // Simulate
        let sim = simulate_with_epistemic(cl_i, v_i, ka_i)

        results.push(sim)
    }

    // Summary maintains epistemic tracking
    let summary = EpistemicSummary::from_population(&results)
    // summary.cmax has propagated confidence

    return EpistemicPopResults {
        individuals: results,
        summary: summary
    }
}
```

## Shrinkage

Shrinkage measures how much individual estimates (EBEs) collapse toward population means:

```sio
/// Compute eta shrinkage: 1 - SD(EBE_eta) / omega
fn compute_shrinkage(
    iiv: &InterIndividualVariability,
    ebe_etas: &Vec<Vec<f64>>
) -> Vec<f64> {
    let n_subjects = ebe_etas.len()
    let n_params = iiv.omega.dimension as usize

    var shrinkage: Vec<f64> = vec![]

    for p in 0..n_params {
        var sum = 0.0
        var sum_sq = 0.0
        var count = 0.0

        for s in 0..n_subjects {
            sum = sum + ebe_etas[s][p]
            sum_sq = sum_sq + ebe_etas[s][p] * ebe_etas[s][p]
            count = count + 1.0
        }

        if count > 1.0 {
            let mean = sum / count
            let var_ebe = (sum_sq - count * mean * mean) / (count - 1.0)
            let sd_ebe = sqrt(var_ebe)
            let omega_sd = sqrt(omega_get(&iiv.omega, p as i64, p as i64))

            if omega_sd > 0.0 {
                shrinkage.push(1.0 - sd_ebe / omega_sd)
            } else {
                shrinkage.push(0.0)
            }
        }
    }

    return shrinkage
}
```

High shrinkage (>30%) indicates:
- Insufficient data for individual estimation
- Potential model misspecification
- EBEs unreliable for individual predictions

## Between-Occasion Variability (BOV)

For parameters that vary within subjects across occasions:

```sio
struct BetweenOccasionVariability {
    gamma: OmegaMatrix,    // BOV variance-covariance
    param_names: Vec<i64>,
    n_occasions: i64
}

fn apply_bov(
    base_eta: f64,         // Subject-level IIV
    gamma_var: f64,        // BOV variance
    occasion: i32,
    seed: i64
) -> f64 {
    // Sample occasion-specific deviation
    let kappa = sample_normal(0.0, sqrt(gamma_var), seed, seed + 1)
    return base_eta + kappa
}
```

## Complete Population PK Example

```sio
use medlang::*
use medlang::population::*

fn main() {
    println("=== Population PK Simulation ===")

    // Population parameters
    let pop = OneCompFixedEffects {
        cl_pop: 10.0,   // L/h
        v_pop: 50.0,    // L
        ka_pop: 1.0     // 1/h
    }

    // Variability
    let omega = OneCompOmega {
        omega_cl: 0.3,   // 30% CV
        omega_v: 0.2,    // 20% CV
        omega_ka: 0.4,   // 40% CV
        rho_cl_v: 0.3    // CL-V correlation
    }

    // Residual error
    let error = residual_combined(0.1, 0.1)

    // Reference patient
    let patient = PatientCovariates {
        id: 0,
        weight: 70.0,
        age: 40.0,
        sex: 0,
        crcl: 120.0,
        albumin: 4.0,
        cyp2d6: 2,
        cyp3a4: 1
    }

    // Simulate 100 subjects
    let results = simulate_population(pop, omega, patient, 100, 12345)

    // Summary statistics
    println("\nPopulation Summary:")
    println("  CL: {:.1} +/- {:.1} L/h", results.summary.cl_mean, results.summary.cl_sd)
    println("  V:  {:.1} +/- {:.1} L", results.summary.v_mean, results.summary.v_sd)
    println("  Ka: {:.2} +/- {:.2} 1/h", results.summary.ka_mean, results.summary.ka_sd)

    println("\nPK Metrics:")
    println("  Cmax: {:.2} +/- {:.2} mg/L", results.summary.cmax_mean, results.summary.cmax_sd)
    println("  AUC:  {:.0} +/- {:.0} mg*h/L", results.summary.auc_mean, results.summary.auc_sd)
    println("  t1/2: {:.1} +/- {:.1} h", results.summary.thalf_mean, results.summary.thalf_sd)

    // Shrinkage
    let shrinkage = compute_shrinkage(&results)
    println("\nShrinkage:")
    println("  CL: {:.1}%", shrinkage[0] * 100.0)
    println("  V:  {:.1}%", shrinkage[1] * 100.0)
    println("  Ka: {:.1}%", shrinkage[2] * 100.0)
}
```

## Next Steps

- [Dosing Protocols](dosing-protocols.md) - Regimen specification
- [Regulatory Compliance](regulatory-compliance.md) - FDA/EMA submission support

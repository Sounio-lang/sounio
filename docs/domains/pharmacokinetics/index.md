# Pharmacokinetics in Sounio

Sounio provides first-class support for pharmacokinetic and pharmacodynamic (PK/PD) modeling through its MedLang domain-specific language, PBPK modeling framework, and population PK analysis tools. This documentation covers Sounio's comprehensive pharmacometric capabilities.

## Why Sounio for PK/PD

Sounio offers unique advantages for pharmacometric modeling that no other language provides:

### Epistemic Computing for Clinical Decisions

Every value in Sounio can carry **uncertainty information** through `Knowledge<T>` types. In PK modeling, this means:

```sio
// Parameters carry confidence and provenance
let cl_hepatic: Knowledge[L/h, confidence >= 0.75] = Knowledge::new(
    value: 30.0,
    confidence: 0.82,
    source: "Population PK study NCT001234"
)

// Uncertainty propagates automatically through computations
let auc = dose / cl_hepatic  // AUC inherits propagated confidence

// Confidence gates for clinical decisions
if auc.confidence >= 0.80 {
    recommend_dosing(auc)
} else {
    require_therapeutic_monitoring()
}
```

### Compile-Time Unit Safety

Sounio's dimensional analysis catches unit errors at compile time:

```sio
let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume  // Type-checked

let clearance: L/h = 10.0
let ke: 1/h = clearance / volume  // Compile error: L/h / mL != 1/h
```

### Regulatory Compliance Built-In

Sounio's provenance tracking and audit trails support FDA 21 CFR Part 11 and EMA PBPK guidelines:

- Automatic provenance chains for every computed value
- Complete audit trails for regulatory submissions
- Validation metrics (GMFE, AFE, AAFE) with acceptance criteria
- Report generation for FDA/EMA submission formats

## Key Modules

Sounio's PK/PD capabilities are organized into these core modules:

| Module | Description |
|--------|-------------|
| `medlang::*` | MedLang DSL for model specification |
| `medlang::pk` | Compartmental PK models |
| `medlang::population` | Population PK (mixed effects) |
| `medlang::dose` | Dosing protocols and regimens |
| `pbpk::*` | Physiologically-based PK |
| `darwin_pbpk::*` | Darwin PBPK platform integration |
| `ode::*` | ODE solvers (Tsit5, BDF, Rosenbrock) |

## Quick Example: One-Compartment Model with Uncertainty

```sio
use medlang::*
use ode::tsit5::*

// Define a one-compartment IV model with epistemic tracking
model OneCompIV {
    // Parameters with priors and uncertainty
    param CL ~ LogNormal(10.0, omega: 0.3)  // Clearance, L/h
    param V ~ LogNormal(70.0, omega: 0.25)  // Volume, L

    // Compartments
    compartment Central(volume: V)

    // Flows
    flow Elimination = CL / V * Central.amount -> External

    // Observation model
    observe Cp = Central.C with Proportional(0.1)
}

// Create patient profile
let patient = Patient {
    weight: 75.0 kg,
    age: 45 years,
    sex: Sex::Male
}

// Create epistemic parameters
let params = PBPKParams {
    cl_hepatic: Knowledge::new(10.0 L/h, confidence: 0.85),
    vd: Knowledge::new(70.0 L, confidence: 0.82),
    ka: Knowledge::new(1.5 1/h, confidence: 0.75)
}

// Run simulation with epistemic tracking
let result = simulate(
    model: OneCompIV,
    dose: 500.0 mg,
    duration: 24.0 h,
    dt: 0.1 h
)

// Result carries propagated confidence
println("Cmax: {} mg/L (confidence: {:.1}%)",
    result.cmax, result.confidence * 100.0)
```

## Learning Path

We recommend the following progression through the PK documentation:

1. **[MedLang Tutorial](medlang-tutorial.md)** - Learn the MedLang DSL syntax for model specification
2. **[PBPK Modeling](pbpk-modeling.md)** - Physiologically-based pharmacokinetic models
3. **[Population PK](population-pk.md)** - Mixed-effects modeling and variability
4. **[Dosing Protocols](dosing-protocols.md)** - Regimen specification and adaptive dosing
5. **[Regulatory Compliance](regulatory-compliance.md)** - FDA/EMA submission support

## Solver Selection Guide

Sounio provides multiple ODE solvers optimized for different PK modeling scenarios:

| Problem Type | Recommended Solver | Use Case |
|--------------|-------------------|----------|
| Non-stiff PK | Tsit5 | Standard compartmental models |
| Stiff PBPK | BDF | 14-compartment PBPK with fast blood circulation |
| Real-time dosing | RK4 | Fixed-step for embedded/control systems |
| Moderate stiffness | Rosenbrock | Faster than BDF for less stiff problems |

### Solver Example

```sio
use ode::*

// For non-stiff one-compartment model
let config = SolverOptions::default()
let solution = solve_tsit5(one_comp_rhs, y0, 0.0, 24.0, &config)

// For stiff 14-compartment PBPK
let bdf_config = bdf_config_default()
let solution = bdf_solve(pbpk14_rhs, y0, 0.0, 72.0, &bdf_config)
```

## Standard PK Types

Sounio provides semantically meaningful wrapper types for pharmacokinetic quantities:

```sio
// Amount types
let dose: mg = 500.0
let amount: DrugAmount = dose_mg(500.0)

// Volume types
let plasma_volume: L = 3.0
let vd: VolumeL = volume_L(70.0)

// Concentration types
let conc: mg/L = dose / plasma_volume
let cp: Concentration = conc_mg_per_L(10.0)

// Clearance types
let cl: L/h = 10.0
let hepatic_cl: HepaticClearance = clearance_L_per_h(30.0)

// Rate constants
let ke: 1/h = cl / vd
let ka: AbsorptionRate = rate_per_h(1.5)

// Time
let tmax: h = 2.0
let half_life: HalfLife = rate_to_halflife(ke)
```

## Validation and Testing

All PK models should be validated against observed data:

```sio
use pbpk::regulatory::*

// Calculate validation metrics
let metrics = ValidationMetrics::calculate(&predicted, &observed)?

// Check FDA acceptance criteria
if metrics.gmfe <= 2.0 && metrics.within_2fold >= 0.80 {
    println("Model passes FDA qualification criteria")
}

// Generate regulatory report
let report = generate_fda_report(
    drug: &midazolam,
    params: &params,
    result: &simulation_result,
    observed_data: &clinical_data,
    config: &report_config
)
```

## Further Resources

- [Sounio Standard Library Reference](/docs/stdlib/index.md)
- [Epistemic Computing Guide](/docs/guides/epistemic-computing.md)
- [Unit System Documentation](/docs/guides/units.md)
- [FDA PBPK Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/physiologically-based-pharmacokinetic-analyses-format-and-content-guidance-industry)
- [EMA PBPK Guideline](https://www.ema.europa.eu/en/reporting-physiologically-based-pharmacokinetic-pbpk-modelling-simulation-scientific-guideline)

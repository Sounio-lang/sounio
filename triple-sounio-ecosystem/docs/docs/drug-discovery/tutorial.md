# Drug Discovery Pipeline Tutorial

A comprehensive guide to the **drug-discovery pipeline** — a 3-stage epistemic system written in pure Sounio that combines virtual screening, pharmacokinetic modeling, and clinical trial simulation.

## Overview

The pipeline automates computational drug discovery workflows:

```
┌──────────────────┐
│  Stage 1         │
│  Screening       │ — Virtual screening with Lipinski's Rule of Five
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Stage 2         │
│  PK/PD Modeling  │ — Pharmacokinetic/pharmacodynamic modeling
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Stage 3         │
│  Trial Sim       │ — Clinical trial outcome prediction
└──────────────────┘
         ↓
    ┌────────┐
    │ Results│ — Efficacy, toxicity, dosing recommendations
    └────────┘
```

Each stage propagates **measurement uncertainty** from molecular properties through patient safety predictions.

---

## Architecture

The pipeline consists of 4 Sounio modules:

| Module | Purpose |
|--------|---------|
| `types.sio` | Domain types (Molecule, PKParameters, Patient, etc.) |
| `screening.sio` | Lipinski filtering + SMILES parsing |
| `pkpd.sio` | Two-compartment PK model with epistemic uncertainty |
| `pipeline.sio` | Orchestration + full workflow |

### File Structure

```
drug-discovery/
├── src/
│   ├── types.sio       # Domain types
│   ├── screening.sio   # Stage 1: Virtual screening
│   ├── pkpd.sio        # Stage 2: PK/PD modeling
│   ├── pipeline.sio    # Stage 3: Clinical simulation + main
│   └── lib.sio         # Shared utilities
├── examples/
│   ├── aspirin.sio
│   ├── ibuprofen.sio
│   └── sample_run.sio
├── tests/
│   └── test_pipeline.sio
└── sounio.toml         # Project manifest
```

---

## Stage 1: Virtual Screening

### Purpose

**Filter compounds** based on drug-like properties before expensive PK studies.

Uses **Lipinski's Rule of Five**:

- Molecular weight < 500 Da
- LogP < 5
- H-bond donors ≤ 5
- H-bond acceptors ≤ 10

### Input

A molecule with:

```sounio
struct Molecule {
    smiles: string,              // SMILES notation
    molecular_weight: f64,       // Daltons
    log_p: f64,                  // Octanol-water partition
    hbd: i64,                    // H-bond donors
    hba: i64                     // H-bond acceptors
}
```

### Example: Aspirin

```sounio
let aspirin: Molecule = Molecule {
    smiles: "CC(=O)Oc1ccccc1C(=O)O",
    molecular_weight: 180.16,
    log_p: 1.19,
    hbd: 2,
    hba: 3
}

// Check Lipinski compliance
let passes: bool = screening_check(aspirin)
print(passes)  // true
```

### Output

```sounio
struct ScreeningResult {
    mol_id: i64,
    passed: bool,
    confidence: f64,
    violations: list[string]
}
```

### Code Example (screening.sio)

```sounio
fn apply_lipinski(mol: Molecule) -> ScreeningResult {
    var violations_count = 0
    var violation_list = []

    // Check MW
    if mol.molecular_weight >= 500.0 {
        violations_count = violations_count + 1
        violation_list = violation_list ++ ["MW >= 500"]
    }

    // Check LogP
    if mol.log_p >= 5.0 {
        violations_count = violations_count + 1
        violation_list = violation_list ++ ["LogP >= 5"]
    }

    // Check HBD
    if mol.hbd > 5 {
        violations_count = violations_count + 1
        violation_list = violation_list ++ ["HBD > 5"]
    }

    // Check HBA
    if mol.hba > 10 {
        violations_count = violations_count + 1
        violation_list = violation_list ++ ["HBA > 10"]
    }

    // Pass if ≤1 violation
    let passed = violations_count <= 1
    let confidence = (5.0 - (violations_count as f64)) / 5.0

    return ScreeningResult {
        mol_id: mol.id,
        passed: passed,
        confidence: confidence,
        violations: violation_list
    }
}
```

---

## Stage 2: PK/PD Modeling

### Purpose

**Predict pharmacokinetics** (absorption, distribution, metabolism, elimination) and **pharmacodynamics** (drug effect over time) with measurement uncertainty.

### Model

A **two-compartment model** with first-order absorption:

```
Input (dose)
   ↓
Central compartment (concentration in blood)
   ↓ ↔ Peripheral compartment (tissues)
   ↓
Elimination
```

### Key Parameters

```sounio
struct PKParameters {
    ka: Knowledge<f64>,           // Absorption rate (1/hr)
    kel: Knowledge<f64>,          // Elimination rate (1/hr)
    volume: Knowledge<f64>,       // Central volume (L)
    clearance: Knowledge<f64>     // Total clearance (mL/min)
}
```

Each parameter is epistemic — includes uncertainty from population variability.

### Epistemic Arithmetic

**GUM rules** propagate uncertainty through the model:

```sounio
let dose: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
let clearance: Knowledge<mL_min> = measure(120.0, uncertainty: 6.0)

// AUC = dose / clearance (with automatic uncertainty)
let auc = dose / clearance
// Result: Knowledge(4.17 ± 0.24 mg·min/mL)
```

### Example: Calculate Exposure

```sounio
fn calculate_auc(
    dose: Knowledge<mg>,
    clearance: Knowledge<mL_min>
) -> Knowledge<mg_min_mL> {
    dose / clearance
}

fn calculate_cmax(
    dose: Knowledge<mg>,
    volume: Knowledge<L>
) -> Knowledge<mg_L> {
    dose / volume
}

// Usage with uncertain measurements
let dose_measured = measure(500.0, uncertainty: 2.5)
let cl_measured = measure(120.0, uncertainty: 6.0)
let vd_measured = measure(50.0, uncertainty: 2.5)

let auc = calculate_auc(dose_measured, cl_measured)
let cmax = calculate_cmax(dose_measured, vd_measured)

print(auc)   // Shows with propagated uncertainty
print(cmax)  // Shows with propagated uncertainty
```

### Time-Dependent Concentration

For a given time *t* after single dose:

```sounio
fn concentration_at_time(
    dose: Knowledge<mg>,
    kel: Knowledge<f64>,  // 1/hr
    t: f64                // hours
) -> Knowledge<mg_L> {
    dose * exp(0.0 - kel.value * t)
}
```

---

## Stage 3: Clinical Trial Simulation

### Purpose

**Predict trial outcomes** given:
- Patient population (weight, age, organ function)
- Drug parameters (PK, efficacy threshold, safety limits)
- Dosing regimen

### Simulation Output

```sounio
struct SimulationResult {
    patient_id: i64,
    dose_mg: f64,
    predicted_auc: Knowledge<mg_min_mL>,
    predicted_cmax: Knowledge<mg_L>,
    efficacy_prob: f64,           // P(effect achieved)
    toxicity_risk: f64,           // P(toxicity)
    recommended_dose: Knowledge<mg>
}
```

### Example: Efficacy Prediction

Assume:
- Efficacy threshold: AUC > 2000 mg·min/mL
- Toxicity threshold: Cmax > 10 mg/L

```sounio
fn predict_efficacy(
    auc: Knowledge<mg_min_mL>,
    efficacy_threshold: f64
) -> f64 {
    // Probability that AUC > threshold (assuming normal distribution)
    // P(X > μ + σ) ≈ 16% for 1σ above mean
    // P(X > μ + 1.96σ) ≈ 2.5%

    let distance = (auc.value - efficacy_threshold) / auc.epsilon

    // Cumulative normal: higher distance = higher probability
    if distance > 2.0 {
        return 0.95  // Very confident
    } else if distance > 0.0 {
        return 0.5 + (distance / 4.0)  // Proportional
    } else {
        return 0.5 - ((-distance) / 4.0)  // Below threshold
    }
}

fn predict_toxicity(
    cmax: Knowledge<mg_L>,
    toxicity_threshold: f64
) -> f64 {
    // Probability that Cmax > toxicity_threshold

    let distance = (cmax.value - toxicity_threshold) / cmax.epsilon

    if distance > 2.0 {
        return 0.05  // Likely safe
    } else if distance > 0.0 {
        return 0.5 - (distance / 4.0)
    } else {
        return 0.5 + ((-distance) / 4.0)
    }
}
```

### Dosing Optimization

Find the dose that maximizes:

**Benefit = P(efficacy) - P(toxicity)**

```sounio
fn optimize_dose(
    patient: PatientData,
    pk_params: PKParameters,
    efficacy_threshold: f64,
    toxicity_threshold: f64,
    dose_range: [f64; 2]  // [min, max]
) -> Knowledge<mg> {
    var best_dose = dose_range[0]
    var best_benefit = 0.0 - 100.0

    var dose = dose_range[0]
    while dose <= dose_range[1] {
        let dose_ep = measure(dose, uncertainty: dose * 0.05)
        let auc = calculate_auc(dose_ep, pk_params.clearance)
        let cmax = calculate_cmax(dose_ep, pk_params.volume)

        let p_efficacy = predict_efficacy(auc, efficacy_threshold)
        let p_toxicity = predict_toxicity(cmax, toxicity_threshold)

        let benefit = p_efficacy - p_toxicity
        if benefit > best_benefit {
            best_benefit = benefit
            best_dose = dose
        }

        dose = dose + 25.0  // Step by 25 mg
    }

    return measure(best_dose, uncertainty: best_dose * 0.05)
}
```

---

## Complete Example: Simulating a Trial

### Setup

```sounio
// Patient cohort
let patients: [PatientData; 3] = [
    PatientData {
        id: 1,
        weight_kg: 70.0,
        age: 45,
        creatinine_clearance: 90.0
    },
    PatientData {
        id: 2,
        weight_kg: 85.0,
        age: 52,
        creatinine_clearance: 75.0
    },
    PatientData {
        id: 3,
        weight_kg: 65.0,
        age: 38,
        creatinine_clearance: 110.0
    }
]

// Drug parameters (from preclinical studies)
let pk_params: PKParameters = PKParameters {
    ka: measure(0.5, uncertainty: 0.05),        // 1/hr
    kel: measure(0.12, uncertainty: 0.012),     // 1/hr
    volume: measure(50.0, uncertainty: 2.5),    // L
    clearance: measure(120.0, uncertainty: 6.0) // mL/min
}

// Thresholds
let efficacy_threshold = 2000.0   // mg·min/mL
let toxicity_threshold = 10.0     // mg/L
let target_dose = 500.0           // mg
```

### Run Simulation

```sounio
fn run_trial(
    patients: [PatientData],
    pk_params: PKParameters,
    dose_target: f64
) -> [SimulationResult] {
    var results = []

    for patient in patients {
        // Personalize dose by weight
        let dose_personalized = dose_target * (patient.weight_kg / 70.0)
        let dose_ep = measure(dose_personalized, uncertainty: dose_personalized * 0.05)

        // Calculate exposure
        let auc = calculate_auc(dose_ep, pk_params.clearance)
        let cmax = calculate_cmax(dose_ep, pk_params.volume)

        // Predict outcomes
        let p_eff = predict_efficacy(auc, efficacy_threshold)
        let p_tox = predict_toxicity(cmax, toxicity_threshold)

        let result = SimulationResult {
            patient_id: patient.id,
            dose_mg: dose_personalized,
            predicted_auc: auc,
            predicted_cmax: cmax,
            efficacy_prob: p_eff,
            toxicity_risk: p_tox,
            recommended_dose: dose_ep
        }

        results = results ++ [result]
    }

    return results
}

// Execute
let trial_results = run_trial(patients, pk_params, target_dose)

// Display results
for result in trial_results {
    print(result.patient_id)
    print(result.efficacy_prob)
    print(result.toxicity_risk)
}
```

---

## Running the Pipeline

### Compile & Execute

```bash
cd triple-sounio-ecosystem/drug-discovery

# Type-check
souc check src/pipeline.sio

# Run (JIT)
souc run src/pipeline.sio

# Compile to native ELF
souc run --native src/pipeline.sio pipeline.elf
./pipeline.elf
```

### From Python

```python
import sounio

# Type-check
result = sounio.check_file("src/pipeline.sio")
if result.ok:
    print("✓ Pipeline is type-safe")

# Run
result = sounio.run_file("src/pipeline.sio")
print(result.stdout)
print(f"Runtime: {result.runtime_seconds:.2f}s")
```

### From Jupyter

```sounio
import src::pipeline

// Run inline
let results = pipeline::run_full_pipeline(molecules, pk_params, patients)
print(results)
```

---

## Interpreting Results

### Efficacy Probability

Higher is better. Represents P(achieving therapeutic effect).

- **>80%**: Likely efficacious
- **50-80%**: Borderline
- **<50%**: Insufficient efficacy

### Toxicity Risk

Lower is better. Represents P(experiencing adverse event).

- **<5%**: Very safe
- **5-15%**: Acceptable
- **>15%**: Concerning

### Confidence Intervals

Each `Knowledge` value shows:

```
AUC = 2500 ± 125 mg·min/mL
```

The ±125 range represents **95% confidence interval** (approximately ±2σ).

**Interpretation:** 95% confident true AUC is between 2375–2625.

### Decision Making

**Choose dose if:**

- Efficacy probability ≥ 70% AND
- Toxicity risk ≤ 10% AND
- Uncertainty bands don't cross critical thresholds

---

## Advanced Topics

### Population Variability

Model inter-individual PK variability:

```sounio
// Clearance with 15% CV (coefficient of variation)
let cl_population = measure(120.0, uncertainty: 120.0 * 0.15)

// Weight adjustment
let cl_patient = cl_population * (patient_weight / 70.0)

// Result: personalized clearance with combined uncertainty
```

### Nonlinear Pharmacokinetics

For saturable metabolism (Michaelis-Menten):

```sounio
fn vm_clearance(
    dose: f64,
    vmax: f64,    // Max metabolic rate
    km: f64       // Michaelis constant
) -> f64 {
    vmax * dose / (km + dose)
}
```

### Biomarker Integration

Tie dosing to measurable biomarkers:

```sounio
fn dose_by_biomarker(
    liver_function: f64,      // 0 = severe impairment, 1 = normal
    renal_function: f64,      // eGFR
    baseline_dose: f64
) -> f64 {
    // Reduce dose if organ function impaired
    baseline_dose * liver_function * (renal_function / 100.0)
}
```

---

## Performance

### JIT Runtime

First execution: 1-2 seconds (compilation)
Subsequent: <100 ms per run

### Optimization Tips

1. **Batch calculations**: Run multiple doses in one program
2. **Native compilation**: Use `souc run --native` for production
3. **Vectorize**: Use arrays instead of individual variables
4. **Profile**: Use `%time` in Jupyter to find bottlenecks

---

## Testing & Validation

### Test File (tests/test_pipeline.sio)

```sounio
fn test_lipinski_aspirin() {
    let aspirin = Molecule { /* ... */ }
    let result = screening_check(aspirin)
    assert result.passed == true
}

fn test_auc_calculation() {
    let dose = measure(500.0, 2.5)
    let cl = measure(120.0, 6.0)
    let auc = calculate_auc(dose, cl)

    // Check value
    assert auc.value > 4.0
    assert auc.value < 5.0

    // Check uncertainty propagation
    assert auc.epsilon > 0.2
}
```

### Run Tests

```bash
souc run tests/test_pipeline.sio
```

---

## Examples

### Example 1: Screen a Batch of Molecules

```sounio
let molecules = [
    // Aspirin
    Molecule { smiles: "CC(=O)Oc1ccccc1C(=O)O", molecular_weight: 180.16, log_p: 1.19, hbd: 2, hba: 3 },
    // Ibuprofen
    Molecule { smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O", molecular_weight: 206.28, log_p: 3.97, hbd: 1, hba: 2 },
    // Naproxen
    Molecule { smiles: "COc1ccc2cc(ccc2c1)C(C)C(=O)O", molecular_weight: 230.26, log_p: 3.18, hbd: 1, hba: 3 }
]

for mol in molecules {
    let result = screening_check(mol)
    print(mol.smiles)
    print(result.passed)
}
```

### Example 2: Personalized Dosing

```sounio
fn personalize_dose(
    patient: PatientData,
    standard_dose: f64
) -> f64 {
    // Weight-based scaling
    let weight_adjusted = standard_dose * (patient.weight_kg / 70.0)

    // Age adjustment (reduce for elderly)
    let age_factor = if patient.age > 65 { 0.8 } else { 1.0 }

    // Renal function (reduce if CL < 30)
    let renal_factor = if patient.creatinine_clearance < 30.0 { 0.5 } else { 1.0 }

    return weight_adjusted * age_factor * renal_factor
}

// Usage
let patient = PatientData { /* ... */ }
let personalized = personalize_dose(patient, 500.0)
print(personalized)
```

---

## Troubleshooting

### "SMILES parsing failed"

The SMILES string is malformed. Verify using [SMILES validator](https://smiles.deaya.com/).

### "Measurement uncertainty too high"

If ε > 50% of value, measurement is questionable. Consider:
- More precise assay
- Larger sample size
- Different measurement method

### "Convergence failed"

Optimization didn't find a dose. Try:
- Wider dose range
- Adjust thresholds (efficacy, toxicity)
- Check PK parameters are reasonable

---

## References

- **Lipinski's Rule of Five** — Lipinski et al., Adv. Drug Deliv. Rev. 1997
- **GUM** — JCGM 100:2008, Guide to the Expression of Uncertainty in Measurement
- **PK Compartment Models** — Rowland & Tozer, "Clinical Pharmacokinetics"
- **Clinical Trial Simulation** — Ette & Williams, "Pharmacometrics"

---

## Next Steps

- [**API Reference**](reference.md) — Detailed function/type reference
- [**sounio-py Integration**](../sounio-py/quickstart.md) — Call pipeline from Python
- [**Jupyter Notebooks**](../sounio-jupyter/usage.md) — Interactive pipeline development
- [**Sounio Language**](https://github.com/sounio-org/sounio/docs/LLM_PROGRAMMING_GUIDE.md) — Learn Sounio

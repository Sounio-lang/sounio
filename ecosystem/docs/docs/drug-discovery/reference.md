# Drug Discovery Pipeline API Reference

Complete reference for types and functions in the drug-discovery pipeline.

---

## Types

### Molecule

Represents a chemical compound with drug-like properties.

```sounio
struct Molecule {
    smiles: string,           // SMILES notation (e.g., "CC(=O)Oc1ccccc1C(=O)O")
    name: i64,                // Molecule ID
    molecular_weight: f64,    // Daltons
    log_p: f64,               // Octanol-water partition coefficient
    hbd: i64,                 // Number of H-bond donors
    hba: i64                  // Number of H-bond acceptors
}
```

**Fields:**

| Field | Type | Unit | Range | Notes |
|-------|------|------|-------|-------|
| smiles | string | - | - | SMILES notation; parsed for properties |
| name | i64 | - | - | Unique identifier |
| molecular_weight | f64 | Da | 0-1000 | Daltons (atomic mass units) |
| log_p | f64 | - | -5 to 10 | Lipophilicity; higher = more lipophilic |
| hbd | i64 | - | 0-10+ | H-bond donors (N-H, O-H) |
| hba | i64 | - | 0-20+ | H-bond acceptors (N, O) |

**Lipinski's Rule of Five:**

- MW < 500 Da
- LogP < 5
- HBD ≤ 5
- HBA ≤ 10

**Example:**

```sounio
let aspirin: Molecule = Molecule {
    smiles: "CC(=O)Oc1ccccc1C(=O)O",
    name: 1,
    molecular_weight: 180.16,
    log_p: 1.19,
    hbd: 2,
    hba: 3
}
```

---

### PatientData

Represents a single patient's demographics and organ function.

```sounio
struct PatientData {
    id: i64,                      // Patient ID
    weight_kg: f64,               // Body weight in kg
    age: i32,                     // Age in years
    creatinine_clearance: f64     // Renal function (mL/min)
}
```

**Fields:**

| Field | Type | Unit | Typical Range | Notes |
|-------|------|------|---|---|
| id | i64 | - | 1-10000 | Unique identifier |
| weight_kg | f64 | kg | 40-150 | Body weight |
| age | i32 | years | 18-100+ | Patient age |
| creatinine_clearance | f64 | mL/min | 0-200 | eGFR; >90=normal, <30=severe CKD |

**Renal Function Categories:**

- **>90 mL/min**: Normal kidney function
- **60-89 mL/min**: Mild reduction
- **30-59 mL/min**: Moderate reduction (CKD stage 3)
- **<30 mL/min**: Severe reduction (CKD stage 4-5)

**Example:**

```sounio
let patient: PatientData = PatientData {
    id: 42,
    weight_kg: 70.0,
    age: 45,
    creatinine_clearance: 90.0
}
```

---

### PKParameters

Pharmacokinetic parameters describing drug disposition.

```sounio
struct PKParameters {
    ka: Knowledge<f64>,          // Absorption rate constant
    kel: Knowledge<f64>,         // Elimination rate constant
    volume: Knowledge<f64>,      // Central compartment volume
    clearance: Knowledge<f64>    // Total body clearance
}
```

**Fields:**

| Field | Type | Unit | Typical Range | Notes |
|-------|------|------|---|---|
| ka | Knowledge | 1/hr | 0.1-2.0 | First-order absorption; higher=faster |
| kel | Knowledge | 1/hr | 0.01-1.0 | Elimination rate; higher=faster clearance |
| volume | Knowledge | L | 5-200 | Central volume of distribution |
| clearance | Knowledge | mL/min | 10-500 | Total body clearance (CL) |

**Relationships:**

- **kel = 0.693 / t_half** (elimination rate from half-life)
- **t_half = 0.693 / kel** (half-life from elimination rate)
- **CL = kel × Vd** (clearance from rate and volume)

**GUM Uncertainty:**

Each field is epistemic — includes standard uncertainty (CV typical: 5-30%).

**Example:**

```sounio
let pk: PKParameters = PKParameters {
    ka: measure(0.5, uncertainty: 0.05),
    kel: measure(0.12, uncertainty: 0.012),
    volume: measure(50.0, uncertainty: 2.5),
    clearance: measure(120.0, uncertainty: 6.0)
}
```

---

### ScreeningResult

Output of virtual screening for a molecule.

```sounio
struct ScreeningResult {
    mol_id: i64,              // Molecule ID
    passed: bool,             // Lipinski compliant?
    confidence: f64,          // 0-1; 1=all rules pass, 0=max violations
    violations: [string]      // List of failed rules
}
```

**Fields:**

| Field | Type | Description |
|-------|------|---|
| mol_id | i64 | Molecule identifier |
| passed | bool | True if ≤1 violation of Lipinski rules |
| confidence | f64 | (5 - num_violations) / 5; higher=more compliant |
| violations | [string] | List like `["MW >= 500", "LogP >= 5"]` |

**Interpretation:**

- **passed=true, confidence=1.0**: Ideal, all rules pass
- **passed=true, confidence=0.8**: 1 rule violated but borderline
- **passed=false, confidence<0.8**: 2+ violations; likely poor absorption/bioavailability

**Example:**

```sounio
let result: ScreeningResult = ScreeningResult {
    mol_id: 1,
    passed: true,
    confidence: 1.0,
    violations: []
}
```

---

### SimulationResult

Predicted clinical outcome for a patient receiving a dose.

```sounio
struct SimulationResult {
    patient_id: i64,                    // Patient ID
    dose_mg: f64,                       // Administered dose
    predicted_auc: Knowledge<f64>,      // Area under concentration curve
    predicted_cmax: Knowledge<f64>,     // Peak concentration
    efficacy_prob: f64,                 // P(achieving therapeutic effect)
    toxicity_risk: f64,                 // P(adverse event)
    recommended_dose: Knowledge<f64>    // Personalized recommendation
}
```

**Fields:**

| Field | Type | Unit | Interpretation |
|-------|------|------|---|
| patient_id | i64 | - | Patient identifier |
| dose_mg | f64 | mg | Input dose |
| predicted_auc | Knowledge | mg·hr/mL | ±uncertainty; exposure metric |
| predicted_cmax | Knowledge | mg/mL | ±uncertainty; peak concentration |
| efficacy_prob | f64 | 0-1 | P(effect); >0.7 = likely efficacious |
| toxicity_risk | f64 | 0-1 | P(harm); <0.1 = acceptable |
| recommended_dose | Knowledge | mg | ±uncertainty; optimized dose |

**Decision Rule:**

```
Recommend dose if:
  efficacy_prob >= 0.70 AND
  toxicity_risk <= 0.10 AND
  confidence >= 0.95
```

**Example:**

```sounio
let result: SimulationResult = SimulationResult {
    patient_id: 42,
    dose_mg: 500.0,
    predicted_auc: measure(2500.0, uncertainty: 125.0),
    predicted_cmax: measure(8.5, uncertainty: 0.4),
    efficacy_prob: 0.85,
    toxicity_risk: 0.08,
    recommended_dose: measure(500.0, uncertainty: 25.0)
}
```

---

### PipelineResult

Complete output from full drug discovery pipeline.

```sounio
struct PipelineResult {
    screening_results: [ScreeningResult],
    pk_models: [PKParameters],
    trial_simulations: [SimulationResult]
}
```

**Fields:**

| Field | Type | Description |
|-------|------|---|
| screening_results | [ScreeningResult] | Per-molecule Lipinski assessment |
| pk_models | [PKParameters] | One per passed molecule; population PK |
| trial_simulations | [SimulationResult] | One per (patient, dose) combination |

**Example:**

```sounio
let final_result: PipelineResult = PipelineResult {
    screening_results: [aspirin_result, ibuprofen_result],
    pk_models: [aspirin_pk, ibuprofen_pk],
    trial_simulations: [sim_p1_aspirin, sim_p2_aspirin, sim_p1_ibuprofen, ...]
}
```

---

## Functions

### Screening Functions

#### apply_lipinski(mol: Molecule) -> ScreeningResult

**Purpose:** Evaluate Lipinski's Rule of Five compliance.

**Parameters:**

- **mol** (Molecule) — Molecule to evaluate

**Returns:** ScreeningResult with pass/fail and violations

**Algorithm:**

1. Check MW < 500 (violation if ≥500)
2. Check LogP < 5 (violation if ≥5)
3. Check HBD ≤ 5 (violation if >5)
4. Check HBA ≤ 10 (violation if >10)
5. Pass if ≤1 violation
6. Confidence = (5 - num_violations) / 5

**Example:**

```sounio
let mol = Molecule { /* ... */ }
let result = apply_lipinski(mol)
if result.passed {
    print("Passed Lipinski!")
}
```

---

### PK/PD Functions

#### calculate_auc(dose: Knowledge<f64>, clearance: Knowledge<f64>) -> Knowledge<f64>

**Purpose:** Calculate area under the concentration-time curve.

**Parameters:**

- **dose** (Knowledge<mg>) — Administered dose
- **clearance** (Knowledge<mL/min>) — Total body clearance

**Returns:** AUC with GUM-propagated uncertainty

**Equation:**

```
AUC = dose / clearance
```

**Uncertainty Propagation:**

Relative uncertainty combines in quadrature:

```
(ε_auc / AUC)² = (ε_dose / dose)² + (ε_cl / cl)²
```

**Example:**

```sounio
let dose = measure(500.0, uncertainty: 2.5)       // ±0.5%
let cl = measure(120.0, uncertainty: 6.0)         // ±5%
let auc = calculate_auc(dose, cl)
// Result: AUC = 4.17 ± 0.24 (±5.7% relative)
```

---

#### calculate_cmax(dose: Knowledge<f64>, volume: Knowledge<f64>) -> Knowledge<f64>

**Purpose:** Calculate peak concentration (Cmax).

**Parameters:**

- **dose** (Knowledge<mg>) — Dose
- **volume** (Knowledge<L>) — Volume of distribution

**Returns:** Cmax with GUM-propagated uncertainty

**Equation:**

```
Cmax = dose / volume
```

**Example:**

```sounio
let dose = measure(500.0, uncertainty: 2.5)
let vd = measure(50.0, uncertainty: 2.5)
let cmax = calculate_cmax(dose, vd)
// Result: Cmax = 10.0 ± 0.4 mg/L
```

---

#### calculate_kel(half_life: f64) -> f64

**Purpose:** Convert half-life to elimination rate constant.

**Parameters:**

- **half_life** (f64) — Half-life in hours

**Returns:** Elimination rate constant (kel) in 1/hr

**Equation:**

```
kel = 0.693 / t_half
```

**Example:**

```sounio
let kel = calculate_kel(5.0)  // t_half = 5 hours
// Result: kel = 0.1386 /hr
```

---

#### concentration_at_time(
    dose: Knowledge<f64>,
    kel: Knowledge<f64>,
    t: f64
) -> Knowledge<f64>

**Purpose:** Predict concentration at a specific time post-dose.

**Parameters:**

- **dose** (Knowledge<mg>) — Initial dose
- **kel** (Knowledge<1/hr>) — Elimination rate constant
- **t** (f64) — Time in hours

**Returns:** Concentration at time t with uncertainty

**Equation (first-order kinetics):**

```
C(t) = (dose / Vd) × exp(-kel × t)
```

**Note:** Vd is absorbed into dose normalization here.

**Example:**

```sounio
let dose = measure(500.0, uncertainty: 2.5)
let kel = measure(0.12, uncertainty: 0.012)

let c_at_4h = concentration_at_time(dose, kel, 4.0)
let c_at_8h = concentration_at_time(dose, kel, 8.0)

print(c_at_4h)  // Concentration at 4 hours
print(c_at_8h)  // Concentration at 8 hours
```

---

### Prediction Functions

#### predict_efficacy(auc: Knowledge<f64>, threshold: f64) -> f64

**Purpose:** Estimate probability of achieving therapeutic effect.

**Parameters:**

- **auc** (Knowledge<f64>) — Predicted AUC
- **threshold** (f64) — Threshold for therapeutic effect

**Returns:** Probability 0-1

**Algorithm:**

Assumes normal distribution; computes P(X > threshold) where X ~ N(auc.value, auc.epsilon²).

```
distance = (auc.value - threshold) / auc.epsilon

if distance > 2.0:
    return 0.95       // High confidence
elif distance > 0.0:
    return 0.5 + (distance / 4.0)  // Linear interpolation
else:
    return 0.5 - ((-distance) / 4.0)  // Below threshold
```

**Interpretation:**

- **>0.8**: Likely efficacious
- **0.5-0.8**: Borderline
- **<0.5**: Insufficient

**Example:**

```sounio
let auc = measure(2500.0, uncertainty: 125.0)
let threshold = 2000.0
let p_eff = predict_efficacy(auc, threshold)
// Result: ~0.85 (85% probability of efficacy)
```

---

#### predict_toxicity(cmax: Knowledge<f64>, threshold: f64) -> f64

**Purpose:** Estimate probability of exceeding toxicity threshold.

**Parameters:**

- **cmax** (Knowledge<f64>) — Predicted peak concentration
- **threshold** (f64) — Toxicity threshold

**Returns:** Probability 0-1 (lower is safer)

**Algorithm:**

Same as predict_efficacy, but lower distance is safer.

```
distance = (cmax.value - threshold) / cmax.epsilon

if distance > 2.0:
    return 0.05       // Very safe
elif distance > 0.0:
    return 0.5 - (distance / 4.0)
else:
    return 0.5 + ((-distance) / 4.0)
```

**Interpretation:**

- **<0.05**: Very safe
- **0.05-0.15**: Acceptable
- **>0.15**: Concerning

**Example:**

```sounio
let cmax = measure(8.5, uncertainty: 0.4)
let tox_threshold = 10.0
let p_tox = predict_toxicity(cmax, tox_threshold)
// Result: ~0.08 (8% risk; acceptable)
```

---

### Optimization Functions

#### optimize_dose(
    pk_params: PKParameters,
    efficacy_threshold: f64,
    toxicity_threshold: f64,
    dose_range: [f64; 2]
) -> Knowledge<f64>

**Purpose:** Find optimal dose maximizing benefit = P(efficacy) - P(toxicity).

**Parameters:**

- **pk_params** (PKParameters) — Patient PK parameters
- **efficacy_threshold** (f64) — Therapeutic target (AUC or Cmax)
- **toxicity_threshold** (f64) — Safety limit
- **dose_range** ([f64; 2]) — [min_dose, max_dose]

**Returns:** Recommended dose with uncertainty

**Algorithm:**

1. Grid search from min_dose to max_dose (step: 25 mg)
2. For each dose:
   - Calculate AUC, Cmax
   - Calculate P(efficacy), P(toxicity)
   - Calculate benefit = P_eff - P_tox
3. Return dose with maximum benefit

**Example:**

```sounio
let best_dose = optimize_dose(
    pk_params,
    efficacy_threshold: 2000.0,
    toxicity_threshold: 10.0,
    dose_range: [250.0, 750.0]
)
print(best_dose)  // Optimized dose with uncertainty
```

---

#### personalize_dose(patient: PatientData, standard_dose: f64) -> f64

**Purpose:** Scale standard dose based on patient demographics and organ function.

**Parameters:**

- **patient** (PatientData) — Patient demographics
- **standard_dose** (f64) — Population standard dose (mg)

**Returns:** Personalized dose (mg)

**Adjustments:**

1. **Weight-based**: `dose × (patient_weight / 70)`
2. **Age**: Reduce by 20% if age > 65
3. **Renal function**: Reduce by 50% if CrCl < 30

**Example:**

```sounio
let patient = PatientData { weight_kg: 85.0, age: 72, creatinine_clearance: 40.0 }
let personalized = personalize_dose(patient, 500.0)
// Result: 500 × (85/70) × 0.8 × 0.9 ≈ 434 mg
```

---

## Constants

### Lipinski Parameters

```sounio
let LIPINSKI_MW_LIMIT = 500.0        // Da
let LIPINSKI_LOGP_LIMIT = 5.0        // dimensionless
let LIPINSKI_HBD_LIMIT = 5           // count
let LIPINSKI_HBA_LIMIT = 10          // count
let LIPINSKI_VIOLATIONS_THRESHOLD = 1 // pass if ≤1 violation
```

### Pharmacokinetic Constants

```sounio
let ELIMINATION_FRACTION = 0.693     // ln(2) for half-life conversion
let DEFAULT_BODY_WEIGHT = 70.0       // kg (standard adult)
let DEFAULT_AGE = 45                 // years
```

### Clinical Thresholds

```sounio
let EFFICACY_THRESHOLD_DEFAULT = 2000.0   // AUC mg·hr/mL
let TOXICITY_THRESHOLD_DEFAULT = 10.0     // Cmax mg/L
let MIN_EFFICACY_PROB = 0.70              // Required efficacy probability
let MAX_TOXICITY_RISK = 0.10              // Acceptable toxicity risk
let MIN_CONFIDENCE_INTERVAL = 0.95        // 95% CI
```

---

## Error Handling

### Validation

Functions validate inputs and return sensible defaults on error:

```sounio
// If dose < 0, return 0
let auc = calculate_auc(measure(-100.0, 5.0), measure(120.0, 6.0))
// Result: AUC = 0 ± 5

// If threshold negative, clamp to 0
let p_eff = predict_efficacy(measure(100.0, 10.0), -50.0)
// Result: 0.99+ (almost certain to exceed negative threshold)
```

### Division by Zero

```sounio
// If clearance = 0, AUC approaches infinity
let auc = calculate_auc(measure(500.0, 2.5), measure(0.0, 0.1))
// Sounio: Div effect; runtime check prevents actual division
```

### Out-of-Range Probability

Probabilities clamped to [0, 1]:

```sounio
let p = predict_efficacy(measure(5000.0, 10.0), 1000.0)
// Always returns value in [0, 1] regardless of distance
```

---

## Examples

### Complete Screen-PK-Trial Workflow

```sounio
// 1. Screen molecules
let aspirin = Molecule { /* ... */ }
let screen_result = apply_lipinski(aspirin)

if not screen_result.passed {
    print("Failed screening")
    return
}

// 2. Assign PK parameters
let pk = PKParameters {
    ka: measure(0.5, 0.05),
    kel: measure(0.12, 0.012),
    volume: measure(50.0, 2.5),
    clearance: measure(120.0, 6.0)
}

// 3. Run trial simulation
let patient = PatientData {
    id: 1,
    weight_kg: 70.0,
    age: 45,
    creatinine_clearance: 90.0
}

let best_dose = optimize_dose(pk, 2000.0, 10.0, [250.0, 750.0])

let dose_ep = measure(best_dose.value, uncertainty: best_dose.epsilon)
let auc = calculate_auc(dose_ep, pk.clearance)
let cmax = calculate_cmax(dose_ep, pk.volume)

let p_eff = predict_efficacy(auc, 2000.0)
let p_tox = predict_toxicity(cmax, 10.0)

let result = SimulationResult {
    patient_id: 1,
    dose_mg: best_dose.value,
    predicted_auc: auc,
    predicted_cmax: cmax,
    efficacy_prob: p_eff,
    toxicity_risk: p_tox,
    recommended_dose: best_dose
}

print(result.recommended_dose)
```

---

## References

- **Lipinski, C. A., et al.** (1997). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." *Advanced Drug Delivery Reviews*, 23(1), 3-25.
- **Rowland, M., & Tozer, T. N.** (2010). *Clinical Pharmacokinetics and Pharmacodynamics*. Lippincott Williams & Wilkins.
- **JCGM** (2008). "Guide to the Expression of Uncertainty in Measurement (GUM)." ISO/IEC GUIDE 98-3.

---

For more examples and tutorials, see the [Tutorial](tutorial.md).

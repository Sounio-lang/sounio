# sounio-py Quickstart

Get started with epistemic computing in Python in 5 minutes.

## Installation

```bash
pip install sounio
```

If you want NumPy and pandas support:

```bash
pip install sounio[numpy,pandas]
```

Verify installation:

```bash
python -c "import sounio; print(sounio.__version__)"
```

---

## 1. Your First Measurement

The `Knowledge` class represents a measured value with uncertainty:

```python
import sounio

# Create a measurement: value ± uncertainty (1σ)
temperature = sounio.Knowledge(
    value=36.5,
    epsilon=0.1,  # Standard uncertainty (1σ) in °C
    provenance="calibrated thermometer"
)

print(temperature)
# Output: Knowledge(36.500 ± 0.100, prov='calibrated thermometer')

# Access components
print(f"Value: {temperature.value}°C")
print(f"Uncertainty: ±{temperature.epsilon}°C")
print(f"Source: {temperature.provenance}")
```

**Key concepts:**

- **value** — your measured number
- **epsilon** — standard uncertainty (GUM, k=1, one sigma)
- **provenance** — where the measurement came from (for traceability)

---

## 2. Uncertainty Propagation (Automatic)

When you do math with `Knowledge` values, uncertainty propagates automatically via proven rules:

### Addition & Subtraction

Uncertainties add in quadrature:

```python
import sounio

# Two lab measurements
glucose_morning = sounio.Knowledge(95.0, 3.0, "finger stick")
glucose_evening = sounio.Knowledge(110.0, 3.0, "finger stick")

# Add them
total_glucose = glucose_morning + glucose_evening
print(total_glucose)
# Output: Knowledge(205.000 ± 4.243, prov='(finger stick)+(finger stick)')
# ε = √(3² + 3²) = 4.243
```

### Multiplication & Division

Relative uncertainties add in quadrature:

```python
# PK calculation: Clearance = Dose / AUC
dose = sounio.Knowledge(500.0, 10.0, "dispensed")      # ±2%
auc = sounio.Knowledge(10000.0, 500.0, "integration")  # ±5%

clearance = dose / auc  # Relative: √(0.02² + 0.05²)
print(clearance)
# Output: Knowledge(0.0500 ± 0.0028, prov='...')
```

### Scalar Operations

Multiplying by a constant scales the uncertainty:

```python
dose_mg = sounio.Knowledge(500.0, 2.5, "pump")
dose_ug = dose_mg * 1000  # Convert to micrograms

print(dose_ug)
# Output: Knowledge(500000.000 ± 2500.000, prov='pump')
# ε scales by 1000
```

---

## 3. Working with Multiple Values

Create several measurements and combine them:

```python
import sounio

# Patient pharmacokinetics
clearance = sounio.Knowledge(120.0, 6.0, "LC-MS CL")      # mL/min
half_life = sounio.Knowledge(5.0, 0.25, "terminal phase")  # hours

# Calculate elimination rate constant
# kel = 0.693 / t_half
kel = 0.693 / half_life
print(f"k_el = {kel.value:.4f} /hr ± {kel.epsilon:.4f}")

# Verify with clearance relationship
volume = clearance / kel
print(f"V_d = {volume.value:.1f} L ± {volume.epsilon:.1f}")
```

---

## 4. NumPy Integration (UncertainArray)

Work with arrays of uncertain measurements:

```python
import sounio
import numpy as np

# Create an array of uncertain dose measurements
dose_values = np.array([95.0, 105.0, 100.0])
dose_errors = np.array([2.0, 2.5, 2.0])  # Individual uncertainties

doses = sounio.UncertainArray(dose_values, dose_errors)

# Element-wise operations
scaled_doses = doses * 1.1
print(f"Scaled doses: {scaled_doses.values}")
print(f"Scaled errors: {scaled_doses.uncertainties}")

# Aggregation with propagated uncertainty
mean_dose = doses.mean()
print(f"Mean dose: {mean_dose.value:.2f} ± {mean_dose.epsilon:.2f} mg")
```

---

## 5. Pandas Integration (EpistemicDataFrame)

Analyze tabular data with measurement uncertainty:

```python
import sounio
import pandas as pd

# Create a clinical dataset with uncertainties
data = {
    'patient_id': [1, 2, 3, 4, 5],
    'dose_mg': [100.0, 150.0, 125.0, 110.0, 140.0],
    'dose_err': [2.0, 3.0, 2.5, 2.0, 3.0],
    'clearance_mL_min': [120.0, 95.0, 110.0, 130.0, 100.0],
    'clearance_err': [6.0, 5.0, 5.5, 6.5, 5.0],
}

df = pd.DataFrame(data)

# Convert to epistemic dataframe
epistemic_df = sounio.EpistemicDataFrame(df)

# Create derived epistemic column
epistemic_df['auc'] = (epistemic_df['dose_mg'] /
                       epistemic_df['clearance_mL_min'])

# Get summary statistics with uncertainty
print(epistemic_df['auc'].epistemic_summary())
# Shows: mean, std, uncertainty propagation
```

---

## 6. Running Sounio Code

For performance-critical calculations, call native Sounio code from Python:

```python
import sounio

# Type-check a Sounio program
result = sounio.check_file('my_model.sio')
if result.ok:
    print("✓ Compilation successful")
else:
    print("✗ Compilation failed:")
    print(result.error)

# Run inline Sounio code
code = '''
fn exponential_clearance(dose: f64, kel: f64, t: f64) -> f64 {
    dose * pipe_exp(0.0 - kel * t)
}

fn main() with IO {
    let result = exponential_clearance(500.0, 0.12, 24.0)
    print_f64(result)
}
'''

result = sounio.run_code(code)
print(f"Output: {result.stdout}")
print(f"Exit code: {result.exit_code}")
```

---

## 7. Domain Types

Use pre-built types for pharmacokinetics, screening, and simulation:

### Molecule Type

```python
import sounio

aspirin = sounio.Molecule(
    name="Acetylsalicylic acid",
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    molecular_weight=180.16,
    log_p=1.19,
    hbd=2,  # H-bond donors
    hba=3   # H-bond acceptors
)

# Check Lipinski's Rule of Five
if aspirin.molecular_weight < 500 and aspirin.log_p < 5:
    print("✓ Aspirin passes Lipinski filter")
```

### PKParameters Type

```python
import sounio

parameters = sounio.PKParameters(
    clearance=120.0,  # mL/min
    volume_of_distribution=50.0,  # L
    absorption_rate=0.5,  # 1/hr
    half_life=5.0  # hours
)

print(f"t_1/2 = {parameters.half_life} hours")
print(f"CL = {parameters.clearance} mL/min")
```

### SimulationResult Type

```python
import sounio

result = sounio.SimulationResult(
    patient_id=42,
    simulated_pk_profile=[100.0, 85.0, 72.0, 61.0],
    simulated_times=[0.0, 4.0, 8.0, 12.0],
    efficacy_probability=0.87,
    toxicity_risk=0.05
)

print(f"Efficacy: {result.efficacy_probability:.1%}")
print(f"Safety risk: {result.toxicity_risk:.1%}")
```

---

## 8. Ontology Integration

Map clinical terms to standardized ontologies:

```python
import sounio

# Resolve a clinical term
term = sounio.ontology.resolve("eGFR 45-59")
print(f"SNOMED: {term.snomed_ct}")
print(f"ICD-10: {term.icd10_cm}")

# Normalize units
normalized = sounio.clinical_normalize("Creatinine 0.9 mg/dL")
print(f"μmol/L: {normalized.value_si}")

# Check relationships
is_ckd = sounio.ontology.is_subclass("CKD stage 3a", "Chronic Kidney Disease")
print(f"Is CKD? {is_ckd}")

# Search for terms
results = sounio.ontology.search("glucose", limit=5)
for term in results:
    print(f"  - {term.preferred_label}")
```

---

## 9. Error Handling

Handle exceptions gracefully:

```python
import sounio

try:
    # Invalid uncertainty (negative)
    bad_measurement = sounio.Knowledge(100.0, -5.0)
except ValueError as e:
    print(f"✗ Invalid Knowledge: {e}")

try:
    # Division by zero
    zero = sounio.Knowledge(0.0, 0.1)
    result = 100.0 / zero
except ZeroDivisionError as e:
    print(f"✗ Division error: {e}")

# Check Sounio compilation errors
result = sounio.check_file('bad_syntax.sio')
if not result.ok:
    print(f"✗ Sounio type error:")
    print(result.error)
```

---

## 10. Complete Example: Drug Dosing Calculation

Put it all together in a realistic scenario:

```python
import sounio

# Patient data with measurement uncertainties
patient_weight = sounio.Knowledge(70.0, 0.5, "weighed on calibrated scale")
creatinine = sounio.Knowledge(1.0, 0.05, "lab assay")

# Drug parameters (from pharmacokinetic study)
clearance_per_kg = sounio.Knowledge(
    1.5,  # mL/min/kg
    0.1,  # CV ~6.7%
    "population PK model"
)
target_concentration = sounio.Knowledge(
    15.0,  # μg/mL
    1.5,   # ±10% therapeutic window
    "efficacy threshold"
)

# Calculate personalized dose
patient_clearance = clearance_per_kg * patient_weight
dose = target_concentration * patient_clearance
dose_scaled = dose / 10.0  # Convert units

print(f"Patient CL: {patient_clearance.value:.1f} ± {patient_clearance.epsilon:.1f} mL/min")
print(f"Recommended dose: {dose_scaled.value:.1f} ± {dose_scaled.epsilon:.1f} mg")
print(f"Confidence (1σ): {dose_scaled.confidence:.1%}")

# Identify high-uncertainty cases
if dose_scaled.relative_uncertainty > 0.15:
    print("⚠ Warning: High uncertainty in dose calculation")
    print(f"  Relative error: {dose_scaled.relative_uncertainty:.1%}")
    print("  Consider additional renal function testing")
```

---

## Troubleshooting

### ImportError: No module named 'sounio'

```bash
pip install sounio
python -c "import sounio"
```

### ImportError: No module named 'numpy'

If you need NumPy arrays:

```bash
pip install sounio[numpy]
```

### "souc binary not found"

The `SounioExecutor` needs the Sounio compiler. Set the path:

```python
import sounio

executor = sounio.SounioExecutor(
    souc_path="/path/to/souc-linux-x86_64-jit"
)
result = executor.run_code("fn main() with IO { print(42) }")
```

Or set environment variable:

```bash
export SOUC=/path/to/souc-linux-x86_64-jit
python your_script.py
```

---

## Next Steps

- [**API Reference**](api.md) — Full documentation of all classes and methods
- [**Examples Repository**](https://github.com/sounio-org/sounio/tree/main/ecosystem/sounio-py/examples) — Real-world examples
- [**Sounio Language Guide**](https://github.com/sounio-org/sounio/docs/LLM_PROGRAMMING_GUIDE.md) — Learn the Sounio language
- [**Jupyter Notebooks**](../sounio-jupyter/usage.md) — Interactive Sounio development

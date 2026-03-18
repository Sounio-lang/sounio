# sounio-py: Python Bindings Overview

**sounio-py** brings Sounio's epistemic computing capabilities to Python, letting you work with measurement uncertainty as a first-class value in familiar pandas, NumPy, and scientific workflows.

## Core Idea

Traditional Python treats floats as exact:

```python
dose = 500.0  # What's the uncertainty? Unknown.
clearance = 120.0  # ±5% or ±50%? Unknown.
result = dose / clearance  # Uncertainty in result? Unknown.
```

With `sounio-py`, uncertainty is explicit:

```python
import sounio

dose = sounio.Knowledge(500.0, epsilon=2.5, provenance="calibrated pump")
clearance = sounio.Knowledge(120.0, epsilon=6.0, provenance="LC-MS assay")
result = dose / clearance  # Propagates uncertainty automatically
print(result)  # Knowledge(4.167 ± 0.068, prov='...')
```

## What's Included

### 1. Knowledge<T> Class

The core epistemic type — a measured value with uncertainty and provenance.

```python
# Create from measured value + uncertainty
temp = sounio.Knowledge(36.5, epsilon=0.1, provenance="thermometer")

# Arithmetic preserves uncertainty
celsius = temp  # 36.5 ± 0.1
fahrenheit = celsius * 9/5 + 32  # 97.7 ± 0.18

# Query properties
print(f"Value: {fahrenheit.value}")
print(f"Uncertainty (1σ): {fahrenheit.epsilon}")
print(f"Source: {fahrenheit.provenance}")
```

### 2. GUM-Compliant Propagation

All arithmetic follows the **Guide to the Expression of Uncertainty in Measurement**:

```python
a = sounio.Knowledge(100.0, 10.0, "sample A")  # 100 ± 10
b = sounio.Knowledge(200.0, 20.0, "sample B")  # 200 ± 20

# Addition/subtraction: ε_z = √(ε_a² + ε_b²)
sum_ab = a + b  # 300 ± 22.4

# Multiplication: ε_z/z = √((ε_a/a)² + (ε_b/b)²)
product = a * b  # 20000 ± 2828

# Division: same relative uncertainty rule
quotient = a / b  # 0.5 ± 0.0707
```

### 3. NumPy Integration (UncertainArray)

Work with arrays of uncertain values:

```python
import sounio
import numpy as np

# Create arrays of uncertain measurements
measurements = sounio.UncertainArray(
    values=[1.0, 2.0, 3.0, 4.0],
    uncertainties=[0.1, 0.15, 0.2, 0.1]
)

# NumPy operations work element-wise with uncertainty
result = measurements * 2.5  # Scales values and uncertainties
mean = measurements.mean()  # Knowledge with propagated uncertainty
```

### 4. Pandas Integration (EpistemicDataFrame)

Store and analyze data with measurement uncertainty:

```python
import sounio
import pandas as pd

# Create a DataFrame where columns are Knowledge values
df = sounio.EpistemicDataFrame({
    'dose_mg': [100.0, 200.0, 150.0],
    'dose_err': [2.0, 4.0, 3.0],
    'clearance_mL_min': [120.0, 150.0, 130.0],
    'clearance_err': [6.0, 8.0, 7.0],
})

# Derive a new epistemic column
df['kel'] = (0.693 / df['half_life']) + df['weight_uncertainty']

# Get summary with uncertainty
print(df.epistemic_summary())
```

### 5. Sounio Code Execution

Call Sounio code from Python for performance-critical computations:

```python
import sounio

# Type-check Sounio code
result = sounio.check_file('my_model.sio')
if result.ok:
    print("✓ Type check passed")

# Run Sounio code (JIT compiled)
result = sounio.run_code('''
fn quadratic(x: f64, a: f64, b: f64, c: f64) -> f64 {
    a * x * x + b * x + c
}

fn main() with IO {
    print_f64(quadratic(2.0, 1.0, 2.0, 3.0))  // 11.0
}
''')

print(result.stdout)  # "11.0"
print(result.exit_code)  # 0
```

### 6. Domain Types

Pre-defined types for scientific domains:

```python
import sounio

molecule = sounio.Molecule(
    name="Aspirin",
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    molecular_weight=180.16,
    log_p=1.19,
    hbd=2,  # H-bond donors
    hba=3   # H-bond acceptors
)

pk_params = sounio.PKParameters(
    clearance=120.0,  # mL/min
    volume_of_distribution=50.0,  # L
    absorption_rate=0.5,  # 1/hr
    half_life=5.0  # hours
)

patient = sounio.PatientData(
    weight_kg=70.0,
    age_years=45,
    creatinine_clearance=90.0,
    liver_function='normal'
)
```

### 7. Ontology Integration

Map domain terms to standardized ontologies:

```python
import sounio

# Resolve clinical terms
term = sounio.ontology.resolve("eGFR < 30")
print(term.snomed_ct)  # SNOMED CT code
print(term.umls)       # UMLS code

# Normalize units
sounio.clinical_normalize("Creatinine 0.9 mg/dL")
# → Creatinine: 79.6 μmol/L (standard unit)

# Check term relationships
is_ckd = sounio.is_subclass("CKD stage 3a", "Chronic Kidney Disease")
```

---

## Comparison: Pure Python vs. Sounio-py

| Task | Pure Python | With sounio-py |
|------|-------------|----------------|
| Create uncertain measurement | `{"value": 100.0, "error": 10.0}` (manual dict) | `Knowledge(100.0, 10.0)` (type-safe) |
| Add two measurements | `sqrt(a_err**2 + b_err**2)` (manual math) | `a + b` (automatic propagation) |
| Multiply measurements | `(a*b) * sqrt((a_err/a)**2 + (b_err/b)**2)` | `a * b` (automatic) |
| Work with arrays | NumPy (ignores uncertainty) | `UncertainArray` (element-wise uncertainty) |
| Scientific DataFrame | pandas (no uncertainty) | `EpistemicDataFrame` (uncertainty tracking) |
| Call high-performance code | ctypes / subprocess | `sounio.run_code()` (direct integration) |

---

## Architecture

```
┌────────────────────────────────────────┐
│         User Python Code               │
│  import sounio                         │
│  x = Knowledge(...)                    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│       sounio-py Python API             │
│  • Knowledge class                     │
│  • UncertainArray (NumPy ext)         │
│  • EpistemicDataFrame (pandas ext)    │
│  • SounioExecutor (souc bridge)       │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│   Native Extension (Rust, optional)    │
│  _sounio_native.pyd (Windows)          │
│  _sounio_native.so (Linux/macOS)       │
│  Falls back to pure Python if missing  │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│       Sounio Compiler (souc)           │
│  Type checking & JIT execution         │
└────────────────────────────────────────┘
```

---

## Installation

```bash
# From PyPI (recommended)
pip install sounio

# From source
git clone https://github.com/sounio-org/sounio.git
cd triple-sounio-ecosystem/sounio-py
pip install -e .
```

### Requirements

- Python 3.8+
- Sounio compiler binary (`souc-linux-x86_64-jit` or equivalent)
- Optional: NumPy, pandas (for integrations)

### Verify Installation

```bash
python -c "import sounio; print(f'sounio v{sounio.__version__}')"
python -c "import sounio; x = sounio.Knowledge(100, 5); print(x)"
```

---

## Next Steps

- [**Quickstart Guide**](quickstart.md) — Create your first epistemic computation
- [**API Reference**](api.md) — Full API docs for all classes
- [**Examples**](https://github.com/sounio-org/sounio/tree/main/triple-sounio-ecosystem/sounio-py/examples) — Runnable examples
- [**Sounio Language**](https://github.com/sounio-org/sounio/docs/LLM_PROGRAMMING_GUIDE.md) — Learn Sounio syntax

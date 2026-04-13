# sounio-py: Python Bindings for Epistemic Computing

Python library for the Sounio epistemic computing language, providing type-safe access to uncertainty quantification via GUM (Guide to the Expression of Uncertainty in Measurement) standard.

## Overview

**sounio-py** enables scientific Python workflows to leverage Sounio's native epistemic types and the full drug discovery pipeline. Instead of hand-crafting uncertainty propagation, declare measured values with uncertainty and let the library handle the mathematics.

```python
import sounio

# Measured temperature from a sensor
temp = sounio.Knowledge(36.5, epsilon=0.1, provenance="thermometer")
# Measured atmospheric pressure
pressure = sounio.Knowledge(1013.25, epsilon=2.5, provenance="barometer")

# Arithmetic automatically propagates uncertainty (GUM rules)
combined = temp + pressure
print(combined)  # Knowledge(1049.75 ± 2.54, prov='(thermometer)+(barometer)')
```

## Installation

### From PyPI (when available)
```bash
pip install sounio
```

### From Source (Development)
```bash
cd ecosystem/sounio-py
pip install -e .
```

### With Native Rust Extension (Faster)
```bash
pip install -e ".[native]"
```
Uses `maturin` to compile the Rust extension for 10x speedup on large computations.

### Verify Installation
```bash
python -c "import sounio; print(sounio.__version__); print('native' if sounio._NATIVE else 'pure python')"
```

## Quick Start

### 1. Epistemic Arithmetic
```python
import sounio

# GUM standard uncertainty propagation
x = sounio.Knowledge(100.0, epsilon=2.0, provenance="sensor_A")
y = sounio.Knowledge(50.0, epsilon=1.0, provenance="sensor_B")

result = x + y  # epsilon = sqrt(2.0^2 + 1.0^2)
result -= x     # subtraction also uses GUM rules
product = x * y  # multiplicative propagation: relative uncertainty

print(result)
print(product)
```

### 2. NumPy Integration
```python
import sounio
import numpy as np

# Wrap NumPy arrays in epistemic uncertainty
measurements = np.array([1.0, 2.0, 3.0, 4.0])
errors = np.array([0.1, 0.15, 0.2, 0.1])

arr = sounio.UncertainArray(measurements, errors)
mean = arr.mean()  # Returns Knowledge with propagated uncertainty
print(f"Mean: {mean}")

# Pandas integration
df = sounio.EpistemicDataFrame(...)  # Convert pandas → epistemic
df['dose_mg'].apply(lambda x: x * 0.5)  # Uncertainty propagates
```

### 3. Run Sounio Code from Python
```python
import sounio

# Check and run .sio files
result = sounio.run_file('examples/screening.sio')
print(result.stdout)
print(result.exit_code)

# Or inline code
code = """
fn main() with IO {
    let x = 42
    print_int(x)
}
"""
result = sounio.run_code(code)
```

### 4. Drug Discovery Pipeline
```python
import sounio

# High-level API wrapping the epistemic drug discovery pipeline
pipeline = sounio.DrugDiscoveryPipeline()

# Stage 1: Virtual screening (Lipinski filter)
screening = pipeline.screen_molecule(
    name="aspirin",
    mol_weight=180.16,
    logp=1.19,
    h_donors=1,
    h_acceptors=3
)
print(f"Screening result: {screening}")

# Stage 2: PK/PD modeling (one-compartment oral)
pk_params = pipeline.fit_pkpd(
    bioavailability=0.85,
    elimination_rate=0.15,
    volume_distribution=50.0
)

# Stage 3: Monte Carlo clinical trial simulation
trial = pipeline.simulate_trial(
    n_patients=1000,
    target_conc=5.0,
    toxic_threshold=20.0
)
print(f"Efficacy: {trial.efficacy_rate}")
print(f"Adverse rate: {trial.adverse_rate}")
```

## Core API

### `Knowledge[T]`
Represents a measured or computed value with standard uncertainty.

```python
class Knowledge:
    value: float           # Central estimate
    epsilon: float         # Standard uncertainty (1-sigma, GUM)
    provenance: str        # Label: source of measurement

    def __add__(self, other) -> Knowledge: ...   # GUM addition
    def __sub__(self, other) -> Knowledge: ...   # GUM subtraction
    def __mul__(self, other) -> Knowledge: ...   # GUM multiplication
    def __truediv__(self, other) -> Knowledge: ... # GUM division
```

### `SounioExecutor`
Run Sounio code with configurable timeouts and environment.

```python
executor = sounio.SounioExecutor(
    souc_path="./bin/souc",
    stdlib_path="./stdlib",
    timeout=30
)
result = executor.run_file("script.sio")
```

### `DrugDiscoveryPipeline`
Epistemic drug discovery in three stages.

```python
pipeline = sounio.DrugDiscoveryPipeline()

# Returns ScreeningResult (Knowledge)
screening = pipeline.screen_molecule(mw, logp, hbd, hba)

# Returns PKParameters (Knowledge values)
pk = pipeline.fit_pkpd(bioavail, ka, cl, vd)

# Returns TrialResult (Knowledge efficacy, adverse, TI)
trial = pipeline.simulate_trial(n_patients, mec, toxic)
```

## Features

- **GUM Uncertainty Propagation** — Automatic standard uncertainty calculation for +, −, ×, ÷
- **Provenance Tracking** — Know the source of every computed value
- **NumPy/Pandas Integration** — Work with arrays and dataframes while preserving uncertainty
- **Ontology Mapping** — CHEBI/DrugBank/PubChem lookups via `sounio.ontology.resolve()`
- **RDKit Cheminformatics** — Molecular property calculation with epistemic confidence
- **Clinical Data Normalization** — Standardize patient records via SNOMED/LOINC
- **Report Generation** — Automatic Markdown/PDF scientific reports with uncertainty
- **Native Compilation** — Rust extension (maturin) for 10x performance

## Ontology & Clinical Integration

```python
import sounio

# Look up drug by CHEBI identifier
aspirin = sounio.ontology.resolve("CHEBI:15365")
print(aspirin.inchikey)  # InChI key

# Normalize clinical term
normalized = sounio.ontology.clinical_normalize("HTN")
print(normalized)  # {SNOMED: "38341003", label: "Hypertension"}

# Map measurement unit
measurement = sounio.ontology.map_term("mg/dL", target="UCUM")
```

## Data Types

```python
from sounio import Molecule, PKParameters, PatientData, SimulationResult

# Molecule properties with epistemic confidence
mol = Molecule(
    name="ibuprofen",
    mol_weight=206.28,
    mw_uncertainty=0.5,
    logp=3.97,
    logp_uncertainty=0.1
)

# PK parameters
pk = PKParameters(
    bioavailability=0.8,
    elimination_rate=0.2,
    half_life=1.86
)

# Patient cohort
patients = PatientData.from_csv("patients.csv")

# Simulation output
result = SimulationResult(
    efficacy_rate=0.72,
    adverse_rate=0.05,
    therapeutic_index=4.0
)
```

## Plotting & Visualization

```python
import sounio

# Plot uncertainty as error bars
fig, ax = sounio.plot_uncertainties(
    values=[x1, x2, x3],
    labels=["Dose A", "Dose B", "Dose C"],
    yaxis="Efficacy",
    save_to="efficacy.png"
)

# Report generation
report = sounio.ReportBuilder(
    title="Drug XYZ Phase II Results",
    pipeline_results=trial_results
)
report.generate_pdf("report.pdf")
```

## Configuration

### Environment Variables
- `SOUC` — Path to souc binary (default: auto-detect)
- `SOUNIO_STDLIB_PATH` — Path to Sounio stdlib (default: auto-detect)

### Executor Options
```python
executor = sounio.SounioExecutor(
    souc_path="...",
    stdlib_path="...",
    timeout=60,
    jit_mode=True,  # Use JIT (faster for one-time runs)
)
```

## Examples

See `examples/` for:
- `basic_knowledge.py` — GUM arithmetic
- `drug_screening.py` — Lipinski filter with uncertainty
- `clinical_trial_sim.py` — Monte Carlo simulation
- `ontology_lookup.py` — CHEBI/DrugBank integration

## Testing

```bash
pytest tests/
pytest tests/ -v --cov=sounio
```

## Performance

| Operation | Pure Python | Native (Rust) |
|-----------|-------------|---------------|
| 1M additions | 320 ms | 18 ms |
| NumPy 100k array | 42 ms | 2 ms |

## Documentation

- [Sounio Language Reference](https://docs.sounio.dev)
- [Epistemic Computing Guide](https://docs.sounio.dev/guide/epistemic)
- [GUM Standard (BIPM)](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf)
- [API Reference](https://sounio-py.readthedocs.io)

## License

Apache-2.0. See LICENSE in the Sounio repository.

# Sounio Ecosystem Examples

Complete end-to-end demonstrations of the Triple Sounio Ecosystem: PubChem molecule data → Knowledge values → GUM-based uncertainty propagation → Report generation.

## Quick Start

### Run the Complete Demo

```bash
cd ecosystem

# With all optional dependencies
PYTHONPATH=sounio-py/python \
SOUC=../bin/souc \
SOUNIO_STDLIB_PATH=../stdlib \
python3 examples/complete_demo.py
```

### Without Pipeline (souc binary not required)

```bash
PYTHONPATH=sounio-py/python \
python3 examples/complete_demo.py
```

The demo gracefully skips the pipeline step if the souc binary is unavailable.

## What the Demo Does

### Step 1: Fetch Molecule from PubChem

Uses the offline cache to fetch molecule properties (aspirin by default):
- SMILES notation
- Molecular weight (with uncertainty: ±0.001 g/mol)
- LogP (with uncertainty: ±0.5)
- Hydrogen-bond donors/acceptors
- Lipinski Rule of Five evaluation

Example output:
```
✓ Molecule: aspirin
  SMILES: CC(=O)Oc1ccccc1C(=O)O
  Molecular Weight: 180.159 ± 0.001 g/mol
  LogP: 1.200 ± 0.500
  H-bond donors: 1, acceptors: 4
  Lipinski rule of five: PASS
```

### Step 2: Knowledge Arithmetic with Uncertainty

Demonstrates GUM (Guide to the Expression of Uncertainty in Measurement) standard:
- Creates Knowledge values with epsilon (uncertainty)
- Performs arithmetic: dose/weight → dose per kg
- Shows relative uncertainty propagation
- Checks reliability threshold (< 15%)

Example:
```
Dose: Knowledge(500.000 ± 10.000, prov='prescribed_dose_mg')
Weight: Knowledge(70.000 ± 0.500, prov='patient_weight_kg')
Dose/kg: Knowledge(7.143 ± 0.152, prov='(prescribed_dose_mg)/(patient_weight_kg)')
  Relative uncertainty: 2.12%
  Confidence: 97.88%
```

### Step 3: Drug Discovery Pipeline

Runs the full pipeline (if souc binary available):
- Screening confidence
- PK parameters (half-life, Tmax, Cmax, AUC)
- Trial outcomes (efficacy, adverse effects)
- Therapeutic index calculation

### Step 4: Report Generation

Generates a reproducible Markdown report with:
- Introduction section
- Knowledge value tables (value, uncertainty, provenance)
- Methodology section (GUM formula explanation)
- Sample output: `demo_report.md`

## GUM Uncertainty Propagation

The demo illustrates the Guide to the Expression of Uncertainty in Measurement rules:

| Operation | Propagation Rule |
|-----------|-----------------|
| Addition/Subtraction | ε = √(ε₁² + ε₂²) |
| Multiplication/Division | ε = \|result\| × √((ε₁/val₁)² + (ε₂/val₂)²) |
| Scalar multiplication | ε = \|factor\| × ε₁ |

## Offline PubChem Cache

The demo uses a bundled offline cache of 10 common drugs to allow tests and CI/CD to run without network access:

- **aspirin** — Acetylsalicylic acid
- **ibuprofen** — Anti-inflammatory
- **metformin** — Diabetes medication
- **paracetamol** — Pain reliever
- **caffeine** — Stimulant
- **penicillin** — Antibiotic
- **insulin** — Diabetes treatment
- **warfarin** — Anticoagulant
- **omeprazole** — Proton pump inhibitor
- **atorvastatin** — Statin

Each entry includes:
- SMILES string
- Molecular weight (with ±0.001 g/mol uncertainty)
- LogP (with ±0.5 uncertainty)
- H-bond donors/acceptors

## File Outputs

The demo generates:

1. **demo_report.md** — Markdown report with:
   - Molecule properties table
   - Dosing calculation table
   - GUM methodology explanation
   - Example execution metadata

The report is saved in the `examples/` directory and can be viewed with any Markdown viewer.

## Integration Tests

Run the expanded integration test suite:

```bash
cd ecosystem

PYTHONPATH=sounio-py/python python3 -m pytest tests/test_integration.py -v
```

Test coverage includes:
- PubChem integration (5 tests)
- Report generation (6 tests)
- Complete workflows (3 tests)
- Knowledge arithmetic (6 tests)
- Knowledge serialization (7 tests)

**All 38 tests pass.** No external dependencies required (souc binary is optional).

## API Usage

### Fetch a Molecule

```python
from sounio.integrations.pubchem import fetch_by_name

mol = fetch_by_name("aspirin", offline=True)
print(f"MW: {mol.molecular_weight.value} ± {mol.molecular_weight.epsilon}")
```

### Knowledge Arithmetic

```python
from sounio import Knowledge

dose = Knowledge(500.0, 10.0, "dose_mg")
weight = Knowledge(70.0, 0.5, "weight_kg")
dose_per_kg = dose / weight

print(f"Dose/kg: {dose_per_kg}")
print(f"Relative uncertainty: {dose_per_kg.relative_uncertainty:.2%}")
```

### Generate Reports

```python
from sounio.report import ReportBuilder

rb = ReportBuilder("My Report", author="Your Name")
rb.add_knowledge_table("Data", {"param": Knowledge(42.0, 0.5, "source")})
rb.save("output.md", format="markdown")
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PYTHONPATH` | Python path to sounio-py | (required) |
| `SOUC` | Path to the souc native wrapper | auto-detect |
| `SOUNIO_STDLIB_PATH` | Path to stdlib | auto-detect |

## Troubleshooting

### Pipeline Step Fails / Skipped

The pipeline requires the souc binary. The demo gracefully continues without it:
- Set `SOUC` environment variable to souc binary path
- Or install souc to a standard location
- Or skip by running demo without `SOUC` set

### ImportError: No module named 'sounio'

Make sure `PYTHONPATH` includes the sounio-py directory:
```bash
export PYTHONPATH="$(pwd)/sounio-py/python:$PYTHONPATH"
```

### pytest fails to find tests

Run pytest from the ecosystem directory with correct PYTHONPATH:
```bash
cd ecosystem
PYTHONPATH=sounio-py/python python3 -m pytest tests/test_integration.py
```

## References

- **GUM Standard**: ISO/IEC Guide 98-3 (Guide to the Expression of Uncertainty in Measurement)
- **PubChem**: https://pubchem.ncbi.nlm.nih.gov/
- **Lipinski Rule of Five**: Lipinski et al. (1997) Drug Discov Today
- **Sounio Epistemic Computing**: https://github.com/demetrios-k/sounio

## License

This demo is part of the Sounio project.

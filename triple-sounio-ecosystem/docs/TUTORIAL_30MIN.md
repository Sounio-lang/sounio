# 30-Minute Tutorial: Epistemic Drug Discovery with Sounio

By the end of this tutorial you will have:
- Measured values with uncertainty using `Knowledge[T]`
- Run a three-stage drug discovery pipeline in Python
- Visualized epistemic uncertainty in Jupyter
- Exported a reproducible research report

**Prerequisites:** Python ≥ 3.8, `pip install sounio sounio-jupyter`

---

## Part 1 — Epistemic Arithmetic (5 min)

Open a Python REPL or Jupyter notebook:

```python
import sounio

# A temperature measurement from a sensor (±0.1 °C)
temp = sounio.Knowledge(36.5, epsilon=0.1, provenance="thermometer")
print(temp)
# Knowledge(36.500 ± 0.100, prov='thermometer')

# Atmospheric pressure from a barometer (±2.5 hPa)
pressure = sounio.Knowledge(1013.25, epsilon=2.5, provenance="barometer")

# Addition automatically propagates uncertainty (GUM rule)
# ε_combined = √(ε_temp² + ε_pressure²)
combined = temp + pressure
print(combined)
# Knowledge(1049.750 ± 2.502, prov='(thermometer)+(barometer)')

# Multiplication also works
ratio = temp * pressure
print(f"relative uncertainty: {ratio.relative_uncertainty:.4f}")
print(f"confidence: {ratio.confidence:.3f}")
```

### Key insight
`Knowledge` is not a tuple — it's a first-class numeric type. Every arithmetic operation automatically computes the correct GUM uncertainty. No manual error propagation formulas.

---

## Part 2 — Provenance Tracking (5 min)

```python
from sounio import Knowledge

# Use .tracked() to enable full DAG provenance
mass = Knowledge.tracked(78.5, epsilon=1.5, provenance="scale", instrument="Mettler-XP")
height = Knowledge.tracked(1.75, epsilon=0.01, provenance="stadiometer")

# BMI calculation — provenance is tracked end-to-end
bmi = mass / (height * height)

# Inspect the provenance graph
print(bmi.chain.summary())
# chain: scale → stadiometer → (mul) → (div)  [3 ancestors]

# Export as JSON
import json
chain_dict = bmi.chain.to_dict()
print(json.dumps(chain_dict, indent=2)[:300])

# Export as Graphviz DOT (pipe to `dot -Tpng -o bmi_chain.png`)
print(bmi.chain.to_dot())

# Walk full ancestry
for node in bmi.ancestry():
    print(f"  {node.operation:8s}  {node.value:.3f} ± {node.epsilon:.3f}  [{node.label}]")
```

---

## Part 3 — Running Sounio Code from Python (5 min)

```python
import sounio

# Run inline Sounio code
result = sounio.run_code("""
    pub fn main() with Mut, Div, Panic {
        let mass:   f64 = 78.5
        let height: f64 = 1.75
        let bmi = mass / (height * height)
        assert(bmi > 20.0 && bmi < 30.0)
    }
""")

print("exit code:", result.exit_code)
print("stdout:", result.stdout)

# Knowledge values are parsed automatically
for kv in result.knowledge_values:
    print(f"  value={kv.value:.3f}  ε={kv.epsilon:.4f}  prov={kv.provenance}")
```

### Async execution (for concurrent pipelines)

```python
import asyncio, sounio

async def run_parallel():
    a, b = await asyncio.gather(
        sounio.async_run_code('pub fn main() with Panic { assert(1 + 2 == 3) }'),
        sounio.async_run_code('pub fn main() with Panic { assert(3 * 4 == 12) }'),
    )
    print("a exit:", a.exit_code, "  b exit:", b.exit_code)

asyncio.run(run_parallel())
```

---

## Part 4 — The Drug Discovery Pipeline (10 min)

### 4a. Define a molecule

```python
from sounio import Molecule, PKParameters, Knowledge

# Aspirin-like compound with measured properties (± uncertainty)
aspirin = Molecule(
    name="ASA-7",
    molecular_weight=Knowledge(180.16, epsilon=0.01, provenance="ms"),
    logp=Knowledge(1.19, epsilon=0.05, provenance="logd_assay"),
    hbd=1,
    hba=4,
)
```

### 4b. Run the three-stage pipeline

```python
from sounio.pipeline import DrugDiscoveryPipeline

pipeline = DrugDiscoveryPipeline()
result = pipeline.run(aspirin)

print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence:.3f}")

# Stage 1: Lipinski screening
print(f"\nScreening: {result.screening.passed}")
print(f"  confidence: {result.screening.confidence:.3f}")

# Stage 2: PK/PD model
pk = result.pk_params
print(f"\nPK Parameters:")
print(f"  Half-life:  {pk.half_life}")
print(f"  Cmax:       {pk.cmax}")
print(f"  AUC:        {pk.auc}")

# Stage 3: Monte Carlo simulation
sim = result.simulation
print(f"\nSimulation:")
print(f"  Efficacy:   {sim.efficacy_rate}")
print(f"  Adverse:    {sim.adverse_event_rate}")
print(f"  Therapeutic index: {sim.therapeutic_index}")
```

### 4c. Generate a report

```python
from sounio.report import ReportBuilder

rb = ReportBuilder("ASA-7 Drug Discovery Report", author="Sounio Demo")
rb.add_section("Background",
    "ASA-7 is an aspirin analog evaluated for anti-inflammatory activity.")
rb.add_knowledge_table("PK Parameters", {
    "Half-life (h)": result.pk_params.half_life,
    "Cmax (mg/L)": result.pk_params.cmax,
    "AUC (mg·h/L)": result.pk_params.auc,
    "Efficacy rate": result.simulation.efficacy_rate,
    "Adverse rate": result.simulation.adverse_event_rate,
})
rb.add_pipeline_summary(result)

# Save as Markdown (and optionally LaTeX)
rb.save("asa7_report.md", format="markdown")
print("Report saved to asa7_report.md")
print(rb.to_markdown()[:600])
```

---

## Part 5 — Jupyter Integration (5 min)

### 5a. Start a Jupyter notebook with the Sounio kernel

```bash
jupyter notebook
```

Create a new notebook and select **Sounio** as the kernel.

### 5b. Epistemic computing in cells

In a Sounio cell:

```sounio
fn main() with IO {
    let dose: mg = 500.0
    let weight: kg = 70.0
    let dose_per_kg = dose / weight
    print_knowledge(dose_per_kg)
}
```

### 5c. Switch to Python with `%python`

In the next cell (still in the Sounio kernel):

```
%python
# k0 is the Knowledge value from the previous cell
print(f"Dose per kg: {k0.value:.3f} ± {k0.epsilon:.4f} mg/kg")
print(f"Confidence: {k0.confidence:.1%}")
```

### 5d. Run the full pipeline

```
%drug_pipeline drug-discovery/examples/full_pipeline.sio
```

The kernel executes the pipeline and renders an HTML uncertainty dashboard inline.

---

## Summary

You have used:

| Feature | How |
|---------|-----|
| Epistemic arithmetic | `Knowledge(v, ε, prov)` + operators |
| GUM uncertainty propagation | automatic in `+`, `-`, `*`, `/` |
| Provenance DAG | `Knowledge.tracked()` + `.chain.to_dot()` |
| Sounio compiler from Python | `sounio.run_code()` |
| Async parallel execution | `asyncio.gather(sounio.async_run_code(...), ...)` |
| Drug discovery pipeline | `DrugDiscoveryPipeline().run(molecule)` |
| Reproducible reports | `ReportBuilder(...).save("report.md")` |
| Jupyter Sounio kernel | `%drug_pipeline`, `%python`, `%check` |

---

## Next Steps

- **API Reference**: [`docs/API_REFERENCE.md`](API_REFERENCE.md)
- **Jupyter User Guide**: [`docs/JUPYTER_USER_GUIDE.md`](JUPYTER_USER_GUIDE.md)
- **Full pipeline example**: [`drug-discovery/examples/full_pipeline.sio`](../drug-discovery/examples/)
- **Integration tests**: [`demo.py`](../demo.py)
- **Dashboard**: `python -m sounio.dashboard --port 8765`

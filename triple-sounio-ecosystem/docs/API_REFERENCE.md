# sounio-py API Reference

Complete reference for all public symbols in the `sounio` Python package.

---

## Core Type: `Knowledge`

```python
class Knowledge(value: float, epsilon: float = 0.0, provenance: str = "")
```

Epistemic type for a scalar measurement with quantified uncertainty. Arithmetic follows GUM (Guide to the Expression of Uncertainty in Measurement) propagation rules.

**Fields**

| Field | Type | Description |
|-------|------|-------------|
| `value` | `float` | Central (best) estimate |
| `epsilon` | `float` | Half-width uncertainty (k=1, ~68% coverage) |
| `provenance` | `str` | Human-readable origin label |

**Derived properties**

| Property | Formula | Description |
|----------|---------|-------------|
| `relative_uncertainty` | `ε / |value|` | Fractional uncertainty |
| `confidence` | `1 − relative_uncertainty` | Confidence score ∈ [0, 1] |

**Arithmetic (GUM standard)**

| Operation | Uncertainty rule |
|-----------|-----------------|
| `a + b` | `ε = √(εa² + εb²)` |
| `a - b` | `ε = √(εa² + εb²)` |
| `a * b` | `ε = |result| · √((εa/a)² + (εb/b)²)` |
| `a / b` | same as multiplication |
| `-a` | `ε = εa` |
| `abs(a)` | `ε = εa` |
| `a * scalar` | `ε = |scalar| · εa` |

**Provenance tracking**

```python
# Default: flat label string, no graph
x = Knowledge(1.0, 0.1, "sensor")

# With full DAG tracking
x = Knowledge.tracked(1.0, 0.1, "sensor", instrument="IMX-500")
y = Knowledge.tracked(2.0, 0.2, "model")
z = x + y
print(z.chain.to_dot())  # Graphviz DOT
print(z.chain.to_json())  # JSON-serializable dict
```

**Class methods**

```python
@classmethod
Knowledge.tracked(value, epsilon=0.0, provenance="", **metadata) -> Knowledge
```
Creates a Knowledge value with a live ProvenanceChain attached. All subsequent arithmetic on this value (and its descendants) records nodes in the DAG.

```python
@classmethod
Knowledge.from_dict(d: dict) -> Knowledge
```
Deserialize from `to_dict()` output.

**Instance methods**

```python
k.to_dict() -> dict
k.ancestry() -> list[ProvenanceNode]  # full ancestor walk (requires .tracked())
```

---

## Provenance: `ProvenanceChain` / `ProvenanceNode`

```python
from sounio import ProvenanceChain, ProvenanceNode
```

### `ProvenanceNode`

```python
@dataclass
class ProvenanceNode:
    id: str           # UUID4
    label: str        # human name
    operation: str    # "source" | "add" | "sub" | "mul" | "div" | "scale" | "neg" | "abs"
    value: float
    epsilon: float
    timestamp: float  # Unix epoch
    parent_ids: list[str]
    metadata: dict
```

### `ProvenanceChain`

```python
chain = ProvenanceChain()

# Introspection
chain.nodes() -> list[ProvenanceNode]
chain.parents_of(node_id: str) -> list[ProvenanceNode]
chain.ancestors_of(node_id: str) -> list[ProvenanceNode]  # full DAG walk
chain.summary() -> str   # human-readable one-liner

# Serialization
chain.to_dict() -> dict
chain.to_json() -> str
chain.to_dot() -> str    # Graphviz DOT, pipe to `dot -Tpng`

@classmethod
ProvenanceChain.from_dict(d: dict) -> ProvenanceChain
```

---

## Executor: `SounioExecutor`

```python
from sounio import SounioExecutor, ExecutionResult, CheckResult

ex = SounioExecutor(souc_path="/path/to/souc", stdlib_path="/path/to/stdlib")
```

**Synchronous methods**

```python
ex.run_file(path: str, timeout: int = 30) -> ExecutionResult
ex.run_code(code: str, timeout: int = 30) -> ExecutionResult
ex.check_file(path: str, show_ast=False, show_types=False) -> CheckResult
```

**Async methods**

```python
await ex.async_run_file(path: str, timeout: float = 60.0) -> ExecutionResult
await ex.async_run_code(code: str, timeout: float = 60.0) -> ExecutionResult
await ex.async_check_file(path: str, timeout: float = 30.0) -> CheckResult
```

### `ExecutionResult`

```python
@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    knowledge_values: list[Knowledge]  # parsed from stdout

    @property
    def ok(self) -> bool: ...
```

### `CheckResult`

```python
@dataclass
class CheckResult:
    passed: bool
    errors: list[str]
    ast: str      # populated when show_ast=True
    types: str    # populated when show_types=True
```

**Module-level convenience**

```python
import sounio

sounio.run_file(path, timeout=30)  -> ExecutionResult
sounio.run_code(code, timeout=30)  -> ExecutionResult
sounio.check_file(path)            -> CheckResult

# Async
await sounio.async_run_file(path)
await sounio.async_run_code(code)
```

---

## Kernel Client: `KernelConnection`

```python
from sounio import launch_jupyter_kernel, KernelConnection, KernelResult
```

```python
def launch_jupyter_kernel(timeout: float = 30.0) -> KernelConnection
```

Starts a Sounio Jupyter kernel subprocess and returns a live connection. Requires `jupyter_client` and the `sounio` kernel spec to be installed.

```python
with launch_jupyter_kernel() as kernel:
    result = kernel.execute("let x: f64 = 42.0")
    values = kernel.get_knowledge_values(
        'let m = measure(500.0, uncertainty: 5.0)'
    )
    kernel.shutdown()
```

### `KernelResult`

```python
@dataclass
class KernelResult:
    stdout: str
    stderr: str
    status: str          # "ok" | "error"
    knowledge_values: list  # parsed Knowledge-like objects
```

---

## Dashboard: `serve_dashboard`

```python
from sounio import serve_dashboard, create_app
```

```python
def serve_dashboard(
    host: str = "127.0.0.1",
    port: int = 8765,
    pipeline_path: str | None = None,
    open_browser: bool = True,
) -> None
```

Starts a blocking Flask server with an embedded single-page dashboard.

```python
def create_app(pipeline_path: str | None = None) -> Flask
```

Returns the Flask app for use with WSGI servers or testing.

**REST endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | HTML dashboard UI |
| `POST` | `/api/pipeline/run` | Run pipeline, returns Knowledge JSON |
| `GET` | `/api/pipeline/status` | Last run status |
| `GET` | `/api/knowledge/<prov_id>` | Single Knowledge value + chain |

---

## NumPy Integration: `UncertainArray`

```python
from sounio.integrations.numpy_ext import UncertainArray
import numpy as np

arr = UncertainArray(
    values=np.array([1.0, 2.0, 3.0]),
    epsilons=np.array([0.1, 0.2, 0.3]),
    provenance="sensor_batch",
)
```

**Arithmetic** — all standard NumPy element-wise operators with GUM propagation.

**Shared memory IPC** — zero-copy transfer between processes:

```python
# Producer process
shm_names = arr.to_shared_memory()
# shm_names = (values_shm_name, epsilons_shm_name, meta_json)

# Consumer process
arr2 = UncertainArray.from_shared_memory(*shm_names)

# Cleanup (producer)
UncertainArray.free_shared_memory(shm_names[0], shm_names[1])
```

---

## Pandas Integration: `EpistemicDataFrame`

```python
from sounio.integrations.pandas_ext import EpistemicDataFrame

edf = EpistemicDataFrame.from_dict({
    "weight_kg":  ([70.0, 80.0], [1.5, 2.0]),   # (values, epsilons)
    "height_cm":  ([175.0, 180.0], [0.5, 0.5]),
}, provenance="clinical_cohort")

# Access uncertainty columns
edf["weight_kg"]          # pandas Series of values
edf.epsilon("weight_kg")  # pandas Series of epsilons
edf.confidence("weight_kg")  # Series of confidence scores

# Epistemic statistics
edf.epistemic_describe()  # DataFrame with value ± ε stats per column
```

---

## Report Generation: `ReportBuilder`

```python
from sounio.report import ReportBuilder
from sounio import Knowledge

rb = ReportBuilder("Drug Discovery Study", author="Dr. Smith")
rb.add_section("Background", "This study investigates compound XY-42...")
rb.add_knowledge_table("PK Parameters", {
    "Half-life (h)": Knowledge(4.62, 0.767, "pk_model"),
    "Clearance (L/h)": Knowledge(12.5, 1.5, "pk_model"),
})
rb.add_pipeline_summary(pipeline_result)

# Save
rb.save("report.md", format="markdown")
rb.save("report.tex", format="latex")
print(rb.to_markdown())
```

---

## Ontology Integration

```python
import sounio

# Search
results = sounio.search("diabetes")         # list[ResolvedOntologyTerm]
term = sounio.resolve("SNOMED:44054006")    # ResolvedOntologyTerm | None

# Hierarchy
ancestors = sounio.ancestors("SNOMED:44054006")  # list[ResolvedOntologyTerm]
is_child = sounio.is_subclass("HPO:0003074", "HPO:0000118")

# Normalize a clinical payload
normalized = sounio.clinical_normalize({
    "patient_id": "pt-001",
    "diagnoses": ["SNOMED:44054006"],
    "labs": ["LOINC:4548-4"],
    "phenotypes": ["HPO:0003074"],
})
```

### `ResolvedOntologyTerm`

```python
@dataclass
class ResolvedOntologyTerm:
    curie: str       # e.g. "SNOMED:44054006"
    label: str       # e.g. "Diabetes mellitus type 2"
    source: str      # ontology name
    confidence: float
    synonyms: list[str]
    ancestors: list[str]

    def to_dict(self) -> dict
```

---

## Domain Types

```python
from sounio import (
    Molecule, PKParameters, PatientData,
    SimulationResult, ScreeningResult, PipelineResult,
)
```

### `Molecule`

```python
@dataclass
class Molecule:
    name: str
    molecular_weight: Knowledge   # Da
    logp: Knowledge               # partition coefficient
    hbd: int                      # H-bond donors
    hba: int                      # H-bond acceptors
    smiles: str = ""
```

### `PKParameters`

```python
@dataclass
class PKParameters:
    bioavailability: Knowledge   # fraction absorbed
    ka: Knowledge                # absorption rate constant (1/h)
    cl: Knowledge                # clearance (L/h)
    vd: Knowledge                # volume of distribution (L)
```

### `PipelineResult`

```python
@dataclass
class PipelineResult:
    molecule: Molecule
    screening: ScreeningResult
    pk_params: PKParameters
    simulation: SimulationResult
    decision: str                # "PROCEED" | "HALT"
    confidence: float
    provenance: ProvenanceChain | None
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOUC` | auto-detected | Path to the `souc` compiler binary |
| `SOUNIO_STDLIB_PATH` | auto-detected | Path to Sounio standard library |

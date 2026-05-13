# sounio-py API Reference

Complete reference for the sounio-py Python API.

---

## Knowledge Class

The core epistemic type — a measured value with GUM-compliant uncertainty.

### Constructor

```python
Knowledge(value: float, epsilon: float = 0.0, provenance: str = "")
```

Create a new `Knowledge` instance.

**Parameters:**

- **value** (float) — Central estimate of the quantity
- **epsilon** (float) — Standard uncertainty under GUM (1σ). Default: 0.0 (exact value)
- **provenance** (str) — Free-form label describing measurement source. Default: ""

**Example:**

```python
import sounio

# Exact value (no uncertainty)
x = sounio.Knowledge(100.0)

# Measured value with uncertainty
temp = sounio.Knowledge(36.5, 0.1, "clinical thermometer")

# String representation
print(temp)  # Knowledge(36.500 ± 0.100, prov='clinical thermometer')
```

### Properties

#### .value

```python
@property
def value() -> float
```

The central estimate (point estimate) of the quantity.

```python
k = sounio.Knowledge(36.5, 0.1)
assert k.value == 36.5
```

#### .epsilon

```python
@property
def epsilon() -> float
```

The standard uncertainty (GUM, k=1, one sigma).

```python
k = sounio.Knowledge(36.5, 0.1)
assert k.epsilon == 0.1
```

#### .provenance

```python
@property
def provenance() -> str
```

Source/method label for traceability.

```python
k = sounio.Knowledge(36.5, 0.1, "IR thermometer")
assert k.provenance == "IR thermometer"
```

#### .relative_uncertainty

```python
@property
def relative_uncertainty() -> float
```

Relative uncertainty: ε / |value|. Represents uncertainty as a fraction of the measurement.

```python
k = sounio.Knowledge(100.0, 5.0)
assert k.relative_uncertainty == 0.05  # 5%
```

#### .confidence

```python
@property
def confidence() -> float
```

Approximate confidence interval assuming normal distribution: 1 - P(|x| > 2σ) ≈ 95.45%.

```python
k = sounio.Knowledge(100.0, 10.0)
c = k.confidence  # ~0.9545 (95% confidence)
```

### Arithmetic Operations

All operations follow GUM rules for uncertainty propagation.

#### Addition

```python
def __add__(self, other: float | Knowledge) -> Knowledge
```

Add `other` to this `Knowledge`. Uncertainties add in quadrature: ε_z = √(ε_a² + ε_b²).

```python
a = sounio.Knowledge(100.0, 5.0, "A")
b = sounio.Knowledge(200.0, 10.0, "B")

c = a + b
assert c.value == 300.0
assert abs(c.epsilon - 11.18) < 0.01  # √(25 + 100)
```

#### Subtraction

```python
def __sub__(self, other: float | Knowledge) -> Knowledge
```

Subtract `other` from this `Knowledge`. Uncertainties add in quadrature (same as addition).

```python
a = sounio.Knowledge(200.0, 10.0)
b = sounio.Knowledge(100.0, 5.0)

diff = a - b
assert diff.value == 100.0
assert abs(diff.epsilon - 11.18) < 0.01
```

#### Multiplication

```python
def __mul__(self, other: float | Knowledge) -> Knowledge
```

Multiply by `other`. Relative uncertainties add in quadrature: (ε_z/z)² = (ε_a/a)² + (ε_b/b)².

```python
a = sounio.Knowledge(100.0, 10.0)  # 100 ± 10 (10% relative)
b = sounio.Knowledge(20.0, 2.0)    # 20 ± 2 (10% relative)

c = a * b
assert c.value == 2000.0
# Relative: √(0.1² + 0.1²) = 0.1414 (14.14%)
assert abs(c.epsilon - 282.8) < 1.0
```

#### Division

```python
def __truediv__(self, other: float | Knowledge) -> Knowledge
```

Divide by `other`. Relative uncertainties add in quadrature (same rule as multiplication).

```python
a = sounio.Knowledge(100.0, 10.0)  # 100 ± 10
b = sounio.Knowledge(5.0, 0.5)     # 5 ± 0.5

c = a / b
assert c.value == 20.0
assert abs(c.epsilon - 4.47) < 0.1  # Relative √(0.1² + 0.1²)
```

#### Negation

```python
def __neg__(self) -> Knowledge
```

Negate the value; uncertainty stays the same.

```python
k = sounio.Knowledge(36.5, 0.1, "thermometer")
neg_k = -k
assert neg_k.value == -36.5
assert neg_k.epsilon == 0.1
```

### Comparison Operations

#### Equality

```python
def __eq__(self, other: float | Knowledge) -> bool
```

Compare for equality. Two `Knowledge` values are equal if both value and epsilon match.

```python
a = sounio.Knowledge(100.0, 5.0)
b = sounio.Knowledge(100.0, 5.0)
assert a == b

c = sounio.Knowledge(100.0, 6.0)
assert a != c
```

#### Less Than / Greater Than

```python
def __lt__(self, other: float | Knowledge) -> bool
def __le__(self, other: float | Knowledge) -> bool
def __gt__(self, other: float | Knowledge) -> bool
def __ge__(self, other: float | Knowledge) -> bool
```

Compare central values only (uncertainty is not considered in comparisons).

```python
a = sounio.Knowledge(100.0, 10.0)
b = sounio.Knowledge(200.0, 5.0)

assert a < b
assert b > a
assert a <= 150.0
```

### String Representation

#### __str__

```python
def __str__(self) -> str
```

Human-readable format: `Knowledge(value ± epsilon, prov='...')`.

```python
k = sounio.Knowledge(36.5, 0.1, "thermometer")
print(k)  # Knowledge(36.500 ± 0.100, prov='thermometer')
```

#### __repr__

```python
def __repr__(self) -> str
```

Machine-readable format suitable for `eval()`.

```python
k = sounio.Knowledge(36.5, 0.1, "thermometer")
s = repr(k)  # "Knowledge(36.5, 0.1, 'thermometer')"
k2 = eval(s)
assert k2 == k
```

---

## SounioExecutor Class

Interface to the Sounio compiler (`souc` binary) for type-checking and executing Sounio code.

### Constructor

```python
SounioExecutor(souc_path: str | None = None, stdlib_path: str | None = None, timeout: int = 30)
```

Create a new executor.

**Parameters:**

- **souc_path** (str, optional) — Path to `souc` binary. Default: search PATH
- **stdlib_path** (str, optional) — Path to Sounio stdlib. Default: environment variable `SOUNIO_STDLIB_PATH`
- **timeout** (int) — Timeout for execution in seconds. Default: 30

**Example:**

```python
import sounio

executor = sounio.SounioExecutor(
    souc_path="/usr/local/bin/souc",
    stdlib_path="/home/user/sounio/stdlib",
    timeout=60
)
```

### Methods

#### check_file(path: str, **kwargs)

```python
def check_file(self, path: str, show_ast: bool = False, show_types: bool = False) -> CheckResult
```

Type-check a `.sio` file without executing it.

**Parameters:**

- **path** (str) — Path to `.sio` file
- **show_ast** (bool) — Include AST in result. Default: False
- **show_types** (bool) — Include inferred types. Default: False

**Returns:** `CheckResult` object

**Example:**

```python
result = executor.check_file('my_program.sio')
if result.ok:
    print("✓ Type check passed")
else:
    print(f"✗ Error: {result.error}")
```

#### check_code(code: str, **kwargs)

```python
def check_code(self, code: str, show_ast: bool = False, show_types: bool = False) -> CheckResult
```

Type-check inline Sounio code.

**Parameters:**

- **code** (str) — Sounio source code
- **show_ast** (bool) — Include AST in result. Default: False
- **show_types** (bool) — Include inferred types. Default: False

**Returns:** `CheckResult` object

#### run_file(path: str, timeout: int | None = None)

```python
def run_file(self, path: str, timeout: int | None = None) -> ExecutionResult
```

JIT-execute a `.sio` file.

**Parameters:**

- **path** (str) — Path to `.sio` file
- **timeout** (int, optional) — Execution timeout in seconds. Default: executor default

**Returns:** `ExecutionResult` object

**Example:**

```python
result = executor.run_file('compute_pk.sio')
print(result.stdout)
print(f"Exit code: {result.exit_code}")
```

#### run_code(code: str, timeout: int | None = None)

```python
def run_code(self, code: str, timeout: int | None = None) -> ExecutionResult
```

JIT-execute inline Sounio code.

**Parameters:**

- **code** (str) — Sounio source code
- **timeout** (int, optional) — Execution timeout. Default: executor default

**Returns:** `ExecutionResult` object

**Example:**

```python
code = '''
fn main() with IO {
    print_f64(sqrt(16.0))
}
'''
result = executor.run_code(code)
print(result.stdout)  # "4.0"
```

---

## ExecutionResult Class

Result of executing Sounio code via `run_file()` or `run_code()`.

### Properties

#### .exit_code

```python
@property
def exit_code() -> int
```

Process exit code (0 = success).

#### .stdout

```python
@property
def stdout() -> str
```

Standard output captured during execution.

#### .stderr

```python
@property
def stderr() -> str
```

Standard error captured during execution.

#### .runtime_seconds

```python
@property
def runtime_seconds() -> float
```

Elapsed execution time in seconds.

#### .success

```python
@property
def success() -> bool
```

True if exit_code == 0.

### Example

```python
result = sounio.run_code('''
fn main() with IO {
    print(42)
}
''')

if result.success:
    print(f"Output: {result.stdout}")
    print(f"Runtime: {result.runtime_seconds:.3f}s")
else:
    print(f"Error: {result.stderr}")
    print(f"Exit code: {result.exit_code}")
```

---

## CheckResult Class

Result of type-checking Sounio code via `check_file()` or `check_code()`.

### Properties

#### .ok

```python
@property
def ok() -> bool
```

True if type-check passed.

#### .error

```python
@property
def error() -> str
```

Error message if check failed (empty string if ok=True).

#### .ast

```python
@property
def ast() -> str | None
```

Abstract syntax tree (if show_ast=True was passed).

#### .types

```python
@property
def types() -> dict[str, str] | None
```

Inferred types for all bindings (if show_types=True was passed).

### Example

```python
result = sounio.check_code('''
let x: i32 = 100.0  // Type error!
''')

if not result.ok:
    print(f"✗ Type error: {result.error}")
```

---

## UncertainArray Class

NumPy-like arrays where each element is a `Knowledge` value.

### Constructor

```python
UncertainArray(values: np.ndarray, uncertainties: np.ndarray, provenance: str = "")
```

Create an array of uncertain measurements.

**Parameters:**

- **values** (np.ndarray) — Central estimates
- **uncertainties** (np.ndarray) — Standard uncertainties (element-wise)
- **provenance** (str) — Source label. Default: ""

**Example:**

```python
import numpy as np
import sounio

values = np.array([100.0, 110.0, 105.0])
errors = np.array([5.0, 5.5, 5.0])

arr = sounio.UncertainArray(values, errors, "assay batch 042")
```

### Properties

#### .values

```python
@property
def values() -> np.ndarray
```

Central estimates (read-only).

#### .uncertainties

```python
@property
def uncertainties() -> np.ndarray
```

Standard uncertainties (read-only).

#### .shape

```python
@property
def shape() -> tuple
```

Array shape (e.g., `(3,)` for 1D array).

#### .dtype

```python
@property
def dtype() -> np.dtype
```

Data type (float64).

### Methods

#### mean()

```python
def mean() -> Knowledge
```

Compute mean with propagated uncertainty.

**Returns:** `Knowledge` value

**Example:**

```python
arr = sounio.UncertainArray([100, 110, 105], [5, 5, 5])
mean = arr.mean()
print(f"Mean: {mean.value:.1f} ± {mean.epsilon:.1f}")
```

#### sum()

```python
def sum() -> Knowledge
```

Compute sum with propagated uncertainty.

#### std()

```python
def std() -> float
```

Compute standard deviation of values (not propagated uncertainty).

#### __add__, __sub__, __mul__, __truediv__

```python
def __add__(self, other: float | Knowledge | UncertainArray) -> UncertainArray
def __sub__(self, other: float | Knowledge | UncertainArray) -> UncertainArray
def __mul__(self, other: float | Knowledge | UncertainArray) -> UncertainArray
def __truediv__(self, other: float | Knowledge | UncertainArray) -> UncertainArray
```

Element-wise arithmetic with uncertainty propagation.

**Example:**

```python
arr = sounio.UncertainArray([100, 200], [5, 10])
scaled = arr * 2.5  # Scales values and errors

# Convert units: mg to μg
arr_ug = arr * 1000
```

---

## EpistemicDataFrame Class

Pandas DataFrame extension where selected columns are `Knowledge` values.

### Constructor

```python
EpistemicDataFrame(data: dict | pd.DataFrame, epistemic_cols: list[str] | None = None)
```

Create an epistemic DataFrame.

**Parameters:**

- **data** (dict or pd.DataFrame) — Input data
- **epistemic_cols** (list, optional) — Columns to treat as epistemic. Default: auto-detect

**Example:**

```python
import sounio
import pandas as pd

data = {
    'patient_id': [1, 2, 3],
    'dose': [100, 150, 125],
    'dose_err': [2, 3, 2.5],
    'ck_baseline': [0.9, 1.1, 0.95],
    'ck_err': [0.05, 0.06, 0.05],
}

edf = sounio.EpistemicDataFrame(data)
edf['derived'] = edf['dose'] / edf['ck_baseline']
```

### Methods

#### epistemic_summary()

```python
def epistemic_summary() -> pd.DataFrame
```

Compute summary statistics with uncertainty.

**Returns:** DataFrame with columns:

- `mean` — mean value
- `std` — standard deviation
- `uncertainty` — propagated uncertainty
- `relative_uncertainty` — relative error

**Example:**

```python
summary = edf[['dose', 'clearance']].epistemic_summary()
print(summary)
```

---

## Domain Types

Pre-defined types for scientific domains.

### Molecule

```python
class Molecule:
    name: str
    smiles: str
    molecular_weight: float
    log_p: float
    hbd: int  # H-bond donors
    hba: int  # H-bond acceptors
```

Represent a chemical compound.

**Example:**

```python
drug = sounio.Molecule(
    name="Ibuprofen",
    smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    molecular_weight=206.28,
    log_p=3.97,
    hbd=1,
    hba=2
)

# Check Lipinski's Rule
if drug.molecular_weight < 500 and drug.log_p < 5:
    print(f"✓ {drug.name} passes Lipinski filter")
```

### PKParameters

```python
class PKParameters:
    clearance: float         # mL/min
    volume_of_distribution: float  # L
    absorption_rate: float   # 1/hr
    half_life: float        # hours
```

Pharmacokinetic parameters.

**Example:**

```python
pk = sounio.PKParameters(
    clearance=120.0,
    volume_of_distribution=50.0,
    absorption_rate=0.5,
    half_life=5.0
)

elimination_rate = 0.693 / pk.half_life
```

### PatientData

```python
class PatientData:
    weight_kg: float
    age_years: int
    creatinine_clearance: float
    liver_function: str
```

Patient demographic and functional data.

### SimulationResult

```python
class SimulationResult:
    patient_id: int
    simulated_pk_profile: list[float]
    simulated_times: list[float]
    efficacy_probability: float
    toxicity_risk: float
```

Result of a clinical trial simulation.

### ScreeningResult

```python
class ScreeningResult:
    molecule_id: int
    passed: bool
    confidence: float
    violations: list[str]
```

Result of virtual screening for a compound.

### PipelineResult

```python
class PipelineResult:
    screening_results: list[ScreeningResult]
    pk_models: list[PKParameters]
    trial_simulations: list[SimulationResult]
```

Complete output from the drug discovery pipeline.

---

## Ontology Module

Clinical terminology integration.

### Functions

#### resolve(term: str) -> ResolvedOntologyTerm

```python
def resolve(term: str) -> ResolvedOntologyTerm
```

Resolve a clinical term to standardized codes.

**Parameters:**

- **term** (str) — Clinical term (e.g., "eGFR 45-59", "CKD stage 3a")

**Returns:** `ResolvedOntologyTerm` with SNOMED-CT, ICD-10-CM, LOINC codes

**Example:**

```python
term = sounio.ontology.resolve("Type 2 Diabetes")
print(f"ICD-10: {term.icd10_cm}")
print(f"SNOMED: {term.snomed_ct}")
```

#### search(query: str, limit: int = 10) -> list[ResolvedOntologyTerm]

```python
def search(query: str, limit: int = 10) -> list[ResolvedOntologyTerm]
```

Search for terms in the ontology.

**Parameters:**

- **query** (str) — Search string
- **limit** (int) — Max results. Default: 10

**Example:**

```python
results = sounio.ontology.search("glucose", limit=5)
for term in results:
    print(f"  {term.preferred_label}")
```

#### clinical_normalize(value_with_unit: str) -> dict

```python
def clinical_normalize(value_with_unit: str) -> dict
```

Convert clinical values to SI units.

**Parameters:**

- **value_with_unit** (str) — e.g., "Creatinine 0.9 mg/dL"

**Returns:** Dictionary with normalized units and SI value

**Example:**

```python
norm = sounio.ontology.clinical_normalize("Glucose 95 mg/dL")
print(f"mmol/L: {norm['value_si']}")
```

---

## Module Functions

### get_executor(**kwargs) -> SounioExecutor

```python
def get_executor(**kwargs) -> SounioExecutor
```

Get the module-level default executor (created lazily).

**Example:**

```python
executor = sounio.get_executor()
result = executor.run_code("fn main() with IO { print(42) }")
```

### reset_executor()

```python
def reset_executor() -> None
```

Discard the cached default executor (useful in tests).

### run_file(path: str, timeout: int = 30) -> ExecutionResult

```python
def run_file(path: str, timeout: int = 30) -> ExecutionResult
```

Convenience function: execute a `.sio` file using the default executor.

### run_code(code: str, timeout: int = 30) -> ExecutionResult

```python
def run_code(code: str, timeout: int = 30) -> ExecutionResult
```

Convenience function: execute inline Sounio code using the default executor.

### check_file(path: str) -> CheckResult

```python
def check_file(path: str) -> CheckResult
```

Convenience function: type-check a `.sio` file using the default executor.

---

## Constants

### __version__

```python
__version__: str
```

sounio-py version string (e.g., "0.1.0").

### _NATIVE

```python
_NATIVE: bool
```

True if native Rust extension is loaded; False if using pure Python fallback.

---

## Exception Hierarchy

### Knowledge-related

```python
try:
    k = sounio.Knowledge(100.0, -5.0)  # Negative uncertainty
except ValueError as e:
    print(f"Invalid Knowledge: {e}")
```

### Executor-related

```python
try:
    result = sounio.run_code("invalid sounio code")
except RuntimeError as e:
    print(f"Sounio compilation failed: {e}")
```

---

## Example: Complete Workflow

```python
import sounio
import numpy as np

# Create measurements
measurements = [
    sounio.Knowledge(95.0, 2.0, "assay 001"),
    sounio.Knowledge(105.0, 2.5, "assay 001"),
    sounio.Knowledge(100.0, 2.0, "assay 001"),
]

# Compute mean with propagated uncertainty
mean_value = sum(measurements[1:], measurements[0])
mean_value.value /= len(measurements)
mean_value.epsilon = np.sqrt(sum(m.epsilon**2 for m in measurements)) / len(measurements)

print(f"Mean: {mean_value.value:.1f} ± {mean_value.epsilon:.1f}")

# Use array
arr = sounio.UncertainArray(
    np.array([m.value for m in measurements]),
    np.array([m.epsilon for m in measurements])
)
arr_mean = arr.mean()
print(f"Array mean: {arr_mean.value:.1f} ± {arr_mean.epsilon:.1f}")

# Type-check Sounio code
check = sounio.check_code("let x: i32 = 42")
if check.ok:
    print("✓ Type check passed")
```

---

For more examples, see the [Quickstart](quickstart.md) and [Examples Repository](https://github.com/sounio-org/sounio/tree/main/ecosystem/sounio-py/examples).

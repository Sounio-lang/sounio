# Sounio Jupyter Kernel — User Guide

Complete guide to running Sounio code in Jupyter notebooks with epistemic uncertainty support.

---

## Installation

```bash
cd triple-sounio-ecosystem/sounio-jupyter
pip install -e .
jupyter kernelspec install kernelspec/ --user
jupyter kernelspec list  # should show "sounio"
```

---

## Basics

Select **Sounio** as the kernel when creating a new notebook. Cells execute Sounio code directly — the compiler is invoked automatically.

### Simple expressions

```sounio
let x: f64 = 42.0
let y: f64 = x * 2.0
```

Output shows the result, with Knowledge values rendered as colored uncertainty cards.

### Functions

```sounio
fn square(x: f64) -> f64 {
    x * x
}

fn main() with IO {
    let s = square(7.0)
    print_f64(s)
}
```

---

## Epistemic Types

```sounio
fn main() with IO {
    let mass = measure(78.5, uncertainty: 0.3)   // Knowledge<kg>
    let height = measure(1.75, uncertainty: 0.01) // Knowledge<m>

    let bmi = mass / (height * height)
    print_knowledge(bmi)
    // Knowledge { value: 25.63 epsilon: 0.24 prov: "mass/(height*height)" }
}
```

The kernel renders Knowledge values with a visual confidence bar in the notebook output.

---

## Magic Commands

All magic commands begin with `%`. They do not execute Sounio code — they interact with the kernel or external tools.

### `%time` — time a statement

```
%time let x: f64 = 1.0 + 2.0 + 3.0
```

Output: `CPU time: 0.0123s` followed by the stdout of the snippet.

---

### `%timeit` — benchmark (multiple runs)

```
%timeit -n 10 let x: f64 = 1.0 + 2.0
```

Output: min / max / avg across N runs.

---

### `%check` — type-check without executing

```
%check let x: f64 = "hello"
```

Output: `✗ Type check failed:  expected f64, found str`

---

### `%ast` — show parse tree

```
%ast let x = 42
```

Prints the AST dump from `souc check --show-ast`.

---

### `%types` — show inferred types

```
%types let x = 42
```

Prints type annotations from `souc check --show-types`.

---

### `%sounio` — kernel information

```
%sounio info     # version, stdlib path, souc binary
%sounio stdlib   # stdlib directory
%sounio souc     # compiler binary path
```

---

### `%python` — run Python with Sounio context

Executes Python code in a subprocess. Knowledge values from the **last Sounio cell** are injected as `k0`, `k1`, ..., `knowledge_values`.

```
%python print(k0.value, "±", k0.epsilon)
```

```
%python import math; print(math.sqrt(k0.value))
```

For multi-line Python, place `%python` on the first line and write Python code in the rest of the cell:

```
%python
import json
summary = {
    "value": k0.value,
    "epsilon": k0.epsilon,
    "confidence": k0.confidence,
    "prov": k0.provenance,
}
print(json.dumps(summary, indent=2))
```

---

### `%drug_pipeline` — run drug discovery pipeline

```
%drug_pipeline                                  # use default pipeline
%drug_pipeline path/to/custom_pipeline.sio     # use specific file
```

Executes the pipeline and renders results as an HTML report with uncertainty cards.

---

### `%ontology_search` — search biomedical ontologies

```
%ontology_search diabetes
%ontology_search hypertension
```

Returns matching terms from SNOMED CT, LOINC, HPO, MeSH.

---

### `%ontology_resolve` — resolve a CURIE

```
%ontology_resolve SNOMED:44054006
%ontology_resolve LOINC:4548-4
%ontology_resolve HPO:0003074
```

Returns the canonical term object as JSON.

---

### `%clinical_normalize` — normalize a patient record

```
%clinical_normalize {"patient_id":"pt-1","diagnoses":["SNOMED:44054006"],"labs":["LOINC:4548-4"],"phenotypes":["HPO:0003074"]}
```

Or from a file:

```
%clinical_normalize data/patient_001.json
```

Returns the normalized payload with resolved ontology labels.

---

### `%%writefile` — write cell to a file

Use as a **cell magic** (double `%%`) to save the cell body to a `.sio` file:

```
%%writefile my_model.sio
fn pk_model(ka: f64, cl: f64, vd: f64) -> f64 {
    ka / (cl / vd)
}
```

---

## Completion and Inspection

- **Tab** — autocomplete Sounio keywords, identifiers defined in previous cells
- **Shift+Tab** — inline documentation for keywords (`let`, `var`, `fn`, `Knowledge`, etc.)

---

## Kernel Configuration

Set environment variables before launching Jupyter:

```bash
export SOUC=/path/to/souc               # compiler binary
export SOUNIO_STDLIB_PATH=/path/to/stdlib  # standard library
jupyter notebook
```

Or use a `.env` file with `python-dotenv`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "souc binary not found" | Set `SOUC` env var or add souc to `$PATH` |
| "stdlib not found" | Set `SOUNIO_STDLIB_PATH` |
| Kernel crashes on large code | Increase JIT memory with `SOUC_JIT_MEMORY=2g` |
| `%python` shows no `k0` | Ensure previous Sounio cell printed a Knowledge value |
| `%drug_pipeline` error | Check pipeline file path; install sounio-py first |

---

## Integration with sounio-py

```python
# In a Python cell of a regular Jupyter notebook (not Sounio kernel)
import sounio

with sounio.launch_jupyter_kernel() as k:
    result = k.execute("""
        let mass = measure(78.5, uncertainty: 0.3)
        let height = measure(1.75, uncertainty: 0.01)
        print_knowledge(mass / (height * height))
    """)
    print(result.knowledge_values)
```

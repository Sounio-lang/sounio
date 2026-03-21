# Sounio Jupyter Kernel

A Jupyter kernel for the **Sounio** programming language with first-class support for **epistemic computing** and uncertainty quantification.

## Overview

The Sounio Jupyter Kernel allows you to run Sounio code directly in Jupyter notebooks and JupyterLab, with native support for:

- **Epistemic types** — `Knowledge<T>` with confidence measures
- **Uncertainty visualization** — HTML-rendered values with color-coded confidence
- **Provenance tracking** — Understand the source of computed values
- **Automatic code wrapping** — Write expressions without boilerplate

## Installation

### Prerequisites

- Python ≥ 3.8
- Jupyter ≥ 4.9 or JupyterLab ≥ 3.0
- Sounio compiler (`souc`) — [install from sounio.dev](https://sounio.dev)

### From Source

```bash
cd triple-sounio-ecosystem/sounio-jupyter
pip install -e .

# Register the kernel with Jupyter
jupyter kernelspec install kernelspec/ --user
```

### Verify Installation

```bash
jupyter kernelspec list | grep sounio
```

You should see:
```
sounio    /path/to/kernelspec/
```

## Quick Start

### Start a Jupyter Notebook

```bash
jupyter notebook
```

In a new notebook cell, select **Sounio** as the kernel.

### Day 1 Example (Type checking)

```sounio
let x = 42
let y = 3.14
```

### Day 2 Example (Simple computation)

```sounio
let sum = 1 + 2 + 3
let product = sum * 10
```

### Day 3 Example (Epistemic values)

```sounio
let measured = measure(100.0, uncertainty: 5.0)
print(measured)
```

Expected output (with color):
```
Knowledge { value: 100.000000, epsilon: 0.952381, prov: "measure" }
```

(Rendered as green box with 95% confidence indicator)

## Architecture

### Day 1: Foundation (✓ Complete)

- **`pyproject.toml`** — Package metadata, dependencies (ipykernel, jupyter-client)
- **`kernel.py`** — `SounioKernel` class inheriting from `ipykernel.kernelbase.Kernel`
- **`executor.py`** — `CellExecutor` subprocess manager (stub for Day 2)
- **`display.py`** — HTML formatters for epistemic values
- **`magics.py`** — IPython magic command stubs
- **`kernel.json`** — Jupyter kernel specification

### Day 2: Executor (Planned)

- Implement `CellExecutor.run_cell()` → wraps code → calls `souc run`
- Auto-wrapper for expressions: `let x = 1 + 1` → `fn main() { ... }`
- Capture stdout/stderr from souc subprocess
- Handle execution errors gracefully

### Day 3: Display & Magics (Planned)

- **Display**: Render `Knowledge<T>` values with confidence coloring
  - Green (#2ecc71): ε ≥ 0.9
  - Orange (#f39c12): ε ≥ 0.7
  - Red (#e74c3c): ε < 0.7
- **Magics**: Implement `%time`, `%sounio info`, etc.

## Key Code Patterns

### Auto-wrapping (Day 2)

```python
if not code.strip().startswith("fn "):
    code = f"fn main() with IO, Mut, Div, Panic {{\n{code}\n}}"
```

### Knowledge HTML rendering (Day 3)

```python
def format_knowledge_html(value: float, epsilon: float, provenance: str) -> str:
    # Returns colored HTML div based on epsilon confidence level
```

## Configuration

### Environment Variables

- **`SOUC`** — Path to souc binary (default: auto-detect)
- **`SOUNIO_STDLIB_PATH`** — Path to Sounio stdlib (default: auto-detect)

### Kernel Metadata (`kernel.json`)

- `epistemic_support`: true
- `uncertainty_visualization`: true
- `provenance_tracking`: true
- `scientific_computing`: true

## Magic Commands

Magic commands (prefixed with `%` or `%%`) provide shortcuts for common tasks.

### `%drug_pipeline`
Run a stage of the epistemic drug discovery pipeline directly from Python sounio-py.

```sounio
%drug_pipeline screen --mol aspirin --mw 180.16 --logp 1.19 --hbd 1 --hba 3
```

Outputs epistemic results via sounio-py integration.

### `%time`
Measure execution time of a cell.

```sounio
%time
let fib_10 = fibonacci(10)
```

### `%sounio info`
Display Sounio compiler and kernel version.

### `%check`
Type-check code without running it.

```sounio
%check
let x: i32 = "wrong"  // Error: string is not i32
```

### `%show-ast`
Dump the abstract syntax tree.

```sounio
%show-ast
let x = 1 + 2
```

### `%show-types`
Display inferred types for all bindings.

```sounio
%show-types
let x = 5
let y = 3.14
let z = x + y
```

## Integration with sounio-py

The kernel automatically detects and uses `sounio-py` for pipeline operations. When you use `%drug_pipeline`, results flow through the epistemic Python API for visualization and further analysis.

```python
# In a Python cell (after sounio-py is installed)
import sounio

# Results from previous Sounio cells are available
pipeline = sounio.DrugDiscoveryPipeline()
# ... use Python for analysis
```

## Known Limitations

- **Streaming output**: No real-time output (JIT must complete before returning)
- **Timeout**: 30s execution timeout per cell
- **No REPL state persistence**: Each cell runs in isolation (by design)
- **Magic commands**: Basic set implemented; advanced IPython magics (`%%cython`, `%%capture`) not supported

## Development

### Run Tests

```bash
pytest tests/
```

### Install in Development Mode

```bash
pip install -e ".[dev]"
```

### Build & Lint

```bash
black sounio_kernel/
ruff check sounio_kernel/
mypy sounio_kernel/
```

## References

- [Sounio Language Docs](https://docs.sounio.dev)
- [Epistemic Programming Guide](https://docs.sounio.dev/guide/epistemic)
- [Jupyter Kernel Spec](https://jupyter-client.readthedocs.io/en/latest/kernels.html)
- [IPykernel Source](https://github.com/ipython/ipykernel)

## License

Apache-2.0
